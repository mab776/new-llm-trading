"""
Optimization driver: loads full BTC history once, precomputes indicators once,
then evaluates configs across walk-forward folds (different market regimes) in ~40ms each.

Metric philosophy: "maximize profits" but robustly. We report each fold's compounded
return and aggregate with the GEOMETRIC mean of (1+fold_return) — this rewards configs
that compound across ALL regimes and punishes any fold that blows up (a -90% fold tanks
the geomean even if another fold is +900%). We also track worst fold and max drawdown.
"""
from __future__ import annotations

import sys, json, time
from dataclasses import dataclass
import numpy as np

from llm_trading_bot.config import load_config, AppConfig
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
import opt.fastbt as fb
from llm_trading_bot.timeframes import timeframe_hours

# Walk-forward folds (start, end). Distinct regimes.
FOLDS = [
    ("2021", "2021-01-01", "2021-12-31"),  # bull -> sharp correction
    ("2022", "2022-01-01", "2022-12-31"),  # bear market
    ("2023", "2023-01-01", "2023-12-31"),  # choppy recovery
    ("2024", "2024-01-01", "2024-12-31"),  # bull
    ("2025", "2025-01-01", "2025-06-01"),  # recent
]

# Finer walk-forward folds (half-years) — used for train/test to reduce overfitting.
HALF_FOLDS = [
    ("21H1", "2021-01-01", "2021-06-30"), ("21H2", "2021-07-01", "2021-12-31"),
    ("22H1", "2022-01-01", "2022-06-30"), ("22H2", "2022-07-01", "2022-12-31"),
    ("23H1", "2023-01-01", "2023-06-30"), ("23H2", "2023-07-01", "2023-12-31"),
    ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
    ("25H1", "2025-01-01", "2025-06-01"),
]
# Interleaved split: train on odd half-years, test on even — both cover all regimes.
TRAIN_FOLDS = [HALF_FOLDS[i] for i in (0, 2, 4, 6, 8)]   # 21H1,22H1,23H1,24H1,25H1
TEST_FOLDS  = [HALF_FOLDS[i] for i in (1, 3, 5, 7)]        # 21H2,22H2,23H2,24H2

_PRE = None
_BASE = None
_FUND = None  # per-bar funding rate sums aligned to _PRE.timestamps
_FMETRIC = None  # causal funding SIGNAL metric: EWM(span=30) of _FUND (all events <= bar i)


@dataclass
class EvaluationContext:
    pre: fb.Precomputed
    config: AppConfig
    funding: list[float] | None
    funding_metric: list[float] | None


def load_context(symbol: str | None = None, config_path: str = "config.json",
                 data_start: str = "2020-10-01", data_end: str = "2025-06-01",
                 funding_end: str = "2025-06-02",
                 extra_timeframes: list[str] | None = None) -> EvaluationContext:
    """Load one independent symbol context (safe for multi-asset callers).

    ``data_end``/``funding_end`` default to the in-sample cutoff; pass a later
    date (e.g. "2026-06-01") to replay genuinely out-of-sample history."""
    cfg = load_config(config_path)
    if extra_timeframes:
        # Research hook (e.g. ["1w"] for a weekly alignment vote). ⚠️ A TF
        # missing from alignment_scale_by_tf votes at the legacy FLAT
        # alignment_scale — callers must pass explicit per-TF weights.
        cfg.trading.timeframes = list(cfg.trading.timeframes) + [
            tf for tf in extra_timeframes if tf not in cfg.trading.timeframes
        ]
    configure_cache(cfg.data_cache.ttl_seconds)
    ds = cfg.data_source
    if symbol:
        ds.exchange_symbol = symbol
    data = fetch_multi_timeframe(
        ds.exchange_symbol, cfg.trading.timeframes,
        start_date=data_start, end_date=data_end,
        warmup_periods=0, source=ds.source, market=ds.market,
    )
    for tf, df in data.items():
        print(f"  {ds.exchange_symbol} {tf}: {len(df)} rows "
              f"{df.index[0].date()} -> {df.index[-1].date()}", file=sys.stderr)
    pre = fb.precompute(data, cfg.trading.primary_timeframe, 200)
    funding = metric = None
    try:
        from llm_trading_bot.funding import fetch_funding_history, aggregate_funding_to_bars
        import pandas as pd
        fund = fetch_funding_history(ds.exchange_symbol,
                                     start_date="2020-08-01", end_date=funding_end)
        tf_hours = timeframe_hours(cfg.trading.primary_timeframe)
        funding = aggregate_funding_to_bars(
            fund, pd.DatetimeIndex(pre.timestamps), tf_hours
        )
        metric = pd.Series(funding).ewm(span=30, adjust=False).mean().tolist()
        print(f"  {ds.exchange_symbol} funding: {len(fund)} settlements loaded", file=sys.stderr)
    except Exception as e:
        print(f"  {ds.exchange_symbol} funding unavailable: {e}", file=sys.stderr)
    return EvaluationContext(pre, cfg, funding, metric)


def setup(symbol: str | None = None, extra_timeframes: list[str] | None = None):
    """Load data + funding and precompute indicators. symbol overrides the config's
    exchange_symbol (e.g. "ETH/USDT:USDT" to evaluate the same strategy on ETH)."""
    global _PRE, _BASE, _FUND, _FMETRIC
    if _PRE is not None:
        return
    ctx = load_context(symbol, extra_timeframes=extra_timeframes)
    _PRE, _BASE, _FUND, _FMETRIC = (
        ctx.pre, ctx.config, ctx.funding, ctx.funding_metric
    )


def build_config(overrides: dict) -> AppConfig:
    """Clone base config and apply a flat override dict of dotted paths."""
    cfg = _BASE.model_copy(deep=True)
    tier = cfg.trading.active_tier
    for k, v in overrides.items():
        if k == "weights":
            # normalize to sum 1.0
            tot = sum(v.values())
            cfg.scoring.weights = {kk: vv / tot for kk, vv in v.items()}
        elif k.startswith("tier."):
            setattr(cfg.trading.leverage_tiers[tier], k[5:], v)
        elif k.startswith("scoring."):
            setattr(cfg.scoring, k[8:], v)
        elif k.startswith("filters."):
            setattr(cfg.filters, k[8:], v)
        elif k.startswith("risk."):
            setattr(cfg.risk_management, k[5:], v)
        elif k.startswith("trailing."):
            setattr(cfg.trading.trailing_stop, k[9:], v)
        elif k.startswith("bt."):
            setattr(cfg.backtesting, k[3:], v)
        elif k.startswith("trading."):
            setattr(cfg.trading, k[8:], v)
        elif k.startswith("sizing."):
            setattr(cfg.position_sizing, k[7:], v)
        else:
            raise KeyError(k)
    return cfg


def evaluate(overrides: dict, folds=FOLDS, slip: float = 0.0,
             model_liquidation: bool = True, strat: dict | None = None,
             funding: bool = False, exit_granularity: str = "primary",
             fund_signal: bool = False, marginal_gate=None) -> dict:
    cfg = build_config(overrides)
    per = {}
    rets = []
    dds = []
    trades = 0
    for name, sd, ed in folds:
        r = fb.simulate(_PRE, cfg, sd, ed, slip=slip, model_liquidation=model_liquidation,
                        strat=strat, funding_by_pos=_FUND if funding else None,
                        exit_granularity=exit_granularity,
                        fund_metric=_FMETRIC if fund_signal else None,
                        marginal_gate=marginal_gate)
        per[name] = {"ret": r.return_pct, "dd": r.max_dd_pct, "tr": r.trades,
                     "wr": round(r.win_rate, 1), "pf": r.profit_factor,
                     "marginal": r.marginal_candidates,
                     "marginal_accepted": r.marginal_accepted,
                     "maker_orders": r.maker_orders,
                     "maker_touches": r.maker_touches,
                     "maker_queue_eligible": r.maker_queue_eligible,
                     "maker_fills": r.maker_fills}
        rets.append(r.return_pct)
        dds.append(r.max_dd_pct)
        trades += r.trades
    # geometric mean of growth factors (floor factor at 0.01 to avoid log(<=0))
    factors = [max(0.01, 1 + x / 100) for x in rets]
    geo = float(np.exp(np.mean(np.log(factors))))
    # full compounded across folds (chained)
    comp = 1.0
    for x in rets:
        comp *= max(0.01, 1 + x / 100)
    return {
        "geo_factor": round(geo, 4),
        "geo_pct": round((geo - 1) * 100, 2),
        "compound_x": round(comp, 3),
        "worst_fold": round(min(rets), 2),
        "mean_ret": round(float(np.mean(rets)), 2),
        "max_dd": round(max(dds), 2),
        "total_trades": trades,
        "per": per,
    }


def fmt(res: dict) -> str:
    line = (f"geo={res['geo_pct']:+7.2f}%/fold  compound={res['compound_x']:8.2f}x  "
            f"worst={res['worst_fold']:+7.1f}%  meanRet={res['mean_ret']:+7.1f}%  "
            f"maxDD={res['max_dd']:5.1f}%  trades={res['total_trades']}")
    detail = "  ".join(f"{k}:{v['ret']:+.0f}%(dd{v['dd']:.0f},t{v['tr']})" for k, v in res['per'].items())
    return line + "\n    " + detail


if __name__ == "__main__":
    setup()
    t0 = time.time()
    base = evaluate({})
    print("BASELINE (current config.json):")
    print(fmt(base))
    print(f"\n(eval took {time.time()-t0:.2f}s for {len(FOLDS)} folds)")
