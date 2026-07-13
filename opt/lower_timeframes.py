"""Research-only comparison of the shipped 4h strategy at 1h and 5m cadence.

This runner deliberately does not create live/paper configuration profiles.  It
transplants the shipped strategy parameters unchanged, changes only the primary
decision cadence and its two trend-alignment timeframes, and evaluates every
cadence on the same Binance USDT-perpetual OHLCV source.  Bitget fees, maker-entry
rules, liquidation, funding, and all risk-management settings remain unchanged.

Historical exchange candles are normalized to bar-open timestamps.  Secondary
indicators are then selected by *bar close*: only a secondary candle whose close
is no later than the primary decision bar's close is visible.  This matters for
1h/5m experiments because selecting a 4h candle merely by its open timestamp
would leak its not-yet-completed high, low, close, and volume.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from llm_trading_bot.config import AppConfig, load_config
from llm_trading_bot.data import configure_cache, fetch_multi_timeframe
from llm_trading_bot.funding import (
    aggregate_funding_to_bars,
    fetch_funding_history,
)
from opt.driver import FOLDS, TEST_FOLDS, TRAIN_FOLDS
from opt.fastbt import Precomputed, build_indicatorsets, precompute, simulate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "opt" / "lower_timeframe_results.json"
START = "2020-10-01"
TEST_START = "2021-01-01"
END = "2025-06-01"

# Keep two alignment timeframes, as in the shipped 4h model.  At lower cadence
# they are both completed higher-timeframe trend confirmations; the primary
# timeframe still supplies every category score and every target.
CADENCES: dict[str, tuple[str, ...]] = {
    "4h": ("1h", "4h", "1d"),
    "1h": ("1h", "4h", "1d"),
    "5m": ("5m", "1h", "4h"),
}

_TF_RE = re.compile(r"^(\d+)([mhdw])$")


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    """Return the wall-clock duration of a simple exchange timeframe."""
    match = _TF_RE.fullmatch(timeframe)
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    count, unit = int(match.group(1)), match.group(2)
    keyword = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[unit]
    return pd.Timedelta(**{keyword: count})


def normalize_to_bar_open(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Normalize either open- or close-stamped archive rows to bar-open stamps.

    Binance archive rows are currently indexed by ``time_close`` (for example
    00:04:59.999 for a 5m bar).  Flooring is safe for UTC-aligned exchange
    intervals and is a no-op for Bitget's already open-stamped rows.
    """
    out = df.copy()
    delta = timeframe_delta(timeframe)
    out.index = out.index.floor(delta)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def causal_precompute(
    data_by_tf: dict[str, pd.DataFrame], primary_tf: str, warmup: int = 200,
    exit_subframe: pd.DataFrame | None = None,
) -> Precomputed:
    """Precompute indicators with close-aware, leakage-free TF alignment."""
    from opt.fastbt import precompute

    # Reuse the canonical precompute for the primary indicators and the special
    # 4h -> 1h exit sub-bars, then replace its open-stamp secondary alignment.
    pre = precompute(data_by_tf, primary_tf, warmup)
    primary_index = pd.DatetimeIndex(pre.timestamps)
    primary_delta = timeframe_delta(primary_tf)

    secondary: dict[str, tuple[pd.DatetimeIndex, list]] = {}
    for tf, frame in data_by_tf.items():
        if tf != primary_tf:
            secondary[tf] = (
                pd.DatetimeIndex(frame.index), build_indicatorsets(frame, tf)
            )

    aligned: list[dict] = []
    for primary_open in primary_index:
        decision_close = primary_open + primary_delta
        row: dict = {}
        for tf, (index, indicators) in secondary.items():
            # A secondary bar is usable iff secondary_open + duration <=
            # primary_close.  Search for the last open satisfying that rule.
            last_open = decision_close - timeframe_delta(tf)
            pos = index.searchsorted(last_open, side="right") - 1
            if pos >= 0 and indicators[pos] is not None and pos + 1 >= 50:
                row[tf] = indicators[pos]
        aligned.append(row)

    subbars = pre.subbars
    if exit_subframe is not None:
        sub_index = pd.DatetimeIndex(exit_subframe.index)
        highs = exit_subframe["High"].to_numpy()
        lows = exit_subframe["Low"].to_numpy()
        closes = exit_subframe["Close"].to_numpy()
        subbars = []
        for pos, primary_open in enumerate(primary_index):
            end = primary_open + primary_delta
            left = sub_index.searchsorted(primary_open, side="left")
            right = sub_index.searchsorted(end, side="left")
            rows = [
                (float(highs[i]), float(lows[i]), float(closes[i]))
                for i in range(left, right)
            ]
            primary = pre.primary[pos]
            if rows and primary is not None and primary.high and primary.low:
                sub_high = max(row[0] for row in rows)
                sub_low = min(row[1] for row in rows)
                if (abs(sub_high - primary.high) / primary.high > 0.005
                        or abs(sub_low - primary.low) / primary.low > 0.005):
                    rows = []
            subbars.append(rows)

    return Precomputed(
        timestamps=pre.timestamps,
        primary=pre.primary,
        sec_by_bar=aligned,
        warmup=pre.warmup,
        subbars=subbars,
    )


def _config_for(base: AppConfig, primary_tf: str) -> AppConfig:
    config = base.model_copy(deep=True)
    config.trading.primary_timeframe = primary_tf
    config.trading.timeframes = list(CADENCES[primary_tf])
    return config


def _result_row(result) -> dict:
    row = asdict(result)
    # Keep the artifact readable and stable; final_balance duplicates return_pct.
    return {
        "return_pct": row["return_pct"],
        "growth_x": max(0.0, 1 + row["return_pct"] / 100),
        "max_dd_pct": row["max_dd_pct"],
        "trades": row["trades"],
        "win_rate": row["win_rate"],
        "profit_factor": row["profit_factor"],
        "sharpe": row["sharpe"],
        "maker_orders": row["maker_orders"],
        "maker_touches": row["maker_touches"],
        "maker_fills": row["maker_fills"],
    }


def _evaluate_folds(pre, config, funding, folds, exit_granularity: str) -> dict:
    per: dict[str, dict] = {}
    factors: list[float] = []
    for label, start, end in folds:
        result = simulate(
            pre, config, start, end,
            slip=0.0002,
            model_liquidation=True,
            funding_by_pos=funding,
            exit_granularity=exit_granularity,
        )
        row = _result_row(result)
        per[label] = row
        factors.append(max(0.01, row["growth_x"]))
    compound = math.prod(factors)
    return {
        "compound_x": compound,
        "geo_growth_x": math.exp(sum(math.log(value) for value in factors) / len(factors)),
        "worst_fold_return_pct": min(row["return_pct"] for row in per.values()),
        "worst_fold_max_dd_pct": max(row["max_dd_pct"] for row in per.values()),
        "trades": sum(row["trades"] for row in per.values()),
        "per": per,
    }


def _evaluate_cadence(
    all_data: dict[str, pd.DataFrame], base: AppConfig, funding_series: pd.Series,
    primary_tf: str,
) -> dict:
    timeframes = CADENCES[primary_tf]
    data = {tf: all_data[tf] for tf in timeframes}
    config = _config_for(base, primary_tf)
    exit_subframe = all_data["5m"] if primary_tf == "1h" else None
    pre = causal_precompute(
        data, primary_tf, warmup=200, exit_subframe=exit_subframe
    )
    funding = aggregate_funding_to_bars(
        funding_series, pd.DatetimeIndex(pre.timestamps),
        timeframe_delta(primary_tf) / pd.Timedelta(hours=1),
    )
    exit_granularity = "sub" if primary_tf in ("4h", "1h") else "primary"

    continuous = simulate(
        pre, config, TEST_START, END,
        slip=0.0002,
        model_liquidation=True,
        funding_by_pos=funding,
        exit_granularity=exit_granularity,
    )
    result = {
        "primary_timeframe": primary_tf,
        "timeframes": list(timeframes),
        "exit_resolution": (
            "1h sub-bars; trailing ratchets once per 4h" if primary_tf == "4h"
            else "5m sub-bars; trailing ratchets once per 1h" if primary_tf == "1h"
            else "5m adverse-first OHLC"
        ),
        "bars": len(pre.timestamps),
        "continuous": _result_row(continuous),
        "annual_reset": _evaluate_folds(
            pre, config, funding, FOLDS, exit_granularity
        ),
        "train": _evaluate_folds(
            pre, config, funding, TRAIN_FOLDS, exit_granularity
        ),
        "held_out_test": _evaluate_folds(
            pre, config, funding, TEST_FOLDS, exit_granularity
        ),
    }

    if primary_tf != "4h":
        def variant(candidate, *, slip=0.0002, funding_by_pos=funding):
            return _result_row(simulate(
                pre, candidate, TEST_START, END,
                slip=slip,
                model_liquidation=True,
                funding_by_pos=funding_by_pos,
                exit_granularity=exit_granularity,
            ))

        no_funding = variant(config, funding_by_pos=None)
        no_slippage = variant(config, slip=0.0)
        zero_costs = config.model_copy(deep=True)
        zero_costs.fees.maker = 0.0
        zero_costs.fees.taker = 0.0
        trailing_off = config.model_copy(deep=True)
        trailing_off.trading.trailing_stop.enabled = False
        trailing_off.backtesting.enable_trailing_stops = False
        conservative = config.model_copy(deep=True)
        conservative.trading.leverage_tiers[
            conservative.trading.active_tier
        ].leverage = 12

        result["diagnostics"] = {
            "no_funding": no_funding,
            "no_market_slippage": no_slippage,
            "zero_fees_slippage_funding": variant(
                zero_costs, slip=0.0, funding_by_pos=None
            ),
            "trailing_disabled": variant(trailing_off),
            "conservative_12x": variant(conservative),
        }
        if primary_tf == "1h":
            result["diagnostics"]["conservative_12x_annual_reset"] = (
                _evaluate_folds(
                    pre, conservative, funding, FOLDS, exit_granularity
                )
            )
            result["diagnostics"]["conservative_12x_held_out_test"] = (
                _evaluate_folds(
                    pre, conservative, funding, TEST_FOLDS, exit_granularity
                )
            )
    return result


def _bitget_alignment_audit(base: AppConfig, funding_series: pd.Series) -> dict:
    """Quantify the existing open-stamp secondary alignment on native data."""
    configure_cache(0)
    ds = base.data_source
    data = fetch_multi_timeframe(
        symbol=ds.exchange_symbol,
        timeframes=base.trading.timeframes,
        start_date=START,
        end_date=END,
        warmup_periods=0,
        source="bitget",
        market="futures",
    )
    rows = {}
    for label, builder in (
        ("legacy_open_stamp_alignment", precompute),
        ("causal_completed_alignment", causal_precompute),
    ):
        pre = builder(data, "4h", 200)
        funding = aggregate_funding_to_bars(
            funding_series, pd.DatetimeIndex(pre.timestamps), 4
        )
        rows[label] = _result_row(simulate(
            pre, base, TEST_START, END,
            slip=0.0002,
            model_liquidation=True,
            funding_by_pos=funding,
            exit_granularity="sub",
        ))
    return {
        "reason": (
            "Bitget candles are bar-open stamped. The legacy fast/full backtests "
            "select secondary rows with secondary_open <= primary_open, exposing "
            "higher-timeframe OHLCV before those bars complete."
        ),
        "rows": rows,
    }


def _load_data(symbol: str) -> dict[str, pd.DataFrame]:
    configure_cache(0)
    raw = fetch_multi_timeframe(
        symbol=symbol,
        timeframes=["5m", "1h", "4h", "1d"],
        start_date=START,
        end_date=END,
        warmup_periods=0,
        source="binance",
        market="futures",
    )
    return {tf: normalize_to_bar_open(frame, tf) for tf, frame in raw.items()}


def _print_summary(results: dict) -> None:
    print("\nUnchanged shipped parameters, common Binance futures data")
    print("(2bps market slippage + funding + liquidation + maker entry)")
    for tf, row in results["cadences"].items():
        cont = row["continuous"]
        test = row["held_out_test"]
        annual = row["annual_reset"]
        print(
            f"{tf:>2}  continuous {cont['growth_x']:>12,.3f}x  "
            f"DD {cont['max_dd_pct']:>6.2f}%  trades {cont['trades']:>7,d}  "
            f"held-out {test['compound_x']:>10,.3f}x  "
            f"worst year {annual['worst_fold_return_pct']:>+8.2f}%"
        )


def run(cadences: list[str], output: Path, symbol: str = "BTC/USDT") -> dict:
    unknown = set(cadences) - set(CADENCES)
    if unknown:
        raise ValueError(f"Unknown cadence(s): {sorted(unknown)}")
    base = load_config(ROOT / "config.json")
    data = _load_data(symbol)
    funding = fetch_funding_history(
        "BTC/USDT:USDT", start_date=START, end_date="2025-06-02"
    )
    artifact = {
        "experiment": "lower-timeframe static transplant",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": symbol,
        "period": {"warmup_start": START, "test_start": TEST_START, "end": END},
        "market_data": "Binance USDT perpetual futures (all cadences)",
        "execution": {
            "entry": "maker at decision close, valid for next primary bar",
            "market_exit_slippage_per_side": 0.0002,
            "fees": {"maker": base.fees.maker, "taker": base.fees.taker},
            "funding": "Binance actual settlements",
            "liquidation": True,
        },
        "alignment": (
            "bar-open normalized; secondary OHLCV visible only after the secondary "
            "bar has completed"
        ),
        "parameter_policy": "all shipped numeric strategy/risk parameters unchanged",
        "cadences": {},
    }
    for cadence in cadences:
        started = time.monotonic()
        print(f"\nEvaluating {cadence}...", file=sys.stderr)
        artifact["cadences"][cadence] = _evaluate_cadence(
            data, base, funding, cadence
        )
        artifact["cadences"][cadence]["elapsed_seconds"] = time.monotonic() - started

    if symbol == "BTC/USDT":
        artifact["bitget_alignment_audit"] = _bitget_alignment_audit(base, funding)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    _print_summary(artifact)
    print(f"\nWrote {output}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cadences", nargs="+", choices=tuple(CADENCES),
        default=list(CADENCES),
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.cadences, args.output, args.symbol)


if __name__ == "__main__":
    main()
