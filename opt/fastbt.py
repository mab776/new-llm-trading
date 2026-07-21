"""
Fast backtest harness for the optimization loop.

Key idea: every indicator is *causal* (ewm/rolling → value at bar i depends only
on bars <= i), so computing an indicator on the full series once and reading row i
is numerically identical to the engine recomputing it on the slice [:i+1] each bar.

We therefore precompute one IndicatorSet per bar ONCE (the expensive part), then run
a lightweight trade simulator per config that reuses the REAL project functions
(compute_composite_score, calculate_targets, apply_pre_trade_filters, Portfolio).
Only the outer loop is re-implemented — and it mirrors backtesting.BacktestEngine.run
step for step. validate.py checks it reproduces the engine exactly.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openwebui_filter import (
    compute_ema, compute_sma, compute_rsi, compute_macd, compute_atr, compute_adx,
    compute_stochastic, compute_bollinger_bands, compute_obv, compute_vwap,
    compute_williams_r, compute_cci, compute_roc,
    calc_trend_score, calc_momentum_score, calc_volume_score, calc_sr_score, calc_risk_score,
)
from llm_trading_bot.scoring import (
    Direction, IndicatorSet, SignalStrength, CategoryScore,
    calculate_targets, apply_pre_trade_filters, compute_composite_score,
    detect_market_regime, score_trend,
)
from llm_trading_bot.portfolio import Portfolio
from llm_trading_bot.entry import PendingEntry, maker_limit_touched
from llm_trading_bot.exposure import (
    anti_martingale_multiplier, cap_risk_pct, update_outcome_streak,
)
from llm_trading_bot.timeframes import decision_close, last_usable_open, timeframe_hours


# ──────────────────────────────────────────────────────────────────────
# Vectorised indicator precompute
# ──────────────────────────────────────────────────────────────────────

def build_indicatorsets(df: pd.DataFrame, tf: str) -> list[IndicatorSet | None]:
    """Return one IndicatorSet per row of df (None for the first <50 rows).

    Reproduces scoring.calculate_indicators exactly, but vectorised over the
    whole series so it is computed once instead of once-per-bar.
    """
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    n = len(df)

    ema9 = compute_ema(c, 9).to_numpy()
    ema21 = compute_ema(c, 21).to_numpy()
    ema50 = compute_ema(c, 50).to_numpy()
    ema200 = compute_ema(c, 200).to_numpy()
    sma200 = compute_sma(c, 200).to_numpy()
    adx_s, pdi_s, mdi_s = compute_adx(h, l, c)
    adx = adx_s.to_numpy(); pdi = pdi_s.to_numpy(); mdi = mdi_s.to_numpy()
    macd_l, macd_s, macd_h = compute_macd(c)
    macd_l = macd_l.to_numpy(); macd_s = macd_s.to_numpy(); macd_h = macd_h.to_numpy()
    rsi = compute_rsi(c, 14).to_numpy()
    sk_s, sd_s = compute_stochastic(h, l, c)
    sk = sk_s.to_numpy(); sd = sd_s.to_numpy()
    cci = compute_cci(h, l, c, 20).to_numpy()
    willr = compute_williams_r(h, l, c).to_numpy()
    roc = compute_roc(c, 10).to_numpy()
    vol = v.to_numpy()
    vol_sma = compute_sma(v, 20).to_numpy()
    obv_s = compute_obv(c, v)
    obv = obv_s.to_numpy()
    obv_sma = compute_sma(obv_s, 20).to_numpy()
    vwap = compute_vwap(h, l, c, v).to_numpy()
    atr = compute_atr(h, l, c, 14).to_numpy()
    bb_up_s, bb_mid_s, bb_low_s = compute_bollinger_bands(c)
    bb_up = bb_up_s.to_numpy(); bb_mid = bb_mid_s.to_numpy(); bb_low = bb_low_s.to_numpy()

    close = c.to_numpy(); opn = o.to_numpy(); high = h.to_numpy(); low = l.to_numpy()

    # prior-candle pivots (vectorised): computed from row i-1
    ph = np.roll(high, 1); pl = np.roll(low, 1); pc = np.roll(close, 1)
    pivot = (ph + pl + pc) / 3.0
    s1 = 2 * pivot - ph
    s2 = pivot - (ph - pl)
    r1 = 2 * pivot - pl
    r2 = pivot + (ph - pl)

    def isna(x):
        return x != x  # NaN check

    out: list[IndicatorSet | None] = []
    for i in range(n):
        if i < 50:  # calculate_indicators raises below 50 candles
            out.append(None)
            continue
        ind = IndicatorSet(timeframe=tf)
        ind.close = float(close[i]); ind.open = float(opn[i])
        ind.high = float(high[i]); ind.low = float(low[i])
        if i >= 1 and close[i-1] != 0:
            ind.change_pct = float((close[i] - close[i-1]) / close[i-1] * 100)
        ind.ema_9 = float(ema9[i]); ind.ema_21 = float(ema21[i]); ind.ema_50 = float(ema50[i])
        if i >= 199:  # len>=200
            ind.ema_200 = float(ema200[i]); ind.sma_200 = float(sma200[i])
        ind.adx = None if isna(adx[i]) else float(adx[i])
        ind.plus_di = None if isna(pdi[i]) else float(pdi[i])
        ind.minus_di = None if isna(mdi[i]) else float(mdi[i])
        ind.macd_line = float(macd_l[i]); ind.macd_signal = float(macd_s[i]); ind.macd_histogram = float(macd_h[i])
        ind.rsi_14 = float(rsi[i]) if not isna(rsi[i]) else None
        ind.stoch_k = None if isna(sk[i]) else float(sk[i])
        ind.stoch_d = None if isna(sd[i]) else float(sd[i])
        ind.cci_20 = None if isna(cci[i]) else float(cci[i])
        ind.williams_r = float(willr[i]) if not isna(willr[i]) else None
        ind.roc_10 = None if isna(roc[i]) else float(roc[i])
        ind.volume = float(vol[i])
        ind.volume_sma_20 = None if isna(vol_sma[i]) else float(vol_sma[i])
        if ind.volume_sma_20 and ind.volume_sma_20 > 0:
            ind.volume_ratio = ind.volume / ind.volume_sma_20
        ind.obv = float(obv[i])
        ind.obv_sma_20 = None if isna(obv_sma[i]) else float(obv_sma[i])
        ind.vwap = None if isna(vwap[i]) else float(vwap[i])
        ind.atr_14 = float(atr[i])
        ind.atr_pct = float(ind.atr_14 / ind.close * 100) if ind.close else None
        ind.bb_upper = None if isna(bb_up[i]) else float(bb_up[i])
        ind.bb_middle = None if isna(bb_mid[i]) else float(bb_mid[i])
        ind.bb_lower = None if isna(bb_low[i]) else float(bb_low[i])
        if ind.bb_upper and ind.bb_lower and ind.bb_middle and ind.bb_middle > 0:
            ind.bb_width = (ind.bb_upper - ind.bb_lower) / ind.bb_middle * 100
        if ind.bb_upper and ind.bb_lower and (ind.bb_upper - ind.bb_lower) > 0:
            raw = (ind.close - ind.bb_lower) / (ind.bb_upper - ind.bb_lower)
            ind.bb_position = max(0.0, min(1.0, raw))
        if i >= 1:
            ind.pivot = float(pivot[i]); ind.support_1 = float(s1[i]); ind.support_2 = float(s2[i])
            ind.resistance_1 = float(r1[i]); ind.resistance_2 = float(r2[i])
            price = ind.close
            supports = [s for s in (ind.support_1, ind.support_2) if s and s < price]
            resistances = [r for r in (ind.resistance_1, ind.resistance_2) if r and r > price]
            ind.nearest_support = max(supports) if supports else ind.support_2
            ind.nearest_resistance = min(resistances) if resistances else ind.resistance_2
        out.append(ind)
    return out


@dataclass
class Precomputed:
    timestamps: list          # 4h bar timestamps (all rows)
    primary: list             # IndicatorSet|None per 4h row
    sec_by_bar: list          # dict{tf:IndicatorSet} per 4h row (secondary tf, asof)
    warmup: int
    subbars: list | None = None  # per 4h row: list of (high, low, close) 1h sub-bars
    #                              in time order (for fine-grained exit replay)


def precompute(data_by_tf: dict[str, pd.DataFrame], primary_tf: str, warmup: int) -> Precomputed:
    primary_df = data_by_tf[primary_tf]
    prim = build_indicatorsets(primary_df, primary_tf)

    # Secondary timeframes: build their IndicatorSets, then map only COMPLETED
    # rows to each primary decision close. All indexes are candle opens.
    sec_inds = {}
    for tf, df in data_by_tf.items():
        if tf == primary_tf:
            continue
        sec_inds[tf] = (df.index, build_indicatorsets(df, tf))

    sec_by_bar = []
    for ts in primary_df.index:
        d = {}
        close_time = decision_close(ts, primary_tf)
        for tf, (idx, inds) in sec_inds.items():
            cutoff = last_usable_open(close_time, tf)
            pos = idx.searchsorted(cutoff, side="right") - 1
            if pos >= 0 and inds[pos] is not None and (pos + 1) >= 50:
                d[tf] = inds[pos]
        sec_by_bar.append(d)

    # 1h sub-bars per primary bar (for fine-grained exit replay): bar at t collects
    # 1h rows with t <= ts < t + 4h, in time order. Sub-bar sets whose extremes
    # disagree with the 4h bar by >0.5% are data holes (e.g. Bitget's 1h perp history
    # is placeholder junk before 2021-01-02) — dropped so the sim falls back to 4h.
    subbars = None
    if primary_tf == "4h" and "1h" in data_by_tf:
        df1 = data_by_tf["1h"]
        h1 = df1["High"].to_numpy(); l1 = df1["Low"].to_numpy(); c1 = df1["Close"].to_numpy()
        idx1 = df1.index
        subbars = []
        delta = pd.Timedelta(hours=4)
        for k, ts in enumerate(primary_df.index):
            a = idx1.searchsorted(ts, side="left")
            b = idx1.searchsorted(ts + delta, side="left")
            subs = [(float(h1[j]), float(l1[j]), float(c1[j])) for j in range(a, b)]
            p = prim[k]
            if subs and p is not None and p.high and p.low:
                s_hi = max(s[0] for s in subs); s_lo = min(s[1] for s in subs)
                if abs(s_hi - p.high) / p.high > 0.005 or abs(s_lo - p.low) / p.low > 0.005:
                    subs = []  # corrupt/incomplete 1h coverage -> 4h fallback
            subbars.append(subs)

    return Precomputed(list(primary_df.index), prim, sec_by_bar, warmup, subbars)


# ──────────────────────────────────────────────────────────────────────
# Trade simulator (mirrors BacktestEngine.run)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    return_pct: float
    final_balance: float
    trades: int
    win_rate: float
    profit_factor: float
    max_dd_pct: float
    sharpe: float
    marginal_candidates: int = 0
    marginal_accepted: int = 0
    maker_orders: int = 0
    maker_touches: int = 0
    maker_queue_eligible: int = 0
    maker_fills: int = 0


def maker_queue_eligible(direction: str, limit_price: float, bar_high: float,
                         bar_low: float, penetration_bps: float = 0.0) -> bool:
    """Return whether price traded far enough through a maker limit to model a fill.

    ``penetration_bps=0`` is exactly the canonical touched-limit rule. Positive
    values are a conservative queue-depth proxy: a long bid must trade below the
    limit and a short offer above it before the order becomes fill-eligible.
    """
    if penetration_bps < 0:
        raise ValueError("maker_queue_penetration_bps must be non-negative")
    distance = penetration_bps / 10_000
    adjusted_limit = (limit_price * (1 - distance) if direction == "LONG"
                      else limit_price * (1 + distance))
    return maker_limit_touched(direction, adjusted_limit, bar_high, bar_low)


def deterministic_maker_fill(probability: float, seed: int, key: str) -> bool:
    """Reproducibly accept an eligible maker order with ``probability``.

    Hashing the immutable order identity avoids evaluation-order and fold-boundary
    effects from a mutable PRNG. Probability 1.0 preserves the shipped baseline.
    """
    if not 0 <= probability <= 1:
        raise ValueError("maker_fill_probability must be between 0 and 1")
    if probability in (0, 1):
        return bool(probability)
    digest = hashlib.blake2b(
        f"{seed}|{key}".encode("utf-8"), digest_size=8
    ).digest()
    sample = int.from_bytes(digest, "big") / 2**64
    return sample < probability


DEFAULT_STRAT = {
    # All defaults preserve the current engine behavior exactly.
    # Entry fill model (backlog #4). "taker": current behavior — market order at the
    # decision bar's close, paying taker fee + `slip`. "maker": rest a limit at that
    # close and fill it only if the NEXT bar trades back to it (LONG: next-bar low <=
    # limit; SHORT: next-bar high >= limit), paying the maker fee with NO slip. Good-
    # for-one-bar: an unfilled limit is cancelled (price ran away -> missed trade).
    # A fill is immediately exposed to adverse-first exits from its touched bar/sub-bar.
    "entry_mode": "taker",        # "taker" | "maker"
    # Maker queue/fill sensitivity controls. These are research-only stressors;
    # defaults preserve the canonical touched-limit behavior exactly.
    "maker_queue_penetration_bps": 0.0,
    "maker_fill_probability": 1.0,
    "maker_fill_seed": 0,
    # Research-only causal regime overlays. Keys are MarketRegime string values;
    # empty mappings preserve the static strategy.
    "regime_threshold_mults": {},
    "regime_leverage_mults": {},
    "regime_trailing_activation_mults": {},
    "regime_trailing_callback_mults": {},
    "vol_target_lev": None,       # e.g. 40.0 -> lev_eff = min(lev, vol_target_lev / atr_pct)
    "trail_mode": "pct",          # "pct" (of entry) | "atr" (multiples of entry ATR)
    "trail_act_atr": 0.5,          # activation distance in ATRs (trail_mode="atr")
    "trail_cb_atr": 0.6,           # callback distance in ATRs (trail_mode="atr")
    "conviction_sizing": None,    # e.g. 1.0 -> risk_pct scaled by (abs_score/strong)^1 capped [0.5, 1.5]
    # Anti-martingale sizing: completed winning trades increase the next trade's
    # margin risk; completed losing trades decrease it.  The signed consecutive
    # outcome streak is causal and resets to the opposite sign when the outcome
    # changes.  step=0 preserves the current strategy exactly.
    "anti_martingale_step": 0.0,
    "anti_martingale_min": 0.5,
    "anti_martingale_max": 1.5,
    "opposite_exit": None,        # e.g. 25.0 -> close when opposite-direction |score| >= this
    # Cross-asset rotation research knobs (opt-in, probe_rotation.py; multi-asset
    # simulate_multi only, NOT wired into live). When a STRONG entry is cap-squeezed
    # below rotate_squeeze_frac of its pre-cap risk, the weakest OTHER symbol's open
    # position (signed support = raw_score for LONG, -raw_score for SHORT) is closed
    # (reason "rotation") iff support <= rotate_weak_support AND the newcomer's
    # |score| - support >= rotate_min_gap; exposure caps then recompute on the freed
    # margin. Both thresholds None (default) = code path untouched.
    "rotate_weak_support": None,
    "rotate_min_gap": None,
    "rotate_squeeze_frac": 0.5,
    # Signal-decay research knobs (opt-in, probe_decay.py; NOT wired into live).
    # Signed score = raw_score for LONG, -raw_score for SHORT.
    "entry_require_rising": None,  # int K: block entry unless signed score non-decreasing over last K transitions
    "entry_slope_min": None,       # block entry if 1-bar signed-score slope < this (0 == require_rising K=1)
    "entry_slope_max": None,       # block entry if slope > this (contrarian: 0 -> only decaying entries)
    "decay_exit_bars": None,       # int N: exit when signed score fell N consecutive bars...
    "decay_exit_floor": None,      # ...AND is below this floor (both must be set)
    "short_threshold_mult": 1.0,  # >1 = stricter shorts (asymmetry)
    "long_threshold_mult": 1.0,
    "max_positions": 1,           # >1 allows pyramiding same-direction entries
    # Ex-ante exposure controls; None preserves the uncapped behavior.
    "global_max_positions": None, # open positions + resting maker entries
    "global_max_margin_pct": None,# committed isolated margin / current balance
    "global_max_notional_pct": None, # committed entry notional / current balance
    "portfolio_risk_multiplier": 1.0, # scales every new trade before caps
    "marginal_size_frac": 1.0,    # size fraction for MARGINAL (vs STRONG) entries
    # Research-only exchange micro-structure modeling (multi_asset simulate_multi):
    # contract minimum quantity + size step per symbol label, and the policy when a
    # computed size falls below the minimum: "skip" (fail closed, mirrors live) or
    # "floor" (bump to the exchange minimum — inflates risk at small balances).
    # None disables the modeling entirely (exact current behavior).
    "min_qty": None,              # e.g. {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
    "size_step": None,            # e.g. {"BTC": 0.0001, "ETH": 0.01, "SOL": 0.1}
    "min_size_policy": "skip",    # "skip" | "floor"
    # Conditional cap-overshoot provision (opt-in, probe_overshoot.py; NOT live).
    # When a skip-policy entry quantizes below the exchange minimum (the live
    # MIN_SIZE_SKIP) but |score| >= min_size_overshoot_score, bump it to the
    # minimum anyway — iff total committed exposure stays within the global
    # margin/notional caps stretched by (1 + min_size_overshoot). Both None
    # (default) = exact fail-closed skip behavior.
    "min_size_overshoot": None,
    "min_size_overshoot_score": None,
    "dd_throttle": None,          # e.g. 0.10 -> while balance DD >= 10%, pyramiding pauses
    "dd_throttle_slots": 1,       # slots allowed while throttled (1 = full pause of pyramiding)
    "dd_throttle_risk": 1.0,      # risk multiplier applied while throttled (e.g. 0.5)
    # Funding-as-signal (backlog #1). fund_metric[i] is a CAUSAL funding metric (EWM of the
    # per-bar funding-rate sum, all events <= bar i — available at the bar's close). Extreme
    # positive funding = crowded longs; extreme negative = crowded shorts. EDA (opt/eda_funding*)
    # showed the effect is trend-confounded: high funding is fine in an uptrend but strongly
    # bearish in a downtrend, so gating is trend-conditioned by default.
    "funding_block_long": None,   # e.g. 1.5e-4 -> skip LONG entries when fund_metric >= this
    "funding_block_short": None,  # e.g. 0.0    -> skip SHORT entries when fund_metric <= this
    "funding_trend_gate": True,   # only block LONG when below ema200 (downtrend) and only block
    #                               SHORT when above ema200 (uptrend); False = block regardless.
    # Funding SHORT-boost: crowded longs in a downtrend (fund_metric >= funding_boost_thr AND
    # below ema200) are the one cell with a real fade edge (EDA: -1.46%/30bar). Ease the SHORT
    # entry thresholds there by this multiplier (<1 = more shorts) to capture it.
    "funding_short_boost": None,  # e.g. 0.7 -> eff short thresholds x0.7 in that regime
    "funding_boost_thr": 1.5e-4,  # funding level that counts as "crowded longs"
    # Funding LONG-boost: very low/negative funding = crowded shorts / capitulation, the
    # strongest raw EDA cell (+2.95%/30bar, robust in both trends). Ease LONG thresholds when
    # fund_metric <= funding_long_thr. funding_long_trend_gate=True restricts to uptrends.
    "funding_long_boost": None,   # e.g. 0.7 -> eff long thresholds x0.7 when funding very low
    "funding_long_thr": 0.0,      # funding level that counts as "crowded shorts"
    "funding_long_trend_gate": False,
    # Multi-timeframe alignment shape (opt-in, probe_alignment.py; NOT wired into
    # live). "discrete" = legacy flat ±alignment_scale sign vote per secondary TF.
    # "continuous" = alignment_scale * tanh(tf_trend / alignment_k) — smooth, kills
    # the ±5 threshold-cliff sensitivity. Defaults reproduce the engine exactly.
    "alignment_mode": "discrete",  # "discrete" | "continuous"
    "alignment_scale": 5.0,        # max per-secondary-TF alignment contribution
    "alignment_k": 30.0,           # continuous tanh smoothing scale (trend-score units)
    "alignment_scale_by_tf": None, # e.g. {"1h": 3, "1d": 8}: per-TF discrete weight
    #                                override. None ⇒ defer to the CONFIG's
    #                                scoring.alignment_scale_by_tf (itself None ⇒
    #                                flat alignment_scale) — keeps engine parity.
    # 1d-TREND OVERLAY (opt-in, probe_daily_trend.py; NOT wired into live). A
    # dedicated daily-regime term added ON TOP of the composite score (beyond the
    # weak ±5 daily alignment vote). beta=0 ⇒ off ⇒ engine-identical.
    "daily_trend_beta": 0.0,       # points added at full daily trend (0 = off)
    "daily_trend_source": "score", # "score"|"ema200"|"ema_stack"|"adx_di" (daily metric)
    "daily_trend_shape": "tanh",   # "sign"|"linear"|"tanh" (magnitude → contribution)
    "daily_trend_k": 40.0,         # saturation scale for linear/tanh (daily-metric units)
    "daily_trend_deadband": 0.0,   # |daily metric| below this contributes 0
    "daily_trend_replace_align": False,  # True ⇒ drop 1d's ±5 alignment vote (no double-count)
}


def _daily_trend_value(dind, source: str, points=None):
    """Derive a daily-regime metric in [-100, 100] from the 1d IndicatorSet, or None."""
    if dind is None:
        return None
    c = dind.close
    if source == "score":
        return score_trend(dind, points).raw_score
    if source == "ema200":
        e = dind.ema_200
        if not e or not c:
            return None
        return max(-100.0, min(100.0, 1000.0 * (c / e - 1.0)))   # ±10% → ±100
    if source == "ema_stack":
        es = (dind.ema_9, dind.ema_21, dind.ema_50, dind.ema_200)
        if any(x is None for x in es):
            return None
        v = sum(33.34 if a > b else -33.34 for a, b in zip(es, es[1:]))
        return max(-100.0, min(100.0, v))
    if source == "adx_di":
        adx, p, m = dind.adx, dind.plus_di, dind.minus_di
        if adx is None or p is None or m is None:
            return None
        return (1.0 if p > m else -1.0) * min(adx / 40.0, 1.0) * 100.0
    return None


def apply_daily_overlay(result, dind, st, points=None) -> None:
    """Add a dedicated daily-trend term on top of ``result.raw_score`` (in place),
    re-deriving direction. beta=0 or missing daily data ⇒ no-op (engine-identical)."""
    beta = st.get("daily_trend_beta", 0.0)
    if not beta:
        return
    d = _daily_trend_value(dind, st.get("daily_trend_source", "score"), points)
    if d is None or abs(d) < st.get("daily_trend_deadband", 0.0):
        return
    shape = st.get("daily_trend_shape", "tanh")
    k = st.get("daily_trend_k", 40.0) or 1.0
    if shape == "sign":
        g = 1.0 if d > 0 else -1.0 if d < 0 else 0.0
    elif shape == "linear":
        g = max(-1.0, min(1.0, d / k))
    else:  # tanh
        g = float(np.tanh(d / k))
    new = max(-100.0, min(100.0, result.raw_score + beta * g))
    result.raw_score = round(new, 2)
    result.direction = (Direction.BULLISH if new > 10 else
                        Direction.BEARISH if new < -10 else Direction.NEUTRAL)


def simulate(pre: Precomputed, config, start_date: str, end_date: str,
             slip: float = 0.0, model_liquidation: bool = True,
             maintenance_margin: float = 0.005, strat: dict | None = None,
             funding_by_pos: "list[float] | None" = None,
             exit_granularity: str = "primary",
             fund_metric: "list[float] | None" = None,
             marginal_gate=None) -> Result:
    """slip: per-side price slippage fraction applied to market fills (entry, SL, time/EOB).
    model_liquidation: if True, a stop placed beyond the isolated-margin liquidation
    distance is capped at the liquidation price (position wipes ~full margin first).
    strat: strategy-variant flags (see DEFAULT_STRAT); defaults reproduce the engine.
    marginal_gate: optional ``(timestamp, scoring_result, targets) -> bool`` callback,
    invoked only when a MARGINAL signal has passed filters and has an available entry
    slot.  ``None`` preserves the engine's current auto-trade behavior.
    """
    st = dict(DEFAULT_STRAT)
    # Config-backed strategy features (engine parity): explicit strat overrides win.
    ps_cfg = config.position_sizing
    st["max_positions"] = getattr(ps_cfg, "max_positions", 1)
    conv = getattr(ps_cfg, "conviction_exponent", 0.0)
    st["conviction_sizing"] = conv if conv > 0 else None
    st["anti_martingale_step"] = getattr(ps_cfg, "anti_martingale_step", 0.0)
    st["anti_martingale_min"] = getattr(ps_cfg, "anti_martingale_min", 0.7)
    st["anti_martingale_max"] = getattr(ps_cfg, "anti_martingale_max", 1.1)
    st["portfolio_risk_multiplier"] = getattr(ps_cfg, "portfolio_risk_multiplier", 1.0)
    gslots = getattr(ps_cfg, "global_max_positions", 0)
    st["global_max_positions"] = gslots if gslots > 0 else None
    gmargin = getattr(ps_cfg, "global_max_margin_pct", 0.0)
    st["global_max_margin_pct"] = gmargin if gmargin > 0 else None
    gnotional = getattr(ps_cfg, "global_max_notional_pct", 0.0)
    st["global_max_notional_pct"] = gnotional if gnotional > 0 else None
    opp = getattr(config.risk_management, "opposite_exit_threshold", 0.0)
    st["opposite_exit"] = opp if opp > 0 else None
    ddt = getattr(config.risk_management, "dd_throttle_threshold", 0.0)
    st["dd_throttle"] = ddt if ddt > 0 else None
    st["dd_throttle_slots"] = getattr(config.risk_management, "dd_throttle_slots", 1)
    st["dd_throttle_risk"] = getattr(config.risk_management, "dd_throttle_risk", 0.5)
    st["entry_mode"] = getattr(config.trading, "entry_mode", "taker")
    if strat:
        st.update(strat)
    if st["maker_queue_penetration_bps"] < 0:
        raise ValueError("maker_queue_penetration_bps must be non-negative")
    if not 0 <= st["maker_fill_probability"] <= 1:
        raise ValueError("maker_fill_probability must be between 0 and 1")
    tr = config.trading
    tier = tr.active_leverage_tier
    sc = config.scoring
    ft = config.filters
    risk = config.risk_management
    bt = config.backtesting
    ps = config.position_sizing

    port = Portfolio(
        initial_balance=bt.initial_balance,
        maker_fee=config.fees.maker, taker_fee=config.fees.taker,
        default_order_type=config.fees.default_order_type,
        use_maker_fee_for_tp=risk.use_maker_fee_for_tp,
    )
    weights = sc.weights
    primary_tf = tr.primary_timeframe
    tf_hours = timeframe_hours(primary_tf)

    idx = pd.DatetimeIndex(pre.timestamps)
    sd = pd.to_datetime(start_date); ed = pd.to_datetime(end_date)
    if idx.tz is not None:
        sd = sd.tz_localize(idx.tz); ed = ed.tz_localize(idx.tz)
    test_mask = (idx >= sd) & (idx <= ed)

    consec_losses = 0
    outcome_streak = 0
    candles_since_loss = 999
    cooldown = 0
    raw_score_hist: list[float] = []  # per-bar composite, for the decay knobs
    pending: PendingEntry | None = None
    last_close = None; last_time = None
    ti = -1  # test-bar counter (mirrors engine's enumerate over test_indices)
    marginal_candidates = 0
    marginal_accepted = 0
    maker_orders = 0
    maker_touches = 0
    maker_queue_eligible_count = 0
    maker_fills = 0

    def loss_penalty():
        if consec_losses == 0:
            return 0.0
        base = min(consec_losses * risk.consecutive_loss_penalty, risk.max_consecutive_loss_penalty)
        decay = risk.loss_penalty_decay_candles
        if decay > 0 and candles_since_loss > decay:
            f = max(0.0, 1.0 - (candles_since_loss - decay) / decay)
            return base * f
        return base

    for i in range(len(pre.timestamps)):
        if not test_mask[i]:
            continue
        ti += 1  # counts every test bar, like enumerate(test_indices)
        if i < pre.warmup:
            continue
        prim = pre.primary[i]
        if prim is None:
            continue

        ts = str(pre.timestamps[i])
        bar_high = prim.high; bar_low = prim.low; bar_close = prim.close
        regime = detect_market_regime(prim).value
        bar_st = dict(st)
        bar_st["_trail_activation_multiplier"] = st[
            "regime_trailing_activation_mults"
        ].get(regime, 1.0)
        bar_st["_trail_callback_multiplier"] = st[
            "regime_trailing_callback_mults"
        ].get(regime, 1.0)
        last_close = bar_close; last_time = ts

        # 0.75 Resolve a maker order from the previous bar.  The fresh fill is
        # checked for exits on THIS bar (or from its first touched sub-bar onward),
        # removing the original screen's one-bar exit-delay optimism.
        fresh_trade = None
        fresh_sub_start = 0
        subs = pre.subbars[i] if (exit_granularity == "sub" and pre.subbars) else None
        if pending is not None:
            touched = False
            queue_eligible = False
            if subs:
                for sub_i, (s_high, s_low, _s_close) in enumerate(subs):
                    if maker_limit_touched(pending.direction, pending.limit_price,
                                           s_high, s_low):
                        touched = True
                    if maker_queue_eligible(
                        pending.direction, pending.limit_price, s_high, s_low,
                        st["maker_queue_penetration_bps"],
                    ):
                        queue_eligible = True
                        fresh_sub_start = sub_i
                        break
            else:
                touched = maker_limit_touched(
                    pending.direction, pending.limit_price, bar_high, bar_low
                )
                queue_eligible = maker_queue_eligible(
                    pending.direction, pending.limit_price, bar_high, bar_low,
                    st["maker_queue_penetration_bps"],
                )
            maker_touches += int(touched)
            maker_queue_eligible_count += int(queue_eligible)
            fill_key = (f"{getattr(config.trading, 'symbol', '')}|"
                        f"{pending.decision_time}|{pending.direction}|"
                        f"{pending.limit_price:.12g}")
            fills_queue = queue_eligible and deterministic_maker_fill(
                st["maker_fill_probability"], st["maker_fill_seed"], fill_key
            )
            if fills_queue:
                fresh_trade = port.open_trade(
                    direction=pending.direction, entry_price=pending.limit_price,
                    entry_time=ts, stop_loss=pending.stop_loss,
                    take_profit_1=pending.take_profit_1,
                    take_profit_2=pending.take_profit_2,
                    leverage=pending.leverage, risk_pct=pending.risk_pct,
                    tp1_exit_pct=pending.tp1_exit_pct, order_type="maker",
                    max_margin_pct=pending.max_margin_pct,
                )
                fresh_trade._atr_entry = pending.atr_at_entry
                maker_fills += 1
            pending = None

        # 1. exits first (records risk events on the portfolio as trades close).
        # exit_granularity="sub": replay the 4h bar's 1h sub-bars in time order for
        # exit sequencing while keeping the trailing stop FIXED intrabar. Ratchet
        # exactly once after the completed 4h bar, matching engine/live cadence.
        # Falls back to the 4h bar when the 1h series has a hole.
        for trade in list(port.open_trades):
            if subs:
                start = fresh_sub_start if trade is fresh_trade else 0
                first = True
                for (s_high, s_low, s_close) in subs[start:]:
                    if not trade.is_open:
                        break
                    _check_exits(port, trade, s_high, s_low, s_close, ts, risk,
                                 tf_hours, bt.enable_partial_exits, bt.enable_trailing_stops,
                                 config.trading.trailing_stop, slip, model_liquidation,
                                 maintenance_margin, bar_st, count_bar=first,
                                 ratchet_trailing=False)
                    first = False
                if (trade.is_open and bt.enable_trailing_stops
                        and config.trading.trailing_stop.enabled):
                    active_subs = subs[start:]
                    favorable = (max(row[0] for row in active_subs)
                                 if trade.direction == "LONG"
                                 else min(row[1] for row in active_subs))
                    _ratchet_trailing_stop(
                        trade, favorable, config.trading.trailing_stop, bar_st
                    )
            else:
                _check_exits(port, trade, bar_high, bar_low, bar_close, ts, risk,
                             tf_hours, bt.enable_partial_exits, bt.enable_trailing_stops,
                             config.trading.trailing_stop, slip, model_liquidation,
                             maintenance_margin, bar_st)

        # 1.5 funding settlement (mirrors engine: survivors of this bar's exits pay;
        # entries happen at the close, after settlement)
        if funding_by_pos is not None and port.open_trades:
            rate_sum = funding_by_pos[i]
            if rate_sum != 0.0:
                from llm_trading_bot.funding import funding_cost
                for trade in port.open_trades:
                    cost = funding_cost(trade.direction, rate_sum, trade.remaining_size, bar_close)
                    port.apply_funding(trade, cost)

        # apply risk-management updates (mirrors _on_trade_closed, which the engine
        # runs *inside* the exit step — i.e. BEFORE the per-bar counter tick)
        upd = getattr(port, "_risk_events", None)
        if upd:
            for ev in upd:
                if not ev.get("streak_applied", False):
                    outcome_streak = update_outcome_streak(
                        outcome_streak, not ev["loss"]
                    )
                if ev["loss"]:
                    consec_losses += 1
                    candles_since_loss = 0
                    if ev["sl"]:
                        cooldown = risk.cooldown_candles_after_sl
                else:
                    consec_losses = 0
                    candles_since_loss = 999
            port._risk_events = []

        # per-bar counter tick
        candles_since_loss += 1
        if cooldown > 0:
            cooldown -= 1

        # 2. score (reuse real composite scorer with secondary-tf alignment)
        inds_by_tf = {primary_tf: prim}
        inds_by_tf.update(pre.sec_by_bar[i])
        result = compute_composite_score(
            indicators_by_tf=inds_by_tf, weights=weights, primary_timeframe=primary_tf,
            confidence_min=sc.confidence_min, confidence_max=sc.confidence_max,
            scoring_points=getattr(sc, "points", None),
            alignment_mode=st["alignment_mode"], alignment_scale=st["alignment_scale"],
            alignment_k=st["alignment_k"],
            alignment_scale_by_tf=(st["alignment_scale_by_tf"]
                                   if st["alignment_scale_by_tf"] is not None
                                   else getattr(sc, "alignment_scale_by_tf", None)),
            exclude_alignment_tfs=({"1d"} if st.get("daily_trend_replace_align") else None),
        )
        apply_daily_overlay(result, inds_by_tf.get("1d"), st, getattr(sc, "points", None))

        targets = calculate_targets(
            indicators=prim, direction=result.direction,
            sl_strategy=tr.stop_loss_strategy, atr_sl_mult=sc.atr_sl_multiplier,
            tp1_rr=tier.tp1_rr, tp2_rr=tier.tp2_rr,
        )

        raw_score_hist.append(result.raw_score)
        abs_score = abs(result.raw_score)
        lp = loss_penalty()
        # Long/short threshold asymmetry (default 1.0 = symmetric)
        side_mult = 1.0
        if result.direction == Direction.BULLISH:
            side_mult = st["long_threshold_mult"]
            # Funding long-boost: ease long thresholds when shorts are crowded (funding very low).
            if (st["funding_long_boost"] is not None and fund_metric is not None):
                fm = fund_metric[i]
                trend_ok = (not st["funding_long_trend_gate"]
                            or (prim.ema_200 is not None and bar_close > prim.ema_200))
                if fm == fm and fm <= st["funding_long_thr"] and trend_ok:
                    side_mult *= st["funding_long_boost"]
        elif result.direction == Direction.BEARISH:
            side_mult = st["short_threshold_mult"]
            # Funding short-boost: ease short thresholds when longs are crowded in a downtrend.
            if (st["funding_short_boost"] is not None and fund_metric is not None
                    and prim.ema_200 is not None and bar_close < prim.ema_200):
                fm = fund_metric[i]
                if fm == fm and fm >= st["funding_boost_thr"]:
                    side_mult *= st["funding_short_boost"]
        side_mult *= st["regime_threshold_mults"].get(regime, 1.0)
        eff_marg = tier.marginal_threshold_low * side_mult + lp
        eff_strong = tier.strong_threshold * side_mult + lp
        if abs_score >= eff_strong:
            signal = SignalStrength.STRONG
        elif abs_score >= eff_marg:
            signal = SignalStrength.MARGINAL
        else:
            signal = SignalStrength.WAIT
        result.signal_strength = signal

        # Opposite-signal exit: composite flipped hard against an open position
        if st["opposite_exit"] is not None and port.open_trades and result.direction != Direction.NEUTRAL:
            want = "LONG" if result.direction == Direction.BULLISH else "SHORT"
            if abs_score >= st["opposite_exit"]:
                for trade in list(port.open_trades):
                    if trade.direction != want:
                        fill = bar_close * (1 - slip) if trade.direction == "LONG" else bar_close * (1 + slip)
                        port.close_trade(trade, fill, ts, "signal_flip")
                        outcome_streak = update_outcome_streak(
                            outcome_streak, trade.net_pnl > 0
                        )
                        if not hasattr(port, "_risk_events"):
                            port._risk_events = []
                        port._risk_events.append({
                            "loss": trade.net_pnl <= 0, "sl": False,
                            "streak_applied": True,
                        })

        # Decay exit (research knob): the signal died without reversing — signed
        # score fell N consecutive bars AND sits below the floor. Softer than the
        # opposite_exit cliff; same fee/streak accounting as signal_flip.
        if (st["decay_exit_bars"] is not None and st["decay_exit_floor"] is not None
                and port.open_trades
                and len(raw_score_hist) > st["decay_exit_bars"]):
            n = st["decay_exit_bars"]
            for trade in list(port.open_trades):
                sgn = 1.0 if trade.direction == "LONG" else -1.0
                sig = [sgn * s for s in raw_score_hist[-(n + 1):]]
                if (all(sig[k] > sig[k + 1] for k in range(n))
                        and sig[-1] < st["decay_exit_floor"]):
                    fill = (bar_close * (1 - slip) if trade.direction == "LONG"
                            else bar_close * (1 + slip))
                    port.close_trade(trade, fill, ts, "decay_exit")
                    outcome_streak = update_outcome_streak(
                        outcome_streak, trade.net_pnl > 0
                    )
                    if not hasattr(port, "_risk_events"):
                        port._risk_events = []
                    port._risk_events.append({
                        "loss": trade.net_pnl <= 0, "sl": False,
                        "streak_applied": True,
                    })

        if signal in (SignalStrength.STRONG, SignalStrength.MARGINAL) and targets:
            if cooldown > 0:
                pass
            else:
                fails = apply_pre_trade_filters(
                    indicators=prim, targets=targets,
                    min_adx=ft.min_adx, min_volatility_pct=ft.min_volatility_pct,
                    fee_rate=config.fees.active_fee_rate, leverage=tier.leverage,
                    check_profit_after_fees=ft.min_profit_after_fees,
                    category_scores=result.category_scores, direction=result.direction,
                    min_category_agreement=ft.min_category_agreement,
                    require_trend_momentum_agree=ft.require_trend_momentum_agree,
                    skip_choppy_regime=ft.skip_choppy_regime,
                    skip_volatile_regime=ft.skip_volatile_regime,
                )
                result.filter_failures = fails
                result.passed_filters = not fails
                direction_str = "LONG" if result.direction == Direction.BULLISH else "SHORT"
                # Funding-as-signal gate: skip entries into a crowded side. Trend-conditioned
                # by default (high funding only fades longs in a downtrend; low funding only
                # blocks shorts in an uptrend) — see DEFAULT_STRAT / opt/eda_funding.
                fund_block = False
                if fund_metric is not None:
                    fm = fund_metric[i]
                    if fm == fm:  # not NaN
                        below = (prim.ema_200 is not None) and (bar_close < prim.ema_200)
                        above = (prim.ema_200 is not None) and (bar_close > prim.ema_200)
                        if (st["funding_block_long"] is not None and direction_str == "LONG"
                                and fm >= st["funding_block_long"]
                                and (below or not st["funding_trend_gate"])):
                            fund_block = True
                        if (st["funding_block_short"] is not None and direction_str == "SHORT"
                                and fm <= st["funding_block_short"]
                                and (above or not st["funding_trend_gate"])):
                            fund_block = True
                # DD-throttle: while balance drawdown >= threshold, pyramiding pauses
                # (slots drop to 1) and risk is optionally cut, until equity recovers.
                slots = st["max_positions"]
                throttled = False
                if st["dd_throttle"] is not None and port.peak_balance > 0:
                    dd = (port.peak_balance - port.balance) / port.peak_balance
                    if dd >= st["dd_throttle"]:
                        slots = min(slots, st["dd_throttle_slots"])
                        throttled = True
                # Entry slot: default single position; pyramiding allows same-direction adds.
                # A resting maker limit counts as a committed slot and must agree in direction.
                committed = len(port.open_trades) + (1 if pending is not None else 0)
                global_limit = st["global_max_positions"]
                global_slot_ok = global_limit is None or committed < global_limit
                pend_ok = pending is None or pending.direction == direction_str
                # Entry freshness (research knob): a threshold-passing score reached
                # on the way DOWN (decaying from a higher peak) is not a fresh signal.
                # Require the direction-signed score to be non-decreasing over the
                # last K bar transitions.
                rising_block = False
                if st["entry_require_rising"] is not None:
                    k = st["entry_require_rising"]
                    if len(raw_score_hist) > k:
                        sgn = 1.0 if direction_str == "LONG" else -1.0
                        sig = [sgn * s for s in raw_score_hist[-(k + 1):]]
                        rising_block = any(sig[j + 1] < sig[j] for j in range(k))
                # Continuous slope band (research): block when the 1-bar signed
                # slope is below min (strictness dial) or above max (contrarian).
                if ((st["entry_slope_min"] is not None or st["entry_slope_max"] is not None)
                        and len(raw_score_hist) >= 2):
                    sgn = 1.0 if direction_str == "LONG" else -1.0
                    delta = sgn * (raw_score_hist[-1] - raw_score_hist[-2])
                    if st["entry_slope_min"] is not None and delta < st["entry_slope_min"]:
                        rising_block = True
                    if st["entry_slope_max"] is not None and delta > st["entry_slope_max"]:
                        rising_block = True
                can_enter = (committed < slots
                             and global_slot_ok
                             and all(t.direction == direction_str for t in port.open_trades)
                             and pend_ok)
                if not fails and can_enter and not fund_block and not rising_block:
                    gate_accept = True
                    if signal == SignalStrength.MARGINAL:
                        marginal_candidates += 1
                        if marginal_gate is not None:
                            gate_accept = bool(marginal_gate(ts, result, targets))
                        if gate_accept:
                            marginal_accepted += 1
                    # Vol-targeted leverage: normalize per-trade risk across vol regimes
                    # A rejected setup remains a normal no-entry bar; snapshot cadence below
                    # is unchanged.
                    if gate_accept:
                        lev_eff = tier.leverage
                        lev_eff = max(1, int(round(
                            lev_eff * st["regime_leverage_mults"].get(regime, 1.0)
                        )))
                        if st["vol_target_lev"] is not None and prim.atr_pct:
                            lev_eff = max(1, min(tier.leverage, int(round(st["vol_target_lev"] / prim.atr_pct))))
                        # Conviction sizing: scale margin with signal strength
                        risk_eff = ps.risk_pct_per_trade
                        if st["conviction_sizing"] is not None and eff_strong > 0:
                            m = (abs_score / eff_strong) ** st["conviction_sizing"]
                            risk_eff *= max(0.5, min(1.5, m))
                        if signal == SignalStrength.MARGINAL:
                            risk_eff *= st["marginal_size_frac"]
                        if throttled:
                            risk_eff *= st["dd_throttle_risk"]
                        risk_eff *= anti_martingale_multiplier(
                            outcome_streak, st["anti_martingale_step"],
                            st["anti_martingale_min"], st["anti_martingale_max"],
                        )
                        committed_margin = sum(
                            t.remaining_size * t.entry_price / t.leverage
                            for t in port.open_trades if t.leverage > 0
                        )
                        committed_notional = sum(
                            t.remaining_size * t.entry_price for t in port.open_trades
                        )
                        if pending is not None:
                            pending_risk = pending.risk_pct
                            if pending.max_margin_pct is not None:
                                pending_risk = min(
                                    pending_risk, pending.max_margin_pct
                                )
                            pending_margin = port.balance * pending_risk
                            committed_margin += pending_margin
                            committed_notional += pending_margin * pending.leverage
                        risk_eff = cap_risk_pct(
                            risk_eff, lev_eff, port.balance,
                            committed_margin, committed_notional,
                            risk_multiplier=st["portfolio_risk_multiplier"],
                            max_margin_pct=st["global_max_margin_pct"] or 0.0,
                            max_notional_pct=st["global_max_notional_pct"] or 0.0,
                        )
                        if risk_eff <= 0:
                            continue
                        if st["entry_mode"] == "maker":
                            # Rest a limit at this bar's close; it fills next bar only if
                            # price trades back to it (checked at the top of the next bar).
                            pending = PendingEntry(
                                direction=direction_str, limit_price=bar_close,
                                stop_loss=targets.stop_loss,
                                take_profit_1=targets.take_profit_1,
                                take_profit_2=targets.take_profit_2,
                                leverage=lev_eff, risk_pct=risk_eff,
                                tp1_exit_pct=tier.tp1_exit_pct,
                                atr_at_entry=prim.atr_14, decision_time=ts,
                                max_margin_pct=ps.max_position_pct,
                            )
                            maker_orders += 1
                        else:
                            entry_eff = bar_close * (1 + slip) if direction_str == "LONG" else bar_close * (1 - slip)
                            trade = port.open_trade(
                                direction=direction_str, entry_price=entry_eff, entry_time=ts,
                                stop_loss=targets.stop_loss, take_profit_1=targets.take_profit_1,
                                take_profit_2=targets.take_profit_2, leverage=lev_eff,
                                risk_pct=risk_eff, tp1_exit_pct=tier.tp1_exit_pct,
                                order_type="taker",
                                max_margin_pct=ps.max_position_pct,
                            )
                            trade._atr_entry = prim.atr_14  # for ATR-based trailing

        if ti % 10 == 0:
            port.record_snapshot(ts, bar_close)

    # close remaining
    if last_close is not None:
        for trade in list(port.open_trades):
            port.close_trade(trade, last_close, last_time, "end_of_backtest")
        port.record_snapshot(last_time, last_close)

    s = port.compute_stats()
    return Result(
        return_pct=s.total_return_pct, final_balance=s.final_balance, trades=s.total_trades,
        win_rate=s.win_rate, profit_factor=s.profit_factor, max_dd_pct=s.max_drawdown_pct,
        sharpe=s.sharpe_ratio, marginal_candidates=marginal_candidates,
        marginal_accepted=marginal_accepted, maker_orders=maker_orders,
        maker_touches=maker_touches,
        maker_queue_eligible=maker_queue_eligible_count,
        maker_fills=maker_fills,
    )


def _ratchet_trailing_stop(trade, favorable, trailing_config, st=None):
    """Ratchet once after a completed primary bar's exit sequence."""
    from llm_trading_bot.trailing import compute_trailing_stop
    if st is None:
        st = DEFAULT_STRAT
    if not trailing_config.enabled or not trade.is_open:
        return
    is_long = trade.direction == "LONG"
    if st["trail_mode"] == "atr" and getattr(trade, "_atr_entry", None):
        act_d = st["trail_act_atr"] * trade._atr_entry
        cb_d = st["trail_cb_atr"] * trade._atr_entry
        new_sl = None
        if is_long:
            if favorable >= trade.entry_price + act_d:
                candidate = favorable - cb_d
                if candidate > trade.stop_loss:
                    new_sl = candidate
        else:
            if favorable <= trade.entry_price - act_d:
                candidate = favorable + cb_d
                if candidate < trade.stop_loss:
                    new_sl = candidate
    else:
        activation_pct = trailing_config.activation_pct * st.get(
            "_trail_activation_multiplier", 1.0
        )
        callback_pct = trailing_config.callback_pct * st.get(
            "_trail_callback_multiplier", 1.0
        )
        new_sl = compute_trailing_stop(
            direction=trade.direction, entry_price=trade.entry_price,
            favorable_extreme=favorable, current_sl=trade.stop_loss,
            activation_pct=activation_pct, callback_pct=callback_pct,
        )
    if new_sl is not None:
        trade.stop_loss = new_sl


def _check_exits(port, trade, bar_high, bar_low, bar_close, bar_time, risk, tf_hours,
                 enable_partial, enable_trailing, trailing_config,
                 slip=0.0, model_liquidation=True, maintenance_margin=0.005, st=None,
                 count_bar=True, ratchet_trailing=True):
    if st is None:
        st = DEFAULT_STRAT
    if not trade.is_open:
        return

    # NOTE on intrabar path: we assume the ADVERSE extreme is reached BEFORE the favorable
    # one (worst case). So exits (SL/TP) are evaluated against the stop as it stands at the
    # START of the bar, and the trailing stop is only ratcheted at the END of the bar (using
    # this bar's favorable extreme) for use on SUBSEQUENT bars. Trailing up first and then
    # checking the low against the raised stop would optimistically credit a top-of-bar exit.

    if count_bar:  # only once per PRIMARY bar (sub-bar replay passes False after the first)
        trade.bars_held += 1
    is_long = trade.direction == "LONG"

    def record_risk(t):
        if not hasattr(port, "_risk_events"):
            port._risk_events = []
        port._risk_events.append({
            "loss": t.net_pnl <= 0,
            "sl": t.exit_reason in ("sl", "trailing_stop"),
            "symbol": getattr(t, "symbol", ""),
        })

    max_hours = risk.max_holding_hours
    if max_hours > 0:
        max_bars = max_hours // tf_hours
        if trade.bars_held >= max_bars:
            fill = bar_close * (1 - slip) if is_long else bar_close * (1 + slip)
            port.close_trade(trade, fill, bar_time, "time_expired")
            record_risk(trade)
            return

    # Effective stop = configured SL, but capped at the liquidation price for the
    # trade's leverage (a stop placed beyond liquidation can't actually be reached —
    # the position is force-closed at liquidation first, wiping ~the full margin).
    eff_stop = trade.stop_loss
    if model_liquidation and trade.leverage > 0:
        liq_dist = (1.0 / trade.leverage) - maintenance_margin
        if liq_dist > 0:
            if is_long:
                liq_price = trade.entry_price * (1 - liq_dist)
                eff_stop = max(trade.stop_loss, liq_price)  # nearer stop hit first on the way down
            else:
                liq_price = trade.entry_price * (1 + liq_dist)
                eff_stop = min(trade.stop_loss, liq_price)

    sl_hit = (bar_low <= eff_stop) if is_long else (bar_high >= eff_stop)
    if sl_hit:
        fill = eff_stop * (1 - slip) if is_long else eff_stop * (1 + slip)
        port.close_trade(trade, fill, bar_time, "sl")
        record_risk(trade)
        return

    if enable_partial and not trade.partial_exits:
        tp1_hit = (bar_high >= trade.take_profit_1) if is_long else (bar_low <= trade.take_profit_1)
        if tp1_hit:
            port.partial_exit(trade, trade.take_profit_1, bar_time, trade.tp1_exit_pct, "tp1")
            if enable_trailing:
                trade.stop_loss = trade.entry_price

    if trade.is_open:
        tp2_hit = (bar_high >= trade.take_profit_2) if is_long else (bar_low <= trade.take_profit_2)
        if tp2_hit:
            port.close_trade(trade, trade.take_profit_2, bar_time, "tp2")
            record_risk(trade)

    # Ratchet the trailing stop LAST — using this bar's favorable extreme — so it only
    # affects subsequent bars (conservative: never rewards an intrabar top-of-bar exit).
    if (ratchet_trailing and enable_trailing and trailing_config.enabled
            and trade.is_open):
        favorable = bar_high if is_long else bar_low
        _ratchet_trailing_stop(trade, favorable, trailing_config, st)
