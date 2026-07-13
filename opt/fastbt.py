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
)
from llm_trading_bot.portfolio import Portfolio


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

    # Secondary timeframes: build their IndicatorSets, then asof-map to each 4h bar.
    sec_inds = {}
    for tf, df in data_by_tf.items():
        if tf == primary_tf:
            continue
        sec_inds[tf] = (df.index, build_indicatorsets(df, tf))

    sec_by_bar = []
    for ts in primary_df.index:
        d = {}
        for tf, (idx, inds) in sec_inds.items():
            pos = idx.searchsorted(ts, side="right") - 1  # latest secondary bar <= ts
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


DEFAULT_STRAT = {
    # All defaults preserve the current engine behavior exactly.
    "vol_target_lev": None,       # e.g. 40.0 -> lev_eff = min(lev, vol_target_lev / atr_pct)
    "trail_mode": "pct",          # "pct" (of entry) | "atr" (multiples of entry ATR)
    "trail_act_atr": 0.5,          # activation distance in ATRs (trail_mode="atr")
    "trail_cb_atr": 0.6,           # callback distance in ATRs (trail_mode="atr")
    "conviction_sizing": None,    # e.g. 1.0 -> risk_pct scaled by (abs_score/strong)^1 capped [0.5, 1.5]
    "opposite_exit": None,        # e.g. 25.0 -> close when opposite-direction |score| >= this
    "short_threshold_mult": 1.0,  # >1 = stricter shorts (asymmetry)
    "long_threshold_mult": 1.0,
    "max_positions": 1,           # >1 allows pyramiding same-direction entries
    "marginal_size_frac": 1.0,    # size fraction for MARGINAL (vs STRONG) entries
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
}


def simulate(pre: Precomputed, config, start_date: str, end_date: str,
             slip: float = 0.0, model_liquidation: bool = True,
             maintenance_margin: float = 0.005, strat: dict | None = None,
             funding_by_pos: "list[float] | None" = None,
             exit_granularity: str = "primary",
             fund_metric: "list[float] | None" = None) -> Result:
    """slip: per-side price slippage fraction applied to market fills (entry, SL, time/EOB).
    model_liquidation: if True, a stop placed beyond the isolated-margin liquidation
    distance is capped at the liquidation price (position wipes ~full margin first).
    strat: strategy-variant flags (see DEFAULT_STRAT); defaults reproduce the engine."""
    st = dict(DEFAULT_STRAT)
    # Config-backed strategy features (engine parity): explicit strat overrides win.
    ps_cfg = config.position_sizing
    st["max_positions"] = getattr(ps_cfg, "max_positions", 1)
    conv = getattr(ps_cfg, "conviction_exponent", 0.0)
    st["conviction_sizing"] = conv if conv > 0 else None
    opp = getattr(config.risk_management, "opposite_exit_threshold", 0.0)
    st["opposite_exit"] = opp if opp > 0 else None
    ddt = getattr(config.risk_management, "dd_throttle_threshold", 0.0)
    st["dd_throttle"] = ddt if ddt > 0 else None
    st["dd_throttle_slots"] = getattr(config.risk_management, "dd_throttle_slots", 1)
    st["dd_throttle_risk"] = getattr(config.risk_management, "dd_throttle_risk", 0.5)
    if strat:
        st.update(strat)
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
    tf_hours = {"1h": 1, "4h": 4, "1d": 24}.get(primary_tf, 4)

    idx = pd.DatetimeIndex(pre.timestamps)
    sd = pd.to_datetime(start_date); ed = pd.to_datetime(end_date)
    if idx.tz is not None:
        sd = sd.tz_localize(idx.tz); ed = ed.tz_localize(idx.tz)
    test_mask = (idx >= sd) & (idx <= ed)

    consec_losses = 0
    candles_since_loss = 999
    cooldown = 0
    last_close = None; last_time = None
    ti = -1  # test-bar counter (mirrors engine's enumerate over test_indices)

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
        last_close = bar_close; last_time = ts

        # 1. exits first (records risk events on the portfolio as trades close).
        # exit_granularity="sub": replay the 4h bar's 1h sub-bars in time order —
        # real intrabar sequencing (trailing ratchets between sub-bars) instead of
        # the worst-case single-bar assumption. Falls back to the 4h bar when the
        # 1h series has a hole.
        subs = pre.subbars[i] if (exit_granularity == "sub" and pre.subbars) else None
        for trade in list(port.open_trades):
            if subs:
                first = True
                for (s_high, s_low, s_close) in subs:
                    if not trade.is_open:
                        break
                    _check_exits(port, trade, s_high, s_low, s_close, ts, risk,
                                 tf_hours, bt.enable_partial_exits, bt.enable_trailing_stops,
                                 config.trading.trailing_stop, slip, model_liquidation,
                                 maintenance_margin, st, count_bar=first)
                    first = False
            else:
                _check_exits(port, trade, bar_high, bar_low, bar_close, ts, risk,
                             tf_hours, bt.enable_partial_exits, bt.enable_trailing_stops,
                             config.trading.trailing_stop, slip, model_liquidation,
                             maintenance_margin, st)

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
        )

        targets = calculate_targets(
            indicators=prim, direction=result.direction,
            sl_strategy=tr.stop_loss_strategy, atr_sl_mult=sc.atr_sl_multiplier,
            tp1_rr=tier.tp1_rr, tp2_rr=tier.tp2_rr,
        )

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
        eff_marg = tier.marginal_threshold_low * side_mult + lp
        eff_strong = tier.strong_threshold * side_mult + lp
        if abs_score >= eff_strong:
            signal = SignalStrength.STRONG
        elif abs_score >= eff_marg:
            signal = SignalStrength.MARGINAL
        else:
            signal = SignalStrength.WAIT

        # Opposite-signal exit: composite flipped hard against an open position
        if st["opposite_exit"] is not None and port.open_trades and result.direction != Direction.NEUTRAL:
            want = "LONG" if result.direction == Direction.BULLISH else "SHORT"
            if abs_score >= st["opposite_exit"]:
                for trade in list(port.open_trades):
                    if trade.direction != want:
                        fill = bar_close * (1 - slip) if trade.direction == "LONG" else bar_close * (1 + slip)
                        port.close_trade(trade, fill, ts, "signal_flip")
                        if not hasattr(port, "_risk_events"):
                            port._risk_events = []
                        port._risk_events.append({"loss": trade.net_pnl <= 0, "sl": False})

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
                # Entry slot: default single position; pyramiding allows same-direction adds
                can_enter = (len(port.open_trades) < slots
                             and all(t.direction == direction_str for t in port.open_trades))
                if not fails and can_enter and not fund_block:
                    # Vol-targeted leverage: normalize per-trade risk across vol regimes
                    lev_eff = tier.leverage
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
                    entry_eff = bar_close * (1 + slip) if direction_str == "LONG" else bar_close * (1 - slip)
                    trade = port.open_trade(
                        direction=direction_str, entry_price=entry_eff, entry_time=ts,
                        stop_loss=targets.stop_loss, take_profit_1=targets.take_profit_1,
                        take_profit_2=targets.take_profit_2, leverage=lev_eff,
                        risk_pct=risk_eff, tp1_exit_pct=tier.tp1_exit_pct,
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
        sharpe=s.sharpe_ratio,
    )


def _check_exits(port, trade, bar_high, bar_low, bar_close, bar_time, risk, tf_hours,
                 enable_partial, enable_trailing, trailing_config,
                 slip=0.0, model_liquidation=True, maintenance_margin=0.005, st=None,
                 count_bar=True):
    from llm_trading_bot.trailing import compute_trailing_stop
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
        port._risk_events.append({"loss": t.net_pnl <= 0, "sl": t.exit_reason in ("sl", "trailing_stop")})

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
    if enable_trailing and trailing_config.enabled and trade.is_open:
        favorable = bar_high if is_long else bar_low
        if st["trail_mode"] == "atr" and getattr(trade, "_atr_entry", None):
            # Distances as multiples of the ATR at entry (adapts to vol regime).
            act_d = st["trail_act_atr"] * trade._atr_entry
            cb_d = st["trail_cb_atr"] * trade._atr_entry
            new_sl = None
            if is_long:
                if favorable >= trade.entry_price + act_d:
                    cand = favorable - cb_d
                    if cand > trade.stop_loss:
                        new_sl = cand
            else:
                if favorable <= trade.entry_price - act_d:
                    cand = favorable + cb_d
                    if cand < trade.stop_loss:
                        new_sl = cand
        else:
            new_sl = compute_trailing_stop(
                direction=trade.direction, entry_price=trade.entry_price,
                favorable_extreme=favorable, current_sl=trade.stop_loss,
                activation_pct=trailing_config.activation_pct, callback_pct=trailing_config.callback_pct,
            )
        if new_sl is not None:
            trade.stop_loss = new_sl
