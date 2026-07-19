"""Scalper research engine — dedicated vectorized backtester for 5m/15m cadence.

RESEARCH ONLY. Nothing here touches the live bot. The live 4h product keeps
using fastbt/backtesting.py; this engine exists because a scalper needs a
different execution/exit model (tight ATR brackets, time stops, mean-touch
exits) and grid searches over ~500k 5m bars, which the per-bar scorer loop
cannot sustain.

Execution model (pessimistic-by-construction where it matters):
- Decision on the CLOSE of bar i; action happens on bar i+1 (no lookahead).
- taker entry: fill at open[i+1] * (1 +/- slip), taker fee.
- maker entry: post-only limit at close[i], good for `maker_ttl` bars. Fills in
  bar j if price trades through limit * (1 -/+ penetration) (long: low <= adj;
  short: high >= adj), maker fee, no slip. Fill bar is exposed to SL from the
  fill onward (adverse-first); TP is NOT granted on the fill bar (extra
  conservative vs fastbt, which grants it after the SL check).
- Exits per bar, adverse-first: SL (stop-market: taker + slip) -> TP (maker or
  taker per `tp_taker`) -> time stop / mean-touch exit at close (taker + slip).
- Breakeven ratchet and ATR trailing update at bar END (effective next bar).
- Funding: settlement events mapped to the bar whose open == settlement time;
  open positions pay rate * notional (longs pay positive rates).
- Liquidation cap: isolated-margin; stop beyond liq distance exits at liq.

Sizing: loss-targeted. notional = balance * loss_pct / sl_dist_frac, capped by
`max_notional_x` * balance (implied leverage cap). This makes the scalp fee
reality explicit: the tighter the stop, the bigger the notional a fixed loss
budget implies, and fees scale with notional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Data loading / aggregation
# ----------------------------------------------------------------------

def load_futures(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Load cached Binance USDT-perp candles (open-stamped UTC index)."""
    from llm_trading_bot.binance_csv import download_binance_csv

    df = download_binance_csv(
        symbol=symbol, timeframe=timeframe, start_date=start, end_date=end,
        warmup_days=0, market="futures",
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def aggregate(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate 5m bars to a coarser cadence (open-stamped, UTC-aligned)."""
    out = df5.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last",
         "Volume": "sum"}
    ).dropna()
    return out


# ----------------------------------------------------------------------
# Vectorized causal indicators (rolling windows end at the current bar)
# ----------------------------------------------------------------------

def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def zscore(close: pd.Series, n: int) -> pd.Series:
    m = close.rolling(n).mean()
    s = close.rolling(n).std(ddof=0)
    return (close - m) / s.replace(0, np.nan)


def rolling_vwap(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = (tp * df["Volume"]).rolling(n).sum()
    vv = df["Volume"].rolling(n).sum()
    return pv / vv.replace(0, np.nan)


def donchian(high: pd.Series, low: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Highest high / lowest low over the PRIOR n bars (excludes current bar)."""
    return high.shift(1).rolling(n).max(), low.shift(1).rolling(n).min()


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman ER: |net move| / path length over n bars. ~1 trending, ~0 chop."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)


# ----------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------

@dataclass
class ExecParams:
    entry_mode: str = "maker"        # "maker" | "taker"
    maker_ttl: int = 1               # bars a maker limit rests
    limit_offset_atr: float = 0.0    # rest the limit this many ATR *into* the
    #                                  extreme (passive fade: long limit below
    #                                  close). Brackets anchor to the limit.
    penetration_bps: float = 0.0     # maker queue-depth stress
    sl_atr: float = 1.5              # stop distance in ATR
    tp_atr: float = 2.5              # take-profit distance in ATR
    tp_taker: bool = False           # True = TP exits pay taker (pessimistic)
    time_stop_bars: int = 0          # 0 = off
    exit_on_mean: bool = False       # mean-reversion: exit at close when z crosses 0
    breakeven_atr: float = 0.0       # move SL to entry after this favorable ATR move (0=off)
    trail_atr: float = 0.0           # chandelier trail distance in ATR (0=off)
    trail_arm_atr: float = 0.0       # arm trail only after this favorable ATR move
    postonly_open_cancel: bool = True  # cancel maker limit if the fill bar OPENS
    #                                     through it (marketable post-only -> live
    #                                     Bitget cancels; matches MAKER_POST_ONLY_
    #                                     CANCELLED observed live). False = fill on
    #                                     touch like fastbt's canonical model.
    maker_fee: float = 0.0002
    taker_fee: float = 0.0006
    slip: float = 0.0002             # per-side market-fill slippage
    loss_pct: float = 0.005          # loss budget per trade (fraction of balance)
    max_notional_x: float = 3.0      # notional cap as multiple of balance
    max_leverage: float = 25.0       # margin efficiency only (isolated)
    cooldown_bars: int = 0           # bars to wait after a stop-out
    allow_long: bool = True
    allow_short: bool = True


@dataclass
class Result:
    growth_x: float
    max_dd_pct: float
    trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    fees_paid: float
    funding_paid: float
    gross_pnl: float
    avg_hold_bars: float
    longs: int
    shorts: int
    maker_orders: int = 0
    maker_fills: int = 0
    exit_counts: dict = field(default_factory=dict)
    equity: np.ndarray | None = None   # per-bar marks (return_equity=True)


def build_subbars(index_primary: pd.DatetimeIndex, tf_minutes: int,
                  df_sub: pd.DataFrame, sub_minutes: int) -> np.ndarray | None:
    """(n_primary, k, 3) array of (high, low, close) sub-bars per primary bar.

    Rows with incomplete sub coverage get NaN and the engine falls back to the
    primary bar's adverse-first OHLC for that bar.
    """
    k = tf_minutes // sub_minutes
    sub_idx = df_sub.index
    sh = df_sub["High"].to_numpy(); sl_ = df_sub["Low"].to_numpy()
    sc = df_sub["Close"].to_numpy()
    n = len(index_primary)
    out = np.full((n, k, 3), np.nan)
    pos = sub_idx.searchsorted(index_primary, side="left")
    delta = pd.Timedelta(minutes=tf_minutes)
    for i in range(n):
        a = pos[i]
        b = sub_idx.searchsorted(index_primary[i] + delta, side="left")
        m = b - a
        if 0 < m <= k:
            out[i, :m, 0] = sh[a:b]
            out[i, :m, 1] = sl_[a:b]
            out[i, :m, 2] = sc[a:b]
    return out


def simulate(
    ohlc: dict[str, np.ndarray],
    atr_arr: np.ndarray,
    long_sig: np.ndarray,
    short_sig: np.ndarray,
    p: ExecParams,
    mean_exit_long: np.ndarray | None = None,
    mean_exit_short: np.ndarray | None = None,
    funding: np.ndarray | None = None,
    start_i: int = 0,
    end_i: int | None = None,
    initial_balance: float = 10_000.0,
    subbars: np.ndarray | None = None,
    return_equity: bool = False,
) -> Result:
    """Run one scalp backtest over precomputed signal arrays.

    ohlc: dict with "open","high","low","close" float64 arrays.
    long_sig/short_sig: bool arrays — signal evaluated at bar i's CLOSE.
    mean_exit_*: bool arrays — "the reversion completed" exit trigger at bar i.
    funding: per-bar funding-rate-sum array (longs pay positive).
    subbars: optional (n, k, 3) high/low/close sub-bar array (build_subbars) —
    resolves SL-vs-TP ordering *across* sub-bars by data; adverse-first remains
    the rule *within* each sub-bar. NaN rows fall back to primary-bar OHLC.
    """
    o = ohlc["open"]; h = ohlc["high"]; l = ohlc["low"]; c = ohlc["close"]
    n = len(c)
    if end_i is None:
        end_i = n
    end_i = min(end_i, n)

    balance = initial_balance
    peak = balance
    max_dd = 0.0

    # position state
    pos_dir = 0            # +1 long, -1 short, 0 flat
    pos_qty = 0.0          # base quantity
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_margin = 0.0
    pos_bars = 0
    pos_atr = 0.0
    trail_armed = False

    # pending maker order
    pend_dir = 0
    pend_limit = 0.0
    pend_sl = 0.0
    pend_tp = 0.0
    pend_atr = 0.0
    pend_ttl = 0
    pend_fresh = False   # True only on the first bar after placement: the
    #                      post-only marketability cancel applies then; once
    #                      resting, price crossing the limit is a FILL.

    cooldown = 0
    trades = 0; wins = 0; longs = 0; shorts = 0
    gross_win = 0.0; gross_loss = 0.0
    fees_paid = 0.0; funding_paid = 0.0; gross_pnl = 0.0
    hold_bars_sum = 0
    maker_orders = 0; maker_fills = 0
    exit_counts: dict[str, int] = {}
    equity_marks: list[float] = []

    liq_frac = max(1.0 / p.max_leverage - 0.005, 0.0)

    def close_position(price: float, fee_rate: float, reason: str, bars_i: int):
        nonlocal balance, pos_dir, pos_qty, trades, wins, gross_win, gross_loss
        nonlocal fees_paid, gross_pnl, hold_bars_sum, longs, shorts, cooldown
        nonlocal trail_armed
        raw = (price - pos_entry) * pos_qty * pos_dir
        fee = (pos_entry * pos_qty) * entry_fee_rate + (price * pos_qty) * fee_rate
        pnl = raw - fee
        balance_new = balance + pnl
        # isolated margin: cannot lose more than margin
        if pnl < -pos_margin:
            pnl = -pos_margin
            balance_new = balance + pnl
        balance = balance_new
        fees_paid += fee
        gross_pnl += raw
        trades += 1
        hold_bars_sum += pos_bars
        if pos_dir > 0:
            longs += 1
        else:
            shorts += 1
        if pnl > 0:
            wins += 1; gross_win += pnl
        else:
            gross_loss += -pnl
            if reason in ("sl", "liq"):
                cooldown = p.cooldown_bars
        exit_counts[reason] = exit_counts.get(reason, 0) + 1
        pos_dir = 0; pos_qty = 0.0
        trail_armed = False

    entry_fee_rate = p.maker_fee if p.entry_mode == "maker" else p.taker_fee
    tp_fee = p.taker_fee if p.tp_taker else p.maker_fee
    pen = p.penetration_bps / 10_000.0

    for i in range(start_i, end_i):
        # ---- 1a. resolve a deferred taker entry at this bar's open
        fresh_fill = False
        fill_sub = 0
        if pend_dir != 0 and pend_limit < 0 and pos_dir == 0:
            fill = o[i] * (1 + p.slip) if pend_dir > 0 else o[i] * (1 - p.slip)
            sl_dist = abs(fill - pend_sl) / fill
            if sl_dist > 0:
                notional = min(balance * p.loss_pct / sl_dist,
                               balance * p.max_notional_x)
                margin = notional / p.max_leverage
                if notional > 0 and margin < balance:
                    pos_dir = pend_dir
                    pos_entry = fill
                    pos_qty = notional / fill
                    pos_sl = pend_sl
                    pos_tp = pend_tp
                    pos_margin = margin
                    pos_atr = pend_atr
                    pos_bars = 0
                    trail_armed = False
                    # taker fill at open: full bar applies, adverse-first (SL
                    # checked before TP below) — same as fastbt's convention.
            pend_dir = 0

        # ---- 1b. resolve pending maker order on this bar
        elif pend_dir != 0 and pos_dir == 0:
            adj = pend_limit * (1 - pen) if pend_dir > 0 else pend_limit * (1 + pen)
            touched = False
            if p.postonly_open_cancel and pend_fresh and (
                (pend_dir > 0 and o[i] < pend_limit)
                or (pend_dir < 0 and o[i] > pend_limit)
            ):
                # marketable post-only at the fill bar's open -> exchange cancels
                pend_dir = 0
                exit_counts["postonly_cancel"] = exit_counts.get("postonly_cancel", 0) + 1
            else:
                use_subs = (subbars is not None
                            and subbars[i, 0, 0] == subbars[i, 0, 0])
                if use_subs:
                    rows = subbars[i]
                    for si in range(rows.shape[0]):
                        rh = rows[si, 0]; rl = rows[si, 1]
                        if rh != rh:
                            break
                        if (pend_dir > 0 and rl <= adj) or (pend_dir < 0 and rh >= adj):
                            touched = True
                            fill_sub = si
                            break
                else:
                    touched = (l[i] <= adj) if pend_dir > 0 else (h[i] >= adj)
            if pend_dir != 0 and touched:
                # size the trade off the *current* balance
                sl_dist = abs(pend_limit - pend_sl) / pend_limit
                if sl_dist > 0:
                    notional = min(balance * p.loss_pct / sl_dist,
                                   balance * p.max_notional_x)
                    margin = notional / p.max_leverage
                    if notional > 0 and margin < balance:
                        pos_dir = pend_dir
                        pos_entry = pend_limit
                        pos_qty = notional / pend_limit
                        pos_sl = pend_sl
                        pos_tp = pend_tp
                        pos_margin = margin
                        pos_atr = pend_atr
                        pos_bars = 0
                        trail_armed = False
                        fresh_fill = True
                        maker_fills += 1
                pend_dir = 0
            else:
                pend_ttl -= 1
                if pend_ttl <= 0:
                    pend_dir = 0
            pend_fresh = False

        # ---- 2. exits (adverse-first)
        if pos_dir != 0:
            pos_bars += 1
            exited = False
            # liquidation cap on the effective stop
            eff_sl = pos_sl
            reason_sl = "sl"
            if liq_frac > 0:
                # with loss-targeted sizing, implied leverage = notional/margin
                # = max_leverage, so the isolated liq distance is liq_frac.
                if pos_dir > 0:
                    liq = pos_entry * (1 - liq_frac)
                    if liq > eff_sl:
                        eff_sl = liq; reason_sl = "liq"
                else:
                    liq = pos_entry * (1 + liq_frac)
                    if liq < eff_sl:
                        eff_sl = liq; reason_sl = "liq"
            use_subs = (subbars is not None
                        and subbars[i, 0, 0] == subbars[i, 0, 0])
            if use_subs:
                # data-resolved SL-vs-TP ordering across 5m sub-bars;
                # adverse-first (SL before TP) *within* each sub-bar. A fresh
                # maker fill is exposed from its fill sub-bar onward, and gets
                # no TP in the fill sub-bar itself (conservative).
                rows = subbars[i]
                start_sub = fill_sub if fresh_fill else 0
                for si in range(start_sub, rows.shape[0]):
                    rh = rows[si, 0]; rl = rows[si, 1]
                    if rh != rh:
                        break
                    tp_ok = not (fresh_fill and si == fill_sub)
                    if pos_dir > 0:
                        if rl <= eff_sl:
                            close_position(eff_sl * (1 - p.slip), p.taker_fee,
                                           reason_sl, i)
                            exited = True
                            break
                        if tp_ok and rh >= pos_tp:
                            close_position(pos_tp, tp_fee, "tp", i)
                            exited = True
                            break
                    else:
                        if rh >= eff_sl:
                            close_position(eff_sl * (1 + p.slip), p.taker_fee,
                                           reason_sl, i)
                            exited = True
                            break
                        if tp_ok and rl <= pos_tp:
                            close_position(pos_tp, tp_fee, "tp", i)
                            exited = True
                            break
            elif pos_dir > 0:
                if l[i] <= eff_sl:
                    close_position(eff_sl * (1 - p.slip), p.taker_fee, reason_sl, i)
                    exited = True
                elif (not fresh_fill) and h[i] >= pos_tp:
                    close_position(pos_tp, tp_fee, "tp", i)
                    exited = True
            else:
                if h[i] >= eff_sl:
                    close_position(eff_sl * (1 + p.slip), p.taker_fee, reason_sl, i)
                    exited = True
                elif (not fresh_fill) and l[i] <= pos_tp:
                    close_position(pos_tp, tp_fee, "tp", i)
                    exited = True

            if not exited and pos_dir != 0:
                # end-of-bar exits: time stop, mean-touch
                if p.time_stop_bars > 0 and pos_bars >= p.time_stop_bars:
                    px = c[i] * (1 - p.slip) if pos_dir > 0 else c[i] * (1 + p.slip)
                    close_position(px, p.taker_fee, "time", i)
                    exited = True
                elif p.exit_on_mean and mean_exit_long is not None:
                    trig = mean_exit_long[i] if pos_dir > 0 else mean_exit_short[i]
                    if trig:
                        px = c[i] * (1 - p.slip) if pos_dir > 0 else c[i] * (1 + p.slip)
                        close_position(px, p.taker_fee, "mean", i)
                        exited = True

            if not exited and pos_dir != 0:
                # funding settlement (survivors pay)
                if funding is not None and funding[i] != 0.0:
                    cost = funding[i] * pos_qty * c[i] * pos_dir
                    balance -= cost
                    funding_paid += cost
                # end-of-bar ratchets (effective from next bar)
                if p.breakeven_atr > 0 and pos_atr > 0:
                    if pos_dir > 0 and h[i] >= pos_entry + p.breakeven_atr * pos_atr:
                        pos_sl = max(pos_sl, pos_entry)
                    elif pos_dir < 0 and l[i] <= pos_entry - p.breakeven_atr * pos_atr:
                        pos_sl = min(pos_sl, pos_entry)
                if p.trail_atr > 0 and pos_atr > 0:
                    if pos_dir > 0:
                        if not trail_armed and h[i] >= pos_entry + p.trail_arm_atr * pos_atr:
                            trail_armed = True
                        if trail_armed:
                            cand = h[i] - p.trail_atr * pos_atr
                            if cand > pos_sl:
                                pos_sl = cand
                    else:
                        if not trail_armed and l[i] <= pos_entry - p.trail_arm_atr * pos_atr:
                            trail_armed = True
                        if trail_armed:
                            cand = l[i] + p.trail_atr * pos_atr
                            if cand < pos_sl:
                                pos_sl = cand

        # ---- 3. cooldown tick
        if cooldown > 0:
            cooldown -= 1

        # ---- 4. entry decision at this bar's close (acts next bar)
        if pos_dir == 0 and pend_dir == 0 and cooldown == 0:
            a = atr_arr[i]
            want = 0
            if long_sig[i] and p.allow_long:
                want = 1
            elif short_sig[i] and p.allow_short:
                want = -1
            if want != 0 and a == a and a > 0 and c[i] > 0:
                anchor = c[i] - want * p.limit_offset_atr * a
                sl_price = anchor - want * p.sl_atr * a
                tp_price = anchor + want * p.tp_atr * a
                if p.entry_mode == "maker":
                    pend_dir = want
                    pend_limit = anchor
                    pend_sl = sl_price
                    pend_tp = tp_price
                    pend_atr = a
                    pend_ttl = p.maker_ttl
                    pend_fresh = True
                    maker_orders += 1
                else:
                    # taker: fill at NEXT bar's open (resolved in step 1a)
                    pend_dir = want
                    pend_limit = -1.0   # sentinel: taker fill at open
                    pend_sl = sl_price
                    pend_tp = tp_price
                    pend_atr = a
                    pend_ttl = 1

        # ---- 5. mark equity at close
        if pos_dir != 0:
            eq = balance + (c[i] - pos_entry) * pos_qty * pos_dir
        else:
            eq = balance
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        equity_marks.append(eq)

    # close any open position at the end
    if pos_dir != 0:
        px = c[end_i - 1] * (1 - p.slip) if pos_dir > 0 else c[end_i - 1] * (1 + p.slip)
        close_position(px, p.taker_fee, "eob", end_i - 1)

    eqs = np.asarray(equity_marks)
    if len(eqs) > 2 and (eqs > 0).all():
        rets = np.diff(np.log(eqs))
        sd = rets.std()
        # annualize by bars/year for readability; comparisons are like-for-like
        bars_per_year = 365 * 24 * 12  # 5m default; only used as a scale factor
        sharpe = float(rets.mean() / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0
    else:
        sharpe = 0.0

    return Result(
        growth_x=balance / initial_balance,
        max_dd_pct=max_dd * 100,
        trades=trades,
        win_rate=(wins / trades * 100) if trades else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        sharpe=sharpe,
        fees_paid=fees_paid,
        funding_paid=funding_paid,
        gross_pnl=gross_pnl,
        avg_hold_bars=(hold_bars_sum / trades) if trades else 0.0,
        longs=longs, shorts=shorts,
        maker_orders=maker_orders, maker_fills=maker_fills,
        exit_counts=exit_counts,
        equity=eqs if return_equity else None,
    )
