"""Signal builders for the scalper research.

Each builder is vectorized and CAUSAL: the signal at index i uses only data
from bars <= i (standard pandas rolling/ewm). The engine acts on bar i+1.

Higher-timeframe context is aligned by COMPLETION time: a 1h bar opened at T
becomes visible at T+1h, and is mapped to the first 5m/15m decision close at
or after that moment (no lookahead — verified in tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from opt.scalp.engine import (
    atr, donchian, efficiency_ratio, ema, rolling_vwap, rsi, sma, zscore,
)


def align_context(ctx: pd.Series, ctx_tf_minutes: int, base_index: pd.DatetimeIndex,
                  base_tf_minutes: int) -> np.ndarray:
    """Map a context-TF series onto base-TF decision closes, completion-aligned."""
    avail = ctx.copy()
    avail.index = avail.index + pd.Timedelta(minutes=ctx_tf_minutes)  # close times
    decision_closes = base_index + pd.Timedelta(minutes=base_tf_minutes)
    return avail.reindex(decision_closes, method="ffill").to_numpy()


def _gates(df: pd.DataFrame, params: dict, ctx: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Common optional gates. Returns (long_ok, short_ok) bool arrays."""
    n = len(df)
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)

    # chop/trend regime via efficiency ratio on the trade TF
    er_max = params.get("er_max")          # mean-reversion wants chop (ER below)
    er_min = params.get("er_min")          # breakout wants trend (ER above)
    if er_max is not None or er_min is not None:
        er = efficiency_ratio(df["Close"], params.get("er_n", 48)).to_numpy()
        if er_max is not None:
            ok = er <= er_max
            long_ok &= ok
            short_ok &= ok
        if er_min is not None:
            ok = er >= er_min
            long_ok &= ok
            short_ok &= ok

    # higher-TF trend gate: +1 up / -1 down from 1h EMA fast-vs-slow
    tg = params.get("trend_gate")  # None | "with" | "against"
    if tg and ctx is not None:
        trend = ctx["trend_1h"]
        if tg == "with":            # longs only in uptrend, shorts only in downtrend
            long_ok &= trend > 0
            short_ok &= trend < 0
        elif tg == "against":       # fade: longs only in downtrend, etc. (contrarian)
            long_ok &= trend < 0
            short_ok &= trend > 0

    # session filter (UTC hours)
    session = params.get("session")  # None | (start_h, end_h)
    if session:
        hours = df.index.hour.to_numpy()
        s, e = session
        in_sess = (hours >= s) & (hours < e) if s < e else (hours >= s) | (hours < e)
        long_ok &= in_sess
        short_ok &= in_sess
    return long_ok, short_ok


def build_context(df_1h: pd.DataFrame, base_index: pd.DatetimeIndex,
                  base_tf_minutes: int) -> dict:
    """Precompute completion-aligned higher-TF context arrays."""
    e_fast = ema(df_1h["Close"], 50)
    e_slow = ema(df_1h["Close"], 200)
    trend = np.sign(e_fast - e_slow)
    return {
        "trend_1h": align_context(trend, 60, base_index, base_tf_minutes),
    }


# ----------------------------------------------------------------------
# Strategy families
# ----------------------------------------------------------------------

def bb_reversion(df: pd.DataFrame, params: dict, ctx: dict | None):
    """Fade z-score extremes; optional mean-touch exit."""
    z = zscore(df["Close"], params["n"]).to_numpy()
    z_in = params["z_in"]
    long_ok, short_ok = _gates(df, params, ctx)
    long_sig = (z < -z_in) & long_ok
    short_sig = (z > z_in) & short_ok
    mean_exit_long = z >= 0
    mean_exit_short = z <= 0
    return long_sig, short_sig, mean_exit_long, mean_exit_short


def rsi_reversion(df: pd.DataFrame, params: dict, ctx: dict | None):
    """Classic short-lookback RSI extreme fade."""
    r = rsi(df["Close"], params["rsi_n"]).to_numpy()
    long_ok, short_ok = _gates(df, params, ctx)
    long_sig = (r < params["lo"]) & long_ok
    short_sig = (r > params["hi"]) & short_ok
    mean_exit_long = r >= 50
    mean_exit_short = r <= 50
    return long_sig, short_sig, mean_exit_long, mean_exit_short


def vwap_reversion(df: pd.DataFrame, params: dict, ctx: dict | None):
    """Fade deviation from rolling VWAP, measured in ATR units."""
    v = rolling_vwap(df, params["n"])
    a = atr(df["High"], df["Low"], df["Close"], 14)
    dev = ((df["Close"] - v) / a.replace(0, np.nan)).to_numpy()
    d_in = params["d_in"]
    long_ok, short_ok = _gates(df, params, ctx)
    long_sig = (dev < -d_in) & long_ok
    short_sig = (dev > d_in) & short_ok
    mean_exit_long = dev >= 0
    mean_exit_short = dev <= 0
    return long_sig, short_sig, mean_exit_long, mean_exit_short


def donchian_breakout(df: pd.DataFrame, params: dict, ctx: dict | None):
    """Break of prior-N-bar extreme, with optional volatility-expansion filter."""
    hi, lo = donchian(df["High"], df["Low"], params["n"])
    close = df["Close"]
    long_sig_s = close > hi
    short_sig_s = close < lo
    long_sig = long_sig_s.fillna(False).to_numpy().copy()
    short_sig = short_sig_s.fillna(False).to_numpy().copy()

    vx = params.get("vol_expand")
    if vx is not None:
        a = atr(df["High"], df["Low"], df["Close"], 14)
        ratio = (a / sma(a, params.get("vol_n", 96)).replace(0, np.nan)).to_numpy()
        ok = ratio >= vx
        long_sig &= ok
        short_sig &= ok

    long_ok, short_ok = _gates(df, params, ctx)
    long_sig &= long_ok
    short_sig &= short_ok
    # momentum: no mean-touch exit
    n = len(df)
    return long_sig, short_sig, np.zeros(n, bool), np.zeros(n, bool)


STRATEGIES = {
    "bb_reversion": bb_reversion,
    "rsi_reversion": rsi_reversion,
    "vwap_reversion": vwap_reversion,
    "donchian_breakout": donchian_breakout,
}
