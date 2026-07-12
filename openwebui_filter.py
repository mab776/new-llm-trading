"""
title: Hybrid Trading Analyzer — Financial Data Injector
description: Multi-timeframe technical analysis with 5-category weighted scoring for crypto trading
author: LLM Trading Bot
version: 2.0.0
license: MIT
requirements: yfinance, pandas, numpy
type: filter
"""

import re
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────
# CACHING
# ──────────────────────────────────────────────────────────────────────

_data_cache: dict = {}
_CACHE_TTL = 60  # seconds


def _get_cached(key: str):
    """Return cached value if still valid, else None."""
    entry = _data_cache.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["val"]
    return None


def _set_cached(key: str, val):
    _data_cache[key] = {"val": val, "ts": time.time()}
    # Evict oldest if cache too large
    if len(_data_cache) > 50:
        oldest = min(_data_cache, key=lambda k: _data_cache[k]["ts"])
        del _data_cache[oldest]


# ──────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS — canonical implementations
# scoring.py imports these; keep self-contained for OpenWebUI copy-paste.
# ──────────────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = compute_ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr = compute_atr(high, low, close, period)
    plus_di = 100 * compute_ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * compute_ema(minus_dm, period) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = compute_ema(dx, period)
    return adx, plus_di, minus_di


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = compute_sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 0  # First value has no diff — avoid NaN cascade
    return (volume * direction).cumsum()


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / denom


def compute_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def compute_roc(series: pd.Series, period: int = 10) -> pd.Series:
    prev = series.shift(period)
    return 100 * (series - prev) / prev.replace(0, np.nan)


def compute_pivot_points(
    high: float, low: float, close: float
) -> dict[str, float]:
    """Classic pivot points from prior period."""
    pivot = (high + low + close) / 3
    return {
        "pivot": pivot,
        "support_1": 2 * pivot - high,
        "support_2": pivot - (high - low),
        "resistance_1": 2 * pivot - low,
        "resistance_2": pivot + (high - low),
    }


# ──────────────────────────────────────────────────────────────────────
# SCORING LOGIC — canonical implementations
# scoring.py imports these; keep self-contained for OpenWebUI copy-paste.
# ──────────────────────────────────────────────────────────────────────

def calc_trend_score(
    *, price=None, ema_9=None, ema_21=None, ema_50=None, ema_200=None,
    adx=None, plus_di=None, minus_di=None,
    macd_hist=None, macd_line=None, macd_signal=None,
) -> tuple[float, dict]:
    """Core trend scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0

    if not price:
        return 0.0, details

    # EMA alignment (9 > 21 > 50 = bullish, reverse = bearish)
    if ema_9 is not None and ema_21 is not None and ema_50 is not None:
        if ema_9 > ema_21 > ema_50:
            score += 30
            details["ema_alignment"] = "bullish_stack"
        elif ema_9 < ema_21 < ema_50:
            score -= 30
            details["ema_alignment"] = "bearish_stack"
        else:
            if ema_9 > ema_21:
                score += 10
            else:
                score -= 10
            details["ema_alignment"] = "mixed"

    # Price vs EMA-200 (long-term trend)
    if ema_200 is not None:
        if price > ema_200:
            score += 15
            details["vs_ema200"] = "above"
        else:
            score -= 15
            details["vs_ema200"] = "below"

    # ADX (trend strength)
    if adx is not None:
        if adx > 40:
            strength_mult = 1.0
            details["adx_strength"] = "very_strong"
        elif adx > 25:
            strength_mult = 0.7
            details["adx_strength"] = "strong"
        elif adx > 20:
            strength_mult = 0.4
            details["adx_strength"] = "moderate"
        else:
            strength_mult = 0.1
            details["adx_strength"] = "weak/ranging"

        # DI direction
        if plus_di and minus_di:
            if plus_di > minus_di:
                score += 20 * strength_mult
                details["di_direction"] = "bullish"
            else:
                score -= 20 * strength_mult
                details["di_direction"] = "bearish"

    # MACD
    if macd_hist is not None:
        if macd_hist > 0:
            score += 15
            details["macd"] = "bullish"
        else:
            score -= 15
            details["macd"] = "bearish"
        # MACD crossover signal
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:
                score += 5
            else:
                score -= 5

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


def calc_momentum_score(
    *, rsi_14=None, stoch_k=None, stoch_d=None,
    cci_20=None, williams_r=None, roc_10=None,
) -> tuple[float, dict]:
    """Core momentum scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0

    # RSI (14)
    if rsi_14 is not None:
        if rsi_14 > 70:
            score -= 20
            details["rsi"] = f"overbought ({rsi_14:.1f})"
        elif rsi_14 > 60:
            score += 15
            details["rsi"] = f"bullish ({rsi_14:.1f})"
        elif rsi_14 > 40:
            score += 0
            details["rsi"] = f"neutral ({rsi_14:.1f})"
        elif rsi_14 > 30:
            score -= 15
            details["rsi"] = f"bearish ({rsi_14:.1f})"
        else:
            score += 20
            details["rsi"] = f"oversold ({rsi_14:.1f})"

    # Stochastic
    if stoch_k is not None and stoch_d is not None:
        if stoch_k > 80:
            score -= 10
            details["stoch"] = "overbought"
        elif stoch_k < 20:
            score += 10
            details["stoch"] = "oversold"
        elif stoch_k > stoch_d:
            score += 10
            details["stoch"] = "bullish_cross"
        else:
            score -= 10
            details["stoch"] = "bearish_cross"

    # CCI
    if cci_20 is not None:
        if cci_20 > 100:
            score += 10
            details["cci"] = "strong_bullish"
        elif cci_20 < -100:
            score -= 10
            details["cci"] = "strong_bearish"

    # Williams %R
    if williams_r is not None:
        if williams_r > -20:
            score -= 10
            details["williams_r"] = "overbought"
        elif williams_r < -80:
            score += 10
            details["williams_r"] = "oversold"

    # ROC
    if roc_10 is not None:
        if roc_10 > 5:
            score += 15
            details["roc"] = f"strong_positive ({roc_10:.1f}%)"
        elif roc_10 > 0:
            score += 5
            details["roc"] = f"positive ({roc_10:.1f}%)"
        elif roc_10 > -5:
            score -= 5
            details["roc"] = f"negative ({roc_10:.1f}%)"
        else:
            score -= 15
            details["roc"] = f"strong_negative ({roc_10:.1f}%)"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


def calc_volume_score(
    *, volume_ratio=None, change_pct=None,
    obv=None, obv_sma_20=None, vwap=None, price=None,
) -> tuple[float, dict]:
    """Core volume scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0

    # Volume ratio
    if volume_ratio is not None:
        details["volume_ratio"] = round(volume_ratio, 2)
        if volume_ratio > 2.0:
            score += 30
        elif volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.0:
            score += 5
        elif volume_ratio > 0.5:
            score -= 10
        else:
            score -= 25

    # Direction: is price move supported by volume?
    if change_pct is not None and volume_ratio is not None:
        price_up = change_pct > 0
        high_vol = volume_ratio > 1.0
        if price_up and high_vol:
            score += 20
            details["vol_confirmation"] = "bullish_confirmed"
        elif not price_up and high_vol:
            score -= 20
            details["vol_confirmation"] = "bearish_confirmed"
        elif price_up and not high_vol:
            score += 0
            details["vol_confirmation"] = "bullish_unconfirmed"
        else:
            score -= 0
            details["vol_confirmation"] = "bearish_unconfirmed"

    # OBV trend
    if obv is not None and obv_sma_20 is not None:
        if obv > obv_sma_20:
            score += 15
            details["obv_trend"] = "accumulation"
        else:
            score -= 15
            details["obv_trend"] = "distribution"

    # VWAP position
    if vwap is not None and price:
        if price > vwap:
            score += 10
            details["vwap_position"] = "above"
        else:
            score -= 10
            details["vwap_position"] = "below"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


def calc_sr_score(
    *, price=None, nearest_support=None, nearest_resistance=None,
    bb_position=None,
) -> tuple[float, dict]:
    """Core support/resistance scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0

    if not price or not nearest_support or not nearest_resistance:
        return 0.0, {"status": "insufficient_data"}

    # Distance to S/R as percentage of price
    dist_support = (price - nearest_support) / price * 100
    dist_resistance = (nearest_resistance - price) / price * 100
    details["dist_to_support_pct"] = round(dist_support, 2)
    details["dist_to_resistance_pct"] = round(dist_resistance, 2)

    # Reward/risk ratio from S/R perspective
    if dist_support > 0:
        sr_ratio = dist_resistance / dist_support
        details["sr_ratio"] = round(sr_ratio, 2)
    else:
        sr_ratio = 0

    # Near support (potential bounce) — bullish bias if trend is up
    if dist_support < 1.0:
        score += 25
        details["proximity"] = "near_support"
    elif dist_resistance < 1.0:
        score -= 25
        details["proximity"] = "near_resistance"

    # Good R:R from S/R standpoint
    if sr_ratio > 3:
        score += 25
        details["sr_quality"] = "excellent"
    elif sr_ratio > 2:
        score += 15
        details["sr_quality"] = "good"
    elif sr_ratio > 1:
        score += 5
        details["sr_quality"] = "fair"
    else:
        score -= 15
        details["sr_quality"] = "poor"

    # Bollinger Band position
    if bb_position is not None:
        if bb_position > 0.95:
            score -= 15
            details["bb"] = "upper_extreme"
        elif bb_position < 0.05:
            score += 15
            details["bb"] = "lower_extreme"
        elif 0.4 < bb_position < 0.6:
            details["bb"] = "middle"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


def calc_risk_score(
    *, atr_pct=None, adx=None, bb_width=None,
) -> tuple[float, dict]:
    """Core risk scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0

    # ATR-based volatility assessment
    if atr_pct is not None:
        if atr_pct > 8:
            score -= 40
            details["volatility"] = f"extreme ({atr_pct:.1f}%)"
        elif atr_pct > 5:
            score -= 20
            details["volatility"] = f"high ({atr_pct:.1f}%)"
        elif atr_pct > 2:
            score += 10
            details["volatility"] = f"healthy ({atr_pct:.1f}%)"
        elif atr_pct > 0.5:
            score += 5
            details["volatility"] = f"moderate ({atr_pct:.1f}%)"
        else:
            score -= 30
            details["volatility"] = f"too_low ({atr_pct:.1f}%)"

    # ADX ranging check
    if adx is not None:
        if adx < 15:
            score -= 30
            details["ranging"] = "strongly_ranging"
        elif adx < 20:
            score -= 15
            details["ranging"] = "possibly_ranging"
        else:
            score += 10
            details["ranging"] = "trending"

    # BB width (squeeze detection)
    if bb_width is not None:
        if bb_width < 2:
            score -= 10
            details["bb_squeeze"] = True
        else:
            details["bb_squeeze"] = False

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


# ──────────────────────────────────────────────────────────────────────
# 4H AGGREGATION
# ──────────────────────────────────────────────────────────────────────

def _aggregate_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df["group"] = df.index.floor("4h")
    agg = df.groupby("group").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    })
    agg.index.name = "Datetime"
    return agg


# ──────────────────────────────────────────────────────────────────────
# DATA FETCHING (with cache)
# ──────────────────────────────────────────────────────────────────────

def _fetch_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with caching."""
    cache_key = f"{symbol}_{timeframe}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    import yfinance as yf

    yf_interval = "1h" if timeframe in ("1h", "4h") else "1d"
    if yf_interval == "1h":
        days = 90
    else:
        days = 365

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(
        interval=yf_interval,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )

    if df.empty:
        return df

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if timeframe == "4h":
        df = _aggregate_4h(df)

    _set_cached(cache_key, df)
    return df


# ──────────────────────────────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def _compute_analysis(df: pd.DataFrame, timeframe: str) -> dict:
    """Calculate all indicators and return a dict for the report."""
    if len(df) < 50:
        return {"error": f"Insufficient data ({len(df)} candles)"}

    c = df["Close"]
    h = df["High"]
    l = df["Low"]  # noqa: E741
    v = df["Volume"]

    result = {"timeframe": timeframe}

    # Price
    result["price"] = round(float(c.iloc[-1]), 2)
    if len(c) >= 2:
        prev = float(c.iloc[-2])
        result["change_pct"] = round((float(c.iloc[-1]) - prev) / prev * 100, 2)

    # ── Trend ──
    result["ema_9"] = round(float(compute_ema(c, 9).iloc[-1]), 2)
    result["ema_21"] = round(float(compute_ema(c, 21).iloc[-1]), 2)
    result["ema_50"] = round(float(compute_ema(c, 50).iloc[-1]), 2)
    result["sma_50"] = round(float(compute_sma(c, 50).iloc[-1]), 2)
    if len(c) >= 200:
        result["ema_200"] = round(float(compute_ema(c, 200).iloc[-1]), 2)
        result["sma_200"] = round(float(compute_sma(c, 200).iloc[-1]), 2)

    adx_s, pdi, mdi = compute_adx(h, l, c)
    if not pd.isna(adx_s.iloc[-1]):
        result["adx"] = round(float(adx_s.iloc[-1]), 1)
        result["plus_di"] = round(float(pdi.iloc[-1]), 1)
        result["minus_di"] = round(float(mdi.iloc[-1]), 1)

    ml, ms, mh = compute_macd(c)
    result["macd"] = round(float(ml.iloc[-1]), 2)
    result["macd_signal"] = round(float(ms.iloc[-1]), 2)
    result["macd_hist"] = round(float(mh.iloc[-1]), 2)

    # ── Momentum ──
    rsi = compute_rsi(c)
    result["rsi_14"] = round(float(rsi.iloc[-1]), 1)
    sk, sd = compute_stochastic(h, l, c)
    if not pd.isna(sk.iloc[-1]):
        result["stoch_k"] = round(float(sk.iloc[-1]), 1)
        result["stoch_d"] = round(float(sd.iloc[-1]), 1)

    # Williams %R
    wr = compute_williams_r(h, l, c)
    if not pd.isna(wr.iloc[-1]):
        result["williams_r"] = round(float(wr.iloc[-1]), 1)

    # CCI
    cci_s = compute_cci(h, l, c, 20)
    if not pd.isna(cci_s.iloc[-1]):
        result["cci_20"] = round(float(cci_s.iloc[-1]), 1)

    # ROC
    roc_s = compute_roc(c, 10)
    if len(c) > 10 and not pd.isna(roc_s.iloc[-1]):
        result["roc_10"] = round(float(roc_s.iloc[-1]), 2)

    # ── Volume — use last COMPLETED candle for ratio ──
    if len(v) >= 22:  # Need at least 20 + 1 completed candle
        v_completed = v.iloc[:-1]  # Exclude current (in-progress) candle
        vol_sma = compute_sma(v_completed, 20)
        if not pd.isna(vol_sma.iloc[-1]) and float(vol_sma.iloc[-1]) > 0:
            result["volume_ratio"] = round(
                float(v_completed.iloc[-1]) / float(vol_sma.iloc[-1]), 2
            )
    obv_s = compute_obv(c, v)
    obv_sma = compute_sma(obv_s, 20)
    result["obv"] = float(obv_s.iloc[-1])
    if not pd.isna(obv_sma.iloc[-1]):
        result["obv_sma_20"] = float(obv_sma.iloc[-1])
        result["obv_trend"] = (
            "accumulation" if result["obv"] > result["obv_sma_20"]
            else "distribution"
        )

    # VWAP
    vwap_s = compute_vwap(h, l, c, v)
    if not pd.isna(vwap_s.iloc[-1]):
        result["vwap"] = round(float(vwap_s.iloc[-1]), 2)

    # ── Volatility ──
    atr_s = compute_atr(h, l, c)
    result["atr_14"] = round(float(atr_s.iloc[-1]), 2)
    result["atr_pct"] = round(float(atr_s.iloc[-1]) / float(c.iloc[-1]) * 100, 2)
    bb_up, bb_mid, bb_low = compute_bollinger_bands(c)
    if not pd.isna(bb_up.iloc[-1]):
        result["bb_upper"] = round(float(bb_up.iloc[-1]), 2)
        result["bb_middle"] = round(float(bb_mid.iloc[-1]), 2)
        result["bb_lower"] = round(float(bb_low.iloc[-1]), 2)
        bb_range = float(bb_up.iloc[-1]) - float(bb_low.iloc[-1])
        if bb_range > 0:
            # Clamp to [0, 1]: on a volatility spike close can sit outside the bands.
            raw_bb_pos = (float(c.iloc[-1]) - float(bb_low.iloc[-1])) / bb_range
            result["bb_position"] = round(max(0.0, min(1.0, raw_bb_pos)), 2)

    # BB width
    if not pd.isna(bb_up.iloc[-1]):
        bb_w = (float(bb_up.iloc[-1]) - float(bb_low.iloc[-1])) / float(bb_mid.iloc[-1]) * 100
        result["bb_width"] = round(bb_w, 2)

    # ── S/R from previous candle + nearest derivation ──
    if len(df) >= 2:
        prev_bar = df.iloc[-2]
        pivots = compute_pivot_points(
            float(prev_bar["High"]), float(prev_bar["Low"]), float(prev_bar["Close"])
        )
        result["pivot"] = round(pivots["pivot"], 2)
        result["support_1"] = round(pivots["support_1"], 2)
        result["support_2"] = round(pivots["support_2"], 2)
        result["resistance_1"] = round(pivots["resistance_1"], 2)
        result["resistance_2"] = round(pivots["resistance_2"], 2)

        # Derive nearest S/R (levels below/above current price)
        price = float(c.iloc[-1])
        supports = [sv for sv in [pivots["support_1"], pivots["support_2"]] if sv < price]
        resistances = [rv for rv in [pivots["resistance_1"], pivots["resistance_2"]] if rv > price]
        result["nearest_support"] = (
            round(max(supports), 2) if supports else result["support_2"]
        )
        result["nearest_resistance"] = (
            round(min(resistances), 2) if resistances else result["resistance_2"]
        )

    return result


# ──────────────────────────────────────────────────────────────────────
# 5-CATEGORY WEIGHTED SCORING  (delegates to calc_*_score functions)
# ──────────────────────────────────────────────────────────────────────

def _score_single_timeframe(a: dict) -> dict:
    """
    Apply 5-category weighted scoring to a single-timeframe analysis dict.
    Delegates to the canonical calc_*_score functions.
    Mutates and returns ``a``.
    """
    if "error" in a:
        return a

    WEIGHTS = {
        "trend": 0.30, "momentum": 0.25, "volume": 0.15,
        "support_resistance": 0.20, "risk": 0.10,
    }

    trend_score, _ = calc_trend_score(
        price=a.get("price"),
        ema_9=a.get("ema_9"), ema_21=a.get("ema_21"), ema_50=a.get("ema_50"),
        ema_200=a.get("ema_200"),
        adx=a.get("adx"), plus_di=a.get("plus_di"), minus_di=a.get("minus_di"),
        macd_hist=a.get("macd_hist"), macd_line=a.get("macd"),
        macd_signal=a.get("macd_signal"),
    )
    mom_score, _ = calc_momentum_score(
        rsi_14=a.get("rsi_14"), stoch_k=a.get("stoch_k"), stoch_d=a.get("stoch_d"),
        cci_20=a.get("cci_20"), williams_r=a.get("williams_r"), roc_10=a.get("roc_10"),
    )
    vol_score, _ = calc_volume_score(
        volume_ratio=a.get("volume_ratio"), change_pct=a.get("change_pct"),
        obv=a.get("obv"), obv_sma_20=a.get("obv_sma_20"),
        vwap=a.get("vwap"), price=a.get("price"),
    )
    sr_score, _ = calc_sr_score(
        price=a.get("price"),
        nearest_support=a.get("nearest_support"),
        nearest_resistance=a.get("nearest_resistance"),
        bb_position=a.get("bb_position"),
    )
    risk_score, _ = calc_risk_score(
        atr_pct=a.get("atr_pct"), adx=a.get("adx"), bb_width=a.get("bb_width"),
    )

    # --- WEIGHTED COMPOSITE -----------------------------------------------
    weighted_total = (
        trend_score * WEIGHTS["trend"]
        + mom_score * WEIGHTS["momentum"]
        + vol_score * WEIGHTS["volume"]
        + sr_score * WEIGHTS["support_resistance"]
        + risk_score * WEIGHTS["risk"]
    )
    composite = max(-100, min(100, weighted_total))

    a["category_scores"] = {
        "trend": round(trend_score, 1),
        "momentum": round(mom_score, 1),
        "volume": round(vol_score, 1),
        "support_resistance": round(sr_score, 1),
        "risk": round(risk_score, 1),
    }
    a["composite_score"] = round(composite, 1)
    if composite > 10:
        a["bias"] = "BULLISH"
    elif composite < -10:
        a["bias"] = "BEARISH"
    else:
        a["bias"] = "NEUTRAL"

    return a


# ──────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME COMBINATION
# ──────────────────────────────────────────────────────────────────────

def _combine_multi_tf(analyses: list[dict], primary_tf: str = "4h") -> dict:
    """
    Produce a combined cross-timeframe score analogous to scoring.py's
    compute_composite_score() alignment-bonus logic.
    """
    primary = None
    for a in analyses:
        if a.get("timeframe") == primary_tf and "error" not in a:
            primary = a
            break
    if not primary:
        for a in analyses:
            if "error" not in a:
                primary = a
                break
    if not primary:
        return {
            "combined_score": 0, "direction": "NEUTRAL",
            "confidence": 50, "signal_strength": "WAIT",
        }

    base_score = primary.get("composite_score", 0)

    # Alignment bonus / penalty (±5 per secondary TF)
    alignment_bonus = 0.0
    alignment_notes: list[str] = []
    for a in analyses:
        if a.get("timeframe") == primary.get("timeframe") or "error" in a:
            continue
        tf_score = a.get("composite_score", 0)
        tf_name = a.get("timeframe", "?").upper()
        if (tf_score > 0 and base_score > 0) or (tf_score < 0 and base_score < 0):
            alignment_bonus += 5
            alignment_notes.append(f"{tf_name} aligns")
        elif (tf_score > 0 and base_score < 0) or (tf_score < 0 and base_score > 0):
            alignment_bonus -= 5
            alignment_notes.append(f"{tf_name} diverges")

    combined = max(-100, min(100, base_score + alignment_bonus))

    # Direction
    if combined > 10:
        direction = "LONG"
    elif combined < -10:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # Confidence: linear map |combined| → [5, 95]
    abs_score = abs(combined)
    confidence = round(5 + (abs_score / 100) * 90, 1)
    confidence = max(5.0, min(95.0, confidence))

    # Signal strength
    if abs_score >= 30:
        signal_strength = "STRONG"
    elif abs_score >= 10:
        signal_strength = "MARGINAL"
    else:
        signal_strength = "WAIT"

    return {
        "combined_score": round(combined, 1),
        "direction": direction,
        "confidence": confidence,
        "signal_strength": signal_strength,
        "alignment_bonus": round(alignment_bonus, 1),
        "alignment_notes": alignment_notes,
        "primary_tf": primary.get("timeframe", "?"),
    }


# ──────────────────────────────────────────────────────────────────────
# TARGETS
# ──────────────────────────────────────────────────────────────────────

def _compute_targets(
    price: float, atr: float, direction: str, atr_mult: float = 1.5,
) -> dict:
    """Compute ATR-based entry / SL / TP targets."""
    if direction == "NEUTRAL" or atr <= 0 or price <= 0:
        return {}
    sl_dist = atr * atr_mult
    tp1_dist = sl_dist * 2.0   # 2:1 R:R
    tp2_dist = sl_dist * 3.5   # 3.5:1 R:R
    if direction == "LONG":
        return {
            "entry": round(price, 2),
            "stop_loss": round(price - sl_dist, 2),
            "tp1": round(price + tp1_dist, 2),
            "tp2": round(price + tp2_dist, 2),
            "sl_pct": round(sl_dist / price * 100, 2),
            "rr_tp1": 2.0, "rr_tp2": 3.5,
        }
    else:  # SHORT
        return {
            "entry": round(price, 2),
            "stop_loss": round(price + sl_dist, 2),
            "tp1": round(price - tp1_dist, 2),
            "tp2": round(price - tp2_dist, 2),
            "sl_pct": round(sl_dist / price * 100, 2),
            "rr_tp1": 2.0, "rr_tp2": 3.5,
        }


# ──────────────────────────────────────────────────────────────────────
# REPORT FORMATTING
# ──────────────────────────────────────────────────────────────────────

def _format_analysis_text(
    analyses: list[dict], combined: dict, targets: dict, symbol: str,
) -> str:
    """Format multi-timeframe analysis into a readable injection block."""
    lines = [
        "=" * 60,
        f"FINANCIAL DATA FOR {symbol}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        "WARNING: These are real calculated values. Do NOT modify or invent numbers.",
        "=" * 60,
    ]

    # ── Combined decision ──
    lines.append("")
    lines.append("━━━━━ COMBINED MULTI-TIMEFRAME DECISION ━━━━━")
    lines.append(f"  Direction: {combined.get('direction', 'N/A')}")
    lines.append(f"  Signal Strength: {combined.get('signal_strength', 'N/A')}")
    lines.append(f"  Combined Score: {combined.get('combined_score', 0):+.1f}")
    lines.append(f"  Confidence: {combined.get('confidence', 50):.1f}%")
    lines.append(f"  Primary TF: {combined.get('primary_tf', '?').upper()}")
    if combined.get("alignment_notes"):
        lines.append(
            f"  TF Alignment: {', '.join(combined['alignment_notes'])} "
            f"(bonus: {combined.get('alignment_bonus', 0):+.1f})"
        )

    # ── Targets ──
    if targets:
        lines.append("")
        lines.append("━━━━━ KEY LEVELS (ATR-based) ━━━━━")
        lines.append(f"  Entry: ${targets['entry']:,.2f}")
        lines.append(f"  Stop Loss: ${targets['stop_loss']:,.2f} ({targets['sl_pct']:.2f}% risk)")
        lines.append(f"  TP1: ${targets['tp1']:,.2f} (R:R {targets['rr_tp1']}:1)")
        lines.append(f"  TP2: ${targets['tp2']:,.2f} (R:R {targets['rr_tp2']}:1)")

    # ── Per-timeframe detail ──
    for a in analyses:
        if "error" in a:
            lines.append(f"\n[{a['timeframe'].upper()}] Error: {a['error']}")
            continue

        tf = a.get("timeframe", "?").upper()
        lines.append(f"\n{'─' * 40}")
        lines.append(f"  {tf} TIMEFRAME")
        lines.append(f"{'─' * 40}")
        lines.append(f"  Price: ${a.get('price', 0):,.2f} ({a.get('change_pct', 0):+.2f}%)")
        comp = a.get("composite_score", 0)
        lines.append(f"  Bias: {a.get('bias', 'N/A')} (composite: {comp:+.1f})")
        cats = a.get("category_scores", {})
        if cats:
            lines.append(
                f"  Scores: trend={cats.get('trend', 0):+.0f}×0.30  "
                f"momentum={cats.get('momentum', 0):+.0f}×0.25  "
                f"volume={cats.get('volume', 0):+.0f}×0.15  "
                f"S/R={cats.get('support_resistance', 0):+.0f}×0.20  "
                f"risk={cats.get('risk', 0):+.0f}×0.10"
            )

        lines.append(f"\n  Trend:")
        lines.append(
            f"    EMA 9/21/50: ${a.get('ema_9', 0):,.2f} / "
            f"${a.get('ema_21', 0):,.2f} / ${a.get('ema_50', 0):,.2f}"
        )
        if "sma_50" in a:
            lines.append(f"    SMA 50: ${a['sma_50']:,.2f}")
        if "ema_200" in a:
            lines.append(f"    EMA 200: ${a['ema_200']:,.2f}")
        if "sma_200" in a:
            lines.append(f"    SMA 200: ${a['sma_200']:,.2f}")
        if "adx" in a:
            lines.append(
                f"    ADX: {a['adx']:.1f} "
                f"(+DI: {a.get('plus_di', 0):.1f}, -DI: {a.get('minus_di', 0):.1f})"
            )
        lines.append(
            f"    MACD: {a.get('macd', 0):.2f} / Signal: "
            f"{a.get('macd_signal', 0):.2f} / Hist: {a.get('macd_hist', 0):.2f}"
        )

        lines.append(f"\n  Momentum:")
        lines.append(f"    RSI(14): {a.get('rsi_14', 0):.1f}")
        if "stoch_k" in a:
            lines.append(f"    Stochastic K/D: {a['stoch_k']:.1f} / {a['stoch_d']:.1f}")
        if "williams_r" in a:
            lines.append(f"    Williams %R: {a['williams_r']:.1f}")
        if "cci_20" in a:
            lines.append(f"    CCI(20): {a['cci_20']:.1f}")
        if "roc_10" in a:
            lines.append(f"    ROC(10): {a['roc_10']:.2f}%")

        lines.append(f"\n  Volume:")
        if "volume_ratio" in a:
            lines.append(f"    Volume Ratio: {a['volume_ratio']:.2f}x average")
        if "obv_trend" in a:
            lines.append(f"    OBV Trend: {a['obv_trend']}")
        if "vwap" in a:
            pos = "above" if a.get("price", 0) > a["vwap"] else "below"
            lines.append(f"    VWAP: ${a['vwap']:,.2f} (price {pos})")

        lines.append(f"\n  Volatility:")
        lines.append(
            f"    ATR(14): ${a.get('atr_14', 0):,.2f} ({a.get('atr_pct', 0):.2f}%)"
        )
        if "bb_upper" in a:
            lines.append(
                f"    BB: ${a['bb_lower']:,.2f} / "
                f"${a['bb_middle']:,.2f} / ${a['bb_upper']:,.2f}"
            )
            if "bb_position" in a:
                lines.append(f"    BB Position: {a['bb_position']:.2f}")
            if "bb_width" in a:
                lines.append(f"    BB Width: {a['bb_width']:.2f}%")

        if "pivot" in a:
            lines.append(f"\n  Support/Resistance:")
            lines.append(f"    Pivot: ${a['pivot']:,.2f}")
            lines.append(f"    S1: ${a['support_1']:,.2f}  S2: ${a['support_2']:,.2f}")
            lines.append(f"    R1: ${a['resistance_1']:,.2f}  R2: ${a['resistance_2']:,.2f}")
            if "nearest_support" in a:
                lines.append(
                    f"    Nearest Support: ${a['nearest_support']:,.2f}  "
                    f"Nearest Resistance: ${a['nearest_resistance']:,.2f}"
                )

    lines.append(f"\n{'=' * 60}")
    lines.append("END FINANCIAL DATA")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# LLM INSTRUCTION PROMPT
# ──────────────────────────────────────────────────────────────────────

_LLM_INSTRUCTIONS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CRITICAL INSTRUCTIONS — READ CAREFULLY  ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE WEIGHTED SCORING IS ALREADY CALCULATED ABOVE.

Look at the "COMBINED MULTI-TIMEFRAME DECISION" section:
- Direction and Signal Strength are already determined
- Confidence level is already calculated
- Category scores and breakdown are provided
- Key levels (entry/SL/TP) are pre-calculated using ATR

YOUR JOB (≤150 WORDS):
1. Review the score (is it reasonable given the data?)
2. Note key factors and conflicts across timeframes
3. Accept or adjust slightly (±10% confidence max)
4. Use the pre-calculated entry/stop/targets from the report
5. Output in the required format below

DO NOT:
❌ Recalculate the score from scratch
❌ Manually go through each indicator one by one
❌ Write more than 150 words in thinking
❌ Explain the scoring system
❌ Invent prices or levels not in the data

DO:
✅ Use the pre-calculated score and targets
✅ Verify they make sense given the raw indicators
✅ Be concise (5 lines max for thinking)
✅ Focus on the single key insight

OUTPUT FORMAT (REQUIRED):

DECISION: [BUY / SHORT / WAIT]
CONFIDENCE: [X]%

KEY LEVELS:
• Entry: $X
• Stop Loss: $X (X% risk)
• TP1: $X (R:R X:1)
• TP2: $X (R:R X:1)

REASONING: [2-3 sentences max explaining why]

CONDITIONS TO INVALIDATE:
[If SHORT: "Close if price > $X" / If BUY: "Close if price < $X"]

NEXT CHECK: [Specific timeframe, e.g., "4H candle close"]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ──────────────────────────────────────────────────────────────────────
# OPENWEBUI FILTER CLASS
# ──────────────────────────────────────────────────────────────────────

class Filter:
    """
    OpenWebUI inlet/outlet filter for Financial Data Injection.

    - Inlet: intercepts user messages, detects market-related queries,
      fetches real-time data, calculates indicators, and injects the
      pre-scored analysis into the message before the LLM sees it.
    - Outlet: prepends the raw financial data as a collapsible HTML
      section in the assistant response so the user can see it.
    """

    class Valves(BaseModel):
        """Filter configuration exposed in OpenWebUI settings."""

        enabled: bool = Field(
            default=True,
            description="Enable/disable financial data injection",
        )
        symbol: str = Field(
            default="BTC-USD",
            description="Default Yahoo Finance ticker symbol",
        )
        timeframes: str = Field(
            default="1h,4h,1d",
            description="Comma-separated timeframes to analyze",
        )
        primary_timeframe: str = Field(
            default="4h",
            description="Primary timeframe for combined scoring",
        )
        always_inject: bool = Field(
            default=False,
            description="Inject data into every message (not just trading queries)",
        )
        trigger_keywords: str = Field(
            default=(
                "btc,bitcoin,crypto,market,analysis,trade,trading,bullish,"
                "bearish,long,short,price,signal,entry,target,support,"
                "resistance,chart,technical,think,manual,what about"
            ),
            description="Comma-separated keywords that trigger data injection",
        )
        atr_multiplier: float = Field(
            default=1.5,
            ge=1.0,
            le=3.0,
            description="Stop-loss distance as ATR multiple",
        )
        priority: int = Field(
            default=0,
            description="Filter priority",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.last_financial_data: Optional[str] = None
        self.last_was_analysis: bool = False

    def _extract_ticker(self, text: str) -> str:
        """Extract ticker from user message, defaulting to configured symbol."""
        text_upper = text.upper()
        common_cryptos = {
            "BITCOIN": "BTC-USD", "BTC": "BTC-USD",
            "ETHEREUM": "ETH-USD", "ETH": "ETH-USD",
            "SOLANA": "SOL-USD", "SOL": "SOL-USD",
            "CARDANO": "ADA-USD", "ADA": "ADA-USD",
            "RIPPLE": "XRP-USD", "XRP": "XRP-USD",
            "DOGECOIN": "DOGE-USD", "DOGE": "DOGE-USD",
            "POLKADOT": "DOT-USD", "DOT": "DOT-USD",
            "BINANCE": "BNB-USD", "BNB": "BNB-USD",
        }
        for name, ticker in common_cryptos.items():
            if name in text_upper:
                return ticker

        # Check for explicit XXX-USD pattern
        ticker_match = re.search(r"\b([A-Z]{2,5})-USD\b", text_upper)
        if ticker_match:
            return ticker_match.group(0)

        return self.valves.symbol

    def _should_inject(self, message: str) -> bool:
        """Check if the message warrants data injection."""
        if self.valves.always_inject:
            return True
        if not self.valves.enabled:
            return False
        msg_lower = message.lower()
        keywords = [k.strip() for k in self.valves.trigger_keywords.split(",")]
        return any(kw in msg_lower for kw in keywords)

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Intercept the user message and inject financial data if relevant."""
        try:
            messages = body.get("messages", [])
            if not messages:
                return body

            last_message = messages[-1]
            if last_message.get("role") != "user":
                return body

            user_content = last_message.get("content", "")

            # Skip format-only requests
            if "output only this format" in user_content.lower():
                self.last_was_analysis = False
                return body

            if not self._should_inject(user_content):
                self.last_was_analysis = False
                return body

            # Detect ticker from message
            symbol = self._extract_ticker(user_content)

            # Fetch and analyze each timeframe
            timeframes = [t.strip() for t in self.valves.timeframes.split(",")]
            analyses: list[dict] = []
            for tf in timeframes:
                try:
                    df = _fetch_data(symbol, tf)
                    if not df.empty:
                        analysis = _compute_analysis(df, tf)
                        _score_single_timeframe(analysis)
                        analyses.append(analysis)
                except Exception as e:
                    analyses.append({"timeframe": tf, "error": str(e)})

            if not analyses:
                return body

            # Combine multi-timeframe scores
            combined = _combine_multi_tf(analyses, self.valves.primary_timeframe)

            # Compute entry / SL / TP targets
            primary_analysis = None
            for a in analyses:
                if a.get("timeframe") == self.valves.primary_timeframe and "error" not in a:
                    primary_analysis = a
                    break
            if not primary_analysis:
                for a in analyses:
                    if "error" not in a:
                        primary_analysis = a
                        break

            targets: dict = {}
            if primary_analysis:
                targets = _compute_targets(
                    primary_analysis.get("price", 0),
                    primary_analysis.get("atr_14", 0),
                    combined.get("direction", "NEUTRAL"),
                    self.valves.atr_multiplier,
                )

            # Format the report
            injection = _format_analysis_text(analyses, combined, targets, symbol)

            # Store for outlet display
            self.last_financial_data = injection
            self.last_was_analysis = True

            # Inject data + instructions into the user message
            enhanced_content = (
                f"{injection}\n{_LLM_INSTRUCTIONS}\nUSER REQUEST: {user_content}"
            )
            messages[-1]["content"] = enhanced_content
            body["messages"] = messages

        except Exception as e:
            # Never break the chat — silently fail
            print(f"[Financial Filter] Error: {e}")

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Post-process the assistant response.
        Prepends the raw financial data as a collapsible HTML block so
        the user can see what data the LLM used for its analysis.
        """
        messages = body.get("messages", [])
        if not messages:
            return body

        last_message = messages[-1]
        if last_message.get("role") != "assistant":
            return body

        assistant_content = last_message.get("content", "")

        # Prepend data if available and not already there
        if (
            self.last_financial_data
            and "Data Used for Analysis" not in assistant_content
        ):
            data_section = (
                "<details>\n"
                "<summary>📊 Data Used for Analysis (Click to Expand)</summary>\n\n"
                f"```\n{self.last_financial_data}\n```\n"
                "</details>\n\n---\n\n"
            )
            messages[-1]["content"] = data_section + assistant_content
            body["messages"] = messages

        # Reset flags
        self.last_was_analysis = False
        self.last_financial_data = None

        return body
