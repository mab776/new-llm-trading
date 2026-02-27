"""
Core scoring engine — single source of truth for all technical calculations.

All indicator computations, scoring logic, target calculations, and pre-trade
filters live here. Every other module imports from this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalStrength(str, Enum):
    STRONG = "STRONG"
    MARGINAL = "MARGINAL"
    WAIT = "WAIT"


@dataclass
class IndicatorSet:
    """All calculated indicators for a single timeframe."""
    timeframe: str

    # Trend
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Momentum
    rsi_14: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    cci_20: Optional[float] = None
    williams_r: Optional[float] = None
    roc_10: Optional[float] = None

    # Volume
    volume: Optional[float] = None
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    obv: Optional[float] = None
    obv_sma_20: Optional[float] = None
    vwap: Optional[float] = None

    # Volatility
    atr_14: Optional[float] = None
    atr_pct: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_position: Optional[float] = None

    # Support / Resistance
    pivot: Optional[float] = None
    support_1: Optional[float] = None
    support_2: Optional[float] = None
    resistance_1: Optional[float] = None
    resistance_2: Optional[float] = None
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None

    # Price context
    close: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    change_pct: Optional[float] = None


@dataclass
class CategoryScore:
    """Score for a single category (trend, momentum, etc.)."""
    name: str
    raw_score: float  # -100 to +100
    weight: float
    weighted_score: float
    details: dict = field(default_factory=dict)


@dataclass
class ScoringResult:
    """Complete scoring output."""
    direction: Direction
    confidence: float  # bounded [confidence_min, confidence_max]
    signal_strength: SignalStrength
    raw_score: float  # -100 to +100
    category_scores: list[CategoryScore] = field(default_factory=list)
    indicators: dict[str, IndicatorSet] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    filter_failures: list[str] = field(default_factory=list)
    passed_filters: bool = True


@dataclass
class TradeTargets:
    """Entry, SL, TP levels."""
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_amount: float  # distance from entry to SL
    reward_1: float  # distance from entry to TP1
    reward_2: float  # distance from entry to TP2
    direction: Direction
    sl_strategy: str = "hybrid"


# ──────────────────────────────────────────────────────────────────────
# Indicator Calculations
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
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
    # Zero out when the other is larger
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


def compute_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / denom


def compute_roc(series: pd.Series, period: int = 10) -> pd.Series:
    prev = series.shift(period)
    return 100 * (series - prev) / prev.replace(0, np.nan)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    tp = (high + low + close) / 3
    cum_tp_vol = (tp * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = compute_sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


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
# Full Indicator Computation
# ──────────────────────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame, timeframe: str) -> IndicatorSet:
    """
    Calculate all indicators from an OHLCV DataFrame.
    Expects columns: Open, High, Low, Close, Volume.
    Returns the latest values as an IndicatorSet.
    """
    if len(df) < 50:
        raise ValueError(f"Need at least 50 candles, got {len(df)}")

    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    ind = IndicatorSet(timeframe=timeframe)

    # Price context
    ind.close = float(c.iloc[-1])
    ind.open = float(o.iloc[-1])
    ind.high = float(h.iloc[-1])
    ind.low = float(l.iloc[-1])
    if len(c) >= 2:
        prev = c.iloc[-2]
        ind.change_pct = float((c.iloc[-1] - prev) / prev * 100) if prev != 0 else 0

    # Trend
    ind.ema_9 = float(compute_ema(c, 9).iloc[-1])
    ind.ema_21 = float(compute_ema(c, 21).iloc[-1])
    ind.ema_50 = float(compute_ema(c, 50).iloc[-1])
    if len(c) >= 200:
        ind.ema_200 = float(compute_ema(c, 200).iloc[-1])
        ind.sma_200 = float(compute_sma(c, 200).iloc[-1])
    ind.sma_50 = float(compute_sma(c, 50).iloc[-1])

    adx_s, pdi_s, mdi_s = compute_adx(h, l, c)
    ind.adx = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else None
    ind.plus_di = float(pdi_s.iloc[-1]) if not pd.isna(pdi_s.iloc[-1]) else None
    ind.minus_di = float(mdi_s.iloc[-1]) if not pd.isna(mdi_s.iloc[-1]) else None

    macd_l, macd_s, macd_h = compute_macd(c)
    ind.macd_line = float(macd_l.iloc[-1])
    ind.macd_signal = float(macd_s.iloc[-1])
    ind.macd_histogram = float(macd_h.iloc[-1])

    # Momentum
    ind.rsi_14 = float(compute_rsi(c, 14).iloc[-1])
    sk, sd = compute_stochastic(h, l, c)
    ind.stoch_k = float(sk.iloc[-1]) if not pd.isna(sk.iloc[-1]) else None
    ind.stoch_d = float(sd.iloc[-1]) if not pd.isna(sd.iloc[-1]) else None
    ind.cci_20 = float(compute_cci(h, l, c, 20).iloc[-1]) if not pd.isna(
        compute_cci(h, l, c, 20).iloc[-1]
    ) else None
    ind.williams_r = float(compute_williams_r(h, l, c).iloc[-1])
    ind.roc_10 = float(compute_roc(c, 10).iloc[-1]) if not pd.isna(
        compute_roc(c, 10).iloc[-1]
    ) else None

    # Volume
    ind.volume = float(v.iloc[-1])
    vol_sma = compute_sma(v, 20)
    ind.volume_sma_20 = float(vol_sma.iloc[-1]) if not pd.isna(vol_sma.iloc[-1]) else None
    if ind.volume_sma_20 and ind.volume_sma_20 > 0:
        ind.volume_ratio = ind.volume / ind.volume_sma_20
    obv_s = compute_obv(c, v)
    ind.obv = float(obv_s.iloc[-1])
    obv_sma = compute_sma(obv_s, 20)
    ind.obv_sma_20 = float(obv_sma.iloc[-1]) if not pd.isna(obv_sma.iloc[-1]) else None
    vwap_s = compute_vwap(h, l, c, v)
    ind.vwap = float(vwap_s.iloc[-1]) if not pd.isna(vwap_s.iloc[-1]) else None

    # Volatility
    atr_s = compute_atr(h, l, c, 14)
    ind.atr_14 = float(atr_s.iloc[-1])
    ind.atr_pct = float(ind.atr_14 / ind.close * 100) if ind.close else None
    bb_up, bb_mid, bb_low = compute_bollinger_bands(c)
    ind.bb_upper = float(bb_up.iloc[-1]) if not pd.isna(bb_up.iloc[-1]) else None
    ind.bb_middle = float(bb_mid.iloc[-1]) if not pd.isna(bb_mid.iloc[-1]) else None
    ind.bb_lower = float(bb_low.iloc[-1]) if not pd.isna(bb_low.iloc[-1]) else None
    if ind.bb_upper and ind.bb_lower and ind.bb_middle and ind.bb_middle > 0:
        ind.bb_width = (ind.bb_upper - ind.bb_lower) / ind.bb_middle * 100
    if ind.bb_upper and ind.bb_lower and (ind.bb_upper - ind.bb_lower) > 0:
        ind.bb_position = (ind.close - ind.bb_lower) / (ind.bb_upper - ind.bb_lower)

    # Support / Resistance (from prior candle)
    if len(df) >= 2:
        prev_row = df.iloc[-2]
        pivots = compute_pivot_points(
            float(prev_row["High"]), float(prev_row["Low"]), float(prev_row["Close"])
        )
        ind.pivot = pivots["pivot"]
        ind.support_1 = pivots["support_1"]
        ind.support_2 = pivots["support_2"]
        ind.resistance_1 = pivots["resistance_1"]
        ind.resistance_2 = pivots["resistance_2"]

        # Find nearest S/R
        price = ind.close
        supports = [s for s in [ind.support_1, ind.support_2] if s and s < price]
        resistances = [r for r in [ind.resistance_1, ind.resistance_2] if r and r > price]
        ind.nearest_support = max(supports) if supports else ind.support_2
        ind.nearest_resistance = min(resistances) if resistances else ind.resistance_2

    return ind


# ──────────────────────────────────────────────────────────────────────
# Scoring Categories
# ──────────────────────────────────────────────────────────────────────

def score_trend(indicators: IndicatorSet) -> CategoryScore:
    """
    Score trend from -100 (strong bearish) to +100 (strong bullish).
    Uses EMA alignment, ADX, MACD, and price vs key MAs.
    """
    details: dict = {}
    score = 0.0

    price = indicators.close
    if not price:
        return CategoryScore("trend", 0, 0, 0, details)

    # EMA alignment (9 > 21 > 50 = bullish, reverse = bearish)
    if indicators.ema_9 and indicators.ema_21 and indicators.ema_50:
        if indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
            score += 30
            details["ema_alignment"] = "bullish_stack"
        elif indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
            score -= 30
            details["ema_alignment"] = "bearish_stack"
        else:
            # Partial alignment
            if indicators.ema_9 > indicators.ema_21:
                score += 10
            else:
                score -= 10
            details["ema_alignment"] = "mixed"

    # Price vs EMA-200 (long-term trend)
    if indicators.ema_200:
        if price > indicators.ema_200:
            score += 15
            details["vs_ema200"] = "above"
        else:
            score -= 15
            details["vs_ema200"] = "below"

    # ADX (trend strength)
    if indicators.adx is not None:
        if indicators.adx > 40:
            strength_mult = 1.0
            details["adx_strength"] = "very_strong"
        elif indicators.adx > 25:
            strength_mult = 0.7
            details["adx_strength"] = "strong"
        elif indicators.adx > 20:
            strength_mult = 0.4
            details["adx_strength"] = "moderate"
        else:
            strength_mult = 0.1
            details["adx_strength"] = "weak/ranging"

        # DI direction
        if indicators.plus_di and indicators.minus_di:
            if indicators.plus_di > indicators.minus_di:
                score += 20 * strength_mult
                details["di_direction"] = "bullish"
            else:
                score -= 20 * strength_mult
                details["di_direction"] = "bearish"

    # MACD
    if indicators.macd_histogram is not None:
        if indicators.macd_histogram > 0:
            score += 15
            details["macd"] = "bullish"
        else:
            score -= 15
            details["macd"] = "bearish"
        # MACD crossover signal
        if indicators.macd_line is not None and indicators.macd_signal is not None:
            if indicators.macd_line > indicators.macd_signal:
                score += 5
            else:
                score -= 5

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return CategoryScore("trend", score, 0, 0, details)


def score_momentum(indicators: IndicatorSet) -> CategoryScore:
    """
    Score momentum from -100 to +100.
    Uses RSI, Stochastic, CCI, Williams %R, ROC.
    """
    details: dict = {}
    score = 0.0

    # RSI (14)
    if indicators.rsi_14 is not None:
        rsi = indicators.rsi_14
        if rsi > 70:
            score -= 20  # Overbought — bearish pressure
            details["rsi"] = f"overbought ({rsi:.1f})"
        elif rsi > 60:
            score += 15  # Bullish momentum
            details["rsi"] = f"bullish ({rsi:.1f})"
        elif rsi > 40:
            score += 0  # Neutral
            details["rsi"] = f"neutral ({rsi:.1f})"
        elif rsi > 30:
            score -= 15
            details["rsi"] = f"bearish ({rsi:.1f})"
        else:
            score += 20  # Oversold — bullish reversal area
            details["rsi"] = f"oversold ({rsi:.1f})"

    # Stochastic
    if indicators.stoch_k is not None and indicators.stoch_d is not None:
        if indicators.stoch_k > 80:
            score -= 10  # Overbought
            details["stoch"] = "overbought"
        elif indicators.stoch_k < 20:
            score += 10  # Oversold
            details["stoch"] = "oversold"
        elif indicators.stoch_k > indicators.stoch_d:
            score += 10
            details["stoch"] = "bullish_cross"
        else:
            score -= 10
            details["stoch"] = "bearish_cross"

    # CCI
    if indicators.cci_20 is not None:
        if indicators.cci_20 > 100:
            score += 10
            details["cci"] = "strong_bullish"
        elif indicators.cci_20 < -100:
            score -= 10
            details["cci"] = "strong_bearish"

    # Williams %R
    if indicators.williams_r is not None:
        if indicators.williams_r > -20:
            score -= 10  # Overbought
            details["williams_r"] = "overbought"
        elif indicators.williams_r < -80:
            score += 10  # Oversold
            details["williams_r"] = "oversold"

    # ROC
    if indicators.roc_10 is not None:
        if indicators.roc_10 > 5:
            score += 15
            details["roc"] = f"strong_positive ({indicators.roc_10:.1f}%)"
        elif indicators.roc_10 > 0:
            score += 5
            details["roc"] = f"positive ({indicators.roc_10:.1f}%)"
        elif indicators.roc_10 > -5:
            score -= 5
            details["roc"] = f"negative ({indicators.roc_10:.1f}%)"
        else:
            score -= 15
            details["roc"] = f"strong_negative ({indicators.roc_10:.1f}%)"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return CategoryScore("momentum", score, 0, 0, details)


def score_volume(indicators: IndicatorSet) -> CategoryScore:
    """
    Score volume confirmation from -100 to +100.
    High volume in trend direction = confirmation.
    """
    details: dict = {}
    score = 0.0

    # Volume ratio
    if indicators.volume_ratio is not None:
        vr = indicators.volume_ratio
        details["volume_ratio"] = round(vr, 2)
        if vr > 2.0:
            score += 30  # Very high volume
        elif vr > 1.5:
            score += 20
        elif vr > 1.0:
            score += 5
        elif vr > 0.5:
            score -= 10  # Below average volume
        else:
            score -= 25  # Very low volume — unreliable signal

    # Direction: is price move supported by volume?
    if indicators.change_pct is not None and indicators.volume_ratio is not None:
        price_up = indicators.change_pct > 0
        high_vol = indicators.volume_ratio > 1.0
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
    if indicators.obv is not None and indicators.obv_sma_20 is not None:
        if indicators.obv > indicators.obv_sma_20:
            score += 15
            details["obv_trend"] = "accumulation"
        else:
            score -= 15
            details["obv_trend"] = "distribution"

    # VWAP position
    if indicators.vwap is not None and indicators.close is not None:
        if indicators.close > indicators.vwap:
            score += 10
            details["vwap_position"] = "above"
        else:
            score -= 10
            details["vwap_position"] = "below"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return CategoryScore("volume", score, 0, 0, details)


def score_support_resistance(indicators: IndicatorSet) -> CategoryScore:
    """
    Score S/R proximity from -100 to +100.
    Near support in uptrend = bullish. Near resistance in uptrend = caution.
    """
    details: dict = {}
    score = 0.0
    price = indicators.close

    if not price or not indicators.nearest_support or not indicators.nearest_resistance:
        return CategoryScore("support_resistance", 0, 0, 0, {"status": "insufficient_data"})

    # Distance to S/R as percentage of price
    dist_support = (price - indicators.nearest_support) / price * 100
    dist_resistance = (indicators.nearest_resistance - price) / price * 100
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
    if indicators.bb_position is not None:
        bbp = indicators.bb_position
        if bbp > 0.95:
            score -= 15  # At upper band — potential reversal
            details["bb"] = "upper_extreme"
        elif bbp < 0.05:
            score += 15  # At lower band — potential bounce
            details["bb"] = "lower_extreme"
        elif 0.4 < bbp < 0.6:
            details["bb"] = "middle"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return CategoryScore("support_resistance", score, 0, 0, details)


def score_risk(indicators: IndicatorSet) -> CategoryScore:
    """
    Risk assessment from -100 (high risk, penalize) to +100 (low risk).
    Looks at volatility extremes, ranging markets, and divergences.
    """
    details: dict = {}
    score = 0.0  # Start neutral, deduct for risks

    # ATR-based volatility assessment
    if indicators.atr_pct is not None:
        atr_pct = indicators.atr_pct
        if atr_pct > 8:
            score -= 40  # Extreme volatility
            details["volatility"] = f"extreme ({atr_pct:.1f}%)"
        elif atr_pct > 5:
            score -= 20
            details["volatility"] = f"high ({atr_pct:.1f}%)"
        elif atr_pct > 2:
            score += 10  # Good volatility for trading
            details["volatility"] = f"healthy ({atr_pct:.1f}%)"
        elif atr_pct > 0.5:
            score += 5
            details["volatility"] = f"moderate ({atr_pct:.1f}%)"
        else:
            score -= 30  # Too low — no profit potential
            details["volatility"] = f"too_low ({atr_pct:.1f}%)"

    # ADX ranging check
    if indicators.adx is not None:
        if indicators.adx < 15:
            score -= 30
            details["ranging"] = "strongly_ranging"
        elif indicators.adx < 20:
            score -= 15
            details["ranging"] = "possibly_ranging"
        else:
            score += 10
            details["ranging"] = "trending"

    # BB width (squeeze detection)
    if indicators.bb_width is not None:
        if indicators.bb_width < 2:
            score -= 10
            details["bb_squeeze"] = True
        else:
            details["bb_squeeze"] = False

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return CategoryScore("risk", score, 0, 0, details)


# ──────────────────────────────────────────────────────────────────────
# Combined Scoring
# ──────────────────────────────────────────────────────────────────────

def compute_composite_score(
    indicators_by_tf: dict[str, IndicatorSet],
    weights: dict[str, float],
    primary_timeframe: str = "4h",
    confidence_min: float = 5,
    confidence_max: float = 95,
) -> ScoringResult:
    """
    Compute a composite score across multiple timeframes.
    Primary timeframe gets full weight, others contribute via alignment bonus/penalty.
    """
    primary = indicators_by_tf.get(primary_timeframe)
    if not primary:
        # Fall back to first available
        primary = next(iter(indicators_by_tf.values()))

    # Score each category on primary timeframe
    cat_funcs = {
        "trend": score_trend,
        "momentum": score_momentum,
        "volume": score_volume,
        "support_resistance": score_support_resistance,
        "risk": score_risk,
    }

    category_scores: list[CategoryScore] = []
    weighted_total = 0.0
    reasons: list[str] = []

    for name, func in cat_funcs.items():
        w = weights.get(name, 0.0)
        cat = func(primary)
        cat.weight = w
        cat.weighted_score = cat.raw_score * w
        weighted_total += cat.weighted_score
        category_scores.append(cat)

        # Build reason string
        if abs(cat.raw_score) > 30:
            direction_word = "bullish" if cat.raw_score > 0 else "bearish"
            reasons.append(f"{name}: {direction_word} ({cat.raw_score:+.0f})")

    # Multi-timeframe alignment bonus/penalty (up to ±15 points)
    alignment_bonus = 0.0
    for tf, ind in indicators_by_tf.items():
        if tf == primary_timeframe:
            continue
        tf_trend = score_trend(ind)
        # If secondary timeframe agrees with primary, small bonus
        if (tf_trend.raw_score > 0 and weighted_total > 0) or (
            tf_trend.raw_score < 0 and weighted_total < 0
        ):
            alignment_bonus += 5
            reasons.append(f"{tf} trend aligns")
        elif (tf_trend.raw_score > 0 and weighted_total < 0) or (
            tf_trend.raw_score < 0 and weighted_total > 0
        ):
            alignment_bonus -= 5
            reasons.append(f"{tf} trend diverges")

    raw_score = max(-100, min(100, weighted_total + alignment_bonus))

    # Direction
    if raw_score > 10:
        direction = Direction.BULLISH
    elif raw_score < -10:
        direction = Direction.BEARISH
    else:
        direction = Direction.NEUTRAL

    # Confidence: map |raw_score| (0-100) → [confidence_min, confidence_max]
    abs_score = abs(raw_score)
    confidence = confidence_min + (confidence_max - confidence_min) * (abs_score / 100)
    confidence = max(confidence_min, min(confidence_max, confidence))

    return ScoringResult(
        direction=direction,
        confidence=round(confidence, 1),
        signal_strength=SignalStrength.WAIT,  # Set by router
        raw_score=round(raw_score, 2),
        category_scores=category_scores,
        indicators=indicators_by_tf,
        reasons=reasons,
    )


# ──────────────────────────────────────────────────────────────────────
# Target Calculation
# ──────────────────────────────────────────────────────────────────────

def calculate_targets(
    indicators: IndicatorSet,
    direction: Direction,
    sl_strategy: str = "hybrid",
    atr_sl_mult: float = 1.5,
    atr_tp1_mult: float = 3.0,
    atr_tp2_mult: float = 5.0,
) -> Optional[TradeTargets]:
    """
    Calculate entry, SL, TP1, TP2 based on ATR and optionally S/R levels.
    Returns None if direction is NEUTRAL.
    """
    if direction == Direction.NEUTRAL:
        return None

    price = indicators.close
    atr = indicators.atr_14
    if not price or not atr or atr == 0:
        return None

    is_long = direction == Direction.BULLISH
    entry = price

    # Stop Loss calculation
    if sl_strategy == "atr":
        sl_distance = atr * atr_sl_mult
    elif sl_strategy == "structure":
        # Use nearest S/R as SL
        if is_long and indicators.nearest_support:
            sl_distance = price - indicators.nearest_support
            sl_distance = max(sl_distance, atr * 0.5)  # Minimum SL
        elif not is_long and indicators.nearest_resistance:
            sl_distance = indicators.nearest_resistance - price
            sl_distance = max(sl_distance, atr * 0.5)
        else:
            sl_distance = atr * atr_sl_mult
    elif sl_strategy == "hybrid":
        atr_sl = atr * atr_sl_mult
        if is_long and indicators.nearest_support:
            structure_sl = price - indicators.nearest_support
            sl_distance = max(min(atr_sl, structure_sl * 1.1), atr * 0.5)
        elif not is_long and indicators.nearest_resistance:
            structure_sl = indicators.nearest_resistance - price
            sl_distance = max(min(atr_sl, structure_sl * 1.1), atr * 0.5)
        else:
            sl_distance = atr_sl
    else:
        sl_distance = atr * atr_sl_mult

    if is_long:
        stop_loss = entry - sl_distance
        tp1 = entry + sl_distance * atr_tp1_mult / atr_sl_mult
        tp2 = entry + sl_distance * atr_tp2_mult / atr_sl_mult
    else:
        stop_loss = entry + sl_distance
        tp1 = entry - sl_distance * atr_tp1_mult / atr_sl_mult
        tp2 = entry - sl_distance * atr_tp2_mult / atr_sl_mult

    return TradeTargets(
        entry=round(entry, 2),
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        risk_amount=round(sl_distance, 2),
        reward_1=round(abs(tp1 - entry), 2),
        reward_2=round(abs(tp2 - entry), 2),
        direction=direction,
        sl_strategy=sl_strategy,
    )


# ──────────────────────────────────────────────────────────────────────
# Pre-Trade Filters
# ──────────────────────────────────────────────────────────────────────

def apply_pre_trade_filters(
    indicators: IndicatorSet,
    targets: Optional[TradeTargets],
    min_adx: float = 20,
    min_volatility_pct: float = 0.3,
    fee_rate: float = 0.0006,
    leverage: int = 5,
    check_profit_after_fees: bool = True,
) -> list[str]:
    """
    Returns a list of filter failure reasons. Empty list = all filters passed.
    """
    failures: list[str] = []

    # ADX check — is the market ranging?
    if indicators.adx is not None and indicators.adx < min_adx:
        failures.append(f"ADX too low ({indicators.adx:.1f} < {min_adx}) — market is ranging")

    # Volatility check
    if indicators.atr_pct is not None and indicators.atr_pct < min_volatility_pct:
        failures.append(
            f"Volatility too low ({indicators.atr_pct:.2f}% < {min_volatility_pct}%)"
        )

    # Profit-after-fees check
    if check_profit_after_fees and targets:
        # Fee cost: entry fee + exit fee, on leveraged notional
        # fee_per_trade = 2 * fee_rate (open + close)
        total_fee_pct = 2 * fee_rate * leverage * 100  # as percentage of position
        profit_at_tp1_pct = (targets.reward_1 / targets.entry * 100 * leverage) if targets.entry else 0

        if profit_at_tp1_pct <= total_fee_pct:
            failures.append(
                f"TP1 profit ({profit_at_tp1_pct:.2f}%) wouldn't cover fees "
                f"({total_fee_pct:.2f}%) at {leverage}x leverage"
            )

    return failures


# ──────────────────────────────────────────────────────────────────────
# Report Generation (text for LLM context)
# ──────────────────────────────────────────────────────────────────────

def format_indicator_report(indicators: IndicatorSet) -> str:
    """Format indicators into a readable text report for LLM context."""
    lines = [f"=== {indicators.timeframe.upper()} Timeframe ==="]
    lines.append(f"Price: ${indicators.close:,.2f} ({indicators.change_pct:+.2f}%)" if indicators.close and indicators.change_pct is not None else "")

    lines.append("\n--- Trend ---")
    if indicators.ema_9:
        lines.append(f"EMA 9/21/50: ${indicators.ema_9:,.2f} / ${indicators.ema_21:,.2f} / ${indicators.ema_50:,.2f}")
    if indicators.ema_200:
        lines.append(f"EMA 200: ${indicators.ema_200:,.2f}")
    if indicators.adx is not None:
        lines.append(f"ADX: {indicators.adx:.1f} (+DI: {indicators.plus_di:.1f}, -DI: {indicators.minus_di:.1f})")
    if indicators.macd_line is not None:
        lines.append(f"MACD: {indicators.macd_line:.2f} / Signal: {indicators.macd_signal:.2f} / Hist: {indicators.macd_histogram:.2f}")

    lines.append("\n--- Momentum ---")
    if indicators.rsi_14 is not None:
        lines.append(f"RSI(14): {indicators.rsi_14:.1f}")
    if indicators.stoch_k is not None:
        lines.append(f"Stochastic K/D: {indicators.stoch_k:.1f} / {indicators.stoch_d:.1f}")
    if indicators.cci_20 is not None:
        lines.append(f"CCI(20): {indicators.cci_20:.1f}")
    if indicators.williams_r is not None:
        lines.append(f"Williams %R: {indicators.williams_r:.1f}")
    if indicators.roc_10 is not None:
        lines.append(f"ROC(10): {indicators.roc_10:.1f}%")

    lines.append("\n--- Volume ---")
    if indicators.volume_ratio is not None:
        lines.append(f"Volume Ratio: {indicators.volume_ratio:.2f}x average")
    if indicators.obv is not None and indicators.obv_sma_20 is not None:
        obv_trend = "accumulation" if indicators.obv > indicators.obv_sma_20 else "distribution"
        lines.append(f"OBV Trend: {obv_trend}")
    if indicators.vwap is not None:
        vwap_pos = "above" if indicators.close and indicators.close > indicators.vwap else "below"
        lines.append(f"VWAP: ${indicators.vwap:,.2f} (price {vwap_pos})")

    lines.append("\n--- Volatility ---")
    if indicators.atr_14 is not None:
        lines.append(f"ATR(14): ${indicators.atr_14:,.2f} ({indicators.atr_pct:.2f}%)")
    if indicators.bb_upper is not None:
        lines.append(f"Bollinger Bands: ${indicators.bb_lower:,.2f} / ${indicators.bb_middle:,.2f} / ${indicators.bb_upper:,.2f}")
    if indicators.bb_position is not None:
        lines.append(f"BB Position: {indicators.bb_position:.2f} (0=lower, 1=upper)")

    lines.append("\n--- Support/Resistance ---")
    if indicators.pivot is not None:
        lines.append(f"Pivot: ${indicators.pivot:,.2f}")
        lines.append(f"Support: S1=${indicators.support_1:,.2f}, S2=${indicators.support_2:,.2f}")
        lines.append(f"Resistance: R1=${indicators.resistance_1:,.2f}, R2=${indicators.resistance_2:,.2f}")

    return "\n".join(line for line in lines if line)


def format_scoring_report(result: ScoringResult, targets: Optional[TradeTargets] = None) -> str:
    """Format the full scoring result into a text report."""
    lines = [
        "╔══════════════════════════════════════╗",
        "║     MARKET ANALYSIS REPORT           ║",
        "╚══════════════════════════════════════╝",
        "",
        f"Direction: {result.direction.value}",
        f"Confidence: {result.confidence}%",
        f"Signal: {result.signal_strength.value}",
        f"Raw Score: {result.raw_score:+.1f}/100",
        "",
        "--- Category Breakdown ---",
    ]

    for cat in result.category_scores:
        lines.append(f"  {cat.name:20s}: {cat.raw_score:+6.1f} × {cat.weight:.0%} = {cat.weighted_score:+6.1f}")
        for k, v in cat.details.items():
            if k != "raw_score":
                lines.append(f"    {k}: {v}")

    if result.reasons:
        lines.append("\n--- Key Reasons ---")
        for r in result.reasons:
            lines.append(f"  • {r}")

    if result.filter_failures:
        lines.append("\n--- Filter Failures ---")
        for f in result.filter_failures:
            lines.append(f"  ✗ {f}")

    if targets:
        lines.append("\n--- Trade Targets ---")
        dir_str = "LONG" if targets.direction == Direction.BULLISH else "SHORT"
        lines.append(f"  Direction: {dir_str}")
        lines.append(f"  Entry: ${targets.entry:,.2f}")
        lines.append(f"  Stop Loss: ${targets.stop_loss:,.2f} (risk: ${targets.risk_amount:,.2f})")
        lines.append(f"  TP1: ${targets.take_profit_1:,.2f} (reward: ${targets.reward_1:,.2f})")
        lines.append(f"  TP2: ${targets.take_profit_2:,.2f} (reward: ${targets.reward_2:,.2f})")
        rr1 = targets.reward_1 / targets.risk_amount if targets.risk_amount else 0
        rr2 = targets.reward_2 / targets.risk_amount if targets.risk_amount else 0
        lines.append(f"  R:R Ratio: TP1={rr1:.1f}:1, TP2={rr2:.1f}:1")
        lines.append(f"  SL Strategy: {targets.sl_strategy}")

    # Indicator reports
    for tf, ind in result.indicators.items():
        lines.append("")
        lines.append(format_indicator_report(ind))

    return "\n".join(lines)
