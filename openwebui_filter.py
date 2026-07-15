"""
Signal library — multi-timeframe technical indicators + 5-category weighted scoring.

This is the single source of truth for all indicator math (``compute_*``) and the
category scorers (``calc_*``), imported by the typed ``scoring`` layer and the fast
backtest harness. It is pure technical analysis — no LLM, no network, no I/O.

(Filename kept for import stability; the former OpenWebUI chat filter was removed.)
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS — canonical implementations
# scoring.py imports these; keep self-contained.
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


# Rolling VWAP window (bars). A cumulative-from-series-start VWAP is NOT
# reproducible in live trading: its value depends on how much history happens to
# be loaded (the live loader only fetches ~warmup_periods+30 candles, while the
# backtest anchors at the first cached candle years earlier), so the same bar
# scores differently live vs backtest. A fixed trailing window computes
# identically on both paths. 100 bars stays comfortably within the live load on
# every timeframe (warmup_periods+30) while remaining a slow VWAP baseline.
VWAP_WINDOW = 100


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    window: int = VWAP_WINDOW,
) -> pd.Series:
    """Rolling volume-weighted average price over a fixed trailing window.

    Causal (row i uses only bars i-window+1..i), so computing it once over the
    full series (fast harness) and recomputing it on each expanding slice
    (engine/live) yield identical values — and live reproduces the backtest
    because the window no longer depends on the loaded history depth.
    """
    tp = (high + low + close) / 3
    pv = (tp * volume).rolling(window=window, min_periods=window).sum()
    vv = volume.rolling(window=window, min_periods=window).sum().replace(0, np.nan)
    return pv / vv


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

# Every hand-tuned point award/penalty lives here.  Callers may supply a partial
# override dict for optimization; omitted keys reproduce the historical strategy
# exactly.  Keeping the values beside the canonical functions preserves the
# standalone OpenWebUI copy/paste contract.
DEFAULT_SCORING_POINTS = {
    "trend.ema_stack": 30, "trend.ema_mixed": 10, "trend.ema200": 15,
    "trend.di": 20, "trend.macd": 15, "trend.macd_cross": 5,
    "momentum.rsi_extreme": 20, "momentum.rsi_trend": 15,
    "momentum.stoch": 10, "momentum.cci": 10, "momentum.williams": 10,
    "momentum.roc_strong": 15, "momentum.roc_weak": 5,
    "volume.ratio_extreme": 30, "volume.ratio_high": 20,
    "volume.ratio_above": 5, "volume.ratio_below": 10,
    "volume.ratio_low": 25, "volume.confirmation": 20,
    "volume.obv": 15, "volume.vwap": 10,
    "sr.proximity": 25, "sr.excellent": 25, "sr.good": 15,
    "sr.fair": 5, "sr.poor": 15, "sr.bb_extreme": 15,
    "risk.atr_extreme": 40, "risk.atr_high": 20,
    "risk.atr_healthy": 10, "risk.atr_moderate": 5, "risk.atr_low": 30,
    "risk.adx_range": 30, "risk.adx_weak": 15, "risk.adx_trend": 10,
    "risk.bb_squeeze": 10,
}


def _scoring_points(overrides=None) -> dict:
    if not overrides:
        return DEFAULT_SCORING_POINTS
    points = dict(DEFAULT_SCORING_POINTS)
    unknown = set(overrides) - set(points)
    if unknown:
        raise ValueError(f"Unknown scoring point keys: {sorted(unknown)}")
    points.update(overrides)
    return points

def calc_trend_score(
    *, price=None, ema_9=None, ema_21=None, ema_50=None, ema_200=None,
    adx=None, plus_di=None, minus_di=None,
    macd_hist=None, macd_line=None, macd_signal=None, points=None,
) -> tuple[float, dict]:
    """Core trend scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0
    p = _scoring_points(points)

    if not price:
        return 0.0, details

    # EMA alignment (9 > 21 > 50 = bullish, reverse = bearish)
    if ema_9 is not None and ema_21 is not None and ema_50 is not None:
        if ema_9 > ema_21 > ema_50:
            score += p["trend.ema_stack"]
            details["ema_alignment"] = "bullish_stack"
        elif ema_9 < ema_21 < ema_50:
            score -= p["trend.ema_stack"]
            details["ema_alignment"] = "bearish_stack"
        else:
            if ema_9 > ema_21:
                score += p["trend.ema_mixed"]
            else:
                score -= p["trend.ema_mixed"]
            details["ema_alignment"] = "mixed"

    # Price vs EMA-200 (long-term trend)
    if ema_200 is not None:
        if price > ema_200:
            score += p["trend.ema200"]
            details["vs_ema200"] = "above"
        else:
            score -= p["trend.ema200"]
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
                score += p["trend.di"] * strength_mult
                details["di_direction"] = "bullish"
            else:
                score -= p["trend.di"] * strength_mult
                details["di_direction"] = "bearish"

    # MACD
    if macd_hist is not None:
        if macd_hist > 0:
            score += p["trend.macd"]
            details["macd"] = "bullish"
        else:
            score -= p["trend.macd"]
            details["macd"] = "bearish"
        # MACD crossover signal
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:
                score += p["trend.macd_cross"]
            else:
                score -= p["trend.macd_cross"]

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


def calc_momentum_score(
    *, rsi_14=None, stoch_k=None, stoch_d=None,
    cci_20=None, williams_r=None, roc_10=None, points=None,
) -> tuple[float, dict]:
    """Core momentum scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0
    p = _scoring_points(points)

    # RSI (14)
    if rsi_14 is not None:
        if rsi_14 > 70:
            score -= p["momentum.rsi_extreme"]
            details["rsi"] = f"overbought ({rsi_14:.1f})"
        elif rsi_14 > 60:
            score += p["momentum.rsi_trend"]
            details["rsi"] = f"bullish ({rsi_14:.1f})"
        elif rsi_14 > 40:
            score += 0
            details["rsi"] = f"neutral ({rsi_14:.1f})"
        elif rsi_14 > 30:
            score -= p["momentum.rsi_trend"]
            details["rsi"] = f"bearish ({rsi_14:.1f})"
        else:
            score += p["momentum.rsi_extreme"]
            details["rsi"] = f"oversold ({rsi_14:.1f})"

    # Stochastic
    if stoch_k is not None and stoch_d is not None:
        if stoch_k > 80:
            score -= p["momentum.stoch"]
            details["stoch"] = "overbought"
        elif stoch_k < 20:
            score += p["momentum.stoch"]
            details["stoch"] = "oversold"
        elif stoch_k > stoch_d:
            score += p["momentum.stoch"]
            details["stoch"] = "bullish_cross"
        else:
            score -= p["momentum.stoch"]
            details["stoch"] = "bearish_cross"

    # CCI
    if cci_20 is not None:
        if cci_20 > 100:
            score += p["momentum.cci"]
            details["cci"] = "strong_bullish"
        elif cci_20 < -100:
            score -= p["momentum.cci"]
            details["cci"] = "strong_bearish"

    # Williams %R
    if williams_r is not None:
        if williams_r > -20:
            score -= p["momentum.williams"]
            details["williams_r"] = "overbought"
        elif williams_r < -80:
            score += p["momentum.williams"]
            details["williams_r"] = "oversold"

    # ROC
    if roc_10 is not None:
        if roc_10 > 5:
            score += p["momentum.roc_strong"]
            details["roc"] = f"strong_positive ({roc_10:.1f}%)"
        elif roc_10 > 0:
            score += p["momentum.roc_weak"]
            details["roc"] = f"positive ({roc_10:.1f}%)"
        elif roc_10 > -5:
            score -= p["momentum.roc_weak"]
            details["roc"] = f"negative ({roc_10:.1f}%)"
        else:
            score -= p["momentum.roc_strong"]
            details["roc"] = f"strong_negative ({roc_10:.1f}%)"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


def calc_volume_score(
    *, volume_ratio=None, change_pct=None,
    obv=None, obv_sma_20=None, vwap=None, price=None, points=None,
) -> tuple[float, dict]:
    """Core volume scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0
    p = _scoring_points(points)

    # Volume ratio
    if volume_ratio is not None:
        details["volume_ratio"] = round(volume_ratio, 2)
        if volume_ratio > 2.0:
            score += p["volume.ratio_extreme"]
        elif volume_ratio > 1.5:
            score += p["volume.ratio_high"]
        elif volume_ratio > 1.0:
            score += p["volume.ratio_above"]
        elif volume_ratio > 0.5:
            score -= p["volume.ratio_below"]
        else:
            score -= p["volume.ratio_low"]

    # Direction: is price move supported by volume?
    if change_pct is not None and volume_ratio is not None:
        price_up = change_pct > 0
        high_vol = volume_ratio > 1.0
        if price_up and high_vol:
            score += p["volume.confirmation"]
            details["vol_confirmation"] = "bullish_confirmed"
        elif not price_up and high_vol:
            score -= p["volume.confirmation"]
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
            score += p["volume.obv"]
            details["obv_trend"] = "accumulation"
        else:
            score -= p["volume.obv"]
            details["obv_trend"] = "distribution"

    # VWAP position
    if vwap is not None and price:
        if price > vwap:
            score += p["volume.vwap"]
            details["vwap_position"] = "above"
        else:
            score -= p["volume.vwap"]
            details["vwap_position"] = "below"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


def calc_sr_score(
    *, price=None, nearest_support=None, nearest_resistance=None,
    bb_position=None, points=None,
) -> tuple[float, dict]:
    """Core support/resistance scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0
    p = _scoring_points(points)

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
        score += p["sr.proximity"]
        details["proximity"] = "near_support"
    elif dist_resistance < 1.0:
        score -= p["sr.proximity"]
        details["proximity"] = "near_resistance"

    # Good R:R from S/R standpoint
    if sr_ratio > 3:
        score += p["sr.excellent"]
        details["sr_quality"] = "excellent"
    elif sr_ratio > 2:
        score += p["sr.good"]
        details["sr_quality"] = "good"
    elif sr_ratio > 1:
        score += p["sr.fair"]
        details["sr_quality"] = "fair"
    else:
        score -= p["sr.poor"]
        details["sr_quality"] = "poor"

    # Bollinger Band position
    if bb_position is not None:
        if bb_position > 0.95:
            score -= p["sr.bb_extreme"]
            details["bb"] = "upper_extreme"
        elif bb_position < 0.05:
            score += p["sr.bb_extreme"]
            details["bb"] = "lower_extreme"
        elif 0.4 < bb_position < 0.6:
            details["bb"] = "middle"

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details


def calc_risk_score(
    *, atr_pct=None, adx=None, bb_width=None, points=None,
) -> tuple[float, dict]:
    """Core risk scoring. Returns (score, details)."""
    details: dict = {}
    score = 0.0
    p = _scoring_points(points)

    # ATR-based volatility assessment
    if atr_pct is not None:
        if atr_pct > 8:
            score -= p["risk.atr_extreme"]
            details["volatility"] = f"extreme ({atr_pct:.1f}%)"
        elif atr_pct > 5:
            score -= p["risk.atr_high"]
            details["volatility"] = f"high ({atr_pct:.1f}%)"
        elif atr_pct > 2:
            score += p["risk.atr_healthy"]
            details["volatility"] = f"healthy ({atr_pct:.1f}%)"
        elif atr_pct > 0.5:
            score += p["risk.atr_moderate"]
            details["volatility"] = f"moderate ({atr_pct:.1f}%)"
        else:
            score -= p["risk.atr_low"]
            details["volatility"] = f"too_low ({atr_pct:.1f}%)"

    # ADX ranging check
    if adx is not None:
        if adx < 15:
            score -= p["risk.adx_range"]
            details["ranging"] = "strongly_ranging"
        elif adx < 20:
            score -= p["risk.adx_weak"]
            details["ranging"] = "possibly_ranging"
        else:
            score += p["risk.adx_trend"]
            details["ranging"] = "trending"

    # BB width (squeeze detection)
    if bb_width is not None:
        if bb_width < 2:
            score -= p["risk.bb_squeeze"]
            details["bb_squeeze"] = True
        else:
            details["bb_squeeze"] = False

    score = max(-100, min(100, score))
    details["raw_score"] = score
    return score, details
