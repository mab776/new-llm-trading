"""
Core scoring engine — typed API layer over canonical calculations.

Indicator computations and scoring logic live in openwebui_filter.py
(the single source of truth). This module provides the typed dataclass API
(IndicatorSet, CategoryScore, ScoringResult, etc.) that the rest of the
package uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Import canonical implementations from openwebui_filter
# ──────────────────────────────────────────────────────────────────────

from openwebui_filter import (  # noqa: E402 — project-root module
    # Indicator computation functions
    compute_ema,
    compute_sma,
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_adx,
    compute_stochastic,
    compute_bollinger_bands,
    compute_obv,
    compute_vwap,
    compute_williams_r,
    compute_cci,
    compute_roc,
    compute_pivot_points,
    # Scoring logic functions
    calc_trend_score,
    calc_momentum_score,
    calc_volume_score,
    calc_sr_score,
    calc_risk_score,
)


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


class MarketRegime(str, Enum):
    TRENDING = "trending"          # ADX > 25, clear direction — ideal
    WEAK_TREND = "weak_trend"      # ADX 20-25, slight direction
    RANGING = "ranging"            # ADX < 20, no direction — avoid
    VOLATILE = "volatile"          # High ATR, wide BB — risky
    CHOPPY = "choppy"              # Low ADX + narrow BB — worst


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
        # Clamp to [0, 1]: on a volatility spike close can sit outside the bands.
        raw_bb_pos = (ind.close - ind.bb_lower) / (ind.bb_upper - ind.bb_lower)
        ind.bb_position = max(0.0, min(1.0, raw_bb_pos))

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

def score_trend(indicators: IndicatorSet, points: dict | None = None) -> CategoryScore:
    """Score trend from -100 (strong bearish) to +100 (strong bullish)."""
    score, details = calc_trend_score(
        price=indicators.close,
        ema_9=indicators.ema_9, ema_21=indicators.ema_21, ema_50=indicators.ema_50,
        ema_200=indicators.ema_200,
        adx=indicators.adx, plus_di=indicators.plus_di, minus_di=indicators.minus_di,
        macd_hist=indicators.macd_histogram, macd_line=indicators.macd_line,
        macd_signal=indicators.macd_signal, points=points,
    )
    return CategoryScore("trend", score, 0, 0, details)


def score_momentum(indicators: IndicatorSet, points: dict | None = None) -> CategoryScore:
    """Score momentum from -100 to +100."""
    score, details = calc_momentum_score(
        rsi_14=indicators.rsi_14,
        stoch_k=indicators.stoch_k, stoch_d=indicators.stoch_d,
        cci_20=indicators.cci_20, williams_r=indicators.williams_r,
        roc_10=indicators.roc_10, points=points,
    )
    return CategoryScore("momentum", score, 0, 0, details)


def score_volume(indicators: IndicatorSet, points: dict | None = None) -> CategoryScore:
    """Score volume confirmation from -100 to +100."""
    score, details = calc_volume_score(
        volume_ratio=indicators.volume_ratio,
        change_pct=indicators.change_pct,
        obv=indicators.obv, obv_sma_20=indicators.obv_sma_20,
        vwap=indicators.vwap, price=indicators.close, points=points,
    )
    return CategoryScore("volume", score, 0, 0, details)


def score_support_resistance(indicators: IndicatorSet, points: dict | None = None) -> CategoryScore:
    """Score S/R proximity from -100 to +100."""
    score, details = calc_sr_score(
        price=indicators.close,
        nearest_support=indicators.nearest_support,
        nearest_resistance=indicators.nearest_resistance,
        bb_position=indicators.bb_position, points=points,
    )
    return CategoryScore("support_resistance", score, 0, 0, details)


def score_risk(indicators: IndicatorSet, points: dict | None = None) -> CategoryScore:
    """Risk assessment from -100 (high risk) to +100 (low risk)."""
    score, details = calc_risk_score(
        atr_pct=indicators.atr_pct,
        adx=indicators.adx,
        bb_width=indicators.bb_width, points=points,
    )
    return CategoryScore("risk", score, 0, 0, details)


# ──────────────────────────────────────────────────────────────────────
# Market Regime Detection
# ──────────────────────────────────────────────────────────────────────

def detect_market_regime(indicators: IndicatorSet) -> MarketRegime:
    """
    Classify the current market regime using ADX, ATR, and Bollinger Band width.

    Returns a regime that can be used to filter or adjust trade sizing.
    """
    adx = indicators.adx if indicators.adx is not None else 15
    atr_pct = indicators.atr_pct if indicators.atr_pct is not None else 1.0
    bb_width = indicators.bb_width if indicators.bb_width is not None else 5.0

    # Choppy: ranging + narrow bands = whipsaw city
    if adx < 18 and bb_width < 3:
        return MarketRegime.CHOPPY

    # Volatile: extreme ATR or very wide BB
    if atr_pct > 5.0 or bb_width > 12:
        return MarketRegime.VOLATILE

    # Ranging: no trend strength
    if adx < 20:
        return MarketRegime.RANGING

    # Weak trend: marginal ADX
    if adx < 25:
        return MarketRegime.WEAK_TREND

    # Trending: strong ADX
    return MarketRegime.TRENDING


# ──────────────────────────────────────────────────────────────────────
# Combined Scoring
# ──────────────────────────────────────────────────────────────────────

def compute_composite_score(
    indicators_by_tf: dict[str, IndicatorSet],
    weights: dict[str, float],
    primary_timeframe: str = "4h",
    confidence_min: float = 5,
    confidence_max: float = 95,
    scoring_points: dict[str, float] | None = None,
    alignment_mode: str = "discrete",
    alignment_scale: float = 5.0,
    alignment_k: float = 30.0,
    alignment_scale_by_tf: dict | None = None,
    exclude_alignment_tfs: set | None = None,
) -> ScoringResult:
    """
    Compute a composite score across multiple timeframes.
    Primary timeframe gets full weight, others contribute via alignment bonus/penalty.

    ``alignment_mode`` controls how each secondary timeframe contributes:
      - "discrete" (default, legacy): a flat +-``alignment_scale`` vote on whether
        the sign of the TF's trend agrees with the primary weighted total.
      - "continuous": the vote magnitude scales with the TF's trend conviction via
        ``alignment_scale * tanh(trend / alignment_k)``, so a near-zero timeframe
        contributes ~0 instead of a full step. This removes the threshold-cliff
        sensitivity where a tiny data change flips a whole +-5 vote across a
        decision boundary. Defaults leave the discrete behavior untouched.
    """
    primary = indicators_by_tf.get(primary_timeframe)
    if not primary:
        raise ValueError(f"Required primary timeframe {primary_timeframe!r} is missing")

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
        cat = func(primary, scoring_points)
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
    prim_sign = 1.0 if weighted_total > 0 else -1.0 if weighted_total < 0 else 0.0
    for tf, ind in indicators_by_tf.items():
        if tf == primary_timeframe:
            continue
        # Opt-in: let a caller take a timeframe out of the alignment vote (e.g. when
        # the daily trend is handled by a dedicated overlay and would double-count).
        if exclude_alignment_tfs and tf in exclude_alignment_tfs:
            continue
        tf_trend = score_trend(ind, scoring_points).raw_score
        # Neutral primary or neutral TF trend contributes nothing (both modes).
        if prim_sign == 0.0 or tf_trend == 0.0:
            continue
        # Per-timeframe weight override (default: the flat alignment_scale for all).
        scale = (alignment_scale_by_tf.get(tf, alignment_scale)
                 if alignment_scale_by_tf else alignment_scale)
        if alignment_mode == "continuous":
            contrib = scale * float(np.tanh(tf_trend / alignment_k)) * prim_sign
            alignment_bonus += contrib
            reasons.append(
                f"{tf} trend {'aligns' if contrib > 0 else 'diverges'} ({contrib:+.1f})")
        else:  # discrete (legacy) — flat ±scale sign vote
            if (tf_trend > 0) == (weighted_total > 0):
                alignment_bonus += scale
                reasons.append(f"{tf} trend aligns")
            else:
                alignment_bonus -= scale
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
    tp1_rr: float = 2.0,
    tp2_rr: float = 3.5,
    # Legacy params (ignored if tp1_rr/tp2_rr provided)
    atr_tp1_mult: float = 3.0,
    atr_tp2_mult: float = 5.0,
) -> Optional[TradeTargets]:
    """
    Calculate entry, SL, TP1, TP2 based on ATR and optionally S/R levels.
    TP1/TP2 are calculated as R:R multiples of the SL distance.
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
        tp1 = entry + sl_distance * tp1_rr
        tp2 = entry + sl_distance * tp2_rr
    else:
        stop_loss = entry + sl_distance
        tp1 = entry - sl_distance * tp1_rr
        tp2 = entry - sl_distance * tp2_rr

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
    category_scores: Optional[list[CategoryScore]] = None,
    direction: Optional[Direction] = None,
    min_category_agreement: int = 0,
    require_trend_momentum_agree: bool = False,
    skip_choppy_regime: bool = True,
    skip_volatile_regime: bool = False,
) -> list[str]:
    """
    Returns a list of filter failure reasons. Empty list = all filters passed.
    """
    failures: list[str] = []

    # ADX check — is the market ranging?
    if indicators.adx is None:
        failures.append("ADX unavailable — required risk filter cannot be evaluated")
    elif indicators.adx < min_adx:
        failures.append(f"ADX too low ({indicators.adx:.1f} < {min_adx}) — market is ranging")

    # Volatility check
    if indicators.atr_pct is None:
        failures.append("ATR unavailable — required volatility filter cannot be evaluated")
    elif indicators.atr_pct < min_volatility_pct:
        failures.append(
            f"Volatility too low ({indicators.atr_pct:.2f}% < {min_volatility_pct}%)"
        )

    # Profit-after-fees check
    if check_profit_after_fees and targets:
        total_fee_pct = 2 * fee_rate * leverage * 100
        profit_at_tp1_pct = (targets.reward_1 / targets.entry * 100 * leverage) if targets.entry else 0

        if profit_at_tp1_pct <= total_fee_pct:
            failures.append(
                f"TP1 profit ({profit_at_tp1_pct:.2f}%) wouldn't cover fees "
                f"({total_fee_pct:.2f}%) at {leverage}x leverage"
            )

    # Category agreement filter — require N categories to agree on direction
    if category_scores and direction and direction != Direction.NEUTRAL and min_category_agreement > 0:
        is_bullish = direction == Direction.BULLISH
        agreeing = sum(
            1 for cat in category_scores
            if (cat.raw_score > 0 and is_bullish) or (cat.raw_score < 0 and not is_bullish)
        )
        if agreeing < min_category_agreement:
            failures.append(
                f"Category agreement too low ({agreeing}/{len(category_scores)} agree, "
                f"need {min_category_agreement})"
            )

    # Trend + momentum must agree on direction
    if require_trend_momentum_agree and category_scores and direction and direction != Direction.NEUTRAL:
        is_bullish = direction == Direction.BULLISH
        trend_cat = next((c for c in category_scores if c.name == "trend"), None)
        momentum_cat = next((c for c in category_scores if c.name == "momentum"), None)
        if trend_cat and momentum_cat:
            trend_agrees = (trend_cat.raw_score > 0) == is_bullish
            momentum_agrees = (momentum_cat.raw_score > 0) == is_bullish
            if not (trend_agrees and momentum_agrees):
                disagree_parts = []
                if not trend_agrees:
                    disagree_parts.append(f"trend={trend_cat.raw_score:+.0f}")
                if not momentum_agrees:
                    disagree_parts.append(f"momentum={momentum_cat.raw_score:+.0f}")
                failures.append(
                    f"Trend/momentum disagree with {direction.value}: {', '.join(disagree_parts)}"
                )

    # Market regime filter
    regime = detect_market_regime(indicators)
    if skip_choppy_regime and regime == MarketRegime.CHOPPY:
        failures.append(f"Market regime is CHOPPY (ADX={indicators.adx:.1f}, BB_width={indicators.bb_width:.1f}) — whipsaw risk")
    if skip_volatile_regime and regime == MarketRegime.VOLATILE:
        failures.append(f"Market regime is VOLATILE (ATR%={indicators.atr_pct:.1f}%) — extreme risk")

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
