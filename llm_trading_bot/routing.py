"""
Signal routing — classifies scoring results and routes to appropriate handler.

3-tier routing (pure technical signal — no LLM):
  STRONG   → Deterministic template response, execute
  MARGINAL → Execute deterministically too (counted as a trade in the backtest)
  WAIT     → Skip trade
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_trading_bot.config import AppConfig, LeverageTier
from llm_trading_bot.scoring import (
    Direction,
    IndicatorSet,
    MarketRegime,
    ScoringResult,
    SignalStrength,
    TradeTargets,
    apply_pre_trade_filters,
    calculate_targets,
    compute_composite_score,
    detect_market_regime,
)


@dataclass
class RoutingDecision:
    """The result of routing a signal through the pipeline."""
    signal_strength: SignalStrength
    scoring_result: ScoringResult
    targets: Optional[TradeTargets]
    template_response: Optional[str] = None
    skip_reason: Optional[str] = None


def classify_signal(
    score: float, tier: LeverageTier
) -> SignalStrength:
    """Classify a raw score into STRONG, MARGINAL, or WAIT."""
    abs_score = abs(score)
    # Map the score to a 0-100 confidence-like value
    # The raw_score is already -100 to +100
    if abs_score >= tier.strong_threshold:
        return SignalStrength.STRONG
    elif abs_score >= tier.marginal_threshold_low:
        return SignalStrength.MARGINAL
    else:
        return SignalStrength.WAIT


def build_template_response(
    result: ScoringResult, targets: TradeTargets, tier: LeverageTier
) -> str:
    """
    Build a deterministic template response for STRONG signals.
    No LLM needed — instant and free.
    """
    dir_str = "LONG" if result.direction == Direction.BULLISH else "SHORT"
    rr1 = targets.reward_1 / targets.risk_amount if targets.risk_amount else 0
    rr2 = targets.reward_2 / targets.risk_amount if targets.risk_amount else 0

    reasons_str = "\n".join(f"  • {r}" for r in result.reasons)

    return f"""
╔══════════════════════════════════════════════╗
║  STRONG SIGNAL — DETERMINISTIC EXECUTION     ║
╚══════════════════════════════════════════════╝

Direction: {dir_str}
Confidence: {result.confidence}%
Raw Score: {result.raw_score:+.1f}/100

TRADE SETUP:
  Entry:     ${targets.entry:,.2f}
  Stop Loss: ${targets.stop_loss:,.2f}
  TP1:       ${targets.take_profit_1:,.2f} (R:R = {rr1:.1f}:1) — Exit {tier.tp1_exit_pct:.0%}
  TP2:       ${targets.take_profit_2:,.2f} (R:R = {rr2:.1f}:1) — Exit remaining
  Leverage:  {tier.leverage}x
  SL Strategy: {targets.sl_strategy}

KEY REASONS:
{reasons_str}

ACTION: Execute trade immediately. Strong conviction based on multi-factor analysis.
""".strip()


def route_signal(
    indicators_by_tf: dict[str, IndicatorSet],
    config: AppConfig,
) -> RoutingDecision:
    """
    Main routing function: score → classify → route.

    Returns a RoutingDecision with everything needed for the next step.
    """
    tier = config.trading.active_leverage_tier
    scoring_cfg = config.scoring
    primary_tf = config.trading.primary_timeframe
    if primary_tf not in indicators_by_tf:
        raise ValueError(f"Required primary timeframe {primary_tf!r} is missing")

    # 1. Score
    result = compute_composite_score(
        indicators_by_tf=indicators_by_tf,
        weights=scoring_cfg.weights,
        primary_timeframe=config.trading.primary_timeframe,
        confidence_min=scoring_cfg.confidence_min,
        confidence_max=scoring_cfg.confidence_max,
        scoring_points=scoring_cfg.points,
    )

    # 2. Calculate targets (use tier R:R ratios)
    primary_ind = indicators_by_tf[primary_tf]
    targets = calculate_targets(
        indicators=primary_ind,
        direction=result.direction,
        sl_strategy=config.trading.stop_loss_strategy,
        atr_sl_mult=scoring_cfg.atr_sl_multiplier,
        tp1_rr=tier.tp1_rr,
        tp2_rr=tier.tp2_rr,
    )

    # 3. Pre-trade filters (with category agreement + regime)
    filter_failures = apply_pre_trade_filters(
        indicators=primary_ind,
        targets=targets,
        min_adx=config.filters.min_adx,
        min_volatility_pct=config.filters.min_volatility_pct,
        fee_rate=config.fees.active_fee_rate,
        leverage=tier.leverage,
        check_profit_after_fees=config.filters.min_profit_after_fees,
        category_scores=result.category_scores,
        direction=result.direction,
        min_category_agreement=config.filters.min_category_agreement,
        require_trend_momentum_agree=config.filters.require_trend_momentum_agree,
        skip_choppy_regime=config.filters.skip_choppy_regime,
        skip_volatile_regime=config.filters.skip_volatile_regime,
    )
    result.filter_failures = filter_failures
    result.passed_filters = len(filter_failures) == 0

    # 4. Classify signal
    signal = classify_signal(result.raw_score, tier)
    result.signal_strength = signal

    # 5. Route
    if not result.passed_filters:
        return RoutingDecision(
            signal_strength=SignalStrength.WAIT,
            scoring_result=result,
            targets=targets,
            skip_reason=f"Filters failed: {'; '.join(filter_failures)}",
        )

    if signal == SignalStrength.STRONG and targets:
        template = build_template_response(result, targets, tier)
        return RoutingDecision(
            signal_strength=SignalStrength.STRONG,
            scoring_result=result,
            targets=targets,
            template_response=template,
        )

    elif signal == SignalStrength.MARGINAL and targets:
        # Marginal setups are traded deterministically (the backtest counts them),
        # so they carry the same template as STRONG — there is no LLM gate.
        return RoutingDecision(
            signal_strength=SignalStrength.MARGINAL,
            scoring_result=result,
            targets=targets,
            template_response=build_template_response(result, targets, tier),
        )

    else:
        return RoutingDecision(
            signal_strength=SignalStrength.WAIT,
            scoring_result=result,
            targets=targets,
            skip_reason=f"Score too low ({result.raw_score:+.1f})",
        )
