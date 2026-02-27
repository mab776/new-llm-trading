"""
Signal routing — classifies scoring results and routes to appropriate handler.

3-tier routing:
  STRONG   → Deterministic template response (instant, free, no LLM)
  MARGINAL → Send to LLM (via OpenWebUI) for multi-bot consensus
  WAIT     → Skip trade
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_trading_bot.config import AppConfig, LeverageTier
from llm_trading_bot.scoring import (
    Direction,
    IndicatorSet,
    ScoringResult,
    SignalStrength,
    TradeTargets,
    apply_pre_trade_filters,
    calculate_targets,
    compute_composite_score,
    format_scoring_report,
)


@dataclass
class RoutingDecision:
    """The result of routing a signal through the pipeline."""
    signal_strength: SignalStrength
    scoring_result: ScoringResult
    targets: Optional[TradeTargets]
    template_response: Optional[str] = None
    needs_llm: bool = False
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


def build_llm_context(result: ScoringResult, targets: Optional[TradeTargets]) -> str:
    """
    Build the context payload to send to the LLM for marginal signals.
    Contains all pre-calculated data — the LLM should NOT invent numbers.
    """
    report = format_scoring_report(result, targets)
    return f"""
[FINANCIAL DATA INJECTION — Pre-calculated indicators below. Do NOT invent or modify these numbers.]

{report}

[END FINANCIAL DATA]

Based on the above pre-calculated technical analysis data, provide your trading recommendation.
You MUST respond in this exact JSON format:
{{
  "decision": "LONG" | "SHORT" | "WAIT",
  "confidence": <number 1-100>,
  "reasoning": "<brief explanation>",
  "entry": <price or null>,
  "stop_loss": <price or null>,
  "take_profit_1": <price or null>,
  "take_profit_2": <price or null>
}}

Important:
- Base your analysis ONLY on the provided data
- Do not hallucinate price levels or indicator values
- If you disagree with the suggested direction, explain why
- A "WAIT" decision is perfectly valid if the setup isn't convincing
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

    # 1. Score
    result = compute_composite_score(
        indicators_by_tf=indicators_by_tf,
        weights=scoring_cfg.weights,
        primary_timeframe=config.trading.primary_timeframe,
        confidence_min=scoring_cfg.confidence_min,
        confidence_max=scoring_cfg.confidence_max,
    )

    # 2. Calculate targets
    primary_tf = config.trading.primary_timeframe
    primary_ind = indicators_by_tf.get(primary_tf) or next(iter(indicators_by_tf.values()))
    targets = calculate_targets(
        indicators=primary_ind,
        direction=result.direction,
        sl_strategy=config.trading.stop_loss_strategy,
        atr_sl_mult=scoring_cfg.atr_sl_multiplier,
        atr_tp1_mult=scoring_cfg.atr_tp1_multiplier,
        atr_tp2_mult=scoring_cfg.atr_tp2_multiplier,
    )

    # 3. Pre-trade filters
    filter_failures = apply_pre_trade_filters(
        indicators=primary_ind,
        targets=targets,
        min_adx=config.filters.min_adx,
        min_volatility_pct=config.filters.min_volatility_pct,
        fee_rate=config.fees.active_fee_rate,
        leverage=tier.leverage,
        check_profit_after_fees=config.filters.min_profit_after_fees,
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

    elif signal == SignalStrength.MARGINAL:
        return RoutingDecision(
            signal_strength=SignalStrength.MARGINAL,
            scoring_result=result,
            targets=targets,
            needs_llm=True,
        )

    else:
        return RoutingDecision(
            signal_strength=SignalStrength.WAIT,
            scoring_result=result,
            targets=targets,
            skip_reason=f"Score too low ({result.raw_score:+.1f})",
        )
