"""Single-model marginal gate: simulator hook and leakage-blinded prompt."""
from __future__ import annotations

import pandas as pd

import opt.fastbt as fb
from llm_trading_bot.config import load_config
from llm_trading_bot.scoring import (
    CategoryScore,
    Direction,
    IndicatorSet,
    ScoringResult,
    SignalStrength,
    TradeTargets,
)
from opt.llm_gate_pilot import _build_blinded_context, _prompt_id


def _indicator(close: float) -> IndicatorSet:
    return IndicatorSet(
        timeframe="4h", close=close, open=close, high=close + 1, low=close - 1,
        change_pct=0, ema_9=close, ema_21=close, ema_50=close, ema_200=close,
        adx=30, plus_di=25, minus_di=15, macd_line=1, macd_signal=0,
        macd_histogram=1, rsi_14=55, stoch_k=55, stoch_d=50, cci_20=10,
        williams_r=-40, roc_10=1, volume=1000, volume_sma_20=1000,
        volume_ratio=1, obv=100, obv_sma_20=90, vwap=close, atr_14=2,
        atr_pct=2, bb_upper=close + 2, bb_middle=close, bb_lower=close - 2,
        bb_width=4, bb_position=.5, pivot=close, support_1=close - 1,
        support_2=close - 2, resistance_1=close + 1, resistance_2=close + 2,
        nearest_support=close - 1, nearest_resistance=close + 1,
    )


def _score(ind: IndicatorSet) -> ScoringResult:
    return ScoringResult(
        direction=Direction.BULLISH, confidence=20,
        signal_strength=SignalStrength.MARGINAL, raw_score=15,
        category_scores=[CategoryScore("trend", 40, 1, 40)],
        indicators={"4h": ind},
    )


def _targets(close: float) -> TradeTargets:
    return TradeTargets(
        entry=close, stop_loss=close - 10, take_profit_1=close + 20,
        take_profit_2=close + 30, risk_amount=10, reward_1=20, reward_2=30,
        direction=Direction.BULLISH, sl_strategy="atr",
    )


def test_blinded_context_rebases_prices_and_omits_timestamp():
    ind = _indicator(54321)
    prompt = _build_blinded_context(_score(ind), _targets(ind.close))

    assert "$100.00" in prompt
    assert "54321" not in prompt
    assert "2024-" not in prompt
    assert "Signal: MARGINAL" in prompt


def test_cache_key_separates_thinking_configuration():
    base = _prompt_id("model", "2024-01-01", "prompt")
    thinking = _prompt_id(
        "model", "2024-01-01", "prompt", think=True, num_predict=8192
    )

    assert base != thinking


def test_marginal_gate_can_reject_without_changing_default(monkeypatch):
    inds = [_indicator(100), _indicator(101)]
    pre = fb.Precomputed(
        timestamps=list(pd.date_range("2024-01-01", periods=2, freq="4h")),
        primary=inds, sec_by_bar=[{}, {}], warmup=0,
    )
    cfg = load_config("config.json")
    cfg.backtesting.initial_balance = 1000
    cfg.backtesting.enable_trailing_stops = False
    cfg.filters.min_adx = 0
    cfg.filters.min_volatility_pct = 0
    cfg.filters.min_profit_after_fees = False
    cfg.filters.min_category_agreement = 0
    cfg.risk_management.cooldown_candles_after_sl = 0
    cfg.risk_management.opposite_exit_threshold = 0
    cfg.position_sizing.max_positions = 1

    monkeypatch.setattr(fb, "compute_composite_score", lambda **kwargs: _score(kwargs["indicators_by_tf"]["4h"]))
    monkeypatch.setattr(fb, "calculate_targets", lambda indicators, **kwargs: _targets(indicators.close))

    default = fb.simulate(pre, cfg, "2024-01-01", "2024-01-02")
    accept = fb.simulate(pre, cfg, "2024-01-01", "2024-01-02", marginal_gate=lambda *_: True)
    reject = fb.simulate(pre, cfg, "2024-01-01", "2024-01-02", marginal_gate=lambda *_: False)

    assert default == accept
    assert default.trades == 1
    assert default.marginal_candidates == default.marginal_accepted == 1
    assert reject.trades == 0
    assert reject.marginal_candidates == 2
    assert reject.marginal_accepted == 0
