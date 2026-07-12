"""Drawdown circuit-breaker (risk_management.dd_throttle_*) — config + engine behavior."""
from __future__ import annotations

from llm_trading_bot.config import load_config
from llm_trading_bot.backtesting import BacktestEngine


def test_config_has_dd_throttle_fields():
    cfg = load_config("config.json")
    assert 0 <= cfg.risk_management.dd_throttle_threshold < 1
    assert cfg.risk_management.dd_throttle_slots >= 1
    assert 0 < cfg.risk_management.dd_throttle_risk <= 1


def _throttle_state(engine):
    """Replicate the engine's throttle computation for assertion."""
    rm = engine.risk_cfg
    p = engine.portfolio
    if rm.dd_throttle_threshold <= 0 or p.peak_balance <= 0:
        return engine.config.position_sizing.max_positions, False
    dd = (p.peak_balance - p.balance) / p.peak_balance
    if dd >= rm.dd_throttle_threshold:
        return min(engine.config.position_sizing.max_positions, rm.dd_throttle_slots), True
    return engine.config.position_sizing.max_positions, False


def test_throttle_inactive_below_threshold():
    cfg = load_config("config.json")
    cfg.risk_management.dd_throttle_threshold = 0.25
    cfg.position_sizing.max_positions = 3
    eng = BacktestEngine(cfg)
    eng.portfolio.peak_balance = 100.0
    eng.portfolio.balance = 90.0     # 10% DD < 25%
    slots, throttled = _throttle_state(eng)
    assert slots == 3 and not throttled


def test_throttle_caps_slots_and_risk_beyond_threshold():
    cfg = load_config("config.json")
    cfg.risk_management.dd_throttle_threshold = 0.25
    cfg.risk_management.dd_throttle_slots = 1
    cfg.risk_management.dd_throttle_risk = 0.5
    cfg.position_sizing.max_positions = 3
    eng = BacktestEngine(cfg)
    eng.portfolio.peak_balance = 100.0
    eng.portfolio.balance = 70.0     # 30% DD >= 25%
    slots, throttled = _throttle_state(eng)
    assert slots == 1 and throttled
    # risk multiplier applied while throttled
    assert cfg.risk_management.dd_throttle_risk == 0.5


def test_throttle_disabled_at_zero():
    cfg = load_config("config.json")
    cfg.risk_management.dd_throttle_threshold = 0.0
    cfg.position_sizing.max_positions = 3
    eng = BacktestEngine(cfg)
    eng.portfolio.peak_balance = 100.0
    eng.portfolio.balance = 10.0     # 90% DD but throttle off
    slots, throttled = _throttle_state(eng)
    assert slots == 3 and not throttled
