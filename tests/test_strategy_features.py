"""
Tests for the strategy features added 2026-07 (pyramiding, conviction sizing,
opposite-signal exit) — config plumbing + engine behavior.
"""
from __future__ import annotations

from llm_trading_bot.config import load_config
from llm_trading_bot.backtesting import BacktestEngine


def _cfg():
    return load_config("config.json")


def test_config_has_strategy_fields():
    cfg = _cfg()
    assert cfg.position_sizing.max_positions >= 1
    assert cfg.position_sizing.conviction_exponent >= 0
    assert cfg.position_sizing.anti_martingale_step >= 0
    assert cfg.position_sizing.global_max_margin_pct >= 0
    assert cfg.position_sizing.global_max_notional_pct >= 0
    assert cfg.risk_management.opposite_exit_threshold >= 0


def test_defaults_reproduce_classic_behavior():
    """max_positions=1 / conviction=0 / opposite_exit=0 must be valid (classic mode)."""
    cfg = _cfg()
    cfg.position_sizing.max_positions = 1
    cfg.position_sizing.conviction_exponent = 0.0
    cfg.risk_management.opposite_exit_threshold = 0.0
    BacktestEngine(cfg)  # constructs fine


def test_pyramiding_same_direction_only():
    """The entry-slot rule: same-direction stacking up to max_positions, never opposite."""
    cfg = _cfg()
    cfg.position_sizing.max_positions = 3
    eng = BacktestEngine(cfg)
    p = eng.portfolio

    def can_enter(direction_str: str) -> bool:
        open_now = p.open_trades
        return (len(open_now) < cfg.position_sizing.max_positions
                and all(t.direction == direction_str for t in open_now))

    assert can_enter("LONG")
    p.open_trade("LONG", 100, "t0", 95, 110, 120, leverage=2, risk_pct=0.02, tp1_exit_pct=0.5)
    assert can_enter("LONG")          # slot 2 of 3
    assert not can_enter("SHORT")     # opposite never stacks
    p.open_trade("LONG", 101, "t1", 96, 111, 121, leverage=2, risk_pct=0.02, tp1_exit_pct=0.5)
    p.open_trade("LONG", 102, "t2", 97, 112, 122, leverage=2, risk_pct=0.02, tp1_exit_pct=0.5)
    assert not can_enter("LONG")      # full


def test_conviction_sizing_clamped():
    """risk_eff = risk_pct * clamp((|score|/strong)^k, 0.5, 1.5)."""
    risk_pct, k, strong = 0.02, 1.0, 20.0
    for score, expected_mult in [(10.0, 0.5), (20.0, 1.0), (25.0, 1.25), (40.0, 1.5), (100.0, 1.5)]:
        m = (score / strong) ** k
        got = max(0.5, min(1.5, m))
        assert abs(got - expected_mult) < 1e-9, (score, got, expected_mult)
        assert 0.5 * risk_pct <= risk_pct * got <= 1.5 * risk_pct


def test_opposite_exit_closes_only_opposing_trades():
    """A hard flip closes positions on the wrong side of the new signal, at bar close."""
    cfg = _cfg()
    cfg.risk_management.opposite_exit_threshold = 20.0
    eng = BacktestEngine(cfg)
    p = eng.portfolio
    t_long = p.open_trade("LONG", 100, "t0", 90, 200, 300, leverage=1, risk_pct=0.02, tp1_exit_pct=0.5)

    # Simulate the engine's 5.5 step for a hard BEARISH flip at close=98
    want = "SHORT"
    for trade in list(p.open_trades):
        if trade.direction != want:
            p.close_trade(trade, 98.0, "t1", "signal_flip")
            eng._on_trade_closed(trade)

    assert not t_long.is_open
    assert t_long.exit_reason == "signal_flip"
    assert t_long.exit_price == 98.0
    # signal_flip is a loss here but must NOT trigger the SL cooldown
    assert eng.cooldown_remaining == 0
    assert eng.consecutive_losses == 1


# ── Geometric structure levels (probe_geometry research knobs) ──

def test_structure_pivot_confirmation_is_causal():
    """A swing low at bar j must be invisible before j+wing and visible at j+wing."""
    import numpy as np
    from opt.fastbt import compute_structure_levels
    lows = np.array([10, 9, 8, 5, 8, 9, 10, 11, 12, 13], dtype=float)
    highs = lows + 1
    wing = 2
    sup, _res, _sl, _rl = compute_structure_levels(lows, highs, wing)
    # pivot low at j=3 (value 5): confirmed at i = j + wing = 5
    assert np.isnan(sup[4]), "pivot visible before confirmation = lookahead"
    assert sup[5] == 5.0
    assert sup[9] == 5.0


def test_structure_trendline_rising_only():
    """Support trendline requires the second pivot low ABOVE the first."""
    import numpy as np
    from opt.fastbt import compute_structure_levels
    # two pivot lows: j=3 (5.0) then j=9 (7.0) -> rising line
    lows = np.array([10, 9, 8, 5, 8, 9, 10, 9, 8, 7, 8, 9, 10, 11], dtype=float)
    highs = lows + 1
    sup, _res, line, _rl = compute_structure_levels(lows, highs, 2)
    # line confirmed once second pivot (j=9) confirms at i=11
    assert np.isnan(line[10])
    slope = (7.0 - 5.0) / (9 - 3)
    assert abs(line[11] - (7.0 + slope * (11 - 9))) < 1e-9
    # falling lows -> no rising support line ever
    lows2 = np.array([13, 12, 11, 8, 11, 12, 10, 9, 7, 5, 7, 8, 9, 10], dtype=float)
    sup2, _r2, line2, _rl2 = compute_structure_levels(lows2, lows2 + 1, 2)
    assert np.isnan(line2).all()
