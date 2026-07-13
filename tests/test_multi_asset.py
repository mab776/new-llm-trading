"""Shared-balance multi-asset simulation tests."""

from __future__ import annotations

import pandas as pd
import pytest

import opt.multi_asset as multi
from llm_trading_bot.config import AppConfig, LeverageTier
from llm_trading_bot.scoring import (
    Direction, IndicatorSet, ScoringResult, SignalStrength, TradeTargets,
)
from opt.fastbt import Precomputed
from opt.multi_asset import (
    AssetInput, apply_exposure_caps, committed_exposure, simulate_multi,
)


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.trading.primary_timeframe = "4h"
    cfg.trading.entry_mode = "taker"
    cfg.trading.leverage_tiers = {
        "x": LeverageTier(leverage=1, strong_threshold=20,
                           marginal_threshold_low=10, tp1_rr=2, tp2_rr=3)
    }
    cfg.trading.active_tier = "x"
    cfg.position_sizing.risk_pct_per_trade = .10
    cfg.position_sizing.max_positions = 1
    cfg.backtesting.initial_balance = 1000
    cfg.backtesting.enable_partial_exits = False
    cfg.backtesting.enable_trailing_stops = False
    cfg.risk_management.max_holding_hours = 0
    cfg.risk_management.opposite_exit_threshold = 0
    cfg.filters.min_profit_after_fees = False
    cfg.filters.min_category_agreement = 0
    cfg.filters.skip_choppy_regime = False
    return cfg


def _pre(price: float) -> Precomputed:
    ts = list(pd.date_range("2024-01-01", periods=2, freq="4h"))
    inds = [IndicatorSet(timeframe="4h", open=price, high=price + 1,
                         low=price - 1, close=price, atr_14=1,
                         atr_pct=1, adx=30) for _ in ts]
    return Precomputed(ts, inds, [{}, {}], warmup=0)


def test_shared_portfolio_tracks_each_symbol_and_one_balance(monkeypatch):
    monkeypatch.setattr(multi, "compute_composite_score", lambda **kw: ScoringResult(
        direction=Direction.BULLISH, confidence=80,
        signal_strength=SignalStrength.STRONG, raw_score=50,
        category_scores=[], indicators=kw["indicators_by_tf"], reasons=[],
    ))
    monkeypatch.setattr(multi, "calculate_targets", lambda ind, *a, **k: TradeTargets(
        ind.close, ind.close - 50, ind.close + 50, ind.close + 100,
        50, 50, 100, Direction.BULLISH,
    ))
    monkeypatch.setattr(multi, "apply_pre_trade_filters", lambda **kw: [])

    result = simulate_multi({
        "BTC": AssetInput(_pre(100), _cfg()),
        "ETH": AssetInput(_pre(50), _cfg()),
    }, "2024-01-01", "2024-01-02")

    assert result.trades == 2  # one per-symbol slot, not one global slot
    assert set(result.per_symbol) == {"BTC", "ETH"}
    assert {t.symbol for t in result.portfolio.trades} == {"BTC", "ETH"}
    btc = next(t for t in result.portfolio.trades if t.symbol == "BTC")
    eth = next(t for t in result.portfolio.trades if t.symbol == "ETH")
    # ETH is alphabetically second and sizes from the same balance after BTC's fee.
    assert eth.size * eth.entry_price < btc.size * btc.entry_price


def test_multi_asset_rejects_mismatched_starting_balances():
    a, b = _cfg(), _cfg()
    b.backtesting.initial_balance = 2000
    try:
        simulate_multi({"A": AssetInput(_pre(100), a),
                        "B": AssetInput(_pre(100), b)},
                       "2024-01-01", "2024-01-02")
    except ValueError as exc:
        assert "initial_balance" in str(exc)
    else:
        raise AssertionError("mismatched balances must be rejected")


def test_committed_exposure_counts_open_and_pending_slots():
    from llm_trading_bot.entry import PendingEntry
    from llm_trading_bot.portfolio import Portfolio

    port = Portfolio(initial_balance=1000)
    trade = port.open_trade(
        "LONG", 100, "t0", 90, 120, 130, leverage=10, risk_pct=.02,
    )
    pending = PendingEntry(
        "LONG", 100, 90, 120, 130, 5, .03, .5, 1, "t1",
    )
    slots, margin, notional = committed_exposure(port, [pending], port.balance)
    expected_pending_margin = port.balance * .03
    assert slots == 2
    assert margin == pytest.approx(
        trade.remaining_size * trade.entry_price / 10 + expected_pending_margin
    )
    assert notional == pytest.approx(
        trade.remaining_size * trade.entry_price + expected_pending_margin * 5
    )


def test_exposure_caps_scale_to_tighter_remaining_capacity():
    strategy = {
        "portfolio_risk_multiplier": .8,
        "global_max_margin_pct": .10,
        "global_max_notional_pct": 2.0,
    }
    # Proposed risk is 2% * .8 = 1.6%. Margin leaves 4%, but the 2x-balance
    # notional cap leaves only 25 / (1000 * 25) = 0.1% risk.
    got = apply_exposure_caps(
        .02, 25, 1000, committed_margin=60, committed_notional=1975,
        strategy=strategy,
    )
    assert got == pytest.approx(.001)


def test_global_slot_cap_includes_resting_maker_orders(monkeypatch):
    monkeypatch.setattr(multi, "compute_composite_score", lambda **kw: ScoringResult(
        direction=Direction.BULLISH, confidence=80,
        signal_strength=SignalStrength.STRONG, raw_score=50,
        category_scores=[], indicators=kw["indicators_by_tf"], reasons=[],
    ))
    monkeypatch.setattr(multi, "calculate_targets", lambda ind, *a, **k: TradeTargets(
        ind.close, ind.close - 50, ind.close + 50, ind.close + 100,
        50, 50, 100, Direction.BULLISH,
    ))
    monkeypatch.setattr(multi, "apply_pre_trade_filters", lambda **kw: [])
    a, b = _cfg(), _cfg()
    a.trading.entry_mode = b.trading.entry_mode = "maker"
    result = simulate_multi(
        {"BTC": AssetInput(_pre(100), a), "ETH": AssetInput(_pre(50), b)},
        "2024-01-01", "2024-01-02",
        strat={"global_max_positions": 1},
    )
    assert result.trades == 1
