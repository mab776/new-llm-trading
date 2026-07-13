"""Research-only maker queue/fill sensitivity guards."""

from __future__ import annotations

import pytest

from opt.fastbt import deterministic_maker_fill, maker_queue_eligible


def test_zero_penetration_matches_touch_rule() -> None:
    assert maker_queue_eligible("LONG", 100, 110, 100, 0)
    assert maker_queue_eligible("SHORT", 100, 100, 90, 0)
    assert not maker_queue_eligible("LONG", 100, 110, 100.01, 0)
    assert not maker_queue_eligible("SHORT", 100, 99.99, 90, 0)


def test_positive_penetration_requires_trade_through_limit() -> None:
    assert not maker_queue_eligible("LONG", 100, 110, 99.99, 2)
    assert maker_queue_eligible("LONG", 100, 110, 99.98, 2)
    assert not maker_queue_eligible("SHORT", 100, 100.01, 90, 2)
    assert maker_queue_eligible("SHORT", 100, 100.02, 90, 2)


def test_probabilistic_fill_is_reproducible_and_bounded() -> None:
    first = deterministic_maker_fill(.5, 73, "BTC|t|LONG|100")
    assert deterministic_maker_fill(.5, 73, "BTC|t|LONG|100") is first
    assert deterministic_maker_fill(0, 73, "x") is False
    assert deterministic_maker_fill(1, 73, "x") is True
    with pytest.raises(ValueError):
        deterministic_maker_fill(1.01, 73, "x")
    with pytest.raises(ValueError):
        maker_queue_eligible("LONG", 100, 101, 99, -1)
