import pytest

from openwebui_filter import calc_trend_score
from opt.search_scoring_points import objective, sample_points


INPUT = dict(
    price=100, ema_9=110, ema_21=105, ema_50=100, ema_200=90,
    adx=45, plus_di=30, minus_di=10, macd_hist=2,
    macd_line=3, macd_signal=1,
)


def test_empty_point_overrides_are_digit_identical():
    assert calc_trend_score(**INPUT) == calc_trend_score(**INPUT, points={})


def test_point_override_changes_only_requested_award():
    base, _ = calc_trend_score(**INPUT)
    changed, _ = calc_trend_score(**INPUT, points={"trend.ema_stack": 35})
    assert changed == base + 5


def test_unknown_point_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown scoring point"):
        calc_trend_score(**INPUT, points={"trend.typo": 1})


def test_scoring_sampler_stays_inside_constrained_surface():
    import random
    points = sample_points(random.Random(1))
    assert len(points) == 9
    assert 20 <= points["trend.ema_stack"] <= 40


def test_scoring_objective_rejects_red_fold():
    assert objective({"worst_fold": -1, "max_dd": 1,
                      "total_trades": 1000, "geo_pct": 100}) < -1e9
