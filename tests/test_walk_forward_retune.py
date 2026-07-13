from opt.walk_forward_retune import TRAIN_WINDOWS, objective, sample_candidate


def test_walk_forward_windows_do_not_overlap_target():
    for _name, train, target in TRAIN_WINDOWS:
        target_start = target[0][1]
        assert all(end < target_start for _label, _start, end in train)


def test_retune_sampler_respects_threshold_order():
    import random
    for _ in range(50):
        candidate = sample_candidate(random.Random(_))
        assert candidate["tier.marginal_threshold_low"] < candidate["tier.strong_threshold"]


def test_objective_rejects_red_training_fold():
    result = {"worst_fold": -0.1, "max_dd": 10, "total_trades": 1000, "geo_pct": 99}
    assert objective(result) < -1e9
