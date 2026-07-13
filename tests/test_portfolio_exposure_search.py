"""Selection tests for the shared-portfolio exposure search."""

from opt.portfolio_exposure import TRAIN_DD_BUFFER, select_on_train


def _row(geo, dd, worst=10):
    return {"strat": {"id": geo}, "train": {
        "geo_pct": geo, "max_dd": dd, "worst_fold": worst,
    }}


def test_selects_highest_return_inside_predeclared_dd_buffer():
    selected = select_on_train([
        _row(100, TRAIN_DD_BUFFER + .01),
        _row(80, TRAIN_DD_BUFFER),
        _row(70, 15),
    ])
    assert selected["strat"]["id"] == 80
