import numpy as np
import pandas as pd
import pytest

from premodel import backtest, config
from premodel.data import devig, load_matches

needs_data = pytest.mark.skipif(not any(config.RAW_DIR.glob("E0_*.csv")),
                                reason="no raw data; run `python -m premodel download`")


def test_devig_rows_sum_to_one():
    odds = np.array([[1.9, 3.5, 4.2], [1.2, 7.0, 13.0]])
    p = devig(odds)
    assert np.allclose(p.sum(axis=1), 1)
    assert (p[:, 0] > p[:, 2]).all()


@needs_data
def test_loaded_matches_are_clean():
    m = load_matches()
    assert m.date.is_monotonic_increasing
    assert set(m.result.unique()) <= {0, 1, 2}
    assert m[["open_h", "close_h"]].notna().all().all()
    assert (m[["open_h", "open_d", "open_a"]] > 1).all().all()


@needs_data
def test_walk_forward_never_sees_the_future(monkeypatch):
    """Every fit must use only matches played before the date being predicted."""
    m = load_matches()
    m = m[m.season.isin(["2016/17", "2017/18", "2018/19"])].reset_index(drop=True)
    target_dates = sorted(m[m.season == "2018/19"].date.unique())[:5]
    seen = []
    real_fit = backtest.dc.fit

    def spy_fit(home, away, hg, ag, days_ago, xi, init=None):
        seen.append(np.min(days_ago))
        return real_fit(home, away, hg, ag, days_ago, xi, init)

    monkeypatch.setattr(backtest.dc, "fit", spy_fit)
    sub = m[m.date <= pd.Timestamp(target_dates[-1])]
    preds = backtest.walk_forward(sub, 0.003, ["2018/19"])
    assert len(seen) == 5
    assert min(seen) >= 1  # the most recent training match is at least a day old
    assert np.allclose(preds[backtest.PROB_COLS].sum(axis=1), 1)
