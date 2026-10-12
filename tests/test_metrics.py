import numpy as np
import pytest

from premodel.metrics import brier, calibration_table, log_loss, rps


def test_perfect_forecast_scores_zero():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert rps(p, y) == pytest.approx(0)
    assert brier(p, y) == pytest.approx(0)
    assert log_loss(p, y) == pytest.approx(0, abs=1e-9)


def test_uniform_forecast_known_values():
    y = np.array([0])
    p = np.full((1, 3), 1 / 3)
    assert brier(p, y) == pytest.approx((2 / 3) ** 2 + 2 * (1 / 3) ** 2)
    assert log_loss(p, y) == pytest.approx(np.log(3))


def test_rps_rewards_near_misses():
    """Predicting an away win when the home side wins is worse than predicting a draw."""
    y = np.array([0])
    draw = np.array([[0.0, 1.0, 0.0]])
    away = np.array([[0.0, 0.0, 1.0]])
    assert rps(draw, y) < rps(away, y)


def test_calibration_table_bins():
    rng = np.random.default_rng(0)
    p_home = rng.uniform(0, 1, 5000)
    p = np.column_stack([p_home, (1 - p_home) / 2, (1 - p_home) / 2])
    y = np.where(rng.uniform(size=5000) < p_home, 0, 2)
    rows = calibration_table(p, y, outcome=0)
    for r in rows:
        assert abs(r["pred"] - r["actual"]) < 0.08
