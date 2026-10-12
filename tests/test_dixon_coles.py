import numpy as np
import pytest
from scipy.optimize import check_grad

from premodel.models import dixon_coles as dc


def simulate_league(n_teams=12, rounds=6, home_adv=0.3, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    att -= att.mean()
    dfn = rng.normal(0, 0.3, n_teams)
    home, away, hg, ag = [], [], [], []
    for _ in range(rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i != j:
                    home.append(teams[i])
                    away.append(teams[j])
                    hg.append(rng.poisson(np.exp(home_adv + att[i] + dfn[j])))
                    ag.append(rng.poisson(np.exp(att[j] + dfn[i])))
    return np.array(home), np.array(away), np.array(hg), np.array(ag), att


def test_analytic_gradient_matches_numerical():
    home, away, hg, ag, _ = simulate_league(n_teams=6, rounds=2)
    teams = sorted(set(home))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = np.array([idx[t] for t in home])
    ai = np.array([idx[t] for t in away])
    w = np.linspace(0.5, 1, len(home))
    theta = np.random.default_rng(1).normal(0, 0.2, 2 * n + 2)
    theta[-1] = -0.05
    args = (hi, ai, hg, ag, w, n)
    err = check_grad(lambda t: dc._neg_ll_and_grad(t, *args)[0],
                     lambda t: dc._neg_ll_and_grad(t, *args)[1], theta)
    assert err < 1e-3


def test_fit_recovers_simulated_parameters():
    home, away, hg, ag, true_att = simulate_league()
    p = dc.fit(home, away, hg, ag, np.zeros(len(home)), xi=0.0)
    assert p.gamma == pytest.approx(0.3, abs=0.1)
    order = np.argsort([int(t[1:]) for t in p.teams])
    assert np.corrcoef(p.att[order], true_att)[0, 1] > 0.8


def test_score_matrix_is_a_distribution():
    m = dc.score_matrix(1.6, 1.1, -0.1)
    assert m.sum() == pytest.approx(1)
    assert (m >= 0).all()


def test_rho_shifts_mass_to_draws():
    neg = dc.score_matrix(1.4, 1.1, -0.1)
    zero = dc.score_matrix(1.4, 1.1, 0.0)
    assert np.trace(neg) > np.trace(zero)


def test_predictions_sum_to_one_and_unseen_teams_use_prior():
    home, away, hg, ag, _ = simulate_league()
    p = dc.fit(home, away, hg, ag, np.zeros(len(home)), xi=0.0)
    prior = dc.promoted_prior(p)
    probs, xg = dc.predict_1x2(p, ["T0", "Newcomer"], ["T1", "T2"], prior)
    assert np.allclose(probs.sum(axis=1), 1)
    assert (xg > 0).all()
