import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from premodel import config
from premodel.data import load_matches
from premodel.experiments import joint as jx
from premodel.models import dixon_coles as dc
from premodel.models import dixon_coles_joint as dj

SEASONS = ["2014/15", "2015/16", "2016/17"]


def two_leagues(n_seasons=3, n_teams=8, n_move=2, rounds=2, seed=0) -> tuple[pd.DataFrame, dict]:
    """E0 (strong) and E1 (weak) on one true scale; after each season the bottom n_move
    of E0 (by true strength) swap with the top n_move of E1."""
    rng = np.random.default_rng(seed)
    truth = {}
    for i in range(n_teams):
        truth[f"A{i}"] = (rng.normal(0.25, 0.15), rng.normal(-0.25, 0.15))
        truth[f"B{i}"] = (rng.normal(-0.35, 0.15), rng.normal(0.35, 0.15))
    league = {"E0": [f"A{i}" for i in range(n_teams)], "E1": [f"B{i}" for i in range(n_teams)]}
    rows = []
    for k, season in enumerate(SEASONS[:n_seasons]):
        for lg, teams in league.items():
            day = pd.Timestamp(f"{2014 + k}-08-09")
            for _ in range(rounds):
                for h in teams:
                    for a in teams:
                        if h != a:
                            (ah, dh), (aa, da) = truth[h], truth[a]
                            rows.append({"season": season, "league": lg, "date": day,
                                         "home": h, "away": a,
                                         "hg": int(rng.poisson(np.exp(0.25 + ah + da))),
                                         "ag": int(rng.poisson(np.exp(aa + dh)))})
                            day += pd.Timedelta(days=1)
        strength = lambda t: truth[t][0] - truth[t][1]  # noqa: E731
        down = sorted(league["E0"], key=strength)[:n_move]
        up = sorted(league["E1"], key=strength)[-n_move:]
        league = {"E0": [t for t in league["E0"] if t not in down] + up,
                  "E1": [t for t in league["E1"] if t not in up] + down}
    m = pd.DataFrame(rows).sort_values(["date", "league", "home"], kind="stable")
    m["result"] = np.select([m.hg > m.ag, m.hg == m.ag], [0, 1], 2)
    return m.reset_index(drop=True), truth


def fit_all(m: pd.DataFrame, date=None, shifts=(), xi=0.0):
    date = date or m.date.max() + pd.Timedelta(days=1)
    tr = m[m.date < date]
    mh, ma = jx.move_directions(tr, tr) if shifts else (None, None)
    days = (date - tr.date).dt.days.to_numpy()
    p = dj.fit(tr.home, tr.away, tr.league, tr.hg, tr.ag, days, xi, mh, ma, shifts)
    return p, (tr.home.to_numpy(), tr.away.to_numpy(), tr.league.to_numpy(),
               tr.hg.to_numpy(), tr.ag.to_numpy(), days, xi, mh, ma)


# --- model ------------------------------------------------------------------------------

def test_gradient_matches_numerical_with_leagues_and_shifts():
    m, _ = two_leagues(n_seasons=2, n_teams=5, rounds=1)
    m = pd.concat([m, m.assign(league="E2", home=m.home + "x", away=m.away + "x")])
    mh, ma = jx.move_directions(m, m)
    p, _ = fit_all(m, shifts=(dj.UP,))
    lay = dj.Layout(len(p.teams), p.leagues, [dj.UP, dj.DOWN])
    idx = p.index()
    args = (np.array([idx[t] for t in m.home]), np.array([idx[t] for t in m.away]),
            np.array([p.leagues.index(lg) for lg in m.league]), mh, ma,
            m.hg.to_numpy(), m.ag.to_numpy(), np.linspace(0.5, 1, len(m)), lay)
    theta = np.random.default_rng(1).normal(0, 0.2, lay.size)
    theta[lay.rho] = -0.05
    err = check_grad(lambda t: dj._neg_ll_and_grad(t, *args)[0],
                     lambda t: dj._neg_ll_and_grad(t, *args)[1], theta)
    assert err < 1e-3


def test_one_league_reduces_to_dc_v1():
    m = load_matches() if any(config.RAW_DIR.glob("E0_*.csv")) else None
    if m is None:
        m, _ = two_leagues(n_seasons=1)
        m = m[m.league == "E0"]
    date = m.date.sort_values().iloc[len(m) // 2]
    tr = m[(m.date < date) & (m.date >= date - pd.Timedelta(days=config.LOOKBACK_DAYS))]
    days = (date - tr.date).dt.days.to_numpy()
    a = dc.fit(tr.home.to_numpy(), tr.away.to_numpy(), tr.hg.to_numpy(), tr.ag.to_numpy(),
               days, 0.003)
    b = dj.fit(tr.home, tr.away, ["E0"] * len(tr), tr.hg, tr.ag, days, 0.003)
    assert np.abs(a.att - b.att).max() < 1e-8 and np.abs(a.dfn - b.dfn).max() < 1e-8
    assert abs(a.gamma - b.gamma["E0"]) < 1e-8 and abs(a.rho - b.rho) < 1e-8
    home = list(tr.home.unique()[:5]) + ["Newcomer"]   # includes the unseen-team fallback
    away = list(tr.away.unique()[5:11])
    pa, _ = dc.predict_1x2(a, home, away, dc.promoted_prior(a))
    pb, _ = dj.predict_1x2(b, home, away, "E0")
    assert np.abs(pa - pb).max() < 1e-9


def test_joint_fit_recovers_movers_and_the_cross_league_scale():
    m, truth = two_leagues(n_teams=10, rounds=4, seed=3)
    p, _ = fit_all(m)
    idx = p.index()
    est = {t: p.att[idx[t]] - p.dfn[idx[t]] for t in p.teams}
    true = {t: truth[t][0] - truth[t][1] for t in p.teams}
    a_only = [t for t in p.teams if p.played_in[t] == {"E0"}]
    b_only = [t for t in p.teams if p.played_in[t] == {"E1"}]
    movers = [t for t in p.teams if len(p.played_in[t]) == 2]
    assert len(movers) == 4

    def centred(d):
        ref = np.mean([d[t] for t in a_only])
        return {t: v - ref for t, v in d.items()}

    est, true = centred(est), centred(true)
    gap_est = np.mean([est[t] for t in b_only])
    gap_true = np.mean([true[t] for t in b_only])
    assert abs(gap_est - gap_true) < 0.2 and gap_true < -0.9  # the leagues are far apart
    assert np.mean([abs(est[t] - true[t]) for t in movers]) < 0.25
    assert p.component["E0"] == p.component["E1"]
    assert abs(p.b["E1"]) < 0.15                               # truth: same goal environment


def test_without_movers_leagues_are_disconnected_and_use_the_fallback():
    m, _ = two_leagues(n_seasons=1, rounds=3)
    p, _ = fit_all(m)
    assert p.component["E0"] != p.component["E1"]
    _, _, from_history = dj.team_rating(p, "B0", "E0")          # E1 history, no link
    assert not from_history
    assert dj.team_rating(p, "B0", "E0")[:2] == dj.league_fallback(p, "E0")
    assert dj.team_rating(p, "A0", "E0")[2]
    # within-league predictions are close to a separate one-league fit; not identical,
    # because rho (the low-score correction) is shared across leagues by design
    e1 = m[m.league == "E1"]
    date = m.date.max() + pd.Timedelta(days=1)
    q = dj.fit(e1.home, e1.away, e1.league, e1.hg, e1.ag, (date - e1.date).dt.days, 0.0)
    pairs = (["B0", "B1", "B2"], ["B3", "B4", "B5"])
    assert np.abs(dj.predict_1x2(p, *pairs, "E1")[0] - dj.predict_1x2(q, *pairs, "E1")[0]
                  ).max() < 0.015


# --- identification of the move-shift variant -------------------------------------------

def smallest_eigenvalues(m, shifts, k=2):
    p, args = fit_all(m, shifts=shifts)
    h, _ = dj.hessian_without_ridge(p, *args, shifts=shifts)
    return np.linalg.eigvalsh(h)[:k]


def test_free_up_and_down_shifts_are_unidentified_when_movers_have_only_one_season():
    # Two seasons: every cross-league link runs through a mover's first season after the
    # move, so (block shift + k, d_up - k, d_down + k) leaves every prediction unchanged.
    m, _ = two_leagues(n_seasons=2, n_teams=8, rounds=3)
    free = smallest_eigenvalues(m, (dj.UP, dj.DOWN))
    constrained = smallest_eigenvalues(m, (dj.UP,))
    assert np.all(np.abs(free) < 1e-6)                # two flat directions (att and dfn)
    assert constrained[0] > 1e-2                      # d_down = 0: identified


def test_constrained_variant_is_identified_with_later_season_movers_too():
    m, _ = two_leagues(n_seasons=3, n_teams=8, rounds=3)
    assert smallest_eigenvalues(m, (dj.UP,))[0] > 1e-2


# --- move directions and leakage --------------------------------------------------------

def test_move_directions_compare_with_the_previous_season():
    m, _ = two_leagues(n_seasons=2)
    s2 = m[m.season == "2015/16"]
    mh, ma = jx.move_directions(s2, m)
    up = set(s2[s2.league == "E0"].home) - set(m[(m.season == "2014/15")
                                                & (m.league == "E0")].home)
    for h, lg, d in zip(s2.home, s2.league, mh, strict=True):
        if h in up and lg == "E0":
            assert d == dj.UP
        elif lg == "E0":
            assert d == dj.NONE
    assert set(mh) == {dj.NONE, dj.UP, dj.DOWN}
    assert (jx.move_directions(m[m.season == "2014/15"], m)[0] == dj.NONE).all()


def test_walk_forward_uses_only_earlier_matches_from_every_league(monkeypatch):
    m, _ = two_leagues(n_seasons=3, n_teams=6, rounds=1)
    m["league"] = m.league.astype(str)
    s3 = m[m.season == "2016/17"]
    stop = s3.date.min() + pd.Timedelta(days=12)
    sub = m[m.date <= stop]
    seen = []
    real = dj.fit

    def spy(home, away, league, hg, ag, days_ago, *a, **k):
        seen.append(int(np.min(days_ago)))
        return real(home, away, league, hg, ag, days_ago, *a, **k)

    monkeypatch.setattr(jx.dj, "fit", spy)
    preds = jx.walk_forward(sub, 0.003, ["2016/17"], shifts=(dj.UP,))
    assert seen and min(seen) >= 1
    monkeypatch.setattr(jx.dj, "fit", real)
    # every league's results on/after a date change nothing before it
    cut = s3.date.min() + pd.Timedelta(days=6)
    poisoned = sub.copy()
    late = poisoned.date >= cut
    poisoned.loc[late, ["hg", "ag"]] = poisoned.loc[late, ["ag", "hg"]].to_numpy() + 2
    a = jx.walk_forward(sub, 0.003, ["2016/17"], shifts=(dj.UP,))
    b = jx.walk_forward(poisoned, 0.003, ["2016/17"], shifts=(dj.UP,))
    early = (a.date <= cut).to_numpy()
    assert np.array_equal(a.loc[early, jx.P].to_numpy(), b.loc[early, jx.P].to_numpy())
    assert len(preds) == len(s3[s3.date <= stop])


@pytest.mark.skipif(not any(config.RAW_DIR.glob("E1_*.csv")), reason="no E1 data")
def test_e1_load_is_unaffected_by_e2_files(tmp_path):
    for p in config.RAW_DIR.glob("E1_*.csv"):
        shutil.copy(p, tmp_path / p.name)
    assert load_matches(league="E1").equals(load_matches(tmp_path, league="E1"))
