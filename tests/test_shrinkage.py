import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from premodel import backtest, config
from premodel.data import load_matches
from premodel.experiments import shrinkage as sh
from premodel.models import dixon_coles as dc
from test_dixon_coles import simulate_league

needs_data = pytest.mark.skipif(not any(config.RAW_DIR.glob("E0_*.csv")),
                                reason="no raw data; run `python -m premodel download`")
SEASONS = ["2014/15", "2015/16", "2016/17"]


def synthetic_seasons(n_teams=8, seed=0) -> pd.DataFrame:
    """Double round-robin seasons; the bottom 3 by points are replaced by new teams."""
    rng = np.random.default_rng(seed)
    strength = {f"T{i}": (rng.normal(0, 0.3), rng.normal(0, 0.3)) for i in range(n_teams)}
    league, rows, new = list(strength), [], n_teams
    for k, season in enumerate(SEASONS):
        day = pd.Timestamp(f"{2014 + k}-08-10")
        for h in league:
            for a in league:
                if h != a:
                    (ah, dh), (aa, da) = strength[h], strength[a]
                    hg, ag = rng.poisson(np.exp(0.25 + ah + da)), rng.poisson(np.exp(aa + dh))
                    rows.append({"season": season, "date": day, "home": h, "away": a,
                                 "hg": int(hg), "ag": int(ag)})
                    day += pd.Timedelta(days=2)
        m = pd.DataFrame(rows)
        down = sh.relegated_teams(m, season)
        for t in down:
            league.remove(t)
            name = f"P{new}"
            strength[name] = (rng.normal(-0.3, 0.1), rng.normal(0.3, 0.1))  # promoted: weaker
            league.append(name)
            new += 1
    m = pd.DataFrame(rows)
    m["result"] = np.select([m.hg > m.ag, m.hg == m.ag], [0, 1], 2)
    return m.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


def args_for(home, away, hg, ag):
    teams = sorted(set(home) | set(away))
    idx = {t: i for i, t in enumerate(teams)}
    return teams, (np.array([idx[t] for t in home]), np.array([idx[t] for t in away]),
                   hg, ag, np.linspace(0.5, 1, len(home)), len(teams))


# --- model variant ----------------------------------------------------------------------

def test_gradient_with_shrinkage_penalty_matches_numerical():
    home, away, hg, ag, _ = simulate_league(n_teams=6, rounds=2)
    teams, args = args_for(home, away, hg, ag)
    rng = np.random.default_rng(3)
    n = len(teams)
    theta = rng.normal(0, 0.2, 2 * n + 2)
    theta[-1] = -0.05
    full = args + (2.5, rng.normal(-0.3, 0.2, n), rng.normal(0.3, 0.2, n))
    err = check_grad(lambda t: dc._neg_ll_and_grad(t, *full)[0],
                     lambda t: dc._neg_ll_and_grad(t, *full)[1], theta)
    assert err < 1e-3


def test_lambda_zero_reproduces_dc_v1_fit_exactly():
    home, away, hg, ag, _ = simulate_league(n_teams=8, rounds=2)
    days = np.arange(len(home))[::-1] + 1
    base = dc.fit(home, away, hg, ag, days, 0.003)
    zero = dc.fit(home, away, hg, ag, days, 0.003,
                  shrink=dc.Shrinkage(0.0, {"T0": (-0.5, 0.5)}))
    assert np.array_equal(base.att, zero.att) and np.array_equal(base.dfn, zero.dfn)
    assert (base.gamma, base.rho) == (zero.gamma, zero.rho)


def test_shrinkage_pulls_a_low_sample_team_to_its_prior_and_barely_moves_the_rest():
    home, away, hg, ag, _ = simulate_league(n_teams=8, rounds=4, seed=2)
    # a newcomer with two matches, both 4-0 defeats: extreme without shrinkage
    home = np.append(home, ["New", "T1"])
    away = np.append(away, ["T0", "New"])
    hg, ag = np.append(hg, [0, 4]), np.append(ag, [4, 0])
    days = np.ones(len(home))
    prior = (-0.4, 0.3)
    free = dc.fit(home, away, hg, ag, days, 0.0)
    mild = dc.fit(home, away, hg, ag, days, 0.0, shrink=dc.Shrinkage(2.0, {"New": prior}))
    strong = dc.fit(home, away, hg, ag, days, 0.0, shrink=dc.Shrinkage(20.0, {"New": prior}))
    i = free.index()["New"]
    est = [free.index()[f"T{k}"] for k in range(8)]  # 56 matches each

    def centred(p):  # ratings relative to the established teams' mean (the sum-to-zero
        return p.att[est] - p.att[est].mean()  # reference shifts when "New" moves)

    assert free.att[i] < -3                           # the problem being fixed
    assert abs(mild.att[i] - free.att[i]) > 3         # mild lam already moves the newcomer
    assert np.abs(centred(mild) - centred(free)).max() < 0.1  # ...but not the others
    assert abs(strong.att[i] - prior[0]) < 0.15 and abs(strong.dfn[i] - prior[1]) < 0.15


# --- promoted labels and prior: no leakage -----------------------------------------------

def test_relegated_and_promoted_teams_follow_the_points_table():
    m = synthetic_seasons()
    down = sh.relegated_teams(m, "2014/15")
    start = m[m.season == "2015/16"].date.min()
    new = sh.season_teams(m, "2015/16") - sh.season_teams(m, "2014/15")
    assert len(down) == 3 and len(new) == 3
    assert set(down) == sh.season_teams(m, "2014/15") - sh.season_teams(m, "2015/16")
    # causal: only promoted teams that have already played count
    assert sh.promoted_teams(m, "2015/16", start) == set()
    late = start + pd.Timedelta(days=6)
    seen = m[(m.season == "2015/16") & (m.date < late)]
    assert sh.promoted_teams(m, "2015/16", late) == new & (set(seen.home) | set(seen.away))
    assert sh.promoted_teams(m, "2014/15", late) == set()  # no previous season


def poison_from(m: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Same history before `date`; everything on/after it scrambled, plus an extra team."""
    p = m.copy()
    late = p.date >= date
    p.loc[late, ["hg", "ag"]] = p.loc[late, ["ag", "hg"]].to_numpy() + 3
    p.loc[late & (p.home == p.home.iloc[-1]), "home"] = "Ghost"
    p["result"] = np.select([p.hg > p.ag, p.hg == p.ag], [0, 1], 2)
    return p


def test_promoted_prior_and_labels_use_nothing_from_on_or_after_the_fit_date():
    m = synthetic_seasons()
    date = m[m.season == "2016/17"].date.min() + pd.Timedelta(days=20)
    p = poison_from(m, date)
    for s in ("2015/16", "2016/17"):
        assert sh.promoted_teams(m, s, date) == sh.promoted_teams(p, s, date)
    assert sh.promoted_priors(m, ["2016/17"], 0.003) == sh.promoted_priors(p, ["2016/17"], 0.003)
    # and the prior really is driven by past data: poisoning before the season changes it
    q = poison_from(m, m[m.season == "2015/16"].date.min())
    assert sh.promoted_priors(q, ["2016/17"], 0.003) != sh.promoted_priors(m, ["2016/17"], 0.003)


def test_prior_is_the_running_mean_of_relegated_ratings():
    m = synthetic_seasons()
    r1, r2 = sh.relegated_rating(m, "2015/16", 0.003), sh.relegated_rating(m, "2016/17", 0.003)
    pri = sh.promoted_priors(m, ["2015/16", "2016/17"], 0.003)
    assert sh.relegated_rating(m, "2014/15", 0.003) is None
    assert pri["2015/16"] == pytest.approx(r1)
    assert pri["2016/17"] == pytest.approx(tuple(np.mean([r1, r2], axis=0)))
    assert r1[0] < 0 < r1[1]  # relegated teams: weak attack, leaky defence


# --- walk-forward harness ---------------------------------------------------------------

def test_walk_forward_without_shrinkage_and_with_lambda_zero_equals_backtest():
    m = synthetic_seasons()
    seasons = ["2016/17"]
    m = m[m.date < m[m.season == "2016/17"].date.min() + pd.Timedelta(days=30)]
    ref = backtest.walk_forward(m, 0.003, seasons)
    priors = sh.promoted_priors(m, seasons, 0.003)
    for got in (sh.walk_forward(m, 0.003, seasons),
                sh.walk_forward(m, 0.003, seasons, lam=0.0, priors=priors)):
        assert np.array_equal(got[sh.PROB_COLS].to_numpy(), ref[backtest.PROB_COLS].to_numpy())


def test_walk_forward_with_shrinkage_never_sees_the_future(monkeypatch):
    m = synthetic_seasons()
    seasons = ["2016/17"]
    m = m[m.date < m[m.season == "2016/17"].date.min() + pd.Timedelta(days=20)]
    priors = sh.promoted_priors(m, seasons, 0.003)
    seen = []
    real = dc.fit

    def spy(home, away, hg, ag, days_ago, xi, init=None, shrink=None):
        seen.append((int(np.min(days_ago)), shrink))
        return real(home, away, hg, ag, days_ago, xi, init=init, shrink=shrink)

    monkeypatch.setattr(sh.dc, "fit", spy)
    sh.walk_forward(m, 0.003, seasons, lam=4.0, priors=priors)
    assert seen and min(d for d, _ in seen) >= 1
    assert all(s.lam == 4.0 for _, s in seen)
    assert any(s.prior for _, s in seen)  # promoted teams get the prior once they've played
    assert all(v == priors["2016/17"] for _, s in seen for v in s.prior.values())


@needs_data
def test_relegated_teams_match_reality_in_every_completed_season():
    m = load_matches()
    seasons = sorted(m.season.unique())
    for prev, s in zip(seasons, seasons[1:], strict=False):
        actual = sh.season_teams(m, prev) - sh.season_teams(m, s)
        assert set(sh.relegated_teams(m, prev)) == actual, prev
