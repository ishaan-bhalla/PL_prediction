import numpy as np
import pandas as pd
import pytest

from premodel import backtest
from premodel.experiments import championship as ch
from premodel.experiments import shrinkage as sh


def odds_frame(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p_true = rng.dirichlet([4, 2.5, 3], n)
    margin = rng.uniform(1.03, 1.08, (n, 1))
    open_ = 1 / (np.clip(p_true + rng.normal(0, 0.03, (n, 3)), 0.03, None) * margin)
    close = 1 / (np.clip(p_true + rng.normal(0, 0.015, (n, 3)), 0.03, None) * margin)
    p = rng.dirichlet([4, 2.5, 3], n)
    y = np.array([rng.choice(3, p=q / q.sum()) for q in p_true])
    return pd.DataFrame({
        "season": "2018/19", "date": pd.Timestamp("2018-08-04")
        + pd.to_timedelta(np.repeat(np.arange(n // 4), 4), unit="D"),
        "home": [f"H{i}" for i in range(n)], "away": [f"A{i}" for i in range(n)],
        "p_h": p[:, 0], "p_d": p[:, 1], "p_a": p[:, 2],
        "open_h": open_[:, 0], "open_d": open_[:, 1], "open_a": open_[:, 2],
        "close_h": close[:, 0], "close_d": close[:, 1], "close_a": close[:, 2],
        "open_src": "pinnacle", "close_src": "pinnacle", "result": y})


def test_model_bets_follow_the_backtest_rule():
    df = odds_frame()
    b = ch.model_bets(df)
    sim = backtest.simulate_bets(df, 0.0, np.random.default_rng(0))
    assert len(b) == sim["n_bets"] and b.clv.mean() == pytest.approx(sim["mean_clv"])
    assert b.pnl.mean() == pytest.approx(sim["roi"])


def test_random_clv_is_the_expectation_of_the_backtest_random_baseline():
    df = odds_frame(n=200)
    expected = ch.model_bets(df, -np.inf).random_clv.mean()
    draws = [backtest.random_bet_baseline(df, np.random.default_rng(s))["mean_clv"]
             for s in range(400)]
    assert np.mean(draws) == pytest.approx(expected, abs=2e-3)


def test_matches_without_odds_are_excluded_from_bets():
    df = odds_frame(n=40)
    df.loc[df.index[:5], "open_h"] = np.nan
    assert not ch.model_bets(df).index.isin(df.index[:5]).any()


def test_blend_weight_zero_is_the_open_market_and_informative_models_get_weight():
    valid = odds_frame(seed=1)
    w, curve = ch.choose_blend(valid)
    open_only = sh.per_match_rps(ch.devig(valid[ch.OPEN].to_numpy()),
                                 valid.result.to_numpy()).mean()
    assert curve[0]["w"] == 0.0 and curve[0]["rps"] == pytest.approx(open_only)
    assert w <= 0.1                            # a noise model earns (almost) no weight
    informed = valid.copy()                    # a model close to the truth earns a lot
    informed[["p_h", "p_d", "p_a"]] = np.eye(3)[informed.result] * 0.6 + 0.4 / 3
    assert ch.choose_blend(informed)[0] >= 0.5


def test_date_block_bootstrap_matches_the_shrinkage_helper_at_95():
    rng_vals = np.random.default_rng(3)
    v = rng_vals.normal(0, 1, 300)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng_vals.integers(0, 60, 300), "D")
    a = ch.date_block_bootstrap(v, dates.to_numpy(), np.random.default_rng(9), n=500)
    b = sh.block_bootstrap(v, dates.to_numpy(), np.random.default_rng(9), n=500)
    assert a["ci95"] == pytest.approx(b["ci95"]) and a["mean"] == pytest.approx(b["mean"])
    assert a["ci983"][0] <= a["ci95"][0] and a["ci983"][1] >= a["ci95"][1]


def test_joint_bootstrap_cancels_shared_date_shocks():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2020-01-01", periods=100).to_numpy(), 4)
    shock = np.repeat(rng.normal(0, 1, 100), 4)
    e1 = shock + 0.05 + rng.normal(0, 0.05, 400)
    e0 = shock + rng.normal(0, 0.05, 400)
    joint = ch.joint_difference_bootstrap(e1, dates, e0, dates, np.random.default_rng(1), 2000)
    indep = ch.independent_difference_bootstrap(e1, dates, e0, dates,
                                                np.random.default_rng(1), 2000)
    assert joint["ci95"][0] > 0                       # the true +0.05 is detected
    assert (joint["ci95"][1] - joint["ci95"][0]) < 0.2 * (indep["ci95"][1] - indep["ci95"][0])


def test_entrants_split_relegated_from_above_and_promoted_from_below():
    def season(s, teams, start):
        rows, day = [], pd.Timestamp(start)
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append({"season": s, "date": day, "home": h, "away": a})
                    day += pd.Timedelta(days=1)
        return rows
    lower = pd.DataFrame(season("2017/18", ["A", "B", "C", "D"], "2017-08-01")
                         + season("2018/19", ["A", "B", "R", "P"], "2018-08-01"))
    upper = pd.DataFrame(season("2017/18", ["R", "X", "Y"], "2017-08-01"))
    test = lower[lower.season == "2018/19"]
    lab = ch.entrants(test, lower, upper, k=2)
    got = {(h, a): v for h, a, v in zip(test.home, test.away, lab, strict=True) if v}
    games = {"R": 0, "P": 0}
    expected = {}
    for h, a in zip(test.home, test.away, strict=True):
        tags = []
        for t in (h, a):
            if t in games:
                games[t] += 1
                if games[t] <= 2:
                    tags.append("from_above" if t == "R" else "from_below")
        if tags:
            expected[(h, a)] = "from_above" if "from_above" in tags else "from_below"
    assert got == expected and set(got.values()) == {"from_above", "from_below"}


def test_figure_renders_from_the_committed_summary(tmp_path):
    from premodel.experiments.championship_plot import render
    summary = ch.OUT_DIR / "summary.json"
    if not summary.exists():
        pytest.skip("no experiment summary yet")
    assert render(summary, tmp_path / "fig.png").stat().st_size > 10_000
