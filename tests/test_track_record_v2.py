"""Scoring, reveal and verify for schema v1 and v2 rounds; head-to-head; live rule."""
import copy
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from fake_chain import FakeSolanaClient, make_signer
from premodel import backtest, config, stats
from premodel import commitment as cm
from premodel import track_record as tr
from premodel.experiments import championship as ch
from premodel.metrics import score_all
from test_track_record import (
    LAST_KICKOFF,
    RID,
    expected_frame,
    payload,
    results_df,
    write_round,
)

JOINT = [(0.55, 0.27, 0.18), (0.25, 0.33, 0.42), (0.40, 0.30, 0.30)]  # dc_joint per fixture


def payload_v2(rid=RID):
    """A schema v2 round: same fixtures, dc_v1 as in the v1 helper plus dc_joint."""
    v1 = payload(rid)
    fixtures = []
    for f, j in zip(v1["fixtures"], JOINT, strict=True):
        g = {k: f[k] for k in ("home", "away", "kickoff_utc", "odds_open", "odds_source")}
        g["probs"] = {"dc_v1": {"p_home": f["p_home"], "p_draw": f["p_draw"],
                                "p_away": f["p_away"]},
                      "dc_joint": dict(zip(("p_home", "p_draw", "p_away"), j, strict=True))}
        fixtures.append(g)
    return {"schema": "premodel/predictions/v2", "round_id": rid, "model_version": "0.2.0",
            "primary_model": "dc_joint", "models": {"dc_joint": {"xi": 0.005},
                                                    "dc_v1": {"xi": 0.003}},
            "generated_at": "2026-10-16T09:00:00Z", "git_commit": "0" * 40,
            "git_dirty": False, "stale_override": False, "staleness_check": "api_completeness",
            "live_evaluation_rule": config.LIVE_EVALUATION["rule"], "data": {},
            "fixtures": fixtures}


@pytest.fixture
def env(tmp_path):
    return {"revealed": tmp_path / "revealed", "pending": tmp_path / "pending",
            "csv": tmp_path / "track_record.csv", "now": LAST_KICKOFF + timedelta(days=2)}


def score(env, results):
    return tr.run_score(results, env["now"], revealed_dir=env["revealed"], path=env["csv"])


# --- scoring ----------------------------------------------------------------------------

def test_v2_round_fills_both_models_columns(env):
    write_round(env["revealed"] / RID, payload_v2())
    row = score(env, results_df()).iloc[0]
    df, y = expected_frame()
    joint = df.assign(p_h=[0.55, 0.25], p_d=[0.27, 0.33], p_a=[0.18, 0.42])
    for mdl, frame in (("dc_v1", df), ("dc_joint", joint)):
        want = score_all(frame[["p_h", "p_d", "p_a"]].to_numpy(), y)
        for m in tr.METRICS:
            assert row[f"{m}_{mdl}"] == pytest.approx(want[m], abs=1e-6)
        sim = backtest.simulate_bets(frame, 0.0, np.random.default_rng(0))
        assert row[f"n_bets_{mdl}"] == sim["n_bets"]
        assert row[f"mean_clv_{mdl}"] == pytest.approx(sim["mean_clv"], abs=1e-6)
    assert row.schema == "premodel/predictions/v2" and row.primary_model == "dc_joint"
    assert (row.n_scored, row.n_unscored) == (2, 1)


def test_v1_round_fills_only_dc_v1_columns(env):
    write_round(env["revealed"] / RID, payload())
    row = score(env, results_df()).iloc[0]
    assert row.primary_model == "dc_v1" and row.schema == "premodel/predictions/v1"
    assert not np.isnan(row.rps_dc_v1)
    assert all(np.isnan(row[f"{c}_dc_joint"])
               for c in ("rps", "brier", "log_loss", "n_bets", "mean_clv"))


@pytest.mark.parametrize("build", [payload, payload_v2])
def test_commit_verify_reveal_and_score_work_for_v1_and_v2(env, build):
    rd = env["pending"] / RID
    rd.mkdir(parents=True)
    (rd / cm.PRED_FILE).write_bytes(cm.canonical_json(build()))
    canon = "Canon1111111111111111111111111111111111111"
    first = cm.kickoffs(build())[0]
    client = FakeSolanaClient(block_time=int(first.timestamp()) - 3600)
    cm.commit_round(rd, client, make_signer(canon), first - timedelta(hours=2),
                    git_dirty=False, repo_root=env["pending"].parent / "repo",
                    canonical_signer=canon)
    assert cm.verify_round(rd, client, signer=canon)[0]
    dest = tr.reveal_round(rd, env["now"], env["revealed"])
    assert cm.verify_round(dest, client, signer=canon)[0]
    row = score(env, results_df()).iloc[0]
    assert row.n_scored == 2 and not np.isnan(row.rps_dc_v1)


# --- legacy track_record.csv ------------------------------------------------------------

LEGACY_HEADER = ("round_id,first_kickoff_utc,n_fixtures,n_scored,n_unscored,rps_model,rps_open,"
                 "rps_close,brier_model,brier_open,brier_close,log_loss_model,log_loss_open,"
                 "log_loss_close,n_bets,mean_clv,clv_pending,signature,scored_at")


def test_legacy_track_record_columns_are_mapped_and_kept(env):
    env["csv"].write_text(LEGACY_HEADER + "\n" + "E0-2026-10-12,2026-10-12T19:00:00Z,1,1,0,"
                          "0.7,0.35,0.38,1.4,0.77,0.83,2.9,1.2,1.3,1,-0.003,0,SIG,"
                          "2026-10-13T03:04:24Z\n")
    old = tr.read_track_record(env["csv"])
    assert list(old.columns) == tr.COLUMNS
    r = old.iloc[0]
    assert (r.rps_dc_v1, r.n_bets_dc_v1, r.mean_clv_dc_v1) == (0.7, 1, -0.003)
    assert r.schema == "premodel/predictions/v1" and r.primary_model == "dc_v1"
    assert np.isnan(r.rps_dc_joint)
    write_round(env["revealed"] / RID, payload_v2())
    table = score(env, results_df())
    assert list(table.round_id) == ["E0-2026-10-12", RID]
    assert table.iloc[0].rps_dc_v1 == 0.7 and table.iloc[0].signature == "SIG"


# --- head-to-head and new teams ---------------------------------------------------------

def test_head_to_head_is_a_date_block_bootstrap_over_v2_matches(env):
    write_round(env["revealed"] / RID, payload_v2())
    write_round(env["revealed"] / "E0-2026-10-24", payload("E0-2026-10-24", [
        ("Leeds", "Hull", "2026-10-24T14:00:00Z", (0.4, 0.3, 0.3), (2.5, 3.3, 3.0))]))
    res = results_df(list(FIX_RESULTS))
    frames = [tr.score_round(d, res, env["now"]).frame for d in tr.revealed_rounds(
        env["revealed"])]
    h2h = tr.head_to_head(frames, res)
    assert h2h["n"] == 2                                   # the v1 round is not included
    v2 = frames[0][frames[0].result.notna()]
    y = v2.result.to_numpy(dtype=int)
    diff = (ch.per_match_rps(v2[tr.p_cols("dc_joint")].to_numpy(), y)
            - ch.per_match_rps(v2[tr.p_cols("dc_v1")].to_numpy(), y))
    assert h2h["diff"] == stats.date_block_bootstrap(diff, v2.date.to_numpy())
    text = tr.summary(frames, res)
    assert "head-to-head on schema v2 matches" in text and "non-decisive" in text
    assert "not yet due (2 of 150" in text
    assert "all       n=2 matches, k=1 dates" in text
    assert "CI not shown: only 1 match dates (need >= 10)" in text and "95% CI" not in text


FIX_RESULTS = [("Arsenal", "Leeds", 2, 1, (2.0, 3.5, 3.9), "csv"),
               ("Hull", "Everton", 1, 1, (3.1, 3.2, 2.45), "csv"),
               ("Leeds", "Hull", 0, 0, (2.4, 3.3, 3.1), "csv")]


def test_stats_bootstrap_matches_the_experiments_helper():
    rng = np.random.default_rng(5)
    v = rng.normal(0, 1, 200)
    d = (pd.Timestamp("2026-08-01") + pd.to_timedelta(rng.integers(0, 40, 200), "D")).to_numpy()
    a = stats.date_block_bootstrap(v, d, np.random.default_rng(1), n=400)
    b = ch.date_block_bootstrap(v, d, np.random.default_rng(1), n=400)
    assert a["ci95"] == pytest.approx(b["ci95"]) and a["mean"] == pytest.approx(b["mean"])
    assert stats.date_block_bootstrap([], [])["n_matches"] == 0


def test_new_team_flags_mark_first_ten_e0_games_after_entering():
    def season(s, teams, start):
        rows, day = [], pd.Timestamp(start)
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append({"season": s, "date": day, "home": h, "away": a})
                    day += pd.Timedelta(days=1)
        return rows
    res = pd.DataFrame(season("2025/26", ["A", "B", "C", "D"], "2025-08-01")
                       + season("2026/27", ["A", "B", "C", "P"], "2026-08-01"))
    cur = res[res.season == "2026/27"].reset_index(drop=True)
    flags = tr.new_team_flags(cur, res, k=2)
    games, expected = 0, []
    for h, a in zip(cur.home, cur.away, strict=True):
        if "P" in (h, a):
            games += 1
        expected.append("P" in (h, a) and games <= 2)
    assert flags.tolist() == expected


# --- the pre-registered live evaluation rule ---------------------------------------------

def h2h(n, lo, hi, last="2027-05-30T15:00:00Z", unscored=0, n_dates=100):
    return {"n": n, "diff": {"mean": (lo + hi) / 2, "ci95": [lo, hi], "n_matches": n,
                             "n_dates": n_dates},
            "last_kickoff": pd.Timestamp(last), "unscored_before_season_end": unscored}


@pytest.mark.parametrize("case, expected", [
    (None, "not yet due (0 of 150"),
    (h2h(149, 0.01, 0.02), "not yet due (149 of 150"),
    (h2h(400, 0.01, 0.02, last="2027-05-23T14:00:00Z"), "not yet due"),
    (h2h(400, 0.01, 0.02, unscored=1), "not yet due"),
    (h2h(150, 0.0001, 0.02), "revert the primary model to dc_v1"),
    (h2h(150, 0.0, 0.02), "keep dc_joint"),
    (h2h(150, -0.01, 0.005), "keep dc_joint"),
    (h2h(150, -0.02, -0.01), "keep dc_joint"),
    (h2h(150, 0.01, 0.02, n_dates=9), "DUE but not evaluable: only 9 match dates"),
    (h2h(150, -0.02, -0.01, n_dates=9), "neither keep nor revert"),
    (h2h(150, 0.01, 0.02, n_dates=10), "revert the primary model to dc_v1"),
    (h2h(149, 0.01, 0.02, n_dates=3), "not yet due"),       # not due: k is irrelevant
])
def test_evaluation_status_boundaries(case, expected):
    assert expected in tr.evaluation_status(case)


def test_rule_text_is_unchanged_from_config():
    rule = copy.deepcopy(config.LIVE_EVALUATION)
    assert rule["min_v2_matches"] == 150 and "entirely above 0" in rule["rule"]


# --- too few date blocks for a CI -------------------------------------------------------

def block(k, n=None, mean=-0.01):
    return {"mean": mean, "ci95": [-0.03, 0.01], "n_matches": n or 3 * k, "n_dates": k}


@pytest.mark.parametrize("k, shown", [(1, False), (3, False), (9, False), (10, True),
                                      (11, True), (40, True)])
def test_ci_is_suppressed_below_min_blocks_and_shown_at_or_above(k, shown):
    assert config.MIN_BOOTSTRAP_BLOCKS == 10
    line = tr.h2h_line("all", block(k))
    assert line.startswith(f"  all       n={3 * k} matches, k={k} dates  -0.0100")
    assert ("95% CI [-0.0300, +0.0100]" in line) is shown
    assert (f"CI not shown: only {k} match dates (need >= 10)" in line) is not shown


def test_h2h_line_with_no_matches_prints_n_and_k_only():
    assert tr.h2h_line("new teams", {"mean": float("nan"), "ci95": [float("nan")] * 2,
                                     "n_matches": 0, "n_dates": 0}) == \
        "  new teams n=0 matches, k=0 dates  -"


def test_min_bootstrap_blocks_is_read_from_config(monkeypatch):
    monkeypatch.setattr(config, "MIN_BOOTSTRAP_BLOCKS", 3)
    assert "95% CI" in tr.h2h_line("all", block(3))
    assert "need >= 3" in tr.h2h_line("all", block(2))
    assert "not evaluable" in tr.evaluation_status(h2h(150, 0.01, 0.02, n_dates=2))
