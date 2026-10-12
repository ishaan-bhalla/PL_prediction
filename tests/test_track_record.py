from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from premodel import backtest
from premodel import commitment as cm
from premodel import track_record as tr
from premodel.commitment import canonical_json
from premodel.data import devig
from premodel.metrics import score_all

RID = "E0-2026-10-17"
LAST_KICKOFF = datetime(2026, 10, 18, 15, 30, tzinfo=timezone.utc)
FIXTURES = [  # (home, away, kickoff, (p_h, p_d, p_a), open odds)
    ("Arsenal", "Leeds", "2026-10-17T11:30:00Z", (0.5, 0.3, 0.2), (2.1, 3.4, 3.6)),
    ("Hull", "Everton", "2026-10-17T14:00:00Z", (0.29, 0.29, 0.42), (3.0, 3.3, 2.5)),
    ("Fulham", "Chelsea", "2026-10-18T15:30:00Z", (0.45, 0.28, 0.27), (2.2, 3.3, 3.4)),
]
# (home, away, hg, ag, closing odds or None, source); Fulham v Chelsea is postponed
RESULTS = [("Arsenal", "Leeds", 2, 1, (2.0, 3.5, 3.9), "csv"),
           ("Hull", "Everton", 1, 1, (3.1, 3.2, 2.45), "csv")]


def payload(rid=RID, fixtures=FIXTURES):
    return {"round_id": rid, "fixtures": [
        {"home": h, "away": a, "kickoff_utc": k, "p_home": p[0], "p_draw": p[1],
         "p_away": p[2], "odds_open": dict(zip(["home", "draw", "away"], o, strict=True)),
         "odds_source": "market_avg"} for h, a, k, p, o in fixtures]}


def write_round(d, data, commit=True):
    d.mkdir(parents=True)
    raw = canonical_json(data)
    (d / cm.PRED_FILE).write_bytes(raw)
    salt = cm.new_salt()
    (d / cm.SALT_FILE).write_text(salt + "\n")
    if commit:
        (d / cm.COMMIT_FILE).write_bytes(canonical_json(
            {"commitment": cm.compute_commitment(salt, raw), "signature": f"sig-{d.name}"}))
    return d


def results_df(rows=RESULTS):
    out = []
    for h, a, hg, ag, close, src in rows:
        c = close or (np.nan,) * 3
        out.append({"season": "2026/27", "date": pd.Timestamp("2026-10-17"), "home": h,
                    "away": a, "hg": hg, "ag": ag,
                    "result": 0 if hg > ag else 1 if hg == ag else 2,
                    "close_h": c[0], "close_d": c[1], "close_a": c[2], "source": src})
    return pd.DataFrame(out)


@pytest.fixture
def env(tmp_path):
    return {"revealed": tmp_path / "revealed", "pending": tmp_path / "pending",
            "csv": tmp_path / "track_record.csv", "now": LAST_KICKOFF + timedelta(days=2)}


def score(env, results, round_id=None):
    return tr.run_score(results, env["now"], round_id=round_id,
                        revealed_dir=env["revealed"], path=env["csv"])


# --- reveal -----------------------------------------------------------------------------

def test_reveal_moves_the_three_files_after_last_kickoff_plus_3h(env):
    src = write_round(env["pending"] / RID, payload())
    files = {n: (src / n).read_bytes() for n in tr.ROUND_FILES}
    dest = tr.reveal_round(src, LAST_KICKOFF + timedelta(hours=3), env["revealed"])
    assert dest == env["revealed"] / RID and not src.exists()
    assert {n: (dest / n).read_bytes() for n in tr.ROUND_FILES} == files
    assert sorted(p.name for p in env["revealed"].iterdir()) == [RID]  # no temp dir left


@pytest.mark.parametrize("now", [LAST_KICKOFF, LAST_KICKOFF + timedelta(hours=3, seconds=-1)])
def test_reveal_refuses_before_last_kickoff_plus_3h_unless_forced(env, now):
    src = write_round(env["pending"] / RID, payload())
    with pytest.raises(tr.RevealRefused, match="2026-10-18T18:30:00Z"):
        tr.reveal_round(src, now, env["revealed"])
    assert src.exists() and not (env["revealed"] / RID).exists()
    assert tr.reveal_round(src, now, env["revealed"], force=True).exists()


def test_reveal_refuses_without_commit_json(env):
    src = write_round(env["pending"] / RID, payload(), commit=False)
    with pytest.raises(tr.RevealRefused, match="commit it before revealing"):
        tr.reveal_round(src, env["now"], env["revealed"], force=True)


def test_reveal_refuses_tampered_files_and_existing_destination(env):
    src = write_round(env["pending"] / RID, payload())
    (src / cm.SALT_FILE).write_text(cm.new_salt() + "\n")
    with pytest.raises(tr.RevealRefused, match="do not match the commitment"):
        tr.reveal_round(src, env["now"], env["revealed"])
    other = write_round(env["pending"] / "E0-2026-10-24", payload("E0-2026-10-24"))
    (env["revealed"] / "E0-2026-10-24").mkdir(parents=True)
    with pytest.raises(tr.RevealRefused, match="already exists"):
        tr.reveal_round(other, env["now"], env["revealed"])


# --- score ------------------------------------------------------------------------------

def expected_frame():
    """The two played matches, in the backtest's column layout."""
    df = pd.DataFrame({"p_h": [0.5, 0.29], "p_d": [0.3, 0.29], "p_a": [0.2, 0.42],
                       "open_h": [2.1, 3.0], "open_d": [3.4, 3.3], "open_a": [3.6, 2.5],
                       "close_h": [2.0, 3.1], "close_d": [3.5, 3.2], "close_a": [3.9, 2.45],
                       "result": [0, 1]})
    return df, df.result.to_numpy()


def test_score_matches_metrics_on_three_matches_with_a_postponed_one(env, capsys):
    write_round(env["revealed"] / RID, payload())
    table = score(env, results_df())
    row = table.iloc[0]
    df, y = expected_frame()
    model = score_all(df[["p_h", "p_d", "p_a"]].to_numpy(), y)
    open_ = score_all(devig(df[["open_h", "open_d", "open_a"]].to_numpy()), y)
    close = score_all(devig(df[["close_h", "close_d", "close_a"]].to_numpy()), y)
    for m in tr.METRICS:
        assert row[f"{m}_model"] == pytest.approx(model[m], abs=1e-6)
        assert row[f"{m}_open"] == pytest.approx(open_[m], abs=1e-6)
        assert row[f"{m}_close"] == pytest.approx(close[m], abs=1e-6)
    sim = backtest.simulate_bets(df, 0.0, np.random.default_rng(0))
    assert row.n_bets == sim["n_bets"] == 2
    assert row.mean_clv == pytest.approx(sim["mean_clv"], abs=1e-6)
    assert (row.n_fixtures, row.n_scored, row.n_unscored, row.clv_pending) == (3, 2, 1, 0)
    assert row.signature == f"sig-{RID}" and row.first_kickoff_utc == "2026-10-17T11:30:00Z"
    out = capsys.readouterr().out
    assert "skipped (no result yet): 2026-10-18 Fulham v Chelsea" in out
    assert list(pd.read_csv(env["csv"]).columns) == tr.COLUMNS


def test_rescoring_replaces_the_round_row_and_keeps_others(env):
    write_round(env["revealed"] / RID, payload())
    other = "E0-2026-10-24"
    write_round(env["revealed"] / other, payload(other, [
        ("Leeds", "Hull", "2026-10-24T14:00:00Z", (0.4, 0.3, 0.3), (2.5, 3.3, 3.0))]))
    score(env, results_df())
    played = results_df(RESULTS + [("Fulham", "Chelsea", 0, 2, (2.3, 3.3, 3.2), "csv")])
    env["now"] += timedelta(days=30)
    table = score(env, played, round_id=RID)
    csv = pd.read_csv(env["csv"])
    assert list(csv.round_id) == [RID, other]
    r = csv.set_index("round_id").loc[RID]
    assert (r.n_scored, r.n_unscored) == (3, 0)
    assert r.scored_at == "2026-11-19T15:30:00Z" and len(table) == 2


def test_clv_pending_when_closing_odds_are_missing_then_filled(env, capsys):
    write_round(env["revealed"] / RID, payload())
    api_only = [RESULTS[0], ("Hull", "Everton", 1, 1, None, "api")]  # CSV has not caught up
    row = score(env, results_df(api_only)).iloc[0]
    df, y = expected_frame()
    assert row.rps_model == pytest.approx(score_all(df[["p_h", "p_d", "p_a"]].to_numpy(),
                                                    y)["rps"], abs=1e-6)
    assert not np.isnan(row.rps_open)
    assert np.isnan(row.rps_close) and np.isnan(row.mean_clv)
    assert row.clv_pending == 1 and row.n_bets == 2
    line = env["csv"].read_text().splitlines()[1]
    assert ",,1," in line  # blank mean_clv, then clv_pending
    assert "1 awaiting closing odds" in capsys.readouterr().out

    row = score(env, results_df()).iloc[0]  # CSV caught up
    assert row.clv_pending == 0 and not np.isnan(row.rps_close) and not np.isnan(row.mean_clv)


def test_score_only_reads_revealed_rounds_and_checks_their_hash(env):
    write_round(env["pending"] / RID, payload())
    with pytest.raises(FileNotFoundError, match="only revealed rounds"):
        score(env, results_df(), round_id=RID)
    d = write_round(env["revealed"] / RID, payload())
    data = payload()
    data["fixtures"][0]["p_home"] = 0.6  # edited after the commitment
    (d / cm.PRED_FILE).write_bytes(canonical_json(data))
    with pytest.raises(ValueError, match="files hash to"):
        score(env, results_df())


def test_summary_pools_matches_across_rounds(env, capsys):
    write_round(env["revealed"] / RID, payload())
    score(env, results_df())
    out = capsys.readouterr().out
    assert "Track record: 1 round(s), 2 scored matches, 1 unscored" in out
    assert "market close" in out and "mean CLV" in out


def test_bets_skip_matches_without_open_odds():
    df = pd.DataFrame({"p_h": [0.9], "p_d": [0.05], "p_a": [0.05], "open_h": [np.nan],
                       "open_d": [np.nan], "open_a": [np.nan], "close_h": [2.0],
                       "close_d": [3.0], "close_a": [4.0]})
    assert tr.bets(df).empty


def test_score_with_no_revealed_rounds_writes_nothing(env, capsys):
    assert score(env, results_df()) is None
    assert not env["csv"].exists() and "nothing to score" in capsys.readouterr().out
