import json

import numpy as np
import pandas as pd
import pytest

from premodel import config, forward
from premodel.commitment import canonical_json

NOW = pd.Timestamp("2026-10-15T09:00:00Z")
GIT = ("abc123", False)
TEAMS = [f"T{i}" for i in range(6)]


def make_matches(start="2026-06-01", end="2026-10-25", seed=0) -> pd.DataFrame:
    """A small synthetic league: one match every day between two random teams."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range(start, end, freq="D"):
        h, a = rng.choice(TEAMS, 2, replace=False)
        rows.append({"season": forward.season_of(d), "date": d, "home": h, "away": a,
                     "hg": int(rng.poisson(1.5)), "ag": int(rng.poisson(1.1))})
    m = pd.DataFrame(rows)
    m["result"] = np.select([m.hg > m.ag, m.hg == m.ag], [0, 1], 2)
    return m


def make_fixtures(rows) -> pd.DataFrame:
    """rows: (home, away, kickoff UTC string)."""
    return pd.DataFrame({
        "home": [r[0] for r in rows],
        "away": [r[1] for r in rows],
        "kickoff_utc": pd.to_datetime([r[2] for r in rows], utc=True),
        "open_h": 2.0, "open_d": 3.4, "open_a": 3.9, "open_src": "market_avg",
    })


FIXTURES = make_fixtures([
    ("T0", "T1", "2026-10-17T14:00:00Z"),
    ("T2", "T3", "2026-10-17T11:30:00Z"),
    ("T4", "T5", "2026-10-18T15:30:00Z"),
    ("T1", "T2", "2026-10-14T19:00:00Z"),  # already kicked off at NOW
])


@pytest.fixture
def dirs(tmp_path):
    return {"pending_dir": tmp_path / "pending", "revealed_dir": tmp_path / "revealed"}


def write_committed_round(pending_dir, rid, fixtures):
    d = pending_dir / rid
    d.mkdir(parents=True)
    (d / "predictions.json").write_bytes(canonical_json({"round_id": rid, "fixtures": fixtures}))
    (d / "commit.json").write_text("{}")


# --- fixtures.csv parsing ---------------------------------------------------------------

def test_load_fixtures_converts_uk_time_to_utc_and_picks_odds(tmp_path):
    csv = tmp_path / "fixtures.csv"
    csv.write_text(
        "﻿Div,Date,Time,HomeTeam,AwayTeam,PSH,PSD,PSA,AvgH,AvgD,AvgA\n"
        "E0,17/10/2026,12:30,Arsenal ,Leeds,1.50,4.50,7.00,1.48,4.40,6.80\n"  # BST
        "E0,07/11/2026,15:00,Hull,Everton,,,,2.90,3.30,2.50\n"                # GMT
        "E1,17/10/2026,15:00,Derby,Stoke,2.0,3.3,3.9,2.0,3.3,3.9\n",
        encoding="utf-8")
    fx = forward.load_fixtures(csv)
    assert list(fx.home) == ["Arsenal", "Hull"]  # E1 dropped, names stripped
    assert [forward.iso_utc(k) for k in fx.kickoff_utc] == [
        "2026-10-17T11:30:00Z", "2026-11-07T15:00:00Z"]
    assert list(fx.open_src) == ["pinnacle", "market_avg"]
    assert fx.loc[0, "open_h"] == 1.50 and fx.loc[1, "open_a"] == 2.50


# --- selecting the round ----------------------------------------------------------------

def test_predict_excludes_fixtures_already_kicked_off(dirs):
    payload = forward.predict_round(make_matches(), FIXTURES, NOW, git=GIT, **dirs)
    pairs = [(f["home"], f["away"]) for f in payload["fixtures"]]
    assert pairs == [("T2", "T3"), ("T0", "T1"), ("T4", "T5")]  # sorted by kickoff
    assert payload["round_id"] == "E0-2026-10-17"


def test_predict_raises_when_nothing_is_upcoming(dirs):
    with pytest.raises(forward.NoFixturesError):
        forward.predict_round(make_matches(), FIXTURES, pd.Timestamp("2026-10-19", tz="UTC"),
                              git=GIT, **dirs)


def test_predict_skips_fixtures_in_committed_rounds_only(dirs):
    committed = [{"home": "T2", "away": "T3", "kickoff_utc": "2026-10-17T11:30:00Z"}]
    write_committed_round(dirs["pending_dir"], "E0-2026-10-17", committed)
    draft = dirs["pending_dir"] / "E0-2026-10-18"  # uncommitted draft: does not count
    draft.mkdir()
    (draft / "predictions.json").write_bytes(canonical_json(
        {"fixtures": [{"home": "T4", "away": "T5", "kickoff_utc": "2026-10-18T15:30:00Z"}]}))

    payload = forward.predict_round(make_matches(), FIXTURES, NOW, git=GIT, **dirs)
    assert [(f["home"], f["away"]) for f in payload["fixtures"]] == [("T0", "T1"), ("T4", "T5")]


def test_unknown_team_name_fails_loudly(dirs):
    fx = make_fixtures([("T0", "Nottm Forest", "2026-10-17T14:00:00Z")])
    with pytest.raises(ValueError, match="Nottm Forest"):
        forward.predict_round(make_matches(), fx, NOW, git=GIT, **dirs)


# --- no leakage -------------------------------------------------------------------------

def test_forward_fit_uses_no_matches_on_or_after_first_kickoff_date(dirs, monkeypatch):
    m = make_matches()
    # A team that only plays on/after the first kickoff date: it must never reach the fit.
    poison = pd.DataFrame({"season": "2026/27", "home": "Poison", "away": "T0", "hg": 9,
                           "ag": 0, "result": 0,
                           "date": pd.to_datetime(["2026-10-17", "2026-10-18", "2026-11-01"])})
    m = pd.concat([m, poison], ignore_index=True)
    seen = {}
    real_fit = forward.dc.fit

    def spy_fit(home, away, hg, ag, days_ago, xi, init=None):
        seen.update(teams=set(home) | set(away), min_days=int(np.min(days_ago)),
                    n=len(home), xi=xi)
        return real_fit(home, away, hg, ag, days_ago, xi, init)

    monkeypatch.setattr(forward.dc, "fit", spy_fit)
    payload = forward.predict_round(m, FIXTURES, NOW, git=GIT, **dirs)

    cutoff = pd.Timestamp("2026-10-17")  # UK date of the first kickoff (11:30Z)
    expected = m[(m.date < cutoff) & (m.date >= cutoff - pd.Timedelta(days=config.LOOKBACK_DAYS))]
    assert "Poison" not in seen["teams"]
    assert seen["min_days"] >= 1
    assert seen["n"] == len(expected) == payload["training"]["n_matches"]
    assert payload["training"]["last_match_date"] == "2026-10-16"
    assert seen["xi"] == config.FORWARD_XI


def test_forward_xi_matches_backtest_choice():
    summary = json.loads((config.REPORTS_DIR / "summary.json").read_text())
    assert config.FORWARD_XI == summary["chosen_xi"]


# --- payload ----------------------------------------------------------------------------

def test_probabilities_are_6dp_and_sum_to_exactly_one():
    p = np.array([[0.3333334, 0.3333333, 0.3333333], [0.71234567, 0.18, 0.10765433]])
    micro = forward.to_micro(p)
    assert (micro.sum(axis=1) == forward.MICRO).all()
    assert micro[1].tolist() == [712346, 180000, 107654]


def test_written_round_is_canonical_and_complete(dirs):
    payload = forward.predict_round(make_matches(), FIXTURES, NOW, git=GIT, **dirs)
    path = forward.write_round(payload, **dirs)
    raw = path.read_bytes()
    assert path == dirs["pending_dir"] / "E0-2026-10-17" / "predictions.json"
    assert raw == canonical_json(payload) == canonical_json(json.loads(raw))

    loaded = json.loads(raw)
    assert loaded["model"] == "dixon-coles" and loaded["git_commit"] == "abc123"
    assert loaded["generated_at"] == "2026-10-15T09:00:00Z"
    for f in loaded["fixtures"]:
        micro = [round(f[k] * forward.MICRO) for k in ("p_home", "p_draw", "p_away")]
        assert sum(micro) == forward.MICRO
        assert f["odds_open"] == {"home": 2.0, "draw": 3.4, "away": 3.9}
        assert f["odds_source"] == "market_avg"


def test_write_round_refuses_to_overwrite_a_committed_round(dirs):
    payload = forward.predict_round(make_matches(), FIXTURES, NOW, git=GIT, **dirs)
    path = forward.write_round(payload, **dirs)
    (path.parent / "commit.json").write_text("{}")
    with pytest.raises(FileExistsError):
        forward.write_round(payload, **dirs)


# --- staleness guard --------------------------------------------------------------------

def test_predict_refuses_stale_training_data_with_both_dates(dirs):
    m = make_matches(end="2026-09-30")  # 17 days before the 2026-10-17 first kickoff
    with pytest.raises(forward.StaleDataError) as e:
        forward.predict_round(m, FIXTURES, NOW, git=GIT, **dirs)
    assert "2026-09-30" in str(e.value) and "2026-10-17" in str(e.value)
    assert "--allow-stale" in str(e.value)


def test_predict_stale_boundary_is_inclusive_of_stale_days(dirs):
    assert config.STALE_DAYS == 16
    m = make_matches(end="2026-10-01")  # exactly STALE_DAYS (16) before: allowed
    payload = forward.predict_round(m, FIXTURES, NOW, git=GIT, **dirs)
    assert payload["stale_override"] is False


def test_allow_stale_predicts_and_records_override(dirs):
    m = make_matches(end="2026-09-30")
    payload = forward.predict_round(m, FIXTURES, NOW, git=GIT, allow_stale=True, **dirs)
    assert payload["stale_override"] is True
    assert payload["training"]["last_match_date"] == "2026-09-30"
    path = forward.write_round(payload, **dirs)
    assert json.loads(path.read_bytes())["stale_override"] is True


def test_fresh_data_records_no_override_even_with_allow_stale(dirs):
    payload = forward.predict_round(make_matches(), FIXTURES, NOW, git=GIT, allow_stale=True,
                                    **dirs)
    assert payload["stale_override"] is False
