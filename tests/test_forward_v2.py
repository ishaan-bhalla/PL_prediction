"""Schema v2 rounds: dc_joint (primary) and dc_v1 (shadow) in one committed file."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from premodel import commitment as cm
from premodel import config, forward, results
from premodel.commitment import canonical_json
from premodel.models import dixon_coles_joint as dj
from test_forward import FIXTURES, GIT, NOW, TEAMS, make_matches

ELC = json.loads((Path(__file__).parent / "data" / "football_data_org_elc.json")
                 .read_text(encoding="utf-8"))


@pytest.fixture
def dirs(tmp_path):
    return {"pending_dir": tmp_path / "pending", "revealed_dir": tmp_path / "revealed"}


def other_leagues(seed=1):
    """E1 (teams U0-U5 plus T5, which also plays in E0) and E2 (W0-W5 plus U5)."""
    out = {}
    for lg, teams, link in (("E1", [f"U{i}" for i in range(6)], "T5"),
                            ("E2", [f"W{i}" for i in range(6)], "U5")):
        m = make_matches(seed=seed)
        rename = dict(zip(TEAMS, teams[:5] + [link], strict=True))
        out[lg] = m.assign(home=m.home.map(rename), away=m.away.map(rename))
        seed += 1
    return out


def predict(dirs, **kw):
    return forward.predict_round(make_matches(), FIXTURES, NOW, git=GIT,
                                 other_leagues=kw.pop("others", other_leagues()), **dirs, **kw)


# --- schema v2 --------------------------------------------------------------------------

def test_v2_round_is_canonical_with_both_models_in_millionths(dirs):
    payload = predict(dirs)
    raw = canonical_json(payload)
    assert raw == canonical_json(json.loads(raw)) == canonical_json(copy.deepcopy(payload))
    assert payload["schema"] == forward.SCHEMA == "premodel/predictions/v2"
    assert payload["primary_model"] == "dc_joint"
    assert payload["live_evaluation_rule"] == config.LIVE_EVALUATION["rule"]
    assert set(payload["models"]["dc_joint"]["training"]) == {"E0", "E1", "E2"}
    for f in payload["fixtures"]:
        assert set(f["probs"]) == {"dc_joint", "dc_v1"}
        for b in f["probs"].values():
            micro = [round(b[k] * forward.MICRO) for k in ("p_home", "p_draw", "p_away")]
            assert sum(micro) == forward.MICRO and all(m > 0 for m in micro)
    assert cm.fixture_probs(payload)[0]["dc_joint"] != cm.fixture_probs(payload)[0]["dc_v1"]


@pytest.mark.parametrize("model", ["dc_joint", "dc_v1"])
@pytest.mark.parametrize("key", ["p_home", "p_draw", "p_away"])
def test_commitment_changes_when_either_models_probability_moves(dirs, model, key):
    payload = predict(dirs)
    salt = "cd" * 32
    base = cm.compute_commitment(salt, canonical_json(payload))
    changed = copy.deepcopy(payload)
    b = changed["fixtures"][1]["probs"][model]
    b[key] = round(b[key] + 1e-6, 6)
    assert cm.compute_commitment(salt, canonical_json(changed)) != base


def test_shadow_dc_v1_block_is_exactly_the_v1_prediction(dirs):
    payload = predict(dirs)
    fx = forward.select_upcoming(FIXTURES, NOW, set())
    p, train = forward.fit_and_predict(make_matches(), fx, config.FORWARD_XI)
    micro = forward.to_micro(p)
    got = [[round(f["probs"]["dc_v1"][k] * forward.MICRO) for k in ("p_home", "p_draw",
                                                                     "p_away")]
           for f in payload["fixtures"]]
    assert got == micro.tolist()
    assert payload["models"]["dc_v1"]["training"]["n_matches"] == len(train)


def test_joint_with_e0_only_matches_dc_v1_at_the_same_xi(dirs, monkeypatch):
    monkeypatch.setattr(config, "FORWARD_JOINT_XI", config.FORWARD_XI)
    fx = forward.select_upcoming(FIXTURES, NOW, set())
    p_joint, _ = forward.fit_joint_and_predict({"E0": make_matches()}, fx, config.FORWARD_XI)
    p_v1, _ = forward.fit_and_predict(make_matches(), fx, config.FORWARD_XI)
    assert np.abs(p_joint - p_v1).max() < 1e-9


def test_fixture_probs_reads_v1_and_v2_and_rejects_unknown(dirs):
    v1 = {"schema": "premodel/predictions/v1",
          "fixtures": [{"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}]}
    assert cm.fixture_probs(v1) == [{"dc_v1": (0.5, 0.3, 0.2)}]
    assert cm.primary_model(v1) == "dc_v1" and cm.primary_model(predict(dirs)) == "dc_joint"
    with pytest.raises(ValueError, match="unknown predictions schema"):
        cm.fixture_probs({"schema": "x/v9", "fixtures": [{}]})


# --- joint live fit: no leakage in any league --------------------------------------------

def poison(m: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    late = m.date >= cutoff
    out = m.copy()
    out.loc[late, ["hg", "ag"]] = out.loc[late, ["ag", "hg"]].to_numpy() + 4
    ghost = pd.DataFrame({"season": "2026/27", "date": [cutoff, cutoff + pd.Timedelta(days=3)],
                          "home": "Ghost", "away": out.home.iloc[0], "hg": 9, "ag": 0,
                          "result": 0})
    return pd.concat([out, ghost], ignore_index=True)


def test_joint_fit_uses_no_match_on_or_after_the_first_kickoff_date(dirs, monkeypatch):
    cutoff = pd.Timestamp("2026-10-17")  # UK date of the first kickoff (11:30Z)
    clean = predict(dirs)
    seen = []
    real = dj.fit

    def spy(home, away, league, hg, ag, days_ago, *a, **k):
        seen.append((set(home) | set(away), int(np.min(days_ago)), set(league)))
        return real(home, away, league, hg, ag, days_ago, *a, **k)

    monkeypatch.setattr(forward.dj, "fit", spy)
    others = {lg: poison(m, cutoff) for lg, m in other_leagues().items()}
    dirty = forward.predict_round(poison(make_matches(), cutoff), FIXTURES, NOW, git=GIT,
                                  other_leagues=others, **dirs)
    teams, min_days, leagues = seen[-1]
    assert "Ghost" not in teams and min_days >= 1 and leagues == {"E0", "E1", "E2"}
    assert [f["probs"] for f in dirty["fixtures"]] == [f["probs"] for f in clean["fixtures"]]
    last = dirty["models"]["dc_joint"]["training"]
    assert all(t["last_match_date"] < "2026-10-17" for t in last.values())


# --- Championship API -------------------------------------------------------------------

def test_elc_names_map_to_the_championship_csv():
    df = results.parse(ELC, config.API_TEAM_NAMES_E1)
    assert list(zip(df.home, df.away, strict=True)) == [("Preston", "Sheffield United"),
                                                        ("Wolves", "Derby")]
    sched = results.schedule(ELC, config.API_TEAM_NAMES_E1)
    assert ("Lincoln", "West Ham") in set(zip(sched.home, sched.away, strict=True))
    assert len(config.API_TEAM_NAMES_E1) == 24


def test_unmapped_championship_name_fails_listing_it():
    doc = copy.deepcopy(ELC)
    doc["response"]["matches"][0]["homeTeam"]["shortName"] = "Hull City"
    with pytest.raises(ValueError, match=r"\['Hull City'\]"):
        results.parse(doc, config.API_TEAM_NAMES_E1)


def test_get_api_results_uses_the_competition_config(tmp_path):
    res, msg = results.get_api_results(True, cache=tmp_path / "c.json",
                                       fetcher=lambda cache: ELC, competition="ELC")
    assert res.live and "(ELC)" in msg and set(res.rows.home) == {"Preston", "Wolves"}
    with pytest.raises(ValueError, match="API_TEAM_NAMES"):  # PL names do not cover E1
        results.get_api_results(True, cache=tmp_path / "d.json", fetcher=lambda cache: ELC)


def test_incomplete_championship_data_is_reported_not_blocking(dirs):
    api = results.ApiResults(results.parse(ELC, config.API_TEAM_NAMES_E1),
                             results.schedule(ELC, config.API_TEAM_NAMES_E1), True, "x")
    payload = predict(dirs, e1_api=api)
    e1 = payload["data"]["E1_completeness"]
    assert e1["checked"] and e1["complete"] is False
    assert any("3 of 552" in p for p in e1["problems"])
    assert payload["data"]["E2_last_match_date"] == "2026-10-16"


# --- configuration ----------------------------------------------------------------------

def test_forward_joint_xi_matches_the_joint_experiment_selection():
    s = json.loads((config.REPORTS_DIR / "experiments" / "joint" / "summary.json").read_text())
    chosen = s["selection"]["chosen"]
    assert config.FORWARD_JOINT_XI == chosen["xi"] and chosen["variant"] == "plain"
    assert config.MODELS["dc_joint"]["variant"] == "plain"


def test_live_evaluation_rule_states_its_power():
    ev = config.LIVE_EVALUATION
    assert ev["min_v2_matches"] == 150 and ev["season_end"] == "2027-05-30"
    assert "0.007" in ev["power"] and "cannot confirm an improvement" in ev["power"]
