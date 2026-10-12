import copy
import io
import json
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from premodel import forward, results
from test_forward import FIXTURES, GIT, NOW, make_matches

DOC = json.loads((Path(__file__).parent / "data" / "football_data_org_finished.json")
                 .read_text(encoding="utf-8"))
# make_matches() reuses (home, away) pairs within a season, unlike a real league, so
# API-only rows below use teams T6-T8 that never appear in the synthetic CSV.
SYNTH_NAMES = {f"Team {i}": f"T{i}" for i in range(9)}


def api_doc(matches) -> dict:
    """matches: (home, away, utcDate, hg, ag) with synthetic 'Team i' names."""
    return {"fetched_at": "2026-10-15T08:00:00Z", "response": {"matches": [
        {"id": i, "utcDate": d, "status": "FINISHED",
         "homeTeam": {"shortName": f"Team {h[1:]}"}, "awayTeam": {"shortName": f"Team {a[1:]}"},
         "score": {"fullTime": {"home": hg, "away": ag}}}
        for i, (h, a, d, hg, ag) in enumerate(matches)]}}


def api_results(matches, live=True) -> results.ApiResults:
    return results.ApiResults(results.parse(api_doc(matches), SYNTH_NAMES), live,
                              "2026-10-15T08:00:00Z")


@pytest.fixture
def dirs(tmp_path):
    return {"pending_dir": tmp_path / "pending", "revealed_dir": tmp_path / "revealed"}


# --- parsing and names ------------------------------------------------------------------

def test_parse_maps_names_dates_and_results():
    df = results.parse(DOC)
    assert list(zip(df.home, df.away, strict=True)) == [
        ("Hull", "Nott'm Forest"), ("Brighton", "Coventry"), ("Liverpool", "Man City")]
    # 23:30 UTC on 22 Aug is 00:30 BST on 23 Aug: dates are UK dates, as in the CSV
    assert [f"{d:%Y-%m-%d}" for d in df.date] == ["2026-08-21", "2026-08-23", "2027-01-02"]
    assert list(df.season) == ["2026/27"] * 3
    assert list(df.result) == [0, 1, 2] and list(df.hg) == [2, 0, 0]
    assert (df.source == "api").all() and df[results.ODDS_COLS].isna().all().all()


def test_unmapped_team_names_fail_loudly_listing_all_of_them():
    doc = copy.deepcopy(DOC)
    doc["response"]["matches"][0]["homeTeam"]["shortName"] = "Hull Tigers"
    doc["response"]["matches"][2]["awayTeam"]["shortName"] = "Manchester Blue"
    with pytest.raises(ValueError, match=r"\['Hull Tigers', 'Manchester Blue'\]"):
        results.parse(doc)


def test_finished_match_without_score_is_an_error():
    doc = copy.deepcopy(DOC)
    doc["response"]["matches"][0]["score"]["fullTime"]["home"] = None
    with pytest.raises(ValueError, match="without a full-time score"):
        results.parse(doc)


# --- merge ------------------------------------------------------------------------------

def test_merge_keeps_csv_rows_and_only_fills_missing_matches():
    csv = pd.DataFrame({"season": ["2026/27"], "date": pd.to_datetime(["2026-08-21"]),
                        "home": ["Hull"], "away": ["Nott'm Forest"], "hg": [5], "ag": [5],
                        "result": [1], "close_h": [2.0]})
    merged = results.merge_results(csv, results.parse(DOC))
    assert len(merged) == 3 and list(merged.source) == ["csv", "api", "api"]
    hull = merged.iloc[0]
    assert (hull.hg, hull.ag, hull.close_h) == (5, 5, 2.0)  # CSV wins on conflict
    assert np.isnan(merged.iloc[1].close_h)
    assert list(merged.columns) == list(csv.columns) + ["source"]
    assert (results.merge_results(csv, None).source == "csv").all()


# --- completeness check in predict ------------------------------------------------------

def test_completeness_passes_when_api_gaps_are_filled(dirs):
    m = make_matches(end="2026-10-10")  # CSV lags; API has the rest of the week
    api = api_results([("T6", "T0", "2026-10-12T14:00:00Z", 1, 0),
                       ("T7", "T8", "2026-10-16T19:00:00Z", 2, 2)])
    payload = forward.predict_round(m, FIXTURES, NOW, git=GIT, api=api, **dirs)
    assert payload["staleness_check"] == "api_completeness"
    assert payload["stale_override"] is False
    assert payload["training"]["n_api_rows"] == 2
    assert payload["training"]["last_match_date"] == "2026-10-16"


def test_completeness_fails_on_a_finished_match_missing_from_training(dirs):
    # Finished at 12:30 UK on the first kickoff's date: excluded by the strictly-before-
    # date training rule, so the model would be missing a result it could have had.
    api = api_results([("T6", "T7", "2026-10-17T09:00:00Z", 3, 0)])
    fx = FIXTURES.iloc[:1]  # T0 v T1 at 14:00Z
    with pytest.raises(forward.StaleDataError, match="2026-10-17 T6 v T7"):
        forward.predict_round(make_matches(), fx, NOW, git=GIT, api=api, **dirs)
    payload = forward.predict_round(make_matches(), fx, NOW, git=GIT, api=api,
                                    allow_stale=True, **dirs)
    assert payload["stale_override"] is True


def test_missing_finished_lists_api_matches_absent_from_train():
    api = api_results([("T6", "T7", "2026-10-01T14:00:00Z", 1, 1),
                       ("T2", "T3", "2026-10-20T14:00:00Z", 1, 1)])  # after kickoff: ignored
    train = make_matches(end="2026-09-30")
    assert forward.missing_finished(api.rows, train, pd.Timestamp("2026-10-17", tz="UTC")) \
        == ["2026-10-01 T6 v T7"]


def test_long_break_with_no_finished_matches_passes(dirs):
    m = make_matches(end="2026-09-20")  # 27 days before kickoff: calendar rule would refuse
    payload = forward.predict_round(m, FIXTURES, NOW, git=GIT, api=api_results([]), **dirs)
    assert payload["staleness_check"] == "api_completeness"
    assert payload["stale_override"] is False and payload["training"]["n_api_rows"] == 0


def test_without_live_api_the_calendar_rule_still_applies(dirs):
    m = make_matches(end="2026-09-20")
    cached = api_results([], live=False)
    with pytest.raises(forward.StaleDataError, match="no live results API"):
        forward.predict_round(m, FIXTURES, NOW, git=GIT, api=cached, **dirs)
    payload = forward.predict_round(make_matches(), FIXTURES, NOW, git=GIT, api=cached, **dirs)
    assert payload["staleness_check"] == "calendar"


# --- fetching, cache, key handling (no network) -----------------------------------------

def test_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nexport A_VAR='from file'\nB_VAR=\"x\"\n")
    monkeypatch.setenv("B_VAR", "from env")
    monkeypatch.delenv("A_VAR", raising=False)
    results.load_dotenv(env)
    import os
    assert os.environ["A_VAR"] == "from file" and os.environ["B_VAR"] == "from env"


def test_fetch_without_key_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv(results.KEY_VAR, raising=False)
    monkeypatch.setattr(results.config, "ENV_PATH", tmp_path / "none")
    monkeypatch.setattr(results, "load_dotenv", lambda path=None: None)
    with pytest.raises(results.ApiUnavailable, match="not set"):
        results.fetch(cache=tmp_path / "c.json")


def test_fetch_sends_key_as_header_caches_and_never_leaks_it(monkeypatch, tmp_path):
    secret = "s3cr3t-key-value"
    monkeypatch.setenv(results.KEY_VAR, secret)
    monkeypatch.setattr(results, "load_dotenv", lambda path=None: None)
    seen = {}

    def ok(req, timeout):
        seen["headers"], seen["url"] = dict(req.header_items()), req.full_url
        return io.BytesIO(json.dumps(DOC["response"]).encode())

    monkeypatch.setattr(results.urllib.request, "urlopen", ok)
    cache = tmp_path / "c.json"
    doc = results.fetch(cache=cache, now=pd.Timestamp("2026-10-12T09:00:00Z"))
    assert seen["headers"]["X-auth-token"] == secret and secret not in seen["url"]
    assert results.load_cache(cache) == doc and doc["fetched_at"] == "2026-10-12T09:00:00Z"
    assert secret not in cache.read_text()

    def fail(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, f"bad token {secret}", {}, None)

    monkeypatch.setattr(results.urllib.request, "urlopen", fail)
    with pytest.raises(results.ApiUnavailable) as e:
        results.fetch(cache=cache)
    assert "403" in str(e.value) and secret not in str(e.value)


def test_get_api_results_falls_back_to_cache_when_fetch_fails(tmp_path):
    cache = tmp_path / "c.json"
    calls = []

    def ok(cache):
        calls.append(1)
        cache.write_text(json.dumps(DOC))
        return DOC

    res, msg = results.get_api_results(True, cache=cache, fetcher=ok)
    assert res.live and len(res.rows) == 3 and "live" in msg and len(calls) == 1

    def down(cache):
        raise results.ApiUnavailable("HTTP 503 from football-data.org")

    res, msg = results.get_api_results(True, cache=cache, fetcher=down)
    assert not res.live and len(res.rows) == 3 and "HTTP 503" in msg and "cached" in msg
    res, msg = results.get_api_results(False, cache=cache, fetcher=down)
    assert not res.live and "--no-refresh" in msg
    assert results.get_api_results(True, cache=tmp_path / "x.json", fetcher=down)[0] is None


def test_fetch_keeps_finished_matches_a_new_response_omits(monkeypatch, tmp_path):
    monkeypatch.setenv(results.KEY_VAR, "k")
    monkeypatch.setattr(results, "load_dotenv", lambda path=None: None)
    cache = tmp_path / "c.json"
    cache.write_text(json.dumps(DOC))
    newer = copy.deepcopy(DOC["response"])
    newer["matches"] = [m for m in newer["matches"] if m["id"] != 3]  # API dropped id 3
    newer["matches"][0]["score"]["fullTime"]["home"] = 4              # and corrected id 1
    respond = lambda body: lambda req, timeout: io.BytesIO(json.dumps(body).encode())  # noqa: E731

    monkeypatch.setattr(results.urllib.request, "urlopen", respond(newer))
    doc = results.fetch(cache=cache)
    df = results.parse(doc)
    assert doc["carried_over"] == 1 and len(df) == 3
    assert df.loc[df.home == "Hull", "hg"].item() == 4  # the newer response wins per id

    next_season = copy.deepcopy(newer)
    next_season["filters"]["season"] = "2027"
    next_season["matches"] = []
    monkeypatch.setattr(results.urllib.request, "urlopen", respond(next_season))
    doc = results.fetch(cache=cache)
    assert doc["carried_over"] == 0 and results.parse(doc).empty
