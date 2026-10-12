"""Finished Premier League results from football-data.org (API v4).

football-data.co.uk's E0.csv lags a few days behind each round, so this is a second
results source. The CSV stays authoritative: API rows only fill matches the CSV does
not have yet (same season, home and away), and they carry no odds.

The API key comes from env FOOTBALL_DATA_API_KEY, or a FOOTBALL_DATA_API_KEY=... line
in .env. It is sent only as the X-Auth-Token header and never printed or logged.
Each run makes at most one request; the response is cached in data/processed/.

The API does not always return every finished match (on 2026-10-12 a response omitted
three matches an earlier response that day had listed as FINISHED). A finished match
never un-finishes, so the cache is cumulative: finished matches from the previous
cached response that a new response omits are carried over (by match id, same season).
"""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from premodel import config
from premodel.data import season_of

KEY_VAR = "FOOTBALL_DATA_API_KEY"
ODDS_COLS = ["open_h", "open_d", "open_a", "close_h", "close_d", "close_a"]


class ApiUnavailable(RuntimeError):
    """No key, or the request failed. Callers fall back to the cache / calendar check."""


@dataclass(frozen=True)
class ApiResults:
    rows: pd.DataFrame  # load_matches() columns + kickoff_utc + source="api"
    live: bool          # fetched during this run (False: read from the cache)
    fetched_at: str


def load_dotenv(path: Path = config.ENV_PATH):
    """Set variables from a simple KEY=VALUE .env file, without overriding the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def fetch(url: str = config.RESULTS_API_URL, cache: Path = config.RESULTS_API_CACHE,
          now: pd.Timestamp | None = None, timeout: float = 30) -> dict:
    """One API request; returns {"fetched_at", "response"} and writes it to `cache`."""
    load_dotenv()
    key = os.environ.get(KEY_VAR)
    if not key:
        raise ApiUnavailable(f"{KEY_VAR} is not set")
    req = urllib.request.Request(url, headers={"X-Auth-Token": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ApiUnavailable(f"HTTP {e.code} from football-data.org") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise ApiUnavailable(f"request failed ({type(e).__name__})") from None
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    carried = _carry_over(body, load_cache(cache))
    doc = {"fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "carried_over": carried,
           "response": body}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    tmp.replace(cache)
    return doc


def _carry_over(body: dict, prev: dict | None) -> int:
    """Append finished matches from `prev` that `body` omits (same season). Returns count."""
    if not prev:
        return 0
    old = prev["response"]
    if old.get("filters", {}).get("season") != body.get("filters", {}).get("season"):
        return 0
    ids = {m["id"] for m in body["matches"]}
    carry = [m for m in old["matches"] if m["status"] == "FINISHED" and m["id"] not in ids]
    body["matches"] = body["matches"] + carry
    return len(carry)


def parse(doc: dict, names: dict[str, str] = config.API_TEAM_NAMES) -> pd.DataFrame:
    """API response -> rows shaped like data.load_matches(), plus kickoff_utc and source."""
    rows, unmapped = [], set()
    for m in doc["response"]["matches"]:
        if m["status"] != "FINISHED":
            continue
        home, away = (names.get(m[t]["shortName"]) for t in ("homeTeam", "awayTeam"))
        unmapped |= {m[t]["shortName"] for t, n in (("homeTeam", home), ("awayTeam", away))
                     if n is None}
        ft = m["score"]["fullTime"]
        if ft["home"] is None or ft["away"] is None:
            raise ValueError(f"FINISHED match without a full-time score: {m.get('id')}")
        kickoff = pd.Timestamp(m["utcDate"])
        date = kickoff.tz_convert(config.KICKOFF_TZ).normalize().tz_localize(None)  # UK date
        rows.append({"season": season_of(date), "date": date, "home": home, "away": away,
                     "hg": int(ft["home"]), "ag": int(ft["away"]), "kickoff_utc": kickoff})
    if unmapped:
        raise ValueError(f"football-data.org team names missing from config.API_TEAM_NAMES: "
                         f"{sorted(unmapped)}")
    df = pd.DataFrame(rows, columns=["season", "date", "home", "away", "hg", "ag",
                                     "kickoff_utc"])
    for c in ODDS_COLS:
        df[c] = np.nan
    df["open_src"] = df["close_src"] = ""
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], [0, 1], 2)
    df["source"] = "api"
    return df.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


def load_cache(cache: Path = config.RESULTS_API_CACHE) -> dict | None:
    return json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None


def get_api_results(refresh: bool = True, cache: Path = config.RESULTS_API_CACHE,
                    fetcher=fetch) -> tuple[ApiResults | None, str]:
    """(results, status message). Live if `refresh` and the request works, else the cache.

    Parsing errors (e.g. an unmapped team name) are raised, never swallowed.
    """
    reason = "--no-refresh"
    if refresh:
        try:
            doc = fetcher(cache=cache)
            res = ApiResults(parse(doc), True, doc["fetched_at"])
            extra = (f" ({doc['carried_over']} kept from the previous response)"
                     if doc.get("carried_over") else "")
            return res, f"results API: live, {len(res.rows)} finished matches{extra}"
        except ApiUnavailable as e:
            reason = str(e)
    doc = load_cache(cache)
    if doc is None:
        return None, f"results API unavailable ({reason}), no cache"
    res = ApiResults(parse(doc), False, doc["fetched_at"])
    return res, (f"results API not live ({reason}); using cached response from "
                 f"{res.fetched_at} ({len(res.rows)} matches)")


def merge_results(csv: pd.DataFrame, api: pd.DataFrame | None) -> pd.DataFrame:
    """CSV rows are authoritative; API rows only add (season, home, away) the CSV lacks."""
    csv = csv.assign(source="csv")
    if api is None or api.empty:
        return csv
    have = set(zip(csv.season, csv.home, csv.away, strict=True))
    extra = api.loc[[k not in have for k in zip(api.season, api.home, api.away, strict=True)],
                    list(csv.columns)]
    if extra.empty:
        return csv
    merged = pd.concat([csv, extra], ignore_index=True)
    return merged.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
