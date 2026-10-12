"""Forward predictions for upcoming fixtures, written as a round ready to commit.

A round is every upcoming E0 fixture in football-data.co.uk's fixtures.csv that
has not kicked off yet and is not already part of a committed round. The model
is fitted on matches played strictly before the date of the round's first
kickoff (same xi and lookback window as the backtest), so no result from the
round itself, or anything after it, can leak into the predictions.

Training results are the season CSVs plus, for matches the CSV does not have yet,
football-data.org API results (premodel.results). Before fitting, the data must be
complete against the live API's season schedule: the schedule itself must be whole,
and every match due to kick off before the round (bar postponed, cancelled or
suspended ones) must be FINISHED and in the training window. Without a live API
response the older calendar rule applies instead (newest match at most STALE_DAYS
before the first kickoff).
"""
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from premodel import __version__, config
from premodel.commitment import COMMIT_FILE, PRED_FILE, canonical_json
from premodel.data import download, load_matches, open_odds, season_of
from premodel.models import dixon_coles as dc
from premodel.models import dixon_coles_joint as dj
from premodel.results import ApiResults, get_api_results, merge_results

SCHEMA = "premodel/predictions/v2"
SCHEMA_V1 = "premodel/predictions/v1"
MICRO = 1_000_000  # probabilities are stored in millionths (6 decimals)


class NoFixturesError(RuntimeError):
    """No upcoming, uncommitted fixtures to predict."""


class StaleDataError(RuntimeError):
    """The newest training match is too long before the round's first kickoff."""


def iso_utc(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def download_fixtures(url: str = config.FIXTURES_URL, dest: Path = config.FIXTURES_PATH):
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"saved  {dest.name}")


def load_fixtures(path: Path = config.FIXTURES_PATH) -> pd.DataFrame:
    """E0 fixtures with kickoff converted from UK local time to UTC, plus open odds."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["Div"] == config.LEAGUE_DIV].dropna(subset=["HomeTeam", "AwayTeam"])
    df = df.reset_index(drop=True)
    if df["Time"].isna().any():
        raise ValueError("fixtures.csv has E0 rows without a kickoff Time")
    local = pd.to_datetime(df["Date"].str.strip() + " " + df["Time"].str.strip(),
                           dayfirst=True, format="mixed")
    kickoff = (local.dt.tz_localize(config.KICKOFF_TZ, ambiguous="raise", nonexistent="raise")
               .dt.tz_convert("UTC"))
    odds, src = open_odds(df)
    return pd.DataFrame({
        "home": df["HomeTeam"].str.strip(),
        "away": df["AwayTeam"].str.strip(),
        "kickoff_utc": kickoff,
        "open_h": odds[:, 0], "open_d": odds[:, 1], "open_a": odds[:, 2],
        "open_src": src,
    })


def committed_fixture_keys(pending_dir: Path, revealed_dir: Path) -> set[tuple[str, str, str]]:
    """(season, home, away) of every fixture in a committed or revealed round."""
    dirs = [d for d in pending_dir.glob("*") if (d / COMMIT_FILE).exists()]
    dirs += [d for d in revealed_dir.glob("*") if d.is_dir()]
    keys = set()
    for d in dirs:
        pred = json.loads((d / PRED_FILE).read_text(encoding="utf-8"))
        for f in pred["fixtures"]:
            keys.add((season_of(pd.Timestamp(f["kickoff_utc"])), f["home"], f["away"]))
    return keys


def select_upcoming(fixtures: pd.DataFrame, now: pd.Timestamp,
                    committed: set[tuple[str, str, str]]) -> pd.DataFrame:
    """Fixtures that kick off after `now` and are not already committed."""
    fx = fixtures[fixtures.kickoff_utc > now]
    keys = [(season_of(k), h, a) for k, h, a in zip(fx.kickoff_utc, fx.home, fx.away, strict=True)]
    fx = fx.loc[np.array([k not in committed for k in keys], dtype=bool)]
    return fx.sort_values(["kickoff_utc", "home"], kind="stable").reset_index(drop=True)


def check_team_names(fixtures: pd.DataFrame, known: set[str]):
    unknown = sorted((set(fixtures.home) | set(fixtures.away)) - known)
    if unknown:
        raise ValueError(f"fixture team names not found in the training data: {unknown}. "
                         "Check for a spelling change in fixtures.csv.")


def training_cutoff(fixtures: pd.DataFrame) -> pd.Timestamp:
    """UK-local date of the first kickoff; training uses matches strictly before it."""
    first = fixtures.kickoff_utc.min()
    return first.tz_convert(config.KICKOFF_TZ).normalize().tz_localize(None)


def training_window(matches: pd.DataFrame, fixtures: pd.DataFrame,
                    lookback_days: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    """Matches in the lookback window ending the day before the round's first kickoff."""
    cutoff = training_cutoff(fixtures)
    train = matches[(matches.date < cutoff)
                    & (matches.date >= cutoff - pd.Timedelta(days=lookback_days))]
    if train.empty:
        raise ValueError(f"no training matches before {cutoff.date()}")
    return train


def is_stale(train: pd.DataFrame, fixtures: pd.DataFrame,
             stale_days: int = config.STALE_DAYS) -> tuple[bool, str]:
    """(stale?, message) comparing the newest training match to the first kickoff date."""
    last, first = train.date.max(), training_cutoff(fixtures)
    gap = (first - last).days
    msg = (f"newest training match is {last:%Y-%m-%d}, {gap} days before the first kickoff "
           f"on {first:%Y-%m-%d} (STALE_DAYS = {stale_days})")
    return gap > stale_days, msg


NOT_DUE_STATUSES = {"POSTPONED", "CANCELLED", "SUSPENDED"}


def schedule_problems(sched: pd.DataFrame, train: pd.DataFrame,
                      first_kickoff: pd.Timestamp, season_matches: int | None = None
                      ) -> list[str]:
    """Why the training data is not complete against the API schedule ([] if it is).

    A response that omits matches cannot vouch for anything, so the schedule must have
    config.SEASON_MATCHES distinct (home, away) pairings. Then every match with kickoff
    before `first_kickoff` must be FINISHED and in `train`, unless postponed, cancelled
    or suspended.
    """
    problems = []
    expected = season_matches or config.SEASON_MATCHES
    n = len(set(zip(sched.home, sched.away, strict=True)))
    if n != expected:
        problems.append(f"API schedule lists {n} of {expected} matches "
                        "(incomplete response)")
    due = sched[(sched.kickoff_utc < first_kickoff) & ~sched.status.isin(NOT_DUE_STATUSES)]
    have = set(zip(train.season, train.home, train.away, strict=True))
    for r in due.itertuples(index=False):
        label = f"{r.date:%Y-%m-%d} {r.home} v {r.away}"
        if r.status != "FINISHED":
            problems.append(f"{label} ({r.status})")
        elif (r.season, r.home, r.away) not in have:
            problems.append(f"{label} (missing from training)")
    return problems


def staleness(train: pd.DataFrame, fixtures: pd.DataFrame, api: ApiResults | None,
              stale_days: int = config.STALE_DAYS) -> tuple[str, bool, str]:
    """(check used, stale?, message): API completeness if live, else the calendar rule."""
    if api is not None and api.live:
        problems = schedule_problems(api.schedule, train, fixtures.kickoff_utc.min())
        msg = (f"training data is incomplete against the results API schedule "
               f"({len(problems)} problem(s)): " + "; ".join(problems))
        return "api_completeness", bool(problems), msg
    stale, msg = is_stale(train, fixtures, stale_days)
    return "calendar", stale, f"no live results API, so using the calendar check: {msg}"


def fit_and_predict(matches: pd.DataFrame, fixtures: pd.DataFrame, xi: float,
                    lookback_days: int = config.LOOKBACK_DAYS):
    """Fit Dixon-Coles on matches before the round and return (probs (n, 3), train)."""
    cutoff = training_cutoff(fixtures)
    train = training_window(matches, fixtures, lookback_days)
    params = dc.fit(train.home.to_numpy(), train.away.to_numpy(),
                    train.hg.to_numpy(), train.ag.to_numpy(),
                    (cutoff - train.date).dt.days.to_numpy(), xi)
    p, _ = dc.predict_1x2(params, fixtures.home.to_numpy(), fixtures.away.to_numpy(),
                          dc.promoted_prior(params))
    return p, train


def to_micro(p: np.ndarray) -> np.ndarray:
    """Round to millionths so each row sums to exactly 1 (away takes the remainder)."""
    hd = np.rint(np.asarray(p)[:, :2] * MICRO).astype(np.int64)
    return np.column_stack([hd, MICRO - hd.sum(axis=1)])


def git_info(root: Path = config.ROOT) -> tuple[str, bool]:
    """HEAD commit, and whether the code that produces predictions has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "src", "pyproject.toml"],
                                cwd=root, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return head, bool(status.strip())


def round_id_for(fixtures: pd.DataFrame) -> str:
    return f"{config.LEAGUE_DIV}-{fixtures.kickoff_utc.min():%Y-%m-%d}"


def training_summary(train: pd.DataFrame) -> dict:
    return {"n_matches": int(len(train)),
            "first_match_date": f"{train.date.min():%Y-%m-%d}" if len(train) else None,
            "last_match_date": f"{train.date.max():%Y-%m-%d}" if len(train) else None,
            "n_api_rows": int((train.source == "api").sum()) if "source" in train else 0}


def fit_joint_and_predict(leagues: dict[str, pd.DataFrame], fixtures: pd.DataFrame,
                          xi: float = config.FORWARD_JOINT_XI,
                          lookback_days: int = config.LOOKBACK_DAYS):
    """dc_joint (plain) on every league's matches strictly before the UK date of the first
    kickoff (same window as dc_v1); predict the E0 fixtures. Returns (probs, trains)."""
    cutoff = training_cutoff(fixtures)
    trains = {lg: m[(m.date < cutoff) & (m.date >= cutoff - pd.Timedelta(days=lookback_days))]
              for lg, m in leagues.items()}
    # same row order as the joint experiment, so the live fit reproduces it exactly
    train = (pd.concat([t.assign(league=lg) for lg, t in trains.items()], ignore_index=True)
             .sort_values(["date", "league", "home"], kind="stable"))
    params = dj.fit(train.home.to_numpy(), train.away.to_numpy(), train.league.to_numpy(),
                    train.hg.to_numpy(), train.ag.to_numpy(),
                    (cutoff - train.date).dt.days.to_numpy(), xi)
    p, _ = dj.predict_1x2(params, fixtures.home.to_numpy(), fixtures.away.to_numpy(),
                          config.LEAGUE_DIV)
    return p, trains


def e1_completeness(e1: pd.DataFrame | None, api: ApiResults | None,
                    fixtures: pd.DataFrame) -> dict:
    """Information only (never blocks predict): is the Championship training data complete
    against the live ELC schedule?"""
    if e1 is None or api is None or not api.live:
        return {"checked": False, "complete": None, "problems": []}
    train = training_window(e1, fixtures)
    problems = schedule_problems(api.schedule, train, fixtures.kickoff_utc.min(),
                                 config.RESULTS_API["ELC"]["season_matches"])
    return {"checked": True, "complete": not problems, "problems": problems[:10]}


def _probs_block(pm) -> dict:
    return {"p_home": pm[0] / MICRO, "p_draw": pm[1] / MICRO, "p_away": pm[2] / MICRO}


def build_payload(fixtures: pd.DataFrame, probs_micro: dict, trains: dict,
                  now: pd.Timestamp, git: tuple[str, bool], stale_override: bool = False,
                  staleness_check: str = "calendar", data_info: dict | None = None) -> dict:
    """Schema v2: one file holds every model's probabilities, so one commitment covers all.

    probs_micro: model -> (n, 3) int array in millionths; trains: "dc_v1" -> frame,
    "dc_joint" -> {league: frame}."""
    rows = []
    for k, f in enumerate(fixtures.itertuples(index=False)):
        odds = [f.open_h, f.open_d, f.open_a]
        has_odds = not any(np.isnan(odds))
        rows.append({
            "home": f.home,
            "away": f.away,
            "kickoff_utc": iso_utc(f.kickoff_utc),
            "probs": {name: _probs_block(pm[k]) for name, pm in probs_micro.items()},
            "odds_open": dict(zip(["home", "draw", "away"], map(float, odds), strict=True))
            if has_odds else None,
            "odds_source": f.open_src if has_odds else None,
        })
    models = {}
    for name, spec in config.MODELS.items():
        if name not in probs_micro:
            continue
        t = trains[name]
        models[name] = {**spec, "lookback_days": config.LOOKBACK_DAYS,
                        "training": ({lg: training_summary(x) for lg, x in t.items()}
                                     if isinstance(t, dict) else training_summary(t))}
    return {
        "schema": SCHEMA,
        "round_id": round_id_for(fixtures),
        "model_version": __version__,
        "primary_model": config.PRIMARY_MODEL,
        "models": models,
        "live_evaluation_rule": config.LIVE_EVALUATION["rule"],
        "git_commit": git[0],
        "git_dirty": git[1],
        "generated_at": iso_utc(now),
        "stale_override": stale_override,
        "staleness_check": staleness_check,
        "data": data_info or {},
        "fixtures": rows,
    }


def predict_round(matches: pd.DataFrame, fixtures: pd.DataFrame, now: pd.Timestamp,
                  pending_dir: Path = config.PENDING_DIR,
                  revealed_dir: Path = config.REVEALED_DIR,
                  xi: float = config.FORWARD_XI, git: tuple[str, bool] | None = None,
                  allow_stale: bool = False, stale_days: int = config.STALE_DAYS,
                  api: ApiResults | None = None, other_leagues: dict | None = None,
                  e1_api: ApiResults | None = None) -> dict:
    """Schema v2 predictions for the next round (pure apart from reading committed rounds).

    `matches` (E0 CSV) are merged with `api` rows; both models use this E0 data, and the
    staleness / completeness check (E0) applies to both. dc_v1 (shadow) is fitted on E0
    exactly as before; dc_joint (primary) also uses `other_leagues` ({"E1": ..., "E2": ...},
    E1 already merged with its API rows). Raises StaleDataError if the E0 training data is
    incomplete, unless `allow_stale`, which records "stale_override": true instead.
    """
    matches = merge_results(matches, api.rows if api is not None else None)
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware")
    fx = select_upcoming(fixtures, now, committed_fixture_keys(pending_dir, revealed_dir))
    if fx.empty:
        raise NoFixturesError("no upcoming uncommitted E0 fixtures in fixtures.csv")
    check_team_names(fx, set(matches.home) | set(matches.away))
    check, stale, msg = staleness(training_window(matches, fx), fx, api, stale_days)
    if stale and not allow_stale:
        raise StaleDataError(f"{msg}. Refresh the data, or pass --allow-stale if this is "
                             "expected.")
    p_v1, train_v1 = fit_and_predict(matches, fx, xi)
    others = other_leagues or {}
    p_joint, trains = fit_joint_and_predict({config.LEAGUE_DIV: matches, **others}, fx)
    e2 = trains.get("E2")
    data_info = {"E1_completeness": e1_completeness(others.get("E1"), e1_api, fx),
                 "E2_last_match_date": f"{e2.date.max():%Y-%m-%d}"
                 if e2 is not None and len(e2) else None}
    return build_payload(fx, {"dc_joint": to_micro(p_joint), "dc_v1": to_micro(p_v1)},
                         {"dc_joint": trains, "dc_v1": train_v1}, now, git or git_info(),
                         stale_override=stale, staleness_check=check, data_info=data_info)


def write_round(payload: dict, pending_dir: Path = config.PENDING_DIR,
                revealed_dir: Path = config.REVEALED_DIR) -> Path:
    rid = payload["round_id"]
    out = pending_dir / rid
    if (out / COMMIT_FILE).exists() or (revealed_dir / rid).exists():
        raise FileExistsError(f"round {rid} is already committed; refusing to overwrite it")
    out.mkdir(parents=True, exist_ok=True)
    path = out / PRED_FILE
    path.write_bytes(canonical_json(payload))
    return path


def load_other_leagues(e1_api: ApiResults | None) -> dict:
    """E1 (CSV merged with the Championship API rows) and E2 (CSV only)."""
    return {"E1": merge_results(load_matches(league="E1"),
                                e1_api.rows if e1_api is not None else None),
            "E2": load_matches(league="E2")}


def run_predict(now: pd.Timestamp | None = None, refresh: bool = True,
                allow_stale: bool = False) -> Path:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if refresh:
        for lg in config.JOINT_LEAGUES:
            download([config.CURRENT_SEASON], overwrite=True, league=lg)
        download_fixtures()
    api, status = get_api_results(refresh)
    print(status)
    e1_api, status = get_api_results(refresh, competition="ELC")
    print(status)
    payload = predict_round(load_matches(), load_fixtures(), now, allow_stale=allow_stale,
                            api=api, other_leagues=load_other_leagues(e1_api), e1_api=e1_api)
    path = write_round(payload)

    v1 = payload["models"]["dc_v1"]["training"]
    joint = payload["models"]["dc_joint"]["training"]
    print(f"staleness check (E0): {payload['staleness_check']}"
          + ("  (calendar fallback: results API not live)"
             if payload["staleness_check"] == "calendar" else "")
          + f"; {v1['n_api_rows']} E0 training rows from the results API")
    print("dc_joint training: " + ", ".join(
        f"{lg} {t['n_matches']} (to {t['last_match_date']}, {t['n_api_rows']} API)"
        for lg, t in joint.items()))
    e1 = payload["data"]["E1_completeness"]
    if e1["checked"] and not e1["complete"]:
        print("  WARNING (non-blocking): Championship training data is incomplete against "
              "the ELC schedule: " + "; ".join(e1["problems"]))
    elif not e1["checked"]:
        print("  note: Championship completeness not checked (ELC API not live)")
    print(f"  E2 (League One, CSV only) last match used: {payload['data']['E2_last_match_date']}")
    primary = payload["primary_model"]
    print(f"\nRound {payload['round_id']}: {len(payload['fixtures'])} fixtures; primary "
          f"{primary} (xi={config.MODELS[primary]['xi']}), shadow dc_v1 "
          f"(xi={config.MODELS['dc_v1']['xi']})")
    for f in payload["fixtures"]:
        line = "  ".join(f"{name}{'*' if name == primary else ''} "
                         f"H {pr['p_home']:.3f} D {pr['p_draw']:.3f} A {pr['p_away']:.3f}"
                         for name, pr in f["probs"].items())
        print(f"  {f['kickoff_utc']}  {f['home']:>16} v {f['away']:<16} {line}")
    if payload["stale_override"]:
        print("  warning: training data is stale; recorded stale_override = true")
    if payload["git_dirty"]:
        print("  warning: src/ has uncommitted changes (git_dirty = true)")
    print(f"wrote {path.relative_to(config.ROOT) if path.is_relative_to(config.ROOT) else path}")
    return path
