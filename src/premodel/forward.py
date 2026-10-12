"""Forward predictions for upcoming fixtures, written as a round ready to commit.

A round is every upcoming E0 fixture in football-data.co.uk's fixtures.csv that
has not kicked off yet and is not already part of a committed round. The model
is fitted on matches played strictly before the date of the round's first
kickoff (same xi and lookback window as the backtest), so no result from the
round itself, or anything after it, can leak into the predictions.
"""
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from premodel import __version__, config
from premodel.commitment import COMMIT_FILE, PRED_FILE, canonical_json
from premodel.data import download, load_matches, open_odds
from premodel.models import dixon_coles as dc

SCHEMA = "premodel/predictions/v1"
MICRO = 1_000_000  # probabilities are stored in millionths (6 decimals)


class NoFixturesError(RuntimeError):
    """No upcoming, uncommitted fixtures to predict."""


class StaleDataError(RuntimeError):
    """The newest training match is too long before the round's first kickoff."""


def iso_utc(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def season_of(kickoff: pd.Timestamp) -> str:
    """Season label in the same format as data.load_season, e.g. "2026/27"."""
    y = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
    return f"{y}/{(y + 1) % 100:02d}"


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


def build_payload(fixtures: pd.DataFrame, probs_micro: np.ndarray, train: pd.DataFrame,
                  xi: float, now: pd.Timestamp, git: tuple[str, bool],
                  stale_override: bool = False) -> dict:
    rows = []
    for f, pm in zip(fixtures.itertuples(index=False), probs_micro, strict=True):
        odds = [f.open_h, f.open_d, f.open_a]
        has_odds = not any(np.isnan(odds))
        rows.append({
            "home": f.home,
            "away": f.away,
            "kickoff_utc": iso_utc(f.kickoff_utc),
            "p_home": pm[0] / MICRO, "p_draw": pm[1] / MICRO, "p_away": pm[2] / MICRO,
            "odds_open": dict(zip(["home", "draw", "away"], map(float, odds), strict=True))
            if has_odds else None,
            "odds_source": f.open_src if has_odds else None,
        })
    return {
        "schema": SCHEMA,
        "round_id": round_id_for(fixtures),
        "model": config.MODEL_NAME,
        "model_version": __version__,
        "git_commit": git[0],
        "git_dirty": git[1],
        "xi": float(xi),
        "lookback_days": config.LOOKBACK_DAYS,
        "generated_at": iso_utc(now),
        "stale_override": stale_override,
        "training": {
            "n_matches": int(len(train)),
            "first_match_date": f"{train.date.min():%Y-%m-%d}",
            "last_match_date": f"{train.date.max():%Y-%m-%d}",
        },
        "fixtures": rows,
    }


def predict_round(matches: pd.DataFrame, fixtures: pd.DataFrame, now: pd.Timestamp,
                  pending_dir: Path = config.PENDING_DIR,
                  revealed_dir: Path = config.REVEALED_DIR,
                  xi: float = config.FORWARD_XI, git: tuple[str, bool] | None = None,
                  allow_stale: bool = False, stale_days: int = config.STALE_DAYS) -> dict:
    """Predictions payload for the next round (pure apart from reading committed rounds).

    Raises StaleDataError if the training data ends more than `stale_days` before the
    first kickoff, unless `allow_stale`, which records "stale_override": true instead.
    """
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware")
    fx = select_upcoming(fixtures, now, committed_fixture_keys(pending_dir, revealed_dir))
    if fx.empty:
        raise NoFixturesError("no upcoming uncommitted E0 fixtures in fixtures.csv")
    check_team_names(fx, set(matches.home) | set(matches.away))
    stale, msg = is_stale(training_window(matches, fx), fx, stale_days)
    if stale and not allow_stale:
        raise StaleDataError(f"{msg}. Refresh the data, or pass --allow-stale if this is "
                             "expected (e.g. an international break).")
    p, train = fit_and_predict(matches, fx, xi)
    return build_payload(fx, to_micro(p), train, xi, now, git or git_info(),
                         stale_override=stale)


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


def run_predict(now: pd.Timestamp | None = None, refresh: bool = True,
                allow_stale: bool = False) -> Path:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if refresh:
        download([config.CURRENT_SEASON], overwrite=True)
        download_fixtures()
    payload = predict_round(load_matches(), load_fixtures(), now, allow_stale=allow_stale)
    path = write_round(payload)

    t = payload["training"]
    print(f"\nRound {payload['round_id']}: {len(payload['fixtures'])} fixtures  "
          f"(trained on {t['n_matches']} matches, {t['first_match_date']} .. "
          f"{t['last_match_date']}, xi={payload['xi']})")
    for f in payload["fixtures"]:
        print(f"  {f['kickoff_utc']}  {f['home']:>16} v {f['away']:<16} "
              f"H {f['p_home']:.3f}  D {f['p_draw']:.3f}  A {f['p_away']:.3f}")
    if payload["stale_override"]:
        print("  warning: training data is stale; recorded stale_override = true")
    if payload["git_dirty"]:
        print("  warning: src/ has uncommitted changes (git_dirty = true)")
    print(f"wrote {path.relative_to(config.ROOT) if path.is_relative_to(config.ROOT) else path}")
    return path
