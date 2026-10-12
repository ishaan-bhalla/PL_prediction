"""Reveal committed rounds and score them: the live, verifiable track record.

reveal  moves predictions/pending/<R>/ (predictions.json, salt.hex, commit.json) to
        predictions/revealed/<R>/ once the round is over (last kickoff + 3h).
score   scores revealed rounds only, from the committed files: the model's committed
        probabilities, the market at the committed open odds, and the market at the
        closing odds, all against the results. Nothing is refitted.

Results come from the season CSV merged with football-data.org rows (premodel.results).
API rows have no odds, so until the CSV catches up a match can be scored for the model
and the open market but not the close: its round shows clv_pending > 0 and blank close
metrics / CLV, which a later re-score fills in. Each re-score replaces the round's row.
"""
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from premodel import config
from premodel.commitment import (
    COMMIT_FILE,
    PRED_FILE,
    SALT_FILE,
    compute_commitment,
    format_time,
    kickoffs,
    read_salt,
)
from premodel.data import devig, season_of
from premodel.metrics import score_all

ROUND_FILES = (PRED_FILE, SALT_FILE, COMMIT_FILE)
METRICS = ("rps", "brier", "log_loss")
SOURCES = ("model", "open", "close")
COLUMNS = (["round_id", "first_kickoff_utc", "n_fixtures", "n_scored", "n_unscored"]
           + [f"{m}_{s}" for m in METRICS for s in SOURCES]
           + ["n_bets", "mean_clv", "clv_pending", "signature", "scored_at"])
P_COLS, OPEN_COLS, CLOSE_COLS = ["p_h", "p_d", "p_a"], ["open_h", "open_d", "open_a"], \
    ["close_h", "close_d", "close_a"]


class RevealRefused(RuntimeError):
    pass


def check_local_commitment(round_dir: Path) -> dict:
    """commit.json, after checking salt.hex + predictions.json hash to its commitment."""
    rec = json.loads((round_dir / COMMIT_FILE).read_bytes())
    got = compute_commitment(read_salt(round_dir / SALT_FILE),
                             (round_dir / PRED_FILE).read_bytes())
    if got != rec["commitment"]:
        raise ValueError(f"{round_dir.name}: files hash to {got}, commit.json says "
                         f"{rec['commitment']}")
    return rec


# --- reveal -----------------------------------------------------------------------------

def reveal_round(round_dir: Path, now: datetime, revealed_dir: Path = config.REVEALED_DIR,
                 force: bool = False) -> Path:
    rid = round_dir.name
    if not (round_dir / COMMIT_FILE).exists():
        raise RevealRefused(f"round {rid} has no {COMMIT_FILE}; commit it before revealing")
    try:
        check_local_commitment(round_dir)
    except (ValueError, KeyError, OSError) as e:
        raise RevealRefused(f"local files do not match the commitment: {e}") from None
    open_at = kickoffs(json.loads((round_dir / PRED_FILE).read_bytes()))[-1] \
        + timedelta(hours=config.REVEAL_DELAY_HOURS)
    if now < open_at and not force:
        raise RevealRefused(f"round {rid} can be revealed from {format_time(open_at)} "
                            f"(last kickoff + {config.REVEAL_DELAY_HOURS}h); --force overrides")
    dest = revealed_dir / rid
    if dest.exists():
        raise RevealRefused(f"{dest} already exists")
    tmp = revealed_dir / f".{rid}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in ROUND_FILES:
        shutil.copy2(round_dir / name, tmp / name)
    tmp.rename(dest)
    for name in ROUND_FILES:
        (round_dir / name).unlink()
    leftovers = [p.name for p in round_dir.iterdir()]
    if not leftovers:
        round_dir.rmdir()
    return dest


# --- score ------------------------------------------------------------------------------

def round_frame(round_dir: Path, results: pd.DataFrame) -> pd.DataFrame:
    """One row per committed fixture: committed probabilities and open odds, plus the
    result and closing odds where known (matched on season, home, away)."""
    pred = json.loads((round_dir / PRED_FILE).read_bytes())
    rows = []
    for f in pred["fixtures"]:
        kickoff = pd.Timestamp(f["kickoff_utc"])
        odds = f["odds_open"] or {}
        rows.append({
            "round_id": round_dir.name, "home": f["home"], "away": f["away"],
            "kickoff_utc": kickoff,
            "season": season_of(kickoff.tz_convert(config.KICKOFF_TZ)),
            "p_h": f["p_home"], "p_d": f["p_draw"], "p_a": f["p_away"],
            "open_h": odds.get("home", np.nan), "open_d": odds.get("draw", np.nan),
            "open_a": odds.get("away", np.nan),
        })
    fx = pd.DataFrame(rows)
    res = results.drop_duplicates(["season", "home", "away"], keep="first")
    res = res.reindex(columns=["season", "home", "away", "result", *CLOSE_COLS])
    return fx.merge(res, on=["season", "home", "away"], how="left", validate="one_to_one")


def bets(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """The backtest rule: at most one bet per match, on the best edge at the committed open
    odds if it exceeds `threshold`; CLV = open odds x de-vigged close - 1 (NaN if no close)."""
    df = df[df[OPEN_COLS].notna().all(axis=1)]
    if df.empty:
        return pd.DataFrame(columns=["pick", "odds", "clv"])
    p, odds = df[P_COLS].to_numpy(), df[OPEN_COLS].to_numpy()
    edge = p * odds - 1
    pick = edge.argmax(axis=1)
    r = np.arange(len(df))
    take = edge[r, pick] > threshold
    close = df[CLOSE_COLS].to_numpy()
    has_close = ~np.isnan(close).any(axis=1)
    fair = np.full_like(close, np.nan)
    fair[has_close] = devig(close[has_close])
    clv = odds[r, pick] * fair[r, pick] - 1
    return pd.DataFrame({"pick": pick, "odds": odds[r, pick], "clv": clv},
                        index=df.index)[take]


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    return score_all(p, y) if len(y) else dict.fromkeys(METRICS, np.nan)


def score_frame(df: pd.DataFrame) -> dict:
    """Metrics for the scored rows of a round_frame (or several pooled)."""
    s = df[df.result.notna()]
    y = s.result.to_numpy(dtype=int)
    has_open = s[OPEN_COLS].notna().all(axis=1).to_numpy()
    has_close = s[CLOSE_COLS].notna().all(axis=1).to_numpy()
    out = {f"{m}_model": v for m, v in _metrics(s[P_COLS].to_numpy(), y).items()}
    for src, cols, ok in (("open", OPEN_COLS, has_open), ("close", CLOSE_COLS, has_close)):
        vals = (_metrics(devig(s[cols].to_numpy()), y) if ok.all() and len(s)
                else dict.fromkeys(METRICS, np.nan))
        out |= {f"{m}_{src}": v for m, v in vals.items()}
    b = bets(s)
    out["n_bets"] = len(b)
    out["clv_pending"] = int((~has_close).sum())
    out["mean_clv"] = float(b.clv.mean()) if len(b) and b.clv.notna().all() else np.nan
    return out


@dataclass
class RoundScore:
    row: dict
    frame: pd.DataFrame
    unscored: list[str]


def score_round(round_dir: Path, results: pd.DataFrame, now: datetime) -> RoundScore:
    rec = check_local_commitment(round_dir)
    df = round_frame(round_dir, results)
    scored = df.result.notna()
    row = {
        "round_id": round_dir.name,
        "first_kickoff_utc": format_time(df.kickoff_utc.min().to_pydatetime()),
        "n_fixtures": len(df), "n_scored": int(scored.sum()),
        "n_unscored": int((~scored).sum()),
        **score_frame(df),
        "signature": rec["signature"],
        "scored_at": format_time(now),
    }
    unscored = [f"{r.kickoff_utc:%Y-%m-%d} {r.home} v {r.away}"
                for r in df[~scored].itertuples(index=False)]
    return RoundScore({k: row[k] for k in COLUMNS}, df, unscored)


def revealed_rounds(revealed_dir: Path = config.REVEALED_DIR,
                    round_id: str | None = None) -> list[Path]:
    if round_id:
        d = revealed_dir / round_id
        if not (d / COMMIT_FILE).exists():
            raise FileNotFoundError(f"no revealed round {round_id} in {revealed_dir} "
                                    "(only revealed rounds are scored)")
        return [d]
    return sorted(d for d in revealed_dir.glob("*") if (d / COMMIT_FILE).exists())


def update_track_record(rows: list[dict], path: Path) -> pd.DataFrame:
    """Replace (or add) each round's row; keep the others."""
    new = pd.DataFrame(rows, columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path, dtype={"signature": str})
        new = pd.concat([old[~old.round_id.isin(new.round_id)], new], ignore_index=True)
    new = new.sort_values("round_id").reset_index(drop=True)[COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return new


def summary(frames: list[pd.DataFrame]) -> str:
    """Pooled over every scored match: model vs open on matches with open odds, and model
    vs close on matches with closing odds, so each comparison uses the same matches."""
    if not frames:
        return "no revealed rounds scored yet"
    all_ = pd.concat(frames, ignore_index=True)
    s = all_[all_.result.notna()]
    lines = [f"Track record: {all_.round_id.nunique()} round(s), {len(s)} scored matches, "
             f"{len(all_) - len(s)} unscored"]
    head = f"  {'':<14}{'n':>5}{'RPS':>10}{'Brier':>10}{'log-loss':>10}"
    for src, cols in (("open", OPEN_COLS), ("close", CLOSE_COLS)):
        sub = s[s[cols].notna().all(axis=1)]
        lines += [f" matches with {src} odds", head]
        if sub.empty:
            lines.append("  (none yet)")
            continue
        y = sub.result.to_numpy(dtype=int)
        for name, p in (("model", sub[P_COLS].to_numpy()),
                        (f"market {src}", devig(sub[cols].to_numpy()))):
            m = score_all(p, y)
            lines.append(f"  {name:<14}{len(sub):>5}{m['rps']:>10.4f}{m['brier']:>10.4f}"
                         f"{m['log_loss']:>10.4f}")
    b = bets(s)
    done = b[b.clv.notna()]
    lines.append(f"  bets (edge > 0 at open odds): {len(b)}; mean CLV "
                 + (f"{done.clv.mean():+.2%} over {len(done)}" if len(done) else "n/a")
                 + (f"; {len(b) - len(done)} awaiting closing odds" if len(done) < len(b)
                    else ""))
    return "\n".join(lines)


def run_score(results: pd.DataFrame, now: datetime, round_id: str | None = None,
              revealed_dir: Path = config.REVEALED_DIR,
              path: Path = config.TRACK_RECORD_PATH) -> pd.DataFrame | None:
    """Score the chosen revealed round (default: all), update the CSV, print the summary
    pooled over every revealed round."""
    chosen = {d.name for d in revealed_rounds(revealed_dir, round_id)}
    if not chosen:
        print(f"no revealed rounds in {revealed_dir}; nothing to score")
        return None
    scores = [score_round(d, results, now) for d in revealed_rounds(revealed_dir)]
    for sc in scores:
        r = sc.row
        if r["round_id"] not in chosen:
            continue
        print(f"{r['round_id']}: scored {r['n_scored']}/{r['n_fixtures']}  "
              f"rps model {r['rps_model']:.4f}"
              + (f"  close {r['rps_close']:.4f}" if not np.isnan(r["rps_close"]) else "")
              + (f"  clv_pending {r['clv_pending']}" if r["clv_pending"] else ""))
        for u in sc.unscored:
            print(f"  skipped (no result yet): {u}")
    table = update_track_record([sc.row for sc in scores if sc.row["round_id"] in chosen],
                                path)
    print(summary([sc.frame for sc in scores]))
    return table
