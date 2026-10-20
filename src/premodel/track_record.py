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
    fixture_probs,
    format_time,
    kickoffs,
    primary_model,
    read_salt,
)
from premodel.data import devig, season_of
from premodel.metrics import score_all
from premodel.stats import date_block_bootstrap

ROUND_FILES = (PRED_FILE, SALT_FILE, COMMIT_FILE)
METRICS = ("rps", "brier", "log_loss")
MODELS = ("dc_v1", "dc_joint")            # columns exist for both; v1 rounds fill dc_v1 only
SOURCES = (*MODELS, "open", "close")
COLUMNS = (["round_id", "schema", "primary_model", "first_kickoff_utc", "n_fixtures",
            "n_scored", "n_unscored"]
           + [f"{m}_{s}" for m in METRICS for s in SOURCES]
           + [f"{k}_{mdl}" for mdl in MODELS for k in ("n_bets", "mean_clv")]
           + ["clv_pending", "signature", "scored_at"])
# track_record.csv written before schema v2 (one model, "model" columns) -> current names
LEGACY_COLUMNS = {"rps_model": "rps_dc_v1", "brier_model": "brier_dc_v1",
                  "log_loss_model": "log_loss_dc_v1", "n_bets": "n_bets_dc_v1",
                  "mean_clv": "mean_clv_dc_v1"}
OPEN_COLS, CLOSE_COLS = ["open_h", "open_d", "open_a"], ["close_h", "close_d", "close_a"]


def p_cols(model: str) -> list[str]:
    return [f"p_h_{model}", f"p_d_{model}", f"p_a_{model}"]


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
    """One row per committed fixture: every committed model's probabilities (schema v1:
    dc_v1 only; v2: dc_joint and dc_v1) and the open odds, plus the result and closing
    odds where known (matched on season, home, away)."""
    pred = json.loads((round_dir / PRED_FILE).read_bytes())
    rows = []
    for f, probs in zip(pred["fixtures"], fixture_probs(pred), strict=True):
        kickoff = pd.Timestamp(f["kickoff_utc"])
        odds = f["odds_open"] or {}
        row = {"round_id": round_dir.name, "schema": pred["schema"],
               "primary_model": primary_model(pred),
               "home": f["home"], "away": f["away"], "kickoff_utc": kickoff,
               "season": season_of(kickoff.tz_convert(config.KICKOFF_TZ)),
               "open_h": odds.get("home", np.nan), "open_d": odds.get("draw", np.nan),
               "open_a": odds.get("away", np.nan)}
        for mdl in MODELS:
            row |= dict(zip(p_cols(mdl), probs.get(mdl, (np.nan,) * 3), strict=True))
        rows.append(row)
    fx = pd.DataFrame(rows)
    res = results.drop_duplicates(["season", "home", "away"], keep="first")
    res = res.reindex(columns=["season", "home", "away", "date", "result", *CLOSE_COLS])
    return fx.merge(res, on=["season", "home", "away"], how="left", validate="one_to_one")


def bets(df: pd.DataFrame, model: str = "dc_v1", threshold: float = 0.0) -> pd.DataFrame:
    """The backtest rule for one model: at most one bet per match, on the best edge at the
    committed open odds if it exceeds `threshold`; CLV = open odds x de-vigged close - 1
    (NaN if no close)."""
    df = df[df[OPEN_COLS].notna().all(axis=1) & df[p_cols(model)].notna().all(axis=1)]
    if df.empty:
        return pd.DataFrame(columns=["pick", "odds", "clv"])
    p, odds = df[p_cols(model)].to_numpy(), df[OPEN_COLS].to_numpy()
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
    """Metrics for the scored rows of a round_frame. A model without probabilities in this
    round (dc_joint in a schema v1 round) gets blank columns."""
    s = df[df.result.notna()]
    y = s.result.to_numpy(dtype=int)
    has_open = s[OPEN_COLS].notna().all(axis=1).to_numpy()
    has_close = s[CLOSE_COLS].notna().all(axis=1).to_numpy()
    out = {}
    for mdl in MODELS:
        present = len(s) and s[p_cols(mdl)].notna().all(axis=1).all()
        vals = _metrics(s[p_cols(mdl)].to_numpy(), y) if present else \
            dict.fromkeys(METRICS, np.nan)
        out |= {f"{m}_{mdl}": v for m, v in vals.items()}
        b = bets(s, mdl) if present else pd.DataFrame(columns=["clv"])
        out[f"n_bets_{mdl}"] = len(b) if present else np.nan
        out[f"mean_clv_{mdl}"] = (float(b.clv.mean()) if len(b) and b.clv.notna().all()
                                  else np.nan)
    for src, cols, ok in (("open", OPEN_COLS, has_open), ("close", CLOSE_COLS, has_close)):
        vals = (_metrics(devig(s[cols].to_numpy()), y) if ok.all() and len(s)
                else dict.fromkeys(METRICS, np.nan))
        out |= {f"{m}_{src}": v for m, v in vals.items()}
    out["clv_pending"] = int((~has_close).sum())
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
        "schema": df.schema.iloc[0],
        "primary_model": df.primary_model.iloc[0],
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


def read_track_record(path: Path) -> pd.DataFrame:
    """The CSV with current column names (files from before schema v2 are mapped)."""
    old = pd.read_csv(path, dtype={"signature": str})
    old = old.rename(columns=LEGACY_COLUMNS)
    if "schema" not in old:
        old["schema"] = "premodel/predictions/v1"
    if "primary_model" not in old:
        old["primary_model"] = "dc_v1"
    return old.reindex(columns=COLUMNS)


def update_track_record(rows: list[dict], path: Path) -> pd.DataFrame:
    """Replace (or add) each round's row; keep the others."""
    new = pd.DataFrame(rows, columns=COLUMNS)
    if path.exists():
        old = read_track_record(path)
        new = pd.concat([old[~old.round_id.isin(new.round_id)], new], ignore_index=True)
    new = new.sort_values("round_id").reset_index(drop=True)[COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return new


# --- head-to-head (schema v2) -----------------------------------------------------------

def new_team_flags(df: pd.DataFrame, results: pd.DataFrame,
                   k: int = 10) -> np.ndarray:
    """Evaluation label (not decisive): the match involves a team in its first k E0 games
    of the season after entering E0 (not in E0 the previous season)."""
    flags = np.zeros(len(df), dtype=bool)
    seasons = sorted(results.season.unique())
    for i, r in enumerate(df.itertuples(index=False)):
        prev = [x for x in seasons if x < r.season]
        if not prev:
            continue
        p = results[results.season == prev[-1]]
        before = set(p.home) | set(p.away)
        cur = results[(results.season == r.season) & (results.date <= r.date)]
        for t in (r.home, r.away):
            if t not in before and ((cur.home == t) | (cur.away == t)).sum() <= k:
                flags[i] = True
    return flags


def head_to_head(frames: list[pd.DataFrame], results: pd.DataFrame) -> dict | None:
    """Pooled over every scored schema v2 match: RPS(dc_joint) - RPS(dc_v1), with a
    date-block bootstrap 95% CI; plus the new-team subgroup (non-decisive)."""
    if not frames:
        return None
    a = pd.concat(frames, ignore_index=True)
    v2 = a[a.schema.str.endswith("/v2") & a.result.notna()
           & a[p_cols("dc_joint")].notna().all(axis=1)]
    if v2.empty:
        return None
    y = v2.result.to_numpy(dtype=int)

    def rps_each(model):
        p = v2[p_cols(model)].to_numpy()
        onehot = np.eye(3)[y]
        cp, co = np.cumsum(p, axis=1)[:, :2], np.cumsum(onehot, axis=1)[:, :2]
        return np.sum((cp - co) ** 2, axis=1) / 2

    diff = rps_each("dc_joint") - rps_each("dc_v1")
    dates = v2.date.to_numpy()
    new = new_team_flags(v2, results)
    return {"n": int(len(v2)), "rps_dc_joint": float(rps_each("dc_joint").mean()),
            "rps_dc_v1": float(rps_each("dc_v1").mean()),
            "diff": date_block_bootstrap(diff, dates),
            "new_teams": date_block_bootstrap(diff[new], dates[new]),
            "last_kickoff": v2.kickoff_utc.max(),
            "unscored_before_season_end": int(
                (a.schema.str.endswith("/v2") & a.result.isna()
                 & (a.kickoff_utc < pd.Timestamp(config.LIVE_EVALUATION["season_end"],
                                                 tz="UTC") + pd.Timedelta(days=1))).sum())}


def ci_shown(block: dict, min_blocks: int | None = None) -> bool:
    """Whether a date-block bootstrap has enough distinct match dates to report its CI."""
    need = config.MIN_BOOTSTRAP_BLOCKS if min_blocks is None else min_blocks
    return block["n_dates"] >= need


def h2h_line(label: str, block: dict) -> str:
    """One head-to-head summary line: n matches and k dates always; the 95% CI only with
    at least config.MIN_BOOTSTRAP_BLOCKS dates (a handful of blocks gives a degenerate
    interval that looks meaningful but is not)."""
    n, k = block["n_matches"], block["n_dates"]
    head = f"  {label:<9} n={n} matches, k={k} dates"
    if n == 0:
        return f"{head}  -"
    if not ci_shown(block):
        return (f"{head}  {block['mean']:+.4f}  CI not shown: only {k} match dates "
                f"(need >= {config.MIN_BOOTSTRAP_BLOCKS})")
    lo, hi = block["ci95"]
    return f"{head}  {block['mean']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"


def evaluation_status(h2h: dict | None, rule: dict = config.LIVE_EVALUATION) -> str:
    """The pre-registered live rule: not yet due / keep dc_joint / revert to dc_v1.

    Once due, the CI must rest on at least config.MIN_BOOTSTRAP_BLOCKS match dates; if it
    does not, the rule reports "not evaluable" rather than keep or revert. (At >= 150
    matches and a season's end this cannot normally happen; the check makes sure of it.)"""
    n = h2h["n"] if h2h else 0
    end = pd.Timestamp(rule["season_end"], tz="UTC")
    season_done = bool(h2h) and h2h["last_kickoff"] >= end \
        and h2h["unscored_before_season_end"] == 0
    if n < rule["min_v2_matches"] or not season_done:
        return (f"not yet due ({n} of {rule['min_v2_matches']} schema-v2 matches scored; "
                f"season end {rule['season_end']} {'reached' if season_done else 'not reached'})")
    if not ci_shown(h2h["diff"]):
        return (f"DUE but not evaluable: only {h2h['diff']['n_dates']} match dates "
                f"(need >= {config.MIN_BOOTSTRAP_BLOCKS}); neither keep nor revert")
    lo, hi = h2h["diff"]["ci95"]
    if lo > 0:
        return f"DUE: revert the primary model to dc_v1 (95% CI [{lo:+.4f}, {hi:+.4f}] > 0)"
    return f"DUE: keep dc_joint (95% CI [{lo:+.4f}, {hi:+.4f}] not entirely above 0)"


def summary(frames: list[pd.DataFrame], results: pd.DataFrame | None = None) -> str:
    """Pooled over every scored match: each model vs the open market on matches with open
    odds, and vs the close on matches with closing odds (same matches per comparison);
    then the dc_joint vs dc_v1 head-to-head over schema v2 matches."""
    if not frames:
        return "no revealed rounds scored yet"
    all_ = pd.concat(frames, ignore_index=True)
    s = all_[all_.result.notna()]
    lines = [f"Track record: {all_.round_id.nunique()} round(s), {len(s)} scored matches, "
             f"{len(all_) - len(s)} unscored"]
    head = f"  {'':<14}{'n':>5}{'RPS':>10}{'Brier':>10}{'log-loss':>10}"
    for src, cols in (("open", OPEN_COLS), ("close", CLOSE_COLS)):
        lines += [f" matches with {src} odds", head]
        for mdl in MODELS:
            sub = s[s[cols].notna().all(axis=1) & s[p_cols(mdl)].notna().all(axis=1)]
            if sub.empty:
                continue
            y = sub.result.to_numpy(dtype=int)
            for name, p in ((mdl, sub[p_cols(mdl)].to_numpy()),
                            (f"market {src}", devig(sub[cols].to_numpy()))):
                m = score_all(p, y)
                lines.append(f"  {name:<14}{len(sub):>5}{m['rps']:>10.4f}"
                             f"{m['brier']:>10.4f}{m['log_loss']:>10.4f}")
    for mdl in MODELS:
        b = bets(s, mdl)
        done = b[b.clv.notna()]
        if len(b):
            lines.append(f"  {mdl} bets (edge > 0 at open odds): {len(b)}; mean CLV "
                         + (f"{done.clv.mean():+.2%} over {len(done)}" if len(done) else "n/a")
                         + (f"; {len(b) - len(done)} awaiting closing odds"
                            if len(done) < len(b) else ""))
    h2h = head_to_head(frames, results if results is not None else s)
    lines.append(" head-to-head on schema v2 matches: RPS(dc_joint) - RPS(dc_v1)")
    if h2h is None:
        lines.append("  (no scored schema v2 matches yet)")
    else:
        lines.append(h2h_line("all", h2h["diff"]))
        lines.append(h2h_line("new teams", h2h["new_teams"])
                     + "  (first 10 games after entering E0; non-decisive)")
    lines.append(f" live evaluation rule: {evaluation_status(h2h)}")
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
        rps = "  ".join(f"{m} {r[f'rps_{m}']:.4f}" for m in MODELS
                        if not np.isnan(r[f"rps_{m}"]))
        print(f"{r['round_id']} ({r['schema'].rsplit('/', 1)[-1]}): scored "
              f"{r['n_scored']}/{r['n_fixtures']}  rps {rps}"
              + (f"  close {r['rps_close']:.4f}" if not np.isnan(r["rps_close"]) else "")
              + (f"  clv_pending {r['clv_pending']}" if r["clv_pending"] else ""))
        for u in sc.unscored:
            print(f"  skipped (no result yet): {u}")
    table = update_track_record([sc.row for sc in scores if sc.row["round_id"] in chosen],
                                path)
    print(summary([sc.frame for sc in scores], results))
    return table
