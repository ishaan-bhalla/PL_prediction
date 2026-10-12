"""Experiment: shrink low-sample team ratings toward a prior ("dc_shrink" vs "dc_v1").

Promoted teams are shrunk toward a promoted-team prior mean, all other teams toward 0,
with penalty lam * sum((att - m_att)^2 + (dfn - m_dfn)^2) (see models.dixon_coles).

No leakage, by construction:
- a team is "promoted" in season S if it has played in S before the fit date and did
  not play in S-1 (so even team names come only from matches already played);
- relegated teams of S-1 are the bottom 3 of S-1's points table, computed from S-1's
  results (this reproduces the actual relegated teams in every season 2014/15-2025/26);
- the promoted-team prior for S is the running mean, over seasons s <= S, of the
  relegated teams' (att, dfn) in a dc_v1 fit on matches strictly before s's first match.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from premodel import backtest, config
from premodel.data import devig, load_matches, season_of
from premodel.metrics import calibration_table, score_all
from premodel.models import dixon_coles as dc

PROB_COLS = ["p_h", "p_d", "p_a"]
N_RELEGATED = 3
LAM_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
# The candidates chosen between on validation RPS, and one diagnostic that never is.
VARIANTS = {"uniform": {"only_prior": False, "zero_prior": False},
            "promoted_only": {"only_prior": True, "zero_prior": False}}
DIAGNOSTIC = ("uniform_zero_prior", {"only_prior": False, "zero_prior": True})
# The published baseline (reports/summary.json): test seasons up to this date, 3,090 matches.
PUBLISHED_TEST_END = pd.Timestamp("2026-09-20")
PUBLISHED_N = 3090
FIRST_GAMES = 10
OUT_DIR = config.REPORTS_DIR / "experiments" / "shrinkage"
LIVE_CASE = {"round_dir": config.PENDING_DIR / "E0-2026-10-12", "home": "Coventry",
             "away": "Newcastle", "date": pd.Timestamp("2026-10-12")}


def previous_season(m: pd.DataFrame, season: str) -> str | None:
    earlier = sorted(s for s in m.season.unique() if s < season)
    return earlier[-1] if earlier else None


def season_teams(m: pd.DataFrame, season: str) -> set[str]:
    g = m[m.season == season]
    return set(g.home) | set(g.away)


def promoted_teams(m: pd.DataFrame, season: str, before: pd.Timestamp) -> set[str]:
    """Teams that played in `season` before `before` but not in the previous season."""
    prev = previous_season(m, season)
    if prev is None:
        return set()
    g = m[(m.season == season) & (m.date < before)]
    return (set(g.home) | set(g.away)) - season_teams(m[m.date < before], prev)


def relegated_teams(m: pd.DataFrame, season: str, k: int = N_RELEGATED) -> list[str]:
    """Bottom k of the season's table by points, goal difference, goals scored."""
    g = m[m.season == season]
    home = pd.DataFrame({"team": g.home, "gf": g.hg, "ga": g.ag,
                         "pts": np.select([g.hg > g.ag, g.hg == g.ag], [3, 1], 0)})
    away = pd.DataFrame({"team": g.away, "gf": g.ag, "ga": g.hg,
                         "pts": np.select([g.ag > g.hg, g.ag == g.hg], [3, 1], 0)})
    t = pd.concat([home, away]).groupby("team").sum()
    t["gd"] = t.gf - t.ga
    return t.sort_values(["pts", "gd", "gf"]).index[:k].tolist()


def fit_before(m: pd.DataFrame, date: pd.Timestamp, xi: float, init=None,
               shrink: dc.Shrinkage | None = None,
               lookback_days: int = config.LOOKBACK_DAYS) -> dc.DCParams:
    """Fit on matches strictly before `date` within the lookback window (as the backtest)."""
    train = m[(m.date < date) & (m.date >= date - pd.Timedelta(days=lookback_days))]
    return dc.fit(train.home.to_numpy(), train.away.to_numpy(), train.hg.to_numpy(),
                  train.ag.to_numpy(), (date - train.date).dt.days.to_numpy(), xi,
                  init=init, shrink=shrink)


def relegated_rating(m: pd.DataFrame, season: str, xi: float) -> tuple[float, float] | None:
    """Mean (att, dfn) of the previous season's relegated teams, from a dc_v1 fit on
    matches strictly before `season`'s first match. None if there is no previous season."""
    prev = previous_season(m, season)
    if prev is None:
        return None
    start = m[m.season == season].date.min()
    past = m[m.date < start]
    params = fit_before(past, start, xi)
    idx = params.index()
    rel = [idx[t] for t in relegated_teams(past, prev)]
    return float(params.att[rel].mean()), float(params.dfn[rel].mean())


def promoted_priors(m: pd.DataFrame, seasons: list[str], xi: float) -> dict:
    """season -> (m_att, m_dfn): running mean of relegated_rating over seasons <= season.

    Every term uses only matches before that season's first match, hence before any
    fit date inside the season."""
    out, history = {}, []
    for s in sorted(m.season.unique()):
        r = relegated_rating(m, s, xi)
        if r is not None:
            history.append(r)
        if s in seasons and history:
            out[s] = tuple(float(v) for v in np.mean(history, axis=0))
    return out


def walk_forward(m: pd.DataFrame, xi: float, seasons: list[str], lam: float | None = None,
                 priors: dict | None = None, zero_prior: bool = False,
                 only_prior: bool = False) -> pd.DataFrame:
    """backtest.walk_forward, optionally with shrinkage (lam=None reproduces dc_v1).

    Promoted teams are shrunk toward priors[season] (or 0 if zero_prior); other teams
    toward 0 ("uniform"), or not at all if only_prior ("promoted-only"). Teams with no
    matches in the window still use dc.promoted_prior() for prediction.
    """
    target = m[m.season.isin(seasons)]
    preds, params = [], None
    for date, day in target.groupby("date", sort=True):
        shrink = None
        if lam is not None:
            season = day.season.iloc[0]
            mean = (0.0, 0.0) if zero_prior else priors[season]
            shrink = dc.Shrinkage(lam, {t: mean for t in promoted_teams(m, season, date)},
                                  only_prior=only_prior)
        params = fit_before(m, date, xi, init=params, shrink=shrink)
        p, xg = dc.predict_1x2(params, day.home.to_numpy(), day.away.to_numpy(),
                               dc.promoted_prior(params))
        out = day.copy()
        out[PROB_COLS] = p
        out[["xg_h", "xg_a"]] = xg
        preds.append(out)
    return pd.concat(preds)


# --- evaluation helpers -----------------------------------------------------------------

def per_match_rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.eye(3)[y]
    cp, co = np.cumsum(p, axis=1)[:, :2], np.cumsum(onehot, axis=1)[:, :2]
    return np.sum((cp - co) ** 2, axis=1) / 2


def block_bootstrap(diff: np.ndarray, dates: np.ndarray, rng: np.random.Generator,
                    n: int = config.BOOTSTRAP_SAMPLES) -> dict:
    """Mean of `diff` with a 95% CI, resampling whole match dates (matches on one date
    share a fit, so they are not independent)."""
    codes, uniq = pd.factorize(pd.Series(dates))
    sums = np.bincount(codes, weights=diff)
    counts = np.bincount(codes).astype(float)
    idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return {"mean": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                                                 float(np.percentile(boot, 97.5))],
            "n_matches": int(len(diff)), "n_dates": int(len(uniq))}


def promoted_early(df: pd.DataFrame, m: pd.DataFrame, k: int = FIRST_GAMES) -> np.ndarray:
    """Evaluation label (not a model input): the match involves a team in its first
    season after promotion, within that team's first k games of the season."""
    flag = np.zeros(len(df), dtype=bool)
    for season, g in df.groupby("season"):
        prev = previous_season(m, season)
        if prev is None:
            continue
        promoted = season_teams(m, season) - season_teams(m, prev)
        sm = m[m.season == season].sort_values(["date", "home"], kind="stable")
        game_no = {}
        for t in promoted:
            mask = (sm.home == t) | (sm.away == t)
            game_no.update({(d, h, a, t): i + 1 for i, (d, h, a) in
                            enumerate(zip(sm[mask].date, sm[mask].home, sm[mask].away,
                                          strict=True))})
        pos = df.index.get_indexer(g.index)
        for j, (d, h, a) in zip(pos, zip(g.date, g.home, g.away, strict=True), strict=True):
            flag[j] = any(game_no.get((d, h, a, t), k + 1) <= k for t in (h, a))
    return flag


def _scores(df: pd.DataFrame) -> dict:
    return score_all(df[PROB_COLS].to_numpy(), df.result.to_numpy())


def tune(m: pd.DataFrame, xi: float, priors: dict) -> dict:
    """Validation curves: {variant: [{lam, rps, brier, log_loss}, ...]}, lam=0 is dc_v1."""
    base = _scores(walk_forward(m, xi, config.VALID_SEASONS))
    curves = {}
    for name, kw in [*VARIANTS.items(), DIAGNOSTIC]:
        rows = [{"lam": 0.0, **base}]
        for lam in LAM_GRID:
            sc = _scores(walk_forward(m, xi, config.VALID_SEASONS, lam=lam, priors=priors, **kw))
            rows.append({"lam": lam, **sc})
            print(f"  {name:<20} lam={lam:<6} rps={sc['rps']:.5f}")
        curves[name] = rows
    return curves


def choose(curves: dict) -> tuple[str, float]:
    """Lowest validation RPS over the candidate variants (never the diagnostic)."""
    best = min(((name, r) for name in VARIANTS for r in curves[name]),
               key=lambda x: x[1]["rps"])
    return (best[0], best[1]["lam"]) if best[1]["lam"] > 0 else ("dc_v1", 0.0)


def compare(base: pd.DataFrame, var: pd.DataFrame, m: pd.DataFrame, seed: int) -> dict:
    """variant vs dc_v1 on the same matches (positive diff = variant worse)."""
    assert base.index.equals(var.index)
    y = base.result.to_numpy()
    d = per_match_rps(var[PROB_COLS].to_numpy(), y) - per_match_rps(base[PROB_COLS].to_numpy(), y)
    rng = np.random.default_rng(seed)
    early = promoted_early(base, m)
    per_season = []
    for s in sorted(base.season.unique()):
        k = (base.season == s).to_numpy()
        per_season.append({"season": s, "n": int(k.sum()),
                           "rps_dc_v1": _scores(base[k])["rps"],
                           "rps_variant": _scores(var[k])["rps"],
                           "rps_diff": block_bootstrap(d[k], base.date.to_numpy()[k], rng)})
    subgroups = {}
    for label, k in (("promoted_first_10_games", early), ("all_other_matches", ~early)):
        subgroups[label] = {"n": int(k.sum()), "dc_v1": _scores(base[k]),
                            "variant": _scores(var[k]),
                            "rps_diff": block_bootstrap(d[k], base.date.to_numpy()[k], rng)}
    bet_rng = np.random.default_rng(config.SEED)
    return {
        "n_matches": int(len(base)),
        "dc_v1": _scores(base), "variant": _scores(var),
        "market_close": score_all(devig(base[backtest.CLOSE_COLS].to_numpy()), y),
        "rps_diff": block_bootstrap(d, base.date.to_numpy(), rng),
        "per_season": per_season,
        "subgroups": subgroups,
        "calibration": {name: {o: calibration_table(df[PROB_COLS].to_numpy(), y, i)
                               for i, o in enumerate(("home", "draw", "away"))}
                        for name, df in (("dc_v1", base), ("variant", var))},
        "betting": {name: [{k: v for k, v in backtest.simulate_bets(df, th, bet_rng).items()
                            if k != "pnl_series"} for th in config.EDGE_THRESHOLDS]
                    for name, df in (("dc_v1", base), ("variant", var))},
        "random_bet_baseline": {k: v for k, v in backtest.random_bet_baseline(
            base, np.random.default_rng(0)).items() if k != "pnl_series"},
    }


def live_case(xi: float, priors: dict, chosen: dict) -> dict:
    """The committed Coventry v Newcastle prediction, refitted on the same data
    (season CSV + cached football-data.org rows) under dc_v1 and each variant."""
    from premodel.results import load_cache, merge_results, parse
    case = LIVE_CASE
    pred = json.loads((case["round_dir"] / "predictions.json").read_bytes())
    fx = next(f for f in pred["fixtures"] if (f["home"], f["away"]) == (case["home"],
                                                                         case["away"]))
    doc = load_cache()
    data = merge_results(load_matches(), parse(doc) if doc else None)
    date = case["date"]
    train = data[(data.date < date) & (data.date >= date - pd.Timedelta(
        days=config.LOOKBACK_DAYS))]
    season = season_of(date)
    out = {"committed": [fx["p_home"], fx["p_draw"], fx["p_away"]],
           "n_training_matches": int(len(train)),
           "committed_n_training_matches": pred["training"]["n_matches"],
           "market_open_devigged": devig(np.array([[fx["odds_open"][k] for k in
                                                    ("home", "draw", "away")]]))[0].tolist(),
           "odds_source": fx["odds_source"], "models": {}}
    promoted = promoted_teams(data, season, date)
    for name, (variant, lam) in {"dc_v1": ("dc_v1", 0.0), **chosen}.items():
        shrink = None
        if variant != "dc_v1":
            kw = VARIANTS.get(variant) or DIAGNOSTIC[1]
            mean = (0.0, 0.0) if kw["zero_prior"] else priors[season]
            shrink = dc.Shrinkage(lam, {t: mean for t in promoted}, only_prior=kw["only_prior"])
        params = fit_before(data, date, xi, shrink=shrink)
        p, _ = dc.predict_1x2(params, [case["home"]], [case["away"]], dc.promoted_prior(params))
        i = params.index()[case["home"]]
        out["models"][name] = {"variant": variant, "lam": lam, "p": p[0].tolist(),
                               "coventry_att": float(params.att[i]),
                               "coventry_dfn": float(params.dfn[i])}
    # committed probabilities are rounded to millionths
    out["dc_v1_reproduces_committed"] = bool(np.abs(
        np.array(out["models"]["dc_v1"]["p"]) - np.array(out["committed"])).max() <= 1.5e-6)
    out["promoted_teams_at_fit"] = sorted(promoted)
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run(out_dir: Path = OUT_DIR) -> dict:
    pub = json.loads((config.REPORTS_DIR / "summary.json").read_text())
    xi, test_seasons = pub["chosen_xi"], pub["test_seasons"]
    m = load_matches()

    print("Checking dc_v1 reproduces the published test set ...")
    base_all = walk_forward(m, xi, test_seasons)
    base = base_all[base_all.date <= PUBLISHED_TEST_END]
    got, ref = _scores(base), pub["benchmarks"]["Dixon-Coles"]
    if len(base) != PUBLISHED_N or any(round(got[k], 4) != round(ref[k], 4) for k in ref):
        raise RuntimeError(f"dc_v1 does not reproduce reports/summary.json: n={len(base)} "
                           f"{got} vs {ref}; stopping")
    print(f"  ok: n={len(base)} rps={got['rps']:.4f} brier={got['brier']:.4f} "
          f"log_loss={got['log_loss']:.4f}")

    priors = promoted_priors(m, sorted(m.season.unique()), xi)
    print("Tuning lam on validation seasons ...")
    curves = tune(m, xi, priors)
    variant, lam = choose(curves)
    best = {name: min(curves[name], key=lambda r: r["rps"])["lam"]
            for name in [*VARIANTS, DIAGNOSTIC[0]]}
    print(f"  -> chosen: {variant}, lam={lam}")

    def test_run(name):
        kw = VARIANTS.get(name) or DIAGNOSTIC[1]
        if best[name] == 0:
            return base_all
        return walk_forward(m, xi, test_seasons, lam=best[name], priors=priors, **kw)

    runs = {name: test_run(name) for name in [*VARIANTS, DIAGNOSTIC[0]]}
    pub_mask = lambda df: df.date <= PUBLISHED_TEST_END  # noqa: E731
    results = {name: compare(base, df[pub_mask(df)], m, config.SEED)
               for name, df in runs.items()}
    extra = base_all[~pub_mask(base_all)]
    extended = ({name: compare(base_all, df, m, config.SEED) for name, df in runs.items()}
                if len(extra) else None)
    secondary = [n for n in VARIANTS if n != variant]
    summary = {
        "xi": xi, "lam_grid": LAM_GRID, "priors_by_season": priors,
        "published_check": {"n": len(base), "dc_v1": got, "published": ref},
        "tuning": curves,
        "best_lam_per_variant": best,
        "chosen": {"variant": variant, "lam": lam},
        "primary_test": {"set": f"published {PUBLISHED_N} matches (test seasons up to "
                                f"{PUBLISHED_TEST_END:%Y-%m-%d})",
                         "comparison": results.get(variant) if variant != "dc_v1" else None},
        "secondary_test": {name: results[name] for name in secondary},
        "diagnostic_test": {DIAGNOSTIC[0]: results[DIAGNOSTIC[0]]},
        "extended_test": ({"n_extra_matches": int(len(extra)), "comparisons": extended}
                          if extended else
                          {"n_extra_matches": 0, "note": "the season CSV has no matches after "
                           f"{PUBLISHED_TEST_END:%Y-%m-%d} yet, so the extended set equals "
                           "the published set"}),
        "live_case": live_case(xi, priors, {name: (name, best[name])
                                            for name in [*VARIANTS, DIAGNOSTIC[0]]}),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")
    _print(summary)
    return summary


def _print(s: dict):
    print("\nValidation (lowest RPS per variant)")
    for name, rows in s["tuning"].items():
        r = min(rows, key=lambda r: r["rps"])
        print(f"  {name:<20} lam={r['lam']:<6} rps={r['rps']:.5f}  (dc_v1 {rows[0]['rps']:.5f})")
    print(f"Chosen: {s['chosen']['variant']} lam={s['chosen']['lam']}")
    for title, block in (("Primary", {s["chosen"]["variant"]: s["primary_test"]["comparison"]}),
                         ("Secondary", s["secondary_test"]), ("Diagnostic", s["diagnostic_test"])):
        for name, c in block.items():
            if c is None:
                continue
            d = c["rps_diff"]
            print(f"{title} test, {name}: rps {c['variant']['rps']:.5f} vs dc_v1 "
                  f"{c['dc_v1']['rps']:.5f}  diff {d['mean']:+.5f} "
                  f"[{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]")
            for k, g in c["subgroups"].items():
                gd = g["rps_diff"]
                print(f"    {k:<24} n={g['n']:<5} diff {gd['mean']:+.5f} "
                      f"[{gd['ci95'][0]:+.5f}, {gd['ci95'][1]:+.5f}]")
    lc = s["live_case"]
    print(f"Coventry v Newcastle: committed {lc['committed']}, market open "
          f"{[round(x, 3) for x in lc['market_open_devigged']]}")
    for name, r in lc["models"].items():
        print(f"  {name:<20} lam={r['lam']:<6} p={[round(x, 3) for x in r['p']]} "
              f"att={r['coventry_att']:+.2f}")
