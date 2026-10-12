"""Experiment: a joint multi-league model (dc_joint) vs dc_v1 in E0 and E1.

One rating per team across E0, E1 and E2 (League One), so a promoted or relegated team
carries its history across leagues. Decision rules are fixed in DECISION_RULES before
any result is computed and written into summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from premodel import backtest, config
from premodel.data import devig, load_matches
from premodel.experiments import championship as ch
from premodel.experiments.shrinkage import _clean, per_match_log_loss, per_match_rps
from premodel.metrics import calibration_table, score_all
from premodel.models import dixon_coles_joint as dj

OUT_DIR = config.REPORTS_DIR / "experiments" / "joint"
LEAGUES = dj.LEAGUE_ORDER                 # all fitted on
PREDICTED = ["E0", "E1"]                  # evaluated
RANK = {lg: r for r, lg in enumerate(LEAGUES)}
P = ["p_h", "p_d", "p_a"]
VARIANTS = {"plain": (), "move_shift": (dj.UP,)}  # move_shift: d_down fixed at 0
XI_GRID = [0.001, 0.0019, 0.003, 0.005, 0.007, 0.01]
DC_V1_XI = {"E0": 0.003, "E1": 0.005}      # published / championship-tuned baselines
KEYS = ["league", "date", "home", "away"]
LIVE_CASE = {"round_dir": config.PENDING_DIR / "E0-2026-10-12", "home": "Coventry",
             "away": "Newcastle", "date": pd.Timestamp("2026-10-12"), "league": "E0"}

# Fixed before any result was computed (approved plan), written into summary.json.
DECISION_RULES = {
    "protocol": "dc_joint fitted on E0+E1+E2 matches strictly before each date on which an "
                "E0 or E1 match is predicted (3-year lookback). Variant (plain, or move "
                "shift with d_down = 0) and xi in {0.001, 0.0019, 0.003, 0.005, 0.007, 0.01} "
                "chosen by the mean of E0 and E1 validation RPS (2015/16-2017/18), equal "
                "weight per league; ties: plain, then smaller xi; a grid-edge optimum is a "
                "limitation, the grid is not extended. Baseline dc_v1 at its own xi (E0 "
                "0.003, E1 0.005) must first reproduce E0 RPS 0.2016 on 3,090 matches and "
                "the championship experiment's E1 RPS on 4,511, else stop. Test 2018/19 to "
                "2026-09-20. CIs: paired date-block bootstrap, 5,000 resamples; 95% decides, "
                "98.3% (Bonferroni, 3 questions) is a robustness line.",
    "Q1": "E0 test: per-match RPS(dc_joint) - RPS(dc_v1). Improvement if the 95% CI is "
          "entirely below 0.",
    "Q2": "E1 test: same.",
    "Q3": "Pre-registered subgroup: matches involving a team in its first 10 games after "
          "entering E0 or E1, both leagues pooled (dates resampled jointly). Same rule. Per "
          "league and per direction of entry reported, not decisive.",
    "interpretation": "Unconstrained up/down shifts and cross-league gaps are not "
                      "interpreted (not identified, or only weakly).",
}


def load_all() -> pd.DataFrame:
    """E0, E1 and E2 matches in one frame with a league column, sorted by date."""
    frames = [load_matches(league=lg).assign(league=lg) for lg in LEAGUES]
    m = pd.concat(frames, ignore_index=True)
    return m.sort_values(["date", "league", "home"], kind="stable").reset_index(drop=True)


def league_by_season(m: pd.DataFrame) -> dict:
    """(team, season) -> league. Each team plays in one league per season."""
    out = {}
    for (s, lg), g in m.groupby(["season", "league"]):
        for t in set(g.home) | set(g.away):
            out[(t, s)] = lg
    return out


def move_directions(df: pd.DataFrame, m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per match in `df`: each side's move direction (dj.UP / DOWN / NONE), comparing the
    match's league with the team's league in the previous season (from `m`). Causal:
    the previous season is complete before the current one starts."""
    seasons = sorted(set(m.season) | set(df.season))
    prev = {s: (seasons[i - 1] if i else None) for i, s in enumerate(seasons)}
    where = league_by_season(m[m.season.isin(set(prev[s] for s in df.season.unique()))])

    def direction(team, season, league):
        p = where.get((team, prev[season])) if prev[season] else None
        if p is None or p == league:
            return dj.NONE
        return dj.UP if RANK[p] > RANK[league] else dj.DOWN

    rows = list(zip(df.home, df.away, df.season, df.league, strict=True))
    return (np.array([direction(h, s, lg) for h, _, s, lg in rows], dtype=int),
            np.array([direction(a, s, lg) for _, a, s, lg in rows], dtype=int))


def fit_before(m: pd.DataFrame, date: pd.Timestamp, xi: float, shifts=(), init=None,
               lookback_days: int = config.LOOKBACK_DAYS) -> dj.JointParams:
    """dc_joint on matches from all leagues strictly before `date` (lookback window)."""
    train = m[(m.date < date) & (m.date >= date - pd.Timedelta(days=lookback_days))]
    mh, ma = move_directions(train, m[m.date < date]) if shifts else (None, None)
    return dj.fit(train.home.to_numpy(), train.away.to_numpy(), train.league.to_numpy(),
                  train.hg.to_numpy(), train.ag.to_numpy(),
                  (date - train.date).dt.days.to_numpy(), xi, mh, ma, shifts, init)


def walk_forward(m: pd.DataFrame, xi: float, seasons: list[str], shifts=(),
                 predict_leagues=PREDICTED) -> pd.DataFrame:
    """Refit before every date with a match to predict (E0 or E1), on all leagues' matches
    strictly before that date; predict that date's E0/E1 matches."""
    target = m[m.season.isin(seasons) & m.league.isin(predict_leagues)]
    preds, params = [], None
    for date, day in target.groupby("date", sort=True):
        params = fit_before(m, date, xi, shifts, init=params)
        mh, ma = move_directions(day, m[m.date < date]) if shifts else (None, None)
        out = day.copy()
        p = np.zeros((len(day), 3))
        xg = np.zeros((len(day), 2))
        for lg in day.league.unique():
            k = (day.league == lg).to_numpy()
            p[k], xg[k] = dj.predict_1x2(params, day.home.to_numpy()[k], day.away.to_numpy()[k],
                                         lg, None if mh is None else mh[k],
                                         None if ma is None else ma[k])
        out[P] = p
        out[["xg_h", "xg_a"]] = xg
        out["n_connecting_movers"] = sum(len(v) > 1 for v in params.played_in.values())
        out["connected"] = len(set(params.component.values())) == 1
        preds.append(out)
    return pd.concat(preds)


def coverage(m: pd.DataFrame) -> list[dict]:
    rows = []
    for (lg, s), g in m.groupby(["league", "season"]):
        rows.append({"league": lg, "season": s, "matches": len(g),
                     "teams": len(set(g.home) | set(g.away)),
                     "first": f"{g.date.min():%Y-%m-%d}", "last": f"{g.date.max():%Y-%m-%d}"})
    return rows


# --- selection --------------------------------------------------------------------------

def league_rps(df: pd.DataFrame) -> dict:
    return {lg: float(per_match_rps(g[P].to_numpy(), g.result.to_numpy()).mean())
            for lg, g in df.groupby("league")}


def choose_config(rows: list[dict]) -> dict:
    """Lowest mean(E0, E1) validation RPS; ties: plain first, then smaller xi."""
    order = list(VARIANTS)
    return min(rows, key=lambda r: (r["score"], order.index(r["variant"]), r["xi"]))


def tune(m: pd.DataFrame) -> tuple[list[dict], dict]:
    rows, frames = [], {}
    for variant, shifts in VARIANTS.items():
        for xi in XI_GRID:
            v = walk_forward(m, xi, config.VALID_SEASONS, shifts)
            r = league_rps(v)
            rows.append({"variant": variant, "xi": xi, "rps_E0": r["E0"], "rps_E1": r["E1"],
                         "score": (r["E0"] + r["E1"]) / 2,
                         "disconnected_fits": int((~v.drop_duplicates("date").connected).sum())})
            frames[(variant, xi)] = v
            print(f"  {variant:<11} xi={xi:<7} E0 {r['E0']:.5f}  E1 {r['E1']:.5f}  "
                  f"mean {rows[-1]['score']:.5f}")
    return rows, frames


# --- comparison -------------------------------------------------------------------------

def attach(base: pd.DataFrame, joint: pd.DataFrame, league: str) -> pd.DataFrame:
    """dc_v1 rows (league's own index) with dc_joint probabilities as p_*_joint."""
    j = joint[joint.league == league][KEYS + P].rename(columns={c: f"{c}_joint" for c in P})
    out = base.assign(league=league).reset_index().merge(j, on=KEYS, how="left",
                                                         validate="one_to_one")
    out = out.set_index("index")
    out.index.name = None
    if out[[f"{c}_joint" for c in P]].isna().any().any():
        raise RuntimeError(f"{league}: dc_joint is missing predictions for some matches")
    return out


def as_model(df: pd.DataFrame, which: str) -> pd.DataFrame:
    """Copy with p_h/p_d/p_a set to dc_v1 ("v1") or dc_joint ("joint") probabilities."""
    return df.assign(**{c: df[f"{c}_joint"] for c in P}) if which == "joint" else df


def diffs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = df.result.to_numpy()
    pj, pv = df[[f"{c}_joint" for c in P]].to_numpy(), df[P].to_numpy()
    return (per_match_rps(pj, y) - per_match_rps(pv, y),
            per_match_log_loss(pj, y) - per_match_log_loss(pv, y))


def compare_league(df: pd.DataFrame, valid_joint: pd.DataFrame, m: pd.DataFrame,
                   upper: pd.DataFrame | None, labels: np.ndarray) -> dict:
    rng = np.random.default_rng(config.SEED + 10)
    d_rps, d_ll = diffs(df)
    dates = df.date.to_numpy()
    y = df.result.to_numpy()
    hc = ch.has(df, ch.CLOSE)
    out = {"n": int(len(df)),
           "dc_v1": score_all(df[P].to_numpy(), y),
           "dc_joint": score_all(df[[f"{c}_joint" for c in P]].to_numpy(), y),
           "rps_diff": ch.date_block_bootstrap(d_rps, dates, rng),
           "log_loss_diff": ch.date_block_bootstrap(d_ll, dates, rng)}
    close = devig(df.loc[hc, ch.CLOSE].to_numpy())
    out["gap_to_close"] = {
        name: ch.date_block_bootstrap(
            per_match_rps(as_model(df, name)[P].to_numpy()[hc], y[hc])
            - per_match_rps(close, y[hc]), dates[hc], rng) for name in ("v1", "joint")}
    out["clv_minus_random_same_match"] = {}
    for name in ("v1", "joint"):
        b = ch.model_bets(as_model(df, name))
        out["clv_minus_random_same_match"][name] = {
            **ch.date_block_bootstrap((b.clv - b.random_clv).to_numpy(), b.date.to_numpy(),
                                      rng), "n_bets": int(len(b))}
    w, curve = ch.choose_blend(valid_joint)
    t = df[ch.has(df, ch.OPEN)]
    po = devig(t[ch.OPEN].to_numpy())
    blend = w * t[[f"{c}_joint" for c in P]].to_numpy() + (1 - w) * po
    yo = t.result.to_numpy()
    out["blend_dc_joint"] = {"w": w, "validation_curve": curve, "rps_diff_blend_minus_open":
                             ch.date_block_bootstrap(per_match_rps(blend, yo)
                                                     - per_match_rps(po, yo),
                                                     t.date.to_numpy(), rng)}
    out["calibration"] = {name: {o: calibration_table(as_model(df, name)[P].to_numpy(), y, i)
                                 for i, o in enumerate(("home", "draw", "away"))}
                          for name in ("v1", "joint")}
    out["per_season"] = []
    for s in sorted(df.season.unique()):
        k = (df.season == s).to_numpy()
        out["per_season"].append({"season": s, "n": int(k.sum()),
                                  "rps_diff": ch.date_block_bootstrap(d_rps[k], dates[k], rng)})
    out["entrant_subgroups"] = {}
    for name in ("from_above", "from_below", ""):
        k = labels == name
        if k.any():
            out["entrant_subgroups"][name or "all_other_matches"] = {
                "n": int(k.sum()),
                "rps_dc_v1": float(per_match_rps(df[P].to_numpy()[k], y[k]).mean()),
                "rps_dc_joint": float(per_match_rps(
                    df[[f"{c}_joint" for c in P]].to_numpy()[k], y[k]).mean()),
                "rps_diff": ch.date_block_bootstrap(d_rps[k], dates[k], rng)}
    return out


def live_case(xi: float, shifts) -> dict:
    """Coventry v Newcastle 2026-10-12, refitted on the committed forecast's E0 data (CSV +
    cached football-data.org rows) plus E1 and E2 CSV matches before the date."""
    from premodel.results import load_cache, merge_results, parse
    case = LIVE_CASE
    pred = json.loads((case["round_dir"] / "predictions.json").read_bytes())
    fx = next(f for f in pred["fixtures"] if (f["home"], f["away"]) == (case["home"],
                                                                         case["away"]))
    doc = load_cache()
    e0 = merge_results(load_matches(), parse(doc) if doc else None).assign(league="E0")
    m = pd.concat([e0] + [load_matches(league=lg).assign(league=lg) for lg in ("E1", "E2")],
                  ignore_index=True).sort_values(["date", "league", "home"], kind="stable")
    date = case["date"]
    params = fit_before(m, date, xi, shifts)
    day = pd.DataFrame({"home": [case["home"]], "away": [case["away"]], "league": ["E0"],
                        "season": ["2026/27"], "date": [date]})
    mh, ma = move_directions(day, m[m.date < date]) if shifts else (None, None)
    p, _ = dj.predict_1x2(params, [case["home"]], [case["away"]], "E0", mh, ma)
    i = params.index()[case["home"]]
    train = m[(m.date < date) & (m.date >= date - pd.Timedelta(days=config.LOOKBACK_DAYS))]
    return {"committed_dc_v1": [fx["p_home"], fx["p_draw"], fx["p_away"]],
            "committed_n_training_matches_e0": pred["training"]["n_matches"],
            "n_training_matches_e0": int((train.league == "E0").sum()),
            "n_training_matches_all_leagues": int(len(train)),
            "dc_joint": p[0].tolist(),
            "coventry": {"att": float(params.att[i]), "dfn": float(params.dfn[i]),
                         "leagues_in_window": sorted(params.played_in[case["home"]]),
                         "move_direction": int(mh[0]) if mh is not None else None},
            "market_open_devigged": devig(np.array([[fx["odds_open"][k] for k in
                                                     ("home", "draw", "away")]]))[0].tolist()}


def run(out_dir: Path = OUT_DIR) -> dict:
    pub = json.loads((config.REPORTS_DIR / "summary.json").read_text())
    champ = json.loads((ch.OUT_DIR / "summary.json").read_text())
    seasons = pub["test_seasons"]
    m = load_all()
    single = {lg: load_matches(league=lg) for lg in PREDICTED}

    print("Reproducing dc_v1 baselines ...")
    base = {}
    for lg, ref, n_ref in (("E0", pub["benchmarks"]["Dixon-Coles"], pub["n_test_matches"]),
                           ("E1", champ["leagues"]["E1"]["scores_all_matches"]["model"],
                            champ["test_window"]["n"]["E1"])):
        t = backtest.walk_forward(single[lg], DC_V1_XI[lg], seasons)
        t = t[t.date <= ch.TEST_END]
        got = score_all(t[P].to_numpy(), t.result.to_numpy())
        if len(t) != n_ref or any(round(got[k], 4) != round(ref[k], 4) for k in ref):
            raise RuntimeError(f"{lg} dc_v1 does not reproduce its baseline: n={len(t)} "
                               f"{got} vs {ref}; stopping")
        base[lg] = t
        print(f"  {lg} ok: n={len(t)} rps={got['rps']:.4f}")

    print("Choosing variant and xi on validation (mean of E0 and E1 RPS) ...")
    rows, frames = tune(m)
    best = choose_config(rows)
    edge = best["xi"] in (min(XI_GRID), max(XI_GRID))
    print(f"  -> {best['variant']}, xi={best['xi']}{'  (grid edge)' if edge else ''}")
    other = min((r for r in rows if r["variant"] != best["variant"]),
                key=lambda r: (r["score"], r["xi"]))

    print("Test walk-forward ...")
    runs = {}
    for cfg in (best, other):
        t = walk_forward(m, cfg["xi"], seasons, VARIANTS[cfg["variant"]])
        runs[cfg["variant"]] = t[t.date <= ch.TEST_END]

    def evaluate(cfg):
        test = runs[cfg["variant"]]
        valid = frames[(cfg["variant"], cfg["xi"])]
        out, pooled = {}, {"rps": [], "dates": [], "labels": []}
        for lg, upper in (("E0", None), ("E1", single["E0"])):
            df = attach(base[lg], test, lg)
            labels = ch.entrants(df, single[lg], upper)
            out[lg] = compare_league(df, valid[valid.league == lg], single[lg], upper, labels)
            d_rps, _ = diffs(df)
            k = labels != ""
            pooled["rps"].append(d_rps[k])
            pooled["dates"].append(df.date.to_numpy()[k])
            pooled["labels"].append(np.array([f"{lg}_{x}" for x in labels[k]]))
        q3 = ch.date_block_bootstrap(np.concatenate(pooled["rps"]),
                                     np.concatenate(pooled["dates"]),
                                     np.random.default_rng(config.SEED + 20))
        conn = test.drop_duplicates("date")
        return {"leagues": out, "q3_pooled_new_teams": q3,
                "connectivity_test_fits": {
                    "n_fits": int(len(conn)), "disconnected": int((~conn.connected).sum()),
                    "connecting_movers_min": int(conn.n_connecting_movers.min()),
                    "connecting_movers_median": float(conn.n_connecting_movers.median())}}

    primary = evaluate(best)
    secondary = evaluate(other)

    def decide(level):
        return {"Q1_e0_improves": bool(primary["leagues"]["E0"]["rps_diff"][level][1] < 0),
                "Q2_e1_improves": bool(primary["leagues"]["E1"]["rps_diff"][level][1] < 0),
                "Q3_new_teams_improve": bool(primary["q3_pooled_new_teams"][level][1] < 0)}

    last = fit_before(m, ch.TEST_END + pd.Timedelta(days=1), best["xi"],
                      VARIANTS[best["variant"]])
    summary = {
        "decision_rules": DECISION_RULES,
        "coverage": coverage(m),
        "baselines": {lg: {"xi": DC_V1_XI[lg], "n": len(base[lg])} for lg in PREDICTED},
        "selection": {"grid": XI_GRID, "rows": rows, "chosen": best, "chosen_at_grid_edge":
                      bool(edge), "best_other_variant": other},
        "primary": {"config": best, **primary},
        "decisions_95": decide("ci95"),
        "decisions_983_robustness": decide("ci983"),
        "secondary_other_variant": {"config": other, **secondary},
        "fitted_at_test_end": {"gamma": last.gamma, "b": last.b,
                               "shift_up": last.shift.get(dj.UP), "rho": last.rho,
                               "note": "b and shifts are identified only through movers; "
                                       "not interpreted as league gaps"},
        "live_case": live_case(best["xi"], VARIANTS[best["variant"]]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")
    _print(summary)
    return summary


def _print(s: dict):
    f = ch._fmt
    p = s["primary"]
    print(f"\nchosen: {p['config']['variant']}, xi={p['config']['xi']}"
          f"{' (grid edge)' if s['selection']['chosen_at_grid_edge'] else ''}")
    for lg, q in (("E0", "Q1"), ("E1", "Q2")):
        c = p["leagues"][lg]
        print(f"{q} {lg}: RPS joint {c['dc_joint']['rps']:.4f} vs v1 {c['dc_v1']['rps']:.4f}  "
              f"diff {f(c['rps_diff'])}")
    print(f"Q3 new teams pooled: {f(p['q3_pooled_new_teams'])}")
    for lg in PREDICTED:
        for g, v in p["leagues"][lg]["entrant_subgroups"].items():
            print(f"   {lg} {g:<18} n={v['n']:<5} diff {f(v['rps_diff'])}")
    print("decisions (95%):", s["decisions_95"])
    print("decisions (98.3%, robustness):", s["decisions_983_robustness"])
    lc = s["live_case"]
    print(f"Coventry v Newcastle: committed dc_v1 {lc['committed_dc_v1']}, dc_joint "
          f"{[round(x, 3) for x in lc['dc_joint']]}, market "
          f"{[round(x, 3) for x in lc['market_open_devigged']]}")
