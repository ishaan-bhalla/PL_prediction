"""Experiment: is the Championship (E1) betting market softer than the Premier League's?

The model is held fixed (dc_v1, the same walk-forward code as the E0 backtest); only
the market changes. Decision rules are fixed in DECISION_RULES before any E1 result
is computed and are written into summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from premodel import backtest, config
from premodel.data import devig, load_matches
from premodel.experiments.shrinkage import _clean, per_match_rps
from premodel.metrics import calibration_table, score_all

OUT_DIR = config.REPORTS_DIR / "experiments" / "championship"
SEASON_MATCHES = {"E0": 380, "E1": 552}
# Same calendar window as the published E0 test set (reports/summary.json).
TEST_END = pd.Timestamp("2026-09-20")
P, OPEN, CLOSE = backtest.PROB_COLS, backtest.OPEN_COLS, backtest.CLOSE_COLS
BLEND_GRID = [round(w, 2) for w in np.linspace(0, 1, 21)]
N_PRIMARY = 3
BONFERRONI = 1 - 0.05 / N_PRIMARY  # 98.33% intervals, robustness only
FIRST_GAMES = 10

# Fixed before any E1 result was computed (approved plan), written into summary.json.
DECISION_RULES = {
    "protocol": "dc_v1 unchanged; walk-forward refit per match date, 3-year lookback; xi "
                "tuned on E1 validation (2015/16-2017/18) by RPS on the E0 grid, a grid-edge "
                "optimum reported as a limitation, grid not extended; test = 2018/19 to "
                "2026-09-20 (the published E0 window) in both leagues; matches without the "
                "odds a comparison needs are excluded from it and counted; 95% CIs from a "
                "date-block bootstrap (5,000 resamples); 98.3% (Bonferroni, 3 primary "
                "questions) reported as robustness only.",
    "Q1": "RPS gap = model - devigged close, per match. E1 gap smaller than E0 gap if the "
          "95% CI of (gap E1 - gap E0), resampling match dates jointly across both leagues, "
          "lies entirely below 0. Independent per-league bootstrap: sensitivity only.",
    "Q2": "For each match the model bets on (edge > 0 at open odds, backtest rule): CLV of "
          "the bet minus the expected CLV of a uniformly random bet on the same match. "
          "E1 market softer if the 95% CI of the mean lies entirely above 0.",
    "Q3": "p = w*model + (1-w)*devigged open, w on a 0..1 grid (step 0.05) chosen by E1 "
          "validation RPS. Model adds information the E1 market lacks if w > 0 and the 95% "
          "CI of test RPS (blend - open) lies entirely below 0.",
}


def coverage_table(m: pd.DataFrame, league: str) -> pd.DataFrame:
    """Per season: matches, teams, dates and where the open / close odds come from."""
    rows = []
    for s, g in m.groupby("season"):
        row = {"season": s, "matches": len(g),
               "expected": SEASON_MATCHES[league] if s < config.CURRENT_SEASON_LABEL else None,
               "teams": len(set(g.home) | set(g.away)), "dates": g.date.nunique(),
               "first": f"{g.date.min():%Y-%m-%d}", "last": f"{g.date.max():%Y-%m-%d}",
               "in_test_window": int((g.date <= TEST_END).sum())
               if s >= config.TEST_START else 0}
        for kind in ("open", "close"):
            src = g[f"{kind}_src"]
            row[f"{kind}_pinnacle"] = int((src == "pinnacle").sum())
            row[f"{kind}_avg"] = int((src == "market_avg").sum())
            row[f"{kind}_missing"] = int((src == "").sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("season")


def print_coverage(leagues=("E0", "E1")):
    for lg in leagues:
        t = coverage_table(load_matches(league=lg), lg)
        print(f"\n{lg}: {int(t.matches.sum())} matches")
        print(t.to_string())


# --- bootstrap --------------------------------------------------------------------------

def _summarise(boot: np.ndarray, mean: float, **extra) -> dict:
    a = (1 - BONFERRONI) / 2 * 100
    return {"mean": float(mean),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "ci983": [float(np.percentile(boot, a)), float(np.percentile(boot, 100 - a))],
            **extra}


def _by_date(values, dates, index: pd.Index):
    codes = index.get_indexer(pd.DatetimeIndex(dates))
    return (np.bincount(codes, weights=values, minlength=len(index)),
            np.bincount(codes, minlength=len(index)).astype(float))


def date_block_bootstrap(values: np.ndarray, dates, rng: np.random.Generator,
                         n: int = config.BOOTSTRAP_SAMPLES) -> dict:
    """Mean of per-match `values`, resampling whole match dates (shared fits / shocks)."""
    idx = pd.Index(pd.unique(pd.DatetimeIndex(dates)))
    s, c = _by_date(values, dates, idx)
    draw = rng.integers(0, len(idx), size=(n, len(idx)))
    boot = s[draw].sum(axis=1) / c[draw].sum(axis=1)
    return _summarise(boot, values.mean(), n_matches=int(len(values)), n_dates=len(idx))


def joint_difference_bootstrap(v1, d1, v0, d0, rng: np.random.Generator,
                               n: int = config.BOOTSTRAP_SAMPLES) -> dict:
    """mean(v1) - mean(v0), resampling the union of match dates jointly, so a date's E0
    and E1 matches enter or leave a resample together."""
    idx = pd.Index(pd.unique(pd.DatetimeIndex(np.concatenate([d1, d0]))))
    s1, c1 = _by_date(v1, d1, idx)
    s0, c0 = _by_date(v0, d0, idx)
    draw = rng.integers(0, len(idx), size=(n, len(idx)))
    boot = s1[draw].sum(1) / c1[draw].sum(1) - s0[draw].sum(1) / c0[draw].sum(1)
    return _summarise(boot, v1.mean() - v0.mean(), n_dates=len(idx))


def independent_difference_bootstrap(v1, d1, v0, d0, rng: np.random.Generator,
                                     n: int = config.BOOTSTRAP_SAMPLES) -> dict:
    def draws(v, d):
        idx = pd.Index(pd.unique(pd.DatetimeIndex(d)))
        s, c = _by_date(v, d, idx)
        k = rng.integers(0, len(idx), size=(n, len(idx)))
        return s[k].sum(1) / c[k].sum(1)
    return _summarise(draws(v1, d1) - draws(v0, d0), v1.mean() - v0.mean())


# --- per-match quantities ---------------------------------------------------------------

def has(df: pd.DataFrame, cols) -> np.ndarray:
    return df[cols].notna().all(axis=1).to_numpy()


def model_bets(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """The backtest rule (backtest.simulate_bets), per match: best edge at open odds if it
    exceeds `threshold`. Also the expected CLV of a random bet on the same match."""
    d = df[has(df, OPEN) & has(df, CLOSE)]
    p, odds = d[P].to_numpy(), d[OPEN].to_numpy()
    fair = devig(d[CLOSE].to_numpy())
    edge = p * odds - 1
    pick = edge.argmax(axis=1)
    r = np.arange(len(d))
    take = edge[r, pick] > threshold
    clv_all = odds * fair - 1
    out = pd.DataFrame({
        "date": d.date.to_numpy(), "pick": pick, "odds": odds[r, pick],
        "clv": clv_all[r, pick], "random_clv": clv_all.mean(axis=1),
        "pnl": np.where(d.result.to_numpy() == pick, odds[r, pick] - 1, -1.0)},
        index=d.index)
    return out[take]


def overround(df: pd.DataFrame) -> dict:
    out = {}
    for kind, cols in (("open", OPEN), ("close", CLOSE)):
        src = df[f"{kind}_src"]
        d = df[has(df, cols)]
        o = (1 / d[cols]).sum(axis=1) - 1
        out[kind] = {"all": float(o.mean()), "n": int(len(o)),
                     **{f"{name}": float(o[src[d.index] == name].mean())
                        for name in ("pinnacle", "market_avg")
                        if (src[d.index] == name).any()}}
    return out


def entrants(df: pd.DataFrame, m: pd.DataFrame, upper: pd.DataFrame | None,
             k: int = FIRST_GAMES) -> np.ndarray:
    """Evaluation label: "from_above" / "from_below" if the match involves a team in its
    first k games after entering the league (relegated from the league above, i.e. in
    `upper` the previous season, or promoted from below), else ""."""
    seasons = sorted(m.season.unique())
    label = np.full(len(df), "", dtype=object)
    pos = {i: j for j, i in enumerate(df.index)}
    for s, g in df.groupby("season"):
        prev = [x for x in seasons if x < s]
        if not prev:
            continue
        prev = prev[-1]
        teams = lambda t, ss: set(t[t.season == ss].home) | set(t[t.season == ss].away)  # noqa: E731
        new = teams(m, s) - teams(m, prev)
        above = teams(upper, prev) if upper is not None else set()
        sm = m[m.season == s].sort_values(["date", "home"], kind="stable")
        count = {t: 0 for t in new}
        early = {}
        for i, h, a in zip(sm.index, sm.home, sm.away, strict=True):
            for t in (h, a):
                if t in count:
                    count[t] += 1
                    if count[t] <= k:
                        early.setdefault(i, []).append("from_above" if t in above
                                                       else "from_below")
        for i in g.index:
            if i in early:  # a match between two entrants takes "from_above" if either is
                label[pos[i]] = "from_above" if "from_above" in early[i] else "from_below"
    return label


# --- per-league evaluation --------------------------------------------------------------

def choose_blend(valid: pd.DataFrame) -> tuple[float, list[dict]]:
    """w minimising validation RPS of w*model + (1-w)*devigged open (ties: smaller w)."""
    v = valid[has(valid, OPEN)]
    y, pm, po = v.result.to_numpy(), v[P].to_numpy(), devig(v[OPEN].to_numpy())
    curve = [{"w": w, "rps": float(per_match_rps(w * pm + (1 - w) * po, y).mean())}
             for w in BLEND_GRID]
    return min(curve, key=lambda r: (r["rps"], r["w"]))["w"], curve


def evaluate_league(test: pd.DataFrame, valid: pd.DataFrame, m: pd.DataFrame,
                    upper: pd.DataFrame | None) -> tuple[dict, dict]:
    """(results, arrays needed for cross-league comparisons)."""
    rng = {k: np.random.default_rng(config.SEED + i)
           for i, k in enumerate(["q1", "q2", "q3", "secondary"])}
    y = test.result.to_numpy()
    dates = test.date.to_numpy()
    hc, ho = has(test, CLOSE), has(test, OPEN)
    rps_model = per_match_rps(test[P].to_numpy(), y)

    # Q1: gap to the closing market (matches with closing odds)
    rps_close = np.full(len(test), np.nan)
    rps_close[hc] = per_match_rps(devig(test.loc[hc, CLOSE].to_numpy()), y[hc])
    gap = rps_model[hc] - rps_close[hc]
    q1 = date_block_bootstrap(gap, dates[hc], rng["q1"])

    # Q2: model-edge bets vs a random bet on the same match
    b = model_bets(test)
    q2 = date_block_bootstrap((b.clv - b.random_clv).to_numpy(), b.date.to_numpy(), rng["q2"])
    both = test[ho & hc]
    q2_context = {
        "n_bets": int(len(b)), "n_matches_with_odds": int(len(both)),
        "mean_clv_model_bets": float(b.clv.mean()),
        "mean_random_clv_same_matches": float(b.random_clv.mean()),
        "mean_random_clv_all_matches": float(model_bets(both, -np.inf).random_clv.mean()),
        "backtest_random_bet_baseline": {
            k: v for k, v in backtest.random_bet_baseline(
                both, np.random.default_rng(0)).items() if k != "pnl_series"},
    }

    # Q3: blend with the open market, weight from validation only
    w, curve = choose_blend(valid)
    t = test[ho]
    p_open = devig(t[OPEN].to_numpy())
    blend = w * t[P].to_numpy() + (1 - w) * p_open
    yo = t.result.to_numpy()
    q3 = date_block_bootstrap(per_match_rps(blend, yo) - per_match_rps(p_open, yo),
                              t.date.to_numpy(), rng["q3"])

    # secondary
    def scores(mask):
        d = test[mask]
        out = {"model": score_all(d[P].to_numpy(), d.result.to_numpy())}
        if has(d, OPEN).all():
            out["open"] = score_all(devig(d[OPEN].to_numpy()), d.result.to_numpy())
        if has(d, CLOSE).all():
            out["close"] = score_all(devig(d[CLOSE].to_numpy()), d.result.to_numpy())
        return {"n": int(mask.sum()), **out}

    per_season = []
    for s in sorted(test.season.unique()):
        k = (test.season == s).to_numpy()
        per_season.append({"season": s, "n": int(k.sum()),
                           "rps_model": float(rps_model[k].mean()),
                           "rps_close": float(np.nanmean(rps_close[k])),
                           "gap": date_block_bootstrap(rps_model[k & hc] - rps_close[k & hc],
                                                       dates[k & hc], rng["secondary"])})
    labels = entrants(test, m, upper)
    subgroups = {}
    for name in ("from_above", "from_below", ""):
        k = labels == name
        if not k.any():
            continue
        kc = k & hc
        subgroups[name or "all_other_matches"] = {
            "n": int(k.sum()),
            "rps_model": float(rps_model[k].mean()), "rps_close": float(rps_close[kc].mean()),
            "gap": date_block_bootstrap(rps_model[kc] - rps_close[kc], dates[kc],
                                        rng["secondary"])}
    results = {
        "n_test": int(len(test)), "n_with_close": int(hc.sum()), "n_with_open": int(ho.sum()),
        "q1_rps_gap_model_minus_close": q1,
        "q2_clv_model_minus_random_same_match": q2, "q2_context": q2_context,
        "q3_blend": {"w": w, "validation_curve": curve, "n_test": int(len(t)),
                     "rps_blend": float(per_match_rps(blend, yo).mean()),
                     "rps_open": float(per_match_rps(p_open, yo).mean()),
                     "rps_diff_blend_minus_open": q3},
        "scores_all_matches": scores(np.ones(len(test), dtype=bool)),
        "scores_matches_with_both_odds": scores(ho & hc),
        "per_season_gap": per_season,
        "roi_threshold_0": date_block_bootstrap(b.pnl.to_numpy(), b.date.to_numpy(),
                                                rng["secondary"]),
        "betting": [{k: v for k, v in backtest.simulate_bets(
            both, th, np.random.default_rng(config.SEED)).items() if k != "pnl_series"}
            for th in config.EDGE_THRESHOLDS],
        "overround": overround(test),
        "calibration": {
            "model": {o: calibration_table(test[P].to_numpy(), y, i)
                      for i, o in enumerate(("home", "draw", "away"))},
            "close_home": calibration_table(devig(test.loc[hc, CLOSE].to_numpy()), y[hc], 0)},
        "entrant_subgroups": subgroups,
    }
    return results, {"gap": gap, "gap_dates": dates[hc]}


def run(out_dir: Path = OUT_DIR) -> dict:
    pub = json.loads((config.REPORTS_DIR / "summary.json").read_text())
    test_seasons = pub["test_seasons"]
    m0, m1 = load_matches(league="E0"), load_matches(league="E1")
    summary = {"decision_rules": DECISION_RULES,
               "coverage": {lg: coverage_table(m, lg).reset_index().to_dict("records")
                            for lg, m in (("E0", m0), ("E1", m1))}}

    print("E0: checking dc_v1 reproduces the published test set ...")
    xi0 = pub["chosen_xi"]
    t0_all = backtest.walk_forward(m0, xi0, test_seasons)
    t0 = t0_all[t0_all.date <= TEST_END]
    got, ref = score_all(t0[P].to_numpy(), t0.result.to_numpy()), pub["benchmarks"]["Dixon-Coles"]
    if len(t0) != pub["n_test_matches"] or any(round(got[k], 4) != round(ref[k], 4)
                                                for k in ref):
        raise RuntimeError(f"E0 dc_v1 does not reproduce reports/summary.json: {got}; stopping")
    print(f"  ok: n={len(t0)} rps={got['rps']:.4f}")

    print("E1: tuning xi on validation seasons ...")
    xi1, tuning = backtest.tune_xi(m1)
    grid = config.XI_GRID
    edge = bool(xi1 in (min(grid), max(grid)))
    print(f"  -> xi = {xi1}{'  (grid edge: limitation, grid not extended)' if edge else ''}")
    t1_all = backtest.walk_forward(m1, xi1, test_seasons)
    t1 = t1_all[t1_all.date <= TEST_END]

    print("Evaluating ...")
    v0 = backtest.walk_forward(m0, xi0, config.VALID_SEASONS)
    v1 = backtest.walk_forward(m1, xi1, config.VALID_SEASONS)
    r0, a0 = evaluate_league(t0, v0, m0, None)
    r1, a1 = evaluate_league(t1, v1, m1, m0)
    q1_joint = joint_difference_bootstrap(a1["gap"], a1["gap_dates"], a0["gap"],
                                          a0["gap_dates"], np.random.default_rng(config.SEED))
    q1_indep = independent_difference_bootstrap(a1["gap"], a1["gap_dates"], a0["gap"],
                                                a0["gap_dates"],
                                                np.random.default_rng(config.SEED))
    q2, q3 = r1["q2_clv_model_minus_random_same_match"], r1["q3_blend"]
    def decide(level):  # the fixed rules, evaluated at a CI level ("ci95" or "ci983")
        return {"Q1_e1_gap_smaller": bool(q1_joint[level][1] < 0),
                "Q2_e1_market_softer": bool(q2[level][0] > 0),
                "Q3_model_adds_information_in_e1": bool(
                    q3["w"] > 0 and q3["rps_diff_blend_minus_open"][level][1] < 0)}

    decisions, robust = decide("ci95"), decide("ci983")
    summary |= {
        "xi": {"E0": xi0, "E1": xi1, "E1_tuning": tuning, "E1_at_grid_edge": edge},
        "test_window": {"seasons": test_seasons, "end": f"{TEST_END:%Y-%m-%d}",
                        "n": {"E0": len(t0), "E1": len(t1)},
                        "extended_e1_matches_after_end": int(len(t1_all) - len(t1))},
        "e0_published_check": {"dc_v1": got, "published": ref},
        "primary": {
            "Q1": {"gap_E0": r0["q1_rps_gap_model_minus_close"],
                   "gap_E1": r1["q1_rps_gap_model_minus_close"],
                   "difference_E1_minus_E0_joint": q1_joint,
                   "difference_E1_minus_E0_independent_sensitivity": q1_indep},
            "Q2": {"E1": q2, "E0": r0["q2_clv_model_minus_random_same_match"]},
            "Q3": {"E1": q3, "E0": r0["q3_blend"]},
        },
        "decisions_95": decisions,
        "decisions_983_robustness": robust,
        "leagues": {"E0": r0, "E1": r1},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")
    _print(summary)
    from premodel.experiments.championship_plot import render
    print(f"wrote {render(out_dir / 'summary.json', out_dir / 'championship.png')}")
    return summary


def _fmt(b: dict, digits: int = 4) -> str:
    f = f"{{:+.{digits}f}}"
    return (f"{f.format(b['mean'])}  95% [{f.format(b['ci95'][0])}, {f.format(b['ci95'][1])}]"
            f"  98.3% [{f.format(b['ci983'][0])}, {f.format(b['ci983'][1])}]")


def _print(s: dict):
    q = s["primary"]
    edge = " (grid edge)" if s["xi"]["E1_at_grid_edge"] else ""
    print(f"\nxi: E0 {s['xi']['E0']}, E1 {s['xi']['E1']}{edge};  "
          f"test n: {s['test_window']['n']}")
    print(f"Q1 RPS gap (model - close)  E0 {_fmt(q['Q1']['gap_E0'])}")
    print(f"                            E1 {_fmt(q['Q1']['gap_E1'])}")
    print(f"   E1 - E0 (joint)             {_fmt(q['Q1']['difference_E1_minus_E0_joint'])}")
    print(f"   E1 - E0 (independent)       "
          f"{_fmt(q['Q1']['difference_E1_minus_E0_independent_sensitivity'])}")
    for lg in ("E1", "E0"):
        print(f"Q2 CLV model - random, {lg}    {_fmt(q['Q2'][lg])}")
    for lg in ("E1", "E0"):
        b = q["Q3"][lg]
        print(f"Q3 blend {lg}: w={b['w']}  blend - open {_fmt(b['rps_diff_blend_minus_open'])}")
    print("decisions (95%):", s["decisions_95"])
    print("decisions (98.3%, robustness):", s["decisions_983_robustness"])
