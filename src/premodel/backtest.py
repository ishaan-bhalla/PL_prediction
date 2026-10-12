"""Walk-forward backtest of Dixon-Coles against the betting market.

Before every match date the model is refitted on matches played strictly
before that date (last LOOKBACK_DAYS, time-decayed), then predicts that day's
matches. No future information is used.
"""
import json

import numpy as np
import pandas as pd

from premodel import config
from premodel.data import devig, load_matches
from premodel.metrics import calibration_table, rps, score_all
from premodel.models import dixon_coles as dc

PROB_COLS = ["p_h", "p_d", "p_a"]
OPEN_COLS = ["open_h", "open_d", "open_a"]
CLOSE_COLS = ["close_h", "close_d", "close_a"]


def walk_forward(m: pd.DataFrame, xi: float, seasons: list[str]) -> pd.DataFrame:
    target = m[m.season.isin(seasons)]
    preds, params = [], None
    for date, day in target.groupby("date", sort=True):
        train = m[(m.date < date) & (m.date >= date - pd.Timedelta(days=config.LOOKBACK_DAYS))]
        params = dc.fit(train.home.to_numpy(), train.away.to_numpy(),
                        train.hg.to_numpy(), train.ag.to_numpy(),
                        (date - train.date).dt.days.to_numpy(), xi, init=params)
        p, xg = dc.predict_1x2(params, day.home.to_numpy(), day.away.to_numpy(),
                               dc.promoted_prior(params))
        out = day.copy()
        out[PROB_COLS] = p
        out[["xg_h", "xg_a"]] = xg
        preds.append(out)
    return pd.concat(preds)


def tune_xi(m: pd.DataFrame) -> tuple[float, list[dict]]:
    rows = []
    for xi in config.XI_GRID:
        v = walk_forward(m, xi, config.VALID_SEASONS)
        sc = score_all(v[PROB_COLS].to_numpy(), v.result.to_numpy())
        rows.append({"xi": xi, "half_life_days": float(np.log(2) / xi) if xi else None, **sc})
        print(f"  xi={xi:<7} rps={sc['rps']:.5f}  log_loss={sc['log_loss']:.5f}")
    best = min(rows, key=lambda r: r["rps"])["xi"]
    return best, rows


def simulate_bets(df: pd.DataFrame, threshold: float, rng: np.random.Generator) -> dict:
    """Flat 1-unit stakes at opening odds; at most one bet per match."""
    p = df[PROB_COLS].to_numpy()
    open_odds = df[OPEN_COLS].to_numpy()
    close_fair = devig(df[CLOSE_COLS].to_numpy())
    edge = p * open_odds - 1
    pick = edge.argmax(axis=1)
    rows = np.arange(len(df))
    take = edge[rows, pick] > threshold
    if not take.any():
        return {"threshold": threshold, "n_bets": 0}
    sel, pk = rows[take], pick[take]
    odds = open_odds[sel, pk]
    pnl = np.where(df["result"].to_numpy()[sel] == pk, odds - 1, -1.0)
    clv = odds * close_fair[sel, pk] - 1  # bet EV if the closing line is the truth
    boot = rng.choice(pnl, size=(config.BOOTSTRAP_SAMPLES, len(pnl))).mean(axis=1)
    return {
        "threshold": threshold,
        "n_bets": int(take.sum()),
        "roi": float(pnl.mean()),
        "roi_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "mean_clv": float(clv.mean()),
        "share_positive_clv": float((clv > 0).mean()),
        "bets_by_outcome": {k: int((pk == i).sum()) for i, k in enumerate("HDA")},
        "pnl_series": np.cumsum(pnl).round(4).tolist(),
    }


def random_bet_baseline(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Bet a uniformly random outcome on every match — CLV here is just the margin."""
    k = rng.integers(0, 3, len(df))
    r = np.arange(len(df))
    odds = df[OPEN_COLS].to_numpy()[r, k]
    pnl = np.where(df.result.to_numpy() == k, odds - 1, -1.0)
    clv = odds * devig(df[CLOSE_COLS].to_numpy())[r, k] - 1
    return {"roi": float(pnl.mean()), "mean_clv": float(clv.mean()),
            "pnl_series": np.cumsum(pnl).round(4).tolist()}


def run() -> dict:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    m = load_matches()
    test_seasons = sorted(s for s in m.season.unique() if s >= config.TEST_START)

    print("Tuning time decay on validation seasons ...")
    best_xi, tuning = tune_xi(m)
    print(f"  -> chosen xi = {best_xi}")

    print("Walk-forward on test seasons ...")
    t = walk_forward(m, best_xi, test_seasons)
    t.to_csv(config.PROCESSED_DIR / "test_predictions.csv", index=False)
    y = t.result.to_numpy()
    p_model = t[PROB_COLS].to_numpy()
    p_close = devig(t[CLOSE_COLS].to_numpy())
    base = m[m.season < config.TEST_START].result.value_counts(normalize=True).sort_index()

    benchmarks = {
        "Naive base rates": score_all(np.tile(base.to_numpy(), (len(t), 1)), y),
        "Dixon-Coles": score_all(p_model, y),
        "Market (opening, devigged)": score_all(devig(t[OPEN_COLS].to_numpy()), y),
        "Market (closing, devigged)": score_all(p_close, y),
    }
    per_season = [{
        "season": s, "n": len(g),
        "rps_model": rps(g[PROB_COLS].to_numpy(), g.result.to_numpy()),
        "rps_close": rps(devig(g[CLOSE_COLS].to_numpy()), g.result.to_numpy()),
    } for s, g in t.groupby("season")]

    rng = np.random.default_rng(config.SEED)
    summary = {
        "n_test_matches": int(len(t)),
        "test_seasons": test_seasons,
        "chosen_xi": best_xi,
        "xi_tuning": tuning,
        "benchmarks": benchmarks,
        "per_season": per_season,
        "betting": [simulate_bets(t, th, rng) for th in config.EDGE_THRESHOLDS],
        "random_bet_baseline": random_bet_baseline(t, np.random.default_rng(0)),
        "calibration_home_model": calibration_table(p_model, y, 0),
        "calibration_home_close": calibration_table(p_close, y, 0),
        "close_source_counts": t.close_src.value_counts().to_dict(),
    }
    (config.REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nForecast accuracy (lower is better)")
    for k, v in benchmarks.items():
        print(f"  {k:<28} rps={v['rps']:.4f}  brier={v['brier']:.4f}  log_loss={v['log_loss']:.4f}")
    print("\nBetting at opening odds (flat stakes)")
    for b in summary["betting"]:
        if b["n_bets"]:
            lo, hi = b["roi_ci95"]
            print(f"  edge>{b['threshold']:.2f}  bets={b['n_bets']:>5}  roi={b['roi']:+.1%} "
                  f"[{lo:+.1%}, {hi:+.1%}]  clv={b['mean_clv']:+.2%}")
    rb = summary["random_bet_baseline"]
    print(f"  random-bet baseline       roi={rb['roi']:+.1%}  clv={rb['mean_clv']:+.2%}")
    return summary
