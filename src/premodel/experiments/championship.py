"""Experiment: is the Championship (E1) betting market softer than the Premier League's?

The model is held fixed (dc_v1, the same walk-forward code as the E0 backtest); only
the market changes. Decision rules are fixed in DECISION_RULES before any E1 result
is computed and are written into summary.json.
"""
import pandas as pd

from premodel import config
from premodel.data import load_matches

OUT_DIR = config.REPORTS_DIR / "experiments" / "championship"
SEASON_MATCHES = {"E0": 380, "E1": 552}
# Same calendar window as the published E0 test set (reports/summary.json).
TEST_END = pd.Timestamp("2026-09-20")


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
