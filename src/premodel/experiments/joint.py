"""Experiment: a joint multi-league model (dc_joint) vs dc_v1 in E0 and E1.

One rating per team across E0, E1 and E2 (League One), so a promoted or relegated team
carries its history across leagues. Decision rules are fixed in DECISION_RULES before
any result is computed and written into summary.json.
"""
import numpy as np
import pandas as pd

from premodel import config
from premodel.data import load_matches
from premodel.models import dixon_coles_joint as dj

OUT_DIR = config.REPORTS_DIR / "experiments" / "joint"
LEAGUES = dj.LEAGUE_ORDER                 # all fitted on
PREDICTED = ["E0", "E1"]                  # evaluated
RANK = {lg: r for r, lg in enumerate(LEAGUES)}
P = ["p_h", "p_d", "p_a"]
VARIANTS = {"plain": (), "move_shift": (dj.UP,)}  # move_shift: d_down fixed at 0


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
