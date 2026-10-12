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
import numpy as np
import pandas as pd

from premodel import config
from premodel.models import dixon_coles as dc

PROB_COLS = ["p_h", "p_d", "p_a"]
N_RELEGATED = 3


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
                 priors: dict | None = None, zero_prior: bool = False) -> pd.DataFrame:
    """backtest.walk_forward, optionally with shrinkage (lam=None reproduces dc_v1).

    Promoted teams are shrunk toward priors[season] (or 0 if zero_prior); others toward 0.
    Teams with no matches in the window still use dc.promoted_prior() for prediction.
    """
    target = m[m.season.isin(seasons)]
    preds, params = [], None
    for date, day in target.groupby("date", sort=True):
        shrink = None
        if lam is not None:
            season = day.season.iloc[0]
            mean = (0.0, 0.0) if zero_prior else priors[season]
            shrink = dc.Shrinkage(lam, {t: mean for t in promoted_teams(m, season, date)})
        params = fit_before(m, date, xi, init=params, shrink=shrink)
        p, xg = dc.predict_1x2(params, day.home.to_numpy(), day.away.to_numpy(),
                               dc.promoted_prior(params))
        out = day.copy()
        out[PROB_COLS] = p
        out[["xg_h", "xg_a"]] = xg
        preds.append(out)
    return pd.concat(preds)
