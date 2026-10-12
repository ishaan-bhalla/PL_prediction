"""Joint multi-league Dixon-Coles ("dc_joint"): one rating per team across leagues.

For a match in league L between home h and away a:

  log lam = gamma_L + b_L + att_h + dfn_a (+ move shifts)
  log mu  =           b_L + att_a + dfn_h (+ move shifts)

with the Dixon-Coles low-score correction tau(rho) (one rho for all leagues), the same
time decay, ridge and soft sum-to-zero on attack as dc_v1.

Identification. Leagues never meet, so they are linked only through teams that move
between them: a mover has one (att, dfn) fitted from its matches in both leagues, which
puts both leagues' teams on one scale. A league's strength is where its teams sit on
that scale; a separate per-league rating offset is not identifiable (absorbed by
non-movers' ratings). gamma_L (home advantage) is identified within each league; b_L
(goal environment) only through movers. Two global invariances remain,
(att + alpha, dfn + beta, b - alpha - beta), fixed by the sum-to-zero on attack over
ALL teams and b = 0 for the reference league (the highest league present). A league
with no movers linking it to the others has an arbitrary position on the scale (set
only by the ridge), so a team's rating is used in a league only if the league is
connected to the team's history; otherwise the per-league fallback applies.

Variant "move shift": a team's matches in its first season after moving league get
(att + d_att, dfn + d_dfn), one pair per direction (up / down), pooled across
boundaries. Optionally d_down is fixed at 0 (see the experiment note on identification).

With one league and no variant the parameter vector, objective and starting point are
exactly dc_v1's.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from premodel.models.dixon_coles import MAX_GOALS, RIDGE, score_matrix

LEAGUE_ORDER = ["E0", "E1", "E2"]  # highest first
NONE, UP, DOWN = 0, 1, 2           # move direction of a team in a match


@dataclass
class Layout:
    """Positions of each parameter block in theta (att, dfn, gamma_ref, rho come first,
    as in dc_v1, so a one-league fit has exactly dc_v1's vector)."""
    n: int
    leagues: list
    shifts: list  # directions with free shift parameters, e.g. [UP, DOWN] or [UP]

    @property
    def gamma(self):  # index of gamma for league j
        return [2 * self.n] + [2 * self.n + 2 + j - 1 for j in range(1, len(self.leagues))]

    @property
    def rho(self):
        return 2 * self.n + 1

    @property
    def b(self):  # index of b for league j (None for the reference league)
        start = 2 * self.n + 2 + len(self.leagues) - 1
        return [None] + [start + j - 1 for j in range(1, len(self.leagues))]

    @property
    def shift(self):  # direction -> (index of d_att, index of d_dfn)
        start = 2 * self.n + 2 + 2 * (len(self.leagues) - 1)
        return {d: (start + 2 * k, start + 2 * k + 1) for k, d in enumerate(self.shifts)}

    @property
    def size(self):
        return 2 * self.n + 2 + 2 * (len(self.leagues) - 1) + 2 * len(self.shifts)


@dataclass
class JointParams:
    teams: list
    att: np.ndarray
    dfn: np.ndarray
    leagues: list
    gamma: dict
    b: dict
    rho: float
    shift: dict = field(default_factory=dict)        # direction -> (d_att, d_dfn)
    last_league: dict = field(default_factory=dict)  # team -> league of its latest match
    played_in: dict = field(default_factory=dict)    # team -> set of leagues in the window
    component: dict = field(default_factory=dict)    # league -> connected-component id

    def index(self):
        return {t: i for i, t in enumerate(self.teams)}


def _unpack(theta, lay: Layout):
    n = lay.n
    gamma = np.array([theta[i] for i in lay.gamma])
    b = np.array([0.0 if i is None else theta[i] for i in lay.b])
    shift = np.zeros((3, 2))
    for d, (ia, idf) in lay.shift.items():
        shift[d] = theta[ia], theta[idf]
    return theta[:n], theta[n:2 * n], gamma, b, theta[lay.rho], shift


def _neg_ll_and_grad(theta, hi, ai, li, hs, as_, x, y, w, lay: Layout):
    att, dfn, gamma, b, rho, shift = _unpack(theta, lay)
    n = lay.n
    log_lam = gamma[li] + att[hi] + dfn[ai]
    log_mu = att[ai] + dfn[hi]
    if len(lay.leagues) > 1:
        log_lam = log_lam + b[li]
        log_mu = log_mu + b[li]
    if lay.shifts:
        log_lam = log_lam + shift[hs, 0] + shift[as_, 1]
        log_mu = log_mu + shift[as_, 0] + shift[hs, 1]
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    tau = np.ones_like(lam)
    dtau_dlam = np.zeros_like(lam)
    dtau_dmu = np.zeros_like(lam)
    dtau_drho = np.zeros_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    dtau_dlam[m00] = -mu[m00] * rho
    dtau_dmu[m00] = -lam[m00] * rho
    dtau_drho[m00] = -lam[m00] * mu[m00]
    tau[m01] = 1 + lam[m01] * rho
    dtau_dlam[m01] = rho
    dtau_drho[m01] = lam[m01]
    tau[m10] = 1 + mu[m10] * rho
    dtau_dmu[m10] = rho
    dtau_drho[m10] = mu[m10]
    tau[m11] = 1 - rho
    dtau_drho[m11] = -1.0
    if np.any(tau <= 0):
        return 1e10, np.zeros_like(theta)

    ll = w * (np.log(tau) + x * log_lam - lam + y * log_mu - mu)
    g_loglam = w * (x - lam + lam * dtau_dlam / tau)
    g_logmu = w * (y - mu + mu * dtau_dmu / tau)

    g = np.zeros_like(theta)
    g_att, g_dfn = np.zeros(n), np.zeros(n)
    np.add.at(g_att, hi, g_loglam)
    np.add.at(g_att, ai, g_logmu)
    np.add.at(g_dfn, ai, g_loglam)
    np.add.at(g_dfn, hi, g_logmu)
    g_gamma = np.bincount(li, weights=g_loglam, minlength=len(lay.leagues))
    for j, i in enumerate(lay.gamma):
        g[i] = -g_gamma[j]
    if len(lay.leagues) > 1:
        g_b = np.bincount(li, weights=g_loglam + g_logmu, minlength=len(lay.leagues))
        for j, i in enumerate(lay.b):
            if i is not None:
                g[i] = -g_b[j]
    for d, (ia, idf) in lay.shift.items():
        g[ia] = -(g_loglam[hs == d].sum() + g_logmu[as_ == d].sum())
        g[idf] = -(g_loglam[as_ == d].sum() + g_logmu[hs == d].sum())
    g[lay.rho] = -np.sum(w * dtau_drho / tau)

    # penalties: ridge on team ratings and move shifts + soft sum-to-zero on all attacks
    s = att.sum()
    pen = RIDGE * (att @ att + dfn @ dfn) + s * s
    g[:n] = -g_att + 2 * RIDGE * att + 2 * s
    g[n:2 * n] = -g_dfn + 2 * RIDGE * dfn
    for ia, idf in lay.shift.values():
        pen += RIDGE * (theta[ia] ** 2 + theta[idf] ** 2)
        g[ia] += 2 * RIDGE * theta[ia]
        g[idf] += 2 * RIDGE * theta[idf]
    return -ll.sum() + pen, g


def components(team_leagues: dict, leagues: list) -> dict:
    """league -> component id; leagues are joined when some team played in both."""
    parent = {lg: lg for lg in leagues}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ls in team_leagues.values():
        ls = sorted(ls)
        for other in ls[1:]:
            parent[find(other)] = find(ls[0])
    return {lg: find(lg) for lg in leagues}


def fit(home, away, league, hg, ag, days_ago, xi, move_home=None, move_away=None,
        shifts: tuple = (), init: JointParams | None = None) -> JointParams:
    """Weighted ML fit. `league` codes per match; `move_home/away` give each side's move
    direction (NONE / UP / DOWN) for the variant; `shifts` lists the free directions."""
    home, away, league = np.asarray(home), np.asarray(away), np.asarray(league)
    teams = sorted(set(home) | set(away))
    idx = {t: i for i, t in enumerate(teams)}
    present = [lg for lg in LEAGUE_ORDER if lg in set(league)] + \
        sorted(set(league) - set(LEAGUE_ORDER))
    lay = Layout(len(teams), present, list(shifts))
    lidx = {lg: j for j, lg in enumerate(present)}
    hi = np.array([idx[t] for t in home])
    ai = np.array([idx[t] for t in away])
    li = np.array([lidx[lg] for lg in league])
    n_obs = len(home)
    hs = np.zeros(n_obs, dtype=int) if move_home is None else np.asarray(move_home, dtype=int)
    as_ = np.zeros(n_obs, dtype=int) if move_away is None else np.asarray(move_away, dtype=int)
    w = np.exp(-xi * np.asarray(days_ago, dtype=float))

    theta0 = np.zeros(lay.size)
    for i in lay.gamma:
        theta0[i] = 0.25
    theta0[lay.rho] = -0.05
    if init is not None:  # warm start
        prev = init.index()
        for t, i in idx.items():
            if t in prev:
                theta0[i] = init.att[prev[t]]
                theta0[lay.n + i] = init.dfn[prev[t]]
        for j, lg in enumerate(present):
            if lg in init.gamma:
                theta0[lay.gamma[j]] = init.gamma[lg]
            if lay.b[j] is not None and lg in init.b:
                theta0[lay.b[j]] = init.b[lg]
        theta0[lay.rho] = init.rho
        for d, (ia, idf) in lay.shift.items():
            if d in init.shift:
                theta0[ia], theta0[idf] = init.shift[d]

    bounds = [(None, None)] * lay.size
    bounds[lay.rho] = (-0.3, 0.3)
    res = minimize(_neg_ll_and_grad, theta0, jac=True, method="L-BFGS-B",
                   args=(hi, ai, li, hs, as_, np.asarray(hg), np.asarray(ag), w, lay),
                   bounds=bounds)
    att, dfn, gamma, b, rho, shift = _unpack(res.x, lay)

    last, played = {}, {t: set() for t in teams}
    order = np.argsort(np.asarray(days_ago, dtype=float))[::-1]  # oldest first
    for k in order:
        for t in (home[k], away[k]):
            last[t] = league[k]
            played[t].add(league[k])
    return JointParams(
        teams, att.copy(), dfn.copy(), present,
        {lg: float(gamma[j]) for j, lg in enumerate(present)},
        {lg: float(b[j]) for j, lg in enumerate(present)},
        float(rho), {d: tuple(float(v) for v in shift[d]) for d in lay.shifts},
        last, played, components(played, present))


def league_fallback(p: JointParams, league: str, k: int = 3) -> tuple[float, float]:
    """Mean ratings of the k weakest teams whose latest match was in `league` (dc_v1's
    promoted_prior when there is one league)."""
    pool = [i for i, t in enumerate(p.teams) if p.last_league.get(t) == league]
    if not pool:
        pool = list(range(len(p.teams)))
    pool = np.array(pool)
    weakest = pool[np.argsort(p.att[pool] - p.dfn[pool], kind="stable")[:k]]
    return float(p.att[weakest].mean()), float(p.dfn[weakest].mean())


def team_rating(p: JointParams, team: str, league: str) -> tuple[float, float, bool]:
    """(att, dfn, from_history): the team's own rating if it has history in a league
    connected to `league`, else the league fallback."""
    i = p.index().get(team)
    if i is not None and league in p.component:
        comp = p.component[league]
        if any(p.component.get(lg) == comp for lg in p.played_in[team]):
            return float(p.att[i]), float(p.dfn[i]), True
    a, d = league_fallback(p, league) if league in p.component else (0.0, 0.0)
    return a, d, False


def predict_1x2(p: JointParams, home, away, league, move_home=None, move_away=None):
    """(n, 3) [P(home), P(draw), P(away)] and expected goals, for matches in `league`."""
    n = len(home)
    mh = np.zeros(n, dtype=int) if move_home is None else np.asarray(move_home, dtype=int)
    ma = np.zeros(n, dtype=int) if move_away is None else np.asarray(move_away, dtype=int)
    out, xg = np.zeros((n, 3)), np.zeros((n, 2))
    for k, (h, a) in enumerate(zip(home, away, strict=True)):
        lg = league[k] if not isinstance(league, str) else league
        ah, dh, _ = team_rating(p, h, lg)
        aa, da, _ = team_rating(p, a, lg)
        sh, sa = p.shift.get(mh[k], (0.0, 0.0)), p.shift.get(ma[k], (0.0, 0.0))
        g, b = p.gamma.get(lg, np.mean(list(p.gamma.values()))), p.b.get(lg, 0.0)
        lam = np.exp(g + b + ah + sh[0] + da + sa[1])
        mu = np.exp(b + aa + sa[0] + dh + sh[1])
        m = score_matrix(lam, mu, p.rho, MAX_GOALS)
        out[k] = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
        xg[k] = [lam, mu]
    return out, xg


def hessian_without_ridge(p: JointParams, home, away, league, hg, ag, days_ago, xi,
                          move_home=None, move_away=None, shifts=(), eps: float = 1e-5):
    """(H, layout): Hessian of the objective with the ridge switched off, at the fitted
    parameters, by central differences of the analytic gradient. Zero eigenvalues mean an
    unidentified direction; the sum-to-zero penalty and b_ref = 0 still apply."""
    global RIDGE
    lay = Layout(len(p.teams), p.leagues, list(shifts))
    idx = p.index()
    hi = np.array([idx[t] for t in home])
    ai = np.array([idx[t] for t in away])
    li = np.array([p.leagues.index(lg) for lg in league])
    n_obs = len(hi)
    hs = np.zeros(n_obs, dtype=int) if move_home is None else np.asarray(move_home, dtype=int)
    as_ = np.zeros(n_obs, dtype=int) if move_away is None else np.asarray(move_away, dtype=int)
    theta = np.zeros(lay.size)
    theta[:lay.n], theta[lay.n:2 * lay.n] = p.att, p.dfn
    for j, lg in enumerate(p.leagues):
        theta[lay.gamma[j]] = p.gamma[lg]
        if lay.b[j] is not None:
            theta[lay.b[j]] = p.b[lg]
    theta[lay.rho] = p.rho
    for d, (ia, idf) in lay.shift.items():
        theta[ia], theta[idf] = p.shift[d]
    w = np.exp(-xi * np.asarray(days_ago, dtype=float))
    args = (hi, ai, li, hs, as_, np.asarray(hg), np.asarray(ag), w, lay)
    saved, RIDGE = RIDGE, 0.0
    try:
        rows = [(_neg_ll_and_grad(theta + eps * e, *args)[1]
                 - _neg_ll_and_grad(theta - eps * e, *args)[1]) / (2 * eps)
                for e in np.eye(lay.size)]
    finally:
        RIDGE = saved
    h = np.array(rows)
    return (h + h.T) / 2, lay
