"""Dixon-Coles (1997) model for football scores, with time-decay weighting.

  home goals X ~ Poisson(lam),  lam = exp(gamma + att[home] + dfn[away])
  away goals Y ~ Poisson(mu),   mu  = exp(att[away] + dfn[home])

`dfn` is defensive *weakness* (higher = concedes more). The low-score
correction tau(x, y; lam, mu, rho) adjusts the 0-0, 0-1, 1-0 and 1-1 cells,
and each past match is weighted by exp(-xi * days_ago).

Fitting maximises the weighted log-likelihood with an analytic gradient
(L-BFGS-B), plus a tiny ridge penalty and a sum-to-zero constraint on attack
ratings for identifiability.

Variant "dc_shrink" (experimental, opt-in via `fit(..., shrink=Shrinkage(...))`):
an extra penalty lam * sum_i[(att_i - m_att_i)^2 + (dfn_i - m_dfn_i)^2] pulls each
team toward a prior mean (0 unless given). One full-weight match carries roughly
1.3 units of information about a team's attack, so lam ~ 1.3k acts like k
pseudo-matches at the prior: large for a team with 5 matches, negligible for one
with 60. With shrink=None (the "dc_v1" baseline) nothing changes.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10
RIDGE = 1e-3


@dataclass
class DCParams:
    teams: list
    att: np.ndarray
    dfn: np.ndarray
    gamma: float
    rho: float

    def index(self):
        return {t: i for i, t in enumerate(self.teams)}


@dataclass(frozen=True)
class Shrinkage:
    """Penalty lam * sum((att - m_att)^2 + (dfn - m_dfn)^2); prior: team -> (m_att, m_dfn).

    Teams not in `prior` are shrunk toward (0, 0), or not penalised at all if only_prior
    (the "promoted-only" variant)."""
    lam: float
    prior: dict = field(default_factory=dict)
    only_prior: bool = False

    def weights_and_means(self, teams: list) -> tuple[np.ndarray, np.ndarray]:
        w = np.array([self.lam if (t in self.prior or not self.only_prior) else 0.0
                      for t in teams], dtype=float)
        means = np.array([self.prior.get(t, (0.0, 0.0)) for t in teams], dtype=float)
        return w, means


def _unpack(theta, n):
    return theta[:n], theta[n:2 * n], theta[2 * n], theta[2 * n + 1]


def _neg_ll_and_grad(theta, hi, ai, x, y, w, n, shrink_lam=0.0, m_att=None, m_dfn=None):
    att, dfn, gamma, rho = _unpack(theta, n)
    log_lam = gamma + att[hi] + dfn[ai]
    log_mu = att[ai] + dfn[hi]
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    # tau and its partial derivatives (zero outside the four low-score cells)
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
    # gradients of ll w.r.t. log_lam and log_mu (chain rule: d/dlog = lam * d/dlam)
    g_loglam = w * (x - lam + lam * dtau_dlam / tau)
    g_logmu = w * (y - mu + mu * dtau_dmu / tau)
    g_rho = np.sum(w * dtau_drho / tau)

    g_att = np.zeros(n)
    g_dfn = np.zeros(n)
    np.add.at(g_att, hi, g_loglam)
    np.add.at(g_att, ai, g_logmu)
    np.add.at(g_dfn, ai, g_loglam)
    np.add.at(g_dfn, hi, g_logmu)
    g_gamma = g_loglam.sum()

    # penalties: ridge on team ratings + soft sum-to-zero on attack
    s = att.sum()
    pen = RIDGE * (att @ att + dfn @ dfn) + s * s
    g_att_pen = 2 * RIDGE * att + 2 * s
    g_dfn_pen = 2 * RIDGE * dfn
    if m_att is not None:  # dc_shrink: pull toward per-team prior means (per-team weight)
        da, dd = att - m_att, dfn - m_dfn
        pen = pen + np.sum(shrink_lam * (da * da + dd * dd))
        g_att_pen = g_att_pen + 2 * shrink_lam * da
        g_dfn_pen = g_dfn_pen + 2 * shrink_lam * dd

    nll = -ll.sum() + pen
    grad = np.concatenate([-g_att + g_att_pen, -g_dfn + g_dfn_pen, [-g_gamma, -g_rho]])
    return nll, grad


def fit(home, away, hg, ag, days_ago, xi, init: DCParams | None = None,
        shrink: Shrinkage | None = None) -> DCParams:
    teams = sorted(set(home) | set(away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = np.array([idx[t] for t in home])
    ai = np.array([idx[t] for t in away])
    w = np.exp(-xi * np.asarray(days_ago, dtype=float))

    theta0 = np.zeros(2 * n + 2)
    theta0[2 * n] = 0.25  # home advantage
    theta0[2 * n + 1] = -0.05
    if init is not None:  # warm start from the previous fit
        prev = init.index()
        for t, i in idx.items():
            if t in prev:
                theta0[i] = init.att[prev[t]]
                theta0[n + i] = init.dfn[prev[t]]
        theta0[2 * n], theta0[2 * n + 1] = init.gamma, init.rho

    bounds = [(None, None)] * (2 * n + 1) + [(-0.3, 0.3)]
    args = (hi, ai, np.asarray(hg), np.asarray(ag), w, n)
    if shrink is not None:
        lam_w, means = shrink.weights_and_means(teams)
        if lam_w.any():  # all-zero weights: exactly the dc_v1 objective
            args += (lam_w, means[:, 0], means[:, 1])
    res = minimize(_neg_ll_and_grad, theta0, jac=True, method="L-BFGS-B",
                   args=args, bounds=bounds)
    att, dfn, gamma, rho = _unpack(res.x, n)
    return DCParams(teams, att.copy(), dfn.copy(), float(gamma), float(rho))


def team_ratings(p: DCParams, team: str, fallback: tuple[float, float]):
    i = p.index().get(team)
    return (p.att[i], p.dfn[i]) if i is not None else fallback


def promoted_prior(p: DCParams, k: int = 3) -> tuple[float, float]:
    """Ratings for a team with no data: mean of the k weakest fitted teams."""
    strength = p.att - p.dfn
    weakest = np.argsort(strength)[:k]
    return float(p.att[weakest].mean()), float(p.dfn[weakest].mean())


def score_matrix(lam, mu, rho, max_goals=MAX_GOALS):
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def predict_1x2(p: DCParams, home, away, prior):
    """Return (n, 3) array of [P(home), P(draw), P(away)] plus expected goals."""
    out = np.zeros((len(home), 3))
    xg = np.zeros((len(home), 2))
    for k, (h, a) in enumerate(zip(home, away, strict=True)):
        ah, dh = team_ratings(p, h, prior)
        aa, da = team_ratings(p, a, prior)
        lam = np.exp(p.gamma + ah + da)
        mu = np.exp(aa + dh)
        m = score_matrix(lam, mu, p.rho)
        out[k] = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
        xg[k] = [lam, mu]
    return out, xg
