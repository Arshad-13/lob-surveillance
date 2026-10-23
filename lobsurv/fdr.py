"""
p-values and false-discovery control for window-level surveillance.

Convention throughout: a larger score means a more unusual window.

  mc_pvalue           Monte Carlo tail p-value of an observed statistic against simulated draws,
                      (1 + #{sim >= obs}) / (1 + n): valid (super-uniform) whenever the draws and
                      the observation are exchangeable under the null.
  conformal_pvalues   split-conformal p-values of test scores against clean calibration scores
                      (Bates, Candes, Lei, Romano and Sesia, Ann. Statist. 51(1), 2023). Valid in
                      finite samples if calibration and null test windows are exchangeable, whatever
                      model produced the scores; the model only decides power.
  cauchy_combine      combines dependent p-values (Liu and Xie, JASA 115(529), 2020); the tail of
                      the combined statistic is insensitive to the dependence among the inputs.
  bh, by              Benjamini-Hochberg (1995) and Benjamini-Yekutieli (2001) step-up procedures.
"""

import numpy as np


def mc_pvalue(observed, simulated, axis=-1):
    """observed [...], simulated [..., n] (draws along `axis`) -> p [...]."""
    sim = np.moveaxis(np.asarray(simulated, dtype=float), axis, -1)
    obs = np.asarray(observed, dtype=float)[..., None]
    return (1.0 + (sim >= obs).sum(-1)) / (1.0 + sim.shape[-1])


def conformal_pvalues(cal_scores, test_scores):
    """p_j = (1 + #{i : cal_i >= test_j}) / (n_cal + 1)."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    test = np.asarray(test_scores, dtype=float)
    n_ge = len(cal) - np.searchsorted(cal, test, side="left")
    return (1.0 + n_ge) / (len(cal) + 1.0)


def cauchy_combine(p, axis=-1, weights=None):
    """Cauchy combination of p-values along `axis`; weights default to equal and are normalised to sum to 1."""
    p = np.moveaxis(np.asarray(p, dtype=float), axis, -1)
    w = np.full(p.shape[-1], 1.0 / p.shape[-1]) if weights is None else np.asarray(weights, float) / np.sum(weights)
    p = np.clip(p, 1e-300, 1.0 - 1e-16)
    # tan((0.5 - p) pi) loses precision for tiny p; 1 / (p pi) is its leading term there
    t = np.where(p < 1e-15, 1.0 / (p * np.pi), np.tan((0.5 - p) * np.pi))
    T = (t * w).sum(-1)
    return np.where(T > 1e15, 1.0 / (T * np.pi), 0.5 - np.arctan(T) / np.pi)


def bh(p, q):
    """Benjamini-Hochberg at level q. Returns a boolean rejection mask with p's shape."""
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    m = len(flat)
    order = np.argsort(flat, kind="stable")
    ok = np.nonzero(flat[order] <= q * np.arange(1, m + 1) / m)[0]
    rej = np.zeros(m, dtype=bool)
    if len(ok):
        rej[order[:ok[-1] + 1]] = True
    return rej.reshape(p.shape)


def by(p, q):
    """Benjamini-Yekutieli: BH at q / H_m, valid under arbitrary dependence (e.g. overlapping windows)."""
    m = np.asarray(p).size
    return bh(p, q / np.sum(1.0 / np.arange(1, m + 1)))


def fdp_and_power(rejected, is_null):
    """Empirical false-discovery proportion and power of one rejection set (ground truth known)."""
    rejected, is_null = np.asarray(rejected, bool), np.asarray(is_null, bool)
    n_rej = rejected.sum()
    fdp = float((rejected & is_null).sum() / n_rej) if n_rej else 0.0
    n_alt = (~is_null).sum()
    power = float((rejected & ~is_null).sum() / n_alt) if n_alt else float("nan")
    return fdp, power
