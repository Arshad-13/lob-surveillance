"""
Window scores and discoveries for the two detectors compared in H2 (proposal Sections 3.3-3.4, 3.6):

  generator    per statistic, the Monte Carlo p-value of the observed window against continuations the frozen
               generator simulates from the window's own context (conditional null)
  uncond       per statistic, the p-value against the same statistic on a reference set of clean real windows
               (unconditional null: baseline B1)

Both combine the per-statistic p-values by the Cauchy combination and use score = -log(combined p); both are
then turned into conformal p-values against the scores of the same clean calibration windows, and
Benjamini-Hochberg picks the discoveries. Only the null differs between the two detectors.
"""

from typing import Dict, List

import numpy as np

from lobsurv.fdr import bh, cauchy_combine, conformal_pvalues, fdp_and_power, mc_pvalue
from lobsurv.statistics import window_stats


def stat_matrix(panel, windows: np.ndarray, names: List[str]) -> np.ndarray:
    """Window statistics [n_windows, n_stats] (windows [n, L, d] in the generator's z-scored space)."""
    st = window_stats(panel, windows)
    return np.stack([st[k] for k in names], axis=1)


def generator_pvalues(sim_stats: np.ndarray, obs_stats: np.ndarray, rng=None) -> np.ndarray:
    """
    sim_stats [M, n_stats] simulated from one context; obs_stats [n_variants, n_stats] -> p [n_variants, n_stats].
    Pass rng to break ties at random (the statistics are partly discrete).
    """
    return np.stack([mc_pvalue(obs_stats[:, j], np.broadcast_to(sim_stats[:, j], (len(obs_stats), len(sim_stats))),
                               rng=rng) for j in range(sim_stats.shape[1])], axis=1)


def uncond_pvalues(ref_stats: np.ndarray, obs_stats: np.ndarray, rng=None) -> np.ndarray:
    """ref_stats [n_ref, n_stats] from clean reference windows -> p [n_obs, n_stats]; rng as in generator_pvalues."""
    return np.stack([conformal_pvalues(ref_stats[:, j], obs_stats[:, j], rng=rng)
                     for j in range(ref_stats.shape[1])], axis=1)


def score(p: np.ndarray) -> np.ndarray:
    """-log of the Cauchy combination over statistics (last axis); larger = more unusual."""
    return -np.log(cauchy_combine(p, axis=-1))


def discoveries(cal_scores: np.ndarray, test_scores: np.ndarray, q: float) -> Dict[str, np.ndarray]:
    p = conformal_pvalues(cal_scores, test_scores)
    return {"p": p, "rejected": bh(p, q)}


def evaluate_mixture(cal_scores, clean_scores, inj_scores, q, prevalence, rng, reps=20) -> Dict[str, float]:
    """
    Power and false-discovery proportion of BH at level q on random mixtures of clean and injected windows at the
    given prevalence (injected and clean windows drawn from different positions), averaged over reps draws.
    """
    n = len(clean_scores)
    n_inj = max(1, int(round(prevalence * n)))
    fdp, power = [], []
    for _ in range(reps):
        idx = rng.permutation(n)
        inj_idx, clean_idx = idx[:n_inj], idx[n_inj:]
        test = np.concatenate([clean_scores[clean_idx], inj_scores[inj_idx]])
        is_null = np.r_[np.ones(len(clean_idx), bool), np.zeros(n_inj, bool)]
        f, pw = fdp_and_power(discoveries(cal_scores, test, q)["rejected"], is_null)
        fdp.append(f)
        power.append(pw)
    return {"fdr": float(np.mean(fdp)), "power": float(np.mean(power)), "n_test": n, "n_injected": n_inj}
