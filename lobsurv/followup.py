"""
Computations of the Stage-0 follow-up (results/stage0_followup.md).

  size_matched_threshold   largest clean p-value whose share among the clean windows is <= alpha (0 if none)
  size_matched_power       share of injected windows at or below that threshold
  auc                      P(injected p < clean p) + P(tie) / 2, over all clean / injected pairs
  matched_resolution       per window-variant generator p-values under the calibration set's refinement rule:
                           all continuations only if that variant itself reached the floor at the first m
  rolling_conformal        conformal p-values of each evaluation day against the clean windows of the day before

Conformal p-values are as in lobsurv.fdr (smaller = more unusual).
"""

from typing import Dict, Sequence

import numpy as np

from lobsurv.fdr import conformal_pvalues
from lobsurv.scoring import score


def size_matched_threshold(p_clean: np.ndarray, alpha: float) -> float:
    u = np.unique(p_clean)
    share = np.searchsorted(np.sort(p_clean), u, side="right") / len(p_clean)
    ok = u[share <= alpha]
    return float(ok.max()) if len(ok) else 0.0


def size_matched_power(p_clean: np.ndarray, p_inj: np.ndarray, alpha: float) -> float:
    return float(np.mean(p_inj <= size_matched_threshold(p_clean, alpha)))


def auc(p_clean: np.ndarray, p_inj: np.ndarray) -> float:
    """Mann-Whitney form: the probability that an injected window looks more unusual than a clean one."""
    c = np.sort(p_clean)
    less = len(c) - np.searchsorted(c, p_inj, side="right")      # clean windows with larger p
    ties = np.searchsorted(c, p_inj, side="right") - np.searchsorted(c, p_inj, side="left")
    return float(np.mean((less + 0.5 * ties) / len(c)))


def matched_resolution(g_first: np.ndarray, g_all: np.ndarray, m: int) -> np.ndarray:
    """g_first, g_all [..., n_stats]: generator p-values from the first m and from all continuations."""
    hit = (g_first <= 2.0 / (m + 1)).any(axis=-1, keepdims=True)
    return np.where(hit, g_all, g_first)


def rolling_conformal(cal_by_day: Dict[int, np.ndarray], scores: np.ndarray, day: np.ndarray) -> np.ndarray:
    """
    cal_by_day {d: clean calibration scores used for evaluation day d}; scores [n, variants]; day [n].
    Returns conformal p-values [n, variants].
    """
    p = np.empty_like(scores, dtype=float)
    for d in np.unique(day):
        rows = day == d
        p[rows] = np.stack([conformal_pvalues(cal_by_day[int(d)], scores[rows, v]) for v in range(scores.shape[1])],
                           axis=1)
    return p


def family_means(values: Dict[str, float], variants: Sequence[str]) -> Dict[str, float]:
    """Average a {variant: value} map over each family's strength grid ('family:params' labels)."""
    fams = dict.fromkeys(v.split(":")[0] for v in variants if v != "clean")
    return {f: float(np.mean([values[v] for v in variants if v.startswith(f + ":")])) for f in fams}


def conformal_from_raw(cal_p: np.ndarray, eval_p: np.ndarray) -> np.ndarray:
    """Per-statistic p-values (cal [n_cal, s], eval [n, variants, s]) -> conformal p [n, variants]."""
    cal, sc = score(cal_p), score(eval_p)
    return np.stack([conformal_pvalues(cal, sc[:, v]) for v in range(sc.shape[1])], axis=1)
