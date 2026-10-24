"""
Computations of the test-day confirmation (results/test_confirmation.md).

  part_days            non-overlapping window starts per day of one split, in the coordinates of a stream that may
                       concatenate several splits
  rolling_from_raw     V3 conformal p-values from raw per-statistic p-values: the first evaluation day against the
                       first calibration day, every later day against the clean copies of the day before
  bootstrap_pooled_diff  percentile interval of a mean over symbols of per-symbol window means, resampling windows
                       within each symbol
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from lobsurv.followup import rolling_conformal
from lobsurv.scoring import score


def part_days(offsets: Dict[str, int], segments: Sequence[int], n_stream: int, part: str, context: int,
              horizon: int) -> List[np.ndarray]:
    """
    offsets {split: stream offset} in stream order; segments: day starts of the whole stream (stream coordinates).
    Windows start `context` events into a day and end before the day does.
    """
    names = list(offsets)
    lo = offsets[part]
    i = names.index(part)
    hi = offsets[names[i + 1]] if i + 1 < len(names) else n_stream
    starts = sorted(int(s) for s in segments if lo <= s < hi)
    bounds = starts + [hi]
    return [np.arange(a + context, b - horizon, horizon) for a, b in zip(bounds[:-1], bounds[1:])]


def rolling_from_raw(cal_p: np.ndarray, eval_p: np.ndarray, day: np.ndarray) -> np.ndarray:
    """
    cal_p [n_cal, s]: per-statistic p-values of the first calibration day's clean windows; eval_p [n, variants, s]
    with variant 0 clean; day [n]: evaluation day index (0, 1, ...). Returns conformal p [n, variants].
    """
    sc = score(eval_p)
    days = sorted(int(d) for d in np.unique(day))
    cal = {days[0]: score(cal_p)}
    for prev, d in zip(days[:-1], days[1:]):
        cal[d] = sc[day == prev, 0]
    return rolling_conformal(cal, sc, day)


def bootstrap_pooled_diff(diffs: Sequence[np.ndarray], reps: int = 2000, seed: int = 0,
                          level: float = 0.95) -> Tuple[float, float, float]:
    """diffs: per symbol, one value per window. Returns (estimate, lower, upper) of the mean of per-symbol means."""
    rng = np.random.default_rng(seed)
    est = float(np.mean([d.mean() for d in diffs]))
    boot = np.zeros(reps)
    for d in diffs:
        boot += d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    boot /= len(diffs)
    lo, hi = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return est, float(lo), float(hi)
