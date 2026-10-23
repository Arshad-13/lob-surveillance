"""
Panel of window statistics for level-1 order-book paths.

Paths are arrays [B, L, d] in the generator's z-scored feature space, feature order
(dmid, spread, log_bid, log_ask[, log_dt]) as in the generator repo's shared_resources/lob_data.py.
Each statistic is a running score [B, L]; the window statistic is its maximum over the horizon,
which is exactly the event {running max >= level} that multilevel splitting (ML/ctesn/splitting.py)
estimates. Statistics that compare against the book at the end of the context (queue_swing,
queue_jump) close over that last context event.

Which footprint each statistic targets:
  range, n_moves     price manipulation (momentum ignition, marking), also ordinary volatility
  max_spread         liquidity withdrawal
  queue_swing        touch spoofing: a large non-bona-fide order shifts best-queue imbalance
                     (the mechanism of Cartea, Jaimungal and Wang, Appl. Math. Finance 2020)
  queue_jump         the add and the cancel of such an order: one-event jumps in a best queue at
                     events that leave both best prices unchanged
  n_fast             quote stuffing: bursts of events with tiny inter-event times
"""

from typing import Callable, Dict

import numpy as np

DMID, SPREAD, LOG_BID, LOG_ASK, LOG_DT = range(5)


def make_panel(view, has_dt: bool, last_context_event: np.ndarray, fast_log_dt: float = None) -> Dict[str, Callable]:
    """
    view: object with to_phys(Z) (the generator repo's DatasetView).
    last_context_event: the final context event, z-scored, shape [d].
    fast_log_dt: log inter-event time below which an event counts as fast (set from training data,
    e.g. its 5th percentile); required for n_fast, which is omitted when the data has no timing.
    """
    ref = view.to_phys(np.asarray(last_context_event, dtype=float)[None])[0]
    imb_ref = ref[LOG_BID] - ref[LOG_ASK]

    def phys(Y):
        return view.to_phys(Y)

    def cum(Y):
        return np.cumsum(phys(Y)[..., DMID], axis=1)

    def queue_jump(Y):
        # Only events that leave both best prices where they were: when a price level changes, a new queue
        # takes over and jumps of 20x or more are routine, so they would drown an added or cancelled order.
        P = phys(Y)
        prev = np.concatenate([np.broadcast_to(ref, (len(P), 1, P.shape[-1])), P[:, :-1]], axis=1)
        same = (np.abs(P[..., DMID]) < 0.25) & (np.abs(P[..., SPREAD] - prev[..., SPREAD]) < 0.5)
        return np.where(same, np.abs(P[..., [LOG_BID, LOG_ASK]] - prev[..., [LOG_BID, LOG_ASK]]).max(-1), 0.0)

    panel = {
        "range": lambda Y: np.maximum.accumulate(cum(Y), axis=1) - np.minimum.accumulate(cum(Y), axis=1),
        "n_moves": lambda Y: np.cumsum(np.abs(phys(Y)[..., DMID]) > 0.25, axis=1).astype(float),
        "max_spread": lambda Y: phys(Y)[..., SPREAD],
        "queue_swing": lambda Y: np.abs(phys(Y)[..., LOG_BID] - phys(Y)[..., LOG_ASK] - imb_ref),
        "queue_jump": queue_jump,
    }
    if has_dt:
        if fast_log_dt is None:
            raise ValueError("fast_log_dt is required when the data has inter-event times")
        panel["n_fast"] = lambda Y: np.cumsum(phys(Y)[..., LOG_DT] < fast_log_dt, axis=1).astype(float)
    return panel


def window_stats(panel: Dict[str, Callable], Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Window statistic (running maximum over the horizon) of every panel member: name -> [B]."""
    return {k: f(Y).max(axis=1) for k, f in panel.items()}
