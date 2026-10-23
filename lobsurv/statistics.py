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
  queue_roundtrip    the add-then-cancel signature: a same-price jump up in a best queue followed within
                     ROUNDTRIP_LAG events by a jump down in the same queue (score: the smaller of the two)
  n_fast             quote stuffing: bursts of events with tiny inter-event times
"""

from typing import Callable, Dict

import numpy as np

DMID, SPREAD, LOG_BID, LOG_ASK, LOG_DT = range(5)
ROUNDTRIP_LAG = 50


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

    def queue_roundtrip(Y):
        # an add followed by a cancel: at events leaving both prices unchanged, a jump up in a best queue and,
        # within ROUNDTRIP_LAG events, a jump down in the same queue; the score is the smaller of the two
        P = phys(Y)
        prev = np.concatenate([np.broadcast_to(ref, (len(P), 1, P.shape[-1])), P[:, :-1]], axis=1)
        same = (np.abs(P[..., DMID]) < 0.25) & (np.abs(P[..., SPREAD] - prev[..., SPREAD]) < 0.5)
        c = np.where(same[..., None], P[..., [LOG_BID, LOG_ASK]] - prev[..., [LOG_BID, LOG_ASK]], 0.0)
        up, down = np.maximum(c, 0.0), np.maximum(-c, 0.0)
        best_up = np.zeros_like(up)                      # largest up-jump in the previous ROUNDTRIP_LAG events
        for d in range(1, ROUNDTRIP_LAG + 1):
            best_up[:, d:] = np.maximum(best_up[:, d:], up[:, :-d])
        return np.minimum(best_up, down).max(-1)

    panel = {
        "range": lambda Y: np.maximum.accumulate(cum(Y), axis=1) - np.minimum.accumulate(cum(Y), axis=1),
        "n_moves": lambda Y: np.cumsum(np.abs(phys(Y)[..., DMID]) > 0.25, axis=1).astype(float),
        "max_spread": lambda Y: phys(Y)[..., SPREAD],
        "queue_swing": lambda Y: np.abs(phys(Y)[..., LOG_BID] - phys(Y)[..., LOG_ASK] - imb_ref),
        "queue_jump": queue_jump,
        "queue_roundtrip": queue_roundtrip,
    }
    if has_dt:
        if fast_log_dt is None:
            raise ValueError("fast_log_dt is required when the data has inter-event times")
        # "at or below": with millisecond timestamps the low quantiles of the inter-event time are the floor (two
        # events in the same millisecond), and a strict inequality would never count anything
        panel["n_fast"] = lambda Y: np.cumsum(phys(Y)[..., LOG_DT] <= fast_log_dt + 1e-9, axis=1).astype(float)
    return panel


def window_stats(panel: Dict[str, Callable], Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Window statistic (running maximum over the horizon) of every panel member: name -> [B]."""
    return {k: f(Y).max(axis=1) for k, f in panel.items()}
