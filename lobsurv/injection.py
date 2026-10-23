"""
Labelled manipulation footprints, written into real level-1 event windows (proposal Section 3.5).

A window is an array [L, 5] of events in physical units, feature order as in the generator repo:
(dmid in ticks, spread in ticks, log best-bid size, log best-ask size, log inter-event seconds). Each row is
the book state after an event, except dmid, which is the change since the previous event. Injections add
events, as a real order would; the window is then cut back to its first L events.

Families (sizes in the symbol's own units, from its training data; see `Units`):
  spoof      an order of k x the median best queue joins the bid (or ask) at an event that leaves both prices
             unchanged, stays while that best price is unchanged, and is cancelled after tau events. If the
             price moves first, the order is no longer at the touch and drops out of the level-1 view without
             a cancel event.
  stuffing   n fast events, each a one-lot flicker of a best queue with prices unchanged.
  ignition   m fast one-tick mid moves in one direction, then m moves back after r further events.
Inserted events use the symbol's fast inter-event time (a low quantile of the training distribution).
All inserted events keep 2 dmid + d(spread) even, so quotes stay on the price lattice.
"""

from dataclasses import dataclass

import numpy as np

DMID, SPREAD, LOG_BID, LOG_ASK, LOG_DT = range(5)


@dataclass
class Units:
    median_bid: float        # median best-bid queue size (lots)
    median_ask: float
    fast_log_dt: float       # log inter-event time of inserted events

    @classmethod
    def from_train(cls, train_phys: np.ndarray, fast_quantile: float = 0.01):
        return cls(float(np.median(np.exp(train_phys[:, LOG_BID]))), float(np.median(np.exp(train_phys[:, LOG_ASK]))),
                   float(np.quantile(train_phys[:, LOG_DT], fast_quantile)))


def _still(prev_row: np.ndarray, units: Units) -> np.ndarray:
    """A new event that leaves both prices where they were (sizes as in prev_row)."""
    e = prev_row.copy()
    e[DMID] = 0.0
    e[LOG_DT] = units.fast_log_dt
    return e


def bid_move(P: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Change of the best bid at each row, in ticks: dmid - d(spread) / 2."""
    sp_prev = np.concatenate([[prev[SPREAD]], P[:-1, SPREAD]])
    return P[:, DMID] - (P[:, SPREAD] - sp_prev) / 2.0


def ask_move(P: np.ndarray, prev: np.ndarray) -> np.ndarray:
    sp_prev = np.concatenate([[prev[SPREAD]], P[:-1, SPREAD]])
    return P[:, DMID] + (P[:, SPREAD] - sp_prev) / 2.0


def spoof(P, prev, t0, side, k, tau, units: Units):
    """Returns (window [L, 5], dict with the inserted event indices in the returned window)."""
    L, col = len(P), (LOG_BID if side == "bid" else LOG_ASK)
    V = k * (units.median_bid if side == "bid" else units.median_ask)
    move = (bid_move if side == "bid" else ask_move)(P, prev)
    base = P[t0 - 1] if t0 > 0 else prev
    add = _still(base, units)
    add[col] = np.log(np.exp(add[col]) + V)
    rows, info = list(P[:t0]) + [add], {"add": t0, "cancel": None, "left_touch": None}
    active, held = True, 0
    for j in range(t0, L):
        r = P[j].copy()
        if active and abs(move[j]) > 1e-9:                # best price moved: the order is no longer at the touch
            active, info["left_touch"] = False, len(rows)
        if active:
            r[col] = np.log(np.exp(r[col]) + V)
            held += 1
        rows.append(r)
        if active and held == tau:
            cancel = _still(P[j], units)                  # back to the size without the order
            rows.append(cancel)
            info["cancel"], active = len(rows) - 1, False
    return np.array(rows[:L]), info


def stuffing(P, prev, t0, n, units: Units, rng):
    base = (P[t0 - 1] if t0 > 0 else prev).copy()
    burst = []
    for i in range(n):
        e = _still(base, units)
        col = LOG_BID if rng.random() < 0.5 else LOG_ASK
        if i % 2 == 0:                                   # one lot added, removed by the next event
            e[col] = np.log(np.exp(base[col]) + 1.0)
        burst.append(e)
    return np.array((list(P[:t0]) + burst + list(P[t0:]))[:len(P)]), {"start": t0, "n": n}


def ignition(P, prev, t0, m, r, direction, units: Units):
    base = (P[t0 - 1] if t0 > 0 else prev).copy()

    def moves(sign, row):
        out = []
        for _ in range(m):
            e = _still(row, units)
            e[DMID] = float(sign)                          # both best prices move one tick: spread unchanged
            out.append(e)
        return out

    rows = list(P[:t0]) + moves(direction, base)
    after = list(P[t0:])
    rows += after[:r]
    rows += moves(-direction, after[r - 1] if r > 0 and len(after) >= r else base)
    rows += after[r:]
    return np.array(rows[:len(P)]), {"start": t0, "m": m, "r": r, "direction": direction}
