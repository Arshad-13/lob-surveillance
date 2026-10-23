import numpy as np

from lobsurv.injection import (DMID, LOG_ASK, LOG_BID, LOG_DT, SPREAD, Units, ask_move, bid_move, ignition, spoof,
                               stuffing)

UNITS = Units(median_bid=10.0, median_ask=10.0, fast_log_dt=np.log(1e-3))


def window(L=100, seed=0):
    """A synthetic level-1 window: mostly still events, occasional joint one-tick moves, spread 1."""
    rng = np.random.default_rng(seed)
    P = np.zeros((L, 5))
    P[:, DMID] = np.where(rng.random(L) < 0.1, rng.choice([-1.0, 1.0], L), 0.0)
    P[:, SPREAD] = 1.0
    P[:, LOG_BID] = np.log(rng.integers(1, 20, L))
    P[:, LOG_ASK] = np.log(rng.integers(1, 20, L))
    P[:, LOG_DT] = np.log(rng.exponential(0.05, L))
    prev = np.array([0.0, 1.0, np.log(5.0), np.log(5.0), np.log(0.1)])
    return P, prev


def on_lattice(P, prev):
    sp_prev = np.concatenate([[prev[SPREAD]], P[:-1, SPREAD]])
    return np.allclose((2 * P[:, DMID] + P[:, SPREAD] - sp_prev) % 2, 0)


def test_spoof_add_hold_cancel():
    P, prev = window()
    P[:, DMID] = 0.0                                   # prices never move: the order is cancelled after tau events
    Q, info = spoof(P, prev, t0=10, side="bid", k=5, tau=20, units=UNITS)
    assert Q.shape == P.shape and on_lattice(Q, prev)
    assert info["add"] == 10 and info["cancel"] == 10 + 20 + 1 and info["left_touch"] is None
    V = 5 * UNITS.median_bid
    np.testing.assert_allclose(np.exp(Q[11:31, LOG_BID]) - np.exp(P[10:30, LOG_BID]), V)
    np.testing.assert_allclose(Q[31, LOG_BID], P[29, LOG_BID])          # cancel restores the size
    assert Q[10, LOG_DT] == UNITS.fast_log_dt and Q[10, DMID] == 0


def test_spoof_leaves_touch_when_bid_moves():
    P, prev = window()
    P[:, DMID] = 0.0
    P[15, DMID] = 1.0                                  # bid and ask move up one tick at original event 15
    Q, info = spoof(P, prev, t0=10, side="bid", k=5, tau=50, units=UNITS)
    assert info["cancel"] is None and info["left_touch"] == 16
    np.testing.assert_allclose(Q[16:, LOG_BID], P[15:99, LOG_BID])      # no size added once off the touch


def test_stuffing_and_ignition_stay_on_lattice_and_keep_length():
    P, prev = window()
    rng = np.random.default_rng(1)
    Q, _ = stuffing(P, prev, t0=20, n=25, units=UNITS, rng=rng)
    assert Q.shape == P.shape and on_lattice(Q, prev)
    assert np.all(Q[20:45, DMID] == 0) and np.all(Q[20:45, LOG_DT] == UNITS.fast_log_dt)
    R, _ = ignition(P, prev, t0=30, m=3, r=10, direction=1, units=UNITS)
    assert R.shape == P.shape and on_lattice(R, prev)
    assert np.all(R[30:33, DMID] == 1) and np.all(R[43:46, DMID] == -1)
    np.testing.assert_allclose(np.cumsum(R[:46, DMID])[-1], np.cumsum(P[:40, DMID])[-1])  # back to the start


def test_moves_decompose_mid_and_spread():
    P, prev = window()
    np.testing.assert_allclose((bid_move(P, prev) + ask_move(P, prev)) / 2, P[:, DMID])
