import numpy as np

from lobsurv.statistics import LOG_BID, ROUNDTRIP_LAG, make_panel


class Identity:
    """Stand-in for the generator's DatasetView: windows are already in physical units."""

    def to_phys(self, Z):
        return np.asarray(Z, dtype=float)


def still_window(L=100):
    W = np.zeros((L, 5))
    W[:, 1] = 1.0                                       # spread 1 tick, no mid moves: every event is same-price
    W[:, 2:4] = np.log(10.0)
    return W


def roundtrip(W):
    panel = make_panel(Identity(), True, W[0], fast_log_dt=np.log(5e-4))
    return panel["queue_roundtrip"](W[None]).max()


def test_add_then_cancel_scores_and_add_alone_does_not():
    W = still_window()
    W[10:31, LOG_BID] = np.log(110.0)                   # 100 lots join the bid at event 10, leave at event 31
    assert np.isclose(roundtrip(W), np.log(11.0))
    A = still_window()
    A[10:, LOG_BID] = np.log(110.0)                     # added, never cancelled
    assert roundtrip(A) == 0.0


def test_pairs_further_apart_than_the_lag_do_not_count():
    W = still_window()
    W[5:5 + ROUNDTRIP_LAG + 1, LOG_BID] = np.log(110.0)  # cancel ROUNDTRIP_LAG + 1 events after the add
    assert roundtrip(W) == 0.0
