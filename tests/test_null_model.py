"""
Integration test against the real generator. Slow (fits the generator on a full dataset), so it runs
only with LOBSURV_INTEGRATION=1 and needs the generator repository (LOB_GENERATOR_REPO) and its data.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("LOBSURV_INTEGRATION") != "1",
                                reason="set LOBSURV_INTEGRATION=1 to run against the generator repository")


def test_null_pvalues_on_real_and_injected_window():
    from lobsurv.null_model import GeneratorNull, generator_provenance
    from lobsurv.statistics import LOG_BID, LOG_DT, make_panel

    prov = generator_provenance()
    assert prov["commit"] and len(prov["file_sha256"]) >= 6
    null = GeneratorNull("BN_BTCUSDT")
    S, view = null.stream, null.view
    s = int(null.offsets["test"]) + 50_000
    fast = float(np.quantile(view.to_phys(S[:null.G["n_train"]])[:, LOG_DT], 0.05))
    panel = make_panel(view, null.has_dt, S[s - 1], fast_log_dt=fast)
    rng = np.random.default_rng(0)
    Y = null.simulate(s, 200, rng)
    assert Y.shape == (200, null.horizon, S.shape[1]) and np.isfinite(Y).all()

    real = S[s:s + null.horizon]
    p_real = null.mc_pvalues(s, real, panel, 200, rng, Y=Y)
    assert all(1 / 201 <= v <= 1 for v in p_real.values())

    # touch spoof: an order of 20x the median best-bid queue joins the bid at an event that leaves both best
    # prices unchanged and is cancelled 10 events later, while the prices still have not moved
    P = view.to_phys(real.copy())
    still = (np.abs(P[:, 0]) < 0.25) & (np.abs(np.diff(P[:, 1], prepend=P[0, 1])) < 0.5)
    t = next(i for i in range(1, len(P) - 11) if still[i:i + 11].all())
    V = 20 * np.median(np.exp(view.to_phys(S[:null.G["n_train"]])[:, LOG_BID]))
    P[t:t + 10, LOG_BID] = np.log(np.exp(P[t:t + 10, LOG_BID]) + V)
    p_inj = null.mc_pvalues(s, view.to_norm(P), panel, 200, rng, Y=Y)
    assert p_inj["queue_jump"] < p_real["queue_jump"] and p_inj["queue_jump"] <= 0.05
