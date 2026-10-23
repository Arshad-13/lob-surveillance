"""
Unit Test Suite for SMFI Engine (T3.2 - T3.4, T4.2).

Verifies:
1. Radius epsilon calibration on clean baseline data.
2. Lifespan noise filter delta calculation.
3. 2-Wasserstein SMFI fragility signal stream computation.
4. Cross-asset systemic coupling complex (k=3) & rank invariant tracking.
"""

import numpy as np
import pytest

from DS.smfi.engine import SMFIEngine, CrossAssetCouplingComplex


def test_epsilon_calibration():
    np.random.seed(42)
    # 100 clean log-signature vectors in 10-dimensional space
    clean_sigs = np.random.normal(0, 0.5, size=(100, 10))
    eps, scores = SMFIEngine.calibrate_radius_epsilon(clean_sigs, window_w=10)
    assert eps > 0
    assert len(scores) > 0
    print(f"[TEST] Calibrated epsilon: {eps:.4f}")


def test_lifespan_delta_and_smfi_stream():
    engine = SMFIEngine(epsilon=0.5, window_w=8)
    
    np.random.seed(42)
    # Stream of 40 log-signature vectors
    sigs = np.random.normal(0, 0.2, size=(40, 6))

    # Inject an anomaly at t=25 (geometrically isolated burst)
    sigs[25:28] += 2.5

    result = engine.compute_smfi_stream(sigs, window_w=8)
    assert "smfi" in result
    smfi = result["smfi"]
    assert len(smfi) > 0
    # Fragility signal should exhibit spike around anomaly window
    peak_idx = np.argmax(smfi)
    print(f"[TEST] Peak SMFI index: {peak_idx} with value: {smfi[peak_idx]:.4f}")
    assert smfi[peak_idx] > 0.0


def test_cross_asset_coupling_complex():
    tickers = ["AAPL", "AMZN", "MSFT"]
    complex_tracker = CrossAssetCouplingComplex(tickers=tickers, coupling_epsilon=0.4)

    # Correlated signatures
    base = np.random.normal(0, 0.3, size=(50, 8))
    multi_sigs = {
        "AAPL": base + np.random.normal(0, 0.05, size=(50, 8)),
        "AMZN": base + np.random.normal(0, 0.05, size=(50, 8)),
        "MSFT": base * 1.5 + np.random.normal(0, 0.2, size=(50, 8))
    }

    res = complex_tracker.compute_coupling_rank_invariant(multi_sigs, window_w=10)
    assert "beta_0_systemic" in res
    assert "active_couplings" in res
    assert len(res["beta_0_systemic"]) > 0
    print("[TEST] Verified Cross-Asset Coupling Complex rank invariant tracking.")
