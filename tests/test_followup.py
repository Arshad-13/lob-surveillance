import numpy as np

from lobsurv.followup import (auc, family_means, matched_resolution, rolling_conformal, size_matched_power,
                              size_matched_threshold)


def test_threshold_is_largest_clean_value_within_alpha_with_ties():
    clean = np.array([0.01, 0.01, 0.02, 0.05, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])   # discrete, with a tie
    assert size_matched_threshold(clean, 0.2) == 0.01                            # 0.02 would give 0.3
    assert size_matched_threshold(clean, 0.3) == 0.02
    assert size_matched_threshold(clean, 0.05) == 0.0                            # tie at the bottom exceeds alpha


def test_matched_power_counts_injected_at_or_below_threshold():
    clean = np.linspace(0.01, 1.0, 100)
    inj = np.array([0.005, 0.05, 0.051, 0.9])
    assert size_matched_power(clean, inj, 0.05) == 0.5


def test_matched_power_ignores_a_shift_of_the_clean_distribution():
    rng = np.random.default_rng(0)
    clean, inj = rng.random(5000), rng.random(5000) * 0.5
    loose, tight = clean * 0.5, inj * 0.5          # same ranking, every p-value halved
    assert abs(size_matched_power(clean, inj, 0.05) - size_matched_power(loose, tight, 0.05)) < 1e-12


def test_auc_values():
    assert auc(np.array([0.5, 0.6]), np.array([0.1, 0.2])) == 1.0
    assert auc(np.array([0.5, 0.6]), np.array([0.7, 0.8])) == 0.0
    assert auc(np.array([0.5]), np.array([0.5])) == 0.5


def test_matched_resolution_uses_all_continuations_only_when_the_variant_hit_the_floor():
    m = 200
    first = np.array([[[0.5, 1 / 201], [0.5, 0.4]]])        # window 0: variant 0 hit the floor, variant 1 did not
    allc = np.array([[[0.45, 1 / 1001], [0.48, 0.41]]])
    out = matched_resolution(first, allc, m)
    np.testing.assert_allclose(out[0, 0], allc[0, 0])
    np.testing.assert_allclose(out[0, 1], first[0, 1])


def test_rolling_conformal_uses_each_days_own_calibration():
    cal = {3: np.array([1.0, 2.0, 3.0]), 4: np.array([10.0, 20.0, 30.0])}
    scores = np.array([[2.5], [2.5]])
    p = rolling_conformal(cal, scores, np.array([3, 4]))
    np.testing.assert_allclose(p[:, 0], [2 / 4, 4 / 4])


def test_family_means():
    v = ["clean", "spoof:k=5", "spoof:k=10", "ignition:m=1"]
    assert family_means({"spoof:k=5": 0.2, "spoof:k=10": 0.4, "ignition:m=1": 0.1}, v) == {"spoof": 0.30000000000000004,
                                                                                            "ignition": 0.1}
