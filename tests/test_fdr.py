import numpy as np

from lobsurv.fdr import bh, by, cauchy_combine, conformal_pvalues, fdp_and_power, mc_pvalue


def test_mc_pvalue_counts_ties_and_observation():
    assert mc_pvalue(3.0, [1.0, 3.0, 5.0]) == (1 + 2) / 4
    assert mc_pvalue(9.0, np.zeros(99)) == 1 / 100


def test_conformal_pvalues_are_valid_under_exchangeability():
    rng = np.random.default_rng(0)
    alpha, reps, hits = 0.05, 4000, 0
    for _ in range(reps):
        x = rng.standard_normal(201)
        hits += conformal_pvalues(x[:200], x[200:])[0] <= alpha
    assert hits / reps <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps)


def test_conformal_pvalue_formula():
    p = conformal_pvalues([1.0, 2.0, 3.0], [0.0, 2.0, 10.0])
    np.testing.assert_allclose(p, [4 / 4, 3 / 4, 1 / 4])


def test_cauchy_combination_fixed_points():
    np.testing.assert_allclose(cauchy_combine([0.03]), 0.03)
    np.testing.assert_allclose(cauchy_combine([0.2, 0.2, 0.2]), 0.2)
    assert cauchy_combine([1e-20, 0.9]) < 1e-19              # a tiny p-value dominates, no underflow
    assert np.isfinite(cauchy_combine([1e-300, 1e-300]))


def test_bh_matches_hand_computation():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    # thresholds q k / m with q = 0.05, m = 8: 0.00625, 0.0125, 0.01875, 0.025, ...; largest k with p_(k) <= is 2
    np.testing.assert_array_equal(bh(p, 0.05), [True, True] + [False] * 6)
    assert by(p, 0.05).sum() <= bh(p, 0.05).sum()


def test_fdp_and_power():
    fdp, power = fdp_and_power([True, True, False, True], [True, False, False, False])
    assert fdp == 1 / 3 and power == 2 / 3
