import numpy as np

from lobsurv.scoring import evaluate_mixture, generator_pvalues, score, uncond_pvalues


def test_generator_pvalues_match_definition():
    sim = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    obs = np.array([[2.0, 10.0], [0.0, 5.0]])
    np.testing.assert_allclose(generator_pvalues(sim, obs), [[3 / 4, 1 / 4], [4 / 4, 4 / 4]])


def test_uncond_pvalues_are_conformal_per_statistic():
    ref = np.array([[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(uncond_pvalues(ref, np.array([[2.5], [9.0]])), [[2 / 4], [1 / 4]])


def test_mixture_fdr_is_controlled_and_power_grows_with_signal():
    rng = np.random.default_rng(0)
    cal = rng.standard_normal(2000)
    clean = rng.standard_normal(2000)
    weak, strong = rng.standard_normal(2000) + 1.0, rng.standard_normal(2000) + 5.0
    a = evaluate_mixture(cal, clean, weak, q=0.1, prevalence=0.05, rng=rng, reps=40)
    b = evaluate_mixture(cal, clean, strong, q=0.1, prevalence=0.05, rng=rng, reps=40)
    assert a["fdr"] <= 0.15 and b["fdr"] <= 0.15
    assert b["power"] > 0.9 and b["power"] > a["power"]


def test_score_is_monotone_in_p():
    assert score(np.array([[0.01, 0.5]])) > score(np.array([[0.2, 0.5]]))
