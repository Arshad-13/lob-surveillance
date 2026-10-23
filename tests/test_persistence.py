import numpy as np

from lobsurv.persistence import consecutive_distances, diagram_stream, mst_edge_lengths, window_diagram


def test_h0_deaths_are_mst_edge_lengths():
    X = np.random.default_rng(1).standard_normal((30, 6))
    H0, _ = window_diagram(X)
    np.testing.assert_allclose(np.sort(H0[:, 1]), mst_edge_lengths(X), rtol=1e-9)


def test_circle_has_one_long_h1_bar():
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    _, H1 = window_diagram(np.column_stack([np.cos(th), np.sin(th)]))
    life = H1[:, 1] - H1[:, 0]
    assert life.max() > 1.0 and (life > 0.5).sum() == 1


def test_identical_windows_have_zero_distance():
    X = np.tile(np.random.default_rng(2).standard_normal((5, 3)), (4, 1))   # period 5
    d = consecutive_distances(diagram_stream(X, window=5)[::5])
    np.testing.assert_allclose(d, 0.0, atol=1e-12)
