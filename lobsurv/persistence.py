"""
Sliding-window persistence of log-signature point clouds: the topological candidate statistic.

This replaces the v1 SMFI engine (archive/v1_smfi). What it is, stated plainly:
  - each window of W consecutive log-signature vectors is a point cloud; its Vietoris-Rips
    persistence (2-skeleton, so H0 and H1) is computed with GUDHI;
  - the signal is the p-Wasserstein distance between the diagrams of consecutive windows.
This is ordinary persistence per window, not zigzag persistence. H0 of a Rips filtration is
single-linkage clustering: its finite death times are the minimum-spanning-tree edge lengths
(`mst_edge_lengths`, checked in the tests). An H0-only signal is therefore a nearest-neighbour
statistic, and only H1 is information a non-topological detector would not have; the proposal's
H4 tests whether it adds anything.
No fallbacks: a missing library raises instead of silently computing a different quantity.
"""

from typing import List, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def window_diagram(points: np.ndarray, max_edge: float = np.inf, min_persistence: float = 0.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Finite (birth, death) pairs of H0 and H1 for one point cloud [W, d]."""
    import gudhi
    st = gudhi.RipsComplex(points=points, max_edge_length=max_edge).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    out = []
    for dim in (0, 1):
        D = np.asarray(st.persistence_intervals_in_dimension(dim), dtype=float).reshape(-1, 2)
        D = D[np.isfinite(D[:, 1])]
        out.append(D[D[:, 1] - D[:, 0] > min_persistence])
    return out[0], out[1]


def diagram_stream(signatures: np.ndarray, window: int, **kw) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Diagrams of every window signatures[t : t + window], t = 0 .. N - window."""
    return [window_diagram(signatures[t:t + window], **kw) for t in range(len(signatures) - window + 1)]


def consecutive_distances(diagrams, dim: int = 0, order: float = 2.0) -> np.ndarray:
    """W_order distance between the dimension-`dim` diagrams of consecutive windows: [len(diagrams) - 1]."""
    from gudhi.wasserstein import wasserstein_distance
    return np.array([wasserstein_distance(a[dim], b[dim], order=order, internal_p=2.0)
                     for a, b in zip(diagrams[:-1], diagrams[1:])])


def mst_edge_lengths(points: np.ndarray) -> np.ndarray:
    """Sorted Euclidean minimum-spanning-tree edge lengths (= finite H0 death times of the Rips filtration)."""
    T = minimum_spanning_tree(squareform(pdist(points)))
    return np.sort(T.data)
