"""
The conditional null model: the frozen level-1 generator from the companion repository.

For a real moment s of the stream, the generator simulates what the next HORIZON events could
have looked like given the actual context before s. A window statistic's tail probability under
those simulations is a p-value for "this window is ordinary for this moment":
  mc_pvalues   plain simulation, resolution 1 / (n + 1)
  ams_pvalue   adaptive multilevel splitting for deep tails (unbiased, not conservative; the
               generator repo's pilot 7 checked it against brute force down to about 1e-3)

The generator is imported from the repository named by the LOB_GENERATOR_REPO environment
variable (default: ../../Research relative to this repository), so this repository holds no copy of
its code. The configuration is the one fixed before the generator's test-split confirmation
(variant slow-s, recent pool of 30,000 moments); do not tune it here.
"""

import os
import subprocess
import sys
from typing import Dict, Optional

import numpy as np

from lobsurv.fdr import mc_pvalue
from lobsurv.statistics import window_stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Research"))
FROZEN_VARIANT, FROZEN_POOL = "slow-s", "W30000"


def generator_repo() -> str:
    repo = os.path.abspath(os.environ.get("LOB_GENERATOR_REPO", DEFAULT_REPO))
    if not os.path.isfile(os.path.join(repo, "ML", "benchmark", "pilot_splitting.py")):
        raise FileNotFoundError(f"generator repository not found at {repo}; set LOB_GENERATOR_REPO")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def generator_provenance() -> Dict:
    """Commit and uncommitted-change count of the generator repository, to store with every result."""
    repo = generator_repo()
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
    return {"repo": repo, "commit": git("rev-parse", "HEAD"),
            "uncommitted_files": len([l for l in git("status", "--porcelain").splitlines() if l.strip()])}


class GeneratorNull:
    def __init__(self, dataset: str, variant: str = FROZEN_VARIANT, pool: str = FROZEN_POOL, seed: int = 0,
                 parts=("train", "val", "test")):
        generator_repo()
        from ML.benchmark.pilot_splitting import build, make_step
        from ML.benchmark.run_benchmark import CONTEXT, HORIZON
        from ML.ctesn.splitting import ams, simulate
        self._make_step, self._simulate, self._ams = make_step, simulate, ams
        self.dataset, self.variant, self.pool, self.seed = dataset, variant, pool, seed
        self.context, self.horizon = CONTEXT, HORIZON
        self.G = build(dataset, variant, seed, parts=parts)
        self.view, self.has_dt, self.offsets = self.G["view"], self.G["has_dt"], self.G["offsets"]
        self.stream = self.G["S"]                 # clean real stream (z-scored); injections never enter it

    def simulate(self, s: int, n: int, rng) -> np.ndarray:
        """n continuations [n, HORIZON, d] of the real context ending just before stream position s."""
        step, state0 = self._make_step(self.G, self.pool, s)
        Y, _ = self._simulate(step, state0, self.horizon, n, rng)
        return Y

    def mc_pvalues(self, s: int, observed: np.ndarray, panel: Dict, n: int, rng,
                   Y: Optional[np.ndarray] = None) -> Dict[str, float]:
        """observed: the real (possibly injected) window [HORIZON, d] that followed the context."""
        Y = self.simulate(s, n, rng) if Y is None else Y
        obs, sim = window_stats(panel, observed[None]), window_stats(panel, Y)
        return {k: float(mc_pvalue(obs[k][0], sim[k])) for k in panel}

    def ams_pvalue(self, s: int, score, level: float, n: int, rng) -> Dict:
        """P(running max of score >= level | context) by multilevel splitting; returns the splitting result."""
        step, state0 = self._make_step(self.G, self.pool, s)
        return self._ams(step, state0, self.horizon, score, level, n=n, rng=rng)
