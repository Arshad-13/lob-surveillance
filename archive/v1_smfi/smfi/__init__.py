"""
Systemic Market Fragility Index (SMFI) Package.

Topological detection of limit order book manipulation via
multiparameter zigzag persistence on log-signature manifolds.
"""

from .engine import SMFIEngine, CrossAssetCouplingComplex

__version__ = "0.1.0"
__all__ = ["SMFIEngine", "CrossAssetCouplingComplex"]
