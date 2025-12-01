"""Reliability-weighted Dempster-Shafer fusion (Lockheed patents)."""

from .dempster_shafer import (
    ReliabilityConfig,
    logistic_reliability,
    compute_mass_function,
    additive_fusion,
    multiplicative_fusion,
    MetaFusionSelector,
    fuse_sources,
)

__all__ = [
    "ReliabilityConfig",
    "logistic_reliability",
    "compute_mass_function",
    "additive_fusion",
    "multiplicative_fusion",
    "MetaFusionSelector",
    "fuse_sources",
]
