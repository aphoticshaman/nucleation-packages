"""
Multi-source signal fusion for LatticeForge.

Combines:
- Reliability-weighted Dempster-Shafer fusion (Lockheed patents)
- Value clustering with basin refinement (92.1% error reduction)
- CIC distance functions for signal comparison

All LLM-free, deterministic, auditable.
"""

from .dempster_shafer import (
    ReliabilityConfig,
    logistic_reliability,
    compute_mass_function,
    additive_fusion,
    multiplicative_fusion,
    MetaFusionSelector,
    fuse_sources,
    hybrid_fusion,
)

from .value_clustering import (
    ValueCluster,
    ClusteringResult,
    relative_distance,
    value_clustering,
    basin_refinement,
    log_weighted_vote,
    fuse_values,
    js_divergence,
    js_divergence_text,
    tail_similarity,
    d_CIC,
)

__all__ = [
    # Dempster-Shafer
    "ReliabilityConfig",
    "logistic_reliability",
    "compute_mass_function",
    "additive_fusion",
    "multiplicative_fusion",
    "MetaFusionSelector",
    "fuse_sources",
    "hybrid_fusion",
    # Value Clustering
    "ValueCluster",
    "ClusteringResult",
    "relative_distance",
    "value_clustering",
    "basin_refinement",
    "log_weighted_vote",
    "fuse_values",
    # CIC Distance
    "js_divergence",
    "js_divergence_text",
    "tail_similarity",
    "d_CIC",
]
