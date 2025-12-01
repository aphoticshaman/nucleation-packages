"""Core types and transfer entropy computation."""

from .types import (
    RegimeState,
    FusionMethod,
    SourceSignal,
    CausalGraph,
    FusedBelief,
    AnomalyScore,
    PhaseTransitionSignal,
    RegimeEstimate,
    CausalEmergence,
    SparseGPState,
    CompressedSignal,
)

from .transfer_entropy import (
    TransferEntropyConfig,
    transfer_entropy,
    build_causal_graph,
    detect_causal_structure_shift,
    compute_intentionality_gradient,
)

__all__ = [
    "RegimeState",
    "FusionMethod",
    "SourceSignal",
    "CausalGraph",
    "FusedBelief",
    "AnomalyScore",
    "PhaseTransitionSignal",
    "RegimeEstimate",
    "CausalEmergence",
    "SparseGPState",
    "CompressedSignal",
    "TransferEntropyConfig",
    "transfer_entropy",
    "build_causal_graph",
    "detect_causal_structure_shift",
    "compute_intentionality_gradient",
]
