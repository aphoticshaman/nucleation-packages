"""Core types, transfer entropy, and Great Attractor theory."""

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

from .great_attractor import (
    AttractorBasin,
    IntentionalityField,
    GreatAttractorState,
    compute_intentionality_field,
    compute_basin_curvature,
    compute_causal_mass,
    compute_attractor_dominance,
    compute_predictive_gain_psi,
    compute_downward_causal_power_delta,
    compute_phase_transition_probability,
    GreatAttractorAnalyzer,
)

__all__ = [
    # Types
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
    # Transfer Entropy
    "TransferEntropyConfig",
    "transfer_entropy",
    "build_causal_graph",
    "detect_causal_structure_shift",
    "compute_intentionality_gradient",
    # Great Attractor (NOVEL - PATENTABLE)
    "AttractorBasin",
    "IntentionalityField",
    "GreatAttractorState",
    "compute_intentionality_field",
    "compute_basin_curvature",
    "compute_causal_mass",
    "compute_attractor_dominance",
    "compute_predictive_gain_psi",
    "compute_downward_causal_power_delta",
    "compute_phase_transition_probability",
    "GreatAttractorAnalyzer",
]
