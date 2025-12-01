"""Core types, transfer entropy, Great Attractor, quantum info, and geometry."""

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

from .quantum_info import (
    DensityOperator,
    SystemState,
    von_neumann_entropy,
    mutual_information_quantum,
    compute_agent_world_mi,
    compute_agent_memory_mi,
    awareness_functional,
    channel_capacity_holevo,
    coherent_causal_capacity,
    decoherence_to_pointer_basis,
    QuantumInfoAnalyzer,
)

from .geometry import (
    RiemannianMetric,
    AttractorBasinGeometry,
    FreeEnergyLandscape,
    compute_fisher_metric,
    compute_ricci_curvature_scalar,
    compute_geodesic,
    detect_basin_boundaries,
    compute_manifold_collapse_rate,
    compute_basin_curvature_tensor,
    estimate_basin_volume,
    ManifoldGeometryAnalyzer,
    great_attractor_curvature,
    possibility_space_metric,
)

from .fep_agent import (
    FEPAgentConfig,
    FEPAgent,
)

from .mckean_vlasov import (
    MVConfig,
    McKeanVlasovEnsemble,
)

from .great_attractor_sim import (
    GreatAttractorConfig,
    GreatAttractorHistory,
    GreatAttractorSimulator,
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
    # Quantum Information Layer
    "DensityOperator",
    "SystemState",
    "von_neumann_entropy",
    "mutual_information_quantum",
    "compute_agent_world_mi",
    "compute_agent_memory_mi",
    "awareness_functional",
    "channel_capacity_holevo",
    "coherent_causal_capacity",
    "decoherence_to_pointer_basis",
    "QuantumInfoAnalyzer",
    # Manifold Geometry Layer
    "RiemannianMetric",
    "AttractorBasinGeometry",
    "FreeEnergyLandscape",
    "compute_fisher_metric",
    "compute_ricci_curvature_scalar",
    "compute_geodesic",
    "detect_basin_boundaries",
    "compute_manifold_collapse_rate",
    "compute_basin_curvature_tensor",
    "estimate_basin_volume",
    "ManifoldGeometryAnalyzer",
    "great_attractor_curvature",
    "possibility_space_metric",
    # FEP Active Inference Agent
    "FEPAgentConfig",
    "FEPAgent",
    # McKean-Vlasov Ensemble
    "MVConfig",
    "McKeanVlasovEnsemble",
    # Great Attractor Simulator
    "GreatAttractorConfig",
    "GreatAttractorHistory",
    "GreatAttractorSimulator",
]
