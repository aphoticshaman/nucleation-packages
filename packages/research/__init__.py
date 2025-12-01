"""
LatticeForge Research Module

Mathematical frameworks for multi-source intelligence fusion and anomaly detection.
Based on expired patents (2022-2024) and overlooked academic research.

CRITICAL components:
- Transfer Entropy Causal Graphs (core.transfer_entropy)
- Lockheed DS Fusion (fusion.dempster_shafer)
- Graph Laplacian Anomaly Detection (anomaly.graph_laplacian)
- Phase Transition / DPT Early Warning (anomaly.phase_transition)

IMPORTANT components:
- Online Sparse GP (inference.sparse_gp)
- MS-VAR Regime Detection (regime.msvar)
- Compressed Sensing / OMP (inference.compressed_sensing)
- Causal Emergence Measures (inference.causal_emergence)

Key expired patents incorporated:
- US6944566B2 (Lockheed) - Reliability-weighted DS fusion
- US6909997B2 (Lockheed) - Meta-fusion selection
- US9805002B2 (IBM) - Graph Laplacian anomaly detection
- US8645304B2 (IBM) - MS-VAR with hierarchical Bayesian group LASSO
- US8190549B2 (Honda) - Online sparse GP with O(n) Givens updates
- US8855431B2 (Stanford) - Compressed sensing / L1 minimization
- US8112340B2 (S&P) - Gaussian copula for tail dependence

Novel insights from Great Attractor framework:
1. Multi-membership Resonance Detection
2. Intentionality Gradient (rate of TE change)
3. Basin Boundary Detection (latent space proximity)
4. Coherence Amplification (spectral concentration)
5. Counter-Attractor Dynamics (fusion strategy shifts)
6. Collapse Velocity (uncertainty reduction rate)
7. Membership Vector Entanglement (sparse feature overlap)
8. Attractor Genesis Detection (transition probability spikes)
9. Causality Inversion Windows (coefficient sign flips)
10. Gravity Gradient Field (basin probability derivatives)
"""

from .core.types import (
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

from .core.transfer_entropy import (
    TransferEntropyConfig,
    transfer_entropy,
    build_causal_graph,
    detect_causal_structure_shift,
    compute_intentionality_gradient,
)

from .fusion.dempster_shafer import (
    ReliabilityConfig,
    logistic_reliability,
    additive_fusion,
    multiplicative_fusion,
    MetaFusionSelector,
    fuse_sources,
)

from .anomaly.graph_laplacian import (
    GraphAnomalyConfig,
    GraphLatentAnomalyDetector,
    spectral_anomaly_detection,
    detect_coherence_amplification,
)

from .anomaly.phase_transition import (
    DPTConfig,
    compute_anomalous_dimension,
    compute_multiscale_anomalous_dimension,
    compute_generalized_hurst_exponent,
    detect_phase_transition_signals,
    compute_attractor_basin_field,
    gravity_gradient_field,
)

from .inference.sparse_gp import (
    SparseGPConfig,
    OnlineSparseGP,
    confidence_score_pipeline,
)

from .inference.compressed_sensing import (
    CompressedSensingConfig,
    generate_measurement_matrix,
    orthogonal_matching_pursuit,
    basis_pursuit_lasso,
    SparseFeatureExtractor,
    cosamp_recovery,
)

from .inference.causal_emergence import (
    CausalEmergenceConfig,
    compute_causal_emergence,
    validate_regime_labels,
    EmergenceMonitor,
)

from .regime.msvar import (
    MSVARConfig,
    MarkovSwitchingVAR,
)

from .core.great_attractor import (
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

from .core.quantum_info import (
    DensityOperator,
    SystemState,
    von_neumann_entropy,
    mutual_information_quantum,
    awareness_functional,
    coherent_causal_capacity,
    QuantumInfoAnalyzer,
)

from .core.geometry import (
    RiemannianMetric,
    AttractorBasinGeometry,
    FreeEnergyLandscape,
    ManifoldGeometryAnalyzer,
    great_attractor_curvature,
    possibility_space_metric,
)

from .engine.outcome_steering import (
    SteeringMode,
    SteeringObjective,
    SteeringAction,
    Phase,
    Roadmap,
    SteeringResult,
    OGSEConfig,
    OutcomeGradientSteeringEngine,
    OGSEngine,
    compute_outcome_gradient,
    evaluate_steering_effectiveness,
)

from .copula.gaussian_copula import (
    GaussianCopula,
    TailDependenceAnalyzer,
    CopulaConfig,
    compute_kendall_tau,
    compute_spearman_rho,
    fit_gaussian_copula,
    simulate_from_copula,
    compute_tail_dependence,
)

from .core.fep_agent import (
    FEPAgentConfig,
    FEPAgent,
)

from .core.mckean_vlasov import (
    MVConfig,
    McKeanVlasovEnsemble,
)

from .core.great_attractor_sim import (
    GreatAttractorConfig,
    GreatAttractorHistory,
    GreatAttractorSimulator,
)

from .attractor.tda_monitor import (
    TDAMonitorConfig,
    TDAMonitorHistory,
    TDAMonitor,
    autocorrelation,
    persistent_entropy,
)

from .attractor.ga_ogse_bridge import (
    GAOGSEBridgeConfig,
    GAOGSEBridge,
)

from .attractor.regime.q_matrix import (
    QMatrixResult,
    build_q_matrix,
    is_valid_q,
    estimate_q_from_counts,
    analyze_q,
    simulate_markov_chain,
)

from .attractor.viz.plot_basins import (
    BasinPlotConfig,
    BasinPlotter,
)

from .attractor.viz.plot_geodesics import (
    GeodesicPlotConfig,
    GeodesicPlotter,
)

from .attractor.regime.regime_sde import (
    RegimeSDEConfig,
    RegimeSDE,
)

from .attractor.viz.plot_persistence import (
    PersistencePlotConfig,
    PersistencePlotter,
    compute_distance_matrix,
)

from .attractor.viz.anim_particles import (
    AnimationConfig,
    ParticleAnimator,
    animate_from_history,
)

from .attractor.api.latticeforge import (
    LatticeForgeConfig,
    LatticeForge,
)

__version__ = "0.1.0"

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
    # Fusion
    "ReliabilityConfig",
    "logistic_reliability",
    "additive_fusion",
    "multiplicative_fusion",
    "MetaFusionSelector",
    "fuse_sources",
    # Anomaly
    "GraphAnomalyConfig",
    "GraphLatentAnomalyDetector",
    "spectral_anomaly_detection",
    "detect_coherence_amplification",
    # Phase Transition
    "DPTConfig",
    "compute_anomalous_dimension",
    "compute_multiscale_anomalous_dimension",
    "compute_generalized_hurst_exponent",
    "detect_phase_transition_signals",
    "compute_attractor_basin_field",
    "gravity_gradient_field",
    # Inference
    "SparseGPConfig",
    "OnlineSparseGP",
    "confidence_score_pipeline",
    "CompressedSensingConfig",
    "generate_measurement_matrix",
    "orthogonal_matching_pursuit",
    "basis_pursuit_lasso",
    "SparseFeatureExtractor",
    "cosamp_recovery",
    "CausalEmergenceConfig",
    "compute_causal_emergence",
    "validate_regime_labels",
    "EmergenceMonitor",
    # Regime
    "MSVARConfig",
    "MarkovSwitchingVAR",
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
    "awareness_functional",
    "coherent_causal_capacity",
    "QuantumInfoAnalyzer",
    # Manifold Geometry Layer
    "RiemannianMetric",
    "AttractorBasinGeometry",
    "FreeEnergyLandscape",
    "ManifoldGeometryAnalyzer",
    "great_attractor_curvature",
    "possibility_space_metric",
    # Outcome Gradient Steering Engine (OGSE)
    "SteeringMode",
    "SteeringObjective",
    "SteeringAction",
    "Phase",
    "Roadmap",
    "SteeringResult",
    "OGSEConfig",
    "OutcomeGradientSteeringEngine",
    "OGSEngine",
    "compute_outcome_gradient",
    "evaluate_steering_effectiveness",
    # Gaussian Copula (S&P Patent)
    "GaussianCopula",
    "TailDependenceAnalyzer",
    "CopulaConfig",
    "compute_kendall_tau",
    "compute_spearman_rho",
    "fit_gaussian_copula",
    "simulate_from_copula",
    "compute_tail_dependence",
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
    # TDA Monitor
    "TDAMonitorConfig",
    "TDAMonitorHistory",
    "TDAMonitor",
    "autocorrelation",
    "persistent_entropy",
    # OGSE-GA Bridge
    "GAOGSEBridgeConfig",
    "GAOGSEBridge",
    # Q-Matrix Utilities
    "QMatrixResult",
    "build_q_matrix",
    "is_valid_q",
    "estimate_q_from_counts",
    "analyze_q",
    "simulate_markov_chain",
    # Visualization - Basins & Geodesics
    "BasinPlotConfig",
    "BasinPlotter",
    "GeodesicPlotConfig",
    "GeodesicPlotter",
    # Regime SDE
    "RegimeSDEConfig",
    "RegimeSDE",
    # Visualization - Persistence
    "PersistencePlotConfig",
    "PersistencePlotter",
    "compute_distance_matrix",
    # Visualization - Animation
    "AnimationConfig",
    "ParticleAnimator",
    "animate_from_history",
    # High-level API
    "LatticeForgeConfig",
    "LatticeForge",
]
