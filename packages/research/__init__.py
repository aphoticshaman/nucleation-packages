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
]
