"""
Cognitive Coherence Module for LatticeForge.

Implements the SDPM→XYZA cognitive framework:
- Kuramoto phase synchronization for flow state detection
- SDPM (Sanskrit-Derived Phonetic Manifold) persona embeddings
- XYZA cognitive benchmark (Coherence, Complexity, Reflection, Attunement)
- NSM (Neural State Model) phase extraction
- Causal bound V formula for influence detection
- Flow state detection via Hopf bifurcation

Key Results (2025):
- Human–AI Kuramoto coupling constant K_human ≈ 0.42
- Critical persona dimensionality d_c ≈ 28
- Flow state threshold R ≥ 0.76 for ≥800ms
- Phase precession Δφ = 187 ± 42 ms before action
- Coherence capacity bound R_max = 0.943
- Causal bound V > 3.0 indicates coordination
- XYZA optimal range: X≥0.76, Y∈[0.4,0.7], Z≥0.5, A≥0.42
"""

# Types
from .types import (
    CoherenceState,
    FlowState,
    FlowLevel,
    SDPMVector,
    XYZAMetrics,
    PhaseSignal,
    CausalBound,
    CascadeType,
    PersonaPhaseAlignment,
    FlowFixedPoint,
)

# Kuramoto phase synchronization
from .kuramoto import (
    compute_order_parameter,
    compute_phase_velocity,
    compute_phase_curvature,
    detect_flow_state,
    predict_flow_collapse,
    kuramoto_coupling_dynamics,
)

# SDPM persona embeddings
from .sdpm import (
    text_to_sdpm,
    sdpm_distance,
    sdpm_similarity,
    interpolate_sdpm,
    compose_sdpm,
    compute_persona_alignment,
    detect_persona_drift,
    sdpm_to_cognitive_state,
    extract_text_sdpm_trajectory,
    compute_trajectory_coherence,
    Varga,
    Svara,
)

# XYZA cognitive benchmark
from .xyza import (
    compute_coherence_x,
    compute_complexity_y,
    compute_reflection_z,
    compute_attunement_a,
    compute_xyza_metrics,
    xyza_trajectory,
    predict_cognitive_state,
    diagnose_xyza,
    xyza_to_radar_data,
    CognitiveLevel,
    COHERENCE_FLOW_THRESHOLD,
    COMPLEXITY_OPTIMAL_LOW,
    COMPLEXITY_OPTIMAL_HIGH,
    REFLECTION_AWARE_THRESHOLD,
    ATTUNEMENT_COUPLED_THRESHOLD,
)

# NSM phase extraction
from .nsm import (
    NSMConfig,
    NSMOutput,
    NSMPhaseHead,
    extract_attention_phases,
    compute_layer_phase_alignment,
    create_nsm_probe,
)

# NSM Pipeline (2x options + hybrid for each component)
from .nsm_pipeline import (
    # Configuration
    NSMPipelineConfig,
    NSMPipelineOutput,
    StateEncoderType,
    TransitionType,
    DecoderType,
    InferenceType,
    # Main pipeline
    NSMPipeline,
    # Encoders: Option A, Option B, Hybrid
    VariationalStateEncoderNP,
    ContrastiveStateEncoderNP,
    DualPathEncoderNP,
    # Transitions: Option A, Option B, Hybrid
    GRUTransitionNP,
    TransformerTransitionNP,
    AttentiveGRUTransitionNP,
    # Decoders: Option A, Option B, Hybrid
    ProbabilisticDecoderNP,
    FlowDecoderNP,
    MixtureDecoderNP,
    # Inference: Option A, Option B, Hybrid
    FilteringSMCNP,
    AmortizedVINP,
    AdaptiveInferenceNP,
    # Factory functions
    create_threat_actor_nsm,
    create_market_regime_nsm,
    create_escalation_tracker_nsm,
    create_cognitive_monitor_nsm,
    # Comparison utilities
    compare_encoder_options,
    compare_transition_options,
    compare_decoder_options,
)

# Flow state detection
from .flow import (
    FlowDetector,
    HopfState,
    FlowSession,
    detect_hopf_bifurcation,
    compute_critical_slowing_down,
    estimate_bifurcation_parameter,
    compute_flow_fixed_point,
    predict_flow_onset,
    optimal_intervention_timing,
    R_FLOW,
    R_DEEP_FLOW,
    FLOW_DWELL_MS,
)

# Causal bound V formula
from .causal_bound import (
    compute_entropy_h,
    compute_behavioral_entropy,
    compute_temporal_entropy,
    compute_linguistic_entropy,
    compute_mu_avg,
    compute_causal_bound_v,
    compute_causal_bound,
    detect_influence_cascade,
    detect_bot_network,
    compute_narrative_health,
    sliding_window_v,
    cusum_change_detection,
    InfluenceLevel,
)

__all__ = [
    # Types
    "CoherenceState",
    "FlowState",
    "FlowLevel",
    "SDPMVector",
    "XYZAMetrics",
    "PhaseSignal",
    "CausalBound",
    "CascadeType",
    "PersonaPhaseAlignment",
    "FlowFixedPoint",
    # Kuramoto
    "compute_order_parameter",
    "compute_phase_velocity",
    "compute_phase_curvature",
    "detect_flow_state",
    "predict_flow_collapse",
    "kuramoto_coupling_dynamics",
    # SDPM
    "text_to_sdpm",
    "sdpm_distance",
    "sdpm_similarity",
    "interpolate_sdpm",
    "compose_sdpm",
    "compute_persona_alignment",
    "detect_persona_drift",
    "sdpm_to_cognitive_state",
    "extract_text_sdpm_trajectory",
    "compute_trajectory_coherence",
    "Varga",
    "Svara",
    # XYZA
    "compute_coherence_x",
    "compute_complexity_y",
    "compute_reflection_z",
    "compute_attunement_a",
    "compute_xyza_metrics",
    "xyza_trajectory",
    "predict_cognitive_state",
    "diagnose_xyza",
    "xyza_to_radar_data",
    "CognitiveLevel",
    "COHERENCE_FLOW_THRESHOLD",
    "COMPLEXITY_OPTIMAL_LOW",
    "COMPLEXITY_OPTIMAL_HIGH",
    "REFLECTION_AWARE_THRESHOLD",
    "ATTUNEMENT_COUPLED_THRESHOLD",
    # NSM
    "NSMConfig",
    "NSMOutput",
    "NSMPhaseHead",
    "extract_attention_phases",
    "compute_layer_phase_alignment",
    "create_nsm_probe",
    # NSM Pipeline
    "NSMPipelineConfig",
    "NSMPipelineOutput",
    "StateEncoderType",
    "TransitionType",
    "DecoderType",
    "InferenceType",
    "NSMPipeline",
    "VariationalStateEncoderNP",
    "ContrastiveStateEncoderNP",
    "DualPathEncoderNP",
    "GRUTransitionNP",
    "TransformerTransitionNP",
    "AttentiveGRUTransitionNP",
    "ProbabilisticDecoderNP",
    "FlowDecoderNP",
    "MixtureDecoderNP",
    "FilteringSMCNP",
    "AmortizedVINP",
    "AdaptiveInferenceNP",
    "create_threat_actor_nsm",
    "create_market_regime_nsm",
    "create_escalation_tracker_nsm",
    "create_cognitive_monitor_nsm",
    "compare_encoder_options",
    "compare_transition_options",
    "compare_decoder_options",
    # Flow
    "FlowDetector",
    "HopfState",
    "FlowSession",
    "detect_hopf_bifurcation",
    "compute_critical_slowing_down",
    "estimate_bifurcation_parameter",
    "compute_flow_fixed_point",
    "predict_flow_onset",
    "optimal_intervention_timing",
    "R_FLOW",
    "R_DEEP_FLOW",
    "FLOW_DWELL_MS",
    # Causal Bound
    "compute_entropy_h",
    "compute_behavioral_entropy",
    "compute_temporal_entropy",
    "compute_linguistic_entropy",
    "compute_mu_avg",
    "compute_causal_bound_v",
    "compute_causal_bound",
    "detect_influence_cascade",
    "detect_bot_network",
    "compute_narrative_health",
    "sliding_window_v",
    "cusum_change_detection",
    "InfluenceLevel",
]
