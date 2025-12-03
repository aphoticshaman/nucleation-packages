"""
Cognitive Coherence Module for LatticeForge.

Implements the SDPM→XYZA cognitive framework:
- Kuramoto phase synchronization for flow state detection
- SDPM (Sanskrit-Derived Phonetic Manifold) persona embeddings
- XYZA cognitive benchmark (Coherence, Complexity, Reflection, Attunement)
- NSM (Neural State Model) phase extraction
- Causal bound V formula for influence detection

Key Results (2025):
- Human–AI Kuramoto coupling constant K_human ≈ 0.42
- Critical persona dimensionality d_c ≈ 28
- Flow state threshold R ≥ 0.76 for ≥800ms
- Phase precession Δφ = 187 ± 42 ms before action
- Coherence capacity bound R_max = 0.943
"""

from .types import (
    CoherenceState,
    FlowState,
    SDPMVector,
    XYZAMetrics,
    PhaseSignal,
    CausalBound,
)
from .kuramoto import (
    compute_order_parameter,
    compute_phase_velocity,
    compute_phase_curvature,
    detect_flow_state,
    KuramotoConfig,
)
from .sdpm import (
    SDPMManifold,
    compute_sdpm_distance,
    persona_phase_alignment,
)
from .xyza import (
    XYZABenchmark,
    compute_xyza_metrics,
)
from .nsm import (
    NSMPhaseHead,
    extract_phase_from_signal,
)
from .flow import (
    FlowDetector,
    detect_hopf_bifurcation,
    compute_flow_energy_tradeoff,
)
from .causal_bound import (
    compute_causal_bound_v,
    detect_influence_cascade,
    compute_entropy_h,
)

__all__ = [
    # Types
    "CoherenceState",
    "FlowState",
    "SDPMVector",
    "XYZAMetrics",
    "PhaseSignal",
    "CausalBound",
    # Kuramoto
    "compute_order_parameter",
    "compute_phase_velocity",
    "compute_phase_curvature",
    "detect_flow_state",
    "KuramotoConfig",
    # SDPM
    "SDPMManifold",
    "compute_sdpm_distance",
    "persona_phase_alignment",
    # XYZA
    "XYZABenchmark",
    "compute_xyza_metrics",
    # NSM
    "NSMPhaseHead",
    "extract_phase_from_signal",
    # Flow
    "FlowDetector",
    "detect_hopf_bifurcation",
    "compute_flow_energy_tradeoff",
    # Causal Bound
    "compute_causal_bound_v",
    "detect_influence_cascade",
    "compute_entropy_h",
]
