"""
LatticeForge Proofs - Proven Mathematical Foundations
======================================================

A collection of proven, tested, and documented algorithms for
geopolitical intelligence analysis and ML inference optimization.

Core Modules:
- cic_core: CIC Functional and supporting algorithms
- prometheus_insights: Novel insights from PROMETHEUS protocol
- cic_integration: TypeScript bridge (see .ts file)

Quick Start:
    from latticeforge_proofs import quick_infer, compute_cic

    samples = [12345, 12346, 12344, 12345, 99999]
    answer, confidence = quick_infer(samples)
    print(f"Answer: {answer}, Confidence: {confidence:.2%}")

Author: LatticeForge Team + Claude PROMETHEUS Synthesis
Version: 1.0.0
"""

from .cic_core import (
    # Constants
    ProvenConstants,

    # Enums
    SystemPhase,

    # Data Classes
    CICState,
    PhaseState,
    Cluster,
    ClusteringResult,
    GrokkingSignal,

    # Core Classes
    CICFunctional,
    ValueClustering,
    PhaseTransitionDetector,
    MicroGrokkingDetector,
    LatticeForgeInference,
    NormalizedCompressionDistance,

    # Convenience Functions
    quick_infer,
    compute_cic,
    detect_phase,
    cluster_values,
    detect_grokking,
)

from .prometheus_insights import (
    # Insight Classes
    VarianceParadox,
    InformationGeometryCriticalPoint,
    ThreeBitPrecisionLimit,
    VariationalFreeEnergyEquivalence,
    PhaseLockingEquivalence,
    AttractorBasinTheory,
    UniversalCollapsePattern,
    FibonacciOptimality,
    FiveHourWindow,
    CompressionCausalityDuality,

    # Engine
    PrometheusEngine,
)

__version__ = "1.0.0"
__author__ = "LatticeForge Team"
__all__ = [
    # Constants
    "ProvenConstants",

    # Enums
    "SystemPhase",

    # Data Classes
    "CICState",
    "PhaseState",
    "Cluster",
    "ClusteringResult",
    "GrokkingSignal",

    # Core Classes
    "CICFunctional",
    "ValueClustering",
    "PhaseTransitionDetector",
    "MicroGrokkingDetector",
    "LatticeForgeInference",
    "NormalizedCompressionDistance",

    # Convenience Functions
    "quick_infer",
    "compute_cic",
    "detect_phase",
    "cluster_values",
    "detect_grokking",

    # PROMETHEUS
    "VarianceParadox",
    "InformationGeometryCriticalPoint",
    "ThreeBitPrecisionLimit",
    "VariationalFreeEnergyEquivalence",
    "PhaseLockingEquivalence",
    "AttractorBasinTheory",
    "UniversalCollapsePattern",
    "FibonacciOptimality",
    "FiveHourWindow",
    "CompressionCausalityDuality",
    "PrometheusEngine",
]
