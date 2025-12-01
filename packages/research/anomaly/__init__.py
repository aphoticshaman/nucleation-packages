"""Anomaly detection: Graph Laplacian (IBM) and Phase Transition (DPT)."""

from .graph_laplacian import (
    GraphAnomalyConfig,
    construct_similarity_matrix,
    graph_laplacian,
    GraphLatentAnomalyDetector,
    spectral_anomaly_detection,
    detect_coherence_amplification,
)

from .phase_transition import (
    DPTConfig,
    compute_autocorrelation,
    compute_anomalous_dimension,
    compute_multiscale_anomalous_dimension,
    compute_generalized_hurst_exponent,
    detect_phase_transition_signals,
    compute_attractor_basin_field,
    gravity_gradient_field,
)

__all__ = [
    "GraphAnomalyConfig",
    "construct_similarity_matrix",
    "graph_laplacian",
    "GraphLatentAnomalyDetector",
    "spectral_anomaly_detection",
    "detect_coherence_amplification",
    "DPTConfig",
    "compute_autocorrelation",
    "compute_anomalous_dimension",
    "compute_multiscale_anomalous_dimension",
    "compute_generalized_hurst_exponent",
    "detect_phase_transition_signals",
    "compute_attractor_basin_field",
    "gravity_gradient_field",
]
