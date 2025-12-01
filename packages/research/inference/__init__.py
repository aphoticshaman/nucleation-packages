"""Inference: Sparse GP (Honda), Compressed Sensing (Stanford), Causal Emergence."""

from .sparse_gp import (
    SparseGPConfig,
    rbf_kernel,
    OnlineSparseGP,
    confidence_score_pipeline,
)

from .compressed_sensing import (
    CompressedSensingConfig,
    generate_measurement_matrix,
    orthogonal_matching_pursuit,
    basis_pursuit_lasso,
    compute_rip_constant,
    SparseFeatureExtractor,
    cosamp_recovery,
)

from .causal_emergence import (
    CausalEmergenceConfig,
    gaussian_entropy,
    gaussian_mutual_information,
    gaussian_conditional_mutual_information,
    compute_psi,
    compute_delta,
    compute_gamma,
    compute_causal_emergence,
    validate_regime_labels,
    EmergenceMonitor,
)

__all__ = [
    "SparseGPConfig",
    "rbf_kernel",
    "OnlineSparseGP",
    "confidence_score_pipeline",
    "CompressedSensingConfig",
    "generate_measurement_matrix",
    "orthogonal_matching_pursuit",
    "basis_pursuit_lasso",
    "compute_rip_constant",
    "SparseFeatureExtractor",
    "cosamp_recovery",
    "CausalEmergenceConfig",
    "gaussian_entropy",
    "gaussian_mutual_information",
    "gaussian_conditional_mutual_information",
    "compute_psi",
    "compute_delta",
    "compute_gamma",
    "compute_causal_emergence",
    "validate_regime_labels",
    "EmergenceMonitor",
]
