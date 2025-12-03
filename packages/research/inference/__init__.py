"""
Inference Module.

Components:
- Sparse GP (Honda): Online Gaussian process for confidence scoring
- Compressed Sensing (Stanford): Sparse recovery algorithms
- Causal Emergence: Φ and emergence detection
- Cognitive LLM: Self-hosted Phi inference with SDPM→XYZA monitoring
"""

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

# Cognitive LLM (lazy import - has heavy deps)
# from .cognitive_llm import CognitiveLLM, CognitiveLLMConfig, CognitiveResponse
# from .client import LatticeForgeClient, LatticeForgeClientSync, InferenceResult

__all__ = [
    # Sparse GP
    "SparseGPConfig",
    "rbf_kernel",
    "OnlineSparseGP",
    "confidence_score_pipeline",
    # Compressed Sensing
    "CompressedSensingConfig",
    "generate_measurement_matrix",
    "orthogonal_matching_pursuit",
    "basis_pursuit_lasso",
    "compute_rip_constant",
    "SparseFeatureExtractor",
    "cosamp_recovery",
    # Causal Emergence
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
    # Cognitive LLM (import directly: from research.inference.cognitive_llm import ...)
    # "CognitiveLLM",
    # "CognitiveLLMConfig",
    # "CognitiveResponse",
    # "LatticeForgeClient",
    # "LatticeForgeClientSync",
]
