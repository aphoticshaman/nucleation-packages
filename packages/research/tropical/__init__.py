"""
Tropical Geometry Module

Implements tropical algebra for transformer expressivity analysis.

Core Theorem: Attention mechanisms in the T→∞ limit become
tropical polynomial evaluators, implying transformers compute
piecewise-linear functions bounded by tropical geometry.
"""

from .tropical_attention import (
    # Constants
    TROPICAL_ZERO,
    TROPICAL_ONE,

    # Semiring operations
    tropical_add,
    tropical_mul,
    tropical_pow,
    tropical_sum,
    tropical_product,

    # Matrix operations
    tropical_matmul,
    tropical_det_2x2,

    # Polynomials
    TropicalPolynomial,

    # Softmax limit
    log_sum_exp,
    temperature_log_sum_exp,
    demonstrate_tropical_limit,

    # Expressivity
    binomial,
    newton_polytope_bound,
    calculate_expressivity_bounds,
    min_layers_for_degree,
    ExpressivityBound,

    # ARC
    estimate_arc_complexity,
    estimate_from_operations,
    ARC_OPERATION_PATTERNS,
    ARCComplexity,

    # Verification
    verify_tropical_theorem,
    TROPICAL_PROOF_STEPS,
    ProofStep,
)

__all__ = [
    'TROPICAL_ZERO',
    'TROPICAL_ONE',
    'tropical_add',
    'tropical_mul',
    'tropical_pow',
    'tropical_sum',
    'tropical_product',
    'tropical_matmul',
    'tropical_det_2x2',
    'TropicalPolynomial',
    'log_sum_exp',
    'temperature_log_sum_exp',
    'demonstrate_tropical_limit',
    'binomial',
    'newton_polytope_bound',
    'calculate_expressivity_bounds',
    'min_layers_for_degree',
    'ExpressivityBound',
    'estimate_arc_complexity',
    'estimate_from_operations',
    'ARC_OPERATION_PATTERNS',
    'ARCComplexity',
    'verify_tropical_theorem',
    'TROPICAL_PROOF_STEPS',
    'ProofStep',
]
