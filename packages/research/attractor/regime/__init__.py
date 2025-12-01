"""Q-matrix utilities for regime transitions and metastability analysis."""

from .q_matrix import (
    QMatrixResult,
    build_q_matrix,
    is_valid_q,
    estimate_q_from_counts,
    analyze_q,
    simulate_markov_chain,
)

__all__ = [
    "QMatrixResult",
    "build_q_matrix",
    "is_valid_q",
    "estimate_q_from_counts",
    "analyze_q",
    "simulate_markov_chain",
]
