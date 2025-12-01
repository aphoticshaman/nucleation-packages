"""
Regime dynamics for Great Attractor.

Components:
- Q-matrix utilities for transition analysis and metastability
- Regime-switching SDE for macro-dynamics
"""

from .q_matrix import (
    QMatrixResult,
    build_q_matrix,
    is_valid_q,
    estimate_q_from_counts,
    analyze_q,
    simulate_markov_chain,
)

from .regime_sde import (
    RegimeSDEConfig,
    RegimeSDE,
)

__all__ = [
    # Q-Matrix
    "QMatrixResult",
    "build_q_matrix",
    "is_valid_q",
    "estimate_q_from_counts",
    "analyze_q",
    "simulate_markov_chain",
    # Regime SDE
    "RegimeSDEConfig",
    "RegimeSDE",
]
