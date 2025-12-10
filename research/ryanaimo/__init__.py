"""
RYANAIMO - Ground-Up AIMO3 Solver
=================================

Purpose-built race car for the AI Mathematical Olympiad.

Core Philosophy:
    F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

Every component optimizes the CIC functional.

Author: Ryan J Cardwell (Archer Phoenix)
"""

__version__ = "0.1.0"
__author__ = "Ryan J Cardwell"

from ryanaimo.solver import RyanAIMOSolver

__all__ = ["RyanAIMOSolver"]
