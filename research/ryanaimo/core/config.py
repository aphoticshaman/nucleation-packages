"""
RYANAIMO Configuration
======================

Unified configuration for the entire pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ProblemType(Enum):
    """Mathematical problem classification."""
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    MIXED = "mixed"


@dataclass
class RyanAIMOConfig:
    """
    Master configuration for RYANAIMO solver.

    All parameters that affect inference behavior.
    """

    # Model
    model_path: str = "/kaggle/input/qwen-72b-math-nf4"
    quantization: str = "nf4"  # "nf4", "int8", "none"
    compute_dtype: str = "bfloat16"

    # Time budget
    total_budget_seconds: int = 280 * 60  # 280 minutes for AIMO3
    min_time_per_problem: float = 60.0    # At least 1 minute
    max_time_per_problem: float = 900.0   # At most 15 minutes

    # Generation
    max_tokens: int = 2048
    num_paths: int = 5  # Independent solution paths
    temperatures: List[float] = field(default_factory=lambda: [0.7, 0.5, 0.3, 0.2, 0.1])

    # Extended thinking
    enable_think_blocks: bool = True
    min_think_tokens: int = 500  # Minimum thinking before code

    # Proof constraints
    enable_bracket_tracking: bool = True
    enable_equation_tracking: bool = True
    enable_repetition_blocking: bool = True
    max_bracket_depth: int = 10

    # Value clustering
    clustering_threshold: float = 0.05  # 5% relative distance
    min_cluster_size: int = 2

    # CIC parameters
    lambda_compress: float = 0.5
    gamma_causal: float = 0.3
    crystallization_threshold: float = 0.1

    # Execution
    code_timeout: int = 30
    enable_sympy_verify: bool = True

    # Answer constraints
    answer_min: int = 0
    answer_max: int = 99999
    fallback_answer: int = 0

    def __post_init__(self):
        """Validate configuration."""
        assert self.answer_min <= self.answer_max
        assert self.total_budget_seconds > 0
        assert 0 < self.clustering_threshold < 1


@dataclass
class ProblemProfile:
    """Profile of a mathematical problem."""
    ptype: ProblemType
    has_modulo: bool = False
    modulo_target: Optional[int] = None
    is_counting: bool = False
    estimated_difficulty: float = 0.5  # 0-1 scale
    keywords: List[str] = field(default_factory=list)


__all__ = ["RyanAIMOConfig", "ProblemProfile", "ProblemType"]
