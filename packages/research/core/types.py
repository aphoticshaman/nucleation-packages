"""
Core type definitions for the LatticeForge research module.
Mathematical frameworks for multi-source intelligence fusion and anomaly detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class RegimeState(Enum):
    """Discrete regime states for phase transition detection."""
    CALM = 0
    BUILDUP = 1
    TRANSITION = 2
    CRISIS = 3
    RECOVERY = 4


class FusionMethod(Enum):
    """Fusion strategies from US6909997B2 (Lockheed, expired 2023)."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class SourceSignal:
    """A single data source signal with quality metrics."""
    name: str
    values: NDArray[np.float64]  # Shape: (T,) or (T, D)
    timestamps: NDArray[np.float64]  # Shape: (T,)
    quality: NDArray[np.float64]  # Shape: (T,) - per-timestep quality metric
    metadata: Dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        """Number of timesteps."""
        return len(self.timestamps)

    @property
    def D(self) -> int:
        """Dimensionality of signal."""
        if self.values.ndim == 1:
            return 1
        return self.values.shape[1]


@dataclass
class CausalGraph:
    """
    Weighted directed graph representing causal relationships.
    Edge weights are transfer entropy values.
    """
    nodes: List[str]  # Source names
    adjacency: NDArray[np.float64]  # Shape: (N, N) - A[i,j] = TE from j to i
    threshold: float = 0.0  # Edges below threshold are zeroed

    def get_edge_weight(self, source: str, target: str) -> float:
        """Get transfer entropy from source to target."""
        i = self.nodes.index(target)
        j = self.nodes.index(source)
        return self.adjacency[i, j]

    def get_laplacian(self) -> NDArray[np.float64]:
        """Compute graph Laplacian L = D - A."""
        degree = np.diag(self.adjacency.sum(axis=1))
        return degree - self.adjacency

    def get_normalized_laplacian(self) -> NDArray[np.float64]:
        """Compute symmetric normalized Laplacian."""
        d = self.adjacency.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        I = np.eye(len(self.nodes))
        return I - D_inv_sqrt @ self.adjacency @ D_inv_sqrt


@dataclass
class FusedBelief:
    """
    Result of reliability-weighted fusion across sources.
    Based on US6944566B2 (Lockheed, expired 2023).
    """
    hypotheses: List[str]  # Hypothesis labels (e.g., regime names)
    probabilities: NDArray[np.float64]  # Shape: (K,) - fused probabilities
    reliabilities: Dict[str, float]  # Per-source reliability scores
    method_used: FusionMethod
    confidence: float  # Overall fusion confidence

    @property
    def map_hypothesis(self) -> str:
        """Maximum a posteriori hypothesis."""
        return self.hypotheses[np.argmax(self.probabilities)]

    @property
    def entropy(self) -> float:
        """Shannon entropy of belief distribution."""
        p = self.probabilities
        p = p[p > 0]  # Avoid log(0)
        return -np.sum(p * np.log2(p))


@dataclass
class AnomalyScore:
    """Anomaly detection result with multiple scoring methods."""
    timestamp: float
    scores: Dict[str, float]  # Method name -> score
    composite_score: float  # Combined score
    is_anomaly: bool
    threshold: float
    contributing_sources: List[str]  # Which sources flagged anomaly


@dataclass
class PhaseTransitionSignal:
    """
    Early warning signal for phase transitions.
    Based on DPT framework (arXiv:2408.06433).
    """
    timestamp: float
    anomalous_dimension: float  # Δ(t,τ)
    volatility: float
    autocorrelation: float
    trend_strength: float  # Strength of upward trend in Δ
    predicted_transition_prob: float
    estimated_time_to_transition: Optional[float] = None


@dataclass
class RegimeEstimate:
    """
    Regime detection result from MS-VAR or particle filter.
    """
    timestamp: float
    state_probabilities: NDArray[np.float64]  # Shape: (K,) for K regimes
    map_state: int  # Most likely state index
    transition_matrix: NDArray[np.float64]  # Shape: (K, K)
    state_duration: int  # Timesteps in current state


@dataclass
class CausalEmergence:
    """
    Causal emergence measures from Mediano et al. (arXiv:2004.08220).
    Quantifies whether macro-features have genuine causal power.
    """
    psi: float  # Ψ > 0 means causally emergent
    delta: float  # Δ > 0 means downward causation
    gamma: float  # Γ = 0 with Ψ > 0 means causal decoupling

    @property
    def is_emergent(self) -> bool:
        return self.psi > 0

    @property
    def has_downward_causation(self) -> bool:
        return self.delta > 0

    @property
    def is_decoupled(self) -> bool:
        return self.psi > 0 and np.isclose(self.gamma, 0, atol=1e-6)


@dataclass
class SparseGPState:
    """
    State for online sparse Gaussian Process.
    Based on US8190549B2 (Honda, expired May 2024).
    """
    inducing_points: NDArray[np.float64]  # Shape: (M, D)
    alpha: NDArray[np.float64]  # Cholesky factor
    kernel_params: Dict[str, float]
    noise_var: float


@dataclass
class CompressedSignal:
    """
    Result of compressed sensing / sparse recovery.
    Based on US8855431B2 (Stanford, expired Aug 2024).
    """
    measurements: NDArray[np.float64]  # y = Φx
    recovered_signal: NDArray[np.float64]  # x̂
    support: NDArray[np.int64]  # Indices of non-zero components
    residual_norm: float
    sparsity: int  # Number of non-zero components
