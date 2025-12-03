"""
Type definitions for the Cognitive Coherence Module.

Mathematical foundations from:
- Kuramoto model: R(t)e^{jΨ(t)} = (1/N)Σ e^{jθ_i(t)}
- Causal bound: V = -log(μ_avg - 1) * (nom - est) / H(z)
- SDPM: 512-dim persona vector p = Σ w_i · φ(akṣara_i)
- XYZA: 4-axis cognitive benchmark [X, Y, Z, A]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class FlowLevel(Enum):
    """Discrete flow state levels."""
    NONE = 0       # R < 0.45
    EMERGING = 1   # 0.45 ≤ R < 0.65
    BUILDING = 2   # 0.65 ≤ R < 0.76
    FLOW = 3       # 0.76 ≤ R < 0.88
    DEEP_FLOW = 4  # R ≥ 0.88


class CascadeType(Enum):
    """Information cascade structure types."""
    STAR = "star"           # Single influential source
    TREE = "tree"           # Hierarchical diffusion (organic)
    FOREST = "forest"       # Multiple independent origins (coordinated)
    RANDOM = "random"       # Purely organic


@dataclass
class PhaseSignal:
    """
    Phase signal from oscillator or NSM extraction.

    θ_i(t) = instantaneous phase of signal i at time t
    """
    phases: NDArray[np.float64]      # Shape: (N,) or (N, T) - phases per oscillator
    timestamps: NDArray[np.float64]  # Shape: (T,)
    frequencies: NDArray[np.float64] # Shape: (N,) - natural frequencies ω_i

    @property
    def N(self) -> int:
        """Number of oscillators."""
        if self.phases.ndim == 1:
            return len(self.phases)
        return self.phases.shape[0]

    @property
    def T(self) -> int:
        """Number of timesteps."""
        return len(self.timestamps)


@dataclass
class CoherenceState:
    """
    Kuramoto order parameter state.

    R(t)e^{jΨ(t)} = (1/N)Σ e^{jθ_i(t)}

    R = magnitude (0-1), coherence level
    Ψ = global phase angle
    """
    R: float                         # Order parameter magnitude [0, 1]
    Psi: float                       # Global phase angle [0, 2π)
    R_dot: float                     # Coherence velocity dR/dt
    R_ddot: float                    # Coherence acceleration d²R/dt²
    sigma_omega: float               # Angular velocity dispersion
    timestamp: float

    @property
    def flow_level(self) -> FlowLevel:
        """Classify current flow state."""
        if self.R < 0.45:
            return FlowLevel.NONE
        elif self.R < 0.65:
            return FlowLevel.EMERGING
        elif self.R < 0.76:
            return FlowLevel.BUILDING
        elif self.R < 0.88:
            return FlowLevel.FLOW
        else:
            return FlowLevel.DEEP_FLOW

    @property
    def is_flow(self) -> bool:
        """Check if in flow state (R ≥ 0.76)."""
        return self.R >= 0.76

    @property
    def near_collapse(self) -> bool:
        """Check if approaching flow collapse (R_dot < -0.1)."""
        return self.R_dot < -0.1 and self.R > 0.5


@dataclass
class FlowState:
    """
    Complete flow state with temporal dynamics.

    Implements flow detection from:
    - R(t) ≥ R_flow for ≥ duration threshold
    - R_dot > 0 (building) or stable
    - Hopf bifurcation detection for collapse prediction
    """
    level: FlowLevel                 # Current flow level
    R: float                         # Order parameter
    dR_dt: float                     # First derivative
    d2R_dt2: float                   # Second derivative
    time_in_state_ms: float          # Time in current level
    stability: float                 # Stability score [0, 1]
    predicted_collapse_ms: Optional[float]  # Time to collapse
    is_flow: bool                    # R ≥ 0.76
    is_deep_flow: bool               # R ≥ 0.88


@dataclass
class SDPMVector:
    """
    Sanskrit-Derived Phonetic Manifold persona vector.

    p = Σ w_i · φ(akṣara_i) where φ is learned phonetic embedding.

    Proven: Distance in SDPM predicts phase alignment (r=0.91).
    Critical dimensionality d_c ≈ 28 (above which alignment plateaus).
    """
    embedding: NDArray[np.float64]           # Shape: (512,) SDPM embedding
    varga_distribution: NDArray[np.float64]  # Shape: (5,) consonant class distribution
    svara_distribution: NDArray[np.float64]  # Shape: (6,) vowel distribution
    cognitive_mode: str                      # Dominant cognitive mode from varga
    emotional_valence: float                 # Emotional valence [-1, 1]

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        return len(self.embedding)


@dataclass
class XYZAMetrics:
    """
    XYZA Cognitive Benchmark metrics.

    Four orthogonal axes:
    - X (Coherence): Phase synchronization quality
    - Y (Complexity): Information density and entropy
    - Z (Reflection): Meta-cognitive depth
    - A (Attunement): Human-AI coupling quality

    Performance = f(X, Y, Z, A) learned from trials.
    """
    coherence_x: float      # X ∈ [0, 1] - from Kuramoto R
    complexity_y: float     # Y ∈ [0, 1] - entropy-based
    reflection_z: float     # Z ∈ [0, 1] - meta-cognitive score
    attunement_a: float     # A ∈ [0, 1] - coupling with human
    timestamp: float        # Timestamp
    cognitive_level: str    # Level classification

    @property
    def combined_score(self) -> float:
        """Combined XYZA score."""
        return (self.coherence_x + self.complexity_y +
                self.reflection_z + self.attunement_a) / 4

    @property
    def vector(self) -> NDArray[np.float64]:
        """Return as numpy vector [X, Y, Z, A]."""
        return np.array([self.coherence_x, self.complexity_y,
                        self.reflection_z, self.attunement_a])

    @property
    def balanced(self) -> bool:
        """Check if axes are balanced (no axis < 0.3)."""
        return min(self.coherence_x, self.complexity_y,
                   self.reflection_z, self.attunement_a) >= 0.3


@dataclass
class CausalBound:
    """
    Causal bound V for influence detection.

    V = -log(μ_avg - 1) * (nom - est) / H(z)

    Components:
    - μ_avg: Average engagement rate (uniformity indicator)
    - nom: Nominal expected engagement
    - est: Estimated observed engagement
    - H(z): Entropy of behavioral distribution

    Detection thresholds:
    - V > 2.0: Approaching cascade threshold
    - V > 3.0: Likely coordination
    - V > 4.0: Imminent cascade / state-sponsored operation
    - V > 5.0: Sophisticated coordinated campaign
    """
    V: float                         # Causal bound value
    mu_avg: float                    # Average engagement rate
    nom: float                       # Nominal expected
    est: float                       # Estimated observed
    H_z: float                       # Behavioral entropy
    timestamp: float

    # Cascade analysis
    cascade_type: CascadeType = CascadeType.RANDOM
    confidence: float = 0.0

    @property
    def is_coordinated(self) -> bool:
        """Check if V indicates coordination (V > 3.0)."""
        return self.V > 3.0

    @property
    def is_state_sponsored(self) -> bool:
        """Check if V indicates state-sponsored operation (V > 5.0)."""
        return self.V > 5.0

    @property
    def cascade_imminent(self) -> bool:
        """Check if cascade threshold approaching (V > 4.0)."""
        return self.V > 4.0

    def intervention_urgency(self) -> str:
        """Determine intervention urgency based on V."""
        if self.V < 2.0:
            return "none"
        elif self.V < 3.0:
            return "monitor"
        elif self.V < 4.0:
            return "prepare_counter"
        else:
            return "immediate"


@dataclass
class PersonaPhaseAlignment:
    """
    Coupling between AI persona and user phase.

    From persona-phase coupling law:
    θ̇_i = ω_i + Σ_k K_k sin(ψ_k(p) - θ_i)

    K_human ≈ 0.42 (fitted on 40k sessions).
    """
    alignment_score: float           # Overall alignment [0, 1]
    phase_difference: float          # Phase difference in radians
    coupling_strength: float         # Effective coupling K
    cognitive_mode_match: bool       # Whether modes match
    stability: float                 # Stability of alignment

    @property
    def aligned(self) -> bool:
        """Check if phases are aligned (|diff| < π/4)."""
        return abs(self.phase_difference) < np.pi / 4


@dataclass
class FlowFixedPoint:
    """
    Hopf bifurcation fixed point characterization.

    The stable flow state corresponds to a fixed point of the
    Kuramoto dynamics: R* where dR/dt = 0.
    """
    R_equilibrium: float             # Equilibrium order parameter
    coupling_strength: float         # Kuramoto coupling K
    stability_eigenvalue: float      # Eigenvalue determining stability
    is_stable: bool                  # Whether fixed point is stable
    bifurcation_distance: float      # Distance to bifurcation

    @property
    def converging(self) -> bool:
        """Check if system is converging to this fixed point."""
        return self.is_stable and self.stability_eigenvalue < 0
