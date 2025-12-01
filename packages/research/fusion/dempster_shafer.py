"""
Reliability-Weighted Dempster-Shafer Fusion.
Based on US6944566B2 (Lockheed Martin, expired April 2023).

This module implements modified Dempster-Shafer theory for multi-source
fusion where sources have varying reliability/quality metrics.

Key innovation from patent:
- Explicit noise belief term in each sensor's mass function
- SNR-dependent reliability functions per sensor
- Additive fusion over reliability functions (not multiplicative DS)
- Avoids catastrophic down-weighting when many sensors are weak
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.types import SourceSignal, FusedBelief, FusionMethod


@dataclass
class ReliabilityConfig:
    """Configuration for reliability mapping."""
    alpha: float = 2.0  # Logistic steepness
    beta: float = 0.5  # Logistic midpoint (quality threshold)
    min_reliability: float = 0.01  # Floor to prevent zero mass
    max_reliability: float = 0.99  # Ceiling


def logistic_reliability(
    quality: float,
    config: ReliabilityConfig = ReliabilityConfig()
) -> float:
    """
    Map source quality metric to reliability using logistic function.

    From US6944566B2: "Define a reliability function r_j(t) = f(q_j(t))"
    where f is monotone increasing.

    Args:
        quality: Source quality metric (e.g., data completeness, model confidence)
        config: Reliability mapping parameters

    Returns:
        Reliability in [min_reliability, max_reliability]
    """
    r = 1.0 / (1.0 + np.exp(-config.alpha * (quality - config.beta)))
    return np.clip(r, config.min_reliability, config.max_reliability)


def compute_mass_function(
    probabilities: NDArray[np.float64],
    reliability: float
) -> Tuple[NDArray[np.float64], float]:
    """
    Compute basic probability assignment (BPA) including noise mass.

    From US6944566B2:
    m_j(H_k; t) = r_j(t) * p_j(H_k | x_j(t))
    m_j(noise; t) = 1 - r_j(t)

    Args:
        probabilities: Probability distribution over hypotheses, shape (K,)
        reliability: Source reliability in [0, 1]

    Returns:
        Tuple of (hypothesis masses, noise mass)
    """
    masses = reliability * probabilities
    noise_mass = 1.0 - reliability
    return masses, noise_mass


def additive_fusion(
    source_probs: List[NDArray[np.float64]],
    reliabilities: List[float],
    hypothesis_names: List[str]
) -> FusedBelief:
    """
    Reliability-weighted additive fusion.

    From US6944566B2:
    R_fused(H_k; t) = Σ_j w_j(t) * R_j(H_k; t)
    where w_j(t) = r_j(t) / Σ_l r_l(t)

    This avoids the catastrophic behavior of multiplicative Dempster's
    rule when many sources have low reliability.

    Args:
        source_probs: List of probability distributions, each shape (K,)
        reliabilities: List of reliability scores
        hypothesis_names: Names of hypotheses

    Returns:
        FusedBelief with combined probabilities
    """
    J = len(source_probs)
    K = len(source_probs[0])

    r = np.array(reliabilities)
    R = np.zeros((J, K))

    # Compute reliability-weighted masses
    for j in range(J):
        masses, _ = compute_mass_function(source_probs[j], r[j])
        R[j, :] = masses

    # Degenerate case: all sources have zero reliability
    if r.sum() < 1e-10:
        uniform = np.ones(K) / K
        return FusedBelief(
            hypotheses=hypothesis_names,
            probabilities=uniform,
            reliabilities={f"source_{j}": r[j] for j in range(J)},
            method_used=FusionMethod.ADDITIVE,
            confidence=0.0
        )

    # Weights proportional to reliability
    w = r / r.sum()

    # Fused reliability
    R_fused = (w[:, None] * R).sum(axis=0)

    # Normalize to probabilities
    p_fused = R_fused / R_fused.sum() if R_fused.sum() > 0 else np.ones(K) / K

    # Confidence: how concentrated is the belief?
    confidence = 1.0 - (-np.sum(p_fused * np.log2(p_fused + 1e-10)) / np.log2(K))

    return FusedBelief(
        hypotheses=hypothesis_names,
        probabilities=p_fused,
        reliabilities={f"source_{j}": float(r[j]) for j in range(J)},
        method_used=FusionMethod.ADDITIVE,
        confidence=float(confidence)
    )


def multiplicative_fusion(
    source_probs: List[NDArray[np.float64]],
    reliabilities: List[float],
    hypothesis_names: List[str]
) -> FusedBelief:
    """
    Reliability-weighted multiplicative fusion.

    From US6909997B2:
    S_mult(H_k; t) = Π_j p_j(H_k | x_j(t))^{r_j(t)}

    Uses exponentiation by reliability to down-weight unreliable sources.
    Can suffer when many sources are weak (product becomes tiny).

    Args:
        source_probs: List of probability distributions
        reliabilities: List of reliability scores
        hypothesis_names: Names of hypotheses

    Returns:
        FusedBelief with combined probabilities
    """
    J = len(source_probs)
    K = len(source_probs[0])

    r = np.array(reliabilities)

    # Log-space computation for numerical stability
    log_S = np.zeros(K)
    for j in range(J):
        p_j = np.clip(source_probs[j], 1e-12, 1.0)
        log_S += r[j] * np.log(p_j)

    # Normalize
    S = np.exp(log_S - log_S.max())  # Subtract max for stability
    p_fused = S / S.sum()

    confidence = 1.0 - (-np.sum(p_fused * np.log2(p_fused + 1e-10)) / np.log2(K))

    return FusedBelief(
        hypotheses=hypothesis_names,
        probabilities=p_fused,
        reliabilities={f"source_{j}": float(r[j]) for j in range(J)},
        method_used=FusionMethod.MULTIPLICATIVE,
        confidence=float(confidence)
    )


class MetaFusionSelector:
    """
    Dynamically selects best fusion method based on performance.
    Based on US6909997B2 (Lockheed Martin, expired August 2023).

    Runs multiple fusion methods and selects the one with best
    recent performance (e.g., highest log-likelihood on held-out data).

    NOVEL INSIGHT #5 from Great Attractor framework:
    The signal isn't one attractor strengthening - it's the relative
    shift between competing fusion strategies. When multiplicative
    starts outperforming additive, it indicates stronger consensus
    forming (single attractor dominating).
    """

    def __init__(
        self,
        methods: List[FusionMethod] = None,
        window_size: int = 20
    ):
        self.methods = methods or [FusionMethod.ADDITIVE, FusionMethod.MULTIPLICATIVE]
        self.window_size = window_size
        self.performance_history: Dict[FusionMethod, List[float]] = {
            m: [] for m in self.methods
        }

    def _score_method(
        self,
        method: FusionMethod,
        source_probs: List[NDArray[np.float64]],
        reliabilities: List[float],
        hypothesis_names: List[str]
    ) -> Tuple[FusedBelief, float]:
        """Apply fusion method and compute quality score."""
        if method == FusionMethod.ADDITIVE:
            belief = additive_fusion(source_probs, reliabilities, hypothesis_names)
        elif method == FusionMethod.MULTIPLICATIVE:
            belief = multiplicative_fusion(source_probs, reliabilities, hypothesis_names)
        else:
            # Default to additive
            belief = additive_fusion(source_probs, reliabilities, hypothesis_names)

        # Score: confidence weighted by average reliability
        avg_reliability = np.mean(reliabilities)
        score = belief.confidence * avg_reliability

        return belief, score

    def fuse(
        self,
        source_probs: List[NDArray[np.float64]],
        reliabilities: List[float],
        hypothesis_names: List[str],
        realized_outcome: Optional[int] = None
    ) -> FusedBelief:
        """
        Perform meta-fusion: run all methods, select best.

        If realized_outcome is provided, update performance history.

        Args:
            source_probs: List of probability distributions
            reliabilities: List of reliability scores
            hypothesis_names: Hypothesis names
            realized_outcome: Index of realized hypothesis (for learning)

        Returns:
            Best FusedBelief according to selection criterion
        """
        candidates = {}
        scores = {}

        for method in self.methods:
            belief, score = self._score_method(
                method, source_probs, reliabilities, hypothesis_names
            )
            candidates[method] = belief

            # Use historical performance if available
            history = self.performance_history[method]
            if len(history) >= self.window_size:
                historical_score = np.mean(history[-self.window_size:])
                scores[method] = 0.7 * historical_score + 0.3 * score
            else:
                scores[method] = score

        # Select best method
        best_method = max(scores, key=scores.get)
        best_belief = candidates[best_method]

        # Update performance history if we have feedback
        if realized_outcome is not None:
            for method, belief in candidates.items():
                # Log probability of realized outcome
                log_prob = np.log(belief.probabilities[realized_outcome] + 1e-10)
                self.performance_history[method].append(float(log_prob))

        return best_belief

    def get_attractor_dominance_signal(self) -> float:
        """
        NOVEL INSIGHT #5: Relative shift between fusion strategies.

        When multiplicative outperforms additive significantly,
        it indicates strong consensus (single attractor dominating).
        When they're equal, multiple competing attractors.

        Returns:
            Dominance signal in [-1, 1]: positive = single attractor,
            negative = competing attractors
        """
        add_hist = self.performance_history.get(FusionMethod.ADDITIVE, [])
        mult_hist = self.performance_history.get(FusionMethod.MULTIPLICATIVE, [])

        if len(add_hist) < 5 or len(mult_hist) < 5:
            return 0.0

        add_recent = np.mean(add_hist[-10:])
        mult_recent = np.mean(mult_hist[-10:])

        # Tanh to bound in [-1, 1]
        return float(np.tanh(mult_recent - add_recent))


def fuse_sources(
    signals: List[SourceSignal],
    models: List[Callable[[NDArray], NDArray]],
    hypothesis_names: List[str],
    reliability_config: ReliabilityConfig = ReliabilityConfig(),
    method: FusionMethod = FusionMethod.ADDITIVE
) -> List[FusedBelief]:
    """
    Fuse multiple source signals over time.

    Args:
        signals: List of source signals
        models: List of model functions, each maps signal values to probabilities
        hypothesis_names: Names of hypotheses (regimes)
        reliability_config: Configuration for reliability mapping
        method: Fusion method to use

    Returns:
        List of FusedBelief objects, one per timestep
    """
    T = min(s.T for s in signals)
    results = []

    for t in range(T):
        source_probs = []
        reliabilities = []

        for signal, model in zip(signals, models):
            # Get probability distribution from model
            if signal.values.ndim == 1:
                x = signal.values[t:t+1]
            else:
                x = signal.values[t:t+1, :]
            probs = model(x)
            source_probs.append(probs.flatten())

            # Map quality to reliability
            q = signal.quality[t]
            r = logistic_reliability(q, reliability_config)
            reliabilities.append(r)

        # Fuse
        if method == FusionMethod.ADDITIVE:
            belief = additive_fusion(source_probs, reliabilities, hypothesis_names)
        elif method == FusionMethod.MULTIPLICATIVE:
            belief = multiplicative_fusion(source_probs, reliabilities, hypothesis_names)
        else:
            belief = additive_fusion(source_probs, reliabilities, hypothesis_names)

        results.append(belief)

    return results
