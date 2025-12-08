"""
CIC Theory Primitives
=====================

Compression-Integration-Causality: The mathematical foundation.

F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

Where:
    Φ(T) = Integrated Information (how much whole exceeds parts)
    H(T|X) = Representation Entropy (disorder/uncertainty)
    C_multi(T) = Multi-scale Causal Power

Intelligence = argmax F[T]
"""

import lzma
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional


def ncd(x: bytes, y: bytes) -> float:
    """
    Normalized Compression Distance - approximates Kolmogorov distance.

    NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

    Range: [0, 1] where 0 = identical, 1 = maximally different

    Key insight: NCD reveals algorithmic similarity, not surface similarity.
    """
    if not x or not y:
        return 1.0

    cx = len(lzma.compress(x))
    cy = len(lzma.compress(y))
    cxy = len(lzma.compress(x + y))

    denominator = max(cx, cy)
    if denominator == 0:
        return 0.0

    return (cxy - min(cx, cy)) / denominator


def phi_integrated_information(traces: List[str]) -> float:
    """
    Φ (Integrated Information) - how much the whole exceeds the parts.

    Computed as: 1 - mean(NCD between all pairs)

    High Φ = traces share deep structure (can't be partitioned)
    Low Φ = traces are independent (no integration)

    This is the "integration" term in CIC.
    """
    if len(traces) < 2:
        return 0.0

    trace_bytes = [t.encode('utf-8', errors='replace') for t in traces]
    ncds = []

    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            ncds.append(ncd(trace_bytes[i], trace_bytes[j]))

    if not ncds:
        return 0.0

    return 1.0 - statistics.mean(ncds)


def representation_entropy(samples: List[int]) -> float:
    """
    H(T|X) - entropy of internal representations.

    Approximated as normalized variance of answers.

    High H = high uncertainty/disorder (bad)
    Low H = crystallized/ordered (good)

    This is the "compression" term in CIC (we want to minimize it).
    """
    if len(samples) < 2:
        return 0.0

    # Normalize by mean to handle different scales
    mean_val = statistics.mean(samples) if samples else 1
    if mean_val == 0:
        mean_val = 1

    normalized = [s / abs(mean_val) for s in samples]
    variance = statistics.variance(normalized)

    # Map to [0, 1] range
    return min(1.0, variance)


def causal_power_multiscale(samples: List[int], target: Optional[int] = None) -> float:
    """
    C_multi(T) - multi-scale causal power.

    Measures ability to influence outcome across scales:
    - Scale 1: Exact match power (consensus)
    - Scale 2: Cluster coherence power (agreement within tolerance)
    - Scale 3: Range constraint power (inverse of spread)

    High C = strong causal influence (answers converge)
    Low C = weak causal influence (answers scatter)
    """
    if not samples:
        return 0.0

    n = len(samples)

    # Scale 1: Exact consensus power
    counter = Counter(samples)
    mode_count = counter.most_common(1)[0][1]
    exact_power = mode_count / n

    # Scale 2: Cluster coherence (within 5% of each other)
    def relative_distance(a: int, b: int) -> float:
        if a == b:
            return 0.0
        if a == 0 or b == 0:
            return 1.0
        return abs(a - b) / max(abs(a), abs(b))

    close_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if relative_distance(samples[i], samples[j]) < 0.05:
                close_pairs += 1

    cluster_power = close_pairs / total_pairs if total_pairs > 0 else 0

    # Scale 3: Range constraint (inverse of spread)
    spread = max(samples) - min(samples) if samples else 0
    center = abs(statistics.mean(samples)) if samples else 1
    range_power = 1.0 / (1.0 + spread / center) if center > 0 else 0

    # Combine scales with Fibonacci-derived weights
    weights = [0.5, 0.3, 0.2]
    C_multi = weights[0] * exact_power + weights[1] * cluster_power + weights[2] * range_power

    return C_multi


@dataclass
class CICState:
    """Complete CIC state for a reasoning system."""
    phi: float           # Integrated information
    entropy: float       # Representation entropy H(T|X)
    causal_power: float  # Multi-scale causal power C_multi
    F: float             # The CIC functional
    confidence: float    # Epistemic confidence (derived)

    def __repr__(self) -> str:
        return (
            f"CICState(Φ={self.phi:.3f}, H={self.entropy:.3f}, "
            f"C={self.causal_power:.3f}, F={self.F:.3f}, conf={self.confidence:.3f})"
        )


def compute_cic_functional(
    samples: List[int],
    traces: Optional[List[str]] = None,
    lambda_compress: float = 0.5,
    gamma_causal: float = 0.3,
) -> CICState:
    """
    Compute the CIC functional:

    F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

    This is the unified objective for intelligent systems.

    Args:
        samples: List of answer candidates
        traces: Optional reasoning traces for Φ computation
        lambda_compress: Weight for compression term (default 0.5)
        gamma_causal: Weight for causal power term (default 0.3)

    Returns:
        CICState with all computed values
    """
    # Compute Φ from traces if available, else from sample similarity
    if traces and len(traces) >= 2:
        phi = phi_integrated_information(traces)
    else:
        # Approximate Φ from answer clustering
        sample_strs = [str(s) for s in samples]
        phi = phi_integrated_information(sample_strs)

    # Compute entropy (disorder)
    H = representation_entropy(samples)

    # Compute causal power
    C_multi = causal_power_multiscale(samples)

    # The CIC functional
    F = phi - lambda_compress * H + gamma_causal * C_multi

    # Epistemic confidence emerges from F
    # High F = high confidence, Low F = low confidence
    # Map to [0.05, 0.95] range
    confidence = max(0.05, min(0.95, 0.5 + 0.5 * F))

    return CICState(
        phi=phi,
        entropy=H,
        causal_power=C_multi,
        F=F,
        confidence=confidence,
    )


def detect_crystallization(history: List[CICState], threshold: float = 0.1) -> bool:
    """
    Detect UIPT (Universal Information Phase Transition).

    Crystallization occurs when: dΦ/dt ≈ λ·dH/dt
    (compression and integration forces balance)

    When detected, the answer has "grokked" - stop generating.

    Args:
        history: List of CICState from successive generations
        threshold: Balance threshold for detection

    Returns:
        True if crystallization detected
    """
    if len(history) < 3:
        return False

    # Compute derivatives
    dphi = history[-1].phi - history[-2].phi
    dH = history[-1].entropy - history[-2].entropy

    # Phase transition: Φ increasing while H decreasing
    lambda_compress = 0.5
    balance = abs(dphi + lambda_compress * dH)

    # Check conditions
    phi_increasing = dphi > 0
    H_decreasing = dH < 0
    balanced = balance < threshold

    return phi_increasing and H_decreasing and balanced


# Convenience exports
__all__ = [
    "ncd",
    "phi_integrated_information",
    "representation_entropy",
    "causal_power_multiscale",
    "compute_cic_functional",
    "detect_crystallization",
    "CICState",
]
