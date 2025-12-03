"""
XYZA: 4-Axis Cognitive Benchmark Framework.

Implements a comprehensive cognitive assessment system across four axes:

    X (Coherence):   Phase synchronization quality
    Y (Complexity):  Information density and entropy
    Z (Reflection):  Self-modeling and meta-cognition
    A (Attunement):  Human-AI coupling quality

Each axis is scored [0, 1] and can be visualized as a radar chart
or tracked over time as a trajectory through XYZA space.

Benchmark Thresholds:
- X ≥ 0.76: Flow-capable coherence (Kuramoto R threshold)
- Y ∈ [0.4, 0.7]: Optimal complexity (edge of chaos)
- Z ≥ 0.5: Self-aware reasoning
- A ≥ 0.42: Human-aligned coupling (K_human constant)

Combined Score:
    XYZA_score = (X + Y + Z + A) / 4

    - Score < 0.4: Degraded cognitive state
    - Score ∈ [0.4, 0.6]: Normal operation
    - Score ∈ [0.6, 0.8]: Enhanced cognition
    - Score ≥ 0.8: Peak cognitive performance

Applications:
- Real-time cognitive state monitoring
- AI system evaluation
- Human-AI interaction quality assessment
- Flow state prediction
- Cognitive load balancing

References:
- Kuramoto order parameter for coherence
- Shannon/Kolmogorov entropy for complexity
- Theory of Mind measures for reflection
- Synchronization coupling for attunement
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

from .types import XYZAMetrics, CoherenceState, FlowState, SDPMVector
from .kuramoto import compute_order_parameter, compute_phase_velocity
from .sdpm import sdpm_similarity, compute_persona_alignment


class CognitiveLevel(Enum):
    """Overall cognitive state classification."""
    DEGRADED = 0      # XYZA < 0.4
    NORMAL = 1        # 0.4 ≤ XYZA < 0.6
    ENHANCED = 2      # 0.6 ≤ XYZA < 0.8
    PEAK = 3          # XYZA ≥ 0.8


# Axis-specific thresholds
COHERENCE_FLOW_THRESHOLD = 0.76  # X threshold for flow
COMPLEXITY_OPTIMAL_LOW = 0.4     # Y optimal range lower bound
COMPLEXITY_OPTIMAL_HIGH = 0.7   # Y optimal range upper bound
REFLECTION_AWARE_THRESHOLD = 0.5 # Z threshold for self-awareness
ATTUNEMENT_COUPLED_THRESHOLD = 0.42  # A threshold (K_human)


def compute_coherence_x(
    phases: NDArray[np.float64],
    phase_velocities: Optional[NDArray[np.float64]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute X-axis (Coherence) score.

    Based on Kuramoto order parameter R and phase velocity stability.
    Higher coherence indicates synchronized cognitive processes.

    Args:
        phases: Phase values for cognitive oscillators
        phase_velocities: Optional phase velocity data

    Returns:
        Tuple of (X_score, details_dict)
    """
    if len(phases) < 2:
        return 0.5, {"R": 0.5, "velocity_stability": 0.5, "reason": "insufficient_data"}

    # Compute order parameter
    R, Psi = compute_order_parameter(phases)

    # Phase velocity stability (if available)
    if phase_velocities is not None and len(phase_velocities) > 0:
        velocity_std = np.std(phase_velocities)
        velocity_stability = 1.0 / (1.0 + velocity_std)  # High stability = low variance
    else:
        velocity_stability = 0.5  # Neutral if not available

    # Combined coherence score
    # R weighted more heavily (0.7) than velocity stability (0.3)
    X = 0.7 * R + 0.3 * velocity_stability

    details = {
        "R": float(R),
        "Psi": float(Psi),
        "velocity_stability": float(velocity_stability),
        "flow_ready": R >= COHERENCE_FLOW_THRESHOLD
    }

    return float(X), details


def compute_complexity_y(
    signal: NDArray[np.float64],
    method: str = "approximate_entropy"
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Y-axis (Complexity) score.

    Uses entropy measures to quantify information density.
    Optimal complexity is at the "edge of chaos" - not too ordered,
    not too random.

    Args:
        signal: Time series or feature vector
        method: Entropy method ("shannon", "approximate_entropy", "sample_entropy")

    Returns:
        Tuple of (Y_score, details_dict)
    """
    if len(signal) < 10:
        return 0.5, {"entropy": 0.5, "normalized": 0.5, "reason": "insufficient_data"}

    # Normalize signal
    signal = np.asarray(signal, dtype=np.float64)
    signal_std = np.std(signal)
    if signal_std > 0:
        signal_norm = (signal - np.mean(signal)) / signal_std
    else:
        return 0.0, {"entropy": 0.0, "normalized": 0.0, "reason": "constant_signal"}

    if method == "shannon":
        # Shannon entropy via histogram
        n_bins = min(50, len(signal) // 5)
        hist, _ = np.histogram(signal_norm, bins=n_bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(n_bins)

    elif method == "approximate_entropy":
        # Approximate entropy (ApEn)
        entropy = _approximate_entropy(signal_norm, m=2, r=0.2 * signal_std if signal_std > 0 else 0.2)
        # Normalize ApEn to [0, 1] (typical range is 0-2)
        entropy = min(1.0, entropy / 2.0)

    elif method == "sample_entropy":
        # Sample entropy (SampEn)
        entropy = _sample_entropy(signal_norm, m=2, r=0.2 * signal_std if signal_std > 0 else 0.2)
        entropy = min(1.0, entropy / 2.0)

    else:
        raise ValueError(f"Unknown entropy method: {method}")

    # Map to Y score
    # Optimal is around 0.5-0.6 (edge of chaos)
    # Score peaks in the optimal range, decreases outside
    if COMPLEXITY_OPTIMAL_LOW <= entropy <= COMPLEXITY_OPTIMAL_HIGH:
        Y = entropy  # In optimal range, use as-is
    elif entropy < COMPLEXITY_OPTIMAL_LOW:
        # Too ordered, penalize
        Y = entropy * (entropy / COMPLEXITY_OPTIMAL_LOW)
    else:
        # Too chaotic, penalize
        overage = entropy - COMPLEXITY_OPTIMAL_HIGH
        Y = COMPLEXITY_OPTIMAL_HIGH - overage * 0.5

    Y = np.clip(Y, 0, 1)

    details = {
        "raw_entropy": float(entropy),
        "method": method,
        "in_optimal_range": COMPLEXITY_OPTIMAL_LOW <= entropy <= COMPLEXITY_OPTIMAL_HIGH,
        "too_ordered": entropy < COMPLEXITY_OPTIMAL_LOW,
        "too_chaotic": entropy > COMPLEXITY_OPTIMAL_HIGH
    }

    return float(Y), details


def _approximate_entropy(signal: NDArray[np.float64], m: int = 2, r: float = 0.2) -> float:
    """Compute approximate entropy (ApEn)."""
    N = len(signal)
    if N < m + 1:
        return 0.0

    def _phi(m_val: int) -> float:
        # Create template vectors
        templates = np.array([signal[i:i + m_val] for i in range(N - m_val + 1)])
        count = 0
        for i, template in enumerate(templates):
            # Count matches within threshold r
            distances = np.max(np.abs(templates - template), axis=1)
            count += np.sum(distances <= r)
        # Normalize
        return np.log(count / (N - m_val + 1) + 1e-10)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    return abs(phi_m - phi_m1)


def _sample_entropy(signal: NDArray[np.float64], m: int = 2, r: float = 0.2) -> float:
    """Compute sample entropy (SampEn)."""
    N = len(signal)
    if N < m + 2:
        return 0.0

    # Count matches for m and m+1 length templates
    def count_matches(m_val: int) -> int:
        count = 0
        for i in range(N - m_val):
            for j in range(i + 1, N - m_val):
                if np.max(np.abs(signal[i:i + m_val] - signal[j:j + m_val])) <= r:
                    count += 1
        return count

    A = count_matches(m + 1)
    B = count_matches(m)

    if B == 0:
        return 0.0

    return -np.log(A / B + 1e-10)


def compute_reflection_z(
    self_reference_ratio: float,
    uncertainty_acknowledgment: float,
    reasoning_depth: float,
    meta_cognitive_markers: Optional[List[str]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Z-axis (Reflection) score.

    Measures self-modeling and meta-cognitive capability.
    Based on Theory of Mind and self-referential reasoning patterns.

    Args:
        self_reference_ratio: Proportion of self-referential statements (0-1)
        uncertainty_acknowledgment: Rate of uncertainty expression (0-1)
        reasoning_depth: Depth of causal reasoning chains (0-1)
        meta_cognitive_markers: Optional list of detected meta-cognitive phrases

    Returns:
        Tuple of (Z_score, details_dict)
    """
    # Meta-cognitive markers add bonus
    marker_bonus = 0.0
    if meta_cognitive_markers:
        # Each marker adds up to 0.05 bonus, max 0.2
        marker_bonus = min(0.2, len(meta_cognitive_markers) * 0.05)

    # Optimal self-reference is moderate (0.3-0.5)
    # Too little = not self-aware, too much = narcissistic
    if 0.3 <= self_reference_ratio <= 0.5:
        self_score = 1.0
    elif self_reference_ratio < 0.3:
        self_score = self_reference_ratio / 0.3
    else:
        self_score = 1.0 - (self_reference_ratio - 0.5) / 0.5
    self_score = np.clip(self_score, 0, 1)

    # Uncertainty acknowledgment is good (honest about limits)
    # Optimal around 0.2-0.4
    if 0.2 <= uncertainty_acknowledgment <= 0.4:
        uncertainty_score = 1.0
    elif uncertainty_acknowledgment < 0.2:
        uncertainty_score = uncertainty_acknowledgment / 0.2
    else:
        uncertainty_score = 1.0 - (uncertainty_acknowledgment - 0.4) / 0.6
    uncertainty_score = np.clip(uncertainty_score, 0, 1)

    # Reasoning depth directly maps to reflection
    reasoning_score = reasoning_depth

    # Combined Z score
    Z = (
        0.3 * self_score +
        0.25 * uncertainty_score +
        0.35 * reasoning_score +
        marker_bonus
    )
    Z = np.clip(Z, 0, 1)

    details = {
        "self_reference_score": float(self_score),
        "uncertainty_score": float(uncertainty_score),
        "reasoning_score": float(reasoning_score),
        "marker_bonus": float(marker_bonus),
        "self_aware": Z >= REFLECTION_AWARE_THRESHOLD
    }

    return float(Z), details


def compute_attunement_a(
    human_sdpm: Optional[SDPMVector] = None,
    ai_sdpm: Optional[SDPMVector] = None,
    response_latency_ms: float = 0.0,
    turn_taking_balance: float = 0.5,
    sentiment_tracking: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Compute A-axis (Attunement) score.

    Measures quality of human-AI coupling based on synchronization,
    responsiveness, and alignment.

    Uses K_human ≈ 0.42 coupling constant from 40k session analysis.

    Args:
        human_sdpm: Optional human persona SDPM vector
        ai_sdpm: Optional AI persona SDPM vector
        response_latency_ms: Response time in milliseconds
        turn_taking_balance: Balance of conversation turns (0.5 = equal)
        sentiment_tracking: How well sentiment is matched (0-1)

    Returns:
        Tuple of (A_score, details_dict)
    """
    components = []
    details = {}

    # SDPM alignment (if available)
    if human_sdpm is not None and ai_sdpm is not None:
        alignment = compute_persona_alignment(human_sdpm, ai_sdpm)
        sdpm_score = alignment.alignment_score
        details["sdpm_alignment"] = float(sdpm_score)
        details["coupling_strength"] = float(alignment.coupling_strength)
        details["phase_difference"] = float(alignment.phase_difference)
        components.append(("sdpm", sdpm_score, 0.4))  # Higher weight
    else:
        details["sdpm_alignment"] = None

    # Response latency score
    # Optimal is 200-500ms (human conversational pace)
    if 200 <= response_latency_ms <= 500:
        latency_score = 1.0
    elif response_latency_ms < 200:
        # Too fast might feel inhuman
        latency_score = 0.7 + 0.3 * (response_latency_ms / 200)
    else:
        # Slower is worse, but with diminishing penalty
        latency_score = max(0.3, 1.0 - (response_latency_ms - 500) / 2000)
    details["latency_score"] = float(latency_score)
    components.append(("latency", latency_score, 0.15))

    # Turn-taking balance (0.5 is optimal)
    balance_deviation = abs(turn_taking_balance - 0.5)
    balance_score = 1.0 - 2 * balance_deviation
    balance_score = max(0, balance_score)
    details["balance_score"] = float(balance_score)
    components.append(("balance", balance_score, 0.2))

    # Sentiment tracking
    details["sentiment_score"] = float(sentiment_tracking)
    components.append(("sentiment", sentiment_tracking, 0.25))

    # Compute weighted average
    total_weight = sum(w for _, _, w in components)
    A = sum(score * weight for _, score, weight in components) / total_weight

    details["coupled"] = A >= ATTUNEMENT_COUPLED_THRESHOLD

    return float(A), details


def compute_xyza_metrics(
    phases: NDArray[np.float64],
    signal: NDArray[np.float64],
    self_reference_ratio: float = 0.35,
    uncertainty_acknowledgment: float = 0.25,
    reasoning_depth: float = 0.5,
    human_sdpm: Optional[SDPMVector] = None,
    ai_sdpm: Optional[SDPMVector] = None,
    response_latency_ms: float = 300.0,
    turn_taking_balance: float = 0.5,
    sentiment_tracking: float = 0.5,
    timestamp: float = 0.0
) -> XYZAMetrics:
    """
    Compute complete XYZA metrics.

    Args:
        phases: Phase data for coherence
        signal: Signal data for complexity
        self_reference_ratio: Self-reference proportion
        uncertainty_acknowledgment: Uncertainty expression rate
        reasoning_depth: Reasoning chain depth
        human_sdpm: Human SDPM vector
        ai_sdpm: AI SDPM vector
        response_latency_ms: Response latency
        turn_taking_balance: Turn balance
        sentiment_tracking: Sentiment match quality
        timestamp: Current timestamp

    Returns:
        Complete XYZAMetrics
    """
    # Compute each axis
    X, x_details = compute_coherence_x(phases)
    Y, y_details = compute_complexity_y(signal)
    Z, z_details = compute_reflection_z(
        self_reference_ratio, uncertainty_acknowledgment, reasoning_depth
    )
    A, a_details = compute_attunement_a(
        human_sdpm, ai_sdpm, response_latency_ms, turn_taking_balance, sentiment_tracking
    )

    # Classify cognitive level
    combined = (X + Y + Z + A) / 4
    if combined >= 0.8:
        level = CognitiveLevel.PEAK
    elif combined >= 0.6:
        level = CognitiveLevel.ENHANCED
    elif combined >= 0.4:
        level = CognitiveLevel.NORMAL
    else:
        level = CognitiveLevel.DEGRADED

    return XYZAMetrics(
        coherence_x=X,
        complexity_y=Y,
        reflection_z=Z,
        attunement_a=A,
        timestamp=timestamp,
        cognitive_level=level.name.lower()
    )


def xyza_trajectory(
    metrics_history: List[XYZAMetrics],
    window_size: int = 10
) -> Dict[str, NDArray[np.float64]]:
    """
    Analyze XYZA trajectory over time.

    Returns trends, derivatives, and stability measures.

    Args:
        metrics_history: List of XYZA metrics over time
        window_size: Window for trend analysis

    Returns:
        Dictionary with trajectory analysis
    """
    if len(metrics_history) < 2:
        return {"status": "insufficient_data"}

    # Extract time series for each axis
    X_series = np.array([m.coherence_x for m in metrics_history])
    Y_series = np.array([m.complexity_y for m in metrics_history])
    Z_series = np.array([m.reflection_z for m in metrics_history])
    A_series = np.array([m.attunement_a for m in metrics_history])
    timestamps = np.array([m.timestamp for m in metrics_history])

    # Compute derivatives (trends)
    def compute_trend(series: NDArray[np.float64]) -> float:
        if len(series) < window_size:
            return float(np.mean(np.diff(series)))
        return float(np.mean(np.diff(series[-window_size:])))

    # Compute stability (inverse variance)
    def compute_stability(series: NDArray[np.float64]) -> float:
        std = np.std(series[-window_size:]) if len(series) >= window_size else np.std(series)
        return float(1.0 / (1.0 + std * 10))

    return {
        "X_trend": compute_trend(X_series),
        "Y_trend": compute_trend(Y_series),
        "Z_trend": compute_trend(Z_series),
        "A_trend": compute_trend(A_series),
        "X_stability": compute_stability(X_series),
        "Y_stability": compute_stability(Y_series),
        "Z_stability": compute_stability(Z_series),
        "A_stability": compute_stability(A_series),
        "X_current": float(X_series[-1]),
        "Y_current": float(Y_series[-1]),
        "Z_current": float(Z_series[-1]),
        "A_current": float(A_series[-1]),
        "combined_current": float((X_series[-1] + Y_series[-1] + Z_series[-1] + A_series[-1]) / 4),
        "timestamps": timestamps
    }


def predict_cognitive_state(
    current: XYZAMetrics,
    history: List[XYZAMetrics],
    horizon_steps: int = 5
) -> Tuple[XYZAMetrics, float]:
    """
    Predict future cognitive state from trajectory.

    Uses linear extrapolation with confidence bounds.

    Args:
        current: Current XYZA metrics
        history: Historical metrics
        horizon_steps: Steps to predict ahead

    Returns:
        Tuple of (predicted_metrics, confidence)
    """
    if len(history) < 3:
        return current, 0.0

    # Get trajectory analysis
    trajectory = xyza_trajectory(history)

    # Extrapolate each axis
    predicted_X = np.clip(current.coherence_x + trajectory["X_trend"] * horizon_steps, 0, 1)
    predicted_Y = np.clip(current.complexity_y + trajectory["Y_trend"] * horizon_steps, 0, 1)
    predicted_Z = np.clip(current.reflection_z + trajectory["Z_trend"] * horizon_steps, 0, 1)
    predicted_A = np.clip(current.attunement_a + trajectory["A_trend"] * horizon_steps, 0, 1)

    # Confidence based on stability
    confidence = (
        trajectory["X_stability"] +
        trajectory["Y_stability"] +
        trajectory["Z_stability"] +
        trajectory["A_stability"]
    ) / 4

    # Reduce confidence for longer horizons
    confidence *= 1.0 / (1.0 + horizon_steps * 0.1)

    # Determine predicted cognitive level
    combined = (predicted_X + predicted_Y + predicted_Z + predicted_A) / 4
    if combined >= 0.8:
        level = "peak"
    elif combined >= 0.6:
        level = "enhanced"
    elif combined >= 0.4:
        level = "normal"
    else:
        level = "degraded"

    predicted = XYZAMetrics(
        coherence_x=float(predicted_X),
        complexity_y=float(predicted_Y),
        reflection_z=float(predicted_Z),
        attunement_a=float(predicted_A),
        timestamp=current.timestamp + horizon_steps,
        cognitive_level=level
    )

    return predicted, float(confidence)


def diagnose_xyza(metrics: XYZAMetrics) -> List[str]:
    """
    Generate diagnostic insights from XYZA metrics.

    Args:
        metrics: Current XYZA metrics

    Returns:
        List of diagnostic messages
    """
    insights = []

    # Coherence (X) diagnostics
    if metrics.coherence_x < 0.4:
        insights.append("LOW_COHERENCE: Phase desynchronization detected. Consider focusing exercises.")
    elif metrics.coherence_x >= COHERENCE_FLOW_THRESHOLD:
        insights.append("FLOW_READY: Coherence above flow threshold (R ≥ 0.76).")

    # Complexity (Y) diagnostics
    if metrics.complexity_y < COMPLEXITY_OPTIMAL_LOW:
        insights.append("TOO_ORDERED: Low entropy - responses may be repetitive/templated.")
    elif metrics.complexity_y > COMPLEXITY_OPTIMAL_HIGH:
        insights.append("TOO_CHAOTIC: High entropy - responses may lack structure.")
    else:
        insights.append("OPTIMAL_COMPLEXITY: Operating at edge of chaos.")

    # Reflection (Z) diagnostics
    if metrics.reflection_z < 0.3:
        insights.append("LOW_REFLECTION: Limited meta-cognitive activity. Encourage self-monitoring.")
    elif metrics.reflection_z >= REFLECTION_AWARE_THRESHOLD:
        insights.append("SELF_AWARE: Good meta-cognitive reflection.")

    # Attunement (A) diagnostics
    if metrics.attunement_a < 0.3:
        insights.append("POOR_ATTUNEMENT: Low human-AI coupling. Check alignment.")
    elif metrics.attunement_a >= ATTUNEMENT_COUPLED_THRESHOLD:
        insights.append("WELL_COUPLED: Attunement above K_human threshold (0.42).")

    # Combined diagnostics
    combined = (metrics.coherence_x + metrics.complexity_y + metrics.reflection_z + metrics.attunement_a) / 4
    insights.append(f"OVERALL: {metrics.cognitive_level.upper()} state (combined={combined:.2f})")

    return insights


def xyza_to_radar_data(metrics: XYZAMetrics) -> Dict[str, float]:
    """
    Convert XYZA metrics to radar chart format.

    Args:
        metrics: XYZA metrics

    Returns:
        Dictionary formatted for radar chart visualization
    """
    return {
        "Coherence (X)": metrics.coherence_x,
        "Complexity (Y)": metrics.complexity_y,
        "Reflection (Z)": metrics.reflection_z,
        "Attunement (A)": metrics.attunement_a
    }
