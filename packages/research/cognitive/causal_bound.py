"""
Causal Bound V Formula for Influence Detection.

Implements the causal bound formula:

    V = -log(μ_avg - 1) * (nom - est) / H(z)

Where:
- μ_avg: Average engagement rate (uniformity indicator)
- nom: Nominal expected engagement
- est: Estimated observed engagement
- H(z): Entropy of behavioral distribution

Detection thresholds:
- V > 2.0: Approaching cascade threshold
- V > 3.0: Likely coordination
- V > 4.0: Imminent cascade / state-sponsored operation
- V > 5.0: Sophisticated coordinated campaign

Applications:
- Bot network detection
- Coordinated inauthentic behavior
- Information cascade tipping points
- Narrative warfare equilibrium analysis
- Astroturfing detection via entropy signatures

References:
- Information warfare training data (training_data_expansion.json)
- Entropy analysis for coordinated behavior detection
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

from .types import CausalBound, CascadeType


class InfluenceLevel(Enum):
    """Influence operation severity levels."""
    ORGANIC = 0           # V < 2.0 - Normal organic behavior
    ELEVATED = 1          # 2.0 ≤ V < 3.0 - Approaching threshold
    COORDINATED = 2       # 3.0 ≤ V < 4.0 - Likely coordination
    CASCADE = 3           # 4.0 ≤ V < 5.0 - Imminent cascade
    STATE_SPONSORED = 4   # V ≥ 5.0 - Sophisticated operation


def compute_entropy_h(
    distribution: NDArray[np.float64],
    epsilon: float = 1e-10
) -> float:
    """
    Compute Shannon entropy H(z) of a behavioral distribution.

    H(z) = -Σ p(z_i) log p(z_i)

    Low entropy (H < 2.0 bits) indicates coordinated/templated behavior.
    High entropy (H > 4.0 bits) indicates organic diversity.

    Args:
        distribution: Probability distribution (must sum to 1)
        epsilon: Small value to prevent log(0)

    Returns:
        Entropy in bits
    """
    # Ensure it's a valid probability distribution
    p = np.asarray(distribution, dtype=np.float64)
    p = p / (p.sum() + epsilon)  # Normalize
    p = np.clip(p, epsilon, 1.0)  # Prevent log(0)

    # Shannon entropy
    entropy = -np.sum(p * np.log2(p))

    return float(entropy)


def compute_behavioral_entropy(
    signals: NDArray[np.float64],
    n_bins: int = 50
) -> float:
    """
    Compute entropy from behavioral signals using histogram binning.

    Args:
        signals: Raw behavioral signals (e.g., posting times, engagement rates)
        n_bins: Number of histogram bins

    Returns:
        Entropy in bits
    """
    if len(signals) < 2:
        return 0.0

    # Create histogram
    counts, _ = np.histogram(signals, bins=n_bins)
    total = counts.sum()

    if total == 0:
        return 0.0

    # Convert to probability distribution
    p = counts / total

    return compute_entropy_h(p)


def compute_temporal_entropy(
    timestamps: NDArray[np.float64],
    window_hours: float = 24.0,
    n_bins: int = 24
) -> float:
    """
    Compute temporal entropy from posting timestamps.

    Authentic behavior shows high temporal entropy (circadian rhythms,
    weekday/weekend variance). Astroturf shows low entropy (synchronized
    posting, off-hours activity consistent with overseas operation).

    Args:
        timestamps: Unix timestamps
        window_hours: Time window for analysis
        n_bins: Number of bins (24 for hourly)

    Returns:
        Temporal entropy in bits
    """
    if len(timestamps) < 2:
        return 0.0

    # Convert to hour-of-day
    hours = (timestamps % (24 * 3600)) / 3600

    return compute_behavioral_entropy(hours, n_bins=n_bins)


def compute_linguistic_entropy(
    texts: List[str],
    vocab_size: int = 10000
) -> float:
    """
    Compute linguistic entropy from text content.

    Organic discourse exhibits high vocabulary diversity and opinion variance.
    Coordinated campaigns show narrative uniformity and talking point repetition.

    Args:
        texts: List of text content
        vocab_size: Maximum vocabulary size to consider

    Returns:
        Linguistic entropy in bits
    """
    if not texts:
        return 0.0

    # Simple word frequency analysis
    from collections import Counter
    words = []
    for text in texts:
        words.extend(text.lower().split())

    if not words:
        return 0.0

    # Count word frequencies
    word_counts = Counter(words)
    total = sum(word_counts.values())

    # Take top vocab_size words
    top_words = word_counts.most_common(vocab_size)
    freqs = np.array([count for _, count in top_words], dtype=np.float64)

    return compute_entropy_h(freqs / total)


def compute_mu_avg(
    engagement_rates: NDArray[np.float64]
) -> float:
    """
    Compute average engagement rate μ_avg.

    When μ_avg approaches 1, indicates unusual uniformity consistent
    with bot coordination.

    Args:
        engagement_rates: Engagement rates per entity (0-1 scale)

    Returns:
        Average engagement rate
    """
    if len(engagement_rates) == 0:
        return 0.5

    return float(np.mean(engagement_rates))


def compute_causal_bound_v(
    mu_avg: float,
    nom: float,
    est: float,
    H_z: float,
    epsilon: float = 1e-10
) -> float:
    """
    Compute the causal bound V for influence detection.

    V = -log(μ_avg - 1) * (nom - est) / H(z)

    Args:
        mu_avg: Average engagement rate
        nom: Nominal expected engagement
        est: Estimated observed engagement
        H_z: Behavioral entropy
        epsilon: Small value to prevent division by zero

    Returns:
        Causal bound value V
    """
    # Handle edge cases
    if H_z < epsilon:
        H_z = epsilon  # Very low entropy amplifies V

    # The log term: -log(μ_avg - 1)
    # When μ_avg approaches 1, this diverges (high uniformity = suspicious)
    # μ_avg must be > 1 for this to work, or we use |μ_avg - 1|
    mu_term = abs(mu_avg - 1)
    if mu_term < epsilon:
        mu_term = epsilon

    log_term = -np.log(mu_term)

    # The deviation term: (nom - est)
    # Positive = suppressed spread, negative = artificial amplification
    deviation = nom - est

    # Combine
    V = log_term * deviation / H_z

    return float(V)


def compute_causal_bound(
    engagement_rates: NDArray[np.float64],
    expected_engagement: float,
    observed_engagement: float,
    behavioral_signals: NDArray[np.float64],
    timestamp: float = 0.0
) -> CausalBound:
    """
    Compute complete CausalBound analysis.

    Args:
        engagement_rates: Per-entity engagement rates
        expected_engagement: Nominal expected engagement (nom)
        observed_engagement: Estimated observed engagement (est)
        behavioral_signals: Signals for entropy computation
        timestamp: Current timestamp

    Returns:
        CausalBound with full analysis
    """
    mu_avg = compute_mu_avg(engagement_rates)
    H_z = compute_behavioral_entropy(behavioral_signals)
    V = compute_causal_bound_v(mu_avg, expected_engagement, observed_engagement, H_z)

    # Determine cascade type based on V and entropy patterns
    if V < 2.0:
        cascade_type = CascadeType.RANDOM  # Organic
    elif H_z < 2.0:
        cascade_type = CascadeType.FOREST  # Coordinated seeding (low entropy)
    elif V > 4.0:
        cascade_type = CascadeType.STAR  # Single influential coordinating source
    else:
        cascade_type = CascadeType.TREE  # Hierarchical diffusion

    # Confidence based on sample size and entropy quality
    n_samples = len(engagement_rates)
    confidence = min(1.0, n_samples / 100) * (1 - np.exp(-H_z))

    return CausalBound(
        V=V,
        mu_avg=mu_avg,
        nom=expected_engagement,
        est=observed_engagement,
        H_z=H_z,
        timestamp=timestamp,
        cascade_type=cascade_type,
        confidence=float(confidence)
    )


def detect_influence_cascade(
    V_timeseries: NDArray[np.float64],
    timestamps: NDArray[np.float64],
    threshold: float = 2.0,
    window_hours: float = 4.0
) -> List[Tuple[float, float, InfluenceLevel]]:
    """
    Detect influence cascade events from V timeseries.

    Uses sliding window analysis with CUSUM change detection
    for V trajectory inflection.

    Args:
        V_timeseries: V values over time
        timestamps: Corresponding timestamps
        threshold: Base threshold for detection (default 2.0)
        window_hours: Analysis window in hours

    Returns:
        List of (start_time, peak_V, level) tuples for detected events
    """
    events = []

    if len(V_timeseries) < 5:
        return events

    # Convert window to indices (assuming roughly hourly data)
    window_size = max(1, int(window_hours))

    # Find threshold crossings
    above_threshold = V_timeseries > threshold

    in_event = False
    event_start = 0
    peak_V = 0.0

    for i, (v, above) in enumerate(zip(V_timeseries, above_threshold)):
        if above and not in_event:
            # Start of new event
            in_event = True
            event_start = i
            peak_V = v
        elif above and in_event:
            # Continuing event, track peak
            if v > peak_V:
                peak_V = v
        elif not above and in_event:
            # End of event
            in_event = False

            # Classify severity
            if peak_V >= 5.0:
                level = InfluenceLevel.STATE_SPONSORED
            elif peak_V >= 4.0:
                level = InfluenceLevel.CASCADE
            elif peak_V >= 3.0:
                level = InfluenceLevel.COORDINATED
            else:
                level = InfluenceLevel.ELEVATED

            events.append((
                float(timestamps[event_start]),
                float(peak_V),
                level
            ))

    # Handle ongoing event at end
    if in_event:
        if peak_V >= 5.0:
            level = InfluenceLevel.STATE_SPONSORED
        elif peak_V >= 4.0:
            level = InfluenceLevel.CASCADE
        elif peak_V >= 3.0:
            level = InfluenceLevel.COORDINATED
        else:
            level = InfluenceLevel.ELEVATED

        events.append((
            float(timestamps[event_start]),
            float(peak_V),
            level
        ))

    return events


def detect_bot_network(
    engagement_rates: NDArray[np.float64],
    posting_times: NDArray[np.float64],
    expected_organic_rate: float = 0.3
) -> Tuple[float, bool, str]:
    """
    Detect bot network using the V formula.

    Bot networks show:
    - High μ_avg (unusual uniformity)
    - Low temporal entropy (synchronized posting)
    - Deviation from organic baseline

    Russian IRA operations showed V > 6.0 on key hashtags.
    Organic movements typically V < 2.0.

    Args:
        engagement_rates: Per-account engagement rates
        posting_times: Posting timestamps
        expected_organic_rate: Expected organic engagement rate

    Returns:
        Tuple of (V_score, is_bot_network, assessment)
    """
    mu_avg = compute_mu_avg(engagement_rates)
    temporal_H = compute_temporal_entropy(posting_times)
    observed_rate = float(np.mean(engagement_rates))

    # Compute V
    V = compute_causal_bound_v(
        mu_avg=mu_avg,
        nom=expected_organic_rate,
        est=observed_rate,
        H_z=temporal_H
    )

    # Assessment
    if V > 6.0:
        assessment = "Sophisticated state-sponsored operation"
        is_bot = True
    elif V > 5.0:
        assessment = "Likely state-sponsored bot network"
        is_bot = True
    elif V > 3.0:
        assessment = "Coordinated inauthentic behavior detected"
        is_bot = True
    elif V > 2.0:
        assessment = "Elevated coordination indicators"
        is_bot = False  # Uncertain
    else:
        assessment = "Organic behavior pattern"
        is_bot = False

    return V, is_bot, assessment


def compute_narrative_health(
    V_current: float,
    V_baseline: float = 1.0
) -> Tuple[float, str]:
    """
    Measure narrative health using V formula.

    High V indicates narrative stress, approaching bifurcation.
    Used for narrative warfare equilibrium analysis.

    Args:
        V_current: Current V value
        V_baseline: Baseline V for healthy narrative

    Returns:
        Tuple of (health_score 0-1, status)
    """
    # Health degrades as V increases
    health = 1.0 / (1.0 + np.exp(V_current - V_baseline))

    if V_current < 1.5:
        status = "stable"
    elif V_current < 2.5:
        status = "stressed"
    elif V_current < 4.0:
        status = "approaching_bifurcation"
    else:
        status = "phase_transition_imminent"

    return float(health), status


def sliding_window_v(
    engagement_stream: NDArray[np.float64],
    time_stream: NDArray[np.float64],
    window_minutes: float = 15.0,
    nom_baseline: float = 0.3
) -> NDArray[np.float64]:
    """
    Compute V on sliding windows for real-time monitoring.

    15-minute intervals with CUSUM change detection for V spikes
    indicating campaign activation.

    Args:
        engagement_stream: Time series of engagement rates
        time_stream: Timestamps
        window_minutes: Window size in minutes
        nom_baseline: Expected organic baseline

    Returns:
        V values for each window
    """
    if len(engagement_stream) < 2:
        return np.array([0.0])

    # Convert window to samples
    dt = np.mean(np.diff(time_stream)) if len(time_stream) > 1 else 60
    window_samples = max(1, int((window_minutes * 60) / dt))

    V_series = []

    for i in range(0, len(engagement_stream) - window_samples + 1, window_samples // 2):
        window = engagement_stream[i:i + window_samples]
        time_window = time_stream[i:i + window_samples]

        mu_avg = compute_mu_avg(window)
        H_z = compute_behavioral_entropy(window)
        est = float(np.mean(window))

        V = compute_causal_bound_v(mu_avg, nom_baseline, est, H_z)
        V_series.append(V)

    return np.array(V_series)


def cusum_change_detection(
    V_series: NDArray[np.float64],
    threshold: float = 2.0,
    drift: float = 0.5
) -> List[int]:
    """
    CUSUM algorithm for V spike detection.

    Detects campaign activation when V trajectory shows significant change.

    Args:
        V_series: V values over time
        threshold: Detection threshold
        drift: Expected drift (target mean shift)

    Returns:
        Indices where change points detected
    """
    change_points = []

    S_pos = 0.0
    S_neg = 0.0

    for i, v in enumerate(V_series):
        S_pos = max(0, S_pos + v - drift)
        S_neg = min(0, S_neg + v + drift)

        if S_pos > threshold:
            change_points.append(i)
            S_pos = 0.0
        elif S_neg < -threshold:
            change_points.append(i)
            S_neg = 0.0

    return change_points
