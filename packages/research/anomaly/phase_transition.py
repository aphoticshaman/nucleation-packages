"""
Phase Transition Detection via Dynamic Phase Transitions (DPT).
Based on arXiv:2408.06433 - "Endogenous Crashes as Phase Transitions"

This module implements early warning signal detection using anomalous
dimension scaling, which outperforms LPPL models with fewer false positives.

Key insight: Crashes are dynamic phase transitions where the noise
distribution evolves (not just drift or volatility changes). The
anomalous dimension Δ(t,τ) shows strong upward trends before crashes
while volatility shows only weak trends.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.ndimage import uniform_filter1d

from ..core.types import PhaseTransitionSignal


@dataclass
class DPTConfig:
    """Configuration for Dynamic Phase Transition detection."""
    tau_min: int = 5  # Minimum lag for autocorrelation
    tau_max: int = 50  # Maximum lag for autocorrelation
    window_size: int = 100  # Rolling window for anomalous dimension
    trend_window: int = 20  # Window for trend detection
    threshold_quantile: float = 0.95  # Anomaly threshold quantile


def compute_autocorrelation(
    x: NDArray[np.float64],
    tau: int
) -> float:
    """
    Compute autocorrelation at lag tau.

    G(t, τ) = ⟨X(t)X(t+τ)⟩

    Args:
        x: Time series
        tau: Lag

    Returns:
        Autocorrelation value
    """
    if tau >= len(x):
        return 0.0

    x_centered = x - np.mean(x)
    n = len(x) - tau

    # Normalized autocorrelation
    numerator = np.sum(x_centered[:n] * x_centered[tau:tau+n])
    denominator = np.sum(x_centered ** 2)

    if denominator < 1e-10:
        return 0.0

    return numerator / denominator


def compute_anomalous_dimension(
    x: NDArray[np.float64],
    tau: int,
    K: float = 1.0
) -> float:
    """
    Compute anomalous dimension Δ(t,τ).

    From arXiv:2408.06433:
    Δ(t,τ) = [K - ln(G(t,τ))] / ln(τ)

    where G(t,τ) is the autocorrelation at lag τ.

    For self-similar processes:
    ⟨X(t)X(t+τ)⟩ = c/|τ|^{2Δ}

    The anomalous dimension captures how the autocorrelation
    decays with lag - changes in Δ indicate structural changes
    in the underlying dynamics.

    Args:
        x: Time series window
        tau: Lag value
        K: Scaling constant

    Returns:
        Anomalous dimension estimate
    """
    if tau <= 1:
        return 0.0

    G = compute_autocorrelation(x, tau)

    # Handle edge cases
    if G <= 0:
        G = 1e-10  # Avoid log of non-positive

    delta = (K - np.log(np.abs(G))) / np.log(tau)

    return delta


def compute_multiscale_anomalous_dimension(
    x: NDArray[np.float64],
    config: DPTConfig = DPTConfig()
) -> Tuple[float, NDArray[np.float64]]:
    """
    Compute anomalous dimension across multiple scales.

    From arXiv:2408.06433:
    "Strong trend is seen in the anomalous dimension" before crashes.

    We fit the power law decay of autocorrelation across multiple
    lags to get a robust estimate.

    Args:
        x: Time series window
        config: DPT configuration

    Returns:
        Tuple of (average Δ, array of Δ per lag)
    """
    taus = np.arange(config.tau_min, min(config.tau_max, len(x) // 2))
    deltas = []

    for tau in taus:
        delta = compute_anomalous_dimension(x, tau)
        deltas.append(delta)

    deltas = np.array(deltas)

    # Filter outliers
    median = np.median(deltas)
    mad = np.median(np.abs(deltas - median))
    valid = np.abs(deltas - median) < 3 * mad

    if valid.sum() > 0:
        avg_delta = np.mean(deltas[valid])
    else:
        avg_delta = median

    return avg_delta, deltas


def compute_generalized_hurst_exponent(
    x: NDArray[np.float64],
    order: int = 2,
    min_scale: int = 4,
    max_scale: int = None
) -> float:
    """
    Compute generalized Hurst exponent for higher-order statistics.

    From arXiv:2408.06433:
    ⟨X(t)^n⟩ = A_n|t|^{Δ_n}
    Δ_n = [K_n + ln⟨X(t)^n⟩] / ln(|t|)

    Multifractality detection: Δ_n ≠ nΔ indicates nonlinearity.

    Args:
        x: Time series
        order: Moment order n
        min_scale: Minimum scale for analysis
        max_scale: Maximum scale

    Returns:
        Generalized Hurst exponent H_n = Δ_n / n
    """
    if max_scale is None:
        max_scale = len(x) // 4

    scales = np.arange(min_scale, max_scale)
    moments = []

    for s in scales:
        # Compute differences at scale s
        diffs = x[s:] - x[:-s]
        # n-th absolute moment
        moment = np.mean(np.abs(diffs) ** order)
        moments.append(moment)

    # Log-log regression to find scaling exponent
    log_scales = np.log(scales)
    log_moments = np.log(np.array(moments) + 1e-10)

    # Fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_moments)

    # H_n = slope / n
    H_n = slope / order

    return H_n


def detect_phase_transition_signals(
    x: NDArray[np.float64],
    timestamps: NDArray[np.float64],
    config: DPTConfig = DPTConfig()
) -> List[PhaseTransitionSignal]:
    """
    Detect early warning signals for phase transitions.

    From arXiv:2408.06433, the DPT framework shows that:
    - Volatility: weak positive trend before crashes
    - Skewness: weak negative trend
    - Lag-1 autocorrelation: no trend
    - Anomalous dimension: STRONG positive trend (best predictor)

    NOVEL INSIGHT #6: Collapse Velocity
    Not just that outcomes are converging, but the RATE at which
    they're converging toward an attractor. Rising Δ trend indicates
    accelerating collapse.

    Args:
        x: Time series (e.g., returns, variance index)
        timestamps: Corresponding timestamps
        config: Detection configuration

    Returns:
        List of PhaseTransitionSignal objects
    """
    T = len(x)
    signals = []

    # Need enough history for rolling windows
    start_idx = config.window_size + config.trend_window

    for t in range(start_idx, T):
        window = x[t - config.window_size:t]
        timestamp = timestamps[t]

        # Compute anomalous dimension
        delta, _ = compute_multiscale_anomalous_dimension(window, config)

        # Compute volatility (standard deviation)
        volatility = np.std(window)

        # Compute lag-1 autocorrelation
        autocorr = compute_autocorrelation(window, 1)

        # Compute trend in anomalous dimension over recent history
        if t >= start_idx + config.trend_window:
            # Get recent Δ values
            recent_deltas = []
            for t_past in range(t - config.trend_window, t):
                past_window = x[t_past - config.window_size:t_past]
                past_delta, _ = compute_multiscale_anomalous_dimension(past_window, config)
                recent_deltas.append(past_delta)

            recent_deltas = np.array(recent_deltas)

            # Trend strength: slope of linear fit normalized by std
            time_idx = np.arange(len(recent_deltas))
            if np.std(recent_deltas) > 1e-10:
                slope, _, r, _, _ = stats.linregress(time_idx, recent_deltas)
                trend_strength = slope * r ** 2  # Weight by R²
            else:
                trend_strength = 0.0

            # Collapse velocity: second derivative of Δ
            delta_grad = np.gradient(recent_deltas)
            collapse_velocity = np.mean(delta_grad[-5:])  # Recent acceleration

        else:
            trend_strength = 0.0
            collapse_velocity = 0.0

        # Predicted transition probability based on trend strength
        # Logistic mapping to [0, 1]
        predicted_prob = 1.0 / (1.0 + np.exp(-5 * (trend_strength - 0.5)))

        # Estimate time to transition (if trend continues linearly)
        if trend_strength > 0.1:
            # Simple extrapolation: when will Δ exceed historical max?
            historical_max_delta = delta * 1.5  # Heuristic threshold
            estimated_time = (historical_max_delta - delta) / (trend_strength + 1e-10)
            estimated_time = max(1, estimated_time)
        else:
            estimated_time = None

        signal = PhaseTransitionSignal(
            timestamp=float(timestamp),
            anomalous_dimension=float(delta),
            volatility=float(volatility),
            autocorrelation=float(autocorr),
            trend_strength=float(trend_strength),
            predicted_transition_prob=float(predicted_prob),
            estimated_time_to_transition=estimated_time
        )
        signals.append(signal)

    return signals


def compute_attractor_basin_field(
    x: NDArray[np.float64],
    n_regimes: int = 3,
    config: DPTConfig = DPTConfig()
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    NOVEL INSIGHT #3: Map the attractor basin field.

    Identify which attractor basin each point belongs to and
    compute the "gravitational gradient" toward each attractor.

    Uses regime clustering in anomalous dimension space, then
    computes probability flows between basins.

    Args:
        x: Time series
        n_regimes: Number of attractor basins to identify
        config: Configuration

    Returns:
        Tuple of (basin_probabilities, basin_assignments)
        basin_probabilities: shape (T, n_regimes) - probability of being in each basin
        basin_assignments: shape (T,) - MAP assignment to basin
    """
    T = len(x)
    start_idx = config.window_size

    # Compute anomalous dimension for each point
    deltas = []
    vols = []
    for t in range(start_idx, T):
        window = x[t - config.window_size:t]
        delta, _ = compute_multiscale_anomalous_dimension(window, config)
        vol = np.std(window)
        deltas.append(delta)
        vols.append(vol)

    deltas = np.array(deltas)
    vols = np.array(vols)

    # Simple k-means in (Δ, volatility) space
    features = np.column_stack([deltas, vols])
    from scipy.cluster.vq import kmeans2, whiten

    # Normalize features
    features_whitened = whiten(features)

    # Cluster
    centroids, assignments = kmeans2(features_whitened, n_regimes, minit='++')

    # Compute soft assignments (distances to centroids)
    distances = np.zeros((len(features), n_regimes))
    for k in range(n_regimes):
        distances[:, k] = np.linalg.norm(features_whitened - centroids[k], axis=1)

    # Convert to probabilities via softmax of negative distances
    neg_dist = -distances
    exp_neg_dist = np.exp(neg_dist - neg_dist.max(axis=1, keepdims=True))
    basin_probs = exp_neg_dist / exp_neg_dist.sum(axis=1, keepdims=True)

    # Pad to full length
    full_probs = np.zeros((T, n_regimes))
    full_probs[start_idx:] = basin_probs
    full_probs[:start_idx] = basin_probs[0]  # Extend first value

    full_assignments = np.zeros(T, dtype=np.int64)
    full_assignments[start_idx:] = assignments
    full_assignments[:start_idx] = assignments[0]

    return full_probs, full_assignments


def gravity_gradient_field(
    basin_probs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    NOVEL INSIGHT #10: Compute gravity gradient field.

    The gradient of basin probability over time indicates how strongly
    the system is being "pulled" toward each attractor.

    Rising gradient = strengthening attractor
    Falling gradient = weakening attractor
    Cross-over = regime transition in progress

    Args:
        basin_probs: Basin probabilities shape (T, K)

    Returns:
        Gradient field shape (T, K)
    """
    # Compute temporal gradient of each basin probability
    gradients = np.zeros_like(basin_probs)

    for k in range(basin_probs.shape[1]):
        gradients[:, k] = np.gradient(basin_probs[:, k])

    return gradients
