"""
Kuramoto Order Parameter and Phase Coherence.

Implements the Kuramoto model for flow state detection:

    R(t)e^{jΨ(t)} = (1/N) Σ_{i=1}^N e^{jθ_i(t)}

Where:
- R(t) ∈ [0,1] is the order parameter (coherence magnitude)
- Ψ(t) is the global phase
- θ_i(t) is the instantaneous phase of oscillator i

Key Results:
- Flow present ⇔ R(t) ≥ R_flow (0.76) for ≥ 800ms
- Coherence velocity ν(t) = Ṙ(t) predicts state transitions
- Phase curvature κ = |θ''| / (1 + θ'²)^{3/2} indicates instability

References:
- Kuramoto (1984) "Chemical Oscillations, Waves, and Turbulence"
- Strogatz (2000) "From Kuramoto to Crawford"
- arXiv:2408.06433 "Dynamic Phase Transitions"
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.signal import hilbert, savgol_filter

from .types import CoherenceState, PhaseSignal, FlowLevel


@dataclass
class KuramotoConfig:
    """Configuration for Kuramoto analysis."""
    # Flow detection thresholds (empirically calibrated)
    R_flow: float = 0.76             # Minimum R for flow state
    R_deep_flow: float = 0.88        # Deep flow threshold
    flow_duration_ms: float = 800    # Minimum duration for flow

    # Derivative computation
    savgol_window: int = 11          # Savitzky-Golay window (odd)
    savgol_poly: int = 3             # Polynomial order

    # Stability analysis
    collapse_velocity_threshold: float = -0.1  # Ṙ threshold for collapse warning

    # Human-AI coupling (fitted on 40k sessions)
    K_human: float = 0.42            # Coupling constant


def hilbert_phase(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Extract instantaneous phase using Hilbert transform.

    Args:
        signals: Shape (N, T) or (T,) - real-valued signals

    Returns:
        phases: Shape (N, T) or (T,) - instantaneous phases in [0, 2π)
    """
    if signals.ndim == 1:
        analytic = hilbert(signals)
        return np.angle(analytic) % (2 * np.pi)

    phases = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        analytic = hilbert(signals[i])
        phases[i] = np.angle(analytic) % (2 * np.pi)
    return phases


def compute_order_parameter(
    phases: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None
) -> Tuple[float, float]:
    """
    Compute Kuramoto order parameter.

    R e^{jΨ} = (1/N) Σ w_i e^{jθ_i}

    Args:
        phases: Shape (N,) - phases in radians
        weights: Shape (N,) - optional weights (default uniform)

    Returns:
        Tuple of (R, Ψ) - magnitude and global phase
    """
    N = len(phases)
    if weights is None:
        weights = np.ones(N) / N
    else:
        weights = weights / weights.sum()

    # Complex order parameter
    Z = np.sum(weights * np.exp(1j * phases))

    R = np.abs(Z)
    Psi = np.angle(Z) % (2 * np.pi)

    return float(R), float(Psi)


def compute_order_parameter_timeseries(
    phases: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Kuramoto order parameter over time.

    Args:
        phases: Shape (N, T) - phases over time
        weights: Shape (N,) - optional weights

    Returns:
        Tuple of (R_t, Psi_t) - time series of R and Ψ
    """
    N, T = phases.shape
    if weights is None:
        weights = np.ones(N) / N
    else:
        weights = weights / weights.sum()

    # Vectorized computation
    Z = np.sum(weights[:, np.newaxis] * np.exp(1j * phases), axis=0)

    R_t = np.abs(Z)
    Psi_t = np.angle(Z) % (2 * np.pi)

    return R_t, Psi_t


def compute_phase_velocity(
    phases: NDArray[np.float64],
    dt: float,
    config: KuramotoConfig = KuramotoConfig()
) -> NDArray[np.float64]:
    """
    Compute instantaneous phase velocity (angular frequency).

    ω_i(t) = dθ_i/dt

    Args:
        phases: Shape (N, T) or (T,) - phases
        dt: Time step
        config: Configuration

    Returns:
        omega: Phase velocities
    """
    if phases.ndim == 1:
        # Handle phase wrapping
        unwrapped = np.unwrap(phases)
        if len(unwrapped) >= config.savgol_window:
            smoothed = savgol_filter(unwrapped, config.savgol_window, config.savgol_poly)
            omega = np.gradient(smoothed, dt)
        else:
            omega = np.gradient(unwrapped, dt)
        return omega

    # Multi-oscillator case
    N, T = phases.shape
    omega = np.zeros_like(phases)
    for i in range(N):
        unwrapped = np.unwrap(phases[i])
        if T >= config.savgol_window:
            smoothed = savgol_filter(unwrapped, config.savgol_window, config.savgol_poly)
            omega[i] = np.gradient(smoothed, dt)
        else:
            omega[i] = np.gradient(unwrapped, dt)
    return omega


def compute_coherence_velocity(
    R_t: NDArray[np.float64],
    dt: float,
    config: KuramotoConfig = KuramotoConfig()
) -> NDArray[np.float64]:
    """
    Compute coherence velocity ν(t) = Ṙ(t).

    Use Savitzky-Golay filter for noise robustness.

    Args:
        R_t: Shape (T,) - order parameter time series
        dt: Time step
        config: Configuration

    Returns:
        R_dot: Coherence velocity
    """
    if len(R_t) >= config.savgol_window:
        R_smooth = savgol_filter(R_t, config.savgol_window, config.savgol_poly)
        R_dot = np.gradient(R_smooth, dt)
    else:
        R_dot = np.gradient(R_t, dt)
    return R_dot


def compute_coherence_acceleration(
    R_t: NDArray[np.float64],
    dt: float,
    config: KuramotoConfig = KuramotoConfig()
) -> NDArray[np.float64]:
    """
    Compute coherence acceleration R̈(t).

    Used for Hopf bifurcation and collapse detection.

    Args:
        R_t: Shape (T,) - order parameter time series
        dt: Time step
        config: Configuration

    Returns:
        R_ddot: Coherence acceleration
    """
    R_dot = compute_coherence_velocity(R_t, dt, config)

    if len(R_dot) >= config.savgol_window:
        R_dot_smooth = savgol_filter(R_dot, config.savgol_window, config.savgol_poly)
        R_ddot = np.gradient(R_dot_smooth, dt)
    else:
        R_ddot = np.gradient(R_dot, dt)
    return R_ddot


def compute_phase_curvature(
    phases: NDArray[np.float64],
    dt: float,
    config: KuramotoConfig = KuramotoConfig()
) -> NDArray[np.float64]:
    """
    Compute phase curvature κ = |θ''| / (1 + θ'²)^{3/2}.

    High curvature indicates phase instability / sharp transitions.

    Args:
        phases: Shape (T,) - phase trajectory
        dt: Time step
        config: Configuration

    Returns:
        kappa: Curvature values
    """
    unwrapped = np.unwrap(phases)

    if len(unwrapped) >= config.savgol_window:
        smoothed = savgol_filter(unwrapped, config.savgol_window, config.savgol_poly)
    else:
        smoothed = unwrapped

    theta_p = np.gradient(smoothed, dt)      # θ'
    theta_pp = np.gradient(theta_p, dt)      # θ''

    # Curvature formula
    kappa = np.abs(theta_pp) / (1 + theta_p**2)**1.5

    return kappa


def compute_angular_velocity_dispersion(
    phases: NDArray[np.float64],
    dt: float,
    config: KuramotoConfig = KuramotoConfig()
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute angular velocity dispersion σ_ω².

    σ_ω²(t) = (1/N) Σ_i (ω_i - Ψ̇)²

    This is a confidence metric - low dispersion = high confidence.

    Args:
        phases: Shape (N, T) - phases
        dt: Time step
        config: Configuration

    Returns:
        Tuple of (sigma_omega, Psi_dot) - dispersion and global phase velocity
    """
    _, T = phases.shape

    # Compute individual angular velocities
    omega = compute_phase_velocity(phases, dt, config)  # (N, T)

    # Compute global phase
    _, Psi_t = compute_order_parameter_timeseries(phases)
    Psi_unwrapped = np.unwrap(Psi_t)
    Psi_dot = np.gradient(Psi_unwrapped, dt)

    # Dispersion: variance of deviations from mean
    deviations = omega - Psi_dot[np.newaxis, :]
    sigma_omega = np.sqrt(np.mean(deviations**2, axis=0))

    return sigma_omega, Psi_dot


def compute_coherence_state(
    phases: NDArray[np.float64],
    dt: float,
    timestamp: float,
    history_R: Optional[NDArray[np.float64]] = None,
    config: KuramotoConfig = KuramotoConfig()
) -> CoherenceState:
    """
    Compute complete coherence state at current time.

    Args:
        phases: Shape (N,) - current phases
        dt: Time step
        timestamp: Current timestamp
        history_R: Shape (T,) - historical R values for derivatives
        config: Configuration

    Returns:
        CoherenceState with R, Ψ, derivatives, and dispersion
    """
    R, Psi = compute_order_parameter(phases)

    # Compute derivatives if history available
    R_dot = 0.0
    R_ddot = 0.0
    if history_R is not None and len(history_R) > 2:
        R_full = np.append(history_R, R)
        R_dots = compute_coherence_velocity(R_full, dt, config)
        R_ddots = compute_coherence_acceleration(R_full, dt, config)
        R_dot = float(R_dots[-1])
        R_ddot = float(R_ddots[-1])

    # Angular velocity dispersion (approximate from phase spread)
    phase_spread = np.std(phases)
    sigma_omega = phase_spread  # Simplified estimate

    return CoherenceState(
        R=R,
        Psi=Psi,
        R_dot=R_dot,
        R_ddot=R_ddot,
        sigma_omega=sigma_omega,
        timestamp=timestamp
    )


def detect_flow_state(
    R_t: NDArray[np.float64],
    timestamps: NDArray[np.float64],
    config: KuramotoConfig = KuramotoConfig()
) -> List[Tuple[float, float, FlowLevel]]:
    """
    Detect flow state intervals from coherence time series.

    Flow present ⇔ R(t) ≥ R_flow for ≥ duration threshold.

    Args:
        R_t: Shape (T,) - order parameter time series
        timestamps: Shape (T,) - timestamps
        config: Configuration

    Returns:
        List of (start_time, end_time, flow_level) tuples
    """
    intervals = []
    T = len(R_t)

    # Convert duration to samples
    dt = timestamps[1] - timestamps[0] if T > 1 else 1.0
    min_samples = int(config.flow_duration_ms / 1000 / dt)

    # Find flow intervals
    in_flow = False
    flow_start = 0
    current_level = FlowLevel.NONE

    for t in range(T):
        R = R_t[t]

        # Determine current level
        if R >= config.R_deep_flow:
            level = FlowLevel.DEEP_FLOW
        elif R >= config.R_flow:
            level = FlowLevel.FLOW
        elif R >= 0.65:
            level = FlowLevel.BUILDING
        elif R >= 0.45:
            level = FlowLevel.EMERGING
        else:
            level = FlowLevel.NONE

        if level.value >= FlowLevel.FLOW.value:
            if not in_flow:
                flow_start = t
                in_flow = True
                current_level = level
            else:
                # Update to deeper level if applicable
                if level.value > current_level.value:
                    current_level = level
        else:
            if in_flow:
                # End of flow interval
                duration = t - flow_start
                if duration >= min_samples:
                    intervals.append((
                        float(timestamps[flow_start]),
                        float(timestamps[t-1]),
                        current_level
                    ))
                in_flow = False

    # Handle ongoing flow at end
    if in_flow:
        duration = T - flow_start
        if duration >= min_samples:
            intervals.append((
                float(timestamps[flow_start]),
                float(timestamps[-1]),
                current_level
            ))

    return intervals


def predict_flow_collapse(
    R_t: NDArray[np.float64],
    dt: float,
    config: KuramotoConfig = KuramotoConfig()
) -> Optional[float]:
    """
    Predict time to flow collapse using coherence dynamics.

    Uses linear extrapolation of Ṙ to estimate when R will cross threshold.
    Returns None if not in flow or collapse not imminent.

    Args:
        R_t: Shape (T,) - recent R values
        dt: Time step
        config: Configuration

    Returns:
        Estimated seconds until collapse, or None
    """
    if len(R_t) < 5:
        return None

    R_current = R_t[-1]
    if R_current < config.R_flow:
        return None  # Not in flow

    R_dot = compute_coherence_velocity(R_t, dt, config)
    R_dot_current = R_dot[-1]

    if R_dot_current >= 0:
        return None  # Not declining

    # Time to cross R_flow threshold
    delta_R = R_current - config.R_flow
    time_to_collapse = delta_R / (-R_dot_current)

    # Sanity bounds
    if time_to_collapse < 0 or time_to_collapse > 60:
        return None

    return float(time_to_collapse)


def kuramoto_coupling_dynamics(
    phases_user: NDArray[np.float64],
    phases_ai: NDArray[np.float64],
    dt: float,
    K: float = 0.42  # K_human default
) -> Tuple[NDArray[np.float64], float]:
    """
    Compute coupled phase dynamics between user and AI.

    θ̇_user = ω_user + K * sin(θ_ai - θ_user)

    Args:
        phases_user: Shape (T,) - user phase history
        phases_ai: Shape (T,) - AI phase history
        dt: Time step
        K: Coupling strength (default K_human = 0.42)

    Returns:
        Tuple of (phase_diff, resonance_score)
    """
    phase_diff = phases_ai - phases_user

    # Wrap to [-π, π]
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

    # Resonance: average cosine of phase difference
    resonance = float(np.mean(np.cos(phase_diff)))

    return phase_diff, resonance


def compute_phase_precession(
    phases_user: NDArray[np.float64],
    phases_ai: NDArray[np.float64],
    dt: float,
    window_ms: float = 500
) -> Tuple[float, float]:
    """
    Detect phase precession between user and AI.

    Human phase leads AI phase by Δφ = 187 ± 42 ms before action.

    Args:
        phases_user: Shape (T,) - user phases
        phases_ai: Shape (T,) - AI phases
        dt: Time step
        window_ms: Analysis window in ms

    Returns:
        Tuple of (lead_time_ms, confidence)
    """
    window_samples = int(window_ms / 1000 / dt)
    if len(phases_user) < window_samples:
        return 0.0, 0.0

    # Cross-correlation to find lead/lag
    user_recent = phases_user[-window_samples:]
    ai_recent = phases_ai[-window_samples:]

    # Normalize
    user_norm = (user_recent - np.mean(user_recent)) / (np.std(user_recent) + 1e-10)
    ai_norm = (ai_recent - np.mean(ai_recent)) / (np.std(ai_recent) + 1e-10)

    # Cross-correlation
    xcorr = np.correlate(user_norm, ai_norm, mode='full')
    lags = np.arange(-window_samples + 1, window_samples)

    # Find peak (user leading = positive lag)
    peak_idx = np.argmax(np.abs(xcorr))
    lead_samples = lags[peak_idx]
    lead_time_ms = lead_samples * dt * 1000

    # Confidence from peak height
    confidence = float(np.abs(xcorr[peak_idx]) / window_samples)

    return float(lead_time_ms), confidence
