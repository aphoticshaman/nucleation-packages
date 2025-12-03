"""
Flow State Detection via Hopf Bifurcation Analysis.

Implements flow state detection using dynamical systems theory.
Flow states emerge through Hopf bifurcation when cognitive oscillators
synchronize beyond a critical threshold.

Hopf Bifurcation:
    At the critical point μ_c, the system transitions from:
    - Fixed point (μ < μ_c): Incoherent, scattered phases
    - Limit cycle (μ > μ_c): Synchronized, coherent oscillation

    The order parameter R follows: R ~ √(μ - μ_c) near bifurcation

Flow State Thresholds:
    - R < 0.45: No flow (NONE)
    - 0.45 ≤ R < 0.65: Emerging flow
    - 0.65 ≤ R < 0.76: Building flow
    - R ≥ 0.76 for ≥800ms: FLOW STATE
    - R ≥ 0.88: Deep flow

The 800ms dwell time requirement filters transient synchronization
from sustained flow states.

Prediction:
    By tracking R trajectory and its derivatives, we can predict:
    - Time to bifurcation crossing
    - Probability of achieving flow
    - Expected flow duration

Applications:
    - Real-time flow state monitoring
    - Intervention timing (nudge toward flow)
    - Flow collapse prediction
    - Optimal challenge calibration

References:
    - Kuramoto model phase transitions
    - Hopf bifurcation normal form
    - Critical slowing down near bifurcation
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time

from .types import FlowState, FlowLevel, FlowFixedPoint, CoherenceState


# Flow thresholds (from empirical 40k session analysis)
R_NONE = 0.45         # Below this: no flow
R_EMERGING = 0.65     # Emerging flow zone
R_BUILDING = 0.76     # Building toward flow
R_FLOW = 0.76         # Flow threshold
R_DEEP_FLOW = 0.88    # Deep flow threshold

# Temporal requirements
FLOW_DWELL_MS = 800   # Minimum time above threshold for flow
DEEP_FLOW_DWELL_MS = 2000  # For deep flow classification

# Bifurcation parameters
MU_CRITICAL = 0.76    # Critical bifurcation parameter
LAMBDA_TYPICAL = 0.1  # Typical approach rate to bifurcation


@dataclass
class HopfState:
    """State of the Hopf bifurcation dynamics."""
    mu: float                     # Bifurcation parameter (≈ R)
    r: float                      # Amplitude (distance from fixed point)
    theta: float                  # Phase angle
    dr_dt: float                  # Amplitude derivative
    dtheta_dt: float              # Phase velocity
    distance_to_bifurcation: float  # μ_c - μ
    predicted_crossing_time: float  # Estimated time to cross μ_c
    is_supercritical: bool        # Above critical point


@dataclass
class FlowSession:
    """Tracking for a flow session."""
    start_time: float
    current_time: float
    r_history: List[float] = field(default_factory=list)
    time_in_flow: float = 0.0
    time_in_deep_flow: float = 0.0
    peak_r: float = 0.0
    flow_achieved: bool = False
    deep_flow_achieved: bool = False
    collapse_count: int = 0


class FlowDetector:
    """
    Real-time flow state detector using Hopf bifurcation analysis.

    Monitors order parameter R and detects transitions through
    the critical bifurcation point.
    """

    def __init__(
        self,
        mu_critical: float = MU_CRITICAL,
        dwell_time_ms: float = FLOW_DWELL_MS,
        history_length: int = 100
    ):
        """
        Initialize flow detector.

        Args:
            mu_critical: Critical bifurcation parameter
            dwell_time_ms: Required dwell time for flow confirmation
            history_length: Length of R history to maintain
        """
        self.mu_critical = mu_critical
        self.dwell_time_ms = dwell_time_ms
        self.history_length = history_length

        # State tracking
        self.r_history: List[Tuple[float, float]] = []  # (timestamp, R)
        self.flow_session: Optional[FlowSession] = None
        self.current_level = FlowLevel.NONE
        self.above_threshold_start: Optional[float] = None

    def update(
        self,
        R: float,
        timestamp: Optional[float] = None
    ) -> FlowState:
        """
        Update flow detector with new order parameter.

        Args:
            R: Current order parameter value
            timestamp: Current timestamp (ms), defaults to current time

        Returns:
            Current FlowState
        """
        if timestamp is None:
            timestamp = time.time() * 1000  # Convert to ms

        # Update history
        self.r_history.append((timestamp, R))
        if len(self.r_history) > self.history_length:
            self.r_history.pop(0)

        # Update flow session
        if self.flow_session is None:
            self.flow_session = FlowSession(
                start_time=timestamp,
                current_time=timestamp
            )
        self.flow_session.current_time = timestamp
        self.flow_session.r_history.append(R)
        self.flow_session.peak_r = max(self.flow_session.peak_r, R)

        # Determine flow level
        prev_level = self.current_level
        self.current_level = self._classify_level(R)

        # Track time above flow threshold
        if R >= R_FLOW:
            if self.above_threshold_start is None:
                self.above_threshold_start = timestamp
            else:
                duration = timestamp - self.above_threshold_start
                self.flow_session.time_in_flow = duration

                if duration >= self.dwell_time_ms:
                    self.flow_session.flow_achieved = True

                if R >= R_DEEP_FLOW and duration >= DEEP_FLOW_DWELL_MS:
                    self.flow_session.time_in_deep_flow = duration
                    self.flow_session.deep_flow_achieved = True
        else:
            # Dropped below threshold
            if self.above_threshold_start is not None:
                if prev_level.value >= FlowLevel.FLOW.value:
                    self.flow_session.collapse_count += 1
                self.above_threshold_start = None

        # Compute derivatives and predictions
        dr_dt, d2r_dt2 = self._compute_derivatives()
        stability = self._compute_stability(R, dr_dt, d2r_dt2)
        predicted_collapse = self._predict_collapse(R, dr_dt)
        hopf_state = self._compute_hopf_state(R, dr_dt)

        # Build flow state
        return FlowState(
            level=self.current_level,
            R=R,
            dR_dt=dr_dt,
            d2R_dt2=d2r_dt2,
            time_in_state_ms=self._time_in_current_level(timestamp),
            stability=stability,
            predicted_collapse_ms=predicted_collapse,
            is_flow=self.current_level.value >= FlowLevel.FLOW.value,
            is_deep_flow=self.current_level == FlowLevel.DEEP_FLOW
        )

    def _classify_level(self, R: float) -> FlowLevel:
        """Classify R value to flow level."""
        if R >= R_DEEP_FLOW:
            return FlowLevel.DEEP_FLOW
        elif R >= R_FLOW:
            return FlowLevel.FLOW
        elif R >= R_BUILDING:
            return FlowLevel.BUILDING
        elif R >= R_EMERGING:
            return FlowLevel.EMERGING
        elif R >= R_NONE:
            return FlowLevel.EMERGING
        else:
            return FlowLevel.NONE

    def _compute_derivatives(self) -> Tuple[float, float]:
        """Compute first and second derivatives of R."""
        if len(self.r_history) < 3:
            return 0.0, 0.0

        recent = self.r_history[-10:] if len(self.r_history) >= 10 else self.r_history

        # Extract R values and timestamps
        times = np.array([t for t, _ in recent])
        Rs = np.array([r for _, r in recent])

        # First derivative (dR/dt)
        dt = np.diff(times)
        dR = np.diff(Rs)

        # Avoid division by zero
        dt = np.where(dt == 0, 1e-10, dt)

        dr_dt = dR / dt
        mean_dr_dt = np.mean(dr_dt) if len(dr_dt) > 0 else 0.0

        # Second derivative (d²R/dt²)
        if len(dr_dt) >= 2:
            d2r_dt2 = np.mean(np.diff(dr_dt) / dt[1:])
        else:
            d2r_dt2 = 0.0

        return float(mean_dr_dt), float(d2r_dt2)

    def _compute_stability(
        self,
        R: float,
        dr_dt: float,
        d2r_dt2: float
    ) -> float:
        """
        Compute stability score for current flow state.

        Uses Lyapunov-like analysis of trajectory.
        """
        # Stability decreases with:
        # - Distance from flow threshold
        # - Negative derivative (approaching collapse)
        # - Negative second derivative (accelerating toward collapse)

        distance_factor = 1.0 - abs(R - R_FLOW) / R_FLOW
        velocity_factor = 1.0 / (1.0 + max(0, -dr_dt * 10))  # Penalize decreasing R
        accel_factor = 1.0 / (1.0 + max(0, -d2r_dt2 * 5))

        stability = 0.4 * distance_factor + 0.35 * velocity_factor + 0.25 * accel_factor
        return float(np.clip(stability, 0, 1))

    def _predict_collapse(self, R: float, dr_dt: float) -> Optional[float]:
        """
        Predict time until flow collapse.

        Returns None if collapse not predicted.
        """
        if R < R_FLOW or dr_dt >= 0:
            return None  # Not in flow or not decreasing

        # Simple linear prediction
        distance_to_threshold = R - R_FLOW
        if dr_dt >= 0:
            return None

        time_to_collapse = -distance_to_threshold / dr_dt * 1000  # Convert to ms
        return float(time_to_collapse) if time_to_collapse > 0 else 0.0

    def _compute_hopf_state(self, R: float, dr_dt: float) -> HopfState:
        """Compute Hopf bifurcation state."""
        # In Hopf normal form: dr/dt = μr - r³
        # Near bifurcation: R ~ √(μ - μ_c)

        distance = self.mu_critical - R
        is_supercritical = R > self.mu_critical

        # Estimate time to crossing based on current velocity
        if dr_dt > 0 and not is_supercritical:
            predicted_crossing = distance / dr_dt * 1000
        else:
            predicted_crossing = float('inf')

        return HopfState(
            mu=R,
            r=R,  # Order parameter as amplitude
            theta=0.0,  # Would need Ψ for true phase
            dr_dt=dr_dt,
            dtheta_dt=0.0,  # Would need dΨ/dt
            distance_to_bifurcation=distance,
            predicted_crossing_time=predicted_crossing,
            is_supercritical=is_supercritical
        )

    def _time_in_current_level(self, current_time: float) -> float:
        """Compute time spent in current flow level."""
        if not self.r_history:
            return 0.0

        # Walk backwards through history to find level change
        duration = 0.0
        prev_time = current_time

        for timestamp, R in reversed(self.r_history):
            level = self._classify_level(R)
            if level != self.current_level:
                break
            duration = current_time - timestamp
            prev_time = timestamp

        return duration

    def get_session_stats(self) -> Dict[str, float]:
        """Get statistics for current flow session."""
        if self.flow_session is None:
            return {"status": "no_session"}

        session = self.flow_session
        duration = session.current_time - session.start_time

        return {
            "session_duration_ms": duration,
            "time_in_flow_ms": session.time_in_flow,
            "time_in_deep_flow_ms": session.time_in_deep_flow,
            "flow_percentage": session.time_in_flow / duration if duration > 0 else 0,
            "peak_R": session.peak_r,
            "flow_achieved": session.flow_achieved,
            "deep_flow_achieved": session.deep_flow_achieved,
            "collapse_count": session.collapse_count,
            "mean_R": np.mean(session.r_history) if session.r_history else 0,
            "std_R": np.std(session.r_history) if len(session.r_history) > 1 else 0,
        }

    def reset_session(self) -> None:
        """Reset flow session tracking."""
        self.flow_session = None
        self.above_threshold_start = None


def detect_hopf_bifurcation(
    R_timeseries: NDArray[np.float64],
    timestamps: NDArray[np.float64],
    mu_critical: float = MU_CRITICAL
) -> List[Tuple[float, str, float]]:
    """
    Detect Hopf bifurcation crossings in R timeseries.

    Args:
        R_timeseries: Order parameter values over time
        timestamps: Corresponding timestamps
        mu_critical: Critical bifurcation parameter

    Returns:
        List of (timestamp, direction, R_value) for each crossing
    """
    crossings = []

    if len(R_timeseries) < 2:
        return crossings

    for i in range(1, len(R_timeseries)):
        prev_R = R_timeseries[i - 1]
        curr_R = R_timeseries[i]

        # Check for crossing
        if prev_R < mu_critical <= curr_R:
            crossings.append((timestamps[i], "up", curr_R))
        elif prev_R >= mu_critical > curr_R:
            crossings.append((timestamps[i], "down", curr_R))

    return crossings


def compute_critical_slowing_down(
    R_timeseries: NDArray[np.float64],
    window_size: int = 20
) -> NDArray[np.float64]:
    """
    Compute critical slowing down indicator.

    Near bifurcation, the system's response time increases
    (autocorrelation increases, variance increases).

    This can be used as an early warning of approaching flow.

    Args:
        R_timeseries: Order parameter time series
        window_size: Window for computing indicators

    Returns:
        Array of slowing down indicators
    """
    if len(R_timeseries) < window_size:
        return np.zeros(len(R_timeseries))

    indicators = np.zeros(len(R_timeseries))

    for i in range(window_size, len(R_timeseries)):
        window = R_timeseries[i - window_size:i]

        # Autocorrelation at lag 1
        if np.std(window) > 1e-10:
            autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
        else:
            autocorr = 0.0

        # Variance (increases near bifurcation)
        variance = np.var(window)

        # Combined indicator
        indicators[i] = 0.5 * (1 + autocorr) + 0.5 * np.tanh(variance * 10)

    return indicators


def estimate_bifurcation_parameter(
    R_timeseries: NDArray[np.float64],
    timestamps: NDArray[np.float64]
) -> Tuple[float, float, float]:
    """
    Estimate Hopf bifurcation parameters from data.

    Fits the normal form: dR/dt = μR - R³

    Args:
        R_timeseries: Order parameter time series
        timestamps: Timestamps

    Returns:
        Tuple of (estimated_mu, lambda_rate, fit_quality)
    """
    if len(R_timeseries) < 10:
        return 0.5, 0.1, 0.0

    # Compute dR/dt
    dt = np.diff(timestamps)
    dR = np.diff(R_timeseries)

    # Avoid division by zero
    dt = np.where(dt == 0, 1e-10, dt)
    dR_dt = dR / dt

    # Fit: dR/dt = μR - R³
    # Linear regression: dR/dt / R = μ - R²
    R_mid = R_timeseries[:-1]
    mask = R_mid > 0.1  # Avoid near-zero

    if np.sum(mask) < 5:
        return 0.5, 0.1, 0.0

    y = dR_dt[mask] / R_mid[mask]
    X = -R_mid[mask] ** 2

    # Fit y = μ + X (intercept is μ)
    X_design = np.column_stack([np.ones(len(X)), X])
    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        mu_estimated = coeffs[0]
        lambda_rate = abs(coeffs[1]) if len(coeffs) > 1 else 0.1

        # Fit quality (R²)
        y_pred = X_design @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return float(mu_estimated), float(lambda_rate), float(r_squared)
    except:
        return 0.5, 0.1, 0.0


def compute_flow_fixed_point(
    R_equilibrium: float,
    coupling_strength: float = 0.42
) -> FlowFixedPoint:
    """
    Compute flow fixed point characteristics.

    The stable flow state corresponds to a fixed point of the
    Kuramoto dynamics: R* where dR/dt = 0.

    Args:
        R_equilibrium: Equilibrium order parameter
        coupling_strength: Kuramoto coupling K

    Returns:
        FlowFixedPoint characterization
    """
    # In Kuramoto model: R = f(K, N)
    # For large N: R → 0 for K < K_c, R → √(1 - K_c/K) for K > K_c

    K_critical = 2.0 / np.pi  # Critical coupling (mean-field)

    is_stable = R_equilibrium > 0.5 and coupling_strength > K_critical

    # Bifurcation distance
    if coupling_strength > K_critical:
        expected_R = np.sqrt(1 - K_critical / coupling_strength)
    else:
        expected_R = 0.0

    deviation = abs(R_equilibrium - expected_R)

    # Eigenvalue of linearization (determines stability)
    # λ ≈ -2K R² for stable fixed point
    eigenvalue = -2 * coupling_strength * R_equilibrium ** 2 if R_equilibrium > 0 else 0

    return FlowFixedPoint(
        R_equilibrium=R_equilibrium,
        coupling_strength=coupling_strength,
        stability_eigenvalue=float(eigenvalue),
        is_stable=is_stable,
        bifurcation_distance=float(deviation)
    )


def predict_flow_onset(
    R_timeseries: NDArray[np.float64],
    timestamps: NDArray[np.float64],
    threshold: float = R_FLOW
) -> Tuple[bool, float, float]:
    """
    Predict if and when flow state will be reached.

    Uses trajectory extrapolation and critical slowing indicators.

    Args:
        R_timeseries: Recent R values
        timestamps: Timestamps
        threshold: Flow threshold

    Returns:
        Tuple of (will_reach_flow, predicted_time_ms, confidence)
    """
    if len(R_timeseries) < 5:
        return False, float('inf'), 0.0

    current_R = R_timeseries[-1]

    # Already in flow
    if current_R >= threshold:
        return True, 0.0, 1.0

    # Compute velocity
    dt = np.diff(timestamps[-5:])
    dR = np.diff(R_timeseries[-5:])
    dt = np.where(dt == 0, 1e-10, dt)
    velocity = np.mean(dR / dt)

    # Not approaching flow
    if velocity <= 0:
        return False, float('inf'), 0.0

    # Time to threshold
    distance = threshold - current_R
    predicted_time = distance / velocity * 1000  # ms

    # Confidence based on:
    # - Consistency of velocity
    # - Distance to threshold
    # - Critical slowing indicator
    velocity_std = np.std(dR / dt)
    velocity_consistency = 1.0 / (1.0 + velocity_std / (abs(velocity) + 1e-10))

    slowing = compute_critical_slowing_down(R_timeseries)
    slowing_factor = slowing[-1] if len(slowing) > 0 else 0.5

    confidence = 0.5 * velocity_consistency + 0.3 * slowing_factor + 0.2 * (1 - distance)
    confidence = float(np.clip(confidence, 0, 1))

    return True, float(predicted_time), confidence


def optimal_intervention_timing(
    flow_state: FlowState,
    target_level: FlowLevel = FlowLevel.FLOW
) -> Dict[str, float]:
    """
    Compute optimal timing for intervention to achieve target flow level.

    Returns recommendations for when to apply challenge/skill adjustments.

    Args:
        flow_state: Current flow state
        target_level: Desired flow level

    Returns:
        Dictionary with intervention timing recommendations
    """
    current_R = flow_state.R
    dr_dt = flow_state.dR_dt
    target_R = _level_to_R(target_level)

    result = {
        "current_R": current_R,
        "target_R": target_R,
        "gap": target_R - current_R,
        "trajectory": "increasing" if dr_dt > 0 else "decreasing" if dr_dt < 0 else "stable"
    }

    if current_R >= target_R:
        # Already at or above target
        result["action"] = "maintain"
        result["urgency"] = 0.0
    elif dr_dt > 0:
        # Approaching target naturally
        time_to_target = (target_R - current_R) / dr_dt * 1000
        result["action"] = "monitor"
        result["estimated_time_ms"] = time_to_target
        result["urgency"] = 0.3
    elif dr_dt < 0:
        # Moving away from target
        result["action"] = "intervene"
        result["urgency"] = min(1.0, abs(dr_dt) * 10 + (target_R - current_R))
    else:
        # Stable below target
        result["action"] = "nudge"
        result["urgency"] = 0.5

    return result


def _level_to_R(level: FlowLevel) -> float:
    """Convert flow level to target R value."""
    mapping = {
        FlowLevel.NONE: 0.0,
        FlowLevel.EMERGING: R_EMERGING,
        FlowLevel.BUILDING: R_BUILDING,
        FlowLevel.FLOW: R_FLOW,
        FlowLevel.DEEP_FLOW: R_DEEP_FLOW
    }
    return mapping.get(level, R_FLOW)
