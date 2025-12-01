"""
streaming_tda.py

Continuous/Streaming TDA Early Warning System.

Implements real-time topological monitoring over sliding windows:
- Rolling persistence diagram computation
- Streaming persistent entropy tracking
- Critical slowing down detection
- Phase transition probability estimation

Key insight: Track the *derivative* of topological features,
not just the features themselves. Rapid changes in persistent
entropy signal approaching phase transitions.

References:
- Scheffer et al. "Early-warning signals for critical transitions" (2009)
- Berwald et al. "Critical transitions and perturbation growth directions" (2017)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from collections import deque
from enum import Enum

Array = NDArray[np.float64]


class AlertLevel(Enum):
    """Early warning alert levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class StreamingTDAConfig:
    """Configuration for streaming TDA monitor."""

    # Window parameters
    window_size: int = 100          # Points per window
    window_overlap: float = 0.5     # Overlap fraction
    n_windows_history: int = 50     # Windows to track

    # TDA parameters
    max_edge_length: float = 2.0    # Rips filtration limit
    n_persistence_features: int = 20  # Top features to track

    # Early warning thresholds
    entropy_spike_threshold: float = 0.3    # Relative change threshold
    variance_spike_threshold: float = 2.0   # Std devs from mean
    autocorr_threshold: float = 0.8         # Critical slowing threshold

    # Smoothing
    smoothing_alpha: float = 0.1    # Exponential smoothing factor


@dataclass
class TDASnapshot:
    """Snapshot of TDA state at one window."""
    timestamp: float
    persistent_entropy: float
    h0_count: int
    h1_count: int
    total_persistence: float
    variance: float
    autocorrelation: float
    centroid: Array


@dataclass
class EarlyWarningSignal:
    """Early warning signal with diagnostics."""
    timestamp: float
    alert_level: AlertLevel
    phase_transition_probability: float
    entropy_trend: float           # Derivative of entropy
    variance_trend: float          # Derivative of variance
    autocorr_trend: float          # Derivative of autocorrelation
    contributing_factors: List[str]
    recommendation: str


class StreamingTDAMonitor:
    """
    Real-time TDA early warning monitor.

    Continuously processes position data over sliding windows,
    computing topological features and detecting critical transitions.

    Example
    -------
    >>> monitor = StreamingTDAMonitor(StreamingTDAConfig())
    >>> for positions in data_stream:
    ...     signal = monitor.update(positions, timestamp)
    ...     if signal.alert_level == AlertLevel.CRITICAL:
    ...         trigger_intervention()
    """

    def __init__(self, config: StreamingTDAConfig = StreamingTDAConfig()):
        self.cfg = config

        # Rolling buffers
        self.position_buffer: deque = deque(maxlen=config.window_size * 2)
        self.snapshot_history: deque = deque(maxlen=config.n_windows_history)

        # Smoothed statistics
        self.smoothed_entropy = 0.0
        self.smoothed_variance = 0.0
        self.smoothed_autocorr = 0.0

        # Baseline statistics (updated during stable periods)
        self.baseline_entropy_mean = 0.0
        self.baseline_entropy_std = 0.1
        self.baseline_variance_mean = 0.0
        self.baseline_variance_std = 0.1
        self.in_baseline_mode = True
        self.baseline_samples = 0

        # Current window
        self.current_window: List[Array] = []
        self.window_count = 0
        self.last_timestamp = 0.0

    def _compute_persistence(self, points: Array) -> Tuple[Array, Array]:
        """Compute persistence diagrams H0 and H1."""
        from scipy.spatial.distance import pdist, squareform

        if len(points) < 3:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        # Distance matrix
        D = squareform(pdist(points))
        n = len(points)
        max_eps = self.cfg.max_edge_length

        # H0 via Union-Find
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] <= max_eps:
                    edges.append((D[i, j], i, j))
        edges.sort()

        deaths = {}
        for eps, i, j in edges:
            pi, pj = find(i), find(j)
            if pi != pj:
                dying = max(pi, pj)
                deaths[dying] = eps
                union(i, j)

        H0 = []
        for i in range(n):
            if parent[i] == i and i in deaths and deaths[i] > 0:
                H0.append([0.0, deaths[i]])

        # Simplified H1 (triangle-based)
        H1 = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    edges_tri = sorted([D[i, j], D[j, k], D[i, k]])
                    if edges_tri[2] <= max_eps:
                        birth, death = edges_tri[1], edges_tri[2]
                        if death > birth:
                            H1.append([birth, death])

        H0 = np.array(H0).reshape(-1, 2) if H0 else np.array([]).reshape(0, 2)
        H1 = np.array(H1).reshape(-1, 2) if H1 else np.array([]).reshape(0, 2)

        return H0, H1

    def _persistent_entropy(self, pairs: Array) -> float:
        """Compute persistent entropy."""
        if len(pairs) == 0:
            return 0.0

        lifetimes = pairs[:, 1] - pairs[:, 0]
        lifetimes = np.maximum(lifetimes, 1e-12)
        total = lifetimes.sum()

        if total < 1e-12:
            return 0.0

        probs = lifetimes / total
        return -float(np.sum(probs * np.log(probs + 1e-12)))

    def _total_persistence(self, pairs: Array) -> float:
        """Total persistence (sum of lifetimes)."""
        if len(pairs) == 0:
            return 0.0
        return float(np.sum(pairs[:, 1] - pairs[:, 0]))

    def _compute_autocorrelation(self, lag: int = 1) -> float:
        """Compute autocorrelation of centroid trajectory."""
        if len(self.snapshot_history) < lag + 2:
            return 0.0

        centroids = [s.centroid for s in self.snapshot_history]
        x_t = np.array(centroids[lag:])
        x_prev = np.array(centroids[:-lag])

        x_t_centered = x_t - x_t.mean(axis=0)
        x_prev_centered = x_prev - x_prev.mean(axis=0)

        num = np.sum(x_t_centered * x_prev_centered)
        den = np.sqrt(np.sum(x_t_centered**2) * np.sum(x_prev_centered**2) + 1e-12)

        return float(num / den)

    def _process_window(self, positions: Array, timestamp: float) -> TDASnapshot:
        """Process a single window and return snapshot."""
        H0, H1 = self._compute_persistence(positions)

        all_pairs = np.vstack([H0, H1]) if len(H0) > 0 or len(H1) > 0 else np.array([]).reshape(0, 2)

        entropy = self._persistent_entropy(all_pairs)
        total_pers = self._total_persistence(all_pairs)
        centroid = positions.mean(axis=0)
        variance = float(np.mean(np.sum((positions - centroid)**2, axis=1)))
        autocorr = self._compute_autocorrelation()

        return TDASnapshot(
            timestamp=timestamp,
            persistent_entropy=entropy,
            h0_count=len(H0),
            h1_count=len(H1),
            total_persistence=total_pers,
            variance=variance,
            autocorrelation=autocorr,
            centroid=centroid,
        )

    def _update_baseline(self, snapshot: TDASnapshot) -> None:
        """Update baseline statistics during stable periods."""
        self.baseline_samples += 1
        alpha = 1.0 / self.baseline_samples

        # Running mean and variance
        delta = snapshot.persistent_entropy - self.baseline_entropy_mean
        self.baseline_entropy_mean += alpha * delta
        self.baseline_entropy_std = np.sqrt(
            (1 - alpha) * self.baseline_entropy_std**2 + alpha * delta**2
        )

        delta_v = snapshot.variance - self.baseline_variance_mean
        self.baseline_variance_mean += alpha * delta_v
        self.baseline_variance_std = np.sqrt(
            (1 - alpha) * self.baseline_variance_std**2 + alpha * delta_v**2
        )

    def _compute_trends(self) -> Tuple[float, float, float]:
        """Compute derivative trends from history."""
        if len(self.snapshot_history) < 3:
            return 0.0, 0.0, 0.0

        recent = list(self.snapshot_history)[-10:]

        # Linear regression for trends
        t = np.arange(len(recent))
        entropy = np.array([s.persistent_entropy for s in recent])
        variance = np.array([s.variance for s in recent])
        autocorr = np.array([s.autocorrelation for s in recent])

        def slope(y):
            if len(y) < 2:
                return 0.0
            return float(np.polyfit(t, y, 1)[0])

        return slope(entropy), slope(variance), slope(autocorr)

    def _compute_phase_transition_probability(
        self,
        snapshot: TDASnapshot,
        entropy_trend: float,
        variance_trend: float,
        autocorr_trend: float,
    ) -> float:
        """
        Estimate probability of approaching phase transition.

        Combines multiple indicators:
        1. Entropy deviation from baseline
        2. Variance increase (critical slowing down)
        3. Autocorrelation increase (critical slowing down)
        4. Rate of change of indicators
        """
        p = 0.0
        weights = 0.0

        # Entropy deviation
        if self.baseline_entropy_std > 1e-6:
            z_entropy = abs(snapshot.persistent_entropy - self.baseline_entropy_mean) / self.baseline_entropy_std
            p += min(z_entropy / 3.0, 1.0) * 0.3
            weights += 0.3

        # Variance increase
        if self.baseline_variance_std > 1e-6:
            z_variance = (snapshot.variance - self.baseline_variance_mean) / self.baseline_variance_std
            if z_variance > 0:
                p += min(z_variance / self.cfg.variance_spike_threshold, 1.0) * 0.25
            weights += 0.25

        # Autocorrelation
        if snapshot.autocorrelation > self.cfg.autocorr_threshold:
            p += (snapshot.autocorrelation - self.cfg.autocorr_threshold) / (1 - self.cfg.autocorr_threshold) * 0.25
            weights += 0.25

        # Trend acceleration
        if entropy_trend > 0:
            p += min(entropy_trend / 0.1, 1.0) * 0.1
        if variance_trend > 0:
            p += min(variance_trend / 0.5, 1.0) * 0.1
        weights += 0.2

        return min(p / max(weights, 0.1), 1.0)

    def _determine_alert_level(self, prob: float) -> AlertLevel:
        """Determine alert level from phase transition probability."""
        if prob < 0.25:
            return AlertLevel.NORMAL
        elif prob < 0.5:
            return AlertLevel.ELEVATED
        elif prob < 0.75:
            return AlertLevel.WARNING
        else:
            return AlertLevel.CRITICAL

    def _generate_recommendation(self, alert_level: AlertLevel, factors: List[str]) -> str:
        """Generate action recommendation based on alert."""
        if alert_level == AlertLevel.NORMAL:
            return "Continue monitoring. System in stable state."
        elif alert_level == AlertLevel.ELEVATED:
            return f"Increased vigilance recommended. Factors: {', '.join(factors)}"
        elif alert_level == AlertLevel.WARNING:
            return f"Prepare intervention. System approaching transition. Factors: {', '.join(factors)}"
        else:
            return f"IMMEDIATE ACTION REQUIRED. Phase transition imminent. Factors: {', '.join(factors)}"

    def update(self, positions: Array, timestamp: float) -> EarlyWarningSignal:
        """
        Process new position data and return early warning signal.

        Parameters
        ----------
        positions : array of shape (n, 2)
            Current particle positions.
        timestamp : float
            Current simulation time.

        Returns
        -------
        EarlyWarningSignal
            Contains alert level, probabilities, and recommendations.
        """
        # Process window
        snapshot = self._process_window(positions, timestamp)
        self.snapshot_history.append(snapshot)

        # Update smoothed values
        alpha = self.cfg.smoothing_alpha
        self.smoothed_entropy = alpha * snapshot.persistent_entropy + (1 - alpha) * self.smoothed_entropy
        self.smoothed_variance = alpha * snapshot.variance + (1 - alpha) * self.smoothed_variance
        self.smoothed_autocorr = alpha * snapshot.autocorrelation + (1 - alpha) * self.smoothed_autocorr

        # Update baseline if in stable mode
        if self.in_baseline_mode and self.baseline_samples < 100:
            self._update_baseline(snapshot)

        # Compute trends
        entropy_trend, variance_trend, autocorr_trend = self._compute_trends()

        # Compute phase transition probability
        prob = self._compute_phase_transition_probability(
            snapshot, entropy_trend, variance_trend, autocorr_trend
        )

        # Identify contributing factors
        factors = []
        if self.baseline_entropy_std > 1e-6:
            z = abs(snapshot.persistent_entropy - self.baseline_entropy_mean) / self.baseline_entropy_std
            if z > 2:
                factors.append(f"entropy_deviation={z:.2f}σ")

        if snapshot.autocorrelation > self.cfg.autocorr_threshold:
            factors.append(f"high_autocorr={snapshot.autocorrelation:.2f}")

        if variance_trend > 0.1:
            factors.append(f"variance_increasing")

        if entropy_trend > 0.05:
            factors.append(f"entropy_rising")

        # Determine alert level
        alert_level = self._determine_alert_level(prob)

        # Exit baseline mode if we see elevated signals
        if alert_level != AlertLevel.NORMAL:
            self.in_baseline_mode = False

        recommendation = self._generate_recommendation(alert_level, factors)

        self.window_count += 1
        self.last_timestamp = timestamp

        return EarlyWarningSignal(
            timestamp=timestamp,
            alert_level=alert_level,
            phase_transition_probability=prob,
            entropy_trend=entropy_trend,
            variance_trend=variance_trend,
            autocorr_trend=autocorr_trend,
            contributing_factors=factors,
            recommendation=recommendation,
        )

    def get_history_dataframe(self) -> Dict[str, List]:
        """Get snapshot history as dict (for DataFrame conversion)."""
        return {
            "timestamp": [s.timestamp for s in self.snapshot_history],
            "entropy": [s.persistent_entropy for s in self.snapshot_history],
            "h0_count": [s.h0_count for s in self.snapshot_history],
            "h1_count": [s.h1_count for s in self.snapshot_history],
            "variance": [s.variance for s in self.snapshot_history],
            "autocorrelation": [s.autocorrelation for s in self.snapshot_history],
        }

    def reset_baseline(self) -> None:
        """Reset baseline statistics for recalibration."""
        self.baseline_entropy_mean = 0.0
        self.baseline_entropy_std = 0.1
        self.baseline_variance_mean = 0.0
        self.baseline_variance_std = 0.1
        self.in_baseline_mode = True
        self.baseline_samples = 0
