"""
tda_monitor.py

Topological Data Analysis (TDA) early-warning indicators
for Great Attractor formation, collapse, or bifurcation.

Uses:
- H0 persistence (component persistence)
- H1 persistence (loop persistence)
- Persistent entropy
- Critical slowing down (variance, autocorrelation)
- Manifold collapse (spectral gap of covariance)

Integrates with the GreatAttractorSimulator via "update" calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from numpy.typing import NDArray

try:
    # optional
    from gtda.homology import VietorisRipsPersistence
    GIOTTO_AVAILABLE = True
except ImportError:
    GIOTTO_AVAILABLE = False


Array = NDArray[np.float64]


# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------

def autocorrelation(x: Array) -> float:
    """
    Lag-1 autocorrelation, robust to NaNs.
    """
    if len(x) < 3:
        return 0.0
    x0 = x[:-1]
    x1 = x[1:]
    x0 = x0 - np.nanmean(x0)
    x1 = x1 - np.nanmean(x1)
    denom = np.sqrt(np.nansum(x0**2) * np.nansum(x1**2)) + 1e-12
    return float(np.nansum(x0 * x1) / denom)


def persistent_entropy(diagram: Array) -> float:
    """
    Compute persistent entropy from a diagram of birth-death intervals.
    diagram: array of shape (n_intervals, 2)
    """
    if len(diagram) == 0:
        return 0.0
    lengths = diagram[:, 1] - diagram[:, 0]
    total = np.sum(lengths) + 1e-12
    p = lengths / total
    return float(-np.sum(p * np.log(p + 1e-12)))


# ----------------------------
# CONFIG & HISTORY STRUCTURES
# ----------------------------

@dataclass
class TDAMonitorConfig:
    max_dimension: int = 1          # compute H0 and H1
    epsilon: float = 0.5            # max radius for VR complex
    compute_every: int = 20         # frequency of updates (steps)
    window: int = 200               # sliding window for time-series signals


@dataclass
class TDAMonitorHistory:
    times: List[float] = field(default_factory=list)
    var_trace: List[float] = field(default_factory=list)
    autocorr: List[float] = field(default_factory=list)
    h0_entropy: List[float] = field(default_factory=list)
    h1_entropy: List[float] = field(default_factory=list)
    spectral_gap: List[float] = field(default_factory=list)


# ----------------------------
# TDA MONITOR
# ----------------------------

class TDAMonitor:
    """
    TDA-based early warning system.
    Can track multiple signals:
    - variance collapse
    - autocorrelation rise
    - persistent entropy drop
    - manifold collapse (via covariance spectrum)
    """

    def __init__(
        self,
        config: TDAMonitorConfig,
        dim: int,
    ) -> None:
        self.config = config
        self.dim = dim
        self.history = TDAMonitorHistory()

        # If Giotto is available, instantiate
        if GIOTTO_AVAILABLE:
            self.vr = VietorisRipsPersistence(
                metric="euclidean",
                homology_dimensions=list(range(config.max_dimension + 1)),
            )
        else:
            self.vr = None

        # For critical slowing down indicators
        self.var_series: List[float] = []
        self.mean_series: List[float] = []
        self.time_series: List[float] = []

    # --------------------------------
    # PUBLIC INTERFACE
    # --------------------------------

    def update(
        self,
        t: float,
        particles: Array,
        cov: Array,
    ) -> None:
        """
        Called each simulation step.

        Parameters
        ----------
        t : float
            simulation time
        particles : Array (N, d)
            particle positions of the MV ensemble
        cov : Array (d, d)
            covariance of the MV ensemble
        """

        self.time_series.append(t)

        # --- variance indicator ---
        var = float(np.trace(cov))
        self.var_series.append(var)

        # --- autocorrelation indicator ---
        if len(self.var_series) > self.config.window:
            window_data = self.var_series[-self.config.window :]
            ac = autocorrelation(np.array(window_data))
        else:
            ac = 0.0

        # --- manifold collapse (spectral gap) ---
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        if len(eigvals) >= 2:
            spec_gap = float(eigvals[0] - eigvals[1])
        else:
            spec_gap = float(eigvals[0])

        # --- TDA: H0 & H1 persistence via Giotto (if available) ---
        if (
            self.vr is not None
            and len(particles) > 3
            and len(particles) < 5000  # safety for demo
        ):
            X = particles.reshape(-1, self.dim)
            diagrams = self.vr.fit_transform(X[None, :, :])[0]

            # Diagram is multi-dimensional:
            # filter by homology dimension:
            diag_h0 = diagrams[diagrams[:, 2] == 0][:, :2]
            diag_h1 = diagrams[diagrams[:, 2] == 1][:, :2]

            h0_ent = persistent_entropy(diag_h0)
            h1_ent = persistent_entropy(diag_h1)
        else:
            # fallback: no TDA available
            h0_ent = 0.0
            h1_ent = 0.0

        # --------------------------------
        # Record all indicators
        # --------------------------------
        self.history.times.append(t)
        self.history.var_trace.append(var)
        self.history.autocorr.append(ac)
        self.history.spectral_gap.append(spec_gap)
        self.history.h0_entropy.append(h0_ent)
        self.history.h1_entropy.append(h1_ent)

    # --------------------------------
    # ANALYTIC METHODS
    # --------------------------------

    def detect_early_warning(self) -> Dict[str, Any]:
        """
        Return qualitative indicators of critical transitions.

        Indicators:
        - rising autocorrelation
        - increasing variance
        - decreasing persistent entropy
        - collapsing spectral gap
        """

        hist = self.history

        return {
            "autocorr_trend": float(np.mean(hist.autocorr[-50:])) if len(hist.autocorr) > 50 else 0.0,
            "var_trend": float(np.mean(hist.var_trace[-50:])) if len(hist.var_trace) > 50 else 0.0,
            "h0_entropy_current": hist.h0_entropy[-1] if hist.h0_entropy else 0.0,
            "h1_entropy_current": hist.h1_entropy[-1] if hist.h1_entropy else 0.0,
            "spectral_gap_current": hist.spectral_gap[-1] if hist.spectral_gap else 0.0,
        }

    def get_history(self) -> TDAMonitorHistory:
        return self.history
