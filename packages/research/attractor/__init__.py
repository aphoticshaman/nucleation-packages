"""
Great Attractor simulation and control module.

Components:
- TDAMonitor: Topological Data Analysis early warning indicators
- GAOGSEBridge: OGSE <-> Great Attractor Simulator bridge for control
- Q-matrix utilities: Regime transition analysis and metastability
- Visualization: Basin plots, geodesic flows
"""

from .tda_monitor import (
    TDAMonitorConfig,
    TDAMonitorHistory,
    TDAMonitor,
    autocorrelation,
    persistent_entropy,
)

from .ga_ogse_bridge import (
    GAOGSEBridgeConfig,
    GAOGSEBridge,
)

from .regime.q_matrix import (
    QMatrixResult,
    build_q_matrix,
    is_valid_q,
    estimate_q_from_counts,
    analyze_q,
    simulate_markov_chain,
)

from .viz.plot_basins import (
    BasinPlotConfig,
    BasinPlotter,
)

from .viz.plot_geodesics import (
    GeodesicPlotConfig,
    GeodesicPlotter,
)

__all__ = [
    # TDA Monitor
    "TDAMonitorConfig",
    "TDAMonitorHistory",
    "TDAMonitor",
    "autocorrelation",
    "persistent_entropy",
    # OGSE Bridge
    "GAOGSEBridgeConfig",
    "GAOGSEBridge",
    # Q-Matrix Utilities
    "QMatrixResult",
    "build_q_matrix",
    "is_valid_q",
    "estimate_q_from_counts",
    "analyze_q",
    "simulate_markov_chain",
    # Visualization
    "BasinPlotConfig",
    "BasinPlotter",
    "GeodesicPlotConfig",
    "GeodesicPlotter",
]
