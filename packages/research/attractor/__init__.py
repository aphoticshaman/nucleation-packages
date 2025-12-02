"""
Great Attractor simulation and control module.

Components:
- TDAMonitor: Topological Data Analysis early warning indicators
- GAOGSEBridge: OGSE <-> Great Attractor Simulator bridge for control
- Q-matrix utilities: Regime transition analysis and metastability
- Regime SDE: Regime-switching stochastic differential equations
- Visualization: Basin plots, geodesic flows, persistence diagrams, animations
- API: High-level LatticeForge interface
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

from .regime.regime_sde import (
    RegimeSDEConfig,
    RegimeSDE,
)

from .viz.plot_basins import (
    BasinPlotConfig,
    BasinPlotter,
)

from .viz.plot_geodesics import (
    GeodesicPlotConfig,
    GeodesicPlotter,
)

from .viz.plot_persistence import (
    PersistencePlotConfig,
    PersistencePlotter,
    persistent_entropy as tda_persistent_entropy,
    compute_distance_matrix,
)

from .viz.anim_particles import (
    AnimationConfig,
    ParticleAnimator,
    animate_from_history,
)

from .api.latticeforge import (
    LatticeForgeConfig,
    LatticeForge,
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
    # Regime SDE
    "RegimeSDEConfig",
    "RegimeSDE",
    # Visualization - Basins
    "BasinPlotConfig",
    "BasinPlotter",
    # Visualization - Geodesics
    "GeodesicPlotConfig",
    "GeodesicPlotter",
    # Visualization - Persistence
    "PersistencePlotConfig",
    "PersistencePlotter",
    "tda_persistent_entropy",
    "compute_distance_matrix",
    # Visualization - Animation
    "AnimationConfig",
    "ParticleAnimator",
    "animate_from_history",
    # High-level API
    "LatticeForgeConfig",
    "LatticeForge",
]
