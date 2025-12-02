"""
Visualization utilities for Great Attractor simulations.

Components:
- plot_basins: Basin boundary and potential field visualization
- plot_geodesics: Geodesic flow on information manifold
- plot_persistence: TDA persistence diagram visualization
- anim_particles: Particle animation utilities
"""

from .plot_basins import (
    BasinPlotConfig,
    BasinPlotter,
)

from .plot_geodesics import (
    GeodesicPlotConfig,
    GeodesicPlotter,
)

from .plot_persistence import (
    PersistencePlotConfig,
    PersistencePlotter,
    persistent_entropy,
    compute_distance_matrix,
)

from .anim_particles import (
    AnimationConfig,
    ParticleAnimator,
    animate_from_history,
)

__all__ = [
    # Basin plots
    "BasinPlotConfig",
    "BasinPlotter",
    # Geodesic flows
    "GeodesicPlotConfig",
    "GeodesicPlotter",
    # Persistence diagrams
    "PersistencePlotConfig",
    "PersistencePlotter",
    "persistent_entropy",
    "compute_distance_matrix",
    # Animation
    "AnimationConfig",
    "ParticleAnimator",
    "animate_from_history",
]
