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

__all__ = [
    "BasinPlotConfig",
    "BasinPlotter",
    "GeodesicPlotConfig",
    "GeodesicPlotter",
]
