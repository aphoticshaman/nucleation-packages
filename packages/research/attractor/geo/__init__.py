"""
Geospatial attractor mapping for nation-level dynamics.

Designed for Google Maps integration with layers/filters/colors.

Components:
- NationAttractor: Per-nation attractor state
- InfluenceEdge: Cross-national influence flows
- GeospatialAttractorSystem: Full system dynamics
- GeoJSON export for visualization
"""

from .geospatial_attractors import (
    AttractorLayer,
    NationAttractor,
    InfluenceEdge,
    GeospatialConfig,
    NationDistanceKernel,
    GeospatialAttractorSystem,
)

__all__ = [
    "AttractorLayer",
    "NationAttractor",
    "InfluenceEdge",
    "GeospatialConfig",
    "NationDistanceKernel",
    "GeospatialAttractorSystem",
]
