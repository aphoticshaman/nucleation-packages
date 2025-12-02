"""
geospatial_attractors.py

Geospatial Attractor Mapping for Nation-Based Analysis.

Maps Great Attractor dynamics onto geographic coordinates for:
- Nation-level attractor basin visualization
- Cross-national influence flows (geodesic paths)
- Regional phase transition risk mapping
- Comparative esteem/perception modeling

Designed for Google Maps integration with layers/filters/colors.

Key concepts:
- Each nation has an "attractor position" in policy/sentiment space
- Cross-national influence is modeled as mean-field interactions
- Cultural/economic proximity defines interaction kernel
- Phase transitions represent major geopolitical shifts
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from enum import Enum
import json

Array = NDArray[np.float64]


class AttractorLayer(Enum):
    """Available visualization layers."""
    BASIN_STRENGTH = "basin_strength"           # Attractor strength/stability
    INFLUENCE_FLOW = "influence_flow"           # Directed influence arrows
    TRANSITION_RISK = "transition_risk"         # Phase transition probability
    ESTEEM_NETWORK = "esteem_network"           # Mutual perception matrix
    REGIME_CLUSTER = "regime_cluster"           # Regime clustering
    GEODESIC_DISTANCE = "geodesic_distance"     # Information-geometric distance


@dataclass
class NationAttractor:
    """Attractor state for a single nation."""
    code: str                      # ISO country code
    name: str
    lat: float                     # Geographic latitude
    lon: float                     # Geographic longitude
    position: Array                # Position in attractor space (policy dims)
    velocity: Array                # Rate of change
    basin_strength: float          # Local stability
    transition_risk: float         # Probability of regime shift
    regime: int                    # Current regime cluster
    influence_radius: float        # Sphere of influence
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceEdge:
    """Directed influence between two nations."""
    source: str                    # Source nation code
    target: str                    # Target nation code
    strength: float                # Influence strength [0, 1]
    direction: Array               # Direction in attractor space
    geodesic_distance: float       # Information-geometric distance
    esteem: float                  # How source views target [-1, 1]


@dataclass
class GeospatialConfig:
    """Configuration for geospatial attractor system."""

    # Attractor space dimensions
    n_dims: int = 4                # Policy/sentiment dimensions

    # Interaction parameters
    interaction_decay: float = 0.001   # Distance decay for influence
    min_influence: float = 0.01        # Minimum influence threshold

    # Visualization
    color_scheme: str = "viridis"
    arrow_scale: float = 1.0
    node_scale: float = 1.0

    # Dynamics
    dt: float = 0.01
    diffusion: float = 0.05


class NationDistanceKernel:
    """Compute cultural/economic distance between nations."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize distance kernel.

        Parameters
        ----------
        weights : dict, optional
            Weights for different distance components:
            - geographic: Physical distance
            - cultural: Language, religion, history
            - economic: Trade, GDP similarity
            - political: Regime type, alliances
        """
        self.weights = weights or {
            "geographic": 0.3,
            "cultural": 0.3,
            "economic": 0.2,
            "political": 0.2,
        }

    def haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Compute great-circle distance in km."""
        R = 6371  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def compute_distance(
        self,
        n1: NationAttractor,
        n2: NationAttractor,
        cultural_matrix: Optional[Array] = None,
    ) -> float:
        """
        Compute weighted distance between two nations.

        Returns normalized distance in [0, 1].
        """
        # Geographic distance (normalized by max ~20000 km)
        geo_dist = self.haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon) / 20000

        # Attractor space distance
        attr_dist = np.linalg.norm(n1.position - n2.position)
        attr_dist = min(attr_dist / 10.0, 1.0)  # Normalize

        # Combined (cultural/economic would come from external data)
        dist = (
            self.weights["geographic"] * geo_dist +
            self.weights["cultural"] * attr_dist * 0.5 +
            self.weights["economic"] * attr_dist * 0.5 +
            self.weights["political"] * attr_dist
        )

        return min(dist, 1.0)


class GeospatialAttractorSystem:
    """
    Geospatial Great Attractor system for nation-level dynamics.

    Example
    -------
    >>> system = GeospatialAttractorSystem(GeospatialConfig())
    >>> system.add_nation("US", "United States", 39.8, -98.6, position=[0.5, 0.3, 0.8, 0.6])
    >>> system.add_nation("CN", "China", 35.0, 105.0, position=[0.7, 0.9, 0.3, 0.5])
    >>> system.step()
    >>> geojson = system.to_geojson(layer=AttractorLayer.BASIN_STRENGTH)
    """

    def __init__(self, config: GeospatialConfig = GeospatialConfig()):
        self.cfg = config
        self.nations: Dict[str, NationAttractor] = {}
        self.edges: List[InfluenceEdge] = []
        self.distance_kernel = NationDistanceKernel()
        self.time = 0.0

        # Esteem matrix: how each nation views others
        self.esteem_matrix: Dict[Tuple[str, str], float] = {}

    def add_nation(
        self,
        code: str,
        name: str,
        lat: float,
        lon: float,
        position: Optional[List[float]] = None,
        regime: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a nation to the system."""
        pos = np.array(position or [0.5] * self.cfg.n_dims)

        self.nations[code] = NationAttractor(
            code=code,
            name=name,
            lat=lat,
            lon=lon,
            position=pos,
            velocity=np.zeros(self.cfg.n_dims),
            basin_strength=1.0,
            transition_risk=0.0,
            regime=regime,
            influence_radius=1.0,
            metadata=metadata or {},
        )

    def set_esteem(self, source: str, target: str, esteem: float) -> None:
        """Set how source nation views target nation."""
        self.esteem_matrix[(source, target)] = np.clip(esteem, -1, 1)

    def get_esteem(self, source: str, target: str) -> float:
        """Get esteem from source to target (default 0 if not set)."""
        return self.esteem_matrix.get((source, target), 0.0)

    def _compute_influence(self, n1: NationAttractor, n2: NationAttractor) -> float:
        """Compute influence strength from n1 to n2."""
        dist = self.distance_kernel.compute_distance(n1, n2)
        influence = np.exp(-dist / self.cfg.interaction_decay)

        # Modulate by esteem
        esteem = self.get_esteem(n1.code, n2.code)
        influence *= (1 + esteem) / 2  # Map esteem [-1,1] to [0,1]

        return max(influence, self.cfg.min_influence) if influence > self.cfg.min_influence else 0.0

    def _update_edges(self) -> None:
        """Recompute influence edges between nations."""
        self.edges = []
        codes = list(self.nations.keys())

        for i, c1 in enumerate(codes):
            for c2 in codes[i+1:]:
                n1, n2 = self.nations[c1], self.nations[c2]

                # Bidirectional influence
                inf_12 = self._compute_influence(n1, n2)
                inf_21 = self._compute_influence(n2, n1)

                if inf_12 > 0:
                    direction = n2.position - n1.position
                    geo_dist = np.linalg.norm(direction)
                    self.edges.append(InfluenceEdge(
                        source=c1,
                        target=c2,
                        strength=inf_12,
                        direction=direction / (geo_dist + 1e-10),
                        geodesic_distance=geo_dist,
                        esteem=self.get_esteem(c1, c2),
                    ))

                if inf_21 > 0:
                    direction = n1.position - n2.position
                    geo_dist = np.linalg.norm(direction)
                    self.edges.append(InfluenceEdge(
                        source=c2,
                        target=c1,
                        strength=inf_21,
                        direction=direction / (geo_dist + 1e-10),
                        geodesic_distance=geo_dist,
                        esteem=self.get_esteem(c2, c1),
                    ))

    def _compute_basin_strength(self, nation: NationAttractor) -> float:
        """Compute local basin stability from incoming influences."""
        incoming = [e for e in self.edges if e.target == nation.code]
        if not incoming:
            return 1.0

        # Stability inversely proportional to diverse incoming influences
        total_influence = sum(e.strength for e in incoming)
        alignment = sum(e.strength * np.dot(e.direction, nation.velocity + 1e-10) for e in incoming)

        stability = 1.0 / (1.0 + total_influence * (1 - alignment / (total_influence + 1e-10)))
        return float(np.clip(stability, 0.1, 1.0))

    def _compute_transition_risk(self, nation: NationAttractor) -> float:
        """Estimate phase transition risk from velocity and basin strength."""
        speed = np.linalg.norm(nation.velocity)
        risk = speed / (nation.basin_strength + 0.1)
        return float(np.clip(risk, 0, 1))

    def step(self) -> None:
        """Advance system by one time step."""
        self._update_edges()
        dt = self.cfg.dt

        # Compute forces on each nation
        forces = {code: np.zeros(self.cfg.n_dims) for code in self.nations}

        for edge in self.edges:
            # Influence pulls target toward source's position
            source = self.nations[edge.source]
            target = self.nations[edge.target]

            pull = edge.strength * (source.position - target.position)
            forces[edge.target] += pull

        # Update positions and velocities
        for code, nation in self.nations.items():
            # Add diffusion noise
            noise = self.cfg.diffusion * np.random.randn(self.cfg.n_dims) * np.sqrt(dt)

            # Update velocity (with damping)
            nation.velocity = 0.9 * nation.velocity + forces[code] * dt

            # Update position
            nation.position = nation.position + nation.velocity * dt + noise

            # Clamp position to [0, 1]
            nation.position = np.clip(nation.position, 0, 1)

            # Update basin strength and transition risk
            nation.basin_strength = self._compute_basin_strength(nation)
            nation.transition_risk = self._compute_transition_risk(nation)

        self.time += dt

    def run(self, n_steps: int) -> List[Dict[str, Any]]:
        """Run simulation for multiple steps, return history."""
        history = []
        for _ in range(n_steps):
            self.step()
            history.append(self.get_state_snapshot())
        return history

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state as dictionary."""
        return {
            "time": self.time,
            "nations": {
                code: {
                    "position": n.position.tolist(),
                    "velocity": n.velocity.tolist(),
                    "basin_strength": n.basin_strength,
                    "transition_risk": n.transition_risk,
                    "regime": n.regime,
                }
                for code, n in self.nations.items()
            },
            "n_edges": len(self.edges),
        }

    def to_geojson(self, layer: AttractorLayer = AttractorLayer.BASIN_STRENGTH) -> Dict[str, Any]:
        """
        Export current state as GeoJSON for map visualization.

        Parameters
        ----------
        layer : AttractorLayer
            Which data layer to encode in properties.

        Returns
        -------
        dict
            GeoJSON FeatureCollection.
        """
        features = []

        for code, nation in self.nations.items():
            # Determine property value based on layer
            if layer == AttractorLayer.BASIN_STRENGTH:
                value = nation.basin_strength
                color_intensity = nation.basin_strength
            elif layer == AttractorLayer.TRANSITION_RISK:
                value = nation.transition_risk
                color_intensity = nation.transition_risk
            elif layer == AttractorLayer.REGIME_CLUSTER:
                value = nation.regime
                color_intensity = nation.regime / 10.0
            else:
                value = nation.basin_strength
                color_intensity = 0.5

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [nation.lon, nation.lat],
                },
                "properties": {
                    "code": code,
                    "name": nation.name,
                    "layer": layer.value,
                    "value": value,
                    "color_intensity": color_intensity,
                    "position": nation.position.tolist(),
                    "basin_strength": nation.basin_strength,
                    "transition_risk": nation.transition_risk,
                    "regime": nation.regime,
                },
            }
            features.append(feature)

        # Add influence edges as LineString features
        if layer == AttractorLayer.INFLUENCE_FLOW:
            for edge in self.edges:
                if edge.strength > 0.1:  # Only significant edges
                    source = self.nations[edge.source]
                    target = self.nations[edge.target]

                    line_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [source.lon, source.lat],
                                [target.lon, target.lat],
                            ],
                        },
                        "properties": {
                            "source": edge.source,
                            "target": edge.target,
                            "strength": edge.strength,
                            "esteem": edge.esteem,
                            "geodesic_distance": edge.geodesic_distance,
                        },
                    }
                    features.append(line_feature)

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "time": self.time,
                "layer": layer.value,
                "n_nations": len(self.nations),
                "n_edges": len(self.edges),
            },
        }

    def get_nation_comparison(self, code1: str, code2: str) -> Dict[str, Any]:
        """Get comparative analysis between two nations."""
        n1 = self.nations.get(code1)
        n2 = self.nations.get(code2)

        if not n1 or not n2:
            return {"error": "Nation not found"}

        # Find edges between them
        edge_12 = next((e for e in self.edges if e.source == code1 and e.target == code2), None)
        edge_21 = next((e for e in self.edges if e.source == code2 and e.target == code1), None)

        return {
            "nations": [code1, code2],
            "distance": {
                "geographic_km": self.distance_kernel.haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon),
                "attractor_space": float(np.linalg.norm(n1.position - n2.position)),
            },
            "mutual_esteem": {
                f"{code1}_views_{code2}": self.get_esteem(code1, code2),
                f"{code2}_views_{code1}": self.get_esteem(code2, code1),
            },
            "influence": {
                f"{code1}_to_{code2}": edge_12.strength if edge_12 else 0.0,
                f"{code2}_to_{code1}": edge_21.strength if edge_21 else 0.0,
            },
            "regime_alignment": n1.regime == n2.regime,
            "position_similarity": float(np.dot(n1.position, n2.position) / (np.linalg.norm(n1.position) * np.linalg.norm(n2.position) + 1e-10)),
        }
