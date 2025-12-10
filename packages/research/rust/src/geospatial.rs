//! Geospatial attractor mapping for nation-level dynamics.
//!
//! Maps Great Attractor dynamics onto geographic coordinates for:
//! - Nation-level attractor basin visualization
//! - Cross-national influence flows
//! - Regional phase transition risk mapping

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Attractor visualization layers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AttractorLayer {
    BasinStrength,
    InfluenceFlow,
    TransitionRisk,
    EsteemNetwork,
    RegimeCluster,
    GeodesicDistance,
}

/// Nation attractor state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NationAttractor {
    pub code: String,
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub position: Vec<f64>,      // Position in attractor space
    pub velocity: Vec<f64>,      // Rate of change
    pub basin_strength: f64,     // Local stability [0, 1]
    pub transition_risk: f64,    // Phase transition probability [0, 1]
    pub regime: usize,           // Current regime cluster
    pub influence_radius: f64,   // Sphere of influence
}

/// Directed influence between nations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluenceEdge {
    pub source: String,
    pub target: String,
    pub strength: f64,
    pub geodesic_distance: f64,
    pub esteem: f64,             // How source views target [-1, 1]
}

/// Geospatial system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeospatialConfig {
    pub n_dims: usize,
    pub interaction_decay: f64,
    pub min_influence: f64,
    pub dt: f64,
    pub diffusion: f64,
}

impl Default for GeospatialConfig {
    fn default() -> Self {
        GeospatialConfig {
            n_dims: 4,
            interaction_decay: 0.001,
            min_influence: 0.01,
            dt: 0.01,
            diffusion: 0.05,
        }
    }
}

/// GeoJSON Feature for map export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonFeature {
    #[serde(rename = "type")]
    pub feature_type: String,
    pub geometry: GeoJsonGeometry,
    pub properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonGeometry {
    #[serde(rename = "type")]
    pub geom_type: String,
    pub coordinates: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoJsonCollection {
    #[serde(rename = "type")]
    pub collection_type: String,
    pub features: Vec<GeoJsonFeature>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Compute haversine distance between two lat/lon points (km)
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371.0; // Earth radius in km

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();

    let a = (dlat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    R * c
}

/// Main geospatial attractor system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeospatialSystem {
    pub config: GeospatialConfig,
    pub nations: HashMap<String, NationAttractor>,
    pub edges: Vec<InfluenceEdge>,
    pub esteem_matrix: HashMap<(String, String), f64>,
    pub time: f64,
}

impl GeospatialSystem {
    pub fn new(config: GeospatialConfig) -> Self {
        GeospatialSystem {
            config,
            nations: HashMap::new(),
            edges: Vec::new(),
            esteem_matrix: HashMap::new(),
            time: 0.0,
        }
    }

    pub fn add_nation(
        &mut self,
        code: &str,
        name: &str,
        lat: f64,
        lon: f64,
        position: Option<Vec<f64>>,
        regime: usize,
    ) {
        let pos = position.unwrap_or_else(|| vec![0.5; self.config.n_dims]);

        self.nations.insert(
            code.to_string(),
            NationAttractor {
                code: code.to_string(),
                name: name.to_string(),
                lat,
                lon,
                position: pos,
                velocity: vec![0.0; self.config.n_dims],
                basin_strength: 1.0,
                transition_risk: 0.0,
                regime,
                influence_radius: 1.0,
            },
        );
    }

    pub fn set_esteem(&mut self, source: &str, target: &str, esteem: f64) {
        let clamped = esteem.max(-1.0).min(1.0);
        self.esteem_matrix
            .insert((source.to_string(), target.to_string()), clamped);
    }

    pub fn get_esteem(&self, source: &str, target: &str) -> f64 {
        *self
            .esteem_matrix
            .get(&(source.to_string(), target.to_string()))
            .unwrap_or(&0.0)
    }

    fn compute_distance(&self, n1: &NationAttractor, n2: &NationAttractor) -> f64 {
        // Geographic distance (normalized)
        let geo_dist = haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon) / 20000.0;

        // Attractor space distance
        let attr_dist: f64 = n1
            .position
            .iter()
            .zip(n2.position.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let attr_dist_norm = (attr_dist / 10.0).min(1.0);

        // Combined distance
        (0.3 * geo_dist + 0.7 * attr_dist_norm).min(1.0)
    }

    fn compute_influence(&self, source: &NationAttractor, target: &NationAttractor) -> f64 {
        let dist = self.compute_distance(source, target);
        let base_influence = (-dist / self.config.interaction_decay).exp();

        // Modulate by esteem
        let esteem = self.get_esteem(&source.code, &target.code);
        let influence = base_influence * (1.0 + esteem) / 2.0;

        if influence > self.config.min_influence {
            influence
        } else {
            0.0
        }
    }

    fn update_edges(&mut self) {
        self.edges.clear();
        let codes: Vec<String> = self.nations.keys().cloned().collect();

        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                let c1 = &codes[i];
                let c2 = &codes[j];

                let n1 = self.nations.get(c1).unwrap().clone();
                let n2 = self.nations.get(c2).unwrap().clone();

                // Bidirectional influence
                let inf_12 = self.compute_influence(&n1, &n2);
                let inf_21 = self.compute_influence(&n2, &n1);

                let geo_dist = haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon);

                if inf_12 > 0.0 {
                    self.edges.push(InfluenceEdge {
                        source: c1.clone(),
                        target: c2.clone(),
                        strength: inf_12,
                        geodesic_distance: geo_dist,
                        esteem: self.get_esteem(c1, c2),
                    });
                }

                if inf_21 > 0.0 {
                    self.edges.push(InfluenceEdge {
                        source: c2.clone(),
                        target: c1.clone(),
                        strength: inf_21,
                        geodesic_distance: geo_dist,
                        esteem: self.get_esteem(c2, c1),
                    });
                }
            }
        }
    }

    fn compute_basin_strength(&self, nation: &NationAttractor) -> f64 {
        let incoming: Vec<&InfluenceEdge> = self
            .edges
            .iter()
            .filter(|e| e.target == nation.code)
            .collect();

        if incoming.is_empty() {
            return 1.0;
        }

        let total_influence: f64 = incoming.iter().map(|e| e.strength).sum();
        let stability = 1.0 / (1.0 + total_influence);

        stability.max(0.1).min(1.0)
    }

    fn compute_transition_risk(&self, nation: &NationAttractor) -> f64 {
        let speed: f64 = nation.velocity.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
        let risk = speed / (nation.basin_strength + 0.1);
        risk.max(0.0).min(1.0)
    }

    pub fn step(&mut self) {
        use rand::Rng;

        self.update_edges();
        let dt = self.config.dt;

        // Compute forces
        let mut forces: HashMap<String, Vec<f64>> = HashMap::new();
        for code in self.nations.keys() {
            forces.insert(code.clone(), vec![0.0; self.config.n_dims]);
        }

        for edge in &self.edges {
            let source = self.nations.get(&edge.source).unwrap();
            let target = self.nations.get(&edge.target).unwrap();

            let pull: Vec<f64> = source
                .position
                .iter()
                .zip(target.position.iter())
                .map(|(s, t)| edge.strength * (s - t))
                .collect();

            if let Some(force) = forces.get_mut(&edge.target) {
                for (f, p) in force.iter_mut().zip(pull.iter()) {
                    *f += p;
                }
            }
        }

        // Update nations
        let mut rng = rand::thread_rng();
        let sqrt_dt = dt.sqrt();

        let codes: Vec<String> = self.nations.keys().cloned().collect();
        for code in codes {
            let force = forces.get(&code).unwrap().clone();

            // First update velocity and position
            {
                let nation = self.nations.get_mut(&code).unwrap();

                // Update velocity with damping
                for (v, f) in nation.velocity.iter_mut().zip(force.iter()) {
                    *v = 0.9 * *v + f * dt;
                }

                // Update position with noise
                for (p, v) in nation.position.iter_mut().zip(nation.velocity.iter()) {
                    let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                    *p += v * dt + self.config.diffusion * sqrt_dt * noise;
                    *p = p.max(0.0).min(1.0);
                }
            }

            // Now compute metrics (immutable borrow of self is ok)
            let nation_ref = self.nations.get(&code).unwrap();
            let basin_strength = self.compute_basin_strength(nation_ref);
            let transition_risk = self.compute_transition_risk(nation_ref);

            // Update metrics
            let nation = self.nations.get_mut(&code).unwrap();
            nation.basin_strength = basin_strength;
            nation.transition_risk = transition_risk;
        }

        self.time += dt;
    }

    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    pub fn to_geojson(&self, layer: AttractorLayer) -> GeoJsonCollection {
        let mut features = Vec::new();

        // Add nation points
        for nation in self.nations.values() {
            let value = match layer {
                AttractorLayer::BasinStrength => nation.basin_strength,
                AttractorLayer::TransitionRisk => nation.transition_risk,
                AttractorLayer::RegimeCluster => nation.regime as f64 / 10.0,
                _ => nation.basin_strength,
            };

            let mut properties = HashMap::new();
            properties.insert("code".to_string(), serde_json::json!(nation.code));
            properties.insert("name".to_string(), serde_json::json!(nation.name));
            properties.insert("value".to_string(), serde_json::json!(value));
            properties.insert(
                "basin_strength".to_string(),
                serde_json::json!(nation.basin_strength),
            );
            properties.insert(
                "transition_risk".to_string(),
                serde_json::json!(nation.transition_risk),
            );
            properties.insert("regime".to_string(), serde_json::json!(nation.regime));
            properties.insert("position".to_string(), serde_json::json!(nation.position));

            features.push(GeoJsonFeature {
                feature_type: "Feature".to_string(),
                geometry: GeoJsonGeometry {
                    geom_type: "Point".to_string(),
                    coordinates: serde_json::json!([nation.lon, nation.lat]),
                },
                properties,
            });
        }

        // Add influence edges as lines
        if layer == AttractorLayer::InfluenceFlow {
            for edge in &self.edges {
                if edge.strength > 0.1 {
                    let source = self.nations.get(&edge.source).unwrap();
                    let target = self.nations.get(&edge.target).unwrap();

                    let mut properties = HashMap::new();
                    properties.insert("source".to_string(), serde_json::json!(edge.source));
                    properties.insert("target".to_string(), serde_json::json!(edge.target));
                    properties.insert("strength".to_string(), serde_json::json!(edge.strength));
                    properties.insert("esteem".to_string(), serde_json::json!(edge.esteem));

                    features.push(GeoJsonFeature {
                        feature_type: "Feature".to_string(),
                        geometry: GeoJsonGeometry {
                            geom_type: "LineString".to_string(),
                            coordinates: serde_json::json!([
                                [source.lon, source.lat],
                                [target.lon, target.lat]
                            ]),
                        },
                        properties,
                    });
                }
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert("time".to_string(), serde_json::json!(self.time));
        metadata.insert(
            "layer".to_string(),
            serde_json::json!(format!("{:?}", layer)),
        );
        metadata.insert(
            "n_nations".to_string(),
            serde_json::json!(self.nations.len()),
        );

        GeoJsonCollection {
            collection_type: "FeatureCollection".to_string(),
            features,
            metadata,
        }
    }

    pub fn get_comparison(&self, code1: &str, code2: &str) -> Option<HashMap<String, serde_json::Value>> {
        let n1 = self.nations.get(code1)?;
        let n2 = self.nations.get(code2)?;

        let mut result = HashMap::new();

        result.insert(
            "geographic_distance_km".to_string(),
            serde_json::json!(haversine_distance(n1.lat, n1.lon, n2.lat, n2.lon)),
        );

        let attr_dist: f64 = n1
            .position
            .iter()
            .zip(n2.position.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        result.insert(
            "attractor_distance".to_string(),
            serde_json::json!(attr_dist),
        );

        result.insert(
            format!("{}_views_{}", code1, code2),
            serde_json::json!(self.get_esteem(code1, code2)),
        );
        result.insert(
            format!("{}_views_{}", code2, code1),
            serde_json::json!(self.get_esteem(code2, code1)),
        );

        result.insert(
            "regime_aligned".to_string(),
            serde_json::json!(n1.regime == n2.regime),
        );

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine() {
        // NYC to London ≈ 5570 km
        let dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278);
        assert!((dist - 5570.0).abs() < 100.0);
    }

    #[test]
    fn test_geospatial_system() {
        let mut system = GeospatialSystem::new(GeospatialConfig::default());

        system.add_nation("US", "United States", 39.8, -98.6, None, 0);
        system.add_nation("CN", "China", 35.0, 105.0, None, 1);
        system.add_nation("GB", "United Kingdom", 51.5, -0.1, None, 0);

        system.set_esteem("US", "GB", 0.8);
        system.set_esteem("GB", "US", 0.7);
        system.set_esteem("US", "CN", -0.3);

        system.run(100);

        assert!(system.time > 0.0);
        assert!(!system.edges.is_empty());

        let geojson = system.to_geojson(AttractorLayer::BasinStrength);
        assert_eq!(geojson.features.len(), 3);
    }
}
