//! WebAssembly bindings for LatticeForge core.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use crate::particles::{Swarm, SwarmConfig, SwarmMetrics};
use crate::persistence::{compute_persistence, persistent_entropy, PersistenceDiagram};
use crate::q_matrix::{analyze_q, build_q_matrix, QMatrixAnalysis};
use crate::geodesics::{integrate_geodesic, fisher_metric_gaussian, GeodesicTrajectory};
use crate::geospatial::{GeospatialSystem, GeospatialConfig, AttractorLayer};

#[cfg(feature = "wasm")]
use console_error_panic_hook;

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "wasm")]
    console_error_panic_hook::set_once();
}

// ============================================================
// Q-Matrix WASM Interface
// ============================================================

/// Build Q-matrix from rates (WASM)
#[wasm_bindgen]
pub fn wasm_build_q_matrix(rates_flat: &[f64], n: usize) -> Result<Vec<f64>, JsValue> {
    use ndarray::Array2;

    let rates = Array2::from_shape_vec((n, n), rates_flat.to_vec())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let q = build_q_matrix(&rates)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(q.into_raw_vec())
}

/// Analyze Q-matrix (WASM)
#[wasm_bindgen]
pub fn wasm_analyze_q(q_flat: &[f64], n: usize) -> Result<JsValue, JsValue> {
    use ndarray::Array2;

    let q = Array2::from_shape_vec((n, n), q_flat.to_vec())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result = analyze_q(&q)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Simulate Markov chain (WASM)
#[wasm_bindgen]
pub fn wasm_simulate_markov_chain(
    q_flat: &[f64],
    n: usize,
    r0: usize,
    total_time: f64,
    dt: f64,
    seed: u64,
) -> Result<JsValue, JsValue> {
    use ndarray::Array2;

    let q = Array2::from_shape_vec((n, n), q_flat.to_vec())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let (times, regimes) = crate::q_matrix::simulate_markov_chain(&q, r0, total_time, dt, seed);

    #[derive(Serialize)]
    struct SimResult {
        times: Vec<f64>,
        regimes: Vec<usize>,
    }

    let result = SimResult { times, regimes };
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ============================================================
// Persistence WASM Interface
// ============================================================

/// Compute persistence diagram (WASM)
#[wasm_bindgen]
pub fn wasm_compute_persistence(
    points_flat: &[f64],
    n_points: usize,
    max_edge: f64,
) -> Result<JsValue, JsValue> {
    let points: Vec<[f64; 2]> = points_flat
        .chunks(2)
        .take(n_points)
        .map(|c| [c[0], c[1]])
        .collect();

    let pd = compute_persistence(&points, max_edge);

    serde_wasm_bindgen::to_value(&pd)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute persistent entropy (WASM)
#[wasm_bindgen]
pub fn wasm_persistent_entropy(
    births: &[f64],
    deaths: &[f64],
) -> f64 {
    use crate::persistence::PersistencePair;

    let pairs: Vec<PersistencePair> = births.iter()
        .zip(deaths.iter())
        .map(|(&b, &d)| PersistencePair { birth: b, death: d, dimension: 0 })
        .collect();

    persistent_entropy(&pairs)
}

// ============================================================
// Geodesic WASM Interface
// ============================================================

/// Integrate geodesic on Fisher metric (WASM)
#[wasm_bindgen]
pub fn wasm_integrate_geodesic_fisher(
    x0: &[f64],
    v0: &[f64],
    dt: f64,
    n_steps: usize,
) -> Result<JsValue, JsValue> {
    if x0.len() < 2 || v0.len() < 2 {
        return Err(JsValue::from_str("x0 and v0 must have at least 2 elements"));
    }

    let traj = integrate_geodesic(
        [x0[0], x0[1]],
        [v0[0], v0[1]],
        fisher_metric_gaussian,
        dt,
        n_steps,
        1e-4,
    );

    serde_wasm_bindgen::to_value(&traj)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ============================================================
// Particle Swarm WASM Interface
// ============================================================

/// WASM-friendly swarm wrapper
#[wasm_bindgen]
pub struct WasmSwarm {
    swarm: Swarm,
    config: SwarmConfig,
    rng_seed: u64,
}

#[wasm_bindgen]
impl WasmSwarm {
    /// Create new swarm
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_particles: usize,
        dt: f64,
        diffusion: f64,
        interaction_strength: f64,
        attractor_x: f64,
        attractor_y: f64,
        attractor_strength: f64,
        seed: u64,
    ) -> WasmSwarm {
        let config = SwarmConfig {
            n_particles,
            dt,
            diffusion,
            interaction_strength,
            attractor_pos: [attractor_x, attractor_y],
            attractor_strength,
        };

        let swarm = Swarm::new(&config, seed);

        WasmSwarm {
            swarm,
            config,
            rng_seed: seed,
        }
    }

    /// Step simulation
    pub fn step(&mut self) {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(self.rng_seed);
        self.swarm.step(&self.config, &mut rng);
        self.rng_seed = self.rng_seed.wrapping_add(1);
    }

    /// Run multiple steps
    pub fn run(&mut self, n_steps: usize) -> JsValue {
        let history = self.swarm.run(&self.config, n_steps, self.rng_seed);
        self.rng_seed = self.rng_seed.wrapping_add(n_steps as u64);

        serde_wasm_bindgen::to_value(&history).unwrap_or(JsValue::NULL)
    }

    /// Get current positions as flat array
    pub fn get_positions(&self) -> Vec<f64> {
        self.swarm.positions_flat()
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> JsValue {
        let metrics = self.swarm.metrics(&self.config.attractor_pos);
        serde_wasm_bindgen::to_value(&metrics).unwrap_or(JsValue::NULL)
    }

    /// Set attractor position
    pub fn set_attractor(&mut self, x: f64, y: f64) {
        self.config.attractor_pos = [x, y];
    }

    /// Get time
    pub fn get_time(&self) -> f64 {
        self.swarm.time
    }

    /// Get number of particles
    pub fn get_n_particles(&self) -> usize {
        self.swarm.particles.len()
    }
}

// ============================================================
// Utility functions
// ============================================================

/// Compute distance matrix (WASM)
#[wasm_bindgen]
pub fn wasm_distance_matrix(points_flat: &[f64], n_points: usize) -> Vec<f64> {
    let points: Vec<[f64; 2]> = points_flat
        .chunks(2)
        .take(n_points)
        .map(|c| [c[0], c[1]])
        .collect();

    let d = crate::persistence::distance_matrix(&points);
    d.into_raw_vec()
}

// ============================================================
// Geospatial System WASM Interface
// ============================================================

/// WASM-friendly geospatial system wrapper
#[wasm_bindgen]
pub struct WasmGeospatialSystem {
    system: GeospatialSystem,
}

#[wasm_bindgen]
impl WasmGeospatialSystem {
    /// Create new geospatial system
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_dims: usize,
        interaction_decay: f64,
        min_influence: f64,
        dt: f64,
        diffusion: f64,
    ) -> WasmGeospatialSystem {
        let config = GeospatialConfig {
            n_dims,
            interaction_decay,
            min_influence,
            dt,
            diffusion,
        };

        WasmGeospatialSystem {
            system: GeospatialSystem::new(config),
        }
    }

    /// Create with default configuration
    #[wasm_bindgen]
    pub fn with_defaults() -> WasmGeospatialSystem {
        WasmGeospatialSystem {
            system: GeospatialSystem::new(GeospatialConfig::default()),
        }
    }

    /// Add a nation to the system
    #[wasm_bindgen]
    pub fn add_nation(
        &mut self,
        code: &str,
        name: &str,
        lat: f64,
        lon: f64,
        regime: usize,
    ) {
        self.system.add_nation(code, name, lat, lon, None, regime);
    }

    /// Add a nation with initial position in attractor space
    #[wasm_bindgen]
    pub fn add_nation_with_position(
        &mut self,
        code: &str,
        name: &str,
        lat: f64,
        lon: f64,
        position: Vec<f64>,
        regime: usize,
    ) {
        self.system.add_nation(code, name, lat, lon, Some(position), regime);
    }

    /// Set esteem relationship between nations
    #[wasm_bindgen]
    pub fn set_esteem(&mut self, source: &str, target: &str, esteem: f64) {
        self.system.set_esteem(source, target, esteem);
    }

    /// Get esteem from source to target
    #[wasm_bindgen]
    pub fn get_esteem(&self, source: &str, target: &str) -> f64 {
        self.system.get_esteem(source, target)
    }

    /// Run one simulation step
    #[wasm_bindgen]
    pub fn step(&mut self) {
        self.system.step();
    }

    /// Run multiple simulation steps
    #[wasm_bindgen]
    pub fn run(&mut self, n_steps: usize) {
        self.system.run(n_steps);
    }

    /// Get current simulation time
    #[wasm_bindgen]
    pub fn get_time(&self) -> f64 {
        self.system.time
    }

    /// Get number of nations
    #[wasm_bindgen]
    pub fn get_nation_count(&self) -> usize {
        self.system.nations.len()
    }

    /// Get number of influence edges
    #[wasm_bindgen]
    pub fn get_edge_count(&self) -> usize {
        self.system.edges.len()
    }

    /// Export to GeoJSON for basin strength visualization
    #[wasm_bindgen]
    pub fn to_geojson_basin(&self) -> JsValue {
        let geojson = self.system.to_geojson(AttractorLayer::BasinStrength);
        serde_wasm_bindgen::to_value(&geojson).unwrap_or(JsValue::NULL)
    }

    /// Export to GeoJSON for transition risk visualization
    #[wasm_bindgen]
    pub fn to_geojson_risk(&self) -> JsValue {
        let geojson = self.system.to_geojson(AttractorLayer::TransitionRisk);
        serde_wasm_bindgen::to_value(&geojson).unwrap_or(JsValue::NULL)
    }

    /// Export to GeoJSON for influence flow visualization
    #[wasm_bindgen]
    pub fn to_geojson_influence(&self) -> JsValue {
        let geojson = self.system.to_geojson(AttractorLayer::InfluenceFlow);
        serde_wasm_bindgen::to_value(&geojson).unwrap_or(JsValue::NULL)
    }

    /// Export to GeoJSON for regime cluster visualization
    #[wasm_bindgen]
    pub fn to_geojson_regime(&self) -> JsValue {
        let geojson = self.system.to_geojson(AttractorLayer::RegimeCluster);
        serde_wasm_bindgen::to_value(&geojson).unwrap_or(JsValue::NULL)
    }

    /// Compare two nations
    #[wasm_bindgen]
    pub fn compare_nations(&self, code1: &str, code2: &str) -> JsValue {
        match self.system.get_comparison(code1, code2) {
            Some(comparison) => serde_wasm_bindgen::to_value(&comparison).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Get nation data as JSON
    #[wasm_bindgen]
    pub fn get_nation(&self, code: &str) -> JsValue {
        match self.system.nations.get(code) {
            Some(nation) => serde_wasm_bindgen::to_value(nation).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    /// Get all nations as JSON array
    #[wasm_bindgen]
    pub fn get_all_nations(&self) -> JsValue {
        let nations: Vec<_> = self.system.nations.values().collect();
        serde_wasm_bindgen::to_value(&nations).unwrap_or(JsValue::NULL)
    }

    /// Get all edges as JSON array
    #[wasm_bindgen]
    pub fn get_all_edges(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.system.edges).unwrap_or(JsValue::NULL)
    }

    /// Serialize entire system state
    #[wasm_bindgen]
    pub fn serialize(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.system)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deserialize system state
    #[wasm_bindgen]
    pub fn deserialize(json: &str) -> Result<WasmGeospatialSystem, JsValue> {
        let system: GeospatialSystem = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmGeospatialSystem { system })
    }
}

/// Compute haversine distance between two points (WASM)
#[wasm_bindgen]
pub fn wasm_haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    crate::geospatial::haversine_distance(lat1, lon1, lat2, lon2)
}
