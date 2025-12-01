//! WebAssembly bindings for LatticeForge core.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use crate::particles::{Swarm, SwarmConfig, SwarmMetrics};
use crate::persistence::{compute_persistence, persistent_entropy, PersistenceDiagram};
use crate::q_matrix::{analyze_q, build_q_matrix, QMatrixAnalysis};
use crate::geodesics::{integrate_geodesic, fisher_metric_gaussian, GeodesicTrajectory};

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
