//! WebAssembly bindings for LatticeForge core.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use crate::particles::{Swarm, SwarmConfig, SwarmMetrics};
use crate::persistence::{compute_persistence, persistent_entropy, PersistenceDiagram};
use crate::q_matrix::{analyze_q, build_q_matrix, QMatrixAnalysis};
use crate::geodesics::{integrate_geodesic, fisher_metric_gaussian, GeodesicTrajectory};
use crate::geospatial::{GeospatialSystem, GeospatialConfig, AttractorLayer};
use crate::cic::{CICConfig, CICState, CICPhase, compute_cic, compute_phi, compute_entropy, compute_coherence, determine_phase, detect_crystallization};
use crate::clustering::{ClusteringConfig, Cluster, FusionResult, gauge_clustering, optimal_answer, fuse_signals, test_gauge_invariance};

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

// ============================================================
// CIC (Compression-Integration-Coherence) WASM Interface
// ============================================================

/// Compute CIC functional from samples and values (WASM)
///
/// samples: JSON array of string samples
/// values: numeric values corresponding to samples
/// lambda: entropy penalty weight (default 0.5)
/// gamma: coherence bonus weight (default 0.3)
#[wasm_bindgen]
pub fn wasm_compute_cic(
    samples_json: &str,
    values: &[f64],
    lambda: f64,
    gamma: f64,
) -> Result<JsValue, JsValue> {
    let samples: Vec<String> = serde_json::from_str(samples_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();

    let config = CICConfig { lambda, gamma };
    let state = compute_cic(&sample_refs, values, &config);

    serde_wasm_bindgen::to_value(&state)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute integrated information Φ from string samples (WASM)
#[wasm_bindgen]
pub fn wasm_compute_phi(samples_json: &str) -> Result<f64, JsValue> {
    let samples: Vec<String> = serde_json::from_str(samples_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
    Ok(compute_phi(&sample_refs))
}

/// Compute entropy of numeric values (WASM)
#[wasm_bindgen]
pub fn wasm_compute_entropy(values: &[f64]) -> f64 {
    compute_entropy(values)
}

/// Compute multi-scale coherence (WASM)
#[wasm_bindgen]
pub fn wasm_compute_coherence(values: &[f64], epsilon: f64) -> f64 {
    compute_coherence(values, epsilon)
}

/// Determine CIC phase from state (WASM)
#[wasm_bindgen]
pub fn wasm_determine_phase(
    phi: f64,
    entropy: f64,
    coherence: f64,
    functional: f64,
    confidence: f64,
) -> String {
    let state = CICState {
        phi,
        entropy,
        coherence,
        functional,
        confidence,
    };
    determine_phase(&state).as_str().to_string()
}

// ============================================================
// Gauge-Theoretic Value Clustering WASM Interface
// ============================================================

/// Perform gauge-theoretic clustering (WASM)
///
/// values: numeric values to cluster
/// epsilon: gauge tolerance (default 0.05)
/// Returns JSON array of clusters
#[wasm_bindgen]
pub fn wasm_gauge_clustering(values: &[f64], epsilon: f64) -> Result<JsValue, JsValue> {
    let config = ClusteringConfig {
        epsilon,
        min_cluster_size: 1,
    };

    let clusters = gauge_clustering(values, &config);

    serde_wasm_bindgen::to_value(&clusters)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get optimal answer via value clustering (WASM)
#[wasm_bindgen]
pub fn wasm_optimal_answer(values: &[f64], epsilon: f64) -> f64 {
    let config = ClusteringConfig {
        epsilon,
        min_cluster_size: 1,
    };
    optimal_answer(values, &config)
}

/// Test gauge invariance of values (WASM)
#[wasm_bindgen]
pub fn wasm_test_gauge_invariance(values: &[f64], epsilon: f64) -> Result<JsValue, JsValue> {
    let result = test_gauge_invariance(values, epsilon);

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Fuse multiple signals into reliable value (WASM)
///
/// values: numeric signals to fuse
/// epsilon: gauge tolerance
/// lambda: CIC entropy weight
/// gamma: CIC coherence weight
#[wasm_bindgen]
pub fn wasm_fuse_signals(
    values: &[f64],
    epsilon: f64,
    lambda: f64,
    gamma: f64,
) -> Result<JsValue, JsValue> {
    let cluster_config = ClusteringConfig {
        epsilon,
        min_cluster_size: 1,
    };
    let cic_config = CICConfig { lambda, gamma };

    let result = fuse_signals(values, &cluster_config, &cic_config);

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Batch fuse multiple signal sets (WASM)
///
/// For high-throughput fusion - processes multiple value arrays at once
#[wasm_bindgen]
pub fn wasm_batch_fuse(
    values_json: &str,
    epsilon: f64,
    lambda: f64,
    gamma: f64,
) -> Result<JsValue, JsValue> {
    let value_sets: Vec<Vec<f64>> = serde_json::from_str(values_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

    let cluster_config = ClusteringConfig {
        epsilon,
        min_cluster_size: 1,
    };
    let cic_config = CICConfig { lambda, gamma };

    let results: Vec<FusionResult> = value_sets
        .iter()
        .map(|values| fuse_signals(values, &cluster_config, &cic_config))
        .collect();

    serde_wasm_bindgen::to_value(&results)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ============================================================
// Graph Laplacian Anomaly Detection WASM Interface
// ============================================================

use crate::graph_laplacian::{
    GraphAnomalyConfig, GraphLatentDetector, AnomalyResult, SpectralAnomalyResult,
    construct_similarity_matrix, graph_laplacian, normalized_laplacian,
    spectral_anomaly_detection, detect_coherence_amplification, embedding_dispersion,
};

/// Create graph Laplacian anomaly detector (WASM)
#[wasm_bindgen]
pub struct WasmGraphLaplacian {
    detector: GraphLatentDetector,
    config: GraphAnomalyConfig,
}

#[wasm_bindgen]
impl WasmGraphLaplacian {
    /// Create new detector with parameters
    #[wasm_bindgen(constructor)]
    pub fn new(
        latent_dim: usize,
        lambda_reg: f64,
        learning_rate: f64,
        max_iter: usize,
    ) -> WasmGraphLaplacian {
        let config = GraphAnomalyConfig {
            latent_dim,
            lambda_reg,
            learning_rate,
            max_iter,
            tol: 1e-6,
            normal_similarity: 1.0,
            anomaly_similarity: -1.0,
            unlabeled_similarity: 0.0,
        };
        WasmGraphLaplacian {
            detector: GraphLatentDetector::new(config.clone()),
            config,
        }
    }

    /// Create with default configuration
    #[wasm_bindgen]
    pub fn with_defaults() -> WasmGraphLaplacian {
        let config = GraphAnomalyConfig::default();
        WasmGraphLaplacian {
            detector: GraphLatentDetector::new(config.clone()),
            config,
        }
    }

    /// Fit detector to data with labeled samples
    /// x_json: JSON 2D array of features
    /// labeled_normal: indices of known normal samples
    /// labeled_anomaly: indices of known anomalous samples
    #[wasm_bindgen]
    pub fn fit(
        &mut self,
        x_json: &str,
        labeled_normal: &[usize],
        labeled_anomaly: &[usize],
    ) -> Result<(), JsValue> {
        use ndarray::Array2;

        let data: Vec<Vec<f64>> = serde_json::from_str(x_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

        if data.is_empty() {
            return Err(JsValue::from_str("Empty data"));
        }

        let n = data.len();
        let d = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();

        let x = Array2::from_shape_vec((n, d), flat)
            .map_err(|e| JsValue::from_str(&format!("Invalid shape: {}", e)))?;

        self.detector.fit(&x, labeled_normal, labeled_anomaly);
        Ok(())
    }

    /// Score samples for anomaly
    #[wasm_bindgen]
    pub fn score(&self, x_json: &str) -> Result<JsValue, JsValue> {
        use ndarray::Array2;

        let data: Vec<Vec<f64>> = serde_json::from_str(x_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

        if data.is_empty() {
            return Ok(JsValue::NULL);
        }

        let n = data.len();
        let d = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();

        let x = Array2::from_shape_vec((n, d), flat)
            .map_err(|e| JsValue::from_str(&format!("Invalid shape: {}", e)))?;

        let scores = self.detector.score(&x);
        serde_wasm_bindgen::to_value(&scores)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get latent embedding
    #[wasm_bindgen]
    pub fn get_embedding(&self) -> JsValue {
        match self.detector.embedding() {
            Some(emb) => serde_wasm_bindgen::to_value(&emb).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
}

/// Spectral anomaly detection (WASM)
#[wasm_bindgen]
pub fn wasm_spectral_anomaly(
    similarity_json: &str,
    n_components: usize,
) -> Result<JsValue, JsValue> {
    use ndarray::Array2;

    let data: Vec<Vec<f64>> = serde_json::from_str(similarity_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

    let n = data.len();
    let flat: Vec<f64> = data.into_iter().flatten().collect();

    let similarity = Array2::from_shape_vec((n, n), flat)
        .map_err(|e| JsValue::from_str(&format!("Invalid shape: {}", e)))?;

    let result = spectral_anomaly_detection(&similarity, n_components);
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Detect coherence amplification (WASM)
#[wasm_bindgen]
pub fn wasm_coherence_amplification(dispersions: &[f64]) -> Vec<f64> {
    detect_coherence_amplification(dispersions)
}

/// Compute embedding dispersion (WASM)
#[wasm_bindgen]
pub fn wasm_embedding_dispersion(embedding: &[f64], n_samples: usize, n_dims: usize) -> f64 {
    embedding_dispersion(embedding, n_samples, n_dims)
}

// ============================================================
// Dempster-Shafer Belief Fusion WASM Interface
// ============================================================

use crate::dempster_shafer::{
    ReliabilityConfig as DSReliabilityConfig, FusionMethod, FusedBelief,
    logistic_reliability, additive_fusion, multiplicative_fusion, auto_fuse,
};

/// Compute reliability from quality metric (WASM)
#[wasm_bindgen]
pub fn wasm_logistic_reliability(
    quality: f64,
    alpha: f64,
    beta: f64,
    min_reliability: f64,
    max_reliability: f64,
) -> f64 {
    let config = DSReliabilityConfig {
        alpha,
        beta,
        min_reliability,
        max_reliability,
    };
    logistic_reliability(quality, &config)
}

/// Additive belief fusion (WASM)
///
/// source_probs_json: JSON array of arrays [[p1,p2,p3], [p1,p2,p3], ...]
/// reliabilities: reliability per source
/// hypothesis_names_json: JSON array of hypothesis names
#[wasm_bindgen]
pub fn wasm_additive_fusion(
    source_probs_json: &str,
    reliabilities: &[f64],
    hypothesis_names_json: &str,
) -> Result<JsValue, JsValue> {
    let source_probs: Vec<Vec<f64>> = serde_json::from_str(source_probs_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid source_probs JSON: {}", e)))?;
    let names: Vec<String> = serde_json::from_str(hypothesis_names_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid names JSON: {}", e)))?;

    let result = additive_fusion(&source_probs, reliabilities, names);
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Multiplicative belief fusion (WASM)
#[wasm_bindgen]
pub fn wasm_multiplicative_fusion(
    source_probs_json: &str,
    reliabilities: &[f64],
    hypothesis_names_json: &str,
) -> Result<JsValue, JsValue> {
    let source_probs: Vec<Vec<f64>> = serde_json::from_str(source_probs_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid source_probs JSON: {}", e)))?;
    let names: Vec<String> = serde_json::from_str(hypothesis_names_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid names JSON: {}", e)))?;

    let result = multiplicative_fusion(&source_probs, reliabilities, names);
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Auto-select fusion method based on reliability (WASM)
#[wasm_bindgen]
pub fn wasm_auto_fuse(
    source_probs_json: &str,
    qualities: &[f64],
    hypothesis_names_json: &str,
    alpha: f64,
    beta: f64,
) -> Result<JsValue, JsValue> {
    let source_probs: Vec<Vec<f64>> = serde_json::from_str(source_probs_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid source_probs JSON: {}", e)))?;
    let names: Vec<String> = serde_json::from_str(hypothesis_names_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid names JSON: {}", e)))?;

    let config = DSReliabilityConfig {
        alpha,
        beta,
        min_reliability: 0.01,
        max_reliability: 0.99,
    };

    let result = auto_fuse(&source_probs, qualities, names, &config);
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ============================================================
// Transfer Entropy Causal Graphs WASM Interface
// ============================================================

use crate::transfer_entropy::{
    TransferEntropyConfig, CausalGraph, StructureShift,
    transfer_entropy, build_causal_graph, detect_structure_shifts, intentionality_gradient,
};

/// Compute transfer entropy from source to target (WASM)
///
/// source, target: time series arrays
/// k_neighbors: k for k-NN estimator
/// lag_x: history length for source
/// lag_y: history length for target
/// normalize: whether to normalize by target entropy
#[wasm_bindgen]
pub fn wasm_transfer_entropy(
    source: &[f64],
    target: &[f64],
    k_neighbors: usize,
    lag_x: usize,
    lag_y: usize,
    normalize: bool,
) -> f64 {
    let config = TransferEntropyConfig {
        k_neighbors,
        lag_x,
        lag_y,
        threshold: 0.0,
        normalize,
    };
    transfer_entropy(source, target, &config)
}

/// Build causal graph from multiple time series (WASM)
///
/// signals_json: JSON array of arrays [[series1], [series2], ...]
/// names_json: JSON array of names ["name1", "name2", ...]
/// k_neighbors, lag_x, lag_y: embedding parameters
/// threshold: minimum TE for edge
#[wasm_bindgen]
pub fn wasm_build_causal_graph(
    signals_json: &str,
    names_json: &str,
    k_neighbors: usize,
    lag_x: usize,
    lag_y: usize,
    threshold: f64,
) -> Result<JsValue, JsValue> {
    let signals: Vec<Vec<f64>> = serde_json::from_str(signals_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid signals JSON: {}", e)))?;
    let names: Vec<String> = serde_json::from_str(names_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid names JSON: {}", e)))?;

    let config = TransferEntropyConfig {
        k_neighbors,
        lag_x,
        lag_y,
        threshold,
        normalize: true,
    };

    let graph = build_causal_graph(&signals, names, &config);
    serde_wasm_bindgen::to_value(&graph)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// WASM wrapper for causal graph analysis
#[wasm_bindgen]
pub struct WasmCausalGraph {
    graph: CausalGraph,
}

#[wasm_bindgen]
impl WasmCausalGraph {
    /// Build from time series data
    #[wasm_bindgen(constructor)]
    pub fn new(
        signals_json: &str,
        names_json: &str,
        k_neighbors: usize,
        lag: usize,
        threshold: f64,
    ) -> Result<WasmCausalGraph, JsValue> {
        let signals: Vec<Vec<f64>> = serde_json::from_str(signals_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid signals JSON: {}", e)))?;
        let names: Vec<String> = serde_json::from_str(names_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid names JSON: {}", e)))?;

        let config = TransferEntropyConfig {
            k_neighbors,
            lag_x: lag,
            lag_y: lag,
            threshold,
            normalize: true,
        };

        let graph = build_causal_graph(&signals, names, &config);
        Ok(WasmCausalGraph { graph })
    }

    /// Get adjacency matrix as flat array
    #[wasm_bindgen]
    pub fn get_adjacency(&self) -> Vec<f64> {
        self.graph.adjacency.clone()
    }

    /// Get node names
    #[wasm_bindgen]
    pub fn get_nodes(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.graph.nodes).unwrap_or(JsValue::NULL)
    }

    /// Get edge weight from source to target
    #[wasm_bindgen]
    pub fn get_edge(&self, source: usize, target: usize) -> f64 {
        self.graph.get_edge(source, target)
    }

    /// Get out-degree (total influence of a node)
    #[wasm_bindgen]
    pub fn out_degree(&self, node: usize) -> f64 {
        self.graph.out_degree(node)
    }

    /// Get in-degree (total influence on a node)
    #[wasm_bindgen]
    pub fn in_degree(&self, node: usize) -> f64 {
        self.graph.in_degree(node)
    }

    /// Get drivers (nodes with highest out-degree)
    #[wasm_bindgen]
    pub fn get_drivers(&self, top_n: usize) -> JsValue {
        let mut degrees: Vec<(usize, f64)> = (0..self.graph.n_nodes)
            .map(|i| (i, self.graph.out_degree(i)))
            .collect();
        degrees.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let result: Vec<(&String, f64)> = degrees
            .iter()
            .take(top_n)
            .map(|(i, d)| (&self.graph.nodes[*i], *d))
            .collect();

        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Get responders (nodes with highest in-degree)
    #[wasm_bindgen]
    pub fn get_responders(&self, top_n: usize) -> JsValue {
        let mut degrees: Vec<(usize, f64)> = (0..self.graph.n_nodes)
            .map(|i| (i, self.graph.in_degree(i)))
            .collect();
        degrees.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let result: Vec<(&String, f64)> = degrees
            .iter()
            .take(top_n)
            .map(|(i, d)| (&self.graph.nodes[*i], *d))
            .collect();

        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Get normalized Laplacian
    #[wasm_bindgen]
    pub fn get_laplacian(&self) -> Vec<f64> {
        self.graph.normalized_laplacian()
    }

    /// Get node count
    #[wasm_bindgen]
    pub fn node_count(&self) -> usize {
        self.graph.n_nodes
    }

    /// Serialize to JSON
    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.graph)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Detect causal structure shifts over time (WASM)
#[wasm_bindgen]
pub fn wasm_structure_shifts(
    signals_json: &str,
    names_json: &str,
    window_size: usize,
    step_size: usize,
    k_neighbors: usize,
    lag: usize,
    threshold: f64,
) -> Result<JsValue, JsValue> {
    let signals: Vec<Vec<f64>> = serde_json::from_str(signals_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid signals JSON: {}", e)))?;
    let names: Vec<String> = serde_json::from_str(names_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid names JSON: {}", e)))?;

    let config = TransferEntropyConfig {
        k_neighbors,
        lag_x: lag,
        lag_y: lag,
        threshold,
        normalize: true,
    };

    let shifts = detect_structure_shifts(&signals, names, window_size, step_size, &config);
    serde_wasm_bindgen::to_value(&shifts)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute intentionality gradient for a target (WASM)
#[wasm_bindgen]
pub fn wasm_intentionality_gradient(
    signals_json: &str,
    target_idx: usize,
    window_size: usize,
    k_neighbors: usize,
    lag: usize,
) -> Result<JsValue, JsValue> {
    let signals: Vec<Vec<f64>> = serde_json::from_str(signals_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

    let config = TransferEntropyConfig {
        k_neighbors,
        lag_x: lag,
        lag_y: lag,
        threshold: 0.0,
        normalize: true,
    };

    let gradient = intentionality_gradient(&signals, target_idx, window_size, &config);
    serde_wasm_bindgen::to_value(&gradient)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ============================================================
// Lyapunov Exponent Regime Stability WASM Interface
// ============================================================

use crate::lyapunov::{
    LyapunovConfig, LyapunovResult, AttractorType, NationStability,
    rosenstein_lyapunov, multivariate_lyapunov, delay_embed, batch_stability_analysis,
};

/// Compute Lyapunov exponent for univariate time series (WASM)
///
/// Rosenstein algorithm for largest Lyapunov exponent.
/// Negative = stable, Near zero = periodic, Positive = chaotic
#[wasm_bindgen]
pub fn wasm_lyapunov_univariate(
    series: &[f64],
    embedding_dim: usize,
    tau: usize,
    min_separation: usize,
    k_neighbors: usize,
    max_iterations: usize,
) -> Result<JsValue, JsValue> {
    let config = LyapunovConfig {
        embedding_dim,
        tau,
        min_separation,
        k_neighbors,
        max_iterations,
        epsilon: 1e-8,
    };

    let result = rosenstein_lyapunov(series, &config);
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute Lyapunov exponent for multivariate state series (WASM)
///
/// states_json: JSON 2D array [[economic, political, social, military, external], ...]
#[wasm_bindgen]
pub fn wasm_lyapunov_multivariate(
    states_json: &str,
    tau: usize,
    min_separation: usize,
    k_neighbors: usize,
) -> Result<JsValue, JsValue> {
    use ndarray::Array2;

    let data: Vec<Vec<f64>> = serde_json::from_str(states_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

    if data.is_empty() {
        return Err(JsValue::from_str("Empty data"));
    }

    let n = data.len();
    let d = data[0].len();
    let flat: Vec<f64> = data.into_iter().flatten().collect();

    let states = Array2::from_shape_vec((n, d), flat)
        .map_err(|e| JsValue::from_str(&format!("Invalid shape: {}", e)))?;

    let config = LyapunovConfig {
        embedding_dim: d,
        tau,
        min_separation,
        k_neighbors,
        max_iterations: 100,
        epsilon: 1e-8,
    };

    let result = multivariate_lyapunov(&states, &config);
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Batch stability analysis for multiple nations (WASM)
///
/// nations_json: JSON object { "USA": [[...], [...]], "CHN": [[...], [...]], ... }
#[wasm_bindgen]
pub fn wasm_batch_stability(
    nations_json: &str,
    tau: usize,
    min_separation: usize,
    k_neighbors: usize,
) -> Result<JsValue, JsValue> {
    use ndarray::Array2;
    use std::collections::HashMap;

    let nations_map: HashMap<String, Vec<Vec<f64>>> = serde_json::from_str(nations_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

    let config = LyapunovConfig {
        embedding_dim: 5,
        tau,
        min_separation,
        k_neighbors,
        max_iterations: 100,
        epsilon: 1e-8,
    };

    let mut results: Vec<NationStability> = Vec::new();

    for (code, data) in nations_map {
        if data.is_empty() || data[0].is_empty() {
            continue;
        }

        let n = data.len();
        let d = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();

        if let Ok(states) = Array2::from_shape_vec((n, d), flat) {
            let result = multivariate_lyapunov(&states, &config);
            results.push(NationStability { code, result });
        }
    }

    serde_wasm_bindgen::to_value(&results)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get basin strength from Lyapunov exponent (WASM)
///
/// Simple helper to convert λ to stability score [0,1]
#[wasm_bindgen]
pub fn wasm_basin_strength(lyapunov_exponent: f64) -> f64 {
    if lyapunov_exponent > 0.0 {
        (-lyapunov_exponent).exp().min(0.5)
    } else if lyapunov_exponent < -0.1 {
        (1.0 - (-lyapunov_exponent / 2.0).min(1.0)).max(0.7)
    } else {
        0.5 // Critical zone
    }
}

/// Classify attractor type from Lyapunov exponent (WASM)
#[wasm_bindgen]
pub fn wasm_classify_attractor(lyapunov_exponent: f64) -> String {
    if lyapunov_exponent < -0.1 {
        "Stable".to_string()
    } else if lyapunov_exponent > 0.1 {
        "Chaotic".to_string()
    } else if lyapunov_exponent.abs() < 0.02 {
        "Periodic".to_string()
    } else {
        "Critical".to_string()
    }
}

// ============================================================
// Cascade Probability via Belief Propagation WASM Interface
// ============================================================

use crate::cascade::{
    CascadeConfig, CascadeChannel, RegionCascade, CascadePrediction,
    CascadeSimulation, RegionCascadeStats,
    predict_cascade, quick_cascade_prediction, simulate_cascades,
};

/// Quick cascade prediction with simplified coupling (WASM)
///
/// neighbors_json: JSON array of [region_name, coupling_strength] pairs
#[wasm_bindgen]
pub fn wasm_quick_cascade(
    origin_event: &str,
    origin_region: &str,
    neighbors_json: &str,
) -> Result<JsValue, JsValue> {
    let neighbors: Vec<(String, f64)> = serde_json::from_str(neighbors_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid JSON: {}", e)))?;

    let neighbor_refs: Vec<(&str, f64)> = neighbors
        .iter()
        .map(|(s, f)| (s.as_str(), *f))
        .collect();

    let result = quick_cascade_prediction(origin_event, origin_region, &neighbor_refs);
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Full cascade prediction with explicit coupling matrices (WASM)
///
/// regions_json: JSON array of region names
/// geo_json: JSON 2D array of geographic distances
/// econ_json: JSON 2D array of economic ties [0,1]
/// ethnic_json: JSON 2D array of ethnic ties [0,1]
/// political_json: JSON 2D array of political alignment [-1,1]
#[wasm_bindgen]
pub fn wasm_predict_cascade(
    origin_event: &str,
    origin_region: &str,
    regions_json: &str,
    geo_json: &str,
    econ_json: &str,
    ethnic_json: &str,
    political_json: &str,
    max_iterations: usize,
    damping: f64,
) -> Result<JsValue, JsValue> {
    use ndarray::Array2;

    let regions: Vec<String> = serde_json::from_str(regions_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid regions JSON: {}", e)))?;

    let n = regions.len();

    let parse_matrix = |json: &str, name: &str| -> Result<Array2<f64>, JsValue> {
        let data: Vec<Vec<f64>> = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("Invalid {} JSON: {}", name, e)))?;
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        Array2::from_shape_vec((n, n), flat)
            .map_err(|e| JsValue::from_str(&format!("Invalid {} shape: {}", name, e)))
    };

    let geo = parse_matrix(geo_json, "geo")?;
    let econ = parse_matrix(econ_json, "econ")?;
    let ethnic = parse_matrix(ethnic_json, "ethnic")?;
    let political = parse_matrix(political_json, "political")?;

    let config = CascadeConfig {
        max_iterations,
        damping,
        convergence_threshold: 1e-6,
        base_prob: 0.1,
        geo_decay: 0.001,
        econ_weight: 0.3,
        ethnic_weight: 0.2,
        political_weight: 0.25,
    };

    let result = predict_cascade(
        origin_event,
        origin_region,
        regions,
        &geo,
        &econ,
        &ethnic,
        &political,
        config,
    );

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Monte Carlo cascade simulation (WASM)
///
/// Runs multiple simulations with perturbed coupling matrices
/// for uncertainty quantification
#[wasm_bindgen]
pub fn wasm_simulate_cascades(
    origin_event: &str,
    origin_region: &str,
    regions_json: &str,
    geo_json: &str,
    econ_json: &str,
    ethnic_json: &str,
    political_json: &str,
    n_simulations: usize,
    perturbation_scale: f64,
) -> Result<JsValue, JsValue> {
    use ndarray::Array2;

    let regions: Vec<String> = serde_json::from_str(regions_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid regions JSON: {}", e)))?;

    let n = regions.len();

    let parse_matrix = |json: &str, name: &str| -> Result<Array2<f64>, JsValue> {
        let data: Vec<Vec<f64>> = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("Invalid {} JSON: {}", name, e)))?;
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        Array2::from_shape_vec((n, n), flat)
            .map_err(|e| JsValue::from_str(&format!("Invalid {} shape: {}", name, e)))
    };

    let geo = parse_matrix(geo_json, "geo")?;
    let econ = parse_matrix(econ_json, "econ")?;
    let ethnic = parse_matrix(ethnic_json, "ethnic")?;
    let political = parse_matrix(political_json, "political")?;

    let result = simulate_cascades(
        origin_event,
        origin_region,
        regions,
        &geo,
        &econ,
        &ethnic,
        &political,
        n_simulations,
        perturbation_scale,
    );

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get cascade channel name as string (WASM helper)
#[wasm_bindgen]
pub fn wasm_cascade_channel_name(channel: u8) -> String {
    match channel {
        0 => "Geographic".to_string(),
        1 => "Economic".to_string(),
        2 => "Ethnic".to_string(),
        3 => "Political".to_string(),
        _ => "Unknown".to_string(),
    }
}
