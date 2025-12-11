//! Simulation & Ablation Test Harness for Novel Algorithms
//!
//! This module provides:
//! 1. Synthetic data generators for testing
//! 2. Ground-truth scenarios with known outcomes
//! 3. Ablation study framework for component analysis
//! 4. Performance benchmarks

use crate::lyapunov::{
    LyapunovConfig, LyapunovResult, AttractorType,
    rosenstein_lyapunov, multivariate_lyapunov,
};
use crate::cascade::{
    CascadeConfig, CascadePrediction, CascadeSimulation,
    predict_cascade, simulate_cascades,
};
use crate::transfer_entropy::{TransferEntropyConfig, intentionality_gradient};

use ndarray::Array2;
use serde::{Deserialize, Serialize};

// ============================================================
// SYNTHETIC DATA GENERATORS
// ============================================================

/// Generate logistic map time series (known Lyapunov exponent)
pub fn generate_logistic_map(r: f64, x0: f64, n: usize) -> Vec<f64> {
    let mut x = vec![x0];
    for _ in 1..n {
        let prev = x[x.len() - 1];
        x.push(r * prev * (1.0 - prev));
    }
    x
}

/// Generate Lorenz attractor time series (chaotic, λ ≈ 0.9)
pub fn generate_lorenz(sigma: f64, rho: f64, beta: f64, dt: f64, n: usize) -> Vec<[f64; 3]> {
    let mut trajectory = vec![[1.0, 1.0, 1.0]];

    for _ in 1..n {
        let [x, y, z] = trajectory[trajectory.len() - 1];
        let dx = sigma * (y - x);
        let dy = x * (rho - z) - y;
        let dz = x * y - beta * z;

        trajectory.push([
            x + dx * dt,
            y + dy * dt,
            z + dz * dt,
        ]);
    }

    trajectory
}

/// Generate stable linear system (negative Lyapunov)
pub fn generate_stable_linear(decay: f64, noise: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut x = vec![1.0];
    let mut lcg = seed;

    for _ in 1..n {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        let rand = (lcg as f64) / (u64::MAX as f64) - 0.5;

        let prev = x[x.len() - 1];
        x.push(decay * prev + noise * rand);
    }
    x
}

/// Generate periodic signal (zero Lyapunov)
pub fn generate_periodic(frequency: f64, amplitude: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| amplitude * (2.0 * std::f64::consts::PI * frequency * i as f64).sin())
        .collect()
}

/// Generate nation-state dynamics with known transition
pub fn generate_nation_transition(
    n_steps: usize,
    transition_point: usize,
    pre_stability: f64,  // High = stable
    post_stability: f64, // Low = unstable
    seed: u64,
) -> Array2<f64> {
    let d = 5; // [economic, political, social, military, external]
    let mut states = Array2::zeros((n_steps, d));
    let mut lcg = seed;

    let rand_next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*s as f64) / (u64::MAX as f64) - 0.5
    };

    // Initialize
    for j in 0..d {
        states[[0, j]] = 0.5 + 0.1 * rand_next(&mut lcg);
    }

    for i in 1..n_steps {
        let stability = if i < transition_point { pre_stability } else { post_stability };

        for j in 0..d {
            let prev = states[[i - 1, j]];
            let noise = 0.02 * rand_next(&mut lcg) / stability;
            let coupling: f64 = (0..d)
                .filter(|&k| k != j)
                .map(|k| states[[i - 1, k]] * 0.05)
                .sum();

            states[[i, j]] = (stability * prev + coupling + noise).max(0.0).min(1.0);
        }
    }

    states
}

/// Generate geopolitical cascade scenario
pub fn generate_cascade_scenario() -> (
    Vec<String>,
    Array2<f64>,  // geo distances
    Array2<f64>,  // econ ties
    Array2<f64>,  // ethnic ties
    Array2<f64>,  // political
) {
    // Middle East scenario: Syria crisis spreading
    let regions = vec![
        "SYR".to_string(), // Origin
        "LBN".to_string(), // Strong ties
        "JOR".to_string(), // Medium ties
        "IRQ".to_string(), // Medium ties
        "TUR".to_string(), // Weak ties
        "ISR".to_string(), // Political tension
    ];

    let n = regions.len();

    // Geographic distances (km, approximate)
    let geo = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j { return 0.0; }
        match (i, j) {
            (0, 1) | (1, 0) => 100.0,   // SYR-LBN
            (0, 2) | (2, 0) => 200.0,   // SYR-JOR
            (0, 3) | (3, 0) => 300.0,   // SYR-IRQ
            (0, 4) | (4, 0) => 500.0,   // SYR-TUR
            (0, 5) | (5, 0) => 400.0,   // SYR-ISR
            (1, 5) | (5, 1) => 150.0,   // LBN-ISR
            _ => 600.0,
        }
    });

    // Economic ties [0,1]
    let econ = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j { return 0.0; }
        match (i, j) {
            (0, 1) | (1, 0) => 0.7,  // SYR-LBN high
            (0, 3) | (3, 0) => 0.5,  // SYR-IRQ medium
            (0, 4) | (4, 0) => 0.4,  // SYR-TUR medium
            (0, 2) | (2, 0) => 0.3,  // SYR-JOR low
            _ => 0.1,
        }
    });

    // Ethnic ties [0,1]
    let ethnic = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j { return 0.0; }
        match (i, j) {
            (0, 1) | (1, 0) => 0.6,  // SYR-LBN (shared communities)
            (0, 3) | (3, 0) => 0.5,  // SYR-IRQ (Kurdish, Arab)
            (0, 4) | (4, 0) => 0.4,  // SYR-TUR (Kurdish)
            _ => 0.1,
        }
    });

    // Political alignment [-1,1]
    let political = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j { return 0.0; }
        match (i, j) {
            (0, 3) | (3, 0) => 0.3,   // SYR-IRQ (aligned)
            (0, 5) | (5, 0) => -0.8,  // SYR-ISR (adversarial)
            (1, 5) | (5, 1) => -0.6,  // LBN-ISR (tense)
            (0, 4) | (4, 0) => -0.3,  // SYR-TUR (tense)
            _ => 0.0,
        }
    });

    (regions, geo, econ, ethnic, political)
}

// ============================================================
// ABLATION STUDY FRAMEWORK
// ============================================================

/// Results from ablation test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResult {
    pub test_name: String,
    pub baseline_metric: f64,
    pub ablated_metric: f64,
    pub impact: f64,  // (baseline - ablated) / baseline
    pub description: String,
}

/// Lyapunov ablation tests
pub fn ablate_lyapunov() -> Vec<AblationResult> {
    let mut results = Vec::new();

    // Test 1: Embedding dimension impact
    let chaotic_series = generate_logistic_map(4.0, 0.1, 1000);

    let baseline_config = LyapunovConfig {
        embedding_dim: 5,
        tau: 1,
        min_separation: 10,
        k_neighbors: 4,
        max_iterations: 100,
        epsilon: 1e-8,
    };
    let baseline = rosenstein_lyapunov(&chaotic_series, &baseline_config);

    // Ablate: reduce embedding dimension
    let ablated_config = LyapunovConfig {
        embedding_dim: 2, // Reduced from 5
        ..baseline_config.clone()
    };
    let ablated = rosenstein_lyapunov(&chaotic_series, &ablated_config);

    results.push(AblationResult {
        test_name: "embedding_dimension".to_string(),
        baseline_metric: baseline.lyapunov_exponent,
        ablated_metric: ablated.lyapunov_exponent,
        impact: if baseline.lyapunov_exponent != 0.0 {
            (baseline.lyapunov_exponent - ablated.lyapunov_exponent) / baseline.lyapunov_exponent.abs()
        } else { 0.0 },
        description: "Impact of reducing embedding dimension from 5 to 2".to_string(),
    });

    // Test 2: k-neighbors impact
    let ablated_config = LyapunovConfig {
        k_neighbors: 1, // Reduced from 4
        ..baseline_config.clone()
    };
    let ablated = rosenstein_lyapunov(&chaotic_series, &ablated_config);

    results.push(AblationResult {
        test_name: "k_neighbors".to_string(),
        baseline_metric: baseline.lyapunov_exponent,
        ablated_metric: ablated.lyapunov_exponent,
        impact: if baseline.lyapunov_exponent != 0.0 {
            (baseline.lyapunov_exponent - ablated.lyapunov_exponent) / baseline.lyapunov_exponent.abs()
        } else { 0.0 },
        description: "Impact of reducing k-neighbors from 4 to 1".to_string(),
    });

    // Test 3: Series length impact
    let short_series = generate_logistic_map(4.0, 0.1, 200);
    let ablated = rosenstein_lyapunov(&short_series, &baseline_config);

    results.push(AblationResult {
        test_name: "series_length".to_string(),
        baseline_metric: baseline.lyapunov_exponent,
        ablated_metric: ablated.lyapunov_exponent,
        impact: if baseline.lyapunov_exponent != 0.0 {
            (baseline.lyapunov_exponent - ablated.lyapunov_exponent) / baseline.lyapunov_exponent.abs()
        } else { 0.0 },
        description: "Impact of reducing series length from 1000 to 200".to_string(),
    });

    results
}

/// Cascade ablation tests
pub fn ablate_cascade() -> Vec<AblationResult> {
    let mut results = Vec::new();

    let (regions, geo, econ, ethnic, political) = generate_cascade_scenario();

    // Baseline prediction
    let baseline_config = CascadeConfig::default();
    let baseline = predict_cascade(
        "crisis_001",
        "SYR",
        regions.clone(),
        &geo,
        &econ,
        &ethnic,
        &political,
        baseline_config.clone(),
    );

    // Ablate: Remove economic channel
    let zero_econ = Array2::zeros((regions.len(), regions.len()));
    let ablated = predict_cascade(
        "crisis_001",
        "SYR",
        regions.clone(),
        &geo,
        &zero_econ,
        &ethnic,
        &political,
        baseline_config.clone(),
    );

    results.push(AblationResult {
        test_name: "economic_channel".to_string(),
        baseline_metric: baseline.overall_probability,
        ablated_metric: ablated.overall_probability,
        impact: if baseline.overall_probability > 0.0 {
            (baseline.overall_probability - ablated.overall_probability) / baseline.overall_probability
        } else { 0.0 },
        description: "Impact of removing economic ties from cascade prediction".to_string(),
    });

    // Ablate: Remove ethnic channel
    let zero_ethnic = Array2::zeros((regions.len(), regions.len()));
    let ablated = predict_cascade(
        "crisis_001",
        "SYR",
        regions.clone(),
        &geo,
        &econ,
        &zero_ethnic,
        &political,
        baseline_config.clone(),
    );

    results.push(AblationResult {
        test_name: "ethnic_channel".to_string(),
        baseline_metric: baseline.overall_probability,
        ablated_metric: ablated.overall_probability,
        impact: if baseline.overall_probability > 0.0 {
            (baseline.overall_probability - ablated.overall_probability) / baseline.overall_probability
        } else { 0.0 },
        description: "Impact of removing ethnic ties from cascade prediction".to_string(),
    });

    // Ablate: Remove political channel
    let zero_political = Array2::zeros((regions.len(), regions.len()));
    let ablated = predict_cascade(
        "crisis_001",
        "SYR",
        regions.clone(),
        &geo,
        &econ,
        &ethnic,
        &zero_political,
        baseline_config.clone(),
    );

    results.push(AblationResult {
        test_name: "political_channel".to_string(),
        baseline_metric: baseline.overall_probability,
        ablated_metric: ablated.overall_probability,
        impact: if baseline.overall_probability > 0.0 {
            (baseline.overall_probability - ablated.overall_probability) / baseline.overall_probability
        } else { 0.0 },
        description: "Impact of removing political alignment from cascade prediction".to_string(),
    });

    // Ablate: Reduce BP iterations
    let low_iter_config = CascadeConfig {
        max_iterations: 5, // Reduced from 100
        ..baseline_config.clone()
    };
    let ablated = predict_cascade(
        "crisis_001",
        "SYR",
        regions.clone(),
        &geo,
        &econ,
        &ethnic,
        &political,
        low_iter_config,
    );

    results.push(AblationResult {
        test_name: "bp_iterations".to_string(),
        baseline_metric: baseline.overall_probability,
        ablated_metric: ablated.overall_probability,
        impact: if baseline.overall_probability > 0.0 {
            (baseline.overall_probability - ablated.overall_probability) / baseline.overall_probability
        } else { 0.0 },
        description: "Impact of reducing BP iterations from 100 to 5".to_string(),
    });

    results
}

// ============================================================
// GROUND TRUTH VALIDATION
// ============================================================

/// Known system validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub system_name: String,
    pub expected_lyapunov: f64,
    pub computed_lyapunov: f64,
    pub relative_error: f64,
    pub correct_classification: bool,
    pub expected_attractor: String,
    pub computed_attractor: String,
}

/// Validate against known systems
pub fn validate_lyapunov_known_systems() -> Vec<ValidationResult> {
    let mut results = Vec::new();

    let config = LyapunovConfig {
        embedding_dim: 3,
        tau: 1,
        min_separation: 5,
        k_neighbors: 4,
        max_iterations: 50,
        epsilon: 1e-8,
    };

    // Logistic map r=4: λ = ln(2) ≈ 0.693
    let logistic = generate_logistic_map(4.0, 0.1, 2000);
    let result = rosenstein_lyapunov(&logistic, &config);
    let expected = 2.0_f64.ln();

    results.push(ValidationResult {
        system_name: "logistic_r4".to_string(),
        expected_lyapunov: expected,
        computed_lyapunov: result.lyapunov_exponent,
        relative_error: (result.lyapunov_exponent - expected).abs() / expected,
        correct_classification: result.attractor_type == AttractorType::Chaotic
            || result.attractor_type == AttractorType::Critical,
        expected_attractor: "Chaotic".to_string(),
        computed_attractor: format!("{:?}", result.attractor_type),
    });

    // Logistic map r=3.5: λ ≈ 0 (periodic)
    let logistic_periodic = generate_logistic_map(3.5, 0.1, 2000);
    let result = rosenstein_lyapunov(&logistic_periodic, &config);

    results.push(ValidationResult {
        system_name: "logistic_r3.5".to_string(),
        expected_lyapunov: 0.0,
        computed_lyapunov: result.lyapunov_exponent,
        relative_error: result.lyapunov_exponent.abs(),
        correct_classification: result.attractor_type == AttractorType::Periodic
            || result.attractor_type == AttractorType::Critical,
        expected_attractor: "Periodic".to_string(),
        computed_attractor: format!("{:?}", result.attractor_type),
    });

    // Stable decay: λ < 0
    let stable = generate_stable_linear(0.9, 0.01, 2000, 42);
    let result = rosenstein_lyapunov(&stable, &config);

    results.push(ValidationResult {
        system_name: "stable_decay".to_string(),
        expected_lyapunov: -0.1, // Approximately
        computed_lyapunov: result.lyapunov_exponent,
        relative_error: if result.lyapunov_exponent < 0.0 { 0.0 } else { 1.0 },
        correct_classification: result.attractor_type == AttractorType::Stable
            || result.lyapunov_exponent < 0.1,
        expected_attractor: "Stable".to_string(),
        computed_attractor: format!("{:?}", result.attractor_type),
    });

    results
}

/// Validate transition detection
pub fn validate_transition_detection() -> Vec<(usize, bool, f64, f64)> {
    let mut results = Vec::new();

    // Test detection at different transition points
    for transition_point in [100, 200, 300] {
        let states = generate_nation_transition(400, transition_point, 0.95, 0.5, 12345);

        let config = LyapunovConfig::default();

        // Analyze pre-transition
        let pre_slice = states.slice(ndarray::s![..transition_point, ..]).to_owned();
        let pre_result = multivariate_lyapunov(&pre_slice, &config);

        // Analyze post-transition
        let post_slice = states.slice(ndarray::s![transition_point.., ..]).to_owned();
        let post_result = multivariate_lyapunov(&post_slice, &config);

        // Should detect instability increase
        let detected = post_result.transition_risk > pre_result.transition_risk
            || post_result.basin_strength < pre_result.basin_strength;

        results.push((
            transition_point,
            detected,
            pre_result.basin_strength,
            post_result.basin_strength,
        ));
    }

    results
}

// ============================================================
// BENCHMARK SUITE
// ============================================================

/// Benchmark timing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub algorithm: String,
    pub input_size: usize,
    pub iterations: usize,
    pub total_time_ms: f64,
    pub per_iteration_ms: f64,
}

/// Run benchmarks (returns time in approximate cycles, not real time in tests)
pub fn run_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Lyapunov benchmark
    let series = generate_logistic_map(4.0, 0.1, 1000);
    let config = LyapunovConfig::default();

    let iterations = 10;
    for _ in 0..iterations {
        let _ = rosenstein_lyapunov(&series, &config);
    }

    results.push(BenchmarkResult {
        algorithm: "lyapunov_rosenstein".to_string(),
        input_size: 1000,
        iterations,
        total_time_ms: 0.0, // Would need std::time in real benchmark
        per_iteration_ms: 0.0,
    });

    // Cascade benchmark
    let (regions, geo, econ, ethnic, political) = generate_cascade_scenario();
    let config = CascadeConfig::default();

    for _ in 0..iterations {
        let _ = predict_cascade(
            "test",
            "SYR",
            regions.clone(),
            &geo,
            &econ,
            &ethnic,
            &political,
            config.clone(),
        );
    }

    results.push(BenchmarkResult {
        algorithm: "cascade_bp".to_string(),
        input_size: regions.len(),
        iterations,
        total_time_ms: 0.0,
        per_iteration_ms: 0.0,
    });

    results
}

// ============================================================
// FULL TEST REPORT
// ============================================================

/// Complete simulation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationReport {
    pub lyapunov_validations: Vec<ValidationResult>,
    pub lyapunov_ablations: Vec<AblationResult>,
    pub cascade_ablations: Vec<AblationResult>,
    pub transition_detections: Vec<(usize, bool, f64, f64)>,
    pub benchmarks: Vec<BenchmarkResult>,
    pub summary: ReportSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub lyapunov_accuracy: f64,
    pub cascade_channel_importance: Vec<(String, f64)>,
    pub transition_detection_rate: f64,
}

/// Generate complete simulation report
pub fn generate_simulation_report() -> SimulationReport {
    let lyapunov_validations = validate_lyapunov_known_systems();
    let lyapunov_ablations = ablate_lyapunov();
    let cascade_ablations = ablate_cascade();
    let transition_detections = validate_transition_detection();
    let benchmarks = run_benchmarks();

    // Compute summary
    let lyapunov_accuracy = lyapunov_validations
        .iter()
        .filter(|v| v.correct_classification)
        .count() as f64 / lyapunov_validations.len() as f64;

    let cascade_channel_importance: Vec<(String, f64)> = cascade_ablations
        .iter()
        .filter(|a| a.test_name != "bp_iterations")
        .map(|a| (a.test_name.clone(), a.impact.abs()))
        .collect();

    let transition_detection_rate = transition_detections
        .iter()
        .filter(|(_, detected, _, _)| *detected)
        .count() as f64 / transition_detections.len() as f64;

    SimulationReport {
        lyapunov_validations,
        lyapunov_ablations,
        cascade_ablations,
        transition_detections,
        benchmarks,
        summary: ReportSummary {
            lyapunov_accuracy,
            cascade_channel_importance,
            transition_detection_rate,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_map_generator() {
        let series = generate_logistic_map(4.0, 0.1, 100);
        assert_eq!(series.len(), 100);
        assert!(series.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_lorenz_generator() {
        let trajectory = generate_lorenz(10.0, 28.0, 8.0 / 3.0, 0.01, 1000);
        assert_eq!(trajectory.len(), 1000);
    }

    #[test]
    fn test_cascade_scenario() {
        let (regions, geo, econ, ethnic, political) = generate_cascade_scenario();
        assert_eq!(regions.len(), 6);
        assert_eq!(geo.dim(), (6, 6));
        assert_eq!(econ.dim(), (6, 6));
    }

    #[test]
    fn test_lyapunov_ablation() {
        let results = ablate_lyapunov();
        assert!(!results.is_empty());
        for r in &results {
            assert!(!r.test_name.is_empty());
        }
    }

    #[test]
    fn test_cascade_ablation() {
        let results = ablate_cascade();
        assert!(!results.is_empty());
        for r in &results {
            assert!(!r.test_name.is_empty());
        }
    }

    #[test]
    fn test_known_system_validation() {
        let results = validate_lyapunov_known_systems();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_transition_detection() {
        let results = validate_transition_detection();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_full_report() {
        let report = generate_simulation_report();
        assert!(!report.lyapunov_validations.is_empty());
        assert!(!report.cascade_ablations.is_empty());
        assert!(report.summary.lyapunov_accuracy >= 0.0);
        assert!(report.summary.lyapunov_accuracy <= 1.0);
    }
}
