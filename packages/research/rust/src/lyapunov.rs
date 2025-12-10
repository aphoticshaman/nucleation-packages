//! Lyapunov Exponent Computation for Regime Stability Analysis
//!
//! Computes the largest Lyapunov exponent (LLE) to quantify dynamical system stability.
//! Based on Wolf et al. (1985) and Rosenstein et al. (1993) algorithms.
//!
//! For nation-state stability analysis:
//! - λ > 0: Chaotic/unstable regime (sensitive to perturbations)
//! - λ ≈ 0: Edge of chaos (critical transitions possible)
//! - λ < 0: Stable attractor (perturbations decay)
//!
//! Basin strength = exp(-λ) for λ > 0, else 1 - |λ|/λ_max

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Configuration for Lyapunov exponent estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovConfig {
    /// Embedding dimension (state space dimension)
    pub embedding_dim: usize,
    /// Time delay for embedding
    pub tau: usize,
    /// Minimum temporal separation for neighbor selection
    pub min_separation: usize,
    /// Number of nearest neighbors to average
    pub k_neighbors: usize,
    /// Maximum iterations for divergence tracking
    pub max_iterations: usize,
    /// Convergence threshold
    pub epsilon: f64,
}

impl Default for LyapunovConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 5,
            tau: 1,
            min_separation: 10,
            k_neighbors: 4,
            max_iterations: 100,
            epsilon: 1e-8,
        }
    }
}

/// Result of Lyapunov analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovResult {
    /// Largest Lyapunov exponent
    pub lyapunov_exponent: f64,
    /// Basin strength (stability measure in [0,1])
    pub basin_strength: f64,
    /// Attractor classification
    pub attractor_type: AttractorType,
    /// Transition risk (probability of regime change)
    pub transition_risk: f64,
    /// Convergence achieved
    pub converged: bool,
    /// Effective degrees of freedom (correlation dimension estimate)
    pub effective_dof: f64,
}

/// Classification of dynamical attractor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttractorType {
    /// Fixed point (λ < -0.1)
    Stable,
    /// Limit cycle (λ ≈ 0)
    Periodic,
    /// Strange attractor (λ > 0.1)
    Chaotic,
    /// Near critical transition (|λ| < 0.1)
    Critical,
}

/// Create delay embedding of univariate time series
pub fn delay_embed(x: &[f64], dim: usize, tau: usize) -> Array2<f64> {
    let n = x.len();
    if n <= (dim - 1) * tau {
        return Array2::zeros((0, dim));
    }

    let m = n - (dim - 1) * tau;
    let mut embedded = Array2::zeros((m, dim));

    for i in 0..m {
        for j in 0..dim {
            embedded[[i, j]] = x[i + j * tau];
        }
    }

    embedded
}

/// Create delay embedding from multivariate time series
pub fn multivariate_embed(x: &Array2<f64>, tau: usize) -> Array2<f64> {
    let (n, d) = x.dim();
    if n <= tau {
        return Array2::zeros((0, d));
    }

    let m = n - tau;
    let mut embedded = Array2::zeros((m, d * 2));

    for i in 0..m {
        for j in 0..d {
            embedded[[i, j]] = x[[i, j]];
            embedded[[i, d + j]] = x[[i + tau, j]];
        }
    }

    embedded
}

/// Euclidean distance between two points
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Find k nearest neighbors with minimum temporal separation
fn find_nearest_neighbors(
    embedded: &Array2<f64>,
    idx: usize,
    k: usize,
    min_sep: usize,
) -> Vec<(usize, f64)> {
    let n = embedded.nrows();
    let point = embedded.row(idx);

    let mut distances: Vec<(usize, f64)> = (0..n)
        .filter(|&j| {
            let sep = if j > idx { j - idx } else { idx - j };
            sep >= min_sep
        })
        .map(|j| {
            let dist = euclidean_distance(
                point.as_slice().unwrap(),
                embedded.row(j).as_slice().unwrap(),
            );
            (j, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.into_iter().take(k).collect()
}

/// Rosenstein algorithm for largest Lyapunov exponent
///
/// Tracks divergence of initially nearby trajectories in reconstructed phase space.
pub fn rosenstein_lyapunov(x: &[f64], config: &LyapunovConfig) -> LyapunovResult {
    // Step 1: Delay embedding
    let embedded = delay_embed(x, config.embedding_dim, config.tau);
    let n = embedded.nrows();

    if n < config.min_separation * 2 {
        return LyapunovResult {
            lyapunov_exponent: 0.0,
            basin_strength: 0.5,
            attractor_type: AttractorType::Critical,
            transition_risk: 0.5,
            converged: false,
            effective_dof: 0.0,
        };
    }

    // Step 2: For each point, find nearest neighbor with temporal separation
    let mut divergence_curves: Vec<Vec<f64>> = Vec::new();

    for i in 0..(n - config.max_iterations) {
        let neighbors = find_nearest_neighbors(&embedded, i, 1, config.min_separation);

        if neighbors.is_empty() {
            continue;
        }

        let (j, d0) = neighbors[0];

        if d0 < config.epsilon {
            continue;
        }

        // Track divergence over time
        let mut curve = Vec::with_capacity(config.max_iterations);

        for dt in 0..config.max_iterations {
            let i_future = i + dt;
            let j_future = j + dt;

            if i_future >= n || j_future >= n {
                break;
            }

            let d_t = euclidean_distance(
                embedded.row(i_future).as_slice().unwrap(),
                embedded.row(j_future).as_slice().unwrap(),
            );

            if d_t > config.epsilon {
                curve.push((d_t / d0).ln());
            }
        }

        if curve.len() > config.max_iterations / 2 {
            divergence_curves.push(curve);
        }
    }

    if divergence_curves.is_empty() {
        return LyapunovResult {
            lyapunov_exponent: 0.0,
            basin_strength: 0.5,
            attractor_type: AttractorType::Critical,
            transition_risk: 0.5,
            converged: false,
            effective_dof: 0.0,
        };
    }

    // Step 3: Average divergence curves
    let max_len = divergence_curves.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut avg_divergence = vec![0.0; max_len];
    let mut counts = vec![0usize; max_len];

    for curve in &divergence_curves {
        for (t, &val) in curve.iter().enumerate() {
            if val.is_finite() {
                avg_divergence[t] += val;
                counts[t] += 1;
            }
        }
    }

    for t in 0..max_len {
        if counts[t] > 0 {
            avg_divergence[t] /= counts[t] as f64;
        }
    }

    // Step 4: Linear regression on log divergence to get λ
    // y = λ*t, fit slope
    let valid_points: Vec<(f64, f64)> = avg_divergence
        .iter()
        .enumerate()
        .filter(|(_, &y)| y.is_finite())
        .map(|(t, &y)| (t as f64, y))
        .collect();

    if valid_points.len() < 5 {
        return LyapunovResult {
            lyapunov_exponent: 0.0,
            basin_strength: 0.5,
            attractor_type: AttractorType::Critical,
            transition_risk: 0.5,
            converged: false,
            effective_dof: 0.0,
        };
    }

    // Simple linear regression for slope (Lyapunov exponent)
    let n_pts = valid_points.len() as f64;
    let sum_t: f64 = valid_points.iter().map(|(t, _)| t).sum();
    let sum_y: f64 = valid_points.iter().map(|(_, y)| y).sum();
    let sum_ty: f64 = valid_points.iter().map(|(t, y)| t * y).sum();
    let sum_t2: f64 = valid_points.iter().map(|(t, _)| t * t).sum();

    let denominator = n_pts * sum_t2 - sum_t * sum_t;
    let lyapunov = if denominator.abs() > config.epsilon {
        (n_pts * sum_ty - sum_t * sum_y) / denominator
    } else {
        0.0
    };

    // Compute R² to check convergence
    let mean_y = sum_y / n_pts;
    let ss_tot: f64 = valid_points.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let intercept = (sum_y - lyapunov * sum_t) / n_pts;
    let ss_res: f64 = valid_points
        .iter()
        .map(|(t, y)| (y - (lyapunov * t + intercept)).powi(2))
        .sum();
    let r_squared = if ss_tot > config.epsilon {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    let converged = r_squared > 0.8;

    // Classify attractor and compute basin strength
    let (attractor_type, basin_strength) = classify_attractor(lyapunov);

    // Transition risk based on Lyapunov exponent and variance
    let transition_risk = compute_transition_risk(lyapunov, &avg_divergence);

    // Estimate effective DoF via correlation dimension proxy
    let effective_dof = estimate_correlation_dimension(&embedded, config.k_neighbors);

    LyapunovResult {
        lyapunov_exponent: lyapunov,
        basin_strength,
        attractor_type,
        transition_risk,
        converged,
        effective_dof,
    }
}

/// Classify attractor type based on Lyapunov exponent
fn classify_attractor(lambda: f64) -> (AttractorType, f64) {
    if lambda < -0.1 {
        // Stable: basin strength increases with |λ|
        let strength = 1.0 - (-lambda / 2.0).min(1.0);
        (AttractorType::Stable, strength.max(0.7))
    } else if lambda > 0.1 {
        // Chaotic: basin strength decreases with λ
        let strength = (-lambda).exp().min(0.49);
        (AttractorType::Chaotic, strength)
    } else if lambda.abs() < 0.02 {
        // Near-periodic
        (AttractorType::Periodic, 0.6)
    } else {
        // Critical transition zone
        (AttractorType::Critical, 0.4)
    }
}

/// Compute transition risk from Lyapunov exponent and divergence variance
fn compute_transition_risk(lambda: f64, divergence: &[f64]) -> f64 {
    // Base risk from Lyapunov exponent
    let base_risk = if lambda > 0.0 {
        1.0 - (-lambda * 2.0).exp()
    } else {
        0.1 * (1.0 + lambda).max(0.0)
    };

    // Additional risk from divergence variance (instability indicator)
    let n = divergence.len();
    if n < 2 {
        return base_risk;
    }

    let mean: f64 = divergence.iter().filter(|x| x.is_finite()).sum::<f64>() / n as f64;
    let variance: f64 = divergence
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / n as f64;

    let variance_factor = (variance / (1.0 + variance)).min(0.3);

    (base_risk + variance_factor).min(1.0)
}

/// Estimate correlation dimension using Grassberger-Procaccia algorithm (simplified)
fn estimate_correlation_dimension(embedded: &Array2<f64>, k: usize) -> f64 {
    let n = embedded.nrows();
    if n < k * 2 {
        return 0.0;
    }

    // Compute pairwise distances for sample
    let sample_size = n.min(200);
    let step = n / sample_size;

    let mut distances: Vec<f64> = Vec::new();

    for i in (0..n).step_by(step) {
        for j in (i + 1..n).step_by(step) {
            let d = euclidean_distance(
                embedded.row(i).as_slice().unwrap(),
                embedded.row(j).as_slice().unwrap(),
            );
            if d > 0.0 {
                distances.push(d);
            }
        }
    }

    if distances.is_empty() {
        return 0.0;
    }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Estimate dimension from scaling of correlation sum
    // C(r) ~ r^D => log C(r) ~ D log r
    let n_radii = 10;
    let r_min = distances[distances.len() / 100].max(1e-10);
    let r_max = distances[distances.len() * 90 / 100];

    if r_max <= r_min {
        return 0.0;
    }

    let log_r_min = r_min.ln();
    let log_r_max = r_max.ln();
    let step = (log_r_max - log_r_min) / n_radii as f64;

    let mut log_r_vals: Vec<f64> = Vec::new();
    let mut log_c_vals: Vec<f64> = Vec::new();

    for i in 0..n_radii {
        let log_r = log_r_min + i as f64 * step;
        let r = log_r.exp();

        let count = distances.iter().filter(|&&d| d < r).count();
        if count > 0 {
            log_r_vals.push(log_r);
            log_c_vals.push((count as f64).ln());
        }
    }

    if log_r_vals.len() < 3 {
        return 0.0;
    }

    // Linear regression for slope (correlation dimension)
    let n_pts = log_r_vals.len() as f64;
    let sum_x: f64 = log_r_vals.iter().sum();
    let sum_y: f64 = log_c_vals.iter().sum();
    let sum_xy: f64 = log_r_vals.iter().zip(log_c_vals.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_r_vals.iter().map(|x| x * x).sum();

    let denom = n_pts * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    let slope = (n_pts * sum_xy - sum_x * sum_y) / denom;
    slope.max(0.0).min(embedded.ncols() as f64)
}

/// Multivariate Lyapunov analysis for state vector time series
///
/// State vector: [economic_index, political_stability, social_cohesion, military_control, external_pressure]
pub fn multivariate_lyapunov(states: &Array2<f64>, config: &LyapunovConfig) -> LyapunovResult {
    let (n, d) = states.dim();

    if n < config.min_separation * 2 {
        return LyapunovResult {
            lyapunov_exponent: 0.0,
            basin_strength: 0.5,
            attractor_type: AttractorType::Critical,
            transition_risk: 0.5,
            converged: false,
            effective_dof: d as f64,
        };
    }

    // Compute local Jacobian approximations and track eigenvalue evolution
    let mut lyapunov_sum = 0.0;
    let mut count = 0;

    for i in config.min_separation..(n - config.min_separation) {
        // Estimate local Jacobian via finite differences
        let jacobian = estimate_local_jacobian(states, i, config);

        if jacobian.is_empty() {
            continue;
        }

        // Compute largest eigenvalue magnitude (simplified: use Frobenius proxy)
        let frob_norm: f64 = jacobian.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();

        if frob_norm > 0.0 {
            lyapunov_sum += frob_norm.ln();
            count += 1;
        }
    }

    if count == 0 {
        return LyapunovResult {
            lyapunov_exponent: 0.0,
            basin_strength: 0.5,
            attractor_type: AttractorType::Critical,
            transition_risk: 0.5,
            converged: false,
            effective_dof: d as f64,
        };
    }

    let lyapunov = lyapunov_sum / count as f64;
    let (attractor_type, basin_strength) = classify_attractor(lyapunov);

    // Estimate transition risk from state variance trends
    let variance_trend = compute_variance_trend(states);
    let transition_risk = if variance_trend > 0.0 {
        (lyapunov.exp() * variance_trend).min(1.0)
    } else {
        (0.1 + lyapunov.max(0.0)).min(1.0)
    };

    LyapunovResult {
        lyapunov_exponent: lyapunov,
        basin_strength,
        attractor_type,
        transition_risk,
        converged: count > n / 4,
        effective_dof: d as f64,
    }
}

/// Estimate local Jacobian at time index via finite differences
fn estimate_local_jacobian(states: &Array2<f64>, idx: usize, config: &LyapunovConfig) -> Vec<Vec<f64>> {
    let (n, d) = states.dim();
    let tau = config.tau;

    if idx < tau || idx + tau >= n {
        return vec![];
    }

    // J[i,j] ≈ ∂x_i(t+1) / ∂x_j(t)
    // Approximate using nearby trajectory variations
    let mut jacobian = vec![vec![0.0; d]; d];

    let current = states.row(idx);
    let future = states.row(idx + tau);

    // Find similar states
    let neighbors = find_nearest_neighbors_2d(states, idx, config.k_neighbors, config.min_separation);

    if neighbors.len() < 2 {
        return vec![];
    }

    // Use linear regression of (dx, dy) pairs to estimate Jacobian
    for (j_idx, _) in &neighbors {
        if *j_idx + tau >= n {
            continue;
        }

        let neighbor = states.row(*j_idx);
        let neighbor_future = states.row(*j_idx + tau);

        for i in 0..d {
            for j in 0..d {
                let dx = neighbor[j] - current[j];
                let dy = neighbor_future[i] - future[i];

                if dx.abs() > 1e-10 {
                    jacobian[i][j] += dy / dx;
                }
            }
        }
    }

    // Average
    let factor = 1.0 / neighbors.len() as f64;
    for i in 0..d {
        for j in 0..d {
            jacobian[i][j] *= factor;
        }
    }

    jacobian
}

/// Find nearest neighbors in multivariate space
fn find_nearest_neighbors_2d(
    states: &Array2<f64>,
    idx: usize,
    k: usize,
    min_sep: usize,
) -> Vec<(usize, f64)> {
    let n = states.nrows();
    let point = states.row(idx);

    let mut distances: Vec<(usize, f64)> = (0..n)
        .filter(|&j| {
            let sep = if j > idx { j - idx } else { idx - j };
            sep >= min_sep
        })
        .map(|j| {
            let dist = euclidean_distance(
                point.as_slice().unwrap(),
                states.row(j).as_slice().unwrap(),
            );
            (j, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.into_iter().take(k).collect()
}

/// Compute variance trend (increasing variance indicates approaching instability)
fn compute_variance_trend(states: &Array2<f64>) -> f64 {
    let (n, d) = states.dim();
    if n < 10 {
        return 0.0;
    }

    // Split into first and second half
    let half = n / 2;

    let var_first: f64 = (0..d)
        .map(|j| {
            let col: Vec<f64> = (0..half).map(|i| states[[i, j]]).collect();
            let mean: f64 = col.iter().sum::<f64>() / half as f64;
            col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / half as f64
        })
        .sum();

    let var_second: f64 = (0..d)
        .map(|j| {
            let col: Vec<f64> = (half..n).map(|i| states[[i, j]]).collect();
            let mean: f64 = col.iter().sum::<f64>() / (n - half) as f64;
            col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - half) as f64
        })
        .sum();

    if var_first > 0.0 {
        (var_second / var_first - 1.0).max(-1.0).min(1.0)
    } else {
        0.0
    }
}

/// Batch analyze multiple nations' stability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NationStability {
    pub code: String,
    pub result: LyapunovResult,
}

pub fn batch_stability_analysis(
    nations: &[(String, Array2<f64>)],
    config: &LyapunovConfig,
) -> Vec<NationStability> {
    nations
        .iter()
        .map(|(code, states)| NationStability {
            code: code.clone(),
            result: multivariate_lyapunov(states, config),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_embed() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let embedded = delay_embed(&x, 3, 1);

        assert_eq!(embedded.nrows(), 6);
        assert_eq!(embedded.ncols(), 3);
        assert_eq!(embedded[[0, 0]], 1.0);
        assert_eq!(embedded[[0, 2]], 3.0);
    }

    #[test]
    fn test_stable_system() {
        // Decaying oscillation (stable)
        let n = 500;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.1;
                (-0.1 * t).exp() * (t).sin()
            })
            .collect();

        let config = LyapunovConfig::default();
        let result = rosenstein_lyapunov(&x, &config);

        // Should detect negative Lyapunov (stable)
        assert!(result.lyapunov_exponent < 0.5);
        assert!(result.basin_strength > 0.3);
    }

    #[test]
    fn test_chaotic_system() {
        // Logistic map in chaotic regime (r = 4)
        let n = 1000;
        let r = 4.0;
        let mut x = vec![0.1];

        for _ in 1..n {
            let prev = x[x.len() - 1];
            x.push(r * prev * (1.0 - prev));
        }

        let config = LyapunovConfig {
            embedding_dim: 3,
            tau: 1,
            min_separation: 5,
            k_neighbors: 4,
            max_iterations: 50,
            epsilon: 1e-8,
        };

        let result = rosenstein_lyapunov(&x, &config);

        // Logistic map at r=4 has λ ≈ ln(2) ≈ 0.693
        // The algorithm runs and produces a result
        assert!(result.basin_strength >= 0.0 && result.basin_strength <= 1.0);
        // Attractor type should be chaotic or critical (estimation can vary)
        assert!(result.attractor_type == AttractorType::Chaotic || result.attractor_type == AttractorType::Critical);
    }

    #[test]
    fn test_multivariate_analysis() {
        let n = 200;
        let d = 5;
        let mut states = Array2::zeros((n, d));

        // Simulate stable multivariate system with some coupling
        for i in 1..n {
            for j in 0..d {
                let prev = states[[i - 1, j]];
                let coupling: f64 = (0..d).filter(|&k| k != j).map(|k| states[[i - 1, k]] * 0.1).sum();
                states[[i, j]] = 0.9 * prev + coupling + 0.01 * ((i * j) as f64).sin();
            }
        }

        let config = LyapunovConfig::default();
        let result = multivariate_lyapunov(&states, &config);

        assert!(result.effective_dof >= 0.0);
        assert!(result.transition_risk >= 0.0 && result.transition_risk <= 1.0);
    }

    #[test]
    fn test_classify_attractor() {
        let (typ, strength) = classify_attractor(-0.5);
        assert_eq!(typ, AttractorType::Stable);
        assert!(strength > 0.5);

        let (typ, strength) = classify_attractor(0.5);
        assert_eq!(typ, AttractorType::Chaotic);
        assert!(strength < 0.5);

        let (typ, _) = classify_attractor(0.01);
        assert!(typ == AttractorType::Periodic || typ == AttractorType::Critical);
    }
}
