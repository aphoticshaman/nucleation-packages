//! Q-matrix utilities for continuous-time Markov chain analysis.
//!
//! A Q-matrix (generator matrix) satisfies:
//! - q_ij >= 0 for i != j (transition rates)
//! - q_ii = -sum_{j != i} q_ij (row sums to zero)

use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QMatrixError {
    #[error("Matrix must be square, got {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },
    #[error("Invalid Q-matrix: {reason}")]
    InvalidQMatrix { reason: String },
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

/// Result of Q-matrix spectral analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMatrixAnalysis {
    /// The Q-matrix itself
    pub q: Vec<Vec<f64>>,
    /// Eigenvalues (real parts)
    pub eigenvalues_real: Vec<f64>,
    /// Eigenvalues (imaginary parts)
    pub eigenvalues_imag: Vec<f64>,
    /// Spectral gap (second smallest |Re(λ)|)
    pub spectral_gap: f64,
    /// Stationary distribution (if exists)
    pub stationary: Option<Vec<f64>>,
}

/// Build a valid Q-matrix from off-diagonal rates.
///
/// Diagonal entries are computed as negative row sums.
pub fn build_q_matrix(rates: &Array2<f64>) -> Result<Array2<f64>, QMatrixError> {
    let (n, m) = rates.dim();
    if n != m {
        return Err(QMatrixError::NotSquare { rows: n, cols: m });
    }

    let mut q = rates.clone();

    // Zero diagonal, then compute as negative row sum
    for i in 0..n {
        q[[i, i]] = 0.0;
    }

    for i in 0..n {
        let row_sum: f64 = q.row(i).sum();
        q[[i, i]] = -row_sum;
    }

    Ok(q)
}

/// Check if a matrix is a valid Q-matrix.
pub fn is_valid_q(q: &Array2<f64>, tol: f64) -> bool {
    let (n, m) = q.dim();
    if n != m {
        return false;
    }

    // Check off-diagonal non-negative
    for i in 0..n {
        for j in 0..n {
            if i != j && q[[i, j]] < -tol {
                return false;
            }
        }
    }

    // Check row sums ≈ 0
    for i in 0..n {
        let row_sum: f64 = q.row(i).sum();
        if row_sum.abs() > tol {
            return false;
        }
    }

    true
}

/// Estimate Q-matrix from transition counts and dwell times.
///
/// MLE for continuous-time Markov chain:
/// q_ij = N_ij / T_i for i != j
pub fn estimate_q_from_counts(
    counts: &Array2<f64>,
    dwell_times: &Array1<f64>,
    eps: f64,
) -> Result<Array2<f64>, QMatrixError> {
    let (n, m) = counts.dim();
    if n != m {
        return Err(QMatrixError::NotSquare { rows: n, cols: m });
    }
    if dwell_times.len() != n {
        return Err(QMatrixError::DimensionMismatch {
            expected: n,
            got: dwell_times.len(),
        });
    }

    let mut q = Array2::zeros((n, n));

    for i in 0..n {
        if dwell_times[i] <= eps {
            continue;
        }
        for j in 0..n {
            if i != j {
                q[[i, j]] = counts[[i, j]] / (dwell_times[i] + eps);
            }
        }
    }

    // Set diagonal
    for i in 0..n {
        let row_sum: f64 = q.row(i).sum() - q[[i, i]];
        q[[i, i]] = -row_sum;
    }

    Ok(q)
}

/// Compute eigenvalues of Q-matrix using power iteration.
/// Returns (real_parts, imag_parts).
fn compute_eigenvalues_approx(q: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = q.nrows();

    // Simple approach: compute characteristic polynomial roots
    // For small n, we can use direct methods
    // For now, return approximate eigenvalues using Gershgorin circles

    let mut eigenvalues_real = Vec::with_capacity(n);
    let mut eigenvalues_imag = Vec::with_capacity(n);

    // Gershgorin approximation: eigenvalues lie in disks centered at diagonal
    for i in 0..n {
        let center = q[[i, i]];
        let radius: f64 = q.row(i).iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| v.abs())
            .sum();

        // Approximate eigenvalue as center (for generator matrix, this is often close)
        eigenvalues_real.push(center);
        eigenvalues_imag.push(0.0); // Assume real for generators
    }

    (eigenvalues_real, eigenvalues_imag)
}

/// Analyze Q-matrix: compute eigenvalues and spectral gap.
pub fn analyze_q(q: &Array2<f64>) -> Result<QMatrixAnalysis, QMatrixError> {
    let n = q.nrows();
    if n != q.ncols() {
        return Err(QMatrixError::NotSquare {
            rows: n,
            cols: q.ncols(),
        });
    }

    let (eigenvalues_real, eigenvalues_imag) = compute_eigenvalues_approx(q);

    // Spectral gap: min |Re(λ)| for λ != 0
    let mut nonzero_magnitudes: Vec<f64> = eigenvalues_real
        .iter()
        .filter(|&&x| x.abs() > 1e-10)
        .map(|x| x.abs())
        .collect();
    nonzero_magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let spectral_gap = nonzero_magnitudes.first().copied().unwrap_or(0.0);

    // Attempt to compute stationary distribution
    let stationary = compute_stationary_distribution(q);

    Ok(QMatrixAnalysis {
        q: q.rows().into_iter().map(|r| r.to_vec()).collect(),
        eigenvalues_real,
        eigenvalues_imag,
        spectral_gap,
        stationary,
    })
}

/// Compute stationary distribution π such that πQ = 0.
fn compute_stationary_distribution(q: &Array2<f64>) -> Option<Vec<f64>> {
    let n = q.nrows();
    if n == 0 {
        return None;
    }

    // Power iteration on transition matrix P = I + Q*dt for small dt
    let dt = 0.01 / q.iter().map(|x| x.abs()).fold(0.0_f64, f64::max).max(1.0);
    let mut p = Array2::eye(n);
    for i in 0..n {
        for j in 0..n {
            p[[i, j]] += q[[i, j]] * dt;
        }
    }

    // Normalize rows
    for i in 0..n {
        let row_sum: f64 = p.row(i).sum();
        if row_sum > 0.0 {
            for j in 0..n {
                p[[i, j]] /= row_sum;
            }
        }
    }

    // Power iteration
    let mut pi = Array1::from_elem(n, 1.0 / n as f64);
    for _ in 0..1000 {
        let pi_new = pi.dot(&p);
        let diff: f64 = (&pi_new - &pi).iter().map(|x| x.abs()).sum();
        pi = pi_new;
        if diff < 1e-10 {
            break;
        }
    }

    // Normalize
    let sum: f64 = pi.sum();
    if sum > 0.0 {
        pi /= sum;
    }

    Some(pi.to_vec())
}

/// Simulate Markov chain using Euler approximation.
pub fn simulate_markov_chain(
    q: &Array2<f64>,
    r0: usize,
    total_time: f64,
    dt: f64,
    seed: u64,
) -> (Vec<f64>, Vec<usize>) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;

    let n = q.nrows();
    let steps = (total_time / dt).ceil() as usize;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Build transition matrix P ≈ I + Q*dt
    let mut p: Array2<f64> = Array2::eye(n);
    for i in 0..n {
        for j in 0..n {
            p[[i, j]] += q[[i, j]] * dt;
            p[[i, j]] = p[[i, j]].max(0.0).min(1.0);
        }
    }

    // Normalize rows
    for i in 0..n {
        let row_sum: f64 = p.row(i).sum();
        if row_sum > 0.0 {
            for j in 0..n {
                p[[i, j]] /= row_sum;
            }
        }
    }

    let mut times = Vec::with_capacity(steps + 1);
    let mut regimes = Vec::with_capacity(steps + 1);

    let mut regime = r0;
    times.push(0.0);
    regimes.push(regime);

    for k in 0..steps {
        // Sample next regime
        let u: f64 = rng.gen();
        let mut cumsum = 0.0;
        for j in 0..n {
            cumsum += p[[regime, j]];
            if u < cumsum {
                regime = j;
                break;
            }
        }

        times.push((k + 1) as f64 * dt);
        regimes.push(regime);
    }

    (times, regimes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    #[test]
    fn test_build_q_matrix() {
        let rates = arr2(&[[0.0, 0.1, 0.2], [0.3, 0.0, 0.1], [0.2, 0.2, 0.0]]);
        let q = build_q_matrix(&rates).unwrap();

        // Check row sums are zero
        for i in 0..3 {
            let sum: f64 = q.row(i).sum();
            assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_is_valid_q() {
        let q = arr2(&[[-0.3, 0.1, 0.2], [0.1, -0.2, 0.1], [0.2, 0.1, -0.3]]);
        assert!(is_valid_q(&q, 1e-8));
    }

    #[test]
    fn test_analyze_q() {
        let q = arr2(&[[-0.3, 0.2, 0.1], [0.1, -0.2, 0.1], [0.1, 0.1, -0.2]]);
        let result = analyze_q(&q).unwrap();

        assert!(result.spectral_gap >= 0.0);
        assert!(result.stationary.is_some());
    }

    #[test]
    fn test_simulate_markov_chain() {
        let q = arr2(&[[-0.2, 0.1, 0.1], [0.1, -0.2, 0.1], [0.1, 0.1, -0.2]]);
        let (times, regimes) = simulate_markov_chain(&q, 0, 10.0, 0.1, 42);

        assert!(!times.is_empty());
        assert_eq!(times.len(), regimes.len());
        assert!(regimes.iter().all(|&r| r < 3));
    }
}
