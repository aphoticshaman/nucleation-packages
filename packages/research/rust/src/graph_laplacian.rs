//! Graph Laplacian Anomaly Detection
//!
//! Based on US9805002B2 (IBM, expired October 31, 2021).
//!
//! Implements semi-supervised anomaly detection using:
//! - Graph Laplacian regularization
//! - Latent variable models
//! - Spectral clustering for basin boundary detection

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Configuration for graph Laplacian anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnomalyConfig {
    /// Latent dimension (D' in the patent)
    pub latent_dim: usize,
    /// Graph Laplacian regularization strength
    pub lambda_reg: f64,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Similarity for normal-normal pairs
    pub normal_similarity: f64,
    /// Similarity for normal-anomaly pairs
    pub anomaly_similarity: f64,
    /// Similarity for unlabeled pairs
    pub unlabeled_similarity: f64,
}

impl Default for GraphAnomalyConfig {
    fn default() -> Self {
        Self {
            latent_dim: 10,
            lambda_reg: 1.0,
            learning_rate: 0.01,
            max_iter: 100,
            tol: 1e-6,
            normal_similarity: 1.0,
            anomaly_similarity: -1.0,
            unlabeled_similarity: 0.0,
        }
    }
}

/// Construct similarity matrix encoding label constraints
///
/// From US9805002B2:
/// - Normal-normal pairs: positive similarity 'a'
/// - Normal-anomalous pairs: non-positive similarity 'b'
pub fn construct_similarity_matrix(
    n_samples: usize,
    labeled_normal: &[usize],
    labeled_anomaly: &[usize],
    config: &GraphAnomalyConfig,
) -> Array2<f64> {
    let mut r = Array2::from_elem((n_samples, n_samples), config.unlabeled_similarity);

    // Normal-normal pairs
    for &i in labeled_normal {
        for &j in labeled_normal {
            if i != j && i < n_samples && j < n_samples {
                r[[i, j]] = config.normal_similarity;
            }
        }
    }

    // Anomaly-anomaly pairs
    for &i in labeled_anomaly {
        for &j in labeled_anomaly {
            if i != j && i < n_samples && j < n_samples {
                r[[i, j]] = config.normal_similarity;
            }
        }
    }

    // Normal-anomaly pairs
    for &i in labeled_normal {
        for &j in labeled_anomaly {
            if i < n_samples && j < n_samples {
                r[[i, j]] = config.anomaly_similarity;
                r[[j, i]] = config.anomaly_similarity;
            }
        }
    }

    r
}

/// Compute graph Laplacian from similarity matrix
///
/// L = D_R - R where D_R is the diagonal degree matrix
pub fn graph_laplacian(r: &Array2<f64>) -> Array2<f64> {
    let n = r.nrows();
    let mut laplacian = -r.clone();

    // Add degree matrix on diagonal
    for i in 0..n {
        let degree: f64 = r.row(i).sum();
        laplacian[[i, i]] = degree;
    }

    laplacian
}

/// Normalized graph Laplacian: L_sym = D^(-1/2) L D^(-1/2)
pub fn normalized_laplacian(r: &Array2<f64>) -> Array2<f64> {
    let n = r.nrows();
    let l = graph_laplacian(r);

    // Compute degree vector
    let degrees: Array1<f64> = r.sum_axis(Axis(1));

    // D^(-1/2)
    let d_inv_sqrt: Array1<f64> = degrees.mapv(|d| {
        if d > 1e-10 {
            1.0 / d.sqrt()
        } else {
            0.0
        }
    });

    // L_sym = D^(-1/2) L D^(-1/2)
    let mut l_sym = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            l_sym[[i, j]] = d_inv_sqrt[i] * l[[i, j]] * d_inv_sqrt[j];
        }
    }

    l_sym
}

/// Result of anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Anomaly scores for each sample
    pub scores: Vec<f64>,
    /// Latent embedding (flattened)
    pub embedding: Vec<f64>,
    /// Latent dimensions
    pub latent_dims: usize,
    /// Basin boundary proximity scores
    pub basin_proximity: Vec<f64>,
}

/// Latent variable model with graph Laplacian regularization
pub struct GraphLatentDetector {
    config: GraphAnomalyConfig,
    w: Option<Array2<f64>>, // Sensor coefficients
    z: Option<Array2<f64>>, // Latent variables
}

impl GraphLatentDetector {
    pub fn new(config: GraphAnomalyConfig) -> Self {
        Self {
            config,
            w: None,
            z: None,
        }
    }

    /// Fit the latent variable model using gradient descent
    ///
    /// From US9805002B2, gradient updates:
    /// W := W - α[{S ⊙ (X - ZW^T)}^T Z + N(WW^T)^{-1}W]
    /// Z := Z - α[{S ⊙ (X - ZW^T)}W + λLZ]
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        labeled_normal: &[usize],
        labeled_anomaly: &[usize],
    ) {
        let (n, d) = (x.nrows(), x.ncols());
        let d_prime = self.config.latent_dim;

        // Initialize
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut w = Array2::from_shape_fn((d, d_prime), |_| rng.gen::<f64>() * 0.1);
        let mut z = Array2::from_shape_fn((n, d_prime), |_| rng.gen::<f64>() * 0.1);

        // Construct similarity and Laplacian
        let r = construct_similarity_matrix(n, labeled_normal, labeled_anomaly, &self.config);
        let l = graph_laplacian(&r);

        let mut prev_loss = f64::INFINITY;

        for _ in 0..self.config.max_iter {
            // Reconstruction error: X - ZW^T
            let reconstruction = z.dot(&w.t());
            let residual = x - &reconstruction;

            // W update
            let grad_w = -residual.t().dot(&z) + &w * self.config.lambda_reg;
            w = w - &grad_w * self.config.learning_rate;

            // Z update with Laplacian regularization
            let grad_z = -residual.dot(&w) + l.dot(&z) * self.config.lambda_reg;
            z = z - &grad_z * self.config.learning_rate;

            // Check convergence
            let loss = residual.mapv(|x| x * x).sum()
                + self.config.lambda_reg * z.t().dot(&l).dot(&z).diag().sum();

            if (prev_loss - loss).abs() < self.config.tol {
                break;
            }
            prev_loss = loss;
        }

        self.w = Some(w);
        self.z = Some(z);
    }

    /// Compute anomaly scores via reconstruction error
    ///
    /// From US9805002B2:
    /// s_n = (I - W(W^TW)^{-1}W^T) · X_n
    pub fn score(&self, x: &Array2<f64>) -> Vec<f64> {
        let w = match &self.w {
            Some(w) => w,
            None => return vec![0.0; x.nrows()],
        };

        let n = x.nrows();
        let d = x.ncols();

        // Compute W(W^TW)^{-1}W^T using pseudo-inverse
        let wtw = w.t().dot(w);

        // Simple regularized inverse
        let mut wtw_inv = Array2::zeros((wtw.nrows(), wtw.ncols()));
        for i in 0..wtw.nrows() {
            for j in 0..wtw.ncols() {
                if i == j {
                    wtw_inv[[i, j]] = 1.0 / (wtw[[i, j]] + 1e-6);
                }
            }
        }

        // Projection matrix onto orthogonal complement
        let p_w = w.dot(&wtw_inv).dot(&w.t());
        let p_orth = Array2::eye(d) - p_w;

        // Orthogonal component magnitude
        let x_orth = x.dot(&p_orth);

        (0..n)
            .map(|i| x_orth.row(i).mapv(|v| v * v).sum().sqrt())
            .collect()
    }

    /// Get latent embedding
    pub fn embedding(&self) -> Option<Vec<f64>> {
        self.z.as_ref().map(|z| z.clone().into_raw_vec())
    }

    /// Compute basin boundary proximity
    ///
    /// Points near basin boundaries have high gradient magnitude
    pub fn basin_proximity(&self, x: &Array2<f64>) -> Vec<f64> {
        let (z, w) = match (&self.z, &self.w) {
            (Some(z), Some(w)) => (z, w),
            _ => return vec![0.0; x.nrows()],
        };

        let n = z.nrows();
        let d_prime = z.ncols();
        let eps = 1e-4;

        let mut gradients = Array2::zeros((n, d_prime));

        for d in 0..d_prime {
            // Perturb in dimension d
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();

            for i in 0..n {
                z_plus[[i, d]] += eps;
                z_minus[[i, d]] -= eps;
            }

            let x_plus = z_plus.dot(&w.t());
            let x_minus = z_minus.dot(&w.t());

            let scores_plus = self.score(&x_plus);
            let scores_minus = self.score(&x_minus);

            for i in 0..n {
                gradients[[i, d]] = (scores_plus[i] - scores_minus[i]) / (2.0 * eps);
            }
        }

        // Gradient magnitude
        (0..n)
            .map(|i| gradients.row(i).mapv(|v| v * v).sum().sqrt())
            .collect()
    }
}

/// Spectral anomaly detection using eigendecomposition
///
/// Uses the Ng-Jordan-Weiss algorithm from US7103225B2 (NEC, expired 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnomalyResult {
    /// Spectral embedding
    pub embedding: Vec<f64>,
    /// Embedding dimensions
    pub n_dims: usize,
    /// Number of samples
    pub n_samples: usize,
    /// Anomaly scores
    pub scores: Vec<f64>,
    /// Eigenvalues used
    pub eigenvalues: Vec<f64>,
}

/// Compute spectral embedding and anomaly scores
///
/// 1. Compute normalized Laplacian from similarity matrix
/// 2. Find first k eigenvectors
/// 3. Anomalies are points far from embedding centroid
pub fn spectral_anomaly_detection(
    similarity: &Array2<f64>,
    n_components: usize,
) -> SpectralAnomalyResult {
    let n = similarity.nrows();
    let l_sym = normalized_laplacian(similarity);

    // Power iteration for dominant eigenvectors
    // (Simplified - for production use nalgebra or ndarray-linalg)
    let k = n_components.min(n);
    let mut embedding = Array2::zeros((n, k));
    let mut eigenvalues = Vec::with_capacity(k);

    // Initialize with random vectors
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for comp in 0..k {
        let mut v = Array1::from_shape_fn(n, |_| rng.gen::<f64>() - 0.5);

        // Power iteration for smallest eigenvalue (largest for I - L_sym)
        let shifted = Array2::eye(n) - &l_sym;

        for _ in 0..50 {
            v = shifted.dot(&v);

            // Orthogonalize against previous eigenvectors
            for prev in 0..comp {
                let prev_vec = embedding.column(prev);
                let proj: f64 = v.iter().zip(prev_vec.iter()).map(|(a, b)| a * b).sum();
                v = v - &(prev_vec.to_owned() * proj);
            }

            // Normalize
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                v = v / norm;
            }
        }

        // Compute eigenvalue (Rayleigh quotient)
        let av = l_sym.dot(&v);
        let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
        eigenvalues.push(eigenvalue);

        for i in 0..n {
            embedding[[i, comp]] = v[i];
        }
    }

    // Normalize rows to unit length
    for i in 0..n {
        let norm: f64 = embedding.row(i).mapv(|x| x * x).sum().sqrt();
        if norm > 1e-10 {
            for j in 0..k {
                embedding[[i, j]] /= norm;
            }
        }
    }

    // Anomaly score: distance from centroid
    let centroid: Array1<f64> = embedding.mean_axis(Axis(0)).unwrap();
    let scores: Vec<f64> = (0..n)
        .map(|i| {
            let row = embedding.row(i);
            row.iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    SpectralAnomalyResult {
        embedding: embedding.into_raw_vec(),
        n_dims: k,
        n_samples: n,
        scores,
        eigenvalues,
    }
}

/// Detect coherence amplification over time
///
/// When distributed agents synchronize, the spectral embedding becomes
/// more concentrated. This indicates an amplified attractor well forming.
pub fn detect_coherence_amplification(dispersions_over_time: &[f64]) -> Vec<f64> {
    if dispersions_over_time.len() < 2 {
        return vec![0.0; dispersions_over_time.len()];
    }

    // Coherence = negative gradient of dispersion
    // Increasing coherence = decreasing dispersion
    let mut coherence = Vec::with_capacity(dispersions_over_time.len());

    // First derivative using central differences
    coherence.push(dispersions_over_time[0] - dispersions_over_time[1]);

    for i in 1..dispersions_over_time.len() - 1 {
        let grad = (dispersions_over_time[i - 1] - dispersions_over_time[i + 1]) / 2.0;
        coherence.push(grad);
    }

    let n = dispersions_over_time.len();
    coherence.push(dispersions_over_time[n - 2] - dispersions_over_time[n - 1]);

    coherence
}

/// Compute embedding dispersion (trace of covariance)
pub fn embedding_dispersion(embedding: &[f64], n_samples: usize, n_dims: usize) -> f64 {
    if n_samples == 0 || n_dims == 0 {
        return 0.0;
    }

    // Compute mean per dimension
    let mut mean = vec![0.0; n_dims];
    for i in 0..n_samples {
        for j in 0..n_dims {
            mean[j] += embedding[i * n_dims + j];
        }
    }
    for j in 0..n_dims {
        mean[j] /= n_samples as f64;
    }

    // Compute trace of covariance (sum of variances)
    let mut trace = 0.0;
    for j in 0..n_dims {
        let mut variance = 0.0;
        for i in 0..n_samples {
            let diff = embedding[i * n_dims + j] - mean[j];
            variance += diff * diff;
        }
        trace += variance / n_samples as f64;
    }

    trace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_laplacian() {
        let r = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 1.0, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.0],
        )
        .unwrap();

        let l = graph_laplacian(&r);

        // Check diagonal is degree
        assert!((l[[0, 0]] - 1.5).abs() < 1e-10);
        assert!((l[[1, 1]] - 1.5).abs() < 1e-10);
        assert!((l[[2, 2]] - 1.0).abs() < 1e-10);

        // Off-diagonal is negative similarity
        assert!((l[[0, 1]] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_matrix() {
        let r = construct_similarity_matrix(
            5,
            &[0, 1],
            &[3, 4],
            &GraphAnomalyConfig::default(),
        );

        // Normal-normal should be positive
        assert!(r[[0, 1]] > 0.0);

        // Normal-anomaly should be negative
        assert!(r[[0, 3]] < 0.0);
        assert!(r[[1, 4]] < 0.0);

        // Unlabeled should be zero
        assert_eq!(r[[0, 2]], 0.0);
    }

    #[test]
    fn test_coherence_amplification() {
        // Decreasing dispersion = increasing coherence
        let dispersions = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        let coherence = detect_coherence_amplification(&dispersions);

        // All should be positive (dispersion decreasing)
        for c in &coherence[1..coherence.len() - 1] {
            assert!(*c > 0.0, "Expected positive coherence, got {}", c);
        }
    }
}
