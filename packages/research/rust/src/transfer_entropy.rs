//! Transfer Entropy computation for causal graph construction
//!
//! Based on arXiv:2312.09478 and US5857978A (Lockheed, expired 2011).
//!
//! Transfer entropy measures directed information flow from source X to target Y,
//! quantifying how much knowing X's past reduces uncertainty about Y's future
//! beyond what Y's own past provides.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Configuration for transfer entropy estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferEntropyConfig {
    /// k for k-NN estimator (Kraskov et al.)
    pub k_neighbors: usize,
    /// History length for source
    pub lag_x: usize,
    /// History length for target
    pub lag_y: usize,
    /// Minimum TE to include edge
    pub threshold: f64,
    /// Normalize by target entropy
    pub normalize: bool,
}

impl Default for TransferEntropyConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 4,
            lag_x: 1,
            lag_y: 1,
            threshold: 0.0,
            normalize: true,
        }
    }
}

/// Create delay embedding of time series
fn embed_time_series(x: &[f64], lag: usize) -> Array2<f64> {
    if x.len() <= lag {
        return Array2::zeros((0, lag));
    }

    let t = x.len() - lag;
    let mut embedded = Array2::zeros((t, lag));

    for i in 0..t {
        for j in 0..lag {
            embedded[[i, j]] = x[i + j];
        }
    }

    embedded
}

/// Compute Euclidean distance between two points
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Chebyshev (max) distance between two points
fn chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Find k-th nearest neighbor distance for each point
fn knn_distances(points: &Array2<f64>, k: usize) -> Vec<f64> {
    let n = points.nrows();
    let mut distances = vec![0.0; n];

    for i in 0..n {
        let mut dists: Vec<f64> = (0..n)
            .filter(|&j| i != j)
            .map(|j| {
                chebyshev_distance(
                    points.row(i).as_slice().unwrap(),
                    points.row(j).as_slice().unwrap(),
                )
            })
            .collect();

        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        distances[i] = if k <= dists.len() {
            dists[k - 1]
        } else if !dists.is_empty() {
            dists[dists.len() - 1]
        } else {
            0.0
        };
    }

    distances
}

/// Count neighbors within radius eps using Chebyshev distance
fn count_neighbors(points: &Array2<f64>, idx: usize, eps: f64) -> usize {
    let n = points.nrows();
    let point = points.row(idx);

    (0..n)
        .filter(|&j| {
            if j == idx {
                return false;
            }
            chebyshev_distance(
                point.as_slice().unwrap(),
                points.row(j).as_slice().unwrap(),
            ) <= eps
        })
        .count()
}

/// Digamma function approximation (psi function)
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Use asymptotic expansion for large x
    if x >= 6.0 {
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;
        return x.ln() - 0.5 * inv_x
            - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 / 252.0));
    }

    // Use recurrence relation for small x
    digamma(x + 1.0) - 1.0 / x
}

/// Kraskov-Stögbauer-Grassberger mutual information estimator
///
/// Based on k-nearest neighbor distances.
pub fn kraskov_mi(x: &Array2<f64>, y: &Array2<f64>, k: usize) -> f64 {
    let n = x.nrows();
    if n == 0 || x.nrows() != y.nrows() {
        return 0.0;
    }

    // Joint space XY
    let d_x = x.ncols();
    let d_y = y.ncols();
    let mut xy = Array2::zeros((n, d_x + d_y));

    for i in 0..n {
        for j in 0..d_x {
            xy[[i, j]] = x[[i, j]];
        }
        for j in 0..d_y {
            xy[[i, d_x + j]] = y[[i, j]];
        }
    }

    // Find k-th neighbor distances in joint space
    let eps_vec = knn_distances(&xy, k);

    // Count neighbors in marginal spaces
    let mut psi_nx_sum = 0.0;
    let mut psi_ny_sum = 0.0;

    for i in 0..n {
        let eps = eps_vec[i];
        let n_x = count_neighbors(x, i, eps);
        let n_y = count_neighbors(y, i, eps);

        psi_nx_sum += digamma((n_x + 1) as f64);
        psi_ny_sum += digamma((n_y + 1) as f64);
    }

    // KSG estimator
    let mi = digamma(k as f64) - (psi_nx_sum + psi_ny_sum) / n as f64 + digamma(n as f64);

    mi.max(0.0) // MI is non-negative
}

/// Compute transfer entropy from source to target
///
/// TE_{X→Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})
///
/// This measures the reduction in uncertainty about Y's future when
/// we also know X's past, beyond what Y's own past provides.
pub fn transfer_entropy(source: &[f64], target: &[f64], config: &TransferEntropyConfig) -> f64 {
    // Embed time series
    let x_past = embed_time_series(source, config.lag_x);
    let y_past = embed_time_series(target, config.lag_y);

    if x_past.nrows() == 0 || y_past.nrows() == 0 {
        return 0.0;
    }

    // Align lengths
    let min_len = x_past.nrows().min(y_past.nrows());
    let x_past = x_past.slice(ndarray::s![x_past.nrows() - min_len.., ..]).to_owned();
    let y_past = y_past.slice(ndarray::s![y_past.nrows() - min_len.., ..]).to_owned();

    // Future of target (one step ahead)
    let target_len = target.len();
    if target_len <= min_len {
        return 0.0;
    }

    let y_future: Array2<f64> = Array2::from_shape_fn((min_len - 1, 1), |(i, _)| {
        target[target_len - min_len + i + 1]
    });

    // Align all arrays
    let x_past = x_past.slice(ndarray::s![..min_len - 1, ..]).to_owned();
    let y_past = y_past.slice(ndarray::s![..min_len - 1, ..]).to_owned();

    if y_future.nrows() == 0 {
        return 0.0;
    }

    // TE = I(Y_future; X_past | Y_past)
    // = I(Y_future; X_past, Y_past) - I(Y_future; Y_past)

    // Concatenate X_past and Y_past
    let d_x = x_past.ncols();
    let d_y = y_past.ncols();
    let n = x_past.nrows();

    let mut xy_past = Array2::zeros((n, d_x + d_y));
    for i in 0..n {
        for j in 0..d_x {
            xy_past[[i, j]] = x_past[[i, j]];
        }
        for j in 0..d_y {
            xy_past[[i, d_x + j]] = y_past[[i, j]];
        }
    }

    let mi_full = kraskov_mi(&y_future, &xy_past, config.k_neighbors);
    let mi_cond = kraskov_mi(&y_future, &y_past, config.k_neighbors);

    let mut te = mi_full - mi_cond;

    if config.normalize && mi_cond > 0.0 {
        te /= mi_cond;
    }

    te.max(0.0)
}

/// Causal graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    /// Node names
    pub nodes: Vec<String>,
    /// Adjacency matrix (flattened, row-major)
    pub adjacency: Vec<f64>,
    /// Number of nodes
    pub n_nodes: usize,
    /// Threshold used
    pub threshold: f64,
}

impl CausalGraph {
    /// Get edge weight from source to target
    pub fn get_edge(&self, source: usize, target: usize) -> f64 {
        if source >= self.n_nodes || target >= self.n_nodes {
            return 0.0;
        }
        self.adjacency[target * self.n_nodes + source]
    }

    /// Get out-degree (total influence of a node)
    pub fn out_degree(&self, node: usize) -> f64 {
        if node >= self.n_nodes {
            return 0.0;
        }
        (0..self.n_nodes)
            .map(|target| self.get_edge(node, target))
            .sum()
    }

    /// Get in-degree (total influence on a node)
    pub fn in_degree(&self, node: usize) -> f64 {
        if node >= self.n_nodes {
            return 0.0;
        }
        (0..self.n_nodes)
            .map(|source| self.get_edge(source, node))
            .sum()
    }

    /// Get normalized Laplacian matrix (flattened)
    pub fn normalized_laplacian(&self) -> Vec<f64> {
        let n = self.n_nodes;
        let mut laplacian = vec![0.0; n * n];

        // Compute degree vector
        let degrees: Vec<f64> = (0..n).map(|i| self.in_degree(i) + self.out_degree(i)).collect();

        // D^(-1/2)
        let d_inv_sqrt: Vec<f64> = degrees
            .iter()
            .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // L = I - D^(-1/2) A D^(-1/2)
        for i in 0..n {
            for j in 0..n {
                let a_ij = self.adjacency[i * n + j] + self.adjacency[j * n + i]; // Symmetrize
                if i == j {
                    laplacian[i * n + j] = 1.0 - d_inv_sqrt[i] * a_ij * d_inv_sqrt[j];
                } else {
                    laplacian[i * n + j] = -d_inv_sqrt[i] * a_ij * d_inv_sqrt[j];
                }
            }
        }

        laplacian
    }
}

/// Build weighted directed graph from transfer entropy between all pairs
pub fn build_causal_graph(
    signals: &[Vec<f64>],
    names: Vec<String>,
    config: &TransferEntropyConfig,
) -> CausalGraph {
    let n = signals.len();
    let mut adjacency = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let te = transfer_entropy(&signals[j], &signals[i], config);
                if te > config.threshold {
                    adjacency[i * n + j] = te;
                }
            }
        }
    }

    CausalGraph {
        nodes: names,
        adjacency,
        n_nodes: n,
        threshold: config.threshold,
    }
}

/// Causal structure shift detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureShift {
    /// Window end index
    pub window_end: usize,
    /// Change score (Frobenius norm of adjacency difference)
    pub change_score: f64,
    /// Causal graph for this window
    pub graph: CausalGraph,
}

/// Detect shifts in causal structure over time using rolling windows
pub fn detect_structure_shifts(
    signals: &[Vec<f64>],
    names: Vec<String>,
    window_size: usize,
    step_size: usize,
    config: &TransferEntropyConfig,
) -> Vec<StructureShift> {
    let t = signals.iter().map(|s| s.len()).min().unwrap_or(0);
    if t < window_size {
        return vec![];
    }

    let mut results = Vec::new();
    let mut prev_adj: Option<Vec<f64>> = None;

    let mut start = 0;
    while start + window_size <= t {
        let end = start + window_size;

        // Extract windowed signals
        let windowed: Vec<Vec<f64>> = signals
            .iter()
            .map(|s| s[start..end].to_vec())
            .collect();

        // Build causal graph for this window
        let graph = build_causal_graph(&windowed, names.clone(), config);

        // Compute structure change from previous window
        let change_score = if let Some(ref prev) = prev_adj {
            // Frobenius norm of difference
            graph
                .adjacency
                .iter()
                .zip(prev.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        } else {
            0.0
        };

        results.push(StructureShift {
            window_end: end,
            change_score,
            graph: graph.clone(),
        });

        prev_adj = Some(graph.adjacency);
        start += step_size;
    }

    results
}

/// Compute intentionality gradient (rate of change in directional TE)
///
/// Rising gradient = strengthening attractor
/// Falling gradient = weakening attractor
pub fn intentionality_gradient(
    signals: &[Vec<f64>],
    target_idx: usize,
    window_size: usize,
    config: &TransferEntropyConfig,
) -> Vec<f64> {
    let t = signals.iter().map(|s| s.len()).min().unwrap_or(0);
    let n = signals.len();

    if t < window_size || target_idx >= n {
        return vec![];
    }

    // Compute TE from each source to target in rolling windows
    let mut te_series: Vec<Vec<f64>> = (0..n)
        .filter(|&i| i != target_idx)
        .map(|_| Vec::new())
        .collect();

    for start in 0..=(t - window_size) {
        let target_window = &signals[target_idx][start..start + window_size];

        let mut source_idx = 0;
        for i in 0..n {
            if i != target_idx {
                let source_window = &signals[i][start..start + window_size];
                let te = transfer_entropy(source_window, target_window, config);
                te_series[source_idx].push(te);
                source_idx += 1;
            }
        }
    }

    // Compute gradients
    let mut gradients: Vec<Vec<f64>> = te_series
        .iter()
        .map(|series| {
            if series.len() < 2 {
                return vec![0.0; series.len()];
            }

            let mut grad = Vec::with_capacity(series.len());
            grad.push(series[1] - series[0]);

            for i in 1..series.len() - 1 {
                grad.push((series[i + 1] - series[i - 1]) / 2.0);
            }

            let n = series.len();
            grad.push(series[n - 1] - series[n - 2]);

            grad
        })
        .collect();

    // Aggregate: sum of absolute gradients
    let n_windows = gradients.first().map(|g| g.len()).unwrap_or(0);
    (0..n_windows)
        .map(|i| {
            gradients
                .iter()
                .map(|g| g.get(i).unwrap_or(&0.0).abs())
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_time_series() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let embedded = embed_time_series(&x, 2);

        assert_eq!(embedded.nrows(), 3);
        assert_eq!(embedded.ncols(), 2);
        assert_eq!(embedded[[0, 0]], 1.0);
        assert_eq!(embedded[[0, 1]], 2.0);
    }

    #[test]
    fn test_digamma() {
        // digamma(1) = -gamma (Euler-Mascheroni constant)
        let psi_1 = digamma(1.0);
        assert!((psi_1 - (-0.5772)).abs() < 0.01);

        // digamma(2) = 1 - gamma
        let psi_2 = digamma(2.0);
        assert!((psi_2 - 0.4228).abs() < 0.01);
    }

    #[test]
    fn test_transfer_entropy_self() {
        // TE from a signal to itself should be low
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let config = TransferEntropyConfig::default();
        let te = transfer_entropy(&x, &x, &config);

        // Self-prediction doesn't add info beyond own past
        assert!(te < 1.0);
    }

    #[test]
    fn test_causal_graph() {
        let signals = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], // Lagged version of first
        ];
        let names = vec!["A".to_string(), "B".to_string()];
        let config = TransferEntropyConfig::default();

        let graph = build_causal_graph(&signals, names, &config);

        assert_eq!(graph.n_nodes, 2);
        assert_eq!(graph.adjacency.len(), 4);
    }
}
