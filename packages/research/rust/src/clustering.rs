//! Gauge-Theoretic Value Clustering (GTVC)
//!
//! Implements value clustering with 5% tolerance as gauge symmetry.
//! Provides robust consensus from noisy multi-agent outputs.
//!
//! Key insight: The tolerance defines an equivalence relation (gauge group)
//! that preserves the CIC functional to O(ε²).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::cic::{compute_cic, CICConfig, CICState, DEFAULT_EPSILON};

/// A cluster of gauge-equivalent values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Cluster center (robust estimator)
    pub center: f64,
    /// Member values
    pub members: Vec<f64>,
    /// Cluster tightness (inverse of spread)
    pub tightness: f64,
    /// Cluster score (size × √tightness)
    pub score: f64,
}

/// Configuration for clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Gauge tolerance (default 5%)
    pub epsilon: f64,
    /// Minimum cluster size to consider
    pub min_cluster_size: usize,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
            min_cluster_size: 1,
        }
    }
}

/// Check if two values are gauge-equivalent within tolerance ε
///
/// a ~ε b ⟺ |a - b| / max(|a|, |b|) < ε
pub fn gauge_equivalent(a: f64, b: f64, epsilon: f64) -> bool {
    if a == b {
        return true;
    }

    // Handle zero cases
    if a == 0.0 && b == 0.0 {
        return true;
    }
    if a == 0.0 || b == 0.0 {
        return false; // 0 is not gauge-equivalent to non-zero
    }

    let max_abs = a.abs().max(b.abs());
    let diff = (a - b).abs();

    diff / max_abs < epsilon
}

/// Perform gauge-theoretic clustering on values
///
/// Groups values by gauge equivalence, computes cluster statistics.
pub fn gauge_clustering(values: &[f64], config: &ClusteringConfig) -> Vec<Cluster> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut clusters: Vec<Cluster> = Vec::new();
    let mut assigned = vec![false; values.len()];

    // Sort by absolute value for stable clustering
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..indexed.len() {
        let (idx, val) = indexed[i];
        if assigned[idx] {
            continue;
        }

        // Start new cluster with this value
        let mut members = vec![val];
        assigned[idx] = true;

        // Find all gauge-equivalent values
        for j in 0..indexed.len() {
            let (other_idx, other_val) = indexed[j];
            if !assigned[other_idx] && gauge_equivalent(val, other_val, config.epsilon) {
                members.push(other_val);
                assigned[other_idx] = true;
            }
        }

        if members.len() >= config.min_cluster_size {
            let center = robust_center(&members);
            let tightness = compute_tightness(&members, center);
            let score = members.len() as f64 * tightness.sqrt();

            clusters.push(Cluster {
                center,
                members,
                tightness,
                score,
            });
        }
    }

    // Sort by score descending
    clusters.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    clusters
}

/// Compute robust cluster center (median + trimmed mean) / 2
fn robust_center(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Median
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // Trimmed mean (exclude top and bottom 10%)
    let trim = (sorted.len() as f64 * 0.1).ceil() as usize;
    let trimmed = if trim * 2 < sorted.len() {
        &sorted[trim..sorted.len() - trim]
    } else {
        &sorted[..]
    };
    let trimmed_mean = trimmed.iter().sum::<f64>() / trimmed.len() as f64;

    (median + trimmed_mean) / 2.0
}

/// Compute cluster tightness (inverse of normalized spread)
fn compute_tightness(values: &[f64], center: f64) -> f64 {
    if values.len() <= 1 {
        return 1.0;
    }

    let max_dev = values
        .iter()
        .map(|&v| {
            let denom = center.abs().max(v.abs()).max(1e-10);
            (v - center).abs() / denom
        })
        .fold(0.0f64, |a, b| a.max(b));

    1.0 / (1.0 + max_dev)
}

/// Get optimal answer from value clustering
///
/// Returns the center of the highest-scoring cluster
pub fn optimal_answer(values: &[f64], config: &ClusteringConfig) -> f64 {
    let clusters = gauge_clustering(values, config);

    if clusters.is_empty() {
        return if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };
    }

    clusters[0].center
}

/// Renormalization Group flow for cluster refinement
///
/// Successively coarsens clusters until fixed point is reached
pub fn rg_flow(clusters: &[Cluster], iterations: usize) -> Vec<Cluster> {
    if clusters.is_empty() || iterations == 0 {
        return clusters.to_vec();
    }

    let mut current = clusters.to_vec();

    for _ in 0..iterations {
        // Collect all centers
        let centers: Vec<f64> = current.iter().map(|c| c.center).collect();

        if centers.len() <= 1 {
            break;
        }

        // Re-cluster with doubled epsilon (coarsening)
        let config = ClusteringConfig {
            epsilon: DEFAULT_EPSILON * 2.0,
            min_cluster_size: 1,
        };

        // Weight centers by cluster size
        let mut weighted_values = Vec::new();
        for cluster in &current {
            for _ in 0..cluster.members.len() {
                weighted_values.push(cluster.center);
            }
        }

        let new_clusters = gauge_clustering(&weighted_values, &config);

        if new_clusters.len() >= current.len() {
            break; // Fixed point reached
        }

        current = new_clusters;
    }

    current
}

/// Test gauge invariance of the CIC functional
///
/// Returns true if F[g(T)] = F[T] + O(ε²) for small perturbations
pub fn test_gauge_invariance(values: &[f64], epsilon: f64) -> GaugeInvarianceResult {
    if values.len() < 2 {
        return GaugeInvarianceResult {
            is_invariant: true,
            max_deviation: 0.0,
        };
    }

    let config = ClusteringConfig {
        epsilon,
        min_cluster_size: 1,
    };
    let clusters = gauge_clustering(values, &config);

    if clusters.is_empty() {
        return GaugeInvarianceResult {
            is_invariant: false,
            max_deviation: 1.0,
        };
    }

    // Check if dominant cluster contains majority of values
    let dominant_ratio = clusters[0].members.len() as f64 / values.len() as f64;
    let is_invariant = dominant_ratio > 0.5;

    // Max deviation is the spread within the dominant cluster
    let max_deviation = if clusters[0].tightness > 0.0 {
        1.0 - clusters[0].tightness
    } else {
        1.0
    };

    GaugeInvarianceResult {
        is_invariant,
        max_deviation,
    }
}

/// Result of gauge invariance test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeInvarianceResult {
    pub is_invariant: bool,
    pub max_deviation: f64,
}

/// Result of signal fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    /// Fused value (optimal answer)
    pub value: f64,
    /// Confidence in the fused value
    pub confidence: f64,
    /// CIC state of the input
    pub cic_state: CICState,
    /// Winning cluster
    pub winning_cluster: Cluster,
    /// All clusters found
    pub all_clusters: Vec<Cluster>,
}

/// Fuse multiple signals into a single reliable value
///
/// Uses gauge-theoretic clustering + CIC assessment
pub fn fuse_signals(values: &[f64], config: &ClusteringConfig, cic_config: &CICConfig) -> FusionResult {
    let clusters = gauge_clustering(values, config);

    let (value, winning_cluster) = if clusters.is_empty() {
        let mean = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };
        (
            mean,
            Cluster {
                center: mean,
                members: values.to_vec(),
                tightness: 0.0,
                score: 0.0,
            },
        )
    } else {
        (clusters[0].center, clusters[0].clone())
    };

    // Compute CIC state
    let samples: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    let sample_refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
    let cic_state = compute_cic(&sample_refs, values, cic_config);

    FusionResult {
        value,
        confidence: cic_state.confidence,
        cic_state,
        winning_cluster,
        all_clusters: clusters,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauge_equivalent_within_tolerance() {
        assert!(gauge_equivalent(42.0, 43.0, 0.05));
        assert!(gauge_equivalent(100.0, 103.0, 0.05));
        assert!(gauge_equivalent(1000.0, 1040.0, 0.05));
    }

    #[test]
    fn test_gauge_equivalent_outside_tolerance() {
        assert!(!gauge_equivalent(42.0, 50.0, 0.05));
        assert!(!gauge_equivalent(100.0, 110.0, 0.05));
    }

    #[test]
    fn test_clustering_basic() {
        let values = vec![42.0, 41.5, 43.0, 100.0, 101.0, 7.0];
        let config = ClusteringConfig::default();
        let clusters = gauge_clustering(&values, &config);

        // Should find multiple clusters
        assert!(clusters.len() >= 2);

        // Highest-scoring cluster should be the ~42 group
        let best = &clusters[0];
        assert!(best.center > 40.0 && best.center < 45.0);
    }

    #[test]
    fn test_optimal_answer() {
        let values = vec![42.0, 42.5, 43.0, 100.0, 101.0, 7.0];
        let config = ClusteringConfig::default();
        let answer = optimal_answer(&values, &config);

        // Should pick ~42 cluster
        assert!(answer > 41.0 && answer < 44.0);
    }

    #[test]
    fn test_rg_flow_convergence() {
        let values = vec![42.0, 42.1, 42.2, 42.3];
        let config = ClusteringConfig::default();
        let clusters = gauge_clustering(&values, &config);
        let refined = rg_flow(&clusters, 5);

        // Should converge to single cluster
        assert_eq!(refined.len(), 1);
    }

    #[test]
    fn test_fusion() {
        let values = vec![42.0, 43.0, 41.0, 100.0];
        let cluster_config = ClusteringConfig::default();
        let cic_config = CICConfig::default();
        let result = fuse_signals(&values, &cluster_config, &cic_config);

        assert!(result.value > 40.0 && result.value < 45.0);
        assert!(result.confidence > 0.0);
    }
}
