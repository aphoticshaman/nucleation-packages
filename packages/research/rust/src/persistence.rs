//! Persistent homology computation for topological data analysis.
//!
//! Implements simplified Vietoris-Rips persistence for 2D point clouds.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Persistence pair (birth, death)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PersistencePair {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
}

impl PersistencePair {
    pub fn lifetime(&self) -> f64 {
        self.death - self.birth
    }
}

/// Persistence diagram result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pub h0: Vec<PersistencePair>,
    pub h1: Vec<PersistencePair>,
}

/// Edge in the Rips complex
#[derive(Debug, Clone, Copy)]
struct Edge {
    i: usize,
    j: usize,
    weight: f64,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl Eq for Edge {}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.weight.partial_cmp(&self.weight).unwrap_or(Ordering::Equal)
    }
}

/// Union-Find structure for H0 computation
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let px = self.find(x);
        let py = self.find(y);
        if px == py {
            return false;
        }
        if self.rank[px] < self.rank[py] {
            self.parent[px] = py;
        } else if self.rank[px] > self.rank[py] {
            self.parent[py] = px;
        } else {
            self.parent[py] = px;
            self.rank[px] += 1;
        }
        true
    }
}

/// Compute pairwise distance matrix
pub fn distance_matrix(points: &[[f64; 2]]) -> Array2<f64> {
    let n = points.len();
    let mut d = Array2::zeros((n, n));

    for i in 0..n {
        for j in i + 1..n {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            let dist = (dx * dx + dy * dy).sqrt();
            d[[i, j]] = dist;
            d[[j, i]] = dist;
        }
    }

    d
}

/// Compute H0 persistence (connected components) using Union-Find
fn compute_h0(points: &[[f64; 2]], max_edge: f64) -> Vec<PersistencePair> {
    let n = points.len();
    if n < 2 {
        return vec![];
    }

    // Build sorted edge list
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            let dist = (dx * dx + dy * dy).sqrt();
            if dist <= max_edge {
                edges.push(Edge { i, j, weight: dist });
            }
        }
    }
    edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());

    // Union-Find to track component deaths
    let mut uf = UnionFind::new(n);
    let mut deaths = vec![None; n];

    for edge in &edges {
        let pi = uf.find(edge.i);
        let pj = uf.find(edge.j);
        if pi != pj {
            // One component dies
            let dying = pi.max(pj);
            deaths[dying] = Some(edge.weight);
            uf.union(edge.i, edge.j);
        }
    }

    // Build persistence pairs (all born at 0)
    let mut pairs = Vec::new();
    for i in 0..n {
        if uf.parent[i] == i {
            if let Some(death) = deaths[i] {
                if death > 0.0 {
                    pairs.push(PersistencePair {
                        birth: 0.0,
                        death,
                        dimension: 0,
                    });
                }
            }
        }
    }

    pairs
}

/// Compute H1 persistence (loops) - simplified heuristic
fn compute_h1(points: &[[f64; 2]], max_edge: f64) -> Vec<PersistencePair> {
    let n = points.len();
    if n < 3 {
        return vec![];
    }

    let mut pairs = Vec::new();

    // Check triangles for loop detection
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let d_ij = distance(&points[i], &points[j]);
                let d_jk = distance(&points[j], &points[k]);
                let d_ik = distance(&points[i], &points[k]);

                let mut edges = [d_ij, d_jk, d_ik];
                edges.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let max = edges[2];
                if max <= max_edge {
                    let birth = edges[1]; // Second edge creates loop
                    let death = max;      // Third edge fills it

                    if death > birth && (death - birth) > 0.01 * max_edge {
                        pairs.push(PersistencePair {
                            birth,
                            death,
                            dimension: 1,
                        });
                    }
                }
            }
        }
    }

    // Deduplicate by keeping longest-lived features
    pairs.sort_by(|a, b| b.lifetime().partial_cmp(&a.lifetime()).unwrap());
    pairs.truncate(n / 3); // Keep top features

    pairs
}

fn distance(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    (dx * dx + dy * dy).sqrt()
}

/// Compute persistence diagram for 2D point cloud
pub fn compute_persistence(points: &[[f64; 2]], max_edge: f64) -> PersistenceDiagram {
    let h0 = compute_h0(points, max_edge);
    let h1 = compute_h1(points, max_edge);

    PersistenceDiagram { h0, h1 }
}

/// Compute persistent entropy
pub fn persistent_entropy(pairs: &[PersistencePair]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }

    let lifetimes: Vec<f64> = pairs.iter().map(|p| p.lifetime().max(1e-12)).collect();
    let total: f64 = lifetimes.iter().sum();

    if total < 1e-12 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for lt in &lifetimes {
        let p = lt / total;
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Total persistence (sum of lifetimes)
pub fn total_persistence(pairs: &[PersistencePair]) -> f64 {
    pairs.iter().map(|p| p.lifetime()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix() {
        let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let d = distance_matrix(&points);

        assert!((d[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((d[[0, 2]] - 1.0).abs() < 1e-10);
        assert!((d[[1, 2]] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_h0_two_clusters() {
        // Two well-separated clusters
        let mut points = Vec::new();
        for _ in 0..10 {
            points.push([0.0, 0.0]);
        }
        for _ in 0..10 {
            points.push([10.0, 0.0]);
        }

        let pd = compute_persistence(&points, 15.0);

        // Should have at least one long-lived H0 feature
        assert!(!pd.h0.is_empty());
    }

    #[test]
    fn test_persistent_entropy() {
        let pairs = vec![
            PersistencePair { birth: 0.0, death: 1.0, dimension: 0 },
            PersistencePair { birth: 0.0, death: 1.0, dimension: 0 },
        ];
        let entropy = persistent_entropy(&pairs);

        // Equal lifetimes -> max entropy for 2 features = ln(2)
        assert!((entropy - 2.0_f64.ln()).abs() < 0.01);
    }
}
