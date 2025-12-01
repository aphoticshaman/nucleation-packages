//! Geodesic computation on Riemannian manifolds.
//!
//! Implements:
//! - Christoffel symbol computation
//! - RK4 geodesic integration
//! - Fisher information metric

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

/// 2D point on manifold
pub type Point2D = [f64; 2];

/// 2x2 metric tensor
pub type Metric2x2 = [[f64; 2]; 2];

/// Geodesic trajectory result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeodesicTrajectory {
    pub points: Vec<Point2D>,
    pub velocities: Vec<Point2D>,
    pub times: Vec<f64>,
}

/// Compute Christoffel symbols numerically.
///
/// Γ^k_{ij} = 0.5 * g^{kl} * (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
pub fn christoffel_symbols<F>(x: Point2D, metric_fn: &F, eps: f64) -> Array3<f64>
where
    F: Fn(Point2D) -> Metric2x2,
{
    let mut gamma = Array3::zeros((2, 2, 2));

    let g = metric_fn(x);
    let g_inv = invert_2x2(g);

    // Compute partial derivatives of metric
    let mut dg = [[[0.0; 2]; 2]; 2]; // dg[l][i][j] = ∂_l g_{ij}

    for l in 0..2 {
        let mut x_plus = x;
        let mut x_minus = x;
        x_plus[l] += eps;
        x_minus[l] -= eps;

        let g_plus = metric_fn(x_plus);
        let g_minus = metric_fn(x_minus);

        for i in 0..2 {
            for j in 0..2 {
                dg[l][i][j] = (g_plus[i][j] - g_minus[i][j]) / (2.0 * eps);
            }
        }
    }

    // Compute Christoffel symbols
    for k in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for l in 0..2 {
                    sum += g_inv[k][l] * (dg[i][j][l] + dg[j][i][l] - dg[l][i][j]);
                }
                gamma[[k, i, j]] = 0.5 * sum;
            }
        }
    }

    gamma
}

/// Invert a 2x2 matrix.
fn invert_2x2(m: Metric2x2) -> Metric2x2 {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-15 {
        return [[1.0, 0.0], [0.0, 1.0]]; // Return identity for singular matrix
    }
    let inv_det = 1.0 / det;
    [
        [m[1][1] * inv_det, -m[0][1] * inv_det],
        [-m[1][0] * inv_det, m[0][0] * inv_det],
    ]
}

/// Geodesic equation right-hand side.
///
/// dθ/dt = v
/// dv/dt = -Γ^k_{ij} v^i v^j
fn geodesic_rhs<F>(x: Point2D, v: Point2D, metric_fn: &F, eps: f64) -> (Point2D, Point2D)
where
    F: Fn(Point2D) -> Metric2x2,
{
    let gamma = christoffel_symbols(x, metric_fn, eps);

    let mut dv = [0.0, 0.0];
    for k in 0..2 {
        let mut sum = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                sum += gamma[[k, i, j]] * v[i] * v[j];
            }
        }
        dv[k] = -sum;
    }

    (v, dv)
}

/// RK4 step for geodesic integration.
fn rk4_step<F>(
    x: Point2D,
    v: Point2D,
    metric_fn: &F,
    dt: f64,
    eps: f64,
) -> (Point2D, Point2D)
where
    F: Fn(Point2D) -> Metric2x2,
{
    // k1
    let (dx1, dv1) = geodesic_rhs(x, v, metric_fn, eps);

    // k2
    let x2 = [x[0] + 0.5 * dt * dx1[0], x[1] + 0.5 * dt * dx1[1]];
    let v2 = [v[0] + 0.5 * dt * dv1[0], v[1] + 0.5 * dt * dv1[1]];
    let (dx2, dv2) = geodesic_rhs(x2, v2, metric_fn, eps);

    // k3
    let x3 = [x[0] + 0.5 * dt * dx2[0], x[1] + 0.5 * dt * dx2[1]];
    let v3 = [v[0] + 0.5 * dt * dv2[0], v[1] + 0.5 * dt * dv2[1]];
    let (dx3, dv3) = geodesic_rhs(x3, v3, metric_fn, eps);

    // k4
    let x4 = [x[0] + dt * dx3[0], x[1] + dt * dx3[1]];
    let v4 = [v[0] + dt * dv3[0], v[1] + dt * dv3[1]];
    let (dx4, dv4) = geodesic_rhs(x4, v4, metric_fn, eps);

    // Combine
    let x_new = [
        x[0] + dt * (dx1[0] + 2.0 * dx2[0] + 2.0 * dx3[0] + dx4[0]) / 6.0,
        x[1] + dt * (dx1[1] + 2.0 * dx2[1] + 2.0 * dx3[1] + dx4[1]) / 6.0,
    ];
    let v_new = [
        v[0] + dt * (dv1[0] + 2.0 * dv2[0] + 2.0 * dv3[0] + dv4[0]) / 6.0,
        v[1] + dt * (dv1[1] + 2.0 * dv2[1] + 2.0 * dv3[1] + dv4[1]) / 6.0,
    ];

    (x_new, v_new)
}

/// Integrate geodesic from initial point and velocity.
pub fn integrate_geodesic<F>(
    x0: Point2D,
    v0: Point2D,
    metric_fn: F,
    dt: f64,
    n_steps: usize,
    eps: f64,
) -> GeodesicTrajectory
where
    F: Fn(Point2D) -> Metric2x2,
{
    let mut points = Vec::with_capacity(n_steps + 1);
    let mut velocities = Vec::with_capacity(n_steps + 1);
    let mut times = Vec::with_capacity(n_steps + 1);

    let mut x = x0;
    let mut v = v0;

    points.push(x);
    velocities.push(v);
    times.push(0.0);

    for i in 0..n_steps {
        let (x_new, v_new) = rk4_step(x, v, &metric_fn, dt, eps);
        x = x_new;
        v = v_new;

        points.push(x);
        velocities.push(v);
        times.push((i + 1) as f64 * dt);
    }

    GeodesicTrajectory {
        points,
        velocities,
        times,
    }
}

/// Fisher information metric for Gaussian with parameters (μ, σ).
pub fn fisher_metric_gaussian(params: Point2D) -> Metric2x2 {
    let sigma = params[1].max(1e-6);
    let sigma_sq = sigma * sigma;

    [
        [1.0 / sigma_sq, 0.0],
        [0.0, 2.0 / sigma_sq],
    ]
}

/// Compute geodesic distance between two points (approximate).
pub fn geodesic_distance<F>(
    x1: Point2D,
    x2: Point2D,
    metric_fn: &F,
) -> f64
where
    F: Fn(Point2D) -> Metric2x2,
{
    // Simple approximation: integrate metric along straight line
    let n_samples = 100;
    let mut total = 0.0;

    for i in 0..n_samples {
        let t = i as f64 / n_samples as f64;
        let x = [
            x1[0] + t * (x2[0] - x1[0]),
            x1[1] + t * (x2[1] - x1[1]),
        ];
        let dx = [
            (x2[0] - x1[0]) / n_samples as f64,
            (x2[1] - x1[1]) / n_samples as f64,
        ];

        let g = metric_fn(x);
        let ds_sq = g[0][0] * dx[0] * dx[0]
            + 2.0 * g[0][1] * dx[0] * dx[1]
            + g[1][1] * dx[1] * dx[1];

        total += ds_sq.max(0.0).sqrt();
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn euclidean_metric(_x: Point2D) -> Metric2x2 {
        [[1.0, 0.0], [0.0, 1.0]]
    }

    #[test]
    fn test_christoffel_flat_space() {
        let gamma = christoffel_symbols([0.0, 0.0], &euclidean_metric, 1e-4);
        // Flat space should have zero Christoffel symbols
        for k in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!(gamma[[k, i, j]].abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_geodesic_flat_space() {
        let traj = integrate_geodesic(
            [0.0, 0.0],
            [1.0, 0.5],
            euclidean_metric,
            0.1,
            100,
            1e-4,
        );

        // In flat space, geodesics are straight lines
        let final_point = traj.points.last().unwrap();
        assert!((final_point[0] - 10.0).abs() < 0.1);
        assert!((final_point[1] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_fisher_metric_positive_definite() {
        let g = fisher_metric_gaussian([0.0, 1.0]);
        let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
        assert!(det > 0.0);
        assert!(g[0][0] > 0.0);
    }
}
