//! Regime-switching dynamics for Great Attractor macro-states.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Regime SDE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    pub n_regimes: usize,
    pub beta: f64,           // Inverse temperature
    pub dt: f64,
    pub diffusion: f64,
    pub switching_scale: f64,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        RegimeConfig {
            n_regimes: 3,
            beta: 1.0,
            dt: 0.01,
            diffusion: 0.1,
            switching_scale: 0.5,
        }
    }
}

/// Regime SDE state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeState {
    pub regime: usize,
    pub position: [f64; 2],
    pub time: f64,
}

/// Regime simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeHistory {
    pub times: Vec<f64>,
    pub regimes: Vec<usize>,
    pub positions: Vec<[f64; 2]>,
    pub switch_times: Vec<f64>,
}

/// Compute switching rate using Arrhenius-like formula
pub fn switching_rate(
    from_regime: usize,
    to_regime: usize,
    position: &[f64; 2],
    potentials: &[fn(&[f64; 2]) -> f64],
    config: &RegimeConfig,
) -> f64 {
    if from_regime == to_regime {
        return 0.0;
    }

    let u_from = potentials[from_regime](position);
    let u_to = potentials[to_regime](position);
    let delta_u = u_to - u_from;

    // Only penalize uphill transitions
    let barrier = delta_u.max(0.0);
    config.switching_scale * (-config.beta * barrier).exp()
}

/// Compute gradient of potential numerically
fn potential_gradient(
    position: &[f64; 2],
    potential: fn(&[f64; 2]) -> f64,
    eps: f64,
) -> [f64; 2] {
    let mut grad = [0.0, 0.0];

    for i in 0..2 {
        let mut pos_plus = *position;
        let mut pos_minus = *position;
        pos_plus[i] += eps;
        pos_minus[i] -= eps;

        grad[i] = (potential(&pos_plus) - potential(&pos_minus)) / (2.0 * eps);
    }

    grad
}

/// Simulate regime-switching SDE
pub fn simulate_regime_sde<R: Rng>(
    initial_state: RegimeState,
    config: &RegimeConfig,
    potentials: &[fn(&[f64; 2]) -> f64],
    n_steps: usize,
    rng: &mut R,
) -> RegimeHistory {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let sqrt_dt = config.dt.sqrt();

    let mut times = Vec::with_capacity(n_steps + 1);
    let mut regimes = Vec::with_capacity(n_steps + 1);
    let mut positions = Vec::with_capacity(n_steps + 1);
    let mut switch_times = Vec::new();

    let mut state = initial_state;
    times.push(state.time);
    regimes.push(state.regime);
    positions.push(state.position);

    for _ in 0..n_steps {
        let current_regime = state.regime;

        // 1. Regime switching
        let total_rate: f64 = (0..config.n_regimes)
            .filter(|&r| r != current_regime)
            .map(|r| switching_rate(current_regime, r, &state.position, potentials, config))
            .sum();

        let p_switch = 1.0 - (-total_rate * config.dt).exp();

        if rng.gen::<f64>() < p_switch && total_rate > 1e-12 {
            // Sample destination regime
            let u: f64 = rng.gen::<f64>() * total_rate;
            let mut cumsum = 0.0;

            for r in 0..config.n_regimes {
                if r == current_regime {
                    continue;
                }
                cumsum += switching_rate(current_regime, r, &state.position, potentials, config);
                if u < cumsum {
                    state.regime = r;
                    switch_times.push(state.time);
                    break;
                }
            }
        }

        // 2. Position dynamics (gradient descent + noise)
        let grad = potential_gradient(&state.position, potentials[state.regime], 1e-5);

        state.position[0] -= grad[0] * config.dt
            + config.diffusion * sqrt_dt * normal.sample(rng);
        state.position[1] -= grad[1] * config.dt
            + config.diffusion * sqrt_dt * normal.sample(rng);

        state.time += config.dt;

        times.push(state.time);
        regimes.push(state.regime);
        positions.push(state.position);
    }

    RegimeHistory {
        times,
        regimes,
        positions,
        switch_times,
    }
}

/// Build Q-matrix at a given position
pub fn local_q_matrix(
    position: &[f64; 2],
    potentials: &[fn(&[f64; 2]) -> f64],
    config: &RegimeConfig,
) -> Vec<Vec<f64>> {
    let n = config.n_regimes;
    let mut q = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                q[i][j] = switching_rate(i, j, position, potentials, config);
            }
        }
        // Set diagonal
        let row_sum: f64 = q[i].iter().sum::<f64>() - q[i][i];
        q[i][i] = -row_sum;
    }

    q
}

/// Example potentials for testing
pub mod example_potentials {
    pub fn quadratic_center_0(pos: &[f64; 2]) -> f64 {
        0.5 * (pos[0].powi(2) + pos[1].powi(2))
    }

    pub fn quadratic_center_1(pos: &[f64; 2]) -> f64 {
        let center = [2.0, 0.0];
        0.5 * ((pos[0] - center[0]).powi(2) + (pos[1] - center[1]).powi(2))
    }

    pub fn quadratic_center_2(pos: &[f64; 2]) -> f64 {
        let center = [0.0, 2.0];
        0.5 * ((pos[0] - center[0]).powi(2) + (pos[1] - center[1]).powi(2))
    }

    pub fn double_well(pos: &[f64; 2]) -> f64 {
        0.25 * (pos[0].powi(2) - 1.0).powi(2) + 0.5 * pos[1].powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_switching_rate() {
        let config = RegimeConfig::default();
        let potentials: Vec<fn(&[f64; 2]) -> f64> = vec![
            example_potentials::quadratic_center_0,
            example_potentials::quadratic_center_1,
            example_potentials::quadratic_center_2,
        ];

        let pos = [0.0, 0.0];
        let rate = switching_rate(0, 1, &pos, &potentials, &config);

        assert!(rate > 0.0);
        assert!(rate <= config.switching_scale);
    }

    #[test]
    fn test_simulate_regime_sde() {
        let config = RegimeConfig {
            n_regimes: 2,
            beta: 1.0,
            dt: 0.01,
            diffusion: 0.1,
            switching_scale: 0.5,
        };

        let potentials: Vec<fn(&[f64; 2]) -> f64> = vec![
            example_potentials::quadratic_center_0,
            example_potentials::quadratic_center_1,
        ];

        let initial = RegimeState {
            regime: 0,
            position: [0.0, 0.0],
            time: 0.0,
        };

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let history = simulate_regime_sde(initial, &config, &potentials, 1000, &mut rng);

        assert_eq!(history.times.len(), 1001);
        assert_eq!(history.regimes.len(), 1001);
        assert!(history.regimes.iter().all(|&r| r < 2));
    }

    #[test]
    fn test_local_q_matrix() {
        let config = RegimeConfig::default();
        let potentials: Vec<fn(&[f64; 2]) -> f64> = vec![
            example_potentials::quadratic_center_0,
            example_potentials::quadratic_center_1,
            example_potentials::quadratic_center_2,
        ];

        let pos = [0.0, 0.0];
        let q = local_q_matrix(&pos, &potentials, &config);

        // Check row sums are zero
        for row in &q {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10);
        }
    }
}
