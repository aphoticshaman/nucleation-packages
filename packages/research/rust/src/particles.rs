//! McKean-Vlasov particle dynamics for Great Attractor simulations.
//!
//! Implements interacting particle systems with mean-field interactions.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Particle state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub pos: [f64; 2],
    pub vel: [f64; 2],
}

/// Particle swarm state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Swarm {
    pub particles: Vec<Particle>,
    pub time: f64,
}

/// Swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub n_particles: usize,
    pub dt: f64,
    pub diffusion: f64,
    pub interaction_strength: f64,
    pub attractor_pos: [f64; 2],
    pub attractor_strength: f64,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        SwarmConfig {
            n_particles: 100,
            dt: 0.01,
            diffusion: 0.1,
            interaction_strength: 1.0,
            attractor_pos: [0.0, 0.0],
            attractor_strength: 0.5,
        }
    }
}

/// Swarm metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetrics {
    pub mean_pos: [f64; 2],
    pub variance: f64,
    pub precision: f64,
    pub order_parameter: f64,
}

impl Swarm {
    /// Create new swarm with random initialization
    pub fn new(config: &SwarmConfig, seed: u64) -> Self {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let particles: Vec<Particle> = (0..config.n_particles)
            .map(|_| Particle {
                pos: [
                    normal.sample(&mut rng),
                    normal.sample(&mut rng),
                ],
                vel: [0.0, 0.0],
            })
            .collect();

        Swarm {
            particles,
            time: 0.0,
        }
    }

    /// Get swarm centroid
    pub fn centroid(&self) -> [f64; 2] {
        let n = self.particles.len() as f64;
        if n == 0.0 {
            return [0.0, 0.0];
        }

        let sum_x: f64 = self.particles.iter().map(|p| p.pos[0]).sum();
        let sum_y: f64 = self.particles.iter().map(|p| p.pos[1]).sum();

        [sum_x / n, sum_y / n]
    }

    /// Compute swarm metrics
    pub fn metrics(&self, attractor_pos: &[f64; 2]) -> SwarmMetrics {
        let n = self.particles.len() as f64;
        if n == 0.0 {
            return SwarmMetrics {
                mean_pos: [0.0, 0.0],
                variance: 0.0,
                precision: 0.0,
                order_parameter: 0.0,
            };
        }

        let mean_pos = self.centroid();

        // Variance
        let variance: f64 = self.particles.iter()
            .map(|p| {
                let dx = p.pos[0] - mean_pos[0];
                let dy = p.pos[1] - mean_pos[1];
                dx * dx + dy * dy
            })
            .sum::<f64>() / n;

        // Precision (inverse variance)
        let precision = if variance > 1e-10 { 1.0 / variance } else { 1e10 };

        // Order parameter (alignment with attractor)
        let to_attractor = [
            attractor_pos[0] - mean_pos[0],
            attractor_pos[1] - mean_pos[1],
        ];
        let dist = (to_attractor[0].powi(2) + to_attractor[1].powi(2)).sqrt();
        let order_parameter = 1.0 / (1.0 + dist);

        SwarmMetrics {
            mean_pos,
            variance,
            precision,
            order_parameter,
        }
    }

    /// Euler-Maruyama step with McKean-Vlasov interactions
    pub fn step<R: Rng>(&mut self, config: &SwarmConfig, rng: &mut R) {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = config.dt.sqrt();
        let centroid = self.centroid();

        for particle in &mut self.particles {
            // Attractor force
            let to_attractor = [
                config.attractor_pos[0] - particle.pos[0],
                config.attractor_pos[1] - particle.pos[1],
            ];

            // Mean-field interaction (toward centroid)
            let to_centroid = [
                centroid[0] - particle.pos[0],
                centroid[1] - particle.pos[1],
            ];

            // Drift
            let drift = [
                config.attractor_strength * to_attractor[0]
                    + config.interaction_strength * to_centroid[0],
                config.attractor_strength * to_attractor[1]
                    + config.interaction_strength * to_centroid[1],
            ];

            // Diffusion
            let noise = [
                config.diffusion * sqrt_dt * normal.sample(rng),
                config.diffusion * sqrt_dt * normal.sample(rng),
            ];

            // Update position
            particle.pos[0] += drift[0] * config.dt + noise[0];
            particle.pos[1] += drift[1] * config.dt + noise[1];
        }

        self.time += config.dt;
    }

    /// Run simulation for multiple steps
    pub fn run(&mut self, config: &SwarmConfig, n_steps: usize, seed: u64) -> Vec<SwarmMetrics> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut history = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            self.step(config, &mut rng);
            history.push(self.metrics(&config.attractor_pos));
        }

        history
    }

    /// Get particle positions as flat array
    pub fn positions_flat(&self) -> Vec<f64> {
        self.particles
            .iter()
            .flat_map(|p| vec![p.pos[0], p.pos[1]])
            .collect()
    }
}

/// Compute pairwise interaction kernel (Gaussian)
pub fn interaction_kernel(r: f64, sigma: f64) -> f64 {
    (-r * r / (2.0 * sigma * sigma)).exp()
}

/// Compute mean-field force on a particle
pub fn mean_field_force(
    pos: &[f64; 2],
    all_positions: &[[f64; 2]],
    interaction_strength: f64,
    kernel_sigma: f64,
) -> [f64; 2] {
    let mut force = [0.0, 0.0];

    for other in all_positions {
        let dx = other[0] - pos[0];
        let dy = other[1] - pos[1];
        let r = (dx * dx + dy * dy).sqrt();

        if r > 1e-10 {
            let w = interaction_kernel(r, kernel_sigma);
            force[0] += w * dx / r * interaction_strength;
            force[1] += w * dy / r * interaction_strength;
        }
    }

    let n = all_positions.len() as f64;
    if n > 0.0 {
        force[0] /= n;
        force[1] /= n;
    }

    force
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_creation() {
        let config = SwarmConfig::default();
        let swarm = Swarm::new(&config, 42);

        assert_eq!(swarm.particles.len(), config.n_particles);
    }

    #[test]
    fn test_swarm_centroid() {
        let mut swarm = Swarm {
            particles: vec![
                Particle { pos: [0.0, 0.0], vel: [0.0, 0.0] },
                Particle { pos: [2.0, 0.0], vel: [0.0, 0.0] },
                Particle { pos: [1.0, 3.0], vel: [0.0, 0.0] },
            ],
            time: 0.0,
        };

        let centroid = swarm.centroid();
        assert!((centroid[0] - 1.0).abs() < 1e-10);
        assert!((centroid[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_swarm_converges_to_attractor() {
        let config = SwarmConfig {
            n_particles: 50,
            dt: 0.1,
            diffusion: 0.01,
            interaction_strength: 0.5,
            attractor_pos: [5.0, 5.0],
            attractor_strength: 1.0,
        };

        let mut swarm = Swarm::new(&config, 42);
        let history = swarm.run(&config, 500, 42);

        // Should converge toward attractor
        let final_pos = history.last().unwrap().mean_pos;
        let dist = ((final_pos[0] - 5.0).powi(2) + (final_pos[1] - 5.0).powi(2)).sqrt();

        assert!(dist < 2.0); // Should be close to attractor
    }

    #[test]
    fn test_interaction_kernel() {
        let k0 = interaction_kernel(0.0, 1.0);
        assert!((k0 - 1.0).abs() < 1e-10);

        let k1 = interaction_kernel(1.0, 1.0);
        assert!(k1 < k0);
        assert!(k1 > 0.0);
    }
}
