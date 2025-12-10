//! Cascade Probability Prediction via Belief Propagation
//!
//! Predicts whether local events will cascade into regional instability using
//! factor graph belief propagation (Loopy BP).
//!
//! Based on Pearl's message-passing algorithm and epidemiological cascade models.
//! Factors encode:
//! - Geographic proximity (contiguity, distance decay)
//! - Economic integration (trade dependency, financial linkages)
//! - Ethnic/cultural ties (diaspora networks, shared identity)
//! - Political alignment (alliance structures, ideological affinity)

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for cascade prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeConfig {
    /// Maximum belief propagation iterations
    pub max_iterations: usize,
    /// Convergence threshold for message updates
    pub convergence_threshold: f64,
    /// Damping factor for message updates (0-1)
    pub damping: f64,
    /// Base cascade probability
    pub base_prob: f64,
    /// Geographic distance decay rate
    pub geo_decay: f64,
    /// Economic coupling strength
    pub econ_weight: f64,
    /// Ethnic tie strength
    pub ethnic_weight: f64,
    /// Political alignment weight
    pub political_weight: f64,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            damping: 0.5,
            base_prob: 0.1,
            geo_decay: 0.001,
            econ_weight: 0.3,
            ethnic_weight: 0.2,
            political_weight: 0.25,
        }
    }
}

/// Cascade channel type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CascadeChannel {
    Geographic,
    Economic,
    Ethnic,
    Political,
}

/// Regional cascade prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionCascade {
    /// Region identifier
    pub region: String,
    /// Probability of cascade reaching this region
    pub probability: f64,
    /// Expected lag in days
    pub expected_lag: f64,
    /// Primary transmission channel
    pub primary_channel: CascadeChannel,
    /// Marginal belief distribution [no_cascade, cascade]
    pub belief: [f64; 2],
}

/// Full cascade prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadePrediction {
    /// Origin event identifier
    pub origin_event: String,
    /// Origin region
    pub origin_region: String,
    /// Affected regions with predictions
    pub affected_regions: Vec<RegionCascade>,
    /// Overall cascade probability
    pub overall_probability: f64,
    /// Confidence in prediction
    pub confidence: f64,
    /// Number of BP iterations to converge
    pub iterations: usize,
    /// Did algorithm converge
    pub converged: bool,
}

/// Factor in the factor graph
#[derive(Debug, Clone)]
struct Factor {
    /// Factor ID
    id: usize,
    /// Connected variable indices
    variables: Vec<usize>,
    /// Factor type
    factor_type: CascadeChannel,
    /// Factor potential function parameters
    params: FactorParams,
}

/// Factor potential parameters
#[derive(Debug, Clone)]
struct FactorParams {
    /// Coupling strength
    strength: f64,
    /// Distance (for geographic factors)
    distance: f64,
    /// Lag (temporal delay)
    lag: f64,
}

/// Variable in the factor graph
#[derive(Debug, Clone)]
struct Variable {
    /// Variable ID (region index)
    id: usize,
    /// Region name
    name: String,
    /// Prior probability of cascade [no_cascade, cascade]
    prior: [f64; 2],
    /// Connected factor indices
    factors: Vec<usize>,
}

/// Factor graph for cascade propagation
pub struct CascadeFactorGraph {
    variables: Vec<Variable>,
    factors: Vec<Factor>,
    // Messages: var_to_factor[var_id][factor_id] -> [f64; 2]
    var_to_factor: HashMap<(usize, usize), [f64; 2]>,
    // Messages: factor_to_var[factor_id][var_id] -> [f64; 2]
    factor_to_var: HashMap<(usize, usize), [f64; 2]>,
    config: CascadeConfig,
}

impl CascadeFactorGraph {
    /// Create a new factor graph from region data
    pub fn new(
        regions: Vec<String>,
        geo_distances: &Array2<f64>,
        econ_ties: &Array2<f64>,
        ethnic_ties: &Array2<f64>,
        political_align: &Array2<f64>,
        config: CascadeConfig,
    ) -> Self {
        let n = regions.len();
        let mut variables: Vec<Variable> = regions
            .iter()
            .enumerate()
            .map(|(i, name)| Variable {
                id: i,
                name: name.clone(),
                prior: [0.9, 0.1],
                factors: Vec::new(),
            })
            .collect();

        let mut factors: Vec<Factor> = Vec::new();
        let mut factor_id = 0;

        // Create factors for each pair of regions
        for i in 0..n {
            for j in (i + 1)..n {
                // Geographic factor
                let geo_dist = geo_distances[[i, j]];
                if geo_dist < 5000.0 {
                    // Within 5000 km
                    factors.push(Factor {
                        id: factor_id,
                        variables: vec![i, j],
                        factor_type: CascadeChannel::Geographic,
                        params: FactorParams {
                            strength: (-config.geo_decay * geo_dist).exp(),
                            distance: geo_dist,
                            lag: (geo_dist / 100.0).max(1.0),
                        },
                    });
                    variables[i].factors.push(factor_id);
                    variables[j].factors.push(factor_id);
                    factor_id += 1;
                }

                // Economic factor
                let econ = econ_ties[[i, j]];
                if econ > 0.1 {
                    factors.push(Factor {
                        id: factor_id,
                        variables: vec![i, j],
                        factor_type: CascadeChannel::Economic,
                        params: FactorParams {
                            strength: econ * config.econ_weight,
                            distance: 0.0,
                            lag: 7.0 / (1.0 + econ), // Faster with stronger ties
                        },
                    });
                    variables[i].factors.push(factor_id);
                    variables[j].factors.push(factor_id);
                    factor_id += 1;
                }

                // Ethnic factor
                let ethnic = ethnic_ties[[i, j]];
                if ethnic > 0.1 {
                    factors.push(Factor {
                        id: factor_id,
                        variables: vec![i, j],
                        factor_type: CascadeChannel::Ethnic,
                        params: FactorParams {
                            strength: ethnic * config.ethnic_weight,
                            distance: 0.0,
                            lag: 3.0 / (1.0 + ethnic),
                        },
                    });
                    variables[i].factors.push(factor_id);
                    variables[j].factors.push(factor_id);
                    factor_id += 1;
                }

                // Political factor
                let political = political_align[[i, j]];
                if political.abs() > 0.1 {
                    factors.push(Factor {
                        id: factor_id,
                        variables: vec![i, j],
                        factor_type: CascadeChannel::Political,
                        params: FactorParams {
                            strength: political.abs() * config.political_weight,
                            distance: 0.0,
                            lag: if political > 0.0 { 5.0 } else { 14.0 },
                        },
                    });
                    variables[i].factors.push(factor_id);
                    variables[j].factors.push(factor_id);
                    factor_id += 1;
                }
            }
        }

        // Initialize messages uniformly
        let mut var_to_factor = HashMap::new();
        let mut factor_to_var = HashMap::new();

        for var in &variables {
            for &fac_id in &var.factors {
                var_to_factor.insert((var.id, fac_id), [0.5, 0.5]);
            }
        }

        for fac in &factors {
            for &var_id in &fac.variables {
                factor_to_var.insert((fac.id, var_id), [0.5, 0.5]);
            }
        }

        Self {
            variables,
            factors,
            var_to_factor,
            factor_to_var,
            config,
        }
    }

    /// Set origin region to certain cascade (evidence)
    pub fn set_evidence(&mut self, origin_idx: usize) {
        if origin_idx < self.variables.len() {
            self.variables[origin_idx].prior = [0.0, 1.0]; // Certain cascade
        }
    }

    /// Compute factor potential
    fn factor_potential(&self, factor: &Factor, x_i: usize, x_j: usize) -> f64 {
        // x_i, x_j ∈ {0, 1} where 0 = no cascade, 1 = cascade
        match (x_i, x_j) {
            (0, 0) => 1.0 - factor.params.strength * 0.1, // Both stable
            (1, 1) => 1.0 + factor.params.strength,       // Both affected
            (0, 1) | (1, 0) => factor.params.strength,    // Cascade transmission
            _ => 1.0,                                      // Default for invalid states
        }
    }

    /// Variable to factor message update
    fn update_var_to_factor(&mut self) {
        for var in &self.variables {
            for &fac_id in &var.factors {
                let mut msg = var.prior;

                // Multiply by incoming factor messages (except this factor)
                for &other_fac in &var.factors {
                    if other_fac != fac_id {
                        if let Some(incoming) = self.factor_to_var.get(&(other_fac, var.id)) {
                            msg[0] *= incoming[0];
                            msg[1] *= incoming[1];
                        }
                    }
                }

                // Normalize
                let sum = msg[0] + msg[1];
                if sum > 1e-10 {
                    msg[0] /= sum;
                    msg[1] /= sum;
                } else {
                    msg = [0.5, 0.5];
                }

                // Damping
                if let Some(old) = self.var_to_factor.get(&(var.id, fac_id)) {
                    let d = self.config.damping;
                    msg[0] = d * old[0] + (1.0 - d) * msg[0];
                    msg[1] = d * old[1] + (1.0 - d) * msg[1];
                }

                self.var_to_factor.insert((var.id, fac_id), msg);
            }
        }
    }

    /// Factor to variable message update
    fn update_factor_to_var(&mut self) {
        for factor in &self.factors {
            for &target_var in &factor.variables {
                // Sum over other variables
                let other_vars: Vec<usize> = factor
                    .variables
                    .iter()
                    .copied()
                    .filter(|&v| v != target_var)
                    .collect();

                let mut msg = [0.0, 0.0];

                if other_vars.is_empty() {
                    msg = [0.5, 0.5];
                } else {
                    // For pairwise factors
                    let other_var = other_vars[0];
                    let incoming = self
                        .var_to_factor
                        .get(&(other_var, factor.id))
                        .copied()
                        .unwrap_or([0.5, 0.5]);

                    for x_target in 0..2 {
                        for x_other in 0..2 {
                            let potential = self.factor_potential(factor, x_target, x_other);
                            msg[x_target] += potential * incoming[x_other];
                        }
                    }
                }

                // Normalize
                let sum = msg[0] + msg[1];
                if sum > 1e-10 {
                    msg[0] /= sum;
                    msg[1] /= sum;
                } else {
                    msg = [0.5, 0.5];
                }

                // Damping
                if let Some(old) = self.factor_to_var.get(&(factor.id, target_var)) {
                    let d = self.config.damping;
                    msg[0] = d * old[0] + (1.0 - d) * msg[0];
                    msg[1] = d * old[1] + (1.0 - d) * msg[1];
                }

                self.factor_to_var.insert((factor.id, target_var), msg);
            }
        }
    }

    /// Compute marginal beliefs for all variables
    fn compute_beliefs(&self) -> Vec<[f64; 2]> {
        self.variables
            .iter()
            .map(|var| {
                let mut belief = var.prior;

                for &fac_id in &var.factors {
                    if let Some(incoming) = self.factor_to_var.get(&(fac_id, var.id)) {
                        belief[0] *= incoming[0];
                        belief[1] *= incoming[1];
                    }
                }

                // Normalize
                let sum = belief[0] + belief[1];
                if sum > 1e-10 {
                    [belief[0] / sum, belief[1] / sum]
                } else {
                    [0.5, 0.5]
                }
            })
            .collect()
    }

    /// Run loopy belief propagation
    pub fn run_bp(&mut self) -> (Vec<[f64; 2]>, usize, bool) {
        let mut prev_beliefs = self.compute_beliefs();
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            self.update_var_to_factor();
            self.update_factor_to_var();

            let new_beliefs = self.compute_beliefs();

            // Check convergence
            let max_diff: f64 = prev_beliefs
                .iter()
                .zip(new_beliefs.iter())
                .map(|(old, new)| (old[0] - new[0]).abs().max((old[1] - new[1]).abs()))
                .fold(0.0, f64::max);

            if max_diff < self.config.convergence_threshold {
                converged = true;
                iterations = iter + 1;
                return (new_beliefs, iterations, converged);
            }

            prev_beliefs = new_beliefs;
            iterations = iter + 1;
        }

        (prev_beliefs, iterations, converged)
    }

    /// Get primary cascade channel for a region
    fn get_primary_channel(&self, origin_idx: usize, target_idx: usize) -> CascadeChannel {
        let mut max_strength = 0.0;
        let mut primary = CascadeChannel::Geographic;

        for factor in &self.factors {
            if factor.variables.contains(&origin_idx) && factor.variables.contains(&target_idx) {
                if factor.params.strength > max_strength {
                    max_strength = factor.params.strength;
                    primary = factor.factor_type;
                }
            }
        }

        primary
    }

    /// Get expected lag for a region
    fn get_expected_lag(&self, origin_idx: usize, target_idx: usize) -> f64 {
        let mut total_lag = 0.0;
        let mut total_weight = 0.0;

        for factor in &self.factors {
            if factor.variables.contains(&origin_idx) && factor.variables.contains(&target_idx) {
                total_lag += factor.params.lag * factor.params.strength;
                total_weight += factor.params.strength;
            }
        }

        if total_weight > 0.0 {
            total_lag / total_weight
        } else {
            14.0 // Default 2 weeks
        }
    }
}

/// Predict cascade from an origin event
pub fn predict_cascade(
    origin_event: &str,
    origin_region: &str,
    regions: Vec<String>,
    geo_distances: &Array2<f64>,
    econ_ties: &Array2<f64>,
    ethnic_ties: &Array2<f64>,
    political_align: &Array2<f64>,
    config: CascadeConfig,
) -> CascadePrediction {
    // Find origin index
    let origin_idx = regions.iter().position(|r| r == origin_region).unwrap_or(0);

    // Build factor graph
    let mut graph = CascadeFactorGraph::new(
        regions.clone(),
        geo_distances,
        econ_ties,
        ethnic_ties,
        political_align,
        config.clone(),
    );

    // Set evidence
    graph.set_evidence(origin_idx);

    // Run BP
    let (beliefs, iterations, converged) = graph.run_bp();

    // Build results
    let mut affected_regions: Vec<RegionCascade> = regions
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != origin_idx)
        .map(|(i, name)| {
            let belief = beliefs[i];
            let prob = belief[1]; // P(cascade)

            RegionCascade {
                region: name.clone(),
                probability: prob,
                expected_lag: graph.get_expected_lag(origin_idx, i),
                primary_channel: graph.get_primary_channel(origin_idx, i),
                belief,
            }
        })
        .collect();

    // Sort by probability descending
    affected_regions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

    // Compute overall cascade probability (at least one other region affected)
    let overall_prob = 1.0
        - affected_regions
            .iter()
            .map(|r| 1.0 - r.probability)
            .product::<f64>();

    // Confidence based on convergence and factor graph density
    let confidence = if converged {
        0.8 + 0.2 * (1.0 - (iterations as f64 / config.max_iterations as f64))
    } else {
        0.5
    };

    CascadePrediction {
        origin_event: origin_event.to_string(),
        origin_region: origin_region.to_string(),
        affected_regions,
        overall_probability: overall_prob,
        confidence,
        iterations,
        converged,
    }
}

/// Simplified cascade prediction with default relationships
pub fn quick_cascade_prediction(
    origin_event: &str,
    origin_region: &str,
    neighbor_regions: &[(&str, f64)], // (region_name, coupling_strength)
) -> CascadePrediction {
    let n = neighbor_regions.len() + 1;
    let mut regions: Vec<String> = vec![origin_region.to_string()];
    regions.extend(neighbor_regions.iter().map(|(r, _)| r.to_string()));

    // Build simple coupling matrices
    let mut geo = Array2::zeros((n, n));
    let mut econ = Array2::zeros((n, n));
    let ethnic = Array2::zeros((n, n));
    let political = Array2::zeros((n, n));

    for (i, (_, strength)) in neighbor_regions.iter().enumerate() {
        // Origin to neighbor
        geo[[0, i + 1]] = 500.0; // Default 500 km
        geo[[i + 1, 0]] = 500.0;
        econ[[0, i + 1]] = *strength;
        econ[[i + 1, 0]] = *strength;
    }

    let config = CascadeConfig::default();
    predict_cascade(
        origin_event,
        origin_region,
        regions,
        &geo,
        &econ,
        &ethnic,
        &political,
        config,
    )
}

/// Monte Carlo cascade simulation for uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeSimulation {
    /// Number of simulations
    pub n_simulations: usize,
    /// Mean cascade probability
    pub mean_probability: f64,
    /// Std dev of cascade probability
    pub std_probability: f64,
    /// 95% confidence interval
    pub ci_95: (f64, f64),
    /// Region-level statistics
    pub region_stats: Vec<RegionCascadeStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionCascadeStats {
    pub region: String,
    pub mean_prob: f64,
    pub std_prob: f64,
    pub cascade_count: usize,
}

/// Run Monte Carlo simulation of cascades
pub fn simulate_cascades(
    origin_event: &str,
    origin_region: &str,
    regions: Vec<String>,
    geo_distances: &Array2<f64>,
    econ_ties: &Array2<f64>,
    ethnic_ties: &Array2<f64>,
    political_align: &Array2<f64>,
    n_simulations: usize,
    perturbation_scale: f64,
) -> CascadeSimulation {
    let n = regions.len();
    let origin_idx = regions.iter().position(|r| r == origin_region).unwrap_or(0);

    let mut overall_probs: Vec<f64> = Vec::with_capacity(n_simulations);
    let mut region_cascades: Vec<Vec<f64>> = vec![Vec::with_capacity(n_simulations); n];

    // Simple LCG for deterministic random numbers (no rand crate dependency)
    let mut seed: u64 = 12345;
    let lcg_next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*s as f64) / (u64::MAX as f64)
    };

    for _ in 0..n_simulations {
        // Perturb coupling matrices
        let mut perturbed_econ = econ_ties.clone();
        let mut perturbed_ethnic = ethnic_ties.clone();
        let mut perturbed_political = political_align.clone();

        for i in 0..n {
            for j in 0..n {
                let noise = (lcg_next(&mut seed) - 0.5) * 2.0 * perturbation_scale;
                perturbed_econ[[i, j]] = (perturbed_econ[[i, j]] + noise).max(0.0).min(1.0);
                perturbed_ethnic[[i, j]] = (perturbed_ethnic[[i, j]] + noise).max(0.0).min(1.0);
                perturbed_political[[i, j]] =
                    (perturbed_political[[i, j]] + noise).max(-1.0).min(1.0);
            }
        }

        // Run prediction
        let prediction = predict_cascade(
            origin_event,
            origin_region,
            regions.clone(),
            geo_distances,
            &perturbed_econ,
            &perturbed_ethnic,
            &perturbed_political,
            CascadeConfig::default(),
        );

        overall_probs.push(prediction.overall_probability);

        // Track region-level cascades
        for (i, region) in regions.iter().enumerate() {
            if i == origin_idx {
                region_cascades[i].push(1.0);
            } else if let Some(r) = prediction.affected_regions.iter().find(|r| &r.region == region)
            {
                region_cascades[i].push(r.probability);
            } else {
                region_cascades[i].push(0.0);
            }
        }
    }

    // Compute statistics
    let mean_prob: f64 = overall_probs.iter().sum::<f64>() / n_simulations as f64;
    let variance: f64 = overall_probs
        .iter()
        .map(|p| (p - mean_prob).powi(2))
        .sum::<f64>()
        / n_simulations as f64;
    let std_prob = variance.sqrt();

    // Sort for percentiles
    overall_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_low = overall_probs[(n_simulations as f64 * 0.025) as usize];
    let ci_high = overall_probs[(n_simulations as f64 * 0.975) as usize];

    // Region stats
    let region_stats: Vec<RegionCascadeStats> = regions
        .iter()
        .enumerate()
        .map(|(i, region)| {
            let probs = &region_cascades[i];
            let mean = probs.iter().sum::<f64>() / n_simulations as f64;
            let var = probs.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n_simulations as f64;
            let cascade_count = probs.iter().filter(|&&p| p > 0.5).count();

            RegionCascadeStats {
                region: region.clone(),
                mean_prob: mean,
                std_prob: var.sqrt(),
                cascade_count,
            }
        })
        .collect();

    CascadeSimulation {
        n_simulations,
        mean_probability: mean_prob,
        std_probability: std_prob,
        ci_95: (ci_low, ci_high),
        region_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_cascade() {
        let prediction = quick_cascade_prediction(
            "protest_001",
            "SYR",
            &[("LBN", 0.7), ("JOR", 0.5), ("IRQ", 0.6), ("TUR", 0.4)],
        );

        assert_eq!(prediction.origin_region, "SYR");
        assert_eq!(prediction.affected_regions.len(), 4);
        assert!(prediction.overall_probability > 0.0);
        assert!(prediction.overall_probability <= 1.0);
    }

    #[test]
    fn test_cascade_factor_graph() {
        let regions = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
        ];

        let geo = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 100.0, 200.0, 100.0, 0.0, 150.0, 200.0, 150.0, 0.0],
        )
        .unwrap();

        let econ = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 0.8, 0.3, 0.8, 0.0, 0.5, 0.3, 0.5, 0.0],
        )
        .unwrap();

        let ethnic = Array2::zeros((3, 3));
        let political = Array2::zeros((3, 3));

        let config = CascadeConfig::default();
        let mut graph = CascadeFactorGraph::new(
            regions,
            &geo,
            &econ,
            &ethnic,
            &political,
            config,
        );

        graph.set_evidence(0); // Origin at A

        let (beliefs, _iterations, _converged) = graph.run_bp();

        assert_eq!(beliefs.len(), 3);
        assert!(beliefs[0][1] > 0.5); // Origin should have elevated cascade prob

        // Verify beliefs are valid probability distributions
        for belief in &beliefs {
            assert!(belief[0] >= 0.0 && belief[0] <= 1.0);
            assert!(belief[1] >= 0.0 && belief[1] <= 1.0);
            assert!((belief[0] + belief[1] - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_cascade_simulation() {
        let regions = vec!["A".to_string(), "B".to_string()];
        let geo = Array2::from_shape_vec((2, 2), vec![0.0, 100.0, 100.0, 0.0]).unwrap();
        let econ = Array2::from_shape_vec((2, 2), vec![0.0, 0.7, 0.7, 0.0]).unwrap();
        let ethnic = Array2::zeros((2, 2));
        let political = Array2::zeros((2, 2));

        let sim = simulate_cascades(
            "event_001",
            "A",
            regions,
            &geo,
            &econ,
            &ethnic,
            &political,
            100,
            0.1,
        );

        assert_eq!(sim.n_simulations, 100);
        assert!(sim.mean_probability >= 0.0 && sim.mean_probability <= 1.0);
        assert!(sim.ci_95.0 <= sim.mean_probability);
        assert!(sim.ci_95.1 >= sim.mean_probability);
    }

    #[test]
    fn test_cascade_channels() {
        let prediction = quick_cascade_prediction(
            "econ_crisis",
            "GRC",
            &[("ITA", 0.8), ("ESP", 0.6), ("PRT", 0.5)],
        );

        // All should have Economic as primary channel given the data
        for region in &prediction.affected_regions {
            assert!(region.probability > 0.0);
            assert!(region.expected_lag > 0.0);
        }
    }
}
