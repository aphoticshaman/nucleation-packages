//! Reliability-Weighted Dempster-Shafer Fusion
//!
//! Based on US6944566B2 (Lockheed Martin, expired April 2023).
//!
//! Implements modified Dempster-Shafer theory for multi-source fusion
//! where sources have varying reliability/quality metrics.
//!
//! Key innovation: Explicit noise belief term, SNR-dependent reliability,
//! additive fusion to avoid catastrophic down-weighting.

use serde::{Deserialize, Serialize};

/// Configuration for reliability mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityConfig {
    /// Logistic steepness
    pub alpha: f64,
    /// Logistic midpoint (quality threshold)
    pub beta: f64,
    /// Floor to prevent zero mass
    pub min_reliability: f64,
    /// Ceiling
    pub max_reliability: f64,
}

impl Default for ReliabilityConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            beta: 0.5,
            min_reliability: 0.01,
            max_reliability: 0.99,
        }
    }
}

/// Fusion method to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionMethod {
    /// Reliability-weighted sum (better for weak sources)
    Additive,
    /// Reliability-weighted product (better for strong consensus)
    Multiplicative,
}

/// Result of belief fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedBelief {
    /// Hypothesis names
    pub hypotheses: Vec<String>,
    /// Fused probability distribution
    pub probabilities: Vec<f64>,
    /// Reliabilities per source
    pub reliabilities: Vec<f64>,
    /// Fusion method used
    pub method: FusionMethod,
    /// Confidence in the fusion (0-1)
    pub confidence: f64,
    /// Noise mass (uncertainty not assigned to hypotheses)
    pub noise_mass: f64,
}

/// Map source quality metric to reliability using logistic function
///
/// From US6944566B2: "Define a reliability function r_j(t) = f(q_j(t))"
pub fn logistic_reliability(quality: f64, config: &ReliabilityConfig) -> f64 {
    let r = 1.0 / (1.0 + (-config.alpha * (quality - config.beta)).exp());
    r.clamp(config.min_reliability, config.max_reliability)
}

/// Compute basic probability assignment (BPA) including noise mass
///
/// From US6944566B2:
/// m_j(H_k; t) = r_j(t) * p_j(H_k | x_j(t))
/// m_j(noise; t) = 1 - r_j(t)
pub fn compute_mass_function(probabilities: &[f64], reliability: f64) -> (Vec<f64>, f64) {
    let masses: Vec<f64> = probabilities.iter().map(|&p| reliability * p).collect();
    let noise_mass = 1.0 - reliability;
    (masses, noise_mass)
}

/// Reliability-weighted additive fusion
///
/// From US6944566B2:
/// R_fused(H_k; t) = Σ_j w_j(t) * R_j(H_k; t)
/// where w_j(t) = r_j(t) / Σ_l r_l(t)
///
/// This avoids catastrophic behavior when many sources have low reliability.
pub fn additive_fusion(
    source_probs: &[Vec<f64>],
    reliabilities: &[f64],
    hypothesis_names: Vec<String>,
) -> FusedBelief {
    let j = source_probs.len();
    if j == 0 || source_probs[0].is_empty() {
        return FusedBelief {
            hypotheses: hypothesis_names,
            probabilities: vec![],
            reliabilities: reliabilities.to_vec(),
            method: FusionMethod::Additive,
            confidence: 0.0,
            noise_mass: 1.0,
        };
    }

    let k = source_probs[0].len();
    let r_sum: f64 = reliabilities.iter().sum();

    // Degenerate case: all sources have zero reliability
    if r_sum < 1e-10 {
        let uniform = vec![1.0 / k as f64; k];
        return FusedBelief {
            hypotheses: hypothesis_names,
            probabilities: uniform,
            reliabilities: reliabilities.to_vec(),
            method: FusionMethod::Additive,
            confidence: 0.0,
            noise_mass: 1.0,
        };
    }

    // Weights proportional to reliability
    let weights: Vec<f64> = reliabilities.iter().map(|r| r / r_sum).collect();

    // Compute reliability-weighted masses per hypothesis
    let mut r_fused = vec![0.0; k];
    for (source_idx, probs) in source_probs.iter().enumerate() {
        let (masses, _) = compute_mass_function(probs, reliabilities[source_idx]);
        for (h_idx, &mass) in masses.iter().enumerate() {
            r_fused[h_idx] += weights[source_idx] * mass;
        }
    }

    // Normalize to probabilities
    let total: f64 = r_fused.iter().sum();
    let p_fused: Vec<f64> = if total > 0.0 {
        r_fused.iter().map(|&r| r / total).collect()
    } else {
        vec![1.0 / k as f64; k]
    };

    // Confidence: how concentrated is the belief? (1 - normalized entropy)
    let max_entropy = (k as f64).ln();
    let entropy: f64 = -p_fused
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f64>();
    let confidence = if max_entropy > 0.0 {
        1.0 - entropy / max_entropy
    } else {
        1.0
    };

    // Average noise mass
    let noise_mass = reliabilities.iter().map(|&r| 1.0 - r).sum::<f64>() / j as f64;

    FusedBelief {
        hypotheses: hypothesis_names,
        probabilities: p_fused,
        reliabilities: reliabilities.to_vec(),
        method: FusionMethod::Additive,
        confidence,
        noise_mass,
    }
}

/// Reliability-weighted multiplicative fusion
///
/// From US6909997B2:
/// S_mult(H_k; t) = Π_j p_j(H_k | x_j(t))^{r_j(t)}
///
/// Uses exponentiation by reliability to down-weight unreliable sources.
pub fn multiplicative_fusion(
    source_probs: &[Vec<f64>],
    reliabilities: &[f64],
    hypothesis_names: Vec<String>,
) -> FusedBelief {
    let j = source_probs.len();
    if j == 0 || source_probs[0].is_empty() {
        return FusedBelief {
            hypotheses: hypothesis_names,
            probabilities: vec![],
            reliabilities: reliabilities.to_vec(),
            method: FusionMethod::Multiplicative,
            confidence: 0.0,
            noise_mass: 1.0,
        };
    }

    let k = source_probs[0].len();

    // Log-space computation for numerical stability
    let mut log_s = vec![0.0; k];
    for (source_idx, probs) in source_probs.iter().enumerate() {
        let r = reliabilities[source_idx];
        for (h_idx, &p) in probs.iter().enumerate() {
            let p_clipped = p.max(1e-12);
            log_s[h_idx] += r * p_clipped.ln();
        }
    }

    // Normalize in log space
    let max_log = log_s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let s: Vec<f64> = log_s.iter().map(|&ls| (ls - max_log).exp()).collect();
    let total: f64 = s.iter().sum();
    let p_fused: Vec<f64> = s.iter().map(|&si| si / total).collect();

    // Confidence
    let max_entropy = (k as f64).ln();
    let entropy: f64 = -p_fused
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f64>();
    let confidence = if max_entropy > 0.0 {
        1.0 - entropy / max_entropy
    } else {
        1.0
    };

    let noise_mass = reliabilities.iter().map(|&r| 1.0 - r).sum::<f64>() / j as f64;

    FusedBelief {
        hypotheses: hypothesis_names,
        probabilities: p_fused,
        reliabilities: reliabilities.to_vec(),
        method: FusionMethod::Multiplicative,
        confidence,
        noise_mass,
    }
}

/// Meta-fusion: dynamically select best method based on performance
pub struct MetaFusionSelector {
    window_size: usize,
    additive_history: Vec<f64>,
    multiplicative_history: Vec<f64>,
}

impl MetaFusionSelector {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            additive_history: Vec::new(),
            multiplicative_history: Vec::new(),
        }
    }

    /// Fuse and optionally learn from realized outcome
    pub fn fuse(
        &mut self,
        source_probs: &[Vec<f64>],
        reliabilities: &[f64],
        hypothesis_names: Vec<String>,
        realized_outcome: Option<usize>,
    ) -> FusedBelief {
        let add_belief =
            additive_fusion(source_probs, reliabilities, hypothesis_names.clone());
        let mult_belief =
            multiplicative_fusion(source_probs, reliabilities, hypothesis_names);

        // Score based on recent performance
        let add_score = self.score_method(&self.additive_history, add_belief.confidence);
        let mult_score = self.score_method(&self.multiplicative_history, mult_belief.confidence);

        // Update performance history if we have feedback
        if let Some(outcome) = realized_outcome {
            if outcome < add_belief.probabilities.len() {
                let add_log_prob = (add_belief.probabilities.get(outcome).unwrap_or(&1e-10) + 1e-10).ln();
                let mult_log_prob = (mult_belief.probabilities.get(outcome).unwrap_or(&1e-10) + 1e-10).ln();

                self.additive_history.push(add_log_prob);
                self.multiplicative_history.push(mult_log_prob);

                // Keep window size
                if self.additive_history.len() > self.window_size {
                    self.additive_history.remove(0);
                }
                if self.multiplicative_history.len() > self.window_size {
                    self.multiplicative_history.remove(0);
                }
            }
        }

        // Return the best method's result
        if mult_score > add_score {
            mult_belief
        } else {
            add_belief
        }
    }

    fn score_method(&self, history: &[f64], current_confidence: f64) -> f64 {
        if history.len() >= self.window_size {
            let historical: f64 =
                history.iter().rev().take(self.window_size).sum::<f64>() / self.window_size as f64;
            0.7 * historical + 0.3 * current_confidence
        } else {
            current_confidence
        }
    }

    /// Get attractor dominance signal
    ///
    /// When multiplicative outperforms additive significantly,
    /// it indicates strong consensus (single attractor dominating).
    pub fn attractor_dominance_signal(&self) -> f64 {
        if self.additive_history.len() < 5 || self.multiplicative_history.len() < 5 {
            return 0.0;
        }

        let add_recent: f64 =
            self.additive_history.iter().rev().take(10).sum::<f64>() / 10.0f64.min(self.additive_history.len() as f64);
        let mult_recent: f64 =
            self.multiplicative_history.iter().rev().take(10).sum::<f64>() / 10.0f64.min(self.multiplicative_history.len() as f64);

        (mult_recent - add_recent).tanh()
    }
}

/// Fuse multiple sources with auto-selected method
pub fn auto_fuse(
    source_probs: &[Vec<f64>],
    qualities: &[f64],
    hypothesis_names: Vec<String>,
    config: &ReliabilityConfig,
) -> FusedBelief {
    // Convert qualities to reliabilities
    let reliabilities: Vec<f64> = qualities
        .iter()
        .map(|&q| logistic_reliability(q, config))
        .collect();

    // Use additive for low average reliability, multiplicative for high
    let avg_reliability: f64 = reliabilities.iter().sum::<f64>() / reliabilities.len().max(1) as f64;

    if avg_reliability < 0.5 {
        additive_fusion(source_probs, &reliabilities, hypothesis_names)
    } else {
        multiplicative_fusion(source_probs, &reliabilities, hypothesis_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_reliability() {
        let config = ReliabilityConfig::default();

        // Quality at midpoint should give ~0.5 reliability
        let r_mid = logistic_reliability(0.5, &config);
        assert!((r_mid - 0.5).abs() < 0.1);

        // High quality should give high reliability
        let r_high = logistic_reliability(0.9, &config);
        assert!(r_high > 0.7);

        // Low quality should give low reliability
        let r_low = logistic_reliability(0.1, &config);
        assert!(r_low < 0.3);
    }

    #[test]
    fn test_additive_fusion() {
        let source1 = vec![0.8, 0.1, 0.1]; // Confident in H0
        let source2 = vec![0.7, 0.2, 0.1]; // Also confident in H0
        let source3 = vec![0.3, 0.4, 0.3]; // Uncertain

        let sources = vec![source1, source2, source3];
        let reliabilities = vec![0.9, 0.8, 0.3];
        let names = vec!["H0".to_string(), "H1".to_string(), "H2".to_string()];

        let result = additive_fusion(&sources, &reliabilities, names);

        // Should favor H0
        assert!(result.probabilities[0] > result.probabilities[1]);
        assert!(result.probabilities[0] > result.probabilities[2]);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_multiplicative_fusion() {
        let source1 = vec![0.9, 0.05, 0.05];
        let source2 = vec![0.85, 0.1, 0.05];

        let sources = vec![source1, source2];
        let reliabilities = vec![0.9, 0.9];
        let names = vec!["H0".to_string(), "H1".to_string(), "H2".to_string()];

        let result = multiplicative_fusion(&sources, &reliabilities, names);

        // Strong consensus should give high confidence
        assert!(result.probabilities[0] > 0.8);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_auto_fuse_selects_method() {
        let sources = vec![vec![0.5, 0.3, 0.2], vec![0.4, 0.4, 0.2]];
        let config = ReliabilityConfig::default();

        // Low quality -> additive
        let low_q = vec![0.2, 0.3];
        let result_low = auto_fuse(&sources, &low_q, vec!["A".into(), "B".into(), "C".into()], &config);
        assert_eq!(result_low.method, FusionMethod::Additive);

        // High quality -> multiplicative
        let high_q = vec![0.9, 0.85];
        let result_high = auto_fuse(&sources, &high_q, vec!["A".into(), "B".into(), "C".into()], &config);
        assert_eq!(result_high.method, FusionMethod::Multiplicative);
    }
}
