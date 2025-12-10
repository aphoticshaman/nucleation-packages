//! CIC (Compression-Integration-Coherence) Framework
//!
//! Implements the CIC functional: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
//!
//! This is the core of LatticeForge's signal reliability assessment:
//! - Φ: Integrated information via compression distance
//! - H: Conditional entropy (uncertainty)
//! - C_multi: Multi-scale coherence across clusters
//!
//! Parameters:
//! - λ = 0.5 (entropy penalty weight)
//! - γ = 0.3 (coherence bonus weight)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default CIC parameters (empirically validated)
pub const DEFAULT_LAMBDA: f64 = 0.5;
pub const DEFAULT_GAMMA: f64 = 0.3;
pub const DEFAULT_EPSILON: f64 = 0.05;

/// CIC state containing all computed components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CICState {
    /// Integrated information (compression coherence)
    pub phi: f64,
    /// Conditional entropy (uncertainty)
    pub entropy: f64,
    /// Multi-scale coherence
    pub coherence: f64,
    /// Final CIC functional value
    pub functional: f64,
    /// Calibrated confidence [0.05, 0.95]
    pub confidence: f64,
}

/// Configuration for CIC computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CICConfig {
    pub lambda: f64,
    pub gamma: f64,
}

impl Default for CICConfig {
    fn default() -> Self {
        Self {
            lambda: DEFAULT_LAMBDA,
            gamma: DEFAULT_GAMMA,
        }
    }
}

/// Compute Normalized Compression Distance between two byte sequences
///
/// NCD(x,y) = (C(x|y) - min(C(x), C(y))) / max(C(x), C(y))
///
/// We approximate C(x) using a simple run-length encoding estimator
/// for speed. For production, consider using zstd or lz4.
pub fn ncd(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 0.0;
    }

    let ca = compressed_size(a);
    let cb = compressed_size(b);

    // Concatenate for joint compression
    let mut ab = Vec::with_capacity(a.len() + b.len());
    ab.extend_from_slice(a);
    ab.extend_from_slice(b);
    let cab = compressed_size(&ab);

    let min_c = ca.min(cb) as f64;
    let max_c = ca.max(cb) as f64;

    if max_c == 0.0 {
        return 0.0;
    }

    // NCD formula
    ((cab as f64) - min_c) / max_c
}

/// Compute NCD for string inputs
pub fn ncd_str(a: &str, b: &str) -> f64 {
    ncd(a.as_bytes(), b.as_bytes())
}

/// Simple compression size estimator using run-length encoding
/// For production, replace with zstd::compress
fn compressed_size(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }

    // Simple RLE-based estimator
    let mut size = 0;
    let mut run_length = 1;
    let mut prev = data[0];

    for &byte in &data[1..] {
        if byte == prev {
            run_length += 1;
        } else {
            // Cost: 1 byte for char + variable for count
            size += 1 + (run_length.min(255) as f64).log2().ceil() as usize;
            run_length = 1;
            prev = byte;
        }
    }
    size += 1 + (run_length.min(255) as f64).log2().ceil() as usize;

    // Add entropy-based adjustment
    let entropy = byte_entropy(data);
    let adjusted = (size as f64 * (0.5 + entropy * 0.5)) as usize;

    adjusted.max(1)
}

/// Compute byte-level entropy
fn byte_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    // Normalize to [0, 1]
    entropy / 8.0
}

/// Compute integrated information Φ from string samples
///
/// Φ measures how much the samples "hang together" via compression
/// High Φ = samples share common structure
/// Low Φ = samples are independent/random
pub fn compute_phi(samples: &[&str]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    if samples.len() == 1 {
        return 1.0;
    }

    // Compute mean pairwise NCD
    let mut total_ncd = 0.0;
    let mut count = 0;

    for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            total_ncd += ncd_str(samples[i], samples[j]);
            count += 1;
        }
    }

    if count == 0 {
        return 1.0;
    }

    let mean_ncd = total_ncd / count as f64;

    // Φ = 1 - mean_ncd (invert so higher = more coherent)
    1.0 - mean_ncd
}

/// Compute entropy of numeric values
///
/// Uses histogram-based estimation with adaptive binning
pub fn compute_entropy(values: &[f64]) -> f64 {
    if values.is_empty() || values.len() == 1 {
        return 0.0;
    }

    // Check if all values are identical
    let first = values[0];
    if values.iter().all(|&v| (v - first).abs() < 1e-10) {
        return 0.0;
    }

    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range < 1e-10 {
        return 0.0;
    }

    // Adaptive bin count (Sturges' rule)
    let n_bins = ((values.len() as f64).log2().ceil() as usize + 1).max(2).min(50);
    let bin_width = range / n_bins as f64;

    let mut bins = vec![0u32; n_bins];
    for &v in values {
        let bin = ((v - min_val) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        bins[bin] += 1;
    }

    // Compute entropy
    let n = values.len() as f64;
    let mut entropy = 0.0;
    for &count in &bins {
        if count > 0 {
            let p = count as f64 / n;
            entropy -= p * p.log2();
        }
    }

    // Normalize by max possible entropy
    let max_entropy = (n_bins as f64).log2();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

/// Compute multi-scale coherence from numeric values
///
/// Measures how well values cluster at multiple scales
pub fn compute_coherence(values: &[f64], epsilon: f64) -> f64 {
    if values.len() < 2 {
        return 1.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

    if variance < 1e-10 {
        return 1.0; // Perfect coherence
    }

    let std_dev = variance.sqrt();
    let cv = std_dev / mean.abs().max(1e-10); // Coefficient of variation

    // Count values within epsilon of each other
    let mut within_epsilon = 0;
    let total_pairs = values.len() * (values.len() - 1) / 2;

    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            let max_val = values[i].abs().max(values[j].abs()).max(1e-10);
            let rel_diff = (values[i] - values[j]).abs() / max_val;
            if rel_diff < epsilon {
                within_epsilon += 1;
            }
        }
    }

    let cluster_ratio = within_epsilon as f64 / total_pairs.max(1) as f64;

    // Combine CV and cluster ratio
    let cv_score = 1.0 / (1.0 + cv);
    (cv_score + cluster_ratio) / 2.0
}

/// Compute the full CIC functional
///
/// F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
pub fn compute_cic(samples: &[&str], values: &[f64], config: &CICConfig) -> CICState {
    let phi = compute_phi(samples);
    let entropy = compute_entropy(values);
    let coherence = compute_coherence(values, DEFAULT_EPSILON);

    // CIC functional
    let functional = phi - config.lambda * entropy + config.gamma * coherence;

    // Calibrated confidence (bounded to [0.05, 0.95])
    let raw_confidence = 0.5 + 0.5 * functional.tanh();
    let confidence = raw_confidence.clamp(0.05, 0.95);

    CICState {
        phi,
        entropy,
        coherence,
        functional,
        confidence,
    }
}

/// Detect crystallization (phase transition to stable state)
///
/// Crystallization occurs when: dΦ/dt ≈ λ·dH/dt
/// This indicates the system has found a stable attractor
pub fn detect_crystallization(history: &[CICState], lambda: f64) -> bool {
    if history.len() < 3 {
        return false;
    }

    // Get recent states
    let recent: Vec<_> = history.iter().rev().take(5).collect();
    if recent.len() < 3 {
        return false;
    }

    // Check if entropy is decreasing and phi is increasing
    let entropy_decreasing = recent.windows(2).all(|w| w[0].entropy <= w[1].entropy + 0.05);
    let phi_stable = recent.windows(2).all(|w| (w[0].phi - w[1].phi).abs() < 0.1);

    // Check crystallization condition
    if recent.len() >= 2 {
        let d_phi = recent[0].phi - recent[1].phi;
        let d_entropy = recent[0].entropy - recent[1].entropy;
        let condition = (d_phi - lambda * d_entropy).abs() < 0.1;

        return entropy_decreasing && (phi_stable || condition);
    }

    false
}

/// Phase of the CIC system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CICPhase {
    /// High disorder, no structure
    Plasma,
    /// Structure forming but unstable
    Nucleating,
    /// Good agreement, moderate disorder
    Supercooled,
    /// Strong consensus, low disorder
    Crystalline,
    /// Transitional state
    Annealing,
}

impl CICPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            CICPhase::Plasma => "plasma",
            CICPhase::Nucleating => "nucleating",
            CICPhase::Supercooled => "supercooled",
            CICPhase::Crystalline => "crystalline",
            CICPhase::Annealing => "annealing",
        }
    }
}

/// Determine phase from CIC state
pub fn determine_phase(state: &CICState) -> CICPhase {
    if state.coherence > 0.8 && state.entropy < 0.3 {
        CICPhase::Crystalline
    } else if state.coherence > 0.6 && state.entropy < 0.5 {
        CICPhase::Supercooled
    } else if state.phi > 0.5 && state.coherence > 0.4 {
        CICPhase::Nucleating
    } else if state.entropy > 0.7 {
        CICPhase::Plasma
    } else {
        CICPhase::Annealing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ncd_identical() {
        assert_eq!(ncd_str("hello", "hello"), 0.0);
    }

    #[test]
    fn test_ncd_different() {
        let d = ncd_str("aaaaaaa", "zzzzzzz");
        assert!(d > 0.3, "Expected NCD > 0.3 for different strings, got {}", d);
    }

    #[test]
    fn test_ncd_similar() {
        let d = ncd_str("hello world", "hello there");
        assert!(d > 0.0 && d < 1.0);
    }

    #[test]
    fn test_phi_coherent() {
        let samples = vec!["42", "42", "42", "43"];
        let phi = compute_phi(&samples);
        assert!(phi > 0.5, "Expected high phi for coherent samples");
    }

    #[test]
    fn test_entropy_uniform() {
        let values = vec![42.0, 42.0, 42.0, 42.0];
        let entropy = compute_entropy(&values);
        assert_eq!(entropy, 0.0, "Expected zero entropy for uniform values");
    }

    #[test]
    fn test_entropy_spread() {
        let values = vec![1.0, 10.0, 100.0, 1000.0];
        let entropy = compute_entropy(&values);
        assert!(entropy > 0.5, "Expected high entropy for spread values");
    }

    #[test]
    fn test_coherence_tight() {
        let values = vec![42.0, 42.1, 42.2, 41.9];
        let coherence = compute_coherence(&values, 0.05);
        assert!(coherence > 0.7, "Expected high coherence for tight cluster");
    }

    #[test]
    fn test_cic_functional() {
        let samples = vec!["42", "43", "41", "42"];
        let values = vec![42.0, 43.0, 41.0, 42.0];
        let config = CICConfig::default();
        let state = compute_cic(&samples, &values, &config);

        assert!(state.phi > 0.0);
        assert!(state.confidence >= 0.05 && state.confidence <= 0.95);
    }

    #[test]
    fn test_phase_detection() {
        let crystalline = CICState {
            phi: 0.9,
            entropy: 0.1,
            coherence: 0.9,
            functional: 0.9,
            confidence: 0.9,
        };
        assert_eq!(determine_phase(&crystalline), CICPhase::Crystalline);

        let plasma = CICState {
            phi: 0.2,
            entropy: 0.9,
            coherence: 0.2,
            functional: 0.1,
            confidence: 0.3,
        };
        assert_eq!(determine_phase(&plasma), CICPhase::Plasma);
    }
}
