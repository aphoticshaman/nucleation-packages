//! Epistemic Uncertainty and Temporal Confidence Decay
//!
//! Models how confidence in predictions decays over time as uncertainty
//! compounds. Provides rigorous confidence bounds for forecasts.
//!
//! Key concepts:
//! - Predictions have temporal validity windows
//! - Confidence decays exponentially with time
//! - Domain-specific decay rates (politics decays faster than geography)
//! - Event-driven confidence shocks
//! - Aggregation of multiple uncertain estimates

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Domain-specific decay half-lives (in hours)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DecayDomain {
    /// Market prices: very fast decay (6 hour half-life)
    Markets,
    /// Political situations: fast decay (24 hour half-life)
    Political,
    /// Economic indicators: medium decay (72 hour half-life)
    Economic,
    /// Security/military: medium-slow decay (168 hour / 1 week)
    Security,
    /// Infrastructure: slow decay (720 hour / 30 days)
    Infrastructure,
    /// Geographic/demographic: very slow decay (2160 hour / 90 days)
    Geographic,
    /// Custom half-life
    Custom(f64),
}

impl DecayDomain {
    /// Get half-life in hours
    pub fn half_life_hours(&self) -> f64 {
        match self {
            Self::Markets => 6.0,
            Self::Political => 24.0,
            Self::Economic => 72.0,
            Self::Security => 168.0,
            Self::Infrastructure => 720.0,
            Self::Geographic => 2160.0,
            Self::Custom(h) => *h,
        }
    }

    /// Decay rate λ where C(t) = C₀ * e^(-λt)
    pub fn decay_rate(&self) -> f64 {
        (2.0_f64).ln() / self.half_life_hours()
    }
}

/// Confidence interval with epistemic uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Point estimate (mean or median)
    pub estimate: f64,
    /// Lower bound (e.g., 5th percentile)
    pub lower: f64,
    /// Upper bound (e.g., 95th percentile)
    pub upper: f64,
    /// Confidence level (e.g., 0.90 for 90% CI)
    pub level: f64,
}

impl ConfidenceInterval {
    pub fn new(estimate: f64, lower: f64, upper: f64, level: f64) -> Self {
        Self { estimate, lower, upper, level }
    }

    /// Create symmetric interval: estimate ± margin
    pub fn symmetric(estimate: f64, margin: f64, level: f64) -> Self {
        Self {
            estimate,
            lower: estimate - margin,
            upper: estimate + margin,
            level,
        }
    }

    /// Width of the interval
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Is value within bounds?
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Merge two intervals (union)
    pub fn union(&self, other: &ConfidenceInterval) -> ConfidenceInterval {
        ConfidenceInterval {
            estimate: (self.estimate + other.estimate) / 2.0,
            lower: self.lower.min(other.lower),
            upper: self.upper.max(other.upper),
            level: self.level.min(other.level),
        }
    }

    /// Intersection of two intervals (if overlapping)
    pub fn intersection(&self, other: &ConfidenceInterval) -> Option<ConfidenceInterval> {
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        if lower <= upper {
            Some(ConfidenceInterval {
                estimate: (lower + upper) / 2.0,
                lower,
                upper,
                level: self.level.min(other.level),
            })
        } else {
            None
        }
    }
}

/// Time-decaying estimate with full uncertainty tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEstimate {
    /// The estimated value
    pub value: f64,
    /// Initial confidence (0-1)
    pub initial_confidence: f64,
    /// Timestamp when estimate was made (Unix seconds)
    pub created_at: i64,
    /// Decay domain
    pub domain: DecayDomain,
    /// Optional expiry time (hard cutoff)
    pub expires_at: Option<i64>,
    /// Source reliability (0-1)
    pub source_reliability: f64,
    /// Initial standard deviation
    pub initial_std: f64,
}

impl TemporalEstimate {
    pub fn new(value: f64, confidence: f64, created_at: i64, domain: DecayDomain) -> Self {
        Self {
            value,
            initial_confidence: confidence,
            created_at,
            domain,
            expires_at: None,
            source_reliability: 0.8,  // Default reliability
            initial_std: 0.1,  // 10% initial uncertainty
        }
    }

    /// Set expiry time
    pub fn with_expiry(mut self, expires_at: i64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Set source reliability
    pub fn with_reliability(mut self, reliability: f64) -> Self {
        self.source_reliability = reliability.clamp(0.0, 1.0);
        self
    }

    /// Set initial standard deviation
    pub fn with_std(mut self, std: f64) -> Self {
        self.initial_std = std.abs();
        self
    }

    /// Is this estimate expired?
    pub fn is_expired(&self, now: i64) -> bool {
        self.expires_at.map(|e| now > e).unwrap_or(false)
    }

    /// Hours since creation
    pub fn age_hours(&self, now: i64) -> f64 {
        (now - self.created_at) as f64 / 3600.0
    }

    /// Current confidence after decay
    pub fn confidence(&self, now: i64) -> f64 {
        if self.is_expired(now) {
            return 0.0;
        }

        let age = self.age_hours(now);
        let decay = (-self.domain.decay_rate() * age).exp();
        (self.initial_confidence * self.source_reliability * decay).clamp(0.0, 1.0)
    }

    /// Current standard deviation (grows with time)
    pub fn std(&self, now: i64) -> f64 {
        if self.is_expired(now) {
            return f64::INFINITY;
        }

        let age = self.age_hours(now);
        // Std grows as sqrt(t) - random walk uncertainty
        let growth_factor = (1.0 + age / self.domain.half_life_hours()).sqrt();
        self.initial_std * growth_factor
    }

    /// Get confidence interval at given time
    pub fn interval(&self, now: i64, level: f64) -> ConfidenceInterval {
        let conf = self.confidence(now);
        let std = self.std(now);

        // z-score for given confidence level (approximation)
        let z = match level {
            l if l >= 0.99 => 2.576,
            l if l >= 0.95 => 1.96,
            l if l >= 0.90 => 1.645,
            l if l >= 0.80 => 1.282,
            _ => 1.0,
        };

        let margin = z * std / conf.max(0.01);
        ConfidenceInterval::symmetric(self.value, margin, level)
    }
}

/// Aggregator for multiple temporal estimates
#[derive(Debug, Default)]
pub struct EpistemicAggregator {
    estimates: Vec<TemporalEstimate>,
}

impl EpistemicAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an estimate
    pub fn add(&mut self, estimate: TemporalEstimate) {
        self.estimates.push(estimate);
    }

    /// Remove expired estimates
    pub fn prune(&mut self, now: i64) {
        self.estimates.retain(|e| !e.is_expired(now));
    }

    /// Confidence-weighted mean
    pub fn weighted_mean(&self, now: i64) -> Option<f64> {
        if self.estimates.is_empty() {
            return None;
        }

        let mut sum_weighted = 0.0;
        let mut sum_weights = 0.0;

        for est in &self.estimates {
            let weight = est.confidence(now);
            sum_weighted += est.value * weight;
            sum_weights += weight;
        }

        if sum_weights > 0.0 {
            Some(sum_weighted / sum_weights)
        } else {
            None
        }
    }

    /// Aggregate confidence (product rule for independent estimates)
    pub fn aggregate_confidence(&self, now: i64) -> f64 {
        if self.estimates.is_empty() {
            return 0.0;
        }

        // Probability that at least one estimate is correct
        let all_wrong_prob: f64 = self.estimates
            .iter()
            .map(|e| 1.0 - e.confidence(now))
            .product();

        1.0 - all_wrong_prob
    }

    /// Combined confidence interval using weighted bootstrap-like aggregation
    pub fn combined_interval(&self, now: i64, level: f64) -> Option<ConfidenceInterval> {
        if self.estimates.is_empty() {
            return None;
        }

        let weighted_mean = self.weighted_mean(now)?;

        // Weighted variance
        let mut sum_weights = 0.0;
        let mut weighted_var = 0.0;

        for est in &self.estimates {
            let weight = est.confidence(now);
            let std = est.std(now);
            weighted_var += weight * std * std;
            sum_weights += weight;
        }

        if sum_weights <= 0.0 {
            return None;
        }

        let combined_std = (weighted_var / sum_weights).sqrt();

        let z = match level {
            l if l >= 0.99 => 2.576,
            l if l >= 0.95 => 1.96,
            l if l >= 0.90 => 1.645,
            _ => 1.0,
        };

        let margin = z * combined_std;
        Some(ConfidenceInterval::symmetric(weighted_mean, margin, level))
    }

    /// Number of active (non-expired) estimates
    pub fn active_count(&self, now: i64) -> usize {
        self.estimates.iter().filter(|e| !e.is_expired(now)).count()
    }
}

/// Tracks confidence over time with event-driven shocks
#[derive(Debug)]
pub struct ConfidenceTracker {
    /// Historical confidence values (timestamp, confidence)
    history: VecDeque<(i64, f64)>,
    /// Maximum history length
    max_history: usize,
    /// Current baseline confidence
    baseline: f64,
    /// Domain for decay
    domain: DecayDomain,
    /// Last update timestamp
    last_update: i64,
}

impl ConfidenceTracker {
    pub fn new(initial_confidence: f64, domain: DecayDomain, now: i64) -> Self {
        let mut history = VecDeque::with_capacity(1000);
        history.push_back((now, initial_confidence));

        Self {
            history,
            max_history: 1000,
            baseline: initial_confidence,
            domain,
            last_update: now,
        }
    }

    /// Get current confidence (with decay)
    pub fn current(&self, now: i64) -> f64 {
        let hours = (now - self.last_update) as f64 / 3600.0;
        let decayed = self.baseline * (-self.domain.decay_rate() * hours).exp();
        decayed.clamp(0.0, 1.0)
    }

    /// Apply a confidence shock (event that affects confidence)
    pub fn shock(&mut self, now: i64, multiplier: f64) {
        // First decay from last update
        self.baseline = self.current(now);
        // Apply shock
        self.baseline = (self.baseline * multiplier).clamp(0.0, 1.0);
        self.last_update = now;

        // Record in history
        self.history.push_back((now, self.baseline));
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Boost confidence (new corroborating evidence)
    pub fn corroborate(&mut self, now: i64, amount: f64) {
        self.baseline = self.current(now);
        self.baseline = (self.baseline + amount * (1.0 - self.baseline)).clamp(0.0, 1.0);
        self.last_update = now;
        self.history.push_back((now, self.baseline));
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Reset to baseline with new confidence
    pub fn reset(&mut self, now: i64, confidence: f64) {
        self.baseline = confidence.clamp(0.0, 1.0);
        self.last_update = now;
        self.history.push_back((now, self.baseline));
    }

    /// Get confidence history
    pub fn history(&self) -> &VecDeque<(i64, f64)> {
        &self.history
    }
}

/// Forecast with temporal validity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalForecast {
    /// Forecast value
    pub value: f64,
    /// When forecast was made
    pub issued_at: i64,
    /// Forecast target time
    pub target_time: i64,
    /// Initial confidence
    pub confidence: f64,
    /// Decay domain
    pub domain: DecayDomain,
}

impl TemporalForecast {
    /// Confidence in forecast at evaluation time
    pub fn confidence_at(&self, now: i64) -> f64 {
        // Confidence decays both with:
        // 1. Time since forecast was issued
        // 2. Remaining time to target (more confident as we approach)

        let age = (now - self.issued_at) as f64 / 3600.0;
        let remaining = (self.target_time - now) as f64 / 3600.0;

        if remaining <= 0.0 {
            // Past target time - rapid decay
            return self.confidence * (-self.domain.decay_rate() * (-remaining)).exp();
        }

        // Age decay
        let age_factor = (-self.domain.decay_rate() * age).exp();

        // Horizon factor: more confident as target approaches
        let total_horizon = (self.target_time - self.issued_at) as f64 / 3600.0;
        let horizon_factor = if total_horizon > 0.0 {
            1.0 - (remaining / total_horizon).powi(2) * 0.3
        } else {
            1.0
        };

        (self.confidence * age_factor * horizon_factor).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_half_life() {
        let domain = DecayDomain::Political;
        let estimate = TemporalEstimate::new(0.85, 0.9, 0, domain);

        // At t=0, full confidence
        let c0 = estimate.confidence(0);
        assert!((c0 - 0.9 * 0.8).abs() < 0.01);  // 0.9 * 0.8 (reliability)

        // At t=24h, should be ~50% of initial (half-life)
        let c24 = estimate.confidence(86400);  // 24 hours in seconds
        assert!((c24 / c0 - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_std_growth() {
        let estimate = TemporalEstimate::new(0.5, 0.9, 0, DecayDomain::Political)
            .with_std(0.1);

        let std0 = estimate.std(0);
        let std24 = estimate.std(86400);

        assert!((std0 - 0.1).abs() < 0.01);
        assert!(std24 > std0);  // Uncertainty grows
    }

    #[test]
    fn test_expiry() {
        let estimate = TemporalEstimate::new(0.5, 0.9, 0, DecayDomain::Political)
            .with_expiry(3600);

        assert!(!estimate.is_expired(1800));  // 30 min - not expired
        assert!(estimate.is_expired(7200));   // 2 hours - expired
        assert_eq!(estimate.confidence(7200), 0.0);
    }

    #[test]
    fn test_aggregator() {
        let mut agg = EpistemicAggregator::new();

        // Two estimates with different confidences
        agg.add(TemporalEstimate::new(0.8, 0.9, 0, DecayDomain::Political));
        agg.add(TemporalEstimate::new(0.6, 0.7, 0, DecayDomain::Political));

        let mean = agg.weighted_mean(0).unwrap();
        // Weighted toward higher confidence estimate
        assert!(mean > 0.7);
        assert!(mean < 0.8);
    }

    #[test]
    fn test_confidence_tracker() {
        let mut tracker = ConfidenceTracker::new(0.9, DecayDomain::Political, 0);

        // Check initial
        assert!((tracker.current(0) - 0.9).abs() < 0.01);

        // After 24 hours
        let c24 = tracker.current(86400);
        assert!(c24 < 0.9 * 0.6);  // Decayed significantly

        // Apply positive shock (corroboration)
        tracker.corroborate(86400, 0.3);
        let after_shock = tracker.current(86400);
        assert!(after_shock > c24);
    }

    #[test]
    fn test_forecast_confidence() {
        let forecast = TemporalForecast {
            value: 0.75,
            issued_at: 0,
            target_time: 86400,  // 24h target
            confidence: 0.8,
            domain: DecayDomain::Political,
        };

        // At issue time
        let c0 = forecast.confidence_at(0);
        assert!(c0 < 0.8);  // Reduced by horizon uncertainty

        // Closer to target
        let c20 = forecast.confidence_at(72000);  // 20h in
        assert!(c20 > c0);  // More confident as we approach

        // Past target
        let c30 = forecast.confidence_at(108000);  // 30h (6h past)
        assert!(c30 < c20);  // Rapid decay after target
    }

    #[test]
    fn test_interval_contains() {
        let ci = ConfidenceInterval::new(0.5, 0.3, 0.7, 0.95);
        assert!(ci.contains(0.5));
        assert!(ci.contains(0.4));
        assert!(!ci.contains(0.2));
        assert!(!ci.contains(0.8));
    }
}
