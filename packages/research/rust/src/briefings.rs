//! Content-Addressed Briefing Storage using Blake3
//!
//! Provides immutable, verifiable storage for intelligence briefings.
//! Each briefing is hashed with Blake3, making it:
//! - Content-addressed: identical content = identical hash
//! - Tamper-evident: any change invalidates the hash
//! - Deduplicated: store once, reference many times
//! - Fast: Blake3 is ~10x faster than SHA-256

use blake3::{Hash, Hasher};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Briefing content hash (32 bytes, hex-encoded to 64 chars)
pub type BriefingHash = String;

/// Phase assessment codes (compact representation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Crystalline: stable, ordered state
    CRYS,
    /// Supercooled: metastable, high risk of sudden transition
    SUPER,
    /// Nucleation: active phase change in progress
    NUC,
    /// Plasma: chaotic, high-energy state
    PLAS,
    /// Annealing: cooling down, approaching stability
    ANN,
}

impl Phase {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "CRYS" | "CRYSTALLINE" => Some(Phase::CRYS),
            "SUPER" | "SUPERCOOLED" => Some(Phase::SUPER),
            "NUC" | "NUCLEATION" => Some(Phase::NUC),
            "PLAS" | "PLASMA" => Some(Phase::PLAS),
            "ANN" | "ANNEALING" => Some(Phase::ANN),
            _ => None,
        }
    }

    pub fn risk_multiplier(&self) -> f64 {
        match self {
            Phase::CRYS => 0.2,
            Phase::SUPER => 1.5,  // High risk - metastable
            Phase::NUC => 2.0,    // Critical - transition active
            Phase::PLAS => 1.8,   // Chaotic but may stabilize
            Phase::ANN => 0.5,    // Cooling down
        }
    }
}

/// CIC metrics embedded in briefing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CICMetrics {
    /// Integrated information Φ
    pub phi: f64,
    /// Representation entropy H
    pub entropy: f64,
    /// Causal coherence C
    pub coherence: f64,
    /// Combined F[T] = Φ - λH + γC
    pub f_score: f64,
}

impl CICMetrics {
    pub fn new(phi: f64, entropy: f64, coherence: f64) -> Self {
        // Default weights: λ=0.3, γ=0.4
        let f_score = phi - 0.3 * entropy + 0.4 * coherence;
        Self { phi, entropy, coherence, f_score }
    }

    pub fn with_weights(phi: f64, entropy: f64, coherence: f64, lambda: f64, gamma: f64) -> Self {
        let f_score = phi - lambda * entropy + gamma * coherence;
        Self { phi, entropy, coherence, f_score }
    }
}

/// Domain-specific risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRisk {
    /// Political risk (0-1)
    pub political: f64,
    /// Economic risk (0-1)
    pub economic: f64,
    /// Security/military risk (0-1)
    pub security: f64,
    /// Cyber risk (0-1)
    pub cyber: f64,
    /// Financial/market risk (0-1)
    pub financial: f64,
}

impl DomainRisk {
    pub fn aggregate(&self) -> f64 {
        // Weighted aggregate - security/political weighted higher
        (self.political * 0.25
         + self.economic * 0.15
         + self.security * 0.30
         + self.cyber * 0.15
         + self.financial * 0.15)
    }

    pub fn max_risk(&self) -> f64 {
        self.political
            .max(self.economic)
            .max(self.security)
            .max(self.cyber)
            .max(self.financial)
    }
}

/// Intelligence briefing with all metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Briefing {
    /// Unix timestamp (seconds since epoch)
    pub timestamp: i64,
    /// ISO country codes of primary actors
    pub actors: Vec<String>,
    /// Current phase assessment
    pub phase: Phase,
    /// CIC metrics for this briefing
    pub cic: CICMetrics,
    /// Domain-specific risks
    pub risks: DomainRisk,
    /// Confidence in assessment (0-1)
    pub confidence: f64,
    /// Compact summary (max 200 chars)
    pub summary: String,
    /// Actionable recommendation (max 100 chars)
    pub action: String,
    /// Source IDs that informed this briefing
    pub sources: Vec<String>,
}

impl Briefing {
    /// Compute Blake3 hash of this briefing
    pub fn hash(&self) -> BriefingHash {
        let mut hasher = Hasher::new();
        // Hash the canonical JSON representation
        let json = serde_json::to_string(self).expect("Briefing serializable");
        hasher.update(json.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Verify this briefing matches a given hash
    pub fn verify(&self, expected_hash: &str) -> bool {
        self.hash() == expected_hash
    }

    /// Calculate time-decayed confidence
    pub fn decayed_confidence(&self, now: i64) -> f64 {
        let age_hours = (now - self.timestamp) as f64 / 3600.0;
        // Exponential decay with 24-hour half-life
        let decay = (-age_hours / 24.0).exp();
        self.confidence * decay
    }

    /// Overall risk score accounting for phase
    pub fn risk_score(&self) -> f64 {
        let base_risk = self.risks.aggregate();
        let phase_multiplier = self.phase.risk_multiplier();
        (base_risk * phase_multiplier).min(1.0)
    }
}

/// Content-addressed briefing store
#[derive(Debug, Default)]
pub struct BriefingStore {
    /// Hash -> Briefing mapping
    briefings: HashMap<BriefingHash, Briefing>,
    /// Temporal index: timestamp -> hashes at that time
    by_time: HashMap<i64, Vec<BriefingHash>>,
    /// Actor index: ISO code -> hashes mentioning that actor
    by_actor: HashMap<String, Vec<BriefingHash>>,
    /// Phase index: phase -> hashes with that phase
    by_phase: HashMap<Phase, Vec<BriefingHash>>,
}

impl BriefingStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a briefing, returns its hash
    pub fn store(&mut self, briefing: Briefing) -> BriefingHash {
        let hash = briefing.hash();

        // Only store if not already present (deduplication)
        if !self.briefings.contains_key(&hash) {
            // Update indices
            self.by_time
                .entry(briefing.timestamp)
                .or_default()
                .push(hash.clone());

            for actor in &briefing.actors {
                self.by_actor
                    .entry(actor.clone())
                    .or_default()
                    .push(hash.clone());
            }

            self.by_phase
                .entry(briefing.phase)
                .or_default()
                .push(hash.clone());

            self.briefings.insert(hash.clone(), briefing);
        }

        hash
    }

    /// Retrieve a briefing by hash
    pub fn get(&self, hash: &str) -> Option<&Briefing> {
        self.briefings.get(hash)
    }

    /// Verify and retrieve
    pub fn get_verified(&self, hash: &str) -> Option<&Briefing> {
        self.briefings.get(hash).filter(|b| b.verify(hash))
    }

    /// Get all briefings for an actor
    pub fn by_actor(&self, actor: &str) -> Vec<&Briefing> {
        self.by_actor
            .get(actor)
            .map(|hashes| {
                hashes
                    .iter()
                    .filter_map(|h| self.briefings.get(h))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all briefings in a phase
    pub fn by_phase(&self, phase: Phase) -> Vec<&Briefing> {
        self.by_phase
            .get(&phase)
            .map(|hashes| {
                hashes
                    .iter()
                    .filter_map(|h| self.briefings.get(h))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get briefings in time range
    pub fn in_range(&self, start: i64, end: i64) -> Vec<&Briefing> {
        self.briefings
            .values()
            .filter(|b| b.timestamp >= start && b.timestamp <= end)
            .collect()
    }

    /// Get high-risk briefings (risk > threshold)
    pub fn high_risk(&self, threshold: f64) -> Vec<&Briefing> {
        self.briefings
            .values()
            .filter(|b| b.risk_score() > threshold)
            .collect()
    }

    /// Total briefings stored
    pub fn len(&self) -> usize {
        self.briefings.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.briefings.is_empty()
    }

    /// Get all hashes (for sync/export)
    pub fn all_hashes(&self) -> Vec<&BriefingHash> {
        self.briefings.keys().collect()
    }

    /// Export all briefings as JSON
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.briefings).expect("Serializable")
    }

    /// Import briefings from JSON, returns count imported
    pub fn import_json(&mut self, json: &str) -> Result<usize, serde_json::Error> {
        let briefings: HashMap<BriefingHash, Briefing> = serde_json::from_str(json)?;
        let count = briefings.len();
        for (_, briefing) in briefings {
            self.store(briefing);
        }
        Ok(count)
    }
}

/// Compute Blake3 hash of arbitrary bytes
pub fn hash_bytes(data: &[u8]) -> BriefingHash {
    blake3::hash(data).to_hex().to_string()
}

/// Compute Blake3 hash of a string
pub fn hash_string(s: &str) -> BriefingHash {
    hash_bytes(s.as_bytes())
}

/// Incremental hasher for streaming data
pub struct StreamHasher {
    hasher: Hasher,
}

impl StreamHasher {
    pub fn new() -> Self {
        Self { hasher: Hasher::new() }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }

    pub fn finalize(self) -> BriefingHash {
        self.hasher.finalize().to_hex().to_string()
    }
}

impl Default for StreamHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_briefing() -> Briefing {
        Briefing {
            timestamp: 1702300000,
            actors: vec!["UKR".to_string(), "RUS".to_string()],
            phase: Phase::SUPER,
            cic: CICMetrics::new(0.85, 0.43, 0.78),
            risks: DomainRisk {
                political: 0.85,
                economic: 0.45,
                security: 0.90,
                cyber: 0.60,
                financial: 0.35,
            },
            confidence: 0.82,
            summary: "SUPERCOOLED. UKR 85%, escalation risk elevated. Coalition stable.".to_string(),
            action: "Monitor UKR daily. Update contingencies.".to_string(),
            sources: vec!["gdelt-001".to_string(), "rss-042".to_string()],
        }
    }

    #[test]
    fn test_hash_determinism() {
        let b1 = make_test_briefing();
        let b2 = make_test_briefing();
        assert_eq!(b1.hash(), b2.hash());
    }

    #[test]
    fn test_hash_changes_on_modification() {
        let mut b1 = make_test_briefing();
        let hash1 = b1.hash();
        b1.confidence = 0.90;
        let hash2 = b1.hash();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_store_deduplication() {
        let mut store = BriefingStore::new();
        let b1 = make_test_briefing();
        let b2 = make_test_briefing();

        let h1 = store.store(b1);
        let h2 = store.store(b2);

        assert_eq!(h1, h2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_verification() {
        let briefing = make_test_briefing();
        let hash = briefing.hash();
        assert!(briefing.verify(&hash));
        assert!(!briefing.verify("invalid_hash"));
    }

    #[test]
    fn test_phase_risk_multiplier() {
        assert!(Phase::CRYS.risk_multiplier() < Phase::SUPER.risk_multiplier());
        assert!(Phase::NUC.risk_multiplier() > Phase::SUPER.risk_multiplier());
    }

    #[test]
    fn test_confidence_decay() {
        let briefing = make_test_briefing();
        let now = briefing.timestamp;
        let later = briefing.timestamp + 86400; // 24 hours later

        let conf_now = briefing.decayed_confidence(now);
        let conf_later = briefing.decayed_confidence(later);

        assert!((conf_now - briefing.confidence).abs() < 0.01);
        assert!(conf_later < conf_now);
        // At 24h half-life, should be ~37% (1/e)
        assert!((conf_later / conf_now - 0.368).abs() < 0.01);
    }

    #[test]
    fn test_actor_index() {
        let mut store = BriefingStore::new();
        let b = make_test_briefing();
        store.store(b);

        let ukr_briefings = store.by_actor("UKR");
        let chn_briefings = store.by_actor("CHN");

        assert_eq!(ukr_briefings.len(), 1);
        assert_eq!(chn_briefings.len(), 0);
    }

    #[test]
    fn test_cic_metrics() {
        let cic = CICMetrics::new(0.85, 0.43, 0.78);
        // F = 0.85 - 0.3*0.43 + 0.4*0.78 = 0.85 - 0.129 + 0.312 = 1.033
        assert!((cic.f_score - 1.033).abs() < 0.001);
    }

    #[test]
    fn test_stream_hasher() {
        let mut h = StreamHasher::new();
        h.update(b"hello ");
        h.update(b"world");
        let hash1 = h.finalize();

        let hash2 = hash_string("hello world");
        assert_eq!(hash1, hash2);
    }
}
