//! Historical Pattern Database with Fuzzy Matching
//!
//! Stores historical geopolitical/economic patterns and finds similar
//! situations using vector similarity and structural matching.
//!
//! Key features:
//! - Pattern encoding as feature vectors
//! - Cosine similarity for pattern matching
//! - Threshold-based retrieval
//! - Temporal weighting (recent patterns weighted higher)

use crate::briefings::{Phase, CICMetrics, DomainRisk};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature vector for pattern matching (normalized to unit length)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternVector {
    /// Political tension (0-1)
    pub political: f64,
    /// Economic stress (0-1)
    pub economic: f64,
    /// Military activity (0-1)
    pub military: f64,
    /// Cyber threat (0-1)
    pub cyber: f64,
    /// Social unrest (0-1)
    pub social: f64,
    /// Resource competition (0-1)
    pub resource: f64,
    /// Alliance cohesion (0-1)
    pub alliance: f64,
    /// Media sentiment (-1 to 1)
    pub sentiment: f64,
}

impl PatternVector {
    /// Create from raw values, normalizes to unit length
    pub fn new(
        political: f64,
        economic: f64,
        military: f64,
        cyber: f64,
        social: f64,
        resource: f64,
        alliance: f64,
        sentiment: f64,
    ) -> Self {
        let mut v = Self {
            political,
            economic,
            military,
            cyber,
            social,
            resource,
            alliance,
            sentiment,
        };
        v.normalize();
        v
    }

    /// Create from domain risks (partial vector)
    pub fn from_risks(risks: &DomainRisk) -> Self {
        Self::new(
            risks.political,
            risks.economic,
            risks.security,  // Map security to military
            risks.cyber,
            0.5,  // Default social
            0.5,  // Default resource
            0.5,  // Default alliance
            0.0,  // Neutral sentiment
        )
    }

    /// As slice for computation
    pub fn as_slice(&self) -> [f64; 8] {
        [
            self.political,
            self.economic,
            self.military,
            self.cyber,
            self.social,
            self.resource,
            self.alliance,
            self.sentiment,
        ]
    }

    /// Magnitude (L2 norm)
    pub fn magnitude(&self) -> f64 {
        self.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 1e-10 {
            self.political /= mag;
            self.economic /= mag;
            self.military /= mag;
            self.cyber /= mag;
            self.social /= mag;
            self.resource /= mag;
            self.alliance /= mag;
            self.sentiment /= mag;
        }
    }

    /// Cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &PatternVector) -> f64 {
        let a = self.as_slice();
        let b = other.as_slice();
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        // Already normalized, so just return dot product
        dot.clamp(-1.0, 1.0)
    }

    /// Euclidean distance
    pub fn euclidean_distance(&self, other: &PatternVector) -> f64 {
        let a = self.as_slice();
        let b = other.as_slice();
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Weighted similarity (custom weights per dimension)
    pub fn weighted_similarity(&self, other: &PatternVector, weights: &[f64; 8]) -> f64 {
        let a = self.as_slice();
        let b = other.as_slice();
        let weighted_dot: f64 = a.iter()
            .zip(b.iter())
            .zip(weights.iter())
            .map(|((x, y), w)| x * y * w)
            .sum();
        let weight_sum: f64 = weights.iter().sum();
        weighted_dot / weight_sum.max(1e-10)
    }
}

/// Historical pattern entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalPattern {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Year/date of occurrence
    pub year: i32,
    /// Primary actors involved (ISO codes)
    pub actors: Vec<String>,
    /// Pattern feature vector
    pub vector: PatternVector,
    /// Phase at time of pattern
    pub phase: Phase,
    /// CIC metrics at time
    pub cic: CICMetrics,
    /// Outcome description
    pub outcome: String,
    /// Outcome severity (0-1)
    pub severity: f64,
    /// Duration in days
    pub duration_days: u32,
    /// Tags for categorical matching
    pub tags: Vec<String>,
}

impl HistoricalPattern {
    /// Create a new historical pattern
    pub fn new(
        id: &str,
        name: &str,
        year: i32,
        actors: Vec<String>,
        vector: PatternVector,
        phase: Phase,
        outcome: &str,
        severity: f64,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            year,
            actors,
            vector,
            phase,
            cic: CICMetrics::new(0.5, 0.5, 0.5),
            outcome: outcome.to_string(),
            severity,
            duration_days: 0,
            tags: Vec::new(),
        }
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<&str>) -> Self {
        self.tags = tags.into_iter().map(String::from).collect();
        self
    }

    /// Set duration
    pub fn with_duration(mut self, days: u32) -> Self {
        self.duration_days = days;
        self
    }

    /// Set CIC metrics
    pub fn with_cic(mut self, cic: CICMetrics) -> Self {
        self.cic = cic;
        self
    }
}

/// Match result with similarity score
#[derive(Debug, Clone)]
pub struct PatternMatch<'a> {
    pub pattern: &'a HistoricalPattern,
    pub similarity: f64,
    pub phase_match: bool,
    pub actor_overlap: f64,
    /// Combined relevance score
    pub relevance: f64,
}

impl<'a> PatternMatch<'a> {
    /// Calculate combined relevance
    fn calculate_relevance(&mut self, current_year: i32) {
        // Time decay: more recent patterns weighted higher
        let age = (current_year - self.pattern.year).max(0) as f64;
        let recency = (-age / 50.0).exp();  // 50-year half-life

        // Phase match bonus
        let phase_bonus = if self.phase_match { 0.2 } else { 0.0 };

        // Combined score
        self.relevance = self.similarity * 0.5
            + recency * 0.2
            + self.actor_overlap * 0.2
            + phase_bonus * 0.1;
    }
}

/// Historical pattern database
#[derive(Debug, Default)]
pub struct PatternDatabase {
    patterns: HashMap<String, HistoricalPattern>,
    /// Index by tag
    by_tag: HashMap<String, Vec<String>>,
    /// Index by actor
    by_actor: HashMap<String, Vec<String>>,
}

impl PatternDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create database with built-in historical patterns
    pub fn with_defaults() -> Self {
        let mut db = Self::new();
        db.add_default_patterns();
        db
    }

    /// Add a pattern
    pub fn add(&mut self, pattern: HistoricalPattern) {
        // Update indices
        for tag in &pattern.tags {
            self.by_tag
                .entry(tag.clone())
                .or_default()
                .push(pattern.id.clone());
        }
        for actor in &pattern.actors {
            self.by_actor
                .entry(actor.clone())
                .or_default()
                .push(pattern.id.clone());
        }
        self.patterns.insert(pattern.id.clone(), pattern);
    }

    /// Find similar patterns to a query vector
    pub fn find_similar(
        &self,
        query: &PatternVector,
        phase: Phase,
        actors: &[String],
        threshold: f64,
        current_year: i32,
    ) -> Vec<PatternMatch<'_>> {
        let mut matches: Vec<PatternMatch<'_>> = self.patterns
            .values()
            .filter_map(|pattern| {
                let similarity = query.cosine_similarity(&pattern.vector);
                if similarity < threshold {
                    return None;
                }

                let phase_match = pattern.phase == phase;

                // Calculate actor overlap (Jaccard similarity)
                let actor_overlap = if actors.is_empty() || pattern.actors.is_empty() {
                    0.0
                } else {
                    let query_set: std::collections::HashSet<_> = actors.iter().collect();
                    let pattern_set: std::collections::HashSet<_> = pattern.actors.iter().collect();
                    let intersection = query_set.intersection(&pattern_set).count();
                    let union = query_set.union(&pattern_set).count();
                    intersection as f64 / union as f64
                };

                let mut m = PatternMatch {
                    pattern,
                    similarity,
                    phase_match,
                    actor_overlap,
                    relevance: 0.0,
                };
                m.calculate_relevance(current_year);
                Some(m)
            })
            .collect();

        // Sort by relevance descending
        matches.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
        matches
    }

    /// Find top-k similar patterns
    pub fn top_k(
        &self,
        query: &PatternVector,
        phase: Phase,
        actors: &[String],
        k: usize,
        current_year: i32,
    ) -> Vec<PatternMatch<'_>> {
        let mut matches = self.find_similar(query, phase, actors, 0.0, current_year);
        matches.truncate(k);
        matches
    }

    /// Find patterns by tag
    pub fn by_tag(&self, tag: &str) -> Vec<&HistoricalPattern> {
        self.by_tag
            .get(tag)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.patterns.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find patterns involving an actor
    pub fn by_actor(&self, actor: &str) -> Vec<&HistoricalPattern> {
        self.by_actor
            .get(actor)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.patterns.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get pattern by ID
    pub fn get(&self, id: &str) -> Option<&HistoricalPattern> {
        self.patterns.get(id)
    }

    /// Total patterns
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Add default historical patterns
    fn add_default_patterns(&mut self) {
        // Cuban Missile Crisis 1962
        self.add(HistoricalPattern::new(
            "cuban-missile-1962",
            "Cuban Missile Crisis",
            1962,
            vec!["USA".into(), "SUN".into(), "CUB".into()],
            PatternVector::new(0.95, 0.3, 0.98, 0.1, 0.6, 0.4, 0.7, -0.8),
            Phase::NUC,
            "Nuclear brinkmanship resolved through negotiation",
            0.95,
        ).with_tags(vec!["nuclear", "cold-war", "brinkmanship"]).with_duration(13));

        // 1973 Oil Crisis
        self.add(HistoricalPattern::new(
            "oil-crisis-1973",
            "1973 Oil Crisis",
            1973,
            vec!["SAU".into(), "USA".into(), "ISR".into()],
            PatternVector::new(0.7, 0.9, 0.4, 0.0, 0.5, 0.95, 0.3, -0.7),
            Phase::SUPER,
            "OPEC embargo caused global recession",
            0.75,
        ).with_tags(vec!["energy", "embargo", "recession"]).with_duration(180));

        // Fall of Berlin Wall 1989
        self.add(HistoricalPattern::new(
            "berlin-wall-1989",
            "Fall of Berlin Wall",
            1989,
            vec!["DEU".into(), "SUN".into()],
            PatternVector::new(0.6, 0.7, 0.2, 0.0, 0.9, 0.3, 0.4, 0.8),
            Phase::ANN,
            "Peaceful transition, German reunification",
            0.3,
        ).with_tags(vec!["transition", "peaceful", "reunification"]).with_duration(330));

        // 2008 Financial Crisis
        self.add(HistoricalPattern::new(
            "financial-2008",
            "2008 Financial Crisis",
            2008,
            vec!["USA".into(), "GBR".into(), "DEU".into()],
            PatternVector::new(0.3, 0.98, 0.1, 0.2, 0.6, 0.2, 0.7, -0.9),
            Phase::NUC,
            "Global financial meltdown, coordinated response",
            0.85,
        ).with_tags(vec!["financial", "banking", "contagion"]).with_duration(545));

        // Crimea Annexation 2014
        self.add(HistoricalPattern::new(
            "crimea-2014",
            "Crimea Annexation",
            2014,
            vec!["RUS".into(), "UKR".into()],
            PatternVector::new(0.9, 0.5, 0.85, 0.6, 0.7, 0.3, 0.4, -0.7),
            Phase::NUC,
            "Territorial change, sanctions, frozen conflict",
            0.70,
        ).with_tags(vec!["annexation", "sanctions", "hybrid-warfare"]).with_duration(30));

        // COVID-19 Pandemic 2020
        self.add(HistoricalPattern::new(
            "covid-2020",
            "COVID-19 Pandemic",
            2020,
            vec!["CHN".into(), "USA".into(), "ITA".into()],
            PatternVector::new(0.5, 0.85, 0.2, 0.4, 0.9, 0.7, 0.6, -0.8),
            Phase::PLAS,
            "Global pandemic, economic disruption, supply chain chaos",
            0.80,
        ).with_tags(vec!["pandemic", "supply-chain", "global"]).with_duration(730));

        // Ukraine Invasion 2022
        self.add(HistoricalPattern::new(
            "ukraine-2022",
            "Ukraine Invasion",
            2022,
            vec!["RUS".into(), "UKR".into()],
            PatternVector::new(0.95, 0.75, 0.98, 0.85, 0.8, 0.6, 0.5, -0.9),
            Phase::NUC,
            "Full-scale invasion, ongoing conflict, energy crisis",
            0.90,
        ).with_tags(vec!["invasion", "war", "energy", "sanctions"]).with_duration(1000));

        // Taiwan Strait Tensions
        self.add(HistoricalPattern::new(
            "taiwan-1996",
            "Third Taiwan Strait Crisis",
            1996,
            vec!["CHN".into(), "TWN".into(), "USA".into()],
            PatternVector::new(0.85, 0.4, 0.9, 0.2, 0.5, 0.3, 0.8, -0.6),
            Phase::SUPER,
            "Missile tests, US carrier deployment, de-escalation",
            0.60,
        ).with_tags(vec!["taiwan", "missiles", "deterrence"]).with_duration(60));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = PatternVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0);
        let sim = v.cosine_similarity(&v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = PatternVector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let v2 = PatternVector::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let sim = v1.cosine_similarity(&v2);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_default_patterns() {
        let db = PatternDatabase::with_defaults();
        assert!(db.len() >= 8);

        // Check Cuban Missile Crisis exists
        let cmc = db.get("cuban-missile-1962");
        assert!(cmc.is_some());
        assert_eq!(cmc.unwrap().year, 1962);
    }

    #[test]
    fn test_find_similar() {
        let db = PatternDatabase::with_defaults();

        // Create a vector similar to the Ukraine 2022 pattern
        let query = PatternVector::new(0.9, 0.7, 0.95, 0.8, 0.75, 0.55, 0.45, -0.85);
        let matches = db.find_similar(
            &query,
            Phase::NUC,
            &["RUS".into(), "UKR".into()],
            0.7,
            2024,
        );

        assert!(!matches.is_empty());
        // Ukraine 2022 should be near the top
        let top_ids: Vec<_> = matches.iter().take(3).map(|m| &m.pattern.id).collect();
        assert!(top_ids.contains(&&"ukraine-2022".to_string()));
    }

    #[test]
    fn test_by_tag() {
        let db = PatternDatabase::with_defaults();
        let nuclear = db.by_tag("nuclear");
        assert!(!nuclear.is_empty());
        assert!(nuclear.iter().any(|p| p.id == "cuban-missile-1962"));
    }

    #[test]
    fn test_actor_overlap() {
        let db = PatternDatabase::with_defaults();
        let rus_patterns = db.by_actor("RUS");
        assert!(rus_patterns.len() >= 2);  // At least Crimea and Ukraine
    }

    #[test]
    fn test_recency_weighting() {
        let db = PatternDatabase::with_defaults();

        let query = PatternVector::new(0.9, 0.7, 0.9, 0.7, 0.7, 0.5, 0.5, -0.8);
        let matches = db.top_k(&query, Phase::NUC, &[], 3, 2024);

        // More recent patterns should have higher relevance
        // Ukraine 2022 should beat Cuban 1962 in relevance
        let ukraine_idx = matches.iter().position(|m| m.pattern.id == "ukraine-2022");
        let cuba_idx = matches.iter().position(|m| m.pattern.id == "cuban-missile-1962");

        if let (Some(u), Some(c)) = (ukraine_idx, cuba_idx) {
            assert!(u < c, "Ukraine 2022 should rank higher than Cuba 1962");
        }
    }
}
