/**
 * SignalProcessor - CPU-first signal processing pipeline
 *
 * The GPU-lovers' blind spot: If you can formalize text into math,
 * CPU multicore absolutely shreds on embarrassingly parallel workloads.
 *
 * This module does:
 * 1. Text → Features (TF-IDF, sentiment, entities) - CPU
 * 2. Features → Clusters (GTVC via Rust WASM) - CPU
 * 3. Clusters → Anomaly Detection (statistical) - CPU
 * 4. ONLY anomalies → LLM interpretation - GPU (rare)
 *
 * Expected savings: 80-95% reduction in LLM calls
 */

// ============================================================
// FEATURE EXTRACTION - Text to Numbers (CPU-bound)
// ============================================================

export interface SignalFeatures {
  // Numeric features (all CPU-computable)
  sentimentScore: number;      // -1 to 1 (rules-based)
  urgencyScore: number;        // 0 to 1 (keyword density)
  magnitudeScore: number;      // 0 to 1 (numbers/percentages detected)
  entityDensity: number;       // 0 to 1 (named entities per token)
  topicVector: number[];       // 8-dim topic embedding (TF-IDF to fixed topics)

  // Categorical (for grouping)
  primaryTopic: string;
  detectedEntities: string[];
  urgencyKeywords: string[];

  // Raw for LLM fallback
  originalText: string;
  wordCount: number;
}

export interface ProcessedSignal {
  id: string;
  source: 'gdelt' | 'rss' | 'api';
  timestamp: Date;
  features: SignalFeatures;

  // Clustering results (from Rust WASM)
  clusterId?: number;
  clusterScore?: number;
  isAnomaly: boolean;
  anomalyReason?: string;
}

// ============================================================
// SENTIMENT ANALYSIS - Rules-based, no LLM needed
// ============================================================

// Curated sentiment lexicon (expandable)
const POSITIVE_WORDS = new Set([
  'agreement', 'alliance', 'breakthrough', 'ceasefire', 'cooperation',
  'deal', 'diplomatic', 'growth', 'improve', 'negotiate', 'peace',
  'progress', 'recovery', 'resolution', 'stable', 'success', 'treaty',
  'upturn', 'win', 'gains', 'surge', 'rally', 'boost', 'strengthen',
]);

const NEGATIVE_WORDS = new Set([
  'attack', 'bankruptcy', 'casualties', 'clash', 'collapse', 'conflict',
  'crisis', 'death', 'decline', 'default', 'destruction', 'disaster',
  'escalate', 'explosion', 'fail', 'fear', 'hostage', 'invasion',
  'kill', 'loss', 'massacre', 'panic', 'protest', 'recession',
  'sanction', 'strike', 'tension', 'terror', 'threat', 'violence',
  'war', 'warning', 'crash', 'plunge', 'slump', 'turmoil', 'unrest',
]);

const URGENCY_WORDS = new Set([
  'breaking', 'urgent', 'immediate', 'emergency', 'critical', 'alert',
  'just in', 'developing', 'flash', 'now', 'live', 'ongoing',
  'imminent', 'sudden', 'unprecedented', 'shock', 'dramatic',
]);

const MAGNITUDE_PATTERNS = [
  /\d+%/g,                        // Percentages
  /\$[\d,]+(?:\.\d+)?[BMK]?/gi,   // Dollar amounts
  /[\d,]+\s*(?:million|billion|trillion)/gi,
  /\b\d{4,}\b/g,                  // Large numbers
  /(?:double|triple|quadruple)d?/gi,
  /(?:surge|plunge|crash|spike)\s+(?:of\s+)?\d+/gi,
];

/**
 * Extract sentiment score from text using rules-based approach
 * Returns -1 (very negative) to 1 (very positive)
 */
export function extractSentiment(text: string): number {
  const words = text.toLowerCase().split(/\W+/);
  let positive = 0;
  let negative = 0;

  for (const word of words) {
    if (POSITIVE_WORDS.has(word)) positive++;
    if (NEGATIVE_WORDS.has(word)) negative++;
  }

  const total = positive + negative;
  if (total === 0) return 0;

  // Normalize to [-1, 1]
  return (positive - negative) / Math.max(total, 1);
}

/**
 * Extract urgency score based on keyword density
 */
export function extractUrgency(text: string): number {
  const words = text.toLowerCase().split(/\W+/);
  let urgencyCount = 0;

  for (const word of words) {
    if (URGENCY_WORDS.has(word)) urgencyCount++;
  }

  // Normalize: 3+ urgency words = max urgency
  return Math.min(1, urgencyCount / 3);
}

/**
 * Extract magnitude score based on numeric content
 */
export function extractMagnitude(text: string): number {
  let matches = 0;

  for (const pattern of MAGNITUDE_PATTERNS) {
    const found = text.match(pattern);
    if (found) matches += found.length;
  }

  // Normalize: 5+ magnitude indicators = max
  return Math.min(1, matches / 5);
}

// ============================================================
// NAMED ENTITY EXTRACTION - Rules-based
// ============================================================

// Common country patterns (ISO codes and full names)
const COUNTRY_PATTERNS = /\b(?:USA|Russia|China|Ukraine|Israel|Iran|NATO|EU|UN|Taiwan|North Korea|South Korea|Japan|Germany|France|UK|Britain|India|Pakistan|Saudi|Turkey|Syria|Lebanon|Gaza|Yemen|Sudan|Myanmar|Afghanistan|Iraq|Libya)\b/gi;

// Organization patterns
const ORG_PATTERNS = /\b(?:Pentagon|Kremlin|White House|Congress|Senate|Parliament|IMF|World Bank|Fed|ECB|OPEC|WHO|WTO|ICC|ICJ|G7|G20|BRICS|ASEAN|Arab League)\b/gi;

// Leader title patterns
const LEADER_PATTERNS = /\b(?:President|Prime Minister|PM|Chancellor|King|Queen|Supreme Leader|General Secretary|Chairman)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?/g;

/**
 * Extract named entities from text
 */
export function extractEntities(text: string): string[] {
  const entities: Set<string> = new Set();

  // Extract countries
  const countries = text.match(COUNTRY_PATTERNS) || [];
  countries.forEach(c => entities.add(c.toUpperCase()));

  // Extract organizations
  const orgs = text.match(ORG_PATTERNS) || [];
  orgs.forEach(o => entities.add(o));

  // Extract leaders
  const leaders = text.match(LEADER_PATTERNS) || [];
  leaders.forEach(l => entities.add(l));

  return Array.from(entities);
}

// ============================================================
// TOPIC CLASSIFICATION - TF-IDF to fixed topic space
// ============================================================

// 8 core topics with defining keywords (can expand)
const TOPIC_KEYWORDS: Record<string, string[]> = {
  military: ['military', 'troops', 'forces', 'army', 'navy', 'weapons', 'missile', 'drone', 'airstrike', 'combat', 'defense', 'offensive'],
  economic: ['economy', 'gdp', 'inflation', 'trade', 'tariff', 'export', 'import', 'market', 'stocks', 'bonds', 'currency', 'investment'],
  political: ['election', 'vote', 'parliament', 'congress', 'legislation', 'policy', 'reform', 'coalition', 'opposition', 'campaign'],
  security: ['terror', 'attack', 'bomb', 'threat', 'intelligence', 'spy', 'surveillance', 'border', 'security', 'police'],
  diplomatic: ['diplomat', 'ambassador', 'summit', 'treaty', 'agreement', 'negotiate', 'talks', 'alliance', 'sanction', 'embargo'],
  humanitarian: ['refugee', 'aid', 'humanitarian', 'crisis', 'famine', 'drought', 'flood', 'earthquake', 'disaster', 'evacuation'],
  energy: ['oil', 'gas', 'pipeline', 'opec', 'energy', 'nuclear', 'power', 'electricity', 'renewable', 'fuel', 'barrel'],
  tech: ['cyber', 'hack', 'ai', 'technology', 'semiconductor', 'chip', 'data', 'internet', 'satellite', 'space'],
};

const TOPICS = Object.keys(TOPIC_KEYWORDS);

/**
 * Compute topic vector using keyword matching (simplified TF-IDF)
 * Returns 8-dimensional vector representing topic distribution
 */
export function computeTopicVector(text: string): number[] {
  const words = new Set(text.toLowerCase().split(/\W+/));
  const vector: number[] = [];

  for (const topic of TOPICS) {
    const keywords = TOPIC_KEYWORDS[topic];
    let matches = 0;
    for (const keyword of keywords) {
      if (words.has(keyword)) matches++;
    }
    // Normalize by keyword count
    vector.push(matches / keywords.length);
  }

  // Normalize vector to sum to 1 (or 0 if no matches)
  const sum = vector.reduce((a, b) => a + b, 0);
  if (sum > 0) {
    return vector.map(v => v / sum);
  }
  return vector;
}

/**
 * Get primary topic from vector
 */
export function getPrimaryTopic(topicVector: number[]): string {
  let maxIdx = 0;
  let maxVal = topicVector[0];

  for (let i = 1; i < topicVector.length; i++) {
    if (topicVector[i] > maxVal) {
      maxVal = topicVector[i];
      maxIdx = i;
    }
  }

  return maxVal > 0 ? TOPICS[maxIdx] : 'general';
}

// ============================================================
// FULL FEATURE EXTRACTION PIPELINE
// ============================================================

/**
 * Extract all features from text signal - PURE CPU, NO LLM
 */
export function extractFeatures(text: string): SignalFeatures {
  const words = text.split(/\W+/);
  const entities = extractEntities(text);
  const topicVector = computeTopicVector(text);

  return {
    sentimentScore: extractSentiment(text),
    urgencyScore: extractUrgency(text),
    magnitudeScore: extractMagnitude(text),
    entityDensity: entities.length / Math.max(words.length, 1),
    topicVector,
    primaryTopic: getPrimaryTopic(topicVector),
    detectedEntities: entities,
    urgencyKeywords: words.filter(w => URGENCY_WORDS.has(w.toLowerCase())),
    originalText: text,
    wordCount: words.length,
  };
}

/**
 * Batch extract features - exploits JavaScript's event loop for pseudo-parallelism
 */
export async function batchExtractFeatures(texts: string[]): Promise<SignalFeatures[]> {
  // Process in chunks to avoid blocking
  const CHUNK_SIZE = 100;
  const results: SignalFeatures[] = [];

  for (let i = 0; i < texts.length; i += CHUNK_SIZE) {
    const chunk = texts.slice(i, i + CHUNK_SIZE);
    const chunkResults = chunk.map(extractFeatures);
    results.push(...chunkResults);

    // Yield to event loop between chunks
    if (i + CHUNK_SIZE < texts.length) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  return results;
}

// ============================================================
// ANOMALY DETECTION - Statistical, no LLM
// ============================================================

export interface BaselineStats {
  sentimentMean: number;
  sentimentStd: number;
  urgencyMean: number;
  urgencyStd: number;
  topicDistribution: number[];  // Rolling average topic vector
  signalCount: number;
  lastUpdated: Date;
}

/**
 * Compute baseline statistics from historical signals
 */
export function computeBaseline(features: SignalFeatures[]): BaselineStats {
  if (features.length === 0) {
    return {
      sentimentMean: 0,
      sentimentStd: 0.5,
      urgencyMean: 0.2,
      urgencyStd: 0.3,
      topicDistribution: TOPICS.map(() => 1 / TOPICS.length),
      signalCount: 0,
      lastUpdated: new Date(),
    };
  }

  // Compute means
  const sentiments = features.map(f => f.sentimentScore);
  const urgencies = features.map(f => f.urgencyScore);

  const sentimentMean = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
  const urgencyMean = urgencies.reduce((a, b) => a + b, 0) / urgencies.length;

  // Compute standard deviations
  const sentimentVariance = sentiments.reduce((sum, s) => sum + Math.pow(s - sentimentMean, 2), 0) / sentiments.length;
  const urgencyVariance = urgencies.reduce((sum, u) => sum + Math.pow(u - urgencyMean, 2), 0) / urgencies.length;

  // Compute average topic distribution
  const topicSum = TOPICS.map(() => 0);
  for (const f of features) {
    for (let i = 0; i < f.topicVector.length; i++) {
      topicSum[i] += f.topicVector[i];
    }
  }
  const topicDistribution = topicSum.map(t => t / features.length);

  return {
    sentimentMean,
    sentimentStd: Math.sqrt(sentimentVariance) || 0.5,
    urgencyMean,
    urgencyStd: Math.sqrt(urgencyVariance) || 0.3,
    topicDistribution,
    signalCount: features.length,
    lastUpdated: new Date(),
  };
}

/**
 * Check if a signal is anomalous compared to baseline
 * Uses simple z-score threshold - no LLM needed!
 */
export function detectAnomaly(
  features: SignalFeatures,
  baseline: BaselineStats,
  thresholdSigma: number = 2.0
): { isAnomaly: boolean; reasons: string[] } {
  const reasons: string[] = [];

  // Z-score for sentiment (negative shift is concerning)
  const sentimentZ = (features.sentimentScore - baseline.sentimentMean) / baseline.sentimentStd;
  if (sentimentZ < -thresholdSigma) {
    reasons.push(`Sentiment shift: ${sentimentZ.toFixed(1)}σ below baseline`);
  }

  // Z-score for urgency (high urgency is concerning)
  const urgencyZ = (features.urgencyScore - baseline.urgencyMean) / baseline.urgencyStd;
  if (urgencyZ > thresholdSigma) {
    reasons.push(`Urgency spike: ${urgencyZ.toFixed(1)}σ above baseline`);
  }

  // Topic drift detection (cosine distance from baseline)
  const topicDrift = cosineDissimilarity(features.topicVector, baseline.topicDistribution);
  if (topicDrift > 0.5) {  // 50% topic drift threshold
    reasons.push(`Topic drift: ${(topicDrift * 100).toFixed(0)}% from baseline`);
  }

  // High magnitude always notable
  if (features.magnitudeScore > 0.7) {
    reasons.push(`High magnitude indicators: ${features.magnitudeScore.toFixed(2)}`);
  }

  // Multiple high-value entities
  if (features.detectedEntities.length >= 5) {
    reasons.push(`Entity density: ${features.detectedEntities.length} actors mentioned`);
  }

  return {
    isAnomaly: reasons.length > 0,
    reasons,
  };
}

/**
 * Cosine dissimilarity between two vectors
 */
function cosineDissimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 1;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 1;

  const similarity = dotProduct / (normA * normB);
  return 1 - similarity;  // Convert to dissimilarity
}

// ============================================================
// SIGNAL DEDUPLICATION - Prevent processing same news twice
// ============================================================

/**
 * Simple hash for deduplication
 */
export function signalHash(text: string): string {
  // Use first 100 chars + word count as cheap hash
  const normalized = text.toLowerCase().replace(/\s+/g, ' ').slice(0, 100);
  const wordCount = text.split(/\s+/).length;

  // Simple string hash
  let hash = 0;
  for (let i = 0; i < normalized.length; i++) {
    const char = normalized.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;  // Convert to 32bit integer
  }

  return `${Math.abs(hash).toString(36)}-${wordCount}`;
}

/**
 * Check similarity between two signals (0 = identical, 1 = completely different)
 */
export function signalSimilarity(a: SignalFeatures, b: SignalFeatures): number {
  // Combine multiple similarity measures
  const topicSim = 1 - cosineDissimilarity(a.topicVector, b.topicVector);
  const sentimentSim = 1 - Math.abs(a.sentimentScore - b.sentimentScore) / 2;
  const entityOverlap = jaccardSimilarity(
    new Set(a.detectedEntities),
    new Set(b.detectedEntities)
  );

  // Weighted average
  return topicSim * 0.4 + sentimentSim * 0.2 + entityOverlap * 0.4;
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;

  const intersection = new Set([...a].filter(x => b.has(x)));
  const union = new Set([...a, ...b]);

  return intersection.size / union.size;
}

// ============================================================
// MAIN PROCESSING PIPELINE
// ============================================================

export interface ProcessingResult {
  processed: ProcessedSignal[];
  anomalies: ProcessedSignal[];
  baseline: BaselineStats;
  stats: {
    totalInputs: number;
    deduplicated: number;
    anomalyCount: number;
    processingTimeMs: number;
  };
}

/**
 * Full CPU-first processing pipeline
 *
 * @param texts - Raw text signals to process
 * @param existingBaseline - Optional baseline for anomaly detection
 * @returns ProcessingResult with anomalies flagged
 */
export async function processSignals(
  texts: string[],
  existingBaseline?: BaselineStats
): Promise<ProcessingResult> {
  const startTime = Date.now();

  // Step 1: Deduplicate
  const seen = new Set<string>();
  const uniqueTexts: string[] = [];

  for (const text of texts) {
    const hash = signalHash(text);
    if (!seen.has(hash)) {
      seen.add(hash);
      uniqueTexts.push(text);
    }
  }

  // Step 2: Extract features (CPU, parallel-ish)
  const features = await batchExtractFeatures(uniqueTexts);

  // Step 3: Compute or update baseline
  const baseline = existingBaseline || computeBaseline(features);

  // Step 4: Detect anomalies
  const processed: ProcessedSignal[] = [];
  const anomalies: ProcessedSignal[] = [];

  for (let i = 0; i < features.length; i++) {
    const detection = detectAnomaly(features[i], baseline);

    const signal: ProcessedSignal = {
      id: signalHash(uniqueTexts[i]),
      source: 'gdelt',  // Default, caller can override
      timestamp: new Date(),
      features: features[i],
      isAnomaly: detection.isAnomaly,
      anomalyReason: detection.reasons.join('; '),
    };

    processed.push(signal);
    if (detection.isAnomaly) {
      anomalies.push(signal);
    }
  }

  return {
    processed,
    anomalies,
    baseline,
    stats: {
      totalInputs: texts.length,
      deduplicated: texts.length - uniqueTexts.length,
      anomalyCount: anomalies.length,
      processingTimeMs: Date.now() - startTime,
    },
  };
}

// ============================================================
// TOKEN BUDGET DECISION
// ============================================================

export type LLMTier = 'none' | 'local' | 'elle';

/**
 * Decide if a signal needs LLM interpretation
 *
 * Returns:
 * - 'none': No LLM needed, use template
 * - 'local': Use local small model (future: Phi/Qwen-0.5B)
 * - 'elle': Use full Elle on RunPod
 */
export function decideLLMTier(signal: ProcessedSignal): LLMTier {
  // No anomaly = no LLM needed
  if (!signal.isAnomaly) {
    return 'none';
  }

  // High urgency + negative sentiment + magnitude = Elle
  if (
    signal.features.urgencyScore > 0.5 &&
    signal.features.sentimentScore < -0.3 &&
    signal.features.magnitudeScore > 0.5
  ) {
    return 'elle';
  }

  // Multiple major entities = Elle (likely complex geopolitical event)
  if (signal.features.detectedEntities.length >= 4) {
    return 'elle';
  }

  // Topic drift but low urgency = local model sufficient
  if (signal.isAnomaly && signal.features.urgencyScore < 0.3) {
    return 'local';
  }

  // Default: local model for moderate anomalies
  return 'local';
}

/**
 * Estimate token savings from this pipeline
 */
export function estimateSavings(result: ProcessingResult): {
  signalsSkipped: number;
  localModelSignals: number;
  elleSignals: number;
  estimatedTokensSaved: number;
  estimatedCostSavedUsd: number;
} {
  let skipped = 0;
  let local = 0;
  let elle = 0;

  for (const signal of result.processed) {
    const tier = decideLLMTier(signal);
    if (tier === 'none') skipped++;
    else if (tier === 'local') local++;
    else elle++;
  }

  // Estimate: Each signal would have cost ~500 tokens on Elle
  const tokensPerSignal = 500;
  const tokensSaved = skipped * tokensPerSignal;

  // At $0.009/second and ~30 tokens/second, roughly $0.00015 per token
  const costPerToken = 0.00015;
  const costSaved = tokensSaved * costPerToken;

  return {
    signalsSkipped: skipped,
    localModelSignals: local,
    elleSignals: elle,
    estimatedTokensSaved: tokensSaved,
    estimatedCostSavedUsd: costSaved,
  };
}
