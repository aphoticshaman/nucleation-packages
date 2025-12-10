# SIGIL Technical Architecture Prompt
## Advanced Mathematical Foundations & System Design for Google AI Studio

---

## PART I: MATHEMATICAL FOUNDATIONS FOR INTELLIGENCE FUSION

### 1.1 Dempster-Shafer Belief Fusion Framework

The core of SIGIL's multi-source intelligence fusion relies on Dempster-Shafer theory, which handles uncertainty more elegantly than Bayesian approaches when sources have incomplete or conflicting information.

**Basic Probability Assignment (BPA):**
For a frame of discernment Θ = {θ₁, θ₂, ..., θₙ} representing possible threat states, each intelligence source provides a mass function m: 2^Θ → [0,1] where:

```
m(∅) = 0
Σ_{A⊆Θ} m(A) = 1
```

**Dempster's Rule of Combination:**
When combining two independent sources with mass functions m₁ and m₂:

```
m₁₂(A) = (1/K) · Σ_{B∩C=A} m₁(B) · m₂(C)

where K = 1 - Σ_{B∩C=∅} m₁(B) · m₂(C)
```

K is the normalization factor accounting for conflicting evidence. When K approaches 0, sources are highly conflicting and the combination should be flagged for analyst review.

**Belief and Plausibility Bounds:**
```
Bel(A) = Σ_{B⊆A} m(B)     // Lower bound - guaranteed support
Pl(A) = Σ_{B∩A≠∅} m(B)    // Upper bound - possible support
```

The uncertainty interval [Bel(A), Pl(A)] provides analysts with epistemic bounds rather than false precision.

**Implementation for SIGIL:**
```typescript
interface BeliefMass {
  hypothesis: string[];  // Subset of frame of discernment
  mass: number;          // Basic probability assignment
  source: string;        // Intelligence source identifier
  reliability: number;   // Source reliability weight [0,1]
}

function dempsterCombine(m1: BeliefMass[], m2: BeliefMass[]): BeliefMass[] {
  const combined: Map<string, number> = new Map();
  let conflict = 0;

  for (const b1 of m1) {
    for (const b2 of m2) {
      const intersection = b1.hypothesis.filter(h => b2.hypothesis.includes(h));
      const product = b1.mass * b2.mass;

      if (intersection.length === 0) {
        conflict += product;
      } else {
        const key = intersection.sort().join(',');
        combined.set(key, (combined.get(key) || 0) + product);
      }
    }
  }

  const K = 1 - conflict;
  if (K < 0.1) {
    throw new HighConflictError('Sources highly contradictory - manual review required');
  }

  return Array.from(combined.entries()).map(([key, mass]) => ({
    hypothesis: key.split(','),
    mass: mass / K,
    source: 'fused',
    reliability: 1.0
  }));
}
```

**Reliability-Weighted Discounting:**
Before combination, discount unreliable sources:
```
m^α(A) = α · m(A)  for A ⊂ Θ
m^α(Θ) = 1 - α + α · m(Θ)
```
Where α ∈ [0,1] is the source reliability coefficient.

### 1.2 Transfer Entropy for Causal Discovery

SIGIL must identify causal relationships between geopolitical events, not just correlations. Transfer entropy quantifies directed information flow between time series.

**Definition:**
Transfer entropy from process X to process Y:
```
T_{X→Y} = Σ p(y_{t+1}, y_t^(k), x_t^(l)) · log[ p(y_{t+1}|y_t^(k), x_t^(l)) / p(y_{t+1}|y_t^(k)) ]
```

Where:
- y_t^(k) = (y_t, y_{t-1}, ..., y_{t-k+1}) is the k-length history of Y
- x_t^(l) = (x_t, x_{t-1}, ..., x_{t-l+1}) is the l-length history of X

**Kraskov-Stögbauer-Grassberger (KSG) Estimator:**
For continuous variables, use the KSG k-nearest-neighbor estimator to avoid binning artifacts:

```
T̂_{X→Y} = ψ(k) - ⟨ψ(n_x + 1) + ψ(n_y + 1) - ψ(n_{xy} + 1)⟩
```

Where ψ is the digamma function and n_x, n_y, n_xy are neighbor counts in marginal and joint spaces.

**Implementation:**
```typescript
interface CausalEdge {
  source: string;
  target: string;
  transferEntropy: number;
  lag: number;
  pValue: number;  // Significance via surrogate testing
}

function buildCausalGraph(
  timeSeries: Map<string, number[]>,
  maxLag: number = 5,
  kNeighbors: number = 4,
  significanceThreshold: number = 0.05
): CausalEdge[] {
  const edges: CausalEdge[] = [];
  const variables = Array.from(timeSeries.keys());

  for (const source of variables) {
    for (const target of variables) {
      if (source === target) continue;

      for (let lag = 1; lag <= maxLag; lag++) {
        const te = ksgTransferEntropy(
          timeSeries.get(source)!,
          timeSeries.get(target)!,
          lag,
          kNeighbors
        );

        // Surrogate testing for significance
        const pValue = surrogateTest(
          timeSeries.get(source)!,
          timeSeries.get(target)!,
          te,
          lag,
          1000  // Number of surrogates
        );

        if (pValue < significanceThreshold) {
          edges.push({ source, target, transferEntropy: te, lag, pValue });
        }
      }
    }
  }

  return pruneTransitiveEdges(edges);  // Remove indirect causation
}
```

### 1.3 Graph Laplacian Anomaly Detection

For detecting anomalous patterns in intelligence networks (communication graphs, transaction networks, movement patterns), spectral methods on the graph Laplacian provide mathematically grounded anomaly scores.

**Graph Laplacian:**
For a weighted adjacency matrix W, the normalized Laplacian:
```
L = I - D^(-1/2) W D^(-1/2)
```
Where D is the degree matrix with D_ii = Σ_j W_ij.

**Spectral Embedding:**
Embed nodes into ℝ^k using the k smallest non-trivial eigenvectors of L:
```
Φ: v_i → (φ_2(i), φ_3(i), ..., φ_{k+1}(i))
```

**Anomaly Score via Embedding Dispersion:**
For a new observation, compute its spectral coordinates and measure distance to expected cluster:
```
anomaly(v) = ||Φ(v) - μ_cluster||²_Σ  // Mahalanobis distance
```

**Temporal Graph Anomaly:**
For time-evolving graphs G_t, detect structural shifts:
```
Δ_spectral(t) = ||λ(L_t) - λ(L_{t-1})||₂
```

Where λ(L) is the spectrum (eigenvalue vector) of L. Large spectral shifts indicate topological anomalies.

**Implementation:**
```typescript
interface GraphAnomaly {
  timestamp: Date;
  spectralShift: number;
  affectedNodes: string[];
  coherenceAmplification: number;  // Fiedler value change
  anomalyType: 'structural' | 'behavioral' | 'emergent';
}

class SpectralAnomalyDetector {
  private baselineSpectrum: number[];
  private baselineEmbedding: Map<string, number[]>;
  private fiedlerBaseline: number;

  constructor(private k: number = 10) {}

  fit(adjacencyMatrix: number[][], nodeIds: string[]): void {
    const laplacian = this.computeNormalizedLaplacian(adjacencyMatrix);
    const { eigenvalues, eigenvectors } = this.eigenDecomposition(laplacian);

    this.baselineSpectrum = eigenvalues.slice(0, this.k);
    this.fiedlerBaseline = eigenvalues[1];  // Algebraic connectivity

    this.baselineEmbedding = new Map();
    for (let i = 0; i < nodeIds.length; i++) {
      this.baselineEmbedding.set(
        nodeIds[i],
        eigenvectors.slice(1, this.k + 1).map(ev => ev[i])
      );
    }
  }

  detectAnomaly(newAdjacency: number[][], nodeIds: string[]): GraphAnomaly | null {
    const laplacian = this.computeNormalizedLaplacian(newAdjacency);
    const { eigenvalues } = this.eigenDecomposition(laplacian);

    const currentSpectrum = eigenvalues.slice(0, this.k);
    const spectralShift = this.l2Distance(this.baselineSpectrum, currentSpectrum);

    // Coherence amplification: change in Fiedler value
    const fiedlerChange = Math.abs(eigenvalues[1] - this.fiedlerBaseline);

    if (spectralShift > this.threshold || fiedlerChange > 0.1) {
      return {
        timestamp: new Date(),
        spectralShift,
        affectedNodes: this.findAffectedNodes(newAdjacency, nodeIds),
        coherenceAmplification: fiedlerChange / this.fiedlerBaseline,
        anomalyType: this.classifyAnomaly(spectralShift, fiedlerChange)
      };
    }

    return null;
  }
}
```

### 1.4 Compression-Integration-Coherence (CIC) Framework

The CIC framework quantifies how well a system (nation-state, organization, network) integrates information. Based on Integrated Information Theory (IIT) concepts adapted for geopolitical analysis.

**Effective Information (EI):**
Measures the causal architecture of a system:
```
EI(S) = H(S^effect | do(S^cause = uniform)) - H(S^effect | do(S^cause = observed))
```

**Integration (Φ):**
The degree to which a system is irreducible to independent parts:
```
Φ = min_{partition} [ I(S) - Σᵢ I(Sᵢ) ]
```

Where the minimum is over all bipartitions of the system.

**Coherence Score:**
For SIGIL, we adapt this to measure regime stability:
```
Coherence(t) = Φ(G_t) / log(|V|)  // Normalized by system size
```

High coherence indicates a well-integrated system resistant to perturbation. Rapid coherence drops predict regime instability.

---

## PART II: SYSTEM ARCHITECTURE SPECIFICATIONS

### 2.1 Technology Stack Deep Dive

**Frontend Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    React Native / Expo                          │
├─────────────────────────────────────────────────────────────────┤
│  State Management: Zustand + React Query                        │
│  ├── useIntelligenceStore (real-time signals)                   │
│  ├── useBriefingCache (offline-first briefings)                 │
│  └── usePersonaContext (role-based filtering)                   │
├─────────────────────────────────────────────────────────────────┤
│  Visualization Layer                                            │
│  ├── react-native-maps (geospatial)                             │
│  ├── victory-native (charts/graphs)                             │
│  ├── react-native-graph (network visualization)                 │
│  └── react-native-skia (custom shaders for heatmaps)            │
├─────────────────────────────────────────────────────────────────┤
│  Offline Compute                                                │
│  ├── WebAssembly modules (Rust-compiled)                        │
│  │   ├── dempster_shafer.wasm                                   │
│  │   ├── transfer_entropy.wasm                                  │
│  │   ├── graph_laplacian.wasm                                   │
│  │   └── signal_processor.wasm                                  │
│  └── SQLite (local intelligence cache)                          │
└─────────────────────────────────────────────────────────────────┘
```

**Backend Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Edge Runtime (Vercel)                        │
├─────────────────────────────────────────────────────────────────┤
│  API Routes (Next.js Edge Functions)                            │
│  ├── /api/signals/ingest      POST  Rate: 1000/min              │
│  ├── /api/signals/stream      WSS   Connections: 10K            │
│  ├── /api/briefing/generate   POST  Timeout: 60s                │
│  ├── /api/fusion/combine      POST  CPU-bound                   │
│  └── /api/graph/anomaly       POST  WASM-accelerated            │
├─────────────────────────────────────────────────────────────────┤
│  CPU-First Processing Pipeline                                  │
│  ├── SignalProcessor (feature extraction, NLP)                  │
│  ├── LogicalAgent (rule-based inference)                        │
│  ├── DempsterShafer (belief fusion)                             │
│  └── AnomalyDetector (spectral methods)                         │
├─────────────────────────────────────────────────────────────────┤
│  LLM Layer (invoked only when necessary)                        │
│  ├── Tier 1: Haiku (signal classification) - $0.25/1M tokens    │
│  ├── Tier 2: Sonnet (briefing generation) - $3/1M tokens        │
│  └── Tier 3: Opus (strategic analysis) - $15/1M tokens          │
│  Decision function: decideLLMTier(anomalyScore, complexity)     │
└─────────────────────────────────────────────────────────────────┘
```

**Data Layer:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Supabase (PostgreSQL + pgvector)             │
├─────────────────────────────────────────────────────────────────┤
│  Tables                                                         │
│  ├── signals                                                    │
│  │   ├── id: uuid PRIMARY KEY                                   │
│  │   ├── source: text (gdelt, acled, osint, humint)             │
│  │   ├── content: text                                          │
│  │   ├── embedding: vector(1536)                                │
│  │   ├── geolocation: geography(POINT)                          │
│  │   ├── timestamp: timestamptz                                 │
│  │   ├── reliability: float4                                    │
│  │   ├── processed: jsonb                                       │
│  │   └── INDEX: signals_embedding_idx USING ivfflat             │
│  │                                                              │
│  ├── briefings                                                  │
│  │   ├── id: uuid PRIMARY KEY                                   │
│  │   ├── user_id: uuid REFERENCES users                         │
│  │   ├── persona: text                                          │
│  │   ├── content: jsonb                                         │
│  │   ├── inferences: jsonb[]                                    │
│  │   ├── confidence_bounds: numrange                            │
│  │   └── generated_at: timestamptz                              │
│  │                                                              │
│  ├── causal_graph                                               │
│  │   ├── id: uuid PRIMARY KEY                                   │
│  │   ├── source_entity: text                                    │
│  │   ├── target_entity: text                                    │
│  │   ├── transfer_entropy: float4                               │
│  │   ├── lag_days: int2                                         │
│  │   ├── p_value: float4                                        │
│  │   └── valid_until: timestamptz                               │
│  │                                                              │
│  └── nation_states                                              │
│      ├── code: text PRIMARY KEY                                 │
│      ├── name: text                                             │
│      ├── coherence_score: float4                                │
│      ├── basin_strength: float4                                 │
│      ├── transition_risk: float4                                │
│      └── regime_embedding: vector(64)                           │
├─────────────────────────────────────────────────────────────────┤
│  Row Level Security (RLS) Policies                              │
│  ├── signals: authenticated users can read                      │
│  ├── briefings: users can only read their own                   │
│  └── nation_states: public read for basic, role-gated for full  │
├─────────────────────────────────────────────────────────────────┤
│  Edge Functions (Deno)                                          │
│  ├── signal-processor: real-time signal ingestion               │
│  ├── belief-fusion: combine multi-source intelligence           │
│  └── anomaly-detector: spectral graph analysis                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Real-Time Streaming Architecture

**WebSocket Protocol Design:**
```typescript
// Message Types
enum SignalMessageType {
  SIGNAL_NEW = 'signal:new',
  SIGNAL_UPDATE = 'signal:update',
  ANOMALY_DETECTED = 'anomaly:detected',
  BRIEFING_READY = 'briefing:ready',
  GRAPH_SHIFT = 'graph:shift',
  COHERENCE_ALERT = 'coherence:alert'
}

interface SignalMessage {
  type: SignalMessageType;
  payload: unknown;
  timestamp: number;
  priority: 'critical' | 'high' | 'medium' | 'low';
  persona_filter?: string[];  // Only deliver to specific personas
}

// Subscription with geographic and thematic filters
interface Subscription {
  userId: string;
  persona: string;
  geoFilter: {
    type: 'radius' | 'polygon' | 'countries';
    value: GeoJSON | string[];
  };
  themeFilter: string[];  // MILITARY, PROTEST, ECONOMIC, etc.
  minPriority: 'critical' | 'high' | 'medium' | 'low';
}
```

**Backpressure Handling:**
```typescript
class SignalStreamManager {
  private subscriptions: Map<string, Subscription> = new Map();
  private messageQueue: PriorityQueue<SignalMessage>;
  private readonly MAX_QUEUE_SIZE = 10000;
  private readonly BATCH_INTERVAL_MS = 100;

  async processIncomingSignal(signal: RawSignal): Promise<void> {
    // CPU-first processing
    const processed = await this.signalProcessor.process(signal);

    // Check for anomalies
    const anomaly = await this.anomalyDetector.check(processed);

    // Determine priority based on anomaly score and source reliability
    const priority = this.calculatePriority(processed, anomaly);

    // Create message
    const message: SignalMessage = {
      type: anomaly ? SignalMessageType.ANOMALY_DETECTED : SignalMessageType.SIGNAL_NEW,
      payload: { signal: processed, anomaly },
      timestamp: Date.now(),
      priority
    };

    // Fan-out to matching subscriptions
    for (const [userId, sub] of this.subscriptions) {
      if (this.matchesSubscription(processed, sub)) {
        await this.enqueueForUser(userId, message);
      }
    }
  }

  private async enqueueForUser(userId: string, message: SignalMessage): Promise<void> {
    const userQueue = this.getUserQueue(userId);

    // Backpressure: drop low-priority messages if queue is full
    if (userQueue.size() >= this.MAX_QUEUE_SIZE) {
      if (message.priority === 'low' || message.priority === 'medium') {
        this.metrics.increment('messages_dropped', { priority: message.priority });
        return;
      }
      // For high/critical: evict lowest priority message
      userQueue.evictLowest();
    }

    userQueue.enqueue(message);
  }
}
```

### 2.3 Offline-First Architecture

**Local Intelligence Cache:**
```typescript
// SQLite schema for offline operation
const OFFLINE_SCHEMA = `
  CREATE TABLE IF NOT EXISTS cached_signals (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB,  -- Float32Array serialized
    processed_json TEXT,
    source TEXT,
    timestamp INTEGER,
    synced INTEGER DEFAULT 0
  );

  CREATE TABLE IF NOT EXISTS local_briefings (
    id TEXT PRIMARY KEY,
    content_json TEXT NOT NULL,
    generated_at INTEGER,
    valid_until INTEGER,
    persona TEXT
  );

  CREATE TABLE IF NOT EXISTS pending_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_type TEXT,  -- 'signal_ack', 'annotation', 'bookmark'
    payload_json TEXT,
    created_at INTEGER,
    retry_count INTEGER DEFAULT 0
  );

  CREATE INDEX idx_signals_timestamp ON cached_signals(timestamp);
  CREATE INDEX idx_signals_source ON cached_signals(source);
  CREATE INDEX idx_pending_created ON pending_operations(created_at);
`;

class OfflineIntelligenceManager {
  private db: SQLiteDatabase;
  private syncInProgress = false;

  async cacheSignal(signal: ProcessedSignal): Promise<void> {
    await this.db.runAsync(
      `INSERT OR REPLACE INTO cached_signals
       (id, content, embedding, processed_json, source, timestamp, synced)
       VALUES (?, ?, ?, ?, ?, ?, 0)`,
      [
        signal.id,
        signal.content,
        this.serializeEmbedding(signal.embedding),
        JSON.stringify(signal.processed),
        signal.source,
        signal.timestamp.getTime()
      ]
    );
  }

  async getLocalSignals(
    since: Date,
    sources?: string[],
    limit: number = 100
  ): Promise<ProcessedSignal[]> {
    let query = `SELECT * FROM cached_signals WHERE timestamp > ?`;
    const params: unknown[] = [since.getTime()];

    if (sources?.length) {
      query += ` AND source IN (${sources.map(() => '?').join(',')})`;
      params.push(...sources);
    }

    query += ` ORDER BY timestamp DESC LIMIT ?`;
    params.push(limit);

    const rows = await this.db.getAllAsync(query, params);
    return rows.map(this.deserializeSignal);
  }

  async performLocalInference(): Promise<LocalBriefing> {
    // When offline, use WASM modules for CPU-only inference
    const signals = await this.getLocalSignals(
      new Date(Date.now() - 24 * 60 * 60 * 1000)
    );

    // Load WASM modules
    const signalProcessor = await loadWasmModule('signal_processor');
    const beliefFusion = await loadWasmModule('dempster_shafer');
    const anomalyDetector = await loadWasmModule('graph_laplacian');

    // Process signals locally
    const processed = signals.map(s => signalProcessor.process(s.content));
    const fused = beliefFusion.combine(processed);
    const anomalies = anomalyDetector.detect(fused);

    return {
      id: generateUUID(),
      content: this.formatLocalBriefing(fused, anomalies),
      generatedAt: new Date(),
      isOffline: true,
      confidence: 0.7  // Lower confidence for offline inference
    };
  }
}
```

### 2.4 WASM Module Architecture

**Rust Core Library Structure:**
```rust
// src/lib.rs - Core SIGIL algorithms compiled to WASM

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================
// DEMPSTER-SHAFER BELIEF FUSION
// ============================================================

#[derive(Serialize, Deserialize)]
pub struct BeliefMass {
    pub hypothesis: Vec<String>,
    pub mass: f64,
    pub source: String,
    pub reliability: f64,
}

#[derive(Serialize, Deserialize)]
pub struct FusionResult {
    pub beliefs: Vec<BeliefMass>,
    pub conflict: f64,
    pub needs_review: bool,
}

#[wasm_bindgen]
pub fn fuse_beliefs(masses_json: &str) -> Result<String, JsValue> {
    let masses: Vec<Vec<BeliefMass>> = serde_json::from_str(masses_json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut result = masses[0].clone();
    let mut total_conflict = 0.0;

    for source_masses in masses.iter().skip(1) {
        // Apply reliability discounting
        let discounted = discount_masses(source_masses);

        // Dempster combination
        let (combined, conflict) = dempster_combine(&result, &discounted)?;
        total_conflict = 1.0 - (1.0 - total_conflict) * (1.0 - conflict);
        result = combined;
    }

    let fusion_result = FusionResult {
        beliefs: result,
        conflict: total_conflict,
        needs_review: total_conflict > 0.5,
    };

    serde_json::to_string(&fusion_result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

fn dempster_combine(
    m1: &[BeliefMass],
    m2: &[BeliefMass]
) -> Result<(Vec<BeliefMass>, f64), JsValue> {
    let mut combined: std::collections::HashMap<Vec<String>, f64> =
        std::collections::HashMap::new();
    let mut conflict = 0.0;

    for b1 in m1 {
        for b2 in m2 {
            let intersection: Vec<String> = b1.hypothesis.iter()
                .filter(|h| b2.hypothesis.contains(h))
                .cloned()
                .collect();

            let product = b1.mass * b2.mass;

            if intersection.is_empty() {
                conflict += product;
            } else {
                let mut key = intersection.clone();
                key.sort();
                *combined.entry(key).or_insert(0.0) += product;
            }
        }
    }

    let k = 1.0 - conflict;
    if k < 0.01 {
        return Err(JsValue::from_str("Total conflict - sources completely contradictory"));
    }

    let result: Vec<BeliefMass> = combined.into_iter()
        .map(|(hypothesis, mass)| BeliefMass {
            hypothesis,
            mass: mass / k,
            source: "fused".to_string(),
            reliability: 1.0,
        })
        .collect();

    Ok((result, conflict))
}

// ============================================================
// TRANSFER ENTROPY (KSG ESTIMATOR)
// ============================================================

#[wasm_bindgen]
pub fn compute_transfer_entropy(
    source_json: &str,
    target_json: &str,
    lag: usize,
    k_neighbors: usize,
) -> Result<f64, JsValue> {
    let source: Vec<f64> = serde_json::from_str(source_json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let target: Vec<f64> = serde_json::from_str(target_json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    if source.len() != target.len() {
        return Err(JsValue::from_str("Time series must have equal length"));
    }

    let n = source.len() - lag - 1;
    if n < k_neighbors * 2 {
        return Err(JsValue::from_str("Insufficient data for KSG estimation"));
    }

    // Build joint and marginal spaces
    let mut joint_points: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut marginal_xy: Vec<[f64; 2]> = Vec::with_capacity(n);
    let mut marginal_y: Vec<f64> = Vec::with_capacity(n);

    for i in lag..source.len() - 1 {
        joint_points.push([target[i + 1], target[i], source[i - lag]]);
        marginal_xy.push([target[i + 1], target[i]]);
        marginal_y.push(target[i]);
    }

    // KSG estimator
    let te = ksg_estimator(&joint_points, &marginal_xy, &marginal_y, k_neighbors);

    Ok(te.max(0.0))  // TE cannot be negative
}

fn ksg_estimator(
    joint: &[[f64; 3]],
    marginal_xy: &[[f64; 2]],
    marginal_y: &[f64],
    k: usize,
) -> f64 {
    let n = joint.len();
    let mut sum = 0.0;

    for i in 0..n {
        // Find k-th nearest neighbor distance in joint space (Chebyshev norm)
        let epsilon = kth_neighbor_distance_3d(joint, i, k);

        // Count neighbors within epsilon in marginal spaces
        let n_xy = count_within_distance_2d(marginal_xy, i, epsilon);
        let n_y = count_within_distance_1d(marginal_y, i, epsilon);

        // Digamma contributions
        sum += digamma(n_xy as f64 + 1.0) + digamma(n_y as f64 + 1.0);
    }

    digamma(k as f64) - sum / n as f64
}

// ============================================================
// GRAPH LAPLACIAN SPECTRAL ANALYSIS
// ============================================================

#[derive(Serialize, Deserialize)]
pub struct SpectralAnomaly {
    pub spectral_shift: f64,
    pub fiedler_change: f64,
    pub affected_nodes: Vec<usize>,
    pub anomaly_score: f64,
}

#[wasm_bindgen]
pub fn detect_spectral_anomaly(
    baseline_adj_json: &str,
    current_adj_json: &str,
    k_eigenvalues: usize,
) -> Result<String, JsValue> {
    let baseline: Vec<Vec<f64>> = serde_json::from_str(baseline_adj_json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let current: Vec<Vec<f64>> = serde_json::from_str(current_adj_json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let n = baseline.len();

    // Compute normalized Laplacians
    let l_baseline = normalized_laplacian(&baseline);
    let l_current = normalized_laplacian(&current);

    // Eigendecomposition (using power iteration for top-k)
    let eig_baseline = top_k_eigenvalues(&l_baseline, k_eigenvalues);
    let eig_current = top_k_eigenvalues(&l_current, k_eigenvalues);

    // Spectral shift (L2 distance between spectra)
    let spectral_shift: f64 = eig_baseline.iter()
        .zip(eig_current.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    // Fiedler value (algebraic connectivity) - second smallest eigenvalue
    let fiedler_change = (eig_current[1] - eig_baseline[1]).abs();

    // Identify affected nodes via embedding distance
    let affected = find_displaced_nodes(&l_baseline, &l_current, k_eigenvalues, 0.1);

    // Combined anomaly score
    let anomaly_score = 0.6 * spectral_shift + 0.4 * fiedler_change;

    let result = SpectralAnomaly {
        spectral_shift,
        fiedler_change,
        affected_nodes: affected,
        anomaly_score,
    };

    serde_json::to_string(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

fn normalized_laplacian(adj: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = adj.len();
    let mut laplacian = vec![vec![0.0; n]; n];

    // Compute degree matrix
    let degrees: Vec<f64> = adj.iter()
        .map(|row| row.iter().sum())
        .collect();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                laplacian[i][j] = 1.0;
            } else if adj[i][j] > 0.0 && degrees[i] > 0.0 && degrees[j] > 0.0 {
                laplacian[i][j] = -adj[i][j] / (degrees[i] * degrees[j]).sqrt();
            }
        }
    }

    laplacian
}
```

---

## PART III: NOVEL ALGORITHMIC CONTRIBUTIONS

### 3.1 Intentionality Gradient Detection

A novel contribution: detecting when entities transition from reactive to intentional behavior patterns.

**Theoretical Foundation:**
Define the intentionality gradient I(e,t) for entity e at time t as:
```
I(e,t) = H(actions_t | context_{t-k:t}) - H(actions_t | context_{t-k:t}, goals_inferred)
```

High intentionality gradient indicates actions are better explained by inferred goals than by reactive context. This distinguishes strategic actors from reactive ones.

**Implementation:**
```typescript
interface IntentionalityScore {
  entity: string;
  gradient: number;          // I(e,t) value
  inferredGoals: string[];   // Most likely objectives
  confidence: number;        // Bayesian posterior
  transitionPoint?: Date;    // When gradient exceeded threshold
}

async function detectIntentionalityShift(
  entityId: string,
  actionHistory: Action[],
  contextHistory: Context[],
  windowSize: number = 30
): Promise<IntentionalityScore> {
  // Compute action entropy given context only
  const contextualEntropy = await computeConditionalEntropy(
    actionHistory.slice(-windowSize),
    contextHistory.slice(-windowSize)
  );

  // Infer possible goals using ILP
  const inferredGoals = await inferGoalsFromActions(
    actionHistory.slice(-windowSize),
    entityId
  );

  // Compute action entropy given context AND inferred goals
  const goalConditionedEntropy = await computeConditionalEntropy(
    actionHistory.slice(-windowSize),
    contextHistory.slice(-windowSize),
    inferredGoals
  );

  const gradient = contextualEntropy - goalConditionedEntropy;

  // Find transition point via change-point detection
  const transitionPoint = await detectChangePoint(
    actionHistory.map((a, i) => computeLocalIntentionality(a, contextHistory[i]))
  );

  return {
    entity: entityId,
    gradient,
    inferredGoals: inferredGoals.map(g => g.description),
    confidence: inferredGoals[0]?.probability || 0,
    transitionPoint
  };
}
```

### 3.2 Cascade Probability via Belief Propagation

For predicting whether a local event will cascade into regional instability:

**Factor Graph Model:**
```
              ┌─────────┐
     ┌───────►│ Event A │◄───────┐
     │        └────┬────┘        │
     │             │             │
┌────┴────┐   ┌────▼────┐   ┌────┴────┐
│ Factor  │   │ Factor  │   │ Factor  │
│  f_geo  │   │ f_econ  │   │ f_ethn  │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     ▼             ▼             ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│Region B │   │Region C │   │Region D │
└─────────┘   └─────────┘   └─────────┘
```

**Belief Propagation Messages:**
```
μ_{v→f}(x_v) = Π_{f' ∈ N(v)\{f}} μ_{f'→v}(x_v)

μ_{f→v}(x_v) = Σ_{x_{N(f)\{v}}} f(x_{N(f)}) Π_{v' ∈ N(f)\{v}} μ_{v'→f}(x_{v'})
```

**Cascade Probability:**
```typescript
interface CascadePrediction {
  originEvent: string;
  affectedRegions: Array<{
    region: string;
    probability: number;
    expectedLag: number;  // days
    primaryChannel: 'economic' | 'ethnic' | 'geographic' | 'political';
  }>;
  overallCascadeProbability: number;
  confidence: number;
}

class CascadePredictor {
  private factorGraph: FactorGraph;
  private historicalCascades: CascadeEvent[];

  async predictCascade(
    originEvent: ProcessedSignal,
    maxIterations: number = 100
  ): Promise<CascadePrediction> {
    // Initialize beliefs
    const beliefs = this.initializeBeliefs(originEvent);

    // Run loopy belief propagation
    for (let i = 0; i < maxIterations; i++) {
      const prevBeliefs = structuredClone(beliefs);

      // Variable to factor messages
      for (const [varId, variable] of this.factorGraph.variables) {
        for (const factor of variable.neighbors) {
          this.sendVariableToFactorMessage(varId, factor.id, beliefs);
        }
      }

      // Factor to variable messages
      for (const [factorId, factor] of this.factorGraph.factors) {
        for (const variable of factor.neighbors) {
          this.sendFactorToVariableMessage(factorId, variable.id, beliefs);
        }
      }

      // Check convergence
      if (this.hasConverged(prevBeliefs, beliefs)) break;
    }

    // Extract marginals
    return this.extractCascadePrediction(originEvent, beliefs);
  }

  private computeFactorPotential(
    factorType: string,
    assignments: Map<string, number>
  ): number {
    switch (factorType) {
      case 'geographic':
        return this.geographicProximityFactor(assignments);
      case 'economic':
        return this.economicIntegrationFactor(assignments);
      case 'ethnic':
        return this.ethnicTiesFactor(assignments);
      case 'political':
        return this.politicalAllianceFactor(assignments);
      default:
        return 1.0;
    }
  }
}
```

### 3.3 Regime Basin Strength via Lyapunov Exponents

Model nation-states as dynamical systems and compute stability via largest Lyapunov exponent.

**State Vector:**
```
x(t) = [economic_index, political_stability, social_cohesion, military_control, external_pressure]
```

**Dynamics (learned from historical data):**
```
x(t+1) = F(x(t)) + η(t)
```

**Largest Lyapunov Exponent:**
```
λ = lim_{n→∞} (1/n) Σ_{i=0}^{n-1} log||DF(x(i))||
```

Where DF is the Jacobian of the dynamics. λ > 0 indicates chaotic/unstable regime; λ < 0 indicates stable basin.

**Basin Strength:**
```
BasinStrength = exp(-λ) if λ > 0 else 1 - |λ|/max_λ
```

```typescript
interface RegimeStability {
  nationCode: string;
  lyapunovExponent: number;
  basinStrength: number;  // 0 = unstable, 1 = maximally stable
  attractor: 'stable' | 'periodic' | 'chaotic';
  transitionRisk: number;
  projectedTrajectory: StateVector[];
}

async function computeRegimeStability(
  nationCode: string,
  historicalStates: StateVector[],
  horizon: number = 90
): Promise<RegimeStability> {
  // Fit neural ODE to learn dynamics F
  const dynamics = await fitNeuralODE(historicalStates);

  // Compute Lyapunov exponent via QR decomposition method
  const lyapunov = await computeLyapunovQR(dynamics, historicalStates.slice(-100));

  // Classify attractor type
  const attractor = classifyAttractor(lyapunov);

  // Basin strength
  const basinStrength = lyapunov > 0
    ? Math.exp(-lyapunov)
    : 1 - Math.abs(lyapunov) / 2.0;

  // Project forward
  const trajectory = await projectTrajectory(
    dynamics,
    historicalStates[historicalStates.length - 1],
    horizon
  );

  // Transition risk = probability of exiting current basin
  const transitionRisk = await estimateTransitionProbability(
    dynamics,
    historicalStates[historicalStates.length - 1],
    horizon
  );

  return {
    nationCode,
    lyapunovExponent: lyapunov,
    basinStrength,
    attractor,
    transitionRisk,
    projectedTrajectory: trajectory
  };
}
```

---

## PART IV: PERFORMANCE & COST OPTIMIZATION

### 4.1 CPU-First Processing Budget

**Cost Model:**
```
TotalCost = C_cpu × T_cpu + C_llm × N_llm × TokensPerCall

Where:
- C_cpu ≈ $0.0001 per 1000 operations (compute)
- C_llm = $0.25 - $15 per 1M tokens (tier-dependent)
- Goal: Minimize N_llm while maintaining quality
```

**Decision Function:**
```typescript
type LLMTier = 'none' | 'haiku' | 'sonnet' | 'opus';

interface ProcessedSignal {
  content: string;
  anomalyScore: number;
  complexity: number;
  uncertainty: [number, number];  // Belief interval
  requiresNuance: boolean;
}

function decideLLMTier(signal: ProcessedSignal): LLMTier {
  const { anomalyScore, complexity, uncertainty, requiresNuance } = signal;
  const uncertaintyWidth = uncertainty[1] - uncertainty[0];

  // CPU-only: low anomaly, low complexity, narrow uncertainty
  if (anomalyScore < 0.3 && complexity < 0.4 && uncertaintyWidth < 0.2) {
    return 'none';
  }

  // Haiku: moderate signals, classification tasks
  if (anomalyScore < 0.6 && !requiresNuance) {
    return 'haiku';
  }

  // Sonnet: high anomaly or complex signals
  if (anomalyScore < 0.85 || (complexity > 0.7 && !requiresNuance)) {
    return 'sonnet';
  }

  // Opus: critical anomalies requiring strategic interpretation
  return 'opus';
}

// Expected distribution: 70% none, 20% haiku, 8% sonnet, 2% opus
// Expected cost reduction: 85-95% vs. always using Sonnet
```

### 4.2 Caching Strategy

**Multi-Level Cache:**
```
┌───────────────────────────────────────────────────┐
│ L1: In-Memory (LRU, 1000 items, <1ms)             │
│     - Recent signal embeddings                     │
│     - Active session briefings                     │
│     - Hot nation-state data                        │
├───────────────────────────────────────────────────┤
│ L2: Redis (100K items, <10ms)                      │
│     - Computed transfer entropy graphs             │
│     - Belief fusion results                        │
│     - Spectral baselines                           │
├───────────────────────────────────────────────────┤
│ L3: PostgreSQL (persistent, <100ms)                │
│     - Historical signals                           │
│     - Causal graph edges                           │
│     - Regime stability time series                 │
└───────────────────────────────────────────────────┘
```

**Cache Invalidation:**
```typescript
interface CachePolicy {
  key: string;
  ttl: number;  // seconds
  invalidateOn: string[];  // event types that invalidate this cache
}

const CACHE_POLICIES: CachePolicy[] = [
  {
    key: 'nation:stability:*',
    ttl: 3600,  // 1 hour
    invalidateOn: ['signal:critical', 'regime:change']
  },
  {
    key: 'causal:graph:*',
    ttl: 86400,  // 24 hours
    invalidateOn: ['causal:edge:new', 'causal:edge:removed']
  },
  {
    key: 'briefing:user:*',
    ttl: 300,  // 5 minutes
    invalidateOn: ['signal:high', 'anomaly:detected']
  }
];
```

### 4.3 Batch Processing for Historical Analysis

```typescript
interface BatchJob {
  id: string;
  type: 'causal_discovery' | 'stability_analysis' | 'cascade_simulation';
  parameters: Record<string, unknown>;
  priority: number;
  estimatedDuration: number;  // seconds
}

class BatchProcessor {
  private queue: PriorityQueue<BatchJob>;
  private workers: Worker[];
  private readonly MAX_CONCURRENT = 4;

  async submitJob(job: BatchJob): Promise<string> {
    this.queue.enqueue(job);
    this.processQueue();
    return job.id;
  }

  private async processQueue(): Promise<void> {
    while (this.queue.size() > 0 && this.activeWorkers() < this.MAX_CONCURRENT) {
      const job = this.queue.dequeue();
      this.executeJob(job);
    }
  }

  private async executeJob(job: BatchJob): Promise<void> {
    const worker = this.getAvailableWorker();

    switch (job.type) {
      case 'causal_discovery':
        await worker.runCausalDiscovery(job.parameters);
        break;
      case 'stability_analysis':
        await worker.runStabilityAnalysis(job.parameters);
        break;
      case 'cascade_simulation':
        await worker.runCascadeSimulation(job.parameters);
        break;
    }

    this.releaseWorker(worker);
    this.processQueue();
  }
}
```

---

## PART V: SECURITY & CLASSIFICATION HANDLING

### 5.1 Data Classification Tiers

```typescript
enum ClassificationLevel {
  UNCLASSIFIED = 0,
  FOUO = 1,          // For Official Use Only
  CONFIDENTIAL = 2,
  SECRET = 3,
  TOP_SECRET = 4
}

interface ClassifiedData<T> {
  payload: T;
  classification: ClassificationLevel;
  caveats: string[];  // e.g., ['NOFORN', 'ORCON']
  originatingAgency: string;
  declassifyOn: Date;
}

// Row-level security based on user clearance
const RLS_POLICY = `
  CREATE POLICY classification_access ON signals
    FOR SELECT
    USING (
      classification <= (
        SELECT clearance_level
        FROM user_clearances
        WHERE user_id = auth.uid()
      )
    );
`;
```

### 5.2 Audit Logging

```typescript
interface AuditEvent {
  timestamp: Date;
  userId: string;
  action: 'view' | 'export' | 'share' | 'annotate' | 'delete';
  resourceType: 'signal' | 'briefing' | 'graph';
  resourceId: string;
  classification: ClassificationLevel;
  ipAddress: string;
  userAgent: string;
  success: boolean;
  details?: Record<string, unknown>;
}

// Immutable audit log (append-only table)
const AUDIT_SCHEMA = `
  CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    event_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    hash TEXT GENERATED ALWAYS AS (
      encode(sha256(event_data::text::bytea), 'hex')
    ) STORED,
    prev_hash TEXT REFERENCES audit_log(hash)
  );

  -- Prevent updates/deletes
  CREATE RULE no_update AS ON UPDATE TO audit_log DO INSTEAD NOTHING;
  CREATE RULE no_delete AS ON DELETE TO audit_log DO INSTEAD NOTHING;
`;
```

---

## CONCLUSION: BUILD INSTRUCTIONS

To build SIGIL with this technical architecture:

1. **Initialize project**: `npx create-expo-app sigil --template tabs`
2. **Add WASM support**: Configure Metro bundler for .wasm files
3. **Implement core algorithms** in Rust, compile to WASM with wasm-pack
4. **Build offline-first SQLite cache** with expo-sqlite
5. **Implement Dempster-Shafer fusion** as primary intelligence combiner
6. **Add transfer entropy** for causal graph construction
7. **Integrate spectral anomaly detection** for network analysis
8. **Implement CPU-first pipeline** with LLM tier decision function
9. **Build real-time WebSocket layer** with geographic/thematic filtering
10. **Deploy backend** on Vercel Edge with Supabase data layer

The mathematical foundations ensure rigorous uncertainty quantification while the architecture optimizes for mobile performance and cost efficiency. The novel algorithms (intentionality gradient, cascade prediction, Lyapunov stability) provide analytical capabilities beyond standard intelligence tooling.
