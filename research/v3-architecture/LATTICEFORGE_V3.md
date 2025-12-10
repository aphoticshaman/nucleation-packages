# LatticeForge V3 Architecture

**Focus**: Geopolitical Intelligence Platform (not ARC Prize)

---

## V2 → V3 Evolution

```
V2 (Current):                    V3 (Target):
─────────────────────────────    ─────────────────────────────
Elle: Single LoRA adapter    →   Elle: MoDE (domain experts)
Guardian: Python validation  →   Guardian: Rust/WASM engine
CIC: Python computation      →   CIC: SIMD-accelerated Rust
3D Nav: Three.js only        →   3D Nav: WASM + WebGPU
Signals: Supabase polling    →   Signals: Edge + streaming
Briefings: On-demand         →   Briefings: Pre-computed + diff
```

---

## Core Components

### 1. ELLE V3: Domain-Routed Experts

```
┌─────────────────────────────────────────────────────────────────┐
│                        ELLE V3 (GPU)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: "Analyze Ukraine-Russia escalation"                    │
│              │                                                   │
│              ▼                                                   │
│   ┌──────────────────────────────────────┐                      │
│   │         DOMAIN ROUTER                 │                      │
│   │   Embedds input → selects experts    │                      │
│   └──────────────────────────────────────┘                      │
│              │                                                   │
│       ┌──────┼──────┬──────────┐                                │
│       ▼      ▼      ▼          ▼                                │
│   ┌──────┐┌──────┐┌──────┐┌──────┐                              │
│   │ POL  ││ ECON ││ MIL  ││ CYBER│  ← Domain Expert LoRAs      │
│   │Expert││Expert││Expert││Expert│    (64-rank each, ~32MB)     │
│   └──────┘└──────┘└──────┘└──────┘                              │
│       │      │      │          │                                 │
│       └──────┴──────┴──────────┘                                │
│              │                                                   │
│              ▼                                                   │
│   ┌──────────────────────────────────────┐                      │
│   │      FORMAT EXPERT (always on)       │                      │
│   │   Ensures JSON compliance            │                      │
│   └──────────────────────────────────────┘                      │
│              │                                                   │
│              ▼                                                   │
│   Output: {"political": "...", "military": "...", ...}          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Domain Experts**:

| Expert | Specialty | Training Data |
|--------|-----------|---------------|
| `elle-pol` | Political risk, elections, governance | GDELT political, ACLED |
| `elle-econ` | Markets, trade, sanctions | Financial news, IMF data |
| `elle-mil` | Military, defense, conflict | Jane's, OSINT, satellites |
| `elle-cyber` | Cyber threats, APTs, infrastructure | CVEs, threat intel feeds |
| `elle-health` | Pandemics, health security | WHO, outbreak data |
| `elle-format` | JSON structure, token efficiency | Synthetic format examples |

**Router Logic**:
```python
class DomainRouter(nn.Module):
    """Routes to 2-3 experts based on input content."""

    def forward(self, input_embeds):
        # Compute domain affinity scores
        scores = self.classifier(input_embeds.mean(dim=1))

        # Always include format expert
        scores[:, self.format_idx] += 10.0

        # Top-3 selection
        top_k = scores.topk(3, dim=-1)
        return top_k.indices, F.softmax(top_k.values, dim=-1)
```

---

### 2. GUARDIAN V3: Rust/WASM Reasoning Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                     GUARDIAN V3 (CPU/WASM)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    CORE ENGINE (Rust)                    │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │                                                          │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│   │  │   CIC    │  │  Phase   │  │ Cascade  │              │   │
│   │  │ Compute  │  │ Detect   │  │ Predict  │              │   │
│   │  │ F[T]=... │  │ L-G      │  │ SIR      │              │   │
│   │  └──────────┘  └──────────┘  └──────────┘              │   │
│   │                                                          │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│   │  │  JSON    │  │ Cluster  │  │Historical│              │   │
│   │  │ Validate │  │ Values   │  │ Correlate│              │   │
│   │  │ + Fix    │  │ 92% err↓ │  │ 500yr DB │              │   │
│   │  └──────────┘  └──────────┘  └──────────┘              │   │
│   │                                                          │   │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│   │  │   3D     │  │ Anomaly  │  │Confidence│              │   │
│   │  │  Nav     │  │ Detect   │  │ Bounds   │              │   │
│   │  │ Pathfind │  │ Outliers │  │ ≤0.95    │              │   │
│   │  └──────────┘  └──────────┘  └──────────┘              │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│         [Browser]       [Server]        [Edge]                  │
│          (WASM)         (Native)        (WASM)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Rust Implementation**:

```rust
// guardian-v3/src/lib.rs

pub mod cic;
pub mod phase;
pub mod cascade;
pub mod validate;
pub mod cluster;
pub mod nav;

/// Main Guardian engine
pub struct Guardian {
    cic: CICEngine,
    phase: PhaseDetector,
    cascade: CascadePredictor,
    validator: JSONValidator,
    clusterer: ValueClusterer,
    nav: NavEngine,
    historical: HistoricalDB,
}

impl Guardian {
    /// Process Elle's output, validate, enhance, return
    pub fn process(&self, elle_output: &str) -> Result<ProcessedBriefing, Error> {
        // 1. Validate JSON structure (< 0.1ms)
        let mut briefing: Briefing = self.validator.parse_and_fix(elle_output)?;

        // 2. Compute CIC metrics (< 1ms)
        let signals = self.fetch_current_signals();
        briefing.cic_f = self.cic.compute_f(&signals);
        briefing.phi = self.cic.compute_phi(&signals);

        // 3. Detect phase state (< 0.1ms)
        briefing.phase = self.phase.detect(signals.temperature(), signals.order());

        // 4. Check cascade risk (< 1ms)
        briefing.cascade_risk = self.cascade.predict(&signals);

        // 5. Find historical parallels (< 5ms)
        briefing.historical = self.historical.find_correlate(&signals);

        // 6. Apply confidence bounds (< 0.1ms)
        briefing.confidence = self.apply_epistemic_bounds(briefing.raw_confidence);

        Ok(ProcessedBriefing {
            briefing,
            validation_time_us: timer.elapsed_us(),
            guardian_intervention: self.validator.had_fixes(),
        })
    }
}

/// WASM bindings for browser
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_briefing(elle_json: &str) -> String {
    let guardian = Guardian::global();
    match guardian.process(elle_json) {
        Ok(result) => serde_json::to_string(&result).unwrap(),
        Err(e) => format!(r#"{{"error": "{}"}}"#, e),
    }
}
```

---

### 3. CIC ENGINE V3: SIMD-Accelerated

```rust
// guardian-v3/src/cic.rs

use std::simd::*;

pub struct CICEngine {
    lambda: f32,      // 0.3 - entropy weight
    gamma: f32,       // 0.25 - causality weight
    t_critical: f32,  // 0.7632
}

impl CICEngine {
    /// F[T] = Φ(T) - λ·H(T|X) + γ·C(T)
    /// SIMD-accelerated, processes 8 signals at once
    #[inline(always)]
    pub fn compute_f(&self, signals: &SignalBatch) -> f32 {
        let phi = self.compute_phi_simd(signals);
        let h = self.compute_entropy_simd(signals);
        let c = self.compute_causality(signals);

        phi - self.lambda * h + self.gamma * c
    }

    #[inline(always)]
    fn compute_phi_simd(&self, signals: &SignalBatch) -> f32 {
        let risks = signals.risks();
        let n = risks.len();

        // SIMD sum for mean
        let mut sum = f32x8::splat(0.0);
        for chunk in risks.chunks_exact(8) {
            sum += f32x8::from_slice(chunk);
        }
        let mean = sum.reduce_sum() / n as f32;

        // SIMD variance
        let mean_vec = f32x8::splat(mean);
        let mut var_sum = f32x8::splat(0.0);
        for chunk in risks.chunks_exact(8) {
            let v = f32x8::from_slice(chunk);
            let diff = v - mean_vec;
            var_sum += diff * diff;
        }
        let variance = var_sum.reduce_sum() / n as f32;

        // Φ = 1 - normalized_variance (higher when more integrated)
        1.0 - variance.min(1.0)
    }

    #[inline(always)]
    fn compute_entropy_simd(&self, signals: &SignalBatch) -> f32 {
        let probs = signals.normalized_risks();
        let n = probs.len();

        // H = -Σ p·log(p)
        let mut h_sum = f32x8::splat(0.0);
        let epsilon = f32x8::splat(1e-10);

        for chunk in probs.chunks_exact(8) {
            let p = f32x8::from_slice(chunk);
            let p_safe = p + epsilon;
            let log_p = p_safe.ln();  // SIMD ln not available, use scalar fallback
            h_sum += p * log_p;
        }

        let h = -h_sum.reduce_sum();
        h / (n as f32).ln()  // Normalize to [0, 1]
    }
}
```

---

### 4. 3D NAV V3: WASM + WebGPU

```rust
// guardian-v3/src/nav.rs

/// 3D navigation engine for threat visualization
pub struct NavEngine {
    graph: SpatialGraph,
    pathfinder: AStarPathfinder,
}

impl NavEngine {
    /// Find path from viewer to threat node
    pub fn find_path(&self, from: Vec3, to: Vec3) -> Option<Path3D> {
        self.pathfinder.find_path(&self.graph, from, to)
    }

    /// Compute threat propagation (cascade visualization)
    pub fn propagate_threat(&self, origin: NodeId, intensity: f32) -> PropagationMap {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = PropagationMap::new();

        queue.push_back((origin, intensity, 0));

        while let Some((node, current_intensity, depth)) = queue.pop_front() {
            if visited.contains(&node) || current_intensity < 0.01 {
                continue;
            }

            visited.insert(node);
            result.insert(node, PropagationState {
                intensity: current_intensity,
                depth,
                arrival_time: depth as f32 * 0.1,
            });

            // Propagate to neighbors with decay
            for (neighbor, edge_weight) in self.graph.neighbors(node) {
                let new_intensity = current_intensity * edge_weight * 0.8;
                queue.push_back((neighbor, new_intensity, depth + 1));
            }
        }

        result
    }
}

/// WebGPU compute shader for parallel propagation
#[cfg(feature = "webgpu")]
pub const PROPAGATION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> edges: array<Edge>;
@group(0) @binding(2) var<storage, read_write> intensities: array<f32>;

@compute @workgroup_size(256)
fn propagate(@builtin(global_invocation_id) id: vec3<u32>) {
    let node_idx = id.x;
    if (node_idx >= arrayLength(&nodes)) { return; }

    var new_intensity = 0.0;

    // Sum incoming edge contributions
    for (var i = 0u; i < arrayLength(&edges); i++) {
        if (edges[i].to == node_idx) {
            new_intensity += intensities[edges[i].from] * edges[i].weight * 0.8;
        }
    }

    intensities[node_idx] = max(intensities[node_idx], new_intensity);
}
"#;
```

---

### 5. SIGNALS V3: Edge-First Streaming

```rust
// guardian-v3/src/signals.rs

/// Edge-deployed signal processor
pub struct SignalProcessor {
    gdelt_buffer: RingBuffer<GDELTEvent, 10000>,
    aggregator: StreamAggregator,
    anomaly_detector: OnlineAnomalyDetector,
}

impl SignalProcessor {
    /// Process incoming GDELT event (runs at edge)
    pub fn ingest(&mut self, event: GDELTEvent) -> Option<Alert> {
        // 1. Add to buffer
        self.gdelt_buffer.push(event.clone());

        // 2. Update running statistics
        self.aggregator.update(&event);

        // 3. Check for anomalies (O(1) online algorithm)
        if let Some(anomaly) = self.anomaly_detector.check(&event) {
            return Some(Alert {
                severity: anomaly.severity,
                source: "gdelt",
                event_id: event.id.clone(),
                message: format!(
                    "Anomaly detected: {} (z-score: {:.2})",
                    event.title, anomaly.z_score
                ),
            });
        }

        None
    }

    /// Get current signal batch for CIC computation
    pub fn current_batch(&self) -> SignalBatch {
        SignalBatch {
            events: self.gdelt_buffer.recent(1000),
            avg_tone: self.aggregator.avg_tone(),
            event_count: self.aggregator.count(),
            risk_scores: self.aggregator.risk_distribution(),
        }
    }
}

/// Runs on Cloudflare Workers / Vercel Edge
#[cfg(feature = "edge")]
pub async fn edge_handler(req: Request) -> Response {
    let processor = PROCESSOR.get_or_init(SignalProcessor::new);

    match req.method() {
        Method::POST => {
            let event: GDELTEvent = req.json().await?;
            if let Some(alert) = processor.ingest(event) {
                // Push to real-time subscribers
                broadcast_alert(alert).await;
            }
            Response::ok("ingested")
        }
        Method::GET => {
            let batch = processor.current_batch();
            Response::json(&batch)
        }
    }
}
```

---

### 6. BRIEFINGS V3: Pre-computed + Diff

```rust
// guardian-v3/src/briefings.rs

/// Pre-computed briefing cache with diff updates
pub struct BriefingCache {
    current: RwLock<Briefing>,
    history: VecDeque<BriefingSnapshot>,
    update_interval: Duration,
}

impl BriefingCache {
    /// Get current briefing (instant, no GPU)
    pub fn get_current(&self) -> Briefing {
        self.current.read().clone()
    }

    /// Get diff since last request
    pub fn get_diff(&self, since: Timestamp) -> BriefingDiff {
        let current = self.current.read();

        // Find snapshot closest to `since`
        let baseline = self.history.iter()
            .rev()
            .find(|s| s.timestamp <= since)
            .map(|s| &s.briefing)
            .unwrap_or(&current);

        // Compute diff
        BriefingDiff {
            changed_fields: self.diff_fields(baseline, &current),
            new_alerts: current.alerts.iter()
                .filter(|a| a.timestamp > since)
                .cloned()
                .collect(),
            phase_changed: baseline.phase != current.phase,
            cic_delta: current.cic_f - baseline.cic_f,
        }
    }

    /// Background update loop (runs on server)
    pub async fn update_loop(&self, elle: &Elle, guardian: &Guardian) {
        loop {
            // 1. Get fresh signals
            let signals = fetch_signals().await;

            // 2. Check if update needed
            let current = self.current.read();
            if !self.needs_update(&current, &signals) {
                tokio::time::sleep(self.update_interval).await;
                continue;
            }
            drop(current);

            // 3. Generate new briefing (GPU)
            let raw = elle.generate(&signals).await;

            // 4. Process through Guardian (CPU)
            let processed = guardian.process(&raw)?;

            // 5. Update cache
            {
                let mut current = self.current.write();
                self.history.push_back(BriefingSnapshot {
                    timestamp: Timestamp::now(),
                    briefing: current.clone(),
                });
                *current = processed.briefing;
            }

            // Keep last 100 snapshots
            while self.history.len() > 100 {
                self.history.pop_front();
            }

            tokio::time::sleep(self.update_interval).await;
        }
    }
}
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LATTICEFORGE V3                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [Cloudflare Edge]                                              │
│   ├── Signal ingestion (GDELT, RSS)                             │
│   ├── Guardian WASM (validation, CIC compute)                   │
│   └── Briefing cache (pre-computed)                             │
│              │                                                   │
│              ▼                                                   │
│   [Vercel/Fly.io]                                               │
│   ├── Next.js frontend                                          │
│   ├── API routes                                                │
│   └── WebSocket for real-time                                   │
│              │                                                   │
│              ▼                                                   │
│   [RunPod/Modal]                                                │
│   ├── Elle V3 (GPU inference)                                   │
│   ├── Domain expert routing                                     │
│   └── Background briefing generation                            │
│              │                                                   │
│              ▼                                                   │
│   [Supabase]                                                     │
│   ├── PostgreSQL (signals, briefings)                           │
│   ├── Realtime subscriptions                                    │
│   └── Auth                                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Targets

| Operation | V2 | V3 Target | Where |
|-----------|----|-----------| ------|
| Briefing generation | 5s | 2s | GPU (RunPod) |
| Briefing retrieval | 2s | <50ms | Edge (cached) |
| CIC computation | 50ms | <1ms | Edge (WASM) |
| JSON validation | 10ms | <0.1ms | Edge (WASM) |
| 3D nav pathfind | 100ms | <10ms | Browser (WASM) |
| Signal ingestion | 500ms | <10ms | Edge |
| Phase detection | 20ms | <0.5ms | Edge (WASM) |

---

## Migration Checklist

```
□ Port CIC engine to Rust
□ Implement SIMD optimizations
□ Build WASM target
□ Deploy to Cloudflare Workers
□ Implement domain expert LoRAs
□ Train domain router
□ Add briefing cache layer
□ Migrate 3D nav to WASM
□ Set up edge signal processing
□ Benchmark and optimize
```

---

*"Make the common case fast, make the fast case invisible."*
