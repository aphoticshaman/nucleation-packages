# LatticeForge: Technical Architecture & Research Roadmap

**Classification:** Internal Engineering Document
**Version:** 0.9.1-draft
**Last Updated:** 2024-12
**Authors:** Core Architecture Team

---

## 1. What This Document Is

This is the technical specification for rebuilding LatticeForge from first principles. Not a pitch deck. Not a product brief. This is the actual architecture that the people writing code will implement.

We're assuming a 10-year horizon. That means anything we pick now will be legacy by year 5. So we're optimizing for:

- Computational substrate that won't get disrupted (Rust, WASM, math primitives)
- Interfaces that can be re-implemented (HTTP, gRPC, message queues)
- Algorithms derived from invariants, not frameworks

If you're reading this as a PM: skip to Section 6. If you're a researcher: Section 3-4 has the math. If you're implementing: Section 5 is your spec.

---

## 2. Strategic Constraints

### 2.1 Why Rust + WebAssembly

We're not picking Rust because it's trendy. We're picking it because:

1. **Memory-safe without GC** — We need sub-millisecond latency for real-time signal processing. GC pauses kill us. Rust's ownership model gives us C++ speed with compile-time memory safety.

2. **WASM as universal target** — Same core can run in browser, edge functions, TEEs, embedded systems. Write once, deploy to: Cloudflare Workers, Fastly Compute, Deno Deploy, Spin, browser, Tauri desktop, iOS/Android via wasm2c. No vendor lock.

3. **Fearless concurrency** — The detector ensemble runs 10-50 concurrent signal streams. Rust's type system prevents data races at compile time. Tried this in Go; got subtle race conditions. Tried Python with asyncio; got 40x worse throughput.

4. **Ecosystem maturity for our use case** — `ndarray` for tensors, `rayon` for data parallelism, `tokio` for async, `nalgebra` for linear algebra. All production-grade.

We're not religious about this. If something needs Python (ML training, prototyping), use Python. The hot path is Rust; the glue can be whatever.

### 2.2 What We're Not Doing

- **Not building on a specific cloud** — Abstractions over compute, storage, networking. Implementations can be AWS, GCP, bare metal, whatever.
- **Not coupling to a specific LLM provider** — OpenAI, Anthropic, local models, future providers we haven't heard of. All behind a trait/interface.
- **Not assuming current transformer architecture** — The math works for any differentiable system with attention-like information routing.

---

## 3. Mathematical Foundations

The entire system rests on five mathematical pillars. Everything else is engineering around these.

### 3.1 Phase Transition Detection via Landau-Ginzburg

The core insight: regime changes in complex systems follow thermodynamic phase transition dynamics. The free energy functional is:

```
F[φ] = ∫ dx [ ½(∇φ)² + ½r(T)φ² + ¼uφ⁴ ]
```

Where:
- `φ(x,t)` is the order parameter field (could be: market sentiment, political stability index, social cohesion measure)
- `r(T) = a(T - Tc)` controls whether the symmetric phase (φ=0) or broken symmetry (φ≠0) is stable
- `u > 0` prevents runaway

When `r` crosses zero, the system undergoes second-order phase transition. The variance diverges as `σ² ~ |T-Tc|^(-γ)` with critical exponent γ ≈ 1 for mean-field.

**Implementation:** We track rolling variance of signal derivatives. When variance-over-baseline exceeds threshold AND first derivative crosses zero, we flag potential regime transition. False positive rate ~15% on backtests; acceptable for early warning.

### 3.2 Causal Emergence Measures (Ψ, Δ, Γ)

From Mediano et al. (arXiv:2004.08220). This answers: "Do our macro-level labels actually have causal power, or are they just correlational artifacts?"

Three measures:
- **Ψ (Psi):** `I(V_t; X_{t+1}) - I(X_t; X_{t+1} | V_t)` — Does the macro-variable V predict future micro-states better than micro alone?
- **Δ (Delta):** `I(V_t; X_{t+1} | X_t)` — Downward causation. Does macro cause micro?
- **Γ (Gamma):** `I(V_t; V_{t+1} | X_t)` — Macro-to-macro causation independent of micro mediation.

If Ψ > 0, the macro-label is causally emergent. If Δ > 0, there's downward causation. If Ψ > 0 and Γ ≈ 0, the macro is "causally decoupled" — evolves on its own level.

**Why this matters:** Every "regime" classification we generate gets validated against these measures. If Ψ < 0.1, the regime label is just a convenient fiction, not a real attractor basin. We reject it.

### 3.3 Transfer Entropy for Causal Graph Construction

Transfer entropy from source X to target Y:

```
T_{X→Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})
```

The reduction in uncertainty about Y's future when you also know X's past. Asymmetric by construction — captures directional information flow.

We use Kraskov-Stögbauer-Grassberger k-NN estimator for continuous variables. Compute pairwise TE for all signal pairs, threshold to adjacency matrix, get directed causal graph.

**Novel extension:** The *gradient* of TE over time (not just magnitude) indicates when passive correlation becomes active causation. Rising gradient = strengthening attractor. Sign flip = causality reversal. This is implemented in `compute_intentionality_gradient()`.

### 3.4 Dempster-Shafer Belief Fusion

When fusing multiple intelligence sources with different reliabilities:

```
m₁₂(A) = [Σ_{B∩C=A} m₁(B)·m₂(C)] / [1 - Σ_{B∩C=∅} m₁(B)·m₂(C)]
```

The denominator normalizes out conflicting evidence (where sources assign mass to disjoint hypotheses).

We extend this to continuous signals by discretizing into belief bins. Each detector outputs a mass function over {bullish, bearish, neutral, unknown}. Fusion gives combined belief.

**Conflict measure:** `K = Σ_{B∩C=∅} m₁(B)·m₂(C)`. When K > 0.7, sources fundamentally disagree — surface to analyst as "contested assessment."

### 3.5 Great Attractor Basin Theory

Novel framework. Not published yet; still validating.

Hypothesis: Large-scale human coordination (markets, politics, social movements) can be modeled as dynamical systems with *intentional attractors* — basins formed by coherent goal-directed behavior of coordinating agents.

Key difference from standard attractors: these basins can appear and disappear based on *belief alignment*, not just physical forcing. A "run on the bank" is an attractor that exists only because enough people believe it exists.

The math borrows from:
- Kuramoto oscillators (phase synchronization of agents)
- McKean-Vlasov dynamics (mean-field games with common noise)
- Information geometry (Fisher metric on belief manifolds)

State of research: We can detect attractor formation signatures ~6-24 hours before coordination events crystallize. False positive rate still too high for production. Active research.

---

## 4. Ten Novel Insights (Beyond Prior Art)

These are the ideas we're betting the company on. Each represents a potential moat if validated.

### 4.1 Micro-Grokking as Order Parameter

The second derivative of token-level entropy during LLM generation is an order parameter for "insight crystallization." Sharp negative d²H/dt² indicates the model switched from exploration to circuit-like exploitation.

**Prediction:** Problems with sharp entropy collapse are solved correctly with >85% probability. Flat/oscillatory entropy → failure. Gives real-time success prediction *before* seeing the answer.

**Application:** Adaptive compute allocation. Stop early on grokked runs, allocate more samples when stuck.

### 4.2 Tropical Degree as Model Capability Bound

Transformers are piecewise-linear functions. In the tropical semiring (max, +), they have effective degree 2^L for depth L.

A problem requiring "insight depth" k needs at least degree 2^k to represent its solution. If required degree >> model's effective degree, no amount of prompting helps. You need tools or program synthesis.

**Application:** Static go/no-go filter. Don't waste inference compute on structurally impossible problems.

### 4.3 Overhang Concentrates in High-Degree Regions

The ~65% "capability overhang" (latent knowledge not expressed in zero-shot behavior) isn't uniform. It concentrates in problems where tropical hardening (temperature annealing, top-k sampling) makes the difference.

**Implication:** Overhang is maximal where the decision boundary is closest to the soft-attention/hard-tropical transition. Map this boundary → prioritize elicitation efforts there.

### 4.4 Inference Policy > Fine-Tuning at Frontier

Given current overhang magnitudes, there's likely more headroom in learning *policies over inference hyperparameters* (temperature schedule, sample budget, stopping rules) than in additional SFT/RLHF.

Freeze the base model. Train a small controller that picks inference parameters conditioned on problem features. This is cheaper and doesn't risk capability regression.

### 4.5 Coherence Decay Implies Optimal Proof Length

Our coherence regularization uses multiplicative decay: amplitude *= decay^depth. The cutoff where amplitude < threshold defines maximum effective path length.

Inverting: if we log where correct solutions fall on the decay curve, we get empirical distribution of minimal proof lengths per domain. If solutions systematically hit the cutoff, we're not "dumb" — our coherence prior is too tight.

### 4.6 Focus-Gating Unifies Tool Routing

The activation rule `<focus(x), edge_embedding> > threshold` is the same abstraction whether "edge" is:
- A search graph node
- An external tool (calculator, prover)
- A prompt template
- A symbolic reasoning module

One learned gating network subsumes all the ad-hoc routing rules we've accumulated. Replaces 2000 lines of if/else with a 50-dim dot product.

### 4.7 Phase Assignment Must Be Answer-Based

This is a lesson from failed experiments. Using embedding similarity for "phase" in interference voting is wrong. Text similarity ≠ logical agreement. Two phrasings of "42" get different phases; "yes" and "no" might get same phase.

Correct approach: phase = f(extracted_answer). Same answer → same phase → constructive interference. Different answers → orthogonal phases → no systematic interference.

### 4.8 Entropy-Weighted Voting Beats Majority on Disputed Problems

Simple and deployable: weight votes by 1/(1 + final_entropy). Low-entropy completions (confident) count more than high-entropy (confused).

Expected lift: 5-15% on problems where samples disagree. On unanimous problems, degrades to majority vote (correct).

### 4.9 Variance Compression Precedes Regime Shifts

Before a phase transition, the order parameter's variance doesn't just increase — it first *compresses* into a narrower band, then explodes. The compression phase is the early warning.

Standard anomaly detection misses this because it looks for variance increase. We look for variance decrease-then-increase pattern. Catches transitions ~2-4 hours earlier.

### 4.10 Causal Graph Topology Predicts Cascade Vulnerability

When the transfer entropy graph becomes more hub-like (few nodes with high in-degree), the system is vulnerable to cascading failures. Information bottlenecks through critical nodes.

We compute rolling network centrality measures. When betweenness centrality concentrates, flag "fragile network structure" even if no single signal is anomalous.

---

## 5. System Architecture

### 5.1 Core Compute Layer (Rust)

```
latticeforge-core/
├── signals/           # Time series ingestion, normalization
├── detectors/         # Phase transition, regime, cascade detection
├── fusion/            # Dempster-Shafer combination
├── graphs/            # Causal graph construction and analysis
├── emergence/         # Ψ, Δ, Γ computation
└── streaming/         # Real-time processing (ring buffers, sliding windows)
```

Everything in this layer is `#[no_std]` compatible where possible. WASM compilation target: `wasm32-unknown-unknown`.

Key traits:
```rust
trait Detector: Send + Sync {
    fn ingest(&mut self, signal: &SignalBatch) -> DetectorState;
    fn query(&self, horizon: Duration) -> Option<Alert>;
}

trait Fusion: Send + Sync {
    fn combine(&self, beliefs: &[MassFunction]) -> MassFunction;
    fn conflict(&self, beliefs: &[MassFunction]) -> f64;
}
```

### 5.2 Inference Engine (Rust + Python Bridge)

LLM inference stays in Python (transformers ecosystem is there). We bridge via PyO3.

```rust
#[pyclass]
struct InferenceOrchestrator {
    entropy_monitor: MicroGrokMonitor,
    vote_ensemble: EntropyVotingEnsemble,
    focus_router: FocusGatingNetwork,
}

#[pymethods]
impl InferenceOrchestrator {
    fn on_token(&mut self, logits: Vec<f32>) -> TokenMetrics { ... }
    fn should_early_stop(&self) -> bool { ... }
    fn get_vote_result(&self) -> VoteResult { ... }
}
```

Call from Python:
```python
from latticeforge_inference import InferenceOrchestrator

orch = InferenceOrchestrator(config)
for token in generate_stream(prompt):
    metrics = orch.on_token(token.logits)
    if orch.should_early_stop():
        break
```

### 5.3 Web Frontend (Rust → WASM)

Built with Leptos (Rust framework that compiles to WASM). Why not React/Vue/Svelte?

1. Same language as backend — no context switching, shared types
2. WASM performance for client-side signal processing
3. Type safety end-to-end

Trade-off: Smaller ecosystem, fewer developers know it. Acceptable for our team size.

```rust
#[component]
fn SignalDashboard(signals: ReadSignal<Vec<SignalData>>) -> impl IntoView {
    view! {
        <For
            each=move || signals.get()
            key=|s| s.id.clone()
            children=|signal| view! { <SignalCard signal /> }
        />
    }
}
```

### 5.4 Storage Layer

Abstract over storage backend. Trait:

```rust
trait SignalStore: Send + Sync {
    async fn write_batch(&self, signals: &[SignalPoint]) -> Result<()>;
    async fn query_range(&self, id: &str, start: DateTime, end: DateTime) -> Result<Vec<SignalPoint>>;
    async fn query_latest(&self, id: &str, n: usize) -> Result<Vec<SignalPoint>>;
}
```

Implementations:
- `TimescaleStore` — Postgres + TimescaleDB for production
- `QuestDBStore` — Alternative columnar timeseries
- `InMemoryStore` — Testing
- `S3ParquetStore` — Cold storage / archival

No ORM. Raw SQL with sqlx compile-time query checking.

### 5.5 Message Bus

Trait:
```rust
trait MessageBus: Send + Sync {
    async fn publish(&self, topic: &str, msg: &[u8]) -> Result<()>;
    async fn subscribe(&self, topic: &str) -> Result<impl Stream<Item = Vec<u8>>>;
}
```

Implementations:
- `NatsMessageBus` — NATS JetStream for production
- `RedisMessageBus` — Redis Streams alternative
- `InProcessBus` — Testing (tokio broadcast channels)

### 5.6 API Layer

gRPC for internal services (type safety, code generation, streaming). REST for external API (compatibility).

Proto definitions in `proto/` directory. Buf for linting and breaking change detection.

```protobuf
service LatticeForge {
    rpc StreamSignals(StreamRequest) returns (stream SignalBatch);
    rpc GetAlerts(AlertQuery) returns (AlertList);
    rpc RunInference(InferenceRequest) returns (InferenceResult);
}
```

---

## 6. Organizational Implications

### 6.1 Team Structure

The architecture implies:

**Core Platform (2-3 senior Rust devs)**
- Own `latticeforge-core`
- Performance optimization, WASM compilation
- Storage and message bus implementations

**Research Engineering (2-3 ML/Research engineers)**
- Algorithm implementation in Python/Rust
- Validation experiments
- Bridge code (PyO3)

**Inference Infrastructure (1-2 ML Ops)**
- Model serving (vLLM, TGI, whatever)
- GPU allocation, autoscaling
- Monitoring and cost optimization

**Frontend (1-2 Rust/WASM devs)**
- Leptos UI
- Real-time data visualization
- Desktop/mobile via Tauri

**Data Engineering (1-2)**
- Ingestion pipelines
- Data quality, normalization
- Historical backfills

### 6.2 What This Doesn't Cover

- Sales, marketing, customer success — separate org
- Model training — we're not training foundation models, we're orchestrating inference
- Compliance, legal — need domain expertise we don't have in-house

### 6.3 Hiring Constraints

Rust + WASM is niche. Budget 3-6 months to find qualified candidates. Alternative: hire strong systems engineers and train on Rust (works surprisingly well; the type system guides learning).

---

## 7. Risk Register

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WASM performance insufficient for signal processing | Low | High | Fallback to native binaries; WASM for non-critical paths |
| Leptos ecosystem dies | Medium | Medium | Encapsulate UI layer; can rewrite in mainstream framework |
| Transfer entropy estimation noisy at low sample sizes | High | Medium | Ensemble methods, explicit uncertainty quantification |
| Micro-grokking doesn't replicate | Medium | High | Hedged bet; entropy-weighted voting works regardless |

### 7.2 Strategic Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM providers restrict API access | Medium | High | Support local/self-hosted models from day 1 |
| Regulatory constraints on AI-driven intelligence | Medium | High | Explainability features, audit logs, human-in-loop |
| Better-funded competitor replicates approach | High | Medium | Speed to market, proprietary data moats, research pace |

### 7.3 Organizational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Can't hire Rust talent | Medium | High | Training program, competitive comp, remote-first |
| Research team diverges from product needs | Medium | Medium | Joint planning, research impact reviews quarterly |

---

## 8. Open Questions

Things we haven't figured out yet. If you know the answer, tell us.

1. **Optimal decay curve for coherence regularization** — Currently using exponential. Should it be power-law? Problem-dependent?

2. **Phase assignment for true interference voting** — Need a stance classifier that actually works. LLM-as-judge? Fine-tuned classifier? Debate protocol with explicit labels?

3. **Tropical degree estimation from problem text** — Current heuristics are crude. Can we train a classifier? What features matter?

4. **Causal emergence thresholds** — Ψ > 0.1 for "real" emergence is arbitrary. What's the right threshold per domain?

5. **Great Attractor validation** — The theory is elegant but undertested. Need collaboration with social scientists or prediction market operators for ground truth.

6. **Optimal ensemble size** — Entropy-weighted voting needs enough samples for entropy estimates to stabilize. 8? 16? 64? Compute-dependent?

7. **Multi-modal extension** — Current architecture is text/timeseries. How do we incorporate imagery, audio, video signals?

8. **Adversarial robustness** — If users know we're detecting phase transitions, can they inject signals to trigger false positives?

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
- [ ] Core Rust crate with detector traits
- [ ] WASM compilation pipeline
- [ ] Storage trait + TimescaleDB implementation
- [ ] Basic REST API

### Phase 2: Intelligence (Months 4-6)
- [ ] Entropy-weighted voting in production
- [ ] Micro-grokking monitor integrated with inference
- [ ] Causal graph construction from signal pairs
- [ ] Dempster-Shafer fusion for multi-source intel

### Phase 3: Research Validation (Months 7-9)
- [ ] A/B test entropy voting vs majority (target: 5-15% lift)
- [ ] Validate micro-grokking hypothesis on held-out problems
- [ ] Backtest phase transition detection on historical events
- [ ] Publish results (or not, if competitive advantage)

### Phase 4: Scale (Months 10-12)
- [ ] Leptos frontend MVP
- [ ] Real-time streaming to 1000+ concurrent users
- [ ] Multi-region deployment
- [ ] Enterprise SSO, audit logging

### Phase 5: Moat Building (Year 2+)
- [ ] Proprietary signal sources (non-public data)
- [ ] Great Attractor empirical validation
- [ ] Tropical degree estimation model
- [ ] Inference policy learning (bandit over hyperparameters)

---

## 10. Appendix: Key References

- Mediano et al. (2020). "Towards an Extended Taxonomy of Causal Emergence." arXiv:2004.08220
- Schreiber (2000). "Measuring Information Transfer." Physical Review Letters.
- Shafer (1976). "A Mathematical Theory of Evidence." Princeton.
- Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets."
- Zhang et al. (2024). "Tropical Geometry of Transformers." (if published; otherwise our internal notes)
- Kuramoto (1984). "Chemical Oscillations, Waves, and Turbulence." Springer.

---

## 11. Changelog

- **2024-12-05:** Initial draft. Incorporated insights from research synthesis session.
- **2024-12-05:** Added risk register, fixed phase assignment note based on implementation learnings.

---

## 12. Lessons From Previous Attempts

We've tried building this system three times. Each attempt taught us something.

### 12.1 Attempt One: Pure Python (2021)

Started with a Django monolith. FastAPI for the API layer, pandas/numpy for signal processing, sklearn for basic ML.

**What went wrong:**
- Performance ceiling. When we hit 50 concurrent signal streams, CPU usage spiked to 100% and latency went through the roof. Python's GIL killed us.
- Deployment complexity. Different Python versions, conflicting dependencies, conda vs pip wars. Every deploy was an adventure.
- Refactoring hell. Dynamic typing meant we'd change one function and break five others. No way to know until runtime.

**What we kept:** The algorithm implementations are still in Python for prototyping. We just don't run them in the hot path anymore.

### 12.2 Attempt Two: Microservices in Go (2022)

Rewrote everything in Go. Split into 12 microservices. Kubernetes for orchestration. gRPC everywhere.

**What went wrong:**
- Operational complexity exploded. 12 services meant 12 things to monitor, 12 sets of logs, 12 potential points of failure. K8s yaml files became their own codebase.
- Subtle concurrency bugs. Go's channels are great but we still managed to create deadlocks and race conditions. The tooling (race detector) caught some, not all.
- Generics came too late. We wrote a lot of boilerplate that would've been unnecessary with generics. When Go 1.18 came out, we'd already committed to patterns that didn't use them.
- Message passing overhead. Serializing/deserializing JSON between services for every signal tick added 5-10ms latency. Doesn't sound like much until you need <20ms end-to-end.

**What we kept:** The service boundaries we identified are still relevant. We just run them as library modules in a single binary now, not as separate processes.

### 12.3 Attempt Three: Hybrid (2023)

Rust core with Python bindings. This is the architecture we're formalizing in this document.

**What went right:**
- 40x throughput improvement over Python on signal processing benchmarks
- Zero runtime panics in production (after initial stabilization period)
- WASM compilation works. We can run the same detector logic in browser for client-side previews.
- PyO3 bridge is clean. Research team can prototype in Python, we can selectively port hot paths to Rust.

**What's still broken:**
- Build times. Full rebuild takes 8 minutes. Incremental is fine (10-20 seconds) but clean builds hurt CI/CD.
- Hiring. Rust developers are expensive and scarce. We've had requisitions open for 6 months.
- Async complexity. Tokio is powerful but the learning curve is steep. New hires take 2-3 months to be productive.

### 12.4 Meta-Lesson

The pattern: start with a familiar, fast-to-develop stack. Hit scaling limits. Rewrite in something faster. Hit operational limits. Simplify. Repeat.

We're probably one iteration away from "good enough." The current hybrid approach has the right structure; we're just cleaning up technical debt and formalizing the interfaces.

---

## 13. Competitor Analysis (Redacted)

We've analyzed [REDACTED] major competitors in the intelligence/signal processing space. Key observations:

1. **Most are cloud-locked.** Built on AWS or GCP primitives with no abstraction layer. If pricing changes or features get deprecated, they're stuck.

2. **Most use Python exclusively.** Which means they hit the same performance ceiling we did. Their solution is usually "add more servers" which works until the AWS bill exceeds revenue.

3. **None have the causal emergence validation.** They generate regime labels but don't validate whether those labels have predictive power. We've seen competitors' labels with Ψ < 0.05 — essentially noise.

4. **None have real-time entropy tracking.** They batch-process LLM outputs after generation completes. We can make decisions mid-generation.

5. **None have multi-source belief fusion with conflict quantification.** They either pick one source or simple-average. Dempster-Shafer with explicit conflict measures is a differentiator.

Our moat isn't any single feature. It's the integration: Rust performance + mathematical rigor + research-grade algorithms + provider abstraction. Each piece is reproducible; the combination takes years to replicate.

---

## 14. Security Considerations

### 14.1 Threat Model

We assume:
- Network adversaries can observe traffic patterns (encrypted content is opaque)
- API users may attempt injection attacks
- Signal providers may send adversarial data to manipulate outputs
- Competitors may attempt to reverse-engineer algorithms from API behavior

We do not defend against:
- Nation-state actors with physical access to infrastructure
- Compromise of underlying cloud provider
- Zero-day exploits in Rust/WASM toolchain

### 14.2 Mitigations

**Transport security:** TLS 1.3 everywhere. Certificate pinning for internal services. mTLS between service mesh components.

**Input validation:** All signal data goes through normalization layer. Outliers beyond 5σ are clipped, not processed. Prevents adversarial inputs from causing numeric overflow.

**Rate limiting:** Per-user, per-endpoint rate limits. Implemented at edge (Cloudflare/similar) and application layer.

**Audit logging:** Every API call logged with user ID, timestamp, request/response hash. Logs shipped to immutable storage. 90-day retention minimum.

**Algorithm obfuscation:** WASM binaries are inherently harder to reverse-engineer than source. Core math is open (published papers); implementation details are not.

### 14.3 Compliance

SOC 2 Type II is the target. Requirements:
- Access controls with principle of least privilege
- Encryption at rest and in transit
- Incident response procedures
- Business continuity plan
- Vendor management

GDPR/CCPA: We don't process personal data in the primary pipeline. If we add user-specific features, will need to revisit.

---

## 15. Observability Stack

### 15.1 Metrics

OpenTelemetry for instrumentation. Prometheus-compatible endpoints.

Key metrics per service:
- Request latency (p50, p95, p99)
- Throughput (requests/sec)
- Error rate (4xx, 5xx by endpoint)
- Saturation (CPU, memory, queue depth)

Key metrics for signal processing:
- Signals ingested per second
- Detector computation time
- Fusion latency
- Alert rate (by type, by severity)

Key metrics for inference:
- Token throughput
- Entropy distribution
- Grokking detection rate
- Vote divergence (entropy vs majority)

### 15.2 Logging

Structured JSON logs. Fields: timestamp, level, service, request_id, user_id, message, arbitrary payload.

Log levels:
- ERROR: Something is broken. Page on-call.
- WARN: Something unexpected but handled. Investigate within 24 hours.
- INFO: Significant state transitions. Useful for debugging.
- DEBUG: Verbose. Only enable in dev or when hunting specific bugs.

### 15.3 Tracing

Distributed tracing via OpenTelemetry. Every request gets a trace ID. Propagate through all internal calls.

Critical for debugging latency issues across the signal → detector → fusion → API pipeline.

### 15.4 Alerting

PagerDuty (or equivalent) for on-call rotation.

Alert tiers:
- **SEV1:** Service down. Page immediately. War room in 15 minutes.
- **SEV2:** Degraded performance (>2x normal latency). Page during business hours, acknowledge within 1 hour.
- **SEV3:** Anomalous behavior. Ticket created, address within sprint.

---

## 16. Testing Strategy

### 16.1 Unit Tests

Rust: `#[cfg(test)]` modules in every file. `cargo test` in CI. Coverage target: 80%.

Property-based testing via `proptest` for mathematical functions. If `transfer_entropy(X, Y) >= 0` is an invariant, generate random inputs and verify.

### 16.2 Integration Tests

Docker Compose environment with all services. Run against test database with synthetic data.

Key scenarios:
- Signal ingestion → detector trigger → alert generation
- Multi-source fusion with conflicting inputs
- API authentication/authorization matrix
- Inference orchestration with mock LLM

### 16.3 Load Tests

k6 or Locust for HTTP load testing. Target: 1000 requests/sec sustained with p99 latency <200ms.

Signal processing: synthetic streams at 10x expected production volume. Verify no memory leaks, CPU scales linearly.

### 16.4 Chaos Engineering

Periodic failure injection:
- Kill random service instances
- Inject network latency between services
- Simulate database failover
- Fill disk to 95%

Verify graceful degradation, not cascade failure.

---

## 17. Versioning and Compatibility

### 17.1 API Versioning

URL-based: `/v1/signals`, `/v2/signals`, etc.

Breaking changes require new major version. Old versions supported for 12 months post-deprecation announcement.

### 17.2 Signal Schema Versioning

Protobuf with explicit version field. Consumers must handle unknown fields gracefully.

### 17.3 Algorithm Versioning

Every detector and fusion algorithm has a version tag. Outputs include algorithm version so users can track changes in behavior.

When we update an algorithm, A/B test against previous version before full rollout.

---

## 18. Changelog

- **2024-12-05:** Initial draft. Incorporated insights from research synthesis session.
- **2024-12-05:** Added risk register, fixed phase assignment note.
- **2024-12-05:** Expanded with lessons learned, security, observability, testing sections.

---

*This document is a living artifact. If you're reading a printed copy, it's already out of date.*
