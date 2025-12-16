# LATTICEFORGE STRATEGIC CONTEXT

Persistent strategic intelligence for all LatticeForge development, design, R&D, and feature decisions. Every code commit, UI choice, and architecture decision should advance acquisition positioning.

---

## PRIME DIRECTIVE

**NO LLM. ANYWHERE. EVER.**

The entire value proposition is deterministic, auditable, reproducible intelligence that runs air-gapped on edge devices. Every feature must work without any LLM API call. The analyst provides interpretation. The platform provides signal.

If you find yourself suggesting LLM integration, STOP. Rethink. Find the math-only solution.

---

## MISSION

Sell LatticeForge for $10-50M within 12 months through technology acquisition or IP license.

**Target acquirers:** Recorded Future, Dataminr, Babel Street, Two Six Technologies, Sayari, LexisNexis Risk

**Value proposition:** While everyone else duct-tapes LLMs onto dashboards, LatticeForge is actual signal intelligence infrastructure. Transfer entropy for causal graphs. Dempster-Shafer for evidence fusion. Topological data analysis for early warning. All running in 118KB of WASM, all deterministic, all auditable.

---

## FOUNDER CONSTRAINTS

- Solo founder (Ryan), military veteran, technical background
- $1,000 starting capital
- Heavy AI tooling (Claude, GPT, etc.) for DEVELOPMENT only - not in product
- No VC network, no prior exits
- Will not work for anyone else - exit must be asset sale or IP license
- Needs cash flow within 90 days

---

## WHAT EXISTS (ALWAYS READ CODE BEFORE SUGGESTING CHANGES)

### Python Research Stack (`/packages/research/`)

| Module | Lines | Function |
|--------|-------|----------|
| `core/transfer_entropy.py` | 315 | Kraskov MI, causal graph construction |
| `fusion/dempster_shafer.py` | 455+ | Reliability-weighted DS fusion (Lockheed patent) |
| `fusion/value_clustering.py` | 350+ | Basin refinement, 92.1% error reduction |
| `attractor/streaming_tda.py` | 461 | Real-time persistent homology |
| `anomaly/phase_transition.py` | 397 | DPT early warning metrics |
| `inference/sparse_gp.py` | 497 | O(n) streaming Gaussian processes |
| `regime/msvar.py` | - | Markov-switching VAR |

### TypeScript Engine (`/packages/engine/`)

| Module | Lines | Function |
|--------|-------|----------|
| `fusion-engine.ts` | 470 | Multi-source fusion with WASM bridge |
| `formulas/phase-transition.ts` | 467 | Proprietary phase classification |
| `core/` | - | FFT, wavelets, statistics, time series |

### WASM Core (`/packages/research/rust/` → 118KB compiled)

- Persistence computation (TDA)
- CIC framework (Compression-Integration-Coherence)
- Gauge-theoretic value clustering
- Q-matrix regime analysis
- Geospatial attractor system (nation-level dynamics)
- Dempster-Shafer fusion
- Transfer entropy
- Markov chain simulation

### Data Sources (`/packages/social-pulse/src/sources/`)

- `gdelt.ts` - GDELT integration (free, no auth, 15-min updates)
- Official sources: CDC, FRED, SEC EDGAR

### API Routes (`/packages/web/app/api/`)

- `analyze/phase-transition/route.ts` - Landau-Ginzburg + Markov-switching
- `analyze/cascades/route.ts` - Cascade probability
- `compute/transfer-entropy/` - Causal graph edges
- `compute/nation-risk/` - Geopolitical risk scoring
- `ingest/gdelt/` - Real-time GDELT polling

---

## WHAT TO BUILD

### Immediate Priority: LLM-Free Demo Pipeline

1. **`/api/signal/analyze` endpoint** that chains:
   - GDELT ingest → Fusion Engine → Phase Transition Model → Structured JSON
   - Returns: phase state, transition probability, causal drivers, confidence
   - Zero LLM, zero external API cost

2. **Analyst Workbench page** (`/workbench`):
   - Region selector, timeframe selector
   - Phase transition probability gauge (0-100%)
   - Regime indicator (STABLE/VOLATILE/CRISIS)
   - Variance timeline chart
   - Causal driver ranking
   - Export buttons (CSV, JSON)
   - **NO PROSE. Numbers and charts only.**

3. **Alert system**:
   - `/api/alerts/webhook` - POST to external URL on threshold
   - Email alerts via SendGrid
   - Configurable thresholds per region

### Month 1-2: Productization

- Landing page with live WASM demo
- Pricing tiers ($49/$199/$999)
- Stripe integration
- User dashboard with saved regions/alerts

### Month 3-6: Scale

- Weekly intelligence brief (auto-generated PDF from template + numbers, NO LLM)
- API documentation
- Enterprise features (white-label, custom integrations)

### Month 7-12: Acquisition

- SBIR/STTR applications
- Case studies from customers
- Data room preparation
- Corp dev outreach

---

## WHAT NOT TO BUILD

- ❌ Any LLM integration (Elle, Guardian, LFBM are DEPRECATED)
- ❌ Natural language summaries or explanations
- ❌ Chat interfaces
- ❌ "AI-powered" anything
- ❌ Features requiring cloud inference
- ❌ Complex visualizations that distract from signal
- ❌ Mobile apps
- ❌ Social/collaboration features

---

## TECHNICAL PRINCIPLES

1. **Deterministic**: Same input → same output, every time
2. **Auditable**: Full provenance tracking, reproducible analysis
3. **Edge-deployable**: Must run air-gapped (WASM + static data)
4. **Low-cost**: Near-zero marginal cost per query
5. **Fast**: Milliseconds, not seconds
6. **Simple**: Complexity in the math, simplicity in the UX

---

## PRICING MODEL

| Tier | Price | Features | Target |
|------|-------|----------|--------|
| Free | $0 | 10 queries/day, watermarked | Tire-kickers |
| Analyst | $49/mo | Unlimited, clean exports, email alerts | Freelance OSINT |
| Professional | $199/mo | API access, webhooks, bulk export | Small consultancies |
| Enterprise | $999/mo | White-label, custom, SLA | Threat intel firms |

---

## EXPIRED PATENTS INCORPORATED (LEGAL TO USE)

| Patent | Owner | Expired | Function |
|--------|-------|---------|----------|
| US6944566B2 | Lockheed | Apr 2023 | Reliability-weighted DS fusion |
| US6909997B2 | Lockheed | Aug 2023 | Meta-fusion selection |
| US9805002B2 | IBM | - | Graph Laplacian anomaly |
| US8645304B2 | IBM | - | MS-VAR Bayesian LASSO |
| US8190549B2 | Honda | - | Online sparse GP |
| US8855431B2 | Stanford | - | Compressed sensing |
| US8112340B2 | S&P | - | Gaussian copula |
| US5857978A | Lockheed | 2011 | Information-theoretic MI |

---

## THE NARRATIVE (FOR ACQUIRERS)

> "While everyone else is duct-taping LLMs onto dashboards and calling it AI, LatticeForge built actual intelligence infrastructure.
>
> Transfer entropy for causal graphs. Dempster-Shafer for evidence fusion. Topological data analysis for early warning. All running in 118KB of WASM, all deterministic, all auditable.
>
> We incorporated methods from 8 expired defense patents - Lockheed, IBM, Honda - reimplemented in modern, deployable code.
>
> Palantir costs $141K/seat and requires consultants. We run on a Raspberry Pi.
>
> We're not selling artificial intelligence. We're selling actual intelligence."

---

## SUCCESS METRICS

| Milestone | Target |
|-----------|--------|
| Week 2 | Working `/api/signal/analyze` endpoint |
| Week 4 | Analyst Workbench page live |
| Month 2 | 10 paying customers |
| Month 3 | $2K MRR |
| Month 6 | $10K MRR |
| Month 9 | First acquisition conversation |
| Month 12 | Signed LOI or term sheet |

---

## DECISION FRAMEWORK

When evaluating any feature or change:

1. **Does it require LLM?** → Don't build it
2. **Does it add complexity without signal?** → Don't build it
3. **Does it delay time-to-revenue?** → Don't build it
4. **Will it impress an acquirer's technical team?** → Build it
5. **Does it prove the math works on real events?** → Build it
6. **Does it move toward $10K MRR?** → Prioritize it

---

## CORE ALGORITHMS TO LEVERAGE

### Signal Fusion Pipeline

```
GDELT → Normalize → Phase Detection (TDA) → DS Fusion → Value Clustering → Output
```

### Key Functions

- `transfer_entropy()` - Causal edge weights
- `additive_fusion()` - Reliability-weighted belief combination
- `value_clustering()` - Multi-source numeric convergence
- `basin_refinement()` - Find "Platonic Form" center
- `hybrid_fusion()` - DS + clustering combined
- `StreamingTDAMonitor.update()` - Real-time early warning

### Output Structure (JSON)

```json
{
  "region": "UKR",
  "timestamp": "2025-01-15T00:00:00Z",
  "phase": {
    "current": "VOLATILE",
    "transition_probability": 0.73,
    "confidence": 0.81
  },
  "regime": {
    "stable": 0.15,
    "volatile": 0.58,
    "crisis": 0.27
  },
  "causal_drivers": [
    {"source": "military", "weight": 0.34},
    {"source": "economic", "weight": 0.21}
  ],
  "alert_level": "WARNING",
  "provenance": {
    "sources": ["GDELT"],
    "method": "hybrid_fusion",
    "deterministic": true
  }
}
```

---

## REMEMBER

Every line of code should answer: **"Does this make LatticeForge more valuable to an acquirer?"**

If the answer is no, don't write it.
