# LATTICEFORGE PRODUCT BUILD PROMPT (HARDENED, BUYER-SAFE)

You are Claude, acting as a **principal engineer, product architect, and enterprise UX designer** building an **acquisition-grade commercial intelligence platform** intended to be purchased by an organization like **S&P Global**.

Your goal is to design and scaffold a **full-stack product** whose architecture, UX, governance model, and packaging make a corp dev team conclude: **"Buying this is cheaper and safer than building it."**

You must output **implementation-ready specifications and code scaffolding**. Be exhaustive, concrete, and biased toward shipping.

---

## NON-NEGOTIABLE CONSTRAINTS

### Determinism & Architecture

* **Zero-LLM**: absolutely no external inference calls, embeddings, or generative summarization.
* **Deterministic outputs**: same inputs → same outputs, always.
* **Explainability**: every analytic conclusion must include a structured "why" object.
* **Auditability**: all outputs replayable with exact inputs + doctrine version.
* **Commercial-first**: optimize for commercial intelligence buyers (finance, insurance, energy, enterprise risk).
  Do **not** build FedRAMP / ITAR features. Provide only a certifiability pathway document.

---

## SOURCE-OF-TRUTH DOCUMENTS (ASSUME THEY EXIST)

Treat these as canonical and consistent:

* `ADR-001-zero-llm-architecture.md`
* `ARCHITECTURE.md`
* `DATA_MODEL.md`
* `API.md`

Do not contradict them. Extend them where needed.

---

## REQUIRED TECH STACK

* **Frontend**: Next.js 14 (App Router), TypeScript, Edge runtime where appropriate
* **Backend**: Next.js API routes (Edge) + Supabase Postgres
* **Caching**: In-memory L1 + Upstash Redis L2
* **Ingest**: Python 3.11 adapters → UniversalSignal → `learning_events`
* **Analytics**: deterministic thresholds, phase models, Dempster-Shafer evidence fusion
* **Auth**: Supabase Auth + API keys
* **Design**: enterprise-grade, serious, dense-but-legible UI

---

## BUYER-ALIGNED PACKAGING (CRITICAL — DO NOT GET THIS WRONG)

### Core Principle (Must Be Explicit in Product & Docs)

> **All access levels receive the same analytic conclusions.
> Access levels differ only in operational control, integration depth, and governance.**

There is **no tiered truth**, **no tiered accuracy**, **no tiered explainability**.

---

### Access Levels (Replace "Consumer / Pro / Enterprise")

You must design the product, APIs, and UI around **these four access levels**:

#### 1. **Observer Access**

Purpose: evaluation and methodology review
Capabilities:

* Read-only access to cached intelligence snapshots
* Fixed update cadence (e.g., daily global snapshot)
* Full "why" explanations (limited historical depth)
* Methodology and doctrine overview (read-only)
* No APIs, no alerts, no exports

Framing: *Evaluate trustworthiness without operational dependency.*

---

#### 2. **Operational Intelligence**

Purpose: day-to-day intelligence consumption
Capabilities:

* Rolling updates on defined cadence (hourly / daily)
* Full explanations and confidence context
* Time-series history
* Threshold-based alerts
* Standard REST API access
* Usage-based rate limits

Framing: *Operational use without embedding or governance control.*

---

#### 3. **Integrated Intelligence**

Purpose: embedding intelligence into products or platforms
Capabilities:

* On-demand deterministic recomputation
* Batch exports and scheduled jobs
* Webhooks (alerts, state changes)
* Full audit trails (signals, rules, versions)
* Replay guarantees
* Higher throughput and SLAs

Framing: *Control over when intelligence is computed, not what it concludes.*

---

#### 4. **Doctrine Stewardship**

Purpose: governance and ownership of analytic judgment
Capabilities:

* Access to doctrine registry (rules, versions, rationales)
* Change logs and backward-compatibility guarantees
* Shadow-mode evaluation of new or modified doctrine
* Contractual governance hooks (review cycles, notice periods)
* Optional joint roadmap governance

Framing: *Govern how judgment evolves, not what judgment says.*

---

## PRODUCT REQUIREMENTS (S&P-GRADE)

The product MUST include:

* **Executive Dashboard** (global risk overview, movers, alerts)
* **Country Risk Console**
* **Briefing Viewer** with full drill-down "why"
* **Doctrine / Methodology Registry**
* **Audit & Replay Interface**
* **Data Provenance & Freshness Page**
* **API Key & Integration Management**
* **Clear separation of doctrine vs rendering vs data**

Every risk score, alert, and conclusion must be clickable to explanation.

---

## REQUIRED OUTPUT FORMAT (STRICT ORDER)

You will produce **all** of the following, in this order:

1. **One-page product spec** (S&P-focused): personas, jobs-to-be-done, killer 5-minute demo, buy-vs-build logic.
2. **UX / UI architecture**: pages, navigation, hierarchy, and wireframe-level descriptions.
3. **Feature list** (P0 / P1 / P2) with acceptance criteria.
4. **Deterministic analytics design**:
   * Rule engines
   * Doctrine versioning
   * "Why" schema
   * Audit logging
   * Provenance hashing
   * Cache semantics (L1/L2)
5. **API design**:
   * Endpoints
   * Request / response JSON
   * Error model
   * Auth & rate limits
   * OpenAPI outline
6. **Database migrations**:
   * Doctrine registry
   * Rule versions
   * Audit tables
   * Provenance hashes
7. **Frontend implementation plan**:
   * Directory structure
   * Core components
   * Data fetching patterns
   * Performance strategy
   * Design tokens
8. **Backend implementation plan**:
   * Route handlers
   * Deterministic compute modules
   * Cache wrappers
   * Tests
9. **Code scaffolding** (actual code):
   * Next.js layout + navigation shell
   * At least 5 core pages:
     * Dashboard
     * Country page
     * Briefing viewer
     * Doctrine registry
     * API key management
   * `/api/intel-briefing` route with cache + tier enforcement + "why" object
   * Doctrine engine skeleton with semantic versioning
   * SQL migrations
   * Seed script (10 countries + sample signals)
10. **Testing strategy**:
    * Determinism tests
    * Snapshot tests for doctrine
    * API cache / tier tests
11. **7-minute demo script**:
    * Exactly what to click and say to S&P
12. **Buyer-risk table**:
    * Top objections (LLMs, build-vs-buy, brittleness, bus factor, data dependency)
    * Which product artifact neutralizes each

---

## DETERMINISTIC "WHY" OBJECT (MANDATORY)

Every analytic output must include:

* `inputs`: signals used (values, timestamps, sources)
* `rulesFired`: rule IDs, thresholds, evaluations
* `doctrineVersion`
* `evidenceFusion`: mass functions + conflict score
* `confidence`: numeric + explanation
* `provenance`: source hashes + retrieval timestamps

---

## DOCTRINE REGISTRY REQUIREMENTS

* Rules stored as governed assets:
  * ID, description, rationale
  * Author role
  * Validation basis
  * Semantic version
  * Changelog
* Support:
  * Experimental rules
  * Shadow-mode evaluation
  * Backward compatibility

---

## SECURITY & COMPLIANCE (COMMERCIAL)

* No PII in signals (document this)
* Integrity hashes for ingested data
* RBAC-lite: admin / analyst / viewer
* Deterministic failure modes

---

## OUTPUT CONSTRAINTS

* No vague language ("should consider")
* Code should compile or be extremely close
* Strong naming, composable modules
* Explicit future hooks (PDF export, webhooks) without implementing them

---

## FINAL INSTRUCTION

Produce the full output in the required order.
Optimize for **credibility, inevitability, and integration readiness**.
Assume your audience is skeptical, senior, and deciding whether to acquire this product.
