# LatticeForge Demo Walkthrough

**Duration:** 5-7 minutes
**Audience:** Potential acquirers, enterprise prospects
**Goal:** Showcase zero-LLM value prop, transparent methodology, enterprise-readiness

---

## Pre-Demo Checklist

- [ ] Browser: Chrome/Firefox, incognito mode recommended
- [ ] Screen: 1920x1080 or higher, dark mode enabled
- [ ] Demo account: `demo@latticeforge.ai` logged in
- [ ] Supabase: Seed data loaded (cascade matrix, country signals, learning events)
- [ ] Cache: Warm (run `/api/intel-briefing` once before demo)

---

## Demo Script

### 1. Landing Page (30 seconds)

**URL:** `https://latticeforge.vercel.app/`

**Key Points:**
- "Zero-LLM Architecture" - no inference costs, no hallucinations
- "190+ nations monitored" - comprehensive coverage
- "Sub-100ms latency" - edge-deployed caching
- Click "View Live Demo" to proceed

**Talking Points:**
> "What you're seeing is a geopolitical intelligence platform that fundamentally differs from
> competitors. We use zero LLM inference for analysis - everything is deterministic templates
> and evidence fusion. This means predictable costs, no hallucinations, and full audit trails."

---

### 2. Consumer Dashboard (90 seconds)

**URL:** `/app`

**Actions:**
1. Show the Intel Briefing panel on the left
2. Toggle between "Basic" → "Analyst" → "Expert" views (top right)
3. Switch presets: Global → NATO → BRICS → Hot Spots
4. Show the interactive map with stability colors
5. Click "Causal" tab to show causal topology view

**Talking Points:**
> "The consumer dashboard provides 26-category intelligence briefings. Notice the skill level
> toggle - we built progressive disclosure so analysts can see methodology details while
> executives get summaries. Every piece of content traces back to source data."

> "The causal view shows transfer entropy flows between global events - this is Dempster-Shafer
> evidence fusion, not LLM generation. Click any node to see the evidence chain."

---

### 3. Enterprise Dashboard (90 seconds)

**URL:** `/dashboard`

**Actions:**
1. Click "Signals" in sidebar - show live feed with source filters
2. Click "Alerts" - show high-risk signal detection
3. Click "Cascades" - show cross-domain propagation patterns
4. Click "Doctrine" - show rule registry (key differentiator)

**Talking Points:**
> "Enterprise customers get API access to our signal feeds. GDELT news, USGS seismic,
> World Bank macro - all ingested and normalized. The freshness indicators show exactly
> when data was last updated."

> "The Doctrine Registry is our key differentiator. Enterprise customers can inspect the
> exact rules governing intelligence computation. They can even propose changes and run
> shadow evaluations to see how new parameters would affect historical assessments."

---

### 4. Doctrine Registry Deep Dive (60 seconds)

**URL:** `/dashboard/doctrine`

**Actions:**
1. Filter by category (signal_interpretation, analytic_judgment)
2. Click "Run Shadow Evaluation" on any doctrine
3. Show the divergence rate calculation

**Talking Points:**
> "This is unprecedented transparency. Traditional intelligence products are black boxes -
> you get an assessment but can't see how it was computed. Here, you can inspect and even
> modify the rules. The shadow evaluation shows exactly how a parameter change would affect
> historical assessments before you commit."

---

### 5. Admin Health & Architecture (45 seconds)

**URL:** `/admin/health`

**Actions:**
1. Show API health checks (all green)
2. Point out scheduled jobs and cache warm status
3. Mention zero-LLM cost structure

**Talking Points:**
> "Our operations team sees real-time health. All endpoints are edge-deployed. The zero-LLM
> architecture means our marginal cost per request is essentially zero - just Supabase reads
> and edge compute. No GPU inference costs, no unpredictable API bills."

---

### 6. Pricing & Tier Summary (30 seconds)

**URL:** `/pricing`

**Actions:**
1. Show four tiers: Observer → Operational → Integrated → Stewardship
2. Highlight Stewardship tier with Doctrine Registry access

**Talking Points:**
> "We're positioned for enterprise with four tiers. The key unlock at Stewardship tier is
> Doctrine Registry access - that's our enterprise moat. No competitor offers this level
> of transparency."

---

## Q&A Preparation

**Common Questions:**

1. **"How do you ensure accuracy without LLMs?"**
   > "We use deterministic templates with Dempster-Shafer evidence fusion. Every assessment
   > traces to source data. Accuracy is calibrated - we publish explicit confidence intervals
   > and never claim certainty we don't have."

2. **"What's your data moat?"**
   > "The moat is methodology, not data. GDELT and World Bank are public. Our value is the
   > fusion algorithms, doctrine registry, and transparent computation. Competitors can copy
   > data; they can't easily replicate our transparency guarantee."

3. **"How does this scale?"**
   > "Edge-deployed with 4-tier caching. L1 is in-memory (60s), L2 is Redis (10min), L3 is
   > Supabase. Sub-100ms p99 at any scale. No GPU capacity planning needed."

4. **"What's the acquisition thesis?"**
   > "Zero-LLM is contrarian now but becomes obvious when LLM costs matter. We're the only
   > provider offering full methodology transparency. Enterprise customers increasingly
   > demand audit trails - we're built for that future."

---

## Post-Demo

- Offer free trial signup
- Share API documentation: `/docs/api`
- Provide architecture overview: `/docs/architecture`
- Schedule technical deep-dive if interested in Stewardship tier
