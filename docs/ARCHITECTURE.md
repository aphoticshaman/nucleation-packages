# LatticeForge Architecture

> **Zero-LLM Geopolitical Signal Intelligence Platform**
>
> All analysis via deterministic computation. No inference calls. No API keys. No external ML dependencies.

## System Overview

```
                                    +------------------+
                                    |   Web UI (Next)  |
                                    |  packages/web/   |
                                    +--------+---------+
                                             |
                              +--------------+--------------+
                              |                             |
                    +---------v---------+         +---------v---------+
                    | /api/intel-briefing|         | /api/us-brief     |
                    | (Vercel Edge)      |         | (Vercel Edge)     |
                    +---------+---------+         +---------+---------+
                              |                             |
                              +-------------+---------------+
                                            |
                              +-------------v-------------+
                              |      Supabase (Postgres)  |
                              |  - nations (state vectors)|
                              |  - country_signals        |
                              |  - learning_events        |
                              |  - briefs                 |
                              +-------------+-------------+
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
          +---------v---------+   +---------v---------+   +---------v---------+
          |  ACLED Adapter    |   |  UCDP Adapter     |   | ReliefWeb Adapter |
          |  (conflict.py)    |   |  (conflict.py)    |   |  (conflict.py)    |
          +-------------------+   +-------------------+   +-------------------+
                    |                       |                       |
                    +-----------+-----------+-----------+-----------+
                                |
                    +-----------v-----------+
                    |   UniversalSignal     |
                    |   (Normalized Format) |
                    +----------+------------+
                               |
                    +----------v------------+
                    |  Dempster-Shafer      |
                    |  Evidence Fusion      |
                    +-----------------------+
```

## Core Principles

### 1. Zero-LLM Architecture
- **NO external inference calls** - all analysis via threshold-based math
- **NO API keys for ML services** - eliminates vendor lock-in and cost volatility
- **Deterministic outputs** - same inputs always produce same results
- **Auditable reasoning** - every conclusion traceable to source data

### 2. Data Flow

```
External APIs → Adapters → UniversalSignal → Fusion → State Vectors → Briefings
     ↓              ↓            ↓              ↓           ↓            ↓
  Raw JSON    Normalized    Typed Schema   Combined    nations DB   Prose Output
```

### 3. Nation State Vectors

Each nation has a computed state vector:

| Field | Type | Description |
|-------|------|-------------|
| `basin_strength` | float [0,1] | Institutional resilience / regime stability |
| `transition_risk` | float [0,1] | Probability of regime change |
| `regime` | int [1-5] | Polity classification |

### 4. Phase Detection

Based on state vectors, nations are classified into phases:

| Phase | basin_strength | transition_risk | Interpretation |
|-------|----------------|-----------------|----------------|
| CRYSTALLINE | > 0.7 | < 0.2 | Stable, low volatility |
| SUPERCOOLED | > 0.5 | 0.2-0.4 | Stable but sensitive to shocks |
| NUCLEATING | 0.3-0.5 | 0.4-0.6 | Early transition indicators |
| PLASMA | < 0.3 | > 0.6 | Active instability |
| ANNEALING | increasing | decreasing | Post-crisis stabilization |

## Component Details

### Web Layer (`packages/web/`)

**Framework**: Next.js 14 (App Router)
**Runtime**: Vercel Edge Functions
**Cache**: Upstash Redis (L2) + In-memory Map (L1)

Key endpoints:
- `POST /api/intel-briefing` - 26-category geopolitical briefing
- `POST /api/us-brief` - US economic regime analysis

### Edge Functions (`supabase/functions/`)

**Runtime**: Deno (Supabase Edge)
**Auth**: Supabase JWT + API key tier validation

- `intel-brief` - Enterprise-tier briefing generation
- `us-brief` - Economic indicator analysis

### Ingest Layer (`packages/research/ingest/`)

**Language**: Python 3.11+
**Pattern**: Adapter per data source

Data sources:
- **ACLED** - Armed Conflict Location & Event Data (requires free API key)
- **UCDP** - Uppsala Conflict Data Program (no auth, academic)
- **ReliefWeb** - UN OCHA humanitarian reports (no auth)

### Evidence Fusion

**Method**: Dempster-Shafer Theory
**Implementation**: `value_clustering.py`

```python
def dempster_combine(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
    """Combine two mass functions using Dempster's rule."""
    # Handles conflict between sources via normalization
```

## Data Model

### Core Tables

```sql
-- Nation state vectors (computed daily)
CREATE TABLE nations (
    code TEXT PRIMARY KEY,        -- ISO 3166-1 alpha-3
    name TEXT NOT NULL,
    basin_strength FLOAT,         -- [0,1] institutional resilience
    transition_risk FLOAT,        -- [0,1] regime change probability
    regime INTEGER,               -- [1-5] polity classification
    updated_at TIMESTAMPTZ
);

-- Economic/social indicators
CREATE TABLE country_signals (
    id UUID PRIMARY KEY,
    country_code TEXT REFERENCES nations(code),
    country_name TEXT,
    indicator TEXT,               -- gdp_growth, inflation, unemployment, etc.
    value FLOAT,
    year INTEGER,
    updated_at TIMESTAMPTZ
);

-- Normalized events from all sources
CREATE TABLE learning_events (
    id UUID PRIMARY KEY,
    session_hash TEXT,            -- Source identifier (e.g., 'acled_ingest')
    domain TEXT,                  -- Country or region code
    data JSONB,                   -- UniversalSignal payload
    timestamp TIMESTAMPTZ
);
```

### UniversalSignal Schema

```typescript
interface UniversalSignal {
    id: string;                   // UUID
    source: string;               // 'acled' | 'ucdp' | 'reliefweb' | ...
    event_type: string;           // Normalized event category
    location: {
        country_code: string;     // ISO 3166-1 alpha-3
        region?: string;
        coordinates?: [number, number];
    };
    timestamp: string;            // ISO 8601
    severity: number;             // [0,1] normalized severity
    confidence: number;           // [0,1] source reliability
    payload: Record<string, unknown>;  // Source-specific data
}
```

## Deployment

### Vercel (Web)
```bash
cd packages/web && vercel --prod
```

### Supabase (Edge Functions)
```bash
supabase functions deploy intel-brief
supabase functions deploy us-brief
```

### Environment Variables

**Required for Web:**
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`

**Required for Ingest (Python):**
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `ACLED_API_KEY` (free registration)

**NOT Required (Zero-LLM):**
- ~~ANTHROPIC_API_KEY~~
- ~~OPENAI_API_KEY~~
- ~~LFBM_ENDPOINT~~
- ~~RUNPOD_API_KEY~~

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Briefing generation | < 100ms | Deterministic template |
| Cache hit (L1) | < 1ms | In-memory Map |
| Cache hit (L2) | 5-50ms | Upstash Redis |
| Inference cost | $0.00 | Zero LLM calls |
| Cold start | < 500ms | Edge runtime |

## Security Model

### Tier-Based Access

| Tier | Access |
|------|--------|
| Consumer | Cached briefings only |
| Pro | Fresh generation on cache miss |
| Enterprise | On-demand + Supabase intel-brief |

### Data Protection

- Session hashing for anonymization
- Quantized risk scores (5-point buckets) to prevent enumeration
- No PII in learning events
- API key validation on every request

## Future Considerations

### Rust/WASM Candidates
- Dempster-Shafer fusion (heavy math)
- Signal processing in ingest pipeline
- Vector similarity computations

### Additional Data Sources
- GDELT (media tone analysis)
- World Bank indicators
- IMF economic data
- Satellite imagery (if budget allows)

---

*Last updated: 2024-12*
*Architecture version: 2.0 (Zero-LLM)*
