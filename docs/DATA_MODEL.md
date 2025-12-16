# LatticeForge Data Model

> Schema documentation for Supabase/Postgres tables and TypeScript interfaces.

## Entity Relationship Diagram

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│     nations     │       │  country_signals │       │  learning_events│
├─────────────────┤       ├──────────────────┤       ├─────────────────┤
│ code (PK)       │◄──────│ country_code (FK)│       │ id (PK)         │
│ name            │       │ id (PK)          │       │ session_hash    │
│ basin_strength  │       │ country_name     │       │ domain          │
│ transition_risk │       │ indicator        │       │ data (JSONB)    │
│ regime          │       │ value            │       │ timestamp       │
│ updated_at      │       │ year             │       └─────────────────┘
└─────────────────┘       │ updated_at       │
         │                └──────────────────┘
         │
         │        ┌──────────────────┐       ┌─────────────────┐
         │        │   nation_risk    │       │     briefs      │
         │        ├──────────────────┤       ├─────────────────┤
         └───────►│ iso_a3 (FK)      │       │ id (PK)         │
                  │ name             │       │ content (TEXT)  │
                  │ overall_risk     │       │ summary         │
                  │ political_risk   │       │ signals_snapshot│
                  │ economic_risk    │       │ model           │
                  │ social_risk      │       │ created_at      │
                  └──────────────────┘       └─────────────────┘

┌─────────────────┐       ┌──────────────────┐
│    api_keys     │       │   user_profiles  │
├─────────────────┤       ├──────────────────┤
│ key_hash (PK)   │       │ id (PK, FK auth) │
│ client_tier     │       │ display_name     │
│ user_id (FK)    │───────│ tier             │
│ created_at      │       │ created_at       │
│ last_used       │       └──────────────────┘
│ revoked         │
└─────────────────┘
```

---

## Core Tables

### nations

Primary table for nation state vectors. Updated daily via ingest pipeline.

```sql
CREATE TABLE nations (
    code TEXT PRIMARY KEY,              -- ISO 3166-1 alpha-3 (e.g., 'USA', 'CHN')
    name TEXT NOT NULL,                 -- Full country name
    basin_strength FLOAT DEFAULT 0.5,   -- [0,1] Institutional resilience
    transition_risk FLOAT DEFAULT 0.3,  -- [0,1] Regime change probability
    regime INTEGER DEFAULT 3,           -- [1-5] Polity classification
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for risk-based queries
CREATE INDEX idx_nations_transition_risk ON nations(transition_risk DESC);
CREATE INDEX idx_nations_basin_strength ON nations(basin_strength);
```

#### Column Descriptions

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `code` | TEXT | ISO 3166-1 | Primary key, 3-letter country code |
| `name` | TEXT | - | Human-readable country name |
| `basin_strength` | FLOAT | [0, 1] | Higher = more stable institutions |
| `transition_risk` | FLOAT | [0, 1] | Higher = more likely regime change |
| `regime` | INTEGER | [1, 5] | 1=Autocracy, 3=Mixed, 5=Democracy |

#### Example Data

```sql
INSERT INTO nations (code, name, basin_strength, transition_risk, regime) VALUES
('USA', 'United States', 0.82, 0.15, 5),
('CHN', 'China', 0.71, 0.22, 1),
('UKR', 'Ukraine', 0.35, 0.68, 4),
('RUS', 'Russia', 0.58, 0.45, 2);
```

---

### country_signals

Economic and social indicators by country. Ingested from World Bank, IMF, etc.

```sql
CREATE TABLE country_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code TEXT REFERENCES nations(code),
    country_name TEXT NOT NULL,
    indicator TEXT NOT NULL,            -- 'gdp_growth', 'inflation', 'unemployment', etc.
    value FLOAT NOT NULL,
    year INTEGER,
    source TEXT,                        -- Data source identifier
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_signals_country ON country_signals(country_code);
CREATE INDEX idx_signals_indicator ON country_signals(indicator);
CREATE INDEX idx_signals_updated ON country_signals(updated_at DESC);
```

#### Supported Indicators

| Indicator | Unit | Source |
|-----------|------|--------|
| `gdp_growth` | % annual | World Bank |
| `inflation` | % annual | IMF |
| `unemployment` | % labor force | ILO |
| `debt_to_gdp` | % | IMF |
| `current_account` | % GDP | World Bank |
| `foreign_reserves` | USD billions | Central banks |
| `population_growth` | % annual | UN |
| `gini_index` | 0-100 | World Bank |

---

### learning_events

Normalized events from all data sources. Used for trend analysis and model training.

```sql
CREATE TABLE learning_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_hash TEXT NOT NULL,         -- Source identifier (e.g., 'acled_ingest')
    domain TEXT,                        -- Country code or region
    data JSONB NOT NULL,                -- UniversalSignal payload
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for time-series queries
CREATE INDEX idx_events_session ON learning_events(session_hash);
CREATE INDEX idx_events_domain ON learning_events(domain);
CREATE INDEX idx_events_timestamp ON learning_events(timestamp DESC);
CREATE INDEX idx_events_data ON learning_events USING GIN(data);
```

#### Session Hash Convention

| Pattern | Source |
|---------|--------|
| `acled_ingest` | ACLED conflict data |
| `ucdp_ingest` | UCDP violence data |
| `reliefweb_ingest` | UN OCHA reports |
| `gdelt_ingest` | GDELT media tone |
| `user_{hash}` | User interactions |

---

### nation_risk

Composite risk scores computed from multiple sources.

```sql
CREATE TABLE nation_risk (
    iso_a3 TEXT PRIMARY KEY REFERENCES nations(code),
    name TEXT NOT NULL,
    overall_risk FLOAT NOT NULL,        -- [0,1] Weighted composite
    political_risk FLOAT,
    economic_risk FLOAT,
    social_risk FLOAT,
    security_risk FLOAT,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_risk_overall ON nation_risk(overall_risk DESC);
```

---

### briefs

Generated intelligence briefs (stored for audit/history).

```sql
CREATE TABLE briefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,              -- Full JSON brief
    summary TEXT,                       -- Executive summary
    signals_snapshot JSONB,             -- Input signals at generation time
    model TEXT DEFAULT 'latticeforge-fusion-v1',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_briefs_created ON briefs(created_at DESC);
```

---

### api_keys

API key management for tiered access.

```sql
CREATE TABLE api_keys (
    key_hash TEXT PRIMARY KEY,          -- SHA-256 hash of key
    user_id UUID REFERENCES auth.users(id),
    client_tier TEXT DEFAULT 'consumer', -- 'consumer', 'pro', 'enterprise'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used TIMESTAMPTZ,
    revoked BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_keys_user ON api_keys(user_id);
CREATE INDEX idx_keys_tier ON api_keys(client_tier);
```

---

## TypeScript Interfaces

### UniversalSignal

Standard format for all ingested events. Full provenance tracking for audit compliance.

```typescript
interface UniversalSignal {
    id: string;                         // Deterministic hash-based ID
    signal_type: SignalType;            // event | news | metric | indicator | alert
    timestamp: string;                  // ISO 8601

    // Spatial
    geo?: {
        country_code: string;           // ISO 3166-1 alpha-2
        country_name?: string;
        region?: string;
        lat?: number;
        lon?: number;
        precision: 'country' | 'region' | 'city' | 'exact';
    };

    // Content
    title?: string;
    content?: string;
    value?: number;
    value_type?: string;
    unit?: string;

    // Confidence
    confidence: ConfidenceLevel;        // unknown | low | medium | high | confirmed
    confidence_score: number;           // [0,1]

    // Classification
    categories: string[];
    tags: string[];

    // PROVENANCE (Critical for audit - see below)
    provenance: Provenance;

    // Raw data (for audit replay)
    raw?: Record<string, unknown>;
}

// Provenance for data integrity verification
interface Provenance {
    source_id: string;                  // e.g., 'acled', 'fred'
    source_name: string;                // Human-readable name
    source_tier: SourceTier;            // official | institutional | open | derived
    source_url?: string;                // Original URL
    fetched_at: string;                 // ISO timestamp of retrieval
    attribution?: string;               // Required attribution text
    license?: string;                   // License type
    transformations: string[];          // Processing chain applied
    original_hash: string;              // SHA-256 of raw response (INTEGRITY)
}

type SignalSource = 'acled' | 'ucdp' | 'reliefweb' | 'gdelt' | 'world_bank' | 'imf' | 'manual';
type SourceTier = 'official' | 'institutional' | 'commercial' | 'open' | 'derived';
```

### Provenance Fields (Acquisition-Critical)

| Field | Purpose | Example |
|-------|---------|---------|
| `source_id` | Unique source identifier | `acled`, `fred` |
| `source_tier` | Reliability classification | `official`, `institutional` |
| `fetched_at` | When data was retrieved | `2024-12-16T10:30:00Z` |
| `original_hash` | SHA-256 of raw response | `a1b2c3d4...` |
| `transformations` | Processing steps | `['normalize', 'geocode']` |

**Why this matters:** Buyers can trace any output to exact source data and verify integrity.

### NationData

Nation state vector as used in briefing generation.

```typescript
interface NationData {
    code: string;                       // ISO 3166-1 alpha-3
    name: string;
    basin_strength: number;             // [0,1]
    transition_risk: number;            // [0,1]
    regime: number;                     // [1-5]
}
```

### ComputedMetrics

Full metrics object for briefing generation.

```typescript
interface ComputedMetrics {
    region: string;
    preset: 'global' | 'nato' | 'brics' | 'conflict';
    timestamp: string;
    categories: Record<CategoryKey, CategoryMetrics>;
    topAlerts: Alert[];
    overallRisk: RiskLevel;
}

interface CategoryMetrics {
    riskLevel: number;                  // 0-100, quantized to 5-point buckets
    trend: 'improving' | 'stable' | 'worsening';
    alertCount: number;
    keyFactors: string[];
}

interface Alert {
    category: string;
    severity: 'watch' | 'warning' | 'critical';
    region: string;
    summary: string;
}

type RiskLevel = 'low' | 'moderate' | 'elevated' | 'high' | 'critical';

type CategoryKey =
    | 'political' | 'economic' | 'security' | 'financial'
    | 'health' | 'scitech' | 'resources' | 'crime'
    | 'cyber' | 'terrorism' | 'domestic' | 'borders'
    | 'infoops' | 'military' | 'space' | 'industry'
    | 'logistics' | 'minerals' | 'energy' | 'markets'
    | 'religious' | 'education' | 'employment' | 'housing'
    | 'crypto' | 'emerging';
```

### CachedBriefing

Redis cache entry format.

```typescript
interface CachedBriefing {
    data: {
        briefings: Record<string, string>;
        metadata: Record<string, unknown>;
    };
    timestamp: number;                  // Unix ms
    generatedAt: string;                // ISO 8601
}
```

---

## Data Flow

### Ingest Pipeline

```
External API → Adapter → UniversalSignal → learning_events
                              ↓
                    Aggregation Job
                              ↓
                         nations (state vectors)
                              ↓
                      nation_risk (composite)
```

### Briefing Generation

```
nations + country_signals → computeCategoryRisk()
                                   ↓
                            ComputedMetrics
                                   ↓
                         Template Generation
                                   ↓
                              briefings{}
                                   ↓
                         Redis Cache + briefs table
```

---

## Migrations

### Initial Schema

```sql
-- Run with Supabase CLI
supabase db push

-- Or manually
psql $DATABASE_URL < migrations/001_initial_schema.sql
```

### Adding New Indicators

```sql
-- Example: Add new signal type
ALTER TABLE country_signals
ADD COLUMN confidence FLOAT DEFAULT 1.0;

-- Update TypeScript types
-- packages/web/lib/types/signals.ts
```

---

## Indexes & Performance

### Query Patterns

| Query | Index Used | Expected Time |
|-------|-----------|---------------|
| Get nation by code | `nations_pkey` | < 1ms |
| List high-risk nations | `idx_nations_transition_risk` | < 10ms |
| Get signals for country | `idx_signals_country` | < 5ms |
| Recent events by source | `idx_events_session` + `idx_events_timestamp` | < 20ms |

### Vacuum Schedule

```sql
-- Run daily
VACUUM ANALYZE nations;
VACUUM ANALYZE country_signals;
VACUUM ANALYZE learning_events;
```

---

## Row-Level Security (RLS)

### Public Read, Authenticated Write

```sql
-- nations: public read
ALTER TABLE nations ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Nations are viewable by everyone"
    ON nations FOR SELECT USING (true);

-- api_keys: owner only
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can only see own keys"
    ON api_keys FOR SELECT
    USING (auth.uid() = user_id);
```

---

*Last updated: 2024-12*
