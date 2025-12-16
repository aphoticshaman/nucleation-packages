# LatticeForge API Reference

> All endpoints use deterministic analysis. No LLM calls. Response times < 100ms.

## Base URLs

| Environment | URL |
|-------------|-----|
| Production (Vercel) | `https://latticeforge.vercel.app` |
| Supabase Edge | `https://<project>.supabase.co/functions/v1` |

---

## Web API Endpoints

### POST /api/intel-briefing

Generate a 26-category geopolitical intelligence briefing.

#### Request

```http
POST /api/intel-briefing
Content-Type: application/json
Cookie: sb-access-token=<jwt>

{
  "preset": "global",
  "region": "US"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `preset` | string | No | One of: `global`, `nato`, `brics`, `conflict`. Default: `global` |
| `region` | string | No | ISO country code for user region. Auto-detected from headers if omitted. |

#### Response (Cache Hit)

```json
{
  "briefings": {
    "political": "Political stability monitoring across 32 nations...",
    "economic": "Economic monitoring active for GLOBAL region...",
    "security": "Security environment requires monitoring...",
    "financial": "Financial stability tracking 32 economies...",
    "cyber": "Cyber threat landscape at baseline...",
    "health": "No significant health security alerts.",
    "scitech": "Innovation metrics stable. Normal operations.",
    "resources": "Resource access stable across monitored regions.",
    "crime": "Crime indices at baseline. No significant changes.",
    "terrorism": "Threat assessment at baseline...",
    "domestic": "Domestic stability indicators nominal...",
    "borders": "Border security posture stable...",
    "infoops": "Information ecosystem baseline...",
    "military": "Force readiness at baseline...",
    "space": "Space operations nominal...",
    "industry": "Manufacturing metrics stable...",
    "logistics": "Supply chain operations nominal...",
    "minerals": "Critical mineral supply stable...",
    "energy": "Energy security baseline...",
    "markets": "Market volatility within normal range...",
    "religious": "Interfaith relations stable...",
    "education": "Education sector stable...",
    "employment": "Employment metrics stable...",
    "housing": "Housing market indicators within normal parameters...",
    "crypto": "Cryptocurrency markets at baseline...",
    "emerging": "No significant emerging trends detected...",
    "summary": "GLOBAL intelligence synthesis across 32 nations...",
    "nsm": "Continue routine monitoring. No immediate escalation required."
  },
  "metadata": {
    "region": "US",
    "preset": "global",
    "timestamp": "2024-12-16T10:30:00.000Z",
    "overallRisk": "moderate",
    "cached": true,
    "cachedAt": "2024-12-16T10:25:00.000Z",
    "cacheAgeSeconds": 300
  }
}
```

#### Response (Cache Warming)

When cache is empty and user is not privileged:

```json
{
  "status": "warming",
  "message": "Intelligence briefing cache is warming up. Please wait...",
  "estimatedWaitSeconds": 60,
  "preset": "global",
  "retryAfterMs": 5000,
  "metadata": {
    "region": "US",
    "preset": "global",
    "timestamp": "2024-12-16T10:30:00.000Z",
    "cached": false,
    "generatedBy": "warmup-pending"
  }
}
```

#### Response (Fresh Generation - Cron/Internal Only)

```json
{
  "briefings": { /* ... same as cache hit ... */ },
  "metadata": {
    "region": "US",
    "preset": "global",
    "timestamp": "2024-12-16T10:30:00.000Z",
    "overallRisk": "moderate",
    "source": "deterministic_template",
    "estimatedCost": "$0.00",
    "reasoning": {
      "confidence": 0.85,
      "engines": ["basin_attractor", "cascade_detection"],
      "conclusion": "stable_equilibrium",
      "computeTimeMs": 45
    },
    "performance": {
      "totalLatencyMs": 87,
      "generationLatencyMs": 42
    },
    "rateLimitRemaining": 999,
    "cached": false
  }
}
```

#### Error Responses

| Status | Condition | Response |
|--------|-----------|----------|
| 400 | Invalid preset | `{"error": "Invalid preset"}` |
| 429 | Rate limited | `{"error": "Rate limit exceeded", "resetAt": "..."}` |
| 500 | Server error | `{"error": "Failed to generate briefing"}` |

---

### POST /api/us-brief

Generate US economic regime analysis.

#### Request

```http
POST /api/us-brief
Content-Type: application/json
Cookie: sb-access-token=<jwt>

{}
```

#### Response

```json
{
  "regime": {
    "current": "LATE_CYCLE",
    "confidence": 0.78,
    "indicators": {
      "gdp_growth": 2.1,
      "unemployment": 3.8,
      "inflation": 3.2,
      "yield_curve": -0.15
    }
  },
  "anomalies": [
    {
      "indicator": "yield_curve",
      "severity": "elevated",
      "description": "Inverted yield curve persists"
    }
  ],
  "positioning": {
    "recommendation": "defensive",
    "rationale": "Late cycle indicators suggest caution",
    "sectors": {
      "overweight": ["utilities", "healthcare"],
      "underweight": ["consumer_discretionary", "tech"]
    }
  },
  "metadata": {
    "timestamp": "2024-12-16T10:30:00.000Z",
    "source": "deterministic_template",
    "dataAsOf": "2024-12-15"
  }
}
```

---

## Supabase Edge Functions

### POST /functions/v1/intel-brief

Enterprise-tier intelligence briefing via Supabase Edge.

#### Request

```http
POST /functions/v1/intel-brief
Authorization: Bearer <supabase_jwt>
x-api-key: <enterprise_api_key>
Content-Type: application/json

{}
```

#### Authentication

Requires BOTH:
1. Valid Supabase JWT (`Authorization: Bearer <jwt>`)
2. Enterprise-tier API key (`x-api-key: <key>`)

#### Response (Success)

```json
{
  "brief_id": "uuid-here",
  "phase": {
    "current": "SUPERCOOLED",
    "confidence": 0.82,
    "basin_strength": 0.65,
    "transition_risk": 0.28
  },
  "signals": [
    {
      "source": "acled",
      "event_count": 142,
      "severity_avg": 0.45,
      "trend": "stable"
    },
    {
      "source": "gdelt",
      "tone_avg": -1.2,
      "volume": 8500,
      "trend": "worsening"
    }
  ],
  "risk_matrix": {
    "political": { "probability": 0.25, "impact": "high" },
    "economic": { "probability": 0.40, "impact": "moderate" },
    "security": { "probability": 0.15, "impact": "critical" }
  },
  "summary": "System in SUPERCOOLED phase with moderate stability...",
  "recommendations": [
    "Monitor economic indicators for transition signals",
    "Maintain baseline security posture"
  ],
  "metadata": {
    "generated_at": "2024-12-16T10:30:00.000Z",
    "model": "latticeforge-fusion-v1",
    "inference_cost": "$0.00"
  }
}
```

#### Error Responses

| Status | Condition | Response |
|--------|-----------|----------|
| 401 | Missing/invalid JWT | `{"error": "Unauthorized"}` |
| 403 | Non-enterprise tier | `{"error": "Intel Brief requires Enterprise tier", "upgrade_url": "/pricing"}` |
| 500 | Server error | `{"error": "Brief generation failed"}` |

---

## Presets

### global

All monitored nations (default ~190 countries)

### nato

NATO member states:
```
USA, CAN, GBR, FRA, DEU, ITA, ESP, POL, NLD, BEL, PRT, GRC, TUR, NOR, DNK,
CZE, HUN, ROU, BGR, SVK, HRV, SVN, LVA, LTU, EST, ALB, MNE, MKD, FIN, SWE, ISL, LUX
```

### brics

BRICS+ members:
```
BRA, RUS, IND, CHN, ZAF, IRN, EGY, ETH, SAU, ARE
```

### conflict

Active conflict zones:
```
UKR, RUS, ISR, PSE, LBN, SYR, YEM, SDN, MMR, AFG, TWN, CHN, PRK, KOR
```

---

## Rate Limits

| Tier | Requests/Hour | Cache TTL |
|------|---------------|-----------|
| Anonymous | 10 | 10 min |
| Consumer | 60 | 10 min |
| Pro | 300 | 5 min |
| Enterprise | Unlimited | On-demand |

---

## Caching Behavior

### L1 Cache (Hot)
- **Storage**: In-memory Map (per edge instance)
- **TTL**: 60 seconds
- **Latency**: < 1ms

### L2 Cache (Warm)
- **Storage**: Upstash Redis (shared)
- **TTL**: 10 minutes
- **Latency**: 5-50ms

### Cache Keys
```
intel-briefing:{preset}
```

---

## Internal Headers

These headers are used by cron jobs and internal services:

| Header | Value | Purpose |
|--------|-------|---------|
| `x-vercel-cron` | `1` | Set by Vercel for cron invocations |
| `x-internal-service` | `<secret>` | Internal service authentication |
| `x-cron-warm` | `1` | Cache warming request |

---

## SDK Usage

### JavaScript/TypeScript

```typescript
// Using fetch
const response = await fetch('/api/intel-briefing', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ preset: 'nato' }),
  credentials: 'include', // Include cookies for auth
});

const data = await response.json();

if (data.status === 'warming') {
  // Retry after delay
  setTimeout(() => refetch(), data.retryAfterMs);
} else {
  // Use briefings
  console.log(data.briefings.summary);
}
```

### Python

```python
import httpx

async def get_briefing(preset: str = "global") -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://latticeforge.vercel.app/api/intel-briefing",
            json={"preset": preset},
            cookies={"sb-access-token": jwt_token}
        )
        return response.json()
```

---

## Webhook Events (Future)

Planned webhook support for:

| Event | Trigger |
|-------|---------|
| `briefing.generated` | New briefing cached |
| `alert.critical` | Critical risk threshold exceeded |
| `phase.transition` | Nation phase change detected |

---

*Last updated: 2024-12*
