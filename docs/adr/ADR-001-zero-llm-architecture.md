# ADR-001: Zero-LLM Architecture

**Status**: Accepted
**Date**: 2024-12
**Decision Makers**: Ryan (Founder)

## Context

LatticeForge is a geopolitical signal intelligence platform targeting acquisition by defense/intel contractors ($10-50M). The original architecture relied on:

1. **Anthropic Claude** - Via Supabase edge functions
2. **LFBM (Self-hosted vLLM)** - On RunPod for cost reduction
3. **OpenAI/HuggingFace** - Various auxiliary models

This created several problems:

- **Cost unpredictability**: Inference costs scaled with usage
- **Vendor dependency**: API changes could break production
- **Latency variance**: 500ms-3s per LLM call
- **Acquisition friction**: Buyers want predictable costs and no third-party AI dependencies

## Decision

**Remove ALL LLM dependencies. Implement 100% deterministic analysis.**

All briefing generation now uses:
- Threshold-based classification
- Pre-computed nation state vectors
- Template-driven prose generation
- Dempster-Shafer evidence fusion

## Consequences

### Positive

| Benefit | Impact |
|---------|--------|
| Zero inference cost | $0.00 per request vs $0.01-0.10 |
| Deterministic outputs | Same inputs = same outputs, always |
| Sub-100ms latency | 10-50x faster than LLM calls |
| No API keys needed | Eliminates vendor lock-in |
| Auditable reasoning | Every conclusion traceable to source data |
| Acquisition-ready | No third-party AI dependencies |

### Negative

| Trade-off | Mitigation |
|-----------|------------|
| No natural language flexibility | Template system with conditional logic |
| Fixed analysis patterns | Extensible rule engine |
| No creative synthesis | Pre-authored expert templates |
| Harder to add new categories | Well-documented template structure |

### Neutral

- Briefing quality depends on template authoring (same as prompt engineering)
- Requires domain expertise to write effective templates
- Analysis depth limited to available structured data

## Implementation

### Removed Components

```diff
- import { Anthropic } from '@anthropic-ai/sdk';
- import { getLFBMClient } from '@/lib/inference/LFBMClient';
- const response = await anthropic.messages.create({...});
- const briefings = await lfbmClient.generateFromMetrics({...});
```

### Added Components

```typescript
// Deterministic briefing generation
briefings['political'] = topRiskNames.length > 0
  ? `Political stability monitoring across ${nationData.length} nations. ` +
    `${topRiskNames.join(', ')} showing elevated transition indicators ` +
    `(avg ${avgTransitionRisk}% risk).`
  : `Political environment stable across ${nationData.length} monitored nations.`;
```

### Purged API Keys

| Key | Service | Status |
|-----|---------|--------|
| `ANTHROPIC_API_KEY` | Claude API | DELETED |
| `LFBM_ENDPOINT` | Self-hosted vLLM | DELETED |
| `LFBM_API_KEY` | Self-hosted vLLM | DELETED |
| `RUNPOD_API_KEY` | GPU hosting | DELETED |
| `OPENAI_API_KEY` | OpenAI | DELETED |
| `HF_TOKEN` | HuggingFace | DELETED |

## Alternatives Considered

### 1. Continue with LFBM (Self-hosted vLLM)

**Rejected because:**
- Still requires GPU hosting ($200-500/mo)
- Maintenance overhead
- Latency still 500ms+
- Acquirer would inherit infrastructure

### 2. Use Cheaper Models (GPT-3.5, Mistral)

**Rejected because:**
- Still has per-request costs
- Quality concerns for geopolitical analysis
- API dependency remains

### 3. Fine-tune Small Model

**Rejected because:**
- Training data requirements
- Ongoing retraining needs
- Still requires inference infrastructure

### 4. Hybrid (LLM for summaries only)

**Rejected because:**
- Complexity without significant benefit
- Partial vendor dependency
- Inconsistent user experience

## Validation

### Performance Metrics

```
Before (LFBM):
- Latency: 800-2000ms
- Cost: ~$0.01 per briefing
- Failure rate: 2-5% (API errors)

After (Deterministic):
- Latency: 50-100ms
- Cost: $0.00 per briefing
- Failure rate: <0.1% (only DB errors)
```

### Quality Assessment

Briefing quality maintained through:
1. Expert-authored templates with conditional logic
2. Real-time data from nation state vectors
3. Threshold-based severity classification
4. 26 category coverage (same as LLM version)

## References

- `/packages/web/app/api/intel-briefing/route.ts` - Main implementation
- `/supabase/functions/intel-brief/index.ts` - Supabase edge function
- `/packages/research/ingest/sources/conflict.py` - Data adapters

---

*This ADR follows the format described in [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) by Michael Nygard.*
