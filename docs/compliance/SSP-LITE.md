# LatticeForge System Security Plan (Lite)

> Abbreviated SSP for acquisition due diligence. Full SSP available upon request.

## 1. System Identification

| Field | Value |
|-------|-------|
| System Name | LatticeForge |
| System Type | Geopolitical Signal Intelligence Platform |
| Classification | Unclassified / Commercial |
| Deployment Model | Cloud (Vercel Edge + Supabase) |
| Data Sensitivity | Public/Open Source Intelligence Only |

## 2. System Boundary

### 2.1 In Scope

- Web application (Next.js on Vercel Edge)
- API endpoints (Edge Functions)
- Database (Supabase Postgres)
- Cache layer (Upstash Redis)
- Ingest pipeline (Python adapters)

### 2.2 Out of Scope

- End-user devices
- Third-party data source infrastructure (ACLED, UCDP, ReliefWeb)
- User authentication provider (Supabase Auth)

## 3. Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM BOUNDARY                              │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │  Ingest  │───►│ Postgres │───►│   API    │───►│  Client  │     │
│  │ Pipeline │    │    DB    │    │  (Edge)  │    │   App    │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│       │                │               │                           │
│       ▼                ▼               ▼                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                     │
│  │  Source  │    │  Redis   │    │  Vercel  │                     │
│  │ Checksum │    │  Cache   │    │   Logs   │                     │
│  └──────────┘    └──────────┘    └──────────┘                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
        ▲                                              │
        │                                              ▼
┌───────┴───────┐                            ┌─────────────────┐
│ External Data │                            │   End Users     │
│   Sources     │                            │  (Web Browser)  │
│ ACLED/UCDP/RW │                            └─────────────────┘
└───────────────┘
```

## 4. Data Classification

| Data Type | Classification | Handling |
|-----------|---------------|----------|
| Nation state vectors | Public derivative | Computed from open sources |
| Economic indicators | Public | World Bank, IMF public data |
| Conflict events | Public | ACLED, UCDP academic data |
| User sessions | Internal | Hashed, no PII stored |
| API keys | Confidential | Hashed at rest |
| Briefing outputs | Internal | Derived analytical assessments |

## 5. Security Controls

### 5.1 Access Control

| Control | Implementation |
|---------|---------------|
| Authentication | Supabase Auth (JWT) |
| Authorization | Tier-based (Consumer/Pro/Enterprise) |
| API Keys | SHA-256 hashed, never stored plaintext |
| Row-Level Security | Enabled on all user-facing tables |

### 5.2 Data Protection

| Control | Implementation |
|---------|---------------|
| Encryption at Rest | Supabase managed (AES-256) |
| Encryption in Transit | TLS 1.3 enforced |
| Data Retention | Configurable per table |
| Backup | Supabase automated daily |

### 5.3 Network Security

| Control | Implementation |
|---------|---------------|
| Edge Distribution | Vercel global edge network |
| DDoS Protection | Vercel/Cloudflare managed |
| Rate Limiting | Per-tier limits enforced |
| IP Allowlisting | Available for Enterprise tier |

## 6. No External AI Dependencies

**Critical Security Feature:**

- Zero external LLM API calls
- No data sent to third-party AI providers
- All analysis computed locally via deterministic algorithms
- No model inference, embeddings, or probabilistic outputs

This eliminates:
- AI supply chain risk
- Data leakage to AI providers
- Non-deterministic output risk
- Model poisoning attack surface

## 7. Incident Response

| Phase | Owner | SLA |
|-------|-------|-----|
| Detection | Automated monitoring | Real-time |
| Triage | On-call engineer | < 1 hour |
| Containment | Security lead | < 4 hours |
| Recovery | Engineering team | < 24 hours |
| Post-mortem | All stakeholders | < 72 hours |

## 8. Compliance Alignment

| Framework | Status | Notes |
|-----------|--------|-------|
| SOC 2 Type II | Aligned | Via Vercel/Supabase |
| FedRAMP | Pathway identified | Requires dedicated deployment |
| GDPR | Compliant | No EU PII processed |
| ITAR | N/A | No controlled technical data |

## 9. Risk Assessment Summary

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data breach | Low | Medium | Encryption + RLS + no PII |
| Service outage | Low | Low | Multi-region edge deployment |
| Supply chain attack | Very Low | High | Zero external AI dependencies |
| Data integrity | Low | Medium | Source checksums + audit logs |

## 10. Continuous Monitoring

- Vercel Analytics (performance, errors)
- Supabase Dashboard (database health)
- Upstash Console (cache metrics)
- Custom learning_events table (anomaly detection)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12 | LatticeForge | Initial SSP-Lite |

---

*This document provides an overview for due diligence purposes. A comprehensive SSP following NIST 800-53 controls is available upon request for formal compliance processes.*
