# LatticeForge Data Classification & Access Control

## Data Classification Levels

### Level 1: Public
Data freely available from open sources.

| Data Type | Source | Examples |
|-----------|--------|----------|
| Economic indicators | World Bank, IMF | GDP growth, inflation rates |
| Conflict events | ACLED, UCDP | Event locations, fatality counts |
| Humanitarian reports | ReliefWeb | Crisis summaries |
| Country metadata | ISO, UN | Country codes, names |

### Level 2: Internal
Computed or aggregated data not publicly available.

| Data Type | Description | Access |
|-----------|-------------|--------|
| Nation state vectors | Computed basin_strength, transition_risk | API consumers |
| Composite risk scores | Aggregated from multiple sources | API consumers |
| Briefing outputs | Generated analytical assessments | Tier-gated |
| Usage analytics | Anonymized request patterns | Internal only |

### Level 3: Confidential
Sensitive operational data.

| Data Type | Description | Access |
|-----------|-------------|--------|
| API keys | User authentication tokens | Owner only (hashed) |
| Session data | Temporary auth state | System only |
| Audit logs | Security event records | Admin only |

### Level 4: Restricted
Not applicable - LatticeForge does not process classified or controlled data.

---

## Access Control Matrix (RACI)

### Data Access

| Role | Public Data | Internal Data | Confidential | Operations |
|------|-------------|---------------|--------------|------------|
| Anonymous User | R | - | - | - |
| Consumer Tier | R | R (cached) | - | - |
| Pro Tier | R | R | - | - |
| Enterprise Tier | R | R (on-demand) | - | - |
| System Admin | R | R/A | R/A | R/A/C |
| Founder | R/A/C/I | R/A/C/I | R/A/C/I | R/A/C/I |

*R=Responsible, A=Accountable, C=Consulted, I=Informed*

### System Operations

| Operation | Consumer | Pro | Enterprise | Admin |
|-----------|----------|-----|------------|-------|
| Read cached briefings | Yes | Yes | Yes | Yes |
| Generate fresh briefings | No | No | Yes | Yes |
| Access raw nation data | No | No | Yes | Yes |
| Modify API keys | Own only | Own only | Own only | All |
| View audit logs | No | No | No | Yes |
| Deploy changes | No | No | No | Yes |

---

## Data Retention Policy

| Data Type | Retention | Justification |
|-----------|-----------|---------------|
| Briefing cache | 10 minutes | Performance optimization |
| Learning events | 90 days | Model improvement |
| Nation vectors | Indefinite | Core data asset |
| Audit logs | 1 year | Compliance requirement |
| User sessions | 7 days | Security best practice |

---

## Data Flow Controls

### Ingestion
```
External Source → Checksum Validation → Normalization → Database
                       ↓
                 Reject if invalid
```

### Processing
```
Database → State Vector Computation → Risk Aggregation → Briefing
              ↓                            ↓                ↓
         Logged             Decision basis recorded    Cached
```

### Output
```
Briefing → Tier Validation → Rate Limit Check → Response
              ↓                    ↓
         403 if invalid     429 if exceeded
```

---

## Export Controls Statement

LatticeForge outputs are classified as:

- **Derived analytical assessments** based on publicly available data
- **Non-operational** - not suitable for targeting or tactical decisions
- **Non-controlled** under ITAR/EAR as they contain no:
  - Defense articles
  - Technical data
  - Controlled technology

Outputs may be shared internationally without export license, subject to:
- Customer contractual terms
- Applicable sanctions (OFAC compliance)

---

## Incident Classification

| Severity | Definition | Response SLA |
|----------|------------|--------------|
| P1 - Critical | Data breach, system compromise | 1 hour |
| P2 - High | Service outage, data integrity issue | 4 hours |
| P3 - Medium | Degraded performance, partial outage | 24 hours |
| P4 - Low | Minor bug, cosmetic issue | 72 hours |

---

*Last updated: 2024-12*
