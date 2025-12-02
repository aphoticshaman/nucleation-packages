# Structured Intelligence Reports
## Standard Formats for Tactical and Strategic Analysis

---

## Overview

Structured reporting formats enable:
- **Speed** - No time deciding what to include
- **Consistency** - Comparable across reporters and time
- **Training** - Easier to develop analyst capabilities
- **Automation** - Machine-readable for processing
- **Quality** - Forced completeness, reduced omissions

This document defines LatticeForge's standard report formats.

---

## REPORT HIERARCHY

```
┌────────────────────────────────────────────────────────────────┐
│                    REPORT ESCALATION                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  FLASH ──→ SPOT ──→ PRELIMINARY ──→ COMPREHENSIVE ──→ SUMMARY │
│   │         │           │               │               │      │
│   │         │           │               │               │      │
│  <5min    <30min      <2hrs          <24hrs         Periodic  │
│   │         │           │               │               │      │
│  Basic    6 W's      Initial         Full          Trends &   │
│  Facts    Answered   Analysis       Exploitation   Patterns   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## SPOT REPORT FORMAT

### The Six Essential Elements

Every tactical intelligence report must answer:

```
┌────────────────────────────────────────────────────────────────┐
│                     SPOT REPORT                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. SIZE/SCOPE                                                 │
│     What is the magnitude?                                     │
│     • Quantitative: Numbers, percentages, metrics              │
│     • Qualitative: Severity level, impact class               │
│                                                                │
│  2. ACTIVITY                                                   │
│     What is happening?                                         │
│     • Current state/action                                     │
│     • Observable behaviors                                     │
│     • Direction of change                                      │
│                                                                │
│  3. LOCATION                                                   │
│     Where is it occurring?                                     │
│     • Geographic coordinates/reference                         │
│     • Domain (cyber, financial, political)                     │
│     • Jurisdiction/boundaries                                  │
│                                                                │
│  4. UNIT/ACTOR                                                 │
│     Who is involved?                                           │
│     • Primary actors                                           │
│     • Affiliations                                             │
│     • Attribution confidence                                   │
│                                                                │
│  5. TIME                                                       │
│     When did/will it occur?                                    │
│     • Date-time of observation                                 │
│     • Duration                                                 │
│     • Projected timeline                                       │
│                                                                │
│  6. EQUIPMENT/RESOURCES                                        │
│     What resources are involved?                               │
│     • Capabilities observed                                    │
│     • Materials/tools identified                               │
│     • Infrastructure used                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Spot Report Template

```markdown
## SPOT REPORT
**DTG:** [YYYY-MM-DD HH:MM:SS Z]
**Report Number:** [Sequential ID]
**Classification:** [UNCLASS/FOUO/etc]

### 1. SIZE/SCOPE
[Magnitude of the event/condition]

### 2. ACTIVITY
[What is happening or was observed]

### 3. LOCATION
[Geographic or domain reference]

### 4. UNIT/ACTOR
[Who is involved, attribution]

### 5. TIME
[When observed, duration, timeline]

### 6. EQUIPMENT/RESOURCES
[Capabilities, materials, infrastructure]

---
**Reporter:** [Name/Org]
**Contact:** [Method]
**Confidence:** [HIGH/MODERATE/LOW]
```

---

## 9-LINE REPORT FORMAT

### Rapid Structured Report

For time-critical situations requiring immediate awareness:

```
┌────────────────────────────────────────────────────────────────┐
│                     9-LINE REPORT                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  LINE 1:  DATE-TIME DISCOVERED                                 │
│           [YYYYMMDDHHMM]                                       │
│                                                                │
│  LINE 2A: REPORTING ACTIVITY                                   │
│           [Organization and location]                          │
│                                                                │
│  LINE 2B: EVENT LOCATION                                       │
│           [Grid reference / coordinates]                       │
│                                                                │
│  LINE 3:  CONTACT METHOD                                       │
│           [How to reach reporter]                              │
│                                                                │
│  LINE 4:  TYPE/CATEGORY                                        │
│           [Classification of event]                            │
│                                                                │
│  LINE 5:  CONTAMINATION/HAZARD                                 │
│           [Secondary risks present]                            │
│                                                                │
│  LINE 6:  RESOURCES THREATENED                                 │
│           [Assets at risk]                                     │
│                                                                │
│  LINE 7:  IMPACT ON OPERATIONS                                 │
│           [Effect on mission/business]                         │
│                                                                │
│  LINE 8:  PROTECTIVE MEASURES                                  │
│           [Actions taken]                                      │
│                                                                │
│  LINE 9:  RECOMMENDED PRIORITY                                 │
│           [IMMEDIATE / INDIRECT / MINOR / NO THREAT]           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## PRELIMINARY TECHNICAL REPORT

### Early-Stage Analysis Format

For initial exploitation and assessment before comprehensive analysis:

```
┌────────────────────────────────────────────────────────────────┐
│               PRELIMINARY TECHNICAL REPORT                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  HEADER                                                        │
│  ├── Priority: [FLASH/IMMEDIATE/PRIORITY/ROUTINE]              │
│  ├── From: [Reporting organization]                            │
│  ├── To: [Primary recipient]                                   │
│  ├── Info: [Secondary recipients]                              │
│  └── Subject: PRELIMINARY REPORT - [Brief description]         │
│                                                                │
│  BODY                                                          │
│                                                                │
│  A. TYPE AND QUANTITY                                          │
│     [What was discovered/observed and how much]                │
│                                                                │
│  B. DATE-TIME OF DISCOVERY                                     │
│     [When the event/item was identified]                       │
│                                                                │
│  C. LOCATION                                                   │
│     [Geographic reference, map coordinates]                    │
│                                                                │
│  D. CIRCUMSTANCES OF DISCOVERY                                 │
│     [How it was found, context]                                │
│                                                                │
│  E. ORIGIN/ATTRIBUTION                                         │
│     [Source assessment, suspected actor]                       │
│                                                                │
│  F. PHYSICAL DESCRIPTION                                       │
│     [Distinguishing marks, characteristics]                    │
│     [Dimensions, appearance, manufacturer if known]            │
│                                                                │
│  G. TECHNICAL CHARACTERISTICS                                  │
│     [Immediate value information]                              │
│     [Capabilities, components identified]                      │
│                                                                │
│  H. REPORT ORIGIN                                              │
│     [Time and source of this message]                          │
│                                                                │
│  I. CURRENT STATUS/DISPOSITION                                 │
│     [Where is the subject now, what's happening with it]       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## COMPREHENSIVE TECHNICAL REPORT

### Full Exploitation Report

For complete analysis after thorough examination:

```
┌────────────────────────────────────────────────────────────────┐
│              COMPREHENSIVE TECHNICAL REPORT                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  SECTION I: ADMINISTRATIVE DATA                                │
│  ├── Report number and DTG                                     │
│  ├── References to prior reports                               │
│  ├── Classification and handling                               │
│  └── Distribution list                                         │
│                                                                │
│  SECTION II: SUMMARY                                           │
│  ├── Key findings (3-5 bullets)                                │
│  ├── Significance assessment                                   │
│  ├── Confidence level                                          │
│  └── Recommended actions                                       │
│                                                                │
│  SECTION III: BACKGROUND                                       │
│  ├── Context and circumstances                                 │
│  ├── Related events/reports                                    │
│  └── Historical precedents                                     │
│                                                                │
│  SECTION IV: TECHNICAL ANALYSIS                                │
│  ├── Component breakdown                                       │
│  │   ├── Trigger analysis                                      │
│  │   ├── Mechanism analysis                                    │
│  │   ├── Effect assessment                                     │
│  │   ├── Power source identification                           │
│  │   ├── Container description                                 │
│  │   └── Enhancement evaluation                                │
│  ├── Forensic findings                                         │
│  ├── Sourcing analysis                                         │
│  └── Signature comparison                                      │
│                                                                │
│  SECTION V: TACTICAL ANALYSIS                                  │
│  ├── Employment method                                         │
│  ├── Target selection                                          │
│  ├── Operational pattern                                       │
│  └── TTP assessment                                            │
│                                                                │
│  SECTION VI: ATTRIBUTION                                       │
│  ├── Actor assessment                                          │
│  ├── Network linkages                                          │
│  ├── Confidence and gaps                                       │
│  └── Alternative hypotheses                                    │
│                                                                │
│  SECTION VII: IMPLICATIONS                                     │
│  ├── Immediate threats                                         │
│  ├── Trend assessment                                          │
│  ├── Countermeasure recommendations                            │
│  └── Intelligence gaps                                         │
│                                                                │
│  SECTION VIII: APPENDICES                                      │
│  ├── Imagery/documentation                                     │
│  ├── Technical specifications                                  │
│  ├── Chronology                                                │
│  └── References                                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## INTELLIGENCE ASSESSMENT FORMAT

### Analytical Product Template

For strategic assessments and forecasts:

```
┌────────────────────────────────────────────────────────────────┐
│                 INTELLIGENCE ASSESSMENT                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  HEADER                                                        │
│  ├── Title: [Descriptive assessment title]                     │
│  ├── DTG: [Date-time of assessment]                            │
│  ├── Timeframe: [Period covered by assessment]                 │
│  └── Classification: [Handling instructions]                   │
│                                                                │
│  SCOPE NOTE                                                    │
│  [What this assessment covers and does not cover]              │
│                                                                │
│  KEY JUDGMENTS                                                 │
│  [3-5 bottom-line assessments with confidence levels]          │
│  • We assess with [HIGH/MODERATE/LOW] confidence that...       │
│  • We judge it [LIKELY/UNLIKELY] that...                       │
│                                                                │
│  DISCUSSION                                                    │
│  [Detailed analysis supporting key judgments]                  │
│                                                                │
│  ├── Current Situation                                         │
│  │   [Present state of affairs]                                │
│  │                                                             │
│  ├── Driving Factors                                           │
│  │   [What is causing/influencing the situation]               │
│  │                                                             │
│  ├── Indicators and Warnings                                   │
│  │   [What to watch for]                                       │
│  │                                                             │
│  └── Scenarios                                                 │
│      [Possible futures with probability assessments]           │
│      • Most Likely:                                            │
│      • Most Dangerous:                                         │
│      • Wild Card:                                              │
│                                                                │
│  ASSUMPTIONS                                                   │
│  [Explicit listing of key assumptions]                         │
│                                                                │
│  INFORMATION GAPS                                              │
│  [What we don't know that would improve the assessment]        │
│                                                                │
│  ANALYTIC CONFIDENCE                                           │
│  [Explanation of confidence level and basis]                   │
│  ├── Source quality:                                           │
│  ├── Corroboration:                                            │
│  ├── Analytical rigor:                                         │
│  └── Key uncertainties:                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## CONFIDENCE STANDARDS

### Likelihood Language

| Term | Probability | Usage |
|------|-------------|-------|
| **Almost Certain** | >90% | Very high confidence the event will occur |
| **Likely/Probably** | 60-90% | More likely than not |
| **Roughly Even Odds** | 40-60% | Could go either way |
| **Unlikely** | 10-40% | Less likely than not |
| **Remote/Highly Unlikely** | <10% | Very improbable |

### Confidence Levels

| Level | Definition | Basis |
|-------|------------|-------|
| **High** | Judgments based on high-quality information and/or strong analytical basis | Multiple independent sources, corroborated, expertise available |
| **Moderate** | Judgments based on credibly sourced and plausible information, but not of sufficient quality or corroboration | Fewer sources, partial corroboration, some gaps |
| **Low** | Judgments based on fragmentary information, questionable sources, or significant analytical gaps | Single source, uncorroborated, major uncertainties |

### Combining Likelihood and Confidence

Always state both:

| Statement | Meaning |
|-----------|---------|
| "We assess with **high confidence** that X is **likely**" | We're sure of our analysis, and the probability is 60-90% |
| "We assess with **low confidence** that X is **likely**" | Our analysis has gaps, but available evidence suggests 60-90% probability |
| "We assess with **high confidence** that X is **unlikely**" | We're sure of our analysis, and the probability is 10-40% |

---

## DOCUMENTATION LOGS

### Evidence Documentation

```
┌────────────────────────────────────────────────────────────────┐
│                   DOCUMENTATION LOG                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IMAGERY LOG                                                   │
│  ┌─────┬──────┬─────────────┬──────────┬──────────┬─────────┐ │
│  │ No. │ View │ Description │ Settings │ Distance │   DTG   │ │
│  ├─────┼──────┼─────────────┼──────────┼──────────┼─────────┤ │
│  │ 001 │ N    │ Overview    │ Auto     │ 10m      │ 0900Z   │ │
│  │ 002 │ S    │ Detail      │ Macro    │ 0.5m     │ 0905Z   │ │
│  │ ... │      │             │          │          │         │ │
│  └─────┴──────┴─────────────┴──────────┴──────────┴─────────┘ │
│                                                                │
│  CHAIN OF CUSTODY                                              │
│  ┌─────────┬──────────┬──────────┬─────────────┬────────────┐ │
│  │   DTG   │ Released │ Received │   Purpose   │ Condition  │ │
│  ├─────────┼──────────┼──────────┼─────────────┼────────────┤ │
│  │ 0900Z   │ Alpha    │ Bravo    │ Transport   │ Sealed     │ │
│  │ 1200Z   │ Bravo    │ Lab      │ Analysis    │ Sealed     │ │
│  │ ...     │          │          │             │            │ │
│  └─────────┴──────────┴──────────┴─────────────┴────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## LATTICEFORGE IMPLEMENTATION

### Auto-Generated Reports

LatticeForge generates reports from structured data:

```typescript
interface ReportConfig {
  type: 'spot' | '9line' | 'preliminary' | 'comprehensive' | 'assessment';

  // Content
  subject: Entity | Event | Condition;
  timeframe: DateRange;
  scope: string[];

  // Format
  output: 'markdown' | 'pdf' | 'html' | 'json';
  classification: string;
  distribution: string[];

  // Analysis options
  include_confidence: boolean;
  include_assumptions: boolean;
  include_alternatives: boolean;
  include_gaps: boolean;
}
```

### Report Generation Flow

```
┌────────────────────────────────────────────────────────────────┐
│                  REPORT GENERATION                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  USER INPUT                                                    │
│  ├── Select entity/event                                       │
│  ├── Choose report type                                        │
│  └── Set parameters                                            │
│           │                                                    │
│           ▼                                                    │
│  DATA ASSEMBLY                                                 │
│  ├── Pull current state data                                   │
│  ├── Pull historical context                                   │
│  ├── Pull related entities                                     │
│  └── Pull analytical metadata                                  │
│           │                                                    │
│           ▼                                                    │
│  TEMPLATE POPULATION                                           │
│  ├── Fill standard fields                                      │
│  ├── Generate narrative sections                               │
│  ├── Calculate confidence metrics                              │
│  └── Add visualizations                                        │
│           │                                                    │
│           ▼                                                    │
│  OUTPUT                                                        │
│  ├── Render in requested format                                │
│  ├── Apply classification markings                             │
│  └── Deliver to distribution list                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Quick Report Interface

```
┌────────────────────────────────────────────────────────────────┐
│                   GENERATE REPORT                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  SUBJECT: [Ukraine Political Stability]                        │
│                                                                │
│  REPORT TYPE:                                                  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ ○ Flash (< 5 min)  - Basic facts only                   │  │
│  │ ○ Spot (< 30 min)  - Six elements answered              │  │
│  │ ● Preliminary      - Initial analysis with confidence   │  │
│  │ ○ Comprehensive    - Full exploitation                  │  │
│  │ ○ Assessment       - Strategic forecast                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  OPTIONS:                                                      │
│  ☑ Include confidence levels                                   │
│  ☑ Include key assumptions                                     │
│  ☑ Include alternative scenarios                               │
│  ☐ Include intelligence gaps                                   │
│                                                                │
│  OUTPUT FORMAT:                                                │
│  ○ Markdown  ● PDF  ○ HTML  ○ JSON                            │
│                                                                │
│  [GENERATE REPORT]                                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## SUMMARY

Structured reporting provides:

1. **Consistency** - Same format across all analysts
2. **Completeness** - Forced consideration of all elements
3. **Comparability** - Reports can be compared over time
4. **Automation** - Machine-readable for processing
5. **Training** - Clear standards for new analysts

**Key Principle:** The format is not bureaucracy - it's a cognitive tool that ensures nothing important is missed under pressure.

---

*Document Version 1.0 | Structured Intelligence Reports*
*Last Updated: December 2024*
