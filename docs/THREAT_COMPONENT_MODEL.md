# Threat Component Model
## A Universal Framework for Decomposing Complex Threats

---

## Overview

Every threat - whether physical, cyber, financial, political, or social - can be decomposed into a standard set of functional components. This decomposition enables:

- **Systematic analysis** across threat types
- **Pattern matching** between different domains
- **Vulnerability identification** at each component
- **Attribution** through component sourcing
- **Prediction** based on component assembly patterns

---

## THE SIX-COMPONENT MODEL

```
┌────────────────────────────────────────────────────────────────┐
│                      THREAT ARCHITECTURE                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│         ┌──────────────────────────────────────────┐          │
│         │            CONTAINER                      │          │
│         │  The structure housing the threat         │          │
│         │  (Physical boundary, organizational form) │          │
│         └──────────────────────────────────────────┘          │
│                           │                                    │
│    ┌──────────────────────┼──────────────────────┐           │
│    │                      │                      │            │
│    ▼                      ▼                      ▼            │
│ ┌────────┐         ┌────────────┐         ┌──────────┐       │
│ │TRIGGER │ ──────→ │ MECHANISM  │ ──────→ │  EFFECT  │       │
│ │        │         │            │         │          │        │
│ │What    │         │How it      │         │What      │        │
│ │starts  │         │operates    │         │happens   │        │
│ │it      │         │            │         │          │        │
│ └────────┘         └────────────┘         └──────────┘       │
│    ↑                      ↑                      ↑            │
│    │                      │                      │            │
│    └──────────────────────┼──────────────────────┘           │
│                           │                                    │
│         ┌──────────────────────────────────────────┐          │
│         │           POWER SOURCE                    │          │
│         │  Resources enabling operation             │          │
│         │  (Money, energy, personnel, access)       │          │
│         └──────────────────────────────────────────┘          │
│                           │                                    │
│         ┌──────────────────────────────────────────┐          │
│         │          ENHANCEMENTS                     │          │
│         │  Modifiers that amplify or alter          │          │
│         │  (Countermeasures, force multipliers)     │          │
│         └──────────────────────────────────────────┘          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## COMPONENT DEFINITIONS

### 1. TRIGGER

**Definition:** The condition, event, or action that initiates the threat's activation.

**Categories:**

| Type | Description | Examples |
|------|-------------|----------|
| **Command** | Deliberate activation by actor | Direct order, remote signal |
| **Time** | Activates after elapsed duration | Scheduled, delayed |
| **Victim-Operated** | Activated by target's action | Tripwire, phishing click |
| **Conditional** | Activates when condition met | Threshold reached, event detected |
| **Hybrid** | Multiple trigger types combined | Safe-arm + firing sequence |

**Analysis Questions:**
- What conditions trigger activation?
- Who/what controls the trigger?
- Can the trigger be detected before activation?
- What indicators precede triggering?

---

### 2. MECHANISM

**Definition:** The operational process that translates trigger into effect.

**Categories:**

| Type | Description | Examples |
|------|-------------|----------|
| **Direct** | Immediate cause-effect | Kinetic action, data deletion |
| **Cascading** | Chain of sequential events | Market contagion, network spread |
| **Amplifying** | Self-reinforcing process | Viral spread, feedback loops |
| **Persistent** | Ongoing operation | Long-term infiltration |
| **Adaptive** | Changes behavior based on context | AI-driven, responsive |

**Analysis Questions:**
- How does the threat produce its effect?
- What are the intermediate steps?
- Where can the mechanism be interrupted?
- How does the mechanism respond to countermeasures?

---

### 3. EFFECT

**Definition:** The intended outcome or impact of the threat.

**Categories:**

| Type | Description | Subtypes |
|------|-------------|----------|
| **Destructive** | Eliminate target | Physical, data, organizational |
| **Disruptive** | Impair operations | Delay, deny, degrade |
| **Extractive** | Remove value | Theft, exfiltration, capture |
| **Coercive** | Change behavior | Intimidation, blackmail |
| **Informational** | Alter perception | Propaganda, deception |
| **Symbolic** | Send message | Political statement, demonstration |

**Analysis Questions:**
- What is the intended outcome?
- What are potential secondary effects?
- Who/what is the target?
- What is the strategic objective beyond immediate effect?

---

### 4. POWER SOURCE

**Definition:** Resources that enable the threat's creation and operation.

**Categories:**

| Type | Description | Examples |
|------|-------------|----------|
| **Financial** | Monetary resources | Funding, cryptocurrency, cash |
| **Material** | Physical components | Raw materials, equipment |
| **Human** | Personnel | Expertise, labor, leadership |
| **Technical** | Knowledge and capability | Know-how, tools, infrastructure |
| **Social** | Network support | Supporters, safe houses, logistics |
| **State** | Government backing | Sanctuary, diplomatic cover |

**Analysis Questions:**
- What resources are required?
- Where do resources originate?
- What is the supply chain?
- Which resources are bottlenecks?

---

### 5. CONTAINER

**Definition:** The physical or organizational structure that houses the threat system.

**Categories:**

| Type | Description | Examples |
|------|-------------|----------|
| **Physical** | Tangible structure | Vehicle, building, device |
| **Organizational** | Group structure | Cell, network, hierarchy |
| **Digital** | Virtual infrastructure | Server, botnet, platform |
| **Geographic** | Territorial base | Region, sanctuary, zone |
| **Institutional** | Legitimate cover | Business, NGO, government |

**Analysis Questions:**
- What structure houses the threat?
- How is the container used for concealment?
- What are the container's vulnerabilities?
- How does the container enable or constrain operations?

---

### 6. ENHANCEMENTS

**Definition:** Modifications that amplify effectiveness or enable survival.

**Categories:**

| Type | Description | Examples |
|------|-------------|----------|
| **Force Multipliers** | Increase effect | Fragmentation, amplification |
| **Countermeasures** | Defeat defenses | Anti-tamper, jamming, evasion |
| **Persistence** | Extend duration | Backup systems, redundancy |
| **Adaptation** | Respond to environment | Learning systems, variant generation |
| **Concealment** | Avoid detection | Camouflage, encryption, cover |

**Analysis Questions:**
- What enhancements are present?
- How do enhancements change the threat profile?
- What capabilities do enhancements indicate?
- Which enhancements suggest sophistication level?

---

## CROSS-DOMAIN APPLICATION

### Cyber Threat

```
┌─────────────────────────────────────────────────────────────┐
│                    CYBER THREAT EXAMPLE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRIGGER:    Phishing email opened, exploit triggered       │
│                                                             │
│  MECHANISM:  Payload executes, establishes persistence,     │
│              moves laterally, exfiltrates data              │
│                                                             │
│  EFFECT:     Data theft, ransomware deployment,             │
│              operational disruption                         │
│                                                             │
│  POWER:      C2 infrastructure, cryptocurrency,             │
│              developer expertise                            │
│                                                             │
│  CONTAINER:  Compromised systems, bulletproof hosting       │
│                                                             │
│  ENHANCE:    Polymorphic code, encrypted comms,             │
│              anti-analysis techniques                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Political Instability

```
┌─────────────────────────────────────────────────────────────┐
│                 POLITICAL THREAT EXAMPLE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRIGGER:    Economic crisis, electoral fraud allegation,   │
│              catalyzing incident                            │
│                                                             │
│  MECHANISM:  Mobilization, protests, institutional          │
│              breakdown, security force defection            │
│                                                             │
│  EFFECT:     Regime change, civil conflict,                 │
│              state failure                                  │
│                                                             │
│  POWER:      Popular support, external backing,             │
│              elite defection, media control                 │
│                                                             │
│  CONTAINER:  Opposition movement, geographic base,          │
│              diaspora network                               │
│                                                             │
│  ENHANCE:    External intervention, social media,           │
│              arms supply, sanctuary                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Financial Threat

```
┌─────────────────────────────────────────────────────────────┐
│                  FINANCIAL THREAT EXAMPLE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRIGGER:    Market event, regulatory change,               │
│              liquidity shock                                │
│                                                             │
│  MECHANISM:  Contagion, margin calls, forced selling,       │
│              counterparty failure                           │
│                                                             │
│  EFFECT:     Asset collapse, institution failure,           │
│              systemic crisis                                │
│                                                             │
│  POWER:      Leverage, interconnectedness,                  │
│              market position                                │
│                                                             │
│  CONTAINER:  Exchange, institution, market structure        │
│                                                             │
│  ENHANCE:    Algorithmic amplification, derivatives,        │
│              opacity, regulatory arbitrage                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## TRIGGER TAXONOMY

### Command Triggers (Actor-Controlled)

```
COMMAND TRIGGERS
├── Direct
│   ├── Wired (physical connection to activation point)
│   └── Wireless (RF, IR, other electromagnetic)
│       ├── High-power (longer range, more detectable)
│       └── Low-power (shorter range, harder to detect)
├── Indirect
│   ├── Proxy (intermediary executes)
│   └── Autonomous (pre-programmed decision)
└── Pull (manual activation with standoff)
```

### Time Triggers (Duration-Based)

```
TIME TRIGGERS
├── Electronic (digital timer, integrated circuit)
├── Mechanical (clock, spring mechanism)
├── Chemical (reaction-based delay)
└── Environmental (natural process)
```

### Victim-Operated Triggers (Target-Activated)

```
VICTIM-OPERATED TRIGGERS
├── Pressure (weight, compression)
├── Pressure-Release (removal of weight)
├── Tension (pull, trip)
├── Tension-Release (cut wire, broken connection)
├── Proximity (infrared, magnetic, radar)
│   ├── Active (emits detection signal)
│   └── Passive (detects change in environment)
├── Movement (trembler, tilt, vibration)
├── Light (photocell, laser break)
└── Target-Selection (discriminating triggers)
    └── Anti-Countermeasure (tamper-activated)
```

---

## COMPONENT SOURCING ANALYSIS

### Supply Chain Intelligence

Every component has a supply chain that can be traced:

```
COMPONENT SOURCING CHAIN
├── Raw Materials → Where do they originate?
├── Manufacturing → Who produces the components?
├── Distribution → How do components reach the actor?
├── Assembly → Where/how is the threat constructed?
└── Employment → How is the threat deployed?
```

### Attribution Through Components

Components provide attribution indicators:

| Component Aspect | Attribution Value |
|------------------|-------------------|
| **Materials** | Geographic origin, supplier network |
| **Construction** | Training origin, tradecraft lineage |
| **Design Pattern** | Group signature, evolution tracking |
| **Tool Marks** | Workshop identification |
| **Sourcing** | Supply chain, financial network |

### Forensic Priorities

```
HIGH ATTRIBUTION VALUE:
├── Unique components (custom-made items)
├── Error signatures (consistent mistakes)
├── Evolution patterns (iterative improvement)
└── Supply chain indicators (packaging, shipping)

MODERATE VALUE:
├── Common components with limited sources
├── Construction techniques
└── Design choices

LOWER VALUE:
├── Widely available components
├── Generic designs
└── Mass-produced materials
```

---

## PATTERN ANALYSIS

### Event Signature Development

Analyze multiple incidents to develop signatures:

```
SIGNATURE DEVELOPMENT PROCESS
│
├── 1. COLLECT component data from multiple events
│
├── 2. CORRELATE common elements across events
│
├── 3. IDENTIFY consistent patterns (signature)
│
├── 4. ATTRIBUTE to actor/network based on signature
│
└── 5. PREDICT future events based on signature evolution
```

### Tactics, Techniques, and Procedures (TTP)

Component analysis reveals TTPs:

| Analysis | Question | Output |
|----------|----------|--------|
| **Tactical** | How is the threat employed? | Employment patterns |
| **Technical** | What is the threat made of? | Component profile |
| **Procedural** | What process produces the threat? | Construction signature |

---

## LATTICEFORGE IMPLEMENTATION

### Component Data Model

```typescript
interface ThreatComponent {
  type: 'trigger' | 'mechanism' | 'effect' | 'power' | 'container' | 'enhancement';

  // Classification
  category: string;
  subcategory: string;

  // Forensic data
  indicators: string[];
  sourcing: {
    origin: string;
    supplier: string;
    method: string;
  };

  // Attribution
  signatures: string[];
  confidence: 'high' | 'moderate' | 'low';

  // Linkage
  related_components: string[];
  associated_actors: string[];
  historical_precedents: string[];
}
```

### Pattern Matching Interface

```
┌────────────────────────────────────────────────────────────┐
│              COMPONENT PATTERN MATCHER                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  INPUT: Observed components                                │
│  ┌──────────────────────────────────────────────────┐     │
│  │ [Trigger: Command-Wireless-LowPower]              │     │
│  │ [Mechanism: Cascading-Persistent]                 │     │
│  │ [Container: Institutional-Cover]                  │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  OUTPUT: Pattern matches                                   │
│  ┌──────────────────────────────────────────────────┐     │
│  │ 87% match: Actor Group Alpha (2019-2023)          │     │
│  │ 72% match: Actor Group Beta (2021-2022)           │     │
│  │ 54% match: Unattributed Cluster C                 │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  EVOLUTION: Component changes over time                    │
│  ┌──────────────────────────────────────────────────┐     │
│  │ 2020: Basic trigger → 2022: Advanced trigger      │     │
│  │ Trend: Increasing sophistication                  │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## INTENDED OUTCOME ANALYSIS

### Tactical Outcomes

| Outcome | Description | Indicators |
|---------|-------------|------------|
| **Anti-Infrastructure** | Target physical systems | Critical infrastructure proximity |
| **Anti-Personnel** | Target individuals | Fragmentation, timing |
| **Anti-Vehicle** | Target transportation | Placement, trigger type |
| **Disruptive** | Impair operations | Timing, location |
| **Coercive** | Force behavior change | Messaging, pattern |

### Strategic Outcomes

| Outcome | Description | Analysis |
|---------|-------------|----------|
| **Political** | Make political statement | Timing with events, target symbolism |
| **TTP Identification** | Probe defenses | Low-consequence events, systematic variation |
| **Experimental** | Test new capabilities | Novel components, unusual design |
| **Obstacle Creation** | Channel movement | Pattern of incidents, geographic analysis |

---

## SUMMARY

The Six-Component Model provides:

1. **Universal Framework** - Applicable across all threat domains
2. **Systematic Analysis** - Consistent methodology
3. **Pattern Recognition** - Cross-incident correlation
4. **Attribution Support** - Component-based sourcing
5. **Prediction Capability** - Evolution tracking

**Key Principle:** Any complex threat is just a system of components. Understand the components, understand the threat.

---

*Document Version 1.0 | Threat Component Model*
*Last Updated: December 2024*
