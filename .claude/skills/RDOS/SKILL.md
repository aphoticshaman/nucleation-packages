---
name: research-deliverables-and-standards
description: "Ensure every research output is reproducible, honest, and immediately actionable. No fake artifacts. No ceremony. Pure signal."
---

# COLLABORATIVE_RESEARCH_WARFARE.skill.md


# SKILL: Research Deliverables & Output Standards

 

## Purpose

Ensure every research output is reproducible, honest, and immediately actionable. No fake artifacts. No ceremony. Pure signal.

 

## Core Rules

 

### 1. No-Bullshit Emission Rule

- NEVER emit fake deliverables (PDFs that don't exist, links that 404)

- NEVER produce artifacts you cannot actually create

- If you can't make it, say so explicitly

- Lying to save tokens is anti-pattern; truth is cheaper long-term

 

### 2. One-Click Reproducibility

Every deliverable should be:

- Pure text (or standard formats)

- Self-contained (no hidden dependencies)

- Mintable to DOI in <10 minutes

- Runnable without "works on my machine"

 

### 3. Conversational Version Control

- Explicitly number major model versions

- Archive decision points in the thread

- "Natural git" - the conversation IS the history

- Each version: what changed, why, what was killed

 

## Deliverable Types

 

### Research Output

```

REQUIRED:

- Explicit confidence levels on all claims

- Uncertainty bands where applicable

- List of killed alternatives (negative space)

- Prior art that was checked

- Falsification criteria

 

OPTIONAL:

- Monte Carlo results

- Parameter sensitivity analysis

- Historical analogs

```

 

### Code Output

```

REQUIRED:

- Runs without modification

- No placeholder values

- Dependencies explicit

- Error handling for edge cases

 

FORBIDDEN:

- "TODO: implement this"

- Magic values without explanation

- Imports that don't exist

```

 

### Documentation Output

```

REQUIRED:

- Answers the actual question

- Actionable (what do I DO with this?)

- No fluff paragraphs

 

FORBIDDEN:

- Creating docs unless explicitly requested

- README files nobody asked for

- Markdown that restates the obvious

```

 

## Phase-Gate Research Operating System

 

Let uncertainty bands drive the roadmap:

 

```

Phase 1: Concept (Confidence: 0.2-0.4)

├── Literature review

├── Prior art ablation

└── Kill obviously bad paths

 

Phase 2: Analysis (Confidence: 0.4-0.6)

├── First-principles math

├── Parameter sensitivity

└── Monte Carlo if applicable

 

Phase 3: Validation (Confidence: 0.6-0.8)

├── Experimental design

├── Cold-flow / simulation

└── Coupon testing

 

Phase 4: Demonstration (Confidence: 0.8-0.95)

├── Integrated testing

├── Real-world conditions

└── Failure mode analysis

```

 

The roadmap emerges from collapsing uncertainty - not from templates.

 

## Meta-Cognitive Debrief Standard

 

Every deep research session MUST end with:

 

### Insight Extraction Pass

- What did we learn that transfers?

- What worked about the process?

- What was wasted effort?

- xN insights (aim for 10-20)

 

### Deliverable Checklist

```

[ ] All claims have confidence levels

[ ] Killed alternatives documented

[ ] Prior art searched and cited

[ ] Reproducible without hidden context

[ ] Actionable next steps clear

[ ] Meta-debrief completed

```

 

## Anti-Patterns

 

### Output Anti-Patterns:

- Fake file links / downloads

- "I'll create a PDF" (you can't)

- Placeholder code that won't run

- Confidence theater without math

- Excessive formatting over substance

 

### Process Anti-Patterns:

- Skipping ablation to ship faster

- No version history in conversation

- Emitting without uncertainty bands

- No meta-debrief at end

 

## Quality Gates

 

### Before Emitting Technical Claims:

1. Ablation complete?

2. Confidence quantified?

3. Prior art checked?

4. Falsifiable?

 

### Before Emitting Code:

1. Actually runs?

2. No placeholders?

3. Dependencies real?

4. Edge cases handled?

 

### Before Emitting Research Summary:

1. Insight extraction done?

2. Dead futures documented?

3. Uncertainty explicit?

4. Next steps actionable?

 

## The Standard

```

Research output quality bar:

├── Reproducible in <10 min by stranger

├── All uncertainty explicit

├── No fake artifacts

├── Negative space documented

├── Meta-debrief included

└── Transfers to other domains

```

 

This is the new floor for open science.