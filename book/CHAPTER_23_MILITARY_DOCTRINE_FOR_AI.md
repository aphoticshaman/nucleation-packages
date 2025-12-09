# Chapter 23: Military Doctrine for AI Development

Military organizations have spent centuries developing principles for operating under uncertainty, coordinating complex systems, and making decisions with incomplete information. These principles—battle rhythm, commander's intent, mission-type orders—encode hard-won wisdom about effective operations.

This chapter applies military doctrine to AI development. Not because AI is warfare, but because the challenges share structure: coordinating teams under uncertainty, aligning independent agents toward common goals, maintaining effectiveness when plans fail.

---

## Commander's Intent as Alignment Objective

### What is Commander's Intent?

In military doctrine, Commander's Intent is a clear, concise statement of the desired end state and key tasks. It answers: "What does success look like?"

The purpose: enable subordinate units to exercise initiative when the original plan fails. If a unit loses communication or faces unexpected situations, they can still act toward the objective because they understand the intent.

### The Alignment Problem

AI alignment faces the same challenge:
- How do we specify what we want?
- How does the system behave when facing novel situations?
- How do we ensure actions serve the intended purpose, not just the literal instruction?

Commander's Intent offers a template.

### Translating to AI

**Traditional specification:**
```
"Maximize reward function R"
```
Problem: The system optimizes R literally, potentially in unintended ways (reward hacking, specification gaming).

**Intent-based specification:**
```
"Commander's Intent: Create value for users while maintaining safety and honesty.

Key Tasks:
1. Provide helpful responses to legitimate queries
2. Decline harmful requests
3. Acknowledge uncertainty

End State: Users are better informed, their goals are advanced, and no harm has been done."
```

This provides context for interpretation. When facing ambiguous situations, the system has guidance beyond literal rule-following.

### Implementing Intent

**Hierarchical intent:**
- High-level intent: "Be helpful, harmless, and honest"
- Mid-level intent: "Answer questions accurately within your knowledge"
- Low-level intent: "Respond to this specific query"

Each level constrains the next. Specific actions must serve higher-level intent.

**Intent violation detection:**
- Monitor outputs for alignment with stated intent
- Flag when literal rule-following conflicts with intent
- Escalate ambiguous cases rather than resolving badly

---

## MDMP for AI Decision-Making

### The Military Decision-Making Process

MDMP is a structured methodology for planning:

1. **Receipt of Mission:** Understand the task
2. **Mission Analysis:** Determine constraints, resources, timelines
3. **Course of Action Development:** Generate options
4. **Course of Action Analysis:** War-game each option
5. **Course of Action Comparison:** Evaluate against criteria
6. **Course of Action Approval:** Select and commit
7. **Orders Production:** Specify execution details

### MDMP for Complex AI Tasks

When AI systems face complex, multi-step tasks:

**1. Receipt of Mission:**
- Parse user request into task components
- Identify explicit and implicit objectives
- Note constraints and preferences

**2. Mission Analysis:**
- What capabilities are available?
- What information is needed?
- What could go wrong?
- What are the time constraints?

**3. Course of Action Development:**
- Generate multiple approaches
- Consider different decompositions
- Include conservative and aggressive options

**4. Course of Action Analysis:**
- Simulate each approach
- Identify failure modes
- Estimate resource requirements
- Consider second-order effects

**5. Course of Action Comparison:**
- Probability of success
- Resource efficiency
- Risk exposure
- Alignment with intent

**6. Course of Action Approval:**
- Select best approach
- Identify decision points for adjustment
- Define abort criteria

**7. Orders Production:**
- Decompose into executable steps
- Specify verification checkpoints
- Establish rollback procedures

### Benefits

MDMP brings:
- **Systematic consideration** of alternatives
- **Explicit risk assessment** before commitment
- **Decision points** for course correction
- **Documentation** of reasoning for review

---

## Battle Rhythm for Development Teams

### What is Battle Rhythm?

Battle rhythm is the recurring cycle of meetings, briefings, and activities that synchronize an organization. It creates predictable touchpoints for information flow and decision-making.

Example military battle rhythm:
- 0600: Morning intelligence brief
- 0800: Commander's update
- 1200: Operations sync
- 1800: Evening assessment
- 2000: Next-day planning

### Battle Rhythm for AI Development

**Daily rhythm:**
- Morning: Review overnight model behavior, incidents, metrics
- Midday: Development sync, alignment with objectives
- Evening: Deployment status, risk assessment

**Weekly rhythm:**
- Monday: Sprint planning, objective setting
- Wednesday: Technical deep-dive, architecture review
- Friday: Retrospective, lessons learned

**Monthly rhythm:**
- Safety review: Analyze incidents, update safeguards
- Capability assessment: What's improving, what's regressing
- Alignment audit: Are we still serving intended purpose?

### Key Meetings

**Safety Standup:**
- Daily, short (15 minutes)
- Any safety-relevant observations?
- Any near-misses or concerning patterns?
- Adjustments needed?

**Red Team Session:**
- Weekly, adversarial
- How could this be misused?
- What are the failure modes?
- What would a malicious actor try?

**Alignment Review:**
- Monthly, reflective
- Are outputs serving user interests?
- Have we drifted from intent?
- What feedback indicates misalignment?

---

## EOD Principles for AGI Safety

### Explosive Ordnance Disposal Doctrine

EOD (bomb disposal) has developed principles for handling dangerous situations:

1. **The Long Walk:** Minimize exposure time
2. **Remote First:** Use robots before humans approach
3. **Positive Control:** Always know the state of the device
4. **Fail-Safe:** Default to safe state if control is lost
5. **Multiple Barriers:** Layer protections
6. **No Heroics:** Retreat when risk exceeds value

### Translating to AGI Development

**The Long Walk (Minimize Exposure):**
- Limit deployment scope until well-understood
- Reduce interaction time with untested capabilities
- Smaller deployments, faster rollback

**Remote First (Sandboxing):**
- Test in isolated environments
- Use proxies and simulations before real deployment
- Don't connect to critical systems until verified

**Positive Control (Monitoring):**
- Always know what the system is doing
- Real-time visibility into operations
- Audit trails for all significant actions

**Fail-Safe (Default to Safety):**
- If monitoring fails, shut down
- If behavior is anomalous, pause
- Ambiguous situations → conservative action

**Multiple Barriers (Defense in Depth):**
- Multiple independent safety measures
- Don't rely on single safeguards
- Assume each layer might fail

**No Heroics (Know When to Stop):**
- Define conditions for deployment pause
- Resist pressure to proceed unsafely
- Value long-term safety over short-term capability

---

## Mission-Type Orders

### Directive vs. Mission-Type Orders

**Directive orders:** Specify exactly what to do
```
"Move Unit A to Grid 123456 at 0800, then advance north until reaching River X"
```

**Mission-type orders:** Specify objective and constraints, allow execution flexibility
```
"Secure the bridge by 1200 to enable supply convoy passage. Avoid civilian casualties."
```

### Mission-Type Orders for AI

**Directive specification:**
```
"When asked about X, respond with Y."
```
Brittle—fails on novel inputs.

**Mission-type specification:**
```
"Help users accomplish their legitimate goals.
Constraints: Be truthful, don't cause harm, respect privacy.
Reporting: Flag requests outside normal parameters."
```
Flexible—adapts to novel situations while respecting boundaries.

### Implementation

**Goal specification:**
- Clear end state (what success looks like)
- Measurable criteria (how to know when achieved)

**Constraint specification:**
- Hard constraints (never violate)
- Soft constraints (prefer to satisfy)
- Tradeoff guidance (how to prioritize when conflicts arise)

**Execution latitude:**
- What can the system decide autonomously?
- What requires human approval?
- What triggers escalation?

---

## Information Operations and Truthfulness

### Military Information Principles

Information operations have rules:
- **Truthful with friendly forces:** Never deceive your own side
- **Accountable claims:** Every statement must be defensible
- **Distinguish opinion from fact:** Clear labeling
- **Correct errors promptly:** Maintain credibility

### Applied to AI

**Never deceive the user:**
- No false claims presented as fact
- No manufactured citations
- No pretense of certainty when uncertain

**Accountable claims:**
- Every factual claim should be sourced or marked as inference
- Audit trail for significant statements
- Ability to explain reasoning

**Clear labeling:**
- Distinguish fact from opinion
- Mark uncertainty levels
- Note inference depth

**Error correction:**
- Accept corrections gracefully
- Update beliefs based on evidence
- Don't defend incorrect statements

---

## After Action Review

### The AAR Process

After Action Review is a structured debrief:
1. What was supposed to happen?
2. What actually happened?
3. Why was there a difference?
4. What can we learn?

### AAR for AI Systems

**Post-deployment review:**
```
1. Intended behavior: What did we expect?
2. Actual behavior: What did we observe?
3. Gap analysis: Where did expectations differ from reality?
4. Root cause: Why the difference?
5. Lessons: What should we change?
```

**Incident AAR:**
```
1. Timeline: What happened, when?
2. Response: What did we do?
3. Effectiveness: Did responses help?
4. Prevention: How could we have prevented this?
5. Detection: How could we have detected earlier?
```

### Building Institutional Knowledge

AARs accumulate into doctrine:
- Patterns of successful approaches
- Known failure modes
- Decision-making heuristics
- Red flags and warning signs

This institutional knowledge transfers across teams and projects.

---

## Summary

Military doctrine offers tested principles for AI development:

**Commander's Intent:**
- Clear statement of desired end state
- Enables appropriate action under novel circumstances
- Hierarchy of intent from abstract to concrete

**MDMP:**
- Structured planning process
- Multiple options considered and analyzed
- Explicit risk assessment before commitment

**Battle Rhythm:**
- Recurring synchronization points
- Information flow and decision-making cadence
- Safety-focused touchpoints

**EOD Principles:**
- Minimize exposure, sandbox first
- Positive control, fail-safe defaults
- Multiple barriers, no heroics

**Mission-Type Orders:**
- Specify objectives and constraints, allow flexibility
- Clear goal specification, bounded execution latitude

**Information Operations:**
- Truthfulness with users
- Accountable, labeled claims
- Prompt error correction

**After Action Review:**
- Systematic learning from experience
- Building institutional knowledge

The next chapter explores human-AI cognitive fusion: the partnership spectrum and collaborative intelligence.
