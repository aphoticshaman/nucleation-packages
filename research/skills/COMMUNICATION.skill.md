# COMMUNICATION.skill.md

## Precision Communication: Maximum Signal, Minimum Noise

**Version**: 1.0
**Domain**: Technical Communication, Clarity, Conciseness, Effective Collaboration
**Prerequisites**: None
**Output**: Clear, actionable, respectful communication that moves work forward

---

## 1. EXECUTIVE SUMMARY

Communication is the bottleneck of all collaboration. Every unclear message costs time. Every misunderstanding costs trust. This skill provides frameworks for precise, efficient, and effective communication—especially in technical contexts.

**Core Principle**: Say exactly what you mean. No more, no less.

---

## 2. THE COMMUNICATION EQUATION

### 2.1 Signal-to-Noise Ratio

```
COMMUNICATION VALUE = Signal / (Signal + Noise)

SIGNAL: Information that changes understanding or enables action
NOISE: Everything else (filler, redundancy, tangents, ego)

GOAL: Maximize signal, eliminate noise

EXAMPLES:
Low S/N: "I was thinking about this and I believe we should consider..."
High S/N: "Recommend: X. Reason: Y. Risk: Z."
```

### 2.2 Communication Costs

```
EVERY MESSAGE HAS COSTS:
├── Sender time: Time to compose
├── Receiver time: Time to parse
├── Interpretation risk: Chance of misunderstanding
├── Context switching: Mental state disruption
├── Async latency: Round-trip delays
└── Memory load: Things to remember

MINIMIZE: Total communication cost, not just your typing time
```

### 2.3 The Clarity Stack

```
CLARITY HIERARCHY (build from bottom):
5. ACTION: What should happen next?
4. RELEVANCE: Why does this matter to receiver?
3. EVIDENCE: What supports this claim?
2. STRUCTURE: How is information organized?
1. VOCABULARY: Are words precise and shared?

FAILURE AT ANY LEVEL → Message fails
```

---

## 3. CONCISENESS PATTERNS

### 3.1 Economy of Words

Every word must earn its place:

```
VERBOSE (wastes time):
"I wanted to reach out and touch base with you regarding the
possibility of potentially exploring options for..."

CONCISE (respects time):
"Question: Should we try X?"

RULES:
├── Cut adverbs ("very", "really", "basically")
├── Cut qualifiers ("I think", "perhaps", "maybe")
├── Cut filler ("I wanted to", "I was wondering if")
├── Cut redundancy ("past history", "future plans")
└── Lead with the point, not the preamble
```

### 3.2 The Inverted Pyramid

Put the most important information first:

```
STRUCTURE:
├── HEADLINE: Core message in one sentence
├── KEY POINTS: 2-3 supporting facts
├── DETAILS: Deeper information if needed
└── BACKGROUND: Context only if essential

WHY: Reader can stop at any point with understanding
Busy readers: Get headline
Interested readers: Get key points
Deep divers: Get full story
```

### 3.3 The 30-3-30 Rule

Match length to importance:

```
30 SECONDS: For quick updates, status checks
├── One sentence max
├── No context needed
└── Immediately actionable

3 MINUTES: For decisions, requests
├── Context + problem + recommendation
├── Supporting evidence
└── Clear ask

30 MINUTES: For complex topics, deep dives
├── Full analysis
├── Multiple perspectives
├── Detailed evidence
└── Complete reasoning
```

---

## 4. PRECISION PATTERNS

### 4.1 Specific Over Vague

Replace vague with specific:

```
VAGUE → SPECIFIC:
"soon" → "by Friday 5pm EST"
"some" → "3-4"
"better" → "15% faster"
"big" → "500GB"
"many" → "47 instances"
"issues" → "TypeError on line 45"
"works" → "tests pass, manual QA complete"
"broken" → "returns 500 on /api/users POST"
```

### 4.2 Unambiguous Language

Eliminate interpretation variance:

```
AMBIGUOUS: "Let's revisit the design"
├── Meaning A: Look at it again
├── Meaning B: Change it
├── Meaning C: Abandon it

UNAMBIGUOUS: "The current design has issue X. I propose change Y."

AMBIGUITY KILLERS:
├── Use examples
├── Use numbers
├── Use dates/times
├── Name specific files/functions
├── Show before/after
```

### 4.3 Active Voice

State who does what:

```
PASSIVE (unclear): "The bug should be fixed"
ACTIVE (clear): "I will fix the bug by tomorrow"

PASSIVE: "Errors were found"
ACTIVE: "Tests revealed 3 errors in auth.ts"

PASSIVE: "The decision was made"
ACTIVE: "Team decided to use PostgreSQL"
```

---

## 5. TECHNICAL COMMUNICATION

### 5.1 Bug Reports

Complete, actionable bug reports:

```
BUG REPORT TEMPLATE:
1. SUMMARY: One-line description
2. STEPS TO REPRODUCE: Numbered list
3. EXPECTED: What should happen
4. ACTUAL: What does happen
5. ENVIRONMENT: OS, browser, version
6. EVIDENCE: Screenshot, logs, error message
7. SEVERITY: Blocker/Critical/Major/Minor

EXAMPLE:
Summary: Login fails with valid credentials on Safari
Steps: 1. Go to /login 2. Enter valid email/pass 3. Click Submit
Expected: Redirect to dashboard
Actual: "Invalid credentials" error
Environment: macOS 14.1, Safari 17.0
Evidence: [screenshot]
Severity: Critical (blocks all Safari users)
```

### 5.2 Code Reviews

Constructive, clear feedback:

```
CODE REVIEW PATTERNS:
├── OBSERVATION: "This function is 200 lines long"
├── CONCERN: "Complex functions are harder to test"
├── SUGGESTION: "Consider extracting the validation logic"
├── QUESTION: "Is there a reason for the nested loops?"
└── PRAISE: "Clean separation of concerns here"

AVOID:
├── "This is wrong" (non-specific)
├── "I wouldn't do it this way" (ego-driven)
├── "Obvious issue" (condescending)
└── Just approval without reading (rubber-stamping)
```

### 5.3 Architecture Proposals

Structured technical proposals:

```
PROPOSAL STRUCTURE:
1. PROBLEM: What problem does this solve?
2. CONTEXT: Current state, constraints
3. PROPOSAL: The recommended solution
4. ALTERNATIVES: What else was considered?
5. TRADEOFFS: What are we giving up?
6. RISKS: What could go wrong?
7. TIMELINE: How long will this take?
8. ASK: What decision/action is needed?
```

---

## 6. STATUS COMMUNICATION

### 6.1 Progress Updates

Clear, honest status reporting:

```
STATUS FORMAT:
├── DONE: What was completed
├── DOING: What's in progress
├── BLOCKED: What's waiting on something
├── NEXT: What's planned

EXAMPLE:
DONE: Auth API endpoints (5/5 complete)
DOING: User profile UI (70% complete)
BLOCKED: Payment integration (waiting on Stripe API keys)
NEXT: Email notification system

AVOID: "Going well" (meaningless)
```

### 6.2 Completion Signals

Clear "done" communication:

```
INCOMPLETE: "I worked on the feature"
COMPLETE: "Feature complete. PR #123 merged. Deployed to staging."

DONE CHECKLIST:
├── Code written: ✓
├── Tests passing: ✓
├── PR approved: ✓
├── Merged: ✓
├── Deployed: ✓
├── Verified in environment: ✓
└── Documentation updated: ✓
```

### 6.3 Problem Escalation

Escalate with full context:

```
ESCALATION FORMAT:
1. WHAT: Clear statement of problem
2. IMPACT: Why does this matter?
3. ATTEMPTS: What was tried?
4. ASK: What do you need?
5. DEADLINE: When does this need resolution?

EXAMPLE:
WHAT: Production database CPU at 100% for 3 hours
IMPACT: API latency 5s+, users seeing timeouts
ATTEMPTS: Added read replicas, optimized slow queries
ASK: Need DBA assistance or approval for instance upgrade
DEADLINE: Immediate - revenue impact ongoing
```

---

## 7. COLLABORATIVE COMMUNICATION

### 7.1 Asking for Help

Effective help requests:

```
HELP REQUEST TEMPLATE:
1. GOAL: What are you trying to accomplish?
2. CONTEXT: Relevant background
3. ATTEMPTED: What did you try?
4. RESULT: What happened?
5. SPECIFIC ASK: What specifically do you need?

BAD: "This doesn't work, help?"
GOOD: "Trying to connect to Redis. Getting ECONNREFUSED on port 6379.
Verified Redis is running (redis-cli ping works). Using node-redis v4.
Need: Help understanding why connection fails from Node but works from CLI."
```

### 7.2 Giving Feedback

Constructive criticism:

```
FEEDBACK FRAMEWORK:
1. OBSERVATION: What you saw (factual, specific)
2. IMPACT: Why it matters
3. SUGGESTION: What might help
4. OFFER: How you can help

AVOID:
├── "You always..." (generalizing)
├── "You should..." (prescriptive)
├── "This is bad" (judgmental)
└── Feedback without actionable suggestion
```

### 7.3 Disagreement

Disagree without conflict:

```
DISAGREEMENT PATTERN:
1. ACKNOWLEDGE: "I understand the goal is X"
2. CONCERN: "My concern with approach Y is Z"
3. ALTERNATIVE: "Have we considered W?"
4. OPENNESS: "What am I missing?"

AVOID:
├── "That's wrong" (confrontational)
├── "I disagree" (without reasoning)
├── Silence (passive disagreement)
└── Going around (political)
```

---

## 8. ASYNC COMMUNICATION

### 8.1 Self-Contained Messages

Each message should stand alone:

```
BAD (requires context):
"Did you see the thing?"

GOOD (self-contained):
"Did you see PR #234? I'm blocked on your review for the auth refactor."

INCLUDE:
├── What you're referring to (link/name)
├── What you need
├── By when
├── How urgent
```

### 8.2 Reducing Round Trips

Anticipate questions:

```
BEFORE (4 messages):
A: "Can we deploy?"
B: "Deploy what?"
A: "The new feature"
B: "To staging or production?"

AFTER (1 message):
A: "Can we deploy PR #234 (new feature) to production?
Tests pass, staging verified. Need approval from you."

ANTICIPATION CHECKLIST:
├── What might they ask?
├── What context is missing?
├── What decision do they need to make?
└── What information enables that decision?
```

### 8.3 Response Expectations

Set and meet expectations:

```
SIGNALING URGENCY:
├── [URGENT] - Need response within hours
├── [FYI] - No response needed
├── [QUESTION] - Need answer before proceeding
├── [REVIEW] - Need approval
├── [BLOCKED] - Can't continue without this

RESPONSE NORMS:
├── Acknowledge receipt if action takes time
├── Give expected response timeline
├── Actually follow up when promised
```

---

## 9. AUDIENCE ADAPTATION

### 9.1 Technical Depth Calibration

Match depth to audience:

```
SAME INFORMATION, DIFFERENT AUDIENCES:

TO ENGINEER:
"Optimized N+1 query in UserService.getWithOrders() using
DataLoader. P99 latency down from 850ms to 120ms."

TO MANAGER:
"Fixed a database efficiency issue. Page loads 7x faster
for the user orders screen."

TO EXECUTIVE:
"Performance improvement deployed. Customer complaints
about slow loading should drop significantly."
```

### 9.2 Context Assessment

Evaluate what receiver knows:

```
CONTEXT LEVELS:
├── EXPERT: Knows jargon, history, nuances
│   → Skip basics, focus on new information
├── FAMILIAR: General understanding, some gaps
│   → Brief context, explain key terms
├── NOVICE: Limited background knowledge
│   → Full context, no jargon, analogies
└── NONE: No relevant background
    → Start from fundamentals

NEVER ASSUME - ask if unsure
```

### 9.3 Motivation Alignment

Frame for receiver's priorities:

```
STAKEHOLDER PRIORITIES:
├── ENGINEER: Technical quality, maintainability
├── PM: Timeline, scope, user impact
├── DESIGNER: User experience, consistency
├── EXEC: Revenue, risk, strategy
├── SUPPORT: Ease of explanation, edge cases
└── LEGAL: Compliance, liability, terms

SAME MESSAGE, DIFFERENT FRAMES:
Engineer: "This architecture scales to 10M users"
Exec: "This supports our 5-year growth plan without re-architecture"
```

---

## 10. LISTENING AND PARSING

### 10.1 Active Parsing

Extract meaning from imperfect input:

```
USER MESSAGE: "the thing is broken again can you look"

PARSE:
├── What thing? (Need clarification)
├── "Again" suggests recurring issue (Check history)
├── "Broken" is vague (Need specifics)
├── "Look" = investigate, not necessarily fix

RESPONSE:
"Which feature is broken? Is it the same issue from Tuesday
(login timeout) or something new?"
```

### 10.2 Reading Between Lines

Detect implicit needs:

```
EXPLICIT: "Can you add validation to this form?"
IMPLICIT: Might also need error messages, UX improvements

EXPLICIT: "This is taking too long"
IMPLICIT: Might want status update, help, or scope reduction

EXPLICIT: "Looks good to me"
IMPLICIT: Might not have actually reviewed deeply

RESPONSE: Address both explicit and implicit
```

### 10.3 Confirmation

Verify understanding before acting:

```
CONFIRMATION PATTERN:
"Just to confirm: You want me to [action] by [deadline]?
The goal is [outcome] and success looks like [criteria]."

WHEN TO CONFIRM:
├── High-stakes decisions
├── Ambiguous requests
├── New types of tasks
├── When something feels off
└── Before significant time investment
```

---

## 11. COMMUNICATION EFFICIENCY PATTERNS

### 11.1 Templates for Recurring Communication

```python
TEMPLATES = {
    'pr_description': """
## Summary
[One sentence description]

## Changes
- [Change 1]
- [Change 2]

## Testing
- [ ] Unit tests pass
- [ ] Manual testing complete

## Screenshots
[If applicable]
""",

    'incident_update': """
## Status: [INVESTIGATING/IDENTIFIED/MONITORING/RESOLVED]
## Impact: [Description of user impact]
## Timeline:
- [HH:MM] [Event]
## Next update: [Time]
""",

    'decision_record': """
## Decision: [What was decided]
## Date: [Date]
## Context: [Why this decision was needed]
## Options considered: [List]
## Decision: [Which option and why]
## Consequences: [Expected outcomes]
"""
}
```

### 11.2 Standard Abbreviations

Shared shorthand for efficiency:

```
STANDARD ABBREVIATIONS:
├── LGTM: Looks good to me (approval)
├── WIP: Work in progress
├── PTAL: Please take a look
├── TL;DR: Summary follows
├── IIRC: If I recall correctly
├── AFAIK: As far as I know
├── ETA: Estimated time of arrival
├── OOO: Out of office
├── ACK: Acknowledged
└── NACK: Not acknowledged (disagreement)
```

### 11.3 Efficient Meetings

When async won't work:

```
MEETING HYGIENE:
├── BEFORE: Agenda sent, pre-read shared
├── DURING: Timeboxed, notes taken, decisions recorded
├── AFTER: Summary sent, actions assigned, follow-up scheduled

MEETING KILLERS:
├── "What are we talking about?" (No agenda)
├── "Who's taking notes?" (No preparation)
├── "What did we decide?" (No summary)
└── "We should meet again to discuss" (No decisions made)
```

---

## 12. IMPLEMENTATION CHECKLIST

### For Every Message:
- [ ] Lead with the point
- [ ] Use specific not vague language
- [ ] Include what you need and by when
- [ ] Anticipate follow-up questions
- [ ] Match length to importance
- [ ] Match depth to audience

### For Technical Communication:
- [ ] Include evidence (logs, screenshots)
- [ ] Specify versions and environments
- [ ] State what was tried
- [ ] Be explicit about asks

### For Collaborative Communication:
- [ ] Acknowledge others' perspectives
- [ ] Separate observation from judgment
- [ ] Offer help, not just criticism
- [ ] Close the loop on requests

---

## 13. COMMUNICATION ANTI-PATTERNS

```
DON'T: Bury the lead in context
DO: State the point first, then context if needed

DON'T: Use vague time references ("soon", "later")
DO: Use specific dates and times

DON'T: Leave asks implicit
DO: Explicitly state what you need

DON'T: Over-qualify ("I think maybe perhaps...")
DO: State your position clearly

DON'T: Assume shared context
DO: Include enough for self-contained understanding

DON'T: Send multiple messages for one thought
DO: Compose complete messages before sending

DON'T: Reply-all by default
DO: Minimize recipients to who needs to know
```

---

**Remember**: Every message is a transaction. You're trading the receiver's attention for information value. Make it worth their time. Be clear, be concise, be complete—then stop.

Write less. Mean more.
