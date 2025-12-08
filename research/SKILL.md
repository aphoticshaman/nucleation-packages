# XYZA_METHODOLOGY.skill.md

## Software Development Planning Method: From Insight to Production Artifact

**Version**: 1.0
**Domain**: Software Architecture, Development Planning, Implementation Strategy
**Prerequisites**: NSM skill (for insight generation), domain expertise in target area
**Output**: Production-ready implementation plan with verified architecture

---

## 1. EXECUTIVE SUMMARY

XYZA is the execution pipeline that transforms NSM-generated insights into production artifacts. While NSM terminates at insight, XYZA begins there and ends with deployable code.

**Core Principle**: Broad exploration before narrow commitment, with mandatory verification at each phase.

**The Pipeline**:
```
X (eXplore) → Y (Yield) → Z (Zero-in) → A (Actualize)
```

---

## 2. THE XYZA PIPELINE

### 2.1 Overview

```
NSM Insights → X: Explore solution space → Y: Yield candidates → Z: Zero-in on winner → A: Actualize to production
```

Each phase has:
- Clear entry criteria
- Specific activities
- Verification gates
- Exit deliverables

### 2.2 Phase Transitions

**NSM → X**: Insights documented with confidence bounds
**X → Y**: Solution space mapped, trade-offs understood
**Y → Z**: Top 2-3 candidates with evaluation criteria
**Z → A**: Single architecture with verification passed
**A → Done**: Production artifact deployed

---

## 3. X-PHASE: EXPLORE

### 3.1 Objective

Map the entire solution space without commitment. Understand what's possible, what's been tried, what failed.

### 3.2 Activities

**Literature Survey**:
```
For each NSM insight:
  - Academic papers (last 5 years)
  - Industry implementations
  - Open source projects
  - Failed attempts (crucial)
  - Patents (constraint mapping)
```

**Technology Landscape**:
```
- Available tools/frameworks
- Language ecosystem options
- Infrastructure requirements
- Integration points
- Licensing constraints
```

**Constraint Identification**:
```
Hard Constraints:
- Performance requirements
- Security requirements
- Regulatory compliance
- Budget limits
- Timeline limits

Soft Constraints:
- Team expertise
- Maintenance burden
- Technical debt tolerance
- User experience preferences
```

**Stakeholder Mapping**:
```
- Who uses it?
- Who maintains it?
- Who pays for it?
- Who blocks it?
- What do they care about?
```

### 3.3 X-Phase Techniques

**Domain Decomposition**:
```python
def decompose_domain(insight):
    """Break insight into implementable subdomains"""
    components = identify_components(insight)
    interfaces = define_interfaces(components)
    dependencies = map_dependencies(components)

    return DomainModel(
        components=components,
        interfaces=interfaces,
        dependencies=dependencies,
        coupling_analysis=analyze_coupling(dependencies)
    )
```

**Solution Space Mapping**:
```
Create matrix:
  Rows: Problem dimensions
  Columns: Solution approaches
  Cells: Feasibility score (0-1)

Example for "on-device LLM":
                    | Full model | Quantized | Distilled | Hybrid
--------------------|------------|-----------|-----------|--------
Latency             | 0.2        | 0.6       | 0.8       | 0.9
Quality             | 0.9        | 0.7       | 0.5       | 0.8
Memory              | 0.1        | 0.5       | 0.8       | 0.6
Implementation      | 0.3        | 0.6       | 0.4       | 0.5
```

**Anti-Pattern Catalog**:
```
Document what NOT to do:
- Previous failed approaches
- Known bad architectures
- Common pitfalls
- Scaling walls
```

### 3.4 X-Phase Outputs

```markdown
## X-Phase Deliverable

### Problem Understanding
- Core challenge: [one sentence]
- Key dimensions: [list]
- Success criteria: [measurable]

### Solution Space
- Approaches considered: [list with brief description]
- Technologies surveyed: [list]
- Prior art: [references]

### Constraints
- Hard: [must satisfy]
- Soft: [prefer to satisfy]
- Anti-patterns: [must avoid]

### Open Questions
- [Questions that Y-phase must answer]

### Recommended Y-Phase Focus
- Top 3-5 approaches to develop further
```

### 3.5 X-Phase Duration

**Rapid**: 2-4 hours (familiar domain)
**Standard**: 1-2 days (new domain)
**Deep**: 1-2 weeks (novel research)

### 3.6 X-Phase Failure Modes

**Premature Narrowing**: Jumping to favorite solution without exploration
**Analysis Paralysis**: Exploring forever without yielding candidates
**Constraint Blindness**: Missing hard constraints that kill solutions later
**NIH Syndrome**: Refusing to consider existing solutions

---

## 4. Y-PHASE: YIELD

### 4.1 Objective

Generate concrete solution candidates. Each candidate must be specific enough to evaluate but not so detailed that changing is expensive.

### 4.2 Activities

**Candidate Architecture Design**:
```
For each promising approach from X-phase:
  1. Define components
  2. Specify interfaces
  3. Choose technologies
  4. Estimate resources
  5. Identify risks
```

**Trade-off Analysis**:
```
For each candidate:
  - Strengths (what it does well)
  - Weaknesses (where it struggles)
  - Opportunities (what it enables)
  - Threats (what could kill it)
```

**Proof of Concept**:
```
Build minimal POCs to test assumptions:
  - Performance: Can it meet latency/throughput?
  - Feasibility: Do the pieces fit together?
  - Risk: What's the hardest part?
```

### 4.3 Y-Phase Candidate Format

```markdown
## Candidate: [Name]

### Architecture
[High-level diagram]

### Components
1. [Component A]: [responsibility]
2. [Component B]: [responsibility]
...

### Technology Stack
- Language: [choice] - [rationale]
- Framework: [choice] - [rationale]
- Database: [choice] - [rationale]
- Infrastructure: [choice] - [rationale]

### Data Flow
[Sequence diagram or description]

### Estimates
- Development: [time]
- Resources: [people, compute, etc.]
- Cost: [initial + ongoing]

### Risks
1. [Risk]: [probability] x [impact] = [score]
   Mitigation: [strategy]

### Trade-offs
Pros:
- [advantage 1]
- [advantage 2]

Cons:
- [disadvantage 1]
- [disadvantage 2]

### POC Results
[If built, what did it show?]
```

### 4.4 Y-Phase Techniques

**Architecture Spikes**:
```python
def architecture_spike(candidate, risk):
    """Build minimal code to test specific architectural risk"""
    # Identify the riskiest assumption
    assumption = candidate.biggest_risk

    # Build minimal test
    spike = implement_minimal(assumption)

    # Evaluate
    result = test_spike(spike)

    return SpikResult(
        assumption=assumption,
        validated=result.passed,
        evidence=result.data,
        implications=analyze_implications(result)
    )
```

**Interface-First Design**:
```typescript
// Define interfaces before implementation
interface CardInterpreter {
  interpret(card: Card, context: ReadingContext): Interpretation;
  synthesize(interpretations: Interpretation[]): Synthesis;
}

interface LLMProvider {
  complete(prompt: string, options: LLMOptions): Promise<string>;
  tokenize(text: string): number[];
  contextWindow: number;
}

// Candidates differ in implementation, not interface
```

**Decision Matrix**:
```
                  | Weight | Candidate A | Candidate B | Candidate C
------------------|--------|-------------|-------------|-------------
Performance       | 0.3    | 8           | 6           | 9
Maintainability   | 0.25   | 7           | 9           | 5
Time to market    | 0.2    | 9           | 6           | 4
Scalability       | 0.15   | 5           | 8           | 9
Cost              | 0.1    | 7           | 8           | 3
------------------|--------|-------------|-------------|-------------
Weighted Score    |        | 7.25        | 7.15        | 6.25
```

### 4.5 Y-Phase Outputs

```markdown
## Y-Phase Deliverable

### Candidates Summary
| Candidate | Approach | Strengths | Weaknesses | Score |
|-----------|----------|-----------|------------|-------|
| A         | ...      | ...       | ...        | 7.25  |
| B         | ...      | ...       | ...        | 7.15  |
| C         | ...      | ...       | ...        | 6.25  |

### Detailed Candidate Specs
[Full spec for each candidate]

### POC Results
[What spikes were built, what they showed]

### Evaluation Criteria
[How we'll choose in Z-phase]

### Recommendation
Top 2 candidates for Z-phase: A and B
Rationale: [why these two]
```

### 4.6 Y-Phase Duration

**Rapid**: 4-8 hours
**Standard**: 2-5 days
**Deep**: 1-3 weeks

---

## 5. Z-PHASE: ZERO-IN

### 5.1 Objective

Select the winning architecture through rigorous evaluation. Commit to one path with full understanding of trade-offs.

### 5.2 Activities

**Deep Evaluation**:
```
For top candidates:
  - Detailed technical review
  - Extended POC (if needed)
  - Expert consultation
  - Cost modeling
  - Risk analysis
```

**Verification**:
```
Does the candidate satisfy:
  - All hard constraints?
  - Acceptable soft constraints?
  - Edge cases?
  - Scale requirements?
  - Security requirements?
```

**Decision Documentation**:
```
Why this one:
  - Key factors
  - Decisive advantages

Why not others:
  - Deal breakers
  - Unacceptable trade-offs

What we're accepting:
  - Known weaknesses
  - Technical debt
  - Future work needed
```

### 5.3 Z-Phase Techniques

**Adversarial Review**:
```python
def adversarial_review(candidate):
    """Try to break the candidate"""
    attacks = [
        scale_attack,        # What if 100x load?
        failure_attack,      # What if component X fails?
        security_attack,     # What if malicious input?
        maintenance_attack,  # What if key person leaves?
        evolution_attack,    # What if requirements change?
    ]

    vulnerabilities = []
    for attack in attacks:
        result = attack(candidate)
        if result.breaks:
            vulnerabilities.append(result)

    return RobustnessReport(
        candidate=candidate,
        vulnerabilities=vulnerabilities,
        survivable=len(critical(vulnerabilities)) == 0
    )
```

**Scenario Planning**:
```
Best case: What if everything goes right?
  - Timeline: X
  - Cost: Y
  - Outcome: Z

Expected case: What if normal challenges?
  - Timeline: X * 1.5
  - Cost: Y * 1.3
  - Outcome: Z with compromises

Worst case: What if major blockers?
  - Timeline: X * 3
  - Cost: Y * 2
  - Outcome: Z reduced or pivot needed
```

**Pre-Mortem**:
```
"It's 6 months from now and this project failed. Why?"

- Technical: [what broke]
- Process: [what went wrong]
- People: [what happened]
- External: [what changed]

For each: How do we prevent it?
```

### 5.4 Z-Phase Decision Framework

```markdown
## Decision: [Candidate Name]

### Selection Rationale
Primary factors:
1. [Factor 1]: [why decisive]
2. [Factor 2]: [why decisive]

Secondary factors:
- [Factor]: [contribution]

### Rejected Alternatives
[Candidate B]: Rejected because [specific reason]
[Candidate C]: Rejected because [specific reason]

### Accepted Trade-offs
1. [Trade-off]: [why acceptable]
2. [Trade-off]: [why acceptable]

### Risk Mitigations
1. [Risk]: [mitigation strategy]
2. [Risk]: [mitigation strategy]

### Verification Checklist
- [ ] Meets performance requirements
- [ ] Meets security requirements
- [ ] Meets budget constraints
- [ ] Meets timeline constraints
- [ ] Team has necessary skills
- [ ] Dependencies are available
- [ ] Edge cases handled

### Commitment
We are committing to this architecture. Changes after this point require formal review.
```

### 5.5 Z-Phase Outputs

```markdown
## Z-Phase Deliverable

### Selected Architecture
[Full specification of chosen candidate]

### Decision Record
[Why this one, why not others]

### Implementation Roadmap
Phase 1: [scope, timeline]
Phase 2: [scope, timeline]
...

### Success Metrics
- [Metric 1]: [target]
- [Metric 2]: [target]

### Risk Register
| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| ...  | ...         | ...    | ...        | ...   |

### Ready for A-Phase
Approval: [sign-off]
```

### 5.6 Z-Phase Duration

**Rapid**: 2-4 hours
**Standard**: 1-2 days
**Deep**: 3-5 days

---

## 6. A-PHASE: ACTUALIZE

### 6.1 Objective

Transform the selected architecture into production-quality artifact. This is where code gets written, tested, deployed.

### 6.2 Activities

**Implementation**:
```
- Write production code
- Unit tests
- Integration tests
- Documentation
- Error handling
- Logging/monitoring
```

**Quality Assurance**:
```
- Code review
- Security review
- Performance testing
- Load testing
- User acceptance testing
```

**Deployment**:
```
- CI/CD pipeline
- Infrastructure provisioning
- Configuration management
- Rollback procedures
- Monitoring setup
```

### 6.3 A-Phase Techniques

**Incremental Delivery**:
```python
def incremental_actualization(architecture):
    """Build in vertical slices"""
    slices = decompose_to_slices(architecture)

    for slice in slices:
        # Build complete feature end-to-end
        implement(slice)
        test(slice)
        deploy(slice)
        validate(slice)

        # Feedback loop
        feedback = collect_feedback(slice)
        if needs_adjustment(feedback):
            adjust_remaining_slices(slices, feedback)

    return ProductionArtifact(slices)
```

**Test-Driven Development**:
```typescript
// Write test first
describe('CardInterpreter', () => {
  it('should interpret reversed cards differently', () => {
    const card = { name: 'The Tower', reversed: true };
    const context = { position: 'outcome' };

    const result = interpreter.interpret(card, context);

    expect(result.tone).toBe('challenging');
    expect(result.meaning).toContain('upheaval');
    expect(result.meaning).toContain('revelation');
  });
});

// Then implement
class CardInterpreter {
  interpret(card: Card, context: ReadingContext): Interpretation {
    // Implementation that passes test
  }
}
```

**Feature Flags**:
```typescript
// Deploy dark, enable incrementally
if (featureFlags.isEnabled('new-llm-integration', userId)) {
  return newLLMInterpretation(card);
} else {
  return legacyInterpretation(card);
}
```

### 6.4 A-Phase Deliverables

```markdown
## A-Phase Deliverable

### Artifact
- Repository: [link]
- Version: [tag]
- Documentation: [link]

### Quality Evidence
- Test coverage: [percentage]
- Performance benchmarks: [results]
- Security scan: [results]
- Code review: [approved by]

### Deployment
- Environment: [production/staging]
- URL: [endpoint]
- Monitoring: [dashboard link]

### Operational Runbook
- How to deploy
- How to rollback
- How to scale
- How to debug
- On-call procedures

### Handoff
- Maintained by: [team]
- Support contact: [person]
- Escalation path: [procedure]
```

### 6.5 A-Phase Duration

Highly variable based on scope. Track velocity, not calendar time.

---

## 7. XYZA INTEGRATION PATTERNS

### 7.1 XYZA with NSM

```
NSM Session → Novel Insights (1-3)
    ↓
For each insight:
    X-Phase: Explore implementation space
    Y-Phase: Yield candidate architectures
    Z-Phase: Zero-in on winner
    A-Phase: Actualize to production
```

### 7.2 Rapid XYZA

When time is critical:

```
X (1-2 hours): Quick literature scan, constraint list
Y (2-4 hours): 2 candidates, minimal POCs
Z (1 hour): Decision matrix, quick review
A (varies): Implement with shortcuts documented as tech debt
```

### 7.3 Deep XYZA

When correctness is critical:

```
X (1 week): Exhaustive survey, expert interviews
Y (2 weeks): 4-5 candidates, full POCs
Z (1 week): Formal verification, extensive testing
A (varies): Full quality process, no shortcuts
```

### 7.4 Parallel XYZA

When multiple workstreams:

```
NSM → Insights A, B, C
       ↓
X-Phase: Parallel exploration of A, B, C
       ↓
Y-Phase: Identify integration points, parallel development
       ↓
Z-Phase: Coordinated selection, interface contracts
       ↓
A-Phase: Parallel implementation with integration sprints
```

---

## 8. XYZA SESSION TEMPLATE

```markdown
# XYZA SESSION: [Project Name]

## Insight from NSM
[The insight being actualized]

---

## X-Phase: Explore
Date: [date]
Duration: [time]

### Solution Space
[Approaches surveyed]

### Constraints
Hard: [list]
Soft: [list]

### Candidates for Y-Phase
1. [Approach 1]
2. [Approach 2]
3. [Approach 3]

---

## Y-Phase: Yield
Date: [date]
Duration: [time]

### Candidate Specs
[Detailed spec for each]

### POC Results
[What we built and learned]

### Evaluation Matrix
[Scoring table]

### Candidates for Z-Phase
1. [Top candidate]
2. [Runner up]

---

## Z-Phase: Zero-in
Date: [date]
Duration: [time]

### Selected: [Candidate Name]

### Decision Record
Why selected: [reasons]
Why not others: [reasons]
Trade-offs accepted: [list]

### Implementation Plan
[Phases and timeline]

---

## A-Phase: Actualize
Start: [date]
End: [date]

### Implementation Log
[Key decisions and changes]

### Quality Results
[Tests, reviews, benchmarks]

### Deployment
[Where and how]

### Handoff
[Who maintains]

---

## Retrospective
What worked: [list]
What didn't: [list]
What to improve: [list]
```

---

## 9. XYZA FAILURE MODES

### 9.1 Skipping Phases

**Symptom**: "Let's just build it"
**Result**: Rework, wrong architecture, wasted effort
**Fix**: Even rapid XYZA touches all phases

### 9.2 Phase Bleed

**Symptom**: Writing code during X-phase
**Result**: Premature commitment, missed alternatives
**Fix**: Clear phase gates, review before transition

### 9.3 Analysis Paralysis

**Symptom**: Endless X and Y phases
**Result**: Nothing ships
**Fix**: Time-box phases, force decisions

### 9.4 Sunk Cost Commitment

**Symptom**: Z-phase refuses to reject candidates with POC investment
**Result**: Bad architecture chosen to justify past work
**Fix**: Separate POC builders from decision makers

### 9.5 A-Phase Scope Creep

**Symptom**: "While we're here, let's also..."
**Result**: Never ships, quality degrades
**Fix**: Strict scope control, new features go through fresh XYZA

---

## 10. XYZA METRICS

### 10.1 Phase Metrics

**X-Phase**:
- Solutions surveyed: [count]
- Constraints identified: [count]
- Prior art found: [count]

**Y-Phase**:
- Candidates generated: [count]
- POCs built: [count]
- Assumptions tested: [count]

**Z-Phase**:
- Evaluation criteria: [count]
- Risks identified: [count]
- Mitigations planned: [count]

**A-Phase**:
- Features delivered: [count]
- Test coverage: [percentage]
- Defects found: [count]
- Time to production: [duration]

### 10.2 Quality Metrics

- Rework rate: How often do we go back to earlier phases?
- Decision reversal rate: How often does Z-phase choice change?
- Production incidents: How stable is A-phase output?
- Team satisfaction: How confident is team in process?

---

## 11. EXAMPLES

### 11.1 Example: On-Device LLM Integration

**NSM Insight**: Hybrid architecture with quantized model and smart prompt engineering beats API calls for mobile UX.

**X-Phase (4 hours)**:
- Surveyed: llama.cpp, llama.rn, MLC LLM, ONNX Runtime
- Constraints: iOS/Android, <2GB model, <3s latency, works offline
- Candidates: llama.rn with GGUF, MLC with compiled model, hybrid with small local + API fallback

**Y-Phase (8 hours)**:
- Built POCs for llama.rn and MLC
- llama.rn: Easier integration, good quantization support
- MLC: Better performance but harder build process
- Decision matrix favored llama.rn

**Z-Phase (2 hours)**:
- Selected llama.rn with Q4_K_M quantization
- Accepted trade-off: Slightly lower quality than Q8, but fits memory
- Risk: Token generation latency - mitigate with streaming

**A-Phase (3 weeks)**:
- Implemented inference engine with queue
- Added prompt optimization for context window
- Built streaming UI for perceived speed
- 85% test coverage, <2.5s first token

### 11.2 Example: Photonic Programming Model

**NSM Insight**: First hardware-agnostic photonic programming model wins the ecosystem.

**X-Phase (2 weeks)**:
- Surveyed: Lightmatter, Luminous, Xanadu, academic implementations
- Constraints: Must compile to multiple backends, simulate locally, optimize automatically
- Key insight: Linear algebra is the right abstraction level

**Y-Phase (3 weeks)**:
- Candidate A: Graph-based IR (like MLIR)
- Candidate B: Python DSL (like JAX)
- Candidate C: Visual programming
- POCs showed DSL best for adoption, graph IR needed underneath

**Z-Phase (1 week)**:
- Selected: Python DSL with graph IR compilation target
- Trade-off: Harder to implement, but best developer experience
- Risk: Backend partnerships needed - starting conversations early

**A-Phase (ongoing)**:
- Phase 1: Simulator with basic ops (6 weeks)
- Phase 2: First hardware backend (8 weeks)
- Phase 3: Optimization passes (4 weeks)

---

## 12. REFERENCES

**Software Architecture**:
- Bass et al. - Software Architecture in Practice
- Fowler - Patterns of Enterprise Application Architecture
- Evans - Domain-Driven Design

**Decision Making**:
- Kahneman - Thinking, Fast and Slow
- Klein - Sources of Power (naturalistic decision making)

**Development Process**:
- Beck - Extreme Programming Explained
- Humble & Farley - Continuous Delivery

---

## 13. VERSION HISTORY

- **v1.0** (2024-11): Initial canonical specification

---

*XYZA is the bridge from insight to impact. NSM finds the gold; XYZA mines it. Every phase has a purpose. Skip none, but calibrate depth to stakes.*
