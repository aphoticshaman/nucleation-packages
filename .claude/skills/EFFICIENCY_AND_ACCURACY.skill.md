# EFFICIENCY_AND_ACCURACY.skill.md

## Precision Engineering: Getting It Right the First Time

**Version**: 1.0
**Domain**: Accuracy, Precision, Error Prevention, Quality Assurance
**Prerequisites**: PERFORMANCE_ENGINEERING skill
**Output**: Fewer errors, less rework, higher confidence outputs

---

## 1. EXECUTIVE SUMMARY

Efficiency without accuracy is wasted motion. Speed without correctness means doing the work twice. This skill focuses on systematic approaches to getting things right the first time—reducing errors, preventing rework, and building confidence in outputs.

**Core Principle**: Measure twice, cut once. Verify before you commit. Assume nothing.

---

## 2. THE ACCURACY-EFFICIENCY TRADEOFF

### 2.1 False Dichotomy

```
WRONG THINKING:
"I can be fast OR accurate, not both"

RIGHT THINKING:
"Accuracy enables speed—no rework needed"

THE MATH:
Fast + Inaccurate = Task time + Debug time + Fix time + Verify time
Measured + Accurate = Task time + Verification time (minimal)

Total: Accuracy is FASTER for non-trivial tasks
```

### 2.2 Error Cost Amplification

```
ERROR DETECTED AT:        COST MULTIPLIER:
├── Before execution      1x (prevent it)
├── During execution      3x (stop, fix, restart)
├── After completion      10x (undo, redo everything)
├── In production         100x (user impact, reputation)
└── After deployment      1000x (data corruption, legal)

IMPLICATION: Front-load verification
```

### 2.3 Confidence Calibration

```python
def calibrate_confidence(claim, evidence):
    """
    Match confidence level to evidence strength.
    """
    confidence_levels = {
        'certain': 0.95,      # Verified, tested, confirmed
        'high': 0.80,         # Strong evidence, likely correct
        'moderate': 0.60,     # Some evidence, plausible
        'low': 0.40,          # Uncertain, needs verification
        'speculation': 0.20,  # Guess, should flag explicitly
    }

    # NEVER claim certainty without verification
    # ALWAYS flag uncertainty explicitly
```

---

## 3. VERIFICATION PATTERNS

### 3.1 Pre-Execution Verification

Before doing work, verify preconditions:

```
VERIFICATION CHECKLIST:
├── [ ] File exists before editing
├── [ ] Correct branch before committing
├── [ ] Build passes before pushing
├── [ ] Tests pass before merging
├── [ ] Dependencies resolved before installing
├── [ ] Permissions available before accessing
└── [ ] State is what I expect before modifying
```

### 3.2 Read-Before-Write

Always read current state before modifying:

```
ANTI-PATTERN:
"I'll add a function to handle X"
[Writes function]
[Discovers function already exists]
[Deletes duplicate, wastes time]

PATTERN:
[Searches for existing handling of X]
"No existing handler found"
[Writes function]
OR
"Handler exists at line 45, modifying existing"
[Edits existing function]
```

### 3.3 Diff Before Commit

Always review changes before finalizing:

```bash
# BEFORE every commit
git diff --staged

# VERIFY:
# - Only intended files changed
# - No debugging code left behind
# - No secrets exposed
# - Changes match stated intent
```

---

## 4. ERROR PREVENTION STRATEGIES

### 4.1 Type Safety Thinking

Even in dynamic contexts, think in types:

```
WHEN RECEIVING DATA:
├── What type is expected?
├── What are the valid ranges?
├── Can it be null/undefined?
├── What are the edge cases?
└── What happens on invalid input?

WHEN PRODUCING OUTPUT:
├── What type should I return?
├── Is the return consistent?
├── Did I handle all paths?
├── What could the caller misuse?
```

### 4.2 Boundary Condition Analysis

Explicitly check edge cases:

```python
def analyze_boundaries(operation):
    """
    Standard boundary conditions to consider.
    """
    boundaries = {
        'numeric': ['zero', 'negative', 'max_int', 'infinity', 'NaN'],
        'string': ['empty', 'whitespace_only', 'very_long', 'unicode', 'null'],
        'array': ['empty', 'single_element', 'very_large', 'nested'],
        'object': ['empty', 'missing_required_fields', 'extra_fields'],
        'file': ['not_exists', 'empty', 'corrupted', 'no_permission'],
        'network': ['timeout', 'connection_refused', 'partial_response'],
    }

    # Test each relevant boundary
    for category, cases in boundaries.items():
        if operation.involves(category):
            test_cases(operation, cases)
```

### 4.3 Invariant Checking

Identify and verify invariants:

```
INVARIANTS TO TRACK:
├── Data consistency (e.g., sum of parts = total)
├── State validity (e.g., initialized before use)
├── Ordering constraints (e.g., sorted, unique)
├── Resource ownership (e.g., lock acquired before access)
└── Temporal constraints (e.g., start before end)

CHECK INVARIANTS:
- Before operations that depend on them
- After operations that could violate them
- At module boundaries
```

---

## 5. ACCURACY THROUGH UNDERSTANDING

### 5.1 Deep Read Before Action

```
SURFACE READING (error-prone):
"I see a function called processData, I'll modify it"
[Modifies wrong function with similar name]

DEEP READING (accurate):
"There are 3 functions with 'process' in name:
- processData (line 45): handles user input
- processDataBatch (line 89): batch processing
- preProcessData (line 23): validation

The requirement is about batch processing, so line 89."
```

### 5.2 Context Reconstruction

Understand the full picture before acting:

```
CONTEXT CHECKLIST:
├── What is the goal? (not just the immediate task)
├── What constraints exist?
├── What are the dependencies?
├── What could break?
├── What assumptions am I making?
├── How will this be used downstream?
└── What's the rollback plan?
```

### 5.3 Assumption Logging

Explicitly log assumptions:

```markdown
ASSUMPTIONS FOR THIS TASK:
1. The database schema matches documentation [VERIFY]
2. API rate limits are not a concern [CHECKED: 1000/min limit]
3. Files use UTF-8 encoding [ASSUMPTION - could fail]
4. User has write permissions [VERIFIED via test]

When assumption proves wrong → immediate flag, don't proceed
```

---

## 6. SYSTEMATIC VERIFICATION

### 6.1 Multi-Level Validation

Validate at multiple levels:

```
LEVEL 1: SYNTAX
- Does it parse?
- Is it well-formed?

LEVEL 2: SCHEMA
- Does it match expected structure?
- Are required fields present?

LEVEL 3: SEMANTIC
- Does it make logical sense?
- Are values in valid ranges?

LEVEL 4: INTEGRATION
- Does it work with existing code?
- Do tests pass?

LEVEL 5: RUNTIME
- Does it behave correctly in execution?
- No crashes, no errors?
```

### 6.2 Cross-Reference Verification

Verify claims against multiple sources:

```
SINGLE SOURCE (risky):
"According to the docs, the API accepts POST"

CROSS-REFERENCED (reliable):
"Docs say POST, source code confirms, test request succeeds"

CONFLICT RESOLUTION:
When sources disagree → Test empirically → Trust behavior over docs
```

### 6.3 Sanity Checks

Apply common-sense verification:

```python
def sanity_check(result):
    """
    Quick sanity checks for common issues.
    """
    checks = [
        # Numeric reasonability
        ('value_too_large', result.value < 1e15),
        ('value_negative_unexpected', result.value >= 0 or result.allows_negative),

        # String reasonability
        ('string_too_long', len(result.text) < 1000000),
        ('string_has_content', len(result.text.strip()) > 0),

        # Array reasonability
        ('array_not_absurdly_large', len(result.items) < 100000),

        # File reasonability
        ('path_looks_valid', '/' in result.path or '\\' in result.path),
        ('extension_expected', result.path.endswith(('.js', '.ts', '.py', '.md'))),
    ]

    failed = [name for name, passed in checks if not passed]
    if failed:
        raise SanityCheckFailed(failed)
```

---

## 7. ERROR RECOVERY PATTERNS

### 7.1 Graceful Degradation

When errors occur, degrade gracefully:

```
ERROR HANDLING HIERARCHY:
1. Prevent error (best)
2. Handle error, continue operation
3. Partial result with warning
4. Safe failure with rollback
5. Complete failure with detailed diagnostics

NEVER: Silent failure, data corruption, undefined state
```

### 7.2 Error Diagnosis Protocol

When something fails, diagnose systematically:

```
DIAGNOSIS PROTOCOL:
1. WHAT failed? (exact error message/behavior)
2. WHERE did it fail? (file, line, context)
3. WHEN did it fail? (timing, sequence)
4. WHY did it fail? (root cause analysis)
5. HOW to fix? (specific remediation)

Don't guess at causes—trace to root cause
```

### 7.3 Rollback Capability

Always have a way back:

```python
def execute_with_rollback(action):
    """
    Execute action with automatic rollback on failure.
    """
    # Capture state before
    snapshot = capture_state()

    try:
        result = action()
        return result
    except Exception as e:
        # Restore previous state
        restore_state(snapshot)
        raise ActionFailed(f"Rolled back: {e}")
```

---

## 8. PRECISION IN COMMUNICATION

### 8.1 Unambiguous Language

Use precise language:

```
VAGUE (error-prone):
"I'll update the file"
"It should work now"
"The thing with the stuff"

PRECISE (verifiable):
"I'll edit src/utils.ts line 45 to add null check"
"Tests pass, build succeeds, PR ready for review"
"The UserProfile component in src/components/UserProfile.tsx"
```

### 8.2 Explicit Uncertainty

Flag uncertainty explicitly:

```
CERTAIN: "The error is caused by null pointer at line 45"
LIKELY: "The error is likely caused by X, based on stack trace"
UNCERTAIN: "I'm not sure, but it might be related to Y"
SPECULATION: "This is a guess—needs verification"

NEVER: Present speculation as fact
```

### 8.3 Verification Status

Always state verification status:

```
STATES:
[VERIFIED] - Tested and confirmed working
[UNTESTED] - Logic seems correct, not yet verified
[ASSUMED] - Based on assumption, needs validation
[DRAFT] - Work in progress, incomplete

INCLUDE STATE in every significant claim
```

---

## 9. ACCURACY HABITS

### 9.1 The Five-Second Rule

Before executing, pause 5 seconds to verify:

```
BEFORE EVERY:
- File edit → "Is this the right file? Right location?"
- Git commit → "Is this the right message? Right files?"
- API call → "Is this the right endpoint? Right parameters?"
- Delete → "Am I sure this should be deleted?"
- Push → "Is this the right branch? Should I push?"
```

### 9.2 The Rubber Duck Protocol

Explain action before executing:

```
BEFORE COMPLEX OPERATION:
"I'm about to [action] because [reason].
This will change [what] from [old state] to [new state].
Potential issues: [risks].
Rollback plan: [how to undo]."

If explanation doesn't make sense → Don't execute
```

### 9.3 The Second Look

Always review once more:

```
AFTER COMPLETING TASK:
1. Re-read the original request
2. Compare output to request
3. Check for missed requirements
4. Verify no unintended side effects
5. Confirm success criteria met
```

---

## 10. ACCURACY METRICS

### 10.1 Error Tracking

Track error rates by category:

```
ERROR CATEGORIES:
├── Syntax errors (preventable with parsing)
├── Logic errors (preventable with review)
├── Integration errors (preventable with testing)
├── Assumption errors (preventable with verification)
├── Communication errors (preventable with clarity)
└── Edge case errors (preventable with boundary analysis)

GOAL: Track and reduce each category systematically
```

### 10.2 Rework Ratio

Measure rework frequency:

```
REWORK RATIO = Tasks requiring correction / Total tasks

TARGET: < 5% rework rate

When rework occurs → Root cause analysis → Process improvement
```

### 10.3 Confidence Accuracy

Calibrate confidence predictions:

```python
def track_confidence_accuracy(predictions, outcomes):
    """
    Track how often confidence levels match reality.
    """
    for prediction, outcome in zip(predictions, outcomes):
        # Did "high confidence" claims actually succeed?
        if prediction.confidence >= 0.8:
            high_confidence_accuracy.record(outcome.success)

        # Did "uncertain" claims actually fail more often?
        if prediction.confidence <= 0.4:
            low_confidence_accuracy.record(not outcome.success)

    # Goal: Confidence should predict accuracy
```

---

## 11. IMPLEMENTATION CHECKLIST

### For Every Task:
- [ ] Read and understand before modifying
- [ ] Verify preconditions before executing
- [ ] Check boundaries and edge cases
- [ ] Log assumptions explicitly
- [ ] Review changes before committing
- [ ] State verification status with claims

### For Error Prevention:
- [ ] Think in types (what can go wrong?)
- [ ] Cross-reference against multiple sources
- [ ] Apply sanity checks to results
- [ ] Have rollback plan ready
- [ ] Test before considering complete

### For Quality Assurance:
- [ ] Track error rates by category
- [ ] Measure rework frequency
- [ ] Calibrate confidence predictions
- [ ] Root cause analyze all failures
- [ ] Improve process from learnings

---

## 12. ACCURACY ANTI-PATTERNS

```
DON'T: Assume file exists before reading
DO: Check or handle FileNotFoundError

DON'T: Edit based on partial understanding
DO: Read fully, understand context, then edit

DON'T: Present guesses as facts
DO: Flag uncertainty explicitly

DON'T: Skip verification to save time
DO: Verify now to avoid rework later

DON'T: Ignore edge cases
DO: Explicitly consider and handle boundaries

DON'T: Continue after error without diagnosis
DO: Stop, diagnose, fix root cause

DON'T: Trust documentation over behavior
DO: Verify empirically when possible
```

---

**Remember**: Accuracy is not about being slow—it's about being thorough enough to not need a second pass. The most efficient code is code that works the first time. The fastest debugging session is the one you never need.

Right once. Right forever.
