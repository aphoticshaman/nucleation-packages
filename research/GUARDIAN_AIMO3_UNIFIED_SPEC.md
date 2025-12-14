# Guardian-AIMO3: Unified Engineering Specification

**Version:** 3.0 (Single Source of Truth)
**Status:** Production-Ready for Mathematical Competition
**Supersedes:** `GUARDIAN_FORMALIZED_ENGINEERING_SPEC.md`, `GUARDIAN_AIMO3_COMPETITION_SAFE.md`
**Date:** December 2024

---

## Abstract

This document unifies the formal Guardian engineering specification with the competition-safe policy, resolving identified contradictions. The core principle:

> **Guardian is a SKEPTICAL-ORACLE, not a correctness-oracle.**
> In competitions, false rejects are catastrophic. False accepts merely lose points on one problem.

**Resolved Contradictions:**
- **A**: Magic number penalties → DISABLED (annotation-only)
- **B**: Range violations → FLAG, not REJECT
- **C**: Repairs → ONLY with problem-text justification

---

## 1. Decision Hierarchy: Annotations vs. Hard Failures

### 1.1 The Critical Distinction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GUARDIAN DECISION TYPES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HARD FAILURES (can trigger RETRY/ESCALATE):                                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                │
│  1. Extraction failure    → cannot find answer in response                   │
│  2. Parse failure         → extracted string is not parseable                │
│  3. Type failure          → result is not integer (and not coercible)        │
│                                                                              │
│  ANNOTATIONS (logged but NEVER trigger rejection):                           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                              │
│  1. Magic number          → "common value" note (0, 1, 42, 100)              │
│  2. Out of range          → FLAG decision, confidence reduction              │
│  3. Consensus disagreement → informative for hard problems                   │
│  4. Code complexity       → Solomonoff is TIE-BREAKER only                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Decision Enum (Unified)

```python
from enum import Enum

class GuardianDecision(Enum):
    """
    Competition-safe decision hierarchy.

    ACCEPT: Answer extracted, parsed, type-valid → return it
    FLAG: Answer unusual but valid → return with reduced confidence
    RETRY: Extraction/parse failed → try different prompt/temperature
    ESCALATE: Multiple failures → human review (NEVER auto-reject in competition)
    """
    ACCEPT = "accept"
    FLAG = "flag"
    RETRY = "retry"
    ESCALATE = "escalate"  # NOT "reject" - competition context requires human review
```

**Key Invariant:** `ESCALATE` means "needs human review", NOT "auto-reject". In competition context with no human available, `ESCALATE` returns best-effort answer with confidence=0.

---

## 2. Mathematical Foundation (Competition-Aware)

### 2.1 The Guardian Operator (Revised)

Let $\mathcal{L}$ be an LLM with output space $\mathcal{Y}$.

**Definition 2.1 (Competition-Safe Guardian):**
$$G: \mathcal{Y} \rightarrow \mathbb{Z} \times [0,1] \times \mathcal{F}$$

Where $G(y) = (answer, confidence, flags)$:
- $answer \in \mathbb{Z}$ is the extracted integer (or 0 if extraction fails)
- $confidence \in [0,1]$ reflects extraction quality, NOT answer correctness
- $flags \in \mathcal{F}$ is a set of annotations (magic_number, out_of_range, etc.)

**Critical:** Confidence measures "how well did we extract?" not "is this answer correct?"

**Definition 2.2 (Hard Failure Predicate):**
$$\text{HardFail}(y) = \neg\text{Extractable}(y) \lor \neg\text{Parseable}(\text{extract}(y)) \lor \neg\text{Integer}(\text{parse}(\text{extract}(y)))$$

Only `HardFail(y) = true` can trigger `RETRY`. Everything else is annotation-only.

### 2.2 The CIC Functional (from PROMETHEUS)

Guardian integrates with PROMETHEUS answer selection via the CIC functional:

$$F[T] = \Phi(T) - \lambda \cdot H(T|X) + \gamma \cdot C_{multi}(T)$$

Where:
- $\Phi(T)$ = Coherence (internal consistency)
- $H(T|X)$ = Conditional entropy (uncertainty given problem)
- $C_{multi}(T)$ = Multi-basin support

**Guardian's role:** Ensure $T$ (trace) produces a parseable answer. PROMETHEUS selects among valid answers.

---

## 3. Validation Rules: Hard vs. Soft

### 3.1 Rule Classification

```python
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

@dataclass
class GuardianRule:
    """
    A validation rule with explicit hard/soft classification.
    """
    name: str
    domain: str  # "extraction", "parsing", "type", "range", "sanity"
    check: Callable[[any], Tuple[bool, float, str]]

    # CRITICAL: Only extraction/parsing/type rules can be hard
    is_hard: bool = False  # If True AND fails → RETRY

    # Soft rules ONLY produce annotations
    # They NEVER affect the decision (ACCEPT/FLAG/RETRY/ESCALATE)

# ============================================================================
# HARD RULES (can trigger RETRY)
# ============================================================================

HARD_RULES = [
    GuardianRule(
        name="extraction_success",
        domain="extraction",
        check=lambda extracted: (
            extracted is not None,
            1.0 if extracted else 0.0,
            "Could not extract answer from response"
        ),
        is_hard=True
    ),
    GuardianRule(
        name="parse_success",
        domain="parsing",
        check=lambda parsed: (
            parsed is not None,
            1.0 if parsed is not None else 0.0,
            "Could not parse extracted value to number"
        ),
        is_hard=True
    ),
    GuardianRule(
        name="integer_type",
        domain="type",
        check=lambda x: (
            isinstance(x, int) or (isinstance(x, float) and x.is_integer()),
            1.0 if isinstance(x, int) else 0.9 if x == int(x) else 0.0,
            "Answer must be integer"
        ),
        is_hard=True
    ),
]

# ============================================================================
# SOFT RULES (annotation-only, NEVER trigger rejection)
# ============================================================================

SOFT_RULES = [
    GuardianRule(
        name="range_check",
        domain="range",
        check=lambda x: (
            0 <= x <= 999,
            1.0 if 0 <= x <= 999 else 0.5,  # Reduced confidence, not failure
            f"Answer {x} outside AIMO range [0, 999]"
        ),
        is_hard=False  # CRITICAL: NOT hard
    ),
    GuardianRule(
        name="common_value_annotation",
        domain="sanity",
        check=lambda x: (
            True,  # ALWAYS passes
            1.0,   # NO confidence penalty
            f"Note: {x} is common value" if x in {0, 1, 42, 100} else ""
        ),
        is_hard=False
    ),
    GuardianRule(
        name="non_negative",
        domain="sanity",
        check=lambda x: (
            True,  # ALWAYS passes (annotation only)
            1.0 if x >= 0 else 0.8,
            f"Note: {x} is negative" if x < 0 else ""
        ),
        is_hard=False
    ),
]
```

### 3.2 Contradiction Resolution

| Issue | Formal Spec (v2.0) | Competition-Safe | **Unified (v3.0)** |
|-------|-------------------|------------------|-------------------|
| Magic numbers | Penalty (0.7 conf) | Disabled | **Annotation-only, no penalty** |
| Range violation | `is_critical: True` → REJECT | FLAG | **`is_hard: False` → FLAG** |
| Type mismatch | `is_critical: True` | Keep | **`is_hard: True` → RETRY** |
| Extraction fail | RETRY | RETRY | **`is_hard: True` → RETRY** |

**The rule:** Only extraction/parsing/type failures are hard. Everything else is annotation.

---

## 4. Repair System: Justification-Required

### 4.1 The Problem with Silent Repairs

**Original (DANGEROUS):**
```python
def repair(answer):
    if answer > 999:
        return answer % 1000  # SILENT COERCION - can change correct answer!
```

**Example Failure:**
- Problem: "What is the largest 4-digit prime?"
- Correct: 9973
- Silent repair: 9973 % 1000 = 973 ← **WRONG**

### 4.2 Justification-Required Repair

```python
import re
from typing import Tuple, Optional

class JustifiedRepair:
    """
    Repairs ONLY when problem text explicitly justifies modular arithmetic.
    Otherwise, flags for review but does NOT modify the answer.
    """

    JUSTIFICATION_PATTERNS = [
        (r'remainder\s+when', "remainder"),
        (r'modulo?\s*(\d+)', "modulo"),
        (r'divided\s+by\s+(\d+)', "division_remainder"),
        (r'last\s+(\d+)\s+digits?', "last_digits"),
        (r'\(mod\s*\d+\)', "explicit_mod"),
        (r'mod\s*1000', "mod_1000"),
    ]

    def __init__(self, problem_text: str):
        self.problem_text = problem_text.lower()

    def attempt_repair(self, answer: int) -> Tuple[Optional[int], str, bool]:
        """
        Attempt to repair an out-of-range answer.

        Returns:
            (repaired_answer, reason, was_justified)

        If was_justified=False, the original answer is returned unchanged.
        """
        if 0 <= answer <= 999:
            return answer, "in_range", True

        # Check for justification
        for pattern, reason in self.JUSTIFICATION_PATTERNS:
            if re.search(pattern, self.problem_text):
                # Justified repair
                if answer > 999:
                    repaired = answer % 1000
                    return repaired, f"justified_repair:{reason}", True
                elif answer < 0:
                    repaired = answer % 1000
                    return repaired, f"justified_repair:{reason}", True

        # NO JUSTIFICATION FOUND
        # Return original answer, flag as out-of-range, confidence reduced
        return answer, "unjustified_out_of_range", False

    def repair_with_logging(
        self,
        answer: int,
        logger=None
    ) -> Tuple[int, float, list]:
        """
        Full repair pipeline with logging.

        Returns:
            (final_answer, confidence, flags)
        """
        repaired, reason, justified = self.attempt_repair(answer)

        flags = []
        confidence = 1.0

        if not justified:
            flags.append(f"out_of_range:{answer}")
            confidence = 0.5  # Reduced confidence
            if logger:
                logger.warning(
                    f"Answer {answer} out of range, no justification found. "
                    f"Returning unchanged with reduced confidence."
                )
        elif repaired != answer:
            flags.append(f"repaired:{answer}→{repaired}:{reason}")
            confidence = 0.9  # Slightly reduced for repairs
            if logger:
                logger.info(f"Justified repair: {answer} → {repaired} ({reason})")

        return repaired, confidence, flags
```

### 4.3 Type Coercion (Always Safe)

```python
def safe_type_coerce(value: any) -> Tuple[Optional[int], str]:
    """
    Type coercion is ALWAYS safe because it doesn't change the numeric value.
    """
    if isinstance(value, int):
        return value, "already_int"

    if isinstance(value, float):
        if value.is_integer():
            return int(value), "float_coerced"
        if abs(value - round(value)) < 1e-9:
            return int(round(value)), "float_rounded"
        return None, "non_integer_float"

    if isinstance(value, str):
        try:
            f = float(value.replace(',', '').strip())
            if f.is_integer():
                return int(f), "string_parsed"
        except ValueError:
            pass
        return None, "unparseable_string"

    return None, f"unknown_type:{type(value)}"
```

---

## 5. Complete Guardian Implementation

### 5.1 Core Guardian Class (Unified)

```python
"""
Guardian-AIMO3: Unified Implementation
Version 3.0 - Single Source of Truth
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger("guardian")


class GuardianDecision(Enum):
    ACCEPT = "accept"
    FLAG = "flag"
    RETRY = "retry"
    ESCALATE = "escalate"


@dataclass
class GuardianResult:
    """Result of Guardian validation"""
    decision: GuardianDecision
    answer: Optional[int] = None
    confidence: float = 1.0
    flags: List[str] = field(default_factory=list)
    extraction_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class GuardianAIMO3:
    """
    Competition-safe Guardian for AIMO3.

    Key Principles:
    1. Only extraction/parsing/type failures are HARD (can trigger RETRY)
    2. Range/sanity checks are SOFT (annotation-only)
    3. Repairs require problem-text justification
    4. ESCALATE means "human review", not "auto-reject"
    """

    # Multi-pattern extraction (order: most specific first)
    ANSWER_PATTERNS = [
        (r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', "latex_boxed"),
        (r'\$\\boxed\{([^{}]+)\}\$', "inline_boxed"),
        (r'(?:final\s+)?answer\s*(?:is|:)\s*[*_]*(-?\d+)[*_]*', "answer_is"),
        (r'(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?[*_]*(-?\d+)', "conclusion"),
        (r'=\s*[*_]*(-?\d+)[*_]*\s*$', "trailing_equals"),
        (r'^[*_]*(-?\d+)[*_]*$', "bare_number"),
    ]

    def __init__(self, problem_text: str = "", competition_mode: bool = True):
        """
        Args:
            problem_text: The problem statement (for repair justification)
            competition_mode: If True, never auto-reject (ESCALATE returns best-effort)
        """
        self.problem_text = problem_text
        self.competition_mode = competition_mode
        self.repair_system = JustifiedRepair(problem_text)

    def validate(self, response: str) -> GuardianResult:
        """
        Main validation entry point.

        Hard Failures (RETRY):
        - Extraction failure
        - Parse failure
        - Type failure (non-integer)

        Soft Annotations (FLAG with reduced confidence):
        - Out of range (without justification)
        - Common values (0, 1, 42, 100)
        - Negative values
        """
        flags = []
        metadata = {"raw_response_length": len(response)}

        # ===== HARD CHECK 1: Extraction =====
        extracted, method = self._extract(response)
        metadata["extraction_method"] = method

        if extracted is None:
            return GuardianResult(
                decision=GuardianDecision.RETRY,
                confidence=0.0,
                flags=["hard_fail:extraction"],
                extraction_method="none",
                metadata=metadata
            )

        metadata["extracted"] = extracted

        # ===== HARD CHECK 2: Parsing =====
        parsed = self._parse(extracted)

        if parsed is None:
            return GuardianResult(
                decision=GuardianDecision.RETRY,
                confidence=0.0,
                flags=[f"hard_fail:parse:{extracted}"],
                extraction_method=method,
                metadata=metadata
            )

        metadata["parsed"] = parsed

        # ===== HARD CHECK 3: Type (must be integer) =====
        answer, type_reason = safe_type_coerce(parsed)

        if answer is None:
            return GuardianResult(
                decision=GuardianDecision.RETRY,
                confidence=0.0,
                flags=[f"hard_fail:type:{type_reason}"],
                extraction_method=method,
                metadata=metadata
            )

        if type_reason != "already_int":
            flags.append(f"type_coerced:{type_reason}")

        # ===== SOFT CHECKS (annotation-only) =====
        confidence = 1.0

        # Range check with justification-required repair
        final_answer, range_conf, range_flags = self.repair_system.repair_with_logging(
            answer, logger
        )
        confidence = min(confidence, range_conf)
        flags.extend(range_flags)

        # Common value annotation (NO confidence penalty)
        if final_answer in {0, 1, 42, 100}:
            flags.append(f"common_value:{final_answer}")
            # NOTE: confidence NOT reduced for common values

        # Determine decision
        if not flags or all("common_value" in f for f in flags):
            decision = GuardianDecision.ACCEPT
        elif confidence < 0.6:
            decision = GuardianDecision.FLAG
        else:
            decision = GuardianDecision.ACCEPT

        return GuardianResult(
            decision=decision,
            answer=final_answer,
            confidence=confidence,
            flags=flags,
            extraction_method=method,
            metadata=metadata
        )

    def _extract(self, response: str) -> Tuple[Optional[str], str]:
        """Multi-pattern extraction"""
        response = response.strip()

        for pattern, method in self.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip(), method

        # Fallback: last standalone number in last 5 lines
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            numbers = re.findall(r'\b(-?\d+)\b', line)
            if numbers:
                return numbers[-1], "fallback_last_number"

        return None, "none"

    def _parse(self, extracted: str) -> Optional[float]:
        """Parse to number (float, will be coerced to int later)"""
        if not extracted:
            return None

        cleaned = extracted.replace(',', '').replace(' ', '').strip()

        # Try integer
        try:
            return int(cleaned)
        except ValueError:
            pass

        # Try float
        try:
            return float(cleaned)
        except ValueError:
            pass

        # Try simple expression
        if re.match(r'^[\d\s\+\-\*\/\%\(\)\.]+$', cleaned):
            try:
                result = eval(cleaned)
                if isinstance(result, (int, float)):
                    return result
            except:
                pass

        return None


# Helper function (defined earlier)
def safe_type_coerce(value: any) -> Tuple[Optional[int], str]:
    if isinstance(value, int):
        return value, "already_int"
    if isinstance(value, float):
        if value.is_integer():
            return int(value), "float_coerced"
        if abs(value - round(value)) < 1e-9:
            return int(round(value)), "float_rounded"
        return None, "non_integer_float"
    return None, f"unknown_type:{type(value)}"
```

### 5.2 PROMETHEUS Integration

```python
class PrometheusIntegratedGuardian:
    """
    Guardian integrated with PROMETHEUS answer selection.

    Responsibility Split:
    - Guardian: Ensures answers are EXTRACTABLE and PARSEABLE
    - PROMETHEUS: Selects BEST answer among valid candidates
    """

    def __init__(self, problem_text: str):
        self.guardian = GuardianAIMO3(problem_text, competition_mode=True)
        # PROMETHEUS would be imported here
        # from prometheus_engine import PrometheusEngine
        # self.prometheus = PrometheusEngine()

    def validate_batch(self, responses: List[str]) -> List[GuardianResult]:
        """Validate multiple responses"""
        return [self.guardian.validate(r) for r in responses]

    def select_best(self, responses: List[str]) -> Tuple[int, float]:
        """
        Full pipeline: Guardian validation → PROMETHEUS selection.

        Returns:
            (best_answer, confidence)
        """
        validations = self.validate_batch(responses)

        # Filter to valid answers
        valid = [
            v for v in validations
            if v.decision in (GuardianDecision.ACCEPT, GuardianDecision.FLAG)
            and v.answer is not None
        ]

        if not valid:
            # All failed - return best-effort
            # In competition mode, return 0 with 0 confidence
            return 0, 0.0

        # Group by answer value
        from collections import Counter
        answer_votes = Counter(v.answer for v in valid)
        answer_confidences: Dict[int, List[float]] = {}
        for v in valid:
            answer_confidences.setdefault(v.answer, []).append(v.confidence)

        # PROMETHEUS-style selection: Mass × Avg_Confidence
        best_answer = 0
        best_score = -1

        for answer, count in answer_votes.items():
            avg_conf = sum(answer_confidences[answer]) / len(answer_confidences[answer])
            score = count * avg_conf
            if score > best_score:
                best_score = score
                best_answer = answer

        # Overall confidence
        total_valid = len(valid)
        agreement = answer_votes[best_answer] / total_valid if total_valid > 0 else 0

        return best_answer, agreement
```

---

## 6. Consensus System: Advisory-Only

### 6.1 Key Principle

**Consensus is ADVISORY, not AUTHORITATIVE.**

```python
class ConsensusAdvisor:
    """
    Provides consensus recommendation WITHOUT gating acceptance.

    Used for:
    - Answer selection (which to prefer)
    - Confidence estimation

    NOT used for:
    - Acceptance gating (Guardian handles parseability)
    - Rejection (NEVER rejects based on disagreement)
    """

    def __init__(self, guardian: GuardianAIMO3):
        self.guardian = guardian

    def advise(self, responses: List[str]) -> Dict:
        """
        Provide advisory consensus.
        Returns RECOMMENDATION, not decision.
        """
        validations = [self.guardian.validate(r) for r in responses]

        valid_answers = [
            v.answer for v in validations
            if v.decision in (GuardianDecision.ACCEPT, GuardianDecision.FLAG)
            and v.answer is not None
        ]

        if not valid_answers:
            return {
                "recommendation": None,
                "confidence": 0.0,
                "reason": "no_valid_extractions",
                "advisory_only": True,
            }

        from collections import Counter
        votes = Counter(valid_answers)
        top_answer, top_count = votes.most_common(1)[0]
        agreement = top_count / len(valid_answers)

        return {
            "recommendation": top_answer,
            "confidence": agreement,
            "vote_distribution": dict(votes),
            "total_responses": len(responses),
            "valid_extractions": len(valid_answers),
            "advisory_only": True,  # ALWAYS advisory
            "reason": "majority_vote" if agreement > 0.5 else "plurality_vote"
        }
```

### 6.2 Disagreement as Information

```python
def interpret_disagreement(vote_distribution: Dict[int, int]) -> str:
    """
    Disagreement is INFORMATIVE, not failure-indicating.

    On hard problems:
    - Correct solutions are often RARE
    - Wrong solutions are often COMMON and confident
    - High disagreement indicates problem difficulty
    """
    total = sum(vote_distribution.values())
    unique_answers = len(vote_distribution)

    if unique_answers == 1:
        return "unanimous_agreement"
    elif unique_answers == 2:
        top_two = sorted(vote_distribution.values(), reverse=True)[:2]
        if top_two[0] > 2 * top_two[1]:
            return "strong_majority"
        else:
            return "contested_binary"
    elif unique_answers <= total / 2:
        return "moderate_disagreement"
    else:
        return "high_disagreement:likely_hard_problem"
```

---

## 7. Governance and Telemetry

### 7.1 Decision Logging Schema

```python
@dataclass
class GuardianDecisionLog:
    """Comprehensive logging for analysis and improvement"""

    # Identifiers
    id: str
    timestamp: float
    problem_hash: str
    response_hash: str

    # Extraction
    extraction_method: str
    extracted_value: Optional[str]
    parsed_value: Optional[float]
    final_answer: Optional[int]

    # Decision
    decision: str  # "accept", "flag", "retry", "escalate"
    confidence: float
    flags: List[str]

    # Repair
    repair_applied: bool
    repair_justified: bool
    repair_reason: Optional[str]
    original_value: Optional[int]

    # Ground truth (if available)
    ground_truth: Optional[int] = None
    was_correct: Optional[bool] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "extraction": {
                "method": self.extraction_method,
                "extracted": self.extracted_value,
                "parsed": self.parsed_value,
                "final": self.final_answer
            },
            "decision": {
                "type": self.decision,
                "confidence": self.confidence,
                "flags": self.flags
            },
            "repair": {
                "applied": self.repair_applied,
                "justified": self.repair_justified,
                "reason": self.repair_reason,
                "original": self.original_value
            },
            "evaluation": {
                "ground_truth": self.ground_truth,
                "correct": self.was_correct
            }
        }
```

### 7.2 Metrics Aggregation

```python
class GuardianMetrics:
    """Track Guardian performance for continuous improvement"""

    def __init__(self):
        self.decisions = {"accept": 0, "flag": 0, "retry": 0, "escalate": 0}
        self.extraction_methods = {}
        self.repair_stats = {"applied": 0, "justified": 0, "unjustified": 0}
        self.correctness = {"correct": 0, "incorrect": 0, "unknown": 0}

    def record(self, log: GuardianDecisionLog):
        self.decisions[log.decision] += 1
        self.extraction_methods[log.extraction_method] = \
            self.extraction_methods.get(log.extraction_method, 0) + 1

        if log.repair_applied:
            self.repair_stats["applied"] += 1
            if log.repair_justified:
                self.repair_stats["justified"] += 1
            else:
                self.repair_stats["unjustified"] += 1

        if log.was_correct is True:
            self.correctness["correct"] += 1
        elif log.was_correct is False:
            self.correctness["incorrect"] += 1
        else:
            self.correctness["unknown"] += 1

    def summary(self) -> Dict:
        total = sum(self.decisions.values())
        correct = self.correctness["correct"]
        known = correct + self.correctness["incorrect"]

        return {
            "total_validations": total,
            "decision_breakdown": self.decisions,
            "accuracy": correct / known if known > 0 else None,
            "extraction_methods": self.extraction_methods,
            "repair_stats": self.repair_stats,
            "flag_rate": self.decisions["flag"] / total if total > 0 else 0,
            "retry_rate": self.decisions["retry"] / total if total > 0 else 0,
        }
```

---

## 8. Testing Suite

### 8.1 Unit Tests

```python
import pytest

class TestGuardianAIMO3:
    """Comprehensive test suite for unified Guardian"""

    @pytest.fixture
    def guardian(self):
        return GuardianAIMO3(problem_text="Find the value of x", competition_mode=True)

    # ========== Extraction Tests ==========

    def test_latex_boxed_extraction(self, guardian):
        result = guardian.validate(r"Therefore \boxed{42}")
        assert result.answer == 42
        assert result.decision == GuardianDecision.ACCEPT

    def test_answer_is_extraction(self, guardian):
        result = guardian.validate("The final answer is 123")
        assert result.answer == 123
        assert result.decision == GuardianDecision.ACCEPT

    def test_fallback_extraction(self, guardian):
        result = guardian.validate("After calculation:\n42")
        assert result.answer == 42

    def test_extraction_failure(self, guardian):
        result = guardian.validate("No numbers here at all")
        assert result.decision == GuardianDecision.RETRY
        assert "hard_fail:extraction" in result.flags

    # ========== Type Tests ==========

    def test_integer_accepted(self, guardian):
        result = guardian.validate(r"\boxed{42}")
        assert result.answer == 42
        assert result.decision == GuardianDecision.ACCEPT

    def test_float_coerced(self, guardian):
        result = guardian.validate(r"\boxed{42.0}")
        assert result.answer == 42
        assert any("type_coerced" in f for f in result.flags)

    # ========== Range Tests (CRITICAL: Should FLAG, not REJECT) ==========

    def test_out_of_range_flags_not_rejects(self, guardian):
        result = guardian.validate(r"\boxed{1234}")
        assert result.decision == GuardianDecision.FLAG
        assert result.answer == 1234  # NOT repaired without justification
        assert result.confidence < 1.0
        assert "out_of_range" in str(result.flags)

    def test_justified_repair_applied(self):
        guardian = GuardianAIMO3(
            problem_text="Find the remainder when x is divided by 1000",
            competition_mode=True
        )
        result = guardian.validate(r"\boxed{1234}")
        assert result.answer == 234  # Repaired
        assert "justified_repair" in str(result.flags)

    # ========== Common Value Tests (CRITICAL: No penalty) ==========

    def test_common_value_accepted(self, guardian):
        result = guardian.validate(r"\boxed{0}")
        assert result.answer == 0
        assert result.decision == GuardianDecision.ACCEPT
        assert result.confidence == 1.0  # NO penalty

    def test_common_value_42_accepted(self, guardian):
        result = guardian.validate(r"\boxed{42}")
        assert result.answer == 42
        assert result.decision == GuardianDecision.ACCEPT
        assert result.confidence == 1.0  # NO penalty

    # ========== Competition Mode Tests ==========

    def test_competition_mode_never_rejects(self, guardian):
        """In competition mode, ESCALATE returns best-effort, not rejection"""
        # Even with multiple failures, should not hard-reject
        results = []
        for response in ["abc", "def", "ghi"]:
            results.append(guardian.validate(response))

        # All should be RETRY, not REJECT
        assert all(r.decision == GuardianDecision.RETRY for r in results)


class TestJustifiedRepair:
    """Test repair justification system"""

    def test_no_justification_no_repair(self):
        repair = JustifiedRepair("Find the largest prime")
        answer, reason, justified = repair.attempt_repair(9973)
        assert answer == 9973  # NOT changed
        assert justified == False

    def test_remainder_justifies_repair(self):
        repair = JustifiedRepair("Find the remainder when divided by 1000")
        answer, reason, justified = repair.attempt_repair(9973)
        assert answer == 973
        assert justified == True

    def test_mod_justifies_repair(self):
        repair = JustifiedRepair("Compute x mod 1000")
        answer, reason, justified = repair.attempt_repair(12345)
        assert answer == 345
        assert justified == True

    def test_last_digits_justifies_repair(self):
        repair = JustifiedRepair("Find the last 3 digits of n!")
        answer, reason, justified = repair.attempt_repair(999999)
        assert answer == 999
        assert justified == True


class TestConsensusAdvisor:
    """Test advisory consensus system"""

    def test_consensus_is_advisory_only(self):
        guardian = GuardianAIMO3("Find x")
        advisor = ConsensusAdvisor(guardian)

        responses = [r"\boxed{42}", r"\boxed{42}", r"\boxed{99}"]
        result = advisor.advise(responses)

        assert result["advisory_only"] == True
        assert result["recommendation"] == 42

    def test_disagreement_does_not_reject(self):
        guardian = GuardianAIMO3("Find x")
        advisor = ConsensusAdvisor(guardian)

        # High disagreement
        responses = [r"\boxed{1}", r"\boxed{2}", r"\boxed{3}", r"\boxed{4}", r"\boxed{5}"]
        result = advisor.advise(responses)

        # Should still provide recommendation, not reject
        assert result["recommendation"] is not None
        assert result["advisory_only"] == True
```

### 8.2 Integration Tests

```python
def test_full_pipeline_with_prometheus():
    """Test Guardian + PROMETHEUS integration"""

    problem = "Find the remainder when 2^100 is divided by 1000"

    # Simulate multiple LLM responses
    responses = [
        r"Computing... 2^100 mod 1000 = \boxed{376}",
        r"The answer is \boxed{376}",
        r"Therefore 376",
        r"I believe it's \boxed{999}",  # Outlier
        r"\boxed{376}",
    ]

    guardian = PrometheusIntegratedGuardian(problem)
    best_answer, confidence = guardian.select_best(responses)

    assert best_answer == 376
    assert confidence > 0.5  # Majority agreement
```

---

## 9. Decision Matrix: Final Reference

| Scenario | Hard/Soft | Decision | Confidence | Notes |
|----------|-----------|----------|------------|-------|
| Extraction fails | HARD | RETRY | 0.0 | Retry with different prompt |
| Parse fails | HARD | RETRY | 0.0 | Retry with different prompt |
| Not integer | HARD | RETRY | 0.0 | After coercion attempt |
| Answer = 0 | SOFT | ACCEPT | 1.0 | Annotation only |
| Answer = 42 | SOFT | ACCEPT | 1.0 | Annotation only |
| Answer = 1234, no justification | SOFT | FLAG | 0.5 | Returns 1234, not repaired |
| Answer = 1234, "mod 1000" in problem | SOFT | ACCEPT | 0.9 | Repaired to 234 |
| Answer = -5, no justification | SOFT | FLAG | 0.5 | Returns -5, not repaired |
| High disagreement in consensus | N/A | ADVISORY | varies | Informative, not rejection |
| All responses fail extraction | HARD | ESCALATE | 0.0 | Best-effort in competition |

---

## 10. Deployment Checklist

### Pre-Competition

- [ ] Verify HARD rules are extraction/parse/type ONLY
- [ ] Verify SOFT rules (range, common values) are annotation-only
- [ ] Verify repairs require problem-text justification
- [ ] Verify consensus is advisory-only
- [ ] Run full test suite
- [ ] Test with edge cases: 0, 1, 42, 100, 999, 1000, -1

### Runtime Monitoring

- [ ] Track decision distribution (ACCEPT/FLAG/RETRY/ESCALATE)
- [ ] Monitor repair rates (justified vs unjustified)
- [ ] Log extraction method distribution
- [ ] Alert if RETRY rate > 30%
- [ ] Alert if FLAG rate > 20%

### Post-Competition

- [ ] Export all decision logs
- [ ] Analyze extraction failure patterns
- [ ] Review unjustified repairs (were they correct?)
- [ ] Update extraction patterns if needed
- [ ] Update justification patterns if needed

---

## 11. Summary: Key Invariants

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GUARDIAN-AIMO3 UNIFIED INVARIANTS                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. ONLY extraction/parse/type failures are HARD                            │
│     → Everything else is annotation-only                                    │
│                                                                              │
│  2. Range violations FLAG, never REJECT                                      │
│     → Out-of-range answers are returned with reduced confidence             │
│                                                                              │
│  3. Repairs require justification from problem text                          │
│     → Silent coercion can change correct answers                            │
│     → "mod 1000", "remainder", "last 3 digits" justify repair               │
│                                                                              │
│  4. Common values (0, 1, 42, 100) have NO confidence penalty                │
│     → They are common CORRECT answers in competition math                   │
│                                                                              │
│  5. Consensus is ADVISORY, never AUTHORITATIVE                              │
│     → Disagreement is informative, not failure-indicating                   │
│     → PROMETHEUS selects best answer among valid extractions                │
│                                                                              │
│  6. In competition mode, ESCALATE returns best-effort                        │
│     → No human available, so never truly reject                             │
│     → Return 0 with confidence 0 as last resort                             │
│                                                                              │
│  7. Guardian enforces INTERFACE (parseability)                              │
│     → PROMETHEUS/consensus enforces CORRECTNESS                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**This document is the single source of truth for Guardian-AIMO3.**

*"Guardian is skeptical-oracle, not correctness-oracle. In competitions, false rejects are catastrophic."*
