# Guardian-AIMO3: Competition-Safe Profile

**Version:** 1.0
**Status:** Production-Ready for Mathematical Competition
**Date:** December 2024

---

## Executive Summary

This document defines a **competition-safe** Guardian configuration that addresses the fragilities identified in the original spec. The key principle:

> **Guardian must be skeptical-oracle, not correctness-oracle.**
> In competitions, false rejects are worse than false accepts.

---

## Critical Modifications from Base Guardian

### ❌ DISABLED: Sanity-Driven Rejection

**Original (DANGEROUS):**
```python
MAGIC_NUMBERS = {0, 1, -1, 42, 69, 100, 1000}
if x in MAGIC_NUMBERS:
    penalty applied → can trigger REJECT
```

**Competition-Safe:**
```python
# DISABLED - Magic numbers are valid AIMO answers
# 0, 1, 42, 100 are COMMON correct answers in competition math
# Never penalize based on answer value alone

# If you want this data for logging:
def annotate_magic_number(answer: int) -> str:
    """Log-only annotation, zero effect on scoring"""
    MAGIC_NUMBERS = {0, 1, 42, 100}
    if answer in MAGIC_NUMBERS:
        return f"Note: {answer} is a common pattern value"
    return ""
```

### ❌ DISABLED: Aggressive Repair

**Original (DANGEROUS):**
```python
def repair(answer):
    if answer > 999:
        return answer % 1000  # SILENT COERCION
```

**Competition-Safe:**
```python
def competition_safe_repair(answer: int, problem_text: str) -> tuple[int | None, str]:
    """
    ONLY repair when problem explicitly justifies it.
    Otherwise, flag for RETRY, never silently coerce.
    """
    if 0 <= answer <= 999:
        return answer, "valid"

    # Check if problem explicitly asks for modular result
    mod_patterns = [
        r'remainder\s+when',
        r'mod\s*\d+',
        r'divided\s+by\s+\d+',
        r'last\s+\d+\s+digits?',
    ]

    problem_lower = problem_text.lower()
    for pattern in mod_patterns:
        if re.search(pattern, problem_lower):
            # Problem explicitly asks for mod - repair is justified
            repaired = answer % 1000
            return repaired, f"justified_repair: problem asks for mod/remainder"

    # NO SILENT REPAIR - flag for investigation
    return None, f"answer {answer} outside range, problem does not justify repair → RETRY"
```

### ❌ DISABLED: Kolmogorov/Solomonoff Primary Ranking

**Original (DANGEROUS):**
```python
score = mass * density * solomonoff_weight(code)  # Short solutions ranked higher
```

**Competition-Safe:**
```python
def competition_score(basin: Basin, codes: list[str]) -> float:
    """
    Length is TIE-BREAKER ONLY, never primary signal.

    On hard problems:
    - Correct solutions are often LONG
    - Wrong solutions are often SHORT and confident
    """
    # PRIMARY: Mass and Density (consensus strength)
    primary_score = basin.mass * (basin.density ** 0.1)

    # SECONDARY: Solomonoff is tie-breaker only
    # Only applied when comparing basins with equal primary scores
    # Weight is negligible (0.01) to ensure tie-breaking only
    tie_breaker = 0.01 * avg_solomonoff(codes)

    return primary_score + tie_breaker
```

### ✅ KEPT: Conservative Type Checking

```python
AIMO3_RULES = [
    # CRITICAL - these CAN trigger reject
    {
        "name": "integer_type",
        "check": lambda x: isinstance(x, int) or (isinstance(x, float) and x.is_integer()),
        "is_critical": True,
        "action": "reject_if_fail"
    },

    # NON-CRITICAL - annotation only
    {
        "name": "range_check",
        "check": lambda x: 0 <= x <= 999,
        "is_critical": False,  # CHANGED from True
        "action": "flag_for_review"  # Not auto-reject
    },
]
```

### ✅ KEPT: Multi-Pattern Extraction

The extraction patterns are robust and competition-safe:

```python
ANSWER_PATTERNS = [
    r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',  # LaTeX boxed - primary
    r'\$\\boxed\{([^{}]+)\}\$',                    # Inline math
    r'(?:final\s+)?answer\s*(?:is|:)\s*[*_]*(\d+)',
    r'(?:therefore|thus|so)\s*,?\s*(\d+)',
    r'=\s*[*_]*(\d+)[*_]*\s*$',
]
```

---

## Competition-Safe Guardian Implementation

```python
"""
Guardian-AIMO3: Competition-Safe Output Validation

Key principles:
1. Sanity → confidence annotation only, never rejection
2. Repair → only when problem explicitly justifies
3. Complexity → tie-breaker only, never primary
4. Rejection → only for parsing failures, never for answer values
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CompetitionDecision(Enum):
    """
    Competition-safe decision hierarchy:

    ACCEPT: Answer extracted and parseable
    FLAG: Answer unusual but valid (log, don't reject)
    RETRY: Could not extract (try different prompt)
    ESCALATE: Multiple failures, needs human review
    """
    ACCEPT = "accept"
    FLAG = "flag"          # NEW: unusual but valid
    RETRY = "retry"
    ESCALATE = "escalate"  # RENAMED from REJECT


@dataclass
class CompetitionValidation:
    """Result of competition-safe validation"""
    decision: CompetitionDecision
    answer: Optional[int] = None
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)  # Annotations, not penalties
    extraction_method: str = ""


class GuardianAIMO3:
    """
    Competition-safe Guardian for AIMO3.

    Key differences from base Guardian:
    - No sanity-based rejection
    - No unjustified repairs
    - Conservative rejection threshold
    """

    ANSWER_PATTERNS = [
        (r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', "latex_boxed"),
        (r'\$\\boxed\{([^{}]+)\}\$', "inline_boxed"),
        (r'(?:final\s+)?answer\s*(?:is|:)\s*[*_]*(\d+)[*_]*', "answer_is"),
        (r'(?:therefore|thus|so)\s*,?\s*(?:the\s+)?(?:answer\s+)?(?:is\s+)?[*_]*(\d+)', "conclusion"),
        (r'=\s*[*_]*(\d+)[*_]*\s*$', "trailing_equals"),
    ]

    def __init__(self, problem_text: str = ""):
        self.problem_text = problem_text

    def validate(self, response: str) -> CompetitionValidation:
        """
        Competition-safe validation.

        NEVER rejects based on:
        - Answer being a "magic number"
        - Answer being outside range (without problem justification)
        - Answer being too simple

        ONLY retries on:
        - Extraction failure
        - Parse failure
        """
        flags = []

        # Step 1: Extract
        extracted, method = self._extract(response)
        if extracted is None:
            return CompetitionValidation(
                decision=CompetitionDecision.RETRY,
                flags=["extraction_failed"],
                extraction_method="none"
            )

        # Step 2: Parse
        answer = self._parse(extracted)
        if answer is None:
            return CompetitionValidation(
                decision=CompetitionDecision.RETRY,
                flags=[f"parse_failed: {extracted}"],
                extraction_method=method
            )

        # Step 3: Range check (ANNOTATION ONLY)
        if not (0 <= answer <= 999):
            # Check if problem justifies repair
            justified, reason = self._check_repair_justification(answer)
            if justified:
                answer = answer % 1000
                flags.append(f"justified_repair: {reason}")
            else:
                flags.append(f"out_of_range: {answer}, no repair justification found")
                # FLAG, not REJECT - let higher-level logic decide
                return CompetitionValidation(
                    decision=CompetitionDecision.FLAG,
                    answer=answer,
                    confidence=0.5,  # Reduced confidence
                    flags=flags,
                    extraction_method=method
                )

        # Step 4: Annotation-only checks (NEVER affect acceptance)
        if answer in {0, 1, 42, 100}:
            flags.append(f"common_value: {answer}")  # Log only
            # confidence NOT reduced

        # Step 5: Accept
        return CompetitionValidation(
            decision=CompetitionDecision.ACCEPT,
            answer=answer,
            confidence=1.0,
            flags=flags,
            extraction_method=method
        )

    def _extract(self, response: str) -> tuple[Optional[str], str]:
        """Multi-pattern extraction"""
        response = response.strip()

        for pattern, method in self.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip(), method

        # Fallback: last standalone number
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            numbers = re.findall(r'\b(\d+)\b', line)
            if numbers:
                return numbers[-1], "fallback_last_number"

        return None, "none"

    def _parse(self, extracted: str) -> Optional[int]:
        """Parse to integer"""
        if not extracted:
            return None

        cleaned = extracted.replace(',', '').replace(' ', '').strip()

        try:
            return int(cleaned)
        except ValueError:
            pass

        try:
            f = float(cleaned)
            if f.is_integer():
                return int(f)
        except ValueError:
            pass

        return None

    def _check_repair_justification(self, answer: int) -> tuple[bool, str]:
        """
        Check if problem text justifies modular repair.
        ONLY repairs when explicitly justified.
        """
        if not self.problem_text:
            return False, "no_problem_text"

        lower = self.problem_text.lower()

        justifications = [
            (r'remainder\s+when', "remainder"),
            (r'modulo?\s*\d+', "modulo"),
            (r'divided\s+by\s+(\d+)', "division_remainder"),
            (r'last\s+(\d+)\s+digits?', "last_digits"),
            (r'\(mod\s*\d+\)', "explicit_mod"),
        ]

        for pattern, reason in justifications:
            if re.search(pattern, lower):
                return True, reason

        return False, "no_justification"


class ConsensusAdvisor:
    """
    ADVISORY consensus, not authoritative.

    Used for:
    - Selection (which answer to prefer)
    - Confidence estimation

    NOT used for:
    - Acceptance gating (Guardian handles that)
    - Rejection (never rejects based on disagreement)
    """

    def __init__(self, guardian: GuardianAIMO3):
        self.guardian = guardian

    def advise(self, responses: list[str]) -> dict:
        """
        Provide advisory consensus.
        Returns recommendation, not decision.
        """
        # Validate all responses
        validations = [self.guardian.validate(r) for r in responses]

        # Collect valid answers
        valid_answers = [
            v.answer for v in validations
            if v.decision in (CompetitionDecision.ACCEPT, CompetitionDecision.FLAG)
            and v.answer is not None
        ]

        if not valid_answers:
            return {
                "recommendation": None,
                "confidence": 0.0,
                "reason": "no_valid_answers",
                "advisory_only": True,
            }

        # Count votes
        from collections import Counter
        votes = Counter(valid_answers)
        top_answer, top_count = votes.most_common(1)[0]

        agreement = top_count / len(valid_answers)

        return {
            "recommendation": top_answer,
            "confidence": agreement,
            "vote_distribution": dict(votes),
            "total_responses": len(responses),
            "valid_responses": len(valid_answers),
            "advisory_only": True,  # ALWAYS advisory
            "reason": "majority_vote" if agreement > 0.5 else "plurality_vote"
        }


# =============================================================================
# VERIFICATION-SCORE CLUSTERING (Replaces ConsensusGuardian)
# =============================================================================

class VerificationScoreClusterer:
    """
    Your existing approach, formalized.

    This is SAFER than ConsensusGuardian because:
    1. Disagreement is informative, not rejection-triggering
    2. No sanity penalties
    3. Verification-gated, not consensus-gated
    """

    def __init__(self, guardian: GuardianAIMO3):
        self.guardian = guardian

    def cluster_and_select(
        self,
        responses: list[str],
        entropy_scores: list[float] = None
    ) -> tuple[int, float]:
        """
        Cluster answers by value, score by verification quality.

        Returns (best_answer, confidence)
        """
        validations = [self.guardian.validate(r) for r in responses]

        # Group by answer
        clusters: dict[int, list[CompetitionValidation]] = {}
        for v in validations:
            if v.decision in (CompetitionDecision.ACCEPT, CompetitionDecision.FLAG):
                if v.answer is not None:
                    clusters.setdefault(v.answer, []).append(v)

        if not clusters:
            return 0, 0.0

        # Score each cluster
        # Score = count × average_confidence
        best_answer = 0
        best_score = -1

        for answer, cluster_validations in clusters.items():
            count = len(cluster_validations)
            avg_conf = sum(v.confidence for v in cluster_validations) / count
            score = count * avg_conf

            if score > best_score:
                best_score = score
                best_answer = answer

        # Overall confidence = best cluster proportion
        total_valid = sum(len(c) for c in clusters.values())
        confidence = len(clusters.get(best_answer, [])) / total_valid if total_valid > 0 else 0

        return best_answer, confidence
```

---

## Decision Matrix: Competition-Safe vs Original

| Scenario | Original Guardian | Competition-Safe |
|----------|-------------------|------------------|
| Answer = 0 | Penalty, possible REJECT | ACCEPT (0 is valid) |
| Answer = 42 | Penalty, possible REJECT | ACCEPT (42 is valid) |
| Answer = 1234 | Auto-repair to 234 | FLAG + check problem text |
| Answer = -5 | Auto-repair to 5 | FLAG + check problem text |
| No consensus | RETRY/REJECT | Select highest-confidence |
| High disagreement | Indicates failure | **Informative** on hard problems |
| Short solution | Ranked higher | No preference |
| Long solution | Ranked lower | No preference |

---

## Integration with Existing Pipeline

```python
# Replace this:
guardian = OutputGuardian()  # Original - dangerous

# With this:
guardian = GuardianAIMO3(problem_text=current_problem)  # Competition-safe

# And replace consensus:
consensus = ConsensusGuardian(guardian)  # Original - can over-reject

# With:
clusterer = VerificationScoreClusterer(guardian)  # Your approach formalized
```

---

## Final Checklist

Before competition:
- [ ] Magic number penalties DISABLED
- [ ] Unjustified repairs DISABLED
- [ ] Solomonoff ranking DEMOTED to tie-breaker only
- [ ] Consensus is ADVISORY only
- [ ] REJECT renamed to ESCALATE (human reviews, never auto-rejects)
- [ ] Range violations FLAG, not REJECT

This configuration ensures Guardian **helps** without **hurting**.

---

*"Guardian-AIMO3: Conservative on rejection, aggressive on extraction, skeptical on everything."*
