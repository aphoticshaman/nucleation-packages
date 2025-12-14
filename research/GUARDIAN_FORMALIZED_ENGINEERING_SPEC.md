# Guardian System: Formalized Engineering Specification

**Version:** 2.0
**Classification:** Advanced Implementation Guide
**Target Audience:** LLM-Assisted Implementation (AIMO3/gpt-oss-120b)
**Author:** LatticeForge Engineering (Extracted from Production Codebase)

---

## Abstract

Guardian is a multi-layer output validation, input sanitization, and self-improving governance system designed to wrap any LLM inference pipeline. This document formalizes the mathematical foundations, architectural patterns, and implementation specifics necessary to construct a Guardian system for mathematical reasoning domains such as AIMO3 competition submissions.

The core insight: **LLMs are probabilistic generators with known failure modes. Guardian provides deterministic validation + adaptive rule evolution to catch and repair failures before they propagate.**

---

## 1. Mathematical Foundations

### 1.1 The Validation Function Space

Let $\mathcal{L}$ be an LLM with input space $\mathcal{X}$ and output space $\mathcal{Y}$.

**Definition 1.1 (Guardian Operator):**
$$G: \mathcal{Y} \rightarrow \mathcal{Y}' \cup \{\perp\}$$

Where:
- $G(y) = y'$ if $y$ passes validation (possibly with repairs)
- $G(y) = \perp$ if $y$ is unfixable (rejection)

**Definition 1.2 (Validation Predicate):**
A validation rule $r \in \mathcal{R}$ is a function:
$$r: \mathcal{Y} \rightarrow \{0, 1\} \times [0,1] \times \mathcal{E}$$

Returning tuple $(valid, confidence, explanation)$.

**Definition 1.3 (Composite Guardian):**
For rule set $\mathcal{R} = \{r_1, ..., r_n\}$ with weights $w_i$:
$$G_\mathcal{R}(y) = \begin{cases}
\text{repair}(y) & \text{if } \sum_{i} w_i \cdot r_i(y)_0 > \tau \\
\perp & \text{otherwise}
\end{cases}$$

Where $\tau$ is the acceptance threshold (typically 0.5-0.7 for mathematical domains).

### 1.2 Error Taxonomy for Mathematical LLMs

Based on empirical observation of 72B+ parameter models on AIMO-style problems:

| Error Class | Formal Definition | Detection Method | Repair Strategy |
|-------------|------------------|------------------|-----------------|
| **Syntactic** | $y \notin \mathcal{G}_{valid}$ (not in grammar) | Parser rejection | Regex extraction |
| **Semantic** | $\text{type}(y) \neq \text{expected\_type}$ | Type checking | Coercion/default |
| **Hallucination** | $\text{ref}(y) \cap \text{grounded\_facts} = \emptyset$ | Fact checking | Rejection + retry |
| **Inconsistency** | $\exists i,j: y_i \land y_j = \bot$ | Contradiction detection | Majority vote |
| **Truncation** | $|y| < |y_{expected}|$ | Length check | Continuation |
| **Format Drift** | $\text{format}(y) \neq \text{spec}$ | Schema validation | JSON repair |

### 1.3 Kolmogorov Complexity for Solution Quality

**Insight:** Among multiple valid solutions, prefer those with lower description length.

$$\text{score}(y) = -K(y | problem) + \alpha \cdot \text{correctness}(y)$$

Where $K(y|x)$ is the conditional Kolmogorov complexity. In practice, approximate via:
$$\hat{K}(y) = \text{len}(\text{gzip}(y))$$

This naturally penalizes:
- Verbose, repetitive solutions
- Hallucinated intermediate steps
- Unnecessary complexity

---

## 2. Architecture Specification

### 2.1 Layer Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER REQUEST                                   │
│                     (Mathematical Problem P)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    GUARDIAN LAYER 1: INPUT SECURITY                      │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │ Injection Det. │  │ Format Validate │  │ Rate Limit (per-user)    │  │
│  │ (regex + ML)   │  │ (JSON schema)   │  │ (token bucket/sliding)   │  │
│  └────────────────┘  └─────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LLM INFERENCE (gpt-oss-120b)                     │
│                                                                          │
│  System Prompt S + User Message P → Response Y                          │
│  Temperature τ, Top-p p, Max tokens T                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    GUARDIAN LAYER 2: OUTPUT VALIDATION                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    VALIDATION PIPELINE                            │   │
│  │                                                                   │   │
│  │  Step 1: EXTRACT      Parse answer from freeform text             │   │
│  │          ↓            Regex: /\\boxed\{([^}]+)\}/                 │   │
│  │          ↓            Or: Final Answer: X                         │   │
│  │                                                                   │   │
│  │  Step 2: TYPECHECK    Verify answer is expected type              │   │
│  │          ↓            int, float, fraction, expression            │   │
│  │          ↓            Parse with sympy/mathjs                     │   │
│  │                                                                   │   │
│  │  Step 3: BOUNDS       Check answer in valid range                 │   │
│  │          ↓            0 ≤ x ≤ 999 for AIMO                        │   │
│  │          ↓            Domain-specific constraints                 │   │
│  │                                                                   │   │
│  │  Step 4: CONSISTENCY  If multiple solutions, check agreement      │   │
│  │          ↓            Majority vote / weighted by confidence      │   │
│  │          ↓            Flag outliers for rejection                 │   │
│  │                                                                   │   │
│  │  Step 5: SANITY       Quick heuristic checks                      │   │
│  │          ↓            "Does this look like a reasonable answer?"  │   │
│  │          ↓            Known impossible values → reject            │   │
│  │                                                                   │   │
│  │  Step 6: REPAIR       If fixable, attempt correction              │   │
│  │                       - Integer coercion: 123.0 → 123             │   │
│  │                       - Modular reduction: 1999 % 1000 → 999      │   │
│  │                       - Sign correction: -42 → 42 (if unsigned)   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    DECISION MATRIX                                │   │
│  │                                                                   │   │
│  │  all_pass ────────────────────────→ ACCEPT (return validated Y')  │   │
│  │  some_fail + repairable ──────────→ REPAIR (return repaired Y'')  │   │
│  │  critical_fail ───────────────────→ RETRY (re-invoke LLM)         │   │
│  │  max_retries_exceeded ────────────→ REJECT (return ⊥ / fallback)  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    GUARDIAN LAYER 3: GOVERNANCE                          │
│                                                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌───────────────────┐  │
│  │ Decision Logging   │  │ Metrics Aggregation│  │ Rule Evolution    │  │
│  │ (for training)     │  │ (accuracy, latency)│  │ (self-improvement)│  │
│  └────────────────────┘  └────────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            VALIDATED OUTPUT Y'
```

### 2.2 State Machine

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┴───┐
│  INPUT   │───▶│ VALIDATE │───▶│  DECIDE  │───▶│   EXECUTE    │
│ RECEIVED │    │  INPUT   │    │ ROUTING  │    │  INFERENCE   │
└──────────┘    └──────────┘    └──────────┘    └──────────────┘
                    │                                      │
                    │ REJECT                               ▼
                    ▼                           ┌──────────────────┐
              ┌──────────┐                      │ VALIDATE OUTPUT  │
              │  BLOCK   │                      └──────────────────┘
              │ RESPONSE │                              │
              └──────────┘                    ┌─────────┼─────────┐
                                              ▼         ▼         ▼
                                         ┌────────┐┌────────┐┌────────┐
                                         │ ACCEPT ││ REPAIR ││ RETRY  │
                                         └────────┘└────────┘└────────┘
                                              │         │         │
                                              │         │    ┌────┘
                                              ▼         ▼    ▼
                                         ┌─────────────────────────┐
                                         │    LOG + RETURN         │
                                         └─────────────────────────┘
```

---

## 3. Implementation: Python Reference

### 3.1 Core Guardian Class

```python
"""
Guardian System for Mathematical LLM Output Validation

Designed for AIMO3 competition with gpt-oss-120b or similar.
"""

import re
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
import logging

logger = logging.getLogger("guardian")


class ValidationResult(Enum):
    ACCEPT = "accept"
    REPAIR = "repair"
    RETRY = "retry"
    REJECT = "reject"


@dataclass
class ValidationOutput:
    """Result of validation pipeline"""
    result: ValidationResult
    answer: Optional[Any] = None
    original: Optional[str] = None
    repaired: bool = False
    confidence: float = 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardianRule:
    """A single validation rule"""
    name: str
    domain: str  # "math", "format", "sanity", etc.
    check: Callable[[Any], Tuple[bool, float, str]]
    weight: float = 1.0
    is_critical: bool = False  # If True, failure = immediate reject


class OutputGuardian:
    """
    Output validation layer for mathematical LLM responses.

    Implements a multi-stage pipeline:
    1. Extract answer from freeform text
    2. Parse and typecheck
    3. Apply validation rules
    4. Attempt repairs if needed
    5. Return validated output or rejection
    """

    # Answer extraction patterns (order matters - try most specific first)
    ANSWER_PATTERNS = [
        r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',  # LaTeX boxed
        r'\$\\boxed\{([^{}]+)\}\$',                    # Inline math boxed
        r'(?:final\s+)?answer\s*(?:is|:)\s*[*_]*(\d+)[*_]*',  # "Answer: 42"
        r'(?:therefore|thus|so)\s*,?\s*(?:the\s+)?(?:answer\s+is\s+)?[*_]*(\d+)[*_]*',
        r'=\s*[*_]*(\d+)[*_]*\s*$',                    # Trailing = X
        r'^[*_]*(\d+)[*_]*$',                          # Just the number
    ]

    def __init__(
        self,
        answer_range: Tuple[int, int] = (0, 999),
        max_retries: int = 3,
        repair_enabled: bool = True,
        strict_mode: bool = False
    ):
        self.answer_range = answer_range
        self.max_retries = max_retries
        self.repair_enabled = repair_enabled
        self.strict_mode = strict_mode
        self.rules: List[GuardianRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Initialize default validation rules for AIMO-style problems"""

        # Rule 1: Type check - must be integer
        self.rules.append(GuardianRule(
            name="integer_type",
            domain="math",
            check=lambda x: (
                isinstance(x, int) or (isinstance(x, float) and x.is_integer()),
                1.0 if isinstance(x, int) else 0.8,
                "Answer must be an integer"
            ),
            weight=2.0,
            is_critical=True
        ))

        # Rule 2: Range check
        self.rules.append(GuardianRule(
            name="range_check",
            domain="math",
            check=lambda x: (
                self.answer_range[0] <= x <= self.answer_range[1],
                1.0 if self.answer_range[0] <= x <= self.answer_range[1] else 0.0,
                f"Answer must be in range [{self.answer_range[0]}, {self.answer_range[1]}]"
            ),
            weight=2.0,
            is_critical=True
        ))

        # Rule 3: Non-negative (for most AIMO problems)
        self.rules.append(GuardianRule(
            name="non_negative",
            domain="math",
            check=lambda x: (
                x >= 0,
                1.0 if x >= 0 else 0.0,
                "Answer must be non-negative"
            ),
            weight=1.5
        ))

        # Rule 4: Sanity check - not a "magic number"
        MAGIC_NUMBERS = {0, 1, -1, 42, 69, 100, 1000}
        self.rules.append(GuardianRule(
            name="magic_number_warning",
            domain="sanity",
            check=lambda x: (
                x not in MAGIC_NUMBERS,
                0.7 if x in MAGIC_NUMBERS else 1.0,
                "Answer is a common 'magic number' - verify carefully"
            ),
            weight=0.3  # Low weight - warning only
        ))

    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract numerical answer from LLM response.

        Tries multiple patterns in order of specificity.
        Returns the extracted string, or None if no match.
        """
        # Clean response
        response = response.strip()

        for pattern in self.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted = match.group(1).strip()
                logger.debug(f"Pattern matched: {pattern[:30]}... → {extracted}")
                return extracted

        # Fallback: try to find any standalone number at end
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):  # Check last 5 lines
            numbers = re.findall(r'\b(\d+)\b', line)
            if numbers:
                logger.debug(f"Fallback extraction: {numbers[-1]}")
                return numbers[-1]

        return None

    def parse_answer(self, extracted: str) -> Optional[int]:
        """
        Parse extracted string to integer.

        Handles:
        - Plain integers: "42"
        - Floats: "42.0" → 42
        - Comma formatting: "1,234" → 1234
        - Negative: "-42" → -42
        """
        if not extracted:
            return None

        # Clean
        cleaned = extracted.replace(',', '').replace(' ', '').strip()

        # Try integer
        try:
            return int(cleaned)
        except ValueError:
            pass

        # Try float → int
        try:
            f = float(cleaned)
            if f.is_integer():
                return int(f)
            # Round if close to integer
            if abs(f - round(f)) < 0.001:
                return int(round(f))
        except ValueError:
            pass

        # Try eval for simple expressions (dangerous in general, but controlled here)
        if re.match(r'^[\d\s\+\-\*\/\%\(\)]+$', cleaned):
            try:
                result = eval(cleaned)
                if isinstance(result, (int, float)):
                    return int(result) if isinstance(result, int) or result.is_integer() else None
            except:
                pass

        return None

    def validate(self, response: str, context: Optional[Dict] = None) -> ValidationOutput:
        """
        Main validation entry point.

        Args:
            response: Raw LLM response text
            context: Optional context (problem info, previous answers, etc.)

        Returns:
            ValidationOutput with result and validated answer
        """
        errors = []
        warnings = []
        metadata = {"context": context or {}}

        # Step 1: Extract
        extracted = self.extract_answer(response)
        if extracted is None:
            return ValidationOutput(
                result=ValidationResult.RETRY,
                original=response,
                errors=["Could not extract answer from response"],
                metadata=metadata
            )

        metadata["extracted"] = extracted

        # Step 2: Parse
        answer = self.parse_answer(extracted)
        if answer is None:
            return ValidationOutput(
                result=ValidationResult.RETRY,
                original=response,
                errors=[f"Could not parse extracted answer: {extracted}"],
                metadata=metadata
            )

        metadata["parsed"] = answer

        # Step 3: Apply rules
        total_score = 0.0
        total_weight = 0.0
        critical_failed = False

        for rule in self.rules:
            passed, confidence, explanation = rule.check(answer)

            if not passed:
                if rule.is_critical:
                    critical_failed = True
                    errors.append(f"[CRITICAL] {rule.name}: {explanation}")
                else:
                    warnings.append(f"{rule.name}: {explanation}")

            total_score += rule.weight * (confidence if passed else 0)
            total_weight += rule.weight

        normalized_score = total_score / total_weight if total_weight > 0 else 0

        # Step 4: Decide
        if critical_failed:
            # Try repair if enabled
            if self.repair_enabled:
                repaired = self._attempt_repair(answer)
                if repaired is not None and repaired != answer:
                    # Re-validate repaired answer
                    return self._validate_repaired(repaired, response, metadata)

            return ValidationOutput(
                result=ValidationResult.REJECT,
                answer=answer,
                original=response,
                confidence=normalized_score,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )

        if normalized_score < 0.5:
            return ValidationOutput(
                result=ValidationResult.RETRY,
                answer=answer,
                original=response,
                confidence=normalized_score,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )

        return ValidationOutput(
            result=ValidationResult.ACCEPT,
            answer=answer,
            original=response,
            confidence=normalized_score,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

    def _attempt_repair(self, answer: int) -> Optional[int]:
        """
        Attempt to repair an invalid answer.

        Strategies:
        1. Modular reduction (if > max)
        2. Absolute value (if negative)
        3. Last N digits (if way out of range)
        """
        min_val, max_val = self.answer_range

        # Strategy 1: Modular reduction
        if answer > max_val:
            mod_val = max_val + 1  # For 0-999, mod 1000
            repaired = answer % mod_val
            if min_val <= repaired <= max_val:
                logger.info(f"Repair: {answer} % {mod_val} = {repaired}")
                return repaired

        # Strategy 2: Absolute value
        if answer < min_val and abs(answer) <= max_val:
            repaired = abs(answer)
            logger.info(f"Repair: |{answer}| = {repaired}")
            return repaired

        # Strategy 3: Last N digits
        if answer > max_val:
            digits = len(str(max_val))
            repaired = int(str(answer)[-digits:])
            if min_val <= repaired <= max_val:
                logger.info(f"Repair: last {digits} digits of {answer} = {repaired}")
                return repaired

        return None

    def _validate_repaired(
        self,
        repaired: int,
        original_response: str,
        metadata: Dict
    ) -> ValidationOutput:
        """Re-validate a repaired answer"""
        errors = []
        warnings = [f"Answer repaired from {metadata['parsed']} to {repaired}"]

        # Quick validation of repaired answer
        for rule in self.rules:
            if rule.is_critical:
                passed, _, explanation = rule.check(repaired)
                if not passed:
                    errors.append(f"Repair failed: {explanation}")
                    return ValidationOutput(
                        result=ValidationResult.REJECT,
                        answer=metadata['parsed'],
                        original=original_response,
                        repaired=False,
                        errors=errors,
                        warnings=warnings,
                        metadata=metadata
                    )

        return ValidationOutput(
            result=ValidationResult.REPAIR,
            answer=repaired,
            original=original_response,
            repaired=True,
            confidence=0.8,  # Slightly lower confidence for repaired answers
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )


class InputGuardian:
    """
    Input validation layer.

    Protects against:
    - Prompt injection
    - Malformed input
    - Rate limiting
    """

    BLOCKED_PATTERNS = [
        r'ignore\s+(previous|above|all)\s+instructions',
        r'disregard\s+(previous|above|all)',
        r'forget\s+(everything|all|previous)',
        r'you\s+are\s+now',
        r'pretend\s+(to\s+be|you\s+are)',
        r'system\s*:\s*',
        r'\[INST\]',
        r'<\|im_start\|>',
    ]

    def __init__(self, max_input_length: int = 10000):
        self.max_input_length = max_input_length
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.BLOCKED_PATTERNS]

    def validate(self, user_input: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user input.

        Returns:
            (allowed, error_message)
        """
        if len(user_input) > self.max_input_length:
            return False, f"Input exceeds maximum length ({self.max_input_length})"

        for pattern in self._patterns:
            if pattern.search(user_input):
                return False, "Input contains blocked pattern"

        return True, None


class ConsensusGuardian:
    """
    Multi-solution consensus for improved reliability.

    Generates multiple solutions and uses voting/agreement
    to select the most likely correct answer.
    """

    def __init__(
        self,
        output_guardian: OutputGuardian,
        min_agreement: float = 0.5,  # Minimum fraction that must agree
        weight_by_confidence: bool = True
    ):
        self.output_guardian = output_guardian
        self.min_agreement = min_agreement
        self.weight_by_confidence = weight_by_confidence

    def aggregate(
        self,
        responses: List[str],
        strategy: str = "majority"
    ) -> ValidationOutput:
        """
        Aggregate multiple LLM responses into a single answer.

        Strategies:
        - "majority": Simple majority vote
        - "weighted": Weight by confidence scores
        - "unanimous": Require all to agree
        """
        if not responses:
            return ValidationOutput(
                result=ValidationResult.REJECT,
                errors=["No responses to aggregate"]
            )

        # Validate each response
        validations = [self.output_guardian.validate(r) for r in responses]

        # Filter to successful extractions
        valid_results = [
            v for v in validations
            if v.result in (ValidationResult.ACCEPT, ValidationResult.REPAIR) and v.answer is not None
        ]

        if not valid_results:
            # All failed - return most informative error
            most_info = max(validations, key=lambda v: len(v.metadata))
            return ValidationOutput(
                result=ValidationResult.REJECT,
                original=responses[0] if responses else None,
                errors=["All solutions failed validation"] + most_info.errors,
                metadata={"all_validations": [v.__dict__ for v in validations]}
            )

        # Count votes
        answer_votes: Dict[int, float] = {}
        answer_details: Dict[int, List[ValidationOutput]] = {}

        for v in valid_results:
            ans = v.answer
            weight = v.confidence if self.weight_by_confidence else 1.0
            answer_votes[ans] = answer_votes.get(ans, 0) + weight
            answer_details.setdefault(ans, []).append(v)

        # Select winner based on strategy
        if strategy == "unanimous":
            if len(answer_votes) == 1:
                winner = list(answer_votes.keys())[0]
            else:
                return ValidationOutput(
                    result=ValidationResult.RETRY,
                    errors=[f"No unanimous agreement. Answers: {list(answer_votes.keys())}"],
                    metadata={"vote_distribution": answer_votes}
                )
        elif strategy == "weighted":
            winner = max(answer_votes.keys(), key=lambda k: answer_votes[k])
        else:  # majority
            winner = max(answer_votes.keys(), key=lambda k: answer_votes[k])

        # Check agreement threshold
        total_weight = sum(answer_votes.values())
        agreement = answer_votes[winner] / total_weight

        if agreement < self.min_agreement:
            return ValidationOutput(
                result=ValidationResult.RETRY,
                answer=winner,
                confidence=agreement,
                warnings=[f"Low agreement ({agreement:.2%}) - consider retry"],
                metadata={"vote_distribution": answer_votes}
            )

        # Return consensus answer
        best_validation = max(answer_details[winner], key=lambda v: v.confidence)

        return ValidationOutput(
            result=ValidationResult.ACCEPT,
            answer=winner,
            original=best_validation.original,
            repaired=any(v.repaired for v in answer_details[winner]),
            confidence=agreement,
            warnings=[f"Consensus from {len(valid_results)}/{len(responses)} solutions"],
            metadata={
                "vote_distribution": answer_votes,
                "agreement": agreement,
                "consensus_size": len(answer_details[winner])
            }
        )
```

### 3.2 Integration with LLM Inference

```python
"""
Guardian-wrapped inference pipeline for AIMO3.
"""

from typing import Optional, Callable
import time


class GuardedInference:
    """
    Wrapper that integrates Guardian with any LLM inference function.
    """

    def __init__(
        self,
        inference_fn: Callable[[str, str], str],  # (system_prompt, user_msg) -> response
        output_guardian: Optional[OutputGuardian] = None,
        input_guardian: Optional[InputGuardian] = None,
        max_retries: int = 3,
        retry_temperature_schedule: List[float] = [0.0, 0.3, 0.7],
        enable_consensus: bool = False,
        consensus_samples: int = 3
    ):
        self.inference_fn = inference_fn
        self.output_guardian = output_guardian or OutputGuardian()
        self.input_guardian = input_guardian or InputGuardian()
        self.max_retries = max_retries
        self.retry_temps = retry_temperature_schedule
        self.enable_consensus = enable_consensus
        self.consensus_samples = consensus_samples

        if enable_consensus:
            self.consensus = ConsensusGuardian(self.output_guardian)

    def solve(
        self,
        problem: str,
        system_prompt: Optional[str] = None
    ) -> ValidationOutput:
        """
        Solve a mathematical problem with Guardian validation.

        Args:
            problem: The problem statement
            system_prompt: Optional custom system prompt

        Returns:
            ValidationOutput with validated answer
        """
        # Input validation
        allowed, error = self.input_guardian.validate(problem)
        if not allowed:
            return ValidationOutput(
                result=ValidationResult.REJECT,
                errors=[f"Input validation failed: {error}"]
            )

        # Default system prompt for math
        if system_prompt is None:
            system_prompt = """You are an expert mathematician solving competition problems.

Think step by step. Show your work clearly.
Express your final answer as: \\boxed{N} where N is an integer.

IMPORTANT:
- Final answer MUST be a non-negative integer between 0 and 999
- Double-check your arithmetic
- If unsure, verify by substitution"""

        # Consensus mode: generate multiple solutions
        if self.enable_consensus:
            responses = []
            for _ in range(self.consensus_samples):
                try:
                    response = self.inference_fn(system_prompt, problem)
                    responses.append(response)
                except Exception as e:
                    responses.append(f"ERROR: {e}")

            return self.consensus.aggregate(responses)

        # Single solution mode with retries
        for attempt in range(self.max_retries):
            temp = self.retry_temps[min(attempt, len(self.retry_temps) - 1)]

            try:
                # Invoke LLM
                response = self.inference_fn(system_prompt, problem)

                # Validate
                validation = self.output_guardian.validate(response)

                if validation.result == ValidationResult.ACCEPT:
                    return validation
                elif validation.result == ValidationResult.REPAIR:
                    return validation
                elif validation.result == ValidationResult.REJECT:
                    # Don't retry rejected (unfixable)
                    return validation
                # RETRY → continue loop

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return ValidationOutput(
                        result=ValidationResult.REJECT,
                        errors=[f"Inference failed after {self.max_retries} attempts: {e}"]
                    )

        return ValidationOutput(
            result=ValidationResult.REJECT,
            errors=[f"Max retries ({self.max_retries}) exceeded"]
        )
```

---

## 4. Self-Improving Governance

### 4.1 Decision Logging Schema

```python
@dataclass
class GuardianDecisionLog:
    """Log entry for training/analysis"""
    id: str
    timestamp: float

    # Input
    problem_hash: str  # SHA256 of problem
    response_hash: str  # SHA256 of response

    # Extraction
    extracted_answer: Optional[str]
    parsed_answer: Optional[int]

    # Validation
    rules_triggered: List[str]
    validation_result: str
    confidence: float

    # Repair
    was_repaired: bool
    repair_details: Optional[str]

    # Ground truth (if available)
    ground_truth: Optional[int] = None
    was_correct: Optional[bool] = None
```

### 4.2 Metrics Collection

```python
class GuardianMetrics:
    """Track Guardian performance for self-improvement"""

    def __init__(self):
        self.total_validations = 0
        self.accepts = 0
        self.repairs = 0
        self.retries = 0
        self.rejects = 0

        self.correct_accepts = 0
        self.incorrect_accepts = 0
        self.correct_repairs = 0
        self.incorrect_repairs = 0

        self.rule_triggers: Dict[str, int] = {}
        self.repair_strategies: Dict[str, int] = {}

    def record(self, log: GuardianDecisionLog):
        """Record a decision for metrics"""
        self.total_validations += 1

        if log.validation_result == "accept":
            self.accepts += 1
            if log.was_correct is True:
                self.correct_accepts += 1
            elif log.was_correct is False:
                self.incorrect_accepts += 1
        elif log.validation_result == "repair":
            self.repairs += 1
            if log.was_correct is True:
                self.correct_repairs += 1
            elif log.was_correct is False:
                self.incorrect_repairs += 1
        elif log.validation_result == "retry":
            self.retries += 1
        else:
            self.rejects += 1

        for rule in log.rules_triggered:
            self.rule_triggers[rule] = self.rule_triggers.get(rule, 0) + 1

    def accuracy(self) -> float:
        """Overall accuracy (correct / total with ground truth)"""
        total_known = (
            self.correct_accepts + self.incorrect_accepts +
            self.correct_repairs + self.incorrect_repairs
        )
        if total_known == 0:
            return 0.0
        return (self.correct_accepts + self.correct_repairs) / total_known

    def precision(self) -> float:
        """Precision: correct accepts / total accepts"""
        if self.accepts == 0:
            return 0.0
        return self.correct_accepts / self.accepts

    def repair_success_rate(self) -> float:
        """How often repairs lead to correct answers"""
        total_repairs = self.correct_repairs + self.incorrect_repairs
        if total_repairs == 0:
            return 0.0
        return self.correct_repairs / total_repairs

    def summary(self) -> Dict[str, Any]:
        """Summary statistics"""
        return {
            "total": self.total_validations,
            "breakdown": {
                "accept": self.accepts,
                "repair": self.repairs,
                "retry": self.retries,
                "reject": self.rejects
            },
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "repair_success": self.repair_success_rate(),
            "top_rules": sorted(
                self.rule_triggers.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
```

### 4.3 Rule Evolution

```python
class RuleEvolver:
    """
    Automatically propose rule changes based on observed patterns.

    Human-in-the-loop: proposals require approval before activation.
    """

    def __init__(self, metrics: GuardianMetrics):
        self.metrics = metrics
        self.proposals: List[Dict] = []

    def analyze_and_propose(self) -> List[Dict]:
        """
        Analyze metrics and propose rule adjustments.

        Patterns detected:
        1. High false positive rate → relax threshold
        2. High false negative rate → tighten threshold
        3. Unused rules → mark for removal
        4. Frequent repair patterns → add as explicit rule
        """
        proposals = []

        # Example: if repair success rate is high, make repairs more aggressive
        if self.metrics.repair_success_rate() > 0.9 and self.metrics.repairs > 20:
            proposals.append({
                "type": "parameter_adjustment",
                "target": "repair_enabled",
                "current": True,
                "proposed": True,
                "rationale": f"High repair success rate ({self.metrics.repair_success_rate():.1%})",
                "confidence": 0.85
            })

        # Example: if a rule never triggers, consider removing
        total = self.metrics.total_validations
        if total > 100:
            for rule, count in self.metrics.rule_triggers.items():
                if count / total < 0.01:  # < 1% trigger rate
                    proposals.append({
                        "type": "rule_removal",
                        "target": rule,
                        "rationale": f"Rule rarely triggers ({count}/{total} = {count/total:.1%})",
                        "confidence": 0.6
                    })

        self.proposals.extend(proposals)
        return proposals
```

---

## 5. AIMO3-Specific Configuration

### 5.1 Competition Constraints

```python
# AIMO3 specific configuration
AIMO3_CONFIG = {
    # Answer constraints
    "answer_range": (0, 999),
    "answer_type": "integer",

    # Validation rules
    "rules": [
        {
            "name": "aimo_range",
            "check": lambda x: 0 <= x <= 999,
            "is_critical": True,
            "repair_strategy": "modular"
        },
        {
            "name": "aimo_integer",
            "check": lambda x: isinstance(x, int) or x == int(x),
            "is_critical": True,
            "repair_strategy": "round"
        }
    ],

    # Consensus settings
    "consensus": {
        "enabled": True,
        "samples": 5,
        "min_agreement": 0.6,
        "temperature_schedule": [0.0, 0.2, 0.4, 0.6, 0.8]
    },

    # Retry settings
    "max_retries": 3,
    "retry_on": ["extraction_failed", "parse_failed"],

    # Logging
    "log_all_decisions": True,
    "track_ground_truth": True
}
```

### 5.2 Integration Example

```python
def solve_aimo3_problem(
    problem: str,
    llm_client,  # Your gpt-oss-120b client
    config: Dict = AIMO3_CONFIG
) -> int:
    """
    Solve an AIMO3 problem with full Guardian protection.

    Returns:
        Integer answer in [0, 999]
    """

    # Initialize Guardian
    guardian = OutputGuardian(
        answer_range=config["answer_range"],
        max_retries=config["max_retries"],
        repair_enabled=True
    )

    # Create guarded inference
    def inference_fn(system: str, user: str) -> str:
        return llm_client.generate(
            system_prompt=system,
            user_message=user,
            max_tokens=2048,
            temperature=0.0
        )

    pipeline = GuardedInference(
        inference_fn=inference_fn,
        output_guardian=guardian,
        enable_consensus=config["consensus"]["enabled"],
        consensus_samples=config["consensus"]["samples"]
    )

    # Solve
    result = pipeline.solve(problem)

    if result.result in (ValidationResult.ACCEPT, ValidationResult.REPAIR):
        return result.answer
    else:
        # Fallback: return 0 or -1 to indicate failure
        return 0
```

---

## 6. Testing and Validation

### 6.1 Unit Tests

```python
import pytest

def test_answer_extraction():
    guardian = OutputGuardian()

    test_cases = [
        (r"Therefore \boxed{42}", "42"),
        ("The answer is 123", "123"),
        ("After calculation, we get 456.", "456"),
        ("= 789", "789"),
        ("No answer here", None),
    ]

    for response, expected in test_cases:
        result = guardian.extract_answer(response)
        assert result == expected, f"Failed for: {response}"


def test_repair_strategies():
    guardian = OutputGuardian(answer_range=(0, 999))

    # Test modular reduction
    assert guardian._attempt_repair(1234) == 234

    # Test absolute value
    assert guardian._attempt_repair(-42) == 42

    # Test last digits
    assert guardian._attempt_repair(123456) == 456


def test_consensus():
    guardian = OutputGuardian()
    consensus = ConsensusGuardian(guardian)

    responses = [
        r"\boxed{42}",
        r"\boxed{42}",
        r"\boxed{42}",
        r"\boxed{99}",  # Outlier
        r"\boxed{42}",
    ]

    result = consensus.aggregate(responses)
    assert result.answer == 42
    assert result.result == ValidationResult.ACCEPT
```

### 6.2 Integration Tests

```python
def test_full_pipeline():
    """Test the complete Guardian pipeline"""

    # Mock LLM that sometimes makes mistakes
    class MockLLM:
        def __init__(self):
            self.call_count = 0
            self.responses = [
                "Let me solve... \\boxed{1234}",  # Out of range
                "Recalculating... \\boxed{234}",   # Correct
            ]

        def generate(self, **kwargs):
            resp = self.responses[min(self.call_count, len(self.responses)-1)]
            self.call_count += 1
            return resp

    llm = MockLLM()
    guardian = OutputGuardian(answer_range=(0, 999))

    pipeline = GuardedInference(
        inference_fn=lambda s, u: llm.generate(),
        output_guardian=guardian,
        max_retries=3
    )

    result = pipeline.solve("What is 1234 mod 1000?")

    # Should either repair to 234 or retry and get 234
    assert result.answer == 234
```

---

## 7. Deployment Checklist

### 7.1 Pre-Deployment

- [ ] All unit tests passing
- [ ] Integration tests with actual LLM
- [ ] Logging pipeline verified
- [ ] Metrics collection working
- [ ] Ground truth validation on test set
- [ ] Repair strategies tested on edge cases

### 7.2 Runtime

- [ ] Monitor validation rates (accept/repair/retry/reject)
- [ ] Track accuracy against known answers
- [ ] Alert on high rejection rate (> 20%)
- [ ] Log all decisions for analysis

### 7.3 Post-Competition

- [ ] Export decision logs
- [ ] Analyze failure patterns
- [ ] Update rules based on findings
- [ ] Retrain if systematic issues found

---

## 8. References

1. **LatticeForge Guardian Codebase**: `packages/web/lib/reasoning/security.ts`
2. **VERA+Guardian Architecture**: `research/ip/methodology/VERA_GUARDIAN_ARCHITECTURE.md`
3. **Database Schema**: `packages/web/supabase/migrations/20241210_guardian_governance.sql`
4. **Model Router**: `packages/web/lib/inference/ModelRouter.ts`

---

*This document was extracted and formalized from the LatticeForge production codebase for the purpose of enabling similar Guardian implementations in other domains.*
