"""
Proof Constraints
=================

Hard and soft constraints for mathematical proof generation.

Prevents:
- Unbalanced brackets/LaTeX
- Variable use before definition
- Numeric hallucinations
- Repetitive loops
"""

import re
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple
from enum import Enum


class ConstraintViolation(Enum):
    """Types of constraint violations."""
    UNBALANCED_BRACKET = "unbalanced_bracket"
    MISMATCHED_BRACKET = "mismatched_bracket"
    UNDEFINED_VARIABLE = "undefined_variable"
    NUMERIC_HALLUCINATION = "numeric_hallucination"
    REPETITION = "repetition"
    EXCESSIVE_DEPTH = "excessive_depth"


@dataclass
class BracketTracker:
    """
    Tracks bracket balance during generation.

    Prevents:
    - Unmatched opening brackets
    - Unmatched closing brackets
    - Mismatched bracket types (e.g., ( closed with ])

    Supports Python, LaTeX, and math notation.
    """

    PAIRS = {
        '(': ')',
        '[': ']',
        '{': '}',
        '\\(': '\\)',
        '\\[': '\\]',
        '\\{': '\\}',
        '\\left(': '\\right)',
        '\\left[': '\\right]',
        '\\left{': '\\right}',
        '\\begin{': '\\end{',
    }

    OPENERS: Set[str] = field(default_factory=lambda: set(BracketTracker.PAIRS.keys()))
    CLOSERS: Set[str] = field(default_factory=lambda: set(BracketTracker.PAIRS.values()))

    stack: List[str] = field(default_factory=list)
    depth: int = 0

    def __post_init__(self):
        self.OPENERS = set(self.PAIRS.keys())
        self.CLOSERS = set(self.PAIRS.values())

    def update(self, token: str) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        Update bracket state with new token.

        Returns:
            (is_valid, violation_type)
        """
        # Check for openers
        for opener in self.OPENERS:
            if opener in token:
                self.stack.append(opener)
                self.depth += 1

        # Check for closers
        for closer in self.CLOSERS:
            if closer in token:
                if not self.stack:
                    return False, ConstraintViolation.UNBALANCED_BRACKET

                expected_opener = None
                for op, cl in self.PAIRS.items():
                    if cl == closer:
                        expected_opener = op
                        break

                if expected_opener and self.stack[-1] != expected_opener:
                    return False, ConstraintViolation.MISMATCHED_BRACKET

                self.stack.pop()
                self.depth -= 1

        return True, None

    def is_balanced(self) -> bool:
        """Check if all brackets are balanced."""
        return len(self.stack) == 0

    def must_close(self) -> Optional[str]:
        """Return required closer if we must close, else None."""
        if self.stack:
            return self.PAIRS.get(self.stack[-1])
        return None

    def clone(self) -> 'BracketTracker':
        """Create a copy for branching."""
        new = BracketTracker()
        new.stack = self.stack.copy()
        new.depth = self.depth
        return new

    def reset(self):
        """Reset state."""
        self.stack = []
        self.depth = 0


@dataclass
class EquationTracker:
    """
    Tracks equation/expression state.

    Ensures:
    - Variables defined before use (soft constraint)
    - Equation sides balance
    - Numeric consistency
    """

    defined_vars: Set[str] = field(default_factory=set)
    in_equation: bool = False
    lhs_vars: Set[str] = field(default_factory=set)
    rhs_vars: Set[str] = field(default_factory=set)
    numbers_seen: List[float] = field(default_factory=list)
    problem_numbers: Set[float] = field(default_factory=set)

    def set_problem_numbers(self, text: str):
        """Extract numbers from problem statement."""
        for match in re.findall(r'-?\d+\.?\d*', text):
            try:
                self.problem_numbers.add(float(match))
            except ValueError:
                pass

    def extract_variables(self, text: str) -> Set[str]:
        """Extract variable names from text."""
        # Single letters that look like variables
        vars_found = set(re.findall(r'\b([a-zA-Z])\b', text))
        # Exclude common non-variables
        vars_found -= {'a', 'A', 'I', 'O', 'e', 'i'}
        return vars_found

    def extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text."""
        numbers = []
        for match in re.findall(r'-?\d+\.?\d*', text):
            try:
                numbers.append(float(match))
            except ValueError:
                pass
        return numbers

    def update(self, token: str) -> Tuple[bool, Optional[ConstraintViolation], float]:
        """
        Update equation state with new token.

        Returns:
            (is_valid, violation_type, penalty)
        """
        penalty = 0.0

        # Track equation boundaries
        if '=' in token and not self.in_equation:
            self.in_equation = True
            self.lhs_vars = self.defined_vars.copy()

        # Extract and track variables
        new_vars = self.extract_variables(token)
        self.defined_vars.update(new_vars)

        # Extract numbers and check consistency
        numbers = self.extract_numbers(token)
        for num in numbers:
            self.numbers_seen.append(num)

            # Penalize very large numbers not in problem
            if abs(num) > 10000 and num not in self.problem_numbers:
                penalty -= 2.0

            # Penalize non-integers in integer problems
            if self.problem_numbers:
                if all(n == int(n) for n in self.problem_numbers):
                    if num != int(num):
                        penalty -= 1.0

        if penalty < -3.0:
            return False, ConstraintViolation.NUMERIC_HALLUCINATION, penalty

        return True, None, penalty

    def clone(self) -> 'EquationTracker':
        """Create a copy for branching."""
        new = EquationTracker()
        new.defined_vars = self.defined_vars.copy()
        new.in_equation = self.in_equation
        new.lhs_vars = self.lhs_vars.copy()
        new.rhs_vars = self.rhs_vars.copy()
        new.numbers_seen = self.numbers_seen.copy()
        new.problem_numbers = self.problem_numbers.copy()
        return new

    def reset(self):
        """Reset state."""
        self.defined_vars = set()
        self.in_equation = False
        self.numbers_seen = []


@dataclass
class RepetitionTracker:
    """
    Tracks and blocks repetitive text.

    Models sometimes loop on the same phrase.
    """

    window_size: int = 100
    threshold: int = 3
    recent_text: str = ""

    def update(self, token: str) -> Tuple[bool, Optional[ConstraintViolation], float]:
        """
        Check for repetition.

        Returns:
            (is_valid, violation_type, penalty)
        """
        self.recent_text += token

        # Keep window size manageable
        if len(self.recent_text) > self.window_size * 2:
            self.recent_text = self.recent_text[-self.window_size:]

        if len(self.recent_text) < self.window_size:
            return True, None, 0.0

        recent = self.recent_text[-self.window_size:]

        # Check for exact repetition
        if len(token) > 3 and token in recent:
            count = recent.count(token)
            if count > self.threshold:
                return False, ConstraintViolation.REPETITION, -2.0 * count

        # Check for loop patterns
        for window in [10, 20, 30]:
            if len(recent) >= window * 2:
                if recent[-window:] == recent[-2*window:-window]:
                    return False, ConstraintViolation.REPETITION, -5.0

        return True, None, 0.0

    def reset(self):
        """Reset state."""
        self.recent_text = ""


class ProofConstraints:
    """
    Unified proof constraint checker.

    Combines all constraint trackers and provides unified interface.
    """

    def __init__(self, max_depth: int = 10):
        self.brackets = BracketTracker()
        self.equations = EquationTracker()
        self.repetition = RepetitionTracker()
        self.max_depth = max_depth
        self.violations: List[ConstraintViolation] = []

    def set_problem(self, problem_text: str):
        """Initialize with problem context."""
        self.equations.set_problem_numbers(problem_text)
        self.reset()

    def check(self, token: str) -> Tuple[bool, float]:
        """
        Check all constraints for a token.

        Args:
            token: The token to check

        Returns:
            (is_valid, penalty) where penalty is log-prob adjustment
        """
        total_penalty = 0.0

        # Bracket check (hard constraint)
        valid, violation = self.brackets.update(token)
        if not valid:
            self.violations.append(violation)
            return False, float('-inf')

        # Depth check (hard constraint)
        if self.brackets.depth > self.max_depth:
            self.violations.append(ConstraintViolation.EXCESSIVE_DEPTH)
            return False, float('-inf')

        # Equation check (soft constraint)
        valid, violation, penalty = self.equations.update(token)
        total_penalty += penalty
        if not valid:
            self.violations.append(violation)
            return False, float('-inf')

        # Repetition check (soft-ish constraint)
        valid, violation, penalty = self.repetition.update(token)
        total_penalty += penalty
        if not valid:
            self.violations.append(violation)
            # Repetition is soft - heavy penalty but don't block
            return True, penalty

        # Proof keyword boost
        total_penalty += self._proof_keyword_boost(token)

        return True, total_penalty

    def _proof_keyword_boost(self, token: str) -> float:
        """Boost proof structure keywords."""
        KEYWORDS = {
            'therefore': 0.5, 'hence': 0.5, 'thus': 0.5,
            'since': 0.3, 'because': 0.3,
            'suppose': 0.4, 'assume': 0.4, 'let': 0.3,
            'given': 0.3, 'prove': 0.4, 'show': 0.3,
            'QED': 1.0, 'contradiction': 0.5,
        }

        text_lower = token.lower()
        bonus = 0.0
        for keyword, boost in KEYWORDS.items():
            if keyword.lower() in text_lower:
                bonus += boost

        return bonus

    def is_valid_end(self) -> bool:
        """Check if current state is valid for ending generation."""
        return self.brackets.is_balanced()

    def get_violations(self) -> List[ConstraintViolation]:
        """Get list of violations encountered."""
        return self.violations

    def reset(self):
        """Reset all trackers."""
        self.brackets.reset()
        self.equations.reset()
        self.repetition.reset()
        self.violations = []

    def clone(self) -> 'ProofConstraints':
        """Create a copy for branching."""
        new = ProofConstraints(max_depth=self.max_depth)
        new.brackets = self.brackets.clone()
        new.equations = self.equations.clone()
        new.violations = self.violations.copy()
        return new


__all__ = [
    "BracketTracker",
    "EquationTracker",
    "RepetitionTracker",
    "ProofConstraints",
    "ConstraintViolation",
]
