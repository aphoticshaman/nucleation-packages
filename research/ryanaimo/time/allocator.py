"""
Time Allocation
===============

Adaptive time management for 5-hour budget across 50 problems.
"""

import time
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeManager:
    """Manages time budget for the competition."""

    total_budget: float  # Total seconds
    start_time: float = 0.0
    problems_completed: int = 0
    total_problems: int = 50

    def __post_init__(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        """Seconds elapsed since start."""
        return time.time() - self.start_time

    def remaining(self) -> float:
        """Seconds remaining in budget."""
        return max(0.0, self.total_budget - self.elapsed())

    def remaining_problems(self) -> int:
        """Problems left to solve."""
        return max(1, self.total_problems - self.problems_completed)

    def time_str(self) -> str:
        """Human-readable remaining time."""
        r = self.remaining()
        return f"{int(r // 60)}m{int(r % 60)}s"

    def mark_completed(self):
        """Mark a problem as completed."""
        self.problems_completed += 1


class TimeAllocator:
    """
    Allocates time per problem based on difficulty and remaining budget.

    Strategy:
    1. Base time = remaining / remaining_problems
    2. Adjust by estimated difficulty
    3. Reserve buffer for hard problems
    4. Minimum and maximum bounds
    """

    def __init__(
        self,
        total_budget: float,
        min_time: float = 60.0,
        max_time: float = 900.0,
        reserve_fraction: float = 0.15,
    ):
        """
        Args:
            total_budget: Total seconds available
            min_time: Minimum seconds per problem
            max_time: Maximum seconds per problem
            reserve_fraction: Fraction to reserve for hard problems
        """
        self.manager = TimeManager(total_budget=total_budget)
        self.min_time = min_time
        self.max_time = max_time
        self.reserve_fraction = reserve_fraction

        # Difficulty history for calibration
        self.difficulty_history: list = []

    def estimate_difficulty(self, problem_text: str) -> float:
        """
        Estimate problem difficulty from text.

        Returns: 0.0 (easy) to 1.0 (hard)
        """
        difficulty = 0.5  # Base

        # Length factor
        length = len(problem_text)
        if length > 1000:
            difficulty += 0.1
        if length > 2000:
            difficulty += 0.1

        text_lower = problem_text.lower()

        # Hard keywords
        hard_keywords = [
            'generating function', 'polynomial', 'prove that',
            'if and only if', 'determine all', 'find all',
            'convex', 'tangent', 'inscribed', 'circumscribed',
            'recursion', 'recurrence', 'functional equation',
        ]
        for kw in hard_keywords:
            if kw in text_lower:
                difficulty += 0.08

        # Easy keywords
        easy_keywords = [
            'how many', 'compute', 'calculate', 'find the value',
            'remainder', 'modulo',
        ]
        for kw in easy_keywords:
            if kw in text_lower:
                difficulty -= 0.05

        # Multi-part problems are harder
        if re.search(r'\(a\).*\(b\)', problem_text, re.DOTALL):
            difficulty += 0.15

        # Bound to [0, 1]
        return max(0.0, min(1.0, difficulty))

    def allocate(self, problem_text: str) -> float:
        """
        Allocate time for a problem.

        Args:
            problem_text: The problem statement

        Returns:
            Seconds allocated for this problem
        """
        difficulty = self.estimate_difficulty(problem_text)
        self.difficulty_history.append(difficulty)

        # Base time
        remaining = self.manager.remaining()
        remaining_problems = self.manager.remaining_problems()

        # Reserve some time for hard problems
        available = remaining * (1 - self.reserve_fraction)
        base_time = available / remaining_problems

        # Difficulty multiplier (0.5x to 2x)
        multiplier = 0.5 + difficulty * 1.5

        allocated = base_time * multiplier

        # Bounds
        allocated = max(self.min_time, min(self.max_time, allocated))

        # Don't allocate more than remaining
        allocated = min(allocated, remaining - 30)  # Keep 30s buffer

        return allocated

    def mark_completed(self):
        """Mark current problem as completed."""
        self.manager.mark_completed()

    def time_remaining(self) -> float:
        """Get remaining time."""
        return self.manager.remaining()

    def time_str(self) -> str:
        """Human-readable time remaining."""
        return self.manager.time_str()


__all__ = ["TimeAllocator", "TimeManager"]
