"""
Symbolic Verification
=====================

SymPy-based verification of mathematical expressions.
"""

import re
from typing import Tuple, Optional


class SymbolicVerifier:
    """
    Hooks for symbolic verification of math expressions.

    Integrates with SymPy for equation checking.
    """

    def __init__(self):
        self.sympy_available = False
        try:
            import sympy
            self.sympy_available = True
            self.sympy = sympy
        except ImportError:
            pass

    def verify_equation(self, equation: str) -> Tuple[bool, str]:
        """
        Verify an equation is mathematically valid.

        Returns:
            (is_valid, reason)
        """
        if not self.sympy_available:
            return True, "SymPy not available"

        try:
            from sympy.parsing.sympy_parser import parse_expr

            if '=' in equation:
                parts = equation.split('=')
                if len(parts) != 2:
                    return True, "Multiple equals signs"

                left, right = parts
                left_expr = parse_expr(left.strip())
                right_expr = parse_expr(right.strip())

                # Check if they're equivalent
                diff = self.sympy.simplify(left_expr - right_expr)
                if diff == 0:
                    return True, "Equation is valid"
                else:
                    return False, f"Sides differ by {diff}"
            else:
                # Just check it parses
                parse_expr(equation)
                return True, "Expression is valid"

        except Exception as e:
            return False, f"Parse error: {e}"

    def simplify_expression(self, expr: str) -> str:
        """Simplify a mathematical expression."""
        if not self.sympy_available:
            return expr

        try:
            from sympy.parsing.sympy_parser import parse_expr
            parsed = parse_expr(expr)
            simplified = self.sympy.simplify(parsed)
            return str(simplified)
        except:
            return expr

    def check_numeric_answer(
        self,
        answer_text: str,
        expected: Optional[float] = None,
    ) -> Tuple[bool, Optional[int]]:
        """
        Extract and optionally verify numeric answer.

        Returns:
            (found_answer, value)
        """
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer_text)
        if boxed_match:
            try:
                value = int(float(boxed_match.group(1)))
                valid = expected is None or abs(value - expected) < 1e-6
                return valid, value
            except ValueError:
                pass

        # Look for "answer is X"
        answer_match = re.search(r'answer\s+is\s+(\d+)', answer_text, re.IGNORECASE)
        if answer_match:
            try:
                value = int(answer_match.group(1))
                valid = expected is None or abs(value - expected) < 1e-6
                return valid, value
            except ValueError:
                pass

        # Look for final numeric answer
        numbers = re.findall(r'-?\d+', answer_text[-200:])
        if numbers:
            try:
                value = int(numbers[-1])
                valid = expected is None or abs(value - expected) < 1e-6
                return valid, value
            except ValueError:
                pass

        return False, None

    def verify_answer_in_range(self, answer: int, min_val: int = 0, max_val: int = 99999) -> bool:
        """Check if answer is in valid range."""
        return min_val <= answer <= max_val


__all__ = ["SymbolicVerifier"]
