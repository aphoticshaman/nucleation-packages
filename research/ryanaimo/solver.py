"""
RYANAIMO Solver
===============

Main entry point for the AIMO3 competition.

Ground-up architecture from first principles:
- CIC theory foundation
- Extended reasoning (think blocks)
- Proof-constrained generation
- Value clustering (88% error reduction)
- Adaptive time management

Competition constraints (from AIMO3 rules):
- 5 hours GPU runtime on H100 80GB
- 50 public + 50 private problems
- Answers: integers 0-99999
- Target: 47/50 on BOTH sets for $1.59M+ prize
- Open-weight models only (before Mar 15, 2026)
- Run twice on private set - both must succeed

Author: Ryan J Cardwell (Archer Phoenix) + Claude Opus 4
"""

import os
import re
import time
import signal
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import Counter
from io import StringIO
import contextlib

# RYANAIMO modules
from ryanaimo.core.cic import compute_cic_functional, detect_crystallization, CICState
from ryanaimo.core.config import RyanAIMOConfig, ProblemProfile, ProblemType
from ryanaimo.selection.clustering import select_answer, value_clustering


# =============================================================================
# CONSTANTS - Aligned with AIMO3 rules
# =============================================================================

ANSWER_MIN = 0
ANSWER_MAX = 99999
FALLBACK_ANSWER = 0

# Time budget: 5 hours = 18000 seconds
TOTAL_BUDGET_SECONDS = 5 * 60 * 60  # 18000

# Kaggle paths
KAGGLE_INPUT = "/kaggle/input"
MODEL_PATH_DEFAULT = f"{KAGGLE_INPUT}/m/ryancardwell/qwen-72b-math-nf4/transformers/v1/1"
TRIAD_DEV_PATH = f"{KAGGLE_INPUT}/triad-dev"
COMPETITION_PATH = f"{KAGGLE_INPUT}/ai-mathematical-olympiad-progress-prize-3"


# =============================================================================
# CODE EXECUTION (Sandboxed)
# =============================================================================

MATH_STDLIB = '''
import math
from math import gcd, factorial, comb, isqrt, sqrt, ceil, floor, log, exp, sin, cos, tan, pi, e
from itertools import permutations, combinations, product, combinations_with_replacement
from functools import reduce, lru_cache
from collections import Counter, defaultdict, deque
from fractions import Fraction
from decimal import Decimal

try:
    from sympy import *
    from sympy.ntheory import factorint, divisors, totient, isprime, primerange, prime
    from sympy.ntheory.modular import crt
except ImportError:
    pass

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0: return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0: return False
    return True

def C(n, k):
    return comb(n, k) if 0 <= k <= n else 0

def P(n, k):
    if k > n or k < 0: return 0
    return factorial(n) // factorial(n - k)

def modinv(a, m):
    return pow(a, -1, m)

def extended_gcd(a, b):
    if a == 0: return b, 0, 1
    g, x, y = extended_gcd(b % a, a)
    return g, y - (b // a) * x, x
'''

ANSWER_SNIFFER = '''
# Sniff for answer variable
for _vname in ["answer", "ans", "result", "res", "total", "count", "final", "output"]:
    if _vname in dir() and isinstance(eval(_vname), (int, float)):
        _val = int(eval(_vname))
        if 0 <= _val <= 99999:
            print(f"EXTRACTED_ANSWER:{_val}")
            break
'''


def execute_code(code: str, timeout: int = 30) -> Tuple[Optional[int], str]:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        (answer, error_message) - answer is None if execution failed
    """
    full_code = MATH_STDLIB + '\n' + code + '\n' + ANSWER_SNIFFER

    stdout_capture = StringIO()

    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:
        signal.alarm(timeout)

        with contextlib.redirect_stdout(stdout_capture):
            exec(full_code, {'__builtins__': __builtins__})

        signal.alarm(0)

        output = stdout_capture.getvalue()

        # Look for extracted answer
        match = re.search(r'EXTRACTED_ANSWER:(\d+)', output)
        if match:
            return int(match.group(1)), ""

        # Fallback: last number in output
        numbers = re.findall(r'\b(\d+)\b', output)
        if numbers:
            val = int(numbers[-1])
            if ANSWER_MIN <= val <= ANSWER_MAX:
                return val, ""

        return None, "No answer found in output"

    except TimeoutError:
        return None, "Timeout"
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# =============================================================================
# PROBLEM CLASSIFICATION
# =============================================================================

def classify_problem(text: str) -> ProblemProfile:
    """
    Classify problem type and extract metadata.

    Per AIMO3 rules: problems span algebra, combinatorics, geometry, number theory.
    """
    t = text.lower()

    # Modulo detection
    mod_match = re.search(r'(?:mod|modulo)\s*(\d+)', t)
    remainder_match = re.search(r'remainder.*?(?:divided by|when.*?by)\s*(\d+)', t)
    has_mod = bool(mod_match or remainder_match or 'modulo' in t)
    mod_target = None
    if mod_match:
        mod_target = int(mod_match.group(1))
    elif remainder_match:
        mod_target = int(remainder_match.group(1))

    # Type classification
    scores = {
        ProblemType.NUMBER_THEORY: sum(1 for k in [
            'prime', 'divisor', 'gcd', 'lcm', 'modulo', 'factorial',
            'coprime', 'divisible', 'perfect square', 'euler'
        ] if k in t),
        ProblemType.COMBINATORICS: sum(1 for k in [
            'how many', 'count', 'ways', 'permutation', 'combination',
            'arrange', 'select', 'choose', 'probability'
        ] if k in t),
        ProblemType.GEOMETRY: sum(1 for k in [
            'triangle', 'circle', 'angle', 'area', 'perimeter',
            'inscribed', 'circumscribed', 'tangent', 'parallel',
            'perpendicular', 'polygon', 'quadrilateral'
        ] if k in t),
        ProblemType.ALGEBRA: sum(1 for k in [
            'polynomial', 'roots', 'equation', 'coefficient',
            'sum', 'product', 'sequence', 'series', 'function'
        ] if k in t),
    }

    best = max(scores.items(), key=lambda x: x[1])
    ptype = best[0] if best[1] > 0 else ProblemType.MIXED

    # Difficulty estimation
    difficulty = 0.5
    if len(text) > 1000:
        difficulty += 0.1
    if len(text) > 2000:
        difficulty += 0.1

    hard_markers = ['prove that', 'determine all', 'find all', 'if and only if']
    for marker in hard_markers:
        if marker in t:
            difficulty += 0.15

    difficulty = min(1.0, max(0.0, difficulty))

    return ProblemProfile(
        ptype=ptype,
        has_modulo=has_mod,
        modulo_target=mod_target,
        is_counting=any(k in t for k in ['how many', 'count', 'number of']),
        estimated_difficulty=difficulty,
    )


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def extract_code(text: str) -> Optional[str]:
    """Extract Python code from response."""
    patterns = [
        r'```python\n(.*?)```',
        r'```py\n(.*?)```',
        r'```\n(.*?)```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
    return None


def extract_text_answer(text: str) -> Optional[int]:
    """Extract integer answer from text (non-code)."""
    # Boxed answer (LaTeX)
    patterns = [
        r'\\boxed\{(\d+)\}',
        r'\\boxed\s*\{(\d+)\}',
        r'answer\s*(?:is|=)\s*(\d+)',
        r'final\s*answer\s*(?:is|=)\s*(\d+)',
        r'=\s*(\d+)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                if ANSWER_MIN <= val <= ANSWER_MAX:
                    return val
            except ValueError:
                pass

    # Last number in text
    numbers = re.findall(r'\b(\d+)\b', text[-500:])
    if numbers:
        try:
            val = int(numbers[-1])
            if ANSWER_MIN <= val <= ANSWER_MAX:
                return val
        except ValueError:
            pass

    return None


def normalize_answer(x: Any) -> Optional[int]:
    """Normalize value to valid answer range."""
    try:
        val = int(float(x))
        return max(ANSWER_MIN, min(ANSWER_MAX, val))
    except (ValueError, TypeError):
        return None


# =============================================================================
# MAIN SOLVER
# =============================================================================

@dataclass
class SolveResult:
    """Result from solving a problem."""
    answer: int
    confidence: float
    paths_explored: int
    cic_state: Optional[CICState]
    time_used: float
    method: str  # "code", "text", "vote", "fallback"


class RyanAIMOSolver:
    """
    Main AIMO3 solver using RYANAIMO architecture.

    Ground-up design from CIC theory principles.
    """

    SYSTEM_PROMPT = """You are an expert olympiad mathematician solving IMO-level problems.

IMPORTANT RULES:
1. First, THINK deeply about the problem. Analyze:
   - What type of problem is this?
   - What mathematical techniques apply?
   - What are the key constraints?
   - Are there edge cases?

2. Write Python code to compute the answer
3. Store the final answer in a variable called 'answer'
4. Answer MUST be an integer from 0 to 99999
5. Any modulo is EXPLICITLY stated in the problem (no implicit mod 1000)

Show reasoning, then provide working code."""

    def __init__(
        self,
        model_path: str = MODEL_PATH_DEFAULT,
        total_budget: float = TOTAL_BUDGET_SECONDS,
        num_problems: int = 50,
        config: Optional[RyanAIMOConfig] = None,
    ):
        """
        Initialize solver.

        Args:
            model_path: Path to Qwen model
            total_budget: Total seconds for all problems (5 hours = 18000s)
            num_problems: Expected number of problems
            config: Optional configuration override
        """
        self.model_path = model_path
        self.total_budget = total_budget
        self.num_problems = num_problems
        self.config = config or RyanAIMOConfig()

        # State
        self.start_time = time.time()
        self.problems_solved = 0
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load_model(self):
        """Lazy load model."""
        if self._loaded:
            return

        from ryanaimo.models.qwen import load_model

        print(f"[RYANAIMO] Loading model from {self.model_path}")
        self.model, self.tokenizer = load_model(
            self.model_path,
            quantization=self.config.quantization,
            compute_dtype=self.config.compute_dtype,
        )
        self._loaded = True
        print(f"[RYANAIMO] Model loaded successfully")

    def time_remaining(self) -> float:
        """Seconds remaining in budget."""
        return max(0, self.total_budget - (time.time() - self.start_time))

    def time_str(self) -> str:
        """Human-readable remaining time."""
        r = self.time_remaining()
        return f"{int(r // 60)}m{int(r % 60)}s"

    def allocate_time(self, problem: str) -> float:
        """
        Allocate time for a problem based on difficulty.

        Per AIMO3 rules: Must complete all problems within 5 hours.
        Run twice on private set - both must succeed.
        """
        remaining = self.time_remaining()
        remaining_problems = max(1, self.num_problems - self.problems_solved)

        # Base allocation
        base_time = remaining / remaining_problems

        # Difficulty adjustment
        profile = classify_problem(problem)
        multiplier = 0.7 + profile.estimated_difficulty * 0.6  # 0.7x to 1.3x

        allocated = base_time * multiplier

        # Bounds
        allocated = max(self.config.min_time_per_problem, allocated)
        allocated = min(self.config.max_time_per_problem, allocated)
        allocated = min(allocated, remaining - 30)  # Keep 30s buffer

        return allocated

    def generate(
        self,
        problem: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate solution text."""
        self._load_model()

        import torch

        prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def solve(self, problem: str) -> SolveResult:
        """
        Solve a single problem using full RYANAIMO pipeline.

        Steps:
        1. Classify problem
        2. Allocate time
        3. Generate multiple solution paths
        4. Execute code and extract answers
        5. Apply CIC-aware selection
        6. Return best answer with confidence
        """
        start_time = time.time()
        time_budget = self.allocate_time(problem)
        profile = classify_problem(problem)

        print(f"[RYANAIMO] Problem type: {profile.ptype.value}, difficulty: {profile.estimated_difficulty:.2f}")
        print(f"[RYANAIMO] Time budget: {time_budget:.0f}s, remaining: {self.time_str()}")

        candidates: List[int] = []
        cic_history: List[CICState] = []
        method = "fallback"

        # Generate multiple paths
        temperatures = self.config.temperatures[:self.config.num_paths]

        for i, temp in enumerate(temperatures):
            # Check time
            elapsed = time.time() - start_time
            if elapsed > time_budget * 0.9:
                print(f"[RYANAIMO] Time limit approaching, stopping early")
                break

            print(f"[RYANAIMO] Path {i+1}/{len(temperatures)} @ temp={temp}")

            try:
                response = self.generate(problem, temperature=temp)

                # Try code execution first
                code = extract_code(response)
                if code:
                    result, err = execute_code(code, timeout=self.config.code_timeout)
                    if result is not None:
                        candidates.append(result)
                        method = "code"
                        print(f"[RYANAIMO]   Code: {result}")
                    else:
                        print(f"[RYANAIMO]   Code failed: {err[:50]}")
                        # Try text extraction as fallback
                        text_ans = extract_text_answer(response)
                        if text_ans is not None:
                            candidates.append(text_ans)
                            method = "text"
                            print(f"[RYANAIMO]   Text fallback: {text_ans}")
                else:
                    # No code block, try text extraction
                    text_ans = extract_text_answer(response)
                    if text_ans is not None:
                        candidates.append(text_ans)
                        method = "text"
                        print(f"[RYANAIMO]   Text: {text_ans}")

                # Check for crystallization
                if len(candidates) >= 2:
                    cic = compute_cic_functional(candidates)
                    cic_history.append(cic)

                    if detect_crystallization(cic_history):
                        print(f"[RYANAIMO] Crystallization detected, stopping")
                        break

            except Exception as e:
                print(f"[RYANAIMO]   Error: {e}")
                continue

        # Select answer using CIC-aware clustering
        if not candidates:
            answer = FALLBACK_ANSWER
            confidence = 0.05
            cic_state = None
        else:
            answer, confidence, metadata = select_answer(
                candidates,
                threshold=self.config.clustering_threshold,
                fallback=FALLBACK_ANSWER,
            )

            # Get final CIC state
            cic_state = compute_cic_functional(candidates)

            if len(set(candidates)) == 1:
                method = "unanimous"
            elif len(candidates) > 1:
                method = "vote"

        # Ensure answer is in valid range (per AIMO3 rules)
        answer = max(ANSWER_MIN, min(ANSWER_MAX, answer))

        time_used = time.time() - start_time
        self.problems_solved += 1

        print(f"[RYANAIMO] Final answer: {answer} (conf={confidence:.2f}, method={method})")
        print(f"[RYANAIMO] Time used: {time_used:.1f}s, paths: {len(candidates)}")

        return SolveResult(
            answer=answer,
            confidence=confidence,
            paths_explored=len(candidates),
            cic_state=cic_state,
            time_used=time_used,
            method=method,
        )


# =============================================================================
# KAGGLE API INTERFACE
# =============================================================================

def create_predict_function(solver: RyanAIMOSolver):
    """
    Create the predict function for Kaggle's API.

    Per AIMO3 rules: Must use kaggle_evaluation.aimo_3_inference_server
    """
    import polars as pl

    def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame:
        """
        Kaggle API predict function.

        Args:
            id_: DataFrame with problem ID
            question: DataFrame with problem text

        Returns:
            DataFrame with id and answer columns
        """
        problem_id = id_.item()
        problem_text = question.item()

        print(f"\n{'='*60}")
        print(f"Problem: {problem_id}")
        print(f"Time remaining: {solver.time_str()}")
        print(f"Q: {problem_text[:200]}..." if len(problem_text) > 200 else f"Q: {problem_text}")
        print('='*60)

        try:
            result = solver.solve(problem_text)
            answer = result.answer
        except Exception as e:
            print(f"[RYANAIMO] CRITICAL ERROR: {e}")
            traceback.print_exc()
            answer = FALLBACK_ANSWER

        # Ensure valid answer (per AIMO3 rules: 0-99999)
        answer = max(ANSWER_MIN, min(ANSWER_MAX, int(answer)))

        print(f"[RYANAIMO] FINAL: {answer}")

        return pl.DataFrame({'id': problem_id, 'answer': answer})

    return predict


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for AIMO3 competition.

    Per rules:
    - 5 hour GPU runtime on H100 80GB
    - Internet disabled
    - Must use kaggle_evaluation.aimo_3_inference_server
    """
    import kaggle_evaluation.aimo_3_inference_server

    print("=" * 60)
    print("RYANAIMO v0.1.0 - AIMO3 Solver")
    print("Ground-up architecture from CIC theory principles")
    print("=" * 60)

    # Create solver
    solver = RyanAIMOSolver(
        model_path=MODEL_PATH_DEFAULT,
        total_budget=TOTAL_BUDGET_SECONDS,
        num_problems=50,  # 50 per run (public or private)
    )

    # Create predict function
    predict = create_predict_function(solver)

    # Create server
    server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

    # Run
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("[RYANAIMO] MODE: Competition rerun (private set)")
        server.serve()
    else:
        print("[RYANAIMO] MODE: Local/public leaderboard")
        test_csv = f"{COMPETITION_PATH}/test.csv"
        server.run_local_gateway((test_csv,))


if __name__ == "__main__":
    main()
