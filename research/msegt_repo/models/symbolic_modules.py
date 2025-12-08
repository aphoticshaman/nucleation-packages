import math
from typing import Optional

def catalan(n: int) -> int:
    if n < 0:
        return 0
    return math.comb(2*n, n) // (n+1)

def legendre_v_p_factorial(n: int, p: int) -> int:
    v = 0
    while n > 0:
        n //= p
        v += n
    return v

def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(max(0, n)):
        a, b = b, a + b
    return a

def divisor_count(n: int) -> int:
    if n <= 0:
        return 0
    x = n
    c = 1
    p = 2
    while p * p <= x:
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            c *= (e + 1)
        p += 1 if p == 2 else 2
    if x > 1:
        c *= 2
    return c

def classify_problem(problem: str) -> str:
    p = problem.lower()
    if any(w in p for w in ["tournament", "round robin", "bracket"]):
        return "tournament"
    if any(w in p for w in ["spiral", "fibonacci", "golden ratio", "phi"]):
        return "geometry_fibo"
    if any(w in p for w in ["cyclotomic", "roots of unity", "x^n - 1"]):
        return "cyclotomic"
    if any(w in p for w in ["divisor", "tau(", "sigma(", "sum of divisors"]):
        return "divisor"
    return "generic"

def symbolic_sanity_score(step_text: str, problem_type: str) -> float:
    s = step_text.lower()
    score = 0.0
    if problem_type == "tournament" and "catalan" in s:
        score += 0.5
    if problem_type == "geometry_fibo" and "fibonacci" in s:
        score += 0.5
    if problem_type == "cyclotomic" and ("x^n - 1" in s or "roots of unity" in s):
        score += 0.5
    if problem_type == "divisor" and ("tau(" in s or "sum of divisors" in s):
        score += 0.5
    return score
