"""
AIMO3 V9 Math Tools Library
===========================
Domain-specific mathematical functions for olympiad-level problems.
These are injected into the code sandbox so LLM-generated code can import and use them.

Covers:
- P5: Tournament/Catalan/Legendre valuation
- P7: Fibonacci/Binet/Spiral geometry  
- P9: Cyclotomic/roots of unity
- P10: Divisor theory/multiplicative functions

All functions are deterministic and numerically stable.
"""

import math
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
from fractions import Fraction
from decimal import Decimal, getcontext

# Set high precision for Decimal operations
getcontext().prec = 50

# =============================================================================
# P5: TOURNAMENT / CATALAN / LEGENDRE VALUATION
# =============================================================================

@lru_cache(maxsize=1024)
def catalan(n: int) -> int:
    """
    Compute the n-th Catalan number.
    C_n = (2n)! / ((n+1)! * n!) = binomial(2n, n) / (n+1)
    
    Used for: bracket counting, ballot sequences, non-crossing partitions,
    binary tree enumeration, mountain ranges, Dyck paths.
    """
    if n < 0:
        return 0
    if n <= 1:
        return 1
    return math.comb(2 * n, n) // (n + 1)


def catalan_list(n: int) -> List[int]:
    """Return list of first n+1 Catalan numbers [C_0, C_1, ..., C_n]."""
    return [catalan(i) for i in range(n + 1)]


@lru_cache(maxsize=1024)
def legendre_v_p(n: int, p: int) -> int:
    """
    Legendre's formula: v_p(n!) = sum_{i>=1} floor(n / p^i)
    Returns the largest power of prime p that divides n!
    
    Used for: prime factorization of factorials, divisibility in binomials,
    tournament bracket problems involving powers of 2.
    """
    if n < 0 or p < 2:
        return 0
    v = 0
    pk = p
    while pk <= n:
        v += n // pk
        pk *= p
    return v


def v_p_binomial(n: int, k: int, p: int) -> int:
    """
    Compute v_p(C(n,k)) = v_p(n!) - v_p(k!) - v_p((n-k)!)
    The p-adic valuation of a binomial coefficient.
    """
    if k < 0 or k > n:
        return 0  # binomial is 0
    return legendre_v_p(n, p) - legendre_v_p(k, p) - legendre_v_p(n - k, p)


def kummer_carries(n: int, k: int, p: int) -> int:
    """
    Kummer's theorem: v_p(C(n,k)) = number of carries when adding k and n-k in base p.
    Alternative to v_p_binomial, sometimes more efficient.
    """
    carries = 0
    carry = 0
    m = n - k
    while k > 0 or m > 0 or carry > 0:
        s = (k % p) + (m % p) + carry
        if s >= p:
            carries += 1
            carry = 1
        else:
            carry = 0
        k //= p
        m //= p
    return carries


def tournament_bracket_count(n: int) -> int:
    """
    Number of distinct tournament brackets for n teams.
    For single elimination with n = 2^k teams, this is C_{n-1} (Catalan).
    """
    # Check if n is power of 2
    if n < 1 or (n & (n - 1)) != 0:
        # Not a power of 2; use general formula
        return catalan(n - 1) if n >= 1 else 0
    return catalan(n - 1)


# =============================================================================
# P7: FIBONACCI / BINET / SPIRAL GEOMETRY
# =============================================================================

@lru_cache(maxsize=2048)
def fibonacci(n: int) -> int:
    """
    Compute the n-th Fibonacci number. F_0 = 0, F_1 = 1.
    Uses matrix exponentiation for large n.
    """
    if n < 0:
        # F_{-n} = (-1)^{n+1} * F_n
        return ((-1) ** (n + 1)) * fibonacci(-n)
    if n <= 1:
        return n
    
    # Matrix exponentiation: [[F_{n+1}, F_n], [F_n, F_{n-1}]] = [[1,1],[1,0]]^n
    def matrix_mult(A, B):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ]
    
    def matrix_pow(M, p):
        if p == 1:
            return M
        if p % 2 == 0:
            half = matrix_pow(M, p // 2)
            return matrix_mult(half, half)
        else:
            return matrix_mult(M, matrix_pow(M, p - 1))
    
    base = [[1, 1], [1, 0]]
    result = matrix_pow(base, n)
    return result[0][1]


@lru_cache(maxsize=2048)
def lucas(n: int) -> int:
    """
    Compute the n-th Lucas number. L_0 = 2, L_1 = 1.
    L_n = F_{n-1} + F_{n+1}
    """
    if n == 0:
        return 2
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n + 1)


def binet_approx(n: int, decimals: int = 15) -> Decimal:
    """
    Binet's formula approximation for F_n.
    F_n = (phi^n - psi^n) / sqrt(5)
    where phi = (1 + sqrt(5))/2, psi = (1 - sqrt(5))/2
    
    Returns Decimal for high precision.
    """
    getcontext().prec = decimals + 10
    sqrt5 = Decimal(5).sqrt()
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    return ((phi ** n) - (psi ** n)) / sqrt5


PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PSI = (1 - math.sqrt(5)) / 2


def golden_ratio() -> float:
    """Return the golden ratio phi = (1 + sqrt(5))/2 ≈ 1.618033988749895"""
    return PHI


def fibonacci_index_approx(f: int) -> int:
    """
    Given a Fibonacci number f, approximate which index n gives F_n = f.
    Uses inverse Binet: n ≈ log_phi(f * sqrt(5))
    """
    if f <= 0:
        return 0
    return round(math.log(f * math.sqrt(5)) / math.log(PHI))


def spiral_ratio(r1: float, r2: float) -> float:
    """
    Check if ratio r2/r1 is close to golden ratio.
    Returns ratio; compare to PHI for spiral similarity check.
    """
    if r1 == 0:
        return float('inf')
    return r2 / r1


def is_fibonacci(n: int) -> bool:
    """
    Check if n is a Fibonacci number.
    n is Fibonacci iff 5n^2 + 4 or 5n^2 - 4 is a perfect square.
    """
    if n < 0:
        return False
    
    def is_perfect_square(x):
        if x < 0:
            return False
        r = int(math.isqrt(x))
        return r * r == x
    
    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)


# =============================================================================
# P9: CYCLOTOMIC / ROOTS OF UNITY
# =============================================================================

@lru_cache(maxsize=512)
def euler_phi(n: int) -> int:
    """
    Euler's totient function phi(n).
    Number of integers in [1, n] coprime to n.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def mobius(n: int) -> int:
    """
    Möbius function μ(n).
    μ(n) = 0 if n has squared prime factor
    μ(n) = (-1)^k if n is product of k distinct primes
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    prime_count = 0
    p = 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            prime_count += 1
            if n % p == 0:  # squared factor
                return 0
        p += 1
    if n > 1:
        prime_count += 1
    return (-1) ** prime_count


def cyclotomic_order(n: int) -> int:
    """
    Degree of the n-th cyclotomic polynomial Φ_n(x).
    This equals phi(n).
    """
    return euler_phi(n)


def primitive_root_count(n: int) -> int:
    """
    Number of primitive n-th roots of unity.
    Equals phi(n).
    """
    return euler_phi(n)


def cyclotomic_divisors(n: int) -> List[int]:
    """
    Return all divisors d of n such that Φ_d(x) divides x^n - 1.
    (This is just all divisors of n.)
    """
    if n <= 0:
        return []
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def nth_roots_of_unity(n: int) -> List[complex]:
    """
    Return all n-th roots of unity as complex numbers.
    e^(2πik/n) for k = 0, 1, ..., n-1
    """
    import cmath
    return [cmath.exp(2j * cmath.pi * k / n) for k in range(n)]


def primitive_nth_roots(n: int) -> List[complex]:
    """
    Return primitive n-th roots of unity.
    These are e^(2πik/n) where gcd(k, n) = 1.
    """
    import cmath
    return [cmath.exp(2j * cmath.pi * k / n) 
            for k in range(n) if math.gcd(k, n) == 1]


def is_cyclotomic_period(sequence: List[int], check_len: int) -> bool:
    """
    Check if sequence has period dividing check_len.
    Used for detecting cyclotomic structure in functional equations.
    """
    if len(sequence) < check_len:
        return False
    for i in range(len(sequence) - check_len):
        if sequence[i] != sequence[i + check_len]:
            return False
    return True


# =============================================================================
# P10: DIVISOR THEORY / MULTIPLICATIVE FUNCTIONS
# =============================================================================

def divisors(n: int) -> List[int]:
    """Return sorted list of all positive divisors of n."""
    if n <= 0:
        return []
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def divisor_count(n: int) -> int:
    """
    τ(n) or d(n): Number of divisors of n.
    For n = p1^a1 * p2^a2 * ..., τ(n) = (a1+1)(a2+1)...
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    count = 1
    p = 2
    while p * p <= n:
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            count *= (exp + 1)
        p += 1
    if n > 1:
        count *= 2  # remaining prime factor with exponent 1
    return count


def divisor_sum(n: int, k: int = 1) -> int:
    """
    σ_k(n): Sum of k-th powers of divisors.
    σ_0(n) = τ(n), σ_1(n) = σ(n) = sum of divisors.
    """
    if n <= 0:
        return 0
    return sum(d ** k for d in divisors(n))


def sigma(n: int) -> int:
    """σ(n): Sum of divisors of n."""
    return divisor_sum(n, 1)


def prime_factorization(n: int) -> Dict[int, int]:
    """
    Return prime factorization as dict {prime: exponent}.
    Example: 12 -> {2: 2, 3: 1}
    """
    if n <= 1:
        return {}
    factors = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def is_prime(n: int) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witnesses to test
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def is_perfect(n: int) -> bool:
    """Check if n is a perfect number (σ(n) = 2n)."""
    return n > 0 and sigma(n) == 2 * n


def is_abundant(n: int) -> bool:
    """Check if n is abundant (σ(n) > 2n)."""
    return n > 0 and sigma(n) > 2 * n


def is_deficient(n: int) -> bool:
    """Check if n is deficient (σ(n) < 2n)."""
    return n > 0 and sigma(n) < 2 * n


def radical(n: int) -> int:
    """
    rad(n): Product of distinct prime factors of n.
    Example: rad(12) = rad(2^2 * 3) = 2 * 3 = 6
    """
    if n <= 1:
        return n if n >= 0 else 1
    factors = prime_factorization(n)
    result = 1
    for p in factors:
        result *= p
    return result


def omega(n: int) -> int:
    """
    ω(n): Number of distinct prime factors.
    Example: ω(12) = ω(2^2 * 3) = 2
    """
    return len(prime_factorization(n))


def big_omega(n: int) -> int:
    """
    Ω(n): Number of prime factors counted with multiplicity.
    Example: Ω(12) = Ω(2^2 * 3) = 3
    """
    factors = prime_factorization(n)
    return sum(factors.values())


def liouville_lambda(n: int) -> int:
    """
    Liouville function λ(n) = (-1)^Ω(n).
    """
    return (-1) ** big_omega(n)


def gcd_extended(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm.
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if b == 0:
        return (a, 1, 0)
    g, x, y = gcd_extended(b, a % b)
    return (g, y, x - (a // b) * y)


def mod_inverse(a: int, m: int) -> Optional[int]:
    """
    Modular inverse of a mod m.
    Returns x such that a*x ≡ 1 (mod m), or None if no inverse exists.
    """
    g, x, _ = gcd_extended(a % m, m)
    if g != 1:
        return None
    return x % m


def chinese_remainder(remainders: List[int], moduli: List[int]) -> Optional[int]:
    """
    Chinese Remainder Theorem.
    Find x such that x ≡ r_i (mod m_i) for all i.
    Returns smallest non-negative x, or None if no solution.
    """
    if len(remainders) != len(moduli):
        return None
    if not moduli:
        return 0
    
    result = remainders[0]
    lcm = moduli[0]
    
    for i in range(1, len(moduli)):
        r, m = remainders[i], moduli[i]
        g, p, _ = gcd_extended(lcm, m)
        if (r - result) % g != 0:
            return None  # No solution
        result += lcm * ((r - result) // g * p % (m // g))
        lcm = lcm // g * m
        result %= lcm
    
    return result


# =============================================================================
# UTILITY / CROSS-DOMAIN FUNCTIONS
# =============================================================================

def is_perfect_power(n: int) -> Tuple[bool, int, int]:
    """
    Check if n = a^b for some integers a, b > 1.
    Returns (True, a, b) if yes, (False, n, 1) otherwise.
    """
    if n <= 1:
        return (False, n, 1)
    
    for b in range(2, n.bit_length() + 1):
        a = round(n ** (1/b))
        for candidate in [a - 1, a, a + 1]:
            if candidate > 1 and candidate ** b == n:
                return (True, candidate, b)
    return (False, n, 1)


def integer_sqrt(n: int) -> int:
    """Integer square root (floor of sqrt(n))."""
    if n < 0:
        raise ValueError("Square root of negative number")
    return math.isqrt(n)


def is_square(n: int) -> bool:
    """Check if n is a perfect square."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def sum_of_squares(n: int, count: int = 2) -> Optional[List[int]]:
    """
    Find representation of n as sum of 'count' squares.
    Returns list of integers [a1, a2, ...] such that sum(ai^2) = n.
    Returns None if no representation found.
    """
    if count == 2:
        # Fermat's theorem: representable iff all p ≡ 3 (mod 4) have even exponent
        factors = prime_factorization(n)
        for p, e in factors.items():
            if p % 4 == 3 and e % 2 == 1:
                return None
        # Find actual representation
        for a in range(int(math.isqrt(n)) + 1):
            b_sq = n - a * a
            if is_square(b_sq):
                return [a, int(math.isqrt(b_sq))]
    return None


def binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def multinomial(n: int, *ks: int) -> int:
    """
    Multinomial coefficient n! / (k1! * k2! * ... * km!)
    where sum(ks) should equal n.
    """
    if sum(ks) != n:
        return 0
    result = math.factorial(n)
    for k in ks:
        result //= math.factorial(k)
    return result


# =============================================================================
# ANSWER VALIDATION UTILITIES
# =============================================================================

def validate_positive_integer(ans: int, lower: int = 0, upper: int = 99999) -> bool:
    """Check if answer is in valid AIMO range."""
    return isinstance(ans, int) and lower <= ans <= upper


def check_divisibility(ans: int, divisor: int) -> bool:
    """Quick check if answer is divisible by given divisor."""
    return divisor != 0 and ans % divisor == 0


def check_modular_constraint(ans: int, remainder: int, modulus: int) -> bool:
    """Check if ans ≡ remainder (mod modulus)."""
    return modulus > 0 and ans % modulus == remainder


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    # Quick sanity checks
    assert catalan(0) == 1
    assert catalan(1) == 1
    assert catalan(5) == 42
    assert catalan(10) == 16796
    
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765
    
    assert euler_phi(1) == 1
    assert euler_phi(10) == 4
    assert euler_phi(12) == 4
    
    assert divisor_count(1) == 1
    assert divisor_count(12) == 6
    assert divisor_count(100) == 9
    
    assert sigma(12) == 28
    assert sigma(28) == 56  # 28 is perfect
    
    assert legendre_v_p(10, 2) == 8  # v_2(10!) = 8
    assert legendre_v_p(10, 5) == 2  # v_5(10!) = 2
    
    print("All math_tools tests passed!")
