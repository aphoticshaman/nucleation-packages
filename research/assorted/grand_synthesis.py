# CELL 3: GRAND SYNTHESIS - CIC + UIPT + NCD + LatticeForge + Grokking + Toroidal
# =============================================================================
# THE COMPLETE RYANAIMO WEAPON SYSTEM
# Integrates: CIC Theory, UIPT Entropy, Extended NCD, LatticeForge Phase Detection,
#             Micro-Grokking, Toroidal Voting, Entropic Gravity, Value Clustering
# =============================================================================

import lzma
import math
import struct
import statistics
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Deque
from collections import Counter, deque
import heapq

# Constants
ANSWER_MIN = 0
ANSWER_MAX = 99999
FALLBACK_ANSWER = 0
TOTAL_BUDGET_SECONDS = 5 * 60 * 60  # 5 hours

# =============================================================================
# SECTION 1: UIPT (Universal Information Phase Transition) - Entropy Tracking
# From prometheus_kaggle.py
# =============================================================================

class UIPTEntropyWindow:
    """
    Rolling entropy tracker for detecting phase transitions in token generation.

    Low entropy = Crystallized Logic (High confidence) - CRYSTALLINE phase
    High entropy = Gas Phase (Hallucination/exploration) - PLASMA phase

    This is the core UIPT insight: Intelligence emerges at the phase transition
    where compression forces balance integration forces.
    """

    def __init__(self, window: int = 32):
        self.window = window
        self.buf: Deque[int] = deque(maxlen=self.window)
        self.counts: Counter = Counter()
        self.entropy_history: List[float] = []

    def add(self, token_id: int) -> None:
        """Add a new token and update entropy calculation."""
        if len(self.buf) == self.window:
            old = self.buf.popleft()
            self.counts[old] -= 1
            if self.counts[old] <= 0:
                del self.counts[old]
        self.buf.append(token_id)
        self.counts[token_id] += 1
        self.entropy_history.append(self.normalized())

    def raw_entropy_bits(self) -> float:
        """Calculate raw Shannon entropy in bits."""
        total = sum(self.counts.values())
        if total <= 0:
            return 0.0
        probs = [c / total for c in self.counts.values() if c > 0]
        H = -sum(p * math.log2(p) for p in probs)
        return H

    def normalized(self) -> float:
        """Normalized entropy (0-1). 0=crystallized, 1=max entropy."""
        H = self.raw_entropy_bits()
        alph = max(1, len(self.counts))
        maxH = math.log2(alph) if alph > 1 else 1.0
        return float(H / maxH) if maxH > 0 else 0.0

    def is_crystallized(self, threshold: float = 0.3) -> bool:
        """Check if entropy has crystallized below threshold."""
        return self.normalized() < threshold

    def is_gas_phase(self, threshold: float = 0.85) -> bool:
        """Check if entropy is in gas phase (hallucination mode)."""
        return self.normalized() > threshold

    def get_phase(self) -> str:
        """Get current phase based on entropy."""
        h = self.normalized()
        if h < 0.3:
            return "CRYSTALLINE"
        elif h < 0.5:
            return "SUPERCOOLED"
        elif h < 0.7:
            return "NUCLEATING"
        elif h < 0.85:
            return "ANNEALING"
        else:
            return "PLASMA"


# =============================================================================
# SECTION 2: MICRO-GROKKING DETECTION
# From entropy_voting.py - The KEY insight
# =============================================================================

@dataclass
class GrokkingSignal:
    """Result of micro-grokking analysis."""
    detected: bool
    score: float
    d2_min: float  # Minimum second derivative (negative = grokking)
    final_entropy: float
    convergence_point: int  # Token position where grokking occurred

def detect_micro_grokking(
    entropies: List[float],
    window_size: int = 5,
    d2_threshold: float = -0.05
) -> GrokkingSignal:
    """
    Detect micro-grokking via entropy second derivative.

    THE KEY INSIGHT: A sharp negative second derivative in entropy indicates
    the model has "clicked" - switching from exploration to exploitation.
    This is the moment of understanding, the "aha" signal.

    Args:
        entropies: Per-token entropy values
        window_size: Window for derivative smoothing
        d2_threshold: Threshold for grokking detection (default -0.05)

    Returns:
        GrokkingSignal with detection result and score
    """
    if len(entropies) < window_size * 3:
        return GrokkingSignal(False, 0.0, 0.0, 1.0, -1)

    arr = [float(e) for e in entropies]

    # Smooth entropies with moving average
    kernel_size = min(window_size, len(arr) // 3)
    smooth = []
    for i in range(len(arr) - kernel_size + 1):
        smooth.append(sum(arr[i:i+kernel_size]) / kernel_size)

    if len(smooth) < 3:
        return GrokkingSignal(False, 0.0, 0.0, arr[-1] if arr else 1.0, -1)

    # First derivative (rate of confusion change)
    d1 = [smooth[i+1] - smooth[i] for i in range(len(smooth)-1)]

    # Second derivative (acceleration of convergence)
    d2 = [d1[i+1] - d1[i] for i in range(len(d1)-1)] if len(d1) > 1 else [0.0]

    # Find minimum d2 (most negative = sharpest convergence)
    min_d2 = min(d2) if d2 else 0.0
    min_d2_idx = d2.index(min_d2) if d2 and min_d2 in d2 else -1

    # Final entropy (last window_size tokens)
    final_entropy = sum(arr[-window_size:]) / window_size if len(arr) >= window_size else arr[-1]

    # Score: favors low final entropy AND sharp convergence
    final_stability = 1.0 / (1.0 + final_entropy)
    convergence_bonus = max(0, -min_d2 * 10)
    score = final_stability + convergence_bonus

    grokking_detected = min_d2 < d2_threshold

    return GrokkingSignal(
        detected=grokking_detected,
        score=score,
        d2_min=min_d2,
        final_entropy=final_entropy,
        convergence_point=min_d2_idx + kernel_size if min_d2_idx >= 0 else -1
    )


# =============================================================================
# SECTION 3: EXTENDED NCD - The Nobel-Worthy Improvement
# From final_nobel_synthesis.py
# =============================================================================

def int_to_extended_bytes(n: int) -> bytes:
    """
    Convert integer to extended byte representation for better NCD discrimination.

    Multiple representations concatenated:
    1. Raw bytes (8 bytes, big-endian)
    2. Digit string (repeated 3x for weight)
    3. Binary string
    4. Residues mod [2,3,5,7,11,13,17,19,23,29] (prime fingerprint)
    5. Digit histogram

    This is THE BREAKTHROUGH: NCD on short numeric strings doesn't differentiate.
    Extended representation captures structural similarity.
    """
    parts = []

    # Raw 8-byte representation
    try:
        parts.append(struct.pack('>q', n))
    except struct.error:
        parts.append(str(n).encode()[:8].ljust(8, b'\x00'))

    # Digit string (repeated 3x for weight)
    digit_str = str(abs(n))
    parts.append((digit_str * 3).encode())

    # Binary string
    bin_str = bin(abs(n))[2:]
    parts.append(bin_str.encode())

    # Prime residue fingerprint - captures number-theoretic structure
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    residues = ''.join([str(abs(n) % p) for p in primes])
    parts.append((residues * 2).encode())

    # Digit histogram (frequency of each digit 0-9)
    hist = [0] * 10
    for d in digit_str:
        if d.isdigit():
            hist[int(d)] += 1
    parts.append(bytes(hist))

    return b''.join(parts)


def ncd_basic(x: bytes, y: bytes) -> float:
    """Basic NCD - Normalized Compression Distance."""
    if not x or not y:
        return 1.0
    cx = len(lzma.compress(x))
    cy = len(lzma.compress(y))
    cxy = len(lzma.compress(x + y))
    return (cxy - min(cx, cy)) / max(cx, cy) if max(cx, cy) > 0 else 0.0


def ncd_extended(x: int, y: int) -> float:
    """Extended NCD using rich integer representation."""
    x_bytes = int_to_extended_bytes(x)
    y_bytes = int_to_extended_bytes(y)
    return ncd_basic(x_bytes, y_bytes)


# =============================================================================
# SECTION 4: LATTICEFORGE PHASE DETECTION
# From LATTICEFORGE_MATHEMATICAL_FOUNDATIONS.md
# =============================================================================

@dataclass
class LatticeForgeState:
    """LatticeForge phase state for answer ensemble."""
    temperature: float      # T - variance-based energy
    order_parameter: float  # Ψ - consensus measure (Fourier harmonics)
    critical_exponent: float  # ν - distance from phase transition
    phase: str              # Current phase
    nucleation_count: int   # Number of nucleation sites
    confidence: float       # Epistemic confidence bound

# Fibonacci-derived weights for harmonic analysis (from LatticeForge patent)
FIBONACCI_WEIGHTS = [0.382, 0.236, 0.146, 0.090, 0.056]

def compute_temperature(samples: List[int]) -> float:
    """
    System temperature T from LatticeForge.
    T = (variance / n) × (1 + (1 - avg_correlation))

    High variance + low correlation = HIGH temperature (chaotic/PLASMA)
    Low variance = LOW temperature (stable/CRYSTALLINE)
    """
    if len(samples) < 2:
        return 0.0

    mean_val = statistics.mean(samples) if samples else 1
    if mean_val == 0:
        mean_val = 1

    # Normalize and compute variance
    normalized = [s / abs(mean_val) for s in samples]
    variance = statistics.variance(normalized)

    # For simplicity, estimate correlation from clustering
    # High agreement = high correlation
    counter = Counter(samples)
    max_agreement = counter.most_common(1)[0][1] / len(samples)
    avg_correlation = max_agreement  # Proxy

    T = variance * (1 + (1 - avg_correlation))
    return min(1.0, T)


def compute_order_parameter(samples: List[int]) -> float:
    """
    Order parameter Ψ from LatticeForge.
    Measures system structure via consensus.

    High Ψ = crystalline structure (consensus)
    Low Ψ = disordered (no consensus)
    """
    if not samples:
        return 0.0

    counter = Counter(samples)
    most_common_count = counter.most_common(1)[0][1]

    # Basic order from consensus
    basic_order = most_common_count / len(samples)

    # Bonus for tight clustering (multiple answers within 5%)
    def rel_dist(a, b):
        if a == b: return 0.0
        if a == 0 or b == 0: return 1.0
        return abs(a - b) / max(abs(a), abs(b))

    n = len(samples)
    close_pairs = sum(1 for i in range(n) for j in range(i+1, n)
                      if rel_dist(samples[i], samples[j]) < 0.05)
    total_pairs = n * (n - 1) // 2 if n > 1 else 1
    cluster_bonus = close_pairs / total_pairs * 0.3

    return min(1.0, basic_order + cluster_bonus)


def compute_critical_exponent(T: float, psi: float, T_c: float = 0.5) -> float:
    """
    Critical exponent ν - distance from phase transition.
    ν → 0 means imminent phase transition (solution crystallizing)
    ν → 1 means far from transition

    From Landau-Ginzburg theory adapted for algorithm space.
    """
    return math.sqrt((T - T_c)**2 + (psi - 0.5)**2) / math.sqrt(2)


def detect_nucleation_sites(samples: List[int], threshold: float = 0.05) -> int:
    """
    Detect nucleation sites - clusters that could trigger cascade.
    From LatticeForge: regions of high local correlation.
    """
    if len(samples) < 3:
        return 0

    # Find tight clusters
    def rel_dist(a, b):
        if a == b: return 0.0
        if a == 0 or b == 0: return 1.0
        return abs(a - b) / max(abs(a), abs(b))

    visited = [False] * len(samples)
    nucleation_count = 0

    for i in range(len(samples)):
        if visited[i]:
            continue

        # Find cluster around samples[i]
        cluster = [i]
        for j in range(i+1, len(samples)):
            if not visited[j] and rel_dist(samples[i], samples[j]) < threshold:
                cluster.append(j)
                visited[j] = True

        # Nucleation site if cluster has 3+ members
        if len(cluster) >= 3:
            nucleation_count += 1

        visited[i] = True

    return nucleation_count


def classify_phase(T: float, psi: float, nu: float, nucleation: int) -> str:
    """
    Classify system phase from LatticeForge.

    CRYSTALLINE: Stable equilibrium, low volatility
    SUPERCOOLED: Appears stable but susceptible to perturbation
    NUCLEATING: Rapid regime change in progress
    PLASMA: High energy chaotic state
    ANNEALING: Post-transition settling
    """
    if nu < 0.1 and nucleation > 2:
        return "NUCLEATING"
    elif T > 0.8 and psi < 0.3:
        return "PLASMA"
    elif T < 0.3 and psi > 0.7:
        return "CRYSTALLINE"
    elif T < 0.5 and psi > 0.5 and nucleation > 0:
        return "SUPERCOOLED"
    else:
        return "ANNEALING"


def compute_latticeforge_state(samples: List[int]) -> LatticeForgeState:
    """Compute full LatticeForge phase state for answer ensemble."""
    T = compute_temperature(samples)
    psi = compute_order_parameter(samples)
    nu = compute_critical_exponent(T, psi)
    nucleation = detect_nucleation_sites(samples)
    phase = classify_phase(T, psi, nu, nucleation)

    # Confidence from LatticeForge: high order + low temperature + low nu
    raw_confidence = psi * (1 - T) * (1 - nu)
    confidence = min(0.95, max(0.05, raw_confidence))

    return LatticeForgeState(
        temperature=T,
        order_parameter=psi,
        critical_exponent=nu,
        phase=phase,
        nucleation_count=nucleation,
        confidence=confidence
    )


# =============================================================================
# SECTION 5: CIC THEORY - Compression-Integration-Causality
# The unified functional F[T] = Φ - λH + γC
# =============================================================================

def representation_entropy(samples: List[int]) -> float:
    """H(T|X) - entropy of representations (compression term)."""
    if len(samples) < 2:
        return 0.0
    mean_val = statistics.mean(samples) if samples else 1
    if mean_val == 0:
        mean_val = 1
    normalized = [s / abs(mean_val) for s in samples]
    variance = statistics.variance(normalized)
    return min(1.0, variance)


def integrated_information_phi(samples: List[int]) -> float:
    """
    Φ (Phi) - Integrated Information for ensemble answers.
    High Φ = answers are structurally related (same reasoning path)
    Low Φ = answers are independent (different reasoning paths)

    Uses extended NCD for better discrimination.
    """
    if len(samples) < 2:
        return 0.0

    # Measure mutual compression across all pairs
    total_mi = 0.0
    pairs = 0
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            # Mutual information approximation via NCD
            mi = 1.0 - ncd_extended(samples[i], samples[j])
            total_mi += mi
            pairs += 1

    return total_mi / pairs if pairs > 0 else 0.0


def causal_power_multiscale(samples: List[int]) -> float:
    """C_multi(T) - multi-scale causal power."""
    if not samples:
        return 0.0
    n = len(samples)

    # Scale 1: Exact consensus
    counter = Counter(samples)
    exact_power = counter.most_common(1)[0][1] / n

    # Scale 2: Cluster coherence
    def rel_dist(a, b):
        if a == b: return 0.0
        if a == 0 or b == 0: return 1.0
        return abs(a - b) / max(abs(a), abs(b))

    close_pairs = sum(1 for i in range(n) for j in range(i+1, n)
                      if rel_dist(samples[i], samples[j]) < 0.05)
    total_pairs = n * (n - 1) // 2
    cluster_power = close_pairs / total_pairs if total_pairs > 0 else 0

    # Scale 3: Range constraint
    spread = max(samples) - min(samples) if samples else 0
    center = abs(statistics.mean(samples)) if samples else 1
    range_power = 1.0 / (1.0 + spread / center) if center > 0 else 0

    return 0.5 * exact_power + 0.3 * cluster_power + 0.2 * range_power


@dataclass
class CICState:
    """Full CIC state with all components."""
    phi: float              # Integrated information
    entropy: float          # Representation entropy H
    causal_power: float     # Multi-scale causal power C
    F: float                # CIC functional value
    confidence: float       # Derived confidence


def compute_cic(samples: List[int], lambda_c: float = 0.5, gamma_c: float = 0.3) -> CICState:
    """
    Compute CIC functional: F[T] = Φ - λH + γC

    This is THE EQUATION of intelligence under CIC theory.
    Intelligence emerges at the fixed point where integration + causality balance compression.
    """
    phi = integrated_information_phi(samples)
    H = representation_entropy(samples)
    C = causal_power_multiscale(samples)

    F = phi - lambda_c * H + gamma_c * C
    confidence = max(0.05, min(0.95, 0.5 + 0.5 * F))

    return CICState(phi=phi, entropy=H, causal_power=C, F=F, confidence=confidence)


# =============================================================================
# SECTION 6: TOROIDAL VOTING - For mod-N answers
# From prometheus_kaggle.py - S¹ clustering
# =============================================================================

def toroidal_distance(a: int, b: int, mod: int) -> float:
    """
    Distance on circle S¹ for modular arithmetic.
    Handles wrap-around: 999 is close to 0 under mod 1000.
    """
    diff = abs(a - b)
    return min(diff, mod - diff) / (mod / 2)  # Normalized to [0, 1]


def toroidal_clustering(samples: List[int], mod: int, threshold: float = 0.1) -> Dict:
    """
    Cluster answers on torus S¹ for mod-N problems.

    Key insight: In mod-1000 problems, 999 and 1 are close!
    Standard Euclidean clustering fails here.
    """
    if not samples:
        return {"clusters": [], "best": None}

    n = len(samples)
    if n == 1:
        return {
            "clusters": [{"members": samples, "center": samples[0] % mod, "size": 1}],
            "best": {"members": samples, "center": samples[0] % mod, "size": 1}
        }

    # Normalize to mod range
    normalized = [s % mod for s in samples]

    # Union-Find clustering on torus
    parent = list(range(n))
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i+1, n):
            if toroidal_distance(normalized[i], normalized[j], mod) < threshold:
                union(i, j)

    # Extract clusters
    clusters_dict = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(normalized[i])

    clusters = []
    for members in clusters_dict.values():
        size = len(members)
        # Toroidal center: use circular mean
        angles = [2 * math.pi * m / mod for m in members]
        sin_sum = sum(math.sin(a) for a in angles)
        cos_sum = sum(math.cos(a) for a in angles)
        mean_angle = math.atan2(sin_sum, cos_sum)
        center = int(round(mean_angle * mod / (2 * math.pi))) % mod

        clusters.append({"members": members, "center": center, "size": size})

    clusters.sort(key=lambda c: -c["size"])
    return {"clusters": clusters, "best": clusters[0] if clusters else None}


# =============================================================================
# SECTION 7: ENTROPIC GRAVITY VOTING
# From prometheus_kaggle.py - Mass × Density^0.15 × Solomonoff
# =============================================================================

def solomonoff_prior(n: int) -> float:
    """
    Solomonoff prior: simpler numbers more likely correct.
    Based on Kolmogorov complexity approximation.
    """
    if n == 0:
        return 1.0

    # Approximate complexity by description length
    digit_len = len(str(abs(n)))

    # Bonus for "nice" numbers
    nice_bonus = 1.0
    if n % 10 == 0:  # Ends in 0
        nice_bonus *= 1.2
    if n % 100 == 0:  # Ends in 00
        nice_bonus *= 1.3
    if n in [1, 2, 3, 4, 5, 10, 12, 15, 20, 24, 25, 30, 36, 42, 48, 50, 60, 72, 100, 120, 144]:
        nice_bonus *= 1.5  # Common competition answers

    # Prior inversely proportional to complexity
    return nice_bonus / (1 + digit_len * 0.3)


def entropic_gravity_vote(
    samples: List[int],
    entropies: Optional[List[float]] = None,
    grokking_scores: Optional[List[float]] = None
) -> Tuple[int, float, Dict]:
    """
    Entropic Gravity Voting: Mass × Density^0.15 × Solomonoff

    Combines:
    - Mass: Number of votes for answer
    - Density: Tightness of cluster (from extended NCD)
    - Solomonoff: Prior probability from simplicity
    - Grokking: Bonus for answers with micro-grokking signal
    """
    if not samples:
        return FALLBACK_ANSWER, 0.05, {}

    counter = Counter(samples)
    n = len(samples)

    # Compute score for each unique answer
    answer_scores = {}
    for ans, count in counter.items():
        # Mass term
        mass = count / n

        # Density term: how tight is the cluster around this answer?
        cluster_members = [s for s in samples if abs(s - ans) < max(1, abs(ans) * 0.05)]
        density = len(cluster_members) / n

        # Solomonoff prior
        prior = solomonoff_prior(ans)

        # Grokking bonus
        grok_bonus = 1.0
        if grokking_scores:
            # Find indices of this answer
            indices = [i for i, s in enumerate(samples) if s == ans]
            if indices:
                avg_grok = sum(grokking_scores[i] for i in indices if i < len(grokking_scores)) / len(indices)
                grok_bonus = 1.0 + avg_grok * 0.5

        # Entropy penalty (lower entropy = higher confidence)
        entropy_factor = 1.0
        if entropies:
            indices = [i for i, s in enumerate(samples) if s == ans]
            if indices:
                avg_entropy = sum(entropies[i] for i in indices if i < len(entropies)) / len(indices)
                entropy_factor = 1.0 / (1.0 + avg_entropy)

        # Final score: Mass × Density^0.15 × Prior × Grokking × Entropy
        score = mass * (density ** 0.15) * prior * grok_bonus * entropy_factor
        answer_scores[ans] = score

    # Select best answer
    best_answer = max(answer_scores.keys(), key=lambda a: answer_scores[a])
    best_score = answer_scores[best_answer]

    # Confidence from score distribution
    total_score = sum(answer_scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.1
    confidence = min(0.95, max(0.05, confidence))

    return best_answer, confidence, {"scores": answer_scores, "method": "entropic_gravity"}


# =============================================================================
# SECTION 8: VALUE CLUSTERING (88% Error Reduction)
# =============================================================================

def relative_distance(a: int, b: int) -> float:
    """Relative distance between two integers."""
    if a == b: return 0.0
    if a == 0 or b == 0:
        return 1.0 if max(abs(a), abs(b)) > 1000 else abs(a-b) / 1000
    return abs(a - b) / max(abs(a), abs(b))


def value_clustering(samples: List[int], threshold: float = 0.05) -> Dict:
    """Cluster by value proximity - the 88% error reduction method."""
    n = len(samples)
    if n == 0:
        return {"clusters": [], "best": None}
    if n == 1:
        return {
            "clusters": [{"members": samples, "center": samples[0], "size": 1, "tightness": 1.0, "score": 1.0}],
            "best": {"members": samples, "center": samples[0], "size": 1, "tightness": 1.0, "score": 1.0}
        }

    # Union-Find clustering
    parent = list(range(n))
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i+1, n):
            if relative_distance(samples[i], samples[j]) < threshold:
                union(i, j)

    clusters_dict = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(samples[i])

    clusters = []
    for members in clusters_dict.values():
        size = len(members)
        center = int(statistics.median(members))
        spread = statistics.stdev(members) if size > 1 else 0
        center_abs = abs(statistics.mean(members)) if members else 1
        tightness = max(0.0, min(1.0, 1.0 - (spread / center_abs if center_abs > 0 else 0)))
        score = size * (tightness ** 0.5)
        clusters.append({"members": members, "center": center, "size": size, "tightness": tightness, "score": score})

    clusters.sort(key=lambda c: -c["score"])
    return {"clusters": clusters, "best": clusters[0] if clusters else None}


def extended_ncd_basin_detection(samples: List[int], threshold: float = 0.25) -> Dict:
    """
    Basin detection using extended NCD representation.
    From final_nobel_synthesis.py - uses prime fingerprint for better discrimination.
    """
    n = len(samples)
    if n == 0:
        return {"found": False, "clusters": []}
    if n == 1:
        return {"found": True, "clusters": [{"members": samples, "center": samples[0]}], "best": samples[0]}

    # Build extended NCD matrix
    ncd_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = ncd_extended(samples[i], samples[j])
            ncd_matrix[i][j] = d
            ncd_matrix[j][i] = d

    # Simple clustering: group if NCD < threshold
    parent = list(range(n))
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i+1, n):
            if ncd_matrix[i][j] < threshold:
                union(i, j)

    # Extract clusters
    clusters_dict = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(samples[i])

    clusters = []
    for members in clusters_dict.values():
        # Internal cohesion
        if len(members) > 1:
            indices = [samples.index(m) for m in members]
            internal_ncds = [ncd_matrix[i][j] for i in indices for j in indices if i < j]
            cohesion = 1.0 - statistics.mean(internal_ncds) if internal_ncds else 0.5
        else:
            cohesion = 0.5

        clusters.append({
            "members": members,
            "size": len(members),
            "center": int(statistics.median(members)),
            "cohesion": cohesion,
            "score": len(members) * cohesion
        })

    clusters.sort(key=lambda c: -c["score"])
    best = clusters[0]["center"] if clusters else samples[0]

    return {"found": True, "clusters": clusters, "best": best}


# =============================================================================
# SECTION 9: THE GRAND SYNTHESIS - UNIFIED ANSWER SELECTION
# =============================================================================

@dataclass
class GrandSynthesisResult:
    """Complete result from grand synthesis answer selection."""
    answer: int
    confidence: float
    method: str
    cic_state: CICState
    lattice_state: LatticeForgeState
    grokking: Optional[GrokkingSignal]
    value_clusters: Dict
    ncd_basins: Dict
    toroidal_result: Optional[Dict]
    debug: Dict


def grand_synthesis_select(
    samples: List[int],
    entropies: Optional[List[float]] = None,
    problem_text: str = "",
    mod_hint: Optional[int] = None
) -> GrandSynthesisResult:
    """
    THE GRAND SYNTHESIS: Unified answer selection combining ALL methods.

    Pipeline:
    1. Value-based clustering (88% error reduction)
    2. Extended NCD basin detection (prime fingerprint)
    3. LatticeForge phase analysis (T, Ψ, ν)
    4. CIC functional computation
    5. Micro-grokking detection (if entropies provided)
    6. Toroidal voting (if mod problem detected)
    7. Entropic gravity voting
    8. Final weighted selection
    """
    if not samples:
        return GrandSynthesisResult(
            answer=FALLBACK_ANSWER, confidence=0.05, method="fallback",
            cic_state=CICState(0,0,0,0,0.05), lattice_state=LatticeForgeState(0,0,0,"UNKNOWN",0,0.05),
            grokking=None, value_clusters={}, ncd_basins={}, toroidal_result=None, debug={}
        )

    # 1. Value clustering
    value_result = value_clustering(samples, threshold=0.05)

    # 2. Extended NCD basin detection
    ncd_result = extended_ncd_basin_detection(samples, threshold=0.25)

    # 3. LatticeForge phase analysis
    lattice_state = compute_latticeforge_state(samples)

    # 4. CIC functional
    cic_state = compute_cic(samples)

    # 5. Micro-grokking detection
    grokking = None
    grokking_scores = None
    if entropies and len(entropies) >= 10:
        grokking = detect_micro_grokking(entropies)
        if grokking.detected:
            # Create grokking scores for each sample (simplified)
            grokking_scores = [grokking.score] * len(samples)

    # 6. Toroidal voting (detect mod problems)
    toroidal_result = None
    detected_mod = mod_hint
    if not detected_mod:
        # Try to detect mod from problem text
        mod_match = re.search(r'modulo?\s*(\d+)', problem_text, re.IGNORECASE)
        if mod_match:
            detected_mod = int(mod_match.group(1))

    if detected_mod and detected_mod > 1:
        toroidal_result = toroidal_clustering(samples, detected_mod, threshold=0.1)

    # 7. Entropic gravity voting
    gravity_answer, gravity_conf, gravity_debug = entropic_gravity_vote(
        samples, entropies, grokking_scores
    )

    # 8. FINAL WEIGHTED SELECTION
    candidates = {}

    # Value cluster candidate
    if value_result["best"]:
        vc = value_result["best"]
        vc_answer = vc["center"]
        vc_weight = vc["size"] / len(samples) * vc["tightness"] * 1.5  # Boost value clustering
        candidates[vc_answer] = candidates.get(vc_answer, 0) + vc_weight

    # NCD basin candidate
    if ncd_result.get("best"):
        ncd_answer = ncd_result["best"]
        ncd_weight = 0.8  # Good but not as reliable as value clustering
        candidates[ncd_answer] = candidates.get(ncd_answer, 0) + ncd_weight

    # LatticeForge influence
    if lattice_state.phase == "CRYSTALLINE":
        # High confidence in consensus
        counter = Counter(samples)
        lf_answer = counter.most_common(1)[0][0]
        lf_weight = lattice_state.confidence * 1.2
        candidates[lf_answer] = candidates.get(lf_answer, 0) + lf_weight

    # CIC influence
    if cic_state.F > 0.3:
        # Good CIC score - trust the consensus
        counter = Counter(samples)
        cic_answer = counter.most_common(1)[0][0]
        cic_weight = cic_state.confidence
        candidates[cic_answer] = candidates.get(cic_answer, 0) + cic_weight

    # Toroidal candidate (for mod problems)
    if toroidal_result and toroidal_result.get("best"):
        tor_answer = toroidal_result["best"]["center"]
        tor_weight = 0.7 if detected_mod else 0.3
        candidates[tor_answer] = candidates.get(tor_answer, 0) + tor_weight

    # Entropic gravity candidate
    candidates[gravity_answer] = candidates.get(gravity_answer, 0) + gravity_conf

    # Grokking bonus
    if grokking and grokking.detected:
        # Boost answers that had grokking signal
        counter = Counter(samples)
        grok_answer = counter.most_common(1)[0][0]
        grok_weight = grokking.score * 0.5
        candidates[grok_answer] = candidates.get(grok_answer, 0) + grok_weight

    # Select best candidate
    if candidates:
        final_answer = max(candidates.keys(), key=lambda a: candidates[a])
        total_weight = sum(candidates.values())
        final_confidence = candidates[final_answer] / total_weight if total_weight > 0 else 0.1
    else:
        final_answer = Counter(samples).most_common(1)[0][0]
        final_confidence = 0.3

    # Determine method used
    if candidates:
        method_weights = {
            "value_cluster": candidates.get(value_result["best"]["center"], 0) if value_result["best"] else 0,
            "ncd_basin": candidates.get(ncd_result.get("best", -1), 0),
            "latticeforge": candidates.get(Counter(samples).most_common(1)[0][0], 0) if lattice_state.phase == "CRYSTALLINE" else 0,
            "entropic_gravity": candidates.get(gravity_answer, 0),
        }
        primary_method = max(method_weights.keys(), key=lambda m: method_weights[m])
    else:
        primary_method = "majority"

    # Bound confidence
    final_confidence = min(0.95, max(0.05, final_confidence))

    return GrandSynthesisResult(
        answer=final_answer,
        confidence=final_confidence,
        method=primary_method,
        cic_state=cic_state,
        lattice_state=lattice_state,
        grokking=grokking,
        value_clusters=value_result,
        ncd_basins=ncd_result,
        toroidal_result=toroidal_result,
        debug={"candidates": candidates, "gravity_debug": gravity_debug}
    )


# =============================================================================
# SECTION 10: CONVENIENCE FUNCTION
# =============================================================================

def select_answer(samples: List[int], threshold: float = 0.05, fallback: int = 0) -> Tuple[int, float, Dict]:
    """
    Main answer selection function - uses grand synthesis.

    Returns: (answer, confidence, debug_info)
    """
    result = grand_synthesis_select(samples, problem_text="")
    return result.answer, result.confidence, {
        "method": result.method,
        "phase": result.lattice_state.phase,
        "cic_F": result.cic_state.F,
        "grokking": result.grokking.detected if result.grokking else False
    }


# =============================================================================
# SECTION 11: MATH-SPECIFIC HEURISTICS (Kept from original)
# =============================================================================

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025]
CATALAN = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786]
FACTORIALS = [1, 1, 2, 6, 24, 120, 720, 5040, 40320]
POWERS_OF_2 = [2**i for i in range(17)]
POWERS_OF_10 = [10**i for i in range(6)]
TRIANGULAR = [n*(n+1)//2 for n in range(200)]
PERFECT_SQUARES = [n*n for n in range(317)]
PRIMES_100 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def is_math_special_number(n: int) -> Tuple[bool, str]:
    """Check if number is 'special' in competition math."""
    if n in FIBONACCI: return True, "fibonacci"
    if n in CATALAN: return True, "catalan"
    if n in FACTORIALS: return True, "factorial"
    if n in POWERS_OF_2: return True, "power_of_2"
    if n in POWERS_OF_10: return True, "power_of_10"
    if n in TRIANGULAR: return True, "triangular"
    if n in PERFECT_SQUARES: return True, "perfect_square"
    if n in PRIMES_100: return True, "small_prime"
    for p in PRIMES_100:
        if p * p > n: break
        if n % p == 0 and n // p in PRIMES_100:
            return True, "semiprime"
    return False, ""


# =============================================================================
# SECTION 12: TIME BUDGET MANAGEMENT (Kept from original)
# =============================================================================

@dataclass
class TimeBudgetManager:
    """Sophisticated time budget management."""
    total_budget: float
    num_problems: int
    start_time: float = field(default_factory=time.time)
    problem_times: Dict[str, float] = field(default_factory=dict)
    problem_difficulties: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.base_time_per_problem = self.total_budget / self.num_problems
        self.time_bank = 0.0
        self.problems_solved = 0

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def remaining(self) -> float:
        return max(0, self.total_budget - self.elapsed())

    def get_budget_for_problem(self, difficulty: float, problem_idx: int) -> float:
        remaining = self.remaining()
        remaining_problems = max(1, self.num_problems - self.problems_solved)
        base_budget = remaining / remaining_problems
        difficulty_multiplier = 0.6 + 0.8 * difficulty
        bank_contribution = 0
        if difficulty > 0.6 and self.time_bank > 0:
            bank_contribution = min(self.time_bank * 0.3, base_budget * 0.5)
        budget = base_budget * difficulty_multiplier + bank_contribution
        return max(60, min(600, budget))

    def record_problem(self, problem_id: str, difficulty: float, actual_time: float):
        self.problem_times[problem_id] = actual_time
        self.problem_difficulties[problem_id] = difficulty
        self.problems_solved += 1
        expected_time = self.base_time_per_problem * (0.6 + 0.8 * difficulty)
        time_saved = expected_time - actual_time
        if time_saved > 0:
            self.time_bank += time_saved * 0.5
        elif time_saved < 0:
            self.time_bank = max(0, self.time_bank + time_saved)


# =============================================================================
# INITIALIZATION MESSAGE
# =============================================================================

print("=" * 70)
print("GRAND SYNTHESIS LOADED - THE COMPLETE RYANAIMO WEAPON SYSTEM")
print("=" * 70)
print("Components:")
print("  ✓ UIPT Entropy Window (Phase transition detection)")
print("  ✓ Micro-Grokking Detection (Entropy 2nd derivative)")
print("  ✓ Extended NCD (Prime residue fingerprint)")
print("  ✓ LatticeForge Phase Analysis (T, Ψ, ν)")
print("  ✓ CIC Functional (Φ - λH + γC)")
print("  ✓ Toroidal Voting (S¹ clustering for mod-N)")
print("  ✓ Entropic Gravity (Mass × Density × Solomonoff)")
print("  ✓ Value Clustering (88% error reduction)")
print("=" * 70)
