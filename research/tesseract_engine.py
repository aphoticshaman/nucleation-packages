"""
T.E.S.S.E.R.A.C.T. ENGINE
Topological Entropy-based Semantic Synthesis & Error-correcting Recursive Architecture for Cognitive Tasks

Implements the Topological Consensus mechanism using:
1. NCD (Normalized Compression Distance) for semantic similarity
2. Simplified Persistent Homology (Hierarchical Clustering Lifetime)
3. CIC-informed scoring (Causal Integration Core functional)

References:
- Riedl & Weidmann (2025) "Quantifying Human-AI Synergy"
- NSM-XYZA-SDPM Framework v3.0

For AIMO3 Competition - Drop-in replacement for select_answer_prometheus
"""

import zlib
import math
import statistics
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CodeTrace:
    """Represents a single solution candidate with its code and output."""
    id: int
    code: str
    output: int
    strategy: str
    kolmogorov_weight: float = 1.0


# =============================================================================
# NCD (NORMALIZED COMPRESSION DISTANCE)
# =============================================================================

class NCDCalculator:
    """Calculates Normalized Compression Distance between code strings."""

    def __init__(self, compression_level: int = 9):
        self.level = compression_level
        self._cache: Dict[Tuple[str, str], float] = {}

    def _normalize(self, text: str) -> str:
        """Normalize text to focus on logic, not whitespace."""
        return ' '.join(text.split())

    def ncd(self, x: str, y: str) -> float:
        """
        Calculate NCD between two strings.

        NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

        Where C(x) is the compressed size of string x.
        """
        if x == y:
            return 0.0
        if not x or not y:
            return 1.0

        # Check cache
        cache_key = (x, y) if x < y else (y, x)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Normalize
        x_norm = self._normalize(x)
        y_norm = self._normalize(y)

        # Compress
        cx = len(zlib.compress(x_norm.encode('utf-8'), self.level))
        cy = len(zlib.compress(y_norm.encode('utf-8'), self.level))
        cxy = len(zlib.compress((x_norm + y_norm).encode('utf-8'), self.level))

        # NCD formula
        result = (cxy - min(cx, cy)) / max(cx, cy)

        # Cache result
        self._cache[cache_key] = result
        return result


# =============================================================================
# TESSERACT SELECTOR
# =============================================================================

class TesseractSelector:
    """
    Implements Topological Consensus mechanism.

    Selects answers based on the topological persistence of their
    derivation logic in the NCD metric space, rather than just
    the frequency of the scalar output.
    """

    def __init__(self, compression_level: int = 9):
        self.ncd_calc = NCDCalculator(compression_level)

        # Strategy priors (tune based on empirical performance)
        self.strategy_priors = {
            'pot_sympy': 1.3,
            'pot_algorithmic': 1.2,
            'pot_bruteforce': 0.9,
            'cot': 0.85,
            'prometheus': 1.0,
            'unknown': 1.0,
        }

    def compute_distance_matrix(self, traces: List[CodeTrace]) -> np.ndarray:
        """Compute NCD distance matrix between all traces."""
        n = len(traces)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = self.ncd_calc.ncd(traces[i].code, traces[j].code)
                matrix[i, j] = d
                matrix[j, i] = d

        return matrix

    def estimate_psi(self, values: List[int]) -> float:
        """
        Calculate Causal Emergence (Ψ) proxy.

        Theory: A valid macro-state (answer) should predict the micro-states
        better than micro predicts itself.

        Ψ = 1 - (Residual_Macro / Variance_Micro)
        """
        if len(values) < 3:
            return 0.1  # Insufficient data

        micro_var = np.var(values)
        if micro_var < 1e-9:
            return 1.0  # Perfect determinism

        macro_pred = int(statistics.median(values))
        residuals = np.var([v - macro_pred for v in values])

        psi = 1.0 - (residuals / micro_var)
        return max(0.0, min(1.0, psi))

    def persistence_analysis(self, traces: List[CodeTrace]) -> Dict:
        """
        Simulates 0-dimensional Persistent Homology via hierarchical clustering.

        Returns the cluster with the highest 'Structural Persistence'.
        """
        if not traces:
            return {"confidence": 0.0, "best_output": None}
        if len(traces) == 1:
            return {"confidence": 1.0, "best_output": traces[0].output}

        # 1. Compute NCD distance matrix
        dist_matrix = self.compute_distance_matrix(traces)

        # 2. Single-Linkage Clustering (Simulating Filtration)
        n = len(traces)
        thresholds = np.linspace(0.05, 0.95, 19)

        cluster_lifetimes = defaultdict(float)

        # Track clusters across filtration scales
        for epsilon in thresholds:
            # Build adjacency graph at this epsilon
            adj = dist_matrix < epsilon

            # Find connected components
            visited = set()
            components = []

            for i in range(n):
                if i not in visited:
                    stack = [i]
                    comp = []
                    while stack:
                        node = stack.pop()
                        if node not in visited:
                            visited.add(node)
                            comp.append(node)
                            neighbors = np.where(adj[node])[0]
                            stack.extend([nb for nb in neighbors if nb not in visited])
                    components.append(comp)

            # Analyze components at this scale
            for comp_indices in components:
                if not comp_indices:
                    continue

                # Get outputs and strategies in this component
                outputs = [traces[i].output for i in comp_indices]
                strategies = set(traces[i].strategy for i in comp_indices)

                # Find majority vote in this semantic cluster
                maj_output, maj_count = Counter(outputs).most_common(1)[0]

                # Calculate purity
                purity = maj_count / len(outputs)

                if purity > 0.6:  # Only count fairly pure clusters
                    # Weight by strategy diversity (Φ proxy)
                    strategy_weight = 1.0 + 0.3 * (len(strategies) - 1)

                    # Weight by strategy quality
                    strategy_bonus = sum(
                        self.strategy_priors.get(traces[i].strategy, 1.0)
                        for i in comp_indices
                    ) / len(comp_indices)

                    # Accumulate lifetime score
                    cluster_lifetimes[maj_output] += (
                        (1.0 / len(thresholds))
                        * len(comp_indices)
                        * purity
                        * strategy_weight
                        * strategy_bonus
                    )

        # 3. Select answer with max persistence + Ψ bonus
        if not cluster_lifetimes:
            # Fallback to simple mode
            outputs = [t.output for t in traces]
            return {
                "confidence": 0.1,
                "best_output": Counter(outputs).most_common(1)[0][0]
            }

        # Add Ψ (causal emergence) bonus to each candidate
        final_scores = {}
        for output, lifetime in cluster_lifetimes.items():
            # Get all traces with this output
            matching_traces = [t for t in traces if t.output == output]
            values = [t.output for t in matching_traces]
            psi = self.estimate_psi(values)

            # CIC-style scoring: Lifetime - λH + γΨ
            # Lifetime already captures Φ (integration)
            # We add Ψ bonus
            final_scores[output] = lifetime + 0.5 * psi * len(matching_traces)

        best_output = max(final_scores.items(), key=lambda x: x[1])[0]
        max_score = final_scores[best_output]

        # Normalize confidence
        confidence = min(0.95, max_score / len(traces))

        return {
            "confidence": confidence,
            "best_output": best_output,
            "persistence_score": cluster_lifetimes.get(best_output, 0),
            "psi_bonus": final_scores[best_output] - cluster_lifetimes.get(best_output, 0),
        }


# =============================================================================
# C.I.C.E.R.O. INTEGRATION
# =============================================================================

def benford_score(n: int) -> float:
    """Calculate Benford's Law score for answer 'niceness'."""
    if n < 0:
        n = -n
    if n == 0:
        return 1.0
    s = str(n)
    first = int(s[0])
    benford_prob = [0.0, 0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    return 1.0 + benford_prob[first]


def cic_score_basin(basin_val: int, members: List[CodeTrace]) -> float:
    """
    The CIC Functional: F = Φ - λH + γC

    Φ (Phi): Integrated Evidence (Mass × Kolmogorov × Strategy Diversity)
    H (Entropy): Internal Dispersion
    C (Causal): Ψ (causal emergence) + niceness priors
    """
    # 1. PHI (Integrated Evidence)
    mass = len(members)
    avg_kolmogorov = statistics.mean([c.kolmogorov_weight for c in members])
    strategies = set(c.strategy for c in members)
    diversity_boost = 1.2 if len(strategies) > 1 else 1.0
    if 'pot_sympy' in strategies and 'pot_bruteforce' in strategies:
        diversity_boost = 1.5  # High-value cross-verification

    Phi = mass * avg_kolmogorov * diversity_boost

    # 2. H (Entropy/Dispersion)
    values = [c.output for c in members]
    if len(values) <= 1:
        H = 0.0
    else:
        dists = []
        for v in values:
            if v == basin_val:
                d = 0.0
            elif v == 0 or basin_val == 0:
                d = 1.0
            else:
                d = abs(v - basin_val) / max(abs(v), abs(basin_val))
            dists.append(d)
        H = statistics.mean(dists)

    # 3. C (Causal Emergence / Ψ)
    Psi = estimate_psi_from_values(values)
    nice_bonus = benford_score(basin_val)

    # Constants
    LAMBDA = 0.8
    GAMMA = 1.5

    # Master Equation
    F = (math.log(1 + Phi) * nice_bonus) - (LAMBDA * H) + (GAMMA * Psi * (mass ** 0.5))

    return F


def estimate_psi_from_values(values: List[int]) -> float:
    """Standalone Ψ calculation from values."""
    if len(values) < 3:
        return 0.1
    micro_var = np.var(values)
    if micro_var < 1e-9:
        return 1.0
    macro_pred = int(statistics.median(values))
    residuals = np.var([v - macro_pred for v in values])
    return max(0.0, min(1.0, 1.0 - (residuals / micro_var)))


# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================

def apply_tesseract(candidates: List[Any], mod_target: Optional[int] = None) -> int:
    """
    Drop-in replacement for select_answer_prometheus.
    Uses topological consensus for selection.

    Args:
        candidates: List of AnswerCandidate objects
        mod_target: Optional modulo target for the answer

    Returns:
        Selected integer answer
    """
    # Filter valid candidates
    valid = [c for c in candidates if c.value is not None]
    if not valid:
        return 0

    # Apply modulo if specified
    if mod_target:
        for c in valid:
            c.value = c.value % mod_target

    # Convert to CodeTrace format
    traces = []
    for i, c in enumerate(valid):
        content = c.code if hasattr(c, 'code') and c.code else str(c.value)
        traces.append(CodeTrace(
            id=i,
            code=content,
            output=c.value,
            strategy=getattr(c, 'strategy', 'unknown'),
            kolmogorov_weight=getattr(c, 'kolmogorov_weight', 1.0),
        ))

    # Run Tesseract
    engine = TesseractSelector()
    result = engine.persistence_analysis(traces)

    best_val = result['best_output']

    # AIMO3 bounds
    if best_val is None:
        return 0
    return max(0, min(99999, best_val))


def select_answer_cic(candidates: List[Any], mod_target: Optional[int] = None) -> int:
    """
    C.I.C.E.R.O. selector - pure CIC functional without topological analysis.
    Lighter weight than full TESSERACT.

    Args:
        candidates: List of AnswerCandidate objects
        mod_target: Optional modulo target

    Returns:
        Selected integer answer
    """
    if not candidates:
        return 0

    # Apply modulo
    if mod_target:
        for c in candidates:
            c.value = c.value % mod_target

    # Convert to traces
    traces = []
    for i, c in enumerate(candidates):
        if c.value is None:
            continue
        content = c.code if hasattr(c, 'code') and c.code else str(c.value)
        traces.append(CodeTrace(
            id=i,
            code=content,
            output=c.value,
            strategy=getattr(c, 'strategy', 'unknown'),
            kolmogorov_weight=getattr(c, 'kolmogorov_weight', 1.0),
        ))

    if not traces:
        return 0

    # Form basins (cluster by proximity)
    basins = defaultdict(list)
    for t in traces:
        assigned = False
        for bval in list(basins.keys()):
            if bval == 0 and t.output == 0:
                basins[bval].append(t)
                assigned = True
                break
            elif bval != 0 and t.output != 0:
                rel_dist = abs(t.output - bval) / max(abs(t.output), abs(bval))
                if rel_dist < 0.01:  # 1% threshold
                    basins[bval].append(t)
                    assigned = True
                    break
        if not assigned:
            basins[t.output].append(t)

    # Score basins via CIC
    scored_basins = []
    for bval, members in basins.items():
        center = int(statistics.median([t.output for t in members]))
        score = cic_score_basin(center, members)
        scored_basins.append((score, center))

    scored_basins.sort(key=lambda x: -x[0])

    best_val = scored_basins[0][1] if scored_basins else 0

    return max(0, min(99999, best_val))


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create mock candidates
    @dataclass
    class MockCandidate:
        value: int
        code: str
        strategy: str
        kolmogorov_weight: float = 1.0

    candidates = [
        MockCandidate(42, "x = solve(eq, x)[0]; return 42", "pot_sympy", 1.2),
        MockCandidate(42, "for i in range(100): if check(i): return 42", "pot_bruteforce", 0.8),
        MockCandidate(56, "result = complex_derivation(); return 56", "pot_sympy", 1.3),
        MockCandidate(56, "answer = 56  # derived", "cot", 0.9),
        MockCandidate(42, "return 42  # guess", "cot", 0.5),
    ]

    print("Testing TESSERACT Engine...")
    result = apply_tesseract(candidates)
    print(f"TESSERACT selected: {result}")

    print("\nTesting C.I.C.E.R.O. Engine...")
    result = select_answer_cic(candidates)
    print(f"C.I.C.E.R.O. selected: {result}")
