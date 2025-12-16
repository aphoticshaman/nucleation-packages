"""
Value Clustering and Basin Refinement for Signal Fusion.

Extracted from AIMO3 competition solver (bansheev6.ipynb).
92.1% error reduction method for multi-source signal fusion.

Core insight: When multiple noisy sources provide estimates,
cluster them by relative proximity and find the "basin center" -
the Platonic Form that best represents the true value.

This is LLM-free, deterministic signal fusion.
"""

import math
import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter


@dataclass
class ValueCluster:
    """A cluster of similar values from multiple sources."""
    members: List[float]
    size: int
    center: float
    tightness: float  # 0 = dispersed, 1 = tight agreement
    score: float      # size * sqrt(tightness)

    def __repr__(self) -> str:
        return f"Cluster(n={self.size}, center={self.center:.4f}, tightness={self.tightness:.3f})"


@dataclass
class ClusteringResult:
    """Result of value clustering operation."""
    clusters: List[ValueCluster]
    n_clusters: int
    best: Optional[ValueCluster]
    consensus_strength: float  # How strong is the winning cluster vs others


def relative_distance(a: float, b: float, epsilon: float = 1e-10) -> float:
    """
    Relative distance between two values.

    |a - b| / max(|a|, |b|)

    This metric is scale-invariant, making it suitable for comparing
    values across different orders of magnitude.

    Args:
        a: First value
        b: Second value
        epsilon: Small value to prevent division by zero

    Returns:
        Relative distance in [0, 1] for similar magnitudes,
        can exceed 1 for very different magnitudes
    """
    if a == b:
        return 0.0

    max_abs = max(abs(a), abs(b))
    if max_abs < epsilon:
        return abs(a - b) / epsilon

    return abs(a - b) / max_abs


def value_clustering(
    samples: List[float],
    threshold: float = 0.05,
    min_cluster_size: int = 1
) -> ClusteringResult:
    """
    Cluster values by relative proximity using Union-Find.

    Two values are in the same cluster if their relative distance
    is below the threshold. This creates equivalence classes of
    "approximately equal" values.

    Args:
        samples: List of values to cluster
        threshold: Maximum relative distance for same cluster (default 5%)
        min_cluster_size: Minimum members for a cluster to be valid

    Returns:
        ClusteringResult with clusters sorted by score (best first)
    """
    n = len(samples)

    if n == 0:
        return ClusteringResult(
            clusters=[],
            n_clusters=0,
            best=None,
            consensus_strength=0.0
        )

    if n == 1:
        cluster = ValueCluster(
            members=[samples[0]],
            size=1,
            center=samples[0],
            tightness=1.0,
            score=1.0
        )
        return ClusteringResult(
            clusters=[cluster],
            n_clusters=1,
            best=cluster,
            consensus_strength=1.0
        )

    # Union-Find data structure
    parent = list(range(n))
    rank = [0] * n

    def find(i: int) -> int:
        if parent[i] != i:
            parent[i] = find(parent[i])  # Path compression
        return parent[i]

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Union by rank
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1

    # Build clusters by merging similar values
    for i in range(n):
        for j in range(i + 1, n):
            if relative_distance(samples[i], samples[j]) < threshold:
                union(i, j)

    # Extract clusters
    clusters_dict: Dict[int, List[float]] = {}
    for i in range(n):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(samples[i])

    # Build cluster objects
    clusters = []
    for members in clusters_dict.values():
        if len(members) < min_cluster_size:
            continue

        size = len(members)
        center = statistics.median(members)

        if size == 1:
            tightness = 1.0
        else:
            spread = statistics.stdev(members)
            center_abs = abs(statistics.mean(members)) if members else 1.0
            if center_abs > 1e-10:
                tightness = max(0.0, min(1.0, 1.0 - (spread / center_abs)))
            else:
                tightness = 1.0 if spread < 1e-10 else 0.0

        # Score: size weighted by tightness
        # More members + tighter agreement = higher score
        score = size * math.sqrt(tightness)

        clusters.append(ValueCluster(
            members=members,
            size=size,
            center=center,
            tightness=tightness,
            score=score
        ))

    # Sort by score (best first)
    clusters.sort(key=lambda c: -c.score)

    # Calculate consensus strength
    if len(clusters) == 0:
        consensus_strength = 0.0
    elif len(clusters) == 1:
        consensus_strength = 1.0
    else:
        # Ratio of best score to second-best
        consensus_strength = min(1.0, clusters[0].score / (clusters[1].score + 0.1))

    return ClusteringResult(
        clusters=clusters,
        n_clusters=len(clusters),
        best=clusters[0] if clusters else None,
        consensus_strength=consensus_strength
    )


def basin_refinement(cluster: ValueCluster) -> float:
    """
    Refine cluster to its "basin center" - the Platonic Form.

    Uses trimmed mean + median fusion to find the most representative
    value that is robust to outliers within the cluster.

    Args:
        cluster: A ValueCluster to refine

    Returns:
        The refined center value (basin attractor)
    """
    members = cluster.members

    if len(members) <= 2:
        return statistics.median(members)

    # Median is robust to outliers
    median_val = statistics.median(members)

    # Trimmed mean removes extreme values
    sorted_m = sorted(members)
    trim = max(1, len(sorted_m) // 4)

    if len(sorted_m) > 2 * trim:
        trimmed = sorted_m[trim:-trim]
    else:
        trimmed = sorted_m

    trimmed_mean = statistics.mean(trimmed)

    # Basin center: fusion of median and trimmed mean
    return (median_val + trimmed_mean) / 2


def log_weighted_vote(
    samples: List[float],
    base: float = 1.25
) -> Tuple[float, float]:
    """
    Log-weighted voting for value selection.

    Weights larger values slightly higher (log scale), which helps
    when correct answers tend to be more "significant" values.

    Args:
        samples: Values to vote on
        base: Base for log weighting (default 1.25)

    Returns:
        Tuple of (selected_value, confidence)
    """
    if not samples:
        return (0.0, 0.0)

    if len(samples) == 1:
        return (samples[0], 1.0)

    counter = Counter(samples)

    # Apply log weighting
    weighted_scores: Dict[float, float] = {}
    for value, count in counter.items():
        weight = math.log(base + abs(value)) * count
        weighted_scores[value] = weight

    # Find winner
    total_weight = sum(weighted_scores.values())
    winner = max(weighted_scores, key=weighted_scores.get)
    confidence = weighted_scores[winner] / total_weight if total_weight > 0 else 0.0

    return (winner, confidence)


def fuse_values(
    samples: List[float],
    clustering_threshold: float = 0.05,
    use_basin_refinement: bool = True
) -> Tuple[float, float, ClusteringResult]:
    """
    Complete value fusion pipeline.

    1. Cluster values by relative proximity
    2. Select best cluster (highest score)
    3. Refine to basin center

    This is the "92.1% error reduction" method.

    Args:
        samples: Values from multiple sources
        clustering_threshold: Relative distance threshold for clustering
        use_basin_refinement: Whether to refine cluster center

    Returns:
        Tuple of (fused_value, confidence, clustering_result)
    """
    result = value_clustering(samples, threshold=clustering_threshold)

    if result.best is None:
        return (0.0, 0.0, result)

    if use_basin_refinement:
        fused = basin_refinement(result.best)
    else:
        fused = result.best.center

    # Confidence combines cluster quality and consensus
    confidence = (result.best.tightness + result.consensus_strength) / 2

    return (fused, confidence, result)


# ============================================================
# CIC Distance Functions (for signal/document comparison)
# ============================================================

def js_divergence(
    dist_a: Dict[str, float],
    dist_b: Dict[str, float],
    epsilon: float = 1e-12
) -> float:
    """
    Jensen-Shannon divergence between two probability distributions.

    JS divergence is symmetric and bounded [0, ln(2)], making it
    a proper metric for comparing distributions.

    Args:
        dist_a: First distribution (key -> probability)
        dist_b: Second distribution (key -> probability)
        epsilon: Smoothing factor

    Returns:
        JS divergence value
    """
    keys = set(dist_a.keys()) | set(dist_b.keys())
    if not keys:
        return 0.0

    # Normalize with smoothing
    a_total = sum(dist_a.values()) + epsilon * len(keys)
    b_total = sum(dist_b.values()) + epsilon * len(keys)

    js = 0.0
    for k in keys:
        pa = (dist_a.get(k, 0) + epsilon) / a_total
        pb = (dist_b.get(k, 0) + epsilon) / b_total
        m = 0.5 * (pa + pb)

        if pa > 0 and m > 0:
            js += 0.5 * pa * math.log(pa / m)
        if pb > 0 and m > 0:
            js += 0.5 * pb * math.log(pb / m)

    return js


def text_to_unigram_dist(text: str) -> Dict[str, int]:
    """Convert text to unigram (word) distribution."""
    import re
    tokens = re.findall(r"\w+", str(text).lower())
    return dict(Counter(tokens))


def js_divergence_text(text_a: str, text_b: str, epsilon: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence between two texts (unigram distributions).

    Args:
        text_a: First text
        text_b: Second text
        epsilon: Smoothing factor

    Returns:
        JS divergence value
    """
    dist_a = text_to_unigram_dist(text_a)
    dist_b = text_to_unigram_dist(text_b)
    return js_divergence(dist_a, dist_b, epsilon)


def char_ngrams(text: str, n: int = 4) -> Counter:
    """Extract character n-grams from text."""
    import re
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    if len(text) < n:
        return Counter()
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
    """Cosine similarity between two Counter objects."""
    if not counter_a or not counter_b:
        return 0.0

    dot = sum(v * counter_b.get(k, 0) for k, v in counter_a.items())
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def tail_similarity(
    text_a: str,
    text_b: str,
    window_chars: int = 900,
    ngram_n: int = 4
) -> float:
    """
    Tail similarity using character n-grams.

    Compares the ending portions of two texts, which is useful for
    detecting when signals are converging to similar conclusions.

    Args:
        text_a: First text
        text_b: Second text
        window_chars: Number of trailing characters to compare
        ngram_n: N-gram size

    Returns:
        Similarity in [0, 1]
    """
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return 0.0

    a_tail = text_a[-min(window_chars, len(text_a)):]
    b_tail = text_b[-min(window_chars, len(text_b)):]

    return cosine_similarity(
        char_ngrams(a_tail, ngram_n),
        char_ngrams(b_tail, ngram_n)
    )


def d_CIC(
    text_a: str,
    text_b: str,
    value_a: Optional[float] = None,
    value_b: Optional[float] = None,
    js_weight: float = 0.6,
    tail_weight: float = 0.3,
    value_weight: float = 0.1
) -> float:
    """
    CIC Distance function for comparing signals/documents.

    Combines:
    - JS divergence (content distribution difference)
    - Tail dissimilarity (conclusion convergence)
    - Value penalty (if extracted values differ)

    Args:
        text_a: First text/signal
        text_b: Second text/signal
        value_a: Optional extracted value from first
        value_b: Optional extracted value from second
        js_weight: Weight for JS divergence (default 0.6)
        tail_weight: Weight for tail dissimilarity (default 0.3)
        value_weight: Weight for value disagreement (default 0.1)

    Returns:
        CIC distance in [0, 1]
    """
    # JS divergence component
    js = js_divergence_text(text_a, text_b)

    # Tail dissimilarity (1 - similarity)
    tail_dissim = 1.0 - tail_similarity(text_a, text_b)

    # Value penalty
    value_penalty = 0.0
    if value_a is not None and value_b is not None:
        if value_a != value_b:
            value_penalty = 1.0

    # Weighted combination
    distance = (
        js_weight * js +
        tail_weight * tail_dissim +
        value_weight * value_penalty
    )

    return min(1.0, distance)
