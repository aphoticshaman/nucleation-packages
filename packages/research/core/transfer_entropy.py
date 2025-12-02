"""
Transfer Entropy computation for causal graph construction.
Based on arXiv:2312.09478 and US5857978A (Lockheed, expired 2011).

Transfer entropy measures directed information flow from source X to target Y,
quantifying how much knowing X's past reduces uncertainty about Y's future
beyond what Y's own past provides.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from scipy.special import digamma
from scipy.spatial import KDTree
from dataclasses import dataclass

from .types import CausalGraph, SourceSignal


@dataclass
class TransferEntropyConfig:
    """Configuration for transfer entropy estimation."""
    k_neighbors: int = 4  # k for k-NN estimator (Kraskov et al.)
    lag_x: int = 1  # History length for source
    lag_y: int = 1  # History length for target
    threshold: float = 0.0  # Minimum TE to include edge
    normalize: bool = True  # Normalize by target entropy


def _embed_time_series(
    x: NDArray[np.float64],
    lag: int
) -> NDArray[np.float64]:
    """
    Create delay embedding of time series.

    Args:
        x: Time series of shape (T,) or (T, D)
        lag: Number of lags to include

    Returns:
        Embedded series of shape (T - lag, lag * D)
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    T, D = x.shape
    embedded = np.zeros((T - lag, lag * D))
    for i in range(lag):
        embedded[:, i*D:(i+1)*D] = x[i:T-lag+i]
    return embedded


def _kraskov_mi(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int = 4
) -> float:
    """
    Kraskov-Stögbauer-Grassberger mutual information estimator.
    Based on k-nearest neighbor distances.

    From US5857978A: Maximum-likelihood information-theoretic estimator
    using local density estimation.

    Args:
        x: First variable, shape (N, Dx)
        y: Second variable, shape (N, Dy)
        k: Number of neighbors

    Returns:
        Estimated mutual information in nats
    """
    N = len(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Joint space
    xy = np.hstack([x, y])

    # Build k-d trees
    tree_xy = KDTree(xy)
    tree_x = KDTree(x)
    tree_y = KDTree(y)

    # Find k-th neighbor distances in joint space
    # Using Chebyshev (max) norm
    dists_xy, _ = tree_xy.query(xy, k=k+1, p=np.inf)
    eps = dists_xy[:, -1]  # k-th neighbor distance

    # Count neighbors within eps in marginal spaces
    n_x = np.array([len(tree_x.query_ball_point(x[i], eps[i], p=np.inf)) - 1
                    for i in range(N)])
    n_y = np.array([len(tree_y.query_ball_point(y[i], eps[i], p=np.inf)) - 1
                    for i in range(N)])

    # KSG estimator (Equation from Kraskov et al. 2004)
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)

    return max(0.0, mi)  # MI is non-negative


def transfer_entropy(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    config: TransferEntropyConfig = TransferEntropyConfig()
) -> float:
    """
    Compute transfer entropy from source to target.

    TE_{X→Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})

    This measures the reduction in uncertainty about Y's future when
    we also know X's past, beyond what Y's own past provides.

    Based on arXiv:2312.09478 Equation:
    T_{J→I} = Σ p(i_t, i_{t-1}^(q), j_{t-1}^(o)) log [p(i_t|i_{t-1}^(q), j_{t-1}^(o)) / p(i_t|i_{t-1}^(q))]

    Args:
        source: Source time series, shape (T,) or (T, D)
        target: Target time series, shape (T,)
        config: Estimation configuration

    Returns:
        Transfer entropy in nats (non-negative)
    """
    # Embed time series
    x_past = _embed_time_series(source, config.lag_x)
    y_past = _embed_time_series(target, config.lag_y)

    # Align lengths
    min_len = min(len(x_past), len(y_past))
    x_past = x_past[-min_len:]
    y_past = y_past[-min_len:]

    # Future of target (one step ahead)
    y_future = target[-(min_len):].reshape(-1, 1) if target.ndim == 1 else target[-(min_len):]

    # Adjust for alignment
    y_future = y_future[1:]
    x_past = x_past[:-1]
    y_past = y_past[:-1]

    # TE = I(Y_future; X_past | Y_past)
    # = I(Y_future; X_past, Y_past) - I(Y_future; Y_past)

    xy_past = np.hstack([x_past, y_past])

    mi_full = _kraskov_mi(y_future, xy_past, k=config.k_neighbors)
    mi_cond = _kraskov_mi(y_future, y_past, k=config.k_neighbors)

    te = mi_full - mi_cond

    if config.normalize and mi_cond > 0:
        te = te / mi_cond  # Normalize by conditional entropy

    return max(0.0, te)


def build_causal_graph(
    signals: List[SourceSignal],
    config: TransferEntropyConfig = TransferEntropyConfig()
) -> CausalGraph:
    """
    Build weighted directed graph from transfer entropy between all pairs.

    From arXiv:2312.09478:
    A_ij = T_{x_j → x_i} if T > threshold, else 0

    This creates adjacency matrix where edge (j→i) has weight equal to
    transfer entropy from source j to source i.

    Args:
        signals: List of source signals to analyze
        config: Transfer entropy configuration

    Returns:
        CausalGraph with TE-weighted edges
    """
    n = len(signals)
    nodes = [s.name for s in signals]
    adjacency = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                te = transfer_entropy(
                    signals[j].values,
                    signals[i].values,
                    config
                )
                if te > config.threshold:
                    adjacency[i, j] = te

    return CausalGraph(nodes=nodes, adjacency=adjacency, threshold=config.threshold)


def detect_causal_structure_shift(
    signals: List[SourceSignal],
    window_size: int = 50,
    step_size: int = 10,
    config: TransferEntropyConfig = TransferEntropyConfig()
) -> List[Tuple[float, CausalGraph, float]]:
    """
    Detect shifts in causal structure over time using rolling windows.

    When causal structure shifts significantly, it indicates:
    - New information pathways forming
    - Existing dependencies breaking down
    - Potential regime change

    This is key for detecting when "attractor wells" form or dissolve -
    when coherent intention enters or exits the system.

    Args:
        signals: List of source signals
        window_size: Size of rolling window
        step_size: Step between windows
        config: TE configuration

    Returns:
        List of (timestamp, CausalGraph, structure_change_score) tuples
    """
    T = min(s.T for s in signals)
    results = []
    prev_adj = None

    for start in range(0, T - window_size, step_size):
        end = start + window_size

        # Extract window
        windowed_signals = [
            SourceSignal(
                name=s.name,
                values=s.values[start:end],
                timestamps=s.timestamps[start:end],
                quality=s.quality[start:end]
            )
            for s in signals
        ]

        # Build causal graph for this window
        graph = build_causal_graph(windowed_signals, config)

        # Compute structure change from previous window
        if prev_adj is not None:
            # Frobenius norm of difference
            change_score = np.linalg.norm(graph.adjacency - prev_adj, 'fro')
        else:
            change_score = 0.0

        timestamp = signals[0].timestamps[end - 1]
        results.append((timestamp, graph, change_score))
        prev_adj = graph.adjacency.copy()

    return results


def compute_intentionality_gradient(
    signals: List[SourceSignal],
    target_idx: int,
    window_size: int = 30,
    config: TransferEntropyConfig = TransferEntropyConfig()
) -> NDArray[np.float64]:
    """
    Compute the rate of change in directional transfer entropy.

    NOVEL INSIGHT #2 from Great Attractor framework:
    The gradient of TE (not just magnitude) indicates when passive
    correlation becomes active causation - when something starts
    "pulling" outcomes toward it.

    Rising gradient = strengthening attractor
    Falling gradient = weakening attractor
    Sign flip = causality reversal

    Args:
        signals: Source signals
        target_idx: Index of target signal to analyze
        window_size: Rolling window size
        config: TE configuration

    Returns:
        Array of intentionality gradient values over time
    """
    T = min(s.T for s in signals)
    n = len(signals)

    te_series = {i: [] for i in range(n) if i != target_idx}

    # Compute TE in rolling windows
    for start in range(0, T - window_size, 1):
        end = start + window_size

        target = signals[target_idx].values[start:end]

        for i in range(n):
            if i != target_idx:
                source = signals[i].values[start:end]
                te = transfer_entropy(source, target, config)
                te_series[i].append(te)

    # Compute gradients (first derivative of TE over time)
    gradients = []
    for i in te_series:
        if len(te_series[i]) > 1:
            grad = np.gradient(te_series[i])
            gradients.append(grad)

    # Aggregate: sum of absolute gradients indicates total intentionality pressure
    if gradients:
        return np.sum(np.abs(gradients), axis=0)
    return np.array([])
