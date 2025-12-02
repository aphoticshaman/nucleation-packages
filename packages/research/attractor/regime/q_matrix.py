"""
q_matrix.py

Utilities for constructing, estimating, and analyzing
continuous-time Markov generator matrices (Q-matrices)
for regime-switching Great Attractor dynamics.

A Q-matrix satisfies:
- q_ij >= 0 for i != j
- q_ii = - sum_{j != i} q_ij

We use Q to model macro-regime dynamics (zeitgeist shifts).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Dict, Any


Array = NDArray[np.float64]


@dataclass
class QMatrixResult:
    Q: Array
    eigenvalues: Array
    spectral_gap: float


# ------------------------------------------------------------
# Q-MATRIX BUILDING & CHECKING
# ------------------------------------------------------------

def build_q_matrix(
    rates: Array,
) -> Array:
    """
    Construct a valid Q-matrix from off-diagonal rates.

    Parameters
    ----------
    rates : (n, n) array
        Off-diagonal entries are proposed transition rates λ_ij >= 0.
        Diagonal entries are ignored/overwritten.

    Returns
    -------
    Q : (n, n) array
        Valid generator matrix.
    """
    rates = np.asarray(rates, dtype=float)
    n, m = rates.shape
    if n != m:
        raise ValueError("rates must be square")

    Q = rates.copy()
    np.fill_diagonal(Q, 0.0)

    row_sums = Q.sum(axis=1)
    np.fill_diagonal(Q, -row_sums)

    return Q


def is_valid_q(Q: Array, tol: float = 1e-8) -> bool:
    """
    Check basic generator properties.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.shape[0] != Q.shape[1]:
        return False

    # Off-diagonal non-negative
    offdiag = Q - np.diag(np.diag(Q))
    if np.any(offdiag < -tol):
        return False

    # Row sums ≈ 0
    if np.max(np.abs(Q.sum(axis=1))) > tol:
        return False

    return True


# ------------------------------------------------------------
# ESTIMATION FROM OBSERVED COUNTS
# ------------------------------------------------------------

def estimate_q_from_counts(
    N: Array,
    dwell_times: Array,
    eps: float = 1e-12,
) -> Array:
    """
    Estimate a Q-matrix from:
      - N[i,j]: number of observed transitions i -> j
      - dwell_times[i]: total time spent in state i

    Simple MLE for continuous-time Markov chains:
      q_ij = N_ij / T_i, i != j
      q_ii = - sum_{j != i} q_ij

    Parameters
    ----------
    N : (n, n) array
        Transition counts between regimes.
    dwell_times : (n,) array
        Total time spent in each regime.

    Returns
    -------
    Q : (n, n) array
    """
    N = np.asarray(N, dtype=float)
    dwell_times = np.asarray(dwell_times, dtype=float)

    n, m = N.shape
    if n != m:
        raise ValueError("N must be square")

    if dwell_times.shape[0] != n:
        raise ValueError("dwell_times must have length n")

    Q = np.zeros_like(N)
    for i in range(n):
        if dwell_times[i] <= eps:
            continue
        for j in range(n):
            if i == j:
                continue
            Q[i, j] = N[i, j] / (dwell_times[i] + eps)

    # set diagonals
    row_sums = Q.sum(axis=1)
    np.fill_diagonal(Q, -row_sums)

    return Q


# ------------------------------------------------------------
# SPECTRAL ANALYSIS (METASTABILITY)
# ------------------------------------------------------------

def analyze_q(Q: Array) -> QMatrixResult:
    """
    Compute eigenvalues + spectral gap of Q.

    For an irreducible CTMC:
      - eigenvalue 0 is always present
      - small second eigenvalue (in magnitude) ↔ metastability

    Spectral gap = min_{λ != 0} |Re(λ)|
    (in magnitude, ignoring numerical noise)
    """
    Q = np.asarray(Q, dtype=float)
    vals = np.linalg.eigvals(Q)

    # sort by real part
    vals_sorted = np.sort_complex(vals)
    # eigenvalue nearest 0
    idx0 = np.argmin(np.abs(vals_sorted))
    lam0 = vals_sorted[idx0]

    # spectral gap: min |Re(λ)| for λ != λ0
    mask = np.ones_like(vals_sorted, dtype=bool)
    mask[idx0] = False
    lam_others = vals_sorted[mask]

    if lam_others.size == 0:
        gap = 0.0
    else:
        gap = float(np.min(np.abs(np.real(lam_others))))

    return QMatrixResult(Q=Q, eigenvalues=vals_sorted, spectral_gap=gap)


# ------------------------------------------------------------
# FORWARD SIMULATION (DISCRETE-STEP)
# ------------------------------------------------------------

def simulate_markov_chain(
    Q: Array,
    r0: int,
    T: float,
    dt: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Simulate regime process r(t) under generator Q via
    explicit Euler approximation:

        P ≈ I + Q dt

    (valid for small dt: dt * max |q_ii| << 1)

    Parameters
    ----------
    Q : (n, n) array
        Generator matrix.
    r0 : int
        Initial regime index.
    T : float
        Total time horizon.
    dt : float
        Time step.
    rng : np.random.Generator, optional

    Returns
    -------
    dict with:
        'times': (K,) array of times
        'regimes': (K,) array of regime indices
    """
    if rng is None:
        rng = np.random.default_rng()

    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    steps = int(np.ceil(T / dt))

    # Discrete transition matrix: P ≈ I + Q dt
    P = np.eye(n) + Q * dt
    # Ensure small numerical negatives don't break sampling
    P = np.clip(P, 0.0, 1.0)
    # Row-normalize
    row_sums = P.sum(axis=1, keepdims=True) + 1e-12
    P = P / row_sums

    regimes = np.zeros(steps + 1, dtype=int)
    times = np.linspace(0, T, steps + 1)

    regimes[0] = r0
    for k in range(steps):
        r = regimes[k]
        regimes[k + 1] = rng.choice(np.arange(n), p=P[r])

    return {"times": times, "regimes": regimes}
