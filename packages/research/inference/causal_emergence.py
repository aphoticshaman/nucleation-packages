"""
Causal Emergence Measures for Multivariate Time Series.
Based on arXiv:2004.08220 (Mediano et al.) and GitHub implementation.

This module implements measures to determine whether macro-level features
(like "regime" labels) have genuine causal power over micro-level components.

Key measures:
- Ψ (Psi): Causal emergence - macro predicts future better than micro alone
- Δ (Delta): Downward causation - macro causes micro
- Γ (Gamma): Causal decoupling - macro affects macro without micro mediation

If Ψ > 0, then V is causally emergent.
If Δ > 0, then V shows downward causation.
If Ψ > 0 and Γ = 0, then V shows causal decoupling.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.stats import entropy
import warnings

from ..core.types import CausalEmergence


@dataclass
class CausalEmergenceConfig:
    """Configuration for causal emergence computation."""
    lag: int = 1  # Time lag for causal analysis
    n_bins: int = 10  # Bins for discretization (if continuous)
    method: str = 'gaussian'  # 'gaussian' or 'discrete'


def gaussian_entropy(X: NDArray[np.float64]) -> float:
    """
    Compute entropy of multivariate Gaussian.

    H(X) = 0.5 * log((2πe)^d |Σ|)

    For numerical stability, uses log-determinant via Cholesky.

    Args:
        X: Data matrix, shape (N, D)

    Returns:
        Entropy in nats
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    N, D = X.shape

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Covariance matrix with regularization
    cov = X_centered.T @ X_centered / (N - 1)
    cov += 1e-6 * np.eye(D)  # Regularization

    # Log-determinant via Cholesky
    try:
        L = np.linalg.cholesky(cov)
        log_det = 2 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        # Fallback to eigenvalues
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        log_det = np.sum(np.log(eigvals))

    # Entropy formula
    H = 0.5 * (D * (1 + np.log(2 * np.pi)) + log_det)

    return H


def gaussian_mutual_information(
    X: NDArray[np.float64],
    Y: NDArray[np.float64]
) -> float:
    """
    Compute mutual information for Gaussian variables.

    I(X; Y) = H(X) + H(Y) - H(X, Y)

    Args:
        X: First variable, shape (N, Dx)
        Y: Second variable, shape (N, Dy)

    Returns:
        Mutual information in nats
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    H_X = gaussian_entropy(X)
    H_Y = gaussian_entropy(Y)
    H_XY = gaussian_entropy(np.hstack([X, Y]))

    MI = H_X + H_Y - H_XY

    return max(0.0, MI)


def gaussian_conditional_mutual_information(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64]
) -> float:
    """
    Compute conditional mutual information for Gaussian variables.

    I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    Args:
        X: First variable
        Y: Second variable
        Z: Conditioning variable

    Returns:
        Conditional MI in nats
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    H_XZ = gaussian_entropy(np.hstack([X, Z]))
    H_YZ = gaussian_entropy(np.hstack([Y, Z]))
    H_Z = gaussian_entropy(Z)
    H_XYZ = gaussian_entropy(np.hstack([X, Y, Z]))

    CMI = H_XZ + H_YZ - H_Z - H_XYZ

    return max(0.0, CMI)


def compute_psi(
    X_t: NDArray[np.float64],
    X_t1: NDArray[np.float64],
    V_t: NDArray[np.float64],
    config: CausalEmergenceConfig = CausalEmergenceConfig()
) -> float:
    """
    Compute Ψ (Psi): Causal emergence measure.

    Ψ = I(V_t; X_{t+1}) - I(X_t; X_{t+1} | V_t)

    Ψ > 0 means the macro-feature V contains predictive information
    about future micro-states X_{t+1} that the micro-states X_t don't
    have when we already know V.

    From arXiv:2004.08220:
    "If Ψ > 0, then V is causally emergent"

    Args:
        X_t: Micro-states at time t, shape (N, D)
        X_t1: Micro-states at time t+1, shape (N, D)
        V_t: Macro-feature at time t, shape (N,) or (N, K)
        config: Configuration

    Returns:
        Psi value (positive indicates emergence)
    """
    if V_t.ndim == 1:
        V_t = V_t.reshape(-1, 1)

    # I(V_t; X_{t+1})
    I_V_Xnext = gaussian_mutual_information(V_t, X_t1)

    # I(X_t; X_{t+1} | V_t)
    I_X_Xnext_given_V = gaussian_conditional_mutual_information(X_t, X_t1, V_t)

    psi = I_V_Xnext - I_X_Xnext_given_V

    return psi


def compute_delta(
    X_t: NDArray[np.float64],
    X_t1: NDArray[np.float64],
    V_t: NDArray[np.float64],
    config: CausalEmergenceConfig = CausalEmergenceConfig()
) -> float:
    """
    Compute Δ (Delta): Downward causation measure.

    Δ = I(V_t; X_{t+1} | X_t)

    Δ > 0 means the macro-feature V provides information about
    future micro-states beyond what current micro-states provide.
    This is "downward causation" - the macro causes the micro.

    From arXiv:2004.08220:
    "If Δ > 0, then V shows downward causation"

    Args:
        X_t: Micro-states at time t
        X_t1: Micro-states at time t+1
        V_t: Macro-feature at time t
        config: Configuration

    Returns:
        Delta value (positive indicates downward causation)
    """
    if V_t.ndim == 1:
        V_t = V_t.reshape(-1, 1)

    # I(V_t; X_{t+1} | X_t)
    delta = gaussian_conditional_mutual_information(V_t, X_t1, X_t)

    return delta


def compute_gamma(
    V_t: NDArray[np.float64],
    V_t1: NDArray[np.float64],
    X_t: NDArray[np.float64],
    config: CausalEmergenceConfig = CausalEmergenceConfig()
) -> float:
    """
    Compute Γ (Gamma): Macro-to-macro causation through micro.

    Γ = I(V_t; V_{t+1} | X_t)

    Γ measures how much the macro-feature at t predicts the macro-feature
    at t+1, beyond what the micro-states provide.

    From arXiv:2004.08220:
    "If Ψ > 0 and Γ = 0, then V shows causal decoupling"
    (macro evolves independently of micro mediation)

    Args:
        V_t: Macro-feature at time t
        V_t1: Macro-feature at time t+1
        X_t: Micro-states at time t
        config: Configuration

    Returns:
        Gamma value
    """
    if V_t.ndim == 1:
        V_t = V_t.reshape(-1, 1)
    if V_t1.ndim == 1:
        V_t1 = V_t1.reshape(-1, 1)

    # I(V_t; V_{t+1} | X_t)
    gamma = gaussian_conditional_mutual_information(V_t, V_t1, X_t)

    return gamma


def compute_causal_emergence(
    X: NDArray[np.float64],
    macro_feature_fn: Callable[[NDArray], NDArray],
    config: CausalEmergenceConfig = CausalEmergenceConfig()
) -> CausalEmergence:
    """
    Compute all causal emergence measures for a time series.

    CRITICAL: The macro-feature V must be a supervenient feature of X,
    meaning V(t) is a deterministic or stochastic function of X(t) only.

    Args:
        X: Multivariate time series, shape (T, D)
        macro_feature_fn: Function that maps X to macro-feature V
        config: Configuration

    Returns:
        CausalEmergence object with Psi, Delta, Gamma
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    T = len(X)
    lag = config.lag

    # Compute macro-features
    V = np.array([macro_feature_fn(X[t:t+1]) for t in range(T)])
    if V.ndim == 1:
        V = V.reshape(-1, 1)

    # Align time series for lag
    X_t = X[:-lag]
    X_t1 = X[lag:]
    V_t = V[:-lag]
    V_t1 = V[lag:]

    # Compute measures
    psi = compute_psi(X_t, X_t1, V_t, config)
    delta = compute_delta(X_t, X_t1, V_t, config)
    gamma = compute_gamma(V_t, V_t1, X_t, config)

    return CausalEmergence(psi=psi, delta=delta, gamma=gamma)


def validate_regime_labels(
    X: NDArray[np.float64],
    regime_labels: NDArray[np.int64],
    config: CausalEmergenceConfig = CausalEmergenceConfig()
) -> Tuple[CausalEmergence, bool]:
    """
    Validate whether regime labels have genuine causal power.

    This is crucial for LatticeForge: if your "regime" classifications
    don't show causal emergence, they're just correlational artifacts,
    not real attractor basins.

    Args:
        X: Multivariate time series, shape (T, D)
        regime_labels: Integer regime labels, shape (T,)
        config: Configuration

    Returns:
        Tuple of (CausalEmergence, is_valid)
        is_valid is True if labels show significant causal power
    """
    # One-hot encode regime labels as macro-feature
    n_regimes = len(np.unique(regime_labels))

    def regime_to_onehot(x_window):
        # This is a hack - we need the regime at this time
        # In practice, you'd pass pre-computed regime labels
        return regime_labels[0:1]  # Placeholder

    # Actually use the provided labels directly
    V = np.zeros((len(regime_labels), n_regimes))
    for t, label in enumerate(regime_labels):
        V[t, label] = 1.0

    T = len(X)
    lag = config.lag

    X_t = X[:-lag]
    X_t1 = X[lag:]
    V_t = V[:-lag]
    V_t1 = V[lag:]

    psi = compute_psi(X_t, X_t1, V_t, config)
    delta = compute_delta(X_t, X_t1, V_t, config)
    gamma = compute_gamma(V_t, V_t1, X_t, config)

    emergence = CausalEmergence(psi=psi, delta=delta, gamma=gamma)

    # Labels are "valid" if they show emergence and downward causation
    is_valid = (psi > 0.1) and (delta > 0.05)

    return emergence, is_valid


class EmergenceMonitor:
    """
    Monitor causal emergence over time.

    Tracks whether macro-level labels maintain causal power
    as new data arrives. If emergence degrades, labels may
    need recalibration.

    NOVEL INSIGHT: When emergence metrics suddenly drop,
    it indicates the attractor landscape has shifted -
    old regime definitions no longer capture the dynamics.
    """

    def __init__(
        self,
        window_size: int = 100,
        config: CausalEmergenceConfig = CausalEmergenceConfig()
    ):
        self.window_size = window_size
        self.config = config
        self.history: list = []

    def update(
        self,
        X_window: NDArray[np.float64],
        V_window: NDArray[np.float64]
    ) -> CausalEmergence:
        """
        Update emergence estimates with new window.

        Args:
            X_window: Recent micro-states, shape (window_size, D)
            V_window: Recent macro-features, shape (window_size,) or (window_size, K)

        Returns:
            Latest CausalEmergence estimate
        """
        if V_window.ndim == 1:
            V_window = V_window.reshape(-1, 1)

        lag = self.config.lag
        X_t = X_window[:-lag]
        X_t1 = X_window[lag:]
        V_t = V_window[:-lag]
        V_t1 = V_window[lag:]

        psi = compute_psi(X_t, X_t1, V_t, self.config)
        delta = compute_delta(X_t, X_t1, V_t, self.config)
        gamma = compute_gamma(V_t, V_t1, X_t, self.config)

        emergence = CausalEmergence(psi=psi, delta=delta, gamma=gamma)
        self.history.append(emergence)

        return emergence

    def get_emergence_trend(self) -> Tuple[float, float, float]:
        """
        Compute trend in emergence metrics.

        Returns:
            Tuple of (psi_trend, delta_trend, gamma_trend)
            Negative trends indicate degrading causal power.
        """
        if len(self.history) < 5:
            return 0.0, 0.0, 0.0

        recent = self.history[-10:]
        psis = [e.psi for e in recent]
        deltas = [e.delta for e in recent]
        gammas = [e.gamma for e in recent]

        psi_trend = np.polyfit(range(len(psis)), psis, 1)[0]
        delta_trend = np.polyfit(range(len(deltas)), deltas, 1)[0]
        gamma_trend = np.polyfit(range(len(gammas)), gammas, 1)[0]

        return psi_trend, delta_trend, gamma_trend

    def needs_recalibration(self, threshold: float = -0.01) -> bool:
        """
        Check if regime definitions need recalibration.

        Returns True if emergence is degrading significantly.
        """
        psi_trend, delta_trend, _ = self.get_emergence_trend()

        return psi_trend < threshold or delta_trend < threshold
