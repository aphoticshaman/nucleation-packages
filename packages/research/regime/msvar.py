"""
Markov-Switching Vector Autoregression (MS-VAR) with Hierarchical Bayesian Group LASSO.
Based on US8645304B2 (IBM, expired February 4, 2018).

This module implements regime detection using:
- Hidden Markov Model for latent regime states
- VAR dynamics per regime
- Hierarchical Bayesian Group LASSO for automatic variable selection
- EM algorithm for inference

Key innovations from patent:
- Change point detection in causal modeling
- Group LASSO selects entire lag structures, not individual coefficients
- Hierarchical prior provides adaptive shrinkage per variable pair
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from scipy.special import logsumexp
from scipy.stats import invgamma

from ..core.types import RegimeEstimate, RegimeState


@dataclass
class MSVARConfig:
    """Configuration for MS-VAR model."""
    n_regimes: int = 3  # K: number of latent states
    n_lags: int = 2  # L: VAR lag order
    max_em_iterations: int = 100
    em_tolerance: float = 1e-4
    group_lasso_a: float = 1.0  # Inverse-gamma prior shape
    group_lasso_b: float = 1.0  # Inverse-gamma prior scale
    regularization_strength: float = 0.1  # Base λ for group LASSO


def embed_var(
    Y: NDArray[np.float64],
    n_lags: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Create VAR-style embedding of multivariate time series.

    Args:
        Y: Multivariate time series, shape (T, D)
        n_lags: Number of lags to include

    Returns:
        Tuple of (Y_target, Y_lagged) where:
        - Y_target: shape (T - n_lags, D)
        - Y_lagged: shape (T - n_lags, D * n_lags)
    """
    T, D = Y.shape
    Y_target = Y[n_lags:]

    Y_lagged = np.zeros((T - n_lags, D * n_lags))
    for l in range(n_lags):
        Y_lagged[:, l*D:(l+1)*D] = Y[n_lags-l-1:T-l-1]

    return Y_target, Y_lagged


class MarkovSwitchingVAR:
    """
    Markov-Switching VAR with Hierarchical Bayesian Group LASSO.

    From US8645304B2:
    Y_{j,t} = Σ_i Σ_l β_{ijkl} Y_{i,t-l} + ε_{j,t}

    where k = S_t is the latent regime state.

    The hierarchical Bayesian group LASSO places:
    - Level 1: θ_{ijkl} ~ N(0, σ²_{ijkl})
    - Level 2: σ²_{ijk} ~ Gamma(L, 2τ²_{ijk})
    - Level 3: τ²_{ijk} ~ Inverse-Gamma(a_{ijk}, b_{ijk})

    NOVEL INSIGHT #8: Attractor Genesis Detection
    The moment a new attractor well forms (before it has mass).
    In MS-VAR terms: when transition probabilities to a new regime
    suddenly increase from near-zero, a new attractor is forming.
    """

    def __init__(self, config: MSVARConfig = MSVARConfig()):
        self.config = config
        self.n_regimes = config.n_regimes
        self.n_lags = config.n_lags

        # Model parameters
        self.transition_matrix: Optional[NDArray] = None  # Shape: (K, K)
        self.initial_probs: Optional[NDArray] = None  # Shape: (K,)
        self.beta: Optional[NDArray] = None  # Shape: (K, D, D*L)
        self.sigma: Optional[NDArray] = None  # Shape: (K, D, D) - noise covariance per regime
        self.group_weights: Optional[NDArray] = None  # Shape: (K, D, D)

        # Inference results
        self.smoothed_probs: Optional[NDArray] = None  # Shape: (T, K)
        self.viterbi_path: Optional[NDArray] = None  # Shape: (T,)

    def _initialize_parameters(self, D: int) -> None:
        """Initialize model parameters."""
        K = self.n_regimes
        L = self.n_lags

        # Transition matrix: slightly favor staying in same state
        self.transition_matrix = np.ones((K, K)) / K
        for k in range(K):
            self.transition_matrix[k, k] += 0.5
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

        # Initial state probabilities
        self.initial_probs = np.ones(K) / K

        # VAR coefficients (small random initialization)
        self.beta = np.random.randn(K, D, D * L) * 0.01

        # Noise covariance (identity for each regime)
        self.sigma = np.stack([np.eye(D) for _ in range(K)])

        # Group LASSO weights
        self.group_weights = np.ones((K, D, D))

    def _forward_pass(
        self,
        Y_target: NDArray,
        Y_lagged: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Forward algorithm for HMM.

        Computes α_t(k) = P(S_t = k, Y_{1:t})

        Returns:
            Tuple of (log_alpha, log_likelihood_per_step)
        """
        T, D = Y_target.shape
        K = self.n_regimes

        log_alpha = np.zeros((T, K))
        log_lik = np.zeros(T)

        # t = 0
        log_obs = self._log_observation_prob(Y_target[0], Y_lagged[0])
        log_alpha[0] = np.log(self.initial_probs + 1e-10) + log_obs
        log_lik[0] = logsumexp(log_alpha[0])
        log_alpha[0] -= log_lik[0]  # Normalize

        # t > 0
        log_trans = np.log(self.transition_matrix + 1e-10)
        for t in range(1, T):
            log_obs = self._log_observation_prob(Y_target[t], Y_lagged[t])

            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_trans[:, k]) + log_obs[k]

            log_lik[t] = logsumexp(log_alpha[t])
            log_alpha[t] -= log_lik[t]

        return log_alpha, log_lik

    def _backward_pass(
        self,
        Y_target: NDArray,
        Y_lagged: NDArray
    ) -> NDArray:
        """
        Backward algorithm for HMM.

        Computes β_t(k) = P(Y_{t+1:T} | S_t = k)
        """
        T, D = Y_target.shape
        K = self.n_regimes

        log_beta = np.zeros((T, K))
        log_trans = np.log(self.transition_matrix + 1e-10)

        # t = T-1: log_beta[T-1] = 0 (already initialized)

        for t in range(T - 2, -1, -1):
            log_obs_next = self._log_observation_prob(Y_target[t+1], Y_lagged[t+1])

            for k in range(K):
                log_beta[t, k] = logsumexp(
                    log_trans[k, :] + log_obs_next + log_beta[t+1]
                )

            # Normalize for numerical stability
            log_beta[t] -= logsumexp(log_beta[t])

        return log_beta

    def _log_observation_prob(
        self,
        y: NDArray,
        y_lagged: NDArray
    ) -> NDArray:
        """
        Compute log observation probability under each regime.

        P(y_t | S_t = k, y_{t-1:t-L}) = N(y_t; β_k @ y_lagged, Σ_k)
        """
        K = self.n_regimes
        D = len(y)
        log_probs = np.zeros(K)

        for k in range(K):
            mean = self.beta[k] @ y_lagged
            residual = y - mean

            # Log of multivariate normal
            # -0.5 * (D*log(2π) + log|Σ| + (y-μ)^T Σ^{-1} (y-μ))
            try:
                L_k = np.linalg.cholesky(self.sigma[k])
                log_det = 2 * np.sum(np.log(np.diag(L_k)))
                z = np.linalg.solve(L_k, residual)
                mahalanobis = np.sum(z**2)
            except np.linalg.LinAlgError:
                # Fallback for non-positive-definite
                log_det = np.log(np.linalg.det(self.sigma[k]) + 1e-10)
                mahalanobis = residual @ np.linalg.pinv(self.sigma[k]) @ residual

            log_probs[k] = -0.5 * (D * np.log(2 * np.pi) + log_det + mahalanobis)

        return log_probs

    def _e_step(
        self,
        Y_target: NDArray,
        Y_lagged: NDArray
    ) -> Tuple[NDArray, NDArray, float]:
        """
        E-step: Compute smoothed state probabilities.

        From US8645304B2:
        L_{tk} = p(S_t = k | Y_{1:T}, D_0, ψ^{(m)})
        H_{t,k'k} = p(S_{t-1} = k', S_t = k | Y_{1:T}, D_0, ψ^{(m)})
        """
        log_alpha, log_lik = self._forward_pass(Y_target, Y_lagged)
        log_beta = self._backward_pass(Y_target, Y_lagged)

        # Smoothed state probabilities (gamma)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Transition probabilities (xi)
        T = len(Y_target)
        K = self.n_regimes
        xi = np.zeros((T - 1, K, K))

        log_trans = np.log(self.transition_matrix + 1e-10)

        for t in range(T - 1):
            log_obs_next = self._log_observation_prob(Y_target[t+1], Y_lagged[t+1])

            for k_prev in range(K):
                for k_next in range(K):
                    xi[t, k_prev, k_next] = (
                        log_alpha[t, k_prev] +
                        log_trans[k_prev, k_next] +
                        log_obs_next[k_next] +
                        log_beta[t+1, k_next]
                    )

            xi[t] = np.exp(xi[t] - logsumexp(xi[t]))

        total_log_lik = np.sum(log_lik)

        return gamma, xi, total_log_lik

    def _m_step(
        self,
        Y_target: NDArray,
        Y_lagged: NDArray,
        gamma: NDArray,
        xi: NDArray
    ) -> None:
        """
        M-step: Update parameters.

        Includes hierarchical Bayesian group LASSO weight update:
        w_{ijk}^{(m+1)} = (a_{ijk} + L) / (||θ_{ijk}^{(m)}||_2 + b_{ijk})
        """
        T, D = Y_target.shape
        K = self.n_regimes
        L = self.n_lags

        # Update initial probabilities
        self.initial_probs = gamma[0] + 1e-10
        self.initial_probs /= self.initial_probs.sum()

        # Update transition matrix
        self.transition_matrix = xi.sum(axis=0) + 1e-10
        self.transition_matrix /= self.transition_matrix.sum(axis=1, keepdims=True)

        # Update VAR coefficients and noise covariance per regime
        for k in range(K):
            weights_k = gamma[:, k]
            total_weight = weights_k.sum() + 1e-10

            # Weighted least squares for β_k
            W = np.diag(weights_k)
            XtWX = Y_lagged.T @ W @ Y_lagged
            XtWY = Y_lagged.T @ W @ Y_target

            # Add group LASSO regularization
            lambda_reg = self.config.regularization_strength
            for i in range(D):
                for j in range(D):
                    # Group: all lags for variable pair (i, j)
                    group_indices = [j + l * D for l in range(L)]
                    group_norm = np.linalg.norm(self.beta[k, i, group_indices])

                    # Update group weight
                    self.group_weights[k, i, j] = (
                        (self.config.group_lasso_a + L) /
                        (group_norm + self.config.group_lasso_b + 1e-10)
                    )

                    # Add regularization to XtWX
                    for idx in group_indices:
                        XtWX[idx, idx] += lambda_reg * self.group_weights[k, i, j]

            # Solve regularized least squares
            try:
                self.beta[k] = np.linalg.solve(XtWX + 1e-6 * np.eye(D * L), XtWY).T
            except np.linalg.LinAlgError:
                self.beta[k] = np.linalg.pinv(XtWX) @ XtWY.T

            # Update noise covariance
            residuals = Y_target - Y_lagged @ self.beta[k].T
            weighted_residuals = residuals * weights_k[:, None]
            self.sigma[k] = (weighted_residuals.T @ residuals) / total_weight
            self.sigma[k] += 1e-4 * np.eye(D)  # Regularization

    def fit(
        self,
        Y: NDArray[np.float64]
    ) -> 'MarkovSwitchingVAR':
        """
        Fit MS-VAR model using EM algorithm.

        Args:
            Y: Multivariate time series, shape (T, D)

        Returns:
            Self
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        T, D = Y.shape
        self._initialize_parameters(D)

        Y_target, Y_lagged = embed_var(Y, self.n_lags)

        prev_log_lik = -np.inf

        for iteration in range(self.config.max_em_iterations):
            # E-step
            gamma, xi, log_lik = self._e_step(Y_target, Y_lagged)

            # Check convergence
            if abs(log_lik - prev_log_lik) < self.config.em_tolerance:
                break
            prev_log_lik = log_lik

            # M-step
            self._m_step(Y_target, Y_lagged, gamma, xi)

        # Store smoothed probabilities
        self.smoothed_probs = np.zeros((T, self.n_regimes))
        self.smoothed_probs[self.n_lags:] = gamma

        # Viterbi path
        self.viterbi_path = np.argmax(self.smoothed_probs, axis=1)

        return self

    def predict_regime(
        self,
        Y: NDArray[np.float64]
    ) -> List[RegimeEstimate]:
        """
        Predict regime at each timestep.

        Args:
            Y: Multivariate time series

        Returns:
            List of RegimeEstimate objects
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Refit or use existing parameters
        Y_target, Y_lagged = embed_var(Y, self.n_lags)
        gamma, _, _ = self._e_step(Y_target, Y_lagged)

        results = []
        current_state = np.argmax(gamma[0])
        state_start = 0

        for t in range(len(gamma)):
            map_state = np.argmax(gamma[t])

            if map_state != current_state:
                current_state = map_state
                state_start = t

            results.append(RegimeEstimate(
                timestamp=float(t + self.n_lags),
                state_probabilities=gamma[t],
                map_state=int(map_state),
                transition_matrix=self.transition_matrix,
                state_duration=t - state_start + 1
            ))

        return results

    def detect_attractor_genesis(
        self,
        window_size: int = 20,
        threshold: float = 0.3
    ) -> List[Tuple[int, int]]:
        """
        NOVEL INSIGHT #8: Detect attractor genesis events.

        Identifies moments when transition probabilities to a regime
        suddenly increase from near-zero, indicating a new attractor
        is forming before it has significant mass.

        Args:
            window_size: Window for computing probability changes
            threshold: Minimum change to flag as genesis event

        Returns:
            List of (timestep, regime_index) pairs for genesis events
        """
        if self.smoothed_probs is None:
            raise ValueError("Model not fitted.")

        genesis_events = []
        T = len(self.smoothed_probs)

        for t in range(window_size, T):
            current_probs = self.smoothed_probs[t]
            past_probs = self.smoothed_probs[t - window_size:t].mean(axis=0)

            for k in range(self.n_regimes):
                # Check for sudden increase from low baseline
                if past_probs[k] < 0.1 and current_probs[k] - past_probs[k] > threshold:
                    genesis_events.append((t, k))

        return genesis_events

    def get_causality_inversion_windows(
        self,
        Y: NDArray[np.float64]
    ) -> List[Tuple[int, int, int]]:
        """
        NOVEL INSIGHT #9: Detect causality inversion windows.

        Brief periods where normal causal flow (indicated by VAR coefficients)
        reverses, indicating external intentionality entering the system.

        Args:
            Y: Time series

        Returns:
            List of (start_t, end_t, variable_pair) tuples
        """
        if self.beta is None:
            raise ValueError("Model not fitted.")

        inversions = []
        D = Y.shape[1] if Y.ndim > 1 else 1
        L = self.n_lags

        # Track coefficient signs over time (based on regime)
        regimes = self.viterbi_path

        for t in range(1, len(regimes)):
            k_prev = regimes[t - 1]
            k_curr = regimes[t]

            if k_prev != k_curr:
                # Check for sign inversions in dominant coefficients
                for i in range(D):
                    for j in range(D):
                        if i != j:
                            coef_prev = self.beta[k_prev, i, j]
                            coef_curr = self.beta[k_curr, i, j]

                            # Sign inversion with significant magnitude
                            if coef_prev * coef_curr < 0 and \
                               abs(coef_prev) > 0.1 and abs(coef_curr) > 0.1:
                                inversions.append((t - 1, t, (i, j)))

        return inversions
