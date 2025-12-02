"""
Compressed Sensing and Sparse Recovery.
Based on US8855431B2 (Stanford University, David Donoho, expired August 2024).

This module implements L1 minimization and Orthogonal Matching Pursuit (OMP)
for sparse signal recovery from compressed measurements.

Key innovations from patent:
- Measurement model: y = Ax + z (compressed observations)
- L1 minimization (Basis Pursuit) for exact recovery under RIP
- OMP for fast greedy approximation
- RIP-based recovery guarantees

Application: Extract sparse predictive features from high-dimensional
text embeddings, financial signals, or sensor data.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings

from ..core.types import CompressedSignal


@dataclass
class CompressedSensingConfig:
    """Configuration for compressed sensing recovery."""
    sparsity: int = 10  # Expected sparsity level K
    max_iterations: int = 100  # Max OMP iterations
    tolerance: float = 1e-6  # Residual tolerance
    lambda_lasso: float = 0.1  # LASSO regularization parameter


def generate_measurement_matrix(
    m: int,
    n: int,
    matrix_type: str = 'gaussian'
) -> NDArray[np.float64]:
    """
    Generate compressed sensing measurement matrix.

    From US8855431B2 (now public domain):
    - Gaussian random: m ≥ C·K·log(N/K) measurements for K-sparse recovery
    - Bernoulli/Rademacher: Same guarantee as Gaussian
    - Partial Fourier: m ≥ C·K·log⁴(N) (can use FFT)

    Args:
        m: Number of measurements (compressed dimension)
        n: Signal dimension
        matrix_type: Type of measurement matrix

    Returns:
        Measurement matrix of shape (m, n)
    """
    if matrix_type == 'gaussian':
        # Gaussian random with proper scaling
        Phi = np.random.randn(m, n) / np.sqrt(m)

    elif matrix_type == 'bernoulli':
        # Rademacher (±1) with proper scaling
        Phi = np.random.choice([-1, 1], size=(m, n)) / np.sqrt(m)

    elif matrix_type == 'fourier':
        # Partial Fourier (random rows of DFT matrix)
        full_dft = np.fft.fft(np.eye(n)) / np.sqrt(n)
        rows = np.random.choice(n, m, replace=False)
        Phi = np.real(full_dft[rows, :])

    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    return Phi


def orthogonal_matching_pursuit(
    y: NDArray[np.float64],
    Phi: NDArray[np.float64],
    config: CompressedSensingConfig = CompressedSensingConfig()
) -> CompressedSignal:
    """
    Orthogonal Matching Pursuit for sparse recovery.

    From US8855431B2 Algorithm (now public domain):
    1. Initialize: r₀ = y, support Λ = ∅
    2. For t = 1 to K:
       (a) Find j* = argmax|⟨Φ_j, r_{t-1}⟩|
       (b) Update support: Λ_t = Λ_{t-1} ∪ {j*}
       (c) Least squares: α̂|_Λt = Φ_Λt† y
       (d) Update residual: r_t = y - Φ_Λt α̂_Λt
    3. Output α̂

    Complexity: O(mNd) general; O(md log d) with FFT for Fourier matrices.

    Args:
        y: Measurements, shape (m,)
        Phi: Measurement matrix, shape (m, n)
        config: Recovery configuration

    Returns:
        CompressedSignal with recovered sparse signal
    """
    m, n = Phi.shape
    K = min(config.sparsity, m - 1)  # Can't recover more than m components

    # Initialize
    residual = y.copy()
    support = []
    x_recovered = np.zeros(n)

    for t in range(config.max_iterations):
        # Find index with maximum correlation to residual
        correlations = np.abs(Phi.T @ residual)

        # Exclude already selected indices
        for idx in support:
            correlations[idx] = -np.inf

        j_star = np.argmax(correlations)

        # Check if we've found enough or correlation is too small
        if correlations[j_star] < config.tolerance:
            break

        support.append(j_star)

        # Least squares on support
        Phi_support = Phi[:, support]
        x_support, _, _, _ = np.linalg.lstsq(Phi_support, y, rcond=None)

        # Update residual
        residual = y - Phi_support @ x_support

        # Check stopping criterion
        if np.linalg.norm(residual) < config.tolerance:
            break

        if len(support) >= K:
            break

    # Construct full solution
    x_recovered = np.zeros(n)
    if len(support) > 0:
        Phi_support = Phi[:, support]
        x_support, _, _, _ = np.linalg.lstsq(Phi_support, y, rcond=None)
        for i, idx in enumerate(support):
            x_recovered[idx] = x_support[i]

    return CompressedSignal(
        measurements=y,
        recovered_signal=x_recovered,
        support=np.array(support, dtype=np.int64),
        residual_norm=float(np.linalg.norm(residual)),
        sparsity=len(support)
    )


def basis_pursuit_lasso(
    y: NDArray[np.float64],
    Phi: NDArray[np.float64],
    config: CompressedSensingConfig = CompressedSensingConfig()
) -> CompressedSignal:
    """
    L1 minimization (LASSO) for sparse recovery.

    From US8855431B2 (now public domain):
    min_x ||Bx||_1 + λ||y - Φx||_2²

    Also known as Basis Pursuit Denoising (BPDN).

    Args:
        y: Measurements, shape (m,)
        Phi: Measurement matrix, shape (m, n)
        config: Recovery configuration

    Returns:
        CompressedSignal with recovered sparse signal
    """
    m, n = Phi.shape
    lambda_reg = config.lambda_lasso

    def objective(x):
        residual = y - Phi @ x
        return 0.5 * np.sum(residual**2) + lambda_reg * np.sum(np.abs(x))

    def gradient(x):
        residual = y - Phi @ x
        grad_ls = -Phi.T @ residual
        grad_l1 = lambda_reg * np.sign(x)
        return grad_ls + grad_l1

    # Use L-BFGS-B with bounds to encourage sparsity
    x0 = np.zeros(n)

    # Soft thresholding via proximal gradient
    # Iterative shrinkage-thresholding algorithm (ISTA)
    step_size = 1.0 / np.linalg.norm(Phi, ord=2)**2
    x = x0.copy()

    for _ in range(config.max_iterations):
        # Gradient step
        residual = y - Phi @ x
        gradient = -Phi.T @ residual

        x_new = x - step_size * gradient

        # Soft thresholding (proximal operator for L1)
        threshold = lambda_reg * step_size
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - threshold, 0)

        # Check convergence
        if np.linalg.norm(x_new - x) < config.tolerance:
            break

        x = x_new

    # Identify support
    support = np.where(np.abs(x) > config.tolerance)[0]
    residual_norm = np.linalg.norm(y - Phi @ x)

    return CompressedSignal(
        measurements=y,
        recovered_signal=x,
        support=support,
        residual_norm=float(residual_norm),
        sparsity=len(support)
    )


def compute_rip_constant(
    Phi: NDArray[np.float64],
    k: int,
    n_samples: int = 1000
) -> float:
    """
    Estimate Restricted Isometry Property constant δ_k.

    From US8855431B2 (now public domain):
    Matrix Φ has RIP of order k if ∃ δ_k < 1 such that for all k-sparse x:
    (1 - δ_k)||x||² ≤ ||Φx||² ≤ (1 + δ_k)||x||²

    This is computationally intensive to compute exactly, so we estimate
    via random sampling.

    Args:
        Phi: Measurement matrix
        k: Sparsity level
        n_samples: Number of random k-sparse vectors to test

    Returns:
        Estimated RIP constant
    """
    m, n = Phi.shape
    max_distortion = 0.0

    for _ in range(n_samples):
        # Generate random k-sparse vector
        support = np.random.choice(n, k, replace=False)
        x = np.zeros(n)
        x[support] = np.random.randn(k)
        x = x / np.linalg.norm(x)  # Unit norm

        # Compute distortion
        y = Phi @ x
        distortion = np.abs(np.linalg.norm(y)**2 - 1.0)
        max_distortion = max(max_distortion, distortion)

    return max_distortion


class SparseFeatureExtractor:
    """
    Extract sparse predictive features from high-dimensional data.

    Uses compressed sensing to identify the most important features
    from text embeddings, financial signals, etc.

    NOVEL INSIGHT #7: Membership Vector Entanglement
    When multiple high-dimensional features compress to the same
    sparse representation, it indicates "entangled" agents whose
    behaviors become correlated beyond environmental noise.
    """

    def __init__(
        self,
        n_features: int,
        compression_ratio: float = 0.3,
        config: CompressedSensingConfig = CompressedSensingConfig()
    ):
        self.n_features = n_features
        self.n_measurements = int(n_features * compression_ratio)
        self.config = config
        self.Phi = generate_measurement_matrix(
            self.n_measurements, n_features, 'gaussian'
        )

    def compress(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compress high-dimensional features.

        Args:
            X: Feature matrix, shape (N, n_features)

        Returns:
            Compressed features, shape (N, n_measurements)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.Phi.T

    def recover(
        self,
        Y: NDArray[np.float64],
        method: str = 'omp'
    ) -> List[CompressedSignal]:
        """
        Recover sparse features from compressed measurements.

        Args:
            Y: Compressed measurements, shape (N, n_measurements)
            method: 'omp' or 'lasso'

        Returns:
            List of CompressedSignal objects
        """
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        results = []
        for y in Y:
            if method == 'omp':
                result = orthogonal_matching_pursuit(y, self.Phi, self.config)
            else:
                result = basis_pursuit_lasso(y, self.Phi, self.config)
            results.append(result)

        return results

    def find_common_support(
        self,
        signals: List[CompressedSignal]
    ) -> Tuple[NDArray[np.int64], float]:
        """
        NOVEL INSIGHT #7: Find entangled features.

        Identifies features that appear in multiple recovered signals,
        indicating correlated/entangled behaviors.

        Args:
            signals: List of recovered sparse signals

        Returns:
            Tuple of (common_support_indices, entanglement_score)
        """
        if len(signals) == 0:
            return np.array([], dtype=np.int64), 0.0

        # Count feature occurrences across signals
        feature_counts = np.zeros(self.n_features)
        for signal in signals:
            feature_counts[signal.support] += 1

        # Find features appearing in majority of signals
        threshold = len(signals) * 0.5
        common_support = np.where(feature_counts >= threshold)[0]

        # Entanglement score: fraction of signals sharing common features
        if len(common_support) == 0:
            entanglement_score = 0.0
        else:
            avg_overlap = feature_counts[common_support].mean() / len(signals)
            entanglement_score = avg_overlap

        return common_support, float(entanglement_score)


def cosamp_recovery(
    y: NDArray[np.float64],
    Phi: NDArray[np.float64],
    sparsity: int,
    max_iter: int = 50,
    tol: float = 1e-6
) -> CompressedSignal:
    """
    CoSaMP (Compressive Sampling Matching Pursuit) algorithm.

    From US8855431B2 recovery guarantee:
    If δ_{4K} ≤ 0.1:
    ||x - x^i||_2 ≤ C · ||x - x_K||_2 + D · ||e||_2

    where x_K is best K-term approximation, e is measurement noise.

    Args:
        y: Measurements
        Phi: Measurement matrix
        sparsity: Target sparsity K
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        CompressedSignal with recovered signal
    """
    m, n = Phi.shape
    K = sparsity

    # Initialize
    x = np.zeros(n)
    residual = y.copy()

    for _ in range(max_iter):
        # Signal proxy: 2K largest correlations
        proxy = Phi.T @ residual
        omega = np.argsort(np.abs(proxy))[-2*K:]

        # Merge with current support
        T = np.union1d(omega, np.where(np.abs(x) > tol)[0])

        # Least squares on merged support
        if len(T) > 0:
            x_T, _, _, _ = np.linalg.lstsq(Phi[:, T], y, rcond=None)

            # Prune to K largest
            if len(x_T) > K:
                top_K = np.argsort(np.abs(x_T))[-K:]
                x_new = np.zeros(n)
                for i, idx in enumerate(T[top_K]):
                    x_new[idx] = x_T[top_K[i]]
            else:
                x_new = np.zeros(n)
                for i, idx in enumerate(T):
                    x_new[idx] = x_T[i]
        else:
            x_new = np.zeros(n)

        # Update residual
        residual = y - Phi @ x_new

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    support = np.where(np.abs(x) > tol)[0]

    return CompressedSignal(
        measurements=y,
        recovered_signal=x,
        support=support,
        residual_norm=float(np.linalg.norm(residual)),
        sparsity=len(support)
    )
