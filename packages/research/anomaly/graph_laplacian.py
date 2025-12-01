"""
Graph Laplacian Anomaly Detection.
Based on US9805002B2 (IBM, expired October 31, 2021).

This module implements semi-supervised anomaly detection using
graph Laplacian regularization and latent variable models.

Key innovations from patent:
- Similarity matrix encodes label constraints (normal-normal positive,
  normal-anomalous negative)
- Latent variable model for dimensionality reduction
- Gradient optimization for joint W and Z learning
- Anomaly score via reconstruction error in latent space
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from ..core.types import CausalGraph, AnomalyScore


@dataclass
class GraphAnomalyConfig:
    """Configuration for graph Laplacian anomaly detection."""
    latent_dim: int = 10  # D' in the patent
    lambda_reg: float = 1.0  # Graph Laplacian regularization strength
    learning_rate: float = 0.01
    max_iter: int = 100
    tol: float = 1e-6
    normal_similarity: float = 1.0  # 'a' in patent
    anomaly_similarity: float = -1.0  # 'b' in patent
    unlabeled_similarity: float = 0.0  # 'c' in patent


def construct_similarity_matrix(
    n_samples: int,
    labeled_normal: List[int],
    labeled_anomaly: List[int],
    config: GraphAnomalyConfig = GraphAnomalyConfig()
) -> NDArray[np.float64]:
    """
    Construct similarity matrix encoding label constraints.

    From US9805002B2:
    - Normal-normal pairs: positive similarity 'a'
    - Normal-anomalous pairs: non-positive similarity 'b'
    - Constraint: b ≤ c ≤ a

    Args:
        n_samples: Total number of samples
        labeled_normal: Indices of samples labeled as normal
        labeled_anomaly: Indices of samples labeled as anomalous
        config: Configuration with similarity values

    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    R = np.full((n_samples, n_samples), config.unlabeled_similarity)

    # Normal-normal pairs
    for i in labeled_normal:
        for j in labeled_normal:
            if i != j:
                R[i, j] = config.normal_similarity

    # Anomaly-anomaly pairs
    for i in labeled_anomaly:
        for j in labeled_anomaly:
            if i != j:
                R[i, j] = config.normal_similarity  # Anomalies similar to each other

    # Normal-anomaly pairs
    for i in labeled_normal:
        for j in labeled_anomaly:
            R[i, j] = config.anomaly_similarity
            R[j, i] = config.anomaly_similarity

    return R


def graph_laplacian(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute graph Laplacian from similarity matrix.

    L = D_R - R

    where D_R is diagonal degree matrix.

    Args:
        R: Similarity matrix

    Returns:
        Graph Laplacian matrix
    """
    D = np.diag(R.sum(axis=1))
    return D - R


class GraphLatentAnomalyDetector:
    """
    Latent variable model with graph Laplacian regularization.

    From US9805002B2:
    X_{n,d} = Z_n · W_d + s · ε

    where Z is latent variable, W is sensor-specific coefficients,
    s varies by label status.

    NOVEL INSIGHT #3 from Great Attractor framework:
    Phase transitions are moments when trajectories cross attractor
    basin boundaries. The latent space Z represents the "possibility
    space" - anomalies are points near basin boundaries where the
    gradient field is unstable.
    """

    def __init__(self, config: GraphAnomalyConfig = GraphAnomalyConfig()):
        self.config = config
        self.W: Optional[NDArray] = None  # Sensor coefficients
        self.Z: Optional[NDArray] = None  # Latent variables

    def fit(
        self,
        X: NDArray[np.float64],
        labeled_normal: List[int] = None,
        labeled_anomaly: List[int] = None,
        similarity_matrix: Optional[NDArray[np.float64]] = None
    ) -> 'GraphLatentAnomalyDetector':
        """
        Fit the latent variable model.

        From US9805002B2, gradient updates:
        W := W - α[{S ⊙ (X - ZW^T)}^T Z + N(WW^T)^{-1}W]
        Z := Z - α[{S ⊙ (X - ZW^T)}W + λLZ]

        Args:
            X: Data matrix of shape (N, D)
            labeled_normal: Indices of normal samples
            labeled_anomaly: Indices of anomalous samples
            similarity_matrix: Pre-computed similarity matrix (optional)

        Returns:
            Self
        """
        N, D = X.shape
        D_prime = self.config.latent_dim

        # Initialize
        self.W = np.random.randn(D, D_prime) * 0.1
        self.Z = np.random.randn(N, D_prime) * 0.1

        # Construct similarity and Laplacian
        if similarity_matrix is None:
            labeled_normal = labeled_normal or []
            labeled_anomaly = labeled_anomaly or []
            R = construct_similarity_matrix(N, labeled_normal, labeled_anomaly, self.config)
        else:
            R = similarity_matrix

        L = graph_laplacian(R)

        # Scale matrix (1 for all in unsupervised case)
        S = np.ones_like(X)

        # Gradient descent
        for iteration in range(self.config.max_iter):
            # Reconstruction error
            residual = X - self.Z @ self.W.T
            scaled_residual = S * residual

            # W update
            grad_W = -scaled_residual.T @ self.Z
            # Regularization term for numerical stability
            reg_W = self.config.lambda_reg * self.W
            self.W -= self.config.learning_rate * (grad_W + reg_W)

            # Z update with Laplacian regularization
            grad_Z = -scaled_residual @ self.W + self.config.lambda_reg * L @ self.Z
            self.Z -= self.config.learning_rate * grad_Z

            # Check convergence
            loss = np.sum(scaled_residual ** 2) + self.config.lambda_reg * np.trace(self.Z.T @ L @ self.Z)
            if iteration > 0 and abs(prev_loss - loss) < self.config.tol:
                break
            prev_loss = loss

        return self

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute anomaly scores via reconstruction error.

        From US9805002B2:
        s_n = (I - W(W^TW)^{-1}W^T) · X_n

        This is the component of X orthogonal to the learned subspace.

        Args:
            X: Data matrix of shape (N, D) or (D,) for single sample

        Returns:
            Anomaly scores of shape (N,)
        """
        if self.W is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Projection matrix onto orthogonal complement of W's column space
        WtW_inv = np.linalg.pinv(self.W.T @ self.W)
        P_orth = np.eye(X.shape[1]) - self.W @ WtW_inv @ self.W.T

        # Orthogonal component magnitude
        X_orth = X @ P_orth
        scores = np.linalg.norm(X_orth, axis=1)

        return scores

    def get_basin_boundary_proximity(
        self,
        X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        NOVEL INSIGHT #3: Proximity to attractor basin boundaries.

        Points near basin boundaries have high gradient magnitude in
        the latent space - small perturbations lead to large changes
        in which attractor they fall toward.

        Computed as local gradient magnitude of anomaly score.

        Args:
            X: Data matrix

        Returns:
            Basin boundary proximity scores
        """
        if self.Z is None:
            raise ValueError("Model not fitted.")

        # Compute gradient of score with respect to latent position
        scores = self.score(X)

        # Numerical gradient estimation via finite differences
        eps = 1e-4
        gradients = np.zeros_like(self.Z)

        for d in range(self.Z.shape[1]):
            Z_plus = self.Z.copy()
            Z_plus[:, d] += eps
            X_plus = Z_plus @ self.W.T

            Z_minus = self.Z.copy()
            Z_minus[:, d] -= eps
            X_minus = Z_minus @ self.W.T

            scores_plus = self.score(X_plus)
            scores_minus = self.score(X_minus)

            gradients[:, d] = (scores_plus - scores_minus) / (2 * eps)

        # Gradient magnitude
        return np.linalg.norm(gradients, axis=1)


def spectral_anomaly_detection(
    causal_graph: CausalGraph,
    features: NDArray[np.float64],
    n_components: int = 5
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Spectral clustering-based anomaly detection using causal graph.

    Uses the Ng-Jordan-Weiss algorithm from US7103225B2 (NEC, expired 2023).

    1. Compute normalized Laplacian from causal graph
    2. Find first k eigenvectors
    3. Cluster in spectral embedding space
    4. Anomalies are points far from cluster centers

    NOVEL INSIGHT #4: Coherence Amplification
    When agents synchronize intentions, they cluster tightly in spectral
    space. Dispersion indicates competing attractors; tight clusters
    indicate amplified coherent intention.

    Args:
        causal_graph: CausalGraph with transfer entropy weights
        features: Additional node features, shape (N, D)
        n_components: Number of spectral components

    Returns:
        Tuple of (spectral embedding, anomaly scores)
    """
    L_sym = causal_graph.get_normalized_laplacian()

    # Eigen decomposition (smallest eigenvalues for Laplacian)
    if L_sym.shape[0] > 100:
        # Use sparse solver for large graphs
        eigenvalues, eigenvectors = eigsh(csr_matrix(L_sym), k=n_components, which='SM')
    else:
        eigenvalues, eigenvectors = eigh(L_sym)
        eigenvectors = eigenvectors[:, :n_components]

    # Normalize rows to unit length
    row_norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    row_norms = np.where(row_norms > 0, row_norms, 1.0)
    embedding = eigenvectors / row_norms

    # Anomaly score: distance from centroid
    centroid = embedding.mean(axis=0)
    scores = np.linalg.norm(embedding - centroid, axis=1)

    return embedding, scores


def detect_coherence_amplification(
    embeddings_over_time: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
    NOVEL INSIGHT #4: Detect coherence amplification over time.

    When distributed agents synchronize, the spectral embedding becomes
    more concentrated. This indicates an amplified attractor well forming.

    Args:
        embeddings_over_time: List of spectral embeddings at each timestep

    Returns:
        Coherence amplification signal (negative = dispersion, positive = concentration)
    """
    dispersions = []

    for embedding in embeddings_over_time:
        # Measure dispersion as trace of covariance
        centered = embedding - embedding.mean(axis=0)
        cov = centered.T @ centered / len(centered)
        dispersion = np.trace(cov)
        dispersions.append(dispersion)

    dispersions = np.array(dispersions)

    # Coherence = negative rate of change of dispersion
    # Increasing coherence = decreasing dispersion
    coherence = -np.gradient(dispersions)

    return coherence
