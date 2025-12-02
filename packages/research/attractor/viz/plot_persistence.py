"""
plot_persistence.py

Persistence diagram visualization for topological data analysis (TDA).

Provides:
- Computation of persistence diagrams (H0, H1) from point clouds
- Persistence diagram plotting with birth-death coordinates
- Persistent entropy computation for phase transition detection
- Betti curve visualization

Uses a simplified Rips complex approach. For production use, consider
integrating with ripser, gudhi, or giotto-tda.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


Array = NDArray[np.float64]


@dataclass
class PersistencePlotConfig:
    """Configuration for persistence diagram visualization."""

    figsize: Tuple[int, int] = (10, 5)
    h0_color: str = "blue"
    h1_color: str = "orange"
    diagonal_color: str = "gray"
    point_size: int = 40
    alpha: float = 0.7
    max_dim: int = 1  # compute up to H1
    max_edge_length: float = 2.0  # maximum filtration value


def persistent_entropy(persistence_pairs: Array, eps: float = 1e-12) -> float:
    """
    Compute persistent entropy of a persistence diagram.

    Persistent entropy measures the "complexity" of topological features.
    High entropy → many features of similar lifetime
    Low entropy → few dominant features

    Parameters
    ----------
    persistence_pairs : array of shape (n, 2)
        Each row is (birth, death).
    eps : float
        Small value to avoid log(0).

    Returns
    -------
    float
        Persistent entropy value.
    """
    if len(persistence_pairs) == 0:
        return 0.0

    lifetimes = persistence_pairs[:, 1] - persistence_pairs[:, 0]
    lifetimes = np.maximum(lifetimes, eps)

    total = lifetimes.sum()
    if total < eps:
        return 0.0

    probs = lifetimes / total
    entropy = -np.sum(probs * np.log(probs + eps))

    return float(entropy)


def compute_distance_matrix(points: Array) -> Array:
    """Compute pairwise Euclidean distance matrix."""
    return squareform(pdist(points, metric="euclidean"))


class PersistencePlotter:
    """
    Persistence diagram computation and visualization.

    Implements a simplified Vietoris-Rips persistence computation.
    For high-dimensional data or large point clouds, use specialized
    libraries like ripser or gudhi.
    """

    def __init__(self, config: PersistencePlotConfig = PersistencePlotConfig()):
        self.cfg = config

    def compute_pd(
        self,
        points: Array,
        max_edge_length: Optional[float] = None,
    ) -> Tuple[Array, Array]:
        """
        Compute persistence diagrams H0 and H1.

        Parameters
        ----------
        points : array of shape (n, d)
            Point cloud.
        max_edge_length : float, optional
            Maximum filtration value.

        Returns
        -------
        H0 : array of shape (k0, 2)
            0-dimensional persistence pairs (connected components).
        H1 : array of shape (k1, 2)
            1-dimensional persistence pairs (loops).
        """
        max_eps = max_edge_length or self.cfg.max_edge_length
        n = len(points)

        if n < 2:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        # Distance matrix
        D = compute_distance_matrix(points)

        # H0: Connected components via Union-Find
        H0 = self._compute_h0(D, n, max_eps)

        # H1: Simplified loop detection
        H1 = self._compute_h1(D, n, max_eps)

        return H0, H1

    def _compute_h0(self, D: Array, n: int, max_eps: float) -> Array:
        """
        Compute H0 (connected components) using Union-Find.

        Each point is born at 0. Components merge when edge appears.
        """
        # Union-Find structure
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> bool:
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True

        # Get all edges sorted by length
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] <= max_eps:
                    edges.append((D[i, j], i, j))
        edges.sort()

        # Track component deaths
        deaths = {}  # component -> death time
        for eps, i, j in edges:
            pi, pj = find(i), find(j)
            if pi != pj:
                # One component dies
                dying = max(pi, pj)  # arbitrary choice
                deaths[dying] = eps
                union(i, j)

        # Build persistence pairs
        # All points are born at 0
        # Components that never merge live to infinity (use max_eps)
        H0 = []
        for i in range(n):
            if parent[i] == i:  # root of a tree
                death = deaths.get(i, max_eps)
                if death > 0:
                    H0.append([0.0, death])

        # Add one infinite component (the final connected component)
        # Usually H0 has exactly n-1 finite bars and 1 infinite
        return np.array(H0).reshape(-1, 2) if H0 else np.array([]).reshape(0, 2)

    def _compute_h1(self, D: Array, n: int, max_eps: float) -> Array:
        """
        Simplified H1 computation (loop detection).

        This is a heuristic approach - for accurate H1, use ripser/gudhi.
        Detects approximate loop births when edges complete triangles.
        """
        if n < 3:
            return np.array([]).reshape(0, 2)

        # Collect triangles and their formation time (max edge length)
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    edges = [D[i, j], D[j, k], D[i, k]]
                    max_edge = max(edges)
                    if max_edge <= max_eps:
                        # Triangle forms when longest edge appears
                        # Loop may be "born" when shorter edges form,
                        # "dies" when triangle fills
                        birth = sorted(edges)[1]  # second longest
                        death = max_edge
                        if death > birth:
                            triangles.append([birth, death])

        # Filter and deduplicate (keep only significant loops)
        if not triangles:
            return np.array([]).reshape(0, 2)

        H1 = np.array(triangles)

        # Remove duplicates and very short-lived features
        min_lifetime = 0.01 * max_eps
        lifetimes = H1[:, 1] - H1[:, 0]
        H1 = H1[lifetimes > min_lifetime]

        return H1 if len(H1) > 0 else np.array([]).reshape(0, 2)

    def plot(
        self,
        H0: Array,
        H1: Array,
        ax: Optional[Any] = None,
        title: str = "Persistence Diagram",
    ) -> Any:
        """
        Plot persistence diagram.

        Parameters
        ----------
        H0 : array of shape (k0, 2)
            0-dimensional persistence pairs.
        H1 : array of shape (k1, 2)
            1-dimensional persistence pairs.
        ax : matplotlib axis, optional
            Axis to plot on.
        title : str
            Plot title.

        Returns
        -------
        ax : matplotlib axis
        """
        cfg = self.cfg

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Determine axis limits
        all_points = np.vstack([H0, H1]) if len(H0) > 0 or len(H1) > 0 else np.array([[0, 1]])
        max_val = all_points.max() * 1.1

        # Diagonal line
        ax.plot([0, max_val], [0, max_val], color=cfg.diagonal_color, linestyle="--", alpha=0.5)

        # Plot H0
        if len(H0) > 0:
            ax.scatter(
                H0[:, 0], H0[:, 1],
                c=cfg.h0_color, s=cfg.point_size, alpha=cfg.alpha,
                label=f"H0 ({len(H0)} features)",
            )

        # Plot H1
        if len(H1) > 0:
            ax.scatter(
                H1[:, 0], H1[:, 1],
                c=cfg.h1_color, s=cfg.point_size, alpha=cfg.alpha,
                label=f"H1 ({len(H1)} features)",
            )

        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect("equal")

        return ax

    def plot_betti_curves(
        self,
        H0: Array,
        H1: Array,
        ax: Optional[Any] = None,
        n_points: int = 100,
    ) -> Any:
        """
        Plot Betti curves (number of features vs filtration value).

        Parameters
        ----------
        H0, H1 : persistence diagrams
        ax : matplotlib axis, optional
        n_points : int
            Number of filtration values to sample.

        Returns
        -------
        ax : matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        # Determine filtration range
        all_points = np.vstack([H0, H1]) if len(H0) > 0 or len(H1) > 0 else np.array([[0, 1]])
        max_val = all_points.max() * 1.1
        eps_range = np.linspace(0, max_val, n_points)

        # Compute Betti numbers at each filtration value
        betti_0 = np.zeros(n_points)
        betti_1 = np.zeros(n_points)

        for i, eps in enumerate(eps_range):
            if len(H0) > 0:
                betti_0[i] = np.sum((H0[:, 0] <= eps) & (H0[:, 1] > eps))
            if len(H1) > 0:
                betti_1[i] = np.sum((H1[:, 0] <= eps) & (H1[:, 1] > eps))

        ax.plot(eps_range, betti_0, color=self.cfg.h0_color, label="β₀", linewidth=2)
        ax.plot(eps_range, betti_1, color=self.cfg.h1_color, label="β₁", linewidth=2)

        ax.set_xlabel("Filtration value (ε)")
        ax.set_ylabel("Betti number")
        ax.set_title("Betti Curves")
        ax.legend()

        return ax

    def analyze(
        self,
        points: Array,
        title: str = "TDA Analysis",
    ) -> Tuple[Array, Array, float]:
        """
        Full TDA analysis: compute PD, plot, and return entropy.

        Parameters
        ----------
        points : array of shape (n, d)
            Point cloud.
        title : str
            Plot title.

        Returns
        -------
        H0, H1 : persistence diagrams
        entropy : float
            Persistent entropy.
        """
        cfg = self.cfg

        # Compute persistence
        H0, H1 = self.compute_pd(points)

        # Compute entropy
        all_pairs = np.vstack([H0, H1]) if len(H0) > 0 or len(H1) > 0 else np.array([]).reshape(0, 2)
        entropy = persistent_entropy(all_pairs)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=cfg.figsize)

        # Persistence diagram
        self.plot(H0, H1, ax=axes[0], title=f"{title}: Persistence Diagram")

        # Betti curves
        self.plot_betti_curves(H0, H1, ax=axes[1])

        # Add entropy annotation
        axes[0].annotate(
            f"Entropy: {entropy:.3f}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        return H0, H1, entropy
