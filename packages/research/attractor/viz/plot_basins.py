"""
plot_basins.py

Visualization utilities for Great Attractor basin boundaries and
effective potential landscapes.

Supports:
- 2D latent spaces (primary)
- Contour plots of U(x)
- Particle overlays
- OGSE influence vector overlay
- Separatrix & basin boundary visualization
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict, Any


Array = NDArray[np.float64]


@dataclass
class BasinPlotConfig:
    """
    Configuration for basin visualization.
    """
    xmin: float = -2.5
    xmax: float = 2.5
    ymin: float = -2.5
    ymax: float = 2.5
    resolution: int = 200
    cmap: str = "inferno"
    n_contours: int = 40
    show_particles: bool = True
    show_vector_field: bool = False
    show_influence_vector: bool = True
    show_separatrix: bool = False  # advanced (requires U1, U2)
    figsize: Tuple[int, int] = (8, 6)


class BasinPlotter:
    """
    Plots the potential landscape U(x) and overlays the MV ensemble particles.

    Usage:
        plotter = BasinPlotter(config)
        plotter.plot(U=..., particles=..., influence_vec=...)
    """

    def __init__(self, config: BasinPlotConfig):
        self.cfg = config

    # ------------------------------------------------------
    # GRID CONSTRUCTION
    # ------------------------------------------------------

    def _make_grid(self) -> Tuple[Array, Array, Array]:
        """Construct 2D grid over latent space."""
        cfg = self.cfg
        x = np.linspace(cfg.xmin, cfg.xmax, cfg.resolution)
        y = np.linspace(cfg.ymin, cfg.ymax, cfg.resolution)
        X, Y = np.meshgrid(x, y)
        return X, Y, np.zeros_like(X)

    # ------------------------------------------------------
    # POTENTIAL EVALUATION
    # ------------------------------------------------------

    def _eval_potential_grid(
        self,
        U: Callable[[Array], float],
        X: Array,
        Y: Array,
    ) -> Array:
        """Evaluate U on 2D grid."""
        Z = np.zeros_like(X)
        shape = X.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                Z[i, j] = U(np.array([X[i, j], Y[i, j]]))
        return Z

    # ------------------------------------------------------
    # SEPARATRIX (OPTIONAL)
    # ------------------------------------------------------

    def _compute_separatrix(
        self,
        U1: Callable[[Array], float],
        U2: Callable[[Array], float],
        X: Array,
        Y: Array,
    ) -> Array:
        """
        Compute separatrix between competing potentials:
            S(x) = U1(x) - U2(x)
        separatrix = S(x) = 0 contour
        """
        S = np.zeros_like(X)
        shape = X.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                x = np.array([X[i, j], Y[i, j]])
                S[i, j] = U1(x) - U2(x)
        return S

    # ------------------------------------------------------
    # MAIN PLOT FUNCTION
    # ------------------------------------------------------

    def plot(
        self,
        U: Callable[[Array], float],
        particles: Optional[Array] = None,
        influence_vec: Optional[Array] = None,
        mean: Optional[Array] = None,
        U_alt: Optional[Callable[[Array], float]] = None,
        title: str = "Great Attractor Basin",
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot basin landscape.

        Parameters
        ----------
        U : potential function
        particles : MV ensemble positions (N,2)
        influence_vec : OGSE influence direction (2,)
        mean : ensemble mean (2,)
        U_alt : optional competing potential for separatrix
        """

        cfg = self.cfg

        if ax is None:
            fig, ax = plt.subplots(figsize=cfg.figsize)

        # --- Grid + U evaluation ---
        X, Y, _ = self._make_grid()
        Z = self._eval_potential_grid(U, X, Y)

        # --- contour plot ---
        cs = ax.contourf(
            X,
            Y,
            Z,
            levels=cfg.n_contours,
            cmap=cfg.cmap,
            alpha=0.85,
        )
        plt.colorbar(cs, ax=ax, shrink=0.7, label="U(x)")

        # --- separatrix (U_alt - U) ---
        if cfg.show_separatrix and U_alt is not None:
            S = self._compute_separatrix(U_alt, U, X, Y)
            ax.contour(
                X,
                Y,
                S,
                levels=[0],
                linewidths=2,
                colors="white",
                linestyles="--",
            )

        # --- particle overlay ---
        if cfg.show_particles and particles is not None:
            ax.scatter(
                particles[:, 0],
                particles[:, 1],
                s=10,
                c="cyan",
                alpha=0.6,
                label="particles",
            )

        # --- influence vector overlay ---
        if cfg.show_influence_vector and influence_vec is not None:
            if mean is None:
                mean = np.array([0.0, 0.0])
            ax.arrow(
                mean[0],
                mean[1],
                influence_vec[0],
                influence_vec[1],
                width=0.02,
                color="red",
                label="influence",
            )

        ax.set_title(title, fontsize=16)
        ax.set_xlim(cfg.xmin, cfg.xmax)
        ax.set_ylim(cfg.ymin, cfg.ymax)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        return ax
