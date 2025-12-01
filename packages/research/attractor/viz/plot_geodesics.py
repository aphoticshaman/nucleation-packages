"""
plot_geodesics.py

Geodesic flow visualization on curved information manifolds.

Supports:
- Numerical computation of Christoffel symbols
- Integration of geodesic ODEs (Runge–Kutta 4)
- Streamplot visualization of geodesic flow
- Curvature-induced basin convergence visualization

For Great Attractor simulations, the metric is typically:
    g = Fisher Information Matrix (FIM)
or any smooth Riemannian metric you pass into the plotter.

Usage example:
    gp = GeodesicPlotter()
    ax = gp.plot_geodesics(metric_fn=..., U=..., particles=...)

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable, Any, Tuple


Array = NDArray[np.float64]


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

@dataclass
class GeodesicPlotConfig:
    xmin: float = -2.5
    xmax: float = 2.5
    ymin: float = -2.5
    ymax: float = 2.5
    resolution: int = 40
    streamline_density: float = 1.2
    max_geodesic_length: int = 300
    dt: float = 0.02
    cmap: str = "magma"
    figsize: Tuple[int, int] = (8, 6)
    show_particles: bool = True
    show_potential: bool = True
    potential_levels: int = 30
    show_start_points: bool = False
    n_random_starts: int = 20


# ------------------------------------------------------------
# GEODESIC PLOTTER
# ------------------------------------------------------------

class GeodesicPlotter:

    def __init__(self, config: GeodesicPlotConfig = GeodesicPlotConfig()):
        self.cfg = config

    # --------------------------------------------------------
    # Utility: Construct grid
    # --------------------------------------------------------

    def _make_grid(self) -> Tuple[Array, Array]:
        cfg = self.cfg
        x = np.linspace(cfg.xmin, cfg.xmax, cfg.resolution)
        y = np.linspace(cfg.ymin, cfg.ymax, cfg.resolution)
        return np.meshgrid(x, y)

    # --------------------------------------------------------
    # Numerical Christoffel symbols
    # --------------------------------------------------------

    def _christoffel_symbols(
        self,
        x: Array,
        metric_fn: Callable[[Array], Array],
        eps: float = 1e-4,
    ) -> Array:
        """
        Numerically compute Christoffel symbols Γ^k_{ij}.

        Parameters
        ----------
        x : Array of shape (2,)
        metric_fn : callable returning g(x) of shape (2,2)

        Returns
        -------
        Gamma : Array of shape (2,2,2)
            Gamma[k,i,j]
        """
        g = metric_fn(x)       # (2,2)
        g_inv = np.linalg.inv(g)
        Gamma = np.zeros((2, 2, 2))

        # numerical partial derivatives of metric
        for i in range(2):
            for j in range(2):
                def g_ij(x0):
                    return metric_fn(x0)[i, j]

                # partial wrt x1
                xp = np.array([x[0] + eps, x[1]])
                xm = np.array([x[0] - eps, x[1]])
                dg_d1 = (g_ij(xp) - g_ij(xm)) / (2 * eps)

                # partial wrt x2
                yp = np.array([x[0], x[1] + eps])
                ym = np.array([x[0], x[1] - eps])
                dg_d2 = (g_ij(yp) - g_ij(ym)) / (2 * eps)

                # assemble Γ^k_{ij}
                for k in range(2):
                    term = 0
                    for ell in range(2):
                        def g_jell(x0):
                            return metric_fn(x0)[j, ell]
                        def g_iell(x0):
                            return metric_fn(x0)[i, ell]
                        def g_ij_fn(x0):
                            return metric_fn(x0)[i, j]

                        xp2 = np.array([x[0] + eps, x[1]])
                        xm2 = np.array([x[0] - eps, x[1]])
                        d_g_jell_dx1 = (g_jell(xp2) - g_jell(xm2)) / (2 * eps)
                        d_g_iell_dx1 = (g_iell(xp2) - g_iell(xm2)) / (2 * eps)
                        d_g_ij_dx1 = (g_ij_fn(xp2) - g_ij_fn(xm2)) / (2 * eps)

                        yp2 = np.array([x[0], x[1] + eps])
                        ym2 = np.array([x[0], x[1] - eps])
                        d_g_jell_dx2 = (g_jell(yp2) - g_jell(ym2)) / (2 * eps)
                        d_g_iell_dx2 = (g_iell(yp2) - g_iell(ym2)) / (2 * eps)
                        d_g_ij_dx2 = (g_ij_fn(yp2) - g_ij_fn(ym2)) / (2 * eps)

                        partial_i_jell = np.array([d_g_jell_dx1, d_g_jell_dx2])
                        partial_i_iell = np.array([d_g_iell_dx1, d_g_iell_dx2])
                        partial_ell_ij = np.array([d_g_ij_dx1, d_g_ij_dx2])

                        term += g_inv[k, ell] * (
                            partial_i_jell[i]
                            + partial_i_iell[j]
                            - partial_ell_ij[ell]
                        )

                    Gamma[k, i, j] = 0.5 * term

        return Gamma

    # --------------------------------------------------------
    # Geodesic ODE integrator
    # --------------------------------------------------------

    def _geodesic_rhs(
        self,
        state: Array,
        metric_fn: Callable[[Array], Array],
    ) -> Array:
        """
        Right-hand side of geodesic equation:

            dθ/dt = v
            dv/dt = -Γ^k_{ij} v^i v^j

        state = [x1, x2, v1, v2]
        returns dstate/dt
        """
        x = state[:2]
        v = state[2:]
        Gamma = self._christoffel_symbols(x, metric_fn)  # (2,2,2)

        dv = np.zeros(2)
        for k in range(2):
            s = 0
            for i in range(2):
                for j in range(2):
                    s += Gamma[k, i, j] * v[i] * v[j]
            dv[k] = -s

        return np.concatenate([v, dv])

    def _rk4_step(
        self,
        state: Array,
        metric_fn: Callable[[Array], Array],
        dt: float,
    ) -> Array:
        """One RK4 step for geodesic integration."""
        k1 = self._geodesic_rhs(state, metric_fn)
        k2 = self._geodesic_rhs(state + 0.5 * dt * k1, metric_fn)
        k3 = self._geodesic_rhs(state + 0.5 * dt * k2, metric_fn)
        k4 = self._geodesic_rhs(state +     dt * k3, metric_fn)
        return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # --------------------------------------------------------
    # Main plot: geodesic flow field
    # --------------------------------------------------------

    def plot_geodesics(
        self,
        metric_fn: Callable[[Array], Array],
        U: Optional[Callable[[Array], float]] = None,
        particles: Optional[Array] = None,
        ax: Optional[Any] = None,
        title: str = "Geodesic Flow on Information Manifold",
    ) -> Any:
        """
        Plot geodesic flow field using streamplots.

        metric_fn: returns metric g(x)
        U: optional potential field for contour shading
        particles: optional particle cloud for overlay
        """
        cfg = self.cfg

        if ax is None:
            fig, ax = plt.subplots(figsize=cfg.figsize)

        # --- grid ---
        X, Y = self._make_grid()

        # --- compute geodesic direction field ---
        Uvec = np.zeros_like(X)
        Vvec = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = np.array([X[i, j], Y[i, j]])
                g = metric_fn(x)
                detg = np.linalg.det(g)
                if detg <= 0:
                    Uvec[i, j] = 0
                    Vvec[i, j] = 0
                    continue
                eps = 1e-3
                dx1 = np.array([x[0] + eps, x[1]])
                dx2 = np.array([x[0], x[1] + eps])

                det1 = np.linalg.det(metric_fn(dx1))
                det2 = np.linalg.det(metric_fn(dx2))

                d_det_dx1 = (det1 - detg) / eps
                d_det_dx2 = (det2 - detg) / eps

                Uvec[i, j] = -d_det_dx1
                Vvec[i, j] = -d_det_dx2

        # normalize
        mag = np.sqrt(Uvec**2 + Vvec**2) + 1e-12
        Uvec /= mag
        Vvec /= mag

        # --- Potential overlay ---
        if cfg.show_potential and U is not None:
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = U(np.array([X[i, j], Y[i, j]]))

            cs = ax.contourf(
                X, Y, Z,
                levels=cfg.potential_levels,
                cmap=cfg.cmap,
                alpha=0.75,
            )
            plt.colorbar(cs, ax=ax, shrink=0.6)

        # --- Streamplot of geodesic drift ---
        ax.streamplot(
            X,
            Y,
            Uvec,
            Vvec,
            density=cfg.streamline_density,
            color="white",
            linewidth=1.0,
            arrowsize=0.8,
        )

        # --- Particle overlay ---
        if cfg.show_particles and particles is not None:
            ax.scatter(
                particles[:, 0],
                particles[:, 1],
                c="cyan",
                s=10,
                alpha=0.6,
                label="particles",
            )

        if cfg.show_start_points:
            starts = np.random.uniform(
                low=[cfg.xmin, cfg.ymin],
                high=[cfg.xmax, cfg.ymax],
                size=(cfg.n_random_starts, 2),
            )
            for x0 in starts:
                self._plot_single_geodesic(ax, x0, metric_fn)

        ax.set_title(title)
        ax.set_xlim(cfg.xmin, cfg.xmax)
        ax.set_ylim(cfg.ymin, cfg.ymax)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        return ax

    # --------------------------------------------------------
    # Optional explicit geodesic integration
    # --------------------------------------------------------

    def _plot_single_geodesic(
        self,
        ax: Any,
        x0: Array,
        metric_fn: Callable[[Array], Array],
    ) -> None:
        """Plot one explicit geodesic from a starting point."""
        cfg = self.cfg

        v0 = np.random.randn(2)
        v0 /= np.linalg.norm(v0) + 1e-12

        state = np.concatenate([x0, v0])

        pts = []
        for _ in range(cfg.max_geodesic_length):
            pts.append(state[:2].copy())
            state = self._rk4_step(
                state,
                metric_fn,
                cfg.dt,
            )

        pts = np.array(pts)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="yellow",
            linewidth=1.2,
            alpha=0.8,
        )
