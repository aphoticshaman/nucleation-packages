"""
latticeforge.py

High-level API for the Great Attractor / LatticeForge system.

This module provides a single entry point:

    from research.attractor.api.latticeforge import LatticeForge

    lf = LatticeForge()
    lf.simulate(steps=1000)
    lf.visualize("basins")
    lf.detect_early_warning()

It wraps:
- GreatAttractorSimulator (core.great_attractor_sim)
- Visualization modules:
    - BasinPlotter (plot_basins.py)
    - GeodesicPlotter (plot_geodesics.py)
    - PersistencePlotter (plot_persistence.py)
    - ParticleAnimator (anim_particles.py)
- Optional regime layer (RegimeSDE, q_matrix utilities)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Callable, List, Tuple

import matplotlib.pyplot as plt

# Core engine
from research.core.great_attractor_sim import (
    GreatAttractorSimulator,
    GreatAttractorConfig,
)

# Regime layer
from research.attractor.regime.q_matrix import (
    build_q_matrix,
    analyze_q,
    QMatrixResult,
)
from research.attractor.regime.regime_sde import RegimeSDE, RegimeSDEConfig

# Visualization
from research.attractor.viz.plot_basins import BasinPlotter, BasinPlotConfig
from research.attractor.viz.plot_geodesics import GeodesicPlotter, GeodesicPlotConfig
from research.attractor.viz.plot_persistence import (
    PersistencePlotter,
    PersistencePlotConfig,
    persistent_entropy,
)
from research.attractor.viz.anim_particles import (
    ParticleAnimator,
    AnimationConfig,
)


Array = NDArray[np.float64]


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

@dataclass
class LatticeForgeConfig:
    """Configuration for the LatticeForge high-level API."""

    # Simulator
    n_agents: int = 500
    dt: float = 0.05
    interaction_strength: float = 2.0
    diffusion: float = 0.1

    # Attractor
    attractor_pos: Array = field(
        default_factory=lambda: np.array([5.0, 5.0], dtype=np.float64)
    )

    # Regime layer
    use_regime: bool = False
    n_regimes: int = 3
    regime_beta: float = 1.0

    # Visualization defaults
    xlim: Tuple[float, float] = (-6.0, 6.0)
    ylim: Tuple[float, float] = (-6.0, 6.0)


# ------------------------------------------------------------
# MAIN CLASS
# ------------------------------------------------------------

class LatticeForge:
    """
    High-level Great Attractor API.

    Provides a unified interface for:
    - Simulation (step, simulate)
    - Steering (steer attractor position/precision)
    - Early warning detection (variance, autocorrelation, TDA)
    - Visualization (basins, geodesics, persistence, animation)

    Example
    -------
    >>> from research.attractor.api.latticeforge import LatticeForge, LatticeForgeConfig
    >>> cfg = LatticeForgeConfig(n_agents=800, use_regime=True)
    >>> lf = LatticeForge(cfg)
    >>> lf.simulate(steps=2000)
    >>> lf.detect_early_warning()
    >>> lf.visualize("basins")
    """

    def __init__(
        self,
        config: Optional[LatticeForgeConfig] = None,
        sim_config_override: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LatticeForge.

        Parameters
        ----------
        config : LatticeForgeConfig
            Global configuration.
        sim_config_override : dict
            Passed directly to GreatAttractorSimulator to override defaults.
        """
        self.cfg = config or LatticeForgeConfig()

        # --- build simulation config ---
        ga_cfg = GreatAttractorConfig(
            n_agents=self.cfg.n_agents,
            interaction_strength=self.cfg.interaction_strength,
            diffusion=self.cfg.diffusion,
            dt=self.cfg.dt,
        )

        # --- core simulator ---
        self.sim = GreatAttractorSimulator(ga_cfg)

        # --- regime engine (optional) ---
        self.regime_sde: Optional[RegimeSDE] = None
        self.regime_history: List[int] = []
        self.regime_times: List[float] = []

        if self.cfg.use_regime:
            # simple default: quadratic attractor potential for all regimes
            def U_base(x: Array) -> float:
                return 0.5 * float(np.sum((x - self.cfg.attractor_pos) ** 2))

            potential_fns = [U_base for _ in range(self.cfg.n_regimes)]
            r_cfg = RegimeSDEConfig(
                n_regimes=self.cfg.n_regimes,
                beta=self.cfg.regime_beta,
                dt=self.cfg.dt,
            )
            self.regime_sde = RegimeSDE(r_cfg, potential_fns)

        # --- visualization helpers ---
        self.basin_plotter = BasinPlotter(
            BasinPlotConfig(
                xmin=self.cfg.xlim[0],
                xmax=self.cfg.xlim[1],
                ymin=self.cfg.ylim[0],
                ymax=self.cfg.ylim[1],
            )
        )
        self.geodesic_plotter = GeodesicPlotter(
            GeodesicPlotConfig(
                xmin=self.cfg.xlim[0],
                xmax=self.cfg.xlim[1],
                ymin=self.cfg.ylim[0],
                ymax=self.cfg.ylim[1],
            )
        )
        self.persistence_plotter = PersistencePlotter(PersistencePlotConfig())

        # history buffers for early-warning & diagnostics
        self.metrics_history: List[Dict[str, Any]] = []
        self.position_history: List[Array] = []

    # --------------------------------------------------------
    # BASIC OPERATIONS
    # --------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """
        Advance simulation by one time step and optionally evolve regime process.

        Returns
        -------
        dict
            Metrics from the simulation step including time, mean position,
            precision, and optionally regime index.
        """
        # Run simulator step
        history = self.sim.run(steps=1)
        metrics = {
            "time": history.times[-1] if history.times else 0.0,
            "mean_pos": history.mean_positions[-1] if history.mean_positions else np.zeros(2),
            "precision": history.precisions[-1] if history.precisions else 0.0,
        }

        # gather positions
        positions = history.positions[-1] if history.positions else np.zeros((self.cfg.n_agents, 2))
        self.position_history.append(positions)
        self.metrics_history.append(metrics)

        # regime evolution (if enabled)
        if self.regime_sde is not None:
            # Drive regime with swarm centroid
            x_centroid = positions.mean(axis=0)
            regime_state = self.regime_sde.step(x_centroid)
            self.regime_history.append(regime_state["regime"])
            self.regime_times.append(regime_state["time"])
            metrics["regime"] = regime_state["regime"]

        return metrics

    def simulate(self, steps: int = 1000, progress: bool = True) -> None:
        """
        Run the simulation for `steps` iterations.

        Parameters
        ----------
        steps : int
            Number of time steps to simulate.
        progress : bool
            If True, print progress updates.
        """
        for k in range(steps):
            m = self.step()
            if progress and (k % max(1, steps // 20) == 0):
                regime_str = f", regime={m.get('regime', 'N/A')}" if self.cfg.use_regime else ""
                print(
                    f"[{k:5d}/{steps}] t={m['time']:.2f}, "
                    f"precision={m['precision']:.3f}{regime_str}"
                )

    # --------------------------------------------------------
    # STEERING
    # --------------------------------------------------------

    def steer(
        self,
        target: Array,
        precision_delta: float = 0.0,
    ) -> None:
        """
        Steer the Great Attractor to a new position.

        Parameters
        ----------
        target : array-like
            New attractor position (2D).
        precision_delta : float
            Additive adjustment to attractor precision (interaction strength).
        """
        # Update the potential center in the simulator
        target_arr = np.asarray(target, dtype=float)
        # This would need integration with the simulator's potential
        # For now, store the target for visualization
        self.cfg.attractor_pos = target_arr

        if precision_delta != 0.0:
            new_strength = max(0.0, self.cfg.interaction_strength + precision_delta)
            self.cfg.interaction_strength = new_strength

    # --------------------------------------------------------
    # EARLY WARNING / DIAGNOSTICS
    # --------------------------------------------------------

    def _compute_variance(self) -> float:
        """Compute current swarm variance."""
        if not self.position_history:
            return 0.0
        pts = self.position_history[-1]
        return float(np.mean(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)))

    def _compute_autocorrelation(self, lag_steps: int = 1) -> float:
        """Compute autocorrelation of centroid position."""
        if len(self.position_history) <= lag_steps:
            return 0.0
        x_t = self.position_history[-1].mean(axis=0)
        x_prev = self.position_history[-1 - lag_steps].mean(axis=0)
        num = float(np.dot(x_t, x_prev))
        den = np.linalg.norm(x_t) * np.linalg.norm(x_prev) + 1e-12
        return num / den

    def detect_early_warning(
        self,
        tda: bool = True,
        lag_steps: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute early-warning indicators of phase transition / attractor flip.

        Parameters
        ----------
        tda : bool
            If True, compute persistent homology entropy.
        lag_steps : int
            Lag for autocorrelation computation.
        verbose : bool
            If True, print results.

        Returns
        -------
        dict
            Contains variance, autocorrelation, and optionally persistent_entropy.
        """
        variance = self._compute_variance()
        autocorr = self._compute_autocorrelation(lag_steps=lag_steps)

        results: Dict[str, Any] = {
            "variance": variance,
            "autocorrelation": autocorr,
        }

        if tda and self.position_history:
            pts = self.position_history[-1]
            h0, h1 = self.persistence_plotter.compute_pd(pts)
            ent = 0.0
            if len(h0) + len(h1) > 0:
                ent = persistent_entropy(np.vstack([h0, h1]))
            results["persistent_entropy"] = ent

        if verbose:
            print("[Early Warning]", results)

        return results

    # --------------------------------------------------------
    # Q-MATRIX ANALYSIS
    # --------------------------------------------------------

    def analyze_regime_transitions(self) -> Optional[QMatrixResult]:
        """
        Analyze regime transition dynamics using Q-matrix spectral analysis.

        Returns
        -------
        QMatrixResult or None
            Contains Q-matrix, eigenvalues, and spectral gap.
            Returns None if regime layer not enabled.
        """
        if not self.cfg.use_regime or not self.regime_history:
            return None

        # Build transition count matrix from history
        n = self.cfg.n_regimes
        N = np.zeros((n, n))
        dwell_times = np.zeros(n)

        for i in range(1, len(self.regime_history)):
            r_prev = self.regime_history[i - 1]
            r_curr = self.regime_history[i]
            if r_prev != r_curr:
                N[r_prev, r_curr] += 1
            dt = self.regime_times[i] - self.regime_times[i - 1] if i < len(self.regime_times) else self.cfg.dt
            dwell_times[r_prev] += dt

        # Build and analyze Q-matrix
        from research.attractor.regime.q_matrix import estimate_q_from_counts
        Q = estimate_q_from_counts(N, dwell_times)
        return analyze_q(Q)

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    def potential_function(self) -> Callable[[Array], float]:
        """
        Get the current potential function (quadratic around attractor).

        Returns
        -------
        callable
            U(x) -> float potential value.
        """
        center = self.cfg.attractor_pos.copy()

        def U(x: Array) -> float:
            return 0.5 * float(np.sum((x - center) ** 2))

        return U

    def visualize(
        self,
        kind: Literal["basins", "geodesics", "persistence", "animation"],
        **kwargs: Any,
    ) -> None:
        """
        Visualize different aspects of the Great Attractor dynamics.

        Parameters
        ----------
        kind : str
            - 'basins'      : potential landscape + particles
            - 'geodesics'   : geodesic flow + optional potential
            - 'persistence' : persistence diagram + entropy
            - 'animation'   : animated swarm evolution
        **kwargs
            Additional arguments passed to the specific visualizer.
        """
        if kind == "basins":
            self._viz_basins(**kwargs)
        elif kind == "geodesics":
            self._viz_geodesics(**kwargs)
        elif kind == "persistence":
            self._viz_persistence(**kwargs)
        elif kind == "animation":
            self._viz_animation(**kwargs)
        else:
            raise ValueError(f"Unknown visualization kind: {kind}")

    # -- visualization helpers ---------------------------------

    def _viz_basins(self, **kwargs: Any) -> None:
        """Visualize attractor basins with particle overlay."""
        if not self.position_history:
            print("[LatticeForge] No position history. Run simulate() first.")
            return

        positions = self.position_history[-1]
        U = self.potential_function()
        mean = positions.mean(axis=0)
        influence = self.cfg.attractor_pos - mean

        ax = self.basin_plotter.plot(
            U=U,
            particles=positions,
            influence_vec=influence,
            mean=mean,
            title="Great Attractor Basins",
        )
        plt.show()

    def _viz_geodesics(self, **kwargs: Any) -> None:
        """Visualize geodesic flow on information manifold."""
        if not self.position_history:
            print("[LatticeForge] No position history. Run simulate() first.")
            return

        positions = self.position_history[-1]
        cov = np.cov(positions.T) + np.eye(2) * 1e-3

        def metric_fn(x: Array) -> Array:
            # Simple constant Fisher metric ≈ Σ^{-1}
            return np.linalg.inv(cov)

        U = self.potential_function()
        ax = self.geodesic_plotter.plot_geodesics(metric_fn, U=U, particles=positions)
        plt.show()

    def _viz_persistence(self, **kwargs: Any) -> None:
        """Visualize persistence diagram and compute entropy."""
        if not self.position_history:
            print("[LatticeForge] No position history. Run simulate() first.")
            return

        pts = self.position_history[-1]
        self.persistence_plotter.analyze(pts, title="Topological Signature")
        plt.show()

    def _viz_animation(self, save_path: Optional[str] = None, **kwargs: Any) -> None:
        """Animate particle swarm evolution."""
        if not self.position_history:
            print("[LatticeForge] No position history. Run simulate() first.")
            return

        anim_cfg = AnimationConfig(
            xlim=self.cfg.xlim,
            ylim=self.cfg.ylim,
            save_path=save_path,
        )

        animator = ParticleAnimator(
            position_history=self.position_history,
            config=anim_cfg,
            potential_fn=self.potential_function(),
        )
        animator.animate()

    # --------------------------------------------------------
    # SUMMARY / STATUS
    # --------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns
        -------
        dict
            Summary of simulation state.
        """
        return {
            "n_agents": self.cfg.n_agents,
            "steps_run": len(self.position_history),
            "use_regime": self.cfg.use_regime,
            "current_regime": self.regime_history[-1] if self.regime_history else None,
            "attractor_pos": self.cfg.attractor_pos.tolist(),
            "last_variance": self._compute_variance(),
        }
