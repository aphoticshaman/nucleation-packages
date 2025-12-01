"""
ga_ogse_bridge.py

OGSE <-> Great Attractor Simulator bridge.
Allows the OGSE engine to actively steer the McKean–Vlasov ensemble
representing the social latent space.

This turns GA simulations into optimal-control attractor engineering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray

from ..core.great_attractor_sim import GreatAttractorSimulator, GreatAttractorConfig
from ..engine.outcome_steering import (
    OutcomeGradientSteeringEngine,
    OGSEConfig,
    PotentialModel,
)


Array = NDArray[np.float64]


@dataclass
class GAOGSEBridgeConfig:
    latent_dim: int = 2
    dt: float = 0.01
    horizon: int = 50
    max_iters: int = 50
    control_clip: float = 0.2
    ogse_random_seed: int = 2024


class GAOGSEBridge:
    """
    Bridge OGSEEngine and GreatAttractorSimulator so that:

    - OGSE sees the ensemble mean as the "state z"
    - OGSE chooses control u_t as influence vector on latent space
    - GA simulator applies u_t as external drift over MV ensemble
    """

    def __init__(
        self,
        ga_sim: GreatAttractorSimulator,
        config: GAOGSEBridgeConfig,
    ) -> None:
        self.ga = ga_sim
        self.cfg = config

        # --- create OGSE engine ---
        latent_dim = config.latent_dim
        control_dim = latent_dim  # influences latent coordinates directly

        def dynamics(z: Array, u: Array) -> Array:
            """Simple integrator: z_{t+1} = z_t + u."""
            return z + u * self.cfg.dt

        # Quadratic potential: want to steer ensemble mean towards target
        self.target = np.zeros(latent_dim)

        ogse_cfg = OGSEConfig(
            horizon=self.cfg.horizon,
            max_iters=self.cfg.max_iters,
            step_size=0.05,
            control_clip=self.cfg.control_clip,
            random_seed=self.cfg.ogse_random_seed,
        )

        self.ogse = OutcomeGradientSteeringEngine(
            state_dim=latent_dim,
            action_dim=control_dim,
            dynamics=dynamics,
            config=ogse_cfg,
            constraints=[],
        )

        # Create potential model for optimization
        def V_state(z: Array, t: int) -> float:
            return float(0.5 * np.linalg.norm(z - self.target) ** 2)

        def V_control(u: Array, t: int) -> float:
            # light penalty for excessive influence
            return float(0.05 * np.linalg.norm(u) ** 2)

        self.potential = PotentialModel(
            state_potential=V_state,
            control_cost=V_control,
        )

    def set_target(self, target: Array) -> None:
        """Set the target state for OGSE steering."""
        self.target = target.copy()

        # Recreate potential with new target
        def V_state(z: Array, t: int) -> float:
            return float(0.5 * np.linalg.norm(z - self.target) ** 2)

        def V_control(u: Array, t: int) -> float:
            return float(0.05 * np.linalg.norm(u) ** 2)

        self.potential = PotentialModel(
            state_potential=V_state,
            control_cost=V_control,
        )

    # ----------------------------------------------------
    # ACTIVE STEERING OF GREAT ATTRACTOR
    # ----------------------------------------------------

    def step(self, t_step: int) -> Dict[str, Any]:
        """
        One step of integrated OGSE+GA control.

        1. Observe the GA ensemble (mean state).
        2. Run OGSE optimization to choose influence u_t.
        3. Apply influence to GA simulator.
        """

        # 1. extract ensemble mean as OGSE state
        mean, cov = self.ga.mv.empirical_moments()
        z = mean.astype(np.float64)

        # 2. run OGSE optimization (single-step rolling horizon)
        controls_opt, states_opt, cost = self.ogse.optimize_controls(
            z0=z,
            potential=self.potential,
            u_init=None,
        )

        # Extract the first control in the optimized sequence
        u_t = controls_opt[0]

        # Apply small influence to all particles
        self.ga.mv.x += u_t * self.cfg.dt

        # Advance GA simulator one MV step (agents push separately)
        avg_dir, avg_F = self.ga.agent_collective_action()
        self.ga.mv.x += avg_dir * self.ga.config.dt
        self.ga.mv.step()

        return {
            "z": z,
            "u_t": u_t,
            "ogse_cost": cost,
            "avg_free_energy": avg_F,
        }

    def run_steering_loop(
        self,
        steps: int,
        tda_monitor: Optional[Any] = None,
        log_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Run full OGSE-controlled GA simulation.

        Parameters
        ----------
        steps : int
            Number of simulation steps
        tda_monitor : TDAMonitor, optional
            If provided, updates TDA early warning indicators
        log_every : int
            Logging frequency

        Returns
        -------
        history : dict with lists of z, u_t, costs, etc.
        """
        history = {
            "z": [],
            "u_t": [],
            "ogse_cost": [],
            "avg_free_energy": [],
        }

        for t_step in range(steps):
            res = self.step(t_step)

            history["z"].append(res["z"])
            history["u_t"].append(res["u_t"])
            history["ogse_cost"].append(res["ogse_cost"])
            history["avg_free_energy"].append(res["avg_free_energy"])

            # TDA monitoring
            if tda_monitor is not None and t_step % tda_monitor.config.compute_every == 0:
                mean, cov = self.ga.mv.empirical_moments()
                tda_monitor.update(t_step * self.cfg.dt, self.ga.mv.x, cov)

            # Logging
            if (t_step + 1) % log_every == 0:
                print(
                    f"[step={t_step+1}] z={res['z']}, "
                    f"||u||={np.linalg.norm(res['u_t']):.4f}, "
                    f"cost={res['ogse_cost']:.4f}"
                )

        return history


# --- Example usage ---
#
# from research.core.great_attractor_sim import GreatAttractorSimulator, GreatAttractorConfig
# from research.attractor.ga_ogse_bridge import GAOGSEBridge, GAOGSEBridgeConfig
# from research.attractor.tda_monitor import TDAMonitor, TDAMonitorConfig
#
# # build GA simulator
# ga_cfg = GreatAttractorConfig(steps=2000)
# ga = GreatAttractorSimulator(ga_cfg)
#
# # build OGSE bridge (active steering)
# bridge_cfg = GAOGSEBridgeConfig(latent_dim=2)
# bridge = GAOGSEBridge(ga, bridge_cfg)
#
# # optionally set a target
# bridge.set_target(np.array([1.0, 0.0]))  # steer toward right well
#
# # build TDA monitor
# tda_cfg = TDAMonitorConfig(max_dimension=1, compute_every=20)
# tda = TDAMonitor(tda_cfg, dim=2)
#
# # run steering loop
# history = bridge.run_steering_loop(steps=2000, tda_monitor=tda, log_every=200)
#
# # Analyze early warning signals
# ews = tda.detect_early_warning()
# print("Early-warning metrics:", ews)
