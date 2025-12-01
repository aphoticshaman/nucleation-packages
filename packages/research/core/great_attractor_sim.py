"""
great_attractor_sim.py

Top-level orchestration for Great Attractor simulations.

This ties together:
- FEP agents (micro Active Inference dynamics),
- a McKean–Vlasov ensemble (mean-field collective dynamics),
- and simple diagnostics for "Great Attractor" formation.

The simplest experiment:
- A 2D latent space x \\in R^2.
- Environmental potential U(x) with one or more wells.
- FEP agents reading noisy sensory projections of x and
  pushing x in directions that minimize their free energy.
- Mean-field interactions coupling agents via a kernel K.

This module can be used in notebooks or scripts as:

    from research.core.great_attractor_sim import GreatAttractorSimulator

    sim = GreatAttractorSimulator(...)
    history = sim.run(T=1000)

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .fep_agent import FEPAgent, FEPAgentConfig
from .mckean_vlasov import McKeanVlasovEnsemble, MVConfig


Array = NDArray[np.float64]


@dataclass
class GreatAttractorConfig:
    """
    High-level configuration for Great Attractor simulations.
    """
    dim_latent: int = 2
    n_agents: int = 50
    n_particles: int = 500
    dt: float = 0.01
    steps: int = 2000
    agent_action_scale: float = 0.05
    log_every: int = 50
    rng_seed: int = 999


@dataclass
class GreatAttractorHistory:
    """Stores time series of aggregate diagnostics."""
    means: List[Array] = field(default_factory=list)
    covs: List[Array] = field(default_factory=list)
    free_energies: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)


class GreatAttractorSimulator:
    """
    Orchestrates FEP agents interacting with a McKean–Vlasov ensemble,
    in a shared latent space x.

    Design simplification:
    - We approximate external latent x with the particle ensemble itself
      (the population lives in the same latent as the agents act on).
    - Agents observe their "local" x via noisy sensory fn,
      update internal beliefs, then push x collectively.
    """

    def __init__(
        self,
        config: GreatAttractorConfig,
        mv_diffusion: float = 0.02,
        mv_interaction_strength: float = 1.0,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.rng_seed)

        # --- Environmental potential U(x) = double-well style in 2D ---

        def U(x: Array) -> float:
            """
            Example potential:
                U(x) = 1/4 (x1^2 - 1)^2 + 1/2 x2^2
            Double-well in x1, quadratic in x2.
            """
            x1, x2 = x[0], x[1] if x.shape[0] > 1 else (x[0], 0.0)
            return 0.25 * (x1**2 - 1.0) ** 2 + 0.5 * x2**2

        def grad_U(x: Array) -> Array:
            """
            Gradient of U:
                dU/dx1 = (x1^2 - 1) * x1
                dU/dx2 = x2
            """
            if x.shape[0] < 2:
                # 1D case fallback
                x1 = x[0]
                return np.array([(x1**2 - 1.0) * x1], dtype=np.float64)
            x1, x2 = x[0], x[1]
            return np.array([(x1**2 - 1.0) * x1, x2], dtype=np.float64)

        def K(x: Array, y: Array) -> float:
            """Isotropic Gaussian kernel; used indirectly by MV ensemble."""
            diff = x - y
            return float(np.exp(-np.dot(diff, diff) / (2.0 * 1.0**2)))

        # --- McKean–Vlasov ensemble ---

        mv_cfg = MVConfig(
            n_particles=config.n_particles,
            dim=config.dim_latent,
            dt=config.dt,
            diffusion=mv_diffusion,
            interaction_strength=mv_interaction_strength,
            rng_seed=config.rng_seed + 1,
        )

        self.mv = McKeanVlasovEnsemble(
            config=mv_cfg,
            U=U,
            grad_U=grad_U,
            K=K,
            rng=self.rng,
        )

        # --- FEP agents ---

        self.agents: List[FEPAgent] = []
        for i in range(config.n_agents):
            agent_cfg = FEPAgentConfig(
                state_dim=config.dim_latent,
                sensory_dim=config.dim_latent,
                dt=config.dt,
                sigma_mu=0.05,
                sigma_s=0.05,
                precision_s=25.0,
                precision_mu=10.0,
                stiffness_prior=1.0,
            )
            mu_init = self.rng.normal(
                loc=0.0,
                scale=0.5,
                size=(config.dim_latent,),
            )
            agent = FEPAgent(
                config=agent_cfg,
                mu_init=mu_init,
                rng=self.rng,
            )
            self.agents.append(agent)

    # --- Coupling between agents and latent ensemble ---

    def sensory_fn(self, x: Array) -> Array:
        """
        Map latent x to sensory space. Here: identity projection, plus
        small observation noise.
        """
        noise = self.rng.normal(scale=0.05, size=x.shape)
        return x + noise

    def agent_collective_action(self) -> Tuple[Array, float]:
        """
        Compute a collective action direction on the latent space
        based on all agents' action directions.

        Strategy:
        - sample M particles from the ensemble as "local states" seen by agents
        - each agent computes its desired action direction for its local x
        - aggregate directions as the mean vector
        """
        if self.mv.x.shape[0] == 0:
            return np.zeros(self.config.dim_latent), 0.0

        # sample as many local x's as agents, with replacement
        idx = self.rng.integers(
            low=0, high=self.mv.x.shape[0], size=len(self.agents)
        )
        local_xs = self.mv.x[idx]

        dirs = []
        Fs = []
        for agent, x_loc in zip(self.agents, local_xs):
            s = self.sensory_fn(x_loc)
            dir_x, F = agent.step(x_loc, sensory_fn=lambda _: s)
            dirs.append(dir_x)
            Fs.append(F)

        dirs_arr = np.stack(dirs, axis=0)
        avg_dir = dirs_arr.mean(axis=0)
        avg_F = float(np.mean(Fs))

        # scale to bounded norm
        norm = np.linalg.norm(avg_dir) + 1e-12
        if norm > 0:
            avg_dir = (
                self.config.agent_action_scale * avg_dir / norm
            )

        return avg_dir, avg_F

    # --- Simulation loop ---

    def run(self, T: Optional[int] = None) -> GreatAttractorHistory:
        """
        Run Great Attractor simulation.

        Parameters
        ----------
        T : int, optional
            Number of time steps. If None, uses config.steps.

        Returns
        -------
        history : GreatAttractorHistory
        """
        steps = T if T is not None else self.config.steps
        hist = GreatAttractorHistory()

        t = 0.0
        for step in range(steps):
            # 1. Agent collective action (average direction in latent space)
            avg_dir, avg_F = self.agent_collective_action()

            # 2. Inject that as a drift bias into the ensemble for this step
            #    (we treat it as an external "control" u_t)
            #    So we hack: move all particles by avg_dir * dt, then MV step.
            self.mv.x += avg_dir * self.config.dt

            # 3. Evolve ensemble by one McKean–Vlasov step
            self.mv.step()

            # 4. Record summary statistics
            mean, cov = self.mv.empirical_moments()
            hist.means.append(mean)
            hist.covs.append(cov)
            hist.free_energies.append(avg_F)
            hist.times.append(t)

            t += self.config.dt

            if (step + 1) % self.config.log_every == 0:
                var = np.trace(cov)
                print(
                    f"[t={t:.3f}] step={step+1} "
                    f"mean={mean}, var_trace={var:.4f}, F~{avg_F:.3f}"
                )

        return hist


# --- Example usage ---
#
# from research.core.great_attractor_sim import (
#     GreatAttractorSimulator,
#     GreatAttractorConfig,
# )
#
# cfg = GreatAttractorConfig(
#     dim_latent=2,
#     n_agents=30,
#     n_particles=400,
#     dt=0.01,
#     steps=1000,
#     agent_action_scale=0.1,
#     log_every=100,
# )
#
# sim = GreatAttractorSimulator(cfg)
# history = sim.run()
#
# # Example: inspect variance over time as basin formation indicator
# import numpy as np
# var_trace = [np.trace(c) for c in history.covs]
# import matplotlib.pyplot as plt
# plt.plot(history.times, var_trace)
# plt.xlabel("time")
# plt.ylabel("trace(cov)")
# plt.title("Ensemble variance (Great Attractor formation)")
# plt.show()
