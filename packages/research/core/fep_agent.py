"""
fep_agent.py

Active Inference Agent for Great Attractor simulations.

Implements a simplified continuous-time Active Inference agent
under a Laplace approximation:

    s = g(psi) + noise
    dpsi/dt = f(psi) + noise

With variational free energy in the quadratic form:

    F ≈ 1/2 (eps_s^T Pi_s eps_s + eps_mu^T Pi_mu eps_mu)

Perception update:
    dmu/dt = D mu - (Pi_mu eps_mu - (dg/dmu)^T Pi_s eps_s)

Action update:
    da/dt = - (ds/da)^T Pi_s eps_s

For simulations here we:
- treat mu as the internal state,
- approximate action as directly modifying a low-dim external latent x,
- expose a step() integrating mu and returning the "desired" action
  direction on the external state x.

This is a research scaffold, not a full cognitive model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass
class FEPAgentConfig:
    """Configuration for an Active Inference agent."""
    state_dim: int
    sensory_dim: int
    dt: float = 0.01
    sigma_mu: float = 0.05         # internal process noise std
    sigma_s: float = 0.05          # sensory noise std
    precision_s: float = 25.0      # scalar sensory precision
    precision_mu: float = 10.0     # scalar prior precision
    stiffness_prior: float = 1.0   # strength of dynamic prior f(mu) ~ -k * mu


class FEPAgent:
    """
    Minimal Active Inference agent.

    Attributes
    ----------
    mu : Array, shape (state_dim,)
        Internal belief state.
    config : FEPAgentConfig
        Hyperparameters.
    """

    def __init__(
        self,
        config: FEPAgentConfig,
        mu_init: Optional[Array] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        if mu_init is None:
            self.mu = self.rng.normal(loc=0.0, scale=0.5,
                                      size=(config.state_dim,))
        else:
            self.mu = mu_init.astype(np.float64)

        # Simple diagonal precisions
        self.Pi_s = config.precision_s * np.eye(config.sensory_dim)
        self.Pi_mu = config.precision_mu * np.eye(config.state_dim)

    # --- Generative model components ---

    def g(self, psi: Array) -> Array:
        """
        Sensory mapping: s = g(psi).
        Here we assume a simple linear observation:
            s = H psi
        with H the identity (or projection if dims differ).
        """
        d_state = self.config.state_dim
        d_sens = self.config.sensory_dim
        if d_sens == d_state:
            H = np.eye(d_state)
        else:
            # project into sensory_dim via fixed random projection
            self._H = getattr(self, "_H", self.rng.normal(
                size=(d_sens, d_state)
            ) / np.sqrt(d_state))
            H = self._H
        return H @ psi

    def f(self, psi: Array) -> Array:
        """
        Dynamic prior: dpsi/dt = f(psi).
        Here a simple linear "spring" toward 0:
            f(psi) = -k * psi
        """
        k = self.config.stiffness_prior
        return -k * psi

    def dg_dmu(self, mu: Array) -> Array:
        """
        Jacobian of g with respect to mu.
        For linear g, it's constant (= H).
        """
        d_state = self.config.state_dim
        d_sens = self.config.sensory_dim
        if d_sens == d_state:
            H = np.eye(d_state)
        else:
            self._H = getattr(self, "_H", self.rng.normal(
                size=(d_sens, d_state)
            ) / np.sqrt(d_state))
            H = self._H
        return H

    # --- Core Free-Energy machinery ---

    def compute_errors(
        self,
        s: Array,
        mu: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Compute sensory and prior prediction errors.

        eps_s = s - g(mu)
        eps_mu = dmu/dt - f(mu)  (we approximate dmu/dt ~ 0 here for simplicity)
        """
        if mu is None:
            mu = self.mu

        eps_s = s - self.g(mu)
        # For simplicity in discrete-time, approximate eps_mu = -f(mu)
        eps_mu = -self.f(mu)
        return eps_s, eps_mu

    def free_energy(
        self,
        s: Array,
        mu: Optional[Array] = None,
    ) -> float:
        """Approximate variational free energy for diagnostics."""
        if mu is None:
            mu = self.mu
        eps_s, eps_mu = self.compute_errors(s, mu)
        F = 0.5 * (
            eps_s.T @ self.Pi_s @ eps_s +
            eps_mu.T @ self.Pi_mu @ eps_mu
        )
        # ignore log det terms for ranking, they are constant if sigmas fixed
        return float(F)

    # --- Update rules ---

    def perception_step(self, s: Array) -> None:
        """
        Gradient descent on F w.r.t. mu (perception).

        dmu/dt ~= - dF/dmu = -(Pi_mu eps_mu - dg^T Pi_s eps_s)
        """
        dt = self.config.dt
        eps_s, eps_mu = self.compute_errors(s, self.mu)
        H = self.dg_dmu(self.mu)

        grad_mu = self.Pi_mu @ eps_mu - H.T @ self.Pi_s @ eps_s
        # Euler step with noise
        noise = self.rng.normal(scale=self.config.sigma_mu,
                                size=self.mu.shape)
        self.mu = self.mu - dt * grad_mu + np.sqrt(dt) * noise

    def action_direction(
        self,
        s: Array,
        jac_s_wrt_x: Optional[Array] = None,
    ) -> Array:
        """
        Compute direction in external latent x that would reduce sensory error.

        da/dt = - (ds/da)^T Pi_s eps_s
        For simplicity we treat action ~ delta x, and ds/dx ~ jac_s_wrt_x.
        If jac_s_wrt_x is None, we assume identity in x-space of sensory dim.
        """
        eps_s, _ = self.compute_errors(s, self.mu)
        d_s = self.config.sensory_dim

        if jac_s_wrt_x is None:
            # assume the external latent x has same dim as sensory
            J = np.eye(d_s)
        else:
            J = jac_s_wrt_x

        # Direction in x-space that reduces sensory error
        dir_x = - J.T @ self.Pi_s @ eps_s
        return dir_x

    def step(
        self,
        x: Array,
        sensory_fn: Optional[Callable[[Array], Array]] = None,
        jac_s_wrt_x: Optional[Array] = None,
    ) -> Tuple[Array, float]:
        """
        Full active-inference update for one time step:

        1. Observe s from x via sensory_fn (or identity).
        2. Update mu (perception step).
        3. Compute action_direction in x-space.

        Returns
        -------
        action_dir : Array
            Proposed delta x direction (not yet scaled).
        free_energy_value : float
            Current approximate F.
        """
        if sensory_fn is None:
            # default: identity mapping from x to s if dims match
            s = x[: self.config.sensory_dim]
        else:
            s = sensory_fn(x)

        # perception
        self.perception_step(s)

        # action direction on x
        dir_x = self.action_direction(s, jac_s_wrt_x=jac_s_wrt_x)
        F = self.free_energy(s, self.mu)
        return dir_x, F
