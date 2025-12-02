"""
mckean_vlasov.py

Particle-based McKean–Vlasov ensemble for Great Attractor simulations.

We approximate the mean-field SDE:

    dX_t = -∇U(X_t) dt + (K * Psi)(X_t, t) dt + sqrt(2D) dW_t

by simulating N interacting particles:

    x_i(t+dt) = x_i(t) + drift_i dt + sqrt(2D dt) * noise

with:

    drift_i = -∇U(x_i) + (1/N) sum_j K(x_i, x_j)

This encapsulates:
- Environmental potential U(x)
- Interaction kernel K(x, y) representing "Will coupling"
- Diffusion D (noise level)

The empirical distribution of particles approximates Psi(x, t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass
class MVConfig:
    """
    Configuration for McKean–Vlasov ensemble simulation.
    """
    n_particles: int
    dim: int
    dt: float = 0.01
    diffusion: float = 0.05       # scalar D (isotropic)
    interaction_strength: float = 1.0
    rng_seed: int = 1234


class McKeanVlasovEnsemble:
    """
    Particle-based McKean–Vlasov ensemble.

    Attributes
    ----------
    x : Array, shape (N, d)
        Current particle positions.
    """

    def __init__(
        self,
        config: MVConfig,
        U: Callable[[Array], float],
        grad_U: Callable[[Array], Array],
        K: Callable[[Array, Array], float],
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config
        self.U = U
        self.grad_U = grad_U
        self.K = K

        self.rng = rng if rng is not None else np.random.default_rng(
            config.rng_seed
        )
        self.x = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=(config.n_particles, config.dim),
        )

    def compute_interaction_forces(self) -> Array:
        """
        Compute interaction drift term:

            F_i = (interaction_strength / N) * sum_j ∇_x K(x_i, x_j)

        Here, for simplicity, we use an isotropic kernel:

            K(x, y) = exp(-||x - y||^2 / (2 * sigma_K^2))

        and ∇_x K = K * (y - x) / sigma_K^2
        """
        N, d = self.x.shape
        sigma_K = 1.0
        sigma2 = sigma_K ** 2

        # pairwise interactions
        F = np.zeros_like(self.x)
        for i in range(N):
            diff = self.x - self.x[i]           # (N, d)
            dist2 = np.sum(diff**2, axis=1)     # (N,)
            k_vals = np.exp(-dist2 / (2.0 * sigma2))  # (N,)
            # gradient w.r.t x_i: sum_j K * (x_j - x_i) / sigma^2
            grad_i = (k_vals[:, None] * diff) / sigma2
            F[i] = self.config.interaction_strength * grad_i.mean(axis=0)

        return F

    def step(self) -> None:
        """
        One Euler–Maruyama step of the ensemble.
        """
        dt = self.config.dt
        N, d = self.x.shape

        # Drift from environmental potential
        gradU = np.zeros_like(self.x)
        for i in range(N):
            gradU[i] = self.grad_U(self.x[i])

        # Interaction drift
        F_int = self.compute_interaction_forces()

        # Total drift
        drift = -gradU + F_int

        # Noise
        noise = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(2 * self.config.diffusion * dt),
            size=(N, d),
        )

        self.x = self.x + drift * dt + noise

    def empirical_moments(self) -> Tuple[Array, Array]:
        """
        Compute empirical mean and covariance of the ensemble.
        """
        mean = self.x.mean(axis=0)
        centered = self.x - mean
        cov = (centered.T @ centered) / (self.x.shape[0] - 1)
        return mean, cov
