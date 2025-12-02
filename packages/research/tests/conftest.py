"""
Pytest configuration and fixtures for LatticeForge Research tests.
"""

import numpy as np
import pytest
from numpy.typing import NDArray


Array = NDArray[np.float64]


# ------------------------------------------------------------
# Random seed fixture
# ------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducibility."""
    return 42


# ------------------------------------------------------------
# Basic data fixtures
# ------------------------------------------------------------

@pytest.fixture
def sample_positions(rng: np.random.Generator) -> Array:
    """Generate sample 2D positions for testing."""
    n_particles = 100
    # Two clusters
    cluster1 = rng.normal(loc=[-1, -1], scale=0.3, size=(n_particles // 2, 2))
    cluster2 = rng.normal(loc=[1, 1], scale=0.3, size=(n_particles // 2, 2))
    return np.vstack([cluster1, cluster2])


@pytest.fixture
def sample_time_series(rng: np.random.Generator) -> Array:
    """Generate sample time series for testing."""
    n_steps = 200
    t = np.linspace(0, 10, n_steps)
    # Noisy sine wave with trend
    series = np.sin(t) + 0.1 * t + 0.2 * rng.standard_normal(n_steps)
    return series


@pytest.fixture
def sample_covariance_matrix(rng: np.random.Generator) -> Array:
    """Generate a valid covariance matrix."""
    dim = 4
    A = rng.standard_normal((dim, dim))
    return A @ A.T + np.eye(dim) * 0.1  # Ensure positive definite


@pytest.fixture
def sample_transition_matrix() -> Array:
    """Generate a valid transition probability matrix."""
    # 3-state transition matrix
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
    ])
    return P


# ------------------------------------------------------------
# Potential function fixtures
# ------------------------------------------------------------

@pytest.fixture
def double_well_potential():
    """Double-well potential U(x) = (x1^2 - 1)^2 / 4 + x2^2 / 2."""
    def U(x: Array) -> float:
        return 0.25 * (x[0]**2 - 1)**2 + 0.5 * x[1]**2
    return U


@pytest.fixture
def quadratic_potential():
    """Simple quadratic potential U(x) = 0.5 * ||x||^2."""
    def U(x: Array) -> float:
        return 0.5 * float(np.sum(x**2))
    return U


@pytest.fixture
def multi_basin_potential():
    """Multiple basin potential for regime testing."""
    centers = [
        np.array([-2.0, 0.0]),
        np.array([2.0, 0.0]),
        np.array([0.0, 2.0]),
    ]

    def U(x: Array) -> float:
        # Sum of inverted Gaussians (creates basins)
        val = 0.0
        for c in centers:
            val -= np.exp(-np.sum((x - c)**2))
        return val
    return U


# ------------------------------------------------------------
# Metric function fixtures
# ------------------------------------------------------------

@pytest.fixture
def identity_metric():
    """Identity metric (flat space)."""
    def g(x: Array) -> Array:
        return np.eye(2)
    return g


@pytest.fixture
def fisher_metric(sample_covariance_matrix: Array):
    """Fisher information metric from covariance."""
    cov_inv = np.linalg.inv(sample_covariance_matrix[:2, :2])

    def g(x: Array) -> Array:
        return cov_inv
    return g


# ------------------------------------------------------------
# Configuration fixtures
# ------------------------------------------------------------

@pytest.fixture
def small_sim_config() -> dict:
    """Small simulation config for fast tests."""
    return {
        "n_agents": 50,
        "dt": 0.1,
        "interaction_strength": 1.0,
        "diffusion": 0.1,
    }


@pytest.fixture
def regime_config() -> dict:
    """Regime SDE configuration."""
    return {
        "n_regimes": 3,
        "beta": 1.0,
        "dt": 0.01,
        "diffusion": 0.1,
    }


# ------------------------------------------------------------
# Skip markers
# ------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
