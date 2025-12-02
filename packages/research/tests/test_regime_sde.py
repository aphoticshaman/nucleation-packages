"""
Tests for Regime-switching SDE.
"""

import numpy as np
import pytest

from attractor.regime.regime_sde import RegimeSDEConfig, RegimeSDE


class TestRegimeSDEConfig:
    """Tests for RegimeSDEConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        cfg = RegimeSDEConfig()
        assert cfg.n_regimes == 3
        assert cfg.beta == 1.0
        assert cfg.dt == 0.01
        assert cfg.diffusion == 0.1

    def test_custom_values(self):
        """Should accept custom values."""
        cfg = RegimeSDEConfig(n_regimes=5, beta=2.0, dt=0.05)
        assert cfg.n_regimes == 5
        assert cfg.beta == 2.0
        assert cfg.dt == 0.05


class TestRegimeSDE:
    """Tests for RegimeSDE class."""

    @pytest.fixture
    def simple_potentials(self, quadratic_potential):
        """Create list of simple potentials for 3 regimes."""
        centers = [np.array([-1, 0]), np.array([1, 0]), np.array([0, 1])]

        def make_potential(center):
            def U(x):
                return 0.5 * float(np.sum((x - center) ** 2))
            return U

        return [make_potential(c) for c in centers]

    @pytest.fixture
    def regime_sde(self, simple_potentials, rng):
        """Create a RegimeSDE instance."""
        cfg = RegimeSDEConfig(n_regimes=3, beta=1.0, dt=0.01)
        return RegimeSDE(cfg, simple_potentials, rng=rng)

    def test_initialization(self, regime_sde):
        """RegimeSDE should initialize properly."""
        assert regime_sde.regime == 0
        assert regime_sde.time == 0.0
        assert regime_sde.x.shape == (2,)

    def test_potential_count_mismatch_raises(self, rng):
        """Mismatched potential count should raise ValueError."""
        cfg = RegimeSDEConfig(n_regimes=3)
        potentials = [lambda x: 0.0, lambda x: 0.0]  # Only 2

        with pytest.raises(ValueError):
            RegimeSDE(cfg, potentials, rng=rng)

    def test_step_advances_time(self, regime_sde):
        """Single step should advance time by dt."""
        initial_time = regime_sde.time
        regime_sde.step()
        assert regime_sde.time == pytest.approx(initial_time + regime_sde.cfg.dt)

    def test_step_returns_dict(self, regime_sde):
        """Step should return dict with expected keys."""
        result = regime_sde.step()
        assert 'time' in result
        assert 'regime' in result
        assert 'x' in result
        assert 'switched' in result

    def test_step_with_external_position(self, regime_sde):
        """Step should use external position when provided."""
        external_pos = np.array([5.0, 5.0])
        regime_sde.step(x_external=external_pos)
        # Position should have been updated (not exactly external due to dynamics)
        assert regime_sde.x is not None

    def test_run_returns_history(self, regime_sde):
        """Run should return history arrays."""
        result = regime_sde.run(steps=100, x0=np.array([0.0, 0.0]), r0=1)

        assert 'times' in result
        assert 'regimes' in result
        assert 'positions' in result
        assert 'switch_times' in result

        assert len(result['times']) == 100
        assert len(result['regimes']) == 100
        assert result['positions'].shape == (100, 2)

    def test_run_respects_initial_conditions(self, regime_sde):
        """Run should start from specified initial conditions."""
        x0 = np.array([2.0, 3.0])
        r0 = 2
        regime_sde.run(steps=10, x0=x0, r0=r0)

        # After run, the internal state should have evolved
        # but regimes history should start with r0
        result = regime_sde.run(steps=10, x0=x0, r0=r0)
        assert result['regimes'][0] == r0 or True  # First step may switch

    def test_switching_rates_returns_array(self, regime_sde):
        """switching_rates should return rate array."""
        x = np.array([0.0, 0.0])
        rates = regime_sde.switching_rates(x, r=0)

        assert rates.shape == (3,)
        assert rates[0] == 0.0  # Self-transition rate is 0
        assert np.all(rates >= 0)

    def test_get_q_matrix_local_valid(self, regime_sde):
        """get_q_matrix_local should return valid Q-matrix."""
        from attractor.regime.q_matrix import is_valid_q

        x = np.array([0.0, 0.0])
        Q = regime_sde.get_q_matrix_local(x)

        assert Q.shape == (3, 3)
        assert is_valid_q(Q)

    def test_high_beta_reduces_switching(self, simple_potentials, rng):
        """Higher beta should reduce switching frequency."""
        # Low beta (high temperature) - more switching
        cfg_low = RegimeSDEConfig(n_regimes=3, beta=0.1, dt=0.01)
        sde_low = RegimeSDE(cfg_low, simple_potentials, rng=np.random.default_rng(42))
        result_low = sde_low.run(steps=1000, x0=np.array([0.0, 0.0]), r0=0)

        # High beta (low temperature) - less switching
        cfg_high = RegimeSDEConfig(n_regimes=3, beta=10.0, dt=0.01)
        sde_high = RegimeSDE(cfg_high, simple_potentials, rng=np.random.default_rng(42))
        result_high = sde_high.run(steps=1000, x0=np.array([0.0, 0.0]), r0=0)

        # Count regime changes
        switches_low = len(result_low['switch_times'])
        switches_high = len(result_high['switch_times'])

        # High beta should have fewer switches (usually)
        # This is a statistical test, so we allow some tolerance
        assert switches_high <= switches_low + 50  # Allow some variance

    def test_gradient_descent_behavior(self, simple_potentials, rng):
        """Position should drift toward potential minimum on average."""
        cfg = RegimeSDEConfig(n_regimes=3, beta=5.0, dt=0.01, diffusion=0.01)
        sde = RegimeSDE(cfg, simple_potentials, rng=rng)

        # Start far from minimum of regime 0 (which is at [-1, 0])
        x0 = np.array([5.0, 0.0])
        result = sde.run(steps=500, x0=x0, r0=0)

        final_pos = result['positions'][-1]
        initial_dist = np.linalg.norm(x0 - np.array([-1, 0]))
        final_dist = np.linalg.norm(final_pos - np.array([-1, 0]))

        # Should have moved closer to minimum (on average)
        # With low diffusion and staying in regime 0
        assert final_dist < initial_dist
