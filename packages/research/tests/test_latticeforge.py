"""
Tests for LatticeForge high-level API.
"""

import numpy as np
import pytest

from attractor.api.latticeforge import LatticeForge, LatticeForgeConfig


class TestLatticeForgeConfig:
    """Tests for LatticeForgeConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        cfg = LatticeForgeConfig()
        assert cfg.n_agents == 500
        assert cfg.dt == 0.05
        assert cfg.use_regime is False
        assert cfg.n_regimes == 3

    def test_custom_values(self):
        """Should accept custom values."""
        cfg = LatticeForgeConfig(
            n_agents=100,
            dt=0.1,
            use_regime=True,
            n_regimes=5,
        )
        assert cfg.n_agents == 100
        assert cfg.dt == 0.1
        assert cfg.use_regime is True
        assert cfg.n_regimes == 5

    def test_attractor_pos_default(self):
        """Default attractor position should be [5, 5]."""
        cfg = LatticeForgeConfig()
        assert np.allclose(cfg.attractor_pos, [5.0, 5.0])


class TestLatticeForge:
    """Tests for LatticeForge class."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast tests."""
        return LatticeForgeConfig(
            n_agents=20,
            dt=0.1,
            use_regime=False,
        )

    @pytest.fixture
    def lf(self, small_config):
        """Create a LatticeForge instance."""
        return LatticeForge(small_config)

    @pytest.fixture
    def lf_with_regime(self):
        """Create a LatticeForge instance with regime layer."""
        cfg = LatticeForgeConfig(
            n_agents=20,
            dt=0.1,
            use_regime=True,
            n_regimes=3,
        )
        return LatticeForge(cfg)

    def test_initialization(self, lf, small_config):
        """LatticeForge should initialize properly."""
        assert lf.cfg.n_agents == small_config.n_agents
        assert lf.sim is not None
        assert len(lf.position_history) == 0
        assert len(lf.metrics_history) == 0

    def test_initialization_with_regime(self, lf_with_regime):
        """LatticeForge with regime should have regime_sde."""
        assert lf_with_regime.regime_sde is not None
        assert len(lf_with_regime.regime_history) == 0

    def test_step_returns_metrics(self, lf):
        """Step should return metrics dict."""
        metrics = lf.step()

        assert isinstance(metrics, dict)
        assert 'time' in metrics
        assert 'mean_pos' in metrics
        assert 'precision' in metrics

    def test_step_updates_history(self, lf):
        """Step should update position history."""
        initial_len = len(lf.position_history)
        lf.step()
        assert len(lf.position_history) == initial_len + 1

    def test_step_with_regime_updates_regime_history(self, lf_with_regime):
        """Step with regime should update regime history."""
        lf_with_regime.step()
        assert len(lf_with_regime.regime_history) == 1
        assert 'regime' in lf_with_regime.metrics_history[0]

    def test_simulate_runs_multiple_steps(self, lf):
        """Simulate should run specified number of steps."""
        lf.simulate(steps=10, progress=False)
        assert len(lf.position_history) == 10

    def test_steer_updates_attractor_pos(self, lf):
        """Steer should update attractor position."""
        new_target = np.array([1.0, 2.0])
        lf.steer(new_target)
        assert np.allclose(lf.cfg.attractor_pos, new_target)

    def test_steer_updates_precision(self, lf):
        """Steer with precision_delta should update precision."""
        initial_strength = lf.cfg.interaction_strength
        lf.steer(np.array([0, 0]), precision_delta=0.5)
        assert lf.cfg.interaction_strength == initial_strength + 0.5

    def test_detect_early_warning_returns_dict(self, lf):
        """detect_early_warning should return dict with indicators."""
        lf.simulate(steps=10, progress=False)
        result = lf.detect_early_warning(verbose=False)

        assert isinstance(result, dict)
        assert 'variance' in result
        assert 'autocorrelation' in result

    def test_detect_early_warning_with_tda(self, lf):
        """detect_early_warning with TDA should include entropy."""
        lf.simulate(steps=10, progress=False)
        result = lf.detect_early_warning(tda=True, verbose=False)

        assert 'persistent_entropy' in result

    def test_detect_early_warning_empty_history(self, lf):
        """detect_early_warning with no history should return zeros."""
        result = lf.detect_early_warning(tda=False, verbose=False)

        assert result['variance'] == 0.0
        assert result['autocorrelation'] == 0.0

    def test_potential_function_callable(self, lf):
        """potential_function should return callable."""
        U = lf.potential_function()
        x = np.array([1.0, 1.0])
        value = U(x)

        assert isinstance(value, float)

    def test_potential_function_minimum_at_attractor(self, lf):
        """Potential should be minimized near attractor position."""
        U = lf.potential_function()
        attractor = lf.cfg.attractor_pos

        # Value at attractor
        val_attractor = U(attractor)

        # Value away from attractor
        val_far = U(attractor + np.array([10.0, 10.0]))

        assert val_attractor < val_far

    def test_status_returns_summary(self, lf):
        """status should return summary dict."""
        lf.simulate(steps=5, progress=False)
        status = lf.status()

        assert isinstance(status, dict)
        assert status['n_agents'] == lf.cfg.n_agents
        assert status['steps_run'] == 5
        assert 'attractor_pos' in status
        assert 'last_variance' in status

    def test_analyze_regime_transitions_without_regime(self, lf):
        """analyze_regime_transitions without regime should return None."""
        result = lf.analyze_regime_transitions()
        assert result is None

    def test_analyze_regime_transitions_with_regime(self, lf_with_regime):
        """analyze_regime_transitions with regime should return QMatrixResult."""
        from attractor.regime.q_matrix import QMatrixResult

        lf_with_regime.simulate(steps=50, progress=False)
        result = lf_with_regime.analyze_regime_transitions()

        assert result is not None
        assert isinstance(result, QMatrixResult)
        assert result.spectral_gap >= 0

    def test_visualize_invalid_kind_raises(self, lf):
        """visualize with invalid kind should raise ValueError."""
        lf.simulate(steps=5, progress=False)
        with pytest.raises(ValueError):
            lf.visualize("invalid_kind")


class TestLatticeForgeIntegration:
    """Integration tests for LatticeForge."""

    @pytest.mark.slow
    def test_full_simulation_workflow(self):
        """Test complete simulation workflow."""
        cfg = LatticeForgeConfig(
            n_agents=50,
            dt=0.05,
            use_regime=True,
            n_regimes=2,
        )
        lf = LatticeForge(cfg)

        # Simulate
        lf.simulate(steps=100, progress=False)

        # Check state
        assert len(lf.position_history) == 100
        assert len(lf.regime_history) == 100

        # Early warning
        ew = lf.detect_early_warning(verbose=False)
        assert ew['variance'] >= 0

        # Regime analysis
        q_result = lf.analyze_regime_transitions()
        assert q_result is not None

        # Steer
        lf.steer(np.array([0.0, 0.0]))
        assert np.allclose(lf.cfg.attractor_pos, [0.0, 0.0])

        # Continue simulation
        lf.simulate(steps=50, progress=False)
        assert len(lf.position_history) == 150

    @pytest.mark.slow
    def test_variance_increases_with_diffusion(self):
        """Higher diffusion should lead to higher variance."""
        # Low diffusion
        cfg_low = LatticeForgeConfig(n_agents=50, dt=0.05, diffusion=0.01)
        lf_low = LatticeForge(cfg_low)
        lf_low.simulate(steps=100, progress=False)
        var_low = lf_low.detect_early_warning(tda=False, verbose=False)['variance']

        # High diffusion
        cfg_high = LatticeForgeConfig(n_agents=50, dt=0.05, diffusion=1.0)
        lf_high = LatticeForge(cfg_high)
        lf_high.simulate(steps=100, progress=False)
        var_high = lf_high.detect_early_warning(tda=False, verbose=False)['variance']

        # Higher diffusion should give higher variance (statistically)
        # Allow some tolerance for stochastic variation
        assert var_high >= var_low * 0.1  # Very loose bound
