"""
Tests for TDA early warning monitor.
"""

import numpy as np
import pytest

from attractor.tda_monitor import (
    TDAMonitorConfig,
    TDAMonitorHistory,
    TDAMonitor,
    autocorrelation,
    persistent_entropy,
)


class TestAutocorrelation:
    """Tests for autocorrelation function."""

    def test_constant_series_autocorr_one(self):
        """Constant series should have autocorrelation 1."""
        series = np.ones(100)
        ac = autocorrelation(series, lag=1)
        assert ac == pytest.approx(1.0, abs=0.01)

    def test_white_noise_low_autocorr(self, rng):
        """White noise should have low autocorrelation."""
        noise = rng.standard_normal(1000)
        ac = autocorrelation(noise, lag=1)
        assert abs(ac) < 0.1  # Should be close to 0

    def test_lag_zero_is_one(self, rng):
        """Autocorrelation at lag 0 should be 1."""
        series = rng.standard_normal(100) + np.linspace(0, 10, 100)
        ac = autocorrelation(series, lag=0)
        assert ac == pytest.approx(1.0)

    def test_sine_wave_periodic_autocorr(self):
        """Sine wave should have periodic autocorrelation."""
        t = np.linspace(0, 4 * np.pi, 200)
        series = np.sin(t)

        # At half period, autocorrelation should be negative
        half_period = 50  # Approximate
        ac_half = autocorrelation(series, lag=half_period)
        assert ac_half < 0

        # At full period, should be positive
        full_period = 100
        ac_full = autocorrelation(series, lag=full_period)
        assert ac_full > 0.5

    def test_ar1_process_positive_autocorr(self, rng):
        """AR(1) process should have positive autocorrelation."""
        n = 500
        phi = 0.8
        series = np.zeros(n)
        series[0] = rng.standard_normal()
        for i in range(1, n):
            series[i] = phi * series[i-1] + rng.standard_normal()

        ac = autocorrelation(series, lag=1)
        assert ac > 0.5  # Should be close to phi


class TestTDAMonitorConfig:
    """Tests for TDAMonitorConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        cfg = TDAMonitorConfig()
        assert cfg.window_size > 0
        assert cfg.n_lags > 0


class TestTDAMonitor:
    """Tests for TDAMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create a TDAMonitor instance."""
        cfg = TDAMonitorConfig(window_size=50, n_lags=5)
        return TDAMonitor(cfg)

    def test_initialization(self, monitor):
        """Monitor should initialize with empty history."""
        assert isinstance(monitor.history, TDAMonitorHistory)

    def test_update_with_positions(self, monitor, sample_positions):
        """update should process positions."""
        result = monitor.update(sample_positions)
        assert isinstance(result, dict)

    def test_detect_early_warning_returns_indicators(self, monitor, sample_positions):
        """detect_early_warning should return indicator dict."""
        # Feed some data
        for _ in range(10):
            monitor.update(sample_positions)

        result = monitor.detect_early_warning()
        assert isinstance(result, dict)

    def test_history_accumulates(self, monitor, rng):
        """History should accumulate over updates."""
        positions1 = rng.normal(size=(50, 2))
        positions2 = rng.normal(size=(50, 2))

        monitor.update(positions1)
        len1 = len(monitor.history.variances)

        monitor.update(positions2)
        len2 = len(monitor.history.variances)

        assert len2 > len1


class TestPersistentEntropyFromMonitor:
    """Tests for persistent_entropy from tda_monitor module."""

    def test_matches_viz_implementation(self):
        """persistent_entropy should match viz implementation."""
        from attractor.viz.plot_persistence import persistent_entropy as viz_pe

        pairs = np.array([
            [0.0, 0.5],
            [0.1, 0.8],
            [0.2, 0.6],
        ])

        ent1 = persistent_entropy(pairs)
        ent2 = viz_pe(pairs)

        # Should be approximately equal
        assert ent1 == pytest.approx(ent2, rel=0.1)
