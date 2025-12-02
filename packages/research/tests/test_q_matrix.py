"""
Tests for Q-matrix utilities (regime transitions and metastability).
"""

import numpy as np
import pytest

from attractor.regime.q_matrix import (
    QMatrixResult,
    build_q_matrix,
    is_valid_q,
    estimate_q_from_counts,
    analyze_q,
    simulate_markov_chain,
)


class TestBuildQMatrix:
    """Tests for build_q_matrix function."""

    def test_basic_construction(self):
        """Test basic Q-matrix construction from rates."""
        rates = np.array([
            [0.0, 0.1, 0.2],
            [0.3, 0.0, 0.1],
            [0.2, 0.2, 0.0],
        ])
        Q = build_q_matrix(rates)

        # Check diagonal is negative sum of off-diagonals
        for i in range(3):
            row_sum = sum(Q[i, j] for j in range(3) if j != i)
            assert np.isclose(Q[i, i], -row_sum)

    def test_row_sums_zero(self):
        """Q-matrix rows should sum to zero."""
        rates = np.array([
            [0.0, 0.5, 0.3],
            [0.2, 0.0, 0.4],
            [0.1, 0.3, 0.0],
        ])
        Q = build_q_matrix(rates)

        row_sums = Q.sum(axis=1)
        assert np.allclose(row_sums, 0.0)

    def test_off_diagonal_non_negative(self):
        """Off-diagonal elements should be non-negative."""
        rates = np.array([
            [0.0, 0.1],
            [0.2, 0.0],
        ])
        Q = build_q_matrix(rates)

        for i in range(2):
            for j in range(2):
                if i != j:
                    assert Q[i, j] >= 0

    def test_non_square_raises(self):
        """Non-square input should raise ValueError."""
        rates = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        with pytest.raises(ValueError):
            build_q_matrix(rates)


class TestIsValidQ:
    """Tests for is_valid_q function."""

    def test_valid_q_matrix(self):
        """Valid Q-matrix should pass validation."""
        Q = np.array([
            [-0.3, 0.1, 0.2],
            [0.1, -0.2, 0.1],
            [0.2, 0.1, -0.3],
        ])
        assert is_valid_q(Q)

    def test_invalid_row_sum(self):
        """Q-matrix with non-zero row sums should fail."""
        Q = np.array([
            [-0.2, 0.1, 0.2],  # row sums to 0.1
            [0.1, -0.2, 0.1],
            [0.2, 0.1, -0.3],
        ])
        assert not is_valid_q(Q)

    def test_negative_off_diagonal(self):
        """Q-matrix with negative off-diagonal should fail."""
        Q = np.array([
            [-0.1, -0.1, 0.2],  # negative off-diagonal
            [0.1, -0.2, 0.1],
            [0.2, 0.1, -0.3],
        ])
        assert not is_valid_q(Q)


class TestEstimateQFromCounts:
    """Tests for estimate_q_from_counts function."""

    def test_basic_estimation(self):
        """Test MLE estimation from counts."""
        # 10 transitions 0->1, 5 transitions 1->0
        # dwell times: 100 in state 0, 50 in state 1
        N = np.array([
            [0.0, 10.0],
            [5.0, 0.0],
        ])
        dwell_times = np.array([100.0, 50.0])

        Q = estimate_q_from_counts(N, dwell_times)

        # q_01 = N_01 / T_0 = 10 / 100 = 0.1
        assert np.isclose(Q[0, 1], 0.1)
        # q_10 = N_10 / T_1 = 5 / 50 = 0.1
        assert np.isclose(Q[1, 0], 0.1)

    def test_result_is_valid_q(self):
        """Estimated Q should be a valid generator matrix."""
        N = np.array([
            [0.0, 5.0, 3.0],
            [2.0, 0.0, 4.0],
            [3.0, 2.0, 0.0],
        ])
        dwell_times = np.array([50.0, 60.0, 40.0])

        Q = estimate_q_from_counts(N, dwell_times)
        assert is_valid_q(Q)


class TestAnalyzeQ:
    """Tests for analyze_q function."""

    def test_returns_result_object(self):
        """Should return QMatrixResult dataclass."""
        Q = np.array([
            [-0.2, 0.1, 0.1],
            [0.1, -0.2, 0.1],
            [0.1, 0.1, -0.2],
        ])
        result = analyze_q(Q)

        assert isinstance(result, QMatrixResult)
        assert hasattr(result, 'Q')
        assert hasattr(result, 'eigenvalues')
        assert hasattr(result, 'spectral_gap')

    def test_zero_eigenvalue_exists(self):
        """Valid Q-matrix should have eigenvalue 0."""
        Q = np.array([
            [-0.3, 0.2, 0.1],
            [0.1, -0.2, 0.1],
            [0.1, 0.1, -0.2],
        ])
        result = analyze_q(Q)

        # At least one eigenvalue should be close to 0
        assert np.any(np.abs(result.eigenvalues) < 1e-6)

    def test_spectral_gap_positive(self):
        """Spectral gap should be positive for irreducible chain."""
        Q = np.array([
            [-0.5, 0.3, 0.2],
            [0.2, -0.4, 0.2],
            [0.3, 0.2, -0.5],
        ])
        result = analyze_q(Q)

        assert result.spectral_gap > 0

    def test_metastable_system_small_gap(self):
        """Metastable system (slow mixing) should have small spectral gap."""
        # Two nearly disconnected clusters
        Q = np.array([
            [-0.01, 0.005, 0.005],
            [0.005, -0.01, 0.005],
            [0.005, 0.005, -0.01],
        ])
        result_slow = analyze_q(Q)

        # Fast mixing system
        Q_fast = Q * 100  # Scale up rates
        result_fast = analyze_q(Q_fast)

        assert result_slow.spectral_gap < result_fast.spectral_gap


class TestSimulateMarkovChain:
    """Tests for simulate_markov_chain function."""

    def test_returns_expected_keys(self, rng):
        """Simulation result should have expected keys."""
        Q = np.array([
            [-0.2, 0.1, 0.1],
            [0.1, -0.2, 0.1],
            [0.1, 0.1, -0.2],
        ])
        result = simulate_markov_chain(Q, r0=0, T=10.0, dt=0.1, rng=rng)

        assert 'times' in result
        assert 'regimes' in result

    def test_initial_regime_correct(self, rng):
        """Initial regime should match r0."""
        Q = np.array([[-0.1, 0.1], [0.1, -0.1]])
        result = simulate_markov_chain(Q, r0=1, T=5.0, dt=0.1, rng=rng)

        assert result['regimes'][0] == 1

    def test_regimes_in_valid_range(self, rng):
        """All regimes should be valid indices."""
        n = 3
        Q = np.array([
            [-0.2, 0.1, 0.1],
            [0.1, -0.2, 0.1],
            [0.1, 0.1, -0.2],
        ])
        result = simulate_markov_chain(Q, r0=0, T=10.0, dt=0.1, rng=rng)

        assert np.all(result['regimes'] >= 0)
        assert np.all(result['regimes'] < n)

    def test_time_array_monotonic(self, rng):
        """Time array should be monotonically increasing."""
        Q = np.array([[-0.1, 0.1], [0.1, -0.1]])
        result = simulate_markov_chain(Q, r0=0, T=5.0, dt=0.1, rng=rng)

        times = result['times']
        assert np.all(np.diff(times) >= 0)

    def test_reproducibility_with_seed(self):
        """Same seed should give same results."""
        Q = np.array([[-0.3, 0.3], [0.3, -0.3]])

        rng1 = np.random.default_rng(123)
        result1 = simulate_markov_chain(Q, r0=0, T=10.0, dt=0.05, rng=rng1)

        rng2 = np.random.default_rng(123)
        result2 = simulate_markov_chain(Q, r0=0, T=10.0, dt=0.05, rng=rng2)

        assert np.array_equal(result1['regimes'], result2['regimes'])
