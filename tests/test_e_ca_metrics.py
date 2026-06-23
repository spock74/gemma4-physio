"""Unit tests for E-CA metrics module.

Tests use synthetic grids with known properties to validate every metric
function.  Random tests seed ``np.random`` for reproducibility and use
generous tolerances for statistical expectations.
"""
from __future__ import annotations

import numpy as np
import pytest

from calibration.e_ca.e_ca_metrics import (
    compare_partitions,
    compute_all_metrics,
    feature_lifetimes,
    grid_autocorrelation,
    jaccard_consecutive,
    per_layer_entropy,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _static_grid(n_layers: int = 10, n_features: int = 20) -> np.ndarray:
    """All features alive at every layer."""
    return np.ones((n_layers, n_features), dtype=np.int64)


def _empty_grid(n_layers: int = 10, n_features: int = 20) -> np.ndarray:
    """All features dead at every layer."""
    return np.zeros((n_layers, n_features), dtype=np.int64)


def _chaotic_grid(
    n_layers: int = 50,
    n_features: int = 500,
    p: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """IID Bernoulli(p) at every cell."""
    rng = np.random.RandomState(seed)
    return (rng.rand(n_layers, n_features) < p).astype(np.int64)


def _glider_grid() -> np.ndarray:
    """Small deterministic grid: 3 features slide one position per layer.

    10 features, 8 layers.
    At layer t, features [t, t+1, t+2] are alive (mod 10).
    """
    n_layers, n_features = 8, 10
    grid = np.zeros((n_layers, n_features), dtype=np.int64)
    for t in range(n_layers):
        for offset in range(3):
            grid[t, (t + offset) % n_features] = 1
    return grid


# ── 1. Static grid ────────────────────────────────────────────────────────

class TestStaticGrid:
    """All features alive at all layers → maximum stability."""

    @pytest.fixture()
    def grid(self) -> np.ndarray:
        return _static_grid(n_layers=10, n_features=20)

    def test_jaccard_all_ones(self, grid: np.ndarray) -> None:
        jac = jaccard_consecutive(grid)
        assert jac.shape == (9,)
        np.testing.assert_allclose(jac, 1.0)

    def test_lifetime_equals_n_layers(self, grid: np.ndarray) -> None:
        lf = feature_lifetimes(grid)
        assert len(lf) == 20  # one run per feature
        np.testing.assert_array_equal(lf, 10)

    def test_entropy_one(self, grid: np.ndarray) -> None:
        ent = per_layer_entropy(grid)
        # p=1 → entropy=0
        np.testing.assert_allclose(ent, 0.0)

    def test_autocorrelation_all_ones(self, grid: np.ndarray) -> None:
        ac = grid_autocorrelation(grid)
        assert ac.shape == (9,)
        np.testing.assert_allclose(ac, 1.0)


# ── 2. Chaotic grid ──────────────────────────────────────────────────────

class TestChaoticGrid:
    """IID Bernoulli(0.5) → specific statistical expectations."""

    @pytest.fixture()
    def grid(self) -> np.ndarray:
        return _chaotic_grid(n_layers=50, n_features=500, seed=42)

    def test_jaccard_around_third(self, grid: np.ndarray) -> None:
        jac = jaccard_consecutive(grid)
        # Expected J ≈ p² / (2p − p²) = 0.25 / 0.75 ≈ 0.333
        np.testing.assert_allclose(np.mean(jac), 1.0 / 3.0, atol=0.05)

    def test_mean_lifetime_around_one(self, grid: np.ndarray) -> None:
        lf = feature_lifetimes(grid)
        # Geometric distribution with p=0.5 → mean run ≈ 1 / (1-p) = 2
        # But feature_lifetimes only records runs of 1s.
        # For iid Bernoulli(0.5), mean consecutive-1 run ≈ 1/(1-0.5) = 2
        # However most runs are length 1, so the mean is close to ~2.
        assert np.mean(lf) < 3.0
        assert np.mean(lf) >= 1.0

    def test_autocorrelation_decays_toward_half(self, grid: np.ndarray) -> None:
        ac = grid_autocorrelation(grid)
        # Independent layers: agreement by chance ≈ p² + (1-p)² = 0.5
        np.testing.assert_allclose(ac[-1], 0.5, atol=0.05)

    def test_entropy_near_one(self, grid: np.ndarray) -> None:
        ent = per_layer_entropy(grid)
        # p ≈ 0.5 → H ≈ 1.0
        np.testing.assert_allclose(np.mean(ent), 1.0, atol=0.05)


# ── 3. Glider grid ───────────────────────────────────────────────────────

class TestGliderGrid:
    """Sliding block of 3 features, verifiable dynamics."""

    @pytest.fixture()
    def grid(self) -> np.ndarray:
        return _glider_grid()

    def test_shape(self, grid: np.ndarray) -> None:
        assert grid.shape == (8, 10)
        # Each layer has exactly 3 active features
        np.testing.assert_array_equal(grid.sum(axis=1), 3)

    def test_jaccard_values(self, grid: np.ndarray) -> None:
        jac = jaccard_consecutive(grid)
        assert jac.shape == (7,)
        # Consecutive layers share 2 of 4 union features → J = 2/4 = 0.5
        np.testing.assert_allclose(jac, 0.5, atol=1e-10)

    def test_lifetime_distribution(self, grid: np.ndarray) -> None:
        lf = feature_lifetimes(grid)
        # Features 0,1,2 are on from layer 0; feature i turns on at layer
        # i-2 (for i>=3).  Each feature is alive for at most 3 consecutive
        # layers (when the glider passes through).
        # Total runs: each feature touched by the glider gets a run of
        # length = min(3, overlap with 8 layers).
        assert len(lf) > 0
        assert np.max(lf) <= 8  # can't exceed n_layers


# ── 4. Empty grid ────────────────────────────────────────────────────────

class TestEmptyGrid:
    """All zeros — edge-case handling."""

    @pytest.fixture()
    def grid(self) -> np.ndarray:
        return _empty_grid()

    def test_jaccard_edge_case(self, grid: np.ndarray) -> None:
        jac = jaccard_consecutive(grid)
        # Both dead → 1.0 by convention
        np.testing.assert_allclose(jac, 1.0)

    def test_entropy_zero(self, grid: np.ndarray) -> None:
        ent = per_layer_entropy(grid)
        np.testing.assert_allclose(ent, 0.0)

    def test_lifetimes_empty(self, grid: np.ndarray) -> None:
        lf = feature_lifetimes(grid)
        assert len(lf) == 0

    def test_autocorrelation_edge(self, grid: np.ndarray) -> None:
        ac = grid_autocorrelation(grid)
        # All zeros match everywhere → 1.0
        np.testing.assert_allclose(ac, 1.0)


# ── 5. compare_partitions ────────────────────────────────────────────────

class TestComparePartitions:
    """Static vs chaotic partitions must separate clearly."""

    def test_structure_and_significance(self) -> None:
        statics = [_static_grid(10, 50) for _ in range(10)]
        chaotics = [_chaotic_grid(10, 50, seed=i) for i in range(10)]

        result = compare_partitions(
            statics, chaotics, labels=("static", "chaotic")
        )

        # Structure checks
        assert "static" in result
        assert "chaotic" in result
        assert "delta_jaccard_mean" in result
        assert "mannwhitney_p" in result
        assert "mannwhitney_u" in result

        # Static Jaccard mean = 1.0, chaotic ≈ 0.33 → big delta
        assert result["delta_jaccard_mean"] > 0.5

        # p-value should be very significant
        assert result["mannwhitney_p"] < 0.01

        # Per-group field checks
        assert result["static"]["n_grids"] == 10
        assert result["chaotic"]["n_grids"] == 10
        assert result["static"]["jaccard_mean"] == pytest.approx(1.0)


# ── 6. Shape validation ──────────────────────────────────────────────────

class TestShapeValidation:
    """Ensure functions reject wrong-shaped inputs."""

    def test_1d_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            jaccard_consecutive(np.array([1, 0, 1]))

    def test_4d_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            feature_lifetimes(np.ones((2, 3, 4, 5), dtype=np.int64))

    def test_3d_accepted(self) -> None:
        # 3-D should be reshaped silently
        grid_3d = np.ones((5, 3, 10), dtype=np.int64)
        jac = jaccard_consecutive(grid_3d)
        assert jac.shape == (4,)
        np.testing.assert_allclose(jac, 1.0)


# ── 7. compute_all_metrics ───────────────────────────────────────────────

class TestComputeAllMetrics:
    """Smoke test on the aggregate helper."""

    def test_keys_present(self) -> None:
        grid = _static_grid(5, 10)
        m = compute_all_metrics(grid)
        expected_keys = {
            "jaccard", "lifetimes", "lifetime_mean", "lifetime_median",
            "entropy", "autocorrelation", "n_layers", "n_features",
            "mean_sparsity",
        }
        assert set(m.keys()) == expected_keys

    def test_types(self) -> None:
        grid = _chaotic_grid(10, 20, seed=99)
        m = compute_all_metrics(grid)
        assert isinstance(m["jaccard"], list)
        assert isinstance(m["lifetimes"], list)
        assert isinstance(m["lifetime_mean"], float)
        assert isinstance(m["lifetime_median"], float)
        assert isinstance(m["entropy"], list)
        assert isinstance(m["autocorrelation"], list)
        assert isinstance(m["n_layers"], int)
        assert isinstance(m["n_features"], int)
        assert isinstance(m["mean_sparsity"], float)

    def test_static_sparsity_one(self) -> None:
        grid = _static_grid(5, 10)
        m = compute_all_metrics(grid)
        assert m["mean_sparsity"] == pytest.approx(1.0)

    def test_empty_sparsity_zero(self) -> None:
        grid = _empty_grid(5, 10)
        m = compute_all_metrics(grid)
        assert m["mean_sparsity"] == pytest.approx(0.0)
