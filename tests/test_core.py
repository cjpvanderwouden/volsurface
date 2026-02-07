"""Tests for core data structures: MarketSlice, VolSurface, FitResult."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.core import MarketSlice, VolSurfacePoint


class TestMarketSlice:
    """Tests for the MarketSlice dataclass."""

    def test_valid_construction(self, synthetic_smile: MarketSlice) -> None:
        assert synthetic_smile.n_strikes == 25
        assert synthetic_smile.ticker == "SYNTH"
        assert synthetic_smile.expiry_years == 0.25

    def test_log_moneyness_shape(self, synthetic_smile: MarketSlice) -> None:
        k = synthetic_smile.log_moneyness
        assert k.shape == (25,)
        # ATM strike should have log_moneyness ≈ 0
        atm_idx = np.argmin(np.abs(synthetic_smile.strikes - 100.0))
        assert abs(k[atm_idx]) < 0.01

    def test_total_variance(self, synthetic_smile: MarketSlice) -> None:
        w = synthetic_smile.total_variance
        assert w.shape == (25,)
        assert np.all(w > 0)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            MarketSlice(
                strikes=np.array([90.0, 100.0, 110.0]),
                ivs=np.array([0.2, 0.18]),
                expiry_years=0.25,
                forward=100.0,
                spot=100.0,
            )

    def test_too_few_strikes_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            MarketSlice(
                strikes=np.array([100.0]),
                ivs=np.array([0.2]),
                expiry_years=0.25,
                forward=100.0,
                spot=100.0,
            )

    def test_non_positive_expiry_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MarketSlice(
                strikes=np.array([90.0, 100.0]),
                ivs=np.array([0.2, 0.18]),
                expiry_years=0.0,
                forward=100.0,
                spot=100.0,
            )

    def test_unsorted_strikes_raises(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            MarketSlice(
                strikes=np.array([110.0, 100.0, 90.0]),
                ivs=np.array([0.22, 0.18, 0.20]),
                expiry_years=0.25,
                forward=100.0,
                spot=100.0,
            )

    def test_negative_iv_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MarketSlice(
                strikes=np.array([90.0, 100.0]),
                ivs=np.array([-0.1, 0.2]),
                expiry_years=0.25,
                forward=100.0,
                spot=100.0,
            )

    def test_non_positive_forward_raises(self) -> None:
        with pytest.raises(ValueError, match="forward"):
            MarketSlice(
                strikes=np.array([90.0, 100.0]),
                ivs=np.array([0.2, 0.18]),
                expiry_years=0.25,
                forward=0.0,
                spot=100.0,
            )


class TestVolSurfacePoint:
    """Tests for the VolSurfacePoint dataclass."""

    def test_total_variance_property(self) -> None:
        pt = VolSurfacePoint(strike=100.0, expiry_years=0.25, iv=0.20)
        expected = 0.20**2 * 0.25
        assert abs(pt.total_variance - expected) < 1e-12
