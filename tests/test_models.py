"""Tests for the Raw SVI model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from volsurface.models.svi import RawSVI, SVIParams

if TYPE_CHECKING:
    from volsurface.core import MarketSlice


class TestSVIParams:
    """Tests for the SVIParams container."""

    def test_roundtrip_array(self) -> None:
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        arr = p.to_array()
        p2 = SVIParams.from_array(arr)
        assert abs(p.a - p2.a) < 1e-12
        assert abs(p.rho - p2.rho) < 1e-12

    def test_as_dict(self) -> None:
        p = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
        d = p.as_dict()
        assert set(d.keys()) == {"a", "b", "rho", "m", "sigma"}
        assert d["rho"] == -0.3


class TestRawSVI:
    """Tests for the RawSVI model."""

    def test_unfitted_model_raises(self) -> None:
        model = RawSVI()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.iv(0.0)

    def test_unfitted_params_raises(self) -> None:
        model = RawSVI()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.params

    def test_n_params(self) -> None:
        assert RawSVI().n_params == 5

    def test_fit_converges(self, synthetic_smile: MarketSlice) -> None:
        model = RawSVI()
        result = model.fit(synthetic_smile)
        assert result.success
        assert result.rmse < 0.01  # should fit synthetic data very well
        assert result.n_iterations > 0

    def test_fit_residuals_shape(self, synthetic_smile: MarketSlice) -> None:
        model = RawSVI()
        result = model.fit(synthetic_smile)
        assert result.residuals.shape == (synthetic_smile.n_strikes,)

    def test_fitted_model_returns_positive_iv(self, synthetic_smile: MarketSlice) -> None:
        model = RawSVI()
        model.fit(synthetic_smile)
        k_grid = np.linspace(-0.3, 0.3, 50)
        ivs = model.iv(k_grid)
        assert np.all(ivs > 0)

    def test_total_variance_increases_from_atm(self, synthetic_smile: MarketSlice) -> None:
        """SVI total variance should form a smile shape (increase away from ATM)."""
        model = RawSVI()
        model.fit(synthetic_smile)
        w_atm = float(model.total_variance(0.0)[0])
        w_far_otm = float(model.total_variance(0.2)[0])
        assert w_far_otm > w_atm

    def test_scalar_and_array_inputs_consistent(self, synthetic_smile: MarketSlice) -> None:
        model = RawSVI()
        model.fit(synthetic_smile)

        k = 0.05
        iv_scalar = model.iv(k)
        iv_array = model.iv(np.array([k]))
        assert abs(float(iv_scalar[0]) - float(iv_array[0])) < 1e-12

    def test_params_accessible_after_fit(self, synthetic_smile: MarketSlice) -> None:
        model = RawSVI()
        model.fit(synthetic_smile)
        p = model.params
        assert isinstance(p, SVIParams)
        assert p.b > 0
        assert abs(p.rho) < 1
        assert p.sigma > 0
