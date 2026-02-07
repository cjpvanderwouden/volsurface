"""Tests for the surface calibration engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from volsurface.calibration.engine import CalibrationResult, calibrate_surface
from volsurface.models.svi import RawSVI

if TYPE_CHECKING:
    from volsurface.core import MarketSlice


class TestCalibrateSurface:
    """Tests for the calibrate_surface function."""

    def test_single_slice(self, synthetic_smile: MarketSlice) -> None:
        result = calibrate_surface([synthetic_smile], RawSVI)
        assert isinstance(result, CalibrationResult)
        assert result.surface.n_expiries == 1
        assert result.surface.ticker is None

    def test_multiple_slices(self, synthetic_surface_slices: list[MarketSlice]) -> None:
        result = calibrate_surface(synthetic_surface_slices, RawSVI, ticker="SYNTH")
        assert result.surface.n_expiries == 4
        assert result.surface.ticker == "SYNTH"
        assert len(result.fit_results) == 4

    def test_all_fits_converge(self, synthetic_surface_slices: list[MarketSlice]) -> None:
        result = calibrate_surface(synthetic_surface_slices, RawSVI)
        for t, fr in result.fit_results.items():
            assert fr.success, f"Fit failed for T={t}"
            assert fr.rmse < 0.01, f"RMSE too high for T={t}: {fr.rmse}"

    def test_arbitrage_report_generated(self, synthetic_surface_slices: list[MarketSlice]) -> None:
        result = calibrate_surface(synthetic_surface_slices, RawSVI, check_arbitrage=True)
        assert result.arbitrage_report is not None

    def test_no_arbitrage_report_when_disabled(self, synthetic_smile: MarketSlice) -> None:
        result = calibrate_surface([synthetic_smile], RawSVI, check_arbitrage=False)
        assert result.arbitrage_report is None

    def test_surface_queryable_after_calibration(
        self, synthetic_surface_slices: list[MarketSlice]
    ) -> None:
        result = calibrate_surface(synthetic_surface_slices, RawSVI)
        surface = result.surface
        # Query at a known expiry
        pt = surface.iv(strike=100.0, expiry_years=0.25)
        assert pt.iv > 0
        # Query between expiries (interpolation)
        pt_interp = surface.iv(strike=100.0, expiry_years=0.3)
        assert pt_interp.iv > 0
