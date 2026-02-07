"""Tests for the arbitrage checking module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from volsurface.arbitrage.checks import (
    ArbitrageReport,
    check_butterfly,
    check_calendar,
    check_slice,
)
from volsurface.calibration.engine import calibrate_surface
from volsurface.models.svi import RawSVI

if TYPE_CHECKING:
    from volsurface.core import MarketSlice


class TestCheckButterfly:
    """Tests for the butterfly (convexity) check."""

    def test_convex_data_passes(self) -> None:
        k = np.linspace(-0.3, 0.3, 20)
        w = 0.04 + 0.5 * k**2  # perfectly convex parabola
        violations = check_butterfly(k, w)
        assert violations == []

    def test_concave_data_detected(self) -> None:
        k = np.linspace(-0.3, 0.3, 20)
        w = 0.04 - 0.5 * k**2  # concave — violates butterfly
        violations = check_butterfly(k, w)
        assert len(violations) > 0

    def test_too_few_points_returns_empty(self) -> None:
        k = np.array([0.0, 0.1])
        w = np.array([0.04, 0.05])
        violations = check_butterfly(k, w)
        assert violations == []


class TestCheckCalendar:
    """Tests for the calendar spread check."""

    def test_clean_surface_passes(self, synthetic_surface_slices: list[MarketSlice]) -> None:
        result = calibrate_surface(synthetic_surface_slices, RawSVI)
        violations = check_calendar(result.surface)
        assert violations == []


class TestCheckSlice:
    """Tests for the single-slice convenience check."""

    def test_returns_report(self, synthetic_smile: MarketSlice) -> None:
        report = check_slice(synthetic_smile)
        assert isinstance(report, ArbitrageReport)

    def test_clean_slice_is_clean(self, synthetic_smile: MarketSlice) -> None:
        report = check_slice(synthetic_smile)
        assert report.is_clean
