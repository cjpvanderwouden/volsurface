"""High-level calibration engine for fitting a full volatility surface.

Provides a single entry point that takes raw market slices, fits a
model to each expiry, runs arbitrage diagnostics, and returns a
populated :class:`VolSurface`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from volsurface.arbitrage.checks import ArbitrageReport, check_butterfly, check_calendar
from volsurface.core import FitResult, MarketSlice, VolSurface

if TYPE_CHECKING:
    from volsurface.models.base import VolModel

    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Aggregate result from fitting a full surface.

    Attributes:
        surface: The fully fitted VolSurface object.
        fit_results: Per-expiry FitResult diagnostics.
        arbitrage_report: Post-fit arbitrage check summary.
    """

    surface: VolSurface
    fit_results: dict[float, FitResult] = field(default_factory=dict)
    arbitrage_report: ArbitrageReport | None = None


def calibrate_surface(
    slices: list[MarketSlice],
    model_factory: type[VolModel],
    *,
    check_arbitrage: bool = True,
    ticker: str | None = None,
) -> CalibrationResult:
    """Fit a volatility model to each expiry and assemble a surface.

    This is the main high-level entry point for surface construction.

    Args:
        slices: List of MarketSlice objects (one per expiry).
        model_factory: The VolModel *class* to instantiate per slice
            (e.g. ``RawSVI``).
        check_arbitrage: If True, run butterfly and calendar checks
            on the fitted surface.
        ticker: Optional ticker symbol for the surface.

    Returns:
        CalibrationResult containing the fitted surface, per-slice
        diagnostics, and an optional arbitrage report.

    Example::

        from volsurface.calibration import calibrate_surface
        from volsurface.models import RawSVI

        result = calibrate_surface(slices, RawSVI)
        surface = result.surface
        print(surface.iv(strike=100, expiry_years=0.25))
    """
    surface = VolSurface(ticker=ticker)
    fit_results: dict[float, FitResult] = {}

    sorted_slices = sorted(slices, key=lambda s: s.expiry_years)

    for ms in sorted_slices:
        model = model_factory()
        logger.info(
            "Fitting %s to T=%.4f (%d strikes)",
            model_factory.__name__,
            ms.expiry_years,
            ms.n_strikes,
        )
        result = surface.add_slice(ms, model)
        fit_results[ms.expiry_years] = result

        if not result.success:
            logger.warning("Fit did not converge for T=%.4f: %s", ms.expiry_years, result.message)
        else:
            logger.info(
                "T=%.4f fitted — RMSE=%.6f (%d iters)",
                ms.expiry_years,
                result.rmse,
                result.n_iterations,
            )

    arb_report: ArbitrageReport | None = None
    if check_arbitrage and surface.n_expiries > 0:
        all_bfly: list[int] = []
        for t, ms in surface.market_data.items():
            slice_model = surface.slices[t]
            k = ms.log_moneyness
            w_model = slice_model.total_variance(k)
            all_bfly.extend(check_butterfly(k, w_model))

        cal_violations = check_calendar(surface) if surface.n_expiries >= 2 else []

        arb_report = ArbitrageReport(
            butterfly_violations=all_bfly,
            calendar_violations=cal_violations,
        )

        if arb_report.is_clean:
            logger.info("No arbitrage violations detected")
        else:
            logger.warning(
                "Arbitrage violations: %d butterfly, %d calendar",
                len(arb_report.butterfly_violations),
                len(arb_report.calendar_violations),
            )

    return CalibrationResult(
        surface=surface,
        fit_results=fit_results,
        arbitrage_report=arb_report,
    )
