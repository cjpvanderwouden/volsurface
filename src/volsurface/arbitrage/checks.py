"""Static no-arbitrage checks for volatility smiles and surfaces.

These checks are based on the necessary conditions derived in
Gatheral & Jacquier (2014) for absence of butterfly and calendar
spread arbitrage in the implied volatility surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from volsurface.core import MarketSlice, VolSurface


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    """Summary of arbitrage violations detected.

    Attributes:
        butterfly_violations: Indices where butterfly (convexity)
            condition is violated.
        calendar_violations: Pairs of (T_i, T_j) expiries where
            calendar spread condition is violated.
        is_clean: True if no violations of any kind were found.
    """

    butterfly_violations: list[int] = field(default_factory=list)
    calendar_violations: list[tuple[float, float]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True if no arbitrage violations were detected."""
        return len(self.butterfly_violations) == 0 and len(self.calendar_violations) == 0


def check_butterfly(
    log_moneyness: npt.NDArray[np.float64],
    total_variance: npt.NDArray[np.float64],
) -> list[int]:
    """Check the butterfly (convexity) no-arbitrage condition.

    For absence of butterfly arbitrage the total variance *w(k)* must
    be convex in log-moneyness *k*. Equivalently, the second finite
    difference of *w* with respect to *k* must be non-negative.

    Args:
        log_moneyness: Sorted array of log(K/F) values.
        total_variance: Corresponding total variance w = σ²T.

    Returns:
        List of interior indices where convexity is violated.
    """
    if len(log_moneyness) < 3:
        return []

    dk = np.diff(log_moneyness)
    dw = np.diff(total_variance)

    # Second finite difference (non-uniform spacing)
    second_deriv = np.zeros(len(log_moneyness) - 2)
    for i in range(len(second_deriv)):
        second_deriv[i] = (dw[i + 1] / dk[i + 1] - dw[i] / dk[i]) / (0.5 * (dk[i] + dk[i + 1]))

    # Allow small numerical tolerance
    violations = np.where(second_deriv < -1e-10)[0]
    return [int(v + 1) for v in violations]  # shift to original indexing


def check_calendar(
    surface: VolSurface,
    k_grid: npt.NDArray[np.float64] | None = None,
    n_grid: int = 50,
) -> list[tuple[float, float]]:
    """Check the calendar spread no-arbitrage condition.

    For absence of calendar arbitrage, total variance must be
    non-decreasing in time at every fixed log-moneyness level:
    w(k, T₁) ≤ w(k, T₂) whenever T₁ < T₂.

    Args:
        surface: A fitted VolSurface with at least 2 expiries.
        k_grid: Optional grid of log-moneyness points to check.
            If None, a uniform grid is generated from the surface data.
        n_grid: Number of grid points if k_grid is not provided.

    Returns:
        List of (T_i, T_j) pairs where a violation was detected.
    """
    expiries = surface.expiries
    if len(expiries) < 2:
        return []

    if k_grid is None:
        # Build grid spanning the intersection of all slices
        all_k = np.concatenate([surface.market_data[t].log_moneyness for t in expiries])
        k_grid = np.linspace(float(all_k.min()), float(all_k.max()), n_grid)

    violations: list[tuple[float, float]] = []

    for i in range(len(expiries) - 1):
        t_lo, t_hi = expiries[i], expiries[i + 1]
        model_lo = surface.slices[t_lo]
        model_hi = surface.slices[t_hi]

        w_lo = model_lo.total_variance(k_grid)
        w_hi = model_hi.total_variance(k_grid)

        if np.any(w_hi - w_lo < -1e-10):
            violations.append((t_lo, t_hi))

    return violations


def check_slice(market_slice: MarketSlice) -> ArbitrageReport:
    """Run butterfly checks on a single market slice.

    This is a convenience function for checking a single expiry
    before fitting.

    Args:
        market_slice: Market data for one expiry.

    Returns:
        ArbitrageReport with any butterfly violations.
    """
    bfly = check_butterfly(market_slice.log_moneyness, market_slice.total_variance)
    return ArbitrageReport(butterfly_violations=bfly)
