"""Utilities for cleaning raw option data into MarketSlice objects.

These functions are data-source agnostic and work with numpy arrays,
making them usable independently of the Yahoo adapter.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from volsurface.core import MarketSlice


def clean_chain(
    strikes: npt.NDArray[np.float64],
    ivs: npt.NDArray[np.float64],
    expiry_years: float,
    forward: float,
    spot: float,
    *,
    rate: float = 0.0,
    ticker: str | None = None,
    min_iv: float = 0.001,
    max_iv: float = 5.0,
    moneyness_range: tuple[float, float] = (0.5, 2.0),
) -> MarketSlice:
    """Clean raw strike/IV arrays and construct a MarketSlice.

    Applies basic sanity filters (positive IVs, moneyness window),
    removes duplicates, and sorts by strike.

    Args:
        strikes: Raw strike prices.
        ivs: Raw implied volatilities corresponding to each strike.
        expiry_years: Time to expiry in years.
        forward: Forward price for this expiry.
        spot: Current spot price.
        rate: Risk-free rate (continuous compounding).
        ticker: Optional ticker symbol.
        min_iv: Minimum plausible IV.
        max_iv: Maximum plausible IV.
        moneyness_range: (low, high) K/F bounds.

    Returns:
        A validated MarketSlice.

    Raises:
        ValueError: If fewer than 2 strikes remain after filtering.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    ivs = np.asarray(ivs, dtype=np.float64)

    # Build boolean mask for valid rows
    moneyness = strikes / forward
    mask = (
        (ivs > min_iv)
        & (ivs < max_iv)
        & (moneyness >= moneyness_range[0])
        & (moneyness <= moneyness_range[1])
        & np.isfinite(strikes)
        & np.isfinite(ivs)
    )

    strikes = strikes[mask]
    ivs = ivs[mask]

    # Sort by strike and remove duplicates
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    ivs = ivs[sort_idx]

    unique_mask = np.concatenate(([True], np.diff(strikes) > 0))
    strikes = strikes[unique_mask]
    ivs = ivs[unique_mask]

    if len(strikes) < 2:
        msg = f"Only {len(strikes)} valid strikes after cleaning (need ≥ 2)"
        raise ValueError(msg)

    return MarketSlice(
        strikes=strikes,
        ivs=ivs,
        expiry_years=expiry_years,
        forward=forward,
        spot=spot,
        rate=rate,
        ticker=ticker,
    )
