"""Shared test fixtures for the volsurface test suite."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.core import MarketSlice


@pytest.fixture()
def synthetic_smile() -> MarketSlice:
    """A synthetic ATM-centred smile for testing.

    Generates a smooth parabolic smile shape typical of equity options,
    with forward = 100 and T = 0.25 years.
    """
    forward = 100.0
    strikes = np.linspace(80, 120, 25)
    log_m = np.log(strikes / forward)
    # Parabolic smile: base vol 0.20, quadratic skew
    ivs = 0.20 + 0.5 * log_m**2 + 0.05 * log_m
    return MarketSlice(
        strikes=strikes,
        ivs=ivs,
        expiry_years=0.25,
        forward=forward,
        spot=100.0,
        ticker="SYNTH",
    )


@pytest.fixture()
def synthetic_surface_slices() -> list[MarketSlice]:
    """Multiple synthetic slices at different expiries for surface tests."""
    forward = 100.0
    spot = 100.0
    expiries = [0.1, 0.25, 0.5, 1.0]
    slices = []

    for T in expiries:
        strikes = np.linspace(80, 120, 25)
        log_m = np.log(strikes / forward)
        # Term structure: ATM vol increases with sqrt(T)
        base_vol = 0.15 + 0.05 * np.sqrt(T)
        ivs = base_vol + 0.4 * log_m**2 + 0.03 * log_m
        slices.append(
            MarketSlice(
                strikes=strikes,
                ivs=ivs,
                expiry_years=T,
                forward=forward,
                spot=spot,
                ticker="SYNTH",
            )
        )

    return slices
