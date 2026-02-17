"""Core data structures for the volsurface package.

This module defines the fundamental types that flow through the entire library:
market data inputs, surface query results, and the central VolSurface object.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# ── Enums ───────────────────────────────────────────────────────────


class OptionKind(str, enum.Enum):
    """Option type identifier."""

    CALL = "call"
    PUT = "put"


class MoneynessKind(str, enum.Enum):
    """Convention used to express moneyness on a smile."""

    LOG_STRIKE = "log_strike"  # log(K/F)
    DELTA = "delta"
    NORMALISED = "normalised"  # K/F


# ── Market Data ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MarketSlice:
    """Cleaned market data for a single expiry.

    Represents one horizontal slice of the volatility surface: a set of
    strikes and their implied volatilities at a fixed time-to-expiry.

    Attributes:
        strikes: Array of strike prices (ascending).
        ivs: Array of implied volatilities corresponding to each strike.
        expiry_years: Time to expiry in years (ACT/365).
        forward: Forward price for this expiry.
        spot: Current spot price of the underlying.
        rate: Risk-free rate used (continuous compounding).
        ticker: Optional underlying ticker symbol.

    Raises:
        ValueError: If array lengths mismatch, values are non-positive
            where required, or strikes are not sorted ascending.
    """

    strikes: npt.NDArray[np.float64]
    ivs: npt.NDArray[np.float64]
    expiry_years: float
    forward: float
    spot: float
    rate: float = 0.0
    ticker: str | None = None

    def __post_init__(self) -> None:
        if len(self.strikes) != len(self.ivs):
            msg = (
                f"strikes and ivs must have equal length, "
                f"got {len(self.strikes)} and {len(self.ivs)}"
            )
            raise ValueError(msg)
        if len(self.strikes) < 2:
            msg = "Need at least 2 data points per slice"
            raise ValueError(msg)
        if self.expiry_years <= 0:
            msg = f"expiry_years must be positive, got {self.expiry_years}"
            raise ValueError(msg)
        if self.forward <= 0:
            msg = f"forward must be positive, got {self.forward}"
            raise ValueError(msg)
        if not np.all(np.diff(self.strikes) > 0):
            msg = "strikes must be sorted in strictly ascending order"
            raise ValueError(msg)
        if np.any(self.ivs <= 0):
            msg = "All implied volatilities must be positive"
            raise ValueError(msg)

    @property
    def log_moneyness(self) -> npt.NDArray[np.float64]:
        """Log-moneyness: ln(K / F) for each strike."""
        return np.log(self.strikes / self.forward)

    @property
    def n_strikes(self) -> int:
        """Number of strike observations in this slice."""
        return len(self.strikes)

    @property
    def total_variance(self) -> npt.NDArray[np.float64]:
        """Total implied variance: σ²·T for each strike."""
        return self.ivs**2 * self.expiry_years


# ── Surface Query Results ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VolSurfacePoint:
    """Result of querying the volatility surface at a single (K, T) point.

    Attributes:
        strike: The queried strike price.
        expiry_years: The queried time to expiry.
        iv: Interpolated / model-implied implied volatility.
        total_variance: iv² × T.
    """

    strike: float
    expiry_years: float
    iv: float

    @property
    def total_variance(self) -> float:
        """Total implied variance at this point."""
        return self.iv**2 * self.expiry_years


# ── Model Protocol ──────────────────────────────────────────────────


@runtime_checkable
class VolModelProtocol(Protocol):
    """Interface that all volatility parameterisation models must satisfy.

    This protocol defines the contract between the calibration engine
    and any model implementation (SVI, SSVI, SABR, etc.).
    """

    @property
    def n_params(self) -> int:
        """Number of free parameters in the model."""
        ...

    def fit(self, market_slice: MarketSlice) -> FitResult:
        """Calibrate model parameters to a market slice."""
        ...

    def iv(self, log_moneyness: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute implied vol for given log-moneyness value(s)."""
        ...

    def total_variance(
        self, log_moneyness: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute total implied variance for given log-moneyness value(s)."""
        ...


# ── Fit Result ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FitResult:
    """Diagnostics returned after fitting a model to a MarketSlice.

    Attributes:
        params: Dict mapping parameter names to fitted values.
        residuals: Per-strike fitting residuals (model IV − market IV).
        rmse: Root mean squared error of the fit.
        success: Whether the optimiser converged.
        message: Optimiser status message.
        n_iterations: Number of optimiser iterations used.
    """

    params: dict[str, float]
    residuals: npt.NDArray[np.float64]
    rmse: float
    success: bool
    message: str
    n_iterations: int


# ── The Surface ─────────────────────────────────────────────────────


@dataclass(slots=True)
class VolSurface:
    """A fitted implied volatility surface.

    The surface is composed of individually fitted smile models (one per
    expiry), stitched together to form a queryable 2D surface over
    (strike, expiry) space.

    Attributes:
        slices: Mapping of expiry (years) → fitted VolModel.
        market_data: The original MarketSlice objects used for fitting.
        ticker: Optional underlying ticker symbol.
    """

    slices: dict[float, VolModelProtocol] = field(default_factory=dict)
    market_data: dict[float, MarketSlice] = field(default_factory=dict)
    ticker: str | None = None

    @property
    def expiries(self) -> list[float]:
        """Sorted list of fitted expiry tenors."""
        return sorted(self.slices.keys())

    @property
    def n_expiries(self) -> int:
        """Number of fitted expiry slices."""
        return len(self.slices)

    def add_slice(
        self,
        market_slice: MarketSlice,
        model: VolModelProtocol,
    ) -> FitResult:
        """Fit a model to a market slice and add it to the surface.

        Args:
            market_slice: Cleaned market data for one expiry.
            model: An (unfitted) model instance to calibrate.

        Returns:
            FitResult with calibration diagnostics.
        """
        result = model.fit(market_slice)
        self.slices[market_slice.expiry_years] = model
        self.market_data[market_slice.expiry_years] = market_slice
        return result

    def iv(self, strike: float, expiry_years: float) -> VolSurfacePoint:
        """Query the surface for implied volatility at (strike, expiry).

        For expiries that lie between fitted slices, linear interpolation
        in total-variance space is used (variance is linear in T under
        most no-arbitrage conditions).  The forward price is also
        linearly interpolated so that both bracketing models are queried
        at the same log-moneyness point.

        Args:
            strike: The strike price to query.
            expiry_years: Time to expiry in years.

        Returns:
            A VolSurfacePoint with the interpolated IV.

        Raises:
            ValueError: If the surface has no fitted slices, or the
                requested expiry is outside the fitted range.
        """
        if not self.slices:
            msg = "Surface has no fitted slices"
            raise ValueError(msg)

        expiries = self.expiries

        if expiry_years < expiries[0] or expiry_years > expiries[-1]:
            msg = (
                f"Requested expiry {expiry_years:.4f}y is outside the fitted "
                f"range [{expiries[0]:.4f}, {expiries[-1]:.4f}]"
            )
            raise ValueError(msg)

        # Exact match — query the slice directly
        if expiry_years in self.slices:
            model = self.slices[expiry_years]
            forward = self.market_data[expiry_years].forward
            log_m = np.log(strike / forward)
            iv_val: float = float(model.iv(log_m)[0])
            return VolSurfacePoint(strike=strike, expiry_years=expiry_years, iv=iv_val)

        # Interpolation in total-variance space
        t_lo, t_hi = self._bracket_expiry(expiry_years)
        model_lo = self.slices[t_lo]
        model_hi = self.slices[t_hi]
        fwd_lo = self.market_data[t_lo].forward
        fwd_hi = self.market_data[t_hi].forward

        # Interpolation weight (used for both forward and variance)
        alpha = (expiry_years - t_lo) / (t_hi - t_lo)

        # Interpolate the forward so both models are queried at the
        # same moneyness point — avoids mixing two different
        # log-moneyness values when fwd_lo ≠ fwd_hi.
        fwd_interp = fwd_lo * (1 - alpha) + fwd_hi * alpha
        log_m = np.log(strike / fwd_interp)

        w_lo: float = float(model_lo.total_variance(log_m)[0])
        w_hi: float = float(model_hi.total_variance(log_m)[0])

        w_interp = w_lo * (1 - alpha) + w_hi * alpha

        iv_interp = np.sqrt(w_interp / expiry_years)
        return VolSurfacePoint(strike=strike, expiry_years=expiry_years, iv=float(iv_interp))

    def _bracket_expiry(self, t: float) -> tuple[float, float]:
        """Find the two fitted expiries that bracket *t*."""
        expiries = self.expiries
        for i in range(len(expiries) - 1):
            if expiries[i] <= t <= expiries[i + 1]:
                return expiries[i], expiries[i + 1]
        msg = f"Cannot bracket expiry {t}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover