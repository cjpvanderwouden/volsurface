"""Raw SVI (Stochastic Volatility Inspired) parameterisation.

The raw SVI parameterisation of the implied total variance smile is:

    w(k) = a + b · [ ρ·(k − m) + √((k − m)² + σ²) ]

where *k* = ln(K/F) is log-moneyness and (a, b, ρ, m, σ) are the five
free parameters.

References:
    Gatheral, J. & Jacquier, A. (2014). "Arbitrage-free SVI volatility
    surfaces." *Quantitative Finance*, 14(1), 59–71.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from volsurface.core import FitResult, MarketSlice
from volsurface.models.base import VolModel


@dataclass
class SVIParams:
    """Container for the five raw SVI parameters.

    Attributes:
        a: Vertical translation of the smile.
        b: Slope tightness (b ≥ 0).
        rho: Rotation / skew (−1 < ρ < 1).
        m: Horizontal translation (smile centre in log-moneyness).
        sigma: Smoothing of the ATM vertex (σ > 0).
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def to_array(self) -> npt.NDArray[np.float64]:
        """Pack parameters into a numpy array."""
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, arr: npt.NDArray[np.float64]) -> SVIParams:
        """Unpack a numpy array into an SVIParams instance."""
        return cls(a=arr[0], b=arr[1], rho=arr[2], m=arr[3], sigma=arr[4])

    def as_dict(self) -> dict[str, float]:
        """Return parameters as a name→value dict."""
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m, "sigma": self.sigma}


class RawSVI(VolModel):
    """Raw SVI smile parameterisation.

    Example::

        from volsurface.models import RawSVI

        model = RawSVI()
        result = model.fit(market_slice)
        ivs = model.iv(log_moneyness_grid)
    """

    def __init__(self) -> None:
        self._params: SVIParams | None = None
        self._expiry_years: float | None = None

    # ── Properties ──────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        return 5

    @property
    def params(self) -> SVIParams:
        """Fitted SVI parameters.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self._params is None:
            msg = "Model has not been fitted yet — call .fit() first"
            raise RuntimeError(msg)
        return self._params

    # ── Core SVI formula ────────────────────────────────────────────

    @staticmethod
    def _w(
        k: npt.NDArray[np.float64],
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the raw SVI total-variance formula."""
        diff = k - m
        return np.asarray(a + b * (rho * diff + np.sqrt(diff**2 + sigma**2)), dtype=np.float64)

    # ── Public interface ────────────────────────────────────────────

    def total_variance(
        self,
        log_moneyness: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute total implied variance w(k) for given log-moneyness.

        Args:
            log_moneyness: Scalar or array of ln(K/F) values.

        Returns:
            Array of total variance values.
        """
        p = self.params  # will raise if not fitted
        k = self._to_array(log_moneyness)
        result = self._w(k, p.a, p.b, p.rho, p.m, p.sigma)
        return np.asarray(result, dtype=np.float64)

    def fit(self, market_slice: MarketSlice) -> FitResult:
        """Calibrate raw SVI to a market smile.

        Uses L-BFGS-B to minimise the sum of squared total-variance
        residuals, with parameter bounds that enforce basic SVI
        constraints.

        Args:
            market_slice: Cleaned market data for one expiry.

        Returns:
            FitResult with calibration diagnostics.
        """
        k = market_slice.log_moneyness
        w_market = market_slice.total_variance
        T = market_slice.expiry_years

        x0 = self._initial_guess(k, w_market)
        bounds = self._param_bounds(k, w_market)

        def objective(x: npt.NDArray[np.float64]) -> float:
            w_model = self._w(k, x[0], x[1], x[2], x[3], x[4])
            return float(np.sum((w_model - w_market) ** 2))

        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-14},
        )

        self._params = SVIParams.from_array(result.x)
        self._expiry_years = T

        w_fitted = self._w(k, *result.x)
        iv_residuals = np.sqrt(np.maximum(w_fitted, 0.0) / T) - market_slice.ivs
        rmse = float(np.sqrt(np.mean(iv_residuals**2)))

        return FitResult(
            params=self._params.as_dict(),
            residuals=iv_residuals,
            rmse=rmse,
            success=bool(result.success),
            message=str(result.message),
            n_iterations=int(result.nit),
        )

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _initial_guess(
        k: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Heuristic starting point for the optimiser.

        Strategy: set *m* at the ATM point, *a* at the ATM total
        variance, and sensible defaults for *b*, *ρ*, *σ*.
        """
        atm_idx = int(np.argmin(np.abs(k)))
        a0 = float(w[atm_idx])
        b0 = 0.1
        rho0 = -0.3  # typical equity skew
        m0 = float(k[atm_idx])
        sigma0 = 0.1
        return np.array([a0, b0, rho0, m0, sigma0])

    @staticmethod
    def _param_bounds(
        k: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64],
    ) -> list[tuple[float, float]]:
        """Parameter bounds for L-BFGS-B.

        These are intentionally somewhat loose to avoid biasing the
        optimiser; tighter no-arbitrage constraints can be applied
        via the arbitrage module post-hoc.
        """
        w_max = float(np.max(w))
        k_range = float(np.ptp(k))
        return [
            (-w_max, 2.0 * w_max),  # a
            (1e-8, 5.0),  # b
            (-0.999, 0.999),  # rho
            (float(k.min()) - k_range, float(k.max()) + k_range),  # m
            (1e-4, 2.0),  # sigma
        ]
