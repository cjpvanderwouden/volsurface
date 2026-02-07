"""Abstract base class for volatility parameterisation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from volsurface.core import FitResult, MarketSlice


class VolModel(ABC):
    """Base class that all volatility smile models inherit from.

    Subclasses must implement :meth:`fit`, :meth:`total_variance`, and
    :attr:`n_params`. The default :meth:`iv` implementation derives
    implied vol from total variance, but subclasses may override it for
    efficiency.
    """

    _expiry_years: float | None = None

    # ── Abstract interface ──────────────────────────────────────────

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of free parameters in this model."""

    @abstractmethod
    def fit(self, market_slice: MarketSlice) -> FitResult:
        """Calibrate the model to observed market data.

        After a successful call the model is *fitted* and :meth:`iv`
        / :meth:`total_variance` can be called.
        """

    @abstractmethod
    def total_variance(
        self,
        log_moneyness: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute total implied variance w(k) = σ²(k)·T.

        Args:
            log_moneyness: Scalar or array of log(K/F) values.

        Returns:
            Array of total variance values.
        """

    # ── Default implementations ─────────────────────────────────────

    def iv(
        self,
        log_moneyness: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute implied volatility from total variance.

        Args:
            log_moneyness: Scalar or array of log(K/F) values.

        Returns:
            Array of implied volatility values.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._expiry_years is None:
            msg = "Model has not been fitted yet — call .fit() first"
            raise RuntimeError(msg)
        w = self.total_variance(log_moneyness)
        return np.sqrt(np.maximum(w, 0.0) / self._expiry_years)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_array(x: float | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Ensure input is a 1-D numpy array."""
        return np.atleast_1d(np.asarray(x, dtype=np.float64))
