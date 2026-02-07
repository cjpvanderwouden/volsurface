"""Tests for the market data cleaning utilities."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.market_data.cleaning import clean_chain


class TestCleanChain:
    """Tests for the clean_chain function."""

    def test_basic_cleaning(self) -> None:
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        ivs = np.array([0.25, 0.22, 0.20, 0.21, 0.24])
        ms = clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)
        assert ms.n_strikes == 5
        assert ms.expiry_years == 0.25

    def test_filters_negative_ivs(self) -> None:
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        ivs = np.array([-0.1, 0.22, 0.20, 0.21, 0.24])
        ms = clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)
        assert ms.n_strikes == 4  # one filtered out

    def test_filters_out_of_moneyness_range(self) -> None:
        strikes = np.array([10.0, 95.0, 100.0, 105.0, 500.0])
        ivs = np.array([0.50, 0.22, 0.20, 0.21, 0.60])
        ms = clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)
        # 10 and 500 are far outside default (0.5, 2.0) moneyness range
        assert ms.n_strikes == 3

    def test_sorts_by_strike(self) -> None:
        strikes = np.array([110.0, 90.0, 100.0])
        ivs = np.array([0.24, 0.25, 0.20])
        ms = clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)
        assert np.all(np.diff(ms.strikes) > 0)

    def test_removes_duplicate_strikes(self) -> None:
        strikes = np.array([90.0, 100.0, 100.0, 110.0])
        ivs = np.array([0.25, 0.20, 0.20, 0.24])
        ms = clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)
        assert ms.n_strikes == 3

    def test_too_few_after_cleaning_raises(self) -> None:
        strikes = np.array([100.0])
        ivs = np.array([0.20])
        with pytest.raises(ValueError, match="valid strikes"):
            clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)

    def test_nan_values_filtered(self) -> None:
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        ivs = np.array([0.25, np.nan, 0.20, 0.21, 0.24])
        ms = clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)
        assert ms.n_strikes == 4
