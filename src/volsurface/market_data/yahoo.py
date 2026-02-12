"""Yahoo Finance option chain fetcher.

This module requires the optional ``yfinance`` dependency::

    pip install volsurface[yahoo]
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from volsurface.core import MarketSlice

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _import_yfinance() -> Any:
    """Lazily import yfinance, raising a clear error if absent."""
    try:
        import yfinance
    except ImportError:
        msg = (
            "yfinance is required for Yahoo data fetching. "
            "Install it with:  pip install volsurface[yahoo]"
        )
        raise ImportError(msg) from None
    return yfinance


def fetch_chain(
    ticker: str,
    *,
    min_volume: int = 0,
    min_open_interest: int = 0,
    max_spread_pct: float = 1.0,
    moneyness_range: tuple[float, float] = (0.7, 1.3),
    min_days_to_expiry: int = 14,
) -> list[MarketSlice]:

    yf = _import_yfinance()
    asset = yf.Ticker(ticker)

    # fast_info is more reliable than .info for spot price
    spot: float = float(asset.fast_info["lastPrice"])
    logger.info("Spot price for %s: %.2f", ticker, spot)

    expiry_strings: tuple[str, ...] = asset.options
    if not expiry_strings:
        msg = f"No option expiries found for {ticker}"
        raise ValueError(msg)

    today = dt.date.today()
    slices: list[MarketSlice] = []

    for exp_str in expiry_strings:
        exp_date = dt.date.fromisoformat(exp_str)
        days_to_exp = (exp_date - today).days
        T = days_to_exp / 365.0

        if days_to_exp < min_days_to_expiry:
            logger.debug("Skipping %s — only %d days to expiry", exp_str, days_to_exp)
            continue

        try:
            chain: pd.DataFrame = asset.option_chain(exp_str).calls
        except Exception:
            logger.warning("Failed to fetch chain for %s %s", ticker, exp_str)
            continue

        chain = _filter_chain(
            chain,
            spot=spot,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct,
            moneyness_range=moneyness_range,
        )

        if len(chain) < 5:
            logger.debug(
                "Skipping %s — only %d strikes after filtering (need 5)",
                exp_str,
                len(chain),
            )
            continue

        strikes = chain["strike"].to_numpy(dtype=np.float64)
        ivs = chain["impliedVolatility"].to_numpy(dtype=np.float64)

        # Use spot as a rough proxy for forward (Yahoo doesn't provide
        # clean forwards).  A future adapter could compute F from
        # put-call parity.
        forward = spot

        try:
            ms = MarketSlice(
                strikes=strikes,
                ivs=ivs,
                expiry_years=T,
                forward=forward,
                spot=spot,
                ticker=ticker,
            )
            slices.append(ms)
            logger.info("Added slice %s (T=%.4f, %d strikes)", exp_str, T, len(strikes))
        except ValueError as e:
            logger.debug("Validation failed for %s: %s", exp_str, e)
            continue

    if not slices:
        msg = f"No valid option slices for {ticker} after filtering"
        raise ValueError(msg)

    return sorted(slices, key=lambda s: s.expiry_years)


def _filter_chain(
    chain: pd.DataFrame,
    *,
    spot: float,
    min_volume: int,
    min_open_interest: int,
    max_spread_pct: float,
    moneyness_range: tuple[float, float],
) -> pd.DataFrame:
    """Apply liquidity and moneyness filters to a raw option chain."""
    df = chain.copy()

    # Moneyness filter (apply first to reduce dataframe size)
    moneyness = df["strike"] / spot
    df = df[(moneyness >= moneyness_range[0]) & (moneyness <= moneyness_range[1])]

    # IV sanity — Yahoo returns 0.00001 for stale/dead contracts
    df = df[df["impliedVolatility"] > 0.01]
    df = df[df["impliedVolatility"] < 5.0]

    # Liquidity filters (only apply if thresholds are > 0)
    if min_volume > 0:
        df = df[df["volume"].fillna(0) >= min_volume]
    if min_open_interest > 0:
        df = df[df["openInterest"].fillna(0) >= min_open_interest]

    # Spread filter — only apply when both bid and ask are quoted
    has_quotes = (df["bid"] > 0) & (df["ask"] > 0)
    if has_quotes.any():
        quoted = df[has_quotes]
        mid = (quoted["bid"] + quoted["ask"]) / 2
        spread_pct = (quoted["ask"] - quoted["bid"]) / mid
        good_spread = quoted[spread_pct <= max_spread_pct]
        # If we have enough quoted contracts, use only those.
        # Otherwise fall back to all contracts with valid IVs.
        if len(good_spread) >= 5:
            df = good_spread
        else:
            logger.debug("Few quoted contracts — using Yahoo IVs directly")

    return df.sort_values("strike").reset_index(drop=True)
