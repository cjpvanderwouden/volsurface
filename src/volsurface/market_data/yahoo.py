"""Yahoo Finance option chain fetcher.

This module requires the optional ``yfinance`` dependency::

    pip install volsurface[yahoo]
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import numpy as np

from volsurface.core import MarketSlice

if TYPE_CHECKING:
    import pandas as pd


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
    min_volume: int = 10,
    min_open_interest: int = 10,
    max_spread_pct: float = 0.50,
    moneyness_range: tuple[float, float] = (0.7, 1.3),
) -> list[MarketSlice]:
    """Fetch and clean option chains from Yahoo Finance.

    Pulls all available expiries for *ticker*, filters for liquid
    contracts, computes mid implied volatilities, and returns a list of
    :class:`MarketSlice` objects ready for fitting.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. ``"AAPL"``).
        min_volume: Minimum daily volume to include a contract.
        min_open_interest: Minimum open interest to include a contract.
        max_spread_pct: Maximum bid-ask spread as a fraction of mid.
        moneyness_range: (low, high) bounds on K/S to include.

    Returns:
        List of MarketSlice objects, one per expiry, sorted by expiry.

    Raises:
        ImportError: If ``yfinance`` is not installed.
        ValueError: If no valid slices can be constructed.
    """
    yf = _import_yfinance()
    asset = yf.Ticker(ticker)
    spot: float = float(asset.info.get("regularMarketPrice", asset.fast_info["lastPrice"]))

    expiry_strings: tuple[str, ...] = asset.options
    if not expiry_strings:
        msg = f"No option expiries found for {ticker}"
        raise ValueError(msg)

    today = dt.date.today()
    slices: list[MarketSlice] = []

    for exp_str in expiry_strings:
        exp_date = dt.date.fromisoformat(exp_str)
        T = (exp_date - today).days / 365.0
        if T <= 0:
            continue

        chain: pd.DataFrame = asset.option_chain(exp_str).calls
        chain = _filter_chain(
            chain,
            spot=spot,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct,
            moneyness_range=moneyness_range,
        )
        if len(chain) < 5:
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
        except ValueError:
            # Validation failed (e.g. non-positive IVs after filtering)
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

    # Liquidity filters
    df = df[df["volume"].fillna(0) >= min_volume]
    df = df[df["openInterest"].fillna(0) >= min_open_interest]

    # Spread filter
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    mid = (df["bid"] + df["ask"]) / 2
    spread_pct = (df["ask"] - df["bid"]) / mid
    df = df[spread_pct <= max_spread_pct]

    # Moneyness filter
    moneyness = df["strike"] / spot
    df = df[(moneyness >= moneyness_range[0]) & (moneyness <= moneyness_range[1])]

    # IV sanity
    df = df[df["impliedVolatility"] > 0.001]
    df = df[df["impliedVolatility"] < 5.0]

    return df.sort_values("strike").reset_index(drop=True)
