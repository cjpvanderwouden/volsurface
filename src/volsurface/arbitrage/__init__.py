"""Static no-arbitrage checks for implied volatility surfaces."""

from volsurface.arbitrage.checks import check_butterfly, check_calendar, check_slice

__all__ = ["check_butterfly", "check_calendar", "check_slice"]
