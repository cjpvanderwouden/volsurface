"""Quick-start example: fetch AAPL options, fit SVI, and plot.

Run with::

    pip install volsurface[all]
    python examples/quickstart.py
"""

from __future__ import annotations

from volsurface.calibration import calibrate_surface
from volsurface.market_data import fetch_chain
from volsurface.models import RawSVI
from volsurface.plotting import plot_smile, plot_surface


def main() -> None:
    # 1. Fetch live option chains
    print("One second.. fetching option chains from Yahoo Finance...")
    slices = fetch_chain("SPY", min_volume=30, moneyness_range=(0.7, 1.3))
    print(f" --> {len(slices)} expiries with sufficient liquidity\n")

    # 2. Calibrate Raw SVI to each expiry
    print("Calibrating Raw SVI to each expiry...")
    result = calibrate_surface(slices, RawSVI, ticker="SPY")

    for t, fr in sorted(result.fit_results.items()):
        status = "checkmark --" if fr.success else "violation --"
        print(f"  {status} T={t:.4f}y  RMSE={fr.rmse:.6f}  ({fr.n_iterations} iters)")

    # 3. Arbitrage diagnostics
    print()
    if result.arbitrage_report and result.arbitrage_report.is_clean:
        print("No arbitrage violations detected checkmark")
    else:
        report = result.arbitrage_report
        if report:
            print(f"Butterfly violations: {len(report.butterfly_violations)}")
            print(f"Calendar violations:  {len(report.calendar_violations)}")

    # 4. Query the surface
    print()
    pt = result.surface.iv(strike=slices[0].forward, expiry_years=slices[0].expiry_years)
    print(f"ATM IV at T={pt.expiry_years:.4f}y: {pt.iv:.4f}")

    # 5. Plot
    # Single smile with market data overlay
    first_expiry = result.surface.expiries[0]
    model = result.surface.slices[first_expiry]
    market = result.surface.market_data[first_expiry]
    plot_smile(market, model=model, show=True)

    # Full surface
    plot_surface(result.surface, kind="smiles", show=True)
    plot_surface(result.surface, kind="3d", show=True)


if __name__ == "__main__":
    main()