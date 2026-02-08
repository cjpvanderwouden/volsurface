# volsurface

A typed, extensible Python toolkit for building, calibrating, and inspecting implied volatility surfaces.

## Overview

`volsurface` provides a clean, well-typed interface for working with implied volatility surfaces in Python. It's designed around a small number of core abstractions — `MarketSlice`, `VolSurface`, and `VolModel` — that compose together to support the full workflow from raw option data to a queryable, arbitrage-checked surface.

## Key Concepts

- **MarketSlice**: Cleaned, validated option chain data for a single expiry
- **VolModel**: A parameterisation of the implied volatility smile (e.g. SVI)
- **VolSurface**: A collection of fitted smile models that form a queryable 2D surface
- **CalibrationResult**: Diagnostics from fitting, including arbitrage checks

## Installation

```bash
pip install volsurface[all]
```

See the [Quick Start](quickstart.md) for a walkthrough using AAPL options.
