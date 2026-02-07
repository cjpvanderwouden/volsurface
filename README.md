# volsurface

[![CI](https://github.com/cjpvanderwouden/volsurface/actions/workflows/ci.yml/badge.svg)](https://github.com/cjpvanderwouden/volsurface/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Typed](https://img.shields.io/badge/typing-mypy%20strict-brightgreen.svg)](https://mypy-lang.org/)

A typed, extensible Python toolkit for building, calibrating, and inspecting implied volatility surfaces.

## Features

- **Clean data structures** — `MarketSlice`, `VolSurface`, `VolSurfacePoint` with full validation
- **SVI parameterisation** — Raw SVI with robust L-BFGS-B calibration and smart initial guesses
- **Arbitrage detection** — Butterfly (convexity) and calendar spread checks
- **Yahoo Finance integration** — Fetch, filter, and clean live option chains
- **Surface interpolation** — Linear interpolation in total-variance space across expiries
- **Visualisation** — 3D surface plots, heatmaps, and smile overlays
- **Fully typed** — `mypy --strict` compliant with `py.typed` marker
- **Extensible** — Add new models by implementing the `VolModel` base class

## Installation

```bash
pip install volsurface            # core only
pip install volsurface[yahoo]     # + Yahoo Finance data
pip install volsurface[plot]      # + matplotlib plotting
pip install volsurface[all]       # everything
```

## Quick Start

```python
from volsurface.market_data import fetch_chain
from volsurface.calibration import calibrate_surface
from volsurface.models import RawSVI
from volsurface.plotting import plot_surface

# Fetch and clean AAPL option chains from Yahoo Finance
slices = fetch_chain("AAPL")

# Fit Raw SVI to each expiry and assemble the surface
result = calibrate_surface(slices, RawSVI, ticker="AAPL")
print(f"Fitted {result.surface.n_expiries} expiries")
print(f"Arbitrage clean: {result.arbitrage_report.is_clean}")

# Query any (strike, expiry) point
point = result.surface.iv(strike=200.0, expiry_years=0.5)
print(f"IV at K=200, T=0.5y: {point.iv:.4f}")

# Visualise
plot_surface(result.surface, kind="3d")
```

## Working with Custom Data

```python
import numpy as np
from volsurface.market_data import clean_chain
from volsurface.models import RawSVI

# Your own data
strikes = np.array([90, 95, 100, 105, 110], dtype=float)
ivs = np.array([0.25, 0.22, 0.20, 0.21, 0.24])

slice_ = clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)

model = RawSVI()
result = model.fit(slice_)
print(f"RMSE: {result.rmse:.6f}")
print(f"Params: {result.params}")
```

## Adding a New Model

Implement the `VolModel` abstract base class:

```python
from volsurface.models.base import VolModel
from volsurface.core import FitResult, MarketSlice

class MyModel(VolModel):
    @property
    def n_params(self) -> int:
        return ...

    def fit(self, market_slice: MarketSlice) -> FitResult:
        ...

    def total_variance(self, log_moneyness):
        ...
```

## Development

```bash
git clone https://github.com/cjpvanderwouden/volsurface.git
cd volsurface
pip install -e ".[dev]"

# Run checks
pytest                  # tests + coverage
ruff check src/ tests/  # linting
ruff format src/ tests/ # formatting
mypy src/               # type checking
```

## License

MIT
