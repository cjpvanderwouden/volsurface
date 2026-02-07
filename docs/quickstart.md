# Quick Start

## Fetch and Fit

```python
from volsurface.market_data import fetch_chain
from volsurface.calibration import calibrate_surface
from volsurface.models import RawSVI

slices = fetch_chain("AAPL")
result = calibrate_surface(slices, RawSVI, ticker="AAPL")

# Query the surface
point = result.surface.iv(strike=200.0, expiry_years=0.5)
print(f"IV: {point.iv:.4f}")
```

## Working with Custom Data

```python
import numpy as np
from volsurface.market_data import clean_chain
from volsurface.models import RawSVI

strikes = np.array([90, 95, 100, 105, 110], dtype=float)
ivs = np.array([0.25, 0.22, 0.20, 0.21, 0.24])

slice_ = clean_chain(strikes, ivs, expiry_years=0.25, forward=100.0, spot=100.0)

model = RawSVI()
result = model.fit(slice_)
```

## Checking for Arbitrage

```python
from volsurface.arbitrage import check_slice

report = check_slice(slice_)
print(f"Clean: {report.is_clean}")
```

## Plotting

```python
from volsurface.plotting import plot_smile, plot_surface

plot_smile(slice_, model=model)
plot_surface(result.surface, kind="3d")
```
