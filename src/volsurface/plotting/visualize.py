"""Plotting functions for volatility smiles and surfaces.

Requires the optional ``matplotlib`` dependency::

    pip install volsurface[plot]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from volsurface.core import MarketSlice, VolSurface


def _import_matplotlib() -> Any:
    """Lazily import matplotlib, raising a clear error if absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        msg = "matplotlib is required for plotting. Install it with:  pip install volsurface[plot]"
        raise ImportError(msg) from None
    return plt


def plot_smile(
    market_slice: MarketSlice,
    model: Any | None = None,
    *,
    n_model_points: int = 200,
    ax: matplotlib.axes.Axes | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a single implied volatility smile with optional model overlay.

    Args:
        market_slice: Market data to plot as scatter points.
        model: Optional fitted VolModel — if provided, its smooth
            smile curve is plotted alongside the market data.
        n_model_points: Number of points for the model curve.
        ax: Optional matplotlib Axes to plot on.
        show: Whether to call ``plt.show()`` at the end.

    Returns:
        The matplotlib Figure.
    """
    plt = _import_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    k = market_slice.log_moneyness
    ax.scatter(k, market_slice.ivs, s=30, color="steelblue", label="Market", zorder=3)

    if model is not None:
        k_fine = np.linspace(float(k.min()) - 0.05, float(k.max()) + 0.05, n_model_points)
        iv_model = model.iv(k_fine)
        ax.plot(k_fine, iv_model, color="crimson", linewidth=1.5, label="Model fit")

    T_label = f"T = {market_slice.expiry_years:.3f}y"
    ticker_label = f" ({market_slice.ticker})" if market_slice.ticker else ""
    ax.set_title(f"Implied Volatility Smile — {T_label}{ticker_label}")
    ax.set_xlabel("Log-moneyness  ln(K/F)")
    ax.set_ylabel("Implied Volatility")
    ax.legend()
    ax.grid(alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return fig  # type: ignore[no-any-return]


def plot_surface(
    surface: VolSurface,
    *,
    n_strike_points: int = 80,
    kind: str = "3d",
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Plot a fitted volatility surface.

    Args:
        surface: A fitted VolSurface with at least 1 expiry.
        n_strike_points: Number of strike points per expiry.
        kind: ``"3d"`` for a 3D surface plot, ``"heatmap"`` for a 2D
            heatmap, ``"smiles"`` for overlaid per-expiry smiles.
        show: Whether to call ``plt.show()`` at the end.

    Returns:
        The matplotlib Figure.

    Raises:
        ValueError: If the surface has no fitted slices or kind is unknown.
    """
    plt = _import_matplotlib()

    if surface.n_expiries == 0:
        msg = "Surface has no fitted slices to plot"
        raise ValueError(msg)

    if kind == "smiles":
        return _plot_smile_overlay(surface, n_strike_points, plt, show)  # type: ignore[no-any-return]
    if kind == "3d":
        return _plot_3d(surface, n_strike_points, plt, show)  # type: ignore[no-any-return]
    if kind == "heatmap":
        return _plot_heatmap(surface, n_strike_points, plt, show)  # type: ignore[no-any-return]

    msg = f"Unknown plot kind: {kind!r}. Use '3d', 'heatmap', or 'smiles'."
    raise ValueError(msg)


# ── Private plotting helpers ────────────────────────────────────────


def _build_k_grid(surface: VolSurface, n_points: int) -> np.ndarray:
    """Build a common log-moneyness grid across all expiries."""
    all_k = np.concatenate([surface.market_data[t].log_moneyness for t in surface.expiries])
    return np.linspace(float(all_k.min()), float(all_k.max()), n_points)


def _plot_smile_overlay(surface: VolSurface, n_points: int, plt: Any, show: bool) -> Any:
    fig, ax = plt.subplots(figsize=(10, 6))
    k_grid = _build_k_grid(surface, n_points)
    cmap = plt.cm.viridis

    for i, t in enumerate(surface.expiries):
        colour = cmap(i / max(len(surface.expiries) - 1, 1))
        model = surface.slices[t]
        iv_vals = model.iv(k_grid)
        ax.plot(k_grid, iv_vals, color=colour, label=f"T={t:.3f}y")

    ax.set_xlabel("Log-moneyness  ln(K/F)")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Volatility Smiles — {surface.ticker or ''}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    return fig


def _plot_3d(surface: VolSurface, n_points: int, plt: Any, show: bool) -> Any:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    k_grid = _build_k_grid(surface, n_points)

    K, T = np.meshgrid(k_grid, surface.expiries)
    IV = np.zeros_like(K)

    for i, t in enumerate(surface.expiries):
        model = surface.slices[t]
        IV[i, :] = model.iv(k_grid)

    ax.plot_surface(K, T, IV, cmap="viridis", alpha=0.85, edgecolor="none")
    ax.set_xlabel("Log-moneyness  ln(K/F)")
    ax.set_ylabel("Expiry (years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"Implied Volatility Surface — {surface.ticker or ''}")

    if show:
        plt.tight_layout()
        plt.show()
    return fig


def _plot_heatmap(surface: VolSurface, n_points: int, plt: Any, show: bool) -> Any:
    fig, ax = plt.subplots(figsize=(12, 6))
    k_grid = _build_k_grid(surface, n_points)

    IV = np.zeros((len(surface.expiries), n_points))
    for i, t in enumerate(surface.expiries):
        model = surface.slices[t]
        IV[i, :] = model.iv(k_grid)

    im = ax.imshow(
        IV,
        aspect="auto",
        origin="lower",
        extent=[k_grid[0], k_grid[-1], surface.expiries[0], surface.expiries[-1]],
        cmap="RdYlBu_r",
    )
    fig.colorbar(im, ax=ax, label="Implied Volatility")
    ax.set_xlabel("Log-moneyness  ln(K/F)")
    ax.set_ylabel("Expiry (years)")
    ax.set_title(f"IV Heatmap — {surface.ticker or ''}")

    if show:
        plt.tight_layout()
        plt.show()
    return fig
