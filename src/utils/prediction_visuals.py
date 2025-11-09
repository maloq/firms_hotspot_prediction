"""Shared plotting and export helpers for prediction pipelines."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

try:  # optional dependency; import lazily if available
    import xarray as xr
except ImportError:  # pragma: no cover - optional dependency
    xr = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

DEFAULT_PLOT_DPI = 300


def to_scalar(obj: object) -> object:
    """Extract the first scalar value from pandas/NumPy containers."""

    if pd.api.types.is_scalar(obj):
        return obj
    if isinstance(obj, pd.Series):
        return obj.iloc[0]
    if isinstance(obj, (np.ndarray, list, tuple)):
        return obj[0]
    return obj


def resolve_color_scale(values_array: np.ndarray) -> tuple[float, float]:
    """Return a robust color scale range for probability-like data."""

    if values_array.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(values_array))
    vmax = float(np.nanmax(values_array))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0, 1.0
    spread = vmax - vmin
    padding = 0.05 * spread if spread > 0 else 0.0
    vmin = max(0.0, vmin - padding)
    vmax = min(1.0, vmax + padding)
    return vmin, vmax


def _figure_and_axes() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(16, 12))


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _attach_colorbar(ax: plt.Axes, mappable: ScalarMappable, label: str) -> Colorbar:
    """Attach a compact colorbar sized to the given axes."""

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)
    colorbar = ax.figure.colorbar(mappable, cax=cax)
    colorbar.set_label(label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    return colorbar


def plot_prediction_map(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    predictions: np.ndarray,
    title: str,
    save_path: str,
    borders_gdf: "gpd.GeoDataFrame | None" = None,
    dpi: int = DEFAULT_PLOT_DPI,
) -> None:
    """Plot interpolated prediction heatmap and persist to disk."""

    fig, ax = _figure_and_axes()
    valid_mask = ~np.isnan(predictions)

    if np.any(valid_mask):
        lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)
        points = np.column_stack((lon_mesh[valid_mask], lat_mesh[valid_mask]))
        values = predictions[valid_mask]
        vmin, vmax = resolve_color_scale(values)

        interp_lon = np.linspace(lon_coords.min(), lon_coords.max(), len(lon_coords) * 5)
        interp_lat = np.linspace(lat_coords.min(), lat_coords.max(), len(lat_coords) * 5)
        interp_lon_mesh, interp_lat_mesh = np.meshgrid(interp_lon, interp_lat)

        interp_linear = griddata(points, values, (interp_lon_mesh, interp_lat_mesh), method="linear", fill_value=np.nan)
        interp_nearest = griddata(points, values, (interp_lon_mesh, interp_lat_mesh), method="nearest")
        interp_values = np.where(np.isnan(interp_linear), interp_nearest, interp_linear)

        mesh = ax.pcolormesh(
            interp_lon,
            interp_lat,
            interp_values,
            cmap="YlOrRd",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
    else:  # pragma: no cover - degenerate case
        vmin, vmax = resolve_color_scale(predictions[valid_mask])
        mesh = ax.pcolormesh(
            lon_coords,
            lat_coords,
            predictions,
            cmap="YlOrRd",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

    _attach_colorbar(ax, mesh, "Prediction Probability")

    if borders_gdf is not None:  # pragma: no branch - simple guard
        borders_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.5)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(lon_coords.min(), lon_coords.max())
    ax.set_ylim(lat_coords.min(), lat_coords.max())

    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _resolve_feature_scale(feature_grid: np.ndarray) -> tuple[float, float]:
    valid_vals = feature_grid[~np.isnan(feature_grid)]
    if valid_vals.size >= 10:
        vmin, vmax = np.nanpercentile(valid_vals, [2, 98])
        if vmin == vmax:
            vmin, vmax = valid_vals.min(), valid_vals.max()
    else:
        vmin, vmax = np.nanmin(feature_grid), np.nanmax(feature_grid)

    if np.isnan(vmin) or np.isnan(vmax):
        return 0.0, 1.0
    return float(vmin), float(vmax)


def plot_feature_map(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    feature_grid: np.ndarray,
    feature_name: str,
    title: str,
    save_path: str,
    borders_gdf: "gpd.GeoDataFrame | None" = None,
    dpi: int = DEFAULT_PLOT_DPI,
) -> None:
    """Plot a grid-based feature heatmap and persist to disk."""

    vmin, vmax = _resolve_feature_scale(feature_grid)

    fig, ax = _figure_and_axes()
    mesh = ax.pcolormesh(
        lon_coords,
        lat_coords,
        feature_grid,
        cmap="viridis",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    _attach_colorbar(ax, mesh, feature_name)

    if borders_gdf is not None:  # pragma: no branch - simple guard
        borders_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.5)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(lon_coords.min(), lon_coords.max())
    ax.set_ylim(lat_coords.min(), lat_coords.max())

    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_feature_maps_group(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    feature_maps: Sequence[tuple[str, np.ndarray, Optional[float]]],
    save_path: str,
    title: str | None = None,
    borders_gdf: "gpd.GeoDataFrame | None" = None,
    dpi: int = DEFAULT_PLOT_DPI,
) -> None:
    """Plot multiple feature grids in a single row of subplots."""

    entries = [(name, grid, importance) for name, grid, importance in feature_maps if grid.size]
    if not entries:
        return

    entries = entries[:3]

    # Choose figure size based on map aspect ratio to minimize whitespace
    lon_span = float(np.nanmax(lon_coords) - np.nanmin(lon_coords))
    lat_span = float(np.nanmax(lat_coords) - np.nanmin(lat_coords))
    aspect = (lat_span / lon_span) if lon_span > 0 else 0.35
    fig_width = 12.0
    row_height = max(2.6, fig_width * aspect)  # ensure enough height for labels and colorbar
    fig_height = row_height * 3 + 0.8  # small extra headroom

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_iter = axes.flatten()

    for idx, ax in enumerate(axes_iter):
        if idx >= len(entries):
            ax.axis("off")
            continue

        feature_name, feature_grid, importance = entries[idx]
        if feature_grid.size == 0 or np.isnan(feature_grid).all():
            ax.axis("off")
            continue

        vmin, vmax = _resolve_feature_scale(feature_grid)
        mesh = ax.pcolormesh(
            lon_coords,
            lat_coords,
            feature_grid,
            cmap="viridis",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        _attach_colorbar(ax, mesh, feature_name)

        if borders_gdf is not None:  # pragma: no branch - simple guard
            borders_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.4)

        importance_str = ""
        if importance is not None and np.isfinite(importance):
            importance_str = f" ({importance:.4f})"
        ax.set_title(f"{feature_name}{importance_str}", fontsize=12)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(axis='both', labelsize=9, direction='out', length=3, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.set_xlim(lon_coords.min(), lon_coords.max())
        ax.set_ylim(lat_coords.min(), lat_coords.max())

    if title:
        fig.suptitle(title, fontsize=14)

    # Compact layout for publication-quality spacing
    fig.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.06, hspace=0.18)

    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_modis_fires(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    modis_data: pd.DataFrame,
    prediction_date: pd.Timestamp,
    n_days: int,
    title: str,
    save_path: str,
    borders_gdf: "gpd.GeoDataFrame | None" = None,
    dpi: int = DEFAULT_PLOT_DPI,
) -> None:
    """Plot MODIS fire detections within a date window."""

    if modis_data is None or modis_data.empty:
        print(
            f"Warning: No MODIS fire data available for plotting around {prediction_date.strftime('%Y-%m-%d')}"
        )
        return

    fig, ax = _figure_and_axes()
    scatter = ax.scatter(
        modis_data["longitude"],
        modis_data["latitude"],
        c=modis_data["brightness"],
        cmap="hot",
        s=5,
        alpha=0.6,
        vmin=modis_data["brightness"].quantile(0.05),
        vmax=modis_data["brightness"].quantile(0.95),
    )

    _attach_colorbar(ax, scatter, "Fire Brightness (K)")

    if borders_gdf is not None:  # pragma: no branch - simple guard
        borders_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.5)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(lon_coords.min(), lon_coords.max())
    ax.set_ylim(lat_coords.min(), lat_coords.max())

    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_prediction_as_netcdf(
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    predictions: np.ndarray,
    date_timestamp: pd.Timestamp,
    save_path: str,
    prediction_type: str = "raw",
) -> None:
    """Persist prediction grid into a NetCDF file."""

    if xr is None:
        raise RuntimeError(
            "xarray is required to export NetCDF files; install xarray to enable this output."
        )

    variable_name = f"fire_probability_{prediction_type}"
    ds = xr.Dataset(
        data_vars={
            variable_name: (["latitude", "longitude"], predictions),
        },
        coords={
            "latitude": lat_coords,
            "longitude": lon_coords,
        },
        attrs={
            "description": f"Fire prediction probabilities ({prediction_type})",
            "date": str(date_timestamp.strftime("%Y-%m-%d")),
            "prediction_type": prediction_type,
            "created": str(pd.Timestamp.now()),
        },
    )

    time_dt64 = np.datetime64(date_timestamp)
    ds = ds.expand_dims(time=[time_dt64])

    ds["latitude"].attrs["units"] = "degrees_north"
    ds["latitude"].attrs["long_name"] = "Latitude"
    ds["longitude"].attrs["units"] = "degrees_east"
    ds["longitude"].attrs["long_name"] = "Longitude"
    ds["time"].attrs["long_name"] = "Time"

    _ensure_parent_dir(save_path)
    ds.to_netcdf(save_path)
    ds.close()
