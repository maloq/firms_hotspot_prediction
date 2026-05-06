"""Plot Baikal-area night-light density and nearest-light feature maps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import box


DEFAULT_BAIKAL_BBOX = (100.0, 49.0, 114.5, 58.5)  # lon_min, lat_min, lon_max, lat_max


def load_feature_store(feature_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, CRS, float]:
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing feature-store manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    features: dict[str, np.ndarray] = {}
    expected_shape = tuple(int(v) for v in manifest["shape"])
    for feature_name, spec in manifest["features"].items():
        array_path = feature_dir / spec["path"]
        if not array_path.exists():
            raise FileNotFoundError(f"Missing feature array: {array_path}")
        arr = np.load(array_path, mmap_mode="r")
        if tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"Feature {feature_name} shape {arr.shape} does not match manifest {expected_shape}"
            )
        features[feature_name] = arr

    transform = np.asarray(manifest["transform"], dtype=np.float64)
    crs = CRS.from_wkt(manifest["crs_wkt"])
    resolution_m = float(manifest["resolution_m"])
    return features, transform, crs, resolution_m


def projected_bbox(
    bbox_lonlat: tuple[float, float, float, float],
    crs: CRS,
) -> tuple[float, float, float, float]:
    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(
        [lon_min, lon_min, lon_max, lon_max],
        [lat_min, lat_max, lat_min, lat_max],
    )
    return min(xs), min(ys), max(xs), max(ys)


def window_from_bounds(
    bounds: tuple[float, float, float, float],
    transform: np.ndarray,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = bounds
    res_x, _, origin_x, _, res_y, origin_y = transform
    height, width = shape

    col0 = max(0, int(math.floor((minx - origin_x) / res_x)))
    col1 = min(width, int(math.ceil((maxx - origin_x) / res_x)))
    row0 = max(0, int(math.floor((origin_y - maxy) / abs(res_y))))
    row1 = min(height, int(math.ceil((origin_y - miny) / abs(res_y))))
    if row0 >= row1 or col0 >= col1:
        raise ValueError("Requested bbox does not overlap the feature raster.")
    return row0, row1, col0, col1


def bounds_from_window(
    window: tuple[int, int, int, int],
    transform: np.ndarray,
) -> tuple[float, float, float, float]:
    row0, row1, col0, col1 = window
    res_x, _, origin_x, _, res_y, origin_y = transform
    left = origin_x + col0 * res_x
    right = origin_x + col1 * res_x
    top = origin_y + row0 * res_y
    bottom = origin_y + row1 * res_y
    return left, bottom, right, top


def load_context_layers(
    countries_path: Path | None,
    lakes_path: Path | None,
    crs: CRS,
    bounds: tuple[float, float, float, float],
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    context_bbox = gpd.GeoSeries([box(*bounds)], crs=crs).to_crs("EPSG:4326").total_bounds
    bbox_tuple = tuple(context_bbox)

    countries = None
    if countries_path and countries_path.exists():
        countries = gpd.read_file(countries_path, bbox=bbox_tuple).to_crs(crs)

    lakes = None
    if lakes_path and lakes_path.exists():
        lakes = gpd.read_file(lakes_path, bbox=bbox_tuple).to_crs(crs)
        name_cols = [col for col in ("name", "NAME", "Name") if col in lakes.columns]
        if name_cols:
            name_col = name_cols[0]
            baikal = lakes[lakes[name_col].astype(str).str.contains("Baikal", case=False, na=False)]
            if not baikal.empty:
                lakes = baikal

    return countries, lakes


def add_lonlat_ticks(
    ax: plt.Axes,
    bbox_lonlat: tuple[float, float, float, float],
    crs: CRS,
    tick_label_size: int,
) -> None:
    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    lon_ticks = np.arange(math.ceil(lon_min / 2) * 2, lon_max + 0.1, 2)
    lat_ticks = np.arange(math.ceil(lat_min), lat_max + 0.1, 1)
    xticks, _ = transformer.transform(lon_ticks, np.full_like(lon_ticks, center_lat))
    _, yticks = transformer.transform(np.full_like(lat_ticks, center_lon), lat_ticks)

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{lon:g}E" for lon in lon_ticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{lat:g}N" for lat in lat_ticks])
    ax.tick_params(labelsize=tick_label_size)
    ax.set_xlabel("")
    ax.set_ylabel("")


def robust_vmax(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return 1.0
    return float(np.percentile(positive, percentile))


def save_map(
    values: np.ndarray,
    extent: tuple[float, float, float, float],
    bbox_lonlat: tuple[float, float, float, float],
    crs: CRS,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    countries: gpd.GeoDataFrame | None,
    lakes: gpd.GeoDataFrame | None,
    dpi: int,
    axis_tick_size: int,
    colorbar_tick_size: int,
    colorbar_label_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 7.8), constrained_layout=False)
    image = ax.imshow(
        values,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if countries is not None and not countries.empty:
        countries.boundary.plot(ax=ax, color="white", linewidth=0.35, alpha=0.75)
    if lakes is not None and not lakes.empty:
        lakes.boundary.plot(ax=ax, color="#8ecae6", linewidth=1.0, alpha=0.95)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    add_lonlat_ticks(ax, bbox_lonlat, crs, axis_tick_size)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.18)
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_tick_size)
    cbar.set_label(colorbar_label, fontsize=colorbar_label_size)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/land_features/night_lights_features_1km"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/land_features/night_light_feature_maps_baikal"))
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BAIKAL_BBOX)
    parser.add_argument("--kernels-km", nargs="+", type=float, default=(5.0, 10.0, 25.0))
    parser.add_argument("--density-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--radiance-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--distance-vmax-km", type=float, default=150.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--axis-tick-size", type=int, default=13)
    parser.add_argument("--colorbar-tick-size", type=int, default=13)
    parser.add_argument("--colorbar-label-size", type=int, default=15)
    parser.add_argument("--countries-shp", type=Path, default=Path("data/countries/ne_110m_admin_0_countries.shp"))
    parser.add_argument("--lakes-shp", type=Path, default=Path("data/natural_earth/ne_10m_lakes/ne_10m_lakes.shp"))
    parser.add_argument("--skip-radiance", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bbox_lonlat = tuple(args.bbox)
    features, transform, crs, _ = load_feature_store(args.input_dir)

    first = next(iter(features.values()))
    display_bounds = projected_bbox(bbox_lonlat, crs)
    display_window = window_from_bounds(display_bounds, transform, first.shape)
    extent_bounds = bounds_from_window(display_window, transform)
    extent = (extent_bounds[0], extent_bounds[2], extent_bounds[1], extent_bounds[3])
    row0, row1, col0, col1 = display_window
    window = np.s_[row0:row1, col0:col1]

    countries, lakes = load_context_layers(args.countries_shp, args.lakes_shp, crs, display_bounds)

    for kernel_km in args.kernels_km:
        label = f"{int(kernel_km)}km" if float(kernel_km).is_integer() else f"{kernel_km:g}km"
        feature_name = f"light_density_gaussian_{label}"
        if feature_name not in features:
            print(f"Skipping missing feature: {feature_name}")
            continue
        density_pct = np.asarray(features[feature_name][window], dtype=np.float32) * 100.0
        vmax = robust_vmax(density_pct, args.density_vmax_percentile)
        save_map(
            values=density_pct,
            extent=extent,
            bbox_lonlat=bbox_lonlat,
            crs=crs,
            output_path=args.output_dir / f"baikal_light_density_gaussian_{int(kernel_km):02d}km.png",
            title=f"Night-light density around Lake Baikal, sigma {kernel_km:g} km",
            colorbar_label="Light-source cell density (%)",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            countries=countries,
            lakes=lakes,
            dpi=args.dpi,
            axis_tick_size=args.axis_tick_size,
            colorbar_tick_size=args.colorbar_tick_size,
            colorbar_label_size=args.colorbar_label_size,
        )

    if "distance_to_light_source_meters" in features:
        distance_km = np.asarray(features["distance_to_light_source_meters"][window], dtype=np.float32) / 1000.0
        save_map(
            values=distance_km,
            extent=extent,
            bbox_lonlat=bbox_lonlat,
            crs=crs,
            output_path=args.output_dir / "baikal_distance_to_nearest_light_source.png",
            title="Distance to nearest night-light source around Lake Baikal",
            colorbar_label="Distance to nearest light source (km)",
            cmap="viridis",
            vmin=0.0,
            vmax=args.distance_vmax_km,
            countries=countries,
            lakes=lakes,
            dpi=args.dpi,
            axis_tick_size=args.axis_tick_size,
            colorbar_tick_size=args.colorbar_tick_size,
            colorbar_label_size=args.colorbar_label_size,
        )

    if not args.skip_radiance and "night_light_radiance_2024" in features:
        radiance = np.asarray(features["night_light_radiance_2024"][window], dtype=np.float32)
        vmax = robust_vmax(radiance, args.radiance_vmax_percentile)
        save_map(
            values=radiance,
            extent=extent,
            bbox_lonlat=bbox_lonlat,
            crs=crs,
            output_path=args.output_dir / "baikal_night_light_radiance_2024.png",
            title="Night-light radiance around Lake Baikal, 2024",
            colorbar_label="VIIRS night-light value",
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            countries=countries,
            lakes=lakes,
            dpi=args.dpi,
            axis_tick_size=args.axis_tick_size,
            colorbar_tick_size=args.colorbar_tick_size,
            colorbar_label_size=args.colorbar_label_size,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
