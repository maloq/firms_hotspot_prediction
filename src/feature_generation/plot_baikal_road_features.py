"""Plot Baikal-area road-density and nearest-road feature maps."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_download.osm_roads_raster import DEFAULT_DRIVABLE_HIGHWAYS, rasterize_extract


DEFAULT_BAIKAL_BBOX = (100.0, 49.0, 114.5, 58.5)  # lon_min, lat_min, lon_max, lat_max
DEFAULT_BAIKAL_PBFS = (
    Path("data/osm/pbf/siberian-fed-district-latest.osm.pbf"),
    Path("data/osm/pbf/far-eastern-fed-district-latest.osm.pbf"),
    Path("data/osm/pbf/mongolia-latest.osm.pbf"),
)


def load_road_raster(npz_path: Path) -> tuple[np.ndarray, np.ndarray, CRS, float]:
    with np.load(npz_path) as data:
        road = data["road_presence"].astype(np.uint8, copy=False)
        transform = data["transform"].astype(float)
        resolution_m = float(data["resolution_m"])
        crs_wkt = str(data["crs_wkt"])
    return road, transform, CRS.from_wkt(crs_wkt), resolution_m


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
        raise ValueError("Requested bbox does not overlap the road raster.")
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


def transform_from_bounds(
    bounds: tuple[float, float, float, float],
    resolution_m: float,
) -> np.ndarray:
    minx, _, _, maxy = bounds
    return np.array([resolution_m, 0.0, minx, 0.0, -resolution_m, maxy], dtype=np.float64)


def expand_bounds(
    bounds: tuple[float, float, float, float],
    buffer_m: float,
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    return minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m


def lonlat_bbox_from_projected(
    bounds: tuple[float, float, float, float],
    crs: CRS,
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs, ys = transformer.transform(
        [minx, minx, maxx, maxx],
        [miny, maxy, miny, maxy],
    )
    return min(xs), min(ys), max(xs), max(ys)


def trim_from_process_window(
    process_window: tuple[int, int, int, int],
    display_window: tuple[int, int, int, int],
) -> tuple[slice, slice]:
    prow0, _, pcol0, _ = process_window
    drow0, drow1, dcol0, dcol1 = display_window
    return slice(drow0 - prow0, drow1 - prow0), slice(dcol0 - pcol0, dcol1 - pcol0)


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


def build_process_raster_without_tracks(
    pbf_paths: list[Path],
    process_bounds: tuple[float, float, float, float],
    crs: CRS,
    resolution_m: float,
    batch_size: int,
) -> np.ndarray:
    minx, miny, maxx, maxy = process_bounds
    width = int(math.ceil((maxx - minx) / resolution_m))
    height = int(math.ceil((maxy - miny) / resolution_m))
    raster_bounds = (minx, miny, minx + width * resolution_m, miny + height * resolution_m)
    raster = np.zeros((height, width), dtype=np.uint8)
    bbox_lonlat = lonlat_bbox_from_projected(raster_bounds, crs)
    highways = tuple(h for h in DEFAULT_DRIVABLE_HIGHWAYS if h != "track")

    print(
        f"Building no-track Baikal crop raster: {width} x {height}, "
        f"{len(highways)} highway classes"
    )
    for pbf_path in pbf_paths:
        if not pbf_path.exists():
            print(f"Skipping missing PBF: {pbf_path}")
            continue
        print(f"Rasterizing no-track roads from {pbf_path}")
        stats = rasterize_extract(
            raster=raster,
            pbf_path=pbf_path,
            bbox_lonlat=bbox_lonlat,
            raster_bounds=raster_bounds,
            resolution_m=resolution_m,
            crs=crs,
            highways=highways,
            batch_size=batch_size,
            skip_private=False,
        )
        print(
            f"  {pbf_path.name}: {stats['features']:,} features, "
            f"{stats['burned_cells']:,} newly occupied cells"
        )
    return raster


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


def save_feature_maps(
    road_process: np.ndarray,
    display_slices: tuple[slice, slice],
    extent: tuple[float, float, float, float],
    bbox_lonlat: tuple[float, float, float, float],
    crs: CRS,
    output_dir: Path,
    prefix: str,
    title_suffix: str,
    kernels_km: list[float],
    resolution_m: float,
    density_vmax_percentile: float,
    distance_vmax_km: float,
    countries: gpd.GeoDataFrame | None,
    lakes: gpd.GeoDataFrame | None,
    dpi: int,
    axis_tick_size: int,
    colorbar_tick_size: int,
    colorbar_label_size: int,
) -> None:
    for kernel_km in kernels_km:
        sigma_pixels = kernel_km * 1000.0 / resolution_m
        density = gaussian_filter(
            road_process.astype(np.float32),
            sigma=sigma_pixels,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
        density_pct = density[display_slices] * 100.0
        vmax = robust_vmax(density_pct, density_vmax_percentile)
        save_map(
            values=density_pct,
            extent=extent,
            bbox_lonlat=bbox_lonlat,
            crs=crs,
            output_path=output_dir / f"{prefix}_road_density_gaussian_{int(kernel_km):02d}km.png",
            title=f"Road density around Lake Baikal {title_suffix}, sigma {kernel_km:g} km",
            colorbar_label="Road-cell density (%)",
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            countries=countries,
            lakes=lakes,
            dpi=dpi,
            axis_tick_size=axis_tick_size,
            colorbar_tick_size=colorbar_tick_size,
            colorbar_label_size=colorbar_label_size,
        )

    distance_km = distance_transform_edt(road_process == 0) * resolution_m / 1000.0
    save_map(
        values=distance_km[display_slices],
        extent=extent,
        bbox_lonlat=bbox_lonlat,
        crs=crs,
        output_path=output_dir / f"{prefix}_distance_to_nearest_road.png",
        title=f"Distance to nearest road around Lake Baikal {title_suffix}",
        colorbar_label="Distance to nearest road (km)",
        cmap="viridis",
        vmin=0.0,
        vmax=distance_vmax_km,
        countries=countries,
        lakes=lakes,
        dpi=dpi,
        axis_tick_size=axis_tick_size,
        colorbar_tick_size=colorbar_tick_size,
        colorbar_label_size=colorbar_label_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/land_features/osm_drivable_roads_1km.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/land_features/road_feature_maps_baikal"))
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BAIKAL_BBOX)
    parser.add_argument("--kernels-km", nargs=3, type=float, default=(5.0, 10.0, 25.0))
    parser.add_argument("--buffer-km", type=float, default=250.0)
    parser.add_argument("--density-vmax-percentile", type=float, default=99.5)
    parser.add_argument("--distance-vmax-km", type=float, default=150.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--axis-tick-size", type=int, default=12)
    parser.add_argument("--colorbar-tick-size", type=int, default=12)
    parser.add_argument("--colorbar-label-size", type=int, default=14)
    parser.add_argument("--countries-shp", type=Path, default=Path("data/countries/ne_110m_admin_0_countries.shp"))
    parser.add_argument("--lakes-shp", type=Path, default=Path("data/natural_earth/ne_10m_lakes/ne_10m_lakes.shp"))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("with_tracks", "without_tracks"),
        default=("with_tracks",),
    )
    parser.add_argument("--no-track-pbfs", nargs="*", type=Path, default=list(DEFAULT_BAIKAL_PBFS))
    parser.add_argument("--pbf-batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bbox_lonlat = tuple(args.bbox)
    road, transform, crs, resolution_m = load_road_raster(args.input)

    display_bounds = projected_bbox(bbox_lonlat, crs)
    process_bounds = expand_bounds(display_bounds, args.buffer_km * 1000.0)
    display_window = window_from_bounds(display_bounds, transform, road.shape)
    process_window = window_from_bounds(process_bounds, transform, road.shape)
    display_slices = trim_from_process_window(process_window, display_window)
    process_bounds_snapped = bounds_from_window(process_window, transform)
    extent_bounds = bounds_from_window(display_window, transform)
    extent = (extent_bounds[0], extent_bounds[2], extent_bounds[1], extent_bounds[3])

    prow0, prow1, pcol0, pcol1 = process_window
    road_process = road[prow0:prow1, pcol0:pcol1]
    countries, lakes = load_context_layers(args.countries_shp, args.lakes_shp, crs, display_bounds)

    if "with_tracks" in args.variants:
        save_feature_maps(
            road_process=road_process,
            display_slices=display_slices,
            extent=extent,
            bbox_lonlat=bbox_lonlat,
            crs=crs,
            output_dir=args.output_dir / "with_tracks",
            prefix="baikal_with_tracks",
            title_suffix="(tracks included)",
            kernels_km=list(args.kernels_km),
            resolution_m=resolution_m,
            density_vmax_percentile=args.density_vmax_percentile,
            distance_vmax_km=args.distance_vmax_km,
            countries=countries,
            lakes=lakes,
            dpi=args.dpi,
            axis_tick_size=args.axis_tick_size,
            colorbar_tick_size=args.colorbar_tick_size,
            colorbar_label_size=args.colorbar_label_size,
        )

    if "without_tracks" in args.variants:
        no_track_process = build_process_raster_without_tracks(
            pbf_paths=list(args.no_track_pbfs),
            process_bounds=process_bounds_snapped,
            crs=crs,
            resolution_m=resolution_m,
            batch_size=args.pbf_batch_size,
        )
        save_feature_maps(
            road_process=no_track_process,
            display_slices=display_slices,
            extent=extent,
            bbox_lonlat=bbox_lonlat,
            crs=crs,
            output_dir=args.output_dir / "without_tracks",
            prefix="baikal_without_tracks",
            title_suffix="(tracks excluded)",
            kernels_km=list(args.kernels_km),
            resolution_m=resolution_m,
            density_vmax_percentile=args.density_vmax_percentile,
            distance_vmax_km=args.distance_vmax_km,
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
