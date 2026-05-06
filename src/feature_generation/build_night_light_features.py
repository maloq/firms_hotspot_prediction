"""Build whole-area night-light feature maps on the model 1 km grid."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt, gaussian_filter


ZENODO_VIIRS_2024_URL = (
    "https://zenodo.org/records/17294744/files/"
    "nightlights.average_viirs.v21_m_500m_s_20240101_20241231_go_epsg4326_v20250904.tif?download=1"
)


def _set_memory_limit(memory_limit_gb: float | None) -> None:
    if memory_limit_gb is None or memory_limit_gb <= 0:
        return
    limit = int(memory_limit_gb * 1024**3)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    new_hard = hard if hard != resource.RLIM_INFINITY else limit
    resource.setrlimit(resource.RLIMIT_AS, (limit, new_hard))
    print(f"Set address-space memory limit to {limit / 1024**3:g} GB")


def _kernel_label(kernel_km: float) -> str:
    if float(kernel_km).is_integer():
        return f"{int(kernel_km)}km"
    return f"{str(kernel_km).replace('.', 'p')}km"


def _load_target_grid(path: Path) -> dict:
    if path.is_dir():
        manifest_path = path / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        return {
            "shape": tuple(int(v) for v in manifest["shape"]),
            "transform": tuple(float(v) for v in manifest["transform"]),
            "crs_wkt": str(manifest["crs_wkt"]),
            "resolution_m": float(manifest["resolution_m"]),
            "bbox_lonlat": manifest.get("bbox_lonlat"),
            "raster_bounds": manifest.get("raster_bounds"),
        }

    with np.load(path) as data:
        shape = tuple(int(v) for v in data["road_presence"].shape)
        return {
            "shape": shape,
            "transform": tuple(float(v) for v in data["transform"].flatten()[:6]),
            "crs_wkt": str(data["crs_wkt"].item() if data["crs_wkt"].shape == () else data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "bbox_lonlat": data["bbox_lonlat"].astype(float).tolist()
            if "bbox_lonlat" in data
            else None,
            "raster_bounds": data["raster_bounds"].astype(float).tolist()
            if "raster_bounds" in data
            else None,
        }


def _source_grid_from_tiff(image: Image.Image) -> dict:
    scale = image.tag_v2[33550]
    tiepoint = image.tag_v2[33922]
    pixel_width = float(scale[0])
    pixel_height = float(scale[1])
    origin_lon = float(tiepoint[3]) - float(tiepoint[0]) * pixel_width
    origin_lat = float(tiepoint[4]) + float(tiepoint[1]) * pixel_height
    nodata_raw = image.tag_v2.get(42113, "-32768")
    nodata = float(str(nodata_raw).split()[0])
    width, height = image.size
    return {
        "width": int(width),
        "height": int(height),
        "origin_lon": origin_lon,
        "origin_lat": origin_lat,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "nodata": nodata,
    }


def _target_centers_for_rows(
    transform: tuple[float, float, float, float, float, float],
    row0: int,
    row1: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d, e, f = transform
    rows = np.arange(row0, row1, dtype=np.float64) + 0.5
    cols = np.arange(width, dtype=np.float64) + 0.5
    xs = c + cols * a
    ys = f + rows * e
    x_grid = np.broadcast_to(xs[None, :], (row1 - row0, width))
    y_grid = np.broadcast_to(ys[:, None], (row1 - row0, width))
    if b != 0.0 or d != 0.0:
        x_grid = x_grid + rows[:, None] * b
        y_grid = y_grid + cols[None, :] * d
    return x_grid, y_grid


def _sample_source_chunk(
    image: Image.Image,
    source_grid: dict,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    light_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    src_cols = np.floor((lons - source_grid["origin_lon"]) / source_grid["pixel_width"]).astype(np.int64)
    src_rows = np.floor((source_grid["origin_lat"] - lats) / source_grid["pixel_height"]).astype(np.int64)
    valid = (
        np.isfinite(lons)
        & np.isfinite(lats)
        & (src_rows >= 0)
        & (src_rows < source_grid["height"])
        & (src_cols >= 0)
        & (src_cols < source_grid["width"])
    )

    radiance = np.zeros(lons.shape, dtype=np.float32)
    if not np.any(valid):
        return radiance, np.zeros(lons.shape, dtype=np.uint8)

    valid_rows = src_rows[valid]
    valid_cols = src_cols[valid]
    row_min = int(valid_rows.min())
    row_max = int(valid_rows.max()) + 1
    col_min = int(valid_cols.min())
    col_max = int(valid_cols.max()) + 1

    crop = image.crop((col_min, row_min, col_max, row_max))
    crop_arr = np.asarray(crop, dtype=np.int32)
    sampled = crop_arr[valid_rows - row_min, valid_cols - col_min].astype(np.float32, copy=False)
    sampled = np.where(sampled == source_grid["nodata"], 0.0, sampled)
    sampled = np.where(sampled > 0.0, sampled, 0.0)
    radiance[valid] = sampled

    presence = (radiance > light_threshold).astype(np.uint8)
    return radiance, presence


def _sample_night_lights_to_target_grid(
    source_tif: Path,
    target_grid: dict,
    output_dir: Path,
    *,
    light_threshold: float,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(source_tif)
    source_grid = _source_grid_from_tiff(image)

    shape = target_grid["shape"]
    radiance_path = output_dir / "night_light_radiance_2024.float32.npy"
    presence_path = output_dir / "night_light_presence_1km.uint8.npy"
    radiance = np.lib.format.open_memmap(radiance_path, mode="w+", dtype=np.float32, shape=shape)
    presence = np.lib.format.open_memmap(presence_path, mode="w+", dtype=np.uint8, shape=shape)

    target_crs = CRS.from_wkt(target_grid["crs_wkt"])
    to_wgs84 = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    height, width = shape
    print(
        f"Sampling night lights to target grid {height:,} x {width:,}; "
        f"source TIFF {source_grid['width']:,} x {source_grid['height']:,}"
    )

    for row0 in range(0, height, row_chunk_size):
        row1 = min(height, row0 + row_chunk_size)
        x_grid, y_grid = _target_centers_for_rows(target_grid["transform"], row0, row1, width)
        lons, lats = to_wgs84.transform(x_grid, y_grid)
        radiance_chunk, presence_chunk = _sample_source_chunk(
            image,
            source_grid,
            lons,
            lats,
            light_threshold=light_threshold,
        )
        radiance[row0:row1] = radiance_chunk
        presence[row0:row1] = presence_chunk
        if row0 == 0 or row1 == height or row1 % max(row_chunk_size * 10, 1) == 0:
            print(f"  sampled rows {row0:,}-{row1:,} / {height:,}")

    radiance.flush()
    presence.flush()
    return radiance, presence, {
        "night_light_radiance_2024": {
            "path": radiance_path.name,
            "dtype": "float32",
            "units": "source raster value",
            "long_name": "2024 VIIRS annual night-light value sampled to model grid",
        },
        "night_light_presence_1km": {
            "path": presence_path.name,
            "dtype": "uint8",
            "units": "0/1",
            "threshold": float(light_threshold),
            "long_name": "night-light source presence in 1 km cell",
        },
    }


def build_night_light_features(
    source_tif: Path,
    target_grid_path: Path,
    output_dir: Path,
    kernels_km: list[float],
    *,
    light_threshold: float,
    row_chunk_size: int,
    memory_limit_gb: float | None,
    truncate: float,
    overwrite: bool,
    keep_temp: bool,
) -> None:
    _set_memory_limit(memory_limit_gb)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; use --overwrite to rebuild")
        shutil.rmtree(output_dir)

    tmp_dir = output_dir.with_name(f"{output_dir.name}.partial")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    target_grid = _load_target_grid(target_grid_path)
    radiance, presence, feature_specs = _sample_night_lights_to_target_grid(
        source_tif,
        target_grid,
        tmp_dir,
        light_threshold=light_threshold,
        row_chunk_size=row_chunk_size,
    )
    shape = target_grid["shape"]
    resolution_m = target_grid["resolution_m"]

    for kernel_km in kernels_km:
        sigma_pixels = kernel_km * 1000.0 / resolution_m
        feature_name = f"light_density_gaussian_{_kernel_label(kernel_km)}"
        density_path = tmp_dir / f"{feature_name}.float32.npy"
        density = np.lib.format.open_memmap(density_path, mode="w+", dtype=np.float32, shape=shape)
        print(
            f"Building {feature_name}: sigma={kernel_km:g} km "
            f"({sigma_pixels:.2f} pixels)"
        )
        gaussian_filter(
            presence,
            sigma=sigma_pixels,
            output=density,
            mode="constant",
            cval=0.0,
            truncate=truncate,
        )
        density.flush()
        feature_specs[feature_name] = {
            "path": density_path.name,
            "dtype": "float32",
            "units": "light-source-cell fraction",
            "kernel": "gaussian",
            "sigma_km": float(kernel_km),
            "truncate": float(truncate),
        }

    distance_path = tmp_dir / "distance_to_light_source_meters.float32.npy"
    if int(presence.sum()) == 0:
        print("No light-source cells found; writing NaN distance map")
        distance = np.lib.format.open_memmap(distance_path, mode="w+", dtype=np.float32, shape=shape)
        distance[:] = np.nan
        distance.flush()
    else:
        print("Building distance_to_light_source_meters")
        temp_distance_path = tmp_dir / "distance_to_light_source_meters.float64.tmp.npy"
        distance64 = np.lib.format.open_memmap(temp_distance_path, mode="w+", dtype=np.float64, shape=shape)
        distance_transform_edt(
            presence == 0,
            sampling=resolution_m,
            return_distances=True,
            return_indices=False,
            distances=distance64,
        )
        distance64.flush()
        distance = np.lib.format.open_memmap(distance_path, mode="w+", dtype=np.float32, shape=shape)
        for row0 in range(0, shape[0], row_chunk_size):
            row1 = min(shape[0], row0 + row_chunk_size)
            distance[row0:row1] = distance64[row0:row1].astype(np.float32, copy=False)
        distance.flush()
        del distance64
        if not keep_temp:
            temp_distance_path.unlink(missing_ok=True)

    feature_specs["distance_to_light_source_meters"] = {
        "path": distance_path.name,
        "dtype": "float32",
        "units": "meters",
        "long_name": "Euclidean distance to nearest night-light source cell",
    }

    manifest = {
        "format": "night_light_feature_map_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tif": str(source_tif),
        "source_url": ZENODO_VIIRS_2024_URL,
        "source_doi": "10.5281/zenodo.17294744",
        "source_title": (
            "Annual time series of global VIIRS nighttime lights for 2000-2024 "
            "at 500-m spatial resolution extrapolated using logistic regression"
        ),
        "target_grid": str(target_grid_path),
        "shape": list(shape),
        "transform": list(target_grid["transform"]),
        "crs_wkt": target_grid["crs_wkt"],
        "resolution_m": resolution_m,
        "bbox_lonlat": target_grid.get("bbox_lonlat"),
        "raster_bounds": target_grid.get("raster_bounds"),
        "light_threshold": float(light_threshold),
        "features": feature_specs,
    }
    with (tmp_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    tmp_dir.rename(output_dir)
    print(
        f"Saved night-light feature maps to {output_dir} in "
        f"{(time.time() - start) / 60.0:.1f} minutes"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-tif",
        type=Path,
        default=Path(
            "data/night_lights/raw/"
            "nightlights.average_viirs.v21_m_500m_s_20240101_20241231_go_epsg4326_v20250904.tif"
        ),
    )
    parser.add_argument(
        "--target-grid",
        type=Path,
        default=Path("data/land_features/osm_drivable_roads_features_1km"),
        help="Road feature store or raw road raster NPZ defining the model grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/land_features/night_lights_features_1km"),
    )
    parser.add_argument("--kernels-km", nargs="+", type=float, default=[5.0, 10.0, 25.0])
    parser.add_argument("--light-threshold", type=float, default=0.0)
    parser.add_argument("--row-chunk-size", type=int, default=256)
    parser.add_argument("--truncate", type=float, default=4.0)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_night_light_features(
        source_tif=args.source_tif,
        target_grid_path=args.target_grid,
        output_dir=args.output_dir,
        kernels_km=list(args.kernels_km),
        light_threshold=args.light_threshold,
        row_chunk_size=args.row_chunk_size,
        memory_limit_gb=args.memory_limit_gb,
        truncate=args.truncate,
        overwrite=args.overwrite,
        keep_temp=args.keep_temp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
