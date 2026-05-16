"""Build whole-area night-light feature maps on the model 1 km grid."""

from __future__ import annotations

import argparse
import json
import os
import re
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


ZENODO_VIIRS_RECORD_URL = "https://zenodo.org/records/17294744"
ANNUAL_VIIRS_PATTERN = re.compile(
    r"nightlights\.average_viirs\..*?_s_(?P<start_year>\d{4})0101_"
    r"(?P<end_year>\d{4})1231_.*?\.tif$"
)
BLACK_MARBLE_FILE_PATTERN = re.compile(
    r"VNP46A4\.A(?P<year>\d{4})001\.h(?P<h>\d{2})v(?P<v>\d{2})\."
    r".*?\.h5$"
)
BLACK_MARBLE_TILE_SIZE = 2400
BLACK_MARBLE_TILE_DEGREES = 10.0
BLACK_MARBLE_DATASET_GROUP = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
BLACK_MARBLE_PRODUCT_URL = (
    "https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A4"
)
BLACK_MARBLE_ARCHIVE_URL = (
    "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP46A4"
)
BLACK_MARBLE_DOI = "10.5067/VIIRS/VNP46A4.002"


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


def _source_year_from_tif(source_tif: Path) -> int | None:
    match = ANNUAL_VIIRS_PATTERN.match(source_tif.name)
    if not match:
        return None
    start_year = int(match.group("start_year"))
    end_year = int(match.group("end_year"))
    return start_year if start_year == end_year else None


def _zenodo_url_for_tif(source_tif: Path) -> str:
    return f"{ZENODO_VIIRS_RECORD_URL}/files/{source_tif.name}?download=1"


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


def _available_black_marble_sources(source_dir: Path) -> dict[tuple[int, int, int], Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Black Marble source directory not found: {source_dir}")

    sources: dict[tuple[int, int, int], Path] = {}
    for h5_path in sorted(source_dir.rglob("VNP46A4.A*.h*.h5")):
        match = BLACK_MARBLE_FILE_PATTERN.match(h5_path.name)
        if not match:
            continue
        year = int(match.group("year"))
        h = int(match.group("h"))
        v = int(match.group("v"))
        sources[(year, h, v)] = h5_path
    if not sources:
        raise FileNotFoundError(f"No VNP46A4 HDF5 tiles found in {source_dir}")
    return sources


def _h5_dataset(file_handle: object, dataset_name: str):
    group = file_handle
    for part in BLACK_MARBLE_DATASET_GROUP.split("/"):
        group = group[part]
    return group[dataset_name]


def _attr_scalar(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="ignore")
    try:
        return float(item)
    except (TypeError, ValueError):
        return default


def _sample_h5_dataset(dataset: object, rows: np.ndarray, cols: np.ndarray, *, as_quality: bool) -> np.ndarray:
    arr = dataset[()]
    sampled = arr[rows, cols]
    fill_value = _attr_scalar(dataset.attrs.get("_FillValue"))
    scale = _attr_scalar(dataset.attrs.get("scale_factor"), 1.0) or 1.0
    offset = _attr_scalar(dataset.attrs.get("offset"), 0.0) or 0.0
    if as_quality:
        values = sampled.astype(np.float32, copy=False)
        if fill_value is not None:
            values = np.where(sampled == fill_value, 255.0, values)
        return values.astype(np.float32, copy=False)

    values = sampled.astype(np.float32, copy=False)
    if fill_value is not None:
        values = np.where(sampled == fill_value, np.nan, values)
    values = values * float(scale) + float(offset)
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    return values.astype(np.float32, copy=False)


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


def _black_marble_pixels_for_coords(lons: np.ndarray, lats: np.ndarray) -> dict[str, np.ndarray]:
    h = np.zeros(lons.shape, dtype=np.int16)
    v = np.zeros(lons.shape, dtype=np.int16)
    rows = np.zeros(lons.shape, dtype=np.int16)
    cols = np.zeros(lons.shape, dtype=np.int16)

    finite = np.isfinite(lons) & np.isfinite(lats)
    valid = np.zeros(lons.shape, dtype=bool)
    if not np.any(finite):
        return {"h": h, "v": v, "row": rows, "col": cols, "valid": valid}

    h_candidate = np.floor((lons[finite] + 180.0) / BLACK_MARBLE_TILE_DEGREES).astype(np.int16)
    v_candidate = np.floor((90.0 - lats[finite]) / BLACK_MARBLE_TILE_DEGREES).astype(np.int16)
    in_tile_range = (
        (h_candidate >= 0)
        & (h_candidate <= 35)
        & (v_candidate >= 0)
        & (v_candidate <= 17)
    )
    finite_positions = np.flatnonzero(finite)
    candidate_positions = finite_positions[in_tile_range]
    if candidate_positions.size == 0:
        return {"h": h, "v": v, "row": rows, "col": cols, "valid": valid}

    h_valid = h_candidate[in_tile_range]
    v_valid = v_candidate[in_tile_range]
    west = -180.0 + h_valid.astype(np.float64) * BLACK_MARBLE_TILE_DEGREES
    north = 90.0 - v_valid.astype(np.float64) * BLACK_MARBLE_TILE_DEGREES
    pixel_size = BLACK_MARBLE_TILE_DEGREES / BLACK_MARBLE_TILE_SIZE
    row_valid = np.floor((north - lats.reshape(-1)[candidate_positions]) / pixel_size).astype(np.int16)
    col_valid = np.floor((lons.reshape(-1)[candidate_positions] - west) / pixel_size).astype(np.int16)
    in_pixel_range = (
        (row_valid >= 0)
        & (row_valid < BLACK_MARBLE_TILE_SIZE)
        & (col_valid >= 0)
        & (col_valid < BLACK_MARBLE_TILE_SIZE)
    )
    final_positions = candidate_positions[in_pixel_range]
    h.reshape(-1)[final_positions] = h_valid[in_pixel_range]
    v.reshape(-1)[final_positions] = v_valid[in_pixel_range]
    rows.reshape(-1)[final_positions] = row_valid[in_pixel_range]
    cols.reshape(-1)[final_positions] = col_valid[in_pixel_range]
    valid.reshape(-1)[final_positions] = True
    return {"h": h, "v": v, "row": rows, "col": cols, "valid": valid}


def _sample_black_marble_chunk(
    sources: dict[tuple[int, int, int], Path],
    source_year: int,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    radiance_sds: str,
    quality_sds: str,
    quality_keep_values: list[int],
    light_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    pixels = _black_marble_pixels_for_coords(lons, lats)
    valid = pixels["valid"]
    radiance = np.zeros(lons.shape, dtype=np.float32)
    if not np.any(valid):
        return radiance, np.zeros(lons.shape, dtype=np.uint8)

    flat_valid_positions = np.flatnonzero(valid.ravel())
    flat_h = pixels["h"].ravel()[flat_valid_positions]
    flat_v = pixels["v"].ravel()[flat_valid_positions]
    flat_rows = pixels["row"].ravel()[flat_valid_positions].astype(np.int64)
    flat_cols = pixels["col"].ravel()[flat_valid_positions].astype(np.int64)
    tile_ids = flat_h.astype(np.int32) * 100 + flat_v.astype(np.int32)
    order = np.argsort(tile_ids, kind="stable")
    ordered_ids = tile_ids[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(ordered_ids)) + 1]
    group_ends = np.r_[group_starts[1:], ordered_ids.size]
    keep_values = np.asarray([int(value) for value in quality_keep_values], dtype=np.int16)

    flat_radiance = radiance.ravel()
    for start, end in zip(group_starts, group_ends):
        group_order = order[start:end]
        h = int(flat_h[group_order[0]])
        v = int(flat_v[group_order[0]])
        source_path = sources.get((source_year, h, v))
        if source_path is None:
            continue

        rows = flat_rows[group_order]
        cols = flat_cols[group_order]
        with h5py.File(source_path, "r") as handle:
            sampled_radiance = _sample_h5_dataset(
                _h5_dataset(handle, radiance_sds),
                rows,
                cols,
                as_quality=False,
            )
            sampled_quality = _sample_h5_dataset(
                _h5_dataset(handle, quality_sds),
                rows,
                cols,
                as_quality=True,
            )
        good_quality = np.isin(sampled_quality.astype(np.int16, copy=False), keep_values)
        sampled_radiance = np.where(good_quality, sampled_radiance, 0.0)
        flat_radiance[flat_valid_positions[group_order]] = sampled_radiance.astype(
            np.float32,
            copy=False,
        )

    presence = (radiance > light_threshold).astype(np.uint8)
    return radiance, presence


def _sample_night_lights_to_target_grid(
    source_tif: Path,
    target_grid: dict,
    output_dir: Path,
    *,
    source_year: int,
    light_threshold: float,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(source_tif)
    source_grid = _source_grid_from_tiff(image)

    shape = target_grid["shape"]
    radiance_name = f"night_light_radiance_{source_year}"
    radiance_path = output_dir / f"{radiance_name}.float32.npy"
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
        radiance_name: {
            "path": radiance_path.name,
            "dtype": "float32",
            "units": "source raster value",
            "source_year": int(source_year),
            "long_name": f"{source_year} VIIRS annual night-light value sampled to model grid",
        },
        "night_light_presence_1km": {
            "path": presence_path.name,
            "dtype": "uint8",
            "units": "0/1",
            "threshold": float(light_threshold),
            "long_name": "night-light source presence in 1 km cell",
        },
    }


def _sample_black_marble_to_target_grid(
    source_dir: Path,
    target_grid: dict,
    output_dir: Path,
    *,
    source_year: int,
    radiance_sds: str,
    quality_sds: str,
    quality_keep_values: list[int],
    light_threshold: float,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    sources = _available_black_marble_sources(source_dir)
    available_tiles = sorted((h, v) for year, h, v in sources if year == source_year)
    if not available_tiles:
        raise FileNotFoundError(
            f"No Black Marble VNP46A4 tiles for {source_year} found in {source_dir}"
        )

    shape = target_grid["shape"]
    radiance_name = f"night_light_radiance_{source_year}"
    radiance_path = output_dir / f"{radiance_name}.float32.npy"
    presence_path = output_dir / "night_light_presence_1km.uint8.npy"
    radiance = np.lib.format.open_memmap(radiance_path, mode="w+", dtype=np.float32, shape=shape)
    presence = np.lib.format.open_memmap(presence_path, mode="w+", dtype=np.uint8, shape=shape)

    target_crs = CRS.from_wkt(target_grid["crs_wkt"])
    to_wgs84 = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)
    height, width = shape
    print(
        f"Sampling Black Marble {source_year} to target grid {height:,} x {width:,}; "
        f"{len(available_tiles):,} tiles available for that year"
    )

    for row0 in range(0, height, row_chunk_size):
        row1 = min(height, row0 + row_chunk_size)
        x_grid, y_grid = _target_centers_for_rows(target_grid["transform"], row0, row1, width)
        lons, lats = to_wgs84.transform(x_grid, y_grid)
        radiance_chunk, presence_chunk = _sample_black_marble_chunk(
            sources,
            source_year,
            lons,
            lats,
            radiance_sds=radiance_sds,
            quality_sds=quality_sds,
            quality_keep_values=quality_keep_values,
            light_threshold=light_threshold,
        )
        radiance[row0:row1] = radiance_chunk
        presence[row0:row1] = presence_chunk
        if row0 == 0 or row1 == height or row1 % max(row_chunk_size * 10, 1) == 0:
            print(f"  sampled rows {row0:,}-{row1:,} / {height:,}")

    radiance.flush()
    presence.flush()
    quality_values_label = ",".join(str(value) for value in quality_keep_values)
    return radiance, presence, {
        radiance_name: {
            "path": radiance_path.name,
            "dtype": "float32",
            "units": "nW cm-2 sr-1",
            "source_year": int(source_year),
            "source_product": "VNP46A4",
            "radiance_sds": radiance_sds,
            "quality_sds": quality_sds,
            "quality_keep_values": [int(value) for value in quality_keep_values],
            "long_name": (
                f"{source_year} NASA Black Marble VNP46A4 annual radiance "
                f"sampled to model grid; quality values kept: {quality_values_label}"
            ),
        },
        "night_light_presence_1km": {
            "path": presence_path.name,
            "dtype": "uint8",
            "units": "0/1",
            "threshold": float(light_threshold),
            "source_product": "VNP46A4",
            "long_name": "quality-filtered Black Marble night-light source presence in 1 km cell",
        },
    }


def build_night_light_features(
    source_tif: Path | None,
    target_grid_path: Path,
    output_dir: Path,
    kernels_km: list[float],
    *,
    black_marble_source_dir: Path | None,
    black_marble_source_year: int,
    black_marble_radiance_sds: str,
    black_marble_quality_sds: str,
    black_marble_quality_keep_values: list[int],
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
    if source_tif is not None:
        source_year = _source_year_from_tif(source_tif) or black_marble_source_year
        radiance, presence, feature_specs = _sample_night_lights_to_target_grid(
            source_tif,
            target_grid,
            tmp_dir,
            source_year=source_year,
            light_threshold=light_threshold,
            row_chunk_size=row_chunk_size,
        )
        source_metadata = {
            "source_type": "zenodo_viirs_geotiff",
            "source_tif": str(source_tif),
            "source_url": _zenodo_url_for_tif(source_tif),
            "source_doi": "10.5281/zenodo.17294744",
            "source_title": (
                "Annual time series of global VIIRS nighttime lights for 2000-2024 "
                "at 500-m spatial resolution extrapolated using logistic regression"
            ),
        }
    else:
        if black_marble_source_dir is None:
            raise ValueError("Either source_tif or black_marble_source_dir must be provided.")
        source_year = int(black_marble_source_year)
        radiance, presence, feature_specs = _sample_black_marble_to_target_grid(
            black_marble_source_dir,
            target_grid,
            tmp_dir,
            source_year=source_year,
            radiance_sds=black_marble_radiance_sds,
            quality_sds=black_marble_quality_sds,
            quality_keep_values=black_marble_quality_keep_values,
            light_threshold=light_threshold,
            row_chunk_size=row_chunk_size,
        )
        source_metadata = {
            "source_type": "nasa_black_marble_vnp46a4",
            "source_dir": str(black_marble_source_dir),
            "source_year": int(source_year),
            "source_url": f"{BLACK_MARBLE_ARCHIVE_URL}/{source_year}/001/",
            "source_product_url": BLACK_MARBLE_PRODUCT_URL,
            "source_doi": BLACK_MARBLE_DOI,
            "source_title": (
                "VIIRS/NPP Lunar BRDF-Adjusted Nighttime Lights Yearly L3 Global "
                "15 arc second Linear Lat Lon Grid"
            ),
            "radiance_sds": black_marble_radiance_sds,
            "quality_sds": black_marble_quality_sds,
            "quality_keep_values": [int(value) for value in black_marble_quality_keep_values],
        }
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
    manifest.update(source_metadata)
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
        default=None,
        help="Legacy annual VIIRS GeoTIFF source. If omitted, NASA Black Marble HDF5 tiles are used.",
    )
    parser.add_argument(
        "--black-marble-source-dir",
        type=Path,
        default=Path("data/night_lights/black_marble_vnp46a4"),
        help="Directory containing downloaded VNP46A4 annual HDF5 tiles.",
    )
    parser.add_argument(
        "--black-marble-source-year",
        type=int,
        default=2024,
        help="VNP46A4 annual year to build the static night-light map from.",
    )
    parser.add_argument(
        "--black-marble-radiance-sds",
        default="NearNadir_Composite_Snow_Free",
    )
    parser.add_argument(
        "--black-marble-quality-sds",
        default="NearNadir_Composite_Snow_Free_Quality",
    )
    parser.add_argument(
        "--black-marble-quality-keep-values",
        nargs="+",
        type=int,
        default=[0],
        help="Quality flag values to keep when writing radiance and presence maps.",
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
        default=Path("data/land_features/night_lights_black_marble_features_1km"),
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
        black_marble_source_dir=args.black_marble_source_dir,
        black_marble_source_year=args.black_marble_source_year,
        black_marble_radiance_sds=args.black_marble_radiance_sds,
        black_marble_quality_sds=args.black_marble_quality_sds,
        black_marble_quality_keep_values=list(args.black_marble_quality_keep_values),
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
