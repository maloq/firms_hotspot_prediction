"""Night-light feature sampling wrappers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

from src.feature_generation.prepare_road_features import get_road_features_for_coords


DEFAULT_RECENT_FEATURE_NAME = "night_light_radiance_recent"
DEFAULT_RECENT_COVERAGE_FEATURE_NAME = "night_light_cf_cvg_recent"
DEFAULT_RECENT_FILTERED_FEATURE_NAME = "night_light_radiance_recent_cf_filtered"
DEFAULT_BLACK_MARBLE_RADIANCE_FEATURE_NAME = "night_light_black_marble_recent"
DEFAULT_BLACK_MARBLE_QUALITY_FEATURE_NAME = "night_light_black_marble_quality_recent"
DEFAULT_BLACK_MARBLE_OBSERVATIONS_FEATURE_NAME = "night_light_black_marble_observations_recent"
DEFAULT_BLACK_MARBLE_FILTERED_FEATURE_NAME = "night_light_black_marble_quality_filtered"
RECENT_CACHE_COORD_SCALE = 1_000_000
RECENT_CACHE_KEY_COLUMNS = ["lat_key", "lon_key", "source_year"]
COORDINATE_COLUMNS = {"lat", "lon", "latitude", "longitude", "lat_rounded", "lon_rounded"}
ANNUAL_VIIRS_PATTERN = re.compile(
    r"nightlights\.average_viirs\..*?_s_(?P<start_year>\d{4})0101_"
    r"(?P<end_year>\d{4})1231_.*?\.tif$"
)
ANNUAL_DATE_RANGE_PATTERN = re.compile(
    r".*?(?P<start_year>\d{4})0101[_-](?P<end_year>\d{4})1231.*?\.tif$"
)
ANNUAL_VNL_YEAR_PATTERN = re.compile(r".*?VNL.*?_(?P<year>\d{4})_.*?\.tif$")
BLACK_MARBLE_FILE_PATTERN = re.compile(
    r"VNP46A4\.A(?P<year>\d{4})001\.h(?P<h>\d{2})v(?P<v>\d{2})\."
    r".*?\.h5$"
)
BLACK_MARBLE_TILE_SIZE = 2400
BLACK_MARBLE_TILE_DEGREES = 10.0
BLACK_MARBLE_DATASET_GROUP = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"


def _source_grid_from_tiff(image: Image.Image) -> dict:
    scale = image.tag_v2[33550]
    tiepoint = image.tag_v2[33922]
    pixel_width = float(scale[0])
    pixel_height = float(scale[1])
    origin_lon = float(tiepoint[3]) - float(tiepoint[0]) * pixel_width
    origin_lat = float(tiepoint[4]) + float(tiepoint[1]) * pixel_height
    nodata_raw = image.tag_v2.get(42113)
    nodata = float(str(nodata_raw).split()[0]) if nodata_raw is not None else None
    width, height = image.size
    block_width = image.tag_v2.get(322)
    block_height = image.tag_v2.get(323)
    return {
        "width": int(width),
        "height": int(height),
        "origin_lon": origin_lon,
        "origin_lat": origin_lat,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "nodata": nodata,
        "block_width": int(block_width) if block_width else 512,
        "block_height": int(block_height) if block_height else 512,
    }


def _year_from_annual_tif_name(filename: str) -> int | None:
    for pattern in (ANNUAL_VIIRS_PATTERN, ANNUAL_DATE_RANGE_PATTERN):
        match = pattern.match(filename)
        if match:
            start_year = int(match.group("start_year"))
            end_year = int(match.group("end_year"))
            return start_year if start_year == end_year else None

    match = ANNUAL_VNL_YEAR_PATTERN.match(filename)
    if match:
        return int(match.group("year"))
    return None


def _available_annual_sources(
    source_dir: str | Path,
    source_glob: str = "*.tif",
) -> dict[int, Path]:
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Night-light annual source directory not found: {source_path}")

    sources: dict[int, Path] = {}
    for tif_path in sorted(source_path.glob(source_glob)):
        year = _year_from_annual_tif_name(tif_path.name)
        if year is None:
            continue
        sources[year] = tif_path

    if not sources:
        raise FileNotFoundError(
            f"No annual VIIRS night-light GeoTIFFs matching {source_glob!r} found in {source_path}"
        )
    return sources


def _nearest_available_years(
    requested_years: Sequence[object],
    available_years: Sequence[int],
) -> np.ndarray:
    available = np.asarray(sorted({int(year) for year in available_years}), dtype=np.int32)
    if available.size == 0:
        raise ValueError("available_years must contain at least one year")

    requested = pd.to_numeric(pd.Series(requested_years), errors="coerce").to_numpy(
        dtype=np.float64
    )
    nearest = np.full(requested.shape, available[-1], dtype=np.int32)
    valid = np.isfinite(requested)
    if not np.any(valid):
        return nearest

    requested_int = np.rint(requested[valid]).astype(np.int32)
    insert_at = np.searchsorted(available, requested_int)
    lower_idx = np.clip(insert_at - 1, 0, available.size - 1)
    upper_idx = np.clip(insert_at, 0, available.size - 1)
    lower_years = available[lower_idx]
    upper_years = available[upper_idx]
    lower_distance = np.abs(requested_int - lower_years)
    upper_distance = np.abs(requested_int - upper_years)

    # Prefer the earlier year on ties so a mid-point does not look into the future.
    choose_upper = upper_distance < lower_distance
    nearest[valid] = np.where(choose_upper, upper_years, lower_years)
    return nearest


def _sample_tiff_for_coords(
    image: Image.Image,
    source_grid: dict,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    src_cols = np.floor((lons - source_grid["origin_lon"]) / source_grid["pixel_width"]).astype(np.int64)
    src_rows = np.floor((source_grid["origin_lat"] - lats) / source_grid["pixel_height"]).astype(np.int64)
    valid = (
        np.isfinite(lats)
        & np.isfinite(lons)
        & (src_rows >= 0)
        & (src_rows < source_grid["height"])
        & (src_cols >= 0)
        & (src_cols < source_grid["width"])
    )

    values = np.full(lats.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return values

    valid_rows = src_rows[valid]
    valid_cols = src_cols[valid]
    valid_positions = np.flatnonzero(valid)
    row_min = int(valid_rows.min())
    row_max = int(valid_rows.max()) + 1
    col_min = int(valid_cols.min())
    col_max = int(valid_cols.max()) + 1
    block_width = int(source_grid.get("block_width", 512))
    block_height = int(source_grid.get("block_height", 512))
    block_cols = valid_cols // block_width
    block_rows = valid_rows // block_height
    block_ids = block_rows * ((source_grid["width"] + block_width - 1) // block_width) + block_cols
    unique_block_count = int(np.unique(block_ids).size)
    bbox_pixels = (row_max - row_min) * (col_max - col_min)
    block_pixels = unique_block_count * block_width * block_height
    if bbox_pixels <= block_pixels * 1.25:
        crop = image.crop((col_min, row_min, col_max, row_max))
        crop_arr = np.asarray(crop, dtype=np.float32)
        sampled = crop_arr[valid_rows - row_min, valid_cols - col_min]
        if source_grid["nodata"] is not None:
            sampled = np.where(sampled == source_grid["nodata"], 0.0, sampled)
        values[valid] = np.where(sampled > 0.0, sampled, 0.0).astype(
            np.float32,
            copy=False,
        )
        return values

    order = np.argsort(block_ids, kind="stable")

    ordered_ids = block_ids[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(ordered_ids)) + 1]
    group_ends = np.r_[group_starts[1:], ordered_ids.size]

    for start, end in zip(group_starts, group_ends):
        group_order = order[start:end]
        row_block = int(block_rows[group_order[0]])
        col_block = int(block_cols[group_order[0]])
        row0 = row_block * block_height
        col0 = col_block * block_width
        row1 = min(row0 + block_height, source_grid["height"])
        col1 = min(col0 + block_width, source_grid["width"])
        crop = image.crop((col0, row0, col1, row1))
        crop_arr = np.asarray(crop, dtype=np.float32)
        sampled = crop_arr[valid_rows[group_order] - row0, valid_cols[group_order] - col0]
        if source_grid["nodata"] is not None:
            sampled = np.where(sampled == source_grid["nodata"], 0.0, sampled)
        values[valid_positions[group_order]] = np.where(sampled > 0.0, sampled, 0.0).astype(
            np.float32,
            copy=False,
        )
    return values


def _cache_keys_for_coords(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_keys = np.rint(lats * RECENT_CACHE_COORD_SCALE).astype(np.int32)
    lon_keys = np.rint(lons * RECENT_CACHE_COORD_SCALE).astype(np.int32)
    return lat_keys, lon_keys


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


def _recent_request_frame(
    lats: np.ndarray,
    lons: np.ndarray,
    chosen_years: np.ndarray,
) -> pd.DataFrame:
    lat_keys, lon_keys = _cache_keys_for_coords(lats, lons)
    return pd.DataFrame(
        {
            "_position": np.arange(lats.size, dtype=np.int64),
            "lat": lats.astype(np.float64, copy=False),
            "lon": lons.astype(np.float64, copy=False),
            "lat_key": lat_keys,
            "lon_key": lon_keys,
            "source_year": chosen_years.astype(np.int16, copy=False),
        }
    )


def _load_recent_cache(cache_path: Path, feature_name: str) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame(columns=[*RECENT_CACHE_KEY_COLUMNS, feature_name])
    columns = [*RECENT_CACHE_KEY_COLUMNS, feature_name]
    try:
        cache = pd.read_parquet(cache_path, columns=columns)
    except (KeyError, ValueError):
        return pd.DataFrame(columns=columns)
    return cache.drop_duplicates(RECENT_CACHE_KEY_COLUMNS, keep="last")


def _write_feature_cache(
    cache_path: Path,
    cache: pd.DataFrame,
    new_rows: pd.DataFrame,
    feature_cols: Sequence[str],
) -> None:
    if new_rows.empty:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_cols = [*RECENT_CACHE_KEY_COLUMNS, "lat", "lon", *feature_cols]
    if cache_path.exists():
        try:
            cache = pd.read_parquet(cache_path).reindex(columns=cache_cols)
        except Exception:
            cache = cache.reindex(columns=cache_cols)
    combined = pd.concat(
        [
            cache.reindex(columns=cache_cols),
            new_rows.reindex(columns=cache_cols),
        ],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(RECENT_CACHE_KEY_COLUMNS, keep="last")
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(cache_path)


def _write_recent_cache(
    cache_path: Path,
    cache: pd.DataFrame,
    new_rows: pd.DataFrame,
    feature_name: str,
) -> None:
    _write_feature_cache(cache_path, cache, new_rows, [feature_name])


def _sample_recent_key_frame(
    key_frame: pd.DataFrame,
    sources: dict[int, Path],
    feature_name: str,
) -> pd.DataFrame:
    values = np.full(len(key_frame), np.nan, dtype=np.float32)
    Image.MAX_IMAGE_PIXELS = None
    for source_year in sorted(key_frame["source_year"].unique()):
        positions = np.flatnonzero(key_frame["source_year"].to_numpy() == source_year)
        tif_path = sources[int(source_year)]
        print(
            f"Sampling {feature_name} from VIIRS annual night lights {int(source_year)} "
            f"for {len(positions):,} unique coordinate(s)."
        )
        with Image.open(tif_path) as image:
            source_grid = _source_grid_from_tiff(image)
            values[positions] = _sample_tiff_for_coords(
                image,
                source_grid,
                key_frame["lat"].to_numpy(dtype=np.float64)[positions],
                key_frame["lon"].to_numpy(dtype=np.float64)[positions],
            )

    sampled = key_frame[[*RECENT_CACHE_KEY_COLUMNS, "lat", "lon"]].copy()
    sampled[feature_name] = values
    return sampled


def _sample_recent_annual_feature(
    coords: np.ndarray,
    years: Sequence[object],
    annual_source_dir: str | Path,
    *,
    feature_name: str,
    source_glob: str = "*.tif",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    if coords.shape[0] != 2:
        raise ValueError("coords must be a 2xN array with latitudes in row 0 and longitudes in row 1")

    n_points = coords.shape[1]
    year_values = list(years)
    if len(year_values) != n_points:
        raise ValueError(
            f"years must contain one value per coordinate ({n_points}), got {len(year_values)}"
        )

    sources = _available_annual_sources(annual_source_dir, source_glob=source_glob)
    chosen_years = _nearest_available_years(year_values, sources.keys())
    lats = np.asarray(coords[0], dtype=np.float64)
    lons = np.asarray(coords[1], dtype=np.float64)
    request = _recent_request_frame(lats, lons, chosen_years)
    unique_keys = request.drop_duplicates(RECENT_CACHE_KEY_COLUMNS, keep="first").reset_index(drop=True)

    cache_file = Path(cache_path) if cache_path else None
    cache = (
        _load_recent_cache(cache_file, feature_name)
        if cache_file is not None
        else pd.DataFrame(columns=[*RECENT_CACHE_KEY_COLUMNS, feature_name])
    )
    resolved = unique_keys.merge(
        cache[[*RECENT_CACHE_KEY_COLUMNS, feature_name]],
        on=RECENT_CACHE_KEY_COLUMNS,
        how="left",
        sort=False,
    )
    missing = resolved[resolved[feature_name].isna()][
        [*RECENT_CACHE_KEY_COLUMNS, "lat", "lon"]
    ].reset_index(drop=True)
    if not missing.empty:
        sampled_missing = _sample_recent_key_frame(missing, sources, feature_name)
        cache_hits = resolved[~resolved[feature_name].isna()][
            [*RECENT_CACHE_KEY_COLUMNS, "lat", "lon", feature_name]
        ]
        resolved = (
            sampled_missing
            if cache_hits.empty
            else pd.concat([cache_hits, sampled_missing], ignore_index=True)
        )
        if cache_file is not None:
            _write_recent_cache(cache_file, cache, sampled_missing, feature_name)
    else:
        print(
            f"Loaded all {len(unique_keys):,} unique {feature_name} values from "
            f"{cache_file}."
        )

    values = request[["_position", *RECENT_CACHE_KEY_COLUMNS]].merge(
        resolved[[*RECENT_CACHE_KEY_COLUMNS, feature_name]],
        on=RECENT_CACHE_KEY_COLUMNS,
        how="left",
        sort=False,
    )
    values = values.sort_values("_position", kind="stable")
    return pd.DataFrame({feature_name: values[feature_name].astype("float32").to_numpy()})


def get_recent_night_light_radiance_for_coords(
    coords: np.ndarray,
    years: Sequence[object],
    annual_source_dir: str | Path,
    recent_feature_name: str = DEFAULT_RECENT_FEATURE_NAME,
    source_glob: str = "*.tif",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Sample the nearest-available annual VIIRS radiance for dated coordinates."""
    return _sample_recent_annual_feature(
        coords=coords,
        years=years,
        annual_source_dir=annual_source_dir,
        feature_name=recent_feature_name,
        source_glob=source_glob,
        cache_path=cache_path,
    )


def _apply_cf_cvg_filter(
    features: pd.DataFrame,
    coords: np.ndarray,
    *,
    radiance_feature_name: str,
    coverage_feature_name: str,
    filtered_feature_name: str,
    min_cf_cvg: float,
    north_lat_min: float,
) -> None:
    lats = np.asarray(coords[0], dtype=np.float64)
    radiance = features[radiance_feature_name].to_numpy(dtype=np.float32)
    cf_cvg = features[coverage_feature_name].to_numpy(dtype=np.float32)
    bad_north_coverage = (lats >= north_lat_min) & (
        ~np.isfinite(cf_cvg) | (cf_cvg < float(min_cf_cvg))
    )
    features[filtered_feature_name] = np.where(
        bad_north_coverage,
        0.0,
        radiance,
    ).astype(np.float32, copy=False)


def _available_black_marble_sources(source_dir: str | Path) -> dict[tuple[int, int, int], Path]:
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Black Marble source directory not found: {source_path}")

    sources: dict[tuple[int, int, int], Path] = {}
    for h5_path in sorted(source_path.rglob("VNP46A4.A*.h*.h5")):
        match = BLACK_MARBLE_FILE_PATTERN.match(h5_path.name)
        if not match:
            continue
        year = int(match.group("year"))
        h = int(match.group("h"))
        v = int(match.group("v"))
        sources[(year, h, v)] = h5_path
    if not sources:
        raise FileNotFoundError(f"No VNP46A4 HDF5 tiles found in {source_path}")
    return sources


def _black_marble_request_frame(
    lats: np.ndarray,
    lons: np.ndarray,
    chosen_years: np.ndarray,
) -> pd.DataFrame:
    lat_keys, lon_keys = _cache_keys_for_coords(lats, lons)
    h = np.floor((lons + 180.0) / BLACK_MARBLE_TILE_DEGREES).astype(np.int16)
    v = np.floor((90.0 - lats) / BLACK_MARBLE_TILE_DEGREES).astype(np.int16)
    west = -180.0 + h.astype(np.float64) * BLACK_MARBLE_TILE_DEGREES
    north = 90.0 - v.astype(np.float64) * BLACK_MARBLE_TILE_DEGREES
    pixel_size = BLACK_MARBLE_TILE_DEGREES / BLACK_MARBLE_TILE_SIZE
    rows = np.floor((north - lats) / pixel_size).astype(np.int16)
    cols = np.floor((lons - west) / pixel_size).astype(np.int16)
    valid = (
        np.isfinite(lats)
        & np.isfinite(lons)
        & (h >= 0)
        & (h <= 35)
        & (v >= 0)
        & (v <= 17)
        & (rows >= 0)
        & (rows < BLACK_MARBLE_TILE_SIZE)
        & (cols >= 0)
        & (cols < BLACK_MARBLE_TILE_SIZE)
    )
    return pd.DataFrame(
        {
            "_position": np.arange(lats.size, dtype=np.int64),
            "lat": lats.astype(np.float64, copy=False),
            "lon": lons.astype(np.float64, copy=False),
            "lat_key": lat_keys,
            "lon_key": lon_keys,
            "source_year": chosen_years.astype(np.int16, copy=False),
            "h": h,
            "v": v,
            "row": rows,
            "col": cols,
            "valid": valid,
        }
    )


def _h5_dataset(file_handle: object, dataset_name: str):
    group = file_handle
    for part in BLACK_MARBLE_DATASET_GROUP.split("/"):
        group = group[part]
    return group[dataset_name]


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


def _sample_black_marble_key_frame(
    key_frame: pd.DataFrame,
    sources: dict[tuple[int, int, int], Path],
    *,
    radiance_sds: str,
    quality_sds: str,
    observations_sds: str | None,
    radiance_feature_name: str,
    quality_feature_name: str,
    observations_feature_name: str,
) -> pd.DataFrame:
    import h5py

    key_frame = key_frame.reset_index(drop=True)
    radiance_values = np.full(len(key_frame), np.nan, dtype=np.float32)
    quality_values = np.full(len(key_frame), 255.0, dtype=np.float32)
    observation_values = np.full(len(key_frame), np.nan, dtype=np.float32)
    valid_mask = key_frame["valid"].to_numpy(dtype=bool)
    if not np.any(valid_mask):
        sampled = key_frame[[*RECENT_CACHE_KEY_COLUMNS, "lat", "lon"]].copy()
        sampled[radiance_feature_name] = radiance_values
        sampled[quality_feature_name] = quality_values
        sampled[observations_feature_name] = observation_values
        return sampled

    valid_frame = key_frame.loc[valid_mask].copy()
    group_cols = ["source_year", "h", "v"]
    for (year, h, v), group in valid_frame.groupby(group_cols, sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        source_path = sources.get((int(year), int(h), int(v)))
        if source_path is None:
            print(f"Missing Black Marble tile for {int(year)} h{int(h):02d}v{int(v):02d}; values set to fill.")
            continue
        print(
            f"Sampling Black Marble {int(year)} h{int(h):02d}v{int(v):02d} "
            f"for {len(group):,} unique coordinate(s)."
        )
        rows = group["row"].to_numpy(dtype=np.int64)
        cols = group["col"].to_numpy(dtype=np.int64)
        with h5py.File(source_path, "r") as handle:
            radiance_values[positions] = _sample_h5_dataset(
                _h5_dataset(handle, radiance_sds),
                rows,
                cols,
                as_quality=False,
            )
            quality_values[positions] = _sample_h5_dataset(
                _h5_dataset(handle, quality_sds),
                rows,
                cols,
                as_quality=True,
            )
            if observations_sds:
                observation_values[positions] = _sample_h5_dataset(
                    _h5_dataset(handle, observations_sds),
                    rows,
                    cols,
                    as_quality=False,
                )

    sampled = key_frame[[*RECENT_CACHE_KEY_COLUMNS, "lat", "lon"]].copy()
    sampled[radiance_feature_name] = radiance_values
    sampled[quality_feature_name] = quality_values
    sampled[observations_feature_name] = observation_values
    return sampled


def _sample_black_marble_features(
    coords: np.ndarray,
    years: Sequence[object],
    source_dir: str | Path,
    *,
    radiance_sds: str,
    quality_sds: str,
    observations_sds: str | None,
    radiance_feature_name: str,
    quality_feature_name: str,
    observations_feature_name: str,
    filtered_feature_name: str | None,
    quality_keep_values: Sequence[int],
    cache_path: str | Path | None,
    missing_tile_strategy: str | None = None,
    fallback_radiance_values: np.ndarray | None = None,
    fallback_quality_value: float = 255.0,
    fallback_observations_value: float | None = 0.0,
) -> pd.DataFrame:
    if coords.shape[0] != 2:
        raise ValueError("coords must be a 2xN array with latitudes in row 0 and longitudes in row 1")

    feature_cols = [radiance_feature_name, quality_feature_name, observations_feature_name]
    cache_file = Path(cache_path) if cache_path else None
    if cache_file is not None and cache_file.exists():
        try:
            cache = pd.read_parquet(cache_file).drop_duplicates(RECENT_CACHE_KEY_COLUMNS, keep="last")
        except Exception:
            cache = pd.DataFrame(columns=[*RECENT_CACHE_KEY_COLUMNS, *feature_cols])
    else:
        cache = pd.DataFrame(columns=[*RECENT_CACHE_KEY_COLUMNS, *feature_cols])
    for col in feature_cols:
        if col not in cache.columns:
            cache[col] = np.nan

    cache_years: set[int] = set()
    if "source_year" in cache.columns and radiance_feature_name in cache.columns:
        cached_source_years = pd.to_numeric(
            cache.loc[cache[radiance_feature_name].notna(), "source_year"],
            errors="coerce",
        ).dropna()
        cache_years = {int(year) for year in cached_source_years}

    try:
        sources = _available_black_marble_sources(source_dir)
    except FileNotFoundError:
        if not cache_years:
            raise
        sources = {}
    source_years = {year for year, _, _ in sources}
    available_years = sorted(source_years | cache_years)
    chosen_years = _nearest_available_years(years, available_years)
    lats = np.asarray(coords[0], dtype=np.float64)
    lons = np.asarray(coords[1], dtype=np.float64)
    request = _black_marble_request_frame(lats, lons, chosen_years)
    unique_keys = request.drop_duplicates(RECENT_CACHE_KEY_COLUMNS, keep="first").reset_index(drop=True)

    resolved = unique_keys.merge(
        cache[[col for col in [*RECENT_CACHE_KEY_COLUMNS, *feature_cols] if col in cache.columns]],
        on=RECENT_CACHE_KEY_COLUMNS,
        how="left",
        sort=False,
    )
    missing = resolved[resolved[radiance_feature_name].isna()].drop(columns=feature_cols, errors="ignore")
    if not missing.empty:
        missing_raw_tiles = sorted(
            {
                (int(row.source_year), int(row.h), int(row.v))
                for row in missing[["source_year", "h", "v"]].itertuples(index=False)
                if (int(row.source_year), int(row.h), int(row.v)) not in sources
            }
        )
        if missing_raw_tiles:
            fallback_enabled = str(missing_tile_strategy or "").lower() in {
                "feature_map",
                "static_feature_map",
                "raster",
            }
            if not fallback_enabled or fallback_radiance_values is None:
                examples = ", ".join(
                    f"{year} h{h:02d}v{v:02d}" for year, h, v in missing_raw_tiles[:10]
                )
                more = "" if len(missing_raw_tiles) <= 10 else f", ... +{len(missing_raw_tiles) - 10} more"
                raise FileNotFoundError(
                    "Black Marble cache is missing requested point samples, and the raw "
                    f"HDF5 tiles are not available: {examples}{more}. "
                    "Regenerate the cache before deleting those raw tiles, or redownload them."
                )
            positions = missing["_position"].to_numpy(dtype=np.int64)
            fallback = np.asarray(fallback_radiance_values, dtype=np.float32)
            if fallback.shape[0] != coords.shape[1]:
                raise ValueError(
                    "Black Marble feature-map fallback length does not match coordinate count: "
                    f"{fallback.shape[0]} vs {coords.shape[1]}."
                )
            sampled_missing = missing[[*RECENT_CACHE_KEY_COLUMNS, "lat", "lon"]].copy()
            sampled_missing[radiance_feature_name] = np.nan_to_num(
                fallback[positions],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
            sampled_missing[quality_feature_name] = np.full(
                len(sampled_missing),
                float(fallback_quality_value),
                dtype=np.float32,
            )
            sampled_missing[observations_feature_name] = np.full(
                len(sampled_missing),
                np.nan if fallback_observations_value is None else float(fallback_observations_value),
                dtype=np.float32,
            )
            examples = ", ".join(
                f"{year} h{h:02d}v{v:02d}" for year, h, v in missing_raw_tiles[:10]
            )
            more = "" if len(missing_raw_tiles) <= 10 else f", ... +{len(missing_raw_tiles) - 10} more"
            print(
                "Warning: Black Marble cache/raw tiles missing for "
                f"{len(sampled_missing):,} unique coordinate-year samples "
                f"({examples}{more}); using feature-map fallback and leaving "
                "the source cache unchanged."
            )
        else:
            sampled_missing = _sample_black_marble_key_frame(
                missing,
                sources,
                radiance_sds=radiance_sds,
                quality_sds=quality_sds,
                observations_sds=observations_sds,
                radiance_feature_name=radiance_feature_name,
                quality_feature_name=quality_feature_name,
                observations_feature_name=observations_feature_name,
            )
            if cache_file is not None:
                _write_feature_cache(cache_file, cache, sampled_missing, feature_cols)
        cache_hits = resolved[~resolved[radiance_feature_name].isna()][
            [*RECENT_CACHE_KEY_COLUMNS, "lat", "lon", *feature_cols]
        ]
        resolved = (
            sampled_missing
            if cache_hits.empty
            else pd.concat([cache_hits, sampled_missing], ignore_index=True)
        )
    else:
        print(f"Loaded all {len(unique_keys):,} unique Black Marble values from {cache_file}.")

    values = request[["_position", *RECENT_CACHE_KEY_COLUMNS]].merge(
        resolved[[*RECENT_CACHE_KEY_COLUMNS, *feature_cols]],
        on=RECENT_CACHE_KEY_COLUMNS,
        how="left",
        sort=False,
    ).sort_values("_position", kind="stable")
    output = pd.DataFrame(
        {
            radiance_feature_name: values[radiance_feature_name].astype("float32").to_numpy(),
            quality_feature_name: values[quality_feature_name].astype("float32").to_numpy(),
            observations_feature_name: values[observations_feature_name].astype("float32").to_numpy(),
        }
    )
    if filtered_feature_name:
        keep = output[quality_feature_name].isin([int(v) for v in quality_keep_values]).to_numpy()
        radiance = output[radiance_feature_name].to_numpy(dtype=np.float32)
        output[filtered_feature_name] = np.where(keep, radiance, 0.0).astype(np.float32, copy=False)
    return output


def get_night_light_features_for_coords(
    coords: np.ndarray,
    feature_map_path: str = "data/land_features/night_lights_black_marble_features_1km",
    legacy_viirs_feature_map_path: str | Path | None = None,
    legacy_viirs_feature_prefix: str = "viirs_",
    years: Sequence[object] | None = None,
    annual_source_dir: str | Path | None = None,
    recent_feature_name: str = DEFAULT_RECENT_FEATURE_NAME,
    recent_source_glob: str = "*.tif",
    recent_cache_path: str | Path | None = None,
    cf_cvg_source_glob: str | None = None,
    cf_cvg_feature_name: str = DEFAULT_RECENT_COVERAGE_FEATURE_NAME,
    cf_cvg_cache_path: str | Path | None = None,
    cf_filtered_feature_name: str | None = DEFAULT_RECENT_FILTERED_FEATURE_NAME,
    min_cf_cvg: float | None = None,
    cf_filter_north_lat_min: float = 58.0,
    black_marble_source_dir: str | Path | None = None,
    black_marble_radiance_sds: str = "NearNadir_Composite_Snow_Free",
    black_marble_quality_sds: str = "NearNadir_Composite_Snow_Free_Quality",
    black_marble_observations_sds: str | None = "NearNadir_Composite_Snow_Free_Num",
    black_marble_radiance_feature_name: str = DEFAULT_BLACK_MARBLE_RADIANCE_FEATURE_NAME,
    black_marble_quality_feature_name: str = DEFAULT_BLACK_MARBLE_QUALITY_FEATURE_NAME,
    black_marble_observations_feature_name: str = DEFAULT_BLACK_MARBLE_OBSERVATIONS_FEATURE_NAME,
    black_marble_filtered_feature_name: str | None = DEFAULT_BLACK_MARBLE_FILTERED_FEATURE_NAME,
    black_marble_quality_keep_values: Sequence[int] = (0,),
    black_marble_cache_path: str | Path | None = None,
    black_marble_missing_tile_fallback: str | None = None,
    black_marble_fallback_feature_name: str = "night_light_radiance_2024",
) -> pd.DataFrame:
    """Sample precomputed night-light feature maps for WGS84 coordinates."""
    needs_years = annual_source_dir is not None or black_marble_source_dir is not None
    if years is None:
        if needs_years:
            raise ValueError(
                "Dated night-light sources require years. Pass one year per coordinate "
                "when annual_source_dir or black_marble_source_dir is configured."
            )
    features = get_road_features_for_coords(coords=coords, npz_path=feature_map_path)
    if legacy_viirs_feature_map_path is not None:
        legacy_features = get_road_features_for_coords(
            coords=coords,
            npz_path=legacy_viirs_feature_map_path,
        )
        legacy_feature_cols = [col for col in legacy_features.columns if col not in COORDINATE_COLUMNS]
        legacy_features = legacy_features[legacy_feature_cols].rename(
            columns={col: f"{legacy_viirs_feature_prefix}{col}" for col in legacy_feature_cols}
        )
        features = pd.concat(
            [features.reset_index(drop=True), legacy_features.reset_index(drop=True)],
            axis=1,
        )
    if years is None:
        return features

    if annual_source_dir is not None:
        recent = _sample_recent_annual_feature(
            coords=coords,
            years=years,
            annual_source_dir=annual_source_dir,
            feature_name=recent_feature_name,
            source_glob=recent_source_glob,
            cache_path=recent_cache_path,
        )
        if recent_feature_name in features.columns:
            features = features.drop(columns=[recent_feature_name])
        features = pd.concat([features.reset_index(drop=True), recent], axis=1)

        if cf_cvg_source_glob:
            cf_cvg = _sample_recent_annual_feature(
                coords=coords,
                years=years,
                annual_source_dir=annual_source_dir,
                feature_name=cf_cvg_feature_name,
                source_glob=cf_cvg_source_glob,
                cache_path=cf_cvg_cache_path,
            )
            if cf_cvg_feature_name in features.columns:
                features = features.drop(columns=[cf_cvg_feature_name])
            features = pd.concat([features.reset_index(drop=True), cf_cvg], axis=1)
            if min_cf_cvg is not None and cf_filtered_feature_name:
                _apply_cf_cvg_filter(
                    features,
                    coords,
                    radiance_feature_name=recent_feature_name,
                    coverage_feature_name=cf_cvg_feature_name,
                    filtered_feature_name=cf_filtered_feature_name,
                    min_cf_cvg=float(min_cf_cvg),
                    north_lat_min=float(cf_filter_north_lat_min),
                )

    if black_marble_source_dir is not None:
        fallback_radiance_values = None
        if str(black_marble_missing_tile_fallback or "").lower() in {
            "feature_map",
            "static_feature_map",
            "raster",
        }:
            if black_marble_fallback_feature_name not in features.columns:
                raise KeyError(
                    "Black Marble feature-map fallback requested, but "
                    f"{black_marble_fallback_feature_name!r} is absent from {feature_map_path}."
                )
            fallback_radiance_values = pd.to_numeric(
                features[black_marble_fallback_feature_name],
                errors="coerce",
            ).to_numpy(dtype=np.float32)
        black_marble = _sample_black_marble_features(
            coords=coords,
            years=years,
            source_dir=black_marble_source_dir,
            radiance_sds=black_marble_radiance_sds,
            quality_sds=black_marble_quality_sds,
            observations_sds=black_marble_observations_sds,
            radiance_feature_name=black_marble_radiance_feature_name,
            quality_feature_name=black_marble_quality_feature_name,
            observations_feature_name=black_marble_observations_feature_name,
            filtered_feature_name=black_marble_filtered_feature_name,
            quality_keep_values=black_marble_quality_keep_values,
            cache_path=black_marble_cache_path,
            missing_tile_strategy=black_marble_missing_tile_fallback,
            fallback_radiance_values=fallback_radiance_values,
        )
        features = pd.concat([features.reset_index(drop=True), black_marble], axis=1)

    return features
