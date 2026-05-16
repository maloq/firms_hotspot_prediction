from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.revision_evaluation.probability_overlays import (
    DEFAULT_NEURAL_CATEGORICAL_COLUMNS,
    DEFAULT_NEURAL_DYNAMIC_COLUMNS,
    DEFAULT_NEURAL_STATIC_COLUMNS,
)
from src.feature_generation.load_climate_data import load_climate_variable_mf, open_climate_fragment
from src.feature_generation.prepare_climate_data import (
    _chunk_size_for_dim,
    _climate_block_source_token,
    _compute_daily_slab,
    _fragment_priority_order,
    _iter_location_chunk_batches,
    _iter_temporal_row_blocks,
    _location_processing_order,
    _nearest_coordinate_indices,
    _rows_for_location_batch,
    _rows_grouped_by_location,
    _subset_for_fragment,
    _time_range_for_subset,
    assign_rows_to_climate_fragments,
    check_dataset_bounds,
    check_fragmented_dataset_bounds,
    climate_fragments_source_token,
    discover_climate_fragments,
    print_dataset_bounds_check,
    print_fragmented_bounds_check,
)


TARGET_METADATA_COLUMNS = [
    "datetime",
    "lat_rounded",
    "lon_rounded",
    "count",
    "negative_stratum",
    "sampling_probability",
    "sample_weight",
    "nearest_positive_distance_cells",
    "nearest_positive_delta_days",
]


def latest_target_cache_files(target_cache_dir: Path) -> list[Path]:
    latest: dict[str, Path] = {}
    for path in sorted(target_cache_dir.glob("target_data_*.parquet")):
        stem = path.stem.removeprefix("target_data_")
        country = stem.rsplit("_", 1)[0]
        current = latest.get(country)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest[country] = path
    return sorted(latest.values())


def read_target_metadata(target_cache_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in latest_target_cache_files(target_cache_dir):
        schema_cols = set(pq.read_schema(path).names)
        columns = [col for col in TARGET_METADATA_COLUMNS if col in schema_cols]
        if not {"datetime", "lat_rounded", "lon_rounded", "count"}.issubset(columns):
            continue
        frame = pd.read_parquet(path, columns=columns)
        if "negative_stratum" not in frame.columns:
            frame["negative_stratum"] = pd.NA
        if "sampling_probability" not in frame.columns:
            frame["sampling_probability"] = np.nan
        if "sample_weight" not in frame.columns:
            frame["sample_weight"] = np.nan
        if "nearest_positive_distance_cells" not in frame.columns:
            frame["nearest_positive_distance_cells"] = np.nan
        if "nearest_positive_delta_days" not in frame.columns:
            frame["nearest_positive_delta_days"] = np.nan
        frames.append(frame[TARGET_METADATA_COLUMNS])
    if not frames:
        raise FileNotFoundError(f"No usable target cache files found in {target_cache_dir}")
    metadata = pd.concat(frames, ignore_index=True)
    metadata["datetime"] = pd.to_datetime(metadata["datetime"], errors="coerce")
    metadata["lat_rounded"] = pd.to_numeric(metadata["lat_rounded"], errors="coerce").astype("float32")
    metadata["lon_rounded"] = pd.to_numeric(metadata["lon_rounded"], errors="coerce").astype("float32")
    metadata["count"] = (pd.to_numeric(metadata["count"], errors="coerce").fillna(0) > 0).astype("int8")
    key_cols = ["datetime", "lat_rounded", "lon_rounded", "count"]
    duplicate_count = int(metadata.duplicated(key_cols).sum())
    if duplicate_count:
        print(f"Warning: dropping {duplicate_count} duplicate target metadata keys.")
        metadata = metadata.drop_duplicates(key_cols, keep="first")
    return metadata


def read_target_order(target_cache_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in latest_target_cache_files(target_cache_dir):
        stem = path.stem.removeprefix("target_data_")
        country = stem.rsplit("_", 1)[0]
        columns = ["datetime", "lat_rounded", "lon_rounded"]
        frame = pd.read_parquet(path, columns=columns)
        frame["_source_country"] = country
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No usable target cache files found in {target_cache_dir}")
    order = pd.concat(frames, ignore_index=True)
    order["datetime"] = pd.to_datetime(order["datetime"], errors="coerce")
    order["lat_rounded"] = pd.to_numeric(order["lat_rounded"], errors="coerce").astype("float32")
    order["lon_rounded"] = pd.to_numeric(order["lon_rounded"], errors="coerce").astype("float32")
    order = order.sort_values(
        ["datetime", "_source_country", "lat_rounded", "lon_rounded"],
        kind="stable",
    ).reset_index(drop=True)
    order["_target_row_idx"] = np.arange(len(order), dtype=np.int64)
    return order


def split_codes(years: np.ndarray) -> np.ndarray:
    split = np.full(years.shape[0], -1, dtype=np.int8)
    split[(years >= 2001) & (years <= 2018)] = 0
    split[(years >= 2019) & (years <= 2020)] = 1
    split[(years >= 2021) & (years <= 2025)] = 2
    return split


def numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )


def fit_scale(matrix: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = matrix[train_mask]
    fill = np.nanmedian(train, axis=0).astype(np.float32)
    fill = np.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
    filled = np.where(np.isnan(matrix), fill, matrix).astype(np.float32)
    mean = filled[train_mask].mean(axis=0).astype(np.float32)
    std = filled[train_mask].std(axis=0).astype(np.float32)
    std = np.where(std <= 0, 1.0, std).astype(np.float32)
    return ((filled - mean) / std).astype(np.float32), fill, mean, std


def fit_scale_daily_channel(matrix: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    train = matrix[train_mask]
    fill = np.nanmedian(train, axis=0).astype(np.float32)
    fill = np.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
    filled = np.where(np.isnan(matrix), fill, matrix).astype(np.float32)
    mean = float(filled[train_mask].mean())
    std = float(filled[train_mask].std())
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    return ((filled - mean) / std).astype(np.float32), fill, np.float32(mean), np.float32(std)


def fit_scale_spatial_daily_channel(
    matrix: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    train = matrix[train_mask]
    fill = np.nanmedian(train, axis=0).astype(np.float32)
    fill = np.nan_to_num(fill, nan=0.0, posinf=0.0, neginf=0.0)
    filled = np.where(np.isnan(matrix), fill, matrix).astype(np.float32)
    mean = float(filled[train_mask].mean())
    std = float(filled[train_mask].std())
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    return ((filled - mean) / std).astype(np.float32), fill, np.float32(mean), np.float32(std)


def write_scaled_spatial_daily_channel(
    matrix: np.ndarray,
    train_mask: np.ndarray,
    out: np.ndarray,
    *,
    row_batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_indices = np.flatnonzero(train_mask)
    if train_indices.size == 0:
        raise ValueError("Cannot fit spatial daily scaler without training rows.")
    if matrix.shape != out.shape:
        raise ValueError(f"Spatial scaler output shape mismatch: matrix={matrix.shape}, out={out.shape}.")

    feature_shape = matrix.shape[1:]
    sums = np.zeros(feature_shape, dtype=np.float64)
    counts = np.zeros(feature_shape, dtype=np.int64)
    row_batch_size = max(1, int(row_batch_size))

    for start in range(0, train_indices.size, row_batch_size):
        idx = train_indices[start : start + row_batch_size]
        block = np.asarray(matrix[idx], dtype=np.float32)
        finite = np.isfinite(block)
        sums += np.where(finite, block, 0.0).sum(axis=0, dtype=np.float64)
        counts += finite.sum(axis=0)

    fill = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    ).astype(np.float32)

    total = 0.0
    total_sq = 0.0
    total_count = 0
    for start in range(0, train_indices.size, row_batch_size):
        idx = train_indices[start : start + row_batch_size]
        block = np.asarray(matrix[idx], dtype=np.float32)
        block = np.where(np.isfinite(block), block, fill)
        total += float(block.sum(dtype=np.float64))
        total_sq += float(np.sum(block * block, dtype=np.float64))
        total_count += int(block.size)

    mean = total / max(total_count, 1)
    variance = max(total_sq / max(total_count, 1) - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0

    for start in range(0, matrix.shape[0], row_batch_size):
        stop = min(matrix.shape[0], start + row_batch_size)
        block = np.asarray(matrix[start:stop], dtype=np.float32)
        block = np.where(np.isfinite(block), block, fill)
        out[start:stop] = ((block - mean) / std).astype(out.dtype, copy=False)

    return fill, np.float32(mean), np.float32(std)


def fill_spatial_gaps_from_point_cache(
    raw: np.ndarray,
    *,
    frame: pd.DataFrame,
    target_cache_dir: Path,
    climate_cache_dir: Path,
    variable: str,
    n_days: int,
    row_batch_size: int = 8192,
) -> dict[str, object]:
    """Fill missing spatial cells with the existing point-level daily climate cache."""

    info: dict[str, object] = {"enabled": False}
    try:
        target_order = read_target_order(target_cache_dir)
        row_idx = align_source_rows_to_target_order(frame, target_order)
        cache_path = latest_cache_for_variable(climate_cache_dir, variable, len(target_order), n_days)
        point_cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        info["error"] = str(exc)
        print(
            f"Warning: could not load point-cache fallback for {variable}; "
            "remaining missing spatial climate values will be scaler-filled. "
            f"Reason: {exc}",
            flush=True,
        )
        return info

    if point_cache.shape != (len(target_order), n_days):
        info["error"] = f"point cache shape {point_cache.shape} does not match {(len(target_order), n_days)}"
        print(f"Warning: {variable} point-cache fallback skipped: {info['error']}", flush=True)
        return info

    rows_with_missing = 0
    missing_values = 0
    filled_values = 0
    remaining_values = 0
    row_batch_size = max(1, int(row_batch_size))

    for start in range(0, raw.shape[0], row_batch_size):
        stop = min(raw.shape[0], start + row_batch_size)
        block = np.asarray(raw[start:stop], dtype=np.float32)
        missing = ~np.isfinite(block)
        if not missing.any():
            continue

        rows_with_missing += int(np.any(missing.reshape(missing.shape[0], -1), axis=1).sum())
        missing_values += int(missing.sum())
        point = np.asarray(point_cache[row_idx[start:stop]], dtype=np.float32)
        fallback = np.broadcast_to(point[:, :, None, None], block.shape)
        fillable = missing & np.isfinite(fallback)
        if fillable.any():
            block[fillable] = fallback[fillable]
            filled_values += int(fillable.sum())

        remaining = ~np.isfinite(block)
        remaining_values += int(remaining.sum())
        raw[start:stop] = block

    if hasattr(raw, "flush"):
        raw.flush()

    info.update(
        {
            "enabled": True,
            "cache_path": str(cache_path),
            "rows_with_missing": rows_with_missing,
            "missing_values": missing_values,
            "filled_values": filled_values,
            "remaining_values": remaining_values,
        }
    )
    if missing_values:
        print(
            f"Filled {filled_values:,}/{missing_values:,} missing {variable} spatial cells "
            f"from point cache ({rows_with_missing:,} rows touched).",
            flush=True,
        )
    return info


def latest_cache_for_variable(cache_dir: Path, variable: str, rows: int, n_days: int) -> Path:
    candidates: list[Path] = []
    for path in sorted(cache_dir.glob(f"{variable}_*.npy")):
        try:
            array = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if array.shape == (rows, n_days):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No cached raw climate matrix for {variable} with shape {(rows, n_days)} under {cache_dir}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def align_source_rows_to_target_order(frame: pd.DataFrame, target_order: pd.DataFrame) -> np.ndarray:
    key_cols = ["datetime", "lat_rounded", "lon_rounded"]
    lookup = target_order[key_cols + ["_target_row_idx"]].drop_duplicates(key_cols, keep="first")
    aligned = frame[key_cols].merge(lookup, on=key_cols, how="left", validate="many_to_one")
    missing = int(aligned["_target_row_idx"].isna().sum())
    if missing:
        raise ValueError(f"Could not align {missing}/{len(frame)} feature rows to target-cache climate rows.")
    return aligned["_target_row_idx"].to_numpy(dtype=np.int64)


def build_daily_dynamic_tensor(
    *,
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    target_cache_dir: Path,
    climate_cache_dir: Path,
    variables: list[str],
    n_days: int,
    output_dtype: str,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    target_order = read_target_order(target_cache_dir)
    row_idx = align_source_rows_to_target_order(frame, target_order)
    x_dyn = np.empty((len(frame), n_days, len(variables)), dtype=np.float32)
    stats: dict[str, dict[str, object]] = {}
    for var_idx, variable in enumerate(variables):
        cache_path = latest_cache_for_variable(climate_cache_dir, variable, len(target_order), n_days)
        print(f"Loading daily climate cache for {variable}: {cache_path}", flush=True)
        raw = np.load(cache_path, mmap_mode="r")[row_idx].astype(np.float32, copy=False)
        scaled, fill, mean, std = fit_scale_daily_channel(raw, train_mask)
        x_dyn[:, :, var_idx] = scaled
        stats[variable] = {
            "cache_path": str(cache_path),
            "fill": fill.astype(float).tolist(),
            "mean": float(mean),
            "std": float(std),
        }
    if output_dtype == "float16":
        return x_dyn.astype(np.float16), stats
    if output_dtype != "float32":
        raise ValueError("--daily-output-dtype must be 'float32' or 'float16'.")
    return x_dyn, stats


def spatial_patch_cache_path(
    cache_dir: Path,
    *,
    climate_data_dir: Path,
    variable: str,
    target_df_pl: pl.DataFrame,
    n_days: int,
    patch_size: int,
    source_token: str | None,
) -> Path:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(climate_data_dir.resolve()).encode("utf-8"))
    if source_token:
        hasher.update(source_token.encode("utf-8"))
    hasher.update(variable.encode("utf-8"))
    hasher.update(str(n_days).encode("utf-8"))
    hasher.update(str(patch_size).encode("utf-8"))

    cache_pd = target_df_pl.select(["acq_date", "lat_rounded", "lon_rounded"]).to_pandas()
    cache_pd["acq_date"] = pd.to_datetime(cache_pd["acq_date"]).astype("datetime64[ns]")
    row_hashes = pd.util.hash_pandas_object(cache_pd, index=False).to_numpy(dtype=np.uint64)
    hasher.update(row_hashes.tobytes())
    return cache_dir / f"{variable}_spatial{patch_size}x{patch_size}_{hasher.hexdigest()}.npy"


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp_path.replace(path)


def expanded_spatial_range(values, step: float | None, radius: int) -> tuple[float, float]:
    lower = float(values.min())
    upper = float(values.max())
    if lower > upper:
        lower, upper = upper, lower
    step_value = float(step or 0.0)
    if not np.isfinite(step_value) or step_value <= 0.0:
        step_value = 0.25
    pad = step_value * max(int(radius), 0)
    return lower - pad, upper + pad


def extract_spatial_climate_timeseries(
    ds,
    variable: str,
    target_df_pl: pl.DataFrame,
    *,
    n_days: int,
    patch_size: int,
    location_batch_size: int | None = None,
    max_time_span_days: int | None = None,
    fill_row_batch_size: int | None = None,
    block_cache_dir: Path | None = None,
    block_cache_source_token: str | None = None,
) -> np.ndarray:
    """Return raw climate patches shaped (rows, days, patch, patch)."""

    if patch_size <= 0 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer.")

    n_rows = target_df_pl.height
    if n_rows == 0:
        return np.empty((0, n_days, patch_size, patch_size), dtype=np.float32)

    vt_full = ds["valid_time"]
    if vt_full.size == 0:
        raise ValueError("Dataset valid_time coordinate is empty.")
    vt_numeric = vt_full.values.astype("datetime64[ns]")
    if np.any(np.diff(vt_numeric) < np.timedelta64(0, "ns")):
        ds = ds.sortby("valid_time")
        vt_full = ds["valid_time"]
        vt_numeric = vt_full.values.astype("datetime64[ns]")

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Dataset must contain 'latitude' and 'longitude' coordinates.")

    lat_vals = np.asarray(ds["latitude"].values, dtype=float)
    lon_vals = np.asarray(ds["longitude"].values, dtype=float)
    out = np.full((n_rows, n_days, patch_size, patch_size), np.nan, dtype=np.float32)

    target_pd = target_df_pl.select(["acq_date", "lat_rounded", "lon_rounded"]).to_pandas()
    acq_dates = pd.to_datetime(target_pd["acq_date"]).to_numpy(dtype="datetime64[D]")
    start_dates = acq_dates - np.timedelta64(n_days - 1, "D")

    xy = target_pd[["lat_rounded", "lon_rounded"]].to_numpy(dtype=np.float64)
    unique_xy, row_to_location = np.unique(xy, axis=0, return_inverse=True)
    n_locations = unique_xy.shape[0]

    lat_indices = _nearest_coordinate_indices(lat_vals, unique_xy[:, 0])
    lon_indices = _nearest_coordinate_indices(lon_vals, unique_xy[:, 1])
    radius = patch_size // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    patch_lat_by_location = np.clip(lat_indices[:, None] + offsets[None, :], 0, len(lat_vals) - 1)
    patch_lon_by_location = np.clip(lon_indices[:, None] + offsets[None, :], 0, len(lon_vals) - 1)

    var_data = ds[variable]
    lat_chunk_size = _chunk_size_for_dim(var_data, "latitude", len(lat_vals))
    lon_chunk_size = _chunk_size_for_dim(var_data, "longitude", len(lon_vals))
    location_order = _location_processing_order(lat_indices, lon_indices, lat_chunk_size, lon_chunk_size)
    row_order, location_offsets = _rows_grouped_by_location(row_to_location, n_locations)

    location_batch_size = max(0, int(location_batch_size or 0))
    max_time_span_days = max(0, int(max_time_span_days if max_time_span_days is not None else 180))
    fill_row_batch_size = max(1, int(fill_row_batch_size if fill_row_batch_size is not None else 50_000))
    max_slab_spatial_cells = max(1, int(250_000))
    day_offsets = np.arange(n_days, dtype=np.int64)

    progress = tqdm(
        total=n_locations,
        desc=f"{variable} spatial climate chunks",
        unit="loc",
        mininterval=0.5,
        leave=False,
    )
    for batch_locations, _lat_bin, _lon_bin in _iter_location_chunk_batches(
        location_order,
        lat_indices,
        lon_indices,
        lat_chunk_size,
        lon_chunk_size,
        location_batch_size,
    ):
        row_indices = _rows_for_location_batch(batch_locations, row_order, location_offsets)
        if row_indices.size == 0:
            progress.update(batch_locations.size)
            continue

        batch_patch_lat = patch_lat_by_location[batch_locations].reshape(-1)
        batch_patch_lon = patch_lon_by_location[batch_locations].reshape(-1)
        lat_start = int(batch_patch_lat.min())
        lat_stop = int(batch_patch_lat.max()) + 1
        lon_start = int(batch_patch_lon.min())
        lon_stop = int(batch_patch_lon.max()) + 1
        if (lat_stop - lat_start) * (lon_stop - lon_start) > max_slab_spatial_cells:
            center_lat = lat_indices[batch_locations]
            center_lon = lon_indices[batch_locations]
            lat_start = max(0, int(center_lat.min()) - radius)
            lat_stop = min(len(lat_vals), int(center_lat.max()) + radius + 1)
            lon_start = max(0, int(center_lon.min()) - radius)
            lon_stop = min(len(lon_vals), int(center_lon.max()) + radius + 1)

        for time_block_rows in _iter_temporal_row_blocks(row_indices, acq_dates, max_time_span_days):
            batch_start_day = start_dates[time_block_rows].min()
            batch_end_day = acq_dates[time_block_rows].max()

            left = int(np.searchsorted(vt_numeric, batch_start_day.astype("datetime64[ns]"), side="left"))
            right = int(np.searchsorted(vt_numeric, batch_end_day.astype("datetime64[ns]"), side="right"))
            if right - left <= 0:
                continue

            cache_path = None
            if block_cache_dir is not None and block_cache_source_token:
                from src.feature_generation.prepare_climate_data import _climate_block_cache_path

                cache_path = _climate_block_cache_path(
                    str(block_cache_dir),
                    block_cache_source_token,
                    variable,
                    batch_start_day,
                    batch_end_day,
                    lat_start,
                    lat_stop,
                    lon_start,
                    lon_stop,
                )

            daily = _compute_daily_slab(
                var_data,
                left,
                right,
                batch_start_day,
                batch_end_day,
                lat_start,
                lat_stop,
                lon_start,
                lon_stop,
                cache_path,
            )

            location_ids = row_to_location[time_block_rows]
            local_patch_lats = patch_lat_by_location[location_ids] - lat_start
            local_patch_lons = patch_lon_by_location[location_ids] - lon_start
            row_offsets_days = (acq_dates[time_block_rows] - batch_start_day).astype("timedelta64[D]").astype(np.int64)

            for fill_start in range(0, time_block_rows.size, fill_row_batch_size):
                fill_stop = min(time_block_rows.size, fill_start + fill_row_batch_size)
                fill_rows = time_block_rows[fill_start:fill_stop]
                fill_offsets = row_offsets_days[fill_start:fill_stop]
                fill_patch_lats = local_patch_lats[fill_start:fill_stop]
                fill_patch_lons = local_patch_lons[fill_start:fill_stop]
                window_indices = fill_offsets[:, None] - (n_days - 1) + day_offsets[None, :]
                out[fill_rows] = daily[
                    window_indices[:, :, None, None],
                    fill_patch_lats[:, None, :, None],
                    fill_patch_lons[:, None, None, :],
                ]

        progress.update(batch_locations.size)
    progress.close()
    return out


def extract_spatial_climate_timeseries_fragmented(
    fragments,
    variable: str,
    target_df_pl: pl.DataFrame,
    *,
    n_days: int,
    patch_size: int,
    assignments: np.ndarray | None = None,
    location_batch_size: int | None = None,
    max_time_span_days: int | None = None,
    fill_row_batch_size: int | None = None,
    block_cache_dir: Path | None = None,
    block_cache_source_token: str | None = None,
) -> np.ndarray:
    out = np.full((target_df_pl.height, n_days, patch_size, patch_size), np.nan, dtype=np.float32)
    if target_df_pl.height == 0:
        return out
    if assignments is None:
        assignments = assign_rows_to_climate_fragments(fragments, target_df_pl, n_days)

    for fragment_idx in _fragment_priority_order(fragments):
        row_indices = np.flatnonzero(assignments == fragment_idx)
        if row_indices.size == 0:
            continue
        fragment = fragments[fragment_idx]
        target_subset = _subset_for_fragment(target_df_pl, row_indices, fragment)
        time_range = _time_range_for_subset(target_subset, n_days)
        radius = patch_size // 2
        lat_range = expanded_spatial_range(target_subset["lat_rounded"], fragment.lat_step, radius)
        lon_range = expanded_spatial_range(target_subset["lon_rounded"], fragment.lon_step, radius)
        print(
            f"Extracting {patch_size}x{patch_size} spatial {variable} from fragment {fragment_idx} "
            f"({row_indices.size:,} rows, {fragment.file_count} file(s))",
            flush=True,
        )
        ds = open_climate_fragment(
            fragment,
            time_range=time_range,
            lat_range=lat_range,
            lon_range=lon_range,
        )
        try:
            fragment_block_token = None
            if block_cache_dir is not None and block_cache_source_token:
                fragment_block_token = _climate_block_source_token(
                    f"{block_cache_source_token}|fragment:{fragment_idx}|{fragment.source_token_payload()}",
                    variable,
                    ds,
                )
            out[row_indices] = extract_spatial_climate_timeseries(
                ds,
                variable,
                target_subset.select(["acq_date", "lat_rounded", "lon_rounded"]),
                n_days=n_days,
                patch_size=patch_size,
                location_batch_size=location_batch_size,
                max_time_span_days=max_time_span_days,
                fill_row_batch_size=fill_row_batch_size,
                block_cache_dir=block_cache_dir,
                block_cache_source_token=fragment_block_token,
            )
        finally:
            ds.close()
    return out


def build_spatial_daily_dynamic_tensor(
    *,
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    target_cache_dir: Path,
    climate_data_dir: Path,
    climate_cache_dir: Path,
    variables: list[str],
    n_days: int,
    patch_size: int,
    output_dtype: str,
    work_dir: Path,
    dynamic_tensor_storage: str,
    spatial_cache_policy: str,
    use_block_cache: bool,
) -> tuple[np.ndarray, dict[str, dict[str, object]], Path | None]:
    if patch_size <= 0 or patch_size % 2 != 1:
        raise ValueError("--spatial-patch-size must be a positive odd integer.")
    dtype = np.float16 if output_dtype == "float16" else np.float32
    if output_dtype not in {"float16", "float32"}:
        raise ValueError("--daily-output-dtype must be 'float32' or 'float16'.")
    if dynamic_tensor_storage not in {"memory", "memmap"}:
        raise ValueError("--dynamic-tensor-storage must be 'memory' or 'memmap'.")
    if spatial_cache_policy not in {"none", "read", "write", "read-write"}:
        raise ValueError("--spatial-cache-policy must be one of: none, read, write, read-write.")

    target_pd = frame[["datetime", "lat_rounded", "lon_rounded"]].copy()
    target_pd = target_pd.rename(columns={"datetime": "acq_date"})
    target_df_pl = pl.from_pandas(target_pd)

    tensor_shape = (len(frame), n_days, patch_size, patch_size, len(variables))
    tensor_gib = np.prod(tensor_shape, dtype=np.float64) * np.dtype(dtype).itemsize / (1024**3)
    if dynamic_tensor_storage == "memmap":
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_path = work_dir / f"x_dyn_daily_spatial_{patch_size}x{patch_size}.tmp.npy"
        print(f"Allocating daily-spatial tensor as memmap: {temp_path} ({tensor_gib:.2f} GiB).", flush=True)
        x_dyn = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=dtype,
            shape=tensor_shape,
        )
    else:
        temp_path = None
        print(f"Allocating daily-spatial tensor in memory ({tensor_gib:.2f} GiB).", flush=True)
        x_dyn = np.empty(tensor_shape, dtype=dtype)

    read_spatial_cache = spatial_cache_policy in {"read", "read-write"}
    write_spatial_cache = spatial_cache_policy in {"write", "read-write"}
    block_cache_dir = climate_cache_dir if use_block_cache else None

    stats: dict[str, dict[str, object]] = {}
    for var_idx, variable in enumerate(variables):
        start = time.time()
        fragments = discover_climate_fragments(str(climate_data_dir), variable)
        source_token = climate_fragments_source_token(fragments)
        cache_path = (
            spatial_patch_cache_path(
                climate_cache_dir,
                climate_data_dir=climate_data_dir,
                variable=variable,
                target_df_pl=target_df_pl,
                n_days=n_days,
                patch_size=patch_size,
                source_token=source_token,
            )
            if read_spatial_cache or write_spatial_cache
            else None
        )
        if read_spatial_cache and cache_path is not None and cache_path.exists():
            raw = np.load(cache_path, allow_pickle=False)
            if raw.shape != (len(frame), n_days, patch_size, patch_size):
                print(f"Ignoring stale spatial cache {cache_path}: got {raw.shape}", flush=True)
                raw = None
            else:
                print(f"Loaded spatial climate cache for {variable}: {cache_path}", flush=True)
        else:
            raw = None

        raw_generated = raw is None
        if raw is None:
            use_fragmented = len(fragments) > 1
            if use_fragmented:
                bounds = check_fragmented_dataset_bounds(fragments, target_df_pl, n_days=n_days)
                print_fragmented_bounds_check(variable, bounds, fragments)
                if not bounds["sufficient"]:
                    print(
                        f"Warning: fragmented dataset for {variable} is incomplete; "
                        "covered rows will use spatial ERA5 and gaps will use point-cache fallback.",
                        flush=True,
                    )
                raw = extract_spatial_climate_timeseries_fragmented(
                    fragments,
                    variable,
                    target_df_pl,
                    n_days=n_days,
                    patch_size=patch_size,
                    assignments=bounds["assignments"],
                    block_cache_dir=block_cache_dir,
                    block_cache_source_token=source_token if use_block_cache else None,
                )
            else:
                fragment = fragments[0]
                radius = patch_size // 2
                lat_range = expanded_spatial_range(target_pd["lat_rounded"], fragment.lat_step, radius)
                lon_range = expanded_spatial_range(target_pd["lon_rounded"], fragment.lon_step, radius)
                time_range = (
                    pd.to_datetime(target_pd["acq_date"]).min() - pd.DateOffset(days=n_days - 1),
                    pd.to_datetime(target_pd["acq_date"]).max(),
                )
                ds = load_climate_variable_mf(
                    str(climate_data_dir),
                    variable,
                    time_range=time_range,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    test_mode=False,
                )
                try:
                    bounds = check_dataset_bounds(ds, target_df_pl, n_days=n_days)
                    print_dataset_bounds_check(variable, bounds)
                    if not bounds["sufficient"]:
                        print(
                            f"Warning: dataset for {variable} is incomplete; "
                            "available rows will use spatial ERA5 and gaps will use point-cache fallback.",
                            flush=True,
                        )
                    block_token = _climate_block_source_token(str(climate_data_dir), variable, ds)
                    raw = extract_spatial_climate_timeseries(
                        ds,
                        variable,
                        target_df_pl,
                        n_days=n_days,
                        patch_size=patch_size,
                        block_cache_dir=block_cache_dir,
                        block_cache_source_token=block_token if use_block_cache else None,
                    )
                finally:
                    ds.close()

        fallback_info = fill_spatial_gaps_from_point_cache(
            raw,
            frame=frame,
            target_cache_dir=target_cache_dir,
            climate_cache_dir=climate_cache_dir,
            variable=variable,
            n_days=n_days,
        )
        if raw_generated and write_spatial_cache and cache_path is not None:
            save_npy_atomic(cache_path, raw.astype(np.float32, copy=False))
            print(f"Saved spatial climate cache for {variable}: {cache_path}", flush=True)

        fill, mean, std = write_scaled_spatial_daily_channel(
            raw,
            train_mask,
            x_dyn[..., var_idx],
        )
        if hasattr(x_dyn, "flush"):
            x_dyn.flush()
        stats[variable] = {
            "cache_path": str(cache_path) if cache_path is not None else None,
            "spatial_cache_policy": spatial_cache_policy,
            "block_cache_enabled": bool(use_block_cache),
            "fill_shape": list(fill.shape),
            "fill": fill.astype(float).tolist(),
            "mean": float(mean),
            "std": float(std),
            "point_cache_fallback": fallback_info,
            "elapsed_seconds": float(time.time() - start),
        }
    return x_dyn, stats, temp_path


def build_categorical_ids(frame: pd.DataFrame, columns: list[str], train_mask: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    parts: list[np.ndarray] = []
    maps: dict[str, dict[str, int]] = {}
    train_idx = np.flatnonzero(train_mask)
    for col in columns:
        values = frame[col].fillna("__missing__").astype(str)
        categories = pd.Index(values.iloc[train_idx].unique()).sort_values().tolist()
        if "__missing__" in categories:
            categories = ["__missing__"] + [value for value in categories if value != "__missing__"]
        else:
            categories = ["__missing__"] + categories
        mapping = {value: idx for idx, value in enumerate(categories)}
        codes = values.map(mapping).fillna(mapping["__missing__"]).astype(np.int32).to_numpy()
        parts.append(codes.reshape(-1, 1))
        maps[col] = mapping
    if not parts:
        return np.zeros((len(frame), 0), dtype=np.int32), maps
    return np.hstack(parts).astype(np.int32), maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-array prepared_data.npz for neural training.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet"),
    )
    parser.add_argument("--target-cache-dir", type=Path, default=Path("data/saved_features/target_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/saved_features/nn_train_data"))
    parser.add_argument("--lead-time-days", type=int, default=30)
    parser.add_argument("--dynamic-mode", choices=["summary", "daily", "daily_spatial"], default="summary")
    parser.add_argument("--climate-cache-dir", type=Path, default=Path("data/saved_features/climate_features_cache"))
    parser.add_argument(
        "--climate-data-dir",
        type=Path,
        default=Path("/home/ids/vmorozov/data/climate_data/climate_features/ERA5"),
        help="Raw climate zarr root used by --dynamic-mode daily_spatial.",
    )
    parser.add_argument("--climate-variables", nargs="+", default=["t2m", "d2m", "tp", "stl1"])
    parser.add_argument("--n-days", type=int, default=128)
    parser.add_argument("--spatial-patch-size", type=int, default=3)
    parser.add_argument("--daily-output-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--dynamic-tensor-storage",
        choices=["memory", "memmap"],
        default="memory",
        help=(
            "Storage for the assembled daily_spatial dynamic tensor. "
            "'memory' avoids the large temporary .npy file; 'memmap' restores the old disk-backed behavior."
        ),
    )
    parser.add_argument(
        "--spatial-cache-policy",
        choices=["none", "read", "write", "read-write"],
        default="read",
        help=(
            "Controls per-variable spatial patch caches for daily_spatial. "
            "'read' reuses existing caches but does not write new large .npy files."
        ),
    )
    parser.add_argument(
        "--enable-climate-block-cache",
        action="store_true",
        help="Allow daily_spatial extraction to write/read intermediate climate block .npy caches.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema_cols = set(pq.read_schema(args.source).names)
    dynamic_cols = [col for col in DEFAULT_NEURAL_DYNAMIC_COLUMNS if col in schema_cols]
    static_cols = [col for col in DEFAULT_NEURAL_STATIC_COLUMNS if col in schema_cols]
    categorical_cols = [col for col in DEFAULT_NEURAL_CATEGORICAL_COLUMNS if col in schema_cols]
    missing = {
        "dynamic": [col for col in DEFAULT_NEURAL_DYNAMIC_COLUMNS if col not in schema_cols]
        if args.dynamic_mode == "summary"
        else [],
        "static": [col for col in DEFAULT_NEURAL_STATIC_COLUMNS if col not in schema_cols],
        "categorical": [col for col in DEFAULT_NEURAL_CATEGORICAL_COLUMNS if col not in schema_cols],
    }
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"Source parquet is missing expected NN columns: {missing}")

    base_cols = ["datetime", "lat_rounded", "lon_rounded", "month", "day", "year", "count", "soft_label"]
    columns = list(dict.fromkeys(base_cols + (dynamic_cols if args.dynamic_mode == "summary" else []) + static_cols + categorical_cols))
    print(f"Reading {len(columns)} columns from {args.source} ...", flush=True)
    frame = pd.read_parquet(args.source, columns=columns)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["lat_rounded"] = pd.to_numeric(frame["lat_rounded"], errors="coerce").astype("float32")
    frame["lon_rounded"] = pd.to_numeric(frame["lon_rounded"], errors="coerce").astype("float32")
    frame["count"] = (pd.to_numeric(frame["count"], errors="coerce").fillna(0) > 0).astype("int8")

    metadata = read_target_metadata(args.target_cache_dir)
    frame = frame.merge(
        metadata,
        on=["datetime", "lat_rounded", "lon_rounded", "count"],
        how="left",
        suffixes=("", "_target"),
        validate="many_to_one",
    )
    matched = frame["negative_stratum"].notna() | frame["sample_weight"].notna()
    print(f"Matched target metadata for {int(matched.sum())}/{len(frame)} rows.", flush=True)

    frame = frame.sort_values(["datetime", "lat_rounded", "lon_rounded"], kind="stable").reset_index(drop=True)
    dates = pd.to_datetime(frame["datetime"], errors="coerce")
    years = pd.to_numeric(frame["year"], errors="coerce").fillna(dates.dt.year).astype(np.int16).to_numpy()
    split = split_codes(years)
    train_mask = split == 0
    if not train_mask.any() or not (split == 1).any():
        raise ValueError("Prepared split must contain train and validation rows.")

    daily_dynamic_stats: dict[str, dict[str, object]] | None = None
    temp_x_dyn_path: Path | None = None
    if args.dynamic_mode == "summary":
        x_dyn_scaled, dyn_fill, dyn_mean, dyn_std = fit_scale(numeric_matrix(frame, dynamic_cols), train_mask)
        x_dyn = x_dyn_scaled.reshape((len(frame), 5, 6)).astype(np.float32)
        dynamic_shape = [None, 5, 6]
    elif args.dynamic_mode == "daily":
        x_dyn, daily_dynamic_stats = build_daily_dynamic_tensor(
            frame=frame,
            train_mask=train_mask,
            target_cache_dir=args.target_cache_dir,
            climate_cache_dir=args.climate_cache_dir,
            variables=list(args.climate_variables),
            n_days=int(args.n_days),
            output_dtype=str(args.daily_output_dtype),
        )
        dyn_fill = dyn_mean = dyn_std = None
        dynamic_cols = [f"{variable}_day_{day:03d}" for day in range(int(args.n_days)) for variable in args.climate_variables]
        dynamic_shape = [None, int(args.n_days), len(args.climate_variables)]
    else:
        x_dyn, daily_dynamic_stats, temp_x_dyn_path = build_spatial_daily_dynamic_tensor(
            frame=frame,
            train_mask=train_mask,
            target_cache_dir=args.target_cache_dir,
            climate_data_dir=args.climate_data_dir,
            climate_cache_dir=args.climate_cache_dir,
            variables=list(args.climate_variables),
            n_days=int(args.n_days),
            patch_size=int(args.spatial_patch_size),
            output_dtype=str(args.daily_output_dtype),
            work_dir=args.output_dir,
            dynamic_tensor_storage=str(args.dynamic_tensor_storage),
            spatial_cache_policy=str(args.spatial_cache_policy),
            use_block_cache=bool(args.enable_climate_block_cache),
        )
        dyn_fill = dyn_mean = dyn_std = None
        radius = int(args.spatial_patch_size) // 2
        dynamic_cols = [
            f"{variable}_day_{day:03d}_dy_{dy:+d}_dx_{dx:+d}"
            for day in range(int(args.n_days))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            for variable in args.climate_variables
        ]
        dynamic_shape = [
            None,
            int(args.n_days),
            int(args.spatial_patch_size),
            int(args.spatial_patch_size),
            len(args.climate_variables),
        ]
    x_static, stat_fill, stat_mean, stat_std = fit_scale(numeric_matrix(frame, static_cols), train_mask)
    x_cat, categorical_maps = build_categorical_ids(frame, categorical_cols, train_mask)

    y = frame["count"].to_numpy(dtype=np.int8)
    soft_label = (
        pd.to_numeric(frame["soft_label"], errors="coerce")
        .fillna(frame["count"].astype(float))
        .clip(0.0, 1.0)
        .to_numpy(dtype=np.float32)
    )
    sample_weight = (
        pd.to_numeric(frame["sample_weight"], errors="coerce")
        .fillna(1.0)
        .replace([np.inf, -np.inf], 1.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float32)
    )
    stratum = frame["negative_stratum"].fillna("positive").astype(str)
    stratum_names = sorted(stratum.unique().tolist())
    stratum_name_to_id = {name: idx for idx, name in enumerate(stratum_names)}
    negative_stratum_id = stratum.map(stratum_name_to_id).astype(np.int16).to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "prepared_data.npz"
    print(f"Writing {out_path} ...", flush=True)
    np.savez_compressed(
        out_path,
        x_dyn=x_dyn,
        x_static=x_static,
        x_cat=x_cat,
        y=y,
        soft_label=soft_label,
        sample_weight=sample_weight,
        negative_stratum_id=negative_stratum_id,
        split=split,
        years=years,
        dates=dates.to_numpy(dtype="datetime64[D]").astype("int64"),
        lat=frame["lat_rounded"].to_numpy(dtype=np.float32),
        lon=frame["lon_rounded"].to_numpy(dtype=np.float32),
        lead_time_days=np.full(len(frame), int(args.lead_time_days), dtype=np.int16),
    )

    split_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(split, return_counts=True))}
    metadata_out = {
        "source_features_path": str(args.source),
        "target_cache_dir": str(args.target_cache_dir),
        "dynamic_mode": args.dynamic_mode,
        "dynamic_source_columns": dynamic_cols,
        "static_columns": static_cols,
        "categorical_columns": categorical_cols,
        "dynamic_shape": dynamic_shape,
        "daily_climate_variables": list(args.climate_variables)
        if args.dynamic_mode in {"daily", "daily_spatial"}
        else None,
        "daily_spatial_patch_size": int(args.spatial_patch_size)
        if args.dynamic_mode == "daily_spatial"
        else None,
        "climate_data_dir": str(args.climate_data_dir) if args.dynamic_mode == "daily_spatial" else None,
        "dynamic_tensor_storage": str(args.dynamic_tensor_storage)
        if args.dynamic_mode == "daily_spatial"
        else None,
        "spatial_cache_policy": str(args.spatial_cache_policy)
        if args.dynamic_mode == "daily_spatial"
        else None,
        "climate_block_cache_enabled": bool(args.enable_climate_block_cache)
        if args.dynamic_mode == "daily_spatial"
        else None,
        "daily_dynamic_stats": daily_dynamic_stats,
        "static_shape": [None, int(x_static.shape[1])],
        "categorical_shape": [None, int(x_cat.shape[1])],
        "negative_stratum_name_to_id": stratum_name_to_id,
        "negative_stratum_id_to_name": {str(value): key for key, value in stratum_name_to_id.items()},
        "split_counts": split_counts,
        "positive_rate": float(y.mean()),
        "soft_label_mean": float(soft_label.mean()),
        "metadata_match_rate": float(matched.mean()),
    }
    (args.output_dir / "prepared_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "dynamic_columns": dynamic_cols,
            "dynamic_fill": dyn_fill,
            "dynamic_mean": dyn_mean,
            "dynamic_std": dyn_std,
            "dynamic_mode": args.dynamic_mode,
            "daily_climate_variables": list(args.climate_variables)
            if args.dynamic_mode in {"daily", "daily_spatial"}
            else None,
            "daily_spatial_patch_size": int(args.spatial_patch_size)
            if args.dynamic_mode == "daily_spatial"
            else None,
            "dynamic_tensor_storage": str(args.dynamic_tensor_storage)
            if args.dynamic_mode == "daily_spatial"
            else None,
            "spatial_cache_policy": str(args.spatial_cache_policy)
            if args.dynamic_mode == "daily_spatial"
            else None,
            "climate_block_cache_enabled": bool(args.enable_climate_block_cache)
            if args.dynamic_mode == "daily_spatial"
            else None,
            "daily_dynamic_stats": daily_dynamic_stats,
            "static_columns": static_cols,
            "static_fill": stat_fill,
            "static_mean": stat_mean,
            "static_std": stat_std,
            "categorical_columns": categorical_cols,
            "categorical_maps": categorical_maps,
            "negative_stratum_name_to_id": stratum_name_to_id,
        },
        args.output_dir / "encoders_meta.joblib",
        compress=("gzip", 3),
    )
    if temp_x_dyn_path is not None and temp_x_dyn_path.exists():
        del x_dyn
        temp_x_dyn_path.unlink()
    print("Split counts:", split_counts, flush=True)
    print("Strata:", stratum.value_counts().to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
