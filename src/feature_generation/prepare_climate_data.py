import hashlib
import os, sys
import re
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
import time as time_lib
import polars as pl
try:
    from dask.distributed import Client
    DASK_AVAILABLE = True
except ImportError:
    Client = None
    DASK_AVAILABLE = False
from tqdm import tqdm
sys.path.append(os.getcwd())
from src.feature_generation.load_climate_data import (
    ClimateFragment,
    climate_fragments_source_token,
    discover_climate_fragments,
    load_climate_variable_mf,
    open_climate_fragment,
)


client = None
server_configuration = True # if True, use server(very big RAM needed) configuration, if False, use local configuration
single_thread_debug = False


def _get_dask_client():
    """Initializes the Dask client if it's not already running."""
    global client
    if not DASK_AVAILABLE:
        return None
    if client is None:
        if server_configuration:
            client = Client(
                n_workers=1,   
                threads_per_worker=16,      
                memory_limit="0",
            )
        else:
            client = Client(
                processes=False,
                threads_per_worker=os.cpu_count(),
                memory_limit="48GB"
            )
        print(client)

    return client


def _safe_cache_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:80]


def _climate_matrix_cache_path(
    cache_dir: str,
    climate_data_dir: str,
    variable: str,
    coords_pl: pl.DataFrame,
    n_days: int,
    source_token: str | None = None,
) -> Path:
    """Build a cache path keyed by target rows and raw climate input params."""

    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(Path(climate_data_dir).resolve()).encode("utf-8"))
    if source_token:
        hasher.update(source_token.encode("utf-8"))
    hasher.update(variable.encode("utf-8"))
    hasher.update(str(n_days).encode("utf-8"))

    cache_pd = coords_pl.select(["acq_date", "lat_rounded", "lon_rounded"]).to_pandas()
    cache_pd["acq_date"] = pd.to_datetime(cache_pd["acq_date"]).astype("datetime64[ns]")
    row_hashes = pd.util.hash_pandas_object(cache_pd, index=False).to_numpy(dtype=np.uint64)
    hasher.update(row_hashes.tobytes())

    return Path(cache_dir) / f"{_safe_cache_token(variable)}_{hasher.hexdigest()}.npy"


def _save_matrix_cache(cache_path: Path, matrix: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.save(fh, matrix, allow_pickle=False)
    os.replace(tmp_path, cache_path)


def _climate_block_source_token(climate_data_dir: str, variable: str, ds: xr.Dataset) -> str:
    """Stable-enough source token for reusable climate slab cache files."""

    payload = "|".join(
        [
            str(Path(climate_data_dir).resolve()),
            variable,
            str(tuple(ds[variable].shape)),
            str(ds[variable].dtype),
            str(ds["valid_time"].values[0]) if ds["valid_time"].size else "",
            str(ds["valid_time"].values[-1]) if ds["valid_time"].size else "",
            str(ds["latitude"].values[0]) if "latitude" in ds.coords and ds["latitude"].size else "",
            str(ds["latitude"].values[-1]) if "latitude" in ds.coords and ds["latitude"].size else "",
            str(ds["longitude"].values[0]) if "longitude" in ds.coords and ds["longitude"].size else "",
            str(ds["longitude"].values[-1]) if "longitude" in ds.coords and ds["longitude"].size else "",
        ]
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _climate_block_cache_path(
    cache_dir: str,
    source_token: str,
    variable: str,
    start_day: np.datetime64,
    end_day: np.datetime64,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
) -> Path:
    start_token = np.datetime_as_string(start_day.astype("datetime64[D]"), unit="D")
    end_token = np.datetime_as_string(end_day.astype("datetime64[D]"), unit="D")
    filename = (
        f"{source_token}_{start_token}_{end_token}_"
        f"lat{lat_start}-{lat_stop}_lon{lon_start}-{lon_stop}.npy"
    )
    return Path(cache_dir) / "climate_block_cache" / _safe_cache_token(variable) / filename


def _load_block_cache(cache_path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    try:
        cached = np.load(cache_path, allow_pickle=False)
    except Exception as exc:
        print(f"Ignoring unreadable climate block cache {cache_path}: {exc}")
        return None
    if cached.shape != expected_shape:
        print(f"Ignoring stale climate block cache {cache_path}: expected {expected_shape}, got {cached.shape}")
        return None
    return cached


def _save_block_cache(cache_path: Path, block: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.save(fh, block, allow_pickle=False)
    os.replace(tmp_path, cache_path)


def _nan_vector(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=np.float32)


def _build_feature_matrix_vectorized(
    ts_matrix_var: np.ndarray,
    variable: str,
    feature_params: dict,
    feature_names: list[str],
) -> np.ndarray | None:
    """Fast matrix-wide equivalent of ``_extract_single_series`` for common features."""

    if feature_params.get("add_autocorr") or feature_params.get("add_fft") or feature_params.get("add_cumu"):
        return None

    ts = np.asarray(ts_matrix_var, dtype=np.float32)
    if ts.ndim != 2:
        return None

    max_length = feature_params.get("max_length")
    if max_length:
        ts = ts[:, -max_length:]
    else:
        ts = ts[:, :0]

    n_rows, n_obs = ts.shape
    var = variable.lower()
    lags = feature_params.get("lags") or []
    windows = feature_params.get("windows") or []
    spans = feature_params.get("spans") or []
    trend_windows = feature_params.get("trend_windows") or []

    columns: list[np.ndarray] = []
    names: list[str] = []

    def add_column(name: str, values: np.ndarray) -> None:
        columns.append(np.asarray(values, dtype=np.float32).reshape(n_rows))
        names.append(name)

    for lag in lags:
        if n_obs >= lag:
            values = ts[:, n_obs - lag]
        else:
            values = _nan_vector(n_rows)
        add_column(f"{var}_lag_{lag}", values)

    for window in windows:
        if n_obs >= window:
            window_values = ts[:, -window:]
            add_column(f"{var}_mean_{window}", window_values.mean(axis=1))
            add_column(f"{var}_std_{window}", window_values.std(axis=1))
            if feature_params.get("add_roll_ext"):
                add_column(f"{var}_min_{window}", window_values.min(axis=1))
                add_column(f"{var}_max_{window}", window_values.max(axis=1))
                add_column(f"{var}_median_{window}", np.median(window_values, axis=1))
        else:
            add_column(f"{var}_mean_{window}", _nan_vector(n_rows))
            add_column(f"{var}_std_{window}", _nan_vector(n_rows))
            if feature_params.get("add_roll_ext"):
                add_column(f"{var}_min_{window}", _nan_vector(n_rows))
                add_column(f"{var}_max_{window}", _nan_vector(n_rows))
                add_column(f"{var}_median_{window}", _nan_vector(n_rows))

    for span in spans:
        if n_obs == 0:
            values = _nan_vector(n_rows)
        else:
            alpha = np.float32(2.0 / (span + 1.0))
            ewma = ts[:, 0].copy()
            for col_idx in range(1, n_obs):
                ewma += alpha * (ts[:, col_idx] - ewma)
            values = ewma
        add_column(f"{var}_ewm_{span}", values)

    if feature_params.get("add_diff") or feature_params.get("add_pct"):
        current = ts[:, -1] if n_obs else _nan_vector(n_rows)
        for lag in lags:
            if n_obs > lag:
                previous = ts[:, -1 - lag]
                if feature_params.get("add_diff"):
                    add_column(f"{var}_diff_{lag}", current - previous)
                if feature_params.get("add_pct"):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pct = np.where(previous != 0.0, (current - previous) / previous, np.nan)
                    add_column(f"{var}_pct_change_{lag}", pct)
            else:
                if feature_params.get("add_diff"):
                    add_column(f"{var}_diff_{lag}", _nan_vector(n_rows))
                if feature_params.get("add_pct"):
                    add_column(f"{var}_pct_change_{lag}", _nan_vector(n_rows))

    if feature_params.get("add_trend"):
        for window in trend_windows:
            if n_obs >= window:
                y = ts[:, -window:].astype(np.float32, copy=False)
                x = np.arange(window, dtype=np.float32)
                sum_x = np.float32(window * (window - 1) / 2.0)
                sum_x2 = np.float32(window * (window - 1) * (2 * window - 1) / 6.0)
                n = np.float32(window)
                denom = n * sum_x2 - sum_x * sum_x
                sum_y = y.sum(axis=1)
                sum_xy = y @ x
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    slope = (n * sum_xy - sum_x * sum_y) / denom
                    intercept = (sum_y - slope * sum_x) / n
                invalid = ~(np.isfinite(slope) & np.isfinite(intercept))
                slope = slope.astype(np.float32, copy=False)
                intercept = intercept.astype(np.float32, copy=False)
                slope[invalid] = np.nan
                intercept[invalid] = np.nan
            else:
                slope = _nan_vector(n_rows)
                intercept = _nan_vector(n_rows)
            add_column(f"{var}_trend_slope_{window}", slope)
            add_column(f"{var}_trend_intercept_{window}", intercept)

    if names != feature_names:
        return None

    if not columns:
        return np.empty((n_rows, 0), dtype=np.float32)

    return np.column_stack(columns).astype(np.float32, copy=False)


def _build_climate_feature_dataframe(
    target_df_pl: pl.DataFrame,
    ts_matrix_per_variable: list[np.ndarray],
    climate_variables: list[str],
    *,
    lags_features: list[int] | None,
    windows_features: list[int] | None,
    spans_features: list[int] | None,
    trend_window_features: list[int] | None,
    max_length_features: int,
    features_to_include_config: dict | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return dataframe with climate features appended and ordered column list."""
    from src.feature_generation.prepare_climate_features import (
        _extract_single_series,
        get_feature_configs_and_names,
    )

    base_df = target_df_pl.to_pandas()
    base_df = base_df.reset_index(drop=True)

    lags = lags_features or []
    windows = windows_features or []
    spans = spans_features or []
    trend_windows_cfg = trend_window_features if trend_window_features is not None else []

    climate_feature_frames: list[pd.DataFrame] = []
    all_climate_feature_names: list[str] = []

    for variable, ts_matrix_var in zip(climate_variables, ts_matrix_per_variable):
        if ts_matrix_var.size == 0:
            continue

        ts_matrix_var = np.asarray(ts_matrix_var, dtype=np.float32)
        if ts_matrix_var.shape[0] == 0:
            continue

        sample_row = None
        for row in ts_matrix_var:
            if np.isfinite(row).any():
                sample_row = row
                break
        if sample_row is None:
            sample_row = ts_matrix_var[0]

        feature_params, feature_names = get_feature_configs_and_names(
            sample_ts_array=np.asarray(sample_row, dtype=np.float32),
            variable_name=variable,
            lags_global=lags,
            windows_global=windows,
            spans_global=spans,
            features_to_include=features_to_include_config,
            trend_window_global=trend_windows_cfg,
            max_length_global=max_length_features,
        )

        if not feature_names:
            continue

        feature_matrix = _build_feature_matrix_vectorized(
            ts_matrix_var,
            variable,
            feature_params,
            feature_names,
        )

        if feature_matrix is None:
            feature_matrix = np.empty((ts_matrix_var.shape[0], len(feature_names)), dtype=np.float32)
            for idx, row in enumerate(ts_matrix_var):
                feature_values = _extract_single_series(
                    row,
                    variable,
                    **feature_params,
                )
                feature_matrix[idx, :] = [
                    np.float32(feature_values[name]) for name in feature_names
                ]

        feature_df = pd.DataFrame(feature_matrix, columns=feature_names)
        climate_feature_frames.append(feature_df)
        all_climate_feature_names.extend(feature_names)

    if climate_feature_frames:
        climate_features_df = pd.concat(climate_feature_frames, axis=1)
        climate_features_df = climate_features_df.reset_index(drop=True)
        base_df = pd.concat([base_df, climate_features_df], axis=1)

    return base_df, all_climate_feature_names

def print_dataset_bounds_check(variable_name, result):
    """Print a formatted, readable version of dataset bounds check results."""
    is_sufficient = result['sufficient']
    details = result['details']
    
    status = "✅ SUFFICIENT" if is_sufficient else "❌ INSUFFICIENT"
    print("\n" + "="*60)
    print(f" DATASET BOUNDS CHECK: {variable_name} - {status}")
    print("="*60)
    
    for dim_name, dim_info in details.items():
        dim_status = "✓" if dim_info['sufficient'] else "✗"
        status_color = "" if dim_info['sufficient'] else "-> MISMATCH"
        
        print(f"\n{dim_name.upper()} {dim_status} {status_color}")
        
        if dim_name == 'time':
            ds_min_str = "N/A"
            ds_max_str = "N/A"
            if isinstance(dim_info['dataset_range'][0], (np.datetime64, pd.Timestamp)):
                ds_min_str = pd.to_datetime(dim_info['dataset_range'][0]).strftime('%Y-%m-%d')
            if isinstance(dim_info['dataset_range'][1], (np.datetime64, pd.Timestamp)):
                ds_max_str = pd.to_datetime(dim_info['dataset_range'][1]).strftime('%Y-%m-%d')
            ds_range_str = f"{ds_min_str} to {ds_max_str}"

            req_min_str = pd.to_datetime(dim_info['required_range'][0]).strftime('%Y-%m-%d')
            req_max_str = pd.to_datetime(dim_info['required_range'][1]).strftime('%Y-%m-%d')
            req_range_str = f"{req_min_str} to {req_max_str}"
        else:
            ds_range_str = f"{dim_info['dataset_range'][0]} to {dim_info['dataset_range'][1]}"
            req_range_str = f"{dim_info['required_range'][0]} to {dim_info['required_range'][1]}"
        
        print(f"  Dataset range: {ds_range_str}")
        print(f"  Required range: {req_range_str}")
    
    print("\n" + "="*60)

    
def check_dataset_bounds(ds, target_df, n_days=129, time_coord_name="valid_time"):
    """
    Check if xarray dataset has sufficient time, latitude and longitude bounds
    to cover the requirements of the target dataframe.
    """
    if not isinstance(target_df, pl.DataFrame):
        target_df = pl.from_pandas(target_df) if isinstance(target_df, pd.DataFrame) else pl.DataFrame(target_df)
    
    if target_df.height == 0:
        print("Warning: Target DataFrame is empty. Skipping bounds check.")
        return {'sufficient': True, 'details': {}}

    end_of_slices = pd.to_datetime(target_df['acq_date'].to_numpy()) # pd.Series for min/max
    start_of_slices = (end_of_slices - pd.DateOffset(days=max(n_days - 1, 0)))
    
    lats_pl = target_df["lat_rounded"]
    lons_pl = target_df["lon_rounded"]
    
    required_time_min = start_of_slices.min().to_datetime64()
    required_time_max = end_of_slices.max().to_datetime64()

    required_lat_min, required_lat_max = lats_pl.min(), lats_pl.max()
    required_lon_min, required_lon_max = lons_pl.min(), lons_pl.max()
    
    # Extract bounds from dataset, ensuring coordinates exist
    ds_time_min, ds_time_max = (None, None)
    if time_coord_name in ds.coords and ds[time_coord_name].size > 0:
        ds_times_arr = ds[time_coord_name].compute().values # Compute for min/max
        ds_time_min, ds_time_max = ds_times_arr.min(), ds_times_arr.max()
    
    ds_lat_min, ds_lat_max = (None, None)
    if 'latitude' in ds.coords and ds.latitude.size > 0:
        ds_lats_arr = ds.latitude.compute().values
        ds_lat_min, ds_lat_max = ds_lats_arr.min(), ds_lats_arr.max()
    
    ds_lon_min, ds_lon_max = (None, None)
    if 'longitude' in ds.coords and ds.longitude.size > 0:
        ds_lons_arr = ds.longitude.compute().values
        ds_lon_min, ds_lon_max = ds_lons_arr.min(), ds_lons_arr.max()

    # ------------------------------------------------------------------
    #  Determine per-dimension tolerances based on dataset spacing
    #  (helps to treat tiny rounding differences as sufficient coverage)
    # ------------------------------------------------------------------
    lat_tol = None
    if 'latitude' in ds.coords and ds.latitude.size > 1:
        # Use half of the median spacing between adjacent latitude points
        lat_diffs = np.abs(np.diff(np.sort(ds.latitude.compute().values)))
        if lat_diffs.size:
            lat_tol = 0.5 * np.median(lat_diffs)
    lon_tol = None
    if 'longitude' in ds.coords and ds.longitude.size > 1:
        lon_diffs = np.abs(np.diff(np.sort(ds.longitude.compute().values)))
        if lon_diffs.size:
            lon_tol = 0.5 * np.median(lon_diffs)
    # Time tolerance: assume regular daily data → ½ day buffer (12 h)
    time_tol = np.timedelta64(12, "h")

    # Handle longitude normalization if necessary (conceptual, needs careful implementation if ranges cross dateline)
    # For simplicity, assuming direct comparison is okay for now or that data is already 0-360 or -180-180 consistently.
    lon_sufficient = False
    if ds_lon_min is not None and ds_lon_max is not None and required_lon_min is not None and required_lon_max is not None:
        # Simplified check: assumes dataset and required longitudes are in a comparable system (e.g., both -180 to 180 or 0 to 360)
        # This might need adjustment if dataset longitudes are e.g. 0-359 and required are -180 to 180.
        # For now, assume they are compatible for a simple range check.
        # A more robust check would handle dateline wrapping if ds spans it (e.g. 270 to 90 degrees)
        # and required range is, say, 350 to 20.
        # Current check:
        if lon_tol is None:
            lon_sufficient = (ds_lon_min <= required_lon_min) and (ds_lon_max >= required_lon_max)
        else:
            lon_sufficient = (ds_lon_min <= required_lon_min + lon_tol) and (ds_lon_max >= required_lon_max - lon_tol)
    
    time_sufficient = False
    if ds_time_min is not None and ds_time_max is not None:
        # Ensure comparable types for time comparison
        ds_time_min_dt64 = np.datetime64(ds_time_min)
        ds_time_max_dt64 = np.datetime64(ds_time_max)
        time_sufficient = (ds_time_min_dt64 <= (required_time_min + time_tol)) and (ds_time_max_dt64 >= (required_time_max - time_tol))

    lat_sufficient = False
    if ds_lat_min is not None and ds_lat_max is not None and required_lat_min is not None and required_lat_max is not None:
        # ds_lat_min/max are numeric extrema, so the coverage check is identical
        # for increasing and decreasing latitude coordinates.
        if lat_tol is None:
            lat_sufficient = (ds_lat_min <= required_lat_min) and (ds_lat_max >= required_lat_max)
        else:
            lat_sufficient = (ds_lat_min <= required_lat_min + lat_tol) and (ds_lat_max >= required_lat_max - lat_tol)

    all_sufficient = time_sufficient and lat_sufficient and lon_sufficient
    
    result = {
        'sufficient': all_sufficient,
        'details': {
            'time': {
                'sufficient': time_sufficient,
                'dataset_range': (ds_time_min, ds_time_max),
                'required_range': (required_time_min, required_time_max)
            },
            'latitude': {
                'sufficient': lat_sufficient,
                'dataset_range': (ds_lat_min, ds_lat_max),
                'required_range': (required_lat_min, required_lat_max)
            },
            'longitude': {
                'sufficient': lon_sufficient,
                'dataset_range': (ds_lon_min, ds_lon_max),
                'required_range': (required_lon_min, required_lon_max)
            }
        }
    }
    return result


def _target_arrays_for_fragment_routing(target_df: pl.DataFrame, n_days: int):
    target_pd = target_df.select(["acq_date", "lat_rounded", "lon_rounded"]).to_pandas()
    acq_dates = pd.to_datetime(target_pd["acq_date"]).to_numpy(dtype="datetime64[D]")
    start_dates = acq_dates - np.timedelta64(max(n_days - 1, 0), "D")
    lats = target_pd["lat_rounded"].to_numpy(dtype=np.float64)
    lons = target_pd["lon_rounded"].to_numpy(dtype=np.float64)
    return acq_dates, start_dates, lats, lons


def _normalize_longitudes_for_fragment(lons: np.ndarray, fragment: ClimateFragment) -> np.ndarray:
    """Convert target longitudes to the fragment's coordinate convention."""

    normalized = np.asarray(lons, dtype=np.float64).copy()
    finite = np.isfinite(normalized)
    if not finite.any():
        return normalized

    if fragment.lon_min >= 0 and np.nanmin(normalized[finite]) < 0:
        normalized[finite] = np.mod(normalized[finite], 360.0)
    elif fragment.lon_max <= 180 and fragment.lon_min < 0 and np.nanmax(normalized[finite]) > 180:
        normalized[finite] = ((normalized[finite] + 180.0) % 360.0) - 180.0
    return normalized


def _fragment_coverage_mask(
    fragment: ClimateFragment,
    acq_dates: np.ndarray,
    start_dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    lat_tol = max(float(fragment.lat_step) / 2.0, 1e-9)
    lon_tol = max(float(fragment.lon_step) / 2.0, 1e-9)
    time_tol = np.timedelta64(12, "h")

    fragment_start = np.datetime64(fragment.time_start, "ns")
    fragment_end = np.datetime64(fragment.time_end, "ns")
    starts_ns = start_dates.astype("datetime64[ns]")
    acq_ns = acq_dates.astype("datetime64[ns]")

    time_ok = (fragment_start <= starts_ns + time_tol) & (fragment_end >= acq_ns - time_tol)
    lat_ok = (lats >= fragment.lat_min - lat_tol) & (lats <= fragment.lat_max + lat_tol)

    routed_lons = _normalize_longitudes_for_fragment(lons, fragment)
    if fragment.lon_min <= fragment.lon_max:
        lon_ok = (routed_lons >= fragment.lon_min - lon_tol) & (routed_lons <= fragment.lon_max + lon_tol)
    else:
        lon_ok = (routed_lons >= fragment.lon_min - lon_tol) | (routed_lons <= fragment.lon_max + lon_tol)

    return time_ok & lat_ok & lon_ok


def _fragment_priority_order(fragments: list[ClimateFragment]) -> list[int]:
    return sorted(
        range(len(fragments)),
        key=lambda idx: (
            fragments[idx].priority,
            fragments[idx].resolution,
            fragments[idx].spatial_area,
            fragments[idx].files[0],
        ),
    )


def _assign_rows_to_climate_fragments_from_arrays(
    fragments: list[ClimateFragment],
    acq_dates: np.ndarray,
    start_dates: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    n_rows = acq_dates.size
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)
    if not fragments:
        return np.full(n_rows, -1, dtype=np.int64)

    assignments = np.full(n_rows, -1, dtype=np.int64)
    for fragment_idx in _fragment_priority_order(fragments):
        unassigned = assignments < 0
        if not unassigned.any():
            break
        coverage = _fragment_coverage_mask(
            fragments[fragment_idx],
            acq_dates,
            start_dates,
            lats,
            lons,
        )
        assignments[unassigned & coverage] = fragment_idx

    return assignments


def assign_rows_to_climate_fragments(
    fragments: list[ClimateFragment],
    target_df: pl.DataFrame,
    n_days: int,
) -> np.ndarray:
    """Assign every target row to the first fragment covering its full window."""

    acq_dates, start_dates, lats, lons = _target_arrays_for_fragment_routing(target_df, n_days)
    return _assign_rows_to_climate_fragments_from_arrays(fragments, acq_dates, start_dates, lats, lons)


def check_fragmented_dataset_bounds(
    fragments: list[ClimateFragment],
    target_df: pl.DataFrame,
    n_days: int,
    assignments: np.ndarray | None = None,
) -> dict:
    """Row-level coverage check for a union of rectangular fragments."""

    if not isinstance(target_df, pl.DataFrame):
        target_df = pl.from_pandas(target_df) if isinstance(target_df, pd.DataFrame) else pl.DataFrame(target_df)

    if target_df.height == 0:
        return {"sufficient": True, "assignments": np.empty(0, dtype=np.int64), "details": {}}

    acq_dates, start_dates, lats, lons = _target_arrays_for_fragment_routing(target_df, n_days)

    if assignments is None:
        assignments = _assign_rows_to_climate_fragments_from_arrays(
            fragments,
            acq_dates,
            start_dates,
            lats,
            lons,
        )

    covered = assignments >= 0
    missing_count = int((~covered).sum())
    fragment_counts = {
        idx: int((assignments == idx).sum())
        for idx in range(len(fragments))
        if int((assignments == idx).sum()) > 0
    }

    required_time_min = start_dates.min().astype("datetime64[ns]")
    required_time_max = acq_dates.max().astype("datetime64[ns]")

    if fragments:
        dataset_time_min = min(np.datetime64(fragment.time_start, "ns") for fragment in fragments)
        dataset_time_max = max(np.datetime64(fragment.time_end, "ns") for fragment in fragments)
        dataset_lat_min = min(fragment.lat_min for fragment in fragments)
        dataset_lat_max = max(fragment.lat_max for fragment in fragments)
        dataset_lon_min = min(fragment.lon_min for fragment in fragments)
        dataset_lon_max = max(fragment.lon_max for fragment in fragments)
    else:
        dataset_time_min = dataset_time_max = None
        dataset_lat_min = dataset_lat_max = None
        dataset_lon_min = dataset_lon_max = None

    missing_examples = []
    if missing_count:
        indexed_target = (
            target_df.with_row_index("__row_nr")
            if hasattr(target_df, "with_row_index")
            else target_df.with_row_count("__row_nr")
        )
        missing_examples = (
            indexed_target
            .filter(pl.Series("__missing", ~covered))
            .select(["__row_nr", "acq_date", "lat_rounded", "lon_rounded"])
            .head(10)
            .to_dicts()
        )

    return {
        "sufficient": missing_count == 0,
        "assignments": assignments,
        "details": {
            "fragment_count": len(fragments),
            "covered_rows": int(covered.sum()),
            "missing_rows": missing_count,
            "missing_fraction": float(missing_count / max(1, target_df.height)),
            "fragment_row_counts": fragment_counts,
            "missing_examples": missing_examples,
            "dataset_union": {
                "time": (dataset_time_min, dataset_time_max),
                "latitude": (dataset_lat_min, dataset_lat_max),
                "longitude": (dataset_lon_min, dataset_lon_max),
            },
            "required": {
                "time": (required_time_min, required_time_max),
                "latitude": (float(np.nanmin(lats)), float(np.nanmax(lats))),
                "longitude": (float(np.nanmin(lons)), float(np.nanmax(lons))),
            },
        },
    }


def print_fragmented_bounds_check(variable_name: str, result: dict, fragments: list[ClimateFragment]) -> None:
    status = "✅ SUFFICIENT" if result["sufficient"] else "❌ INSUFFICIENT"
    details = result.get("details", {})
    print("\n" + "=" * 60)
    print(f" FRAGMENTED DATASET COVERAGE: {variable_name} - {status}")
    print("=" * 60)
    print(f"  Fragments: {details.get('fragment_count', 0)}")
    print(f"  Covered rows: {details.get('covered_rows', 0)}")
    print(f"  Missing rows: {details.get('missing_rows', 0)}")
    if details.get("missing_rows", 0):
        print(f"  Missing fraction: {details.get('missing_fraction', 0.0):.4%}")

    union = details.get("dataset_union", {})
    required = details.get("required", {})
    if union and required:
        time_union = union.get("time", (None, None))
        time_required = required.get("time", (None, None))
        print(
            "  Union time: "
            f"{pd.to_datetime(time_union[0]).date() if time_union[0] is not None else 'N/A'} "
            f"to {pd.to_datetime(time_union[1]).date() if time_union[1] is not None else 'N/A'}"
        )
        print(
            "  Required time: "
            f"{pd.to_datetime(time_required[0]).date()} to {pd.to_datetime(time_required[1]).date()}"
        )
        print(f"  Union latitude bounds: {union.get('latitude')}")
        print(f"  Required latitude bounds: {required.get('latitude')}")
        print(f"  Union longitude bounds: {union.get('longitude')}")
        print(f"  Required longitude bounds: {required.get('longitude')}")

    row_counts = details.get("fragment_row_counts", {})
    for idx, count in sorted(row_counts.items()):
        fragment = fragments[idx]
        print(
            f"  Fragment {idx}: {count:,} rows, {fragment.file_count} file(s), "
            f"lat {fragment.lat_min:.2f}..{fragment.lat_max:.2f}, "
            f"lon {fragment.lon_min:.2f}..{fragment.lon_max:.2f}, "
            f"time {pd.to_datetime(fragment.time_start).date()}..{pd.to_datetime(fragment.time_end).date()}"
        )

    if details.get("missing_examples"):
        print("  Missing examples:")
        for example in details["missing_examples"]:
            print(f"    {example}")
    print("=" * 60)

import logging  
from textwrap import indent


def _nearest_coordinate_indices(coordinate_values: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Vectorised nearest-neighbour lookup for monotonic climate coordinates."""

    values = np.asarray(coordinate_values, dtype=np.float64)
    requested = np.asarray(requested, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Coordinate values must be a non-empty one-dimensional array.")
    if requested.size == 0:
        return np.empty(0, dtype=np.int64)
    if values.size == 1:
        return np.zeros(requested.size, dtype=np.int64)

    diffs = np.diff(values)
    if np.all(diffs >= 0):
        sorted_values = values
        reverse = False
    elif np.all(diffs <= 0):
        sorted_values = values[::-1]
        reverse = True
    else:
        return np.fromiter(
            (int(np.abs(values - val).argmin()) for val in requested),
            dtype=np.int64,
            count=requested.size,
        )

    idx = np.searchsorted(sorted_values, requested)
    idx = np.clip(idx, 0, sorted_values.size - 1)
    prev_idx = np.clip(idx - 1, 0, sorted_values.size - 1)

    prev_dist = np.abs(sorted_values[prev_idx] - requested)
    curr_dist = np.abs(sorted_values[idx] - requested)
    use_prev = (prev_dist < curr_dist) & (idx > 0)
    idx = np.where(use_prev, prev_idx, idx)

    if reverse:
        idx = sorted_values.size - 1 - idx

    return idx.astype(np.int64, copy=False)


def _chunk_size_for_dim(data_array: xr.DataArray, dim_name: str, fallback_size: int) -> int:
    chunksizes = getattr(data_array, "chunksizes", None)
    if chunksizes and dim_name in chunksizes and chunksizes[dim_name]:
        return max(1, int(chunksizes[dim_name][0]))
    return max(1, int(fallback_size))


def _location_processing_order(
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    lat_chunk_size: int,
    lon_chunk_size: int,
) -> np.ndarray:
    """Sort locations so each batch tends to hit the same backing array chunks."""

    lat_bins = lat_indices // max(1, lat_chunk_size)
    lon_bins = lon_indices // max(1, lon_chunk_size)
    return np.lexsort((lon_indices, lat_indices, lon_bins, lat_bins))


def _iter_location_chunk_batches(
    location_order: np.ndarray,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    lat_chunk_size: int,
    lon_chunk_size: int,
    location_batch_size: int,
):
    """Yield location ids grouped by backing spatial chunk."""

    pos = 0
    n_locations = location_order.size
    while pos < n_locations:
        first_loc = location_order[pos]
        lat_bin = lat_indices[first_loc] // max(1, lat_chunk_size)
        lon_bin = lon_indices[first_loc] // max(1, lon_chunk_size)
        end = pos + 1
        while end < n_locations:
            loc = location_order[end]
            if (
                lat_indices[loc] // max(1, lat_chunk_size) != lat_bin
                or lon_indices[loc] // max(1, lon_chunk_size) != lon_bin
            ):
                break
            end += 1

        chunk_locations = location_order[pos:end]
        if location_batch_size > 0 and chunk_locations.size > location_batch_size:
            for batch_start in range(0, chunk_locations.size, location_batch_size):
                yield chunk_locations[batch_start:batch_start + location_batch_size], int(lat_bin), int(lon_bin)
        else:
            yield chunk_locations, int(lat_bin), int(lon_bin)
        pos = end


def _slab_bounds_for_locations(
    batch_locations: np.ndarray,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    lat_bin: int,
    lon_bin: int,
    lat_chunk_size: int,
    lon_chunk_size: int,
    lat_size: int,
    lon_size: int,
    max_spatial_cells: int,
) -> tuple[int, int, int, int]:
    chunk_lat_start = lat_bin * max(1, lat_chunk_size)
    chunk_lon_start = lon_bin * max(1, lon_chunk_size)
    chunk_lat_stop = min(chunk_lat_start + max(1, lat_chunk_size), lat_size)
    chunk_lon_stop = min(chunk_lon_start + max(1, lon_chunk_size), lon_size)
    chunk_cells = (chunk_lat_stop - chunk_lat_start) * (chunk_lon_stop - chunk_lon_start)

    if chunk_cells <= max_spatial_cells:
        return chunk_lat_start, chunk_lat_stop, chunk_lon_start, chunk_lon_stop

    lat_start = int(lat_indices[batch_locations].min())
    lat_stop = int(lat_indices[batch_locations].max()) + 1
    lon_start = int(lon_indices[batch_locations].min())
    lon_stop = int(lon_indices[batch_locations].max()) + 1
    return lat_start, lat_stop, lon_start, lon_stop


def _rows_grouped_by_location(row_to_location: np.ndarray, n_locations: int) -> tuple[np.ndarray, np.ndarray]:
    row_order = np.argsort(row_to_location, kind="stable")
    counts = np.bincount(row_to_location, minlength=n_locations)
    offsets = np.empty(n_locations + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return row_order, offsets


def _rows_for_location_batch(
    batch_locations: np.ndarray,
    row_order: np.ndarray,
    location_offsets: np.ndarray,
) -> np.ndarray:
    parts = [
        row_order[location_offsets[location_id]:location_offsets[location_id + 1]]
        for location_id in batch_locations
        if location_offsets[location_id] < location_offsets[location_id + 1]
    ]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _iter_temporal_row_blocks(
    row_indices: np.ndarray,
    acq_dates: np.ndarray,
    max_time_span_days: int,
):
    """Yield rows sorted into bounded acquisition-date spans."""

    if row_indices.size == 0:
        return

    sorted_rows = row_indices[np.argsort(acq_dates[row_indices], kind="stable")]
    if max_time_span_days <= 0:
        yield sorted_rows
        return

    block_start = 0
    block_start_date = acq_dates[sorted_rows[0]]
    for pos in range(1, sorted_rows.size):
        span_days = (acq_dates[sorted_rows[pos]] - block_start_date).astype("timedelta64[D]").astype(int)
        if span_days > max_time_span_days:
            yield sorted_rows[block_start:pos]
            block_start = pos
            block_start_date = acq_dates[sorted_rows[pos]]

    yield sorted_rows[block_start:]


def _compute_daily_slab(
    var_data: xr.DataArray,
    start_left: int,
    end_right: int,
    batch_start_day: np.datetime64,
    batch_end_day: np.datetime64,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
    cache_path: Path | None,
) -> np.ndarray:
    time_rng = pd.date_range(
        start=pd.Timestamp(batch_start_day),
        end=pd.Timestamp(batch_end_day),
        freq="D",
    )
    expected_shape = (len(time_rng), lat_stop - lat_start, lon_stop - lon_start)

    if cache_path is not None:
        cached = _load_block_cache(cache_path, expected_shape)
        if cached is not None:
            return cached

    ts = var_data.isel(valid_time=slice(start_left, end_right))
    if ts.sizes.get("valid_time", 0) == 0:
        return np.full(expected_shape, np.nan, dtype=np.float32)

    vt_vals_slice = ts["valid_time"].values
    if vt_vals_slice.size != np.unique(vt_vals_slice).size:
        _, uniq_idx = np.unique(vt_vals_slice, return_index=True)
        ts = ts.isel(valid_time=np.sort(uniq_idx))

    ts = ts.sortby("valid_time")
    slab = ts.isel(
        latitude=slice(lat_start, lat_stop),
        longitude=slice(lon_start, lon_stop),
    ).transpose("valid_time", "latitude", "longitude")

    daily = (
        slab.interp(
            valid_time=("valid_time", time_rng),
            kwargs={"fill_value": None, "bounds_error": False},
        )
        .compute()
        .values
    )

    if cache_path is not None:
        _save_block_cache(cache_path, daily)

    return daily



def extract_climate_timeseries(
    ds: xr.Dataset,
    variable: str,
    target_df_pl: pl.DataFrame,
    n_days: int = 120,
    location_batch_size: int | None = None,
    max_time_span_days: int | None = None,
    fill_row_batch_size: int | None = None,
    block_cache_dir: str | None = None,
    block_cache_source_token: str | None = None,
    debug: bool = False,          # ← flip to True to get detailed output
) -> np.ndarray:
    """
    Return an (n_rows × n_days) NumPy array with raw values of `variable`.
    Keeps all rows from `target_df_pl`, handling duplicates in space, time,
    and acquisition date.  Missing windows are returned as NaN.

    Set `debug=True` to dump detailed stats for each step.
    """

    # ───────────── set up logger (only when debug=True) ──────────────
    if debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        log = logging.getLogger("extract_climate_timeseries")
        log.info("VARIABLE      : %s  (units: %s)",
                 variable, ds[variable].attrs.get("units", "unknown"))
        log.info("DATASET time range : %s … %s",
                 pd.to_datetime(ds.valid_time.min().values),
                 pd.to_datetime(ds.valid_time.max().values))
        log.info("DATASET lat range  : %.3f … %.3f",
                 float(ds.latitude.min()), float(ds.latitude.max()))
        log.info("DATASET lon range  : %.3f … %.3f",
                 float(ds.longitude.min()), float(ds.longitude.max()))
    else:
        # dummy logger that does nothing
        log = logging.getLogger("dummy")
        log.addHandler(logging.NullHandler())

    # ───────────── guard against empty target table ─────────────
    n_rows = target_df_pl.height
    if n_rows == 0:
        raise ValueError("Target DataFrame is empty. Cannot extract timeseries.")

    # ensure time axis monotonic so slice operations stay cheap
    vt_full = ds["valid_time"]
    if vt_full.size == 0:
        raise ValueError("Dataset valid_time coordinate is empty.")

    vt_vals = vt_full.values
    if vt_vals.ndim != 1:
        raise ValueError("valid_time coordinate must be one-dimensional.")

    vt_numeric = vt_vals.astype("datetime64[ns]")
    if np.any(np.diff(vt_numeric) < np.timedelta64(0, "ns")):
        ds = ds.sortby("valid_time")
        vt_full = ds["valid_time"]
        vt_numeric = vt_full.values.astype("datetime64[ns]")

    # Cache latitude/longitude lookup globally and read by location batches. This
    # avoids re-reading the same overlapping windows for every acquisition date.
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Dataset must contain 'latitude' and 'longitude' coordinates.")

    lat_vals = np.asarray(ds["latitude"].values, dtype=float)
    lon_vals = np.asarray(ds["longitude"].values, dtype=float)

    # choose a dtype that can hold NaN
    out_dtype = (ds[variable].dtype
                 if np.issubdtype(ds[variable].dtype, np.floating)
                 else np.float32)
    out = np.full((n_rows, n_days), np.nan, dtype=out_dtype)

    target_pd = target_df_pl.select(["acq_date", "lat_rounded", "lon_rounded"]).to_pandas()
    acq_dates = pd.to_datetime(target_pd["acq_date"]).to_numpy(dtype="datetime64[D]")
    start_dates = acq_dates - np.timedelta64(n_days - 1, "D")

    xy = target_pd[["lat_rounded", "lon_rounded"]].to_numpy(dtype=np.float64)
    unique_xy, row_to_location = np.unique(xy, axis=0, return_inverse=True)
    n_locations = unique_xy.shape[0]

    lat_indices = _nearest_coordinate_indices(lat_vals, unique_xy[:, 0])
    lon_indices = _nearest_coordinate_indices(lon_vals, unique_xy[:, 1])

    var_data = ds[variable]
    lat_chunk_size = _chunk_size_for_dim(var_data, "latitude", len(lat_vals))
    lon_chunk_size = _chunk_size_for_dim(var_data, "longitude", len(lon_vals))
    location_order = _location_processing_order(lat_indices, lon_indices, lat_chunk_size, lon_chunk_size)
    row_order, location_offsets = _rows_grouped_by_location(row_to_location, n_locations)

    if location_batch_size is None:
        location_batch_size = int(os.environ.get("CLIMATE_LOCATION_BATCH_SIZE", "0"))
    location_batch_size = max(0, int(location_batch_size))

    if max_time_span_days is None:
        max_time_span_days = int(os.environ.get("CLIMATE_BATCH_MAX_SPAN_DAYS", "180"))
    max_time_span_days = max(0, int(max_time_span_days))

    if fill_row_batch_size is None:
        fill_row_batch_size = int(os.environ.get("CLIMATE_FILL_ROW_BATCH_SIZE", "100000"))
    fill_row_batch_size = max(1, int(fill_row_batch_size))

    max_slab_spatial_cells = int(os.environ.get("CLIMATE_MAX_SLAB_SPATIAL_CELLS", "250000"))
    max_slab_spatial_cells = max(1, max_slab_spatial_cells)

    if debug:
        log.info(
            "unique locations: %d, location batch size: %d, max time span days: %d, fill row batch size: %d",
            n_locations,
            location_batch_size,
            max_time_span_days,
            fill_row_batch_size,
        )

    progress_bar = None
    if not debug and n_locations > 1:
        progress_bar = tqdm(
            total=n_locations,
            desc=f"{variable} climate chunks",
            unit="loc",
            mininterval=0.5,
            leave=False,
        )

    day_offsets = np.arange(n_days, dtype=np.int64)

    for batch_locations, lat_bin, lon_bin in _iter_location_chunk_batches(
        location_order,
        lat_indices,
        lon_indices,
        lat_chunk_size,
        lon_chunk_size,
        location_batch_size,
    ):
        row_indices = _rows_for_location_batch(batch_locations, row_order, location_offsets)
        if row_indices.size == 0:
            if progress_bar:
                progress_bar.update(batch_locations.size)
            continue

        lat_start, lat_stop, lon_start, lon_stop = _slab_bounds_for_locations(
            batch_locations,
            lat_indices,
            lon_indices,
            lat_bin,
            lon_bin,
            lat_chunk_size,
            lon_chunk_size,
            len(lat_vals),
            len(lon_vals),
            max_slab_spatial_cells,
        )
        local_lat_by_location = lat_indices - lat_start
        local_lon_by_location = lon_indices - lon_start

        for time_block_rows in _iter_temporal_row_blocks(row_indices, acq_dates, max_time_span_days):
            block_locations = np.unique(row_to_location[time_block_rows])
            batch_start_day = start_dates[time_block_rows].min()
            batch_end_day = acq_dates[time_block_rows].max()

            if debug:
                log.info(
                    "chunk lat=%d lon=%d (%d loc, %d rows, %s … %s)",
                    lat_bin,
                    lon_bin,
                    block_locations.size,
                    time_block_rows.size,
                    pd.Timestamp(batch_start_day).date(),
                    pd.Timestamp(batch_end_day).date(),
                )

            left = int(np.searchsorted(vt_numeric, batch_start_day.astype("datetime64[ns]"), side="left"))
            right = int(np.searchsorted(vt_numeric, batch_end_day.astype("datetime64[ns]"), side="right"))

            if right - left <= 0:
                if debug:
                    log.warning("  • batch window empty - skipping (no overlapping timestamps)")
                continue

            cache_path = None
            if block_cache_dir and block_cache_source_token:
                cache_path = _climate_block_cache_path(
                    block_cache_dir,
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

            if debug:
                log.info(
                    indent(
                        f"""daily: min={np.nanmin(daily):.2f}  max={np.nanmax(daily):.2f}"""
                        f"""  nan%={np.isnan(daily).mean()*100:.1f}%""",
                        prefix="    ",
                    )
                )

            local_lats = local_lat_by_location[row_to_location[time_block_rows]]
            local_lons = local_lon_by_location[row_to_location[time_block_rows]]
            row_offsets_days = (acq_dates[time_block_rows] - batch_start_day).astype("timedelta64[D]").astype(np.int64)

            for fill_start in range(0, time_block_rows.size, fill_row_batch_size):
                fill_stop = min(time_block_rows.size, fill_start + fill_row_batch_size)
                fill_rows = time_block_rows[fill_start:fill_stop]
                fill_lats = local_lats[fill_start:fill_stop]
                fill_lons = local_lons[fill_start:fill_stop]
                fill_offsets = row_offsets_days[fill_start:fill_stop]
                window_indices = fill_offsets[:, None] - (n_days - 1) + day_offsets[None, :]
                out[fill_rows] = daily[window_indices, fill_lats[:, None], fill_lons[:, None]]

        if progress_bar:
            progress_bar.update(batch_locations.size)

    if progress_bar:
        progress_bar.close()

    if debug:
        log.info("=== FINAL MATRIX ===")
        log.info("shape : %s", out.shape)
        log.info("min/max : %.2f / %.2f", np.nanmin(out), np.nanmax(out))
        log.info("nan%%   : %.1f%%", np.isnan(out).mean() * 100)

    return out


def _subset_for_fragment(target_df_pl: pl.DataFrame, row_indices: np.ndarray, fragment: ClimateFragment) -> pl.DataFrame:
    subset = target_df_pl[row_indices.tolist()]
    lons = subset["lon_rounded"].to_numpy().astype(np.float64)
    normalized_lons = _normalize_longitudes_for_fragment(lons, fragment)
    if not np.allclose(lons, normalized_lons, equal_nan=True):
        subset = subset.with_columns(pl.Series("lon_rounded", normalized_lons))
    return subset


def _time_range_for_subset(target_subset: pl.DataFrame, n_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    acq_dates = pd.to_datetime(target_subset["acq_date"].to_numpy())
    start_dates = acq_dates - pd.DateOffset(days=max(n_days - 1, 0))
    return start_dates.min(), acq_dates.max()


def extract_climate_timeseries_fragmented(
    fragments: list[ClimateFragment],
    variable: str,
    target_df_pl: pl.DataFrame,
    n_days: int = 120,
    assignments: np.ndarray | None = None,
    location_batch_size: int | None = None,
    max_time_span_days: int | None = None,
    fill_row_batch_size: int | None = None,
    block_cache_dir: str | None = None,
    block_cache_source_token: str | None = None,
    persist_dataset: bool = False,
) -> np.ndarray:
    """Extract climate windows from a union of rectangular fragments."""

    n_rows = target_df_pl.height
    out = np.full((n_rows, n_days), np.nan, dtype=np.float32)
    if n_rows == 0:
        return out

    if assignments is None:
        assignments = assign_rows_to_climate_fragments(fragments, target_df_pl, n_days)

    for fragment_idx in _fragment_priority_order(fragments):
        row_indices = np.flatnonzero(assignments == fragment_idx)
        if row_indices.size == 0:
            continue

        fragment = fragments[fragment_idx]
        target_subset = _subset_for_fragment(target_df_pl, row_indices, fragment)
        subset_coords = target_subset.select(["acq_date", "lat_rounded", "lon_rounded"])
        time_range = _time_range_for_subset(target_subset, n_days)
        lat_range = (target_subset["lat_rounded"].min(), target_subset["lat_rounded"].max())
        lon_range = (target_subset["lon_rounded"].min(), target_subset["lon_rounded"].max())

        print(
            f"Extracting {variable} from fragment {fragment_idx} "
            f"({row_indices.size:,} rows, {fragment.file_count} file(s))"
        )
        ds = open_climate_fragment(
            fragment,
            time_range=time_range,
            lat_range=lat_range,
            lon_range=lon_range,
        )
        try:
            if persist_dataset:
                dask_client = _get_dask_client()
                persist_start = time_lib.time()
                ds = ds.persist()
                client_desc = "distributed" if dask_client else "local"
                print(
                    f"Fragment dataset persisted in memory "
                    f"({time_lib.time() - persist_start:.2f}s) using {client_desc} client."
                )
            else:
                print("Fragment dataset-wide persist disabled; streaming location batches from disk.")

            fragment_block_token = None
            if block_cache_dir and block_cache_source_token:
                fragment_block_token = _climate_block_source_token(
                    f"{block_cache_source_token}|fragment:{fragment_idx}|{fragment.source_token_payload()}",
                    variable,
                    ds,
                )

            subset_matrix = extract_climate_timeseries(
                ds,
                variable,
                subset_coords,
                n_days=n_days,
                location_batch_size=location_batch_size,
                max_time_span_days=max_time_span_days,
                fill_row_batch_size=fill_row_batch_size,
                block_cache_dir=block_cache_dir,
                block_cache_source_token=fragment_block_token,
            )
            out[row_indices] = subset_matrix.astype(out.dtype, copy=False)
        finally:
            ds.close()

    return out


def prepare_data(
    climate_data_dir: str, 
    climate_variables: list[str], 
    target_df: pd.DataFrame | pl.DataFrame, 
    n_days: int = 120, # Lookback for raw data slices
    prep_climate: bool = True, 
    test_mode: bool = False,
    max_length_features: int = 120, 
    cache_dir: str = 'data/saved_features/climate_features_cache',
    lags_features: list = None,
    windows_features: list = None,
    spans_features: list = None,
    trend_window_features: list = None,
    features_to_include_config: dict = None,
    use_cached_files: bool = False,
    return_features_df: bool = False,
    location_batch_size: int | None = None,
    max_time_span_days: int | None = None,
    persist_dataset: bool = False,
    strict_climate_bounds: bool = True,
):
    """
    Loads climate data, extracts time series, computes features, and merges them.

    Returns
    -------
    - If ``return_features_df`` is ``False`` (default): ``np.ndarray`` with stacked
      time-series for every variable.
    - If ``return_features_df`` is ``True``: tuple of
      (``pandas.DataFrame`` with climate features appended, ordered list of
      climate feature column names, stacked time-series matrix).

    Parameters
    ----------
    single_thread_debug : bool
        If True, disables Dask parallelism during feature extraction and executes
        computations sequentially. This is useful for debugging, since any
        exceptions raised inside the feature-extraction logic will surface
        immediately in the main thread instead of being wrapped by Dask.
    """
    if not prep_climate:
        if isinstance(target_df, pl.DataFrame):
            target_pd = target_df.to_pandas()
        else:
            target_pd = target_df.copy()
        if return_features_df:
            empty_matrix = np.empty((len(target_pd), 0), dtype=np.float32)
            return target_pd.reset_index(drop=True), [], empty_matrix
        return target_pd

    # Standardize target_df to Polars DataFrame and ensure 'id' column
    if isinstance(target_df, pd.DataFrame):
        # If 'id' not present or not unique, reset and create one
        if 'id' not in target_df.columns or not target_df['id'].is_unique:
            target_df_pl = pl.from_pandas(target_df.reset_index(drop=True).reset_index().rename(columns={'index': 'id'}))
        else:
            target_df_pl = pl.from_pandas(target_df)
    elif isinstance(target_df, pl.DataFrame):
        if 'id' not in target_df.columns or not target_df['id'].is_unique():
            target_df_pl = target_df.drop("id") if "id" in target_df.columns else target_df
            target_df_pl = target_df_pl.with_row_count('id')
        else:
            target_df_pl = target_df
    else:
        raise TypeError("target_df must be a Pandas or Polars DataFrame.")


    print("\n--- Generating features and saving to cache ---")
    start_time_prep = time_lib.time()
    
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory '{cache_dir}' will be used for saving intermediate results.")
    
    print(f'Input target DataFrame shape for feature generation: {target_df_pl.shape}')
    if target_df_pl.height == 0:
        print("Target DataFrame is empty. Returning original (or empty if no prep_climate).")
        empty_pd = target_df_pl.to_pandas()
        if return_features_df:
            empty_matrix = np.empty((0, 0), dtype=np.float32)
            return empty_pd, [], empty_matrix
        return empty_pd

    if target_df_pl.height > 0 :
        end_of_slices_pd_series = pd.to_datetime(target_df_pl['acq_date'].to_numpy())
        start_of_slices_pd_series = (end_of_slices_pd_series - pd.DateOffset(days=n_days))
        
        time_range_load = (start_of_slices_pd_series.min(), end_of_slices_pd_series.max())
        lat_range_load = (target_df_pl["lat_rounded"].min(), target_df_pl["lat_rounded"].max())
        lon_range_load = (target_df_pl["lon_rounded"].min(), target_df_pl["lon_rounded"].max())
    else:
        print("Warning: Target DataFrame is empty, cannot determine data ranges. Climate features will be empty.")
        time_range_load, lat_range_load, lon_range_load = None, None, None

    final_df_pl = target_df_pl 
    ts_matrix_var_list = []
    coords_pl = target_df_pl.select(["acq_date", "lat_rounded", "lon_rounded"])

    total_variables = len(climate_variables)
    elapsed_per_variable: list[float] = []
    use_matrix_cache = use_cached_files and os.environ.get("CLIMATE_MATRIX_CACHE", "1") != "0"
    use_block_cache = use_cached_files and os.environ.get("CLIMATE_BLOCK_CACHE", "1") != "0"

    for idx, variable in enumerate(climate_variables, start=1):
        print(f"\n[{idx}/{total_variables}] Processing variable: {variable}")
        load_start = time_lib.time()

        try:
            fragments = discover_climate_fragments(climate_data_dir, variable)
        except (FileNotFoundError, ValueError, IOError) as e:
            raise ValueError(f"Error discovering climate fragments for {variable}: {e}") from e

        fragment_source_token = climate_fragments_source_token(fragments)
        use_fragmented_loader = len(fragments) > 1
        if use_fragmented_loader:
            print(f"Discovered {len(fragments)} spatial fragments for {variable}; using fragmented loading.")
        else:
            print(f"Discovered {len(fragments)} spatial fragment for {variable}; using standard loading.")

        matrix_cache_path = (
            _climate_matrix_cache_path(
                cache_dir,
                climate_data_dir,
                variable,
                coords_pl,
                n_days,
                source_token=fragment_source_token,
            )
            if use_matrix_cache
            else None
        )

        if matrix_cache_path is not None and matrix_cache_path.exists():
            cache_start = time_lib.time()
            ts_matrix_var = np.load(matrix_cache_path, allow_pickle=False)
            if ts_matrix_var.shape == (target_df_pl.height, n_days):
                print(
                    f"Loaded cached raw climate matrix for {variable} from "
                    f"{matrix_cache_path} in {time_lib.time() - cache_start:.2f}s"
                )
                ts_matrix_var_list.append(ts_matrix_var)
                elapsed = time_lib.time() - load_start
                elapsed_per_variable.append(elapsed)
                remaining = total_variables - idx
                if remaining > 0:
                    avg_time = sum(elapsed_per_variable) / len(elapsed_per_variable)
                    eta_minutes = (avg_time * remaining) / 60
                    print(f"Completed {variable} in {elapsed:.2f}s. Estimated remaining: {eta_minutes:.2f} minutes for {remaining} variables.")
                else:
                    print(f"Completed {variable} in {elapsed:.2f}s.")
                continue

            print(
                f"Ignoring stale cache for {variable}: expected "
                f"{(target_df_pl.height, n_days)}, got {ts_matrix_var.shape}"
            )

        if use_fragmented_loader:
            bounds_check_result = check_fragmented_dataset_bounds(
                fragments,
                target_df_pl,
                n_days=n_days,
            )
            print_fragmented_bounds_check(variable, bounds_check_result, fragments)
            if not bounds_check_result["sufficient"]:
                message = (
                    f"Fragmented dataset for {variable} does not fully cover every "
                    "target row and lookback window. Climate features would contain "
                    "missing or partial windows."
                )
                if strict_climate_bounds:
                    raise ValueError(message)
                print(f"Warning: {message}")

            raw_start = time_lib.time()
            ts_matrix_var = extract_climate_timeseries_fragmented(
                fragments,
                variable,
                target_df_pl,
                n_days=n_days,
                assignments=bounds_check_result["assignments"],
                location_batch_size=location_batch_size,
                max_time_span_days=max_time_span_days,
                block_cache_dir=cache_dir if use_block_cache else None,
                block_cache_source_token=fragment_source_token if use_block_cache else None,
                persist_dataset=persist_dataset,
            )

            print(f"Fragmented raw-series extraction time for {variable}: {time_lib.time() - raw_start:.2f}s")
            print(f"variable {variable} time series matrix shape: {ts_matrix_var.shape}")
            ts_matrix_var_list.append(ts_matrix_var)
            if matrix_cache_path is not None:
                cache_write_start = time_lib.time()
                _save_matrix_cache(matrix_cache_path, ts_matrix_var)
                print(f"Saved raw climate matrix cache for {variable} in {time_lib.time() - cache_write_start:.2f}s")

            elapsed = time_lib.time() - load_start
            elapsed_per_variable.append(elapsed)
            remaining = total_variables - idx
            if remaining > 0:
                avg_time = sum(elapsed_per_variable) / len(elapsed_per_variable)
                eta_minutes = (avg_time * remaining) / 60
                print(f"Completed {variable} in {elapsed:.2f}s. Estimated remaining: {eta_minutes:.2f} minutes for {remaining} variables.")
            else:
                print(f"Completed {variable} in {elapsed:.2f}s.")
            continue

        try:
            ds = load_climate_variable_mf(
                climate_data_dir, variable, 
                time_range=time_range_load, 
                lat_range=lat_range_load, 
                lon_range=lon_range_load,
                test_mode=test_mode
            )
        except (FileNotFoundError, ValueError, IOError) as e:
            raise ValueError(f"Error loading data for variable {variable}: {e}")
            
        print(f"Initial dataset load/reference for {variable}: {time_lib.time() - load_start:.2f}s")
        block_cache_source_token = (
            _climate_block_source_token(f"{climate_data_dir}|{fragment_source_token}", variable, ds)
            if use_block_cache
            else None
        )

        if persist_dataset:
            dask_client = _get_dask_client()
            persist_start = time_lib.time()
            ds = ds.persist()
            persist_time = time_lib.time() - persist_start
            client_desc = "distributed" if dask_client else "local"
            print(
                f"Dataset persisted in distributed memory ({persist_time:.2f}s) using {client_desc} client."
            )
        else:
            print("Dataset-wide persist disabled; streaming location batches from disk.")

        # Perform bounds check (optional, but good for diagnostics)
        bounds_check_result = check_dataset_bounds(ds, target_df_pl, n_days=n_days)
        print_dataset_bounds_check(variable, bounds_check_result)
        if not bounds_check_result['sufficient']:
            message = (
                f"Dataset for {variable} does not fully cover required target "
                "time/latitude/longitude ranges. Climate features would contain "
                "missing or partial windows."
            )
            if strict_climate_bounds:
                raise ValueError(message)
            print(f"Warning: {message}")

        raw_start = time_lib.time()

        # 1. get the (row × n_days) matrix
        ts_matrix_var = extract_climate_timeseries(
            ds, variable,
            coords_pl,
            n_days=n_days,
            location_batch_size=location_batch_size,
            max_time_span_days=max_time_span_days,
            block_cache_dir=cache_dir if use_block_cache else None,
            block_cache_source_token=block_cache_source_token,
        )

        print(f"Raw‐series extraction time for {variable}: {time_lib.time() - raw_start:.2f}s")
        print(f"variable {variable} time series matrix shape: {ts_matrix_var.shape}")
        ts_matrix_var_list.append(ts_matrix_var)
        if matrix_cache_path is not None:
            cache_write_start = time_lib.time()
            _save_matrix_cache(matrix_cache_path, ts_matrix_var)
            print(f"Saved raw climate matrix cache for {variable} in {time_lib.time() - cache_write_start:.2f}s")
        ds.close()

        elapsed = time_lib.time() - load_start
        elapsed_per_variable.append(elapsed)
        remaining = total_variables - idx
        if remaining > 0:
            avg_time = sum(elapsed_per_variable) / len(elapsed_per_variable)
            eta_minutes = (avg_time * remaining) / 60
            print(f"Completed {variable} in {elapsed:.2f}s. Estimated remaining: {eta_minutes:.2f} minutes for {remaining} variables.")
        else:
            print(f"Completed {variable} in {elapsed:.2f}s.")


    end_time_prep = time_lib.time()
    total_time_secs = end_time_prep - start_time_prep
    print(f"\nClimate data preparation and feature generation completed in {total_time_secs:.2f} seconds ({total_time_secs/60:.2f} minutes).")

    if ts_matrix_var_list:
        ts_matrix = np.concatenate(ts_matrix_var_list, axis=1)
    else:
        ts_matrix = np.empty((target_df_pl.height, 0), dtype=np.float32)
    print(f"time series matrix shape: {ts_matrix.shape}")

    if not return_features_df:
        return ts_matrix

    df_with_features, climate_feature_columns = _build_climate_feature_dataframe(
        target_df_pl,
        ts_matrix_var_list,
        climate_variables,
        lags_features=lags_features,
        windows_features=windows_features,
        spans_features=spans_features,
        trend_window_features=trend_window_features,
        max_length_features=max_length_features,
        features_to_include_config=features_to_include_config,
    )

    return df_with_features, climate_feature_columns, ts_matrix
