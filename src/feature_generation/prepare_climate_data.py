import pandas as pd
import os, sys
import numpy as np
import xarray as xr
import time as time_lib
import polars as pl
from pandas import to_datetime
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from tqdm import tqdm
from functools import lru_cache
sys.path.append(os.getcwd())
from src.feature_generation.load_climate_data import load_climate_variable_mf


client = None
server_configuration = True # if True, use server(very big RAM needed) configuration, if False, use local configuration
single_thread_debug = False


def _get_dask_client():
    """Initializes the Dask client if it's not already running."""
    global client
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
    start_of_slices = (end_of_slices - pd.DateOffset(days=n_days))
    
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
        # Latitude is simpler as it doesn't wrap.
        # Check for decreasing latitude coordinate order in ds
        if 'latitude' in ds.coords and ds.latitude.size > 1 and ds.latitude.values[0] > ds.latitude.values[-1]:
             # Dataset latitude is decreasing (e.g. 90 to -90)
             if lat_tol is None:
                 lat_sufficient = (ds_lat_max <= required_lat_min) and (ds_lat_min >= required_lat_max)
             else:
                 lat_sufficient = (ds_lat_max <= required_lat_min + lat_tol) and (ds_lat_min >= required_lat_max - lat_tol)
        else: # Dataset latitude is increasing (e.g. -90 to 90) or single point
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

import logging  
from textwrap import indent



def extract_climate_timeseries(
    ds: xr.Dataset,
    variable: str,
    target_df_pl: pl.DataFrame,
    n_days: int = 120,
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

    # cache latitude/longitude lookup so we do not interpolate full grids repeatedly
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Dataset must contain 'latitude' and 'longitude' coordinates.")

    lat_vals = np.asarray(ds["latitude"].values, dtype=float)
    lon_vals = np.asarray(ds["longitude"].values, dtype=float)
    lat_lookup = {round(float(val), 5): idx for idx, val in enumerate(lat_vals)}
    lon_lookup = {round(float(val), 5): idx for idx, val in enumerate(lon_vals)}

    @lru_cache(maxsize=None)
    def _coord_to_idx(lat_val: float, lon_val: float) -> tuple[int, int]:
        lat_q = round(float(lat_val), 5)
        lon_q = round(float(lon_val), 5)

        ilat = lat_lookup.get(lat_q)
        if ilat is None:
            ilat = int(np.abs(lat_vals - float(lat_val)).argmin())
            lat_lookup[lat_q] = ilat

        ilon = lon_lookup.get(lon_q)
        if ilon is None:
            ilon = int(np.abs(lon_vals - float(lon_val)).argmin())
            lon_lookup[lon_q] = ilon
        return ilat, ilon

    # choose a dtype that can hold NaN
    out_dtype = (ds[variable].dtype
                 if np.issubdtype(ds[variable].dtype, np.floating)
                 else np.float32)
    out = np.full((n_rows, n_days), np.nan, dtype=out_dtype)

    # keep original order
    target_df_pl = target_df_pl.with_row_count(name="_row_idx")

    # ───────────── main loop over acquisition dates ─────────────
    n_groups = int(target_df_pl.get_column("acq_date").n_unique())
    progress_bar = None
    if not debug and n_groups > 1:
        progress_bar = tqdm(
            total=n_groups,
            desc=f"{variable} windows",
            unit="window",
            mininterval=0.5,
            leave=False,
        )

    var_data = ds[variable]
    for key, group in target_df_pl.group_by("acq_date", maintain_order=True):
        acq_date = pd.Timestamp(key[0])
        end      = acq_date
        start    = end - pd.Timedelta(days=n_days - 1)
        time_rng = pd.date_range(end=end, periods=n_days, freq="D")

        if debug:
            log.info("--- acq_date %s  (rows %d)", acq_date.date(), group.height)

        # 1 ▸ slice by time (dataset already monotonic)
        start64 = np.datetime64(start)
        end64 = np.datetime64(end)
        left = int(np.searchsorted(vt_numeric, start64, side="left"))
        right = int(np.searchsorted(vt_numeric, end64, side="right"))

        if right - left <= 0:
            if debug:
                log.warning("  • window empty - skipping (no overlapping timestamps)")
            if progress_bar:
                progress_bar.update(1)
            continue

        ts = var_data.isel(valid_time=slice(left, right))

        if ts.sizes.get("valid_time", 0) == 0:
            if debug:
                log.warning("  • window empty - skipping (all NaNs left intact)")
            if progress_bar:
                progress_bar.update(1)
            continue

        # 1b ▸ deduplicate & sort
        vt_vals_slice = ts["valid_time"].values
        if vt_vals_slice.size != np.unique(vt_vals_slice).size:
            _, uniq_idx = np.unique(vt_vals_slice, return_index=True)
            ts = ts.isel(valid_time=np.sort(uniq_idx))
            if debug:
                log.info("  • dropped %d duplicate timestamps",
                         vt_vals_slice.size - uniq_idx.size)

        ts = ts.sortby("valid_time")

        if debug:
            log.info("  • time slice size: %d  (%s … %s)",
                     ts.sizes["valid_time"],
                     pd.to_datetime(ts.valid_time.min().values),
                     pd.to_datetime(ts.valid_time.max().values))

        # 2 ▸ unique spatial lookup
        xy          = np.column_stack([group["lat_rounded"], group["lon_rounded"]])
        uniq_xy, inv = np.unique(xy, axis=0, return_inverse=True)
        idx_pairs = np.array([_coord_to_idx(lat, lon) for lat, lon in uniq_xy], dtype=int)
        lat_idx = xr.DataArray(idx_pairs[:, 0], dims="u_loc")
        lon_idx = xr.DataArray(idx_pairs[:, 1], dims="u_loc")

        grid = ts.isel(latitude=lat_idx, longitude=lon_idx).transpose("u_loc", "valid_time")

        # quick stats before daily re-gridding
        if debug:
            gdat = grid.values
            log.info(
                indent(
                    f"""grid: min={np.nanmin(gdat):.2f}  max={np.nanmax(gdat):.2f}"""
                    f"""  nan%={np.isnan(gdat).mean()*100:.1f}%""",
                    prefix="    ",
                )
            )

        # 3 ▸ daily re-grid
        daily = (
            grid.interp(
                valid_time=("valid_time", time_rng),
                kwargs={"fill_value": None, "bounds_error": False},
            )
            .compute()
            .values[inv]
        )
        if debug:
            log.info(
                indent(
                    f"""daily: min={np.nanmin(daily):.2f}  max={np.nanmax(daily):.2f}"""
                    f"""  nan%={np.isnan(daily).mean()*100:.1f}%""",
                    prefix="    ",
                )
            )

        # 4 ▸ write into output
        out[group["_row_idx"].to_numpy()] = daily

        if progress_bar:
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    if debug:
        log.info("=== FINAL MATRIX ===")
        log.info("shape : %s", out.shape)
        log.info("min/max : %.2f / %.2f", np.nanmin(out), np.nanmax(out))
        log.info("nan%%   : %.1f%%", np.isnan(out).mean() * 100)

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

    for idx, variable in enumerate(climate_variables, start=1):
        print(f"\n[{idx}/{total_variables}] Processing variable: {variable}")
        load_start = time_lib.time()

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

        dask_client = _get_dask_client()
        persist_start = time_lib.time()
        ds = ds.persist()
        persist_time = time_lib.time() - persist_start
        client_desc = "distributed" if dask_client else "local"
        print(
            f"Dataset persisted in distributed memory ({persist_time:.2f}s) using {client_desc} client."
        )

        # Perform bounds check (optional, but good for diagnostics)
        bounds_check_result = check_dataset_bounds(ds, target_df_pl, n_days=n_days)
        print_dataset_bounds_check(variable, bounds_check_result)
        if not bounds_check_result['sufficient']:
                print(f"Warning: Dataset for {variable} may not fully cover required data ranges. Results might be incomplete.")
                # Decide if to continue or raise error. For now, continue with warning.

        raw_start = time_lib.time()

        # 1. get the (row × n_days) matrix
        ts_matrix_var = extract_climate_timeseries(
            ds, variable,
            coords_pl,
            n_days=n_days,
        )

        print(f"Raw‐series extraction time for {variable}: {time_lib.time() - raw_start:.2f}s")
        print(f"variable {variable} time series matrix shape: {ts_matrix_var.shape}")
        ts_matrix_var_list.append(ts_matrix_var)
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
