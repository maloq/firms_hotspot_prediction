import time as time_lib
import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
import yaml
try:
    import dask
    from dask.diagnostics import ProgressBar
except ImportError:
    dask = None
    ProgressBar = None
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_trend_slope(data_array, time_dim_name):
    """
    Vectorised, Dask-friendly computation of the slope of a linear
    regression along `time_dim_name`.
    """
    # 1.  Convert the time coordinate to a numeric axis
    t = data_array[time_dim_name]
    if np.issubdtype(t.dtype, np.datetime64):
        t = xr.DataArray(
            t.dt.year + (t.dt.dayofyear - 1) / 366.0,
            coords=t.coords,
            dims=t.dims,
        )

    # 2.  Centre the time axis (better numerics)
    t = t - t.mean(dim=time_dim_name)

    # 3.  Element-wise slope helper (operates on 1-D vectors)
    def _slope(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return np.nan
        xm = x[mask] - x[mask].mean()
        ym = y[mask] - y[mask].mean()
        return (xm * ym).sum() / (xm * xm).sum()

    # 4.  Apply across the whole cube, fully parallelisable
    return xr.apply_ufunc(
        _slope,
        t,
        data_array,
        input_core_dims=[[time_dim_name], [time_dim_name]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[data_array.dtype],
    )


def calculate_fire_index_features(fire_index_files: List[str], target_resolution: float = 0.1,
                           lat_bounds: Optional[tuple] = None, lon_bounds: Optional[tuple] = None,
                           output_npz_path: Optional[str] = None) -> tuple:
    """
    Loads fire index NetCDF files, calculates monthly statistical features (mean, std, min, max, median)
    and the linear trend of monthly mean, max, and median values across years.
    Interpolates results to a grid with specified resolution.
    
    Args:
        fire_index_files (List[str]): List of paths to fire index NetCDF files (e.g., 
                                    ['fire_index_2000-2010.nc', 'fire_index_2010-2023.nc'])
        target_resolution (float): Resolution in degrees for output grid (default: 0.1)
        lat_bounds (Optional[tuple]): Optional latitude bounds (min, max) for the output grid
        lon_bounds (Optional[tuple]): Optional longitude bounds (min, max) for the output grid
        output_npz_path (Optional[str]): If provided, save results to this NPZ file path
        
    Returns:
        tuple:
            xr.Dataset: Dataset containing interpolated monthly fire index features (dims: month, latitude, longitude)
                        Includes mean, std, min, max, median, and mean_trend, max_trend, median_trend.
            list: List of base feature names (suffixes like _mean, _std, _mean_trend etc. correspond to monthly stats)
    """
    if dask is None or ProgressBar is None:
        raise ImportError("dask is required to calculate fire index features. Install requirements.txt first.")

    print(f"Loading fire index files: {fire_index_files}")
    start_time = time_lib.time()
    
    combined_ds = xr.open_mfdataset(
        fire_index_files,
        combine="by_coords",
        parallel=True,         
        engine="netcdf4",
        chunks={"time": 365, "latitude": -1, "longitude": -1},
        persist=True
    )

    time_dim_name = next((d for d in ["time", "valid_time"] if d in combined_ds.dims), None)
    if not time_dim_name:
        raise ValueError("Could not find a recognizable time dimension ('time' or 'valid_time')")

    # -------- 2.  BASIC METADATA  --------
    original_lats = combined_ds.latitude.values
    original_lons = combined_ds.longitude.values
    print(f"Original latitude range: {original_lats.min()}, {original_lats.max()}")
    print(f"Original longitude range: {original_lons.min()}, {original_lons.max()}")

    # -------- 3.  MAIN FEATURE LOOP  --------
    feature_ds   = {}
    feature_names = []
    var_names    = list(combined_ds.data_vars)
    time_dim     = time_dim_name

    print(f"Processing {len(var_names)} variables: {var_names}")
    for var_name in tqdm(var_names, desc="Processing fire index variables"):
        var_data = combined_ds[var_name]

        # --- monthly statistics (lazy) ---------------------------------------
        monthly = var_data.groupby(f"{time_dim}.month")
        mean_overall_monthly   = monthly.mean(dim=time_dim, skipna=True)
        std_overall_monthly    = monthly.std(dim=time_dim,  skipna=True)
        min_overall_monthly    = monthly.min(dim=time_dim,  skipna=True)
        max_overall_monthly    = monthly.max(dim=time_dim,  skipna=True)
        median_overall_monthly = monthly.median(dim=time_dim, skipna=True)

        feature_ds[f"{var_name}_mean"]   = mean_overall_monthly
        feature_ds[f"{var_name}_std"]    = std_overall_monthly
        feature_ds[f"{var_name}_min"]    = min_overall_monthly
        feature_ds[f"{var_name}_max"]    = max_overall_monthly
        feature_ds[f"{var_name}_median"] = median_overall_monthly
        feature_names.extend(
            [f"{var_name}_{s}" for s in ["mean", "std", "min", "max", "median"]]
        )

        # --- trend per calendar month ----------------------------------------
        monthly_mean_ts   = var_data.resample({time_dim: "1M"}).mean(skipna=True)
        monthly_max_ts    = var_data.resample({time_dim: "1M"}).max(skipna=True)
        monthly_median_ts = var_data.resample({time_dim: "1M"}).median(skipna=True)

        months_full_range = np.arange(1, 13)
        nan_trend_array   = xr.DataArray(
            np.nan,
            coords={"month": months_full_range, "latitude": original_lats, "longitude": original_lons},
            dims=["month", "latitude", "longitude"],
        )

        if monthly_mean_ts[time_dim].size >= 2:
            mean_trend_monthly = monthly_mean_ts.groupby(f"{time_dim}.month") \
                                                .apply(calculate_trend_slope, time_dim_name=time_dim) \
                                                .reindex(month=months_full_range)
            max_trend_monthly  = monthly_max_ts.groupby(f"{time_dim}.month") \
                                                .apply(calculate_trend_slope, time_dim_name=time_dim) \
                                                .reindex(month=months_full_range)
            median_trend_monthly = monthly_median_ts.groupby(f"{time_dim}.month") \
                                                    .apply(calculate_trend_slope, time_dim_name=time_dim) \
                                                    .reindex(month=months_full_range)
        else:
            mean_trend_monthly   = nan_trend_array
            max_trend_monthly    = nan_trend_array
            median_trend_monthly = nan_trend_array

        feature_ds[f"{var_name}_mean_trend"]   = mean_trend_monthly
        feature_ds[f"{var_name}_max_trend"]    = max_trend_monthly
        feature_ds[f"{var_name}_median_trend"] = median_trend_monthly
        feature_names.extend(
            [f"{var_name}_{s}_trend" for s in ["mean", "max", "median"]]
        )

    # -------- 4.  MERGE & INTERPOLATE  ---------------------------------------
    months_full_range = np.arange(1, 13)
    stats_ds = xr.Dataset(feature_ds).reindex(month=months_full_range)

    # build target grid
    lat_min, lat_max = lat_bounds or (original_lats.min(), original_lats.max())
    lon_min, lon_max = lon_bounds or (original_lons.min(), original_lons.max())
    target_lats = np.arange(lat_min, lat_max + target_resolution, target_resolution)
    target_lons = np.arange(lon_min, lon_max + target_resolution, target_resolution)

    interpolated_ds = stats_ds.interp(
        latitude=target_lats,
        longitude=target_lons,
        method="linear",
    )

    # -------- 5.  FINAL COMPUTE & SAVE  --------------------------------------
    if output_npz_path:
        print(f"Computing Dask graph and saving features to NPZ file: {output_npz_path}")
        with ProgressBar():
            interpolated_ds = interpolated_ds.compute()


        feature_arrays = {
            "month":     interpolated_ds.month.values,
            "latitude":  interpolated_ds.latitude.values,
            "longitude": interpolated_ds.longitude.values,
        }

        for f in sorted(set(feature_names)):
            if f not in interpolated_ds:
                raise ValueError(f"Feature {f} missing in interpolated dataset.")
            if np.all(np.isnan(interpolated_ds[f].values)):
                raise ValueError(f"Feature {f} consists entirely of NaNs.")
            feature_arrays[f] = interpolated_ds[f].values

        feature_arrays["feature_names"] = np.array(sorted(set(feature_names)), dtype=object)
        np.savez_compressed(output_npz_path, **feature_arrays)
        print(f"Successfully saved {len(feature_names)} features ➜ {output_npz_path}")

    total_time = time_lib.time() - start_time
    print(f"calculate_fire_index_features completed in {total_time:.2f} s")

    return interpolated_ds, feature_names



def get_fire_index_features(npz_path: str, target_df: pd.DataFrame, 
                           lat_col: str = "lat_rounded", lon_col: str = "lon_rounded") -> pd.DataFrame:
    """
    Loads precomputed **monthly** fire index features (including trends) from an NPZ file,
    and for each row in target_df finds and returns the feature value from the nearest grid point
    **for the month specified in target_df['month']**.
    
    The NPZ file is expected to contain:
      - 'latitude': a 1D numpy array of latitude values
      - 'longitude': a 1D numpy array of longitude values
      - 'month': a 1D numpy array of month numbers (e.g., 1-12)
      - For each feature name in 'feature_names': a 3D numpy array with shape (len(month), len(latitude), len(longitude))
      - 'feature_names': an array-like list of feature names (e.g., 'drtstr_mean', 'drtstr_mean_trend')
    
    For each row of target_df (with coordinates given in lat_col and lon_col and a month in 'month'),
    the nearest indices in the stored grid are determined. Then, for each base feature (e.g., 'drtstr_mean_trend'),
    one new column is created (e.g., 'drtstr_mean_trend') containing the value for the specified month
    at that nearest grid point.
    
    Args:
        npz_path (str): Path to the NPZ file with precomputed monthly fire index features.
        target_df (pd.DataFrame): DataFrame containing at least the coordinate columns and a 'month' column.
        lat_col (str): Name of the latitude column in target_df (default "lat_rounded").
        lon_col (str): Name of the longitude column in target_df (default "lon_rounded").
    
    Returns:
        tuple:
            pd.DataFrame: A new DataFrame with the original target_df columns plus additional columns 
                          for each fire index feature for the specified month.
            list: A list of the names of the added feature columns.
    """
    import numpy as np
    import pandas as pd

    # Load the NPZ file
    data = np.load(npz_path, allow_pickle=True)
    grid_lats = data["latitude"]   # 1D array of latitudes
    grid_lons = data["longitude"]  # 1D array of longitudes
    grid_months = data["month"]    # 1D array of months
    feature_names = list(data["feature_names"])  # Base feature names (includes trends)

    # Extract target coordinates and months from the DataFrame
    target_lats = target_df[lat_col].values
    target_lons = target_df[lon_col].values
    target_months = target_df['month'].values  # Month for each row

    # Find nearest latitude indices
    lat_indices = np.searchsorted(grid_lats, target_lats)
    lat_indices = np.clip(lat_indices, 0, len(grid_lats) - 1)
    lat_indices[target_lats >= grid_lats[-1]] = len(grid_lats) - 1
    lat_indices_prev = np.clip(lat_indices - 1, 0, len(grid_lats) - 1)
    lat_dist_prev = np.abs(grid_lats[lat_indices_prev] - target_lats)
    lat_dist_curr = np.abs(grid_lats[lat_indices] - target_lats)
    lat_indices = np.where((lat_dist_prev < lat_dist_curr) & (lat_indices > 0), lat_indices_prev, lat_indices)

    # Find nearest longitude indices
    lon_indices = np.searchsorted(grid_lons, target_lons)
    lon_indices = np.clip(lon_indices, 0, len(grid_lons) - 1)
    lon_indices[target_lons >= grid_lons[-1]] = len(grid_lons) - 1
    lon_indices_prev = np.clip(lon_indices - 1, 0, len(grid_lons) - 1)
    lon_dist_prev = np.abs(grid_lons[lon_indices_prev] - target_lons)
    lon_dist_curr = np.abs(grid_lons[lon_indices] - target_lons)
    lon_indices = np.where((lon_dist_prev < lon_dist_curr) & (lon_indices > 0), lon_indices_prev, lon_indices)

    # Initialize list to store added feature names and dictionary for new columns
    added_feature_names = []
    new_columns = {}

    # Process each base feature
    for base_feature in feature_names:
        if base_feature not in data:
            print(f"Warning: Base feature {base_feature} not found in NPZ file {npz_path}. Skipping.")
            continue
            
        f_array_3d = data[base_feature]
        expected_shape = (len(grid_months), len(grid_lats), len(grid_lons))
        if f_array_3d.shape != expected_shape:
            print(f"Warning: Shape mismatch for feature {base_feature}. Expected {expected_shape}, got {f_array_3d.shape}. Skipping.")
            continue

        # Map target months to indices in grid_months (assuming grid_months is 1-based [1, 2, ..., 12])
        month_indices = np.searchsorted(grid_months, target_months)
        month_indices = np.clip(month_indices, 0, len(grid_months) - 1)
        # Verify that the month matches
        month_indices[target_months > grid_months[-1]] = len(grid_months) - 1
        # Ensure the selected index corresponds to the correct month
        month_indices = np.where(target_months == grid_months[month_indices], month_indices, 
                                 np.clip(month_indices - 1, 0, len(grid_months) - 1))

        # Extract the feature value for the specific month at each point
        values_at_points = np.array([f_array_3d[month_idx, lat_idx, lon_idx]
                                     for month_idx, lat_idx, lon_idx
                                     in zip(month_indices, lat_indices, lon_indices)])

        # Add the feature column
        new_columns[f"fire_index_{base_feature}"] = values_at_points
        added_feature_names.append(f"fire_index_{base_feature}")

    # Create a new DataFrame with all new columns at once
    new_features_df = pd.DataFrame(new_columns, index=target_df.index)
    features_df = pd.concat([target_df.copy(), new_features_df], axis=1)

    return features_df, added_feature_names


def validate_feature_dataset(output_npz_path: str, nan_ratio_threshold: float = 0.1) -> None:
    """
    Sanity-check the computed feature dataset.

    Hard checks (raise ValueError):
      • Feature missing from dataset.
      • All values are NaN.
      • Global mean or std is non-finite.

    Soft check (warning only):
      • Fraction of NaNs exceeds `nan_ratio_threshold`.

    Parameters
    ----------
    dataset : xr.Dataset
        Fully-realised (computed) dataset with feature variables.
    feature_names : List[str]
        Expected feature variable names.
    nan_ratio_threshold : float, optional
        Acceptable NaN fraction before a warning is emitted (default 0.1 = 10 %).
    """
    data = np.load(output_npz_path, allow_pickle=True)
    feature_names = list(data["feature_names"])
    dataset = xr.Dataset(data)

    issues = []
    for f in feature_names:
        if f not in dataset:
            issues.append(f"Feature '{f}' is missing from the dataset.")
            continue

        arr = dataset[f].values  # numpy array (no dask after .compute())
        if arr.size == 0:
            issues.append(f"Feature '{f}' is empty.")
            continue

        nan_fraction = np.isnan(arr).mean()
        if nan_fraction == 1.0:
            issues.append(f"Feature '{f}' consists entirely of NaNs.")
            continue

        if nan_fraction > nan_ratio_threshold:
            warnings.warn(
                f"Feature '{f}' contains {nan_fraction:.1%} NaNs "
                f"(threshold {nan_ratio_threshold:.1%})."
            )

        mean_val = np.nanmean(arr)
        std_val  = np.nanstd(arr)

        if not np.isfinite(mean_val):
            issues.append(f"Feature '{f}' has non-finite mean ({mean_val}).")
        if not np.isfinite(std_val):
            issues.append(f"Feature '{f}' has non-finite std ({std_val}).")

    if issues:
        raise ValueError("Feature validation failed:\n" + "\n".join(" • " + m for m in issues))


if __name__ == "__main__":
    import glob
    config_path = 'configs/target_config.yaml'
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    fire_index_files = glob.glob("data/fire_index_*.nc")
    target_resolution =  config['spatial_coarseness']
    lat_bounds = (35, 80)
    lon_bounds = (1, 179)
    output_npz_path = "data/land_features/fire_index_features.npz"
    print(f"Saving features to NPZ file: {output_npz_path}")
    calculate_fire_index_features(fire_index_files, target_resolution, lat_bounds, lon_bounds, output_npz_path)

    validate_feature_dataset(output_npz_path)
