
import argparse
import os
from typing import Tuple, Optional

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt


def _find_common_variable(ds1: xr.Dataset, ds2: xr.Dataset, preferred: Optional[str] = None) -> str:
    """Return the first common variable name between the two datasets.

    If *preferred* is provided and exists in both, it is returned. Otherwise, the
    first intersection element is returned. Raises *ValueError* when no common
    variable is found.
    """
    if preferred and preferred in ds1.data_vars and preferred in ds2.data_vars:
        return preferred

    common = list(set(ds1.data_vars).intersection(ds2.data_vars))
    if not common:
        raise ValueError("No common data variables found between the two NetCDF files.")
    # Sort to have deterministic order
    common.sort()
    return common[0]


def _load_dataset(path: str) -> xr.Dataset:
    """Load a NetCDF file using *xarray*. Raises if the file cannot be found."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NetCDF file not found: {path}")
    return xr.open_dataset(path)


def _compute_metrics(arr1: xr.DataArray, arr2: xr.DataArray) -> Tuple[float, float, float]:
    """Compute MAE, RMSE, and maximum absolute difference between two arrays."""
    diff = arr1 - arr2
    # Convert to numpy masked array ignoring NaNs
    diff_values = diff.values
    # Flatten & mask NaNs
    mask = ~np.isnan(diff_values)
    if not np.any(mask):
        raise ValueError("All grid values are NaN after alignment; cannot compute metrics.")

    diff_flat = diff_values[mask]
    mae = float(np.mean(np.abs(diff_flat)))
    rmse = float(np.sqrt(np.mean(diff_flat ** 2)))
    max_diff = float(np.max(np.abs(diff_flat)))
    return mae, rmse, max_diff


def _save_diff_netcdf(diff_da: xr.DataArray, path: str) -> None:
    """Save DataArray *diff_da* to NetCDF at *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    diff_da.to_dataset().to_netcdf(path)
    print(f"Saved difference grid to {path}")


def _save_diff_png(diff_da: xr.DataArray, path: str) -> None:
    """Save the difference array as a PNG heatmap. Uses a diverging colormap centred at 0."""
    # Compute symmetric bounds around zero for colour scale
    max_abs = float(np.nanmax(np.abs(diff_da.values)))
    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 1e-6  # avoid zero-range colour scale

    plt.figure(figsize=(10, 6))

    if {'latitude', 'longitude'}.issubset(diff_da.coords):
        lon = diff_da['longitude'].values
        lat = diff_da['latitude'].values
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        plt.pcolormesh(lon_mesh, lat_mesh, diff_da.values, cmap='RdBu_r', shading='auto', vmin=-max_abs, vmax=max_abs)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    else:
        # Fallback: rely on xarray's default plotting (may not have geospatial axes)
        diff_da.plot(cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)

    plt.title(f"Difference heatmap: {diff_da.name}")
    plt.colorbar(label='Difference')
    plt.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved difference heatmap to {path}")


def compare_netcdf(file1: str, file2: str, variable: Optional[str] = None, save_diff: Optional[str] = None) -> None:
    """Load *file1* and *file2*, compare their *variable*, and print difference metrics.

    *variable*: name of the data variable to compare. If omitted, the first common
    variable present in both datasets is used.

    *save_diff*: optional path to save the difference grid. The file extension
    determines the output format: ".nc" → NetCDF, image extensions (".png",
    ".jpg", ".jpeg") → heatmap PNG.
    """
    ds1 = _load_dataset(file1)
    ds2 = _load_dataset(file2)

    try:
        var_name = _find_common_variable(ds1, ds2, preferred=variable)
    except ValueError as e:
        ds1.close()
        ds2.close()
        raise

    arr1_raw = ds1[var_name]
    arr2_raw = ds2[var_name]

    # If 'time' is a dimension of size 1, remove it to allow spatial comparison
    if 'time' in arr1_raw.dims and arr1_raw.sizes['time'] == 1:
        arr1_raw = arr1_raw.squeeze('time', drop=True)
    if 'time' in arr2_raw.dims and arr2_raw.sizes['time'] == 1:
        arr2_raw = arr2_raw.squeeze('time', drop=True)

    # Align along all coordinates/dimensions; inner join keeps overlapping area only
    arr1, arr2 = xr.align(arr1_raw, arr2_raw, join="inner")

    if arr1.size == 0:
        raise ValueError("Datasets have no overlapping coordinates to compare.")

    mae, rmse, max_diff = _compute_metrics(arr1, arr2)

    print("=== NetCDF Comparison Report ===")
    print(f"Variable compared    : {var_name}")
    print(f"Grid shape compared  : {arr1.shape}")
    print(f"Mean Abs. Error (MAE): {mae:.6f}")
    print(f"Root MSE (RMSE)     : {rmse:.6f}")
    print(f"Max Abs. Diff       : {max_diff:.6f}")

    if save_diff:
        diff_da = (arr1 - arr2).rename(f"{var_name}_diff")
        ext = os.path.splitext(save_diff)[1].lower()
        if ext == '.nc':
            _save_diff_netcdf(diff_da, save_diff)
        elif ext in {'.png', '.jpg', '.jpeg'}:
            _save_diff_png(diff_da, save_diff)
        else:
            print(f"Warning: Unrecognized extension '{ext}'. Supported: .nc, .png, .jpg, .jpeg. Skipping save.")

    # Close datasets explicitly to free resources
    ds1.close()
    ds2.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two NetCDF outputs from prediction_pipeline and report their differences.")
    
    parser.add_argument("file1", type=str, nargs="?", help="Path to the first NetCDF file.",
                        default="outputs/forecast_run_30d/2025-07-11/netcdf/forecast_raw_2025-07-11_Russian_Federation.nc")
    
    parser.add_argument("file2", type=str, nargs="?", help="Path to the second NetCDF file.",
                        default="outputs/forecast_run_30d/2025-07-12/netcdf/forecast_raw_2025-07-12_Russian_Federation.nc")
    
    parser.add_argument("--var", type=str, default=None,
                        help="Name of variable to compare. Defaults to the first common variable.")
    
    parser.add_argument("--save-diff", type=str, default="outputs/diff.png",
                        help="Optional path to save the difference grid. Extension determines format (.nc or image such as .png).")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compare_netcdf(args.file1, args.file2, variable=args.var, save_diff=args.save_diff)
