from pathlib import Path
import numpy as np
import xarray as xr
import argparse

try:
    import xesmf as xe
except ImportError as e:   # xESMF is optional but strongly recommended
    xe = None
    print("xESMF not found falling back to xarray.interp.")

def interpolate_netcdf(
    in_path: str | Path,
    target_res_km: float = 5.0,
    out_path: str | Path | None = None,
    var_name: str | None = None,
    method: str = "bilinear",
    chunks: dict | None = None,
) -> Path:
    """
    Upsample a 2-D probability field from a 10×10 km grid (or any spacing)  
    to a finer grid (default 5 × 5 km) and save the result as NetCDF.

    Parameters
    ----------
    in_path : str | Path
        Path to the source NetCDF file.
    target_res_km : float, default 5.0
        Desired grid spacing in **kilometres**.
    out_path : str | Path | None, default None
        Where to write the new file.  If None, appends `_{target_res_km}km`
        before the original extension of the input filename.
    var_name : str | None
        Name of the variable to interpolate.  If None, the first data
        variable that contains `lat` and `lon` dimensions is used.
    method : {'bilinear', 'nearest_s2d', 'conservative', 'linear', 'nearest', 'cubic'}
        Interpolation scheme.  
        *When xESMF is available* the first three are accepted  
        (default **bilinear**).  Otherwise xarray's `'linear'`, `'nearest'`,
        `'cubic'`.
    chunks : dict | None
        Dask chunk mapping to speed up large files,
        e.g. `chunks={'lat': 256, 'lon': 256}`.

    Returns
    -------
    Path
        Path to the saved NetCDF file.

    Notes
    -----
    • Probabilities are clipped to the [0, 1] range after interpolation.  
    • CRS is assumed to be regular geographic (WGS-84).  For true-
      distance grids, re-project first (see rioxarray docs).

    """
    in_path = Path(in_path)
    if out_path is None:
        # Preserve the original extension (suffix) and simply append
        # _{target_res_km}km to the filename stem.
        out_path = in_path.with_name(
            f"{in_path.stem}_{int(target_res_km)}km{in_path.suffix}"
        )
    out_path = Path(out_path)

    # 1. Load
    ds = xr.open_dataset(in_path, chunks=chunks)
    if var_name is None:
        # pick the first variable that has lat/lon among its dimensions
        for name, da in ds.data_vars.items():
            if {"latitude", "longitude"}.issubset(da.dims):
                var_name = name
                break
        else:
            raise ValueError("Could not find a variable with ('lat','lon') dims.")
    prob = ds[var_name]

    # 2. Build target grid (simple geographic grid – 1° ≈ 111 km)
    ddeg = target_res_km / 111.0
    lat_new = np.arange(prob.latitude.min().item(), prob.latitude.max().item() + ddeg/2, ddeg)
    lon_new = np.arange(prob.longitude.min().item(), prob.longitude.max().item() + ddeg/2, ddeg)

    # 3. Interpolate
    if xe and method in {"bilinear", "nearest_s2d", "conservative"}:
        ds_in  = prob.to_dataset(name="prob")
        ds_out = xr.Dataset({"latitude": (["latitude"], lat_new), "longitude": (["longitude"], lon_new)})
        regridder = xe.Regridder(ds_in, ds_out, method, reuse_weights=False)
        prob_out = regridder(ds_in)["prob"]
    else:  # fallback to xarray.interp
        interp_method = {"bilinear": "linear"}.get(method, method)  # map if needed
        prob_out = prob.interp(latitude=lat_new, longitude=lon_new, method=interp_method)

    # 4. Post-process
    prob_out = prob_out.clip(0, 1)
    prob_out.attrs.update(prob.attrs)           # keep variable metadata
    ds_out = prob_out.to_dataset(name=var_name)
    ds_out.attrs.update(ds.attrs)              # keep global metadata
    # Record how the file was generated. Use 'valid_time' if available for timestamp.
    if "valid_time" in ds:
        time_str = str(ds["valid_time"].values)
        ds_out.attrs["history"] = (
            f"Interpolated from {in_path.name} to {target_res_km} km using {method} "
            f"on {time_str}"
        )
    else:
        ds_out.attrs["history"] = (
            f"Interpolated from {in_path.name} to {target_res_km} km using {method}"
        )

    # 5. Save
    comp = {"zlib": True, "complevel": 4}
    ds_out.to_netcdf(out_path, encoding={var_name: comp})

    return out_path


# -------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------

# Helper utilities ---------------------------------------------------


def _expected_out_path(in_path: Path, target_res_km: float) -> Path:
    """Return the pathname that :func:`interpolate_netcdf` would generate."""
    return in_path.with_name(f"{in_path.stem}_{int(target_res_km)}km{in_path.suffix}")


def _interpolate_if_needed(
    in_path: Path,
    target_res_km: float = 5.0,
    method: str = "bilinear",
    overwrite: bool = False,
) -> Path | None:
    """Interpolate *in_path* unless the target file already exists.

    Parameters
    ----------
    in_path : Path
        Source NetCDF file.
    target_res_km : float, default 5.0
        Desired output resolution in kilometres.
    method : str, default 'bilinear'
        Interpolation method (see :func:`interpolate_netcdf`).
    overwrite : bool, default False
        Recompute even if the output file already exists.
    """

    out_path = _expected_out_path(in_path, target_res_km)
    if out_path.exists() and not overwrite:
        print(f"✓ {out_path} already exists — skipping.")
        return out_path

    try:
        return interpolate_netcdf(
            in_path,
            target_res_km=target_res_km,
            out_path=out_path,
            method=method,
        )
    except Exception as exc:
        # Do not halt the entire batch on a single failure.
        print(f"✗ Failed to interpolate {in_path}: {exc}")
        return None


def _discover_forecast_files(root: Path) -> list[Path]:
    """Recursively locate raw forecast NetCDF files below *root*."""
    return sorted(root.rglob("forecast_raw_*.nc"))


# CLI entry-point -----------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpolate raw forecast NetCDF files onto a finer grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Path to a single NetCDF file to process. If omitted, the script scans the forecast directory tree.",
    )
    parser.add_argument(
        "--root",
        default="outputs/forecast_run",
        help="Root directory to search for forecast files when <file> is not provided.",
    )
    parser.add_argument(
        "--resolution",
        "--target_res_km",
        dest="target_res_km",
        type=float,
        default=5.0,
        help="Target grid spacing in kilometres.",
    )
    parser.add_argument(
        "--method",
        choices=[
            "bilinear",
            "nearest_s2d",
            "conservative",
            "linear",
            "nearest",
            "cubic",
        ],
        default="bilinear",
        help="Interpolation scheme to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-interpolate even if the target file already exists.",
    )

    args = parser.parse_args()

    if args.file is not None:
        # Process a single file supplied by the user.
        _interpolate_if_needed(
            Path(args.file),
            target_res_km=args.target_res_km,
            method=args.method,
            overwrite=args.overwrite,
        )
    else:
        # Scan the forecast run directory.
        root_dir = Path(args.root)
        files = _discover_forecast_files(root_dir)
        if not files:
            print(f"No raw forecast files found under {root_dir}.")
            return

        print(f"Found {len(files)} file(s) to process.")
        for path in files:
            _interpolate_if_needed(
                path,
                target_res_km=args.target_res_km,
                method=args.method,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    _main()