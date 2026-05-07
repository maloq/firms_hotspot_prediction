#!/usr/bin/env python3
"""Create missing ERA5 NetCDF links and processed zarrs for 2000-2008.

The files under /home/ids/vmorozov/era5 have a historical quirk: many files are
named ``*.grib`` but are HDF5/NetCDF files produced by cfgrib.  This script opens
them with xarray's NetCDF reader, creates missing ``*.nc`` symlinks for clarity,
and writes the processed zarr schema expected by the feature builder:

    /home/ids/vmorozov/data/climate_data/climate_features/ERA5/{var}/{var}_{year}.zarr

New zarrs are reindexed onto the existing 2009 ERA5 grid for each variable so
that xarray can combine 2000-2018 without the post-2019 grid mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class Era5Variable:
    short_name: str
    raw_dir: str
    raw_prefix: str


VARIABLES = [
    Era5Variable("t2m", "2m_temperature", "2m_temperature"),
    Era5Variable("d2m", "2m_dewpoint_temperature", "2m_dewpoint_temperature"),
    Era5Variable("tp", "total_precipitation", "total_precipitation"),
    Era5Variable("stl1", "soil_temperature_level_1", "soil_temperature_level_1"),
]


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_nc_link(raw_root: Path, spec: Era5Variable, year: int) -> tuple[Path, str]:
    raw_dir = raw_root / spec.raw_dir
    nc_path = raw_dir / f"{spec.raw_prefix}_{year}.nc"
    grib_path = raw_dir / f"{spec.raw_prefix}_{year}.grib"
    if nc_path.exists():
        return nc_path, "existing_nc"
    if not grib_path.exists():
        raise FileNotFoundError(f"Missing raw source for {spec.short_name} {year}: {nc_path} or {grib_path}")
    nc_path.symlink_to(grib_path.name)
    return nc_path, "created_nc_symlink_to_hdf5_grib_suffix"


def reference_grid(processed_root: Path, var: str) -> tuple[np.ndarray, np.ndarray]:
    ref = processed_root / var / f"{var}_2009.zarr"
    if not ref.exists():
        raise FileNotFoundError(f"Missing reference zarr for grid alignment: {ref}")
    ds = xr.open_zarr(ref, chunks=None)
    try:
        return ds["latitude"].values.copy(), ds["longitude"].values.copy()
    finally:
        ds.close()


def normalize_dataset(ds: xr.Dataset, var: str, year: int, lat_grid: np.ndarray, lon_grid: np.ndarray) -> xr.Dataset:
    if "valid_time" in ds.coords:
        source_time = "valid_time"
    elif "time" in ds.coords:
        source_time = "time"
    else:
        raise ValueError(f"{var} {year} missing valid_time/time coordinate")
    if var not in ds.data_vars:
        # Keep this explicit; silent wrong-variable conversion would poison the experiment.
        raise ValueError(f"{var} {year} missing expected variable {var}; data vars={list(ds.data_vars)}")

    ds = ds[[var]].sortby(source_time)
    start = np.datetime64(f"{year}-01-01")
    end = np.datetime64(f"{year}-12-31")
    ds = ds.sel({source_time: slice(start, end)})
    if ds.sizes.get(source_time, 0) == 0:
        raise ValueError(f"{var} {year} has no dates in expected year")
    if source_time != "time":
        ds = ds.rename({source_time: "time"})

    # Reindex to the 2009 processed ERA5 grid. For the newer raw downloads,
    # longitude sometimes ends at 179 rather than 180; the 180 column is outside
    # the ECMWF comparison footprint but retained for schema compatibility.
    ds = ds.reindex(latitude=lat_grid, longitude=lon_grid, method="nearest")
    ds[var] = ds[var].astype("float32")
    ds = ds.transpose("time", "latitude", "longitude")
    return ds


def write_zarr(ds: xr.Dataset, out: Path, force: bool) -> str:
    if out.exists():
        if not force:
            return "existing_zarr"
        shutil.rmtree(out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"chunks": tuple(ds[name].shape if ds[name].ndim == 0 else ds[name].shape)}
        for name in []
    }
    # Use compact, predictable chunks. xarray will ignore var encoding if not needed.
    var = next(iter(ds.data_vars))
    time_dim = "time" if "time" in ds.sizes else "valid_time"
    encoding[var] = {"chunks": (min(366, ds.sizes[time_dim]), ds.sizes["latitude"], ds.sizes["longitude"])}
    try:
        ds.to_zarr(tmp, mode="w", zarr_format=2, encoding=encoding)
    except TypeError:
        ds.to_zarr(tmp, mode="w", zarr_version=2, encoding=encoding)
    tmp.rename(out)
    return "created_zarr"


def verify_zarr(path: Path, var: str, year: int) -> dict[str, Any]:
    ds = xr.open_zarr(path, chunks=None)
    try:
        tc = "time" if "time" in ds.coords else "valid_time"
        return {
            "path": path,
            "variable": var,
            "year": year,
            "time_start": str(pd.Timestamp(ds[tc].values[0]).date()),
            "time_end": str(pd.Timestamp(ds[tc].values[-1]).date()),
            "time_count": int(ds.sizes[tc]),
            "lat_min": float(ds.latitude.min()),
            "lat_max": float(ds.latitude.max()),
            "lat_count": int(ds.sizes["latitude"]),
            "lon_min": float(ds.longitude.min()),
            "lon_max": float(ds.longitude.max()),
            "lon_count": int(ds.sizes["longitude"]),
            "dtype": str(ds[var].dtype),
        }
    finally:
        ds.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/home/ids/vmorozov/era5"))
    parser.add_argument("--processed-root", type=Path, default=Path("/home/ids/vmorozov/data/climate_data/climate_features/ERA5"))
    parser.add_argument("--years", type=int, nargs="*", default=list(range(2000, 2009)))
    parser.add_argument("--manifest", type=Path, default=Path("results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/era5_2000_2008_conversion_manifest.json"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for spec in VARIABLES:
        lat_grid, lon_grid = reference_grid(args.processed_root, spec.short_name)
        for year in args.years:
            started = time.perf_counter()
            nc_path, nc_status = ensure_nc_link(args.raw_root, spec, year)
            out = args.processed_root / spec.short_name / f"{spec.short_name}_{year}.zarr"
            if out.exists() and not args.force:
                status = "existing_zarr"
                verify = verify_zarr(out, spec.short_name, year)
            else:
                ds = xr.open_dataset(nc_path)
                try:
                    ds_norm = normalize_dataset(ds, spec.short_name, year, lat_grid, lon_grid)
                    status = write_zarr(ds_norm, out, args.force)
                    ds_norm.close()
                finally:
                    ds.close()
                verify = verify_zarr(out, spec.short_name, year)
            row = {
                "variable": spec.short_name,
                "year": year,
                "nc_path": nc_path,
                "nc_status": nc_status,
                "zarr_path": out,
                "zarr_status": status,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                **verify,
            }
            rows.append(row)
            print(f"{spec.short_name} {year}: {nc_status}, {status}, {row['elapsed_seconds']}s", flush=True)
    write_json(args.manifest, {"created_at": pd.Timestamp.now(), "rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
