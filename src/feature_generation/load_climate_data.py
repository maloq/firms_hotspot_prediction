from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import time as time_lib
from pathlib import Path
import pathlib
import glob

import pandas as pd
import xarray as xr
import numpy as np

try:
    from numcodecs import get_codec
except ImportError:  # pragma: no cover - only used for metadata fallback
    get_codec = None


def find_climate_files(climate_data_dir: str, variable: str):
    """
    Return (file_list, variable_dir) for *variable*.
    New logic: if <climate_data_dir>/<variable> is missing, look one level
    deeper and search recursively.
    """
    # ―― primary location: <dir>/<variable>/variable_*.zarr ――――――――――――――――――
    variable_dir = os.path.join(climate_data_dir, variable)
    pattern      = os.path.join(variable_dir, f"{variable}_*.zarr")
    files        = glob.glob(pattern)

    # ―― fallback: search one level deeper (e.g. …/ECMWF/<variable>) ――――――
    if not files:
        pattern = os.path.join(climate_data_dir, "**", f"{variable}_*.zarr")
        files   = glob.glob(pattern, recursive=True)
        if files:
            # derive the “variable_dir” from the first hit (for logging only)
            variable_dir = str(pathlib.Path(files[0]).parent)

    if not files:
        raise FileNotFoundError(
            f"No NetCDFs named '{variable}_*.zarr' under {climate_data_dir}"
        )
    return files, variable_dir


@dataclass(frozen=True)
class ClimateFragment:
    """A spatial climate fragment, usually backed by several temporal shards."""

    variable: str
    files: tuple[str, ...]
    time_start: np.datetime64
    time_end: np.datetime64
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    lat_step: float
    lon_step: float
    lat_size: int
    lon_size: int
    dtype: str
    priority: int = 0
    time_coord_name: str = "valid_time"

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def spatial_area(self) -> float:
        return max(0.0, self.lat_max - self.lat_min) * max(0.0, self.lon_max - self.lon_min)

    @property
    def resolution(self) -> float:
        steps = [step for step in (self.lat_step, self.lon_step) if np.isfinite(step) and step > 0]
        return max(steps) if steps else float("inf")

    def source_token_payload(self) -> str:
        parts = [
            self.variable,
            str(self.time_start),
            str(self.time_end),
            f"{self.lat_min:.8f}",
            f"{self.lat_max:.8f}",
            f"{self.lon_min:.8f}",
            f"{self.lon_max:.8f}",
            f"{self.lat_step:.8f}",
            f"{self.lon_step:.8f}",
            str(self.lat_size),
            str(self.lon_size),
            self.dtype,
            str(self.priority),
        ]
        for file_name in self.files:
            path = Path(file_name)
            try:
                stat_target = path / ".zmetadata" if path.is_dir() and (path / ".zmetadata").exists() else path
                stat = stat_target.stat()
                parts.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                parts.append(str(path))
        return "|".join(parts)


@dataclass(frozen=True)
class _ClimateFileMeta:
    path: str
    variable: str
    time_start: np.datetime64
    time_end: np.datetime64
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    lat_step: float
    lon_step: float
    lat_size: int
    lon_size: int
    dtype: str
    time_coord_name: str


def _dask_array_available() -> bool:
    try:
        import dask.array  # noqa: F401

        return True
    except ImportError:
        return False


def _time_coord_from_dataset(ds: xr.Dataset) -> str:
    if "valid_time" in ds.coords:
        return "valid_time"
    if "time" in ds.coords:
        return "time"
    raise ValueError("Climate dataset has neither 'valid_time' nor 'time' coordinate.")


def _coord_step(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    diffs = np.diff(np.sort(values))
    diffs = np.abs(diffs[np.isfinite(diffs) & (diffs != 0)])
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _metadata_from_xarray(path: Path, variable: str) -> _ClimateFileMeta:
    chunks = "auto" if _dask_array_available() else None
    ds = xr.open_dataset(path, decode_timedelta=False, chunks=chunks)
    try:
        time_coord = _time_coord_from_dataset(ds)
        data_var = variable if variable in ds.data_vars else next(iter(ds.data_vars))
        time_values = pd.to_datetime(ds[time_coord].values).to_numpy(dtype="datetime64[ns]")
        lat_values = np.asarray(ds["latitude"].values, dtype=float)
        lon_values = np.asarray(ds["longitude"].values, dtype=float)
        return _ClimateFileMeta(
            path=str(path),
            variable=variable,
            time_start=np.nanmin(time_values),
            time_end=np.nanmax(time_values),
            lat_min=float(np.nanmin(lat_values)),
            lat_max=float(np.nanmax(lat_values)),
            lon_min=float(np.nanmin(lon_values)),
            lon_max=float(np.nanmax(lon_values)),
            lat_step=_coord_step(lat_values),
            lon_step=_coord_step(lon_values),
            lat_size=int(lat_values.size),
            lon_size=int(lon_values.size),
            dtype=str(ds[data_var].dtype),
            time_coord_name="valid_time",
        )
    finally:
        ds.close()


def _load_zarr_metadata(store: Path) -> dict:
    consolidated = store / ".zmetadata"
    if consolidated.exists():
        with consolidated.open("r", encoding="utf-8") as fh:
            return json.load(fh).get("metadata", {})

    metadata: dict[str, dict] = {}
    root_v3 = store / "zarr.json"
    if root_v3.exists():
        with root_v3.open("r", encoding="utf-8") as fh:
            root_doc = json.load(fh)
        inline = root_doc.get("consolidated_metadata", {}).get("metadata", {})
        for name, node in inline.items():
            metadata[f"{name}/zarr.json"] = node
        if metadata:
            return metadata

    for meta_file in store.rglob("zarr.json"):
        if meta_file == root_v3:
            continue
        rel = meta_file.relative_to(store).as_posix()
        with meta_file.open("r", encoding="utf-8") as fh:
            metadata[rel] = json.load(fh)
    for meta_file in store.rglob(".zarray"):
        rel = meta_file.relative_to(store).as_posix()
        with meta_file.open("r", encoding="utf-8") as fh:
            metadata[rel] = json.load(fh)
    for attrs_file in store.rglob(".zattrs"):
        rel = attrs_file.relative_to(store).as_posix()
        with attrs_file.open("r", encoding="utf-8") as fh:
            metadata[rel] = json.load(fh)
    return metadata


def _metadata_entry(metadata: dict, array_name: str, suffix: str) -> dict | None:
    if suffix == ".zarray":
        return metadata.get(f"{array_name}/.zarray") or metadata.get(f"{array_name}/zarr.json")
    if suffix == ".zattrs":
        v2_attrs = metadata.get(f"{array_name}/.zattrs")
        if v2_attrs is not None:
            return v2_attrs
        v3_node = metadata.get(f"{array_name}/zarr.json")
        if v3_node is not None:
            return v3_node.get("attributes", {})
    return metadata.get(f"{array_name}/{suffix}")


def _read_zarr_1d_array(store: Path, metadata: dict, array_name: str) -> np.ndarray | None:
    if get_codec is None:
        return None

    zarray = _metadata_entry(metadata, array_name, ".zarray")
    if not zarray or len(zarray.get("shape", [])) != 1:
        return None
    if zarray.get("zarr_format") == 3 or "chunks" not in zarray:
        return None
    if zarray.get("filters"):
        return None

    shape = int(zarray["shape"][0])
    chunk_size = int(zarray["chunks"][0])
    dtype = np.dtype(zarray["dtype"])
    order = zarray.get("order", "C")
    compressor_config = zarray.get("compressor")
    compressor = get_codec(compressor_config) if compressor_config else None
    result = np.empty(shape, dtype=dtype)

    for chunk_idx, start in enumerate(range(0, shape, chunk_size)):
        chunk_path = store / array_name / str(chunk_idx)
        if not chunk_path.exists():
            return None
        raw = chunk_path.read_bytes()
        decoded = compressor.decode(raw) if compressor is not None else raw
        chunk = np.frombuffer(decoded, dtype=dtype)
        chunk = chunk.reshape((chunk_size,), order=order)
        stop = min(start + chunk_size, shape)
        result[start:stop] = chunk[: stop - start]

    return result


def _decode_cf_time(values: np.ndarray, attrs: dict | None) -> np.ndarray | None:
    if values is None:
        return None
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")

    attrs = attrs or {}
    units = attrs.get("units")
    if not units:
        return None

    match = re.match(r"^\s*([A-Za-z]+)\s+since\s+(.+?)\s*$", str(units))
    if not match:
        return None

    unit_name, origin = match.groups()
    unit_map = {
        "day": "D",
        "days": "D",
        "hour": "h",
        "hours": "h",
        "minute": "m",
        "minutes": "m",
        "second": "s",
        "seconds": "s",
        "millisecond": "ms",
        "milliseconds": "ms",
        "microsecond": "us",
        "microseconds": "us",
        "nanosecond": "ns",
        "nanoseconds": "ns",
    }
    pd_unit = unit_map.get(unit_name.lower())
    if pd_unit is None:
        return None

    try:
        origin_ts = pd.Timestamp(origin)
        decoded = origin_ts + pd.to_timedelta(values.astype(float), unit=pd_unit)
        return decoded.to_numpy(dtype="datetime64[ns]")
    except Exception:
        return None


def _date_bounds_from_filename(path: Path) -> tuple[np.datetime64, np.datetime64] | None:
    name = path.name
    range_match = re.search(r"_(\d{8})-(\d{8})(?:_|\.zarr$)", name)
    if range_match:
        start, end = range_match.groups()
        return np.datetime64(pd.Timestamp(start).to_datetime64(), "ns"), np.datetime64(
            pd.Timestamp(end).to_datetime64(), "ns"
        )

    year_match = re.search(r"_(\d{4})(?:_|\.zarr$)", name)
    if year_match:
        year = int(year_match.group(1))
        return np.datetime64(f"{year:04d}-01-01", "ns"), np.datetime64(f"{year:04d}-12-31", "ns")

    return None


def _lat_lon_bounds_from_filename(path: Path) -> tuple[float, float, float, float] | None:
    match = re.search(
        r"latitude(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_longitude(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)",
        path.name,
    )
    if not match:
        return None
    lat0, lat1, lon0, lon1 = map(float, match.groups())
    return min(lat0, lat1), max(lat0, lat1), min(lon0, lon1), max(lon0, lon1)


def _metadata_from_zarr_store(path: Path, variable: str) -> _ClimateFileMeta:
    metadata = _load_zarr_metadata(path)
    data_var = variable if _metadata_entry(metadata, variable, ".zarray") else None
    if data_var is None:
        data_vars = [
            key.split("/", 1)[0]
            for key, value in metadata.items()
            if (key.endswith("/.zarray") or key.endswith("/zarr.json")) and len(value.get("shape", [])) >= 3
        ]
        if not data_vars:
            raise ValueError(f"No data variable metadata found in {path}")
        data_var = sorted(data_vars)[0]

    var_zarray = _metadata_entry(metadata, data_var, ".zarray") or {}
    var_zattrs = _metadata_entry(metadata, data_var, ".zattrs") or {}

    lat_values = _read_zarr_1d_array(path, metadata, "latitude")
    lon_values = _read_zarr_1d_array(path, metadata, "longitude")

    if lat_values is not None:
        lat_min = float(np.nanmin(lat_values))
        lat_max = float(np.nanmax(lat_values))
        lat_step = _coord_step(lat_values)
        lat_size = int(lat_values.size)
    else:
        lat_first = var_zattrs.get("GRIB_latitudeOfFirstGridPointInDegrees")
        lat_last = var_zattrs.get("GRIB_latitudeOfLastGridPointInDegrees")
        lat_size = int(var_zarray.get("shape", [0, 0, 0])[-2])
        if lat_first is None or lat_last is None:
            filename_bounds = _lat_lon_bounds_from_filename(path)
            if filename_bounds is None:
                raise ValueError(f"Cannot infer latitude bounds for {path}")
            lat_min, lat_max = filename_bounds[:2]
        else:
            lat_min, lat_max = sorted((float(lat_first), float(lat_last)))
        lat_step = abs(lat_max - lat_min) / max(1, lat_size - 1)

    if lon_values is not None:
        lon_min = float(np.nanmin(lon_values))
        lon_max = float(np.nanmax(lon_values))
        lon_step = _coord_step(lon_values)
        lon_size = int(lon_values.size)
    else:
        lon_first = var_zattrs.get("GRIB_longitudeOfFirstGridPointInDegrees")
        lon_last = var_zattrs.get("GRIB_longitudeOfLastGridPointInDegrees")
        lon_size = int(var_zarray.get("shape", [0, 0, 0])[-1])
        if lon_first is None or lon_last is None:
            filename_bounds = _lat_lon_bounds_from_filename(path)
            if filename_bounds is None:
                raise ValueError(f"Cannot infer longitude bounds for {path}")
            lon_min, lon_max = filename_bounds[2:]
        else:
            lon_min, lon_max = sorted((float(lon_first), float(lon_last)))
        lon_step = abs(lon_max - lon_min) / max(1, lon_size - 1)

    time_coord = "valid_time" if _metadata_entry(metadata, "valid_time", ".zarray") else "time"
    time_values = _read_zarr_1d_array(path, metadata, time_coord)
    time_attrs = _metadata_entry(metadata, time_coord, ".zattrs") or {}
    decoded_time = _decode_cf_time(time_values, time_attrs) if time_values is not None else None
    if decoded_time is not None and decoded_time.size:
        time_start = np.nanmin(decoded_time)
        time_end = np.nanmax(decoded_time)
    else:
        filename_bounds = _date_bounds_from_filename(path)
        if filename_bounds is None:
            raise ValueError(f"Cannot infer time bounds for {path}")
        time_start, time_end = filename_bounds

    return _ClimateFileMeta(
        path=str(path),
        variable=variable,
        time_start=np.datetime64(time_start, "ns"),
        time_end=np.datetime64(time_end, "ns"),
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_step=lat_step,
        lon_step=lon_step,
        lat_size=lat_size,
        lon_size=lon_size,
        dtype=str(np.dtype(var_zarray.get("dtype") or var_zarray.get("data_type", "float32"))),
        time_coord_name="valid_time",
    )


def _read_climate_file_metadata(path: str | Path, variable: str) -> _ClimateFileMeta:
    path = Path(path)
    try:
        return _metadata_from_xarray(path, variable)
    except Exception as xarray_error:
        if path.suffix == ".zarr" and path.is_dir():
            try:
                return _metadata_from_zarr_store(path, variable)
            except Exception as zarr_error:
                raise ValueError(
                    f"Could not read climate metadata from {path}. "
                    f"xarray error: {xarray_error}; zarr metadata error: {zarr_error}"
                ) from zarr_error
        raise


def _spatial_group_key(meta: _ClimateFileMeta) -> tuple:
    return (
        meta.variable,
        round(meta.lat_min, 6),
        round(meta.lat_max, 6),
        round(meta.lon_min, 6),
        round(meta.lon_max, 6),
        meta.lat_size,
        meta.lon_size,
        round(meta.lat_step, 8),
        round(meta.lon_step, 8),
    )


def discover_climate_fragments(climate_data_dir: str, variable: str) -> list[ClimateFragment]:
    """Discover climate zarr files and group temporal shards by spatial grid."""

    files, _ = find_climate_files(climate_data_dir, variable)
    metas = [_read_climate_file_metadata(path, variable) for path in sorted(files)]
    grouped: dict[tuple, list[_ClimateFileMeta]] = {}
    for meta in metas:
        grouped.setdefault(_spatial_group_key(meta), []).append(meta)

    fragments: list[ClimateFragment] = []
    for priority, group in enumerate(sorted(grouped.values(), key=lambda values: values[0].path)):
        group_sorted = sorted(group, key=lambda item: (item.time_start, item.path))
        first = group_sorted[0]
        fragments.append(
            ClimateFragment(
                variable=variable,
                files=tuple(item.path for item in group_sorted),
                time_start=min(item.time_start for item in group_sorted),
                time_end=max(item.time_end for item in group_sorted),
                lat_min=first.lat_min,
                lat_max=first.lat_max,
                lon_min=first.lon_min,
                lon_max=first.lon_max,
                lat_step=first.lat_step,
                lon_step=first.lon_step,
                lat_size=first.lat_size,
                lon_size=first.lon_size,
                dtype=first.dtype,
                priority=priority,
                time_coord_name=first.time_coord_name,
            )
        )

    return sorted(
        fragments,
        key=lambda item: (item.priority, item.resolution, item.spatial_area, item.files[0]),
    )


def climate_fragments_source_token(fragments: list[ClimateFragment]) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for fragment in sorted(fragments, key=lambda item: item.files[0]):
        hasher.update(fragment.source_token_payload().encode("utf-8"))
    return hasher.hexdigest()


def _coordinate_half_step(coord) -> float:
    values = np.asarray(coord.compute().values if hasattr(coord, "compute") else coord.values, dtype=float)
    if values.size <= 1:
        return 0.0
    diffs = np.diff(np.sort(values))
    diffs = np.abs(diffs[np.isfinite(diffs) & (diffs != 0)])
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs) / 2.0)


def _padded_coord_range(requested_range, coord) -> tuple[float, float]:
    """Pad target bounds enough to keep nearest climate cells at chunk edges."""

    lower, upper = (float(requested_range[0]), float(requested_range[1]))
    if lower > upper:
        lower, upper = upper, lower
    pad = _coordinate_half_step(coord)
    eps = max(pad, 1e-9) * 1e-6
    return lower - pad - eps, upper + pad + eps


def _open_climate_files(files, variable, time_range=None, lat_range=None, lon_range=None, chunks_spec="auto", source_label=None):
    load_start_time = time_lib.time()
    files = [str(path) for path in files]
    variable_dir = source_label or str(pathlib.Path(files[0]).parent)
    chunks_to_use = chunks_spec if _dask_array_available() else None

    print(f"Opening {len(files)} climate file(s) for {variable} …")
    time_coord_name = "valid_time"

    def _preprocess_dataset(ds):
        if time_coord_name not in ds.coords and "time" in ds.coords:
            ds = ds.rename({"time": time_coord_name})
        if time_coord_name in ds.coords:
            ds = ds.sortby(time_coord_name)
        if lat_range and "latitude" in ds.coords:
            lat0, lat1 = _padded_coord_range(lat_range, ds.latitude)
            decreasing = ds.latitude[0] > ds.latitude[-1]
            ds = ds.sel(latitude=slice(lat1, lat0) if decreasing else slice(lat0, lat1))
        if lon_range and "longitude" in ds.coords:
            lon0, lon1 = _padded_coord_range(lon_range, ds.longitude)
            ds = ds.sel(longitude=slice(lon0, lon1))
        return ds

    try:
        if len(files) == 1:
            ds = xr.open_dataset(
                files[0],
                decode_timedelta=False,
                chunks=chunks_to_use,
            )
            ds = _preprocess_dataset(ds)
        else:
            # xarray ≥ 0.23 ➜ no concat_dim when combine="by_coords"
            ds = xr.open_mfdataset(
                files,
                combine="by_coords",
                decode_timedelta=False,
                chunks=chunks_to_use,
                parallel=bool(chunks_to_use),
                preprocess=_preprocess_dataset,
            )
            ds = _preprocess_dataset(ds)

    except TypeError as e:
        print(" • Falling back to combine='nested' + explicit concat_dim "
              f"because: {e}")
        ds = xr.open_mfdataset(
            files,
            concat_dim=time_coord_name,
            combine="nested",
            decode_timedelta=False,
            chunks=chunks_to_use,
            parallel=bool(chunks_to_use),
            preprocess=_preprocess_dataset,
        )
        ds = _preprocess_dataset(ds)

    except Exception as e:
        print(f"Error opening climate files in {variable_dir}: {e}")
        print("Check:")
        print(f" - Time coordinate name used: '{time_coord_name}' (is it correct?)")
        print(f" - File integrity in list: {files[:3]}...")
        print(f" - Chunk specification: {chunks_to_use}")
        print(f" - Available memory and Dask worker logs (if using a cluster)")
        raise IOError(f"Failed to open dataset for {variable}") from e

    if lat_range and 'latitude' in ds.coords:
        lat0, lat1 = _padded_coord_range(lat_range, ds.latitude)
        decreasing = ds.latitude[0] > ds.latitude[-1]

        if decreasing:
            ds = ds.sel(latitude=slice(lat1, lat0))
        else:
            ds = ds.sel(latitude=slice(lat0, lat1))
    if lon_range and 'longitude' in ds.coords:
        lon0, lon1 = _padded_coord_range(lon_range, ds.longitude)
        ds = ds.sel(longitude=slice(lon0, lon1))

    if time_range and time_coord_name in ds.coords:
       try:
            coord_dtype = ds[time_coord_name].dtype
            start_time = pd.Timestamp(time_range[0]).to_datetime64().astype(coord_dtype)
            end_time = pd.Timestamp(time_range[1]).to_datetime64().astype(coord_dtype)
            ds = ds.sel({time_coord_name: slice(start_time, end_time)})
       except Exception as e:
           print(f"Warning: Could not apply precise time slice ({time_range}) using .sel(): {e}. Relying on file filtering.")
           print(f"Dataset time coordinate type: {ds[time_coord_name].dtype}")

    print("\n" + "="*50)
    print(f"✅ LAZY CLIMATE DATASET CREATED: {variable}")
    print("="*50)
    print(f"  Load Function Time: {time_lib.time() - load_start_time:.2f}s")
    try:
        def safe_min_max_str(coord, fmt='%Y-%m-%d %H:%M:%S'):
            if coord.size > 0:
                 min_val = pd.Timestamp(coord.min().compute().item()) if hasattr(coord.min(), "compute") else pd.Timestamp(coord.min().item())
                 max_val = pd.Timestamp(coord.max().compute().item()) if hasattr(coord.max(), "compute") else pd.Timestamp(coord.max().item())
                 return f"{min_val.strftime(fmt)} to {max_val.strftime(fmt)}"
            return "N/A or Empty"

        def safe_min_max_float(coord, fmt='.2f'):
             if coord.size > 0:
                  min_raw = coord.min()
                  max_raw = coord.max()
                  min_val = min_raw.compute().item() if hasattr(min_raw, "compute") else min_raw.item()
                  max_val = max_raw.compute().item() if hasattr(max_raw, "compute") else max_raw.item()
                  return f"{min_val:{fmt}} to {max_val:{fmt}}"
             return "N/A or Empty"

        time_str = safe_min_max_str(ds[time_coord_name]) if time_coord_name in ds else "N/A"
        lat_str = safe_min_max_float(ds.latitude) if 'latitude' in ds else "N/A"
        lon_str = safe_min_max_float(ds.longitude) if 'longitude' in ds else "N/A"

        print(f"📅 Approx Time Range (lazy): {time_str}")
        print(f"🌐 Approx Lat Range (lazy):  {lat_str}")
        print(f"🌐 Approx Lon Range (lazy):  {lon_str}")
        print(f"  Dimensions (lazy): {dict(ds.sizes)}")
        print(f"  Chunking reported by xarray: {ds.chunks}")
    except Exception as e:
        print(f"Could not print all dataset details (might be empty or error): {e}")
        print(f"Dataset Coords: {list(ds.coords)}")
        print(f"Dataset Variables: {list(ds.data_vars)}")

    print("="*50 + "\n")

    return ds


def open_climate_fragment(fragment: ClimateFragment, time_range=None, lat_range=None, lon_range=None, chunks_spec="auto"):
    return _open_climate_files(
        fragment.files,
        fragment.variable,
        time_range=time_range,
        lat_range=lat_range,
        lon_range=lon_range,
        chunks_spec=chunks_spec,
        source_label=f"fragment:{fragment.files[0]}",
    )


def load_climate_variable_mf(climate_data_dir, variable, time_range=None, lat_range=None, lon_range=None, test_mode=False, chunks_spec="auto"):
    """
    Loads data for a single climate variable using open_mfdataset for efficiency.

    Args:
        climate_data_dir: Directory containing climate data files.
        variable: Name of the climate variable directory/prefix.
        time_range: Optional tuple (start_time, end_time) pandas Timestamps/datetimes.
        lat_range: Optional tuple (min_lat, max_lat).
        lon_range: Optional tuple (min_lon, max_lon).
        test_mode: (Currently unused, consider purpose).
        chunks_spec: Chunk specification for Dask (e.g., "auto", {'latitude': 50, 'longitude': 50}).

    Returns:
        xarray Dataset (Dask-backed) containing the loaded variable data.

    Raises:
        ValueError: If data cannot be found or loaded.
        IOError: If xr.open_mfdataset fails.
    """
    # 1. Find files, pre-filtering by filename convention
    try:
        files, variable_dir = find_climate_files(climate_data_dir, variable)
    except (FileNotFoundError, ValueError) as e:
        # Reraise as ValueError for the calling function to handle
        raise ValueError(f"Could not find data for variable {variable}: {e}") from e

    return _open_climate_files(
        files,
        variable,
        time_range=time_range,
        lat_range=lat_range,
        lon_range=lon_range,
        chunks_spec=chunks_spec,
        source_label=variable_dir,
    )
