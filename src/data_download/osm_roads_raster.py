"""Download Geofabrik OSM extracts and build an unsmoothed drivable-roads raster.

The script intentionally avoids the slow OpenStreetMap API. It downloads country
PBF extracts from Geofabrik, streams OSM road ways in batches through GDAL/pyogrio,
and burns them into a compact uint8 raster.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyogrio
import shapely
import yaml
from numba import njit
from PIL import Image
from pyproj import CRS, Transformer

try:
    import xarray as xr
except ImportError:  # pragma: no cover - optional output
    xr = None


GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1-nogeom.json"
DEFAULT_WORKER_TIMEOUT_SEC = 45 * 60
DEFAULT_WORKER_MEMORY_GB = 8.0
DEFAULT_MAX_RASTER_CELLS = 500_000_000
DEFAULT_MAX_PBF_SIZE_MB = 2500.0
DEFAULT_BATCH_SIZE = 8192

DEFAULT_DRIVABLE_HIGHWAYS = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
    "track",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
)

COUNTRY_ALIASES = {
    "dem_rep_korea": "north-korea",
    "russian_federation": "russia",
    "republic_of_korea": "south-korea",
    "czech_republic": "czech-republic",
    "macedonia_former_yugoslav_republic_of": "macedonia",
    "bosnia_and_herzegovina": "bosnia-herzegovina",
    "united_states": "us",
}

SAFE_WORKER_ENV = {
    "CPL_DEBUG": "OFF",
    "GDAL_CACHEMAX": "128",
    "OGR_INTERLEAVED_READING": "YES",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


@njit(cache=True)
def _burn_polyline(raster: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> int:
    """Burn a polyline represented by integer row/col arrays into raster."""

    height, width = raster.shape
    burned = 0
    for i in range(rows.size - 1):
        r0 = rows[i]
        c0 = cols[i]
        r1 = rows[i + 1]
        c1 = cols[i + 1]

        min_r = r0 if r0 < r1 else r1
        max_r = r1 if r0 < r1 else r0
        min_c = c0 if c0 < c1 else c1
        max_c = c1 if c0 < c1 else c0
        if max_r < 0 or max_c < 0 or min_r >= height or min_c >= width:
            continue

        dr = r1 - r0
        dc = c1 - c0
        steps = abs(dr)
        if abs(dc) > steps:
            steps = abs(dc)

        if steps == 0:
            if 0 <= r0 < height and 0 <= c0 < width:
                if raster[r0, c0] == 0:
                    burned += 1
                raster[r0, c0] = 1
            continue

        for step in range(steps + 1):
            rr = int(math.floor(r0 + dr * step / steps + 0.5))
            cc = int(math.floor(c0 + dc * step / steps + 0.5))
            if 0 <= rr < height and 0 <= cc < width:
                if raster[rr, cc] == 0:
                    burned += 1
                raster[rr, cc] = 1

    return burned


def normalize_country_name(country: str) -> str:
    key = country.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    return key.replace("_", "-")


def clean_geofabrik_name(name: str) -> str:
    name = re.sub(r"<[^>]+>", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def load_config_countries(config_path: Path, key: str) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    countries = config.get(key)
    if not countries:
        raise ValueError(f"No countries found in {config_path} at key '{key}'")
    return list(countries)


def parse_coordinate_bounds(config_path: Path) -> tuple[float, float, float, float] | None:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    bounds = config.get("coordinate_bounds")
    if not bounds:
        return None
    if len(bounds) != 4:
        raise ValueError("coordinate_bounds must be [lat_min, lon_min, lat_max, lon_max]")
    lat_min, lon_min, lat_max, lon_max = map(float, bounds)
    return lon_min, lat_min, lon_max, lat_max


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    print(f"Downloading {url}", flush=True)
    if partial_path.exists():
        print(f"  resuming {partial_path.stat().st_size / 1024 / 1024:.1f} MB partial", flush=True)

    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "-C",
        "-",
        "-o",
        str(partial_path),
        url,
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("curl is required for resilient Geofabrik downloads") from exc
    partial_path.replace(output_path)


def load_geofabrik_index(index_path: Path, refresh: bool = False) -> dict:
    if refresh or not index_path.exists():
        download_file(GEOFABRIK_INDEX_URL, index_path, chunk_size=256 * 1024)
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_extracts(index: dict, countries: Iterable[str], expand_parents: bool = True) -> list[dict]:
    by_id = {feature["properties"]["id"]: feature["properties"] for feature in index["features"]}
    by_parent: dict[str, list[dict]] = {}
    for feature in index["features"]:
        props = feature["properties"]
        parent = props.get("parent")
        if parent:
            by_parent.setdefault(parent, []).append(props)

    resolved = []
    missing = []
    for country in countries:
        geofabrik_id = normalize_country_name(country)
        props = by_id.get(geofabrik_id)
        if not props:
            missing.append((country, geofabrik_id))
            continue
        url = props.get("urls", {}).get("pbf")
        if not url:
            missing.append((country, geofabrik_id))
            continue
        children = [
            child
            for child in by_parent.get(geofabrik_id, [])
            if child.get("urls", {}).get("pbf")
        ]
        if expand_parents and children:
            print(
                f"Expanding {country} ({geofabrik_id}) into "
                f"{len(children)} Geofabrik sub-extracts for safer processing."
            )
            for child in sorted(children, key=lambda item: item["id"]):
                child_id = child["id"]
                resolved.append(
                    {
                        "country": f"{country}:{child_id}",
                        "source_country": country,
                        "geofabrik_id": child_id,
                        "name": clean_geofabrik_name(child["name"]),
                        "url": child["urls"]["pbf"],
                    }
                )
        else:
            resolved.append(
                {
                    "country": country,
                    "source_country": country,
                    "geofabrik_id": geofabrik_id,
                    "name": clean_geofabrik_name(props["name"]),
                    "url": url,
                }
            )
    if missing:
        msg = ", ".join(f"{country}->{gid}" for country, gid in missing)
        raise ValueError(f"Could not resolve Geofabrik PBF URLs for: {msg}")
    return resolved


def calculate_raster_grid(
    bbox_lonlat: tuple[float, float, float, float],
    crs: CRS,
    resolution_m: float,
    max_cells: int = DEFAULT_MAX_RASTER_CELLS,
) -> tuple[tuple[int, int], tuple[float, float, float, float], np.ndarray]:
    lon_min, lat_min, lon_max, lat_max = bbox_lonlat
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = transformer.transform(
        [lon_min, lon_min, lon_max, lon_max],
        [lat_min, lat_max, lat_min, lat_max],
    )
    minx = math.floor(min(xs) / resolution_m) * resolution_m
    maxx = math.ceil(max(xs) / resolution_m) * resolution_m
    miny = math.floor(min(ys) / resolution_m) * resolution_m
    maxy = math.ceil(max(ys) / resolution_m) * resolution_m
    width = int(round((maxx - minx) / resolution_m))
    height = int(round((maxy - miny) / resolution_m))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster dimensions: {width} x {height}")
    cells = width * height
    if cells > max_cells:
        raise ValueError(
            f"Raster would contain {cells:,} cells, exceeding --max-raster-cells "
            f"({max_cells:,}). Increase resolution or explicitly raise the limit."
        )
    print(
        f"Raster grid: {width} x {height} cells "
        f"({cells / 1_000_000:.1f}M, {cells / 1024 / 1024:.1f} MiB as uint8)"
    )
    transform = np.array([resolution_m, 0.0, minx, 0.0, -resolution_m, maxy], dtype=np.float64)
    return (height, width), (minx, miny, maxx, maxy), transform


def build_raster_grid(
    bbox_lonlat: tuple[float, float, float, float],
    crs: CRS,
    resolution_m: float,
    max_cells: int = DEFAULT_MAX_RASTER_CELLS,
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]:
    shape, raster_bounds, transform = calculate_raster_grid(
        bbox_lonlat=bbox_lonlat,
        crs=crs,
        resolution_m=resolution_m,
        max_cells=max_cells,
    )
    height, width = shape
    raster = np.zeros((height, width), dtype=np.uint8)
    return raster, raster_bounds, transform


def open_raster_memmap(
    raster_path: Path,
    shape: tuple[int, int],
    resume: bool,
) -> np.memmap:
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(shape[0]) * int(shape[1])
    should_initialize = True
    if resume and raster_path.exists():
        actual_bytes = raster_path.stat().st_size
        if actual_bytes == expected_bytes:
            should_initialize = False
            print(f"Reusing work raster {raster_path}")
        else:
            print(
                f"Work raster size mismatch ({actual_bytes} bytes); "
                "reinitializing it for the current grid."
            )

    if should_initialize:
        print(f"Initializing work raster {raster_path} ({expected_bytes / 1024 / 1024:.1f} MiB)")
        with raster_path.open("wb") as f:
            f.truncate(expected_bytes)

    return np.memmap(raster_path, dtype=np.uint8, mode="r+", shape=shape)


def osm_where_clause(highways: Iterable[str]) -> str:
    quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in highways)
    return f"highway IN ({quoted})"


def should_skip_access(other_tags: str | None, skip_private: bool) -> bool:
    if not skip_private or not other_tags:
        return False
    tags = str(other_tags)
    blocked_values = ("no", "private")
    for key in ("access", "motor_vehicle", "motorcar", "vehicle"):
        for value in blocked_values:
            if f'"{key}"=>"{value}"' in tags:
                return True
    return False


def iter_arrow_batches(
    path: Path,
    layer: str,
    where: str,
    bbox: tuple[float, float, float, float],
    batch_size: int,
):
    with pyogrio.open_arrow(
        path,
        layer=layer,
        columns=["highway", "other_tags"],
        where=where,
        bbox=bbox,
        batch_size=batch_size,
        use_pyarrow=True,
    ) as (metadata, reader):
        if not metadata.get("fields") is None and "highway" not in metadata["fields"]:
            return
        for batch in reader:
            yield batch


def burn_geometry(
    raster: np.ndarray,
    geom,
    transformer: Transformer,
    raster_bounds: tuple[float, float, float, float],
    resolution_m: float,
) -> int:
    if geom is None or shapely.is_empty(geom):
        return 0

    _, _, _, maxy = raster_bounds
    burned = 0
    geom_type = shapely.get_type_id(geom)

    if geom_type == 1:  # LineString
        parts = (geom,)
    elif geom_type == 5:  # MultiLineString
        parts = shapely.get_parts(geom)
    else:
        return 0

    for part in parts:
        coords = shapely.get_coordinates(part)
        if coords.shape[0] < 2:
            continue
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        cols = np.floor((np.asarray(x) - raster_bounds[0]) / resolution_m).astype(np.int64)
        rows = np.floor((maxy - np.asarray(y)) / resolution_m).astype(np.int64)
        burned += _burn_polyline(raster, rows, cols)
    return burned


def rasterize_extract(
    raster: np.ndarray,
    pbf_path: Path,
    bbox_lonlat: tuple[float, float, float, float],
    raster_bounds: tuple[float, float, float, float],
    resolution_m: float,
    crs: CRS,
    highways: tuple[str, ...],
    batch_size: int,
    skip_private: bool,
) -> dict[str, int]:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    where = osm_where_clause(highways)
    stats = {"features": 0, "burned_cells": 0}

    for layer in ("lines", "multilinestrings"):
        try:
            info = pyogrio.read_info(pbf_path, layer=layer)
        except Exception:
            continue
        if "highway" not in info.get("fields", []):
            continue

        print(f"  reading layer '{layer}'")
        layer_features = 0
        layer_burned = 0
        for batch in iter_arrow_batches(pbf_path, layer, where, bbox_lonlat, batch_size):
            names = batch.schema.names
            wkb_col = "wkb_geometry" if "wkb_geometry" in names else names[-1]
            other_tags_col = batch.column("other_tags").to_pylist() if "other_tags" in names else [None] * batch.num_rows
            geometries = shapely.from_wkb(batch.column(wkb_col).to_numpy(zero_copy_only=False))

            for geom, other_tags in zip(geometries, other_tags_col):
                if should_skip_access(other_tags, skip_private):
                    continue
                layer_features += 1
                layer_burned += burn_geometry(raster, geom, transformer, raster_bounds, resolution_m)

            if layer_features and layer_features % (batch_size * 10) == 0:
                print(
                    f"    {layer_features:,} features, "
                    f"{layer_burned:,} new raster cells in this layer"
                )

        print(f"  {layer}: {layer_features:,} features, {layer_burned:,} new cells")
        stats["features"] += layer_features
        stats["burned_cells"] += layer_burned

    return stats


def save_npz(
    output_path: Path,
    raster: np.ndarray,
    transform: np.ndarray,
    crs: CRS,
    bbox_lonlat: tuple[float, float, float, float],
    raster_bounds: tuple[float, float, float, float],
    resolution_m: float,
    extracts: list[dict],
    highways: tuple[str, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving NPZ raster to {output_path}")
    np.savez_compressed(
        output_path,
        road_presence=raster,
        transform=transform,
        crs_wkt=crs.to_wkt(),
        bbox_lonlat=np.array(bbox_lonlat, dtype=np.float64),
        raster_bounds=np.array(raster_bounds, dtype=np.float64),
        resolution_m=np.array(resolution_m, dtype=np.float64),
        countries=np.array([item["country"] for item in extracts], dtype=str),
        geofabrik_ids=np.array([item["geofabrik_id"] for item in extracts], dtype=str),
        source_urls=np.array([item["url"] for item in extracts], dtype=str),
        highways=np.array(highways, dtype=str),
    )


def save_netcdf(
    output_path: Path,
    raster: np.ndarray,
    raster_bounds: tuple[float, float, float, float],
    resolution_m: float,
    crs: CRS,
    bbox_lonlat: tuple[float, float, float, float],
) -> None:
    if xr is None:
        print("xarray is not installed; skipping NetCDF output")
        return
    minx, _, maxx, maxy = raster_bounds
    x = minx + (np.arange(raster.shape[1], dtype=np.float64) + 0.5) * resolution_m
    y = maxy - (np.arange(raster.shape[0], dtype=np.float64) + 0.5) * resolution_m
    data = xr.Dataset(
        {
            "road_presence": (
                ("y", "x"),
                raster,
                {
                    "long_name": "unsmoothed OSM drivable road presence",
                    "description": "1 means at least one selected OSM highway crosses the pixel.",
                },
            )
        },
        coords={"x": x, "y": y},
        attrs={
            "crs_wkt": crs.to_wkt(),
            "resolution_m": float(resolution_m),
            "bbox_lonlat": json.dumps(bbox_lonlat),
        },
    )
    chunksizes = (min(512, raster.shape[0]), min(512, raster.shape[1]))
    encoding = {"road_presence": {"zlib": True, "complevel": 4, "chunksizes": chunksizes}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving NetCDF raster to {output_path}")
    data.to_netcdf(output_path, encoding=encoding)


def save_preview_png(output_path: Path, raster: np.ndarray, max_side: int = 2200) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = (raster * 255).astype(np.uint8)
    img = Image.fromarray(image, mode="L")
    scale = min(max_side / img.width, max_side / img.height, 1.0)
    if scale < 1.0:
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(size, resample=Image.Resampling.NEAREST)
    img.save(output_path)
    print(f"Saved preview PNG to {output_path}")


def write_manifest(output_path: Path, manifest: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def make_worker_preexec(memory_gb: float | None, nice: int):
    def _preexec() -> None:
        if nice:
            os.nice(nice)
        if memory_gb and memory_gb > 0:
            try:
                import resource

                limit = int(memory_gb * 1024**3)
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except Exception as exc:  # pragma: no cover - platform dependent
                print(f"Warning: failed to set worker memory limit: {exc}", file=sys.stderr)

    return _preexec


def terminate_process_group(proc: subprocess.Popen, grace_sec: int = 30) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def cleanup_worker_temp_files(log_dir: Path) -> None:
    for temp_path in log_dir.glob("osm_tmp_nodes_*"):
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def check_pbf_size(path: Path, max_size_mb: float, allow_large: bool) -> None:
    if allow_large or max_size_mb <= 0:
        return
    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > max_size_mb:
        raise RuntimeError(
            f"{path} is {size_mb:.1f} MB, larger than --max-pbf-size-mb={max_size_mb:.1f}. "
            "This is intentionally blocked in safe mode. Use regional Geofabrik extracts "
            "or pass --allow-large-pbf if you really want to try it."
        )


def run_worker_for_extract(
    extract: dict,
    raster_path: Path,
    raster_shape: tuple[int, int],
    bbox_lonlat: tuple[float, float, float, float],
    raster_bounds: tuple[float, float, float, float],
    resolution_m: float,
    crs: str,
    highways: tuple[str, ...],
    batch_size: int,
    skip_private: bool,
    timeout_sec: int,
    memory_gb: float,
    nice: int,
    log_dir: Path,
) -> dict:
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stats_path = log_dir / f"{extract['geofabrik_id']}.stats.json"
    log_path = log_dir / f"{extract['geofabrik_id']}.log"
    stats_path.unlink(missing_ok=True)
    worker_extract = dict(extract)
    worker_extract["local_path"] = str(Path(extract["local_path"]).resolve())
    raster_path = Path(raster_path).resolve()

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-extract-json",
        json.dumps(worker_extract),
        "--worker-raster-path",
        str(raster_path),
        "--worker-raster-shape",
        str(raster_shape[0]),
        str(raster_shape[1]),
        "--worker-raster-bounds",
        *(str(value) for value in raster_bounds),
        "--worker-stats-output",
        str(stats_path),
        "--bbox",
        *(str(value) for value in bbox_lonlat),
        "--resolution-m",
        str(resolution_m),
        "--crs",
        crs,
        "--batch-size",
        str(batch_size),
    ]
    if skip_private:
        command.append("--skip-private-access")
    command.extend(["--highways", *highways])

    env = os.environ.copy()
    env.update(SAFE_WORKER_ENV)
    env["CPL_TMPDIR"] = str(log_dir)
    env["TMPDIR"] = str(log_dir)
    print(
        f"Rasterizing {extract['country']} ({extract['name']}) in guarded worker; "
        f"log: {log_path}"
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(log_dir),
            start_new_session=True,
            preexec_fn=make_worker_preexec(memory_gb, nice),
        )
        try:
            returncode = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            terminate_process_group(proc)
            cleanup_worker_temp_files(log_dir)
            return {
                "status": "timeout",
                "features": 0,
                "burned_cells": 0,
                "message": f"Worker exceeded timeout of {timeout_sec} seconds.",
                "log_path": str(log_path),
            }

    if returncode != 0:
        cleanup_worker_temp_files(log_dir)
        return {
            "status": "failed",
            "features": 0,
            "burned_cells": 0,
            "message": f"Worker exited with code {returncode}.",
            "log_path": str(log_path),
        }
    if not stats_path.exists():
        cleanup_worker_temp_files(log_dir)
        return {
            "status": "failed",
            "features": 0,
            "burned_cells": 0,
            "message": "Worker completed without writing stats.",
            "log_path": str(log_path),
        }

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    cleanup_worker_temp_files(log_dir)
    stats["log_path"] = str(log_path)
    return stats


def worker_main(args: argparse.Namespace) -> int:
    extract = json.loads(args.worker_extract_json)
    raster_shape = tuple(args.worker_raster_shape)
    raster_bounds = tuple(args.worker_raster_bounds)
    bbox_lonlat = tuple(args.bbox)
    raster = np.memmap(args.worker_raster_path, dtype=np.uint8, mode="r+", shape=raster_shape)
    stats = rasterize_extract(
        raster=raster,
        pbf_path=Path(extract["local_path"]),
        bbox_lonlat=bbox_lonlat,
        raster_bounds=raster_bounds,
        resolution_m=args.resolution_m,
        crs=CRS.from_user_input(args.crs),
        highways=tuple(args.highways),
        batch_size=args.batch_size,
        skip_private=args.skip_private_access,
    )
    raster.flush()
    stats.update(
        {
            "status": "ok",
            "geofabrik_id": extract["geofabrik_id"],
            "country": extract["country"],
        }
    )
    args.worker_stats_output.parent.mkdir(parents=True, exist_ok=True)
    with args.worker_stats_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/features_config_30d.yaml"))
    parser.add_argument("--countries-key", default="prediction_countries")
    parser.add_argument("--countries", nargs="*", help="Override countries from config.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Raster/download filter bbox in lon/lat. Defaults to config coordinate_bounds.",
    )
    parser.add_argument("--resolution-m", type=float, default=1000.0)
    parser.add_argument("--crs", default="EPSG:3857")
    parser.add_argument("--osm-dir", type=Path, default=Path("data/osm/pbf"))
    parser.add_argument("--index-path", type=Path, default=Path("data/osm/geofabrik-index-v1-nogeom.json"))
    parser.add_argument("--output", type=Path, default=Path("data/land_features/osm_drivable_roads_1km.npz"))
    parser.add_argument("--netcdf-output", type=Path, default=Path("data/land_features/osm_drivable_roads_1km.nc"))
    parser.add_argument("--preview-output", type=Path, default=Path("data/land_features/osm_drivable_roads_1km_preview.png"))
    parser.add_argument("--manifest-output", type=Path, default=Path("data/land_features/osm_drivable_roads_1km_manifest.json"))
    parser.add_argument("--work-raster", type=Path, help="Temporary uint8 memmap raster path.")
    parser.add_argument("--log-dir", type=Path, default=Path("data/land_features/osm_roads_raster_logs"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--worker-timeout-sec", type=int, default=DEFAULT_WORKER_TIMEOUT_SEC)
    parser.add_argument("--worker-memory-gb", type=float, default=DEFAULT_WORKER_MEMORY_GB)
    parser.add_argument("--worker-nice", type=int, default=10)
    parser.add_argument("--max-raster-cells", type=int, default=DEFAULT_MAX_RASTER_CELLS)
    parser.add_argument("--max-pbf-size-mb", type=float, default=DEFAULT_MAX_PBF_SIZE_MB)
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-netcdf", action="store_true")
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument("--skip-private-access", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-expand-parent-extracts", action="store_true")
    parser.add_argument("--allow-large-pbf", action="store_true")
    parser.add_argument("--continue-on-worker-failure", action="store_true")
    parser.add_argument("--unsafe-in-process", action="store_true")
    parser.add_argument("--highways", nargs="*", default=list(DEFAULT_DRIVABLE_HIGHWAYS))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-extract-json", help=argparse.SUPPRESS)
    parser.add_argument("--worker-raster-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-raster-shape", nargs=2, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-raster-bounds", nargs=4, type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-stats-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)

    config_path = args.config
    countries = args.countries or load_config_countries(config_path, args.countries_key)
    bbox = tuple(args.bbox) if args.bbox else parse_coordinate_bounds(config_path)
    if bbox is None:
        raise ValueError("Provide --bbox or coordinate_bounds in config")

    crs = CRS.from_user_input(args.crs)
    highways = tuple(args.highways)
    index = load_geofabrik_index(args.index_path, refresh=args.refresh_index)
    extracts = resolve_extracts(
        index,
        countries,
        expand_parents=not args.no_expand_parent_extracts,
    )
    if args.dry_run:
        print(f"Planned extracts: {len(extracts)}")
        for extract in extracts:
            print(f"  {extract['geofabrik_id']:32s} {extract['country']} -> {extract['url']}")
        return 0

    previous_manifest = {} if args.no_resume else load_manifest(args.manifest_output)
    country_stats = previous_manifest.get("country_stats", {}) if previous_manifest else {}
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bbox_lonlat": bbox,
        "resolution_m": args.resolution_m,
        "crs": args.crs,
        "highways": highways,
        "skip_private_access": bool(args.skip_private_access),
        "safe_mode": {
            "worker_timeout_sec": args.worker_timeout_sec,
            "worker_memory_gb": args.worker_memory_gb,
            "worker_nice": args.worker_nice,
            "max_raster_cells": args.max_raster_cells,
            "max_pbf_size_mb": args.max_pbf_size_mb,
            "expand_parent_extracts": not args.no_expand_parent_extracts,
        },
        "extracts": extracts,
        "country_stats": country_stats,
    }

    for extract in extracts:
        pbf_path = args.osm_dir / f"{extract['geofabrik_id']}-latest.osm.pbf"
        extract["local_path"] = str(pbf_path.resolve())
        if not args.skip_download and not pbf_path.exists():
            download_file(extract["url"], pbf_path)
        elif pbf_path.exists():
            print(f"Using existing {pbf_path}")
        else:
            raise FileNotFoundError(f"Missing {pbf_path}; rerun without --skip-download")
        check_pbf_size(pbf_path, args.max_pbf_size_mb, args.allow_large_pbf)

    write_manifest(args.manifest_output, manifest)
    if args.download_only:
        print("Download-only mode complete.")
        return 0

    raster_shape, raster_bounds, transform = calculate_raster_grid(
        bbox_lonlat=bbox,
        crs=crs,
        resolution_m=args.resolution_m,
        max_cells=args.max_raster_cells,
    )
    work_raster_path = args.work_raster or args.output.with_suffix(args.output.suffix + ".work.uint8")
    resume = not args.no_resume
    expected_raster_bytes = int(raster_shape[0]) * int(raster_shape[1])
    can_reuse_work_raster = (
        resume
        and work_raster_path.exists()
        and work_raster_path.stat().st_size == expected_raster_bytes
    )
    if country_stats and not can_reuse_work_raster:
        print("Existing manifest stats found, but the matching work raster is absent; restarting rasterization.")
        country_stats = {}
        manifest["country_stats"] = country_stats
    raster = open_raster_memmap(work_raster_path, raster_shape, resume=resume)
    raster.flush()

    for extract in extracts:
        geofabrik_id = extract["geofabrik_id"]
        previous_stats = country_stats.get(geofabrik_id)
        if resume and previous_stats and previous_stats.get("status") == "ok":
            print(f"Skipping {geofabrik_id}; already completed in manifest.")
            continue

        if args.unsafe_in_process:
            print(f"Rasterizing {extract['country']} ({extract['name']}) in current process")
            stats = rasterize_extract(
                raster=raster,
                pbf_path=Path(extract["local_path"]),
                bbox_lonlat=bbox,
                raster_bounds=raster_bounds,
                resolution_m=args.resolution_m,
                crs=crs,
                highways=highways,
                batch_size=args.batch_size,
                skip_private=args.skip_private_access,
            )
            stats.update({"status": "ok", "geofabrik_id": geofabrik_id, "country": extract["country"]})
            raster.flush()
        else:
            stats = run_worker_for_extract(
                extract=extract,
                raster_path=work_raster_path,
                raster_shape=raster_shape,
                bbox_lonlat=bbox,
                raster_bounds=raster_bounds,
                resolution_m=args.resolution_m,
                crs=args.crs,
                highways=highways,
                batch_size=args.batch_size,
                skip_private=args.skip_private_access,
                timeout_sec=args.worker_timeout_sec,
                memory_gb=args.worker_memory_gb,
                nice=args.worker_nice,
                log_dir=args.log_dir,
            )

        country_stats[geofabrik_id] = stats
        manifest["country_stats"] = country_stats
        manifest["raster_bounds"] = raster_bounds
        manifest["transform"] = transform.tolist()
        manifest["work_raster_path"] = str(work_raster_path)
        write_manifest(args.manifest_output, manifest)

        if stats.get("status") != "ok":
            print(f"Worker did not finish {geofabrik_id}: {stats.get('message')}")
            if not args.continue_on_worker_failure:
                print("Stopping safely. Rerun with the same command after changing limits to resume.")
                return 2

        print(
            f"Finished {geofabrik_id}: {stats.get('features', 0):,} features, "
            f"{stats.get('burned_cells', 0):,} newly occupied cells"
        )

    manifest["raster_bounds"] = raster_bounds
    manifest["transform"] = transform.tolist()
    manifest["occupied_cells"] = int(raster.sum())
    manifest["country_stats"] = country_stats
    manifest["work_raster_path"] = str(work_raster_path)
    write_manifest(args.manifest_output, manifest)

    save_npz(args.output, raster, transform, crs, bbox, raster_bounds, args.resolution_m, extracts, highways)
    if not args.skip_netcdf:
        save_netcdf(args.netcdf_output, raster, raster_bounds, args.resolution_m, crs, bbox)
    if not args.skip_preview:
        save_preview_png(args.preview_output, raster)

    print(f"Occupied road cells: {int(raster.sum()):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
