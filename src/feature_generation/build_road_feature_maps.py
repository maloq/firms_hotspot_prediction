"""Build whole-area road feature maps from the OSM drivable road raster.

The output is a directory of ``.npy`` arrays plus ``manifest.json``. Arrays are
kept uncompressed so feature generation can mmap them and sample only requested
cells instead of reading multi-GB maps into RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _set_memory_limit(memory_limit_gb: float | None) -> None:
    if memory_limit_gb is None or memory_limit_gb <= 0:
        return
    limit = int(memory_limit_gb * 1024**3)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    new_hard = hard if hard != resource.RLIM_INFINITY else limit
    resource.setrlimit(resource.RLIMIT_AS, (limit, new_hard))
    print(f"Set address-space memory limit to {limit / 1024**3:g} GB")


def _kernel_label(kernel_km: float) -> str:
    if float(kernel_km).is_integer():
        return f"{int(kernel_km)}km"
    return f"{str(kernel_km).replace('.', 'p')}km"


def _copy_array_to_npy(
    array: np.ndarray,
    path: Path,
    *,
    dtype: np.dtype,
    chunk_rows: int,
) -> np.ndarray:
    out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=array.shape)
    for row0 in range(0, array.shape[0], chunk_rows):
        row1 = min(array.shape[0], row0 + chunk_rows)
        out[row0:row1] = array[row0:row1].astype(dtype, copy=False)
    out.flush()
    return out


def _load_source_raster(input_path: Path) -> tuple[np.ndarray, dict]:
    with np.load(input_path) as data:
        if "road_presence" not in data:
            raise KeyError(f"{input_path} does not contain 'road_presence'")
        road = data["road_presence"].astype(np.uint8, copy=False)
        metadata = {
            "shape": list(road.shape),
            "transform": data["transform"].astype(float).tolist(),
            "crs_wkt": str(data["crs_wkt"].item() if data["crs_wkt"].shape == () else data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
            "bbox_lonlat": data["bbox_lonlat"].astype(float).tolist()
            if "bbox_lonlat" in data
            else None,
            "raster_bounds": data["raster_bounds"].astype(float).tolist()
            if "raster_bounds" in data
            else None,
            "source_highways": data["highways"].astype(str).tolist()
            if "highways" in data
            else None,
        }
    return road, metadata


def build_road_feature_maps(
    input_path: Path,
    output_dir: Path,
    kernels_km: list[float],
    *,
    overwrite: bool,
    memory_limit_gb: float | None,
    truncate: float,
    chunk_rows: int,
    keep_temp: bool,
) -> None:
    _set_memory_limit(memory_limit_gb)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; use --overwrite to rebuild")
        shutil.rmtree(output_dir)

    tmp_dir = output_dir.with_name(f"{output_dir.name}.partial")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print(f"Loading road raster from {input_path}")
    road, metadata = _load_source_raster(input_path)
    shape = tuple(metadata["shape"])
    resolution_m = float(metadata["resolution_m"])
    print(f"Road raster shape: {shape[0]:,} x {shape[1]:,}; occupied cells: {int(road.sum()):,}")

    feature_specs: dict[str, dict] = {}

    print("Saving road_presence_1km mmap")
    road_path = tmp_dir / "road_presence_1km.uint8.npy"
    _copy_array_to_npy(road, road_path, dtype=np.uint8, chunk_rows=chunk_rows)
    feature_specs["road_presence_1km"] = {
        "path": road_path.name,
        "dtype": "uint8",
        "units": "0/1",
        "long_name": "OSM drivable road presence in 1 km cell",
    }

    for kernel_km in kernels_km:
        sigma_pixels = kernel_km * 1000.0 / resolution_m
        feature_name = f"road_density_gaussian_{_kernel_label(kernel_km)}"
        density_path = tmp_dir / f"{feature_name}.float32.npy"
        density = np.lib.format.open_memmap(
            density_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        print(
            f"Building {feature_name}: sigma={kernel_km:g} km "
            f"({sigma_pixels:.2f} pixels)"
        )
        gaussian_filter(
            road,
            sigma=sigma_pixels,
            output=density,
            mode="constant",
            cval=0.0,
            truncate=truncate,
        )
        density.flush()
        feature_specs[feature_name] = {
            "path": density_path.name,
            "dtype": "float32",
            "units": "road-cell fraction",
            "kernel": "gaussian",
            "sigma_km": float(kernel_km),
            "truncate": float(truncate),
        }

    distance_path = tmp_dir / "distance_to_road_meters.float32.npy"
    if road.sum() == 0:
        print("No road cells found; writing NaN distance map")
        distance = np.lib.format.open_memmap(
            distance_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        distance[:] = np.nan
        distance.flush()
    else:
        print("Building distance_to_road_meters")
        mask = road == 0
        temp_distance_path = tmp_dir / "distance_to_road_meters.float64.tmp.npy"
        distance64 = np.lib.format.open_memmap(
            temp_distance_path,
            mode="w+",
            dtype=np.float64,
            shape=shape,
        )
        distance_transform_edt(
            mask,
            sampling=resolution_m,
            return_distances=True,
            return_indices=False,
            distances=distance64,
        )
        distance64.flush()

        distance = np.lib.format.open_memmap(
            distance_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        for row0 in range(0, shape[0], chunk_rows):
            row1 = min(shape[0], row0 + chunk_rows)
            distance[row0:row1] = distance64[row0:row1].astype(np.float32, copy=False)
        distance.flush()
        del distance64
        if not keep_temp:
            temp_distance_path.unlink(missing_ok=True)

    feature_specs["distance_to_road_meters"] = {
        "path": distance_path.name,
        "dtype": "float32",
        "units": "meters",
        "long_name": "Euclidean distance to nearest OSM drivable road cell",
    }

    manifest = {
        "format": "road_feature_map_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_raster": str(input_path),
        "shape": metadata["shape"],
        "transform": metadata["transform"],
        "crs_wkt": metadata["crs_wkt"],
        "resolution_m": metadata["resolution_m"],
        "bbox_lonlat": metadata["bbox_lonlat"],
        "raster_bounds": metadata["raster_bounds"],
        "source_highways": metadata["source_highways"],
        "features": feature_specs,
    }
    with (tmp_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    tmp_dir.rename(output_dir)
    print(
        f"Saved road feature maps to {output_dir} in "
        f"{(time.time() - start) / 60.0:.1f} minutes"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/land_features/osm_drivable_roads_1km.npz"),
        help="Input raw road_presence NPZ from osm_roads_raster.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/land_features/osm_drivable_roads_features_1km"),
        help="Output directory for mmap feature maps.",
    )
    parser.add_argument("--kernels-km", nargs="+", type=float, default=[5.0, 10.0, 25.0])
    parser.add_argument("--truncate", type=float, default=4.0)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=16.0,
        help="Address-space limit. Use 0 to disable.",
    )
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_road_feature_maps(
        input_path=args.input,
        output_dir=args.output_dir,
        kernels_km=list(args.kernels_km),
        overwrite=args.overwrite,
        memory_limit_gb=args.memory_limit_gb,
        truncate=args.truncate,
        chunk_rows=args.chunk_rows,
        keep_temp=args.keep_temp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
