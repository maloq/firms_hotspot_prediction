"""Sample precomputed raster feature stores at model grid coordinates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


COORD_FEATURE_COLUMNS = {"lat", "lon", "latitude", "longitude", "lat_rounded", "lon_rounded"}
LEGACY_METADATA_KEYS = {
    "transform",
    "crs_wkt",
    "bbox_lonlat",
    "raster_bounds",
    "resolution_m",
    "countries",
    "geofabrik_ids",
    "source_urls",
    "highways",
    "kernels_km",
    "source_raster",
    "created_at_utc",
    "description",
}


def _as_scalar_string(value: np.ndarray | str) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value)
    return str(value)


def _load_transform(transform_data: np.ndarray | list[float] | tuple[float, ...]) -> tuple[float, float, float, float, float, float]:
    if isinstance(transform_data, np.ndarray):
        coeffs = transform_data.flatten()[:6]
    else:
        coeffs = list(transform_data)[:6]

    if len(coeffs) < 6:
        raise ValueError(f"Road feature transform must contain six coefficients, got {coeffs}")
    return tuple(float(v) for v in coeffs)


def _rowcol_from_transform(
    transform: tuple[float, float, float, float, float, float],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if det == 0:
        raise ValueError(f"Road feature affine transform is not invertible: {transform}")
    x_shift = x_coords - c
    y_shift = y_coords - f
    cols_float = (e * x_shift - b * y_shift) / det
    rows_float = (-d * x_shift + a * y_shift) / det
    return np.floor(rows_float).astype(np.int64), np.floor(cols_float).astype(np.int64)


def _load_feature_directory(
    feature_dir: Path,
) -> tuple[dict[str, np.ndarray], tuple[float, float, float, float, float, float], str, tuple[int, int]]:
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Road feature directory '{feature_dir}' does not contain manifest.json"
        )

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    transform = _load_transform(manifest["transform"])
    crs_wkt = str(manifest["crs_wkt"])
    shape = tuple(int(v) for v in manifest["shape"])

    features: dict[str, np.ndarray] = {}
    for feature_name, spec in manifest.get("features", {}).items():
        rel_path = spec["path"] if isinstance(spec, Mapping) else spec
        array_path = feature_dir / rel_path
        if not array_path.exists():
            raise FileNotFoundError(f"Road feature array missing: {array_path}")
        arr = np.load(array_path, mmap_mode="r")
        if tuple(arr.shape) != shape:
            raise ValueError(
                f"Road feature '{feature_name}' shape {arr.shape} does not match manifest {shape}"
            )
        features[feature_name] = arr

    if not features:
        raise ValueError(f"No feature arrays listed in {manifest_path}")

    return features, transform, crs_wkt, shape


def _load_feature_store(
    path: Path,
) -> tuple[dict[str, np.ndarray], tuple[float, float, float, float, float, float], str, tuple[int, int], object | None]:
    if path.is_dir():
        features, transform, crs_wkt, shape = _load_feature_directory(path)
        return features, transform, crs_wkt, shape, None

    npz = np.load(path)
    try:
        features, transform, crs_wkt, shape = _load_legacy_npz_from_handle(npz, path)
    except Exception:
        npz.close()
        raise
    return features, transform, crs_wkt, shape, npz


def _load_legacy_npz_from_handle(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
) -> tuple[dict[str, np.ndarray], tuple[float, float, float, float, float, float], str, tuple[int, int]]:
    if "transform" not in data or "crs_wkt" not in data:
        raise ValueError(f"Road feature NPZ is missing transform/crs_wkt metadata: {npz_path}")

    transform = _load_transform(data["transform"])
    crs_wkt = _as_scalar_string(data["crs_wkt"])

    features: dict[str, np.ndarray] = {}
    if "distance_meters" in data:
        features["distance_to_road_meters"] = data["distance_meters"]
    if "distance_to_road_meters" in data:
        features["distance_to_road_meters"] = data["distance_to_road_meters"]
    if "road_presence" in data:
        features["road_presence_1km"] = data["road_presence"]

    for key in data.files:
        if key in LEGACY_METADATA_KEYS or key in {"distance_meters", "distance_to_road_meters"}:
            continue
        if key.startswith(("density_", "road_density_")):
            features[key] = data[key]

    if not features:
        raise ValueError(
            f"Road feature NPZ '{npz_path}' does not contain distance/density feature arrays. "
            "Run src/feature_generation/build_road_feature_maps.py first."
        )

    first_shape = tuple(next(iter(features.values())).shape)
    for name, arr in features.items():
        if tuple(arr.shape) != first_shape:
            raise ValueError(f"Feature '{name}' has shape {arr.shape}, expected {first_shape}")

    return features, transform, crs_wkt, first_shape


def get_road_features_for_coords(
    coords: np.ndarray,
    npz_path: str = "data/land_features/osm_drivable_roads_features_1km",
) -> pd.DataFrame:
    """
    Extract precomputed road features for WGS84 latitude/longitude coordinates.

    ``npz_path`` may be either the legacy single-file NPZ or the newer directory
    store produced by ``build_road_feature_maps.py``. Directory stores are opened
    with mmap so training and prediction sample only requested cells.
    """
    feature_path = Path(npz_path)
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Road features data not found at {feature_path}")

    if coords.shape[0] != 2:
        raise ValueError("coords must be a 2xN array with latitudes in row 0 and longitudes in row 1")

    print(f"Loading raster features from {feature_path}...")
    npz_handle = None
    try:
        features, transform, crs_wkt, raster_shape, npz_handle = _load_feature_store(feature_path)
        feature_names = [name for name in features if name not in COORD_FEATURE_COLUMNS]
        print(f"Loaded {len(feature_names)} raster feature maps. Shape: {raster_shape}")

        lats = np.asarray(coords[0], dtype=np.float64)
        lons = np.asarray(coords[1], dtype=np.float64)
        n_points = coords.shape[1]

        source_crs = CRS("EPSG:4326")
        target_crs = CRS.from_wkt(crs_wkt)
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        x_coords, y_coords = transformer.transform(lons, lats)

        rows, cols = _rowcol_from_transform(transform, x_coords, y_coords)

        valid_mask = (
            (rows >= 0) & (rows < raster_shape[0]) &
            (cols >= 0) & (cols < raster_shape[1])
        )
        num_invalid = int(n_points - np.sum(valid_mask))
        if num_invalid:
            print(
                f"Warning: {num_invalid} raster feature coordinates fall outside the raster bounds; "
                "values will be NaN."
            )

        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        valid_positions = np.flatnonzero(valid_mask)
        if valid_rows.size:
            linear_order = np.argsort(valid_rows * raster_shape[1] + valid_cols, kind="stable")
        else:
            linear_order = np.empty(0, dtype=np.int64)
        results: dict[str, np.ndarray] = {"lat": lats.astype(np.float32), "lon": lons.astype(np.float32)}

        for feature_name in feature_names:
            feature_map = features[feature_name]
            values = np.full(n_points, np.nan, dtype=np.float32)
            if valid_rows.size:
                sampled = feature_map[
                    valid_rows[linear_order],
                    valid_cols[linear_order],
                ].astype(np.float32, copy=False)
                values[valid_positions[linear_order]] = sampled
            results[feature_name] = values

        df_results = pd.DataFrame(results)
        print(f"Created raster feature DataFrame with shape: {df_results.shape}")
        return df_results
    finally:
        if npz_handle is not None:
            npz_handle.close()
