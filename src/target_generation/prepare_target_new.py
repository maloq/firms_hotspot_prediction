import copy
import math
import os
from typing import Any, Sequence

import pandas as pd
import geopandas as gpd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.vectorized import contains
import yaml
from tqdm import tqdm
import datetime
from datetime import timedelta
import itertools
from pathlib import Path
import hashlib

from .stationary_points import DEFAULT_OUTPUT_DIR, drop_stationary_points

config_path = 'configs/target_config.yaml'
default_config = {
    'spatial_coarseness': 0.1,
    'brightness_threshold': 380,
    'confidence_threshold': 90,
    'brightness_threshold_high_lat': 360,  # Lower threshold for brightness in high-latitude areas
    'confidence_threshold_high_lat': 80,   # Lower threshold for confidence in high-latitude areas
    'samples_per_area_per_year': 10,
    'filter_stationary_points': True,
    'stationary_points_dir': str(DEFAULT_OUTPUT_DIR),
    'use_high_latitude_filter': True,  # Set to False to use standard thresholds for all latitudes
    'negative_sampling': {
        'strategy': 'legacy_area',
        'target_positive_fraction': 0.10,
        'stratum_weights': {
            'near_fire_hard': 0.30,
            'same_season': 0.20,
            'same_ecoregion': 0.20,
            'same_burnable_landcover': 0.20,
            'random_background': 0.10,
        },
        'exclude_positive_buffer_cells': 1,
        'exclude_positive_buffer_days': 1,
        'near_fire_min_cells': 2,
        'near_fire_max_cells': 5,
        'near_fire_day_window': 7,
        'match_month_for_static_strata': True,
        'max_attempt_multiplier': 80,
        'landseamask_land_threshold': 70,
        'use_ecoregion': True,
        'use_landcover': True,
        'burnable_feature_columns': [
            'forest_cover',
            'tree_cover',
            'tvl',
            'tvh',
            'type_of_low_vegetation',
            'type_of_high_vegetation',
        ],
        'landcover_match_columns': ['tvl', 'tvh', 'forest_cover', 'tree_cover'],
        'landcover_file_keywords': [
            'forest',
            'vegetation',
            'land_sea',
            'landsea',
            'landseamask',
            'type_of_high_vegetation',
            'type_of_low_vegetation',
        ],
    },
    'soft_labels': {
        'enabled': True,
        'column': 'soft_label',
        'max_negative_label': 0.35,
        'spatial_decay_cells': 2.0,
        'temporal_decay_days': 7.0,
        'max_distance_cells': 5,
        'max_delta_days': 7,
        'distance_column': 'nearest_positive_distance_cells',
        'delta_days_column': 'nearest_positive_delta_days',
    },
    'recent_fire_history': {
        'enabled': True,
        'radii_cells': [0, 1, 2],
        'min_lag_days': 30,
        'count_windows': [
            {'name': 'last_month', 'start_days': 30, 'end_days': 60},
            {'name': 'last_year', 'start_days': 30, 'end_days': 395},
        ],
        'include_days_since': False,
        'days_since_radii_cells': [],
        'days_since_cap': 365,
    },
}
try:
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    if config is None: # Handle empty config file
        print(f"Warning: Config file {config_path} is empty. Using default values.")
        config = default_config
except FileNotFoundError:
    print(f"Warning: Config file not found at {config_path}. Using default values.")
    config = default_config
except yaml.YAMLError as e:
    print(f"Error parsing config file {config_path}: {e}. Using default values.")
    config = default_config
except Exception as e:
    print(f"Unexpected error loading config file: {e}. Using default values.")
    config = default_config

SPATIAL_COARSENESS = config.get('spatial_coarseness', default_config['spatial_coarseness'])
BRIGHTNESS_THRESHOLD = config.get('brightness_threshold', default_config['brightness_threshold'])
CONFIDENCE_THRESHOLD = config.get('confidence_threshold', default_config['confidence_threshold'])
BRIGHTNESS_THRESHOLD_HIGH_LAT = config.get('brightness_threshold_high_lat', default_config['brightness_threshold_high_lat'])
CONFIDENCE_THRESHOLD_HIGH_LAT = config.get('confidence_threshold_high_lat', default_config['confidence_threshold_high_lat'])
SAMPLES_PER_AREA_PER_YEAR = config.get('samples_per_area_per_year', default_config['samples_per_area_per_year'])
FILTER_STATIONARY_POINTS = config.get('filter_stationary_points', default_config['filter_stationary_points'])
STATIONARY_POINTS_DIR = Path(config.get('stationary_points_dir', default_config['stationary_points_dir']))
USE_HIGH_LATITUDE_FILTER = config.get('use_high_latitude_filter', default_config['use_high_latitude_filter'])
GLOBAL_SEED = int(config.get('global_seed', 42))
NEGATIVE_SAMPLING_CONFIG = config.get('negative_sampling', default_config['negative_sampling'])
rounding_precision = int(-np.log10(SPATIAL_COARSENESS)) if SPATIAL_COARSENESS > 0 else 0

country_mapping = {
        'Russian_Federation': 'Russia',
        'United_Kingdom': 'United Kingdom',
        'Czech_Republic': 'Czechia',
        'Bosnia_and_Herzegovina': 'Bosnia and Herzegovina',
        'Serbia': 'Republic of Serbia',
        'Dem_Rep_Korea': 'North Korea',
        'Republic_of_Korea': 'South Korea',
        'Macedonia_Former_Yugoslav_Republic_of': 'North Macedonia',
        'Bosnia_and_Herzegovina': 'Bosnia and Herzegovina',
    }


def detect_confidence_scale(confidence: pd.Series | np.ndarray) -> str:
    """Detect whether FIRMS confidence values are stored as fractions or percentages."""

    values = pd.to_numeric(pd.Series(confidence), errors='coerce').dropna()
    if values.empty:
        return 'unknown'
    q95 = float(values.quantile(0.95))
    max_value = float(values.max())
    if q95 > 1.0 or max_value > 1.0:
        return 'percent_0_100'
    return 'fraction_0_1'


def normalize_confidence_threshold_for_series(
    confidence: pd.Series | np.ndarray,
    threshold: float,
) -> tuple[float, str]:
    """Put a configured confidence threshold on the same scale as FIRMS data.

    Historical configs in this project use both 0-1 thresholds (0.85) and
    0-100 thresholds (85/90). This helper prevents silently accepting nearly
    every detection when the data are percentages but the config is fractional.
    """

    scale = detect_confidence_scale(confidence)
    threshold_value = float(threshold)
    if scale == 'percent_0_100' and threshold_value <= 1.0:
        threshold_value *= 100.0
    elif scale == 'fraction_0_1' and threshold_value > 1.0:
        threshold_value /= 100.0
    return threshold_value, scale


def deterministic_negative_seed(global_seed: int, country: str, date_start, date_end, split_name: str = "target") -> int:
    text = f"{global_seed}|{country}|{date_start}|{date_end}|{split_name}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


STRATIFIED_NEGATIVE_STRATA = [
    "near_fire_hard",
    "same_season",
    "same_ecoregion",
    "same_burnable_landcover",
    "random_background",
]


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""

    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _negative_sampling_settings(feature_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return target negative-sampling settings plus data paths from the feature config."""

    settings = _deep_update(default_config["negative_sampling"], NEGATIVE_SAMPLING_CONFIG or {})
    feature_config = feature_config or {}
    land_params = feature_config.get("land_data_params") or {}

    if "country_shapes_path" not in settings and feature_config.get("country_shapes_path"):
        settings["country_shapes_path"] = feature_config["country_shapes_path"]
    settings.setdefault("country_shapes_path", feature_config.get("country_shapes_path", "data/countries"))

    if "wwf_shp_path" not in settings and land_params.get("wwf_shp_path"):
        settings["wwf_shp_path"] = land_params["wwf_shp_path"]
    settings.setdefault("wwf_shp_path", "data/wwf_terr_ecos")

    if "land_data_files" not in settings and land_params.get("land_data_files"):
        settings["land_data_files"] = land_params["land_data_files"]
    settings.setdefault("land_data_files", [])
    settings["land_data_files"] = list(settings.get("land_data_files") or [])

    landsea_mask_path = land_params.get("landsea_mask_path")
    if landsea_mask_path and landsea_mask_path not in settings["land_data_files"]:
        settings["land_data_files"] = list(settings["land_data_files"]) + [landsea_mask_path]

    return settings


def _target_section_settings(
    section_name: str,
    feature_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return merged target-generation settings for an optional config section."""

    settings = _deep_update(default_config.get(section_name, {}), config.get(section_name, {}) or {})
    feature_config = feature_config or {}
    if section_name in feature_config:
        settings = _deep_update(settings, feature_config.get(section_name) or {})
    return settings


def _desired_negative_count(positive_count: int, target_positive_fraction: float) -> int:
    """Number of negatives needed so positives are the requested final fraction."""

    positive_count = int(positive_count)
    target_positive_fraction = float(target_positive_fraction)
    if positive_count <= 0:
        return 0
    if not 0.0 < target_positive_fraction < 1.0:
        raise ValueError("negative_sampling.target_positive_fraction must be in (0, 1).")
    return int(math.ceil(positive_count * (1.0 - target_positive_fraction) / target_positive_fraction))


def _normalise_stratum_weights(weights: dict[str, Any] | None) -> dict[str, float]:
    weights = weights or {}
    cleaned = {
        stratum: max(0.0, float(weights.get(stratum, 0.0)))
        for stratum in STRATIFIED_NEGATIVE_STRATA
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        cleaned = copy.deepcopy(default_config["negative_sampling"]["stratum_weights"])
        total = sum(cleaned.values())
    return {key: value / total for key, value in cleaned.items()}


def _allocate_by_weights(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Deterministically allocate an integer total according to fractional weights."""

    total = int(total)
    if total <= 0:
        return {key: 0 for key in weights}
    raw = {key: total * float(value) for key, value in weights.items()}
    allocation = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(allocation.values())
    ranked = sorted(raw, key=lambda key: (raw[key] - allocation[key], weights[key]), reverse=True)
    for key in ranked[:remainder]:
        allocation[key] += 1
    return allocation


def _filter_to_coordinate_bounds(
    frame: pd.DataFrame,
    coordinate_bounds: tuple[float, float, float, float] | list[float] | None,
) -> pd.DataFrame:
    if coordinate_bounds is None or frame.empty:
        return frame
    min_lat, min_lon, max_lat, max_lon = (float(value) for value in coordinate_bounds)
    mask = (
        frame["lat_rounded"].between(min_lat, max_lat)
        & frame["lon_rounded"].between(min_lon, max_lon)
    )
    return frame.loc[mask].reset_index(drop=True)


def _coordinate_keys(frame: pd.DataFrame, resolution: float = SPATIAL_COARSENESS) -> pd.DataFrame:
    scale = 1.0 / float(resolution)
    return pd.DataFrame(
        {
            "_lat_key": np.rint(pd.to_numeric(frame["lat_rounded"], errors="coerce").to_numpy(dtype=float) * scale).astype(np.int64),
            "_lon_key": np.rint(pd.to_numeric(frame["lon_rounded"], errors="coerce").to_numpy(dtype=float) * scale).astype(np.int64),
        },
        index=frame.index,
    )


def _normalise_group_value(value: Any) -> Any:
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalise_group_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return tuple(_normalise_group_value(item) for item in value)
    return (_normalise_group_value(value),)


def _country_geometry(world: gpd.GeoDataFrame, country: str):
    mapped_country_name = country_mapping.get(country, country)
    name_columns = [col for col in ["SOVEREIGNT", "ADMIN", "NAME", "NAME_EN"] if col in world.columns]
    mask = np.zeros(len(world), dtype=bool)
    for col in name_columns:
        values = world[col].astype(str)
        mask |= values.eq(mapped_country_name).to_numpy()
        mask |= values.eq(country).to_numpy()
    matches = world.loc[mask]
    if matches.empty:
        raise ValueError(f"Fatal: Geometry for '{mapped_country_name}' not found.")
    country_geom = matches.geometry.unary_union
    if not country_geom.is_valid:
        print(f"Warning: Invalid geometry for {mapped_country_name}, attempting buffer(0).")
        country_geom = country_geom.buffer(0)
        if not country_geom.is_valid:
            raise ValueError(f"Fatal: Unfixable invalid geometry for '{mapped_country_name}'.")
    return country_geom


def _candidate_cells_for_country(
    country_geom,
    country: str,
    coordinate_bounds: tuple[float, float, float, float] | list[float] | None,
    resolution: float,
) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = country_geom.bounds
    if coordinate_bounds is not None:
        b_min_lat, b_min_lon, b_max_lat, b_max_lon = (float(value) for value in coordinate_bounds)
        min_lat = max(min_lat, b_min_lat)
        max_lat = min(max_lat, b_max_lat)
        min_lon = max(min_lon, b_min_lon)
        max_lon = min(max_lon, b_max_lon)
        if min_lat > max_lat or min_lon > max_lon:
            return pd.DataFrame(columns=["lat_rounded", "lon_rounded", "country"])

    lat = np.round(
        np.arange(
            np.floor(min_lat / resolution) * resolution,
            np.ceil(max_lat / resolution) * resolution + resolution / 2.0,
            resolution,
        ),
        10,
    )
    lon = np.round(
        np.arange(
            np.floor(min_lon / resolution) * resolution,
            np.ceil(max_lon / resolution) * resolution + resolution / 2.0,
            resolution,
        ),
        10,
    )
    if len(lat) == 0 or len(lon) == 0:
        return pd.DataFrame(columns=["lat_rounded", "lon_rounded", "country"])

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask = contains(country_geom, lon_grid, lat_grid)
    cells = pd.DataFrame(
        {
            "lat_rounded": lat_grid[mask].astype(np.float32),
            "lon_rounded": lon_grid[mask].astype(np.float32),
            "country": country,
        }
    )
    return cells.drop_duplicates(["lat_rounded", "lon_rounded", "country"]).reset_index(drop=True)


def _candidate_cells_for_countries(
    countries: list[str],
    *,
    country_shapes_path: str,
    coordinate_bounds: tuple[float, float, float, float] | list[float] | None,
    resolution: float = SPATIAL_COARSENESS,
) -> pd.DataFrame:
    print(f"\nBuilding stratified negative candidate grid from '{country_shapes_path}'...")
    world = gpd.read_file(country_shapes_path)
    parts = []
    for country in countries:
        geom = _country_geometry(world, country)
        cells = _candidate_cells_for_country(geom, country, coordinate_bounds, resolution)
        print(f"Candidate cells for {country}: {len(cells)}")
        if not cells.empty:
            parts.append(cells)
    if not parts:
        raise ValueError("No candidate grid cells available for stratified negative sampling.")
    cells = pd.concat(parts, ignore_index=True).drop_duplicates(["lat_rounded", "lon_rounded", "country"])
    return cells.reset_index(drop=True)


def _existing_paths(paths: list[str]) -> list[str]:
    existing = []
    for path in paths or []:
        if path and os.path.exists(path):
            existing.append(path)
        elif path:
            print(f"Warning: static stratum file not found and will be skipped: {path}")
    return existing


def _landcover_relevant_paths(paths: list[str], settings: dict[str, Any]) -> list[str]:
    keywords = [str(item).lower() for item in settings.get("landcover_file_keywords", [])]
    if not keywords:
        return paths
    relevant = []
    for path in paths:
        path_lower = os.path.basename(str(path)).lower()
        if any(keyword in path_lower for keyword in keywords):
            relevant.append(path)
    skipped = sorted(set(paths) - set(relevant))
    if skipped:
        print(
            "Skipping static files not needed for burnable/land-cover strata: "
            + ", ".join(os.path.basename(path) for path in skipped)
        )
    return relevant


def _enrich_cells_with_static_strata(cells: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Add ecoregion and burnable/land-cover fields to candidate cells when available."""

    enriched = cells.reset_index(drop=True).copy()

    if bool(settings.get("use_ecoregion", True)):
        wwf_path = settings.get("wwf_shp_path")
        if wwf_path and os.path.exists(wwf_path):
            try:
                from src.feature_generation.prepare_land import assign_ecoregion

                eco_df, eco_names = assign_ecoregion(enriched[["lat_rounded", "lon_rounded"]], wwf_shp=wwf_path)
                for name in eco_names:
                    enriched[name] = eco_df[name].reset_index(drop=True)
                print(f"Added ecoregion strata from {wwf_path}.")
            except Exception as exc:
                print(f"Warning: failed to add ecoregion strata: {exc}")
        else:
            print(f"Warning: ecoregion path unavailable; same_ecoregion will fall back to country strata: {wwf_path}")

    if "ecoregion_name" not in enriched.columns:
        enriched["ecoregion_name"] = "Unknown"
    if "ecoregion_realm" not in enriched.columns:
        enriched["ecoregion_realm"] = "Unknown"

    land_files = _existing_paths(_landcover_relevant_paths(list(settings.get("land_data_files") or []), settings))
    if bool(settings.get("use_landcover", True)) and land_files:
        try:
            from src.feature_generation.prepare_land import prepare_land_data

            land_df, land_names = prepare_land_data(
                land_data_files=land_files,
                target_df=enriched[["lat_rounded", "lon_rounded"]],
                radius_meters=None,
            )
            for name in land_names:
                if name in land_df.columns:
                    enriched[name] = land_df[name].reset_index(drop=True)
            print(f"Added land-cover/burnable strata from {len(land_files)} static file(s).")
        except Exception as exc:
            print(f"Warning: failed to add land-cover/burnable strata: {exc}")
    elif bool(settings.get("use_landcover", True)):
        print("Warning: no static land-cover files available; burnable stratum will use land-only fallback.")

    land_threshold = float(settings.get("landseamask_land_threshold", 70))
    burnable = np.ones(len(enriched), dtype=bool)
    if "landseamask" in enriched.columns:
        burnable &= pd.to_numeric(enriched["landseamask"], errors="coerce").fillna(100).to_numpy(dtype=float) < land_threshold

    burnable_cols = [col for col in settings.get("burnable_feature_columns", []) if col in enriched.columns]
    if burnable_cols:
        vegetation_like = np.zeros(len(enriched), dtype=bool)
        for col in burnable_cols:
            values = pd.to_numeric(enriched[col], errors="coerce")
            vegetation_like |= values.fillna(0).to_numpy(dtype=float) > 0
        burnable &= vegetation_like
    enriched["burnable"] = burnable

    match_cols = [col for col in settings.get("landcover_match_columns", []) if col in enriched.columns]
    if match_cols:
        enriched["landcover_key"] = [
            "|".join(str(_normalise_group_value(value)) for value in row)
            for row in enriched[match_cols].itertuples(index=False, name=None)
        ]
    else:
        enriched["landcover_key"] = np.where(enriched["burnable"], "burnable", "nonburnable")

    return enriched


def _merge_positive_static_strata(positives: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    pos = positives.copy()
    pos_keys = _coordinate_keys(pos)
    pos["_lat_key"] = pos_keys["_lat_key"].to_numpy()
    pos["_lon_key"] = pos_keys["_lon_key"].to_numpy()
    cell_keys = _coordinate_keys(cells)
    cell_info = cells.copy()
    cell_info["_lat_key"] = cell_keys["_lat_key"].to_numpy()
    cell_info["_lon_key"] = cell_keys["_lon_key"].to_numpy()
    static_cols = [
        col for col in ["ecoregion_name", "ecoregion_realm", "burnable", "landcover_key"]
        if col in cell_info.columns
    ]
    merged = pos.merge(
        cell_info[["country", "_lat_key", "_lon_key"] + static_cols].drop_duplicates(["country", "_lat_key", "_lon_key"]),
        on=["country", "_lat_key", "_lon_key"],
        how="left",
    )
    merged["ecoregion_name"] = merged.get("ecoregion_name", pd.Series("Unknown", index=merged.index)).fillna("Unknown")
    merged["ecoregion_realm"] = merged.get("ecoregion_realm", pd.Series("Unknown", index=merged.index)).fillna("Unknown")
    merged["burnable"] = merged.get("burnable", pd.Series(True, index=merged.index)).fillna(True).astype(bool)
    merged["landcover_key"] = merged.get("landcover_key", pd.Series("burnable", index=merged.index)).fillna("burnable")
    return merged


def _build_cell_groups(cells: pd.DataFrame, columns: list[str]) -> dict[tuple[Any, ...], np.ndarray]:
    groups: dict[tuple[Any, ...], np.ndarray] = {}
    if not columns:
        return groups
    for key, idx in cells.groupby(columns, dropna=False, observed=True).indices.items():
        groups[_normalise_group_key(key)] = np.asarray(idx, dtype=np.int64)
    return groups


def _build_blocked_keys(
    positives: pd.DataFrame,
    *,
    buffer_cells: int,
    buffer_days: int,
    resolution: float = SPATIAL_COARSENESS,
) -> set[tuple[str, int, int, int]]:
    if positives.empty:
        return set()
    pos = positives.copy()
    keys = _coordinate_keys(pos, resolution)
    pos["_lat_key"] = keys["_lat_key"].to_numpy()
    pos["_lon_key"] = keys["_lon_key"].to_numpy()
    pos["acq_date"] = pd.to_datetime(pos["acq_date"]).dt.date
    blocked: set[tuple[str, int, int, int]] = set()
    cell_offsets = range(-int(buffer_cells), int(buffer_cells) + 1)
    day_offsets = range(-int(buffer_days), int(buffer_days) + 1)
    for country_value, acq_date, lat_key, lon_key in zip(
        pos["country"],
        pos["acq_date"],
        pos["_lat_key"],
        pos["_lon_key"],
    ):
        base_ord = acq_date.toordinal()
        country = str(country_value)
        for dday in day_offsets:
            date_ord = base_ord + dday
            for dlat in cell_offsets:
                for dlon in cell_offsets:
                    blocked.add((country, date_ord, int(lat_key) + dlat, int(lon_key) + dlon))
    return blocked


def _dates_by_month(date_start, date_end) -> tuple[pd.DatetimeIndex, dict[int, np.ndarray]]:
    dates = pd.date_range(start=date_start, end=date_end, freq="D")
    month_map: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        month_dates = dates[dates.month == month]
        month_map[month] = month_dates.to_numpy()
    return dates, month_map


def _negative_row(
    *,
    country: str,
    lat: float,
    lon: float,
    date_value,
    stratum: str,
    sample_weight: float,
    sampling_probability: float,
    nearest_positive_distance_cells: float | None = None,
    nearest_positive_delta_days: float | None = None,
) -> dict[str, Any]:
    ts = pd.Timestamp(date_value)
    return {
        "lat_rounded": round(float(lat), rounding_precision),
        "lon_rounded": round(float(lon), rounding_precision),
        "brightness": 0,
        "confidence": 100,
        "country": country,
        "count": 0,
        "day": int(ts.day),
        "month": int(ts.month),
        "year": int(ts.year),
        "acq_date": ts.date(),
        "negative_stratum": stratum,
        "sampling_probability": sampling_probability,
        "sample_weight": sample_weight,
        "nearest_positive_distance_cells": nearest_positive_distance_cells,
        "nearest_positive_delta_days": nearest_positive_delta_days,
    }


def _append_negative_candidate(
    rows: list[dict[str, Any]],
    used_keys: set[tuple[str, int, int, int]],
    blocked_keys: set[tuple[str, int, int, int]],
    *,
    cell_idx: int,
    date_value,
    stratum: str,
    cells: pd.DataFrame,
    lat_keys: np.ndarray,
    lon_keys: np.ndarray,
    all_dates_count: int,
    requested_in_stratum: int,
    nearest_positive_distance_cells: float | None = None,
    nearest_positive_delta_days: float | None = None,
) -> bool:
    date_ts = pd.Timestamp(date_value)
    country = str(cells["country"].iat[cell_idx])
    key = (country, date_ts.date().toordinal(), int(lat_keys[cell_idx]), int(lon_keys[cell_idx]))
    if key in blocked_keys or key in used_keys:
        return False
    used_keys.add(key)
    universe = max(1, len(cells) * max(1, all_dates_count))
    probability = min(1.0, max(0.0, float(requested_in_stratum) / float(universe)))
    sample_weight = float("nan") if probability <= 0 else 1.0 / probability
    rows.append(
        _negative_row(
            country=country,
            lat=float(cells["lat_rounded"].iat[cell_idx]),
            lon=float(cells["lon_rounded"].iat[cell_idx]),
            date_value=date_ts,
            stratum=stratum,
            sample_weight=sample_weight,
            sampling_probability=probability,
            nearest_positive_distance_cells=nearest_positive_distance_cells,
            nearest_positive_delta_days=nearest_positive_delta_days,
        )
    )
    return True


def _random_date_for_template(
    template: pd.Series,
    stratum: str,
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    month_dates: dict[int, np.ndarray],
    *,
    match_month_for_static_strata: bool,
):
    use_month = stratum == "same_season" or (
        match_month_for_static_strata
        and stratum in {"same_ecoregion", "same_burnable_landcover"}
    )
    if use_month:
        candidates = month_dates.get(int(template["month"]), np.array([], dtype="datetime64[ns]"))
        if len(candidates):
            return rng.choice(candidates)
    return rng.choice(dates.to_numpy())


def _sample_generic_stratum(
    *,
    stratum: str,
    take: int,
    positives: pd.DataFrame,
    cells: pd.DataFrame,
    country_groups: dict[tuple[Any, ...], np.ndarray],
    ecoregion_groups: dict[tuple[Any, ...], np.ndarray],
    landcover_groups: dict[tuple[Any, ...], np.ndarray],
    dates: pd.DatetimeIndex,
    month_dates: dict[int, np.ndarray],
    rng: np.random.Generator,
    rows: list[dict[str, Any]],
    used_keys: set[tuple[str, int, int, int]],
    blocked_keys: set[tuple[str, int, int, int]],
    lat_keys: np.ndarray,
    lon_keys: np.ndarray,
    settings: dict[str, Any],
) -> int:
    if take <= 0 or cells.empty or len(dates) == 0:
        return 0

    before = len(rows)
    max_attempts = max(1000, int(take) * int(settings.get("max_attempt_multiplier", 80)))
    match_month = bool(settings.get("match_month_for_static_strata", True))
    positives_for_templates = positives if not positives.empty else None

    for _ in range(max_attempts):
        if len(rows) - before >= take:
            break

        if stratum == "random_background" or positives_for_templates is None:
            cell_idx = int(rng.integers(0, len(cells)))
            date_value = rng.choice(dates.to_numpy())
        else:
            template = positives_for_templates.iloc[int(rng.integers(0, len(positives_for_templates)))]
            country_key = (str(template["country"]),)
            group = country_groups.get(country_key)

            if stratum == "same_ecoregion":
                eco_key = (str(template["country"]), _normalise_group_value(template.get("ecoregion_name", "Unknown")))
                group = ecoregion_groups.get(eco_key, group)
            elif stratum == "same_burnable_landcover":
                land_key = (
                    str(template["country"]),
                    bool(template.get("burnable", True)),
                    _normalise_group_value(template.get("landcover_key", "burnable")),
                )
                group = landcover_groups.get(land_key, group)

            if group is None or len(group) == 0:
                group = np.arange(len(cells), dtype=np.int64)
            cell_idx = int(rng.choice(group))
            date_value = _random_date_for_template(
                template,
                stratum,
                rng,
                dates,
                month_dates,
                match_month_for_static_strata=match_month,
            )

        _append_negative_candidate(
            rows,
            used_keys,
            blocked_keys,
            cell_idx=cell_idx,
            date_value=date_value,
            stratum=stratum,
            cells=cells,
            lat_keys=lat_keys,
            lon_keys=lon_keys,
            all_dates_count=len(dates),
            requested_in_stratum=take,
        )

    return len(rows) - before


def _sample_near_fire_stratum(
    *,
    take: int,
    positives: pd.DataFrame,
    cells: pd.DataFrame,
    cell_lookup: dict[tuple[str, int, int], int],
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    rows: list[dict[str, Any]],
    used_keys: set[tuple[str, int, int, int]],
    blocked_keys: set[tuple[str, int, int, int]],
    lat_keys: np.ndarray,
    lon_keys: np.ndarray,
    settings: dict[str, Any],
) -> int:
    if take <= 0 or positives.empty or cells.empty or len(dates) == 0:
        return 0

    before = len(rows)
    min_cells = max(1, int(settings.get("near_fire_min_cells", 2)))
    max_cells = max(min_cells, int(settings.get("near_fire_max_cells", 5)))
    day_window = max(0, int(settings.get("near_fire_day_window", 7)))
    max_attempts = max(1000, int(take) * int(settings.get("max_attempt_multiplier", 80)))
    offsets = [
        (dlat, dlon, max(abs(dlat), abs(dlon)))
        for dlat in range(-max_cells, max_cells + 1)
        for dlon in range(-max_cells, max_cells + 1)
        if min_cells <= max(abs(dlat), abs(dlon)) <= max_cells
    ]
    date_min = pd.Timestamp(dates.min()).date()
    date_max = pd.Timestamp(dates.max()).date()

    for _ in range(max_attempts):
        if len(rows) - before >= take:
            break

        template = positives.iloc[int(rng.integers(0, len(positives)))]
        dlat, dlon, distance_cells = offsets[int(rng.integers(0, len(offsets)))]
        day_delta = int(rng.integers(-day_window, day_window + 1)) if day_window else 0
        base_date = pd.Timestamp(template["acq_date"]).date()
        date_value = base_date + timedelta(days=day_delta)
        if date_value < date_min or date_value > date_max:
            continue

        cell_key = (
            str(template["country"]),
            int(template["_lat_key"]) + dlat,
            int(template["_lon_key"]) + dlon,
        )
        cell_idx = cell_lookup.get(cell_key)
        if cell_idx is None:
            continue

        _append_negative_candidate(
            rows,
            used_keys,
            blocked_keys,
            cell_idx=cell_idx,
            date_value=date_value,
            stratum="near_fire_hard",
            cells=cells,
            lat_keys=lat_keys,
            lon_keys=lon_keys,
            all_dates_count=len(dates),
            requested_in_stratum=take,
            nearest_positive_distance_cells=float(distance_cells),
            nearest_positive_delta_days=float(abs(day_delta)),
        )

    return len(rows) - before


def _sample_stratified_negatives_from_cells(
    *,
    positives: pd.DataFrame,
    cells: pd.DataFrame,
    date_start,
    date_end,
    settings: dict[str, Any],
    seed: int,
    resolution: float = SPATIAL_COARSENESS,
) -> pd.DataFrame:
    positive_count = len(positives)
    target_negative_count = _desired_negative_count(
        positive_count,
        float(settings.get("target_positive_fraction", 0.10)),
    )
    if target_negative_count <= 0:
        return pd.DataFrame()

    cells = cells.reset_index(drop=True).copy()
    cell_keys = _coordinate_keys(cells, resolution)
    cells["_lat_key"] = cell_keys["_lat_key"].to_numpy()
    cells["_lon_key"] = cell_keys["_lon_key"].to_numpy()

    positives = _merge_positive_static_strata(positives, cells)
    positives["acq_date"] = pd.to_datetime(positives["acq_date"]).dt.date
    month_from_date = pd.to_datetime(positives["acq_date"]).map(lambda value: value.month)
    month_values = positives["month"] if "month" in positives.columns else pd.Series(np.nan, index=positives.index)
    positives["month"] = pd.to_numeric(month_values, errors="coerce").fillna(month_from_date).astype(int)

    weights = _normalise_stratum_weights(settings.get("stratum_weights"))
    allocations = _allocate_by_weights(target_negative_count, weights)
    print(
        "Stratified negative target:"
        f" positives={positive_count}, negatives={target_negative_count},"
        f" final positive fraction~{positive_count / (positive_count + target_negative_count):.3f}"
    )
    print(f"Negative stratum allocation: {allocations}")

    dates, month_dates = _dates_by_month(date_start, date_end)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    used_keys: set[tuple[str, int, int, int]] = set()
    blocked_keys = _build_blocked_keys(
        positives,
        buffer_cells=int(settings.get("exclude_positive_buffer_cells", 1)),
        buffer_days=int(settings.get("exclude_positive_buffer_days", 1)),
        resolution=resolution,
    )

    lat_keys = cells["_lat_key"].to_numpy(dtype=np.int64)
    lon_keys = cells["_lon_key"].to_numpy(dtype=np.int64)
    cell_lookup = {
        (str(country), int(lat_key), int(lon_key)): idx
        for idx, (country, lat_key, lon_key) in enumerate(
            zip(cells["country"], cells["_lat_key"], cells["_lon_key"])
        )
    }
    country_groups = _build_cell_groups(cells, ["country"])
    ecoregion_groups = _build_cell_groups(cells, ["country", "ecoregion_name"])
    landcover_groups = _build_cell_groups(cells, ["country", "burnable", "landcover_key"])

    sampled_counts: dict[str, int] = {}
    sampled_counts["near_fire_hard"] = _sample_near_fire_stratum(
        take=allocations.get("near_fire_hard", 0),
        positives=positives,
        cells=cells,
        cell_lookup=cell_lookup,
        dates=dates,
        rng=rng,
        rows=rows,
        used_keys=used_keys,
        blocked_keys=blocked_keys,
        lat_keys=lat_keys,
        lon_keys=lon_keys,
        settings=settings,
    )

    for stratum in ["same_season", "same_ecoregion", "same_burnable_landcover", "random_background"]:
        sampled_counts[stratum] = _sample_generic_stratum(
            stratum=stratum,
            take=allocations.get(stratum, 0),
            positives=positives,
            cells=cells,
            country_groups=country_groups,
            ecoregion_groups=ecoregion_groups,
            landcover_groups=landcover_groups,
            dates=dates,
            month_dates=month_dates,
            rng=rng,
            rows=rows,
            used_keys=used_keys,
            blocked_keys=blocked_keys,
            lat_keys=lat_keys,
            lon_keys=lon_keys,
            settings=settings,
        )

    shortfall = target_negative_count - len(rows)
    if shortfall > 0:
        print(f"Warning: stratified pools were short by {shortfall}; refilling from random background.")
        sampled_counts["random_background_refill"] = _sample_generic_stratum(
            stratum="random_background",
            take=shortfall,
            positives=positives,
            cells=cells,
            country_groups=country_groups,
            ecoregion_groups=ecoregion_groups,
            landcover_groups=landcover_groups,
            dates=dates,
            month_dates=month_dates,
            rng=rng,
            rows=rows,
            used_keys=used_keys,
            blocked_keys=blocked_keys,
            lat_keys=lat_keys,
            lon_keys=lon_keys,
            settings=settings,
        )

    print(f"Sampled negative counts by stratum: {sampled_counts}")
    if not rows:
        raise ValueError("Stratified negative sampler could not generate any negatives.")
    negative_df = pd.DataFrame(rows)
    if len(negative_df) < target_negative_count:
        print(
            f"Warning: generated {len(negative_df)}/{target_negative_count} requested negatives "
            "after exhausting configured sampling attempts."
        )
    return negative_df.reset_index(drop=True)


def sample_stratified_negative_samples(
    *,
    positives: pd.DataFrame,
    countries: list[str],
    date_start,
    date_end,
    coordinate_bounds: tuple[float, float, float, float] | list[float] | None,
    settings: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    cells = _candidate_cells_for_countries(
        countries,
        country_shapes_path=settings.get("country_shapes_path", "data/countries"),
        coordinate_bounds=coordinate_bounds,
        resolution=SPATIAL_COARSENESS,
    )
    cells = _enrich_cells_with_static_strata(cells, settings)
    return _sample_stratified_negatives_from_cells(
        positives=positives,
        cells=cells,
        date_start=date_start,
        date_end=date_end,
        settings=settings,
        seed=seed,
        resolution=SPATIAL_COARSENESS,
    )


def load_modis_data(data_dir='data/modis/', countries = ['Russian_Federation'], start_date = '2001-01-01', end_date = '2024-12-31'):
    print(f'Loading data from {str(start_date)} to {str(end_date)}')
    print(f'Loading data for countries: {countries}')
    df_country_list = []
    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])

    for country in countries:
        dataframes = []
        for year in range(start_year, end_year + 1):
            file_path = os.path.join(data_dir, str(year), f'modis_{year}_{country}.csv')
            if os.path.exists(file_path):
                df_year = pd.read_csv(file_path)
                df_year['year'] = year
                dataframes.append(df_year)
            else:
                print(f'Warning: File {file_path} does not exist')

        if not dataframes:
             print(f"No data files found for {country} between {start_year} and {end_year}. Skipping country.")
             continue

        df_country = pd.concat(dataframes, ignore_index=True)
        df_country['country'] = country
        df_country_list.append(df_country)

    if not df_country_list:
        raise FileNotFoundError("Error: No MODIS data loaded for any specified country and date range.")

    df = pd.concat(df_country_list, ignore_index=True)

    required_cols = ['brightness', 'confidence', 'acq_date', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Error: Missing required columns in loaded data: {missing_cols}")

    print(f"Number of records before brightness and confidence thresholds: {len(df)}")
    df['acq_date'] = pd.to_datetime(df['acq_date']).dt.date
    start_date_obj = pd.to_datetime(start_date).date()
    end_date_obj = pd.to_datetime(end_date).date()

    confidence_threshold, confidence_scale = normalize_confidence_threshold_for_series(
        df['confidence'],
        CONFIDENCE_THRESHOLD,
    )
    confidence_threshold_high_lat, confidence_scale_high_lat = normalize_confidence_threshold_for_series(
        df['confidence'],
        CONFIDENCE_THRESHOLD_HIGH_LAT,
    )
    print(
        "Detected FIRMS confidence scale:"
        f" {confidence_scale}; using thresholds {confidence_threshold}"
        f" / {confidence_threshold_high_lat} (configured"
        f" {CONFIDENCE_THRESHOLD} / {CONFIDENCE_THRESHOLD_HIGH_LAT})."
    )

    if USE_HIGH_LATITUDE_FILTER:
        # Apply latitude-dependent thresholds: lower thresholds for |lat| > 58
        high_lat_mask = df['latitude'].abs() > 58

        # Low-latitude filter (|lat| ≤ 60) – use default thresholds
        low_lat_filter = (
            (~high_lat_mask)
            & (df['brightness'] > BRIGHTNESS_THRESHOLD)
            & (df['confidence'] > confidence_threshold)
        )

        # High-latitude filter (|lat| > 58) – use relaxed thresholds
        high_lat_filter = (
            high_lat_mask
            & (df['brightness'] > BRIGHTNESS_THRESHOLD_HIGH_LAT)
            & (df['confidence'] > confidence_threshold_high_lat)
        )

        # --- THRESHOLDS for far-north & western longitudes (lat > 58 and lon < 45) ---
        special_region_mask = (df['latitude'] > 58) & (df['longitude'] < 45)

        SPECIAL_BRIGHTNESS_THRESHOLD = max(BRIGHTNESS_THRESHOLD_HIGH_LAT - 30, 0)
        if confidence_threshold_high_lat > 1:
            SPECIAL_CONFIDENCE_THRESHOLD = max(confidence_threshold_high_lat - 15, 0)
        else:                                          
            SPECIAL_CONFIDENCE_THRESHOLD = max(confidence_threshold_high_lat - 0.15, 0.0)

        special_region_filter = (
            special_region_mask
            & (df['brightness'] > SPECIAL_BRIGHTNESS_THRESHOLD)
            & (df['confidence'] > SPECIAL_CONFIDENCE_THRESHOLD)
        )
        # --------------------------------------------------------------------------------------------

        # Combine all filters
        df = df[low_lat_filter | high_lat_filter | special_region_filter]
        print(f"Applied latitude-dependent thresholds (high-latitude filter enabled)")
    else:
        # Use standard thresholds for all latitudes
        df = df[
            (df['brightness'] > BRIGHTNESS_THRESHOLD)
            & (df['confidence'] > confidence_threshold)
        ]
        print(f"Applied standard thresholds for all latitudes (high-latitude filter disabled)")
    df = df[df['acq_date'] >= start_date_obj]
    df = df[df['acq_date'] <= end_date_obj]
    print(f"Number of fire points after brightness and confidence thresholds: {len(df)}")

    if FILTER_STATIONARY_POINTS:
        df, removed = drop_stationary_points(
            df,
            stationary_dir=STATIONARY_POINTS_DIR,
            country_col='country',
            lat_col='latitude',
            lon_col='longitude'
        )
        if removed:
            print(
                f"Filtered out {removed} stationary detections using catalogue at"
                f" {STATIONARY_POINTS_DIR}"
            )
        else:
            print("No stationary detections removed (catalogue empty or no matches).")
    else:
        print("Stationary-point filtering disabled via config.")

    if len(df) > 0:
        print("\n" + "="*60)
        print(f"🔥 LOADED TARGET (Positive Points):")
        print(f"Target time range: {str(df['acq_date'].min())}-{str(df['acq_date'].max())}")
        print(f"Target lat range: {df['latitude'].min()}-{df['latitude'].max()}")
        print(f"Target lon range: {df['longitude'].min()}-{df['longitude'].max()}")
        print(f"Target df length: {len(df)}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("🔥 WARNING: NO POSITIVE TARGET POINTS LOADED after filtering.")
        print("="*60)

    return df


def expand_positive_points(data: pd.DataFrame,
                           spatial_coarseness: float,
                           lat_col: str = 'lat_rounded',
                           lon_col: str = 'lon_rounded',
                           count_col: str = 'count') -> pd.DataFrame:
    """
    Expands points with count > 1 by adding new points around them.

    For each point with count > 1:
    - Keeps the original point but sets its count to 1.
    - Adds 'count - 1' new points (up to a max of 4).
    - New points are positioned 'spatial_coarseness' distance (in degrees)
      from the original point in cardinal/diagonal directions.
    - New points inherit data from the original point and have count = 1.

    Args:
        data (pd.DataFrame): Input dataframe, typically after grouping.
        spatial_coarseness (float): The offset distance in degrees for new points.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.
        count_col (str): Name of the count column.

    Returns:
        pd.DataFrame: Dataframe with points expanded. All points in the
                      returned dataframe will have count = 1.

    Raises:
        KeyError: If required columns (lat, lon, count) are missing.
    """
    print(f"\nExpanding positive points with count > 1...")
    print(f"Using spatial_coarseness (offset): {spatial_coarseness}")

    # Check required columns
    required_expand_cols = [lat_col, lon_col, count_col]
    for col in required_expand_cols:
        if col not in data.columns:
            raise KeyError(f"Missing required column for expansion: '{col}'")

    # Separate points to expand from others
    # Ensure count column is numeric and handle potential NaNs
    data[count_col] = pd.to_numeric(data[count_col], errors='coerce')
    to_expand = data[data[count_col] > 1].copy()
    others = data[(data[count_col] <= 1) | (data[count_col].isna())].copy() # Keep points with count=1 or invalid counts

    if to_expand.empty:
        print("No points found with count > 1. No expansion needed.")
        # Ensure all counts in 'others' are 1 if they were validly <= 1
        others.loc[others[count_col] == 1, count_col] = 1
        return others

    print(f"Found {len(to_expand)} points with count > 1 to expand.")

    expanded_points_list = []
    modified_originals_list = []
    S = spatial_coarseness # Alias for brevity

    print("Generating expanded points...")
    for _, row in tqdm(to_expand.iterrows(), total=len(to_expand), desc="Expanding Points"):
        original_lat = row[lat_col]
        original_lon = row[lon_col]
        original_count = int(row[count_col]) # Assumes count is integer after check

        # 1. Modify the original point (set count to 1) and add to list
        modified_original_row = row.copy()
        modified_original_row[count_col] = 1
        modified_originals_list.append(modified_original_row)

        # 2. Determine number of new points to add
        num_new_points = min(original_count - 1, 4)

        # 3. Generate coordinates for new points based on num_new_points
        new_coords = []
        if num_new_points >= 1: # North
            new_coords.append({'lat': original_lat + S, 'lon': original_lon})
        if num_new_points >= 2: # South
            new_coords.append({'lat': original_lat - S, 'lon': original_lon})
        if num_new_points >= 3: # East
            new_coords.append({'lat': original_lat, 'lon': original_lon + S})
        if num_new_points >= 4: # West
            new_coords.append({'lat': original_lat, 'lon': original_lon - S})
            # If we wanted diagonal for 3, logic would be more complex. Cardinal is simpler.

        # 4. Create new rows for these points
        for coord in new_coords:
            new_row_dict = row.to_dict() # Copy data from original row
            new_row_dict[lat_col] = round(coord['lat'], rounding_precision) # Use new lat, round it
            new_row_dict[lon_col] = round(coord['lon'], rounding_precision) # Use new lon, round it
            new_row_dict[count_col] = 1 # Set count to 1
            # Potentially update other derived fields if needed, but usually inheriting is fine
            expanded_points_list.append(new_row_dict)

    # Convert lists of rows/dicts to DataFrames
    modified_originals_df = pd.DataFrame(modified_originals_list)
    newly_expanded_df = pd.DataFrame(expanded_points_list)

    total_new_points = len(newly_expanded_df)
    print(f"Generated {total_new_points} new points from expansion.")
    print(f"Original {len(to_expand)} points modified to have count=1.")

    # Combine the original points that weren't expanded, the modified originals, and the new points
    final_df = pd.concat([others, modified_originals_df, newly_expanded_df], ignore_index=True)

    # Final check: Ensure all counts are integer 1 (or handle NaNs if they exist in 'others')
    final_df[count_col] = final_df[count_col].fillna(0).astype(int) # Example: fill NaN counts with 0, ensure int
    # Or if you expect only 1s and 0s (negatives later)
    # final_df.loc[final_df[count_col] > 0, count_col] = 1

    print(f"Expansion complete. Final dataset size after expansion: {len(final_df)}")
    print(f"Value counts for '{count_col}' after expansion:")
    print(final_df[count_col].value_counts())


    return final_df


def filter_negative_neighbors(data: pd.DataFrame,
                              lat_col: str = 'lat_rounded',
                              lon_col: str = 'lon_rounded',
                              date_col: str = 'acq_date',
                              count_col: str = 'count',
                              neighbor_dist: float = 0.1,
                              neighbor_days: int = 1) -> pd.DataFrame:
    """
    Vectorized replacement: filters out negative samples whose (lat, lon, date)
    matches any positive sample within +/- neighbor_dist degrees and +/- neighbor_days days.
    """
    # --- validations & split ---
    required = [lat_col, lon_col, date_col, count_col]
    for c in required:
        if c not in data.columns:
            raise KeyError(f"Missing required column for filtering: '{c}'")
    # ensure date_col is datetime.date
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])
    data[date_col] = data[date_col].dt.date

    pos = data[data[count_col] > 0]
    neg = data[data[count_col] == 0]
    if pos.empty or neg.empty:
        # nothing to filter
        return data.copy()

    # --- build the expanded-positive grid × date shifts ---
    # shifts in lat/lon: -1,0,+1 cells
    offsets = [-1, 0, 1]
    pos_expanded_parts = []
    for dlat, dlon, dday in itertools.product(offsets, offsets, offsets):
        df = pos[[lat_col, lon_col, date_col]].copy()
        # shift coordinates and date
        df[lat_col] = df[lat_col] + dlat * neighbor_dist
        df[lon_col] = df[lon_col] + dlon * neighbor_dist
        df[date_col] = df[date_col] + timedelta(days=dday)
        pos_expanded_parts.append(df)
    pos_expanded = (
        pd.concat(pos_expanded_parts, ignore_index=True)
          .drop_duplicates()
    )

    # --- merge negatives against the expanded positives ---
    neg_idx = neg.reset_index().rename(columns={'index':'orig_idx'})
    merged = neg_idx.merge(
        pos_expanded,
        on=[lat_col, lon_col, date_col],
        how='left',
        indicator=True
    )

    # keep only negatives with no match in pos_expanded
    filtered_neg = (
        merged[merged['_merge']=='left_only']
          .drop(columns=['_merge'])
          .set_index('orig_idx')
          .loc[:, neg.columns]  # restore original column order
    )

    # --- recombine positives and filtered negatives ---
    result = pd.concat([pos, filtered_neg], ignore_index=True)
    return result.reset_index(drop=True)


def _initial_positive_counts(latitude, longitude) -> np.ndarray:
    """Return target weights for positive detections before cell/date grouping."""

    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    special_region_mask_NW = (lat > 60) & (lon < 45)
    special_region_mask_W = (lat > 55) & (lon < 60)
    return np.select(
        [special_region_mask_NW, special_region_mask_W],
        [3, 2],
        default=1,
    ).astype(int)


def _as_positive_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        values = default
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    cleaned = sorted({int(item) for item in values if int(item) >= 0})
    return cleaned or list(default)


def _safe_feature_token(value: Any) -> str:
    token = str(value).strip().lower()
    token = "".join(char if char.isalnum() else "_" for char in token)
    token = "_".join(part for part in token.split("_") if part)
    return token or "window"


def _fire_count_windows(settings: dict[str, Any]) -> list[dict[str, int | str]]:
    configured = settings.get("count_windows")
    if configured:
        windows = configured if isinstance(configured, (list, tuple)) else [configured]
        parsed: list[dict[str, int | str]] = []
        for idx, raw in enumerate(windows):
            if not isinstance(raw, dict):
                raise ValueError("recent_fire_history.count_windows entries must be mappings.")
            start_days = max(0, int(raw.get("start_days", settings.get("min_lag_days", 30))))
            end_days = max(start_days + 1, int(raw.get("end_days", raw.get("window_days", 0))))
            raw_name = raw.get("name") or f"{start_days}_{end_days}d"
            parsed.append(
                {
                    "name": _safe_feature_token(raw_name),
                    "start_days": start_days,
                    "end_days": end_days,
                }
            )
        return parsed

    min_lag_days = max(0, int(settings.get("min_lag_days", 0)))
    return [
        {
            "name": f"{min_lag_days}_{min_lag_days + int(window)}d",
            "start_days": min_lag_days,
            "end_days": min_lag_days + int(window),
        }
        for window in _as_positive_int_list(settings.get("windows_days"), [7, 14, 30])
        if int(window) > 0
    ]


def _spatial_offsets(radius_cells: int) -> list[tuple[int, int]]:
    radius_cells = max(0, int(radius_cells))
    return [
        (dlat, dlon)
        for dlat in range(-radius_cells, radius_cells + 1)
        for dlon in range(-radius_cells, radius_cells + 1)
        if max(abs(dlat), abs(dlon)) <= radius_cells
    ]


def _positive_event_arrays_by_cell(
    positives: pd.DataFrame,
    radius_cells: int,
) -> dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]]:
    if positives.empty:
        return {}

    pos = positives.copy()
    pos_keys = _coordinate_keys(pos, SPATIAL_COARSENESS)
    pos["_lat_key"] = pos_keys["_lat_key"].to_numpy()
    pos["_lon_key"] = pos_keys["_lon_key"].to_numpy()
    pos["acq_date"] = pd.to_datetime(pos["acq_date"], errors="coerce")
    pos = pos[pos["acq_date"].notna()]
    if pos.empty:
        return {}

    pos["_date_ord"] = pos["acq_date"].dt.date.map(lambda value: value.toordinal()).astype(np.int64)
    pos["_event_count"] = pd.to_numeric(pos.get("count", 1), errors="coerce").fillna(1.0).clip(lower=1.0)
    offsets = np.asarray(_spatial_offsets(radius_cells), dtype=np.int64)
    offset_count = len(offsets)
    expanded = pd.DataFrame(
        {
            "country": np.tile(pos["country"].astype(str).to_numpy(), offset_count),
            "_lat_key": (
                pos["_lat_key"].to_numpy(dtype=np.int64)[None, :]
                + offsets[:, 0, None]
            ).ravel(),
            "_lon_key": (
                pos["_lon_key"].to_numpy(dtype=np.int64)[None, :]
                + offsets[:, 1, None]
            ).ravel(),
            "_date_ord": np.tile(pos["_date_ord"].to_numpy(dtype=np.int64), offset_count),
            "_event_count": np.tile(
                pos["_event_count"].to_numpy(dtype=np.float32),
                offset_count,
            ),
        }
    )
    grouped = (
        expanded.groupby(["country", "_lat_key", "_lon_key", "_date_ord"], observed=True, sort=False)["_event_count"]
        .sum()
        .reset_index()
        .sort_values(["country", "_lat_key", "_lon_key", "_date_ord"], kind="stable")
    )

    events: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, group in grouped.groupby(["country", "_lat_key", "_lon_key"], observed=True, sort=False):
        date_ord = group["_date_ord"].to_numpy(dtype=np.int64)
        cumulative = np.cumsum(group["_event_count"].to_numpy(dtype=np.float64))
        events[(str(key[0]), int(key[1]), int(key[2]))] = (date_ord, cumulative)
    return events


def _recent_count_from_events(
    target_keys: pd.DataFrame,
    events: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]],
    window_days: int,
) -> np.ndarray:
    result = np.zeros(len(target_keys), dtype=np.float32)
    if not events or len(target_keys) == 0:
        return result

    window_days = max(1, int(window_days))
    target_dates = target_keys["_date_ord"].to_numpy(dtype=np.int64)
    for key, indices in target_keys.groupby(["country", "_lat_key", "_lon_key"], observed=True, sort=False).indices.items():
        event = events.get((str(key[0]), int(key[1]), int(key[2])))
        if event is None:
            continue
        event_dates, cumulative = event
        idx = np.asarray(indices, dtype=np.int64)
        dates = target_dates[idx]
        right = np.searchsorted(event_dates, dates, side="left")
        left = np.searchsorted(event_dates, dates - window_days, side="left")
        right_values = np.where(right > 0, cumulative[right - 1], 0.0)
        left_values = np.where(left > 0, cumulative[left - 1], 0.0)
        counts = right_values - left_values
        result[idx] = counts.astype(np.float32)
    return result


def _days_since_from_events(
    target_keys: pd.DataFrame,
    events: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]],
    cap_days: int,
) -> np.ndarray:
    cap_days = max(1, int(cap_days))
    missing_value = float(cap_days + 1)
    result = np.full(len(target_keys), missing_value, dtype=np.float32)
    if not events or len(target_keys) == 0:
        return result

    target_dates = target_keys["_date_ord"].to_numpy(dtype=np.int64)
    for key, indices in target_keys.groupby(["country", "_lat_key", "_lon_key"], observed=True, sort=False).indices.items():
        event = events.get((str(key[0]), int(key[1]), int(key[2])))
        if event is None:
            continue
        event_dates, _ = event
        idx = np.asarray(indices, dtype=np.int64)
        dates = target_dates[idx]
        previous_idx = np.searchsorted(event_dates, dates, side="left") - 1
        has_previous = previous_idx >= 0
        if has_previous.any():
            days = dates[has_previous] - event_dates[previous_idx[has_previous]]
            result[idx[has_previous]] = np.minimum(days, cap_days + 1).astype(np.float32)
    return result


def _target_indices_by_cell(target_keys: pd.DataFrame) -> dict[tuple[str, int, int], np.ndarray]:
    return {
        (str(key[0]), int(key[1]), int(key[2])): np.asarray(indices, dtype=np.int64)
        for key, indices in target_keys.groupby(
            ["country", "_lat_key", "_lon_key"],
            observed=True,
            sort=False,
        ).indices.items()
    }


def _values_before_positions(cumulative: np.ndarray, positions: np.ndarray) -> np.ndarray:
    values = np.zeros(len(positions), dtype=np.float64)
    valid = positions > 0
    if valid.any():
        values[valid] = cumulative[positions[valid] - 1]
    return values


def _recent_history_for_radius(
    *,
    target_dates: np.ndarray,
    target_groups: dict[tuple[str, int, int], np.ndarray],
    events: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]],
    count_windows: Sequence[dict[str, int | str]],
    include_days_since: bool,
    min_lag_days: int,
    cap_days: int,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    count_results = {
        str(window["name"]): np.zeros(len(target_dates), dtype=np.float32)
        for window in count_windows
    }
    days_since = (
        np.full(len(target_dates), float(cap_days + 1), dtype=np.float32)
        if include_days_since
        else None
    )
    if not events or len(target_dates) == 0:
        return count_results, days_since

    for key, idx in target_groups.items():
        event = events.get(key)
        if event is None:
            continue
        event_dates, cumulative = event
        dates = target_dates[idx]

        for window in count_windows:
            name = str(window["name"])
            start_days = max(0, int(window["start_days"]))
            end_days = max(start_days + 1, int(window["end_days"]))
            right = np.searchsorted(event_dates, dates - start_days + 1, side="left")
            left = np.searchsorted(event_dates, dates - end_days, side="left")
            right_values = _values_before_positions(cumulative, right)
            count_results[name][idx] = (
                right_values - _values_before_positions(cumulative, left)
            ).astype(np.float32)

        if days_since is not None:
            right = np.searchsorted(event_dates, dates - max(0, int(min_lag_days)) + 1, side="left")
            previous_idx = right - 1
            has_previous = previous_idx >= 0
            if has_previous.any():
                previous_days = dates[has_previous] - event_dates[previous_idx[has_previous]]
                days_since[idx[has_previous]] = np.minimum(
                    previous_days,
                    cap_days + 1,
                ).astype(np.float32)

    return count_results, days_since


def _add_recent_fire_history_features(
    target: pd.DataFrame,
    positives: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if not bool(settings.get("enabled", True)) or target.empty:
        return target

    count_windows = _fire_count_windows(settings)
    radii = _as_positive_int_list(settings.get("radii_cells"), [0, 1, 2])
    days_since_radii = _as_positive_int_list(settings.get("days_since_radii_cells"), radii)
    min_lag_days = max(0, int(settings.get("min_lag_days", 30)))
    cap_days = int(settings.get("days_since_cap", 365))

    result = target.copy()
    keyed = result.copy()
    target_keys = _coordinate_keys(keyed, SPATIAL_COARSENESS)
    keyed["_lat_key"] = target_keys["_lat_key"].to_numpy()
    keyed["_lon_key"] = target_keys["_lon_key"].to_numpy()
    date_values = pd.to_datetime(keyed["acq_date"], errors="coerce")
    keyed["_date_ord"] = date_values.dt.date.map(
        lambda value: value.toordinal() if pd.notna(value) else -1
    ).astype(np.int64)
    valid_date_mask = keyed["_date_ord"].to_numpy(dtype=np.int64) >= 0
    working_keys = keyed.loc[valid_date_mask, ["country", "_lat_key", "_lon_key", "_date_ord"]].copy()
    target_dates = working_keys["_date_ord"].to_numpy(dtype=np.int64)
    target_groups = _target_indices_by_cell(working_keys)

    all_radii = sorted(set(radii) | set(days_since_radii))
    event_cache = {
        radius: _positive_event_arrays_by_cell(positives, radius)
        for radius in all_radii
    }
    include_days_since = bool(settings.get("include_days_since", True))

    for radius in all_radii:
        count_results, days_since = _recent_history_for_radius(
            target_dates=target_dates,
            target_groups=target_groups,
            events=event_cache.get(radius, {}),
            count_windows=count_windows if radius in radii else [],
            include_days_since=include_days_since and radius in days_since_radii,
            min_lag_days=min_lag_days,
            cap_days=cap_days,
        )
        for window_name, compact_values in count_results.items():
            name = f"past_fire_count_r{radius}_{window_name}"
            values = np.zeros(len(result), dtype=np.float32)
            values[valid_date_mask] = compact_values
            result[name] = values
        if days_since is not None:
            name = f"days_since_fire_r{radius}"
            values = np.full(len(result), float(cap_days + 1), dtype=np.float32)
            values[valid_date_mask] = days_since
            result[name] = values

    added = [
        col
        for col in result.columns
        if col.startswith("past_fire_count_")
        or col.startswith("recent_fire_count_")
        or col.startswith("days_since_fire_")
    ]
    print(f"Added recent fire-history features: {added}")
    return result


def _assign_soft_fire_labels(target: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    column = str(settings.get("column", "soft_label"))
    result = target.copy()
    hard_positive = pd.to_numeric(result.get("count", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0
    soft = hard_positive.astype(np.float32)

    if bool(settings.get("enabled", True)):
        distance_col = str(settings.get("distance_column", "nearest_positive_distance_cells"))
        delta_col = str(settings.get("delta_days_column", "nearest_positive_delta_days"))
        if distance_col in result.columns and delta_col in result.columns:
            distance = pd.to_numeric(result[distance_col], errors="coerce").to_numpy(dtype=float)
            delta = pd.to_numeric(result[delta_col], errors="coerce").to_numpy(dtype=float)
            max_distance = float(settings.get("max_distance_cells", 5))
            max_delta = float(settings.get("max_delta_days", 7))
            spatial_decay = max(float(settings.get("spatial_decay_cells", 2.0)), 1e-6)
            temporal_decay = max(float(settings.get("temporal_decay_days", 7.0)), 1e-6)
            max_negative_label = float(settings.get("max_negative_label", 0.35))
            eligible = (
                ~hard_positive
                & np.isfinite(distance)
                & np.isfinite(delta)
                & (distance <= max_distance)
                & (delta <= max_delta)
            )
            if eligible.any():
                softened = max_negative_label * np.exp(
                    -np.maximum(distance[eligible] - 1.0, 0.0) / spatial_decay
                ) * np.exp(-np.maximum(delta[eligible], 0.0) / temporal_decay)
                soft[eligible] = np.maximum(soft[eligible], softened.astype(np.float32))

    result[column] = np.clip(soft, 0.0, 1.0).astype(np.float32)
    soft_negative_count = int(((~hard_positive) & (result[column].to_numpy(dtype=float) > 0.0)).sum())
    print(
        f"Assigned soft labels in '{column}': positives={int(hard_positive.sum())}, "
        f"softened_negatives={soft_negative_count}"
    )
    return result



def prepare_target_data(
    data: pd.DataFrame,
    countries: list,
    samples_per_area_per_year: float,
    coordinate_bounds: tuple[float, float, float, float] | list[float] | None = None,
    negative_sampling_feature_config: dict[str, Any] | None = None,
):
    '''Prepares target data: aggregates positives, EXPANDS high-count positives, adds negatives, filters negatives near positives.'''

    if data.empty:
        raise ValueError("Input data to prepare_target_data is empty. Cannot proceed.")

    data = data[data['country'].isin(countries)].copy()
    if data.empty:
        raise ValueError(f"No data found for specified countries: {countries} after initial filtering.")
    print(f"Using rounding precision: {rounding_precision}, lat and lon step is {SPATIAL_COARSENESS}")
    data['lat_rounded'] = data['latitude'].round(rounding_precision)
    data['lon_rounded'] = data['longitude'].round(rounding_precision)

    if not isinstance(data['acq_date'].iloc[0], datetime.date):
         raise TypeError("Column 'acq_date' is not composed of date objects in prepare_target_data.")

    data['day'] = data['acq_date'].apply(lambda d: d.day)
    data['month'] = data['acq_date'].apply(lambda d: d.month)
    data['year'] = data['acq_date'].apply(lambda d: d.year)

    date_start = data['acq_date'].min()
    date_end = data['acq_date'].max()
    
    # Initialize count: 1 for normal points, 2 for points in the special region
    special_region_mask_NW = (data['latitude'] > 60) & (data['longitude'] < 45)
    special_region_mask_W = (data['latitude'] > 55) & (data['longitude'] < 60)
    data['count'] = _initial_positive_counts(data['latitude'], data['longitude'])
    print(f"\nTriple base count for {special_region_mask_NW.sum()} points in the special region (lat>60, lon<45).")
    print(f"\nDoubled base count for {(special_region_mask_W & ~special_region_mask_NW).sum()} points in the special region (lat>55, lon<60).")

    # --- Step 1: Group positive points ---
    data_grouped = data.groupby(['lat_rounded', 'lon_rounded', 'acq_date'], observed=True).agg(
        brightness=('brightness', 'mean'),
        confidence=('confidence', 'mean'),
        count=('count', 'sum'), # This sums the initial '1's, getting the true count per group
        month=('month', 'first'),
        year=('year', 'first'),
        country=('country', 'first'),
        day=('day', 'first'),
    ).reset_index()
    print(f"\nGrouped positive points. Size: {len(data_grouped)}")
    print("Value counts for 'count' after grouping:")
    print(data_grouped['count'].value_counts().head()) # Show distribution

    # --- Step 2: Expand high-count positive points ---
    # Apply the expansion function to the grouped data
    data_expanded = expand_positive_points(
        data=data_grouped,
        spatial_coarseness=SPATIAL_COARSENESS,
        lat_col='lat_rounded',
        lon_col='lon_rounded',
        count_col='count'
    )
    data_expanded_for_sampling = _filter_to_coordinate_bounds(data_expanded, coordinate_bounds)

    # --- Step 3: Add Negative Samples ---
    negative_settings = _negative_sampling_settings(negative_sampling_feature_config)
    negative_strategy = str(negative_settings.get("strategy", "legacy_area")).lower()

    if negative_strategy in {"stratified", "stratified_case_control", "real_world_strata"}:
        if data_expanded_for_sampling.empty:
            raise ValueError("No positive target rows remain after applying coordinate bounds.")
        print("\nAdding stratified negative samples...")
        negative_samples_df = sample_stratified_negative_samples(
            positives=data_expanded_for_sampling,
            countries=countries,
            date_start=date_start,
            date_end=date_end,
            coordinate_bounds=coordinate_bounds,
            settings=negative_settings,
            seed=deterministic_negative_seed(GLOBAL_SEED, ",".join(sorted(countries)), date_start, date_end, "stratified"),
        )
        print(f"Total stratified negative samples added: {len(negative_samples_df)}")
        data_filtered = pd.concat([data_expanded_for_sampling, negative_samples_df], ignore_index=True)
    else:
        print(f"\nReading country geometries from 'data/countries'...")
        world = gpd.read_file('data/countries')
        print("Country geometries loaded.")

        country_areas = {}
        country_sizes = {}
        # Use date_start/end from the original *grouped* data before expansion
        num_years = max(1, (date_end - date_start).days / 365.25)

        valid_countries_for_neg_samples = []
        for country in countries: # Iterate through original requested countries
            mapped_country_name = country_mapping.get(country, country)
            country_geom_series = world[world['SOVEREIGNT'] == mapped_country_name]['geometry']
            if not country_geom_series.empty:
                country_geom = country_geom_series.iloc[0]
                if not country_geom.is_valid:
                     print(f"Warning: Invalid geometry for {mapped_country_name}, attempting buffer(0).")
                     country_geom = country_geom.buffer(0)
                     if not country_geom.is_valid:
                          raise ValueError(f"Fatal: Unfixable invalid geometry for '{mapped_country_name}'.")

                country_areas[country] = float(country_geom.area)
                num_samples = int(country_areas[country] * samples_per_area_per_year * num_years)
                country_sizes[country] = max(num_samples, 1) if samples_per_area_per_year > 0 else 0
                if country_sizes[country] > 0:
                    valid_countries_for_neg_samples.append(country)
                else:
                     print(f"Calculated 0 negative samples for {country}. Skipping.")
            else:
                raise ValueError(f"Fatal: Geometry for '{mapped_country_name}' not found.")

        random_data_list = []
        if valid_countries_for_neg_samples:
            print("\nAdding negative samples using ProcessPoolExecutor...")
            futures = []
            executor = None
            try:
                executor = ProcessPoolExecutor()
                for country in valid_countries_for_neg_samples:
                    mapped_country_name = country_mapping.get(country, country)
                    world_subset = world[world['SOVEREIGNT'] == mapped_country_name].iloc[0:1].copy()

                    futures.append(executor.submit(
                        add_negative_samples,
                        date_start=date_start, # Use original start/end for neg sampling range
                        date_end=date_end,
                        size=country_sizes[country],
                        world_data_subset=world_subset,
                        country_name=country,
                        seed=deterministic_negative_seed(GLOBAL_SEED, country, date_start, date_end),
                    ))

                for future in tqdm(as_completed(futures), total=len(futures), desc="Negative Sampling"):
                    result_df = future.result() # Raises exceptions from worker
                    if result_df is not None and not result_df.empty:
                        random_data_list.append(result_df)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Cancelling negative sampling workers.")
                for future in futures:
                    future.cancel()
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:
                for future in futures:
                    future.cancel()
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                print(f"\nError during parallel processing for negative samples: {e}")
                raise RuntimeError("Failed to generate negative samples.") from e
            else:
                if executor is not None:
                    executor.shutdown()

        if random_data_list:
             negative_samples_df = pd.concat(random_data_list, ignore_index=True)
             print(f"Total negative samples added: {len(negative_samples_df)}")
             # Combine expanded positives and new negatives
             data_with_negatives = pd.concat([data_expanded, negative_samples_df], ignore_index=True)
        else:
            raise ValueError("No negative samples were added. Check the country geometries and sampling parameters.")

        # --- Step 4: Apply the negative filtering function ---
        # Filter the negative points that are too close to the (now expanded) positive points
        data_filtered = filter_negative_neighbors(
            data=data_with_negatives,
            lat_col='lat_rounded',
            lon_col='lon_rounded',
            date_col='acq_date',
            count_col='count', # Should be 1 for positives, 0 for negatives
            neighbor_dist=0.1,
            neighbor_days=1
        )

    # --- Step 5: Add datetime column ---
    # Ensure year, month, day are valid before creating datetime
    required_dt_cols = ['year', 'month', 'day']
    for col in required_dt_cols:
         if col not in data_filtered.columns:
              raise KeyError(f"Missing column required for datetime creation: {col}")
         # Ensure they are numeric, handle potential issues from concat/expansion
         data_filtered[col] = pd.to_numeric(data_filtered[col], errors='coerce').fillna(-1).astype(int)
         if (data_filtered[col] < 1).any() and col != 'day': # Basic check
             print(f"Warning: Invalid values found in column '{col}' before datetime creation.")

    # Check for invalid dates before conversion
    invalid_dates = data_filtered[
        (data_filtered['month'] < 1) | (data_filtered['month'] > 12) |
        (data_filtered['day'] < 1) | (data_filtered['day'] > 31) 
    ]
    if not invalid_dates.empty:
        print(f"Warning: Found {len(invalid_dates)} rows with potentially invalid month/day values.")
        raise ValueError(f"Invalid date components found in data before final datetime conversion. Example row index: {invalid_dates.index[0]}")



    data_filtered['datetime'] = pd.to_datetime(data_filtered[['year', 'month', 'day']].assign(hour=0))

    soft_label_settings = _target_section_settings(
        "soft_labels",
        negative_sampling_feature_config,
    )
    data_filtered = _assign_soft_fire_labels(data_filtered, soft_label_settings)

    fire_history_settings = _target_section_settings(
        "recent_fire_history",
        negative_sampling_feature_config,
    )
    data_filtered = _add_recent_fire_history_features(
        data_filtered,
        positives=data_expanded_for_sampling,
        settings=fire_history_settings,
    )

    print("\nPreparation pipeline complete.")
    return data_filtered


def add_negative_samples(date_start,
                         date_end,
                         size,
                         world_data_subset,
                         country_name,
                         seed: int | None = None):
    '''Generates random negative samples within country bounds.'''

    if world_data_subset.empty:
        raise ValueError(f"No geometry data provided for negative sampling for {country_name}")

    country_polygon = world_data_subset.geometry.iloc[0]
    if not country_polygon.is_valid:
        print(f"Warning: Invalid geometry for {country_name} in worker, attempting buffer(0).")
        country_polygon = country_polygon.buffer(0)
        if not country_polygon.is_valid:
            raise ValueError(f"Fatal: Could not fix invalid geometry for {country_name} in worker.")

    min_lon, min_lat, max_lon, max_lat = country_polygon.bounds
    lat_range = (min_lat, max_lat)
    lon_range = (min_lon, max_lon)

    generated_count = 0
    max_attempts = 10
    attempts = 0
    valid_lat = []
    valid_lon = []
    rng = np.random.default_rng(seed)

    while generated_count < size and attempts < max_attempts:
        needed = size - generated_count
        buffer_needed = int(needed * 1.2) + 10
        random_lat = rng.uniform(lat_range[0], lat_range[1], size=buffer_needed)
        random_lon = rng.uniform(lon_range[0], lon_range[1], size=buffer_needed)

        mask = contains(country_polygon, random_lon, random_lat) # Can raise GEOSException
        new_valid_lat = random_lat[mask]
        new_valid_lon = random_lon[mask]

        take_count = min(needed, len(new_valid_lat))
        if take_count > 0:
            valid_lat.extend(new_valid_lat[:take_count])
            valid_lon.extend(new_valid_lon[:take_count])
            generated_count += take_count
        attempts += 1

    if generated_count < size:
        print(f"Warning: Worker could only generate {generated_count}/{size} samples for {country_name} ({attempts} attempts).")
    if generated_count == 0:
        raise ValueError(f"Worker could not generate any valid negative samples for {country_name}.")

    final_size = generated_count
    random_lat_arr = np.array(valid_lat)
    random_lon_arr = np.array(valid_lon)

    random_dates = pd.to_datetime(rng.choice(pd.date_range(start=date_start, end=date_end), size=final_size))

    random_data = pd.DataFrame({
        'lat_rounded': np.round(random_lat_arr, decimals=rounding_precision),
        'lon_rounded': np.round(random_lon_arr, decimals=rounding_precision),
        'brightness': 0,
        'confidence': 100,
        'country': country_name,
        'count': 0, # Negative samples always have count 0
        'day': random_dates.day,
        'month': random_dates.month,
        'year': random_dates.year,
        'acq_date': random_dates.date
    })
    # print(f"Worker generated {len(random_data)} samples for {country_name}") # Reduce verbosity
    return random_data


def print_country_names(geojson_path='data/countries'):
    """ Prints country names and returns them. Raises error on failure. """
    world = gpd.read_file(geojson_path)
    countries = sorted(world['SOVEREIGNT'].unique())
    print("Available country names in 'SOVEREIGNT' column:")
    for country in countries:
        print(f"- {country}")
    return countries


if __name__ == '__main__':
    available_countries = print_country_names()
    print("-" * 30)

    example_countries = ['Russia'] 
    example_start = '2019-01-01'   
    example_end = '2021-12-31'

    modis_load_countries = [k for k, v in country_mapping.items() if v in example_countries]
    modis_load_countries.extend([c for c in example_countries if c not in country_mapping.values() and c in available_countries])

    print(f"Runnin for countries: {example_countries} (loading as {modis_load_countries})")
    print(f"Date range: {example_start} to {example_end}")
    print(f"Using SPATIAL_COARSENESS: {SPATIAL_COARSENESS}")
    print(f"Using SAMPLES_PER_AREA_PER_YEAR: {SAMPLES_PER_AREA_PER_YEAR}")

    raw_fire_data = load_modis_data(
        countries=modis_load_countries,
        start_date=example_start,
        end_date=example_end
    )

    if raw_fire_data is not None:
            target_data = prepare_target_data(
                data=raw_fire_data,
                countries=modis_load_countries,
                samples_per_area_per_year=SAMPLES_PER_AREA_PER_YEAR
            )

            if not target_data.empty:
                print(f"Shape: {target_data.shape}")
                print(f"Columns: {target_data.columns.tolist()}")
                print("\nValues for 'count':")
                print(target_data['count'].value_counts())
                print("\nHead:")
                print(target_data.head())
                print("\nTail:")
                print(target_data.tail())
                print("\nInfo:")
                target_data.info()
            else:
                print("Final target data is empty.")
    else:
        raise ValueError
