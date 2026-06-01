"""Utilities for generating NN prediction features over full prediction grids."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from src.feature_generation.make_features_nn import generate_all_features
from src.target_generation.prepare_target_new import SPATIAL_COARSENESS
from src.target_generation.create_grid_target import country_mapping


DEFAULT_CACHE_DIR = "data/saved_features/climate_features_cache"
ANCHOR_COLUMNS = [
    "datetime",
    "acq_date",
    "lat_rounded",
    "lon_rounded",
    "latitude",
    "longitude",
    "month",
    "day",
    "year",
    "count",
]


@dataclass(slots=True)
class GeneratedFeatures:
    """Container with file paths of generated artefacts."""

    features_path: str
    climate_path: str


def _resolve_prediction_dates(
    forecast_dates: Sequence[pd.Timestamp | str | np.datetime64],
) -> list[pd.Timestamp]:
    if not forecast_dates:
        raise ValueError(
            "At least one forecast date must be provided to generate prediction features."
        )
    normalized = [pd.to_datetime(dt).normalize() for dt in forecast_dates]
    normalized.sort()
    return normalized


def _resolve_coordinate_bounds(config: dict) -> tuple[float, float, float, float]:
    bounds = config.get("coordinate_bounds")
    if not bounds or len(bounds) != 4:
        raise KeyError(
            "Config must define 'coordinate_bounds' as [min_lat, min_lon, max_lat, max_lon]"
        )
    min_lat, min_lon, max_lat, max_lon = map(float, bounds)
    if min_lat > max_lat or min_lon > max_lon:
        raise ValueError(f"Invalid coordinate bounds: {bounds}")
    return min_lat, min_lon, max_lat, max_lon


def _rounded_range(start: float, stop: float, step: float, *, decimals: int) -> np.ndarray:
    values = np.arange(start, stop + step / 2, step)
    return np.round(values, decimals=decimals)


def _filter_by_countries(
    grid_df: pd.DataFrame,
    prediction_countries: Sequence[str] | None,
    *,
    borders_gdf: gpd.GeoDataFrame | None,
    country_shapes_path: str | os.PathLike[str] | None,
) -> pd.DataFrame:
    if not prediction_countries:
        return grid_df

    if borders_gdf is not None:
        shapes = borders_gdf.copy()
    else:
        if not country_shapes_path:
            raise KeyError(
                "Config must define 'country_shapes_path' when 'prediction_countries' is set."
            )
        if not os.path.exists(country_shapes_path):
            raise FileNotFoundError(f"Country shapes path not found: {country_shapes_path}")
        shapes = gpd.read_file(country_shapes_path)

    if shapes.empty:
        raise ValueError("Country shapes are empty; cannot filter prediction grid.")
    if shapes.crs is None:
        raise ValueError("Country shapes must define a CRS before prediction grid filtering.")
    shapes = shapes.to_crs("EPSG:4326")

    candidate_columns = [
        col
        for col in ("SOVEREIGNT", "ADMIN", "NAME", "COUNTRY", "CNTRY_NAME")
        if col in shapes.columns
    ]
    if not candidate_columns:
        raise ValueError(
            "No usable country-name column found in country shapes. "
            "Expected one of: SOVEREIGNT, ADMIN, NAME, COUNTRY, CNTRY_NAME."
        )

    requested_names = list(prediction_countries)
    mapped_names = [country_mapping.get(name, name) for name in requested_names]
    names_to_match = set(requested_names) | set(mapped_names)
    mask = pd.Series(False, index=shapes.index)
    for column in candidate_columns:
        mask |= shapes[column].isin(names_to_match)

    selected_shapes = shapes[mask]
    if selected_shapes.empty:
        raise ValueError(
            f"prediction_countries {requested_names} not found in country shapes "
            f"using columns {candidate_columns}."
        )

    points = gpd.GeoDataFrame(
        grid_df,
        geometry=gpd.points_from_xy(grid_df["lon_rounded"], grid_df["lat_rounded"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, selected_shapes, how="inner", predicate="within")
    if joined.empty:
        raise ValueError(
            "Country filtering removed all grid points; check coordinate_bounds and "
            "prediction_countries overlap."
        )

    filtered = (
        joined.drop(columns=["geometry", "index_right"]).drop_duplicates().reset_index(drop=True)
    )
    return filtered.loc[:, grid_df.columns]


def _build_prediction_grid(
    config: dict,
    forecast_dates: Sequence[pd.Timestamp],
    *,
    borders_gdf: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    min_lat, min_lon, max_lat, max_lon = _resolve_coordinate_bounds(config)

    step = float(SPATIAL_COARSENESS)
    if step <= 0:
        raise ValueError(f"Invalid SPATIAL_COARSENESS value: {SPATIAL_COARSENESS}")
    decimals = max(0, int(round(-np.log10(step)))) if step < 1 else 0

    lat_vals = _rounded_range(min_lat, max_lat, step, decimals=decimals)
    lon_vals = _rounded_range(min_lon, max_lon, step, decimals=decimals)

    lat_mesh, lon_mesh = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    grid = pd.DataFrame({
        "lat_rounded": lat_mesh.ravel(),
        "lon_rounded": lon_mesh.ravel(),
    })
    grid["latitude"] = grid["lat_rounded"]
    grid["longitude"] = grid["lon_rounded"]

    grid = _filter_by_countries(
        grid,
        config.get("prediction_countries"),
        borders_gdf=borders_gdf,
        country_shapes_path=config.get("country_shapes_path"),
    )

    dates_frame = pd.DataFrame({"datetime": forecast_dates})
    dates_frame["acq_date"] = dates_frame["datetime"]
    dates_frame["month"] = dates_frame["datetime"].dt.month
    dates_frame["day"] = dates_frame["datetime"].dt.day
    dates_frame["year"] = dates_frame["datetime"].dt.year

    grid["_key"] = 1
    dates_frame["_key"] = 1
    expanded = grid.merge(dates_frame, on="_key").drop(columns="_key")
    expanded["count"] = 0

    expanded = expanded.loc[:, ANCHOR_COLUMNS]
    expanded = expanded.sort_values(["datetime", "lat_rounded", "lon_rounded"]).reset_index(drop=True)
    return expanded


def _determine_output_paths(output_dir: str, dates: Sequence[pd.Timestamp]) -> GeneratedFeatures:
    os.makedirs(output_dir, exist_ok=True)
    start_label = dates[0].strftime("%Y%m%d")
    end_label = dates[-1].strftime("%Y%m%d")
    suffix = start_label if len(dates) == 1 else f"{start_label}_{end_label}"
    base = os.path.join(output_dir, f"prediction_features_{suffix}")
    return GeneratedFeatures(features_path=f"{base}.parquet", climate_path=f"{base}.npz")


def generate_prediction_features(
    config: dict,
    forecast_dates: Sequence[pd.Timestamp | str | np.datetime64],
    *,
    output_dir: str,
    borders_gdf: gpd.GeoDataFrame | None = None,
    use_cached_climate: bool | None = None,
) -> GeneratedFeatures:
    """Generate NN prediction features for the provided dates and persist them."""

    resolved_dates = _resolve_prediction_dates(forecast_dates)
    prediction_grid = _build_prediction_grid(
        config,
        resolved_dates,
        borders_gdf=borders_gdf,
    )

    if prediction_grid.empty:
        raise RuntimeError(
            "Prediction grid generation produced no rows; verify coordinate bounds and filters."
        )

    climate_params = (
        config.get("climate_data_params_prediction")
        or config.get("climate_data_params")
    )
    if not climate_params:
        raise KeyError(
            "Config must define 'climate_data_params_prediction' or 'climate_data_params'."
        )

    land_params = config.get("land_data_params")
    if not land_params:
        raise KeyError("Config must define 'land_data_params' for feature generation.")

    elevation_params = config.get("elevation_data_params")
    if not elevation_params:
        raise KeyError("Config must define 'elevation_data_params' for feature generation.")

    road_params = config.get("road_data_params", {})
    night_light_params = config.get("night_light_data_params", {})

    cache_dir = config.get("climate_features_cache_dir", DEFAULT_CACHE_DIR)
    use_cached = bool(use_cached_climate if use_cached_climate is not None else True)

    features_df, climate_matrix = generate_all_features(
        df_target=prediction_grid,
        climate_data_dir=climate_params["climate_data_dir"],
        climate_variables=climate_params["climate_variables"],
        climate_n_days=int(climate_params["n_days"]),
        elevation_file_path=elevation_params["elevation_file"],
        elevation_window_sizes=elevation_params.get("window_size", [0.25]),
        road_feature_map_path=road_params.get("feature_map_path", ""),
        use_road_features=bool(road_params.get("use_road_features", False)),
        night_light_feature_map_path=night_light_params.get("feature_map_path"),
        use_night_light_features=bool(night_light_params.get("use_night_light_features", False)),
        night_light_legacy_viirs_feature_map_path=night_light_params.get(
            "legacy_viirs_feature_map_path"
        ),
        night_light_legacy_viirs_feature_prefix=night_light_params.get(
            "legacy_viirs_feature_prefix", "viirs_"
        ),
        night_light_annual_source_dir=night_light_params.get("annual_source_dir"),
        night_light_recent_feature_name=night_light_params.get(
            "recent_feature_name", "night_light_radiance_recent"
        ),
        night_light_recent_source_glob=night_light_params.get("recent_source_glob", "*.tif"),
        night_light_recent_cache_path=night_light_params.get("recent_cache_path"),
        night_light_cf_cvg_source_glob=night_light_params.get("cf_cvg_source_glob"),
        night_light_cf_cvg_feature_name=night_light_params.get(
            "cf_cvg_feature_name", "night_light_cf_cvg_recent"
        ),
        night_light_cf_cvg_cache_path=night_light_params.get("cf_cvg_cache_path"),
        night_light_cf_filtered_feature_name=night_light_params.get(
            "cf_filtered_feature_name", "night_light_radiance_recent_cf_filtered"
        ),
        night_light_min_cf_cvg=night_light_params.get("min_cf_cvg"),
        night_light_cf_filter_north_lat_min=night_light_params.get(
            "cf_filter_north_lat_min", 58.0
        ),
        night_light_black_marble_source_dir=night_light_params.get("black_marble_source_dir"),
        night_light_black_marble_cache_path=night_light_params.get("black_marble_cache_path"),
        fire_index_npz_path=land_params.get(
            "fire_index_npz_path", "data/land_features/fire_index_features.npz"
        ),
        land_data_files=land_params["land_data_files"],
        wwf_shp_path=land_params["wwf_shp_path"],
        landsea_mask_path=land_params.get("landsea_mask_path"),
        landsea_distance_path=land_params.get("landsea_distance_path"),
        landsea_mask_threshold=land_params.get("landsea_mask_threshold", 70),
        anchor_cols=ANCHOR_COLUMNS,
        test_mode=True,
        skip_climate=False,
        use_cached_files=use_cached,
        cache_dir=cache_dir,
    )

    if features_df.empty:
        raise RuntimeError("Generated features dataframe is empty after pipeline execution.")
    if climate_matrix is None or climate_matrix.size == 0:
        raise RuntimeError("Climate matrix is empty; cannot proceed with NN predictions.")

    duplicated_cols = features_df.columns[features_df.columns.duplicated()].unique()
    if duplicated_cols.size:
        print(
            "Warning: Duplicate feature columns detected after generation; keeping first occurrence for: "
            f"{list(duplicated_cols)}"
        )
        features_df = features_df.loc[:, ~features_df.columns.duplicated(keep="first")]

    output_paths = _determine_output_paths(output_dir, resolved_dates)

    features_df.to_parquet(output_paths.features_path)
    np.savez(output_paths.climate_path, climate_features=climate_matrix)

    print(
        f"Generated NN prediction features for {len(resolved_dates)} day(s); saved to {output_paths.features_path}"
    )

    return output_paths
