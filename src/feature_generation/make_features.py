import os
import sys
import time
import argparse
import yaml
import pandas as pd
import numpy as np

sys.path.append(os.getcwd()) 

from src.target_generation.prepare_target_new import (
    SPATIAL_COARSENESS,
    _add_recent_fire_history_features,
    _fire_count_windows,
    _initial_positive_counts,
    _target_section_settings,
    expand_positive_points,
    load_modis_data,
    prepare_target_data,
)
from src.feature_generation.prepare_climate_data import prepare_data as prepare_climate_data_func
from src.feature_generation.prepare_land import (
    get_elevation_stats,
    prepare_land_data,
    assign_ecoregion,
    landsea_distance,
)
from src.feature_generation.prepare_night_light_features import (
    DEFAULT_BLACK_MARBLE_FILTERED_FEATURE_NAME,
    DEFAULT_BLACK_MARBLE_OBSERVATIONS_FEATURE_NAME,
    DEFAULT_BLACK_MARBLE_QUALITY_FEATURE_NAME,
    DEFAULT_BLACK_MARBLE_RADIANCE_FEATURE_NAME,
    DEFAULT_RECENT_COVERAGE_FEATURE_NAME,
    DEFAULT_RECENT_FILTERED_FEATURE_NAME,
    DEFAULT_RECENT_FEATURE_NAME,
    get_night_light_features_for_coords,
)
from src.feature_generation.prepare_road_features import get_road_features_for_coords
from src.feature_generation.prepare_fire_index_features import get_fire_index_features


# --- TARGET DATA GENERATION ---
def crop_target_df(df, coordinate_bounds):
    min_lat, min_lon, max_lat, max_lon = coordinate_bounds[0], coordinate_bounds[1], coordinate_bounds[2], coordinate_bounds[3]
    
    df_cropped = df[
        (df['lat_rounded'] >= min_lat) & (df['lat_rounded'] <= max_lat) &
        (df['lon_rounded'] >= min_lon) & (df['lon_rounded'] <= max_lon)
    ]
    print(f'Target df shape: {df.shape[0]} rows -> after cropping: {df_cropped.shape[0]} rows for bounds {coordinate_bounds}')
    return df_cropped

def generate_target_data(
    modis_data_path,
    modis_countries,
    target_samples_per_area_per_year,
    coordinate_bounds, # [min_lat, min_lon, max_lat, max_lon]
    start_date,
    end_date,
    use_cached: bool = False,
    cache_path: str | None = None,
    feature_config: dict | None = None,
):
    print(f"--- Generating Target Data ({start_date} to {end_date}) ---")

    if use_cached and cache_path and os.path.exists(cache_path):
        print(f"Loading cached target data from '{cache_path}'...")
        t_start = time.time()
        df_target_sorted = pd.read_parquet(cache_path)
        print(f"Loaded target data from cache. Shape: {df_target_sorted.shape}. Time: {time.time() - t_start:.2f}s")
        return df_target_sorted

    t_start = time.time()
    df_modis = load_modis_data(modis_data_path, modis_countries, start_date, end_date)
    print(f"Loaded MODIS data shape: {df_modis.shape}")
    
    df_target = prepare_target_data(
        df_modis,
        modis_countries,
        samples_per_area_per_year=target_samples_per_area_per_year,
        coordinate_bounds=coordinate_bounds,
        negative_sampling_feature_config=feature_config,
    )
    print(f"Target data shape after adding negative samples: {df_target.shape}")
    
    df_target_cropped = crop_target_df(df_target, coordinate_bounds)
    df_target_sorted = df_target_cropped.sort_values(by='datetime').reset_index(drop=True)
    
    print(f"Final target data shape: {df_target_sorted.shape}")
    print(f"Target data generation took {time.time() - t_start:.2f} seconds.")

    if cache_path:
        print(f"Saving target data to '{cache_path}'...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_target_sorted.to_parquet(cache_path)

    return df_target_sorted

# --- INDIVIDUAL FEATURE PREPARATION WRAPPERS (largely as in original) ---
def _prepare_unique_feature_rows(
    target_df: pd.DataFrame,
    key_cols: list[str],
    feature_func,
    label: str,
):
    """Run an expensive row-aligned feature function once per unique key."""

    unique_target = target_df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    if len(unique_target) != len(target_df):
        print(
            f"{label}: using {len(unique_target)} unique key rows instead of "
            f"{len(target_df)} target rows."
        )

    feature_df, feature_names = feature_func(unique_target)
    if not feature_names:
        return feature_df, feature_names

    if len(unique_target) == len(target_df):
        return feature_df.reset_index(drop=True), feature_names

    feature_block = pd.concat(
        [
            unique_target[key_cols].reset_index(drop=True),
            feature_df[feature_names].reset_index(drop=True),
        ],
        axis=1,
    )
    aligned = target_df[key_cols].merge(feature_block, on=key_cols, how="left", sort=False)
    return aligned.reset_index(drop=True), feature_names


def prepare_elevation_features_wrapper(target_df: pd.DataFrame, elevation_file: str, window_sizes: list[float]):
    return _prepare_unique_feature_rows(
        target_df,
        ["lat_rounded", "lon_rounded"],
        lambda unique_df: get_elevation_stats(elevation_file, unique_df, window_sizes),
        "Elevation features",
    )

def prepare_road_features_wrapper(target_df: pd.DataFrame, feature_map_path: str):
    def _compute(unique_df: pd.DataFrame):
        coords_array = unique_df[['lat_rounded', 'lon_rounded']].to_numpy()
        df_road = get_road_features_for_coords(coords=coords_array.T, npz_path=feature_map_path)
        road_feature_names = [
            col for col in df_road.columns
            if col not in ['lat', 'lon', 'latitude', 'longitude', 'lat_rounded', 'lon_rounded']
        ]
        return df_road, road_feature_names

    return _prepare_unique_feature_rows(
        target_df,
        ["lat_rounded", "lon_rounded"],
        _compute,
        "Road features",
    )

def _target_year_column(target_df: pd.DataFrame) -> pd.Series | None:
    if "year" in target_df.columns:
        return target_df["year"]
    if "datetime" in target_df.columns:
        return pd.to_datetime(target_df["datetime"], errors="coerce").dt.year
    return None


def prepare_night_light_features_wrapper(
    target_df: pd.DataFrame,
    feature_map_path: str,
    annual_source_dir: str | None = None,
    recent_feature_name: str = DEFAULT_RECENT_FEATURE_NAME,
    recent_source_glob: str = "*.tif",
    recent_cache_path: str | None = None,
    cf_cvg_source_glob: str | None = None,
    cf_cvg_feature_name: str = DEFAULT_RECENT_COVERAGE_FEATURE_NAME,
    cf_cvg_cache_path: str | None = None,
    cf_filtered_feature_name: str | None = DEFAULT_RECENT_FILTERED_FEATURE_NAME,
    min_cf_cvg: float | None = None,
    cf_filter_north_lat_min: float = 58.0,
    black_marble_source_dir: str | None = None,
    black_marble_cache_path: str | None = None,
    black_marble_missing_tile_fallback: str | None = None,
    black_marble_fallback_feature_name: str = "night_light_radiance_2024",
):
    working_df = target_df
    key_cols = ["lat_rounded", "lon_rounded"]
    use_dated_night_lights = annual_source_dir is not None or black_marble_source_dir is not None
    use_recent = use_dated_night_lights
    if use_recent:
        year_values = _target_year_column(target_df)
        if year_values is None:
            raise ValueError(
                "Dated night-light features are configured, but target rows do not "
                "contain 'year' or 'datetime'."
            )
        working_df = target_df.copy()
        working_df["_night_light_target_year"] = year_values
        key_cols.append("_night_light_target_year")

    def _compute(unique_df: pd.DataFrame):
        coords_array = unique_df[['lat_rounded', 'lon_rounded']].to_numpy()
        years = (
            unique_df["_night_light_target_year"].to_numpy()
            if use_recent
            else None
        )
        df_lights = get_night_light_features_for_coords(
            coords=coords_array.T,
            feature_map_path=feature_map_path,
            years=years,
            annual_source_dir=annual_source_dir,
            recent_feature_name=recent_feature_name,
            recent_source_glob=recent_source_glob,
            recent_cache_path=recent_cache_path,
            cf_cvg_source_glob=cf_cvg_source_glob,
            cf_cvg_feature_name=cf_cvg_feature_name,
            cf_cvg_cache_path=cf_cvg_cache_path,
            cf_filtered_feature_name=cf_filtered_feature_name,
            min_cf_cvg=min_cf_cvg,
            cf_filter_north_lat_min=cf_filter_north_lat_min,
            black_marble_source_dir=black_marble_source_dir,
            black_marble_cache_path=black_marble_cache_path,
            black_marble_missing_tile_fallback=black_marble_missing_tile_fallback,
            black_marble_fallback_feature_name=black_marble_fallback_feature_name,
        )
        light_feature_names = [
            col for col in df_lights.columns
            if col not in ['lat', 'lon', 'latitude', 'longitude', 'lat_rounded', 'lon_rounded']
        ]
        return df_lights, light_feature_names

    return _prepare_unique_feature_rows(
        working_df,
        key_cols,
        _compute,
        "Night-light features",
    )

def prepare_land_features_wrapper(target_df: pd.DataFrame, land_data_files: list[str]):
    return _prepare_unique_feature_rows(
        target_df,
        ["lat_rounded", "lon_rounded"],
        lambda unique_df: prepare_land_data(land_data_files=land_data_files, target_df=unique_df, radius_meters=10000),
        "Land features",
    )

def prepare_fire_index_features_wrapper(target_df: pd.DataFrame, fire_index_npz_path: str):
    return _prepare_unique_feature_rows(
        target_df,
        ["lat_rounded", "lon_rounded", "month"],
        lambda unique_df: get_fire_index_features(fire_index_npz_path, unique_df),
        "Fire index features",
    )

def prepare_ecoregion_wrapper(target_df: pd.DataFrame, wwf_shp_path: str):
    return _prepare_unique_feature_rows(
        target_df,
        ["lat_rounded", "lon_rounded"],
        lambda unique_df: assign_ecoregion(df=unique_df, wwf_shp=wwf_shp_path),
        "Ecoregion features",
    )

def prepare_landsea_distance_wrapper(
    target_df: pd.DataFrame,
    mask_path: str | None = None,
    dist_path: str | None = None,
):
    landsea_kwargs = {}
    if mask_path:
        landsea_kwargs["mask_path"] = mask_path
    if dist_path:
        landsea_kwargs["dist_path"] = dist_path
    return _prepare_unique_feature_rows(
        target_df,
        ["lat_rounded", "lon_rounded"],
        lambda unique_df: landsea_distance(
            unique_df,
            **landsea_kwargs,
        ),
        "Land-sea distance features",
    )

def merge_features(
    df_with_climate_features_and_target_anchors: pd.DataFrame | None,
    all_climate_feature_names: list[str],
    elevation_features: pd.DataFrame | None, 
    elevation_feature_names: list[str], 
    road_features: pd.DataFrame | None, 
    road_feature_names: list[str], 
    night_light_features: pd.DataFrame | None,
    night_light_feature_names: list[str],
    fire_index_features: pd.DataFrame | None,
    fire_index_feature_names: list[str],
    land_df: pd.DataFrame | None, 
    land_feature_names: list[str], 
    ecoregion_features: pd.DataFrame | None,
    ecoregion_feature_names: list[str],
    landsea_distance_features: pd.DataFrame | None,
    landsea_distance_feature_names: list[str],
    anchor_cols: list[str],
    drop_by_sea_mask: bool = True,
    landsea_mask_threshold: float = 70,
):
    all_dataframes_to_concat = []
    final_feature_columns_ordered = []

    if df_with_climate_features_and_target_anchors is not None:
        df_anchors = df_with_climate_features_and_target_anchors[anchor_cols].reset_index(drop=True)
        all_dataframes_to_concat.append(df_anchors)
        final_feature_columns_ordered.extend(anchor_cols)

        if all_climate_feature_names:
            df_clim_feats_only = df_with_climate_features_and_target_anchors[all_climate_feature_names].reset_index(drop=True)
            all_dataframes_to_concat.append(df_clim_feats_only)
            final_feature_columns_ordered.extend(all_climate_feature_names)
    else:
        raise ValueError("Warning: df_with_climate_features_and_target_anchors is None. This shouldn't happen if handled correctly by caller.")


    feature_sets = [
        (elevation_features, elevation_feature_names),
        (road_features, road_feature_names),
        (night_light_features, night_light_feature_names),
        (fire_index_features, fire_index_feature_names),
        (land_df, land_feature_names), 
        (ecoregion_features, ecoregion_feature_names),
        (landsea_distance_features, landsea_distance_feature_names),
    ]

    for features_df, feature_names_list in feature_sets:
        if feature_names_list and features_df is not None and not features_df.empty:
            all_dataframes_to_concat.append(features_df[feature_names_list].reset_index(drop=True))
            final_feature_columns_ordered.extend(feature_names_list)

    if not all_dataframes_to_concat:
        print("Warning: No feature dataframes available for merging.")
        return pd.DataFrame()
    
    final_features_df = pd.concat(all_dataframes_to_concat, axis=1)
    final_features_df = final_features_df.loc[:, ~final_features_df.columns.duplicated(keep='first')]
    
    existing_cols_in_order = [col for col in final_feature_columns_ordered if col in final_features_df.columns]
    final_features_df = final_features_df[existing_cols_in_order]

    if drop_by_sea_mask:
        if 'landseamask' not in final_features_df.columns:
            print("WARNING: 'landseamask' column not found in final features df. Skipping sea mask drop.")
        else:
            rows_before_sea_mask_drop = final_features_df.shape[0]
            final_features_df = final_features_df[
                final_features_df['landseamask'] < float(landsea_mask_threshold)
            ].reset_index(drop=True)
            rows_after_sea_mask_drop = final_features_df.shape[0]
            print(f"🌊 Using landseamask, dropped {rows_before_sea_mask_drop - rows_after_sea_mask_drop} rows due to sea_mask < {landsea_mask_threshold:g} "
                f"Final features df shape: {final_features_df.shape}")
            
    if 'count' not in final_features_df.columns and 'count' in anchor_cols:
        print("Warning: 'count' column specified in anchor_cols is missing from the final merged DataFrame.")
    
    return final_features_df


def generate_all_features(
    df_target: pd.DataFrame,
    # Climate data loading params
    climate_data_dir: str,
    climate_variables: list[str],
    climate_n_days: int,
    # Climate feature generation params
    generate_climate_max_length: int,
    generate_climate_lags: list[int],
    generate_climate_windows: list[int],
    generate_climate_spans: list[int],
    generate_climate_trend_window: int,
    generate_climate_features_to_include: dict,
    # Elevation params
    elevation_file_path: str,
    elevation_window_sizes: list[float],
    # Road params
    road_feature_map_path: str,
    use_road_features: bool,
    # Night-light params
    night_light_feature_map_path: str | None,
    use_night_light_features: bool,
    # Fire index params
    fire_index_npz_path: str,
    # Land data params
    land_data_files: list[str],
    # Ecoregion params
    wwf_shp_path: str,
    # Other controls
    anchor_cols: list[str],
    climate_location_batch_size: int | None = None,
    climate_max_time_span_days: int | None = None,
    persist_climate_dataset: bool = False,
    strict_climate_bounds: bool = True,
    landsea_mask_path: str | None = None,
    landsea_distance_path: str | None = None,
    landsea_mask_threshold: float = 70,
    test_mode: bool = False,
    skip_climate: bool = False,
    use_cached_files: bool = False,
    cache_dir: str = 'data/saved_features/climate_features_cache',
    night_light_annual_source_dir: str | None = None,
    night_light_recent_feature_name: str = DEFAULT_RECENT_FEATURE_NAME,
    night_light_recent_source_glob: str = "*.tif",
    night_light_recent_cache_path: str | None = None,
    night_light_cf_cvg_source_glob: str | None = None,
    night_light_cf_cvg_feature_name: str = DEFAULT_RECENT_COVERAGE_FEATURE_NAME,
    night_light_cf_cvg_cache_path: str | None = None,
    night_light_cf_filtered_feature_name: str | None = DEFAULT_RECENT_FILTERED_FEATURE_NAME,
    night_light_min_cf_cvg: float | None = None,
    night_light_cf_filter_north_lat_min: float = 58.0,
    night_light_black_marble_source_dir: str | None = None,
    night_light_black_marble_cache_path: str | None = None,
    night_light_black_marble_missing_tile_fallback: str | None = None,
    night_light_black_marble_fallback_feature_name: str = "night_light_radiance_2024",
) -> pd.DataFrame:
    """
    Orchestrates the full feature generation pipeline.
    """
    total_script_start_time = time.time()
    print(f"--- Starting Full Feature Generation ---")
    
    df_processed_target_with_climate: pd.DataFrame | None = None
    all_climate_feature_names_list: list[str] = []

    if not skip_climate:
        print("--- Generating Climate Features ---")
        climate_start_time = time.time()
        df_processed_target_with_climate, all_climate_feature_names_list, ts_matrix = prepare_climate_data_func(
            climate_data_dir=climate_data_dir,
            climate_variables=climate_variables,
            target_df=df_target,
            n_days=climate_n_days,
            test_mode=test_mode,
            max_length_features=generate_climate_max_length,
            lags_features=generate_climate_lags,
            windows_features=generate_climate_windows,
            spans_features=generate_climate_spans,
            trend_window_features=generate_climate_trend_window,
            features_to_include_config=generate_climate_features_to_include,
            use_cached_files=use_cached_files,
            cache_dir=cache_dir,
            return_features_df=True,
            location_batch_size=climate_location_batch_size,
            max_time_span_days=climate_max_time_span_days,
            persist_dataset=persist_climate_dataset,
            strict_climate_bounds=strict_climate_bounds,
        )
        assert ts_matrix.shape[0] == len(df_target)
        print(
            f"Climate features generated ({len(all_climate_feature_names_list)} columns). "
            f"Time: {time.time() - climate_start_time:.2f}s"
        )
        print(f"DataFrame shape after climate processing: {df_processed_target_with_climate.shape}")
    else:
        print("--- Skipping Climate Feature Generation ---")
        # df_target already contains anchor columns. This will be the base.
        df_processed_target_with_climate = df_target.copy()
        all_climate_feature_names_list = []

    # Generate other features using the original df_target for coordinate references
    # These functions are expected to return DataFrames that can be aligned (e.g., via index or merge keys)
    # or just the feature columns that are row-aligned.
    # The merge_features function expects feature DFs that might include lat/lon, and it selects only new feature columns.

    print("--- Generating Elevation Features ---")
    elev_start_time = time.time()
    df_elevation, elevation_feature_names = prepare_elevation_features_wrapper(
        target_df=df_target,
        elevation_file=elevation_file_path, 
        window_sizes=elevation_window_sizes
    )
    print(f"Elevation features generated ({len(elevation_feature_names)} columns). Shape: {df_elevation.shape}. Time: {time.time() - elev_start_time:.2f}s")

    df_road, road_feature_names = None, []
    if use_road_features:
        print("--- Generating Road Features ---")
        road_start_time = time.time()
        df_road, road_feature_names = prepare_road_features_wrapper(
            target_df=df_target,
            feature_map_path=road_feature_map_path
        )
        print(f"Road features generated ({len(road_feature_names)} columns). Shape: {df_road.shape}. Time: {time.time() - road_start_time:.2f}s")
    else:
        print("--- Skipping Road Features ---")

    df_night_light, night_light_feature_names = None, []
    if use_night_light_features:
        if not night_light_feature_map_path:
            raise ValueError("Night-light features enabled but night_light_feature_map_path is empty.")
        print("--- Generating Night-Light Features ---")
        night_light_start_time = time.time()
        df_night_light, night_light_feature_names = prepare_night_light_features_wrapper(
            target_df=df_target,
            feature_map_path=night_light_feature_map_path,
            annual_source_dir=night_light_annual_source_dir,
            recent_feature_name=night_light_recent_feature_name,
            recent_source_glob=night_light_recent_source_glob,
            recent_cache_path=night_light_recent_cache_path,
            cf_cvg_source_glob=night_light_cf_cvg_source_glob,
            cf_cvg_feature_name=night_light_cf_cvg_feature_name,
            cf_cvg_cache_path=night_light_cf_cvg_cache_path,
            cf_filtered_feature_name=night_light_cf_filtered_feature_name,
            min_cf_cvg=night_light_min_cf_cvg,
            cf_filter_north_lat_min=night_light_cf_filter_north_lat_min,
            black_marble_source_dir=night_light_black_marble_source_dir,
            black_marble_cache_path=night_light_black_marble_cache_path,
            black_marble_missing_tile_fallback=night_light_black_marble_missing_tile_fallback,
            black_marble_fallback_feature_name=night_light_black_marble_fallback_feature_name,
        )
        print(
            f"Night-light features generated ({len(night_light_feature_names)} columns). "
            f"Shape: {df_night_light.shape}. Time: {time.time() - night_light_start_time:.2f}s"
        )
    else:
        print("--- Skipping Night-Light Features ---")
        
    print("--- Generating Land Data Features ---")
    land_start_time = time.time()
    df_land, land_feature_names = prepare_land_features_wrapper(
        target_df=df_target,
        land_data_files=land_data_files
    )
    print(f"Land features generated ({len(land_feature_names)} columns). Shape: {df_land.shape}. Time: {time.time() - land_start_time:.2f}s")

    print("--- Generating Fire Index Features ---")
    fire_idx_start_time = time.time()
    df_fire_index, fire_index_feature_names = prepare_fire_index_features_wrapper(
        target_df=df_target,
        fire_index_npz_path=fire_index_npz_path
    )
    print(f"Fire index features generated ({len(fire_index_feature_names)} columns). Shape: {df_fire_index.shape}. Time: {time.time() - fire_idx_start_time:.2f}s")
    
    print("--- Generating Ecoregions ---")
    ecoregion_start_time = time.time()
    df_ecoregion, ecoregion_feature_names = prepare_ecoregion_wrapper(
        target_df=df_target,
        wwf_shp_path=wwf_shp_path
    )
    print(f"Ecoregions generated ({len(ecoregion_feature_names)} columns). Shape: {df_ecoregion.shape}. Time: {time.time() - ecoregion_start_time:.2f}s")

    print("--- Generating Landsea Distance ---")
    landsea_distance_start_time = time.time()
    df_landsea_distance, landsea_distance_feature_names = prepare_landsea_distance_wrapper(
        target_df=df_target,
        mask_path=landsea_mask_path,
        dist_path=landsea_distance_path,
    )
    print(f"Landsea distance features generated ({len(landsea_distance_feature_names)} columns). Shape: {df_landsea_distance.shape}. Time: {time.time() - landsea_distance_start_time:.2f}s")

    print("--- Merging All Features ---")
    merge_start_time = time.time()
    final_df = merge_features(
        df_with_climate_features_and_target_anchors=df_processed_target_with_climate,
        all_climate_feature_names=all_climate_feature_names_list,
        elevation_features=df_elevation, 
        elevation_feature_names=elevation_feature_names,
        road_features=df_road, 
        road_feature_names=road_feature_names,
        night_light_features=df_night_light,
        night_light_feature_names=night_light_feature_names,
        fire_index_features=df_fire_index,
        fire_index_feature_names=fire_index_feature_names,
        land_df=df_land, 
        land_feature_names=land_feature_names, 
        ecoregion_features=df_ecoregion,
        ecoregion_feature_names=ecoregion_feature_names,
        landsea_distance_features=df_landsea_distance,
        landsea_distance_feature_names=landsea_distance_feature_names,
        anchor_cols=anchor_cols,
        drop_by_sea_mask=True,
        landsea_mask_threshold=landsea_mask_threshold,
    )
    print(f"Merging took {time.time() - merge_start_time:.2f}s")
    
    # Drop temporary 'id' column if it exists and is not an anchor_col
    temp_id_col = 'id' # Assuming 'id' is the name of the temporary column from climate processing
    if temp_id_col in final_df.columns and (anchor_cols is None or temp_id_col not in anchor_cols):
        print(f"Dropping temporary '{temp_id_col}' column.")
        final_df = final_df.drop(columns=[temp_id_col])
        
    print(f"Final merged DataFrame shape: {final_df.shape}")
    total_script_duration = time.time() - total_script_start_time
    print(f"Total feature generation pipeline took {total_script_duration:.2f} seconds ({total_script_duration/60:.2f} minutes).")

    float_cols = final_df.select_dtypes(include=["float64"]).columns
    int_cols = final_df.select_dtypes(include=["int64"]).columns
    final_df[float_cols] = final_df[float_cols].astype("float32")
    final_df[int_cols] = final_df[int_cols].astype("int32")

    return final_df


def _load_config(config_or_path: str | dict) -> dict:
    if isinstance(config_or_path, str):
        with open(config_or_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Using loaded config file: {config_or_path}")
        return config
    if isinstance(config_or_path, dict):
        print(f"Using config dictionary: {config_or_path}")
        return config_or_path
    raise TypeError("config_or_path must be a string path or a config dictionary.")


def _expected_recent_fire_history_columns(settings: dict) -> list[str]:
    if not bool(settings.get("enabled", True)):
        return []
    radii = sorted({int(radius) for radius in settings.get("radii_cells", [0, 1, 2]) if int(radius) >= 0})
    windows = _fire_count_windows(settings)
    columns = [
        f"past_fire_count_r{radius}_{window['name']}"
        for radius in radii
        for window in windows
    ]
    if bool(settings.get("include_days_since", True)):
        days_since_radii = sorted(
            {
                int(radius)
                for radius in settings.get("days_since_radii_cells", radii)
                if int(radius) >= 0
            }
        )
        columns.extend(f"days_since_fire_r{radius}" for radius in days_since_radii)
    return columns


def _positive_history_frame(raw: pd.DataFrame, resolution: float = SPATIAL_COARSENESS) -> pd.DataFrame:
    columns = ["acq_date", "lat_rounded", "lon_rounded", "country", "count"]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rounding_precision = int(-np.log10(resolution)) if resolution > 0 else int(-np.log10(SPATIAL_COARSENESS))
    data = raw.copy()
    data["lat_rounded"] = data["latitude"].round(rounding_precision)
    data["lon_rounded"] = data["longitude"].round(rounding_precision)
    data["acq_date"] = pd.to_datetime(data["acq_date"]).dt.date
    data["count"] = _initial_positive_counts(data["latitude"], data["longitude"])
    grouped = (
        data.groupby(["lat_rounded", "lon_rounded", "acq_date"], observed=True)
        .agg(count=("count", "sum"), country=("country", "first"))
        .reset_index()
    )
    expanded = expand_positive_points(
        grouped,
        spatial_coarseness=resolution,
        lat_col="lat_rounded",
        lon_col="lon_rounded",
        count_col="count",
    )
    expanded["acq_date"] = pd.to_datetime(expanded["acq_date"])
    expanded["count"] = 1
    return expanded[columns]


def _add_recent_fire_history_for_prediction(
    target: pd.DataFrame,
    config: dict,
    *,
    resolution: float = SPATIAL_COARSENESS,
) -> pd.DataFrame:
    settings = _target_section_settings("recent_fire_history", config)
    expected = _expected_recent_fire_history_columns(settings)
    if not expected or target.empty or all(col in target.columns for col in expected):
        return target

    date_source = target["acq_date"] if "acq_date" in target.columns else target.get("datetime")
    dates = pd.to_datetime(date_source, errors="coerce")
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return target

    windows = _fire_count_windows(settings)
    max_lookback_days = max([int(window["end_days"]) for window in windows] + [int(settings.get("days_since_cap", 365))])
    history_start = valid_dates.min() - pd.Timedelta(days=max_lookback_days + 1)
    history_end = valid_dates.max()
    if "country" in target.columns and target["country"].notna().any():
        countries = sorted(str(value) for value in target["country"].dropna().unique())
    else:
        countries = list(config.get("modis_countries") or config.get("prediction_countries") or [])
    if not countries:
        return target

    print(
        "Adding recent fire-history features for prediction rows "
        f"({history_start.date()} to {history_end.date()}, countries={countries})."
    )
    raw = load_modis_data(
        config.get("modis_data_path", "data/modis"),
        countries,
        history_start.strftime("%Y-%m-%d"),
        history_end.strftime("%Y-%m-%d"),
    )
    positives = _positive_history_frame(raw, resolution=resolution)
    history_target = target.copy()
    if "country" not in history_target.columns:
        fallback_country = countries[0] if len(countries) == 1 else "__unknown__"
        history_target["country"] = fallback_country
    else:
        history_target["country"] = history_target["country"].fillna("__unknown__").astype(str)
    enriched = _add_recent_fire_history_features(history_target, positives=positives, settings=settings)
    missing = [col for col in expected if col not in enriched.columns]
    if missing:
        raise KeyError(f"Recent fire-history generation missed expected columns: {missing}")
    return enriched


def _build_anchor_columns(df_target_processed: pd.DataFrame, extra_anchor_cols: list[str] | None = None) -> list[str]:
    anchor_columns = ['datetime', 'lat_rounded', 'lon_rounded', 'month', 'day', 'year', 'count']
    target_metadata_columns = {
        'soft_label',
        'negative_stratum',
        'sampling_probability',
        'sample_weight',
        'nearest_positive_distance_cells',
        'nearest_positive_delta_days',
        'country',
    }
    target_derived_prefixes = (
        'past_fire_count_',
        'recent_fire_count_',
        'days_since_fire_',
    )
    for col in df_target_processed.columns:
        if col in target_metadata_columns or col.startswith(target_derived_prefixes):
            if col not in anchor_columns:
                anchor_columns.append(col)
    if extra_anchor_cols:
        for col in extra_anchor_cols:
            if col not in anchor_columns:
                anchor_columns.append(col)

    for col in anchor_columns:
        if col not in df_target_processed.columns:
            if col == 'year' and 'datetime' in df_target_processed.columns:
                print(f"Generating 'year' column from 'datetime' for anchors.")
                df_target_processed['year'] = pd.to_datetime(df_target_processed['datetime']).dt.year
            else:
                print(f"Warning: Anchor column '{col}' not found in df_target_processed. It will be missing if not generated by features.")

    return [col for col in anchor_columns if col in df_target_processed.columns]


def make_features_from_target_df(
    config_or_path: str | dict,
    df_target_processed: pd.DataFrame,
    test_mode: bool,
    use_cached_files: bool = False,
    cache_dir: str = 'climate_features_cache',
    extra_anchor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Generate the final feature dataframe from an already prepared target table."""

    config = _load_config(config_or_path)

    current_climate_params = config.get('climate_data_params_test', config['climate_data_params']) \
        if test_mode else config['climate_data_params']

    generate_climate_params_cfg = config['generate_climate_params']
    trend_window_for_climate = generate_climate_params_cfg.get('trend_window', [21, 90])
    if 'trend_window' not in generate_climate_params_cfg:
         print(f"Using default trend_window: {trend_window_for_climate} for climate features generation.")

    elevation_params_cfg = config['elevation_data_params']
    road_params_cfg = config['road_data_params']
    night_light_params_cfg = config.get('night_light_data_params', {})
    land_params_cfg = config['land_data_params']

    skip_climate_cfg = config.get('skip_climate', False)
    strict_climate_bounds_cfg = config.get('strict_climate_bounds', True)
    df_target_processed = _add_recent_fire_history_for_prediction(
        df_target_processed,
        config,
        resolution=float(config.get("spatial_coarseness", SPATIAL_COARSENESS)),
    )
    anchor_columns = _build_anchor_columns(df_target_processed, extra_anchor_cols=extra_anchor_cols)

    return generate_all_features(
        df_target=df_target_processed,
        # Climate data loading
        climate_data_dir=current_climate_params["climate_data_dir"],
        climate_variables=current_climate_params["climate_variables"],
        climate_n_days=current_climate_params["n_days"],
        climate_location_batch_size=current_climate_params.get("location_batch_size"),
        climate_max_time_span_days=current_climate_params.get("max_time_span_days"),
        persist_climate_dataset=current_climate_params.get("persist_dataset", False),
        # Climate feature generation
        generate_climate_max_length=generate_climate_params_cfg["max_length"],
        generate_climate_lags=generate_climate_params_cfg["lags"],
        generate_climate_windows=generate_climate_params_cfg["windows"],
        generate_climate_spans=generate_climate_params_cfg["spans"],
        generate_climate_trend_window=trend_window_for_climate,
        generate_climate_features_to_include=generate_climate_params_cfg["features_to_include"],
        # Elevation
        elevation_file_path=elevation_params_cfg["elevation_file"],
        elevation_window_sizes=elevation_params_cfg.get("window_size", [0.25]),
        # Road
        road_feature_map_path=road_params_cfg["feature_map_path"],
        use_road_features=road_params_cfg.get("use_road_features", False),
        # Night lights
        night_light_feature_map_path=night_light_params_cfg.get("feature_map_path"),
        use_night_light_features=night_light_params_cfg.get("use_night_light_features", False),
        night_light_annual_source_dir=night_light_params_cfg.get("annual_source_dir"),
        night_light_recent_feature_name=night_light_params_cfg.get(
            "recent_feature_name", DEFAULT_RECENT_FEATURE_NAME
        ),
        night_light_recent_source_glob=night_light_params_cfg.get("recent_source_glob", "*.tif"),
        night_light_recent_cache_path=night_light_params_cfg.get("recent_cache_path"),
        night_light_cf_cvg_source_glob=night_light_params_cfg.get("cf_cvg_source_glob"),
        night_light_cf_cvg_feature_name=night_light_params_cfg.get(
            "cf_cvg_feature_name", DEFAULT_RECENT_COVERAGE_FEATURE_NAME
        ),
        night_light_cf_cvg_cache_path=night_light_params_cfg.get("cf_cvg_cache_path"),
        night_light_cf_filtered_feature_name=night_light_params_cfg.get(
            "cf_filtered_feature_name", DEFAULT_RECENT_FILTERED_FEATURE_NAME
        ),
        night_light_min_cf_cvg=night_light_params_cfg.get("min_cf_cvg"),
        night_light_cf_filter_north_lat_min=night_light_params_cfg.get(
            "cf_filter_north_lat_min", 58.0
        ),
        night_light_black_marble_source_dir=night_light_params_cfg.get("black_marble_source_dir"),
        night_light_black_marble_cache_path=night_light_params_cfg.get("black_marble_cache_path"),
        night_light_black_marble_missing_tile_fallback=night_light_params_cfg.get(
            "black_marble_missing_tile_fallback"
        ),
        night_light_black_marble_fallback_feature_name=night_light_params_cfg.get(
            "black_marble_fallback_feature_name", "night_light_radiance_2024"
        ),
        # Fire Index
        fire_index_npz_path=land_params_cfg.get("fire_index_npz_path", "data/land_features/fire_index_features.npz"),
        # Land
        land_data_files=land_params_cfg["land_data_files"],
        landsea_mask_path=land_params_cfg.get("landsea_mask_path"),
        landsea_distance_path=land_params_cfg.get("landsea_distance_path"),
        landsea_mask_threshold=land_params_cfg.get("landsea_mask_threshold", 70),
        # Ecoregion
        wwf_shp_path=land_params_cfg["wwf_shp_path"],
        # Other controls
        anchor_cols=anchor_columns,
        test_mode=test_mode,
        skip_climate=skip_climate_cfg,
        strict_climate_bounds=strict_climate_bounds_cfg,
        use_cached_files=use_cached_files,
        cache_dir=cache_dir
    )


def make_features_and_save(config_or_path: str | dict, output_file: str, test_mode: bool, use_cached_files: bool = False, cache_dir: str = 'climate_features_cache', use_cached_target: bool = False):
    """
    Loads configuration, generates target data, generates features, and saves the result.
    """
    config = _load_config(config_or_path)

    # Target data parameters
    modis_data_path = config['modis_data_path']
    modis_countries = config['modis_countries']
    target_samples_per_area_per_year = config['target_samples_per_area_per_year']
    coordinate_bounds_cfg = tuple(config['coordinate_bounds']) # [min_lat, min_lon, max_lat, max_lon]
    
    target_start_date_cfg = config['target_start_date']
    target_end_date_cfg = config['target_end_date']
    test_start_date_cfg = config.get('test_start_date', "2020-01-01")
    test_end_date_cfg = config.get('test_end_date', "2020-12-31")

    # Determine date range for target generation
    current_start_date = test_start_date_cfg if test_mode else target_start_date_cfg
    current_end_date = test_end_date_cfg if test_mode else target_end_date_cfg

    target_cache_path = config.get('target_cache_path', 'data/saved_features/target_data.parquet')

    df_target_processed = generate_target_data(
        modis_data_path=modis_data_path,
        modis_countries=modis_countries,
        target_samples_per_area_per_year=target_samples_per_area_per_year,
        coordinate_bounds=coordinate_bounds_cfg,
        start_date=current_start_date,
        end_date=current_end_date,
        use_cached=use_cached_target,
        cache_path=target_cache_path,
        feature_config=config,
    )

    final_features_df = make_features_from_target_df(
        config,
        df_target_processed,
        test_mode=test_mode,
        use_cached_files=use_cached_files,
        cache_dir=cache_dir,
    )

    print(f"Final features DataFrame shape before saving: {final_features_df.shape}")
    if not final_features_df.empty:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_features_df.to_parquet(output_file)
        print(f"Final features saved to '{output_file}'.")
    else:
        print(f"Warning: Final features DataFrame is empty. Not saving to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features for fire prediction model.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/features_config_30d.yaml", # Assuming your config file is YAML
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/saved_features_boost/train_features_boost_30d.parquet",
        help="Path to save the final features Parquet file."
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode (uses test_start_date, test_end_date from config)."
    )
    parser.add_argument(
        "--use_cached_target",
        action="store_true",
        help="Load target data from cache if available."
    )
    parser.add_argument(
        "--use_cached_climate_files",
        action="store_true",
        help="Use cached climate feature files if available."
    )
    
    args = parser.parse_args()
    
    # Example: If your src modules are in a directory 'src' at the same level as this script's dir
    # current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(current_script_dir) 
    # sys.path.append(project_root) # Add project root to allow `from src. ...`
    # This sys.path.append might be needed if you run this script directly and `src` is not in PYTHONPATH
    # If `src` is a package installed in your environment, this is not needed.

    # For direct execution, ensure current working directory allows finding `src`
    # This is often handled by running python -m your_module.make_features or setting PYTHONPATH
    if os.getcwd() not in sys.path: # A common way to ensure local modules are found
        sys.path.append(os.getcwd())


    print(f"Running feature generation with config: {args.config}")
    print(f"Output will be saved to: {args.output}")
    print(f"Test mode: {args.test_mode}")
    print(f"Use cached target: {args.use_cached_target}")
    print(f"Use cached climate files: {args.use_cached_climate_files}")

    make_features_and_save(
        config_or_path=args.config,
        output_file=args.output,
        test_mode=args.test_mode,
        use_cached_target=args.use_cached_target,
        use_cached_files=args.use_cached_climate_files
    )
