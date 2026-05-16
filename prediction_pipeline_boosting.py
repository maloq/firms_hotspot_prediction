import argparse
import os
import sys
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import geopandas as gpd
from catboost import CatBoostClassifier
from shapely.geometry import Point
from scipy.spatial import cKDTree

from src.feature_generation.make_features import generate_all_features
from src.feature_generation.precompute_climate_features import (
    load_precomputed_features_for_day,
    precompute_features_from_config,
)
from src.target_generation.create_grid_target import country_mapping, load_modis_data_for_period
from src.utils.prediction_adjustments import (
    DEFAULT_DEPLOY_PRIOR,
    DEFAULT_TRAIN_PRIOR,
    adjust_probabilities_for_prior,
)
from src.utils.prediction_visuals import (
    plot_feature_maps_group,
    plot_modis_fires,
    plot_prediction_map,
    save_prediction_as_netcdf,
    to_scalar,
)

# debug options
save_feature_maps = True
print_features_mean = False

# MODIS plot options
MODIS_PLOT_DAYS_WINDOW = 3  # ±N days before and after prediction date

# Prior correction defaults (model trained with downsampled negatives)
TRAIN_PRIOR = DEFAULT_TRAIN_PRIOR
DEPLOY_PRIOR = DEFAULT_DEPLOY_PRIOR  # Switch to 1e-4 when the true positive rate is 0.01%
ENABLE_PRIOR_CORRECTION = True

# Mutable runtime configuration (can be overridden via config in __main__)
train_prior = TRAIN_PRIOR
deploy_prior = DEPLOY_PRIOR
enable_prior_correction = ENABLE_PRIOR_CORRECTION

# --- HELPER FUNCTIONS ---



def _save_feature_maps(
    features_df: pd.DataFrame,
    model: CatBoostClassifier,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    plot_dir: str,
    date_str: str,
    borders_data: gpd.GeoDataFrame | None,
) -> None:
    """
    Saves plots of feature maps for a given forecast day.
    """
    feature_plot_dir = os.path.join(plot_dir, "feature_maps")
    os.makedirs(feature_plot_dir, exist_ok=True)

    lat_map = {lat: i for i, lat in enumerate(lat_coords)}
    lon_map = {lon: j for j, lon in enumerate(lon_coords)}

    try:
        raw_importances = model.get_feature_importance(type="PredictionValuesChange")
    except Exception as exc:  # pragma: no cover - CatBoost API fallback
        print(f"Warning: Could not retrieve feature importance ({exc}). Using model order.")
        raw_importances = np.full(len(model.feature_names_), np.nan)

    feature_with_importance: list[tuple[str, float]] = []
    for idx, feature_name in enumerate(model.feature_names_):
        importance = float(raw_importances[idx]) if idx < len(raw_importances) else np.nan
        if feature_name not in features_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(features_df[feature_name]):
            print(f"Skipping non-numeric feature '{feature_name}' for map plot.")
            continue
        feature_with_importance.append((feature_name, importance))

    if not feature_with_importance:
        print("Warning: No numeric feature columns available for feature importance plots.")
        return

    feature_with_importance.sort(key=lambda item: (-(item[1]) if np.isfinite(item[1]) else float('inf'), item[0]))

    feature_entries: list[tuple[str, np.ndarray, Optional[float]]] = []
    for feature_name, importance in feature_with_importance:
        feature_grid = np.full((len(lat_coords), len(lon_coords)), np.nan, dtype=float)
        for _, row in features_df[['lat_rounded', 'lon_rounded', feature_name]].iterrows():
            lat = float(to_scalar(row['lat_rounded']))
            lon = float(to_scalar(row['lon_rounded']))
            val = to_scalar(row[feature_name])

            if pd.isna(val) or not isinstance(val, (int, float, np.number)):
                continue

            if lat in lat_map and lon in lon_map:
                i, j = lat_map[lat], lon_map[lon]
                feature_grid[i, j] = float(val)

        if np.isnan(feature_grid).all():
            continue

        feature_entries.append((feature_name, feature_grid, importance if np.isfinite(importance) else None))

    if not feature_entries:
        print("Warning: Feature importance grids are empty; skipping feature map plots.")
        return

    for group_idx in range(0, len(feature_entries), 3):
        group = feature_entries[group_idx:group_idx + 3]
        group_path = os.path.join(
            feature_plot_dir,
            f"feature_maps_group_{group_idx // 3 + 1}_{date_str}.png",
        )
        group_title = f"{date_str} Feature Maps (Group {group_idx // 3 + 1})"
        plot_feature_maps_group(
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            feature_maps=group,
            save_path=group_path,
            title=group_title,
            borders_gdf=borders_data,
        )
        print(f"Saved feature map group: {group_path}")


def _prepare_daily_output_dirs(base_dir: str, date_str: str) -> dict[str, str]:
    """Ensure per-day output directories exist and return their paths."""

    day_dir = os.path.join(base_dir, date_str)
    paths = {
        "root": day_dir,
        "csv": os.path.join(day_dir, "csv"),
        "netcdf": os.path.join(day_dir, "netcdf"),
        "plots": os.path.join(day_dir, "plots"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _generate_features_for_forecast_day(df_target_day: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Internal helper to call the main feature generation function with parameters from config.
    """
    
    # Climate data loading params
    try:
        climate_params = config['climate_data_params_prediction'] 
    except KeyError:
        raise KeyError("climate_data_params_prediction not found in config, this is required for prediction")
    
    # Climate feature generation params
    gen_climate_params = config['generate_climate_params']
    # Get trend_window from config, with fallback to default if not specified
    trend_window_default = gen_climate_params.get('trend_window', [21, 90])
    
    # Elevation params
    elevation_params = config['elevation_data_params']
    
    # Road params
    road_params = config['road_data_params']
    night_light_params = config.get('night_light_data_params', {})

    # Land data params
    land_data_params = config['land_data_params']

    anchor_cols = ['datetime', 'lat_rounded', 'lon_rounded', 'month', 'day', 'year', 'count']

    for col in anchor_cols:
        if col not in df_target_day.columns:
            if col == 'year' and 'datetime' in df_target_day:
                 df_target_day['year'] = pd.to_datetime(df_target_day['datetime']).dt.year
            elif col == 'count': # 'count' is a placeholder for prediction
                 df_target_day['count'] = 0
            else:
                # This should not happen if df_target_day is prepared correctly
                raise ValueError(f"Required anchor column '{col}' is missing in df_target_day for feature generation.")

    # >>>>>>>>> Precompute climate features <<<<<<<<<<
    # date of the forecast
    date_str = df_target_day['datetime'].iloc[0].strftime('%Y-%m-%d')
    print(f"Precomputing climate features for {date_str}")
    precompute_features_from_config(config=config, date_str=date_str)

        # >>>>>>>>> Load precomputed features <<<<<<<<<<
    print(f"Loading precomputed features for {date_str}")
    climate_features_df = load_precomputed_features_for_day(
        df=df_target_day,
        config=config,
        prediction_date=pd.to_datetime(date_str),
        lat_col='lat_rounded',
        lon_col='lon_rounded',
    )

    # Print mean for every climate_features_df column
    if print_features_mean:
        print(climate_features_df.mean().to_dict())

    # >>>>>>>>> Generate features <<<<<<<<<<
    features_df = generate_all_features(
        df_target=df_target_day,
        # Climate data loading
        climate_data_dir=climate_params["climate_data_dir"],
        climate_variables=climate_params["climate_variables"],
        climate_n_days=climate_params["n_days"],
        # Climate feature generation
        generate_climate_max_length=gen_climate_params["max_length"],
        generate_climate_lags=gen_climate_params["lags"],
        generate_climate_windows=gen_climate_params["windows"],
        generate_climate_spans=gen_climate_params["spans"],
        generate_climate_trend_window=trend_window_default,
        generate_climate_features_to_include=gen_climate_params["features_to_include"],
        # Elevation
        elevation_file_path=elevation_params["elevation_file"],
        elevation_window_sizes=elevation_params.get("window_size", [0.25]),
        # Road
        road_feature_map_path=road_params["feature_map_path"],
        use_road_features=road_params.get("use_road_features", False),
        # Night lights
        night_light_feature_map_path=night_light_params.get("feature_map_path"),
        use_night_light_features=night_light_params.get("use_night_light_features", False),
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
        # Fire Index
        fire_index_npz_path=land_data_params.get("fire_index_npz_path", "data/land_features/fire_index_features.npz"),
        # Land
        land_data_files=land_data_params["land_data_files"],
        landsea_mask_path=land_data_params.get("landsea_mask_path"),
        landsea_distance_path=land_data_params.get("landsea_distance_path"),
        # Ecoregion
        wwf_shp_path=land_data_params["wwf_shp_path"],
        # Other controls
        anchor_cols=anchor_cols,
        test_mode=True, 
        skip_climate=True, # Precomputed features are used instead
        use_cached_files=False,
    )



    # merge features_df and climate_features_df
    # Using nearest-neighbor matching because generate_all_features can filter points,
    # leading to coordinate mismatches that break a direct merge.
    
    # Ensure dataframes have clean indices before we start
    features_df = features_df.reset_index(drop=True)
    climate_features_df = climate_features_df.reset_index(drop=True)

    # Coordinates from the feature dataframe (which may have been filtered)
    feature_coords = features_df[['lat_rounded', 'lon_rounded']].values

    # Coordinates from the climate dataframe (which is complete)
    climate_coords = climate_features_df[['lat_rounded', 'lon_rounded']].values

    if climate_coords.shape[0] > 0:
        # Build a KD-Tree for efficient nearest-neighbor search on the climate data
        tree = cKDTree(climate_coords)
        
        # For each point in features_df, find the index of the closest point in climate_features_df
        distances, indices = tree.query(feature_coords, k=1) # k=1 for single nearest neighbor
        
        print(f"Max distance between feature points and matched climate data: {distances.max():.6f} degrees")
        
        # Select the corresponding climate features
        matched_climate_features = climate_features_df.iloc[indices]
        
        # Reset index of matched features to align with features_df for concatenation
        matched_climate_features = matched_climate_features.reset_index(drop=True)
        
        # Drop coordinate columns from climate features to avoid duplication
        climate_cols_to_add = matched_climate_features.drop(columns=['lat_rounded', 'lon_rounded'], errors='ignore')

        # Concatenate the original features with the matched climate features
        features_df = pd.concat([features_df, climate_cols_to_add], axis=1)
    else:
        print("Warning: climate_features_df is empty. Cannot merge climate features.")

    if print_features_mean:
        print(features_df.select_dtypes(include=['number']).mean().to_dict())

    # Remove any duplicate columns that could cause issues (keep first occurrence)
    if features_df.columns.duplicated().any():
        dup_cols = features_df.columns[features_df.columns.duplicated()].unique()
        print(f"Warning: Duplicate feature columns detected and removed: {list(dup_cols)}")
        features_df = features_df.loc[:, ~features_df.columns.duplicated()].copy()

    return features_df


# --- MAIN FORECAST FUNCTION ---

def make_n_day_forecast(n_days: int, 
                        min_lat: float, max_lat: float, 
                        min_lon: float, max_lon: float, 
                        config_path: str, 
                        model_path: str, 
                        output_base_dir: str, 
                        country_filter_names: list[str] | None = None,
                        start_date: str | None = None):
    """
    Generates fire predictions for the next n days.
    
    Args:
        start_date: Optional start date in 'dd-mm-yy' format. If None, uses today.
    """
    if not 1 <= n_days <= 30:
        raise ValueError("n_days must be between 1 and 30.")

    print(f"--- Starting {n_days}-Day Forecast ---")

    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  

    borders_data = None
    try:
        world_shapes_path = config.get('country_shapes_path', 'data/countries')
        if not os.path.exists(world_shapes_path):
            print(f"Warning: Country shapes file/directory not found at {world_shapes_path}. Plots will not have borders.")
        else:
            borders_data = gpd.read_file(world_shapes_path)
            print("Loaded border data for plotting.")
    except Exception as e:
        print(f"Warning: Could not load border data from {world_shapes_path} due to error: {e}. Plots will not have borders.")

    # 2. Define Forecast Period and Grid
    if start_date:
        # Parse the date in dd-mm-yy format
        try:
            today = pd.to_datetime(start_date, format='%d-%m-%y').normalize().date()
            print(f"Using specified start date: {today}")
        except ValueError as e:
            raise ValueError(f"Invalid date format '{start_date}'. Expected format: dd-mm-yy (e.g., 15-03-24)") from e
    else:
        today = pd.Timestamp.now().normalize().date() # Use normalize for midnight
        print(f"Using today as start date: {today}")
    
    forecast_dates = [today + timedelta(days=i) for i in range(n_days)]
    print(f"Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
    target_config_path = 'configs/target_config.yaml'
    with open(target_config_path, 'r') as config_file:
        target_config = yaml.safe_load(config_file)
    grid_resolution = target_config['spatial_coarseness']
    lat_coords = np.arange(min_lat, max_lat + grid_resolution/2, grid_resolution) # Ensure max_lat is included
    lon_coords = np.arange(min_lon, max_lon + grid_resolution/2, grid_resolution) # Ensure max_lon is included

    # 3. Load Country Data for Filtering (if specified)
    apply_country_filter = False
    filter_country_geoms = None
    if country_filter_names:
        if borders_data is not None:
            try:
                # Map provided names to names in shapefile (e.g., using country_mapping)
                mapped_country_names = [country_mapping.get(c, c) for c in country_filter_names] # Use original name if no mapping
                
                filter_country_geoms = borders_data[borders_data['SOVEREIGNT'].isin(mapped_country_names)] # Adjust column name if different
                
                if filter_country_geoms.empty:
                    print(f"Warning: No matching countries found for {country_filter_names} (mapped: {mapped_country_names}). Proceeding without country filtering.")
                else:
                    print(f"Found {len(filter_country_geoms)} geometries for filtering: {mapped_country_names}")
                    apply_country_filter = True
            except Exception as e:
                print(f"Error processing country data for filtering: {e}. Proceeding without country filtering.")
                apply_country_filter = False # Ensure it's false on error
        else:
            print("Warning: Country filtering requested, but no border data was loaded. Skipping filtering.")

    # 4. Load Model
    model = CatBoostClassifier()
    model.load_model(model_path)

    # 5. Loop Through Each Forecast Day
    for forecast_date_obj in forecast_dates:
        current_date_dt = pd.to_datetime(forecast_date_obj) # Ensure Timestamp
        date_str = current_date_dt.strftime('%Y-%m-%d')
        print(f"\n--- Processing forecast for: {date_str} ---")

        # 5a. Create Forecast Skeleton DataFrame
        lat_mesh, lon_mesh = np.meshgrid(lat_coords, lon_coords, indexing='ij')
        
        df_forecast_target = pd.DataFrame({
            'lat_rounded': lat_mesh.ravel(),
            'lon_rounded': lon_mesh.ravel(),
            'datetime': current_date_dt
        })
        
        if apply_country_filter and filter_country_geoms is not None and not filter_country_geoms.empty:
            geometry = [Point(lon, lat) for lon, lat in zip(df_forecast_target['lon_rounded'], df_forecast_target['lat_rounded'])]
            gdf_points = gpd.GeoDataFrame(df_forecast_target, geometry=geometry, crs=filter_country_geoms.crs) # Use same CRS
            
            # Perform spatial join
            points_in_countries = gpd.sjoin(gdf_points, filter_country_geoms, how="inner", predicate="within")
            df_forecast_target = points_in_countries[['lat_rounded', 'lon_rounded', 'datetime']].drop_duplicates()
            print(f"After country filtering: {df_forecast_target.shape[0]} grid points remaining.")
            if df_forecast_target.empty:
                print(f"Warning: No grid points remaining after country filter for {date_str}. Skipping this date.")
                continue

        df_forecast_target['month'] = current_date_dt.month
        df_forecast_target['day'] = current_date_dt.day
        df_forecast_target['year'] = current_date_dt.year # Added year
        df_forecast_target['acq_date'] = current_date_dt.date() # For compatibility if some old code expects it
        df_forecast_target['count'] = 0 # Placeholder for target variable
        
        if df_forecast_target.empty:
             print(f"Skipping {date_str} due to empty forecast target (possibly after filtering).")
             continue

        # 5b. Generate Features
        features_df = _generate_features_for_forecast_day(df_forecast_target.copy(), config)
        
        if features_df.shape[0] != df_forecast_target.shape[0]:
             print(f"Warning: Feature generation output shape ({features_df.shape[0]}) "
                   f"mismatches skeleton ({df_forecast_target.shape[0]}). "
                   "Leaving missing grid cells as NaN instead of nearest-filling them.")

        # 5c. Prepare Data and Predict
        # Ensure all model features are present, fill NaNs
        X_pred = pd.DataFrame(columns=model.feature_names_) # Create empty DF with model's feature order
        for col in model.feature_names_:
            if col in features_df.columns:
                X_pred[col] = features_df[col]
            else:
                print(f"Warning: Model feature '{col}' not found in generated features. Will be NaN.")
                X_pred[col] = np.nan # Explicitly add as NaN if missing

        # Fill NaNs
        # Get categorical feature indices from the model
        cat_feature_indices = model.get_cat_feature_indices()
        cat_feature_names = [model.feature_names_[i] for i in cat_feature_indices]
        
        # This handles the special case for population. Consider integrating into a general config.
        if 'population' in X_pred.columns:
            X_pred['population'] = X_pred['population'].fillna(0)

        # Process all columns, ensuring correct types and filling NaNs.
        for col in X_pred.columns:
            if col in cat_feature_names:
                # This is a categorical feature
                if X_pred[col].dtype == 'object' or pd.api.types.is_string_dtype(X_pred[col]):
                    if X_pred[col].isnull().any():
                        X_pred[col] = X_pred[col].fillna("Unknown")
                        print(f"Filled NaNs in categorical feature '{col}' (string type) with 'Unknown'")
                else: # Should be integer-based categorical
                    if X_pred[col].isnull().any():
                        X_pred[col] = X_pred[col].fillna(99)
                        print(f"Filled NaNs in categorical feature '{col}' (numeric type) with 99")
                    
                    # Ensure the final type is integer for CatBoost. This is the key fix.
                    if not pd.api.types.is_integer_dtype(X_pred[col]):
                        X_pred[col] = X_pred[col].astype(int)
                        print(f"Casted float categorical feature '{col}' to int.")
            else:
                # This is a numerical feature
                if X_pred[col].isnull().any():
                    mean_val = X_pred[col].mean()
                    if pd.isna(mean_val): # If all values were NaN
                        mean_val = 0 # Fallback to 0
                    X_pred[col] = X_pred[col].fillna(mean_val)
                    print(f"Filled NaNs in numerical feature '{col}' with mean/0: {mean_val}")
        
        if X_pred.isnull().any().any():
            print("Critical Warning: Missing values persist in X_pred after fillna. Check feature generation and filling logic.")
            # Fallback: fill all remaining NaNs with 0 to prevent CatBoost error
            X_pred = X_pred.fillna(0)

        if X_pred.empty:
            print(f"Warning: No data to predict for {date_str}. Skipping prediction.")
            y_pred_proba = np.array([])
        else:
            positive_class_scores = model.predict_proba(X_pred)[:, 1]
            if enable_prior_correction:
                y_pred_proba = adjust_probabilities_for_prior(
                    positive_class_scores,
                    train_prior=train_prior,
                    deploy_prior=deploy_prior,
                    assume_logits=False,
                )
            else:
                y_pred_proba = positive_class_scores
        
        if len(y_pred_proba) == len(features_df):
            features_df['prediction'] = y_pred_proba
        elif not X_pred.empty: # y_pred_proba exists but length mismatch
            print(f"Warning: Prediction length mismatch ({len(y_pred_proba)}) vs features_df ({len(features_df)}). "
                  "Predictions may not be correctly assigned to features_df for CSV. Check alignment.")
            if X_pred.index.equals(features_df.index):
                 features_df['prediction'] = pd.Series(y_pred_proba, index=X_pred.index)
            else: 
                 raise ValueError("Prediction length mismatch and cannot align. Check feature generation and alignment logic.")

        else:
            raise ValueError("No predictions made. Check feature generation and alignment logic.")

        # 5d. Reshape Predictions to Grid
        pred_grid = np.full((len(lat_coords), len(lon_coords)), np.nan)  # Initialize with NaN

        lat_step = float(np.nanmedian(np.diff(lat_coords))) if len(lat_coords) > 1 else 0.0
        lon_step = float(np.nanmedian(np.diff(lon_coords))) if len(lon_coords) > 1 else 0.0
        lat_tolerance = abs(lat_step) / 2.0 + 1e-8
        lon_tolerance = abs(lon_step) / 2.0 + 1e-8

        # Use 'lat_rounded', 'lon_rounded' and 'prediction' from features_df
        for _, row in features_df.iterrows():
            if pd.isna(row['prediction']):
                continue

            lat = float(to_scalar(row['lat_rounded']))
            lon = float(to_scalar(row['lon_rounded']))
            if not np.isfinite(lat) or not np.isfinite(lon) or lat_step == 0.0 or lon_step == 0.0:
                continue

            # Snap only existing feature rows back to the native output grid.
            # This tolerates tiny coordinate drift without inventing predictions
            # for cells that feature generation dropped.
            i = int(round((lat - float(lat_coords[0])) / lat_step))
            j = int(round((lon - float(lon_coords[0])) / lon_step))

            if (
                0 <= i < len(lat_coords)
                and 0 <= j < len(lon_coords)
                and abs(lat - float(lat_coords[i])) <= lat_tolerance
                and abs(lon - float(lon_coords[j])) <= lon_tolerance
            ):
                pred_grid[i, j] = to_scalar(row['prediction'])

        # Keep NaNs for unassigned grid cells so plotting/interpolation and color scaling
        # are driven by actual predicted values rather than zeros that flatten the colormap.

        # 5e. Save and Plot Results
        output_dirs = _prepare_daily_output_dirs(output_base_dir, date_str)
        nc_dir = output_dirs["netcdf"]
        plot_dir = output_dirs["plots"]
        csv_dir = output_dirs["csv"]

        country_suffix = ""
        if apply_country_filter and country_filter_names:
            if len(country_filter_names) == 1:
                country_suffix = f"_{country_filter_names[0].replace(' ', '_')}"
            else: # Multiple countries or generic filter
                country_suffix = "_filtered_countries"
        
        common_filename_base = f"forecast_raw_{date_str}"

        nc_path = os.path.join(nc_dir, f"{common_filename_base}.nc")
        save_prediction_as_netcdf(
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            predictions=pred_grid,
            date_timestamp=current_date_dt,
            save_path=nc_path,
            prediction_type="raw",
        )
        print(f"Saved raw prediction NetCDF: {nc_path}")

        plot_path_raw = os.path.join(plot_dir, f"{common_filename_base}.png")
        title = f"Fire Probability Forecast for {date_str}"
        plot_prediction_map(
            lat_coords=lat_coords,
            lon_coords=lon_coords,
            predictions=pred_grid,
            title=title,
            save_path=plot_path_raw,
            borders_gdf=borders_data,
        )
        print(f"Saved prediction plot: {plot_path_raw}")

        # Plot MODIS fire data for ±N days around the prediction date
        try:
            modis_data_dir = config.get('modis_data_path', 'data/modis/')
            modis_countries = config.get('prediction_countries', config.get('modis_countries'))
            
            # Calculate date range
            start_date_modis = (current_date_dt - timedelta(days=MODIS_PLOT_DAYS_WINDOW)).date()
            end_date_modis = (current_date_dt + timedelta(days=MODIS_PLOT_DAYS_WINDOW)).date()
            
            # Load MODIS data for the date range
            modis_fire_data = load_modis_data_for_period(
                data_dir=modis_data_dir,
                start_date=start_date_modis,
                end_date=end_date_modis,
                countries=modis_countries
            )
            
            if not modis_fire_data.empty:
                # Filter to the coordinate bounds
                modis_fire_data = modis_fire_data[
                    (modis_fire_data['latitude'] >= lat_coords.min()) &
                    (modis_fire_data['latitude'] <= lat_coords.max()) &
                    (modis_fire_data['longitude'] >= lon_coords.min()) &
                    (modis_fire_data['longitude'] <= lon_coords.max())
                ]
                
                if not modis_fire_data.empty:
                    modis_plot_path = os.path.join(plot_dir, f"modis_fires_{date_str}.png")
                    modis_title = f"MODIS Fires (±{MODIS_PLOT_DAYS_WINDOW} days around {date_str})"
                    plot_modis_fires(
                        lat_coords=lat_coords,
                        lon_coords=lon_coords,
                        modis_data=modis_fire_data,
                        prediction_date=current_date_dt,
                        n_days=MODIS_PLOT_DAYS_WINDOW,
                        title=modis_title,
                        save_path=modis_plot_path,
                        borders_gdf=borders_data,
                    )
                    print(f"Saved MODIS fire plot: {modis_plot_path}")
                else:
                    print(f"Warning: No MODIS fire data found in the coordinate bounds for {date_str} (±{MODIS_PLOT_DAYS_WINDOW} days)")
            else:
                print(f"Warning: No MODIS fire data available for {date_str} (±{MODIS_PLOT_DAYS_WINDOW} days). Skipping MODIS plot.")
        except Exception as e:
            print(f"Warning: Could not load or plot MODIS fire data for {date_str}: {e}")
    
        if save_feature_maps:
            _save_feature_maps(
                features_df=features_df,
                model=model,
                lat_coords=lat_coords,
                lon_coords=lon_coords,
                plot_dir=plot_dir,
                date_str=date_str,
                borders_data=borders_data,
            )

        save_csv = False
        if save_csv: # TODO: Add this to the config
            csv_path_forecast = os.path.join(csv_dir, f"data_with_predictions_{date_str}{country_suffix}.csv")
            try:
                cols_to_save = ['datetime', 'lat_rounded', 'lon_rounded', 'prediction'] + \
                            [col for col in model.feature_names_ if col in features_df.columns]
                relevant_features_df = features_df[cols_to_save].copy()
                relevant_features_df.to_csv(csv_path_forecast, index=False)
                print(f"Saved daily forecast data to CSV: {csv_path_forecast}")
            except Exception as e:
                print(f"Error saving daily forecast data to CSV {csv_path_forecast}: {e}")

    print(f"\n--- {n_days}-Day Forecast Complete ---")
    print(f"Outputs saved in base directory: {output_base_dir}")




if __name__ == "__main__":
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    parser = argparse.ArgumentParser(description="Run N-day fire forecast.")
    parser.add_argument('--config', type=str, default='configs/features_config_30d.yaml', 
                        help='Path to the YAML configuration file for feature generation and model paths.')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to forecast (e.g., 1-30).')
    parser.add_argument('--output', type=str, default='outputs/forecast_run_30d',
                        help='Base directory for forecast outputs.')
    parser.add_argument('--start-date', type=str, default="09-06-25",
                        help='Start date for forecast in dd-mm-yy format (e.g., 15-03-24). If not provided, uses today.')

    args = parser.parse_args()

    print("\n=== Running N-Day Forecast ===")
    
    if not os.path.exists(args.config):
         raise FileNotFoundError(f"Forecast config file {args.config} not found.")
    
    with open(args.config, 'r') as f:
         forecast_config_data = yaml.safe_load(f)

    model_file_path = forecast_config_data.get('model_path')
    if not model_file_path or not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found. Path specified in config ('model_path'): {model_file_path}")

    train_prior = float(forecast_config_data.get('train_prior', TRAIN_PRIOR))
    deploy_prior = float(forecast_config_data.get('deploy_prior', DEPLOY_PRIOR))
    enable_prior_correction = forecast_config_data.get('enable_prior_correction', ENABLE_PRIOR_CORRECTION)
    if isinstance(enable_prior_correction, str):
        enable_prior_correction = enable_prior_correction.strip().lower() in {"1", "true", "yes", "on"}
    else:
        enable_prior_correction = bool(enable_prior_correction)
    if enable_prior_correction:
        print(f"Prior correction enabled: train_prior={train_prior}, deploy_prior={deploy_prior}")
    else:
        print("Prior correction disabled via configuration.")

    # Coordinate bounds for the forecast grid (can be overridden by command-line in a more complex setup)
    # The config 'coordinate_bounds' = [min_lat, min_lon, max_lat, max_lon]
    coord_bounds_cfg = forecast_config_data.get('coordinate_bounds')
    if not coord_bounds_cfg or len(coord_bounds_cfg) != 4:
        raise ValueError("Config 'coordinate_bounds' [min_lat, min_lon, max_lat, max_lon] is missing or invalid.")
    
    min_lat_fc, min_lon_fc, max_lat_fc, max_lon_fc = coord_bounds_cfg

    print(f"Forecast region from config: Lat [{min_lat_fc}, {max_lat_fc}], Lon [{min_lon_fc}, {max_lon_fc}]")

    countries_to_filter = forecast_config_data.get('prediction_countries', 
                                                forecast_config_data.get('modis_countries'))

    make_n_day_forecast(
        n_days=args.days,
        min_lat=min_lat_fc,
        max_lat=max_lat_fc,
        min_lon=min_lon_fc,
        max_lon=max_lon_fc,
        config_path=args.config, 
        model_path=model_file_path,   
        output_base_dir=args.output,
        country_filter_names=countries_to_filter,
        start_date=args.start_date
    )

    print("\n--- Script Finished ---")
