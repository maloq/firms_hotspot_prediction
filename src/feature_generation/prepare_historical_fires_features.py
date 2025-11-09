from sklearn.neighbors import BallTree
import pandas as pd
import numpy as np
from typing import Optional, Union, List

# IMPORTANT:
# NOT USED(because of overfitting)

def get_fire_features(target_df: pd.DataFrame, 
                      modis_df: pd.DataFrame, 
                      density_radii_km: Union[float, List[float]] = 50.0,
                      query_lat_col: str = "lat_rounded",
                      query_lon_col: str = "lon_rounded",
                      fire_time_limit: Optional[Union[str, pd.Timestamp]] = "2021-01-01") -> pd.DataFrame:
    """
    For each coordinate in target_df (using query_lat_col and query_lon_col), compute fire features based on MODIS fire events.
    
    Features computed:
      1. The distance (in km) to the nearest fire.
      2. For each density radius provided:
         - The count of fires within that radius.
         - The density of fires (fires per km²) within that radius.
         
    If fire_time_limit is provided, only fire events with a timestamp at or before the given limit will be considered.
    
    Args:
        target_df (pd.DataFrame): DataFrame containing query coordinates.
        modis_df (pd.DataFrame): DataFrame with MODIS fire event data.
        density_radii_km (Union[float, List[float]]): Radius or list of radii (in km) for density calculations.
        query_lat_col (str): Column name for latitude in target_df (default "lat_rounded").
        query_lon_col (str): Column name for longitude in target_df (default "lon_rounded").
        fire_time_limit (Optional[Union[str, pd.Timestamp]]): Maximum time threshold; only consider fire events with a 
                                                              timestamp at or before this limit. If None, all events are used.
    
    Returns:
        pd.DataFrame: DataFrame with computed fire features.
    """
    # If a fire time limit is provided, filter modis_df based on a recognized time column.
    if fire_time_limit is not None:
        time_col = None
        for col in ['acq_date', 'acquisition_date', 'datetime']:
            if col in modis_df.columns:
                time_col = col
                break
        if time_col is not None:
            modis_df = modis_df.copy()  # avoid modifying the original dataframe
            modis_df[time_col] = pd.to_datetime(modis_df[time_col])
            fire_time_limit_dt = pd.to_datetime(fire_time_limit)
            modis_df = modis_df[modis_df[time_col] <= fire_time_limit_dt]
        else:
            import warnings
            warnings.warn(
                "fire_time_limit provided but no recognized time column "
                "(acq_date, acquisition_date, or datetime) found in modis_df. "
                "No time filtering applied."
            )

    query_lats = target_df[query_lat_col]
    query_lons = target_df[query_lon_col]
    
    if 'latitude' in modis_df.columns and 'longitude' in modis_df.columns:
        fire_coords = modis_df[['latitude', 'longitude']].to_numpy()
    elif 'lat_rounded' in modis_df.columns and 'lon_rounded' in modis_df.columns:
        fire_coords = modis_df[['lat_rounded', 'lon_rounded']].to_numpy()
    else:
        raise ValueError("modis_df must contain either 'latitude' and 'longitude' or 'lat_rounded' and 'lon_rounded' columns.")
    
    fire_coords_rad = np.radians(fire_coords)
    query_points = np.column_stack((query_lats, query_lons))
    query_points_rad = np.radians(query_points)
    
    tree = BallTree(fire_coords_rad, metric='haversine')
    earth_radius_km = 6371.0
    
    distances, _ = tree.query(query_points_rad, k=1)
    nearest_distance_km = distances.flatten() * earth_radius_km
    
    features = {
        'latitude': query_lats,
        'longitude': query_lons,
        'distance_to_nearest_fire_km': nearest_distance_km,
    }
    
    if not isinstance(density_radii_km, list):
        density_radii_km = [density_radii_km]
    
    for r in density_radii_km:
        radius_radians = r / earth_radius_km
        indices = tree.query_radius(query_points_rad, r=radius_radians)
        fire_counts = np.array([len(ind) for ind in indices])
        area = np.pi * r**2
        density = fire_counts / area
        features[f'fire_count_within_radius_{r}'] = fire_counts
        features[f'fire_density_per_km2_{r}'] = density

    feature_df = pd.DataFrame(features)
    return feature_df