import pandas as pd
import geopandas as gpd
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.vectorized import contains
from shapely.errors import GEOSException
import yaml
from tqdm import tqdm
import datetime
import copy
from datetime import timedelta
import itertools
from pathlib import Path

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

    if USE_HIGH_LATITUDE_FILTER:
        # Apply latitude-dependent thresholds: lower thresholds for |lat| > 58
        high_lat_mask = df['latitude'].abs() > 58

        # Low-latitude filter (|lat| ≤ 60) – use default thresholds
        low_lat_filter = (
            (~high_lat_mask)
            & (df['brightness'] > BRIGHTNESS_THRESHOLD)
            & (df['confidence'] > CONFIDENCE_THRESHOLD)
        )

        # High-latitude filter (|lat| > 58) – use relaxed thresholds
        high_lat_filter = (
            high_lat_mask
            & (df['brightness'] > BRIGHTNESS_THRESHOLD_HIGH_LAT)
            & (df['confidence'] > CONFIDENCE_THRESHOLD_HIGH_LAT)
        )

        # --- THRESHOLDS for far-north & western longitudes (lat > 58 and lon < 45) ---
        special_region_mask = (df['latitude'] > 58) & (df['longitude'] < 45)

        SPECIAL_BRIGHTNESS_THRESHOLD = max(BRIGHTNESS_THRESHOLD_HIGH_LAT - 30, 0)
        if CONFIDENCE_THRESHOLD_HIGH_LAT > 1:         
            SPECIAL_CONFIDENCE_THRESHOLD = max(CONFIDENCE_THRESHOLD_HIGH_LAT - 15, 0)
        else:                                          
            SPECIAL_CONFIDENCE_THRESHOLD = max(CONFIDENCE_THRESHOLD_HIGH_LAT - 0.15, 0.0)

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
            & (df['confidence'] > CONFIDENCE_THRESHOLD)
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



def prepare_target_data(data: pd.DataFrame, countries: list, samples_per_area_per_year: float):
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
    data['count'] = np.where(special_region_mask_NW, 3, 1)
    data['count'] = np.where(special_region_mask_W,  2, 1)
    print(f"\nTriple base count for {special_region_mask_NW.sum()} points in the special region (lat>60, lon<45).")
    print(f"\nDoubled base count for {special_region_mask_W.sum()} points in the special region (lat>55, lon<60).")

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

    # --- Step 3: Add Negative Samples ---
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
        try:
            with ProcessPoolExecutor() as executor:
                for country in valid_countries_for_neg_samples:
                    mapped_country_name = country_mapping.get(country, country)
                    world_subset = world[world['SOVEREIGNT'] == mapped_country_name].iloc[0:1].copy()

                    futures.append(executor.submit(
                        add_negative_samples,
                        date_start=date_start, # Use original start/end for neg sampling range
                        date_end=date_end,
                        size=country_sizes[country],
                        world_data_subset=world_subset,
                        country_name=country
                    ))

                for future in tqdm(as_completed(futures), total=len(futures), desc="Negative Sampling"):
                    result_df = future.result() # Raises exceptions from worker
                    if result_df is not None and not result_df.empty:
                        random_data_list.append(result_df)
        except Exception as e:
            print(f"\nError during parallel processing for negative samples: {e}")
            raise RuntimeError("Failed to generate negative samples.") from e

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

    print("\nPreparation pipeline complete.")
    return data_filtered


def add_negative_samples(date_start,
                         date_end,
                         size,
                         world_data_subset,
                         country_name):
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

    while generated_count < size and attempts < max_attempts:
        needed = size - generated_count
        buffer_needed = int(needed * 1.2) + 10
        random_lat = np.random.uniform(lat_range[0], lat_range[1], size=buffer_needed)
        random_lon = np.random.uniform(lon_range[0], lon_range[1], size=buffer_needed)

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

    random_dates = pd.to_datetime(np.random.choice(pd.date_range(start=date_start, end=date_end), size=final_size))

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
