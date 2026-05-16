import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import datetime
import yaml

try:
    from .prepare_target_new import normalize_confidence_threshold_for_series
except ImportError:  # pragma: no cover - supports direct script execution
    from src.target_generation.prepare_target_new import normalize_confidence_threshold_for_series

config_path = 'configs/target_config.yaml'
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)
SPATIAL_COARSENESS = config['spatial_coarseness']
BRIGHTNESS_THRESHOLD = config['brightness_threshold']
CONFIDENCE_THRESHOLD = config['confidence_threshold']
BRIGHTNESS_THRESHOLD_HIGH_LAT = config.get('brightness_threshold_high_lat', 360)
CONFIDENCE_THRESHOLD_HIGH_LAT = config.get('confidence_threshold_high_lat', 80)
rounding_precision = int(-np.log10(SPATIAL_COARSENESS))


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

def load_modis_data_for_period(data_dir, start_date, end_date, countries=None):
    """
    Load MODIS data for a specific date range (inclusive of start and end dates).
    
    Args:
        data_dir (str): Directory containing MODIS data.
        start_date (str or datetime): Start date of the period (YYYY-MM-DD).
        end_date (str or datetime): End date of the period (YYYY-MM-DD).
        countries (list, optional): List of countries to include. If None, all countries are included.
    
    Returns:
        pd.DataFrame: Filtered MODIS data for the specified date range.
    """
    # Convert dates to datetime if they're strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    
    # Get the years covered by the date range
    years = list(range(start_date.year, end_date.year + 1))
    
    # Initialize list to hold dataframes
    dataframes = []
    
    # If no countries specified, determine from available files
    countries_by_year = {}
    if countries is None:
        for year in years:
            year_dir = os.path.join(data_dir, str(year))
            if os.path.exists(year_dir):
                files = [f for f in os.listdir(year_dir) if f.startswith(f'modis_{year}_') and f.endswith('.csv')]
                countries_by_year[year] = [f.split('_')[2].replace('.csv', '') for f in files]
    
    # Process each year
    for year in years:
        # Determine which countries to use for this year
        year_countries = countries if countries is not None else countries_by_year.get(year, [])
        
        # Load data for each country
        for country in year_countries:
            file_path = os.path.join(data_dir, str(year), f'modis_{year}_{country}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['country'] = country
                
                # Convert date column to datetime
                df['acq_date'] = pd.to_datetime(df['acq_date']).dt.date
                
                # Filter by date range
                df = df[(df['acq_date'] >= start_date) & (df['acq_date'] <= end_date)]
                
                # Apply latitude-dependent thresholds
                high_lat_mask = df['latitude'].abs() > 60
                confidence_threshold, confidence_scale = normalize_confidence_threshold_for_series(
                    df['confidence'],
                    CONFIDENCE_THRESHOLD,
                )
                confidence_threshold_high_lat, _ = normalize_confidence_threshold_for_series(
                    df['confidence'],
                    CONFIDENCE_THRESHOLD_HIGH_LAT,
                )
                print(
                    "Detected FIRMS confidence scale:"
                    f" {confidence_scale}; using thresholds {confidence_threshold}"
                    f" / {confidence_threshold_high_lat}."
                )

                low_lat_filter = (
                    (~high_lat_mask)
                    & (df['brightness'] > BRIGHTNESS_THRESHOLD)
                    & (df['confidence'] > confidence_threshold)
                )

                high_lat_filter = (
                    high_lat_mask
                    & (df['brightness'] > BRIGHTNESS_THRESHOLD_HIGH_LAT)
                    & (df['confidence'] > confidence_threshold_high_lat)
                )

                df = df[low_lat_filter | high_lat_filter]

                print(f"Number of fire points after brightness/confidence thresholds: {len(df)}")
                
                dataframes.append(df)
    print(f"Number of fires in this period: {len(dataframes)}")
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame()



def create_grid_target_for_period(min_lat, max_lat, min_lon, max_lon, start_date, end_date, data_dir='data/modis/', geojson_path='data/countries', countries=None):
    """
    Create a target grid for a given coordinate limits and date range,
    filtered to only include points within the specified countries.
    
    Args:
        min_lat (float): Minimum latitude.
        max_lat (float): Maximum latitude.
        min_lon (float): Minimum longitude.
        max_lon (float): Maximum longitude.
        start_date (str or datetime): Start date of the period (YYYY-MM-DD).
        end_date (str or datetime): End date of the period (YYYY-MM-DD).
        data_dir (str, optional): Directory containing MODIS data.
        geojson_path (str, optional): Path to the GeoJSON file with country boundaries.
        countries (list, optional): List of countries to include. If None, all countries are considered.
    
    Returns:
        tuple: (grid, lat_coords, lon_coords)
            - grid: 2D numpy array where 1 represents fire and 0 represents no fire
            - lat_coords: array of latitude coordinates corresponding to grid rows
            - lon_coords: array of longitude coordinates corresponding to grid columns
    """
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point
    import time

    # Load fire data for the specified date range
    fire_data = load_modis_data_for_period(data_dir, start_date, end_date, countries)
    
    # Calculate the grid size based on the coordinate limits and spatial coarseness
    # Round the min/max coordinates to the grid
    min_lat_rounded = np.floor(min_lat / SPATIAL_COARSENESS) * SPATIAL_COARSENESS
    max_lat_rounded = np.ceil(max_lat / SPATIAL_COARSENESS) * SPATIAL_COARSENESS
    min_lon_rounded = np.floor(min_lon / SPATIAL_COARSENESS) * SPATIAL_COARSENESS
    max_lon_rounded = np.ceil(max_lon / SPATIAL_COARSENESS) * SPATIAL_COARSENESS
    
    # Create coordinate arrays
    lat_coords = np.arange(min_lat_rounded, max_lat_rounded + SPATIAL_COARSENESS/2, SPATIAL_COARSENESS)
    lon_coords = np.arange(min_lon_rounded, max_lon_rounded + SPATIAL_COARSENESS/2, SPATIAL_COARSENESS)
    
    # Create empty grid
    grid = np.zeros((len(lat_coords), len(lon_coords)), dtype=int)
    
    # If we have fire data, populate the grid
    if not fire_data.empty:
        # Round the fire coordinates to match the grid
        fire_data['lat_rounded'] = np.round(fire_data['latitude'] / SPATIAL_COARSENESS) * SPATIAL_COARSENESS
        fire_data['lon_rounded'] = np.round(fire_data['longitude'] / SPATIAL_COARSENESS) * SPATIAL_COARSENESS
        
        # Group by rounded coordinates and count occurrences
        fire_counts = fire_data.groupby(['lat_rounded', 'lon_rounded']).size().reset_index(name='count')
        
        # For each fire location, set the corresponding grid cell to 1
        for _, row in fire_counts.iterrows():
            lat_idx = np.abs(lat_coords - row['lat_rounded']).argmin()
            lon_idx = np.abs(lon_coords - row['lon_rounded']).argmin()
            grid[lat_idx, lon_idx] = 1
    
    # COUNTRY FILTERING FOR ALL GRID CELLS
    if countries:
        start_time = time.time()
        print(f"Starting country filtering for all grid cells. Grid shape: {grid.shape}")
        
        # Load country boundaries
        world = gpd.read_file(geojson_path)
        
        # Create combined country geometry
        combined_geometry = None
        for country in countries:

            search_country = country_mapping.get(country, country)
            
            country_data = world[world['SOVEREIGNT'] == search_country]
            if country_data.empty:
                print(f"Warning: Country '{search_country}' not found in GeoJSON file.")
                continue
                
            if combined_geometry is None:
                combined_geometry = country_data.geometry.unary_union
            else:
                combined_geometry = combined_geometry.union(country_data.geometry.unary_union)
        
        if combined_geometry is not None:
            # Create a mask for the entire grid
            country_mask = np.zeros_like(grid, dtype=bool)
            
            # Process the grid in batches of rows to optimize memory usage
            batch_size = 100  # Number of rows to process at once
            total_rows = len(lat_coords)
            
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                print(f"Processing rows {batch_start} to {batch_end-1} of {total_rows}")
                
                # For each row in the batch
                for i in range(batch_start, batch_end):
                    # Create points for an entire row at once
                    points = [Point(lon, lat_coords[i]) for lon in lon_coords]
                    
                    # Convert to GeoDataFrame for spatial operations
                    points_gdf = gpd.GeoDataFrame(
                        {'geometry': points, 'lon_idx': range(len(lon_coords))},
                        crs=world.crs
                    )
                    
                    # Use spatial join to efficiently find points within country
                    country_gdf = gpd.GeoDataFrame(geometry=[combined_geometry], crs=world.crs)
                    points_in_country = points_gdf.sjoin(
                        country_gdf,
                        how="inner",
                        predicate="within"
                    )
                    
                    # Mark these points in the mask
                    for lon_idx in points_in_country['lon_idx']:
                        country_mask[i, lon_idx] = True
            
            # Apply the mask to the grid - only keep points within country boundaries
            grid = grid * country_mask.astype(int)
            
            elapsed_time = time.time() - start_time
            print(f"Country filtering completed in {elapsed_time:.2f} seconds")
    
    return grid, lat_coords, lon_coords



def visualize_grid(grid, lat_coords, lon_coords, date_or_period, save_path=None):
    """
    Visualize the fire grid and optionally save to a file.
    
    Args:
        grid (np.ndarray): 2D grid where 1 represents fire and 0 represents no fire
        lat_coords (np.ndarray): Array of latitude coordinates
        lon_coords (np.ndarray): Array of longitude coordinates
        date_or_period: Either a single date or a tuple of (start_date, end_date)
        save_path (str, optional): Path to save the plot. If None, the plot is displayed but not saved.
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Convert date to string for display
    if isinstance(date_or_period, tuple):
        # It's a period
        start_date, end_date = date_or_period
        if isinstance(start_date, (datetime.date, datetime.datetime)):
            start_date_str = start_date.strftime('%Y-%m-%d')
        else:
            start_date_str = str(start_date)
            
        if isinstance(end_date, (datetime.date, datetime.datetime)):
            end_date_str = end_date.strftime('%Y-%m-%d')
        else:
            end_date_str = str(end_date)
            
        date_str = f"{start_date_str} to {end_date_str}"
        title_str = f'Fire Grid for period: {date_str}'
    else:
        # It's a single date
        if isinstance(date_or_period, (datetime.date, datetime.datetime)):
            date_str = date_or_period.strftime('%Y-%m-%d')
        else:
            date_str = str(date_or_period)
        title_str = f'Fire Grid for {date_str}'
    
    # Create a custom colormap (white for no fire, red for fire)
    colors = ['#FFFFFF', '#FF3300']  # White for 0, Red for 1
    cmap = ListedColormap(colors)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the grid as an image
    img = ax.imshow(grid, cmap=cmap, origin='lower', 
                   extent=[lon_coords.min(), lon_coords.max(), 
                           lat_coords.min(), lat_coords.max()])
    
    # Add grid lines
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors[0], label='No Fire'),
        mpatches.Patch(color=colors[1], label='Fire')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title and labels
    ax.set_title(title_str, fontsize=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add some stats
    fire_count = np.sum(grid)
    total_cells = grid.size
    fire_percentage = (fire_count / total_cells) * 100
    stats_text = f'Fire cells: {fire_count} ({fire_percentage:.2f}% of total)'
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12)
    
    # Make the plot tight
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig

def print_fire_coordinates(grid, lat_coords, lon_coords, save_to_file=None):
    """
    Print the coordinates of all fire cells in the grid.
    
    Args:
        grid (np.ndarray): 2D grid where 1 represents fire and 0 represents no fire
        lat_coords (np.ndarray): Array of latitude coordinates
        lon_coords (np.ndarray): Array of longitude coordinates
        save_to_file (str, optional): Path to save the coordinates to a CSV file
    
    Returns:
        list: List of (lat, lon) tuples for all fire cells
    """
    # Find indices where grid == 1 (fire locations)
    fire_indices = np.where(grid == 1)
    
    # Extract corresponding lat/lon coordinates
    fire_coordinates = []
    for i, j in zip(fire_indices[0], fire_indices[1]):
        lat = lat_coords[i]
        lon = lon_coords[j]
        fire_coordinates.append((lat, lon))
    
    print(f"\nFound {len(fire_coordinates)} fire locations:")
    print("Latitude, Longitude")
    print("-" * 30)
    
    fire_coordinates.sort()
    
    for lat, lon in fire_coordinates:
        print(f"{lat:.6f}, {lon:.6f}")
    
    if save_to_file:
        df = pd.DataFrame(fire_coordinates, columns=['latitude', 'longitude'])
        os.makedirs(os.path.dirname(save_to_file) or '.', exist_ok=True)
        df.to_csv(save_to_file, index=False)
        print(f"\nFire coordinates saved to {save_to_file}")
    
    return fire_coordinates

def get_week_dates(date):
    """
    Get the start and end dates for a week containing the given date.
    
    Args:
        date (str or datetime): A date within the desired week
    
    Returns:
        tuple: (start_date, end_date) as datetime.date objects
    """
    if isinstance(date, str):
        date = pd.to_datetime(date).date()
    
    # Calculate start date (Monday of the week)
    start_date = date - datetime.timedelta(days=date.weekday())
    
    # Calculate end date (Sunday of the week)
    end_date = start_date + datetime.timedelta(days=6)
    
    return start_date, end_date

if __name__ == "__main__":
    # Example usage with a one-week period
    min_lat, max_lat = 40, 70 
    min_lon, max_lon = 100, 140
    
    # start_date, end_date = get_week_dates(reference_date)
    start_date, end_date = "2021-07-01", "2021-07-07"

    print(f"Analyzing fire data for week: {start_date} to {end_date}")
    
    # Create grid for the entire week
    grid, lat_coords, lon_coords = create_grid_target_for_period(
        min_lat, max_lat, min_lon, max_lon, start_date, end_date, countries=['Russian_Federation']
    )
    print(f"Grid shape: {grid.shape}")
    print(f"Number of fire cells: {np.sum(grid)}")
    
    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filenames with the date range and region info
    date_range_str = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    image_filename = f"fire_grid_week_{date_range_str}_lat{min_lat}-{max_lat}_lon{min_lon}-{max_lon}.png"
    coords_filename = f"fire_coordinates_week_{date_range_str}_lat{min_lat}-{max_lat}_lon{min_lon}-{max_lon}.csv"
    
    image_path = os.path.join(output_dir, image_filename)
    coords_path = os.path.join(output_dir, coords_filename)
    
    visualize_grid(grid, lat_coords, lon_coords, (start_date, end_date), save_path=image_path)
    fire_coordinates = print_fire_coordinates(grid, lat_coords, lon_coords, save_to_file=coords_path)
    
 
