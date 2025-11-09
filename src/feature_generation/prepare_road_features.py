import glob
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine # Import Affine explicitly
from pyproj import Transformer, CRS # Use pyproj directly for speed


def get_road_features_for_coords(coords: np.ndarray, 
                                npz_path: str = "data/roads_features.npz") -> pd.DataFrame:
    """
    Extract road feature data for specific geographic coordinates using optimized methods.
    
    Args:
        coords: NumPy array with shape (2, N) where row 0 contains latitudes 
               and row 1 contains longitudes. Coordinates should be in EPSG:4326 (WGS84).
        npz_path: Path to the NPZ file containing road features data.
        
    Returns:
        pandas DataFrame with columns for distance to nearest road (meters) 
        and road densities at different window sizes for each input coordinate.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Road features data not found at {npz_path}")

    # Load the road features data using a context manager
    print(f"Loading road features from {npz_path}...")
    with np.load(npz_path) as data:
        
        # Check what features are available
        distance_map = data['distance_meters']
        raster_shape = distance_map.shape
        print(f"Loaded distance map. Shape: {raster_shape}")

        # --- Load and reconstruct the Affine transform ---
        # This part is slightly more robust to how Affine might be stored
        transform_data = data['transform']
        try:
            # If saved as an object array containing Affine
            item = transform_data.item(0) if transform_data.ndim > 0 else transform_data.item()
            if isinstance(item, Affine):
                transform = item
            # If saved as a flat array of coefficients [a, b, c, d, e, f]
            elif isinstance(transform_data, np.ndarray) and transform_data.size >= 6:
                coeffs = transform_data.flatten()[:6]
                transform = Affine(*coeffs)
            else:
                raise TypeError(f"Loaded transform data is of unexpected type or size: {type(transform_data)}, size {transform_data.size}. Expected Affine object or coefficient array.")
        except Exception as e:
             raise ValueError(f"Failed to load or reconstruct the Affine transform: {e}")
        print(f"Reconstructed transform: {transform}")
        
        crs_wkt = str(data['crs_wkt'])
        if not crs_wkt:
            raise ValueError("CRS information is missing from the NPZ file.")
            
        # Find the density columns
        density_keys = sorted([k for k in data.keys() if k.startswith('density_')])
        print(f"Found {len(density_keys)} density maps: {density_keys}")
        
        # --- Optimized Coordinate Transformation using pyproj ---
        print("Converting geographic coordinates to raster coordinates using pyproj...")
        if coords.shape[0] != 2:
            raise ValueError("coords must be a 2xN array with latitudes in row 0 and longitudes in row 1")
            
        n_points = coords.shape[1]
        lats = coords[0]
        lons = coords[1]
        
        # Create a transformer from WGS84 (EPSG:4326) to the raster's CRS
        source_crs = CRS("EPSG:4326")
        target_crs = CRS.from_wkt(crs_wkt) # Use the CRS from the NPZ file
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True) # Ensure (lon, lat) -> (x, y) order

        # Perform the transformation in a single call
        # Note: transformer.transform expects x, y order, so pass lons, lats
        x_coords, y_coords = transformer.transform(lons, lats)
        print(f"Transformed {n_points} coordinates.")

        # --- Vectorized Row/Col Calculation ---
        print("Calculating raster row/column indices...")
        # Convert projected coordinates to pixel coordinates (rows, cols)
        # Note: rasterio.transform.rowcol handles arrays directly
        rows, cols = rasterio.transform.rowcol(transform, x_coords, y_coords)
        
        # Ensure rows and cols are integer arrays for indexing
        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)
        print("Calculated row/column indices.")

        # --- Vectorized Data Extraction ---
        print("Extracting feature values...")
        
        # Initialize results dictionary with NaNs
        results = {
            'lat': lats, # Store original coords for reference if needed
            'lon': lons,
            'distance_to_road_meters': np.full(n_points, np.nan, dtype=distance_map.dtype)
        }
        for key in density_keys:
            # Ensure density maps are loaded if not already
            density_map = data[key]
            results[key] = np.full(n_points, np.nan, dtype=density_map.dtype)
            
        # Create a mask for coordinates that fall *inside* the raster bounds
        valid_mask = (
            (rows >= 0) & (rows < raster_shape[0]) &
            (cols >= 0) & (cols < raster_shape[1])
        )
        
        num_valid = np.sum(valid_mask)
        num_invalid = n_points - num_valid
        if num_invalid > 0:
             print(f"Warning: {num_invalid} points fall outside the raster bounds and will have NaN values.")

        # Get the valid row/col indices
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        
        # Extract data using advanced indexing (vectorized) ONLY for valid points
        results['distance_to_road_meters'][valid_mask] = distance_map[valid_rows, valid_cols]
        
        for key in density_keys:
            density_map = data[key] # Access the density map array
            results[key][valid_mask] = density_map[valid_rows, valid_cols]
            
        print("Feature extraction complete.")

    # Return as pandas DataFrame
    df_results = pd.DataFrame(results)
    print(f"Created DataFrame with shape: {df_results.shape}")
    return df_results
