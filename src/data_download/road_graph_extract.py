import geopandas as gpd
import pandas as pd
import os
import glob
import rasterio
from rasterio.features import rasterize
import rasterio
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, uniform_filter


def load_shapefiles(pattern: str) -> list[gpd.GeoDataFrame]:
    """Loads all shapefiles matching the given glob pattern."""
    shapefile_paths = glob.glob(pattern)
    gdf_list = []
    for path in shapefile_paths:
        try:
            gdf = gpd.read_file(path)
            gdf_list.append(gdf)
            print(f"Loaded {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return gdf_list


def combine_geodataframes(gdf_list: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame | None:
    """Combines a list of GeoDataFrames into a single one."""
    if not gdf_list:
        print("No shapefiles were loaded.")
        return None

    combined_gdf = pd.concat(gdf_list, ignore_index=True)
    print("Combined all regional road data.")

    # Ensure the combined GeoDataFrame has a CRS set, potentially from the first file
    if combined_gdf.crs is None and gdf_list:
         combined_gdf.crs = gdf_list[0].crs # Or choose an appropriate CRS
    return combined_gdf


def ensure_crs(gdf: gpd.GeoDataFrame, target_epsg: int) -> gpd.GeoDataFrame | None:
    """Ensures the GeoDataFrame is in the target CRS, reprojecting if necessary."""
    if gdf.crs is None:
        print("Error: Combined GeoDataFrame has no CRS. Cannot reproject.")
        # Handle error: You might need to manually set a CRS if the source files lack it
        # For example: gdf.set_crs("EPSG:4326", inplace=True) # Assuming original is WGS84
        return None # Indicate failure or handle differently
    elif gdf.crs.to_epsg() != target_epsg:
         print(f"Reprojecting combined data from {gdf.crs} to EPSG:{target_epsg}...")
         gdf = gdf.to_crs(epsg=target_epsg)
    return gdf


def calculate_raster_params(gdf: gpd.GeoDataFrame, resolution: float) -> tuple:
    """Calculates raster dimensions and transform based on GeoDataFrame bounds and resolution."""
    minx, miny, maxx, maxy = gdf.total_bounds
    print(f"Combined Bounding Box: {minx}, {miny}, {maxx}, {maxy}")

    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)

    print(f"Raster Size: {width} x {height}")
    print(f"Transform: {transform}") 
    return width, height, transform


def rasterize_geodataframe(gdf: gpd.GeoDataFrame, width: int, height: int, transform, default_value: int = 255, fill_value: int = 0) -> np.ndarray:
    """Rasterizes GeoDataFrame geometries."""
    shapes = [(geom, 1) for geom in gdf.geometry] 
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=fill_value,         
        default_value=default_value,
        dtype=rasterio.uint8
    )
    return raster


def save_raster(raster: np.ndarray, output_path: str, width: int, height: int, crs, transform):
    """Saves the numpy raster array to a GeoTIFF file."""
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(raster, 1)
    print(f"Raster saved as {output_path}")


def make_road_raster():
    shapefile_directory_pattern = "data/russia_map/russia_parts/*/*.shp"
    target_epsg = 3857 
    raster_resolution = 1000
    raster_output_path = "data/combined_roads_raster.tif"

    # 1. Load data
    all_roads_gdf_list = load_shapefiles(shapefile_directory_pattern)
    # 2. Combine data
    roads_gdf = combine_geodataframes(all_roads_gdf_list)
    if roads_gdf is None:
        exit() 
    # 3. Ensure CRS
    roads_gdf = ensure_crs(roads_gdf, target_epsg)
    if roads_gdf is None:
        exit() 
    print(roads_gdf.head())
    # 4. Calculate raster parameters
    width, height, transform = calculate_raster_params(roads_gdf, raster_resolution)
    # 5. Rasterize
    raster_data = rasterize_geodataframe(roads_gdf, width, height, transform, default_value=255, fill_value=0)
    # 6. Save raster
    save_raster(raster_data, raster_output_path, width, height, roads_gdf.crs, transform)


def load_raster_file(raster_path: str) -> tuple[np.ndarray, rasterio.Affine, str, float] | None:
    """
    Loads a raster file and returns its data, transform, CRS, and resolution.
    
    Args:
        raster_path: Path to the raster file
        
    Returns:
        Tuple containing (raster_data, transform, crs_wkt, resolution) or None if loading fails
    """
    if not os.path.exists(raster_path):
        print(f"Error: Input raster file not found at {raster_path}")
        return None
    
    try:
        with rasterio.open(raster_path) as src:
            print("Loading raster data...")
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs
            print(f"Raster loaded. Shape: {raster_data.shape}, CRS: {crs}")
            
            # Process CRS
            crs_wkt = crs.to_wkt() if crs else None
            if not crs:
                warnings.warn(f"Raster file {raster_path} does not have a CRS defined. "
                            "Assuming distances calculated are based on the units of the "
                            "undefined coordinate system (likely meters if EPSG:3857 was used).")
            
            # Get pixel resolution
            resolution = abs(transform.a)  # Use width for distance calculation
            if abs(transform.a) != abs(transform.e):
                warnings.warn(f"Raster pixels are not square ({transform.a} x {transform.e}). "
                            "Using width ({transform.a}) for distance calculation. "
                            "Results might be slightly inaccurate depending on orientation.")
            print(f"Pixel resolution: {resolution} meters/pixel")
            
            return raster_data, transform, crs_wkt, resolution
            
    except rasterio.RasterioIOError as e:
        print(f"Error opening or reading raster file {raster_path}: {e}")
        
    return None

def calculate_distance_map(raster_data: np.ndarray, resolution: float) -> np.ndarray:
    """
    Calculates distance to the nearest road in meters.
    
    Args:
        raster_data: Binary raster where 0 represents roads
        resolution: Pixel resolution in meters
        
    Returns:
        A numpy array with distances in meters
    """
    print("Calculating Euclidean Distance Transform (EDT)...")
    distance_pixels = distance_transform_edt(raster_data == 0)
    print("Distance transform in pixels calculated.")
    
    distance_meters = distance_pixels * resolution
    print("Distance converted to meters.")
    
    return distance_meters



def calculate_density(road_raster_data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculates road density using a uniform filter (moving window average).

    Args:
        road_raster_data: NumPy array where non-zero values represent roads and 0 represents non-roads.
        window_size: The side length of the square window in pixels.

    Returns:
        A NumPy array of the same shape as the input, where each pixel value
        represents the proportion (0.0 to 1.0) of road pixels within its
        neighborhood defined by the window size.
    """
    print(f"Calculating road density with a {window_size}x{window_size} pixel window...")
    assert window_size > 1, "Window size must be greater than 1"

    binary_raster = (road_raster_data != 0).astype(np.float32)
    density_map = uniform_filter(binary_raster, size=window_size, mode='constant', cval=0.0)

    print("Density calculation complete.")
    return density_map


def verify_saved_features(npz_path: str):
    """
    Verifies the contents of the saved NPZ file containing road features.

    Args:
        npz_path: Path to the NPZ file.
    """
    if not os.path.exists(npz_path):
        print(f"Error: File not found for verification: {npz_path}")
        return

    print(f"\nVerifying saved file: {npz_path}")
    try:
        loaded_data = np.load(npz_path)
        print("Keys in the saved file:", list(loaded_data.keys()))

        # Verify distance map
        if 'distance_meters' in loaded_data:
            dist_map = loaded_data['distance_meters']
            print("\n--- Verifying Distance Map ---")
            print("Loaded distance map shape:", dist_map.shape)
            print("Example distances (min, max):", np.min(dist_map), np.max(dist_map))
        else:
            print("Error: 'distance_meters' key not found in saved file.")

        # Verify density maps
        print("\n--- Verifying Density Maps ---")
        density_keys = [k for k in loaded_data.keys() if k not in ['distance_meters', 'transform', 'crs_wkt']]
        if not density_keys:
            print("No density maps found in the file.")
        else:
            for key in density_keys:
                if key in loaded_data: # Double check key exists
                    density_map = loaded_data[key]
                    print(f"Loaded density map '{key}' shape:", density_map.shape)
                    print(f"Example densities for '{key}' (min, max):", np.min(density_map), np.max(density_map))
                else: # Should not happen with list comprehension above, but good practice
                     print(f"Warning: Key '{key}' listed but not found during verification.")


        print("\n--- Verifying Metadata ---")
        if 'transform' in loaded_data:
            print("Loaded transform:", loaded_data['transform'])
        else:
             print("Warning: 'transform' key not found in saved file.")

        if 'crs_wkt' in loaded_data:
            print("Loaded CRS (WKT):", loaded_data['crs_wkt'])
        else:
            print("Warning: 'crs_wkt' key not found in saved file.")

        loaded_data.close()
        print("\nVerification complete.")
    except Exception as e:
        print(f"Error verifying saved file: {e}")


if __name__ == "__main__":

    create_raster = True
    raster_filepath = "data/combined_roads_raster.tif" 
    output_npz_path = "data/roads_features.npz"
    window_size_pixels = [10, 50, 100, 300, 500]

    if not os.path.exists(raster_filepath):
        print("Road raster not found, creating...")
        if create_raster:
            make_road_raster() 
        else:
            print("Road raster not found, exiting...")
            exit()
    else:
        print("Road raster found, skipping creation...")

    road_raster_data, transform, crs_wkt, resolution = load_raster_file(raster_filepath)
    distance_meters = calculate_distance_map(road_raster_data, resolution)
    density_maps = {}
    for window_size in window_size_pixels:
        density_window_size = window_size
        window_size_meters = density_window_size * resolution

        print(f"\nCalculating road density using a {density_window_size}x{density_window_size} pixel window "
                f"(approx {window_size_meters:.1f} x {window_size_meters:.1f} meters)...")
        density_map = calculate_density(road_raster_data, density_window_size)
        density_maps[f"density_{density_window_size}"] = density_map
    
    output_dir = os.path.dirname(output_npz_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Saving features to compressed NumPy array: {output_npz_path}")
    np.savez_compressed(
        output_npz_path,
        distance_meters=distance_meters,
        transform=np.array(transform),
        crs_wkt=crs_wkt,
        **density_maps
    )
    print("Features saved successfully.")

    verify_saved_features(output_npz_path)
