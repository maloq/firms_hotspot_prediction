import xarray as xr
import numpy as np
import os
from pathlib import Path
import traceback 
import warnings
warnings.filterwarnings("ignore")

def find_matching_wind_files(base_data_path="data/climate_data_files/ECMWF"):
    """
    Find all matching u10 and v10 zarr files in the specified directory.
    
    Parameters:
    -----------
    base_data_path : str
        Base path to the ECMWF climate data files
        
    Returns:
    --------
    list : List of tuples containing (u10_file, v10_file, base_filename)
    """
    u10_dir = os.path.join(base_data_path, "u10")
    v10_dir = os.path.join(base_data_path, "v10")
    
    if not os.path.exists(u10_dir):
        raise FileNotFoundError(f"u10 directory not found: {u10_dir}")
    if not os.path.exists(v10_dir):
        raise FileNotFoundError(f"v10 directory not found: {v10_dir}")
    
    # Find all u10 zarr files
    u10_files = []
    for item in os.listdir(u10_dir):
        item_path = os.path.join(u10_dir, item)
        if os.path.isdir(item_path) and item.endswith('.zarr'):
            u10_files.append(item)
    
    print(f"Found {len(u10_files)} u10 files")
    
    matching_pairs = []
    
    for u10_file in u10_files:
        # Extract the time and coordinate part from u10 filename
        # Pattern: u10_20000102-20250601_latitude35.00_80.00_longitude25.00_179.00.zarr
        if u10_file.startswith('u10_'):
            # Remove 'u10_' prefix to get the base filename
            base_filename = u10_file[4:]  # Remove 'u10_' prefix
            
            # Construct corresponding v10 filename
            v10_file = f"v10_{base_filename}"
            v10_path = os.path.join(v10_dir, v10_file)
            u10_path = os.path.join(u10_dir, u10_file)
            
            # Check if corresponding v10 file exists
            if os.path.exists(v10_path):
                matching_pairs.append((u10_path, v10_path, base_filename))
                print(f"Found matching pair: {u10_file} <-> {v10_file}")
            else:
                print(f"Warning: No matching v10 file found for {u10_file}")
    
    if not matching_pairs:
        raise ValueError("No matching u10/v10 file pairs found")
    
    print(f"\nTotal matching pairs found: {len(matching_pairs)}")
    return matching_pairs


def calculate_absolute_wind_speed_batch(file_pairs=None, 
                                       base_data_path="data/climate_data_files/ECMWF",
                                       output_var_name="w_abs_10"):
    """
    Calculate absolute wind speed from u10 and v10 components for multiple file pairs.
    
    Parameters:
    -----------
    file_pairs : list
        List of tuples (u10_path, v10_path, base_filename). If None, automatically find all pairs.
    base_data_path : str
        Base path to the ECMWF climate data files
    output_var_name : str
        Name for the output variable and folder
        
    Returns:
    --------
    list : List of paths to the saved output zarr files
    """
    
    if file_pairs is None:
        file_pairs = find_matching_wind_files(base_data_path)
    
    output_paths = []
    
    for i, (u10_path, v10_path, base_filename) in enumerate(file_pairs, 1):
        print(f"\n{'='*60}")
        print(f"Processing pair {i}/{len(file_pairs)}")
        print(f"u10: {u10_path}")
        print(f"v10: {v10_path}")
        print(f"{'='*60}")
        
        try:
            output_path = calculate_absolute_wind_speed_single(
                u10_path, v10_path, base_filename, base_data_path, output_var_name
            )
            output_paths.append(output_path)
            print(f"✓ Successfully processed pair {i}")
            
        except Exception as e:
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"Successfully processed: {len(output_paths)}/{len(file_pairs)} pairs")
    print(f"{'='*60}")
    
    return output_paths


def calculate_absolute_wind_speed_single(u10_path, v10_path, base_filename, 
                                        base_data_path, output_var_name):
    """
    Calculate absolute wind speed from a single pair of u10 and v10 files.
    
    Parameters:
    -----------
    u10_path : str
        Path to the u10 zarr file
    v10_path : str
        Path to the v10 zarr file
    base_filename : str
        Base filename without u10/v10 prefix
    base_data_path : str
        Base path to the ECMWF climate data files
    output_var_name : str
        Name for the output variable
        
    Returns:
    --------
    str : Path to the saved output zarr file
    """
    
    # Load the zarr datasets
    try:
        u10_ds = xr.open_zarr(u10_path, chunks='auto')
        v10_ds = xr.open_zarr(v10_path, chunks='auto')
        
        print("Successfully loaded u10 and v10 datasets")
        print(f"u10 dataset dimensions: {dict(u10_ds.dims)}")
        print(f"v10 dataset dimensions: {dict(v10_ds.dims)}")
        
    except Exception as e:
        raise Exception(f"Error loading zarr files: {e}")
    
    # Extract the wind components (assuming the variable names are 'u10' and 'v10')
    u10_var = u10_ds['u10'] if 'u10' in u10_ds.data_vars else u10_ds[list(u10_ds.data_vars)[0]]
    v10_var = v10_ds['v10'] if 'v10' in v10_ds.data_vars else v10_ds[list(v10_ds.data_vars)[0]]
    
    print(f"u10 variable shape: {u10_var.shape}")
    print(f"v10 variable shape: {v10_var.shape}")
    
    # Check if dimensions match
    if u10_var.shape != v10_var.shape:
        raise ValueError(f"Shape mismatch: u10 {u10_var.shape} vs v10 {v10_var.shape}")
    
    # Calculate absolute wind speed: sqrt(u^2 + v^2)
    print("Calculating absolute wind speed...")
    w_abs_data = np.sqrt(u10_var**2 + v10_var**2)
    
    # Create a new dataset with the calculated wind speed
    # Use the same coordinates as the input datasets
    coords = u10_var.coords
    dims = u10_var.dims
    
    w_abs_ds = xr.Dataset(
        {output_var_name: (dims, w_abs_data.data)},
        coords=coords,
        attrs={
            'description': f'Absolute wind speed calculated from u10 and v10 components',
            'units': 'm/s',
            'long_name': 'Absolute wind speed at 10m height',
            'calculation': 'sqrt(u10^2 + v10^2)',
            'source_u10': u10_path,
            'source_v10': v10_path
        }
    )
    
    # Create output directory
    output_dir = os.path.join(base_data_path, output_var_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_filename = f"{output_var_name}_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Saving absolute wind speed to: {output_path}")
    
    # Save as zarr
    try:
        # Remove existing zarr store if it exists
        if os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)
            print(f"Removed existing zarr store: {output_path}")
        
        # Save the dataset
        w_abs_ds.to_zarr(output_path, mode='w')
        print(f"Successfully saved absolute wind speed to {output_path}")
        
        # Print summary statistics
        print("\nSummary statistics:")
        print(f"Min wind speed: {float(w_abs_data.min().compute()):.3f} m/s")
        print(f"Max wind speed: {float(w_abs_data.max().compute()):.3f} m/s")
        print(f"Mean wind speed: {float(w_abs_data.mean().compute()):.3f} m/s")
        
    except Exception as e:
        raise Exception(f"Error saving zarr file: {e}")
    
    return output_path


def load_wind_data(file_path):
    """
    Load wind data from a zarr file.
    
    Parameters:
    -----------
    file_path : str
        Path to the zarr file
        
    Returns:
    --------
    xarray.Dataset : The loaded dataset
    """
    try:
        ds = xr.open_zarr(file_path, chunks='auto')
        print(f"Loaded dataset from {file_path}")
        print(f"Variables: {list(ds.data_vars)}")
        print(f"Dimensions: {ds.dims}")
        return ds
    except Exception as e:
        raise Exception(f"Error loading zarr file {file_path}: {e}")


# Example usage
if __name__ == "__main__":
    # Find all matching wind file pairs and calculate absolute wind speed
    try:
        print("Finding matching u10 and v10 files...")
        file_pairs = find_matching_wind_files()
        
        print("\nCalculating absolute wind speed for all pairs...")
        output_paths = calculate_absolute_wind_speed_batch(file_pairs)
        
        print(f"\nCompleted! Processed {len(output_paths)} files:")
        for path in output_paths:
            print(f"  - {path}")
        
        # Load and verify the first output file
        if output_paths:
            print(f"\nVerifying first output file...")
            result_ds = load_wind_data(output_paths[0])
            print(f"Variables: {list(result_ds.data_vars)}")
            print(f"Shape: {result_ds[list(result_ds.data_vars)[0]].shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        
