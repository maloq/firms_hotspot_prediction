import xarray as xr
import pandas as pd
import pathlib
import sys
import os
sys.path.append(os.getcwd())


def normalize_longitudes(ds):
    """
    Normalize longitude values to the range -180 to 180.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing longitude coordinates

    Returns:
    --------
    xarray.Dataset
        Dataset with normalized longitude coordinates in the range [-180, 180]
    """
    if 'longitude' in ds.coords:
        # Using modulo arithmetic to wrap around values into the [-180, 180] range
        ds['longitude'] = ((ds['longitude'] + 180) % 360) - 180
    return ds


def print_dataset_info(ds, variable_name=None):
    """
    Prints information about a dataset 
    """
    
    if 'time' in ds.dims or 'valid_time' in ds.dims:
        time_dim = 'time' if 'time' in ds.dims else 'valid_time'
        time_min = pd.to_datetime(ds[time_dim].values.min())
        time_max = pd.to_datetime(ds[time_dim].values.max())
        print(f"Loaded {variable_name} with time range ({time_min}, {time_max})", end=" ")
    
    if 'latitude' in ds.coords:
        lat_min = float(ds.latitude.min().values)
        lat_max = float(ds.latitude.max().values)
        print(f"and lat range ({lat_min}, {lat_max})", end=" ")
    
    if 'longitude' in ds.coords:
        lon_min = float(ds.longitude.min().values)
        lon_max = float(ds.longitude.max().values)
        print(f"and lon range ({lon_min}, {lon_max})")
    else:
        print()  


def load_and_normalize_dataset(file_path, variable_name):
    """
    Loads the dataset from the given file path based on its extension,
    normalizes the longitudes, and prints dataset info.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset file (.nc, .zarr, or .grib)
    variable_name : str
        Name of the variable to be processed (for informational printing)
        
    Returns:
    --------
    xarray.Dataset
        The loaded and normalized dataset
    """
    print(f"Loading dataset from {file_path}")
    if file_path.endswith('.nc'):
        ds = xr.open_dataset(file_path, decode_timedelta=False)
    elif file_path.endswith('.zarr'):
        ds = xr.open_zarr(file_path, decode_timedelta=False)
    elif file_path.endswith('.grib'):
        ds = xr.open_dataset(file_path, engine='cfgrib', decode_timedelta=False)
    else:
        raise ValueError("File must be .nc, .zarr, or .grib format")
    
    ds = normalize_longitudes(ds)
    print_dataset_info(ds, variable_name)
    return ds


def _process_tp_variable(variable_ds, time_dim='valid_time'):
    """
    Processes the 'tp' (Total Precipitation) variable:
    - Checks for monotonicity along the time dimension.
    - Converts cumulative forecast values to daily totals.
    - Updates metadata accordingly.

    Parameters:
    -----------
    variable_ds : xarray.Dataset
        Dataset containing the 'tp' variable and its coordinates.
        Must have a time dimension (specified by time_dim).
    time_dim : str, optional
        Name of the time dimension/coordinate (default: 'valid_time').

    Returns:
    --------
    xarray.Dataset
        The dataset with 'tp' converted to daily totals and metadata updated.
    """
    print("\nProcessing 'tp' variable: Checking monotonicity and converting to daily totals.")
    variable_name = 'tp' # Explicitly setting for clarity within this function

    # Ensure the variable exists in the input dataset
    if variable_name not in variable_ds:
        raise ValueError(f"'{variable_name}' variable not found in the input dataset.")
    if time_dim not in variable_ds.coords and time_dim not in variable_ds.dims: # Check dims too for safety
         raise ValueError(f"Time coordinate/dimension '{time_dim}' not found in the input dataset for 'tp' processing.")

    tp_data_array = variable_ds[variable_name]

    # Check if there's more than one time step to compare/difference
    if variable_ds[time_dim].size > 1:
        # Check for monotonicity
        diffs = tp_data_array.diff(dim=time_dim)
        if not (diffs >= 0).all():
            print(f"Warning: 'tp' variable is not monotonically increasing along the '{time_dim}' dimension.")
            # Optionally, decide how to handle non-monotonic data

        # Convert cumulative sum to daily totals
        print(f"Converting cumulative 'tp' to daily totals along dimension '{time_dim}'.")
        
        # Calculate the difference between consecutive time steps.
        # label='upper' assigns the difference tp[i] - tp[i-1] to coordinate time[i].
        daily_tp_diff_array = tp_data_array.diff(dim=time_dim, label='upper') 
        
        # The first time step's value is the original cumulative value.
        first_step_tp = tp_data_array.isel({time_dim: 0})
        
        # Concatenate the first step with the differenced values.
        # first_step_tp (coord T0) + daily_tp_diff_array (coords T1, T2, ...)
        # gives coords T0, T1, T2, ...
        daily_tp_array = xr.concat([first_step_tp.expand_dims(dim=time_dim), daily_tp_diff_array], dim=time_dim)


        # Ensure non-negative precipitation
        daily_tp_array = daily_tp_array.where(daily_tp_array >= 0, 0)

        # Update attributes to reflect the change
        daily_tp_array.attrs = tp_data_array.attrs.copy() # Copy original attributes
        daily_tp_array.attrs['long_name'] = f"Daily {daily_tp_array.attrs.get('long_name', 'Total Precipitation')}"
        original_units = daily_tp_array.attrs.get('units', 'm')
        daily_tp_array.attrs['units'] = f"{original_units}/day" # Or adjust logic based on original units
        daily_tp_array.attrs['processing_note'] = 'Converted from cumulative forecast sum to daily total.'

        processed_ds = variable_ds.copy() 
        processed_ds[variable_name] = daily_tp_array
        print("'tp' variable processed into daily totals.")

    else:
        print("Only one time step found for 'tp', skipping daily conversion.")
        processed_ds = variable_ds 

    return processed_ds


def _save_variable_zarr(data_to_save, variable_name, time_dim, output_dir):
    """
    Helper function to generate filename and save a variable dataset to Zarr.

    Parameters:
    -----------
    data_to_save : xarray.Dataset
        The dataset containing the variable to save.
    variable_name : str
        The name of the variable.
    time_dim : str
        The name of the time dimension coordinate.
    output_dir : str
        The base directory to save the variable.

    Returns:
    --------
    str
        The full path to the saved file.

    Raises:
    ------
    ValueError: If latitude or longitude coordinates are not found.
    """
    # --- Filename Generation ---
    start_date = pd.to_datetime(data_to_save[time_dim].values.min()).strftime('%Y%m%d')
    end_date = pd.to_datetime(data_to_save[time_dim].values.max()).strftime('%Y%m%d')
    time_info = f"{start_date}-{end_date}"

    coord_bounds = {}
    # Determine coordinate names robustly
    lat_coord_name = next((c for c in ['latitude', 'lat'] if c in data_to_save.coords), None)
    lon_coord_name = next((c for c in ['longitude', 'lon'] if c in data_to_save.coords), None)

    if lat_coord_name:
        min_lat = float(data_to_save[lat_coord_name].min().values)
        max_lat = float(data_to_save[lat_coord_name].max().values)
        coord_bounds['latitude'] = f"{min_lat:.2f}_{max_lat:.2f}"
    else:
        raise ValueError(f"Latitude coordinate ('latitude' or 'lat') not found in dataset for {variable_name}.")

    if lon_coord_name:
        min_lon = float(data_to_save[lon_coord_name].min().values)
        max_lon = float(data_to_save[lon_coord_name].max().values)
        coord_bounds['longitude'] = f"{min_lon:.2f}_{max_lon:.2f}"
    else:
        raise ValueError(f"Longitude coordinate ('longitude' or 'lon') not found in dataset for {variable_name}.")

    coords_str = "_".join([f"{k}{v}" for k, v in coord_bounds.items()])
    output_path = pathlib.Path(output_dir) / variable_name
    output_path.mkdir(parents=True, exist_ok=True)

    output_filename = f"{variable_name}_{time_info}_{coords_str}.zarr"
    full_output_path = output_path / output_filename

    # --- Saving ---
    print(f"\nSaving {variable_name} ({time_info}) to {full_output_path} (Zarr format)")
    
    # Define chunks
    chunks_dict = {}
    # Time dimension chunking: -1 means a single chunk for the entire dimension
    chunks_dict[time_dim] = -1 
    
    # Spatial dimension chunking: 'auto' or a small number
    # Let's find other dimensions of the data variable and set them to 'auto'
    # Assuming the main variable_name is the one to determine relevant dims for chunking
    if variable_name in data_to_save.data_vars:
        for dim in data_to_save[variable_name].dims:
            if dim != time_dim:
                chunks_dict[dim] = 'auto' # xarray will try to determine a good chunk size
    else:
        # Fallback if variable_name is not in data_vars for some reason,
        # though it should be. This is a defensive measure.
        # We might try to infer from coordinates, but it's safer to chunk only if var is present.
        print(f"Warning: Variable {variable_name} not found directly in dataset for chunking. Chunking might be incomplete.")


    print(f"Using chunks: {chunks_dict}")
    
    # Rechunk the dataset before saving
    # Note: Xarray's auto-chunking might not behave as expected for all dimensions if not all are specified.
    # If specific small chunk sizes are needed for non-time dimensions, replace 'auto' with integer values.
    data_to_save_chunked = data_to_save.chunk(chunks_dict)
    
    # For Zarr, encoding is primarily handled by chunks and compressors configured directly.
    data_to_save_chunked.to_zarr(full_output_path, mode='w', consolidated=True)

    return str(full_output_path)


def process_variable_ECMWF(full_ds_from_file, variable_name):
    """
    Processes a single variable from an ECMWF GRIB dataset:
    - Computes ensemble mean.
    - Transforms dimensions/coordinates to valid_time, latitude, longitude.
    - Verifies variable existence.
    - Extracts the variable as a dataset.
    - If `latest_forecast` is True, splits the ~62-day forecast into two ~1-month datasets.
    - Handles 'tp' variable specific processing.

    Parameters:
    -----------
    full_ds_from_file : xarray.Dataset
        The loaded GRIB dataset (using cfgrib engine) for a single file.
    variable_name : str
        The name of the variable to process (e.g., 't2m', 'd2m').
    latest_forecast : bool, optional
        If True, assumes the input `full_ds_from_file` might cover ~62 days and
        splits it into two output datasets based on the time dimension midpoint.
        Defaults to False.

    Returns:
    --------
    list[xarray.Dataset]
        A list containing the processed variable dataset(s).
        Typically one dataset, or two if `latest_forecast` caused a split.
    """
    print(f"Processing variable: {variable_name} from GRIB input")

    # --- GRIB Preprocessing Steps ---
    ds_proc = full_ds_from_file.mean(dim='number')

    # 1. Drop scalar forecast reference 'time' coordinate if it exists.
    #    This is often the GRIB1 forecastReferenceTime.
    if 'time' in ds_proc.coords and ds_proc['time'].ndim == 0:
        ds_proc = ds_proc.drop_vars('time')

    # 2. Standardize time dimension to 'valid_time'.
    #    cfgrib usually provides 'valid_time' as a coordinate, calculated from
    #    the forecast reference time and the 'step' (forecast lead time).
    #    The 'step' is often the dimension for 'valid_time'.
    if 'valid_time' in ds_proc.coords and 'step' in ds_proc.dims:
        # Check if 'valid_time' coordinate's dimension is indeed 'step'
        if list(ds_proc['valid_time'].dims) == ['step']:
            # Promote 'valid_time' to be the dimension coordinate by swapping 'step' dimension
            # with the 'valid_time' coordinate.
            ds_proc = ds_proc.swap_dims({'step': 'valid_time'})
            # After swap_dims, 'valid_time' is the dimension.
            # The original 'step' coordinate might remain as a non-indexed coordinate/variable.
            # It should be dropped if it's still present and not the main dimension.
            if 'step' in ds_proc.coords: # 'step' might be a coordinate name
                 ds_proc = ds_proc.drop_vars('step', errors='ignore')
        else:
            # 'valid_time' coord exists, 'step' dim exists, but 'valid_time' not on 'step' dim.
            raise ValueError(
                f"GRIB 'valid_time' coordinate exists but its dimensions are {ds_proc['valid_time'].dims}, not ('step',)."
            )
    elif 'valid_time' in ds_proc.dims and 'valid_time' in ds_proc.coords:
        # If 'valid_time' is already a dimension and a coordinate, we are good.
        # Ensure it's an indexed coordinate, though it should be if it's a dimension coordinate.
        if not isinstance(ds_proc.indexes.get('valid_time'), pd.Index):
            ds_proc = ds_proc.set_index(valid_time='valid_time') # Make it an indexed coord
    else:
        # If the GRIB structure isn't as expected (e.g., 'valid_time' missing, or 'step' dim missing when needed).
        err_msg = (
            "Could not standardize GRIB time dimension. Expected 'valid_time' coordinate on 'step' dimension, "
            "or 'valid_time' already as a dimension coordinate. "
            f"Current state: Dims={list(ds_proc.dims)}, Coords={list(ds_proc.coords.keys())}. "
            f"'valid_time' in coords: {'valid_time' in ds_proc.coords}. "
            f"'step' in dims: {'step' in ds_proc.dims}. "
        )
        if 'valid_time' in ds_proc.coords:
            err_msg += f"Dims of 'valid_time' coord: {ds_proc['valid_time'].dims}."
        raise ValueError(err_msg)

    # 3. Drop other unwanted coordinates like 'surface'.
    coords_to_drop = ['surface']
    # Any other form of 'time' that isn't 'valid_time' and isn't a dimension can also be dropped.
    if 'time' in ds_proc.coords and 'time' not in ds_proc.dims:
        coords_to_drop.append('time')

    final_processed_ds = ds_proc.drop_vars(coords_to_drop, errors='ignore')
    
    time_dim = 'valid_time' # This should now be the main time dimension/coordinate

    # Final check: 'valid_time' must be a dimension and an indexed coordinate.
    if not (time_dim in final_processed_ds.dims and \
            time_dim in final_processed_ds.coords and \
            isinstance(final_processed_ds.indexes.get(time_dim), pd.Index)):
        # If not, try to set 'valid_time' as index one last time if it's a coordinate and dimension
        if time_dim in final_processed_ds.dims and time_dim in final_processed_ds.coords:
            final_processed_ds = final_processed_ds.set_index({time_dim: time_dim})
            if not isinstance(final_processed_ds.indexes.get(time_dim), pd.Index): # Check again
                 raise ValueError(f"Failed to set '{time_dim}' as an indexed dimension coordinate.")
        else:
            raise ValueError(
                f"Post-processing, '{time_dim}' is not a dimension and/or coordinate as expected. "
                f"Dims: {list(final_processed_ds.dims)}, Coords: {list(final_processed_ds.coords.keys())}"
            )
    # --- End of GRIB Preprocessing ---

    if variable_name not in final_processed_ds.data_vars:
        available_vars = list(final_processed_ds.data_vars)
        raise ValueError(f"Variable '{variable_name}' not found in GRIB processed dataset. Available: {available_vars}")

    print(f"\nExtracting variable '{variable_name}'...")
    variable_ds = final_processed_ds[[variable_name]].copy() # Ensure it's a new dataset object

    if variable_name == 'tp':
        variable_ds = _process_tp_variable(variable_ds, time_dim)
        return [variable_ds] # Ensure a list containing the dataset is returned
    else:
        return [variable_ds]


def process_variable_ERA5(full_ds_from_file, variable_name):
    """
    Processes a single variable from an ERA5 dataset:
    - Verifies its existence.
    - Extracts the variable as a dataset.
    - Ensures the time coordinate is named 'valid_time'.
    
    Parameters:
    -----------
    full_ds_from_file : xarray.Dataset
        The loaded dataset for a single file.
    variable_name : str
        The name of the variable to process.
        
    Returns:
    --------
    xarray.Dataset
        The processed variable dataset with 'valid_time' as the time coordinate.
    """
    print(f"Processing variable: {variable_name} from ERA5 input")
    
    if variable_name not in full_ds_from_file:
        raise ValueError(f"Variable '{variable_name}' not found in ERA5 dataset. Available variables: {list(full_ds_from_file.data_vars)}")
    
    variable_ds = full_ds_from_file[[variable_name]].copy() # Use [[var]] to keep it as Dataset, and .copy()
    
    # Standardize time coordinate to 'valid_time'
    if 'time' in variable_ds.coords and 'valid_time' not in variable_ds.coords:
        variable_ds = variable_ds.rename({'time': 'valid_time'})
        print("Renamed 'time' coordinate to 'valid_time'.")
    elif 'valid_time' not in variable_ds.coords:
        # Check common time coordinate names
        potential_time_coords = [tc for tc in ['t', 'datetime', 'date'] if tc in variable_ds.coords]
        if potential_time_coords:
            original_time_coord = potential_time_coords[0]
            variable_ds = variable_ds.rename({original_time_coord: 'valid_time'})
            print(f"Renamed '{original_time_coord}' coordinate to 'valid_time'.")
        else:
            raise ValueError("A recognized time coordinate ('time', 'valid_time', 't', 'datetime', 'date') not found in ERA5 variable dataset.")

    if 'valid_time' not in variable_ds.coords:
         raise ValueError("Failed to set 'valid_time' coordinate in ERA5 variable dataset.")

    print(f"Dataset {variable_name} shape: {variable_ds[variable_name].shape}")
    # 'tp' variable is not specially processed for ERA5 in this script's original form,
    # but if it were, the call would go here.
    
    return variable_ds


def extract_and_save_variables(file_paths, variable_names, output_dir='climate_data_files', source='ERA5'):
    """
    Loads data from multiple files, extracts specific variables, merges them across files,
    and saves each consolidated variable to a NetCDF file.
    
    Parameters:
    -----------
    file_paths : list[str]
        List of paths to the dataset files.
    variable_names : list[str]
        List of variable names to extract and consolidate.
    output_dir : str
        Base directory where the extracted data will be saved.
    source : str
        The source of the data, e.g., 'ERA5' or 'ECMWF'.
    latest_forecast_ecmwf : bool
        Flag for ECMWF source, indicating if files (especially long ones)
        should be subject to splitting logic.
        
    Returns:
    --------
    list[str]
        Paths to the saved consolidated files (one per variable).
    """
    all_saved_paths = []
    time_dim_name = 'valid_time' # Standardized time dimension name

    for variable_name in variable_names:
        print(f"\n--- Consolidating data for variable: {variable_name} ---")
        datasets_for_variable = []

        for file_idx, file_path in enumerate(file_paths):
            # Pass a generic name or the specific variable if only one, for load_and_normalize_dataset's print
            # Using the current variable_name for context, though the file might contain more.
            print(f"\nReading file {file_idx + 1}/{len(file_paths)}: {file_path} for variable {variable_name}")
            # load_and_normalize_dataset normalizes longitudes and prints info
            raw_ds = load_and_normalize_dataset(file_path, variable_name)
            
            processed_datasets_from_file = []
            if source == 'ERA5':
                processed_ds = process_variable_ERA5(raw_ds, variable_name)
                processed_datasets_from_file.append(processed_ds)
            elif source == 'ECMWF':
                processed_datasets_from_file = process_variable_ECMWF(raw_ds, variable_name)
            else:
                raw_ds.close() # Ensure closure if error
                raise ValueError(f"Invalid source: {source}")
            
            datasets_for_variable.extend(processed_datasets_from_file)
            raw_ds.close() # Close the raw dataset after processing for this variable

        if not datasets_for_variable:
            print(f"No data found for variable '{variable_name}' across all files. Skipping.")
            continue

        print(f"\nConcatenating {len(datasets_for_variable)} dataset(s) for variable '{variable_name}' along '{time_dim_name}' dimension.")
        
        # Check for time dimension consistency before concatenation
        for i, ds_chunk in enumerate(datasets_for_variable):
            if time_dim_name not in ds_chunk.coords and time_dim_name not in ds_chunk.dims:
                err_msg = f"Dataset chunk {i} for variable '{variable_name}' from an unknown file does not have the expected time dimension '{time_dim_name}'. Coords: {list(ds_chunk.coords)}, Dims: {list(ds_chunk.dims)}"
                print(f"Problematic dataset chunk ({i}):")
                print(ds_chunk)
                raise ValueError(err_msg)
        
        try:
            merged_ds = xr.concat(datasets_for_variable, dim=time_dim_name)
        except Exception as e:
            print(f"Error during concatenation for variable {variable_name}: {e}")
            print("Details of datasets attempted to concatenate:")
            for i, d in enumerate(datasets_for_variable):
                print(f"--- Dataset {i} ---")
                print(d)
                if time_dim_name in d.coords:
                    print(f"Time coord '{time_dim_name}': {d[time_dim_name].min().values} to {d[time_dim_name].max().values}")
                else:
                    print(f"Time coord '{time_dim_name}' NOT FOUND.")
            raise

        print(f"Sorting merged data for '{variable_name}' by '{time_dim_name}'.")
        merged_ds = merged_ds.sortby(time_dim_name)
        
        # Optional: Check for and handle duplicate time steps if necessary
        # Example: unique_times, counts = np.unique(merged_ds[time_dim_name], return_counts=True)
        # if (counts > 1).any(): print(f"Warning: Duplicate time steps found in merged {variable_name}")

        saved_path = _save_variable_zarr(merged_ds, variable_name, time_dim_name, output_dir)
        all_saved_paths.append(saved_path)
        
        merged_ds.close() # Close the merged dataset after saving

    return all_saved_paths


def check_saved_file(file_path):
    if os.path.exists(file_path): # For Zarr, this checks if the directory exists
        print("="*50)
        print(f"File {file_path} exists")
        # Open Zarr dataset
        ds = xr.open_zarr(file_path, consolidated=True, decode_timedelta=False)
        print_dataset_info(ds, str(pathlib.Path(file_path).name)) # Use pathlib for name
    else:
        print(f"File {file_path} does not exist")


if __name__ == "__main__":

    import glob

    mode = "ECMWF_all" # Change this to test different modes
    # Determine if the "latest forecast" logic (splitting of 62-day files) should apply
    # This is a placeholder; in a real scenario, this might be determined by configuration or file properties.
    apply_latest_forecast_splitting_for_ecmwf = False 

    # Option to limit the number of files for testing  
    max_files_to_process_in_test_mode = 2    # Number of files to process if limiting is enabled

    if mode == "ECMWF":
        # Example for a single ECMWF file, but now uses the new consolidated logic
        file_paths = ["data/raw_climate/ECMWF/seasonal_forecast_2023_02.nc"] # Ensure this file exists for testing
        output_dir = "data/climate_data_files/ECMWF_single"
        variable_names = ["tp", "t2m", "d2m", "stl1"] # "stl1"]
        source_type = 'ECMWF'
        # For a single file, latest_forecast might be relevant if it's a long one
        # apply_latest_forecast_splitting_for_ecmwf = True # if this specific file is a 62-day one

    elif mode == "ECMWF_all":
        file_paths = sorted(glob.glob("data/raw_climate/ECMWF/*.grib")) # Sort for chronological order if names allow
        if not file_paths:
            print("No GRIB files found for ECMWF_all mode. Please check the path.")
            sys.exit(1)
        output_dir = "data/climate_data_files/ECMWF"
        variable_names = ["tp", "t2m", "d2m", "stl1"] # add "stl1"]
        source_type = 'ECMWF'
        # apply_latest_forecast_splitting_for_ecmwf could be True if any of these are "latest" 62-day files.
        # Or, this flag could be dynamically set per file if process_variable_ECMWF was adapted to auto-detect.
        # For now, it's a global flag for the batch.

    elif mode == "ERA5":
        # Example for ERA5, potentially multiple files if glob matches more
        file_paths = sorted(glob.glob("data/raw_climate/ERA5/tp*.nc")) # Example: tp_2020.nc, tp_2021.nc
        if not file_paths:
            print("No NetCDF files found for ERA5 mode. Please check the path.")
            sys.exit(1)
        output_dir = "data/climate_data_files/ERA5"
        variable_names = ["tp"] 
        source_type = 'ERA5'
    else:
        raise ValueError(f"Invalid mode: {mode}")
    limit_files_for_testing = False
    # Apply file limiting if enabled for testing
    if limit_files_for_testing and file_paths and len(file_paths) > max_files_to_process_in_test_mode:
        print(f"\n*** TEST MODE ENABLED: Limiting processing to the first {max_files_to_process_in_test_mode} files. ***\n")
        file_paths = file_paths[:max_files_to_process_in_test_mode]

    print(f"Running in mode: {mode}")
    print(f"Input files: {file_paths}")
    print(f"Variables to process: {variable_names}")
    print(f"Output directory: {output_dir}")
   
    if file_paths: # Proceed only if files are found
        # Call extract_and_save_variables ONCE with all file paths for the selected mode
        consolidated_files_paths = extract_and_save_variables(
            file_paths,
            variable_names,
            output_dir,
            source=source_type,
        )
        
        print("\n--- Checking all consolidated saved files ---")
        for path in consolidated_files_paths:
            if path: # If a path was returned (i.e., saving was successful)
                check_saved_file(path)
    else:
        print("No files to process.")
