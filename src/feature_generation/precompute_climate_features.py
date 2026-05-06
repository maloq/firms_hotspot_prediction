import argparse
import os
import time as time_lib
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib.pyplot as plt

import sys
sys.path.append(os.getcwd())

from src.feature_generation.prepare_climate_data import _get_dask_client
from src.feature_generation.prepare_climate_features import (
    _extract_single_series,
    get_selective_feature_configs_and_names,
    parse_selected_features,
)

def precompute_window_features(
    climate_data_dir: str,
    climate_variables: list[str],
    output_dir: str,
    prediction_date_str: str = 'today',
    window_config: dict = None,
    selected_config: dict | None = None,
    max_length_global: int = 120,
    test_mode: bool = False,
    plot_features: bool = False,
):
    """
    Pre-computes and saves window features for all grid points.
    If a `selected_config` is provided, it only computes the features from that selection.
    """
    if not os.path.exists(climate_data_dir):
        print(f"Error: Climate data directory not found at {climate_data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    _get_dask_client()

    prediction_date = pd.to_datetime(prediction_date_str)
    date_folder_name = prediction_date.strftime('%Y-%m-%d')
    start_date = prediction_date - pd.Timedelta(days=max_length_global)

    # Use a dummy time series to get feature names
    sample_ts = np.random.rand(max_length_global)
    
    for i, var in enumerate(climate_variables):
        print(f"\n--- Pre-computing features for variable: {var} ---")
        start_time_var = time_lib.time()

        # Check if the output file already exists to avoid re-computation
        var_output_dir = os.path.join(output_dir, var, date_folder_name)
        output_path = os.path.join(var_output_dir, 'precomputed_ECWMF.nc')
        if os.path.exists(output_path):
            print(f"Precomputed file for {var} already exists at {output_path}. Skipping.")
            continue

        try:
            # Dynamically import to avoid circular dependency issues if this file grows
            from src.feature_generation.load_climate_data import load_climate_variable_mf
            ds = load_climate_variable_mf(
                climate_data_dir, var, 
                time_range=(start_date, prediction_date),
                test_mode=test_mode
            )
        except (FileNotFoundError, IOError) as e:
            print(f"Could not load data for {var}. Skipping. Error: {e}")
            continue

        if i == 0:
            lat_res = abs(ds['latitude'][1] - ds['latitude'][0]).item()
            lon_res = abs(ds['longitude'][1] - ds['longitude'][0]).item()
            print(f"Data resolution: {lat_res:.4f} (lat) x {lon_res:.4f} (lon) degrees")

        # Get feature configuration for windows, using selective logic if available
        feature_params, feature_names = get_selective_feature_configs_and_names(
            sample_ts_array=sample_ts,
            variable_name=var,
            selected_config=selected_config,
            max_length_global=max_length_global,
            # Pass window parameters for the fallback case (when selected_config is None)
            lags_global=window_config['lags_global'],
            windows_global=window_config['windows_global'],
            spans_global=window_config['spans_global'],
            trend_window_global=window_config['trend_window_global'],
        )

        if not feature_names:
            print(f"No window features to compute for {var}. Skipping.")
            ds.close()
            continue

        # print(f"Will compute the following {len(feature_names)} features for '{var}':")
        # for name in feature_names:
        #     print(f"  - {name}")

        # Define a wrapper to apply feature extraction over the dataset
        def feature_extraction_wrapper(ts_data, variable_name, **params):
            features = _extract_single_series(ts_data, variable_name, **params)
            return np.array(list(features.values()), dtype=np.float32)

        # Use xarray's apply_ufunc to apply the function over all pixels
        precomputed_features_da = xr.apply_ufunc(
            feature_extraction_wrapper,
            ds[var].chunk({'valid_time': -1}),
            input_core_dims=[['valid_time']],
            output_core_dims=[['feature']],
            vectorize=True,
            dask='parallelized',
            dask_gufunc_kwargs={'output_sizes': {'feature': len(feature_names)}},
            output_dtypes=[np.float32],
            kwargs={'variable_name': var, **feature_params}
        )

        precomputed_features_da['feature'] = feature_names
        
        # Convert to a dataset for saving
        precomputed_ds = precomputed_features_da.to_dataset(dim='feature')

        os.makedirs(var_output_dir, exist_ok=True)

        saved_features = list(precomputed_ds.data_vars)
        # print(f"Saving the following {len(saved_features)} features to {output_path}:")
        for feature_name in saved_features:
            print(f"  - {feature_name}")

        precomputed_ds.to_netcdf(output_path)
        
        # --- Plotting Feature Maps ---
        if plot_features:
            plots_dir = os.path.join(var_output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            print(f"\nSaving feature plots to {plots_dir}...")
            for feature_name in saved_features:
                plt.figure(figsize=(12, 8))
                data_to_plot = precomputed_ds[feature_name]

                if data_to_plot.isnull().all():
                    print(f"  - Skipping plot for '{feature_name}' as it contains only NaN values.")
                    plt.close()
                    continue
                
                data_to_plot.plot(cmap='viridis')
                plt.title(f'Feature: {feature_name}\nVariable: {var}')
                plot_path = os.path.join(plots_dir, f'{feature_name}.png')
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                plt.close()

        ds.close()
        print(f"Variable {var} took {time_lib.time() - start_time_var:.2f}s")

def precompute_features_from_config(config: dict, date_str: str, test_mode: bool = False, plot: bool = False):
    """
    Runs the full pre-computation pipeline for window features based on a config file.
    """
    print("\n=== Pre-computing Climate Features ===")

    # Extract parameters from config
    try:
        climate_params = config['climate_data_params_prediction']
    except KeyError:
        raise KeyError("'climate_data_params_prediction' not found in config. This is required for prediction.")
    
    climate_data_dir = climate_params.get('climate_data_dir')
    climate_variables = climate_params.get('climate_variables')

    if not climate_data_dir or not climate_variables:
        raise ValueError("'climate_data_dir' and 'climate_variables' must be specified in 'climate_data_params_prediction'.")

    gen_climate_params = config.get('generate_climate_params', {})
    max_length_global = gen_climate_params.get('max_length', 128)
    lags_global = gen_climate_params.get('lags', [])
    windows_global = gen_climate_params.get('windows', [])
    spans_global = gen_climate_params.get('spans', [])
    trend_window_global = gen_climate_params.get('trend_window', [])
    
    # Determine output directory
    output_dir = config.get('precomputed_features_path', 'data/precomputed_climate_features')

    # Load selected features config if it exists
    selected_features_path = "configs/selected_climate.txt"
    if os.path.exists(selected_features_path):
        print(f"Loading selected features from {selected_features_path}")
        selected_config = parse_selected_features(selected_features_path)
    else:
        print("No selected_climate.txt found, will compute all features based on config.")
        selected_config = None

    window_conf = {
        'lags_global': lags_global,
        'windows_global': windows_global,
        'spans_global': spans_global,
        'trend_window_global': trend_window_global,
    }

    print("--- Starting Pre-computation of Window Features ---")
    print(f"Climate Data Dir: {climate_data_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Variables: {climate_variables}")
    print(f"End Date: {date_str}")
    
    precompute_window_features(
        climate_data_dir=climate_data_dir,
        climate_variables=climate_variables,
        output_dir=output_dir,
        prediction_date_str=date_str,
        window_config=window_conf,
        selected_config=selected_config,
        max_length_global=max_length_global,
        test_mode=test_mode,
        plot_features=plot
    )
    
    print("\n--- Pre-computation complete. ---")


def load_precomputed_features_for_day(
    df: pd.DataFrame,
    prediction_date: pd.Timestamp,
    config: dict,
    lat_col: str = 'lat',
    lon_col: str = 'lon'
) -> pd.DataFrame:
    """
    Loads precomputed climate features for a given day and returns them as a new dataframe.

    Args:
        df (pd.DataFrame): DataFrame with latitude and longitude columns.
        prediction_date (pd.Timestamp): The date for which to load features.
        config (dict): The configuration dictionary, containing 'precomputed_features_path'
                       and 'climate_data_params_prediction'.'climate_variables'.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.

    Returns:
        pd.DataFrame: A DataFrame with the new feature columns, with an index matching the input DataFrame.
    """
    try:
        climate_params = config['climate_data_params_prediction']
        output_dir = config.get('precomputed_features_path', 'data/precomputed_climate_features')
        climate_variables = climate_params.get('climate_variables', [])
    except KeyError:
        print("Error: 'climate_data_params_prediction' not found in config.")
        return pd.DataFrame(index=df.index)

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Input DataFrame must have '{lat_col}' and '{lon_col}' columns.")

    date_folder_name = prediction_date.strftime('%Y-%m-%d')
    
    all_feature_ds = []

    for var in climate_variables:
        file_path = os.path.join(output_dir, var, date_folder_name, 'precomputed_ECWMF.nc')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Precomputed file not found for variable '{var}' on {date_folder_name}. Skipping.")
            
        try:
            ds = xr.open_dataset(file_path)
            all_feature_ds.append(ds)
        except Exception as e:
            print(f"Warning: Could not load precomputed file for {var}. Error: {e}. Skipping.")
            continue
    
    if not all_feature_ds:
        print("No precomputed features were loaded.")
        return pd.DataFrame(index=df.index)

    merged_ds = xr.merge(all_feature_ds)


    # For point-wise interpolation, we should provide DataArrays for the coordinates
    # that share a common dimension. Here we use the index of the dataframe.
    selected_features_ds = merged_ds.interp(
        latitude=xr.DataArray(df[lat_col], dims="station", coords={"station": df.index}),
        longitude=xr.DataArray(df[lon_col], dims="station", coords={"station": df.index}),
        method='linear',
        kwargs={'fill_value': None}
    )

    features_df = selected_features_ds.to_dataframe()

    coords_to_drop = ['valid_time', 'station']
    for coord in coords_to_drop:
        if coord in features_df.columns:
            features_df = features_df.drop(columns=coord)

    features_df = features_df.rename(columns={'latitude': lat_col, 'longitude': lon_col})
    
    features_df = features_df.reset_index(drop=True)
    features_df.index = df.index
    

    return features_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute large-window climate features.")
    parser.add_argument('--config', type=str, default='configs/features_config_30d.yaml',
                        help='Path to the YAML configuration file for feature generation.')
    parser.add_argument('--date', type=str, default='today',
                        help="Prediction date for feature calculation (YYYY-MM-DD or 'today').")
    parser.add_argument('--test_mode', action='store_true',
                        help="Run in test mode with smaller data.")
    parser.add_argument('--plot', action='store_true', help="Plot and save feature maps.")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    precompute_features_from_config(
        config=config,
        date_str=args.date,
        test_mode=args.test_mode,
        plot=args.plot
    )
