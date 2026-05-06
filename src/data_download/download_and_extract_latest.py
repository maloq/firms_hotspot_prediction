import yaml
from pathlib import Path
import asyncio
from download_forecast import download_seasonal_forecast_batch
import datetime 
import logging 
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def latest_available_forecast_date(forecast_date: datetime.date) -> datetime.date:
    """
    Returns the latest available forecast date.
    If today is the 6th of the month or later, it returns the current month.
    Otherwise, it returns the previous month.
    """
    if forecast_date.day >= 6:
        return forecast_date
    else:
        # Go to the first day of the current month
        first_day_of_current_month = forecast_date.replace(day=1)
        # Subtract one day to get to the last day of the previous month
        last_day_of_previous_month = first_day_of_current_month - datetime.timedelta(days=1)
        # Return the first day of that previous month
        return last_day_of_previous_month.replace(day=1)



def get_target_month_starts(base_date: datetime.date, num_prior: int, num_after: int) -> list[tuple[str, str, str]]:
    """
    Generates a list of (year_str, month_str, "01") tuples for target months.
    Includes num_prior months before the base_date's month, and num_after months after.
    """
    targets = []
    
    # Generate for prior months
    for i in range(num_prior, 0, -1): # num_prior, num_prior-1, ..., 1
        year = base_date.year
        month = base_date.month - i
        
        while month <= 0:
            month += 12
            year -= 1
        targets.append((str(year), f"{month:02d}", "01"))

    # Generate for after months
    for i in range(1, num_after + 1): # 1, ..., num_after
        year = base_date.year
        month = base_date.month + i

        while month > 12:
            month -= 12
            year += 1
        targets.append((str(year), f"{month:02d}", "01"))
        
    return targets


if __name__ == "__main__":

    try:
        with open('configs/download_config.yaml', 'r') as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError:
        logging.error("Error: configs/download_config.yaml not found.")
        exit(1) 
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configs/download_config.yaml: {e}")
        exit(1)

    # Validate required config keys exist
    required_keys = ['data_path_prediction', 'area', 'variable', 'system', 'originating_centre']
    if not all(key in config for key in required_keys):
        logging.error(f"Config file must contain keys: {', '.join(required_keys)}")
        exit(1)

    seasonal_storage_path = Path(config['data_path_prediction'])
    seasonal_storage_path.mkdir(exist_ok=True, parents=True)

    today = datetime.date.today()
    # Determine the 1st day of the month that is considered "latest available" for forecast issuance
    latest_forecast_issuance_month_start = latest_available_forecast_date(today).replace(day=1)
    
    # Get targets for the 6 months prior to latest_forecast_issuance_month_start
    previous_six_months_targets = get_target_month_starts(
        base_date=latest_forecast_issuance_month_start,
        num_prior=6, # 6 months before latest_forecast_issuance_month_start
        num_after=0  # No months after
    )
    logging.info(f"Identified {len(previous_six_months_targets)} previous months for standard download: {previous_six_months_targets}")

    # Target for the latest month itself (with extended lead time)
    latest_month_target_tuple = (
        str(latest_forecast_issuance_month_start.year),
        f"{latest_forecast_issuance_month_start.month:02d}",
        "01"
    )
    logging.info(f"Identified latest month for extended download: {latest_month_target_tuple}")

    leadtime_hours_standard = [str(i * 24) for i in range(1, 31)] # For 1 to 30 days
    leadtime_hours_extended = [str(i * 24) for i in range(1, 66)] # For 1 to 65 days

    try:
        with open('configs/features_config_30d.yaml', 'r') as model_config_file:
            model_config = yaml.safe_load(model_config_file)
    except FileNotFoundError:
        logging.error("Error: configs/features_config_30d.yaml not found.")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configs/features_config_30d.yaml: {e}")
        exit(1)

    from extract_climate_data import extract_and_save_variables
    all_extracted_paths = [] # Initialize list for successfully downloaded/existing GRIB files
    downloaded_grib_files = [] # Initialize list for successfully downloaded/existing GRIB files

    # --- Download loop for the 6 previous months (standard lead time) ---
    for year_str, month_str, day_str in previous_six_months_targets:
        current_date_tuple = (year_str, month_str, day_str)
        logging.info(f"Processing forecast for {year_str}-{month_str}-{day_str} (standard lead time: 1-30 days)")

        output_filename = f"seasonal_forecast_latest_{year_str}_{month_str}_{day_str}.grib"
        output_filepath = seasonal_storage_path / output_filename

        if output_filepath.exists():
            logging.info(f"File {output_filepath} already exists. Skipping download, will be used for extraction.")
            downloaded_grib_files.append(output_filepath)
        else:
            logging.info(f"Attempting to download {output_filepath} with standard lead time (1-30 days).")
            try:
                asyncio.get_event_loop().run_until_complete(
                    download_seasonal_forecast_batch(
                        dates=[current_date_tuple],
                        leadtime_hours=leadtime_hours_standard, # Standard lead time
                        area=config['area'],
                        variables=config['variable'],
                        system=config['system'],
                        output_filepath=output_filepath,
                        originating_centre=config['originating_centre'],
                        max_workers=config.get('max_workers', 1)
                    )
                )
                logging.info(f"Download process finished for {output_filepath}.")
                if output_filepath.exists(): # Verify file was actually created
                    downloaded_grib_files.append(output_filepath)
                else:
                    logging.error(f"File {output_filepath} was not created after download attempt.")
            except Exception as e:
                logging.error(f"An error occurred during the download process for {output_filepath}: {e}")
    
    # --- Download for the latest month (extended lead time) ---
    year_latest, month_latest, day_latest = latest_month_target_tuple
    current_date_tuple_latest = (year_latest, month_latest, day_latest)
    logging.info(f"Processing forecast for {year_latest}-{month_latest}-{day_latest} (extended lead time: 1-65 days)")

    output_filename_latest = f"seasonal_forecast_latest_{year_latest}_{month_latest}_{day_latest}.grib"
    output_filepath_latest = seasonal_storage_path / output_filename_latest

    if output_filepath_latest.exists():
        logging.info(f"File {output_filepath_latest} already exists. Skipping download, will be used for extraction.")
        downloaded_grib_files.append(output_filepath_latest)
    else:
        logging.info(f"Attempting to download {output_filepath_latest} with extended lead time (1-65 days).")
        try:
            asyncio.get_event_loop().run_until_complete(
                download_seasonal_forecast_batch(
                    dates=[current_date_tuple_latest],
                    leadtime_hours=leadtime_hours_extended, # Extended lead time
                    area=config['area'],
                    variables=config['variable'],
                    system=config['system'],
                    output_filepath=output_filepath_latest,
                    originating_centre=config['originating_centre'],
                    max_workers=config.get('max_workers', 1)
                )
            )
            logging.info(f"Download process finished for {output_filepath_latest}.")
            if output_filepath_latest.exists(): # Verify file was actually created
                downloaded_grib_files.append(output_filepath_latest)
            else:
                logging.error(f"File {output_filepath_latest} was not created after download attempt.")
        except Exception as e:
            logging.error(f"An error occurred during the download process for {output_filepath_latest}: {e}")

    # New extraction logic, after all downloads are complete
    if downloaded_grib_files:
        logging.info(f"Proceeding with consolidated extraction for {len(downloaded_grib_files)} GRIB file(s): {[str(f) for f in downloaded_grib_files]}")
        try:
            # Convert all Path objects in downloaded_grib_files to strings
            string_grib_file_paths = [str(p) for p in downloaded_grib_files]

            # Call extract_and_save_variables ONCE with all downloaded GRIB files
            extracted_paths_result = extract_and_save_variables(
                file_paths=string_grib_file_paths,
                variable_names=model_config['climate_data_params_prediction']['climate_variables'],
                source='ECMWF', # Assuming source is ECMWF based on this script's context
                output_dir= model_config['climate_data_params_prediction']['climate_data_dir'],
            )
            all_extracted_paths = extracted_paths_result # Assign the list of paths
            logging.info(f"Consolidated extraction complete. Saved variables to: {all_extracted_paths}")

        except Exception as e:
            logging.error(f"An error occurred during the consolidated extraction process: {e}")
            logging.error(traceback.format_exc())
            # Re-raise to indicate critical failure in extraction, similar to original behavior
            raise ValueError(f"Critical error during consolidated extraction: {e}")
    else:
        logging.warning("No GRIB files were available or successfully downloaded for extraction in this run.")


    if all_extracted_paths:
        logging.info(f"Summary: All extracted and saved variable paths: {all_extracted_paths}")
    else:
        if downloaded_grib_files: 
             logging.error("Extraction process completed, but no variables were successfully extracted from the consolidated set.")
        else: 
             logging.warning("No data was extracted as no GRIB files were processed.")
            
    logging.info("Data processing complete for the specified months.")
