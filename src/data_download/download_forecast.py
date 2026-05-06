import asyncio
import datetime
import logging
import sys
from pathlib import Path
import os
from dateutil.relativedelta import relativedelta
import cdsapi
from yaml import safe_load
import argparse
sys.path.append(os.getcwd())

def get_cdsapi_client():
    """Get an instance of the CDS API client.

    Returns:
        cdsapi.Client: An instance of the CDS API client.

    """
    return cdsapi.Client()


async def download_seasonal_forecast_async(client, dataset, request, output_file, retries=3):
    """Asynchronous function to download Seasonal Forecast data with retry logic.

    Args:
        client (cdsapi.Client): The CDS API client.
        dataset (str): The dataset to retrieve.
        request (dict): The request parameters.
        output_file (str): The output file path.
        retries (int): The number of retries for failed downloads.

    """
    output_file_str = str(output_file) # CDS API client expects a string path
    for attempt in range(retries):
        try:
            # Log the start of the download attempt with key request details
            logging.info(
                f"Starting async download for dataset: {dataset}, request: "
                f"{{'area': {request['area']}, 'variable': {request['variable']}, 'year': {request['year'][0]}, 'month': {request['month'][0]}, 'day': {request['day'][0]}}}, "
                f"target: {output_file_str}, attempt: {attempt + 1}"
            )
            # Ensure output directory exists just before download attempt
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(client.retrieve, dataset, request, output_file_str)
            logging.info(f"Successfully downloaded {output_file_str}")
            return output_file_str # Return the file path on success

        except Exception as e:
            # Log specific errors encountered during download attempt
            logging.error(
                f"Failed to download for {request['year'][0]}-{request['month'][0]}, target: {output_file_str}, attempt {attempt + 1}: {e}"
            )
        # Retry logic
        if attempt < retries - 1:
            await asyncio.sleep(5) # Add a small delay before retrying
            logging.info(
                f"Retrying download for {request['year'][0]}-{request['month'][0]}, target: {output_file_str}, attempt {attempt + 2}"
            )
        else:
            # Log failure after all retries are exhausted
            logging.error(f"All retry attempts failed for {request['year'][0]}-{request['month'][0]}, target: {output_file_str}")
            # Optionally, try to remove the potentially incomplete file
            try:
                if Path(output_file).exists():
                    os.remove(output_file_str)
                    logging.info(f"Removed potentially incomplete file: {output_file_str}")
            except OSError as rm_err:
                logging.error(f"Error removing incomplete file {output_file_str}: {rm_err}")

    return None # Indicate download failure


# Define a wrapper function to manage the semaphore for each download task
async def download_with_semaphore(semaphore, client, dataset, request, output_file, retries=3):
    """Acquires semaphore before downloading."""
    async with semaphore:
        # Call the original download function once the semaphore is acquired and
        # return its result so the caller can determine success or failure.
        return await download_seasonal_forecast_async(
            client, dataset, request, output_file, retries
        )


def build_seasonal_forecast_request_template(year, month, day, system, variables, leadtime_hours, area, originating_centre):
    """Build the request template for Seasonal Forecast data from the ECMWF.

    Args:
        year (str): The year for which to build the request.
        month (str): The month for which to build the request.
        day (str): The day for which to build the request.
        system (str): The forecast system to use (e.g., SEAS5).
        variables (list of str): The list of variables to request.
        leadtime_hours (list of str): The list of lead time hours to request.
        area (list of float): The geographical area of interest defined as [north, west, south, east].

    Returns:
        dict: The request template populated with the specified parameters.

    """
    # Ensure parameters are strings and wrapped in lists as expected by the API
    return {
        "originating_centre": originating_centre,
        "system": system,
        "variable": variables, # Assumes variables is already a list of strings
        "year": [str(year)],
        "month": [str(month)],
        "day": [str(day)], # Day should be 'DD' format string, put in a list
        "leadtime_hour": leadtime_hours, # Assumes leadtime_hours is already a list of strings
        "format": "grib",
        "area": area, # Assumes area is already a list of floats/numbers
    }


async def download_seasonal_forecast_batch(
    dates: list[tuple[str, str, str]], # Expecting tuples of strings (year, month, day)
    system: str,
    leadtime_hours: list[str],
    area: list[float],
    variables: list[str],
    originating_centre: str,
    max_workers: int = 1,
    output_filepath: Path | None = None, # Added parameter for specific output file path
) -> None:
    """Asynchronously download Seasonal Forecast data for specified dates.

    If output_filepath is provided, assumes a single forecast start date in 'dates'
    and downloads to that specific file path.

    Args:
        dates (list of tuples): List of (year, month, day) tuples (expects strings).
                                  For 'latest' mode, should contain only one tuple.
        system (str): Forecast system (e.g., 'seas5').
        leadtime_hours (list of str): List of lead time hours (as strings).
        area (list of float): Geographical area [north, west, south, east].
        variables (list of str): List of variable names.
        originating_centre (str): Originating centre (e.g., 'ecmwf').
        max_workers (int): Maximum number of concurrent downloads.
        output_filepath (Path | None, optional): Specific full path for the output file.
                                                  Required when downloading a single 'latest' forecast. Defaults to None.
    """
    # Check if output_filepath is provided (required for the 'latest' download scenario)
    if not output_filepath:
        logging.error("Output filepath must be provided for this operation.")
        # Or consider raising an error: raise ValueError("Output filepath must be provided.")
        return

    # Validate that 'dates' contains exactly one entry when output_filepath is specified
    if len(dates) != 1:
        logging.warning(
            f"Expected exactly one date in 'dates' when output_filepath is specified, but got {len(dates)}. Using the first date: {dates[0]}"
        )
        # Optionally raise an error instead of proceeding:
        # raise ValueError("Expected exactly one start date when output_filepath is specified.")
    
    if not dates:
        logging.error("No dates provided for download.")
        return

    client = get_cdsapi_client()
    tasks = []
    semaphore = asyncio.Semaphore(max_workers)

    # Use the single date provided in the list
    year, month, day = dates[0]

    # Build the request using the single date tuple
    request = build_seasonal_forecast_request_template(
        year, month, day, system, variables, leadtime_hours, area, originating_centre=originating_centre
    )
    logging.info(f"Built request: {request}") # Log the constructed request

    # Ensure the parent directory for the output file exists
    try:
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {output_filepath.parent}: {e}")
        return # Cannot proceed if directory creation fails

    # Create the download task using the semaphore wrapper
    tasks.append(
        download_with_semaphore(
            semaphore, client, "seasonal-original-single-levels", request, output_filepath # Pass the Path object
        )
    )

    # Run the download task(s) concurrently (just one task in this case)
    logging.info(f"Starting download task for {output_filepath}...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle and log any exceptions caught by asyncio.gather
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Specific error is logged within download_seasonal_forecast_async
            logging.error(f"Download task {i+1} failed. Check previous logs for details: {result}")
        # else: # Optional success confirmation
        #    logging.info(f"Download task {i+1} completed.")


# Helper function to calculate past dates (assuming forecast day is '01')
def get_last_n_months_dates(n_months: int) -> list[tuple[str, str, str]]:
    """Calculates the start dates (YYYY, MM, DD='01') for the last n months."""
    dates = []
    # Use current date to find the start of the current month
    current_month_start = datetime.date.today().replace(day=1)

    for i in range(n_months):
        # Calculate the start date for each of the past n months
        target_date = current_month_start - relativedelta(months=i)
        year = str(target_date.year)
        month = f"{target_date.month:02d}" # Pad month with leading zero
        day = "01" # Assuming forecasts are issued on the 1st
        dates.append((year, month, day))

    # Return in chronological order (oldest first)
    return dates[::-1]


# New function to download data for the last N months
async def download_last_n_months_seasonal_forecast(
    n_months: int,
    system: str,
    leadtime_hours: list[str],
    area: list[float],
    variables: list[str],
    base_output_dir: Path,
    max_workers: int = 4,
    originating_centre: str = "ecmwf",
    retries: int = 3,
) -> None:
    """
    Asynchronously downloads Seasonal Forecast data for the last N months.

    Downloads data starting from the 1st of each of the last N months
    (including the current month) into a structured directory.

    Args:
        n_months (int): Number of past months to download data for.
        system (str): Forecast system (e.g., 'seas5').
        leadtime_hours (list of str): List of lead time hours (as strings).
        area (list of float): Geographical area [north, west, south, east].
        variables (list of str): List of variable names.
        originating_centre (str): Originating centre (e.g., 'ecmwf').
        base_output_dir (Path): The base directory to save downloaded files.
                                Files will be saved under base_output_dir/system/year/month/.
        max_workers (int): Maximum number of concurrent downloads.
        retries (int): Number of download retries for each file.
    """
    if n_months <= 0:
        logging.error("Number of months (n_months) must be positive.")
        return

    # Calculate the start dates for the last N months
    dates = get_last_n_months_dates(n_months)
    if not dates:
        logging.warning("No dates generated for download.")
        return

    logging.info(f"Planning to download data for the following start dates: {dates}")

    client = get_cdsapi_client()
    tasks = []
    semaphore = asyncio.Semaphore(max_workers)
    dataset = "seasonal-original-single-levels" 
    for year, month, day in dates:
        output_dir = base_output_dir
        output_filename = f"forecast_{originating_centre}_{year}_{month}.grib"
        output_filepath = output_dir / output_filename

        # Build the request for this specific date
        request = build_seasonal_forecast_request_template(
            year, month, day, system, variables, leadtime_hours, area, originating_centre
        )

        # Log the request being prepared
        logging.debug(f"Preparing download task for {output_filepath} with request: {request}")

        # Create the download task using the semaphore wrapper
        # Pass the number of retries to the download function
        tasks.append(
            download_with_semaphore(
                semaphore, client, dataset, request, output_filepath, retries=retries
            )
        )

    logging.info(f"Starting {len(tasks)} download tasks with max {max_workers} concurrent workers...")
    # Run tasks concurrently and collect results/exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and log summary
    successful_downloads = 0
    failed_downloads = 0
    for i, result in enumerate(results):
        date_str = f"{dates[i][0]}-{dates[i][1]}-{dates[i][2]}"
        if isinstance(result, Exception):
            # Error logged within download_seasonal_forecast_async
            logging.error(f"Download task for date {date_str} failed with an exception. See previous logs.")
            failed_downloads += 1
        elif result is None: # download_seasonal_forecast_async returns None on failure after retries
             logging.error(f"Download task for date {date_str} failed after all retries. See previous logs.")
             failed_downloads += 1
        else:
            # Success is logged within download_seasonal_forecast_async
            # logging.info(f"Download task for date {date_str} completed successfully.")
            successful_downloads += 1

    logging.info(f"Batch download summary: {successful_downloads} successful, {failed_downloads} failed.")


# New function to download data for specific historical dates from config
async def download_historical_seasonal_forecast(
    dates: list[tuple[str, str, str]], # Expecting tuples of strings (year, month, day)
    system: str,
    leadtime_hours: list[str],
    area: list[float],
    variables: list[str],
    base_output_dir: Path,
    max_workers: int = 4,
    originating_centre: str = "ecmwf",
    retries: int = 3,
) -> None:
    """
    Asynchronously downloads Seasonal Forecast data for specific dates.

    Downloads data for each date specified in the 'dates' list
    into a structured directory (base_output_dir/system/year/month/).

    Args:
        dates (list of tuples): List of (year, month, day) tuples to download.
        system (str): Forecast system (e.g., 'seas5').
        leadtime_hours (list of str): List of lead time hours (as strings).
        area (list of float): Geographical area [north, west, south, east].
        variables (list of str): List of variable names.
        base_output_dir (Path): The base directory to save downloaded files.
        max_workers (int): Maximum number of concurrent downloads.
        originating_centre (str): Originating centre (e.g., 'ecmwf').
        retries (int): Number of download retries for each file.
    """
    if not dates:
        logging.warning("No dates provided for historical download.")
        return

    logging.info(f"Planning to download historical data for {len(dates)} specific dates.")

    client = get_cdsapi_client()
    tasks = []
    semaphore = asyncio.Semaphore(max_workers)
    dataset = "seasonal-original-single-levels" # Standard dataset for this type

    for year, month, day in dates:
        # Define the output path for this specific date
        output_dir = base_output_dir
        # Example filename structure: YYYYMMDD_system_forecast.grib
        output_filename = f"{originating_centre}_{year}_{month}_forecast.grib"
        output_filepath = output_dir / output_filename

        # Build the request for this specific date
        request = build_seasonal_forecast_request_template(
            year, month, day, system, variables, leadtime_hours, area, originating_centre
        )

        # Log the request being prepared
        logging.debug(f"Preparing download task for {output_filepath} with request: {request}")

        # Create the download task using the semaphore wrapper
        tasks.append(
            download_with_semaphore(
                semaphore, client, dataset, request, output_filepath, retries=retries
            )
        )

    logging.info(f"Starting {len(tasks)} download tasks with max {max_workers} concurrent workers...")
    # Run tasks concurrently and collect results/exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and log summary
    successful_downloads = 0
    failed_downloads = 0
    for i, result in enumerate(results):
        date_str = f"{dates[i][0]}-{dates[i][1]}-{dates[i][2]}"
        if isinstance(result, Exception):
            logging.error(f"Download task for date {date_str} failed with an exception. See previous logs.")
            failed_downloads += 1
        elif result is None: # download_seasonal_forecast_async returns None on failure after retries
             logging.error(f"Download task for date {date_str} failed after all retries. See previous logs.")
             failed_downloads += 1
        else:
            successful_downloads += 1

    logging.info(f"Historical batch download summary: {successful_downloads} successful, {failed_downloads} failed.")


async def main(args):
     """Main execution function controlled by command-line arguments."""
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

     # Load configuration from YAML file
     try:
         with open(args.config, 'r') as f:
             config = safe_load(f)
         logging.info(f"Loaded configuration from {args.config}")
     except FileNotFoundError:
         logging.error(f"Configuration file not found: {args.config}")
         return
     except Exception as e:
         logging.error(f"Error loading configuration file {args.config}: {e}")
         return

     # --- Extract common parameters from config ---
     # Use get() with defaults for robustness
     forecast_system = config.get("system", "seas5") # Example default system
     # Ensure leadtime_hours is present in config or handle appropriately
     lead_times = config.get("leadtime_hours")
     originating_centre = config.get("originating_centre", "ecmwf")
     if lead_times is None:
          logging.error("Missing 'leadtime_hours' in configuration file.")
          # Example: Provide a default if appropriate, or exit
          # lead_times = ["24", "48", "72", "96", "120", "144", "168"] # Example default
          # logging.warning("Using default lead times: {lead_times}")
          return # Exit if lead times are essential and not provided

     download_area = config.get("area")
     if download_area is None:
         logging.error("Missing 'area' in configuration file.")
         return
     variables_to_download = config.get("variable")
     if variables_to_download is None:
         logging.error("Missing 'variable' in configuration file.")
         return
     base_output_dir = Path(config.get("data_path", "data/raw_climate/ECMWF")) # Default output path

     # --- Optional parameters (can be overridden or added to config) ---
     concurrent_downloads = config.get("max_workers", 5)
     download_retries = config.get("retries", 3) 

     # --- Determine download mode ---
     if args.last_n_months is not None and args.last_n_months > 0:
         # --- Download Last N Months Mode ---
         logging.info(f"Starting download for the last {args.last_n_months} months.")
         await download_last_n_months_seasonal_forecast(
             n_months=args.last_n_months,
             system=forecast_system,
             leadtime_hours=lead_times,
             area=download_area,
             variables=variables_to_download,
             base_output_dir=base_output_dir,
             max_workers=concurrent_downloads,
             originating_centre=originating_centre,
             retries=download_retries,
         )
     else:
         # --- Download Historical Dates from Config Mode ---
         logging.info("Starting download based on dates specified in the configuration file.")
         # Extract dates from config
         years = config.get("year", [])
         months = config.get("month", [])
         days = config.get("day", ["01"]) # Default to day '01' if not specified

         if not years or not months:
             logging.error("Missing 'year' or 'month' lists in configuration file for historical download.")
             return

         # Generate list of date tuples (YYYY, MM, DD)
         historical_dates = []
         for year in years:
             for month in months:
                 for day in days: # Iterate through specified days
                     historical_dates.append((str(year), f"{int(month):02d}", f"{int(day):02d}"))

         if not historical_dates:
             logging.warning("No valid historical dates generated from config.")
             return

         await download_historical_seasonal_forecast(
             dates=historical_dates,
             system=forecast_system,
             leadtime_hours=lead_times,
             area=download_area,
             variables=variables_to_download,
             base_output_dir=base_output_dir,
             max_workers=concurrent_downloads,
             retries=download_retries,
             originating_centre=originating_centre,
         )


if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Download ECMWF Seasonal Forecast Data.")
     parser.add_argument(
         "--config",
         type=str,
         default="configs/download_config.yaml",
         help="Path to the YAML configuration file."
     )
     parser.add_argument(
         "--last-n-months",
         type=int,
         help="Download data for the last N months (overrides dates in config)"
     )

     parsed_args = parser.parse_args()

     asyncio.run(main(parsed_args))
