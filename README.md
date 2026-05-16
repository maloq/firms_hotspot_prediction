# Wildfire Prediction System

This project provides a system for historical analysis and forecasting of wildfire risks using machine learning models and climate data.

## Installation

### Prerequisites
- Python 3.9+
- CDS API account (for accessing ECMWF climate data)
- Access to the DWD ICON open data server (for ICON forecasts)


## Data Structure

The project expects the following data structure:

```
data/
├── climate_data_files/
│   ├── ECMWF/           # ECMWF seasonal forecast climate data
│   │   ├── t2m/         # 2m temperature data
│   │   ├── tp/          # Total precipitation data
│   │   ├── stl1/          # Soil temperature at 1m depth
│   │   └── d2m/         # 2m dewpoint temperature data
│   └── ECMWF_prediction/ # ECMWF prediction data
├── land_features/       # Static geographical features
│   ├── GMTED2010_15n015_00625deg.nc     # Elevation data
│   ├── forest_data.nc                   # Forest cover data
│   ├── topography.nc                    
│   ├── IMERG_land_sea_mask.nc             
│   ├── kontur_population_20231101_r6.gpkg # Population density data
│   └── type_of_high_vegetation_0_daily-mean.nc 
│   └── type_of_low_vegetation_0_daily-mean.nc 
│   └── soil_type_stream-oper_daily-mean.nc
│   └── fire_index_features.npz    
├── modis/               # Fire detection data from MODIS (by year)
│   ├── 2000/            # Fire data for year 2000
│   ├── 2001/            # Fire data for year 2001
│   ├── ...              
│   ├── 2024/            # Most recent fire data
├── countries/           # Country boundary data
├── wwf_terr_ecos/       # WWF Terrestrial Ecoregions data
│   ├── wwf_terr_ecos.shp               # Ecoregions shapefile
│   ├── wwf_terr_ecos.dbf               # Ecoregions database
│   └── wwf_terr_ecos.*                 # Associated shapefile components
├── WeightsAndGrid/      # Grid and interpolation weights
│   ├── icon_weights.nc                 # ICON model interpolation weights
│   └── grid_world_0125.txt             # Global grid definition
├── raw_climate/         # Raw downloaded climate data before processing
│   └── [climate data files before extraction]
├── saved_features/      # Generated features saved for reuse
    └── train_test_features_all_30d.parquet # Pre-computed feature datasets
```

**Note**: The `outputs/` directory will be created automatically when running predictions and will contain:
```
outputs/             # Prediction outputs (created during execution)
└── forecast_run_30d/
    ├── forecast_netcdf/  # NetCDF prediction files
    └── forecast_plots/   # Visualization plots
```


## Usage

Revision evaluation keeps deployment-like calibrated full-grid testing in its
own full-grid study, while the main model-comparison tables remain sampled
case-control diagnostics with random-error estimates. See
[`docs/revision_evaluation_calibrated.md`](docs/revision_evaluation_calibrated.md)
for the split, calibration, and count-interpretation details.

### Step 1: Download Historical Data

To download historical climate data:

```bash
python src/data_download/download_forecast.py --config configs/download_config.yaml
```
To download last N months

```bash
python src/data_download/download_forecast.py --config configs/download_config.yaml --last-n-months 7
```

Extract the downloaded data:

```bash
python src/data_download/extract_climate_data.py
```

### Step 2: Download Latest Forecast Data

For generating a current forecast, download the latest ECMWF and ICON data:

```bash
# ECMWF seasonal forecasts
python src/data_download/download_and_extract_latest.py
```

### Step 3: Run Prediction Pipeline

The main script supports both historical analysis and forecasting:

```bash
# For a 30-day forecast using default configuration
python prediction_pipeline.py --config configs/features_config_30d.yaml --days 30 --output outputs/forecast_run_30d
```

Command-line options:
- `--config`: Path to the configuration file
- `--days`: Number of days to forecast (1-30)
- `--output`: Output directory for forecast results
- `--threshold`: Probability threshold for binary classification (default: 0.9)


## Configuration Files

### Main Configuration

Primary configuration files available:

- `configs/features_config_30d.yaml`: Configuration for 30-day forecasts
- `configs/target_config.yaml`: Target generation configuration
- `configs/download_config.yaml`: Climate data download configuration
- `configs/selected_columns_30d.txt`: Selected feature columns for 30-day models
- `configs/selected_climate.txt`: Selected climate variables for 30-day models, subset of variables in `configs/features_config_30d.yaml`



Example configuration file (e.g., `configs/features_config_30d.yaml`):

```yaml
# Prediction parameters
prediction_countries:
  - 'Russian_Federation'
model_path: "models/catboost_fire_model_30d.cbm"
outputs_dir: "outputs/test"

# Feature generation configuration
modis_data_path: 'data/modis/'
coordinate_bounds: [40, 70, 100, 140]  # [min_lat, max_lat, min_lon, max_lon]

climate_data_params:
  climate_data_dir: "data/climate_data_files/ECMWF"
  climate_variables: ["t2m", "d2m", "tp"]
  n_days: 128
```

### Download Configuration

For climate data download (`configs/download_config.yaml`):

```yaml
# Data sources, variables, and parameters for downloading 
dataset: "seasonal-original-single-levels"
originating_centre: "ecmwf"
system: "51"
variable:
  - "2m_dewpoint_temperature"
  - "2m_temperature"
  - "total_precipitation"
area: [35, 75, 25, 179]  # [north, west, south, east]
```

## Outputs

The system produces the following outputs:

1. **NetCDF Files**: Gridded probability maps for each forecast day
2. **Visualization Plots**: Heatmaps showing fire probability across the region
3. **Data Files**: Feature datasets and prediction results (Parquet format)

## Train

To train a new model, first run the script to create the training data:

```bash
python make_train_data.py
```
If you have large RAM(more than 300GB) and big CPU, you can tweak dask Client in `src/feature_generation/prepare_climate_data.py` in function `_get_dask_client`

Then use notebook to train the model `train_test.ipynb`

You can also train the CatBoost model directly with validation-based early stopping:

```bash
python train_catboost.py \
  --validation-start-date 2019-01-01 \
  --test-start-date 2022-01-01 \
  --iterations 2000 \
  --early-stopping-rounds 150 \
  --no-feature-importance-analysis
```

The script keeps the latest dates as test data, uses the validation period for
`use_best_model`/early stopping, and tunes the binary probability threshold on
validation for the reported precision/recall/F1 metrics.


## Models

Pre-trained models should be placed in the `models/` directory. The default configuration expects:
- `models/catboost_fire_model_30d.cbm`: CatBoost model for 30-day forecasts

## Data Sources

The project integrates multiple data sources:

- **Climate Data**: ECMWF seasonal forecasts for temperature, precipitation, and dewpoint
- **Fire Detection**: MODIS FIRMS data organized by year (2000-2024)
- **Geographic Features**: 
  - Elevation data from GMTED2010
  - Forest cover data
  - Population density from Kontur
  - Land/sea mask from IMERG
- **Administrative Boundaries**: Natural Earth country boundaries
- **Ecoregions**: WWF Terrestrial Ecoregions dataset
