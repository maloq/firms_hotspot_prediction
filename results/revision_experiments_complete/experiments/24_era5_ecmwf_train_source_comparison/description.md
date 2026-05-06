# ERA5 vs ECMWF Training Source Comparison

## Purpose
This experiment compares three CatBoost training sources while holding validation and test fixed to the same ECMWF/SEAS5 matrix:

- ECMWF train -> ECMWF validation/test.
- ERA5 train -> ECMWF validation/test.
- ERA5 + ECMWF duplicated train rows -> ECMWF validation/test.

Thresholds are selected only on ECMWF validation years 2019-2020 and then applied unchanged to the ECMWF 2021-2025 test rows.

## Common Footprint
- Latitude: `35.00` to `75.00`.
- Longitude: `10.00` to `179.00`.
- ERA5 processed time range: `2009-01-01` to `2018-12-31`.
- Training interval with 128-day lookback support: `2009-05-09` to `2018-12-31`.

## ERA5 Source
ERA5 climate features were generated from processed zarr files under `/home/ids/vmorozov/data/climate_data/climate_features/ERA5`. The raw ERA5 directory `/home/ids/vmorozov/era5` was audited and is saved in `artifacts/raw_era5_audit.json`. The conda-run raw GRIB audit could not find `grib_ls`; processed ERA5 zarrs were used for the reproducible training matrix.

## Folder Layout
- `tables/`: readable CSV tables, at most six columns each.
- `plots/png/`: high-resolution PNG plots.
- `plots/pdf/`: PDF plots.
- `artifacts/`: ERA5 training parquet, raw metrics JSONL, models, predictions, schema, logs, and environment metadata.
