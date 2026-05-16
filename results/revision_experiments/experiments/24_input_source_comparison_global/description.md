# Input Source Comparison Global

## Purpose
Compares ERA5 and SEAS5/ECMWF source settings, including retrospective upper bound, operational setting, domain shift, and mixed training.

## Source Tables
- `input_source_comparison.csv`
- `input_source_comparison_by_year.csv`
- `era5_feature_schema.csv`
- `era5_seas5_common_schema.csv`

## Notes
- Rows with status completed_available_domain are real ERA5-domain source-comparison runs on sampled case-control rows.
- Rows with blocked/schema status document unavailable full-domain ERA5 parity rather than hiding the attempted experiment.
