# Calibrated Full-Grid Prediction Study

## Purpose
Keeps deployment-grid calibrated prediction in one place: prevalence audit, sampled-vs-full-grid contrast, risk concentration, count calibration, and spatial-scale evaluation.

## Source Tables
- `primary_full_grid_calibrated/model_comparison.csv`
- `primary_full_grid_calibrated/probability_metrics.csv`
- `primary_full_grid_calibrated/reliability_bins.csv`
- `primary_full_grid_calibrated/monthly_count_calibration.csv`
- `primary_full_grid_calibrated/country_count_calibration.csv`
- `primary_full_grid_calibrated/prevalence_audit.csv`
- `primary_full_grid_calibrated/risk_concentration.csv`
- `primary_full_grid_calibrated/count_correction.csv`
- `primary_full_grid_calibrated/spatial_scale_evaluation.csv`

## Notes
- Classification F1 is expected to be tiny at the deployment base rate; risk-ranking and count-calibration tables carry the operational interpretation.
- The target is a MODIS/FIRMS-derived fire-positive grid cell-day, not a unique ignition event.
- Full-grid rows are not copied into the sampled model-comparison experiments.
