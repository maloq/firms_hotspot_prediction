# Representative Model Ablation ERA5 To ECMWF No Tp

## Purpose
Compares a compact set of distinct model families on the ERA5-trained, SEAS5/ECMWF-tested no-tp source-shift setting.

## Source Tables
- `representative_model_ablation_era5_to_ecmwf_no_tp.csv`
- `representative_model_ablation_era5_to_ecmwf_no_tp_by_year.csv`

## Notes
- The model set is intentionally limited to representative families: boosted trees, bagged trees, linear classification, point-process GLM, and the best neural model.
- All rows use ERA5 validation thresholds and SEAS5/ECMWF test rows with `tp` removed.
