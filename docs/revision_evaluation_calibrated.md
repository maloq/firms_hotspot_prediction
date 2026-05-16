# Revision Evaluation: Calibrated Deployment-Grid Testing

Model fitting still uses random case-control negative sampling to reduce computational cost and to keep enough positive examples in each training run. That sampling changes the class prevalence, so raw CatBoost margins, neural-network logits, and sampled-distribution probabilities are not treated as deployment probabilities.

The revision evaluation suite now separates two evaluation roles:

- Calibrated full-grid study: deployment-like full-grid or weighted-grid calibrated testing under `src/revision_evaluation`. A model predicts raw scores on a held-out calibration grid, a model-specific post-hoc calibrator is fit there, and calibrated probability/ranking/count/spatial diagnostics are reported on a separate held-out test grid.
- Legacy evaluation: the old undersampled-negative/case-control validation and test metrics are still saved for comparison with older runs.

Primary outputs are written under:

- `primary_full_grid_calibrated/model_comparison.csv`
- `primary_full_grid_calibrated/probability_metrics.csv`
- `primary_full_grid_calibrated/reliability_bins.csv`
- `primary_full_grid_calibrated/monthly_count_calibration.csv`
- `primary_full_grid_calibrated/country_count_calibration.csv`
- `primary_full_grid_calibrated/prevalence_audit.csv`
- `primary_full_grid_calibrated/risk_concentration.csv`
- `primary_full_grid_calibrated/count_correction.csv`
- `primary_full_grid_calibrated/spatial_scale_evaluation.csv`

Root tables `main_model_comparison.csv` and `main_model_comparison_by_year.csv` remain sampled/case-control model diagnostics with random-error estimates. Full-grid calibrated metrics are intentionally kept out of those model-comparison, ablation, neural, label, lead-time, and input-source experiment folders; the organized report studies them only under `experiments/04_primary_full_grid_calibrated/`.

The modeled target should be described as the probability of a MODIS/FIRMS-derived fire-positive grid cell-day. It should not be described as the probability of a true ignition unless a separate unique-ignition extraction step is implemented. Similarly, summing calibrated probabilities over the grid estimates the expected number of fire-positive grid cell-days under the configured label definition; if labels are spatially expanded or dilated, that sum is not the number of unique real fires.

Calibration uses held-out deployment-like rows with valid `eval_weight`. The sampled training prevalence is not used as the implied real fire probability. The full-grid study is organized as five checks: prevalence audit, sampled-vs-full-grid metric contrast, risk concentration, raw-vs-calibrated count correction, and spatial-scale evaluation.
