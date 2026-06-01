## Experimental Setup
We evaluated all feasible revision experiments on the existing precomputed feature matrix using a fixed chronological split: 2001-2018 for training, 2019-2020 for threshold selection and validation, and 2021-2025 for testing. Binary decision thresholds were selected only on validation by maximizing F1 and then applied unchanged to the test split and all regional/yearly subsets. Metric error columns are estimated with five stratified bootstrap trials over saved predictions for efficient uncertainty summaries.

## Dataset Statistics
The saved dataset uses a detected grid spacing of 0.09999847412109375 deg lat x 0.09999847412109375 deg lon (~11.11 km latitude spacing). Negatives are sampled rather than full-grid negatives, and the feature merge retains land rows according to the configured land/sea mask rule. The global train, validation, and test sample counts are reported in `dataset_statistics.csv`.

## Main Performance
The best completed global test model is `CatBoost` with PR-AUC 0.548 and F1 0.522. The full CatBoost model achieves PR-AUC 0.546 and F1 0.526. Relative to the FWI-only baseline, the full model changes PR-AUC by 0.236; relative to weather-only CatBoost by 0.195; relative to linear logistic by 0.304; and relative to Poisson point-process by 0.095.

## Feature Ablations
The CatBoost ablations quantify data-fusion value by removing or isolating feature sources while keeping the same validation thresholding rule. The ablation plots report absolute PR-AUC/F1 next to the full-minus-variant delta, so positive deltas mean the variant scored lower than the full fused CatBoost model and negative deltas mean the variant scored higher.

## ERA5 / SEAS5
SEAS5/ECMWF -> SEAS5/ECMWF is the clean operationally matched setting available from the existing feature matrix. ERA5->ERA5 would represent a retrospective upper-bound setting, while ERA5->SEAS5 measures input-source domain shift, not simply model quality. The raw ERA5 files are readable, but exact feature-schema parity was blocked by the absence of a precomputed ERA5-derived feature parquet.

## Interpretability
Native CatBoost importance ranks the following features highest: night_light_radiance_2024, past_fire_count_r2_last_year, fire_index_ffmcode_median, fire_index_fwinx_median, fire_index_fdsrte_median. Grouped permutation importance shows the largest PR-AUC drop for `Weather / meteorology history` (0.170), while climate-window permutation is strongest for `90-day climate window` (0.048). These attributions are model explanations, not causal effects.

## Lead Time
Lead-time sensitivity was not directly supported by the saved 30-day aggregate feature matrix because it does not retain forecast lead-time metadata.

## Limitations
Neural embedding/fusion ablations, no-dilation label sensitivity, stricter target-threshold sensitivity, and full ERA5 parity require regenerated intermediate datasets. These blockers are listed explicitly in `failures.md`.
