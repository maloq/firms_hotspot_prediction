# Full-grid fire prediction improvements summary

Date: 2026-06-01

## What changed

- Full-grid calibrated evaluation is now the model-selection target. Selection artifacts are written to `primary_full_grid_calibrated/model_selection.csv` and `best_model.json`.
- Reliability was added as scalar metrics: `reliability_ece`, `reliability_mce`, and `reliability_rmse`.
- Calendar context features were added to training and deployment features: weekend today/yesterday/or and holiday today/yesterday/or.
- A deployment-distribution CatBoost experiment was added using capped square-root `sample_weight`.
- The spatial TSN neural model now has a dense full-grid evaluator; the smoke run completed with real full-grid predictions.
- Summary-feature neural MLPs are now true full-grid-evaluable: dense neural evaluation passes dynamic weather columns through, and NN training can optionally consume prepared deployment `sample_weight` with power/cap/normalization.
- Daily-spatial neural evaluation now uses the model metadata climate source first, so the ERA5 no-TP model is evaluated on ERA5 rather than the generic feature-config source. Dense daily-spatial tensor extraction was optimized with full-slab climate reads and exact daily timestamp indexing.
- RF best-model reruns are reproducible through YAML subset/parameter controls.

## Main bounded full-grid experiment

Config: `configs/revision_evaluation_full_grid_improvements.yaml`

Scope: Russian Federation, 2021-05 calibration, 2022-05 test, bounded to `[50.0, 80.0, 70.0, 140.0]` because the initial unbounded all-Russia run exceeded available climate coverage.

| Model | Full-grid AP | ROC-AUC | Reliability ECE | Expected/Observed count |
| --- | ---: | ---: | ---: | ---: |
| Spatial climate TSN-MLP no tp (ERA5 global full) | 0.001291 | 0.939580 | 0.000079 | 0.339118 |
| Random Forest | 0.000495 | 0.818653 | 0.000083 | 0.196999 |
| Logistic Regression | 0.000355 | 0.854597 | 0.276278 | 2777.237317 |
| FWI-only CatBoost | 0.000308 | 0.753046 | 0.000094 | 0.196078 |
| CatBoost | 0.000195 | 0.743569 | 0.000084 | 0.191871 |
| Weather-only CatBoost | 0.000163 | 0.661115 | 0.000089 | 0.193439 |
| CatBoost deployment-weighted | 0.000102 | 0.466352 | 0.000086 | 0.196038 |
| Poisson point-process GLM | 0.000100 | 0.243479 | 0.000120 | 0.210541 |

Decision: select the Spatial climate TSN-MLP no-TP ERA5 model as the current full-grid AP model. It improves AP by about 161% over the Random Forest and about 562% over the full CatBoost baseline, while also improving ROC-AUC and reliability ECE. It still underpredicts total event count, but less severely than the Random Forest.

## Deployment-distribution training

The deployment-weighted CatBoost did not improve the selected metric:

- CatBoost AP: `0.000195`
- Deployment-weighted CatBoost AP: `0.000102`

It slightly moved the expected/observed count ratio toward 1, but the ranking loss is too large under the current full-grid AP target. Do not adopt this variant as the default.

## Neural MLP optimization

Configs:

- `configs/nn_global_full_minimal_mlp_fullgrid_opt.yaml`
- `configs/nn_global_full_minimal_mlp_fullgrid_rank_opt.yaml`
- `configs/revision_evaluation_neural_full_grid_mlp_opt.yaml`
- `configs/revision_evaluation_neural_full_grid_mlp_rank_opt.yaml`

Output: `results/neural_full_grid_mlp_opt_bounded/primary_full_grid_calibrated`

All rows below are true calibrated full-grid test metrics on the same bounded May 2022 grid as the tabular experiment.

| Neural model | Full-grid AP | ROC-AUC | Reliability ECE | Expected/Observed count | Sampled val AP | Sampled test AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Minimal MLP full-grid rank optimized | 0.000320 | 0.838358 | 0.000079 | 0.216877 | 0.400532 | 0.452981 |
| Minimal MLP baseline | 0.000318 | 0.844416 | 0.000078 | 0.233539 | 0.399058 | 0.470244 |
| Minimal MLP deployment-weighted/focal | 0.000313 | 0.835645 | 0.000085 | 0.193730 | 0.404928 | 0.457224 |

Decision: select `Minimal MLP full-grid rank optimized` as the best summary-feature MLP under the full-grid AP target. It is a small global AP lift over the old MLP baseline, but it is superseded by the daily-spatial TSN model on the proper ERA5 full-grid comparison.

The deployment-weighted neural variant improved count calibration and regional AP in several regions, especially Far East, but it lost global AP. Keep it as an experiment, not the default neural model.

## Best-model rerun

Config: `configs/revision_evaluation_rf_best_rerun.yaml`

The stronger RF rerun used 300k training rows, latitude/longitude enabled, 180 trees, depth 22, min leaf 10, positive class weight 6.

| Model | Full-grid AP | ROC-AUC | Reliability ECE | Expected/Observed count |
| --- | ---: | ---: | ---: | ---: |
| Selected RF rerun | 0.000344 | 0.845371 | 0.000071 | 0.305453 |
| Original selected RF | 0.000495 | 0.818653 | 0.000083 | 0.196999 |

Decision: do not replace the selected AP model. The rerun improved ROC-AUC, reliability ECE, and count calibration, but AP decreased.

## Neural full-grid evaluation

Configs:

- `configs/revision_evaluation_neural_full_grid_smoke.yaml`
- `configs/revision_evaluation_spatial_neural_full_grid_proper.yaml`

The spatial TSN no-TP model completed a true unsampled full-grid evaluation on a small bounded grid/window with positives in both calibration and test. This verifies the dense neural full-grid adapter and writes normal full-grid artifacts.

Global smoke metrics:

- AP: `0.001577`
- ROC-AUC: `0.533203`
- Reliability ECE: `0.001392`
- Expected/Observed count ratio: `0.848994`

The proper bounded May full-grid run now also completes on the same grid as the tabular comparison:

| Neural model | Region | Full-grid AP | ROC-AUC | Reliability ECE | Expected/Observed count |
| --- | --- | ---: | ---: | ---: | ---: |
| Spatial climate TSN-MLP no tp (ERA5 global full) | Global | 0.001291 | 0.939580 | 0.000079 | 0.339118 |
| Spatial climate TSN-MLP no tp (ERA5 global full) | Eastern Siberia | 0.001274 | 0.940101 | 0.000070 | 0.358834 |
| Spatial climate TSN-MLP no tp (ERA5 global full) | Far East | 0.023925 | 0.995727 | 0.000202 | 0.083435 |
| Spatial climate TSN-MLP no tp (ERA5 global full) | Central Asia | 0.000734 | 0.788758 | 0.000068 | 0.728403 |

Optimization notes:

- `scripts/build_prepared_nn_data.py` now uses a full-slab fast path for daily-spatial climate extraction when the requested deployment chunk fits under the configured slab-cell limit.
- `src/feature_generation/prepare_climate_data.py` now uses exact daily `isel` indexing when requested dates already exist in the climate data, avoiding unnecessary interpolation.
- `src/revision_evaluation/probability_overlays.py` now prefers the trained model metadata `climate_data_dir` before the generic feature config. This fixed the ERA5 no-TP model comparison.
- `configs/revision_evaluation_spatial_neural_full_grid_proper.yaml` runs the spatial TSN against the main `results/full_grid_improvements_bounded` output so `model_selection.csv` and `best_model.json` compare neural and tabular models directly.

## Tests

Command:

```bash
conda run -n pointnet pytest -q tests/test_calendar_features.py tests/test_probability_metrics.py tests/test_neural_full_grid.py tests/test_neural_training_sample_weight.py tests/test_tabular_sample_weight.py tests/test_revision_evaluation_calibrated.py
```

Result: `32 passed, 4 warnings`.

## Research notes

- Selecting and tuning on the deployment-like full grid is aligned with covariate-shift guidance: standard validation can be biased when train/test input distributions differ; importance-weighted validation/training is one remedy, but should be validated against the target distribution. Source: Sugiyama et al., 2007, JMLR, https://www.jmlr.org/beta/papers/v8/sugiyama07a.html
- Spatial/temporal dependence can make random or case-control validation over-optimistic for mapped ecological predictions, supporting full-grid and structured validation. Source: Roberts et al., 2017, Ecography, https://doi.org/10.1111/ecog.02881
- Reliability/ECE is appropriate for probability quality, especially for neural and high-capacity models where ranking quality and probability calibration can diverge. Source: Guo et al., 2017, ICML, https://proceedings.mlr.press/v70/guo17a.html
- The deployment-weighted CatBoost experiment follows rare-event/subsampled-data motivation, but in this run the empirical full-grid AP result was negative. Source: Fithian and Hastie, 2014, Annals of Statistics, https://pubmed.ncbi.nlm.nih.gov/25492979/
