# Revision Experiment Repo Audit

- Created at: `2026-05-07T14:54:59.875229`
- Commit hash: `3a48ef85483049608b88aa3e16c77f43ba2c3606`
- Dirty worktree entries: `M configs/revision_evaluation_all_models.yaml
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/analysis.md
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/command.txt
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/environment.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/era5_train_common_features.metadata.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/models/ecmwf_train_ecmwf_test.cbm
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/models/ecmwf_train_ecmwf_test_features.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/models/era5_train_ecmwf_test.cbm
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/models/era5_train_ecmwf_test_features.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/models/mixed_era5_ecmwf_train_ecmwf_test.cbm
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/models/mixed_era5_ecmwf_train_ecmwf_test_features.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/processed_era5_schema_all_years.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/processed_era5_schema_train_years.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/raw_era5_audit.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/split_and_feature_metadata.json
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/description.md
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/plots/pdf/input_source_train_f1.pdf
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/plots/pdf/input_source_train_pr_auc.pdf
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/plots/pdf/input_source_train_yearly_f1.pdf
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/plots/pdf/input_source_train_yearly_pr_auc.pdf
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/plots/png/input_source_train_f1.png
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/plots/png/input_source_train_pr_auc.png
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/plots/png/input_source_train_yearly_f1.png
 M results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/plots/png/input_source_train_yearly_pr_auc.png
 M scripts/run_era5_ecmwf_source_comparison.py
 M src/neural_net/train_nn.py
 M src/revision_evaluation/commands.py
 M src/revision_evaluation/config.py
 M src/revision_evaluation/followups.py
 M src/revision_evaluation/neural_metrics.py
 M src/revision_evaluation/result_library.py
 M src/revision_evaluation/tabular.py
?? results/revision_experiments_complete/commands_used.txt
?? results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/commands_used.md
?? results/revision_experiments_complete/experiments/24_era5_ecmwf_train_source_comparison/artifacts/era5_2000_2008_conversion_manifest.json
?? scripts/prepare_missing_era5_2000_2008.py`

## Main Scripts
- `make_nn_train_data.py`
- `make_train_data.py`
- `src/evaluation/evaluate_boosting.py`
- `src/evaluation/evaluate_log_regression.py`
- `src/evaluation/evaluate_nn.py`
- `src/evaluation/run_combined_evaluations.py`
- `src/log_regression/train_log_regression.py`
- `src/neural_net/train_nn.py`
- `train_catboost.py`
- `tune_catboost.py`

## Configs
- `configs/catboost_train_config.yaml`
- `configs/catboost_tune_config.yaml`
- `configs/download_config.yaml`
- `configs/features_config_30d.yaml`
- `configs/features_config_30d_LSTM.yaml`
- `configs/features_config_30d_LSTM_early_fusion.yaml`
- `configs/features_config_30d_MLP.yaml`
- `configs/features_config_7d_local.yaml`
- `configs/nn_global_full_ft_transformer.yaml`
- `configs/nn_global_full_lstm_attention.yaml`
- `configs/nn_global_full_lstm_gated_moe.yaml`
- `configs/nn_global_full_lstm_static_concat.yaml`
- `configs/nn_global_full_minimal_mlp.yaml`
- `configs/nn_global_full_tsn.yaml`
- `configs/regions_example.yaml`
- `configs/revision_evaluation_all_models.yaml`
- `configs/target_config.yaml`

## Data Paths
- `features_path`: `data/saved_features_boost/train_test_features_30d_all_extended_north_14cb.parquet`
- `feature_config`: `configs/features_config_30d.yaml`
- `target_config`: `configs/target_config.yaml`
- `regions_file`: `configs/regions_example.yaml`
- `era5_raw_dir`: `/home/ids/vmorozov/era5`
- `ecmwf_climate_features_dir`: `/home/ids/vmorozov/data/climate_data/climate_features/ECMWF`
- `processed_era5_features_dir`: `/home/ids/vmorozov/data/climate_data/climate_features/ERA5`

## ERA5
- Raw ERA5 path exists/readable: `True` / `True`
- Raw ERA5 variables found: `2m_dewpoint_temperature, 2m_temperature, mean_sea_level_pressure, soil_temperature_level_1, total_precipitation`
- Processed ERA5 zarr dir exists: `True`

## SEAS5 / ECMWF
- ECMWF climate feature path exists: `True`
- Sample files found: `8`

## Existing Outputs And Models
- Output: `outputs/catboost_train_20260429_231721_122742_bd496e80`
- Output: `outputs/catboost_train_20260429_232521_884351_4450b317`
- Output: `outputs/catboost_train_20260429_232729_741065_86b58a2e`
- Model/artifact: `models/catboost_fire_model_30d.cbm`
- Model/artifact: `models/catboost_fire_model_30d_en.cbm`
- Model/artifact: `models/catboost_fire_model_30d_old.cbm`
- Model/artifact: `models/log_regression`
- Model/artifact: `models/lstm_ml_fire-lstm_early_fusion-epoch=00-val_ap=0.6569.ckpt`
- Model/artifact: `models/lstm_ml_fire-lstm_early_fusion-epoch=04-sel_ap=1.5267.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v1.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=02-val_f1=0.7090.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087-v1.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7230.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7232.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834-v1.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=03-val_f1=0.6871.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v1.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=00-val_f1=0.7237.ckpt`

## Installed Packages In `pointnet`
- `catboost`: available=`True`, version=`1.2.10`
- `xgboost`: available=`False`, version=`None`
- `lightgbm`: available=`False`, version=`None`
- `sklearn`: available=`True`, version=`1.8.0`
- `torch`: available=`True`, version=`2.11.0+cu128`
- `shap`: available=`False`, version=`None`
- `pandas`: available=`True`, version=`3.0.2`
- `numpy`: available=`True`, version=`2.4.4`
- `yaml`: available=`True`, version=`6.0.3`
- `pyarrow`: available=`True`, version=`24.0.0`
- `geopandas`: available=`True`, version=`1.1.3`
- `xarray`: available=`True`, version=`2026.4.0`
- `netCDF4`: available=`True`, version=`1.7.4`

## Directly Supported Experiments
- Reviewer chronological split from saved feature parquet
- CatBoost full, weather-only, FWI-only, and drop-group ablations
- Linear logistic baseline using train-only ordinal encoding
- Poisson point-process GLM baseline using train-only ordinal encoding
- Spline logistic baseline using train-only spline-expanded numeric features
- Random Forest baseline using train-only ordinal encoding and capped bootstrap rows
- Minimal MLP and FT-Transformer NN baselines via the shared neural training registry when NN inputs are present
- Native CatBoost feature importance
- Grouped permutation importance
- CatBoost-native SHAP values if feasible

## Blockers / Needed Adapters
- Full ERA5-vs-SEAS5 matrix requires an ERA5-derived feature parquet with schema parity; raw ERA5 GRIB is present but not a drop-in feature matrix.
- Neural embedding/fusion ablations require prepared_data.npz metadata, which is not present in the current workspace.
- No-dilation target sensitivity requires rebuilding target caches with a modified target-generation function.

## Small Adapters Added
- `src/revision_evaluation/tabular.py`
