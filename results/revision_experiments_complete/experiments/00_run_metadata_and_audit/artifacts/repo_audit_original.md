# Revision Experiment Repo Audit

- Created at: `2026-05-05T20:14:32.699290`
- Commit hash: `afc61bf008662d03b54dfdd7bdab191057952d80`
- Dirty worktree entries: `M src/feature_generation/load_climate_data.py
 M src/neural_net/models/architectures.py
 M src/neural_net/models/lightning.py
 M src/neural_net/train_nn.py
 M train_catboost.py
?? analyze_year_difficulty.py
?? configs/catboost_train_config.yaml
?? configs/catboost_tune_config.yaml
?? configs/nn_global_full_ft_transformer.yaml
?? configs/nn_global_full_lstm_attention.yaml
?? configs/nn_global_full_lstm_gated_moe.yaml
?? configs/nn_global_full_lstm_static_concat.yaml
?? configs/nn_global_full_minimal_mlp.yaml
?? configs/nn_global_full_tsn.yaml
?? configs/revision_evaluation_all_models.yaml
?? models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977.ckpt
?? models/nn_global_full_ft_transformer-ft_transformer-epoch=02-val_f1=0.7090.ckpt
?? models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087.ckpt
?? models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7232.ckpt
?? models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834.ckpt
?? models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=03-val_f1=0.6871.ckpt
?? models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024.ckpt
?? results/
?? scripts/
?? src/revision_evaluation/
?? tune_catboost.py`

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
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=02-val_f1=0.7090.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7232.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=03-val_f1=0.6871.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024.ckpt`

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
- `scripts/run_revision_experiments.py`
