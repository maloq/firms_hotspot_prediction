# Revision Experiment Repo Audit

- Created at: `2026-05-31T22:36:36.799959`
- Commit hash: `614edf7245cf5d390a054187f9f256147e87c468`
- Dirty worktree entries: `M configs/features_config_30d.yaml
 M configs/features_config_30d_LSTM.yaml
 M configs/features_config_30d_LSTM_early_fusion.yaml
 M configs/features_config_30d_MLP.yaml
 M configs/features_config_30d_era5.yaml
 M configs/revision_evaluation_all_models_with_nns.yaml
 M configs/selected_columns_30d.txt
 M prediction_pipeline_boosting.py
 M results/rev/README.md
 M results/rev/experiments/04_primary_full_grid_calibrated/tables/01_prevalence_audit.csv
 M results/rev/experiments/04_primary_full_grid_calibrated/tables/02_sampled_vs_full_grid_metric_contrast.csv
 M results/rev/experiments/19_label_sensitivity_by_region/tables/regional_label_sensitivity.csv
 M results/rev/experiments/experiment_index.csv
 D results/rev/shared_artifacts/era5_source_comparison/era5_source_registry_no_tp.csv
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_central_asia_rank01_20230111_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_central_asia_rank02_20230313_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_central_asia_rank03_20220323_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_eastern_siberia_rank01_20220310_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_eastern_siberia_rank02_20220908_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_eastern_siberia_rank03_20230813_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_europe_rank01_20230111_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_europe_rank02_20230313_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_europe_rank03_20220303_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_far_east_rank01_20210322_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_far_east_rank02_20220501_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_far_east_rank03_20240313_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_global_rank01_20220310_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_global_rank02_20220928_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/plots/png/ECMWF_probability_overlay_Spatial_climate_TSN-MLP_global_full_global_rank03_20230113_3d.png
 D results/rev/shared_artifacts/probability_overlays/ECMWF/tables/period_probability_metrics.csv
 D results/rev/shared_artifacts/probability_overlays/ECMWF/tables/selected_probability_periods.csv
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_central_asia_rank01_20230111_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_central_asia_rank02_20230313_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_central_asia_rank03_20220323_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_eastern_siberia_rank01_20220310_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_eastern_siberia_rank02_20220908_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_eastern_siberia_rank03_20230813_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_europe_rank01_20230111_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_europe_rank02_20230313_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_europe_rank03_20220303_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_far_east_rank01_20210322_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_far_east_rank02_20220501_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_far_east_rank03_20240313_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_global_rank01_20220310_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_global_rank02_20220928_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/plots/png/ERA5_probability_overlay_Spatial_climate_TSN-MLP_global_full_global_rank03_20230113_3d.png
 D results/rev/shared_artifacts/probability_overlays/ERA5/tables/period_probability_metrics.csv
 D results/rev/shared_artifacts/probability_overlays/ERA5/tables/selected_probability_periods.csv
 D results/rev/shared_artifacts/representative_model_ablation/representative_model_ablation_era5_to_ecmwf_no_tp_registry.csv
 D results/rev/shared_artifacts/source_plots_mixed/climate_window_permutation_importance.png
 D results/rev/shared_artifacts/source_plots_mixed/embedding_fusion_f1.png
 D results/rev/shared_artifacts/source_plots_mixed/embedding_fusion_pr_auc.png
 D results/rev/shared_artifacts/source_plots_mixed/feature_ablation_f1_drop.png
 D results/rev/shared_artifacts/source_plots_mixed/feature_ablation_pr_auc_drop.png
 D results/rev/shared_artifacts/source_plots_mixed/grouped_permutation_importance.png
 D results/rev/shared_artifacts/source_plots_mixed/input_source_comparison.png
 D results/rev/shared_artifacts/source_plots_mixed/input_source_f1.png
 D results/rev/shared_artifacts/source_plots_mixed/input_source_pr_auc.png
 D results/rev/shared_artifacts/source_plots_mixed/lead_time_f1.png
 D results/rev/shared_artifacts/source_plots_mixed/lead_time_pr_auc.png
 D results/rev/shared_artifacts/source_plots_mixed/native_feature_importance_top30.png
 D results/rev/shared_artifacts/source_plots_mixed/neural_feature_ablation_f1.png
 D results/rev/shared_artifacts/source_plots_mixed/neural_feature_ablation_pr_auc.png
 D results/rev/shared_artifacts/source_plots_mixed/neural_feature_importance_top30.png
 D results/rev/shared_artifacts/source_plots_mixed/pr_curves_global.png
 D results/rev/shared_artifacts/source_plots_mixed/pr_curves_regions.png
 D results/rev/shared_artifacts/source_plots_mixed/representative_model_ablation_era5_to_ecmwf_no_tp_f1.png
 D results/rev/shared_artifacts/source_plots_mixed/representative_model_ablation_era5_to_ecmwf_no_tp_pr_auc.png
 D results/rev/shared_artifacts/source_plots_mixed/shap_summary.png
 M src/data_download/download_black_marble_vnp46a4.py
 M src/feature_generation/cache_black_marble_features.py
 M src/feature_generation/make_features.py
 M src/feature_generation/make_features_nn.py
 M src/feature_generation/prepare_night_light_features.py
 M src/neural_net/prediction_features_builder.py
 M src/revision_evaluation/README.md
 M src/revision_evaluation/config.py
 M src/revision_evaluation/deployment_grid.py
 M src/revision_evaluation/experiment_library.py
 M src/revision_evaluation/full_grid_evaluation.py
 M src/revision_evaluation/probability_metrics.py
 M src/revision_evaluation/probability_overlays.py
 M src/revision_evaluation/stages.py
 M src/revision_evaluation/tabular.py
 M src/revision_evaluation/workflow.py
 M tests/test_prepare_night_light_features.py
?? configs/revision_evaluation_full_grid_improvements.yaml
?? links.txt
?? results/full_grid_improvements_bounded/
?? results/full_grid_improvements_may/
?? results/rev/experiments/33_fire_period_timeline_plots/
?? results/rev/fire_weather_index_ranking/
?? results/rev/shared_artifacts/fire_period_timelines/
?? results/rev/shared_artifacts/prediction_diagnostics/
?? src/feature_generation/calendar_features.py
?? src/revision_evaluation/fire_period_timelines.py
?? src/revision_evaluation/fire_weather_index_evaluation.py
?? src/revision_evaluation/neural_full_grid.py
?? src/revision_evaluation/prediction_diagnostics.py
?? tests/test_calendar_features.py
?? tests/test_fire_weather_index_evaluation.py
?? tests/test_neural_full_grid.py
?? tests/test_prediction_diagnostics.py
?? tests/test_probability_metrics.py
?? tests/test_tabular_sample_weight.py`

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
- `configs/catboost_coast_ablation_with_coast.yaml`
- `configs/catboost_coast_ablation_without_coast.yaml`
- `configs/catboost_train_config.yaml`
- `configs/catboost_train_config_era5.yaml`
- `configs/catboost_train_config_no_terrain.yaml`
- `configs/catboost_tune_config.yaml`
- `configs/download_config.yaml`
- `configs/download_ecmwf_north_america.yaml`
- `configs/download_ecmwf_rest_of_europe.yaml`
- `configs/download_era5_north_america.yaml`
- `configs/download_era5_rest_of_europe.yaml`
- `configs/features_config_30d.yaml`
- `configs/features_config_30d_LSTM.yaml`
- `configs/features_config_30d_LSTM_early_fusion.yaml`
- `configs/features_config_30d_MLP.yaml`
- `configs/features_config_30d_era5.yaml`
- `configs/features_config_7d_local.yaml`
- `configs/night_light_eog_v22.yaml`
- `configs/night_light_nasa_black_marble.yaml`
- `configs/nn_global_full_ft_transformer.yaml`
- `configs/nn_global_full_lstm_attention.yaml`
- `configs/nn_global_full_lstm_gated_moe.yaml`
- `configs/nn_global_full_lstm_static_concat.yaml`
- `configs/nn_global_full_minimal_mlp.yaml`
- `configs/nn_global_full_spatial_tsn.yaml`
- `configs/nn_global_full_spatial_tsn_ecmwf.yaml`
- `configs/nn_global_full_spatial_tsn_no_tp.yaml`
- `configs/nn_global_full_tsn.yaml`
- `configs/regions_europe_illustration.yaml`
- `configs/regions_example.yaml`
- `configs/revision_evaluation_all_models_with_nns.yaml`
- `configs/revision_evaluation_full_grid_improvements.yaml`
- `configs/target_config.yaml`

## Data Paths
- `features_path`: `data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet`
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
- Output: `outputs/catboost_train_20260514_213154_430400_3b41bc0e`
- Output: `outputs/catboost_train_20260517_215546_745305_8ff31767`
- Output: `outputs/catboost_train_20260521_170618_343652_2dba38e0`
- Output: `outputs/catboost_train_20260521_171418_991984_2eb8a999`
- Output: `outputs/catboost_train_no_terrain_20260515_140140_186470_3064a3c4`
- Model/artifact: `models/catboost_no_terrain.cbm`
- Model/artifact: `models/catboost_no_terrain_metrics.json`
- Model/artifact: `models/catboost_no_terrain_probability_calibrator.joblib`
- Model/artifact: `models/catboost_no_terrain_training_config.json`
- Model/artifact: `models/log_regression`
- Model/artifact: `models/lstm_ml_fire-lstm_early_fusion-epoch=00-val_ap=0.6569.ckpt`
- Model/artifact: `models/lstm_ml_fire-lstm_early_fusion-epoch=04-sel_ap=1.5267.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_dynamic_sequence_only-tsn_mlp-epoch=02-val_ap=0.1993.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_dynamic_sequence_only-tsn_mlp-epoch=04-val_ap=0.3011.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_dynamic_sequence_only-tsn_mlp-epoch=08-val_f1=0.6568.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_categorical_features-tsn_mlp-epoch=00-val_ap=0.3469.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_categorical_features-tsn_mlp-epoch=00-val_ap=0.3675.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_categorical_features-tsn_mlp-epoch=01-val_f1=0.7050.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_dynamic_sequence-tsn_mlp-epoch=03-val_ap=0.2383.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_dynamic_sequence-tsn_mlp-epoch=06-val_f1=0.4798.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_dynamic_sequence-tsn_mlp-epoch=12-val_ap=0.2984.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_static_features-tsn_mlp-epoch=00-val_ap=0.3203.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_static_features-tsn_mlp-epoch=00-val_ap=0.3381.ckpt`
- Model/artifact: `models/nn_feature_ablation_tsn_no_static_features-tsn_mlp-epoch=01-val_f1=0.6984.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.4462.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v1.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v2.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v3.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v4.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v5.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v6.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v7.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977-v8.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.6977.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.7118-v1.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=00-val_f1=0.7118.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=02-val_f1=0.7090.ckpt`
- Model/artifact: `models/nn_global_full_ft_transformer-ft_transformer-epoch=05-val_f1=0.4691.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=00-val_f1=0.4686.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.4936.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087-v1.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087-v2.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087-v3.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087-v4.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087-v5.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087-v6.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087-v7.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=01-val_f1=0.7087.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=03-val_f1=0.6938-v1.ckpt`
- Model/artifact: `models/nn_global_full_lstm_attention-lstm_attention-epoch=03-val_f1=0.6938.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=00-val_f1=0.4626.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=00-val_f1=0.4844.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=00-val_f1=0.7130.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7170.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7187.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7188.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7204.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7230.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7232.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=01-val_f1=0.7240.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=02-val_f1=0.7114.ckpt`
- Model/artifact: `models/nn_global_full_lstm_gated_moe-lstm_gated_moe-epoch=02-val_f1=0.7125.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=00-val_f1=0.7122-v1.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=00-val_f1=0.7122.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834-v1.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834-v2.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834-v3.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834-v4.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834-v5.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834-v6.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834-v7.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=01-val_f1=0.6834.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=02-val_f1=0.5004.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=03-val_f1=0.4573.ckpt`
- Model/artifact: `models/nn_global_full_lstm_static_concat-lstm_mlp-epoch=03-val_f1=0.6871.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=01-val_f1=0.4673.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v1.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v2.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v3.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v4.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v5.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v6.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v7.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024-v8.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=04-val_f1=0.7024.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=07-val_f1=0.4687.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=08-val_f1=0.4890.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=09-val_f1=0.7066-v1.ckpt`
- Model/artifact: `models/nn_global_full_minimal_mlp-minimal_mlp-epoch=09-val_f1=0.7066.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn-spatial_climate_tsn_mlp-epoch=00-val_ap=0.6388-v1.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn-spatial_climate_tsn_mlp-epoch=00-val_ap=0.6388.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn-spatial_climate_tsn_mlp-epoch=01-val_ap=0.3782.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn-spatial_climate_tsn_mlp-epoch=01-val_ap=0.5844.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn-spatial_tsn_mlp-epoch=00-val_ap=0.3316.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn_ecmwf-spatial_climate_tsn_mlp-epoch=01-val_ap=0.3782.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn_ecmwf-spatial_climate_tsn_mlp-epoch=03-val_ap=0.3311.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn_ecmwf-spatial_climate_tsn_mlp-epoch=06-val_ap=0.3293.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn_no_tp-spatial_climate_tsn_mlp-epoch=02-val_ap=0.6082.ckpt`
- Model/artifact: `models/nn_global_full_spatial_tsn_no_tp-spatial_climate_tsn_mlp-epoch=03-val_ap=0.5763.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=00-val_ap=0.3466.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=00-val_ap=0.3559.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=00-val_f1=0.7163.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=00-val_f1=0.7177.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=00-val_f1=0.7197.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=00-val_f1=0.7237.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=01-val_f1=0.4435.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=01-val_f1=0.6873.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=01-val_f1=0.6883.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=03-val_f1=0.7088.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=03-val_f1=0.7112.ckpt`
- Model/artifact: `models/nn_global_full_tsn-tsn_mlp-epoch=03-val_f1=0.7169.ckpt`
- Model/artifact: `models/nn_tsn_daily_focal_ap-tsn_mlp-epoch=00-val_ap=0.3457.ckpt`
- Model/artifact: `models/nn_tsn_daily_hard_bce_ap-tsn_mlp-epoch=01-val_ap=0.3343.ckpt`
- Model/artifact: `models/nn_tsn_focal_hard_alpha75_ap-tsn_mlp-epoch=02-val_ap=0.3750.ckpt`
- Model/artifact: `models/nn_tsn_focal_hard_ap-tsn_mlp-epoch=02-val_ap=0.3821.ckpt`
- Model/artifact: `models/nn_tsn_hard_bce_ap-tsn_mlp-epoch=02-val_ap=0.3799.ckpt`
- Model/artifact: `models/nn_tsn_soft_bce_ap-tsn_mlp-epoch=02-val_ap=0.3761.ckpt`
- Model/artifact: `models/nn_tsn_soft_bce_ap-tsn_mlp-epoch=02-val_ap=0.3808.ckpt`
- Model/artifact: `models/nn_tsn_soft_strata_ap-tsn_mlp-epoch=02-val_ap=0.3848.ckpt`

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
