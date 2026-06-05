from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ALL_NN_MODELS = [
    "minimal_mlp",
    "minimal_mlp_fullgrid_opt",
    "minimal_mlp_fullgrid_rank_opt",
    "ft_transformer",
    "tsn",
    "tsn_embedding_fusion",
    "spatial_tsn",
    "spatial_tsn_embedding_fusion",
    "spatial_tsn_no_tp",
    "spatial_tsn_ecmwf",
    "lstm_static_concat",
    "lstm_embedding_fusion",
    "lstm_attention",
    "lstm_gated_moe",
]

NN_LABELS = {
    "minimal_mlp": "Minimal MLP (global full)",
    "minimal_mlp_fullgrid_opt": "Minimal MLP full-grid optimized",
    "minimal_mlp_fullgrid_rank_opt": "Minimal MLP full-grid rank optimized",
    "ft_transformer": "FT-Transformer (global full)",
    "tsn": "TemporalConvNet / TSN-MLP (global full)",
    "tsn_embedding_fusion": "TemporalConvNet embedding fusion (global full)",
    "spatial_tsn": "Spatial climate TSN-MLP (global full)",
    "spatial_tsn_embedding_fusion": "Spatial climate TSN embedding fusion (global full)",
    "spatial_tsn_no_tp": "Spatial climate TSN-MLP no tp (ERA5 global full)",
    "spatial_tsn_ecmwf": "Spatial climate TSN-MLP (ECMWF global full)",
    "lstm_static_concat": "LSTM static concat (global full)",
    "lstm_embedding_fusion": "LSTM embedding fusion (global full)",
    "lstm_attention": "LSTM attention (global full)",
    "lstm_gated_moe": "LSTM gated MoE (global full)",
}


@dataclass
class EvaluationConfig:
    output_dir: Path = Path("results/revision_experiments_complete")
    python: str = sys.executable
    overwrite_output_dir: bool = False
    seed: int = 17

    run_main_tabular: bool = True
    run_sensitivity_experiments: bool = True
    run_new_nn_models: bool = True
    import_nn_metrics: bool = True
    run_neural_full_grid_evaluation: bool = False
    run_neural_feature_importance: bool = True
    run_era5_source_comparison: bool = False
    run_representative_model_ablation: bool = False
    run_probability_overlays: bool = False
    run_fire_period_timelines: bool = False
    run_prediction_diagnostics: bool = False
    run_fire_weather_index_evaluation: bool = False
    run_organizer: bool = True

    run_legacy_sampled_evaluation: bool = True
    run_full_grid_evaluation: bool = True
    full_grid_is_primary: bool = False
    fail_on_full_grid_error: bool = False

    calibration_start_date: str | None = None
    calibration_end_date: str | None = None
    test_start_date: str | None = None
    test_end_date: str | None = None

    deployment_grid_resolution: float = 0.1
    deployment_grid_universe: str = "land_or_burnable"
    deployment_grid_chunk_by: list[str] = field(default_factory=lambda: ["country", "month"])
    deployment_grid_countries: list[str] | None = None
    deployment_grid_coordinate_bounds: list[float] | None = None
    deployment_grid_clip_to_feature_bounds: bool = True

    full_grid_mode: str = "full_grid"
    weighted_grid_sample: bool = False
    weighted_grid_sample_fraction: float | None = None
    weighted_grid_sample_strata: list[str] = field(default_factory=lambda: ["country", "month"])
    weighted_grid_sample_include_all_positives: bool = True

    calibration_method: str = "platt_month"
    n_reliability_bins: int = 20
    reliability_binning: str = "equal_count"
    full_grid_selection_metric: str = "average_precision"
    full_grid_selection_direction: str | None = None
    full_grid_selection_region: str = "global"
    full_grid_selection_split: str = "test"
    save_full_grid_predictions: bool = True
    save_calibrated_predictions: bool = True
    max_grid_rows_per_chunk: int | None = None
    cache_full_grid_features: bool = False

    use_lat_lon_features: bool = True
    use_ecoregion_features: bool = True
    use_historical_fire_features: bool = True

    features_path: Path = Path("data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet")
    seven_day_features_path: Path = Path("data/saved_features/train_test_features_7d_all.parquet")
    feature_config: Path = Path("configs/features_config_30d.yaml")
    target_config: Path = Path("configs/target_config.yaml")
    catboost_config: Path = Path("configs/catboost_train_config.yaml")
    regions_file: Path = Path("configs/regions_example.yaml")
    era5_dir: Path = Path("/home/ids/vmorozov/era5")
    era5_feature_root: Path = Path("/home/ids/vmorozov/data/climate_data/climate_features/ERA5")
    ecmwf_feature_root: Path = Path("/home/ids/vmorozov/data/climate_data/climate_features/ECMWF")
    era5_source_cache_dir: Path = Path("data/saved_features/revision_evaluation/era5_source_comparison")

    catboost_iterations: int = 450
    sensitivity_catboost_iterations: int = 260
    catboost_depth: int = 5
    catboost_learning_rate: float = 0.03
    catboost_task_type: str = "GPU"
    catboost_verbose: int = 100
    rf_max_train_rows: int = 300_000
    rf_n_estimators: int = 120
    rf_max_depth: int | None = 18
    rf_min_samples_leaf: int = 20
    rf_positive_class_weight: float = 4.0
    rf_max_features: str | float | None = None
    linear_epochs: int = 4
    point_process_max_train_rows: int = 500_000
    point_process_alpha: float = 1e-4
    point_process_max_iter: int = 200
    prediction_batch_size: int = 100_000
    random_error_trials: int = 5
    random_error_sample_size: int = 50_000
    permutation_trials: int = 5
    permutation_sample_size: int = 50_000
    shap_sample_size: int = 8_000
    skip_shap: bool = False
    run_label_sensitivity: bool = True
    run_lead_time_sensitivity: bool = True
    tabular_model_subset: list[str] | None = None
    new_nn_models: list[str] = field(default_factory=lambda: list(ALL_NN_MODELS))
    main_nn_model: str = "tsn"
    run_neural_feature_ablation: bool = True
    neural_feature_ablation_model: str | None = None
    neural_feature_ablation_variants: list[str] = field(
        default_factory=lambda: [
            "no_dynamic_sequence",
            "no_static_features",
            "no_categorical_features",
            "dynamic_sequence_only",
        ]
    )
    nn_data_path: Path | None = None
    nn_dry_run: bool = False
    nn_parallel_jobs: int | str = "auto"
    nn_parallel_devices: list[str] | str | None = "auto"
    skip_existing_nn_models: bool = False
    nn_metrics_glob: str = "outputs/nn_global_full_*/metrics.json"
    nn_feature_ablation_metrics_glob: str = "outputs/nn_feature_ablation_*/metrics.json"
    neural_importance_sample_size: int = 50_000
    neural_importance_batch_size: int = 8192
    neural_importance_device: str = "auto"
    neural_full_grid_models: list[str] = field(default_factory=lambda: ["spatial_tsn_no_tp"])
    neural_full_grid_training_features: Path = Path(
        "data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet"
    )
    neural_full_grid_batch_size: int = 8192
    neural_full_grid_rows_per_prediction_batch: int | str | None = None
    neural_full_grid_max_tensor_batch_bytes: int = 512 * 1024 * 1024
    neural_full_grid_device: str = "auto"
    neural_full_grid_masked_climate_variables: list[str] = field(default_factory=list)
    neural_full_grid_dense_cache_dir: Path | None = None
    neural_full_grid_dense_cache_policy: str = "read-write"
    neural_full_grid_dense_block_cache_dir: Path | None = None
    neural_full_grid_dense_use_block_cache: bool = True
    neural_full_grid_dense_location_batch_size: int | None = None
    neural_full_grid_dense_max_time_span_days: int | None = None
    neural_full_grid_dense_fill_row_batch_size: int | None = None
    neural_full_grid_dense_max_slab_spatial_cells: int | None = None

    era5_train_start_year: int = 2001
    era5_validation_start_year: int = 2019
    era5_test_start_year: int = 2021
    era5_test_end_year: int | None = None
    force_rebuild_era5_features: bool = False
    skip_mixed_source: bool = False
    drop_climate_variables: list[str] = field(default_factory=list)
    only_era5_source_experiments: list[str] | None = None
    merge_existing_input_source: bool = False
    update_organized_results: bool = False
    input_source_include_best_neural: bool = False
    input_source_neural_model: str = "best_neural"
    input_source_neural_training_features: Path = Path(
        "data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet"
    )
    input_source_neural_batch_size: int = 8192
    input_source_neural_device: str = "auto"
    input_source_neural_rows_per_prediction_batch: int | str | None = None
    input_source_neural_max_tensor_batch_bytes: int = 512 * 1024 * 1024
    input_source_neural_dense_cache_dir: Path | None = None
    input_source_neural_dense_cache_policy: str = "read-write"
    input_source_neural_dense_block_cache_dir: Path | None = None
    input_source_neural_dense_use_block_cache: bool = True
    input_source_neural_dense_location_batch_size: int | None = None
    input_source_neural_dense_max_time_span_days: int | None = None
    input_source_neural_dense_fill_row_batch_size: int | None = None
    input_source_neural_dense_max_slab_spatial_cells: int | None = None
    input_source_neural_masked_climate_variables: list[str] = field(default_factory=list)

    probability_overlay_source: str = "legacy"
    probability_overlay_model: str = "best_neural"
    probability_overlay_best_model_scope: str = "per_period"
    probability_overlay_source_runs: list[dict[str, Any]] = field(default_factory=list)
    probability_overlay_selection_metric: str = "average_precision"
    probability_overlay_min_wildfires: int = 7
    probability_overlay_spatial_tolerance_degrees: float = 0.0
    probability_overlay_window_days: int = 3
    probability_overlay_top_periods: int = 1
    probability_overlay_max_period_end: str | None = None
    probability_overlay_allow_overlapping_periods: bool = False
    probability_overlay_regions: list[str] | None = None
    probability_overlay_include_global: bool = True
    probability_overlay_allow_partial_periods: bool = False
    probability_overlay_map_summary: str = "sum"
    probability_overlay_surface_source: str = "dense-neural"
    probability_overlay_dense_model_path: Path | None = None
    probability_overlay_dense_neural_model_path: Path | None = None
    probability_overlay_dense_neural_training_features: Path = Path(
        "data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet"
    )
    probability_overlay_dense_neural_batch_size: int = 8192
    probability_overlay_dense_neural_device: str = "auto"
    probability_overlay_dense_neural_rows_per_prediction_batch: int | str | None = None
    probability_overlay_dense_neural_max_tensor_batch_bytes: int = 512 * 1024 * 1024
    probability_overlay_dense_neural_cache_dir: Path | None = None
    probability_overlay_dense_neural_cache_policy: str = "read-write"
    probability_overlay_dense_neural_block_cache_dir: Path | None = None
    probability_overlay_dense_neural_use_block_cache: bool = True
    probability_overlay_dense_neural_location_batch_size: int | None = None
    probability_overlay_dense_neural_max_time_span_days: int | None = None
    probability_overlay_dense_neural_fill_row_batch_size: int | None = None
    probability_overlay_dense_neural_max_slab_spatial_cells: int | None = None
    probability_overlay_overwrite_dense: bool = False
    probability_overlay_grid_resolution: float | None = None
    probability_overlay_interpolation_factor: int = 5
    probability_overlay_prior_correction: bool = True
    probability_overlay_train_prior: float = 0.15
    probability_overlay_deploy_prior: float = 0.001
    probability_overlay_colormap: str = "YlOrRd"
    probability_overlay_color_floor: float | None = None
    probability_overlay_color_vmax: float | None = None
    probability_overlay_verbose_feature_generation: bool = False
    probability_overlay_country_shapes: Path = Path("data/countries")
    probability_overlay_output_dir: Path | None = None
    probability_overlay_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    probability_overlay_dpi: int = 320
    probability_overlay_keep_existing_plots: bool = False

    fire_period_timeline_source: str | None = None
    fire_period_timeline_model: str | None = None
    fire_period_timeline_selection_metric: str | None = None
    fire_period_timeline_min_wildfires: int | None = None
    fire_period_timeline_spatial_tolerance_degrees: float | None = None
    fire_period_timeline_window_days: int = 28
    fire_period_timeline_top_periods: int = 1
    fire_period_timeline_allow_overlapping_periods: bool | None = None
    fire_period_timeline_common_windows: bool = False
    fire_period_timeline_reference_source: str | None = None
    fire_period_timeline_regions: list[str] | None = None
    fire_period_timeline_include_global: bool | None = None
    fire_period_timeline_allow_partial_periods: bool | None = None
    fire_period_timeline_excluded_months: list[int] = field(default_factory=lambda: [12, 1, 2])
    fire_period_timeline_prefer_centered_activity: bool = True
    fire_period_timeline_center_peak_min_fraction: float = 0.25
    fire_period_timeline_center_peak_max_fraction: float = 0.75
    fire_period_timeline_max_start_activity_fraction: float = 0.15
    fire_period_timeline_min_middle_activity_fraction: float = 0.50
    fire_period_timeline_output_dir: Path | None = None
    fire_period_timeline_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    fire_period_timeline_dpi: int = 320
    fire_period_timeline_lead_column: str | None = None
    fire_period_timeline_max_lead_days: int = 10
    fire_period_timeline_burned_area_label: str = "Burnt Area"
    fire_period_timeline_count_colormap: str = "fire_risk"
    fire_period_timeline_count_norm_gamma: float = 0.42
    fire_period_timeline_count_vmax_percentile: float = 95.0
    fire_period_timeline_generate_overlay_maps: bool = True
    fire_period_timeline_overlay_surface_source: str | None = None
    fire_period_timeline_overlay_window_days: int | None = 3
    fire_period_timeline_overlay_center_on_observed_peak: bool = True

    prediction_diagnostics_source: str = "legacy"
    prediction_diagnostics_model: str = "best_neural"
    prediction_diagnostics_regions: list[str] | None = None
    prediction_diagnostics_include_global: bool = True
    prediction_diagnostics_time_frequency: str = "month"
    prediction_diagnostics_output_dir: Path | None = None
    prediction_diagnostics_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    prediction_diagnostics_dpi: int = 260
    prediction_diagnostics_grid_resolution: float | None = None
    prediction_diagnostics_plot_interpolation_resolution: float | None = None
    prediction_diagnostics_country_shapes: Path = Path("data/countries")
    prediction_diagnostics_error_colormap: str = "risk_residual"
    prediction_diagnostics_ground_truth_smoothing_sigma_cells: float = 0.0
    prediction_diagnostics_recalibrate_on_test: bool = True
    prediction_diagnostics_require_full_grid_predictions: bool = True
    prediction_diagnostics_generate_full_grid_predictions: bool = False
    prediction_diagnostics_full_grid_model_path: Path | None = None
    prediction_diagnostics_full_grid_feature_schema_path: Path | None = None
    prediction_diagnostics_sample_prediction_days_per_month: int | None = None
    prediction_diagnostics_months_per_feature_chunk: int = 6
    prediction_diagnostics_max_grid_rows_per_chunk: int | None = None

    fire_weather_index_dir: Path = Path("fire_weather_indexes")
    fire_weather_index_output_dir: Path | None = None
    fire_weather_index_variables: list[str] | None = None
    fire_weather_index_save_predictions: bool = False
    fire_weather_index_run_logistic_regression: bool = True
    fire_weather_index_logistic_train_year: int = 2022

    @classmethod
    def from_yaml(cls, path: Path) -> "EvaluationConfig":
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping in {path}")
        data: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                data[field_name] = raw[field_name]

        for key in [
            "output_dir",
            "features_path",
            "seven_day_features_path",
            "feature_config",
            "target_config",
            "catboost_config",
            "regions_file",
            "era5_dir",
            "era5_feature_root",
            "ecmwf_feature_root",
            "era5_source_cache_dir",
            "input_source_neural_training_features",
            "input_source_neural_dense_cache_dir",
            "input_source_neural_dense_block_cache_dir",
            "nn_data_path",
            "neural_full_grid_training_features",
            "neural_full_grid_dense_cache_dir",
            "neural_full_grid_dense_block_cache_dir",
            "probability_overlay_dense_model_path",
            "probability_overlay_dense_neural_model_path",
            "probability_overlay_dense_neural_training_features",
            "probability_overlay_dense_neural_cache_dir",
            "probability_overlay_dense_neural_block_cache_dir",
            "probability_overlay_country_shapes",
            "probability_overlay_output_dir",
            "fire_period_timeline_output_dir",
            "prediction_diagnostics_output_dir",
            "prediction_diagnostics_country_shapes",
            "prediction_diagnostics_full_grid_model_path",
            "prediction_diagnostics_full_grid_feature_schema_path",
            "fire_weather_index_dir",
            "fire_weather_index_output_dir",
        ]:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        if data.get("python") in {None, ""}:
            data["python"] = sys.executable
        return cls(**data)
