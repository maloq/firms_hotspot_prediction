from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ALL_NN_MODELS = [
    "minimal_mlp",
    "ft_transformer",
    "tsn",
    "spatial_tsn",
    "spatial_tsn_no_tp",
    "spatial_tsn_ecmwf",
    "lstm_static_concat",
    "lstm_attention",
    "lstm_gated_moe",
]

NN_LABELS = {
    "minimal_mlp": "Minimal MLP (global full)",
    "ft_transformer": "FT-Transformer (global full)",
    "tsn": "TemporalConvNet / TSN-MLP (global full)",
    "spatial_tsn": "Spatial climate TSN-MLP (global full)",
    "spatial_tsn_no_tp": "Spatial climate TSN-MLP no tp (ERA5 global full)",
    "spatial_tsn_ecmwf": "Spatial climate TSN-MLP (ECMWF global full)",
    "lstm_static_concat": "LSTM static concat (global full)",
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
    run_neural_feature_importance: bool = True
    run_era5_source_comparison: bool = False
    run_representative_model_ablation: bool = False
    run_probability_overlays: bool = False
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
    nn_metrics_glob: str = "outputs/nn_global_full_*/metrics.json"
    nn_feature_ablation_metrics_glob: str = "outputs/nn_feature_ablation_*/metrics.json"
    neural_importance_sample_size: int = 50_000
    neural_importance_batch_size: int = 8192
    neural_importance_device: str = "auto"

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
            "nn_data_path",
            "probability_overlay_dense_model_path",
            "probability_overlay_dense_neural_model_path",
            "probability_overlay_dense_neural_training_features",
            "probability_overlay_country_shapes",
            "probability_overlay_output_dir",
        ]:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        if data.get("python") in {None, ""}:
            data["python"] = sys.executable
        return cls(**data)
