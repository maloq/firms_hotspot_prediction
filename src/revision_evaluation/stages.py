from __future__ import annotations

import shutil
from pathlib import Path

from .config import EvaluationConfig


def prepare_output_dir(config: EvaluationConfig) -> None:
    if config.overwrite_output_dir and config.output_dir.exists():
        shutil.rmtree(config.output_dir)


def run_main_tabular(config: EvaluationConfig) -> None:
    from .tabular import default_args, run

    args = default_args(
        features_path=config.features_path,
        feature_config=config.feature_config,
        target_config=config.target_config,
        catboost_config=config.catboost_config,
        regions_file=config.regions_file,
        output_dir=config.output_dir,
        era5_dir=config.era5_dir,
        seed=config.seed,
        catboost_iterations=config.catboost_iterations,
        catboost_depth=config.catboost_depth,
        catboost_learning_rate=config.catboost_learning_rate,
        catboost_task_type=config.catboost_task_type,
        catboost_verbose=config.catboost_verbose,
        rf_max_train_rows=config.rf_max_train_rows,
        linear_epochs=config.linear_epochs,
        point_process_max_train_rows=config.point_process_max_train_rows,
        point_process_alpha=config.point_process_alpha,
        point_process_max_iter=config.point_process_max_iter,
        prediction_batch_size=config.prediction_batch_size,
        permutation_sample_size=config.permutation_sample_size,
        permutation_trials=config.permutation_trials,
        random_error_trials=config.random_error_trials,
        random_error_sample_size=config.random_error_sample_size,
        shap_sample_size=config.shap_sample_size,
        skip_shap=config.skip_shap,
        run_legacy_sampled_evaluation=config.run_legacy_sampled_evaluation,
        run_full_grid_evaluation=config.run_full_grid_evaluation,
        full_grid_is_primary=config.full_grid_is_primary,
        fail_on_full_grid_error=config.fail_on_full_grid_error,
        calibration_start_date=config.calibration_start_date,
        calibration_end_date=config.calibration_end_date,
        test_start_date=config.test_start_date,
        test_end_date=config.test_end_date,
        deployment_grid_resolution=config.deployment_grid_resolution,
        deployment_grid_universe=config.deployment_grid_universe,
        deployment_grid_chunk_by=config.deployment_grid_chunk_by,
        deployment_grid_countries=config.deployment_grid_countries,
        deployment_grid_coordinate_bounds=config.deployment_grid_coordinate_bounds,
        deployment_grid_clip_to_feature_bounds=config.deployment_grid_clip_to_feature_bounds,
        full_grid_mode=config.full_grid_mode,
        weighted_grid_sample=config.weighted_grid_sample,
        weighted_grid_sample_fraction=config.weighted_grid_sample_fraction,
        weighted_grid_sample_strata=config.weighted_grid_sample_strata,
        calibration_method=config.calibration_method,
        n_reliability_bins=config.n_reliability_bins,
        reliability_binning=config.reliability_binning,
        save_full_grid_predictions=config.save_full_grid_predictions,
        save_calibrated_predictions=config.save_calibrated_predictions,
        max_grid_rows_per_chunk=config.max_grid_rows_per_chunk,
        cache_full_grid_features=config.cache_full_grid_features,
        use_lat_lon_features=config.use_lat_lon_features,
        use_ecoregion_features=config.use_ecoregion_features,
        use_historical_fire_features=config.use_historical_fire_features,
    )
    run(args, command="config-driven revision_evaluation.tabular")


def run_sensitivity_experiments(config: EvaluationConfig) -> None:
    from .sensitivity_experiments import run_from_evaluation_config

    run_from_evaluation_config(config)


def run_new_nn_models(config: EvaluationConfig) -> None:
    from .neural_training import run_from_evaluation_config

    run_from_evaluation_config(config)


def run_era5_source_comparison(config: EvaluationConfig) -> None:
    from .era5_source_comparison import default_args, run

    args = default_args(
        features_path=config.features_path,
        feature_config=config.feature_config,
        catboost_config=config.catboost_config,
        regions_file=config.regions_file,
        output_dir=config.output_dir,
        era5_feature_root=config.era5_feature_root,
        ecmwf_feature_root=config.ecmwf_feature_root,
        cache_dir=config.era5_source_cache_dir,
        train_start_year=config.era5_train_start_year,
        validation_start_year=config.era5_validation_start_year,
        test_start_year=config.era5_test_start_year,
        test_end_year=config.era5_test_end_year,
        catboost_iterations=config.sensitivity_catboost_iterations,
        catboost_depth=config.catboost_depth,
        catboost_learning_rate=config.catboost_learning_rate,
        catboost_task_type=config.catboost_task_type,
        catboost_verbose=config.catboost_verbose,
        random_error_trials=config.random_error_trials,
        random_error_sample_size=config.random_error_sample_size,
        seed=config.seed,
        use_lat_lon_features=config.use_lat_lon_features,
        force_rebuild_era5_features=config.force_rebuild_era5_features,
        skip_mixed_source=config.skip_mixed_source,
        drop_climate_variables=config.drop_climate_variables,
        only_experiment=config.only_era5_source_experiments,
        merge_existing_input_source=config.merge_existing_input_source,
        update_organized_results=config.update_organized_results,
        input_source_include_best_neural=config.input_source_include_best_neural,
        input_source_neural_model=config.input_source_neural_model,
        input_source_neural_training_features=config.input_source_neural_training_features,
        input_source_neural_batch_size=config.input_source_neural_batch_size,
        input_source_neural_device=config.input_source_neural_device,
        input_source_neural_masked_climate_variables=config.input_source_neural_masked_climate_variables,
    )
    run(args)


def run_representative_model_ablation(config: EvaluationConfig) -> None:
    from .representative_model_ablation import default_args, run

    args = default_args(
        features_path=config.features_path,
        feature_config=config.feature_config,
        catboost_config=config.catboost_config,
        regions_file=config.regions_file,
        output_dir=config.output_dir,
        era5_feature_root=config.era5_feature_root,
        ecmwf_feature_root=config.ecmwf_feature_root,
        cache_dir=config.era5_source_cache_dir,
        train_start_year=config.era5_train_start_year,
        validation_start_year=config.era5_validation_start_year,
        test_start_year=config.era5_test_start_year,
        test_end_year=config.era5_test_end_year,
        catboost_iterations=config.sensitivity_catboost_iterations,
        catboost_depth=config.catboost_depth,
        catboost_learning_rate=config.catboost_learning_rate,
        catboost_task_type=config.catboost_task_type,
        catboost_verbose=config.catboost_verbose,
        rf_max_train_rows=config.rf_max_train_rows,
        linear_epochs=config.linear_epochs,
        point_process_max_train_rows=config.point_process_max_train_rows,
        point_process_alpha=config.point_process_alpha,
        point_process_max_iter=config.point_process_max_iter,
        prediction_batch_size=config.prediction_batch_size,
        random_error_trials=config.random_error_trials,
        random_error_sample_size=config.random_error_sample_size,
        seed=config.seed,
        use_lat_lon_features=config.use_lat_lon_features,
        force_rebuild_era5_features=config.force_rebuild_era5_features,
        drop_climate_variables=config.drop_climate_variables or ["tp"],
        neural_model=config.input_source_neural_model,
    )
    run(args)


def run_organizer(config: EvaluationConfig) -> None:
    from .experiment_library import organize_results

    organize_results(config.output_dir)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
