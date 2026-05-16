from __future__ import annotations

from pathlib import Path

import pandas as pd

from .stages import (
    prepare_output_dir,
    run_era5_source_comparison,
    run_representative_model_ablation,
    run_sensitivity_experiments,
    run_main_tabular,
    run_new_nn_models,
    run_organizer,
)
from .config import EvaluationConfig, NN_LABELS
from .neural_metrics import import_neural_metrics
from .neural_importance import run_neural_feature_importance
from .probability_overlays import ProbabilityOverlayConfig, run_probability_overlays, safe_slug


def _resolve_best_neural_model(config: EvaluationConfig) -> str:
    model = str(config.probability_overlay_model or "")
    normalized = safe_slug(model).lower()
    if normalized not in {"best_neural", "best_nn", "best_nns"}:
        return model
    if str(config.probability_overlay_best_model_scope).lower() not in {"global", "global_best", "single"}:
        return model

    neural_path = config.output_dir / "embedding_fusion_ablation.csv"
    if not neural_path.exists():
        return model
    table = pd.read_csv(neural_path)
    if table.empty:
        return model
    if "region" in table.columns:
        table = table[table["region"].astype(str).eq("global")].copy()
    if "period" in table.columns:
        preferred = table[table["period"].astype(str).eq("2021-2025")].copy()
        if not preferred.empty:
            table = preferred
    metric = str(config.probability_overlay_selection_metric or "average_precision")
    if metric not in table.columns:
        return model
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    table = table.dropna(subset=[metric])
    if table.empty:
        return model
    best = table.sort_values(metric, ascending=False).iloc[0]
    label = str(best.get("experiment") or best.get("model") or "")
    label_to_key = {label_value: key for key, label_value in NN_LABELS.items()}
    key = label_to_key.get(label)
    if key is None:
        source_metrics = str(best.get("source_metrics") or "")
        stem = Path(source_metrics).parent.name
        if stem.startswith("nn_global_full_"):
            key = stem.removeprefix("nn_global_full_")
    return f"nn_global_full_{key}" if key else model


def probability_overlay_config(
    config: EvaluationConfig,
    *,
    feature_config: Path | None = None,
    output_dir: Path | None = None,
    source_label: str | None = None,
    dense_neural_training_features: Path | None = None,
    dense_neural_model_path: Path | None = None,
) -> ProbabilityOverlayConfig:
    return ProbabilityOverlayConfig(
        results_dir=config.output_dir,
        regions_file=config.regions_file,
        feature_config=feature_config or config.feature_config,
        target_config=config.target_config,
        source=config.probability_overlay_source,
        model=_resolve_best_neural_model(config),
        selection_metric=config.probability_overlay_selection_metric,
        min_wildfires=config.probability_overlay_min_wildfires,
        window_days=config.probability_overlay_window_days,
        top_periods=config.probability_overlay_top_periods,
        allow_overlapping_periods=config.probability_overlay_allow_overlapping_periods,
        regions=config.probability_overlay_regions,
        include_global=config.probability_overlay_include_global,
        allow_partial_periods=config.probability_overlay_allow_partial_periods,
        map_summary=config.probability_overlay_map_summary,
        surface_source=config.probability_overlay_surface_source,
        dense_model_path=config.probability_overlay_dense_model_path,
        dense_neural_model_path=dense_neural_model_path or config.probability_overlay_dense_neural_model_path,
        dense_neural_training_features=dense_neural_training_features or config.probability_overlay_dense_neural_training_features,
        dense_neural_batch_size=config.probability_overlay_dense_neural_batch_size,
        dense_neural_device=config.probability_overlay_dense_neural_device,
        overwrite_dense=config.probability_overlay_overwrite_dense,
        grid_resolution=config.probability_overlay_grid_resolution,
        interpolation_factor=config.probability_overlay_interpolation_factor,
        prior_correction=config.probability_overlay_prior_correction,
        train_prior=config.probability_overlay_train_prior,
        deploy_prior=config.probability_overlay_deploy_prior,
        verbose_feature_generation=config.probability_overlay_verbose_feature_generation,
        country_shapes=config.probability_overlay_country_shapes,
        output_dir=output_dir or config.probability_overlay_output_dir,
        source_label=source_label,
        formats=config.probability_overlay_formats,
        dpi=config.probability_overlay_dpi,
        keep_existing_plots=config.probability_overlay_keep_existing_plots,
    )


def probability_overlay_configs(config: EvaluationConfig) -> list[ProbabilityOverlayConfig]:
    runs = list(config.probability_overlay_source_runs or [])
    if not runs:
        return [probability_overlay_config(config)]

    base_output = config.probability_overlay_output_dir or (
        config.output_dir / "shared_artifacts" / "probability_overlays"
    )
    configs: list[ProbabilityOverlayConfig] = []
    for idx, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError("probability_overlay_source_runs entries must be mappings.")
        label = str(run.get("name") or run.get("source_label") or f"source_{idx}")
        run_output = Path(run["output_dir"]) if run.get("output_dir") else base_output / safe_slug(label)
        run_feature_config = Path(run.get("feature_config") or config.feature_config)
        run_training_features = Path(
            run.get("dense_neural_training_features")
            or run.get("training_features")
            or config.probability_overlay_dense_neural_training_features
        )
        model_path = run.get("dense_neural_model_path") or run.get("model_path")
        configs.append(
            probability_overlay_config(
                config,
                feature_config=run_feature_config,
                output_dir=run_output,
                source_label=label,
                dense_neural_training_features=run_training_features,
                dense_neural_model_path=Path(model_path) if model_path else None,
            )
        )
    return configs


def run_evaluation(config: EvaluationConfig) -> None:
    prepare_output_dir(config)

    if config.run_main_tabular:
        run_main_tabular(config)
    if config.run_sensitivity_experiments:
        run_sensitivity_experiments(config)
    if config.run_new_nn_models:
        run_new_nn_models(config)
    if config.import_nn_metrics:
        import_neural_metrics(config)
    if config.run_neural_feature_importance:
        run_neural_feature_importance(config)
    if config.run_era5_source_comparison:
        run_era5_source_comparison(config)
    if config.run_representative_model_ablation:
        run_representative_model_ablation(config)
    if config.run_probability_overlays:
        for overlay_config in probability_overlay_configs(config):
            run_probability_overlays(overlay_config)
    if config.run_organizer:
        run_organizer(config)
