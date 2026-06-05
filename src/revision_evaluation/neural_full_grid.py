from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .calibration import logit
from .config import NN_LABELS
from .full_grid_evaluation import (
    evaluate_model_full_grid_calibrated,
    primary_dir,
    write_full_grid_failure,
)
from .probability_overlays import (
    infer_dense_neural_model_path,
    load_dense_neural_predictor,
)
from .tabular import load_regions


_MODEL_OUTPUT_DIRS = {
    "minimal_mlp": "nn_global_full_minimal_mlp",
    "minimal_mlp_fullgrid_opt": "nn_global_full_minimal_mlp_fullgrid_opt",
    "minimal_mlp_fullgrid_rank_opt": "nn_global_full_minimal_mlp_fullgrid_rank_opt",
    "nn_global_full_minimal_mlp": "nn_global_full_minimal_mlp",
    "nn_global_full_minimal_mlp_fullgrid_opt": "nn_global_full_minimal_mlp_fullgrid_opt",
    "nn_global_full_minimal_mlp_fullgrid_rank_opt": "nn_global_full_minimal_mlp_fullgrid_rank_opt",
    "spatial_tsn": "nn_global_full_spatial_tsn",
    "spatial_tsn_embedding_fusion": "nn_global_full_spatial_tsn_embedding_fusion",
    "spatial_tsn_no_tp": "nn_global_full_spatial_tsn_no_tp",
    "spatial_tsn_ecmwf": "nn_global_full_spatial_tsn_ecmwf",
    "nn_global_full_spatial_tsn": "nn_global_full_spatial_tsn",
    "nn_global_full_spatial_tsn_embedding_fusion": "nn_global_full_spatial_tsn_embedding_fusion",
    "nn_global_full_spatial_tsn_no_tp": "nn_global_full_spatial_tsn_no_tp",
    "nn_global_full_spatial_tsn_ecmwf": "nn_global_full_spatial_tsn_ecmwf",
    "tsn_embedding_fusion": "nn_global_full_tsn_embedding_fusion",
    "nn_global_full_tsn_embedding_fusion": "nn_global_full_tsn_embedding_fusion",
    "lstm_embedding_fusion": "nn_global_full_lstm_embedding_fusion",
    "nn_global_full_lstm_embedding_fusion": "nn_global_full_lstm_embedding_fusion",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


def _canonical_model_key(model: str) -> str:
    value = str(model).strip()
    if value.startswith("outputs/") or value.endswith(".parquet"):
        stem = Path(value).parts[1] if value.startswith("outputs/") and len(Path(value).parts) > 1 else Path(value).stem
        return stem
    return value


def neural_prediction_path(model: str) -> Path:
    if str(model).strip().endswith(".parquet"):
        return Path(str(model).strip())
    key = _canonical_model_key(model)
    output_name = _MODEL_OUTPUT_DIRS.get(key, key)
    return Path("outputs") / output_name / "legacy_sampled_predictions" / "test_predictions.parquet"


def neural_model_label(model: str) -> str:
    key = _canonical_model_key(model)
    if key.startswith("nn_global_full_"):
        key = key.removeprefix("nn_global_full_")
    return NN_LABELS.get(key, str(model))


def dense_neural_feature_columns(predictor: Any) -> list[str]:
    dynamic_mode = str(getattr(predictor, "dynamic_mode", "") or "").lower()
    dynamic_columns = [] if dynamic_mode in {"daily", "daily_spatial"} else list(
        getattr(predictor, "dynamic_columns", []) or []
    )
    columns = [
        "datetime",
        "lat_rounded",
        "lon_rounded",
        *dynamic_columns,
        *list(getattr(predictor, "static_columns", []) or []),
        *list(getattr(predictor, "categorical_columns", []) or []),
    ]
    return list(dict.fromkeys(str(col) for col in columns if str(col)))


def _predictor_batch_rows(
    predictor: Any,
    rows_per_prediction_batch: int | str | None,
    max_tensor_batch_bytes: int | None = None,
) -> int:
    if hasattr(predictor, "resolve_tensor_batch_rows"):
        return int(
            predictor.resolve_tensor_batch_rows(
                rows_per_prediction_batch,
                max_tensor_batch_bytes=max_tensor_batch_bytes,
            )
        )
    if rows_per_prediction_batch is not None:
        if isinstance(rows_per_prediction_batch, str):
            normalized = rows_per_prediction_batch.strip().lower()
            if normalized not in {"", "auto", "none"}:
                return max(1, int(normalized))
        elif int(rows_per_prediction_batch) > 0:
            return max(1, int(rows_per_prediction_batch))
    return 8192


def make_dense_neural_raw_predict_fn(
    predictor: Any,
    rows_per_prediction_batch: int | str | None,
    max_tensor_batch_bytes: int | None = None,
):
    batch_rows = _predictor_batch_rows(
        predictor,
        rows_per_prediction_batch,
        max_tensor_batch_bytes=max_tensor_batch_bytes,
    )

    def _predict(frame: pd.DataFrame) -> dict[str, Any]:
        probs: list[np.ndarray] = []
        for start in range(0, len(frame), batch_rows):
            end = min(start + batch_rows, len(frame))
            probs.append(np.asarray(predictor.predict(frame.iloc[start:end]), dtype=np.float32).reshape(-1))
        prob = np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)
        return {
            "prob_raw": prob,
            "raw_score": logit(prob).astype(np.float32),
            "raw_score_source": "dense_neural_predictor_logit",
        }

    return _predict


def run_neural_full_grid_evaluation(config: Any) -> list[dict[str, Any]]:
    models: Sequence[str] = getattr(config, "neural_full_grid_models", []) or []
    if not models:
        return []

    feature_config_path = Path(getattr(config, "feature_config"))
    target_config_path = Path(getattr(config, "target_config"))
    feature_config = _load_yaml(feature_config_path)
    target_config = _load_yaml(target_config_path)
    regions = load_regions(Path(getattr(config, "regions_file")))
    training_features = Path(getattr(config, "neural_full_grid_training_features", getattr(config, "features_path")))
    batch_size = int(getattr(config, "neural_full_grid_batch_size", 8192))
    rows_per_prediction_batch = getattr(config, "neural_full_grid_rows_per_prediction_batch", None)
    max_tensor_batch_bytes = getattr(config, "neural_full_grid_max_tensor_batch_bytes", None)
    device = str(getattr(config, "neural_full_grid_device", "auto"))
    masked_dynamic_variables = list(getattr(config, "neural_full_grid_masked_climate_variables", []) or [])
    output_dir = Path(getattr(config, "output_dir"))
    dense_cache_dir = getattr(config, "neural_full_grid_dense_cache_dir", None)
    dense_cache_dir = Path(dense_cache_dir) if dense_cache_dir else None
    dense_cache_policy = str(getattr(config, "neural_full_grid_dense_cache_policy", "none") or "none")
    if dense_cache_dir is None and dense_cache_policy.lower() in {"read", "write", "read-write"}:
        dense_cache_dir = output_dir / "shared_artifacts" / "dense_neural_dynamic_cache"
    dense_block_cache_dir = getattr(config, "neural_full_grid_dense_block_cache_dir", None)
    dense_block_cache_dir = Path(dense_block_cache_dir) if dense_block_cache_dir else None
    if dense_block_cache_dir is None and dense_cache_dir is not None:
        dense_block_cache_dir = dense_cache_dir / "blocks"

    rows: list[dict[str, Any]] = []
    for model in models:
        prediction_path = neural_prediction_path(str(model))
        model_name = neural_model_label(str(model))
        try:
            if not prediction_path.exists():
                raise FileNotFoundError(f"Neural sampled prediction artifact not found: {prediction_path}")
            results_dir = prediction_path.parent.parent
            model_path = infer_dense_neural_model_path(results_dir, prediction_path, None)
            predictor = load_dense_neural_predictor(
                results_dir=results_dir,
                prediction_path=prediction_path,
                training_features_path=training_features,
                model_path=model_path,
                batch_size=batch_size,
                device=device,
                feature_config_path=feature_config_path,
                feature_config=feature_config,
                masked_dynamic_variables=masked_dynamic_variables,
                daily_dynamic_cache_dir=dense_cache_dir,
                daily_dynamic_cache_policy=dense_cache_policy,
                daily_dynamic_block_cache_dir=dense_block_cache_dir,
                daily_dynamic_use_block_cache=bool(getattr(config, "neural_full_grid_dense_use_block_cache", True)),
                daily_dynamic_location_batch_size=getattr(config, "neural_full_grid_dense_location_batch_size", None),
                daily_dynamic_max_time_span_days=getattr(config, "neural_full_grid_dense_max_time_span_days", None),
                daily_dynamic_fill_row_batch_size=getattr(config, "neural_full_grid_dense_fill_row_batch_size", None),
                daily_dynamic_max_slab_spatial_cells=getattr(
                    config,
                    "neural_full_grid_dense_max_slab_spatial_cells",
                    None,
                ),
            )
            feature_columns = dense_neural_feature_columns(predictor)
            metrics = evaluate_model_full_grid_calibrated(
                model_name=model_name,
                model_type="Neural",
                feature_columns=feature_columns,
                categorical_columns=list(getattr(predictor, "categorical_columns", []) or []),
                config=config,
                output_dir=output_dir,
                predict_raw_fn=make_dense_neural_raw_predict_fn(
                    predictor,
                    rows_per_prediction_batch,
                    max_tensor_batch_bytes=max_tensor_batch_bytes,
                ),
                feature_config=feature_config,
                target_config=target_config,
                regions=regions,
                model_path=model_path,
                feature_set="daily spatial neural dense full-grid",
            )
            rows.append(
                {
                    "model": str(model),
                    "model_name": model_name,
                    "prediction_path": str(prediction_path),
                    "model_path": str(model_path),
                    "status": "completed",
                    "feature_columns": feature_columns,
                    "metrics": metrics,
                }
            )
            logging.info("Full-grid dense neural evaluation complete for %s", model_name)
        except Exception as exc:
            failure_path = write_full_grid_failure(
                output_dir,
                model_name=model_name,
                model_type="Neural",
                exc=exc,
            )
            rows.append(
                {
                    "model": str(model),
                    "model_name": model_name,
                    "prediction_path": str(prediction_path),
                    "status": "failed",
                    "failure_path": str(failure_path),
                    "reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            logging.exception("Full-grid dense neural evaluation failed for %s", model_name)
            if bool(getattr(config, "fail_on_full_grid_error", False)):
                raise

    out = primary_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "neural_full_grid_manifest.json").write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )
    try:
        from .experiment_library import refresh_primary_full_grid_experiment

        refresh_primary_full_grid_experiment(output_dir)
    except Exception:
        logging.exception("Failed to refresh experiment 04 after dense neural full-grid evaluation.")
        if bool(getattr(config, "fail_on_full_grid_error", False)):
            raise
    return rows
