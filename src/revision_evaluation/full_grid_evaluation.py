from __future__ import annotations

import json
import logging
import math
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.feature_generation.calendar_features import add_calendar_context_features
from src.feature_generation.make_features import make_features_from_target_df

from .calibration import (
    apply_calibrator,
    build_calibration_metadata,
    fit_calibrator,
    logit,
    save_calibrator,
    sigmoid,
    write_calibration_metadata,
)
from .deployment_grid import REQUIRED_DEPLOYMENT_COLUMNS, iter_deployment_grid_chunks
from .probability_metrics import (
    calibration_slope_intercept,
    daily_expected_observed_mae,
    expected_observed_count_ratio,
    make_reliability_bins,
    max_weighted_f1,
    reliability_summary,
    weighted_average_precision,
    weighted_brier_score,
    weighted_log_loss,
    weighted_prevalence,
    weighted_roc_auc,
)


PredictionFn = Callable[[pd.DataFrame], dict[str, Any] | tuple[np.ndarray, np.ndarray]]
RISK_TOP_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
SPATIAL_COARSE_RESOLUTIONS = (0.5, 1.0)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def primary_dir(output_dir: Path) -> Path:
    return output_dir / "primary_full_grid_calibrated"


def failure_path(output_dir: Path, model_name: str) -> Path:
    return primary_dir(output_dir) / "failures" / f"{safe_slug(model_name)}.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return None if not math.isfinite(scalar) else scalar
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _raw_jsonl_for_csv(path: Path) -> Path | None:
    if path.suffix != ".csv":
        return None
    if path.parent.name != "primary_full_grid_calibrated":
        return None
    output_dir = path.parent.parent
    return output_dir / "shared_artifacts" / "raw_tables_jsonl" / f"primary_full_grid_calibrated_{path.stem}.jsonl.gz"


def _raw_schema_for_csv(path: Path) -> Path | None:
    if path.suffix != ".csv":
        return None
    if path.parent.name != "primary_full_grid_calibrated":
        return None
    output_dir = path.parent.parent
    return output_dir / "shared_artifacts" / "raw_table_schemas" / f"primary_full_grid_calibrated_{path.stem}.schema.json"


def _read_existing_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    raw_path = _raw_jsonl_for_csv(path)
    if raw_path is not None and raw_path.exists():
        return pd.read_json(raw_path, orient="records", lines=True, compression="gzip")
    return pd.DataFrame()


def _sync_organized_raw_table(path: Path, table: pd.DataFrame) -> None:
    raw_path = _raw_jsonl_for_csv(path)
    if raw_path is None or not raw_path.parent.exists():
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_json(raw_path, orient="records", lines=True, compression="gzip")

    schema_path = _raw_schema_for_csv(path)
    if schema_path is not None:
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema = {
            "source_csv": f"primary_full_grid_calibrated/{path.name}",
            "rows": int(len(table)),
            "columns": [{"name": col, "dtype": str(table[col].dtype)} for col in table.columns],
        }
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _upsert_csv(path: Path, rows: pd.DataFrame, key_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = _read_existing_table(path)
    if not old.empty:
        if not old.empty and all(col in old.columns for col in key_cols) and all(col in rows.columns for col in key_cols):
            old_key = old[key_cols].astype(str).agg("\x1f".join, axis=1)
            new_key = rows[key_cols].astype(str).agg("\x1f".join, axis=1)
            old = old.loc[~old_key.isin(set(new_key))]
        combined = pd.concat([old, rows], ignore_index=True)
    else:
        combined = rows
    combined.to_csv(path, index=False)
    _sync_organized_raw_table(path, combined)


def _selection_direction(metric_name: str, requested: str | None = None) -> str:
    requested_key = str(requested or "").strip().lower()
    if requested_key in {"min", "max"}:
        return requested_key
    name = str(metric_name).lower()
    smaller_is_better = (
        "loss",
        "logloss",
        "brier",
        "error",
        "ece",
        "mce",
        "rmse",
        "mae",
    )
    return "min" if any(token in name for token in smaller_is_better) else "max"


def write_full_grid_model_selection(output_dir: Path, config: Any) -> pd.DataFrame:
    out = primary_dir(output_dir)
    comparison_path = out / "model_comparison.csv"
    if not comparison_path.exists():
        return pd.DataFrame()
    comparison = pd.read_csv(comparison_path)
    metric_name = str(getattr(config, "full_grid_selection_metric", "average_precision") or "average_precision")
    direction = _selection_direction(metric_name, getattr(config, "full_grid_selection_direction", None))
    region = str(getattr(config, "full_grid_selection_region", "global") or "global")
    split = str(getattr(config, "full_grid_selection_split", "test") or "test")
    rows = comparison.copy()
    if "region" in rows.columns:
        rows = rows[rows["region"].astype(str).eq(region)]
    if "split" in rows.columns:
        rows = rows[rows["split"].astype(str).eq(split)]
    if rows.empty or metric_name not in rows.columns:
        return pd.DataFrame()
    rows["_selection_metric_value"] = pd.to_numeric(rows[metric_name], errors="coerce")
    rows = rows.dropna(subset=["_selection_metric_value"]).copy()
    if rows.empty:
        return pd.DataFrame()
    rows = rows.sort_values(
        "_selection_metric_value",
        ascending=direction == "min",
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    rows.insert(0, "selection_rank", np.arange(1, len(rows) + 1))
    rows.insert(1, "selection_metric", metric_name)
    rows.insert(2, "selection_direction", direction)
    rows.insert(3, "selection_target", f"primary_full_grid_calibrated:{region}:{split}")
    rows = rows.rename(columns={"_selection_metric_value": "selection_metric_value"})
    rows.to_csv(out / "model_selection.csv", index=False)

    best = rows.iloc[0].to_dict()
    payload = {
        "selection_target": best.get("selection_target"),
        "selection_metric": metric_name,
        "selection_direction": direction,
        "model_name": best.get("model_name"),
        "model_type": best.get("model_type"),
        "feature_set": best.get("feature_set"),
        "metric_value": best.get("selection_metric_value"),
        "source_table": str(comparison_path),
    }
    (out / "best_model.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return rows


def _stable_seed(value: str, offset: int = 0) -> int:
    base = sum((idx + 1) * ord(char) for idx, char in enumerate(str(value)))
    return int((base + offset) % (2**32 - 1))


def _error_trials(config: Any) -> int:
    return int(getattr(config, "random_error_trials", 1) or 1)


def _error_sample_size(config: Any) -> int:
    return int(getattr(config, "random_error_sample_size", 0) or 0)


def _as_metric_arrays(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(score, dtype=float).reshape(-1)
    if sample_weight is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
    if not (len(y) == len(p) == len(w)):
        raise ValueError("Metric arrays must have the same length.")
    finite = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    return y[finite].astype(int), np.clip(p[finite], 0.0, 1.0), w[finite]


def _binary_metric_values(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None,
) -> dict[str, Any]:
    y, p, w = _as_metric_arrays(y_true, score, sample_weight)
    if len(y) == 0:
        return {
            "support": 0,
            "weighted_support": 0.0,
            "positives": 0.0,
            "observed_prevalence": None,
            "mean_score": None,
            "average_precision": None,
            "roc_auc": None,
            "weighted_brier_score": None,
            "weighted_logloss": None,
            "max_f1": None,
            "precision_at_max_f1": None,
            "recall_at_max_f1": None,
            "threshold_at_max_f1": None,
            "predicted_positive_grid_cells_at_max_f1": None,
            "expected_fire_positive_grid_cells": None,
            "observed_fire_positive_grid_cells": None,
            "expected_observed_count_ratio": None,
        }
    f1 = max_weighted_f1(y, p, w)
    count = expected_observed_count_ratio(y, p, w)
    return {
        "support": int(len(y)),
        "weighted_support": float(np.sum(w)),
        "positives": float(np.sum(y * w)),
        "observed_prevalence": weighted_prevalence(y, w),
        "mean_score": float(np.average(p, weights=w)) if len(y) else None,
        "average_precision": weighted_average_precision(y, p, w),
        "roc_auc": weighted_roc_auc(y, p, w),
        "weighted_brier_score": weighted_brier_score(y, p, w),
        "weighted_logloss": weighted_log_loss(y, p, w),
        **f1,
        **count,
    }


def _reliability_values(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None,
    config: Any | None,
) -> dict[str, float | None]:
    return reliability_summary(
        y_true,
        score,
        sample_weight,
        n_bins=int(getattr(config, "n_reliability_bins", 20) if config is not None else 20),
        strategy=str(getattr(config, "reliability_binning", "equal_count") if config is not None else "equal_count"),
    )


def _bootstrap_index_samples(
    y_true: np.ndarray | pd.Series,
    *,
    trials: int,
    sample_size: int,
    seed: int,
) -> Iterable[np.ndarray]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if trials <= 1 or len(y) == 0:
        return
    n = min(len(y), int(sample_size)) if sample_size and sample_size > 0 else len(y)
    if n <= 1:
        return
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    pos_frac = len(pos) / len(y) if len(y) else 0.0
    all_idx = np.arange(len(y))
    for _ in range(trials):
        if len(pos) and len(neg):
            n_pos = min(max(1, int(round(n * pos_frac))), n - 1)
            n_neg = n - n_pos
            idx = np.concatenate(
                [
                    rng.choice(pos, size=n_pos, replace=True),
                    rng.choice(neg, size=n_neg, replace=True),
                ]
            )
            rng.shuffle(idx)
        else:
            idx = rng.choice(all_idx, size=n, replace=True)
        yield idx


def _std_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else None


def _bootstrap_array_metric_errors(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None,
    *,
    config: Any,
    seed: int,
    metric_names: list[str],
) -> dict[str, float | None]:
    trials = _error_trials(config)
    if trials <= 1:
        return {f"{name}_error": None for name in metric_names}
    y, p, w = _as_metric_arrays(y_true, score, sample_weight)
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for idx in _bootstrap_index_samples(
        y,
        trials=trials,
        sample_size=_error_sample_size(config),
        seed=seed,
    ):
        metrics = _binary_metric_values(y[idx], p[idx], w[idx])
        for name in metric_names:
            value = metrics.get(name)
            if value is not None and math.isfinite(float(value)):
                values[name].append(float(value))
    return {f"{name}_error": _std_or_none(vals) for name, vals in values.items()}


def _coerce_prediction_result(result: dict[str, Any] | tuple[np.ndarray, np.ndarray], n_rows: int) -> tuple[np.ndarray, np.ndarray, str]:
    if isinstance(result, dict):
        raw = result.get("raw_score")
        prob = result.get("prob_raw")
        source = str(result.get("raw_score_source") or "predict_raw_fn")
    else:
        raw, prob = result
        source = "predict_raw_fn"
    if prob is None and raw is not None:
        prob = sigmoid(raw)
    if raw is None and prob is not None:
        raw = logit(prob)
        source = "logit_predict_proba"
    if raw is None or prob is None:
        raise ValueError("predict_raw_fn must return raw_score and/or prob_raw.")
    raw_arr = np.asarray(raw, dtype=float).reshape(-1)
    prob_arr = np.asarray(prob, dtype=float).reshape(-1)
    if len(raw_arr) != n_rows or len(prob_arr) != n_rows:
        raise ValueError(
            f"Prediction length mismatch: raw={len(raw_arr)}, prob={len(prob_arr)}, expected={n_rows}."
        )
    return raw_arr, np.clip(prob_arr, 0.0, 1.0), source


def _align_feature_columns(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in features.columns]
    if missing:
        raise KeyError(f"Deployment features missing {len(missing)} model columns; first missing columns: {missing[:10]}")
    return features[feature_columns].copy()


def _make_features_for_chunk(
    chunk_rows: pd.DataFrame,
    *,
    feature_config: dict[str, Any] | str | Path,
    feature_columns: list[str],
    test_mode: bool = True,
) -> pd.DataFrame:
    rows = add_calendar_context_features(chunk_rows.copy())
    if "acq_date" not in rows.columns and "datetime" in rows.columns:
        rows["acq_date"] = pd.to_datetime(rows["datetime"]).dt.normalize()
    if all(col in rows.columns for col in feature_columns):
        return rows
    return make_features_from_target_df(
        str(feature_config) if isinstance(feature_config, Path) else feature_config,
        rows,
        test_mode=test_mode,
        use_cached_files=False,
        cache_dir="data/saved_features/deployment_grid_feature_cache",
        extra_anchor_cols=["acq_date", "country", "is_fire", "eval_weight"],
    )


def _feature_cache_path(output_dir: Path, split_name: str, country: str | None, period: str) -> Path:
    root_path = (
        primary_dir(output_dir)
        / "feature_cache"
        / f"{safe_slug(split_name)}_{safe_slug(country or 'all')}_{safe_slug(period)}.parquet"
    )
    shared_path = (
        output_dir
        / "shared_artifacts"
        / "primary_full_grid_calibrated"
        / "feature_cache"
        / root_path.name
    )
    if shared_path.exists():
        return shared_path
    return root_path


def _feature_cache_matches_chunk(
    features: pd.DataFrame,
    chunk_rows: pd.DataFrame,
    *,
    resolution: float = 0.1,
) -> bool:
    key_cols = ["datetime", "lat_rounded", "lon_rounded", "country", "is_fire", "eval_weight"]
    for col in key_cols:
        if col not in features.columns or col not in chunk_rows.columns:
            return False

    def key_frame(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=range(len(frame)))
        out["datetime"] = (
            pd.to_datetime(frame["datetime"])
            .astype("datetime64[ns]")
            .astype("int64")
            .reset_index(drop=True)
        )
        out["lat_key"] = np.rint(
            pd.to_numeric(frame["lat_rounded"], errors="coerce").to_numpy(dtype=float)
            / float(resolution)
        ).astype("int64")
        out["lon_key"] = np.rint(
            pd.to_numeric(frame["lon_rounded"], errors="coerce").to_numpy(dtype=float)
            / float(resolution)
        ).astype("int64")
        out["country"] = frame["country"].astype(str).reset_index(drop=True)
        out["is_fire"] = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).reset_index(drop=True)
        out["eval_weight"] = np.round(
            pd.to_numeric(frame["eval_weight"], errors="coerce").to_numpy(dtype=float),
            3,
        )
        return out

    feature_keys = key_frame(features)
    chunk_keys = key_frame(chunk_rows)
    if len(feature_keys) == len(chunk_keys):
        return feature_keys.equals(chunk_keys)
    if len(feature_keys) > len(chunk_keys):
        return False

    # Feature generation can deterministically drop rows after land/sea masking.
    # It can also reorder rows during merges. A cache is still valid when its
    # anchor rows are an order-independent subset of the current deployment
    # chunk; the cached rows carry their own coordinates, labels, and weights.
    # Guard against the old exact-float label-join caches by requiring the cache
    # to retain a substantial share of positive rows when the chunk has labels.
    feature_index = pd.MultiIndex.from_frame(feature_keys)
    chunk_index = pd.MultiIndex.from_frame(chunk_keys)
    feature_positive_count = int(feature_keys["is_fire"].sum())
    chunk_positive_count = int(chunk_keys["is_fire"].sum())
    if chunk_positive_count > 0 and feature_positive_count < max(1, int(math.ceil(chunk_positive_count * 0.5))):
        return False
    if feature_index.is_unique and chunk_index.is_unique:
        return bool(feature_index.isin(chunk_index).all())
    feature_counts = feature_index.value_counts()
    chunk_counts = chunk_index.value_counts()
    aligned_chunk_counts = chunk_counts.reindex(feature_counts.index, fill_value=0)
    return bool((feature_counts <= aligned_chunk_counts).all())


def _features_for_chunk(
    chunk_rows: pd.DataFrame,
    *,
    config: Any,
    output_dir: Path,
    split_name: str,
    country: str | None,
    period: str,
    feature_config: dict[str, Any] | str | Path,
    feature_columns: list[str],
    test_mode: bool = True,
) -> pd.DataFrame:
    cache_enabled = bool(getattr(config, "cache_full_grid_features", False))
    cache_path = _feature_cache_path(output_dir, split_name, country, period)
    if cache_enabled and cache_path.exists():
        logging.info("Loading cached deployment features from %s", cache_path)
        cached = add_calendar_context_features(pd.read_parquet(cache_path))
        if _feature_cache_matches_chunk(
            cached,
            chunk_rows,
            resolution=float(getattr(config, "deployment_grid_resolution", 0.1) or 0.1),
        ):
            return cached
        logging.info(
            "Cached deployment features at %s do not match the current grid rows; regenerating.",
            cache_path,
        )

    features = _make_features_for_chunk(
        chunk_rows,
        feature_config=feature_config,
        feature_columns=feature_columns,
        test_mode=test_mode,
    )
    if cache_enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path, index=False)
    return features


def _prediction_frame_from_features(
    features: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    split_name: str,
    feature_columns: list[str],
    predict_raw_fn: PredictionFn,
    feature_config_path: Path | None,
    target_config_path: Path | None,
    model_path: Path | str | None,
    calibration_method: str,
) -> pd.DataFrame:
    X = _align_feature_columns(features, feature_columns)
    raw, prob_raw, source = _coerce_prediction_result(predict_raw_fn(X), len(features))
    keep_cols = [col for col in REQUIRED_DEPLOYMENT_COLUMNS if col in features.columns]
    pred = features[keep_cols].copy()
    if "is_fire" not in pred.columns and "count" in features.columns:
        pred["is_fire"] = (pd.to_numeric(features["count"], errors="coerce").fillna(0) > 0).astype(np.int8)
    if "eval_weight" not in pred.columns:
        pred["eval_weight"] = 1.0
    if "month" not in pred.columns:
        pred["month"] = pd.to_datetime(pred["datetime"]).dt.month
    if "year" not in pred.columns:
        pred["year"] = pd.to_datetime(pred["datetime"]).dt.year
    pred["model_name"] = model_name
    pred["model_type"] = model_type
    pred["split_name"] = split_name
    pred["raw_score"] = raw.astype(np.float32)
    pred["prob_raw"] = prob_raw.astype(np.float32)
    pred["raw_score_source"] = source
    pred["feature_config_path"] = str(feature_config_path) if feature_config_path is not None else None
    pred["target_config_path"] = str(target_config_path) if target_config_path is not None else None
    pred["model_path"] = str(model_path) if model_path is not None else None
    pred["calibration_method"] = calibration_method
    return pred


def _generate_predictions_for_split(
    *,
    config: Any,
    feature_config: dict[str, Any] | str | Path,
    feature_config_payload: dict[str, Any],
    target_config_payload: dict[str, Any],
    feature_columns: list[str],
    predict_raw_fn: PredictionFn,
    model_name: str,
    model_type: str,
    split_name: str,
    output_dir: Path,
    model_path: Path | str | None,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    feature_config_path = getattr(config, "feature_config", None)
    target_config_path = getattr(config, "target_config", None)
    for chunk in iter_deployment_grid_chunks(
        config=config,
        feature_config=feature_config_payload,
        target_config=target_config_payload,
        split_name=split_name,
    ):
        if chunk.rows.empty:
            continue
        logging.info(
            "Full-grid %s %s chunk %s/%s rows=%d",
            model_name,
            split_name,
            chunk.country,
            chunk.period,
            len(chunk.rows),
        )
        features = _features_for_chunk(
            chunk.rows,
            config=config,
            output_dir=output_dir,
            split_name=split_name,
            country=chunk.country,
            period=chunk.period,
            feature_config=feature_config,
            feature_columns=feature_columns,
            test_mode=True,
        )
        if features.empty:
            continue
        pred = _prediction_frame_from_features(
            features,
            model_name=model_name,
            model_type=model_type,
            split_name=split_name,
            feature_columns=feature_columns,
            predict_raw_fn=predict_raw_fn,
            feature_config_path=Path(feature_config_path) if feature_config_path else None,
            target_config_path=Path(target_config_path) if target_config_path else None,
            model_path=model_path,
            calibration_method=str(getattr(config, "calibration_method", "platt_month")),
        )
        chunks.append(pred)

    if not chunks:
        raise RuntimeError(f"No deployment-grid prediction rows were generated for split={split_name}.")
    frame = pd.concat(chunks, ignore_index=True)
    pred_path = (
        primary_dir(output_dir)
        / "predictions"
        / f"{safe_slug(model_name)}_{safe_slug(split_name)}_predictions.parquet"
    )
    if bool(getattr(config, "save_full_grid_predictions", True)):
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(pred_path, index=False)
    return frame


def _metric_row(
    frame: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    feature_set: str,
    region: str,
    region_display: str,
    split_label: str,
    calibration_method: str,
    test_universe: str,
    test_period: str,
    error_config: Any | None = None,
    error_seed: int = 0,
) -> dict[str, Any]:
    y = frame["is_fire"].astype(int).to_numpy()
    p = pd.to_numeric(frame["prob_calibrated"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    count = expected_observed_count_ratio(y, p, w)
    slope = calibration_slope_intercept(y, p, w)
    f1 = max_weighted_f1(y, p, w)
    average_precision = weighted_average_precision(y, p, w)
    roc_auc = weighted_roc_auc(y, p, w)
    brier = weighted_brier_score(y, p, w)
    reliability = _reliability_values(y, p, w, error_config)
    row = {
        "Model": model_name,
        "model": model_name,
        "model_name": model_name,
        "model_type": model_type,
        "Feature set": feature_set,
        "feature_set": feature_set,
        "Region": region_display,
        "region": region,
        "region_display": region_display,
        "split": split_label,
        "evaluation_type": "primary_full_grid_calibrated",
        "is_primary": True,
        "calibration_method": calibration_method,
        "test_universe": test_universe,
        "test_period": test_period,
        "support": int(len(frame)),
        "weighted_support": float(np.sum(w)),
        "positives": float(np.sum(y * w)),
        "observed_prevalence": weighted_prevalence(y, w),
        "mean_calibrated_predicted_probability": float(np.average(p, weights=w)) if len(frame) else None,
        "PR-AUC": average_precision,
        "average_precision": average_precision,
        "ROC-AUC": roc_auc,
        "roc_auc": roc_auc,
        "weighted_brier_score": brier,
        "Brier": brier,
        "weighted_logloss": weighted_log_loss(y, p, w),
        **reliability,
        "daily_expected_observed_count_mae": daily_expected_observed_mae(frame),
        **f1,
        **slope,
        **count,
    }
    if error_config is not None:
        row.update(
            _bootstrap_array_metric_errors(
                y,
                p,
                w,
                config=error_config,
                seed=error_seed,
                metric_names=[
                    "observed_prevalence",
                    "mean_score",
                    "average_precision",
                    "roc_auc",
                    "max_f1",
                    "weighted_brier_score",
                    "weighted_logloss",
                    "expected_observed_count_ratio",
                ],
            )
        )
        if "mean_score_error" in row:
            row["mean_calibrated_predicted_probability_error"] = row.pop("mean_score_error")
    return row


def _regional_metric_rows(
    frame: pd.DataFrame,
    *,
    regions: Iterable[Any] | None,
    model_name: str,
    model_type: str,
    feature_set: str,
    calibration_method: str,
    test_universe: str,
    test_period: str,
    error_config: Any | None = None,
) -> pd.DataFrame:
    rows = [
        _metric_row(
            frame,
            model_name=model_name,
            model_type=model_type,
            feature_set=feature_set,
            region="global",
            region_display="Global",
            split_label="test",
            calibration_method=calibration_method,
            test_universe=test_universe,
            test_period=test_period,
            error_config=error_config,
            error_seed=_stable_seed(model_name, 10),
        )
    ]
    if regions is not None and {"lat_rounded", "lon_rounded"}.issubset(frame.columns):
        for region_idx, region in enumerate(regions):
            mask = region.mask(frame)
            if np.asarray(mask).any():
                rows.append(
                    _metric_row(
                        frame.loc[mask],
                        model_name=model_name,
                        model_type=model_type,
                        feature_set=feature_set,
                        region=region.name,
                        region_display=region.display_name,
                        split_label="test",
                        calibration_method=calibration_method,
                        test_universe=test_universe,
                        test_period=test_period,
                        error_config=error_config,
                        error_seed=_stable_seed(f"{model_name}:{region.name}", 20 + region_idx),
                    )
                )
    return pd.DataFrame(rows)


def _by_year_rows(base: pd.DataFrame, *, error_config: Any | None = None, model_name: str, **kwargs: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "year" not in base.columns:
        base = base.copy()
        base["year"] = pd.to_datetime(base["datetime"]).dt.year
    for year, group in base.groupby("year", observed=True):
        row = _metric_row(
            group,
            split_label=f"test_{int(year)}",
            test_period=str(int(year)),
            model_name=model_name,
            error_config=error_config,
            error_seed=_stable_seed(f"{model_name}:{int(year)}", 100),
            **kwargs,
        )
        row["period"] = str(int(year))
        rows.append(row)
    return pd.DataFrame(rows)


def _count_calibration_table(
    frame: pd.DataFrame,
    group_cols: list[str],
    *,
    model_name: str,
    model_type: str,
    table_type: str,
) -> pd.DataFrame:
    work = frame.copy()
    work["expected"] = pd.to_numeric(work["prob_calibrated"], errors="coerce") * pd.to_numeric(
        work["eval_weight"], errors="coerce"
    )
    work["observed"] = pd.to_numeric(work["is_fire"], errors="coerce") * pd.to_numeric(
        work["eval_weight"], errors="coerce"
    )
    rows = []
    for keys, group in work.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        data = {
            "model_name": model_name,
            "model_type": model_type,
            "table_type": table_type,
            "expected_fire_positive_grid_cells": float(group["expected"].sum()),
            "observed_fire_positive_grid_cells": float(group["observed"].sum()),
        }
        for col, value in zip(group_cols, keys):
            data[col] = value
        observed = data["observed_fire_positive_grid_cells"]
        data["expected_observed_count_ratio"] = data["expected_fire_positive_grid_cells"] / observed if observed > 0 else None
        rows.append(data)
    return pd.DataFrame(rows)


def _prevalence_audit_table(
    frame: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    config: Any,
) -> pd.DataFrame:
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    w = pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    rows = [
        {
            "model_name": model_name,
            "model_type": model_type,
            "denominator": "deployment_weighted_grid_cell_days",
            "support": int(len(frame)),
            "denominator_count": float(np.sum(w)),
            "event_count": float(np.sum(y * w)),
            "event_rate": float(np.sum(y * w) / np.sum(w)) if np.sum(w) > 0 else None,
            "notes": "Uses eval_weight to represent the configured deployment grid or weighted-grid sample.",
        },
        {
            "model_name": model_name,
            "model_type": model_type,
            "denominator": "evaluated_rows_unweighted",
            "support": int(len(frame)),
            "denominator_count": float(len(frame)),
            "event_count": float(np.sum(y)),
            "event_rate": float(np.mean(y)) if len(y) else None,
            "notes": "Unweighted rows actually scored after chunking, sampling, and feature generation.",
        },
    ]
    trials = _error_trials(config)
    if trials > 1:
        for row in rows:
            values: list[float] = []
            weight_for_rate = w if row["denominator"] == "deployment_weighted_grid_cell_days" else np.ones_like(w)
            for idx in _bootstrap_index_samples(
                y,
                trials=trials,
                sample_size=_error_sample_size(config),
                seed=_stable_seed(f"{model_name}:{row['denominator']}", 300),
            ):
                denom = float(np.sum(weight_for_rate[idx]))
                if denom > 0:
                    values.append(float(np.sum(y[idx] * weight_for_rate[idx]) / denom))
            row["event_rate_error"] = _std_or_none(values)
    else:
        for row in rows:
            row["event_rate_error"] = None
    return pd.DataFrame(rows)


def _count_correction_values(frame: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    w = pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    raw = pd.to_numeric(frame.get("prob_raw", frame["prob_calibrated"]), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    calibrated = pd.to_numeric(frame["prob_calibrated"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_counts = expected_observed_count_ratio(y, raw, w)
    calibrated_counts = expected_observed_count_ratio(y, calibrated, w)
    observed = calibrated_counts.get("observed_fire_positive_grid_cells")
    return {
        "raw_mean_predicted_probability": float(np.average(raw, weights=w)) if len(raw) else None,
        "calibrated_mean_predicted_probability": float(np.average(calibrated, weights=w)) if len(calibrated) else None,
        "raw_expected_fire_positive_grid_cells": raw_counts.get("expected_fire_positive_grid_cells"),
        "calibrated_expected_fire_positive_grid_cells": calibrated_counts.get("expected_fire_positive_grid_cells"),
        "observed_fire_positive_grid_cells": observed,
        "raw_expected_observed_count_ratio": raw_counts.get("expected_observed_count_ratio"),
        "calibrated_expected_observed_count_ratio": calibrated_counts.get("expected_observed_count_ratio"),
    }


def _count_correction_table(
    frame: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    calibration_method: str,
    config: Any,
) -> pd.DataFrame:
    row = {
        "model_name": model_name,
        "model_type": model_type,
        "calibration_method": calibration_method,
        **_count_correction_values(frame),
    }
    raw_ratio = row.get("raw_expected_observed_count_ratio")
    calibrated_ratio = row.get("calibrated_expected_observed_count_ratio")
    row["eo_ratio_reduction_factor"] = (
        float(raw_ratio) / float(calibrated_ratio)
        if raw_ratio is not None
        and calibrated_ratio is not None
        and math.isfinite(float(raw_ratio))
        and math.isfinite(float(calibrated_ratio))
        and float(calibrated_ratio) != 0
        else None
    )
    trials = _error_trials(config)
    metric_names = ["raw_expected_observed_count_ratio", "calibrated_expected_observed_count_ratio"]
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    if trials > 1:
        y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
        for idx in _bootstrap_index_samples(
            y,
            trials=trials,
            sample_size=_error_sample_size(config),
            seed=_stable_seed(model_name, 400),
        ):
            sample = frame.iloc[idx]
            metrics = _count_correction_values(sample)
            for name in metric_names:
                value = metrics.get(name)
                if value is not None and math.isfinite(float(value)):
                    values[name].append(float(value))
    for name in metric_names:
        row[f"{name}_error"] = _std_or_none(values[name])
    return pd.DataFrame([row])


def _risk_concentration_values(
    y_true: np.ndarray | pd.Series,
    score: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None,
    top_fraction: float,
) -> dict[str, Any]:
    y, p, w = _as_metric_arrays(y_true, score, sample_weight)
    if len(y) == 0:
        return {
            "top_fraction_requested": float(top_fraction),
            "top_fraction_observed": None,
            "top_weighted_support": None,
            "captured_fire_positive_grid_cells": None,
            "observed_fire_positive_grid_cells": None,
            "recall_at_top_q": None,
            "precision_at_top_q": None,
            "lift_at_q": None,
            "average_precision": None,
            "observed_prevalence": None,
            "ap_lift": None,
        }
    total_weight = float(np.sum(w))
    total_pos = float(np.sum(y * w))
    order = np.argsort(-p, kind="mergesort")
    w_sorted = w[order]
    y_sorted = y[order]
    cutoff = max(0.0, min(1.0, float(top_fraction))) * total_weight
    cumulative = np.cumsum(w_sorted)
    if cutoff <= 0:
        top_n = 1
    else:
        top_n = int(np.searchsorted(cumulative, cutoff, side="left")) + 1
    top_n = max(1, min(top_n, len(order)))
    top_weight = float(np.sum(w_sorted[:top_n]))
    captured = float(np.sum(y_sorted[:top_n] * w_sorted[:top_n]))
    actual_fraction = top_weight / total_weight if total_weight > 0 else None
    recall = captured / total_pos if total_pos > 0 else None
    precision = captured / top_weight if top_weight > 0 else None
    lift = recall / actual_fraction if recall is not None and actual_fraction and actual_fraction > 0 else None
    prevalence = weighted_prevalence(y, w)
    ap = weighted_average_precision(y, p, w)
    return {
        "top_fraction_requested": float(top_fraction),
        "top_fraction_observed": actual_fraction,
        "top_weighted_support": top_weight,
        "captured_fire_positive_grid_cells": captured,
        "observed_fire_positive_grid_cells": total_pos,
        "recall_at_top_q": recall,
        "precision_at_top_q": precision,
        "lift_at_q": lift,
        "average_precision": ap,
        "observed_prevalence": prevalence,
        "ap_lift": float(ap / prevalence) if ap is not None and prevalence and prevalence > 0 else None,
    }


def _risk_concentration_table(
    frame: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    config: Any,
) -> pd.DataFrame:
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(frame["prob_calibrated"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w = pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    trials = _error_trials(config)
    for q_idx, fraction in enumerate(RISK_TOP_FRACTIONS):
        row = {
            "model_name": model_name,
            "model_type": model_type,
            "q_label": f"top_{100.0 * fraction:g}pct",
            **_risk_concentration_values(y, p, w, fraction),
        }
        metric_names = ["recall_at_top_q", "precision_at_top_q", "lift_at_q", "ap_lift"]
        values: dict[str, list[float]] = {name: [] for name in metric_names}
        if trials > 1:
            for idx in _bootstrap_index_samples(
                y,
                trials=trials,
                sample_size=_error_sample_size(config),
                seed=_stable_seed(f"{model_name}:risk:{fraction}", 500 + q_idx),
            ):
                metrics = _risk_concentration_values(y[idx], p[idx], w[idx], fraction)
                for name in metric_names:
                    value = metrics.get(name)
                    if value is not None and math.isfinite(float(value)):
                        values[name].append(float(value))
        for name in metric_names:
            row[f"{name}_error"] = _std_or_none(values[name])
        rows.append(row)
    return pd.DataFrame(rows)


def _date_lat_lon_keys(frame: pd.DataFrame, *, resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    date_key = pd.to_datetime(frame["datetime"]).values.astype("datetime64[D]").astype("int64")
    lat_key = np.rint(pd.to_numeric(frame["lat_rounded"], errors="coerce").to_numpy(dtype=float) / resolution).astype("int64")
    lon_key = np.rint(pd.to_numeric(frame["lon_rounded"], errors="coerce").to_numpy(dtype=float) / resolution).astype("int64")
    return date_key, lat_key, lon_key


def _neighborhood_positive_labels(frame: pd.DataFrame, *, resolution: float, radius_cells: int = 1) -> np.ndarray:
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    if not y.any() or not {"datetime", "lat_rounded", "lon_rounded"}.issubset(frame.columns):
        return y
    date_key, lat_key, lon_key = _date_lat_lon_keys(frame, resolution=resolution)
    positives = pd.MultiIndex.from_arrays([date_key[y == 1], lat_key[y == 1], lon_key[y == 1]])
    expanded = np.zeros(len(frame), dtype=bool)
    for dlat in range(-radius_cells, radius_cells + 1):
        for dlon in range(-radius_cells, radius_cells + 1):
            probe = pd.MultiIndex.from_arrays([date_key, lat_key + dlat, lon_key + dlon])
            expanded |= probe.isin(positives)
    return expanded.astype(int)


def _spatial_metric_row(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray,
    *,
    model_name: str,
    model_type: str,
    scale: str,
    spatial_resolution_degrees: float | None,
    config: Any,
    seed: int,
    observed_fire_positive_grid_cells: float | None = None,
    expected_fire_positive_grid_cells: float | None = None,
) -> dict[str, Any]:
    metrics = _binary_metric_values(y_true, score, sample_weight)
    row = {
        "model_name": model_name,
        "model_type": model_type,
        "scale": scale,
        "spatial_resolution_degrees": spatial_resolution_degrees,
        "support_units": metrics["support"],
        "weighted_support": metrics["weighted_support"],
        "positive_units": metrics["positives"],
        "observed_prevalence": metrics["observed_prevalence"],
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "max_f1": metrics["max_f1"],
        "precision_at_max_f1": metrics["precision_at_max_f1"],
        "recall_at_max_f1": metrics["recall_at_max_f1"],
        "threshold_at_max_f1": metrics["threshold_at_max_f1"],
        "weighted_brier_score": metrics["weighted_brier_score"],
        "weighted_logloss": metrics["weighted_logloss"],
        "observed_fire_positive_grid_cells": (
            observed_fire_positive_grid_cells
            if observed_fire_positive_grid_cells is not None
            else metrics["observed_fire_positive_grid_cells"]
        ),
        "expected_fire_positive_grid_cells": (
            expected_fire_positive_grid_cells
            if expected_fire_positive_grid_cells is not None
            else metrics["expected_fire_positive_grid_cells"]
        ),
        "expected_observed_count_ratio": metrics["expected_observed_count_ratio"],
    }
    row.update(
        _bootstrap_array_metric_errors(
            y_true,
            score,
            sample_weight,
            config=config,
            seed=seed,
            metric_names=["observed_prevalence", "average_precision", "roc_auc", "max_f1"],
        )
    )
    return row


def _coarse_spatial_frame(frame: pd.DataFrame, *, resolution: float) -> pd.DataFrame:
    work = frame.copy()
    work["date_key"] = pd.to_datetime(work["datetime"]).values.astype("datetime64[D]").astype("int64")
    work["lat_bin"] = np.floor(pd.to_numeric(work["lat_rounded"], errors="coerce").astype(float) / resolution) * resolution
    work["lon_bin"] = np.floor(pd.to_numeric(work["lon_rounded"], errors="coerce").astype(float) / resolution) * resolution
    work["weight"] = pd.to_numeric(work["eval_weight"], errors="coerce").fillna(1.0).astype(float)
    work["observed"] = pd.to_numeric(work["is_fire"], errors="coerce").fillna(0).astype(float) * work["weight"]
    work["expected"] = pd.to_numeric(work["prob_calibrated"], errors="coerce").fillna(0.0).astype(float) * work["weight"]
    grouped = (
        work.groupby(["date_key", "lat_bin", "lon_bin"], observed=True, dropna=False)
        .agg(
            support_units=("is_fire", "size"),
            weighted_support=("weight", "sum"),
            observed_fire_positive_grid_cells=("observed", "sum"),
            expected_fire_positive_grid_cells=("expected", "sum"),
        )
        .reset_index()
    )
    grouped["label"] = (grouped["observed_fire_positive_grid_cells"] > 0).astype(int)
    grouped["score"] = np.divide(
        grouped["expected_fire_positive_grid_cells"].to_numpy(dtype=float),
        grouped["weighted_support"].to_numpy(dtype=float),
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["weighted_support"].to_numpy(dtype=float) > 0,
    )
    return grouped


def _spatial_scale_table(
    frame: pd.DataFrame,
    *,
    model_name: str,
    model_type: str,
    config: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(frame["prob_calibrated"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w = pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    base_resolution = float(getattr(config, "deployment_grid_resolution", 0.1) or 0.1)
    rows.append(
        _spatial_metric_row(
            y,
            p,
            w,
            model_name=model_name,
            model_type=model_type,
            scale="exact_0.1_degree_cell_day",
            spatial_resolution_degrees=base_resolution,
            config=config,
            seed=_stable_seed(model_name, 600),
        )
    )
    if {"datetime", "lat_rounded", "lon_rounded"}.issubset(frame.columns):
        neigh_y = _neighborhood_positive_labels(frame, resolution=base_resolution, radius_cells=1)
        rows.append(
            _spatial_metric_row(
                neigh_y,
                p,
                w,
                model_name=model_name,
                model_type=model_type,
                scale="neighborhood_3x3_cell_day",
                spatial_resolution_degrees=base_resolution * 3.0,
                config=config,
                seed=_stable_seed(model_name, 700),
            )
        )
        for idx, coarse_resolution in enumerate(SPATIAL_COARSE_RESOLUTIONS):
            coarse = _coarse_spatial_frame(frame, resolution=coarse_resolution)
            rows.append(
                _spatial_metric_row(
                    coarse["label"].to_numpy(dtype=int),
                    coarse["score"].to_numpy(dtype=float),
                    coarse["weighted_support"].to_numpy(dtype=float),
                    model_name=model_name,
                    model_type=model_type,
                    scale=f"coarse_{coarse_resolution:g}_degree_cell_day",
                    spatial_resolution_degrees=float(coarse_resolution),
                    config=config,
                    seed=_stable_seed(model_name, 800 + idx),
                    observed_fire_positive_grid_cells=float(coarse["observed_fire_positive_grid_cells"].sum()),
                    expected_fire_positive_grid_cells=float(coarse["expected_fire_positive_grid_cells"].sum()),
                )
            )
    return pd.DataFrame(rows)


def _write_metric_tables(
    *,
    output_dir: Path,
    test_predictions: pd.DataFrame,
    model_name: str,
    model_type: str,
    feature_set: str,
    regions: Iterable[Any] | None,
    config: Any,
) -> dict[str, Any]:
    out = primary_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    method = str(getattr(config, "calibration_method", "platt_month"))
    universe = str(getattr(config, "deployment_grid_universe", "land_or_burnable"))
    start = getattr(config, "test_start_date", None) or "2022-01-01"
    end = getattr(config, "test_end_date", None) or "2025-10-02"
    period = f"{start} to {end}"
    comparison = _regional_metric_rows(
        test_predictions,
        regions=regions,
        model_name=model_name,
        model_type=model_type,
        feature_set=feature_set,
        calibration_method=method,
        test_universe=universe,
        test_period=period,
        error_config=config,
    )
    _upsert_csv(out / "model_comparison.csv", comparison, ["model_name", "region", "split"])

    by_year = _by_year_rows(
        test_predictions,
        model_name=model_name,
        model_type=model_type,
        feature_set=feature_set,
        region="global",
        region_display="Global",
        calibration_method=method,
        test_universe=universe,
        error_config=config,
    )
    if not by_year.empty:
        _upsert_csv(out / "model_comparison_by_year.csv", by_year, ["model_name", "region", "period"])

    prob_cols = [
        "model_name",
        "model_type",
        "evaluation_type",
        "is_primary",
        "calibration_method",
        "test_universe",
        "test_period",
        "support",
        "weighted_support",
        "positives",
        "observed_prevalence",
        "mean_calibrated_predicted_probability",
        "average_precision",
        "roc_auc",
        "max_f1",
        "precision_at_max_f1",
        "recall_at_max_f1",
        "threshold_at_max_f1",
        "predicted_positive_grid_cells_at_max_f1",
        "weighted_brier_score",
        "weighted_brier_score_error",
        "weighted_logloss",
        "weighted_logloss_error",
        "reliability_ece",
        "reliability_mce",
        "reliability_rmse",
        "calibration_intercept",
        "calibration_slope",
        "expected_observed_count_ratio",
        "expected_observed_count_ratio_error",
        "expected_fire_positive_grid_cells",
        "observed_fire_positive_grid_cells",
        "daily_expected_observed_count_mae",
        "observed_prevalence_error",
        "mean_calibrated_predicted_probability_error",
        "average_precision_error",
        "roc_auc_error",
        "max_f1_error",
    ]
    probability = comparison[comparison["region"].eq("global")].copy()
    for col in prob_cols:
        if col not in probability.columns:
            probability[col] = np.nan
    _upsert_csv(out / "probability_metrics.csv", probability[prob_cols], ["model_name"])

    global_metrics = probability.iloc[0].to_dict() if not probability.empty else {}
    registry_metrics = []
    for metric_name in [
        "average_precision",
        "roc_auc",
        "max_f1",
        "precision_at_max_f1",
        "recall_at_max_f1",
        "threshold_at_max_f1",
        "weighted_brier_score",
        "weighted_logloss",
        "reliability_ece",
        "reliability_mce",
        "reliability_rmse",
        "calibration_intercept",
        "calibration_slope",
        "expected_observed_count_ratio",
        "daily_expected_observed_count_mae",
    ]:
        registry_metrics.append(
            {
                "model_name": model_name,
                "model_type": model_type,
                "evaluation_type": "primary_full_grid_calibrated",
                "is_primary": True,
                "calibration_method": method,
                "test_universe": universe,
                "test_period": period,
                "metric_name": metric_name,
                "metric_value": global_metrics.get(metric_name),
            }
        )
    _upsert_csv(out / "experiment_registry.csv", pd.DataFrame(registry_metrics), ["model_name", "metric_name"])

    reliability = make_reliability_bins(
        test_predictions["is_fire"],
        test_predictions["prob_calibrated"],
        test_predictions["eval_weight"],
        n_bins=int(getattr(config, "n_reliability_bins", 20)),
        strategy=str(getattr(config, "reliability_binning", "equal_count")),
    )
    reliability.insert(0, "model_type", model_type)
    reliability.insert(0, "model_name", model_name)
    _upsert_csv(out / "reliability_bins.csv", reliability, ["model_name", "bin"])

    monthly = _count_calibration_table(test_predictions, ["month"], model_name=model_name, model_type=model_type, table_type="month")
    country_cols = ["country"] if "country" in test_predictions.columns else ["region"]
    country = _count_calibration_table(test_predictions, country_cols, model_name=model_name, model_type=model_type, table_type="country")
    _upsert_csv(out / "monthly_count_calibration.csv", monthly, ["model_name", "month"])
    _upsert_csv(out / "country_count_calibration.csv", country, ["model_name", country_cols[0]])
    region = comparison[
        [
            "model_name",
            "model_type",
            "region",
            "region_display",
            "expected_fire_positive_grid_cells",
            "observed_fire_positive_grid_cells",
            "expected_observed_count_ratio",
        ]
    ].copy()
    _upsert_csv(out / "region_count_calibration.csv", region, ["model_name", "region"])

    prevalence = _prevalence_audit_table(test_predictions, model_name=model_name, model_type=model_type, config=config)
    _upsert_csv(out / "prevalence_audit.csv", prevalence, ["model_name", "denominator"])

    risk = _risk_concentration_table(test_predictions, model_name=model_name, model_type=model_type, config=config)
    _upsert_csv(out / "risk_concentration.csv", risk, ["model_name", "q_label"])

    correction = _count_correction_table(
        test_predictions,
        model_name=model_name,
        model_type=model_type,
        calibration_method=method,
        config=config,
    )
    _upsert_csv(out / "count_correction.csv", correction, ["model_name"])

    spatial = _spatial_scale_table(test_predictions, model_name=model_name, model_type=model_type, config=config)
    _upsert_csv(out / "spatial_scale_evaluation.csv", spatial, ["model_name", "scale"])
    write_full_grid_model_selection(output_dir, config)

    global_row = comparison[comparison["region"].eq("global")].iloc[0].to_dict()
    return global_row


def refresh_primary_comparison_tables(output_dir: Path) -> bool:
    """Deprecated compatibility hook.

    Full-grid calibrated rows intentionally stay under
    ``primary_full_grid_calibrated`` so sampled model-comparison experiments do
    not mix base rates. Kept as a no-op for older wrappers that imported it.
    """

    return False


def evaluate_model_full_grid_calibrated(
    *,
    model: Any | None = None,
    model_name: str,
    model_type: str,
    feature_columns: list[str],
    categorical_columns: list[str] | None = None,
    config: Any,
    split_info: dict[str, Any] | None = None,
    output_dir: Path,
    predict_raw_fn: PredictionFn | None = None,
    feature_config: dict[str, Any] | str | Path | None = None,
    target_config: dict[str, Any] | None = None,
    regions: Iterable[Any] | None = None,
    model_path: Path | str | None = None,
    feature_set: str = "full features",
) -> dict[str, Any]:
    if not bool(getattr(config, "run_full_grid_evaluation", True)):
        return {}
    if predict_raw_fn is None:
        raise ValueError("predict_raw_fn is required for revision full-grid calibrated evaluation.")
    if feature_config is None:
        feature_config = getattr(config, "feature_config", None)
    if feature_config is None:
        raise ValueError("feature_config is required.")
    if isinstance(feature_config, (str, Path)):
        import yaml

        with Path(feature_config).open("r", encoding="utf-8") as handle:
            feature_payload = yaml.safe_load(handle) or {}
    else:
        feature_payload = dict(feature_config)
    target_payload = dict(target_config or {})

    out = primary_dir(output_dir)
    (out / "calibrators").mkdir(parents=True, exist_ok=True)
    method = str(getattr(config, "calibration_method", "platt_month"))
    cal = _generate_predictions_for_split(
        config=config,
        feature_config=feature_config,
        feature_config_payload=feature_payload,
        target_config_payload=target_payload,
        feature_columns=feature_columns,
        predict_raw_fn=predict_raw_fn,
        model_name=model_name,
        model_type=model_type,
        split_name="calibration",
        output_dir=output_dir,
        model_path=model_path,
    )
    calibrator = fit_calibrator(cal, method)
    calibrator_path = out / "calibrators" / f"{safe_slug(model_name)}_{safe_slug(method)}.joblib"
    save_calibrator(calibrator, calibrator_path)
    metadata = build_calibration_metadata(
        frame=cal,
        model_name=model_name,
        model_type=model_type,
        calibrator_type=method,
        calibration_start_date=getattr(config, "calibration_start_date", None) or "2021-01-01",
        calibration_end_date=getattr(config, "calibration_end_date", None) or "2021-12-31",
        test_start_date=getattr(config, "test_start_date", None) or "2022-01-01",
        test_end_date=getattr(config, "test_end_date", None) or "2025-10-02",
        calibrator_path=calibrator_path,
        calibrator=calibrator,
    )
    metadata_path = out / "calibration_metadata" / f"{safe_slug(model_name)}.json"
    write_calibration_metadata(metadata_path, metadata)

    combined_meta_path = out / "calibration_metadata.json"
    existing_meta: dict[str, Any] = {}
    if combined_meta_path.exists():
        existing_meta = json.loads(combined_meta_path.read_text(encoding="utf-8"))
    existing_meta[model_name] = json.loads(metadata_path.read_text(encoding="utf-8"))
    combined_meta_path.write_text(json.dumps(existing_meta, indent=2, default=_json_default), encoding="utf-8")

    test = _generate_predictions_for_split(
        config=config,
        feature_config=feature_config,
        feature_config_payload=feature_payload,
        target_config_payload=target_payload,
        feature_columns=feature_columns,
        predict_raw_fn=predict_raw_fn,
        model_name=model_name,
        model_type=model_type,
        split_name="test",
        output_dir=output_dir,
        model_path=model_path,
    )
    test["prob_calibrated"] = apply_calibrator(calibrator, test)
    if bool(getattr(config, "save_calibrated_predictions", True)):
        path = out / "predictions" / f"{safe_slug(model_name)}_test_predictions.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        test.to_parquet(path, index=False)

    metrics = _write_metric_tables(
        output_dir=output_dir,
        test_predictions=test,
        model_name=model_name,
        model_type=model_type,
        feature_set=feature_set,
        regions=regions,
        config=config,
    )
    stale_failure = failure_path(output_dir, model_name)
    if stale_failure.exists():
        stale_failure.unlink()
    return metrics


def write_full_grid_failure(
    output_dir: Path,
    *,
    model_name: str,
    model_type: str,
    exc: BaseException,
) -> Path:
    path = failure_path(output_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "model_type": model_type,
        "evaluation_type": "primary_full_grid_calibrated",
        "status": "failed",
        "reason": str(exc),
        "traceback": traceback.format_exc(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
