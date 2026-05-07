#!/usr/bin/env python3
"""Compare ERA5-trained, ECMWF-trained, and mixed-source CatBoost models.

The comparison keeps validation and test data fixed to the operational
ECMWF/SEAS5 feature matrix.  ERA5 is used to rebuild the dynamic climate
features for the same training target rows over the common spatial footprint.
The mixed model duplicates each training target row once with ECMWF climate
features and once with ERA5 climate features.

No test-set threshold tuning is performed: thresholds are selected on the
ECMWF validation split and then applied to the ECMWF 2021-2025 test split.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_generation.prepare_climate_data import prepare_data
from src.revision_evaluation.filesystem import prune_empty_dirs
from src.revision_evaluation.tabular import (
    DATE_COLUMN,
    LAT_COLUMN,
    LON_COLUMN,
    TARGET_COLUMN,
    Region,
    compute_metric_errors,
    load_regions,
    model_feature_columns,
    normalize_cat_columns,
    positive_labels,
)


SEED = 42
N_DAYS = 128
CLIMATE_VARIABLES = ["t2m", "d2m", "tp", "stl1"]
LAGS = [7, 14, 30, 90, 120]
WINDOWS = [7, 14, 30, 90, 120]
SPANS = [7, 14, 30, 90, 120]
TREND_WINDOWS = [21, 90]
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_FEATURE_WEIGHTS = {
    "ecoregion_name": 0.4,
    "lat_rounded": 0.4,
    "lon_rounded": 0.4,
}


@dataclass
class Footprint:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    era5_min_time: pd.Timestamp
    era5_max_time: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return None if not math.isfinite(x) else x
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def safe_score(func, *args) -> float | None:
    try:
        value = float(func(*args))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, Any]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5, {"reason": "validation_has_one_class", "validation_f1": None}
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5, {"reason": "no_thresholds", "validation_f1": None}
    f1 = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if not np.isfinite(f1).any():
        return 0.5, {"reason": "no_finite_f1", "validation_f1": None}
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), {
        "reason": "validation_f1_max",
        "validation_f1": float(f1[idx]),
        "validation_precision": float(precision[idx]),
        "validation_recall": float(recall[idx]),
    }


def metric_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return {
            "support": 0,
            "positives": 0,
            "negatives": 0,
            "positive_rate": np.nan,
            "predicted_positives": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "average_precision": None,
            "roc_auc": None,
            "brier_score": None,
            "threshold": threshold,
        }
    y_pred = (y_prob >= threshold).astype(np.int8)
    both = len(np.unique(y_true)) == 2
    return {
        "support": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "predicted_positives": int(y_pred.sum()),
        "precision": safe_score(lambda a, b: precision_score(a, b, zero_division=0), y_true, y_pred),
        "recall": safe_score(lambda a, b: recall_score(a, b, zero_division=0), y_true, y_pred),
        "f1": safe_score(lambda a, b: f1_score(a, b, zero_division=0), y_true, y_pred),
        "average_precision": safe_score(average_precision_score, y_true, y_prob) if both else None,
        "roc_auc": safe_score(roc_auc_score, y_true, y_prob) if both else None,
        "brier_score": safe_score(brier_score_loss, y_true, y_prob) if both else None,
        "threshold": float(threshold),
    }


def metric_dict_with_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    trials: int,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    metrics = metric_dict(y_true, y_prob, threshold)
    metrics.update(
        compute_metric_errors(
            y_true,
            y_prob,
            threshold,
            trials=trials,
            sample_size=sample_size,
            seed=seed,
        )
    )
    return metrics


def catboost_pool(X: pd.DataFrame, y: np.ndarray | None, cat_features: list[str]) -> Pool:
    cats = [c for c in cat_features if c in X.columns]
    return Pool(X, label=y, cat_features=cats) if y is not None else Pool(X, cat_features=cats)


def predict_proba(model: CatBoostClassifier, X: pd.DataFrame, cat_features: list[str]) -> np.ndarray:
    return np.asarray(model.predict_proba(catboost_pool(X, None, cat_features)))[:, 1].astype(np.float32)


def catboost_feature_weights(feature_columns: list[str], catboost_config: dict[str, Any]) -> dict[str, float]:
    configured = (
        catboost_config.get("catboost_train", {})
        .get("features", {})
        .get("feature_weights", DEFAULT_FEATURE_WEIGHTS)
    )
    if not isinstance(configured, dict):
        configured = DEFAULT_FEATURE_WEIGHTS
    return {str(k): float(v) for k, v in configured.items() if str(k) in feature_columns}


def raw_era5_audit(raw_root: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for var_dir in sorted(raw_root.glob("*")):
        if not var_dir.is_dir():
            continue
        rows.append(
            {
                "variable_dir": var_dir.name,
                "nc_files": len(list(var_dir.glob("*.nc"))),
                "grib_files": len(list(var_dir.glob("*.grib"))),
                "sample_nc": next((p.name for p in sorted(var_dir.glob("*.nc"))), None),
                "sample_grib": next((p.name for p in sorted(var_dir.glob("*.grib"))), None),
            }
        )
    audit = {"raw_root": raw_root, "variables": rows, "sample_grib_checks": []}
    samples = [
        raw_root / "total_precipitation" / "total_precipitation_2025.grib",
        raw_root / "2m_dewpoint_temperature" / "2m_dewpoint_temperature_2025.grib",
        raw_root / "soil_temperature_level_1" / "soil_temperature_level_1_2025.grib",
    ]
    for sample in samples:
        if not sample.exists():
            continue
        cmd = ["timeout", "10s", "grib_ls", "-p", "shortName,date,time", str(sample)]
        completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        audit["sample_grib_checks"].append(
            {
                "path": str(sample),
                "returncode": int(completed.returncode),
                "output_excerpt": completed.stdout[:2000],
                "unreadable_messages": "ERROR: unreadable message" in completed.stdout,
            }
        )
    write_json(output_dir / "artifacts" / "raw_era5_audit.json", audit)
    return audit


def processed_era5_schema(
    era5_root: Path,
    *,
    year_start: int | None = None,
    year_end: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for var in CLIMATE_VARIABLES:
        paths_all = sorted((era5_root / var).glob(f"{var}_*.zarr"))
        paths = []
        for path in paths_all:
            try:
                year = int(path.stem.split("_")[-1])
            except Exception:
                year = None
            if year_start is not None and year is not None and year < year_start:
                continue
            if year_end is not None and year is not None and year > year_end:
                continue
            paths.append(path)
        lat_min = lat_max = lon_min = lon_max = None
        time_min = time_max = None
        years = []
        for path in paths:
            years.append(path.stem.split("_")[-1])
            ds = xr.open_zarr(path, chunks=None)
            time_coord = "time" if "time" in ds.coords else "valid_time"
            t0 = pd.Timestamp(ds[time_coord].values[0])
            t1 = pd.Timestamp(ds[time_coord].values[-1])
            la_min = float(np.nanmin(ds["latitude"].values))
            la_max = float(np.nanmax(ds["latitude"].values))
            lo_min = float(np.nanmin(ds["longitude"].values))
            lo_max = float(np.nanmax(ds["longitude"].values))
            ds.close()
            time_min = t0 if time_min is None else min(time_min, t0)
            time_max = t1 if time_max is None else max(time_max, t1)
            lat_min = la_min if lat_min is None else max(lat_min, la_min)
            lat_max = la_max if lat_max is None else min(lat_max, la_max)
            lon_min = lo_min if lon_min is None else max(lon_min, lo_min)
            lon_max = lo_max if lon_max is None else min(lon_max, lo_max)
        rows.append(
            {
                "variable": var,
                "zarr_files": len(paths),
                "years": ",".join(sorted(set(years))),
                "time_min": time_min,
                "time_max": time_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            }
        )
    return pd.DataFrame(rows)


def make_filtered_era5_root(source_root: Path, target_root: Path, years: range) -> Path:
    """Create a lightweight zarr directory view limited to the years used for training."""
    for var in CLIMATE_VARIABLES:
        (target_root / var).mkdir(parents=True, exist_ok=True)
        for old in (target_root / var).glob("*.zarr"):
            if old.is_symlink():
                old.unlink()
        for year in years:
            src = source_root / var / f"{var}_{year}.zarr"
            if not src.exists():
                raise FileNotFoundError(f"Missing processed ERA5 zarr for {var} {year}: {src}")
            dst = target_root / var / src.name
            if dst.exists() or dst.is_symlink():
                continue
            dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
    return target_root


def common_footprint(schema: pd.DataFrame, base_df: pd.DataFrame) -> Footprint:
    if schema.empty or schema["zarr_files"].min() <= 0:
        raise RuntimeError("Processed ERA5 zarr files are missing for at least one common climate variable.")
    lat_min = max(float(schema["lat_min"].max()), float(base_df[LAT_COLUMN].min()))
    lat_max = min(float(schema["lat_max"].min()), float(base_df[LAT_COLUMN].max()))
    lon_min = max(float(schema["lon_min"].max()), float(base_df[LON_COLUMN].min()))
    lon_max = min(float(schema["lon_max"].min()), float(base_df[LON_COLUMN].max()))
    t_min = max(pd.to_datetime(schema["time_min"]))
    t_max = min(pd.to_datetime(schema["time_max"]))
    train_start = max(pd.Timestamp("2001-01-01"), t_min + pd.Timedelta(days=N_DAYS))
    train_end = min(pd.Timestamp("2018-12-31"), t_max)
    if train_start > train_end:
        raise RuntimeError(f"No ERA5 training interval after lag support: {train_start} > {train_end}")
    return Footprint(lat_min, lat_max, lon_min, lon_max, t_min, t_max, train_start, train_end)


def mask_between(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    values = pd.to_datetime(series)
    return ((values >= start) & (values <= end)).to_numpy()


def footprint_masks(df: pd.DataFrame, fp: Footprint) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(df[DATE_COLUMN])
    spatial = (
        (pd.to_numeric(df[LAT_COLUMN]) >= fp.lat_min)
        & (pd.to_numeric(df[LAT_COLUMN]) <= fp.lat_max)
        & (pd.to_numeric(df[LON_COLUMN]) >= fp.lon_min)
        & (pd.to_numeric(df[LON_COLUMN]) <= fp.lon_max)
    ).to_numpy()
    years = dates.dt.year
    return {
        "train": spatial & mask_between(df[DATE_COLUMN], fp.train_start, fp.train_end),
        "validation": spatial & ((years >= 2019) & (years <= 2020)).to_numpy(),
        "test": spatial & ((years >= 2021) & (years <= 2025)).to_numpy(),
        "spatial": spatial,
    }


def climate_columns(feature_columns: list[str]) -> list[str]:
    return [c for c in feature_columns if any(c.startswith(f"{var}_") for var in CLIMATE_VARIABLES)]


def row_hash(df: pd.DataFrame) -> str:
    import hashlib

    cols = df[[DATE_COLUMN, LAT_COLUMN, LON_COLUMN]].copy()
    cols[DATE_COLUMN] = pd.to_datetime(cols[DATE_COLUMN]).astype("datetime64[ns]")
    values = pd.util.hash_pandas_object(cols, index=False).to_numpy(dtype=np.uint64)
    h = hashlib.blake2b(digest_size=16)
    h.update(values.tobytes())
    return h.hexdigest()


def build_era5_training_frame(
    *,
    base_df: pd.DataFrame,
    train_positions: np.ndarray,
    feature_columns: list[str],
    feature_config: dict[str, Any],
    era5_root: Path,
    output_dir: Path,
    force: bool,
) -> pd.DataFrame:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = artifact_dir / "era5_train_common_features.parquet"
    metadata_path = artifact_dir / "era5_train_common_features.metadata.json"
    train_base = base_df.iloc[train_positions].reset_index(drop=True)
    expected_hash = row_hash(train_base)
    if matrix_path.exists() and metadata_path.exists() and not force:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("row_hash") == expected_hash and meta.get("feature_count") == len(feature_columns):
            print(f"Reusing cached ERA5 training matrix: {matrix_path}", flush=True)
            return pd.read_parquet(matrix_path)

    target = train_base[[DATE_COLUMN, LAT_COLUMN, LON_COLUMN]].rename(columns={DATE_COLUMN: "acq_date"}).copy()
    target["row_position"] = np.arange(len(target), dtype=np.int64)
    climate_feature_parts: list[pd.DataFrame] = []
    cache_dir = artifact_dir / "era5_climate_cache"
    feature_include = feature_config.get("generate_climate_params", {}).get("features_to_include", {})

    for var in CLIMATE_VARIABLES:
        start = time.perf_counter()
        print(f"Building ERA5 climate features for {var} on {len(target):,} training rows", flush=True)
        feat_df, generated_cols, _ = prepare_data(
            climate_data_dir=str(era5_root),
            climate_variables=[var],
            target_df=target,
            n_days=N_DAYS,
            prep_climate=True,
            max_length_features=N_DAYS,
            cache_dir=str(cache_dir),
            lags_features=LAGS,
            windows_features=WINDOWS,
            spans_features=SPANS,
            trend_window_features=TREND_WINDOWS,
            features_to_include_config=feature_include,
            use_cached_files=True,
            return_features_df=True,
            location_batch_size=0,
            max_time_span_days=180,
            persist_dataset=False,
            strict_climate_bounds=True,
        )
        generated_cols = [c for c in generated_cols if c in feature_columns]
        missing = [c for c in feature_columns if c.startswith(f"{var}_") and c not in generated_cols]
        if missing:
            raise RuntimeError(f"ERA5 feature generation for {var} missed columns: {missing[:20]}")
        climate_feature_parts.append(feat_df[generated_cols].reset_index(drop=True))
        var_block_cache = cache_dir / "climate_block_cache" / var
        if var_block_cache.exists():
            shutil.rmtree(var_block_cache)
        print(f"Finished {var}: {len(generated_cols)} columns in {(time.perf_counter() - start) / 60:.1f} min", flush=True)

    era5_train = train_base.copy()
    generated_climate = pd.concat(climate_feature_parts, axis=1)
    for col in climate_columns(feature_columns):
        if col not in generated_climate.columns:
            raise RuntimeError(f"Missing ERA5 climate column {col}")
        era5_train[col] = generated_climate[col].to_numpy(dtype=np.float32)

    block_cache = cache_dir / "climate_block_cache"
    if block_cache.exists():
        shutil.rmtree(block_cache)

    era5_train.to_parquet(matrix_path, index=False, compression="zstd")
    metadata = {
        "created_at": pd.Timestamp.now().isoformat(),
        "row_hash": expected_hash,
        "rows": int(len(era5_train)),
        "feature_count": len(feature_columns),
        "climate_columns": climate_columns(feature_columns),
        "source": str(era5_root),
        "n_days": N_DAYS,
        "lags": LAGS,
        "windows": WINDOWS,
        "spans": SPANS,
        "trend_windows": TREND_WINDOWS,
    }
    write_json(metadata_path, metadata)
    return era5_train


def fit_source_model(
    *,
    experiment_id: str,
    train_df: pd.DataFrame,
    train_y: np.ndarray,
    val_df: pd.DataFrame,
    val_y: np.ndarray,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    cat_features: list[str],
    numerical_cat_features: list[str],
    output_dir: Path,
    iterations: int,
    task_type: str,
    training_mode: str,
    eval_metric: str,
    early_stopping_rounds: int,
    min_tree_count: int,
    selection_metric: str,
    feature_weights: dict[str, float],
) -> dict[str, Any]:
    print(f"Training {experiment_id}: rows={len(train_df):,}, features={len(feature_columns)}", flush=True)
    X_train = normalize_cat_columns(train_df[feature_columns], cat_features, numerical_cat_features)
    X_val = normalize_cat_columns(val_df[feature_columns], cat_features, numerical_cat_features)
    X_test = normalize_cat_columns(test_df[feature_columns], cat_features, numerical_cat_features)

    def candidate_specs() -> list[dict[str, Any]]:
        if training_mode == "single":
            return [
                {
                    "name": f"{eval_metric.lower()}_early_stop",
                    "eval_metric": eval_metric,
                    "use_best_model": True,
                    "early_stopping_rounds": early_stopping_rounds,
                }
            ]
        if training_mode == "robust":
            return [
                {
                    "name": "logloss_early_stop",
                    "eval_metric": "Logloss",
                    "use_best_model": True,
                    "early_stopping_rounds": early_stopping_rounds,
                },
                {
                    "name": "logloss_full_iterations",
                    "eval_metric": "Logloss",
                    "use_best_model": False,
                    "early_stopping_rounds": 0,
                },
            ]
        return [
            {
                "name": "f1_early_stop",
                "eval_metric": "F1",
                "use_best_model": True,
                "early_stopping_rounds": early_stopping_rounds,
            },
            {
                "name": "prauc_early_stop",
                "eval_metric": "PRAUC",
                "use_best_model": True,
                "early_stopping_rounds": early_stopping_rounds,
            },
            {
                "name": "logloss_early_stop",
                "eval_metric": "Logloss",
                "use_best_model": True,
                "early_stopping_rounds": early_stopping_rounds,
            },
            {
                "name": "logloss_full_iterations",
                "eval_metric": "Logloss",
                "use_best_model": False,
                "early_stopping_rounds": 0,
            },
        ]

    base_params = {
        "iterations": iterations,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 0.35,
        "min_data_in_leaf": 80,
        "loss_function": "Logloss",
        "class_weights": [1.0, 4.0],
        "random_seed": SEED,
        "random_strength": 1.0,
        "verbose": 100,
        "allow_writing_files": False,
    }
    if feature_weights:
        base_params["feature_weights"] = feature_weights

    train_pool = catboost_pool(X_train, train_y, cat_features)
    val_pool = catboost_pool(X_val, val_y, cat_features)
    candidate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for spec in candidate_specs():
        params = dict(base_params)
        params["eval_metric"] = spec["eval_metric"]
        if task_type:
            params["task_type"] = task_type
        fit_kwargs: dict[str, Any] = {
            "eval_set": val_pool,
            "use_best_model": bool(spec["use_best_model"]),
        }
        if spec["early_stopping_rounds"]:
            fit_kwargs["early_stopping_rounds"] = int(spec["early_stopping_rounds"])

        def fit_with_params(candidate_params: dict[str, Any]) -> CatBoostClassifier:
            candidate_model = CatBoostClassifier(**candidate_params)
            candidate_model.fit(train_pool, **fit_kwargs)
            return candidate_model

        try:
            model = fit_with_params(params)
        except Exception as exc:
            if params.get("task_type") == "GPU":
                print(
                    f"GPU failed for {experiment_id}/{spec['name']}: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else exc}; retrying CPU",
                    flush=True,
                )
                params.pop("task_type", None)
                try:
                    model = fit_with_params(params)
                except Exception as cpu_exc:
                    candidate_rows.append(
                        {
                            "candidate": spec["name"],
                            "status": "failed",
                            "error": f"{type(cpu_exc).__name__}: {cpu_exc}",
                            "eval_metric": spec["eval_metric"],
                        }
                    )
                    print(f"Candidate failed for {experiment_id}/{spec['name']}: {cpu_exc}", flush=True)
                    continue
            else:
                candidate_rows.append(
                    {
                        "candidate": spec["name"],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "eval_metric": spec["eval_metric"],
                    }
                )
                print(f"Candidate failed for {experiment_id}/{spec['name']}: {exc}", flush=True)
                continue

        val_prob = predict_proba(model, X_val, cat_features)
        threshold, threshold_info = choose_threshold(val_y, val_prob)
        val_metrics = metric_dict(val_y, val_prob, threshold)
        tree_count = int(getattr(model, "tree_count_", 0) or 0)
        eligible = tree_count >= min_tree_count
        score_key = "f1" if selection_metric == "validation_f1" else "average_precision"
        score = val_metrics.get(score_key)
        row = {
            "candidate": spec["name"],
            "status": "completed",
            "eligible": eligible,
            "eval_metric": spec["eval_metric"],
            "use_best_model": spec["use_best_model"],
            "tree_count": tree_count,
            "best_iteration": model.get_best_iteration(),
            "validation_f1": val_metrics.get("f1"),
            "validation_pr_auc": val_metrics.get("average_precision"),
            "validation_threshold": threshold,
            "params": params,
        }
        candidate_rows.append(row)
        print(
            f"Candidate {experiment_id}/{spec['name']}: trees={tree_count}, val_f1={val_metrics.get('f1')}, val_pr_auc={val_metrics.get('average_precision')}, eligible={eligible}",
            flush=True,
        )
        if score is None or not math.isfinite(float(score)):
            continue
        if not eligible and any(r.get("eligible") for r in candidate_rows):
            continue
        if best is None:
            best = {"model": model, "row": row, "threshold": threshold, "threshold_info": threshold_info, "score": float(score)}
            continue
        best_eligible = bool(best["row"].get("eligible"))
        if eligible and not best_eligible:
            best = {"model": model, "row": row, "threshold": threshold, "threshold_info": threshold_info, "score": float(score)}
        elif eligible == best_eligible and float(score) > float(best["score"]):
            best = {"model": model, "row": row, "threshold": threshold, "threshold_info": threshold_info, "score": float(score)}

    if best is None:
        raise RuntimeError(f"No CatBoost candidate completed for {experiment_id}")

    model = best["model"]
    threshold = float(best["threshold"])
    threshold_info = dict(best["threshold_info"])
    threshold_info["selected_candidate"] = best["row"]["candidate"]
    threshold_info["selection_metric"] = selection_metric
    threshold_info["candidate_tree_count"] = best["row"]["tree_count"]
    test_prob = predict_proba(model, X_test, cat_features)
    model_dir = output_dir / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{experiment_id}.cbm"
    model.save_model(model_path)
    write_json(
        model_dir / f"{experiment_id}_features.json",
        {
            "features": feature_columns,
            "categorical_features": cat_features,
            "params": best["row"]["params"],
            "selected_candidate": best["row"],
            "candidate_rows": candidate_rows,
        },
    )
    return {
        "model": model,
        "model_path": model_path,
        "val_prob": predict_proba(model, X_val, cat_features),
        "test_prob": test_prob,
        "threshold": threshold,
        "threshold_info": threshold_info,
        "best_iteration": model.get_best_iteration(),
        "train_rows": int(len(train_df)),
        "params": best["row"]["params"],
        "selected_candidate": best["row"],
        "candidate_rows": candidate_rows,
    }


def evaluate_experiment(
    *,
    experiment: str,
    train_source: str,
    result: dict[str, Any],
    val_df: pd.DataFrame,
    val_y: np.ndarray,
    test_df: pd.DataFrame,
    test_y: np.ndarray,
    regions: list[Region],
    random_error_trials: int = 5,
    random_error_sample_size: int = 50_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = float(result["threshold"])
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "experiment": experiment,
            "train_source": train_source,
            "split": "validation",
            "region": "global",
            "region_display": "Global",
            "period": "2019-2020",
            **metric_dict_with_error(
                val_y,
                result["val_prob"],
                threshold,
                trials=random_error_trials,
                sample_size=random_error_sample_size,
                seed=SEED,
            ),
            "train_rows": result["train_rows"],
            "best_iteration": result["best_iteration"],
            "threshold_source": result["threshold_info"].get("reason"),
            "validation_f1_at_threshold": result["threshold_info"].get("validation_f1"),
        }
    )

    def add_test(region: str, display: str, period: str, mask: np.ndarray) -> None:
        rows.append(
            {
                "experiment": experiment,
                "train_source": train_source,
                "split": "test",
                "region": region,
                "region_display": display,
                "period": period,
                **metric_dict_with_error(
                    test_y[mask],
                    result["test_prob"][mask],
                    threshold,
                    trials=random_error_trials,
                    sample_size=random_error_sample_size,
                    seed=SEED + len(rows),
                ),
                "train_rows": result["train_rows"],
                "best_iteration": result["best_iteration"],
                "threshold_source": result["threshold_info"].get("reason"),
                "validation_f1_at_threshold": result["threshold_info"].get("validation_f1"),
            }
        )

    years = pd.to_datetime(test_df[DATE_COLUMN]).dt.year.to_numpy()
    for year in TEST_YEARS:
        year_mask = years == year
        if year_mask.any():
            add_test("global", "Global", str(year), year_mask)
            for region in regions:
                region_year = year_mask & region.mask(test_df)
                if region_year.any():
                    add_test(region.name, region.display_name, str(year), region_year)
    for label, mask in [
        ("2021-2023", (years >= 2021) & (years <= 2023)),
        ("2021-2025", (years >= 2021) & (years <= 2025)),
    ]:
        if mask.any():
            add_test("global", "Global", label, mask)
            for region in regions:
                region_mask = mask & region.mask(test_df)
                if region_mask.any():
                    add_test(region.name, region.display_name, label, region_mask)

    pred_rows = [
        {
            "experiment": experiment,
            "train_source": train_source,
            "split": "validation",
            "frame": val_df,
            "y": val_y,
            "prob": result["val_prob"],
            "threshold": threshold,
        },
        {
            "experiment": experiment,
            "train_source": train_source,
            "split": "test",
            "frame": test_df,
            "y": test_y,
            "prob": result["test_prob"],
            "threshold": threshold,
        },
    ]
    return rows, pred_rows


def save_predictions(output_dir: Path, pred_specs: list[dict[str, Any]]) -> None:
    if not pred_specs:
        return
    pred_dir = output_dir / "artifacts" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for spec in pred_specs:
        frame = spec["frame"]
        keep = [c for c in [DATE_COLUMN, LAT_COLUMN, LON_COLUMN, "month", "year"] if c in frame.columns]
        pred = frame[keep].copy()
        pred["target_binary"] = np.asarray(spec["y"]).astype(np.int8)
        pred["pred_proba"] = np.asarray(spec["prob"]).astype(np.float32)
        pred["pred_binary"] = (pred["pred_proba"].to_numpy() >= float(spec["threshold"])).astype(np.int8)
        name = safe_filename(f"{spec['experiment']}_{spec['split']}_predictions.parquet")
        pred.to_parquet(pred_dir / name, index=False)


def safe_filename(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum() or ch in {".", "_", "-"}:
            out.append(ch)
        elif ch in {" ", "+", "/", ">"}:
            out.append("_")
    cleaned = "".join(out)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(sanitize(row), sort_keys=True) + "\n")


def round_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out


def write_table(path: Path, df: pd.DataFrame) -> None:
    if len(df.columns) > 6:
        raise ValueError(f"{path} has {len(df.columns)} columns; presentation limit is six.")
    path.parent.mkdir(parents=True, exist_ok=True)
    round_table(df).to_csv(path, index=False)


def make_tables(output_dir: Path, metrics: pd.DataFrame, fp: Footprint) -> None:
    tables = output_dir / "tables"
    test_combined = metrics[(metrics["split"].eq("test")) & (metrics["region"].eq("global")) & (metrics["period"].eq("2021-2025"))]
    write_table(
        tables / "global_metrics.csv",
        test_combined[
            ["experiment", "train_source", "f1", "f1_error", "average_precision", "average_precision_error"]
        ].rename(columns={"experiment": "Experiment", "train_source": "Train Source", "f1": "F1", "f1_error": "F1 Error", "average_precision": "PR-AUC", "average_precision_error": "PR-AUC Error"}),
    )
    write_table(
        tables / "global_precision_recall.csv",
        test_combined[
            ["experiment", "precision", "recall", "roc_auc", "brier_score", "threshold"]
        ].rename(columns={"experiment": "Experiment", "precision": "Precision", "recall": "Recall", "roc_auc": "ROC-AUC", "brier_score": "Brier", "threshold": "Threshold"}),
    )
    regional = metrics[(metrics["split"].eq("test")) & (~metrics["region"].eq("global")) & (metrics["period"].eq("2021-2025"))]
    write_table(
        tables / "regional_metrics.csv",
        regional[
            ["region_display", "experiment", "f1", "f1_error", "average_precision", "average_precision_error"]
        ].rename(columns={"region_display": "Region", "experiment": "Experiment", "f1": "F1", "f1_error": "F1 Error", "average_precision": "PR-AUC", "average_precision_error": "PR-AUC Error"}),
    )
    yearly = metrics[(metrics["split"].eq("test")) & (metrics["region"].eq("global")) & (metrics["period"].isin([str(y) for y in TEST_YEARS]))]
    write_table(
        tables / "yearly_global_metrics.csv",
        yearly[
            ["period", "experiment", "f1", "f1_error", "average_precision", "average_precision_error"]
        ].rename(columns={"period": "Year", "experiment": "Experiment", "f1": "F1", "f1_error": "F1 Error", "average_precision": "PR-AUC", "average_precision_error": "PR-AUC Error"}),
    )
    footprint_rows = pd.DataFrame(
        [
            {"Item": "Latitude bounds", "Value": f"{fp.lat_min:.2f} to {fp.lat_max:.2f}"},
            {"Item": "Longitude bounds", "Value": f"{fp.lon_min:.2f} to {fp.lon_max:.2f}"},
            {"Item": "ERA5 processed time", "Value": f"{fp.era5_min_time.date()} to {fp.era5_max_time.date()}"},
            {"Item": "ERA5 train interval", "Value": f"{fp.train_start.date()} to {fp.train_end.date()}"},
            {"Item": "Validation source", "Value": "ECMWF/SEAS5 2019-2020"},
            {"Item": "Test source", "Value": "ECMWF/SEAS5 2021-2025"},
        ]
    )
    write_table(tables / "common_footprint.csv", footprint_rows)


def make_plots(output_dir: Path, metrics: pd.DataFrame) -> None:
    png_dir = output_dir / "plots" / "png"
    pdf_dir = output_dir / "plots" / "pdf"

    combined = metrics[(metrics["split"].eq("test")) & (metrics["region"].eq("global")) & (metrics["period"].eq("2021-2025"))].copy()
    if not combined.empty:
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for metric, ylabel, stem in [
            ("average_precision", "PR-AUC", "input_source_train_pr_auc"),
            ("f1", "F1", "input_source_train_f1"),
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            plot_df = combined.sort_values(metric)
            ax.barh(plot_df["experiment"], plot_df[metric].astype(float), color=["#5b8def", "#e07054", "#55a868"])
            ax.set_xlabel(ylabel)
            ax.set_title(f"ECMWF Test {ylabel} By Training Source")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(png_dir / f"{stem}.png", dpi=320)
            fig.savefig(pdf_dir / f"{stem}.pdf")
            plt.close(fig)

    yearly = metrics[(metrics["split"].eq("test")) & (metrics["region"].eq("global")) & (metrics["period"].isin([str(y) for y in TEST_YEARS]))].copy()
    if not yearly.empty:
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for metric, ylabel, stem in [
            ("average_precision", "PR-AUC", "input_source_train_yearly_pr_auc"),
            ("f1", "F1", "input_source_train_yearly_f1"),
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 4.4))
            for exp, group in yearly.groupby("experiment", sort=False):
                group = group.sort_values("period")
                ax.plot(group["period"].astype(str), group[metric].astype(float), marker="o", label=exp)
            ax.set_xlabel("ECMWF test year")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Yearly ECMWF Test {ylabel}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(png_dir / f"{stem}.png", dpi=320)
            fig.savefig(pdf_dir / f"{stem}.pdf")
            plt.close(fig)


def write_description(output_dir: Path, fp: Footprint, raw_audit: dict[str, Any]) -> None:
    grib_note = "Some raw GRIB samples were unreadable by ecCodes; processed ERA5 zarrs were used."
    if any(item.get("returncode") == 127 for item in raw_audit.get("sample_grib_checks", [])):
        grib_note = "The conda-run raw GRIB audit could not find `grib_ls`; processed ERA5 zarrs were used for the reproducible training matrix."
    elif not any(item.get("unreadable_messages") for item in raw_audit.get("sample_grib_checks", [])):
        grib_note = "Raw GRIB sample checks did not report unreadable messages."
    write_text(
        output_dir / "description.md",
        f"""# ERA5 vs ECMWF Training Source Comparison

## Purpose
This experiment compares three CatBoost training sources while holding validation and test fixed to the same ECMWF/SEAS5 matrix:

- ECMWF train -> ECMWF validation/test.
- ERA5 train -> ECMWF validation/test.
- ERA5 + ECMWF duplicated train rows -> ECMWF validation/test.

Thresholds are selected only on ECMWF validation years 2019-2020 and then applied unchanged to the ECMWF 2021-2025 test rows.

## Common Footprint
- Latitude: `{fp.lat_min:.2f}` to `{fp.lat_max:.2f}`.
- Longitude: `{fp.lon_min:.2f}` to `{fp.lon_max:.2f}`.
- ERA5 processed time range: `{fp.era5_min_time.date()}` to `{fp.era5_max_time.date()}`.
- Training interval with 128-day lookback support: `{fp.train_start.date()}` to `{fp.train_end.date()}`.

## ERA5 Source
ERA5 climate features were generated from processed zarr files under `/home/ids/vmorozov/data/climate_data/climate_features/ERA5`. The raw ERA5 directory `/home/ids/vmorozov/era5` was audited and is saved in `artifacts/raw_era5_audit.json`. {grib_note}

## Folder Layout
- `tables/`: readable CSV tables, at most six columns each.
- `plots/png/`: high-resolution PNG plots.
- `plots/pdf/`: PDF plots.
- `artifacts/`: ERA5 training parquet, raw metrics JSONL, models, predictions, schema, logs, and environment metadata.
""",
    )


def write_analysis(output_dir: Path, metrics: pd.DataFrame) -> None:
    combined = metrics[(metrics["split"].eq("test")) & (metrics["region"].eq("global")) & (metrics["period"].eq("2021-2025"))].copy()
    if combined.empty:
        body = "- No combined global test metrics were produced."
    else:
        best_pr = combined.loc[combined["average_precision"].astype(float).idxmax()]
        best_f1 = combined.loc[combined["f1"].astype(float).idxmax()]
        baseline = combined[combined["experiment"].eq("ECMWF train -> ECMWF test")]
        lines = [
            f"- Highest ECMWF-test PR-AUC: {best_pr['experiment']} ({float(best_pr['average_precision']):.4f}).",
            f"- Highest ECMWF-test F1: {best_f1['experiment']} ({float(best_f1['f1']):.4f}).",
        ]
        if not baseline.empty:
            b = baseline.iloc[0]
            for _, row in combined.iterrows():
                if row["experiment"] == b["experiment"]:
                    continue
                lines.append(
                    f"- {row['experiment']} vs ECMWF baseline: Delta PR-AUC={float(row['average_precision']) - float(b['average_precision']):+.4f}, Delta F1={float(row['f1']) - float(b['f1']):+.4f}."
                )
        lines.append("- ERA5-trained performance on ECMWF test should be interpreted as input-source domain transfer, not an ERA5 retrospective upper bound.")
        lines.append("- The mixed-source model tests whether duplicating targets with both climate sources improves operational robustness on ECMWF inputs.")
        body = "\n".join(lines)
    write_text(output_dir / "analysis.md", "# Analysis\n\n" + body)


def update_index(output_dir: Path, result_root: Path) -> None:
    index = result_root / "experiments" / "index.md"
    rel = output_dir.relative_to(result_root / "experiments")
    line = f"- [ERA5 vs ECMWF Training Source Comparison]({rel}/description.md): Compares ERA5, ECMWF, and duplicated ERA5+ECMWF training rows on the same ECMWF validation/test footprint."
    if index.exists():
        text = index.read_text(encoding="utf-8")
        if str(rel) not in text:
            write_text(index, text.rstrip() + "\n" + line)
    table = result_root / "experiments" / "experiment_index.csv"
    if table.exists():
        df = pd.read_csv(table)
        if not df["Folder"].astype(str).eq(str(rel)).any():
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "Experiment": "ERA5 vs ECMWF Training Source Comparison",
                                "Folder": str(rel),
                                "Purpose": "Compares ERA5, ECMWF, and duplicated ERA5+ECMWF training rows on the same ECMWF validation/test footprint.",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            if len(df.columns) <= 6:
                df.to_csv(table, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=Path("data/saved_features_boost/train_test_features_30d_all_extended_north_14cb.parquet"))
    parser.add_argument("--feature-config", type=Path, default=Path("configs/features_config_30d.yaml"))
    parser.add_argument("--catboost-config", type=Path, default=Path("configs/catboost_train_config.yaml"))
    parser.add_argument("--regions-file", type=Path, default=Path("configs/regions_example.yaml"))
    parser.add_argument("--era5-root", type=Path, default=Path("/home/ids/vmorozov/data/climate_data/climate_features/ERA5"))
    parser.add_argument("--raw-era5-root", type=Path, default=Path("/home/ids/vmorozov/era5"))
    parser.add_argument("--result-root", type=Path, default=Path("results/revision_experiments_complete"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=450)
    parser.add_argument("--task-type", default="GPU")
    parser.add_argument("--training-mode", choices=["single", "robust", "best"], default="best")
    parser.add_argument("--eval-metric", default="F1")
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--min-tree-count", type=int, default=20)
    parser.add_argument("--selection-metric", choices=["validation_f1", "average_precision"], default="validation_f1")
    parser.add_argument("--random-error-trials", type=int, default=5)
    parser.add_argument("--random-error-sample-size", type=int, default=50_000)
    parser.add_argument("--force-era5-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(SEED)
    output_dir = args.output_dir or args.result_root / "experiments" / "24_era5_ecmwf_train_source_comparison"
    for stale in [output_dir / "tables", output_dir / "plots", output_dir / "artifacts" / "models", output_dir / "artifacts" / "predictions"]:
        if stale.exists():
            shutil.rmtree(stale)
    command = " ".join(["conda", "run", "-n", "pointnet", "python", "scripts/run_era5_ecmwf_source_comparison.py", *sys.argv[1:]])
    write_text(output_dir / "artifacts" / "command.txt", command)
    write_json(
        output_dir / "artifacts" / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "seed": SEED,
            "cwd": Path.cwd(),
            "command": command,
        },
    )

    print(f"Loading ECMWF feature matrix: {args.features_path}", flush=True)
    df = pd.read_parquet(args.features_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    raw_audit = raw_era5_audit(args.raw_era5_root, output_dir)
    schema_all = processed_era5_schema(args.era5_root)
    schema_train = processed_era5_schema(args.era5_root, year_start=2000, year_end=2018)
    for stale_schema in (output_dir / "artifacts").glob("processed_era5_schema*.csv"):
        stale_schema.unlink()
    write_json(output_dir / "artifacts" / "processed_era5_schema_all_years.json", schema_all.to_dict(orient="records"))
    write_json(output_dir / "artifacts" / "processed_era5_schema_train_years.json", schema_train.to_dict(orient="records"))
    feature_config = read_yaml(args.feature_config)
    catboost_config = read_yaml(args.catboost_config)
    regions = load_regions(args.regions_file)
    ignored = catboost_config.get("catboost_train", {}).get("features", {}).get(
        "ignored",
        ["datetime", "day", "latitude", "longitude", "year"],
    )
    feature_columns = model_feature_columns(df, ignored)
    cat_features = [c for c in feature_config.get("cat_features", []) if c in feature_columns]
    numerical_cat_features = [c for c in feature_config.get("numerical_cat_features", []) if c in feature_columns]
    feature_weights = catboost_feature_weights(feature_columns, catboost_config)
    climate_cols = climate_columns(feature_columns)
    if not climate_cols:
        raise RuntimeError("No shared climate columns found in ECMWF feature matrix.")

    fp = common_footprint(schema_train, df)
    lookback_start_year = (fp.train_start - pd.Timedelta(days=N_DAYS)).year
    filtered_era5_root = make_filtered_era5_root(
        args.era5_root,
        output_dir / "artifacts" / "era5_zarr_train_year_view",
        range(lookback_start_year, fp.train_end.year + 1),
    )
    masks = footprint_masks(df, fp)
    train_pos = np.flatnonzero(masks["train"])
    val_pos = np.flatnonzero(masks["validation"])
    test_pos = np.flatnonzero(masks["test"])
    if len(train_pos) == 0 or len(val_pos) == 0 or len(test_pos) == 0:
        raise RuntimeError(f"Empty split after common footprint: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")

    y_all = positive_labels(df[TARGET_COLUMN])
    split_info = {
        "common_footprint": fp.__dict__,
        "rows": {
            "train": int(len(train_pos)),
            "validation": int(len(val_pos)),
            "test": int(len(test_pos)),
            "spatial_all_years": int(masks["spatial"].sum()),
        },
        "positives": {
            "train": int(y_all[train_pos].sum()),
            "validation": int(y_all[val_pos].sum()),
            "test": int(y_all[test_pos].sum()),
        },
        "climate_columns": climate_cols,
        "catboost_training": {
            "iterations": args.iterations,
            "training_mode": args.training_mode,
            "candidate_eval_metrics": (
                ["Logloss"]
                if args.training_mode == "robust"
                else ([args.eval_metric] if args.training_mode == "single" else ["F1", "PRAUC", "Logloss"])
            ),
            "early_stopping_rounds": args.early_stopping_rounds,
            "min_tree_count": args.min_tree_count,
            "selection_metric": args.selection_metric,
            "feature_weights": feature_weights,
        },
    }
    write_json(output_dir / "artifacts" / "split_and_feature_metadata.json", split_info)

    era5_train = build_era5_training_frame(
        base_df=df,
        train_positions=train_pos,
        feature_columns=feature_columns,
        feature_config=feature_config,
        era5_root=filtered_era5_root,
        output_dir=output_dir,
        force=args.force_era5_features,
    )

    ecmwf_train = df.iloc[train_pos].reset_index(drop=True)
    val_df = df.iloc[val_pos].reset_index(drop=True)
    test_df = df.iloc[test_pos].reset_index(drop=True)
    y_train = y_all[train_pos].astype(np.int8)
    y_val = y_all[val_pos].astype(np.int8)
    y_test = y_all[test_pos].astype(np.int8)

    mixed_train = pd.concat([ecmwf_train, era5_train], ignore_index=True)
    mixed_y = np.concatenate([y_train, y_train]).astype(np.int8)

    specs = [
        ("ecmwf_train_ecmwf_test", "ECMWF train -> ECMWF test", "ECMWF", ecmwf_train, y_train),
        ("era5_train_ecmwf_test", "ERA5 train -> ECMWF test", "ERA5", era5_train, y_train),
        ("mixed_era5_ecmwf_train_ecmwf_test", "ERA5 + ECMWF train -> ECMWF test", "ERA5 + ECMWF duplicates", mixed_train, mixed_y),
    ]

    all_metric_rows: list[dict[str, Any]] = []
    all_pred_specs: list[dict[str, Any]] = []
    model_selection_rows: list[dict[str, Any]] = []
    for experiment_id, label, train_source, train_df, train_y in specs:
        result = fit_source_model(
            experiment_id=experiment_id,
            train_df=train_df,
            train_y=train_y,
            val_df=val_df,
            val_y=y_val,
            test_df=test_df,
            feature_columns=feature_columns,
            cat_features=cat_features,
            numerical_cat_features=numerical_cat_features,
            output_dir=output_dir,
            iterations=args.iterations,
            task_type=args.task_type,
            training_mode=args.training_mode,
            eval_metric=args.eval_metric,
            early_stopping_rounds=args.early_stopping_rounds,
            min_tree_count=args.min_tree_count,
            selection_metric=args.selection_metric,
            feature_weights=feature_weights,
        )
        selected_candidate = result.get("selected_candidate", {}).get("candidate")
        for row in result.get("candidate_rows", []):
            model_selection_rows.append(
                {
                    "experiment": label,
                    "train_source": train_source,
                    "selected": row.get("candidate") == selected_candidate,
                    **row,
                }
            )
        metric_rows, pred_specs = evaluate_experiment(
            experiment=label,
            train_source=train_source,
            result=result,
            val_df=val_df,
            val_y=y_val,
            test_df=test_df,
            test_y=y_test,
            regions=regions,
            random_error_trials=args.random_error_trials,
            random_error_sample_size=args.random_error_sample_size,
        )
        all_metric_rows.extend(metric_rows)
        all_pred_specs.extend(pred_specs)

    metrics = pd.DataFrame(all_metric_rows)
    write_raw_jsonl(output_dir / "artifacts" / "input_source_train_metrics_raw.jsonl.gz", all_metric_rows)
    write_raw_jsonl(output_dir / "artifacts" / "model_selection_raw.jsonl.gz", model_selection_rows)
    metrics.to_parquet(output_dir / "artifacts" / "input_source_train_metrics.parquet", index=False)
    if model_selection_rows:
        selection_df = pd.DataFrame(model_selection_rows)
        write_table(
            output_dir / "tables" / "model_selection.csv",
            selection_df[
                ["experiment", "candidate", "selected", "tree_count", "validation_f1", "validation_pr_auc"]
            ].rename(
                columns={
                    "experiment": "Experiment",
                    "candidate": "Candidate",
                    "selected": "Selected",
                    "tree_count": "Trees",
                    "validation_f1": "Val F1",
                    "validation_pr_auc": "Val PR-AUC",
                }
            ),
        )
    save_predictions(output_dir, all_pred_specs)
    make_tables(output_dir, metrics, fp)
    make_plots(output_dir, metrics)
    write_description(output_dir, fp, raw_audit)
    write_analysis(output_dir, metrics)
    update_index(output_dir, args.result_root)
    prune_empty_dirs(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
