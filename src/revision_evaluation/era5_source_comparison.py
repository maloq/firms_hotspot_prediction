from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from src.feature_generation.prepare_climate_data import (
    check_fragmented_dataset_bounds,
    discover_climate_fragments,
    prepare_data as prepare_climate_data_func,
)

from .tabular import (
    DEFAULT_FEATURES_PATH,
    DEFAULT_IGNORED_FEATURES,
    DEFAULT_RESULTS_DIR,
    DATE_COLUMN,
    LAT_COLUMN,
    LON_COLUMN,
    SEED,
    TARGET_COLUMN,
    CatBoostClassifier,
    CATBOOST_IMPORT_ERROR,
    ExperimentFailure,
    build_feature_sets,
    catboost_categorical_features,
    catboost_pool,
    choose_threshold_by_f1,
    evaluate_predictions,
    feature_group,
    load_regions,
    load_yaml,
    model_feature_columns,
    normalize_cat_columns,
    plot_input_source,
    positive_labels,
    predict_catboost,
    save_predictions,
    validate_no_leakage_features,
    write_json,
)
from .probability_overlays import load_dense_neural_predictor


ERA5_FEATURE_ROOT = Path("/home/ids/vmorozov/data/climate_data/climate_features/ERA5")
ECMWF_FEATURE_ROOT = Path("/home/ids/vmorozov/data/climate_data/climate_features/ECMWF")


@dataclass(frozen=True)
class Era5Inventory:
    variables: list[str]
    common_years: list[int]
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    rows: list[dict[str, Any]]


def default_args(**overrides: Any) -> SimpleNamespace:
    data = {
        "features_path": DEFAULT_FEATURES_PATH,
        "feature_config": Path("configs/features_config_30d.yaml"),
        "catboost_config": Path("configs/catboost_train_config.yaml"),
        "regions_file": Path("configs/regions_example.yaml"),
        "output_dir": DEFAULT_RESULTS_DIR,
        "era5_feature_root": ERA5_FEATURE_ROOT,
        "ecmwf_feature_root": ECMWF_FEATURE_ROOT,
        "cache_dir": Path("data/saved_features/revision_evaluation/era5_source_comparison"),
        "train_start_year": 2001,
        "validation_start_year": 2019,
        "test_start_year": 2021,
        "test_end_year": None,
        "catboost_iterations": 260,
        "catboost_depth": 5,
        "catboost_learning_rate": 0.03,
        "catboost_task_type": "GPU",
        "catboost_verbose": 100,
        "random_error_trials": 5,
        "random_error_sample_size": 50_000,
        "seed": SEED,
        "use_lat_lon_features": False,
        "force_rebuild_era5_features": False,
        "skip_mixed_source": False,
        "drop_climate_variables": [],
        "only_experiment": None,
        "merge_existing_input_source": False,
        "update_organized_results": False,
        "input_source_include_best_neural": False,
        "input_source_neural_model": "best_neural",
        "input_source_neural_training_features": DEFAULT_FEATURES_PATH,
        "input_source_neural_batch_size": 8192,
        "input_source_neural_device": "auto",
        "input_source_neural_masked_climate_variables": [],
    }
    data.update(overrides)
    data["drop_climate_variables"] = normalize_string_list(data.get("drop_climate_variables"))
    data["input_source_neural_masked_climate_variables"] = normalize_string_list(
        data.get("input_source_neural_masked_climate_variables")
    )
    return SimpleNamespace(**data)


def normalize_string_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [str(item).strip() for item in value if str(item).strip()]


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "shared_artifacts" / "logs" / "era5_source_comparison.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )


def _year_from_path(path: Path) -> int | None:
    match = re.search(r"_(\d{4})(?:\.zarr)?$", path.name)
    return int(match.group(1)) if match else None


def _open_zarr_summary(path: Path) -> dict[str, Any]:
    ds = xr.open_zarr(path, consolidated=False)
    try:
        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        return {
            "time_coord": time_coord,
            "time_min": str(pd.Timestamp(ds[time_coord].min().values).date()) if time_coord in ds.coords else None,
            "time_max": str(pd.Timestamp(ds[time_coord].max().values).date()) if time_coord in ds.coords else None,
            "lat_min": float(ds["latitude"].min().values),
            "lat_max": float(ds["latitude"].max().values),
            "lon_min": float(ds["longitude"].min().values),
            "lon_max": float(ds["longitude"].max().values),
        }
    finally:
        ds.close()


def era5_inventory(root: Path, variables: Sequence[str]) -> Era5Inventory:
    rows: list[dict[str, Any]] = []
    year_sets: list[set[int]] = []
    lat_mins: list[float] = []
    lat_maxs: list[float] = []
    lon_mins: list[float] = []
    lon_maxs: list[float] = []

    for variable in variables:
        files = sorted((root / variable).glob(f"{variable}_*.zarr"))
        if not files:
            raise FileNotFoundError(f"No ERA5 zarr files found for {variable} under {root}")
        years = sorted(year for path in files if (year := _year_from_path(path)) is not None)
        if not years:
            raise ValueError(f"Could not infer ERA5 years for {variable} under {root / variable}")
        summaries = [_open_zarr_summary(path) for path in files]
        summary = {
            "lat_min": max(item["lat_min"] for item in summaries),
            "lat_max": min(item["lat_max"] for item in summaries),
            "lon_min": max(item["lon_min"] for item in summaries),
            "lon_max": min(item["lon_max"] for item in summaries),
        }
        year_sets.append(set(years))
        lat_mins.append(summary["lat_min"])
        lat_maxs.append(summary["lat_max"])
        lon_mins.append(summary["lon_min"])
        lon_maxs.append(summary["lon_max"])
        rows.append(
            {
                "variable": variable,
                "era5_zarr_files": len(files),
                "era5_years": ",".join(str(year) for year in years),
                "era5_lat_min": summary["lat_min"],
                "era5_lat_max": summary["lat_max"],
                "era5_lon_min": summary["lon_min"],
                "era5_lon_max": summary["lon_max"],
                "status": "common variable available",
            }
        )

    common_years = sorted(set.intersection(*year_sets))
    if not common_years:
        raise ValueError("ERA5 variables have no common years.")
    return Era5Inventory(
        variables=list(variables),
        common_years=common_years,
        lat_min=max(lat_mins),
        lat_max=min(lat_maxs),
        lon_min=max(lon_mins),
        lon_max=min(lon_maxs),
        rows=rows,
    )


def write_schema_tables(
    output_dir: Path,
    inventory: Era5Inventory,
    era5_root: Path,
    ecmwf_root: Path,
) -> None:
    rows = []
    for item in inventory.rows:
        variable = item["variable"]
        ecmwf_files = sorted((ecmwf_root / variable).glob(f"{variable}_*.zarr"))
        row = dict(item)
        row["ecmwf_zarr_files"] = len(ecmwf_files)
        row["available_domain"] = (
            f"lat {inventory.lat_min:g}..{inventory.lat_max:g}, "
            f"lon {inventory.lon_min:g}..{inventory.lon_max:g}"
        )
        rows.append(row)
    schema = pd.DataFrame(rows)
    write_raw_table(output_dir, "era5_feature_schema.csv", schema)
    write_raw_table(output_dir, "era5_seas5_common_schema.csv", schema)

    log_text = "\n".join(
        [
            "# ERA5 Feature Build Log",
            "",
            f"- ERA5 zarr root: `{era5_root}`.",
            f"- Common ERA5 years used: `{min(inventory.common_years)}-{max(inventory.common_years)}`.",
            f"- ERA5-covered comparison domain: lat `{inventory.lat_min:g}-{inventory.lat_max:g}`, lon `{inventory.lon_min:g}-{inventory.lon_max:g}`.",
            "- The source-comparison rerun uses only sampled feature-table rows inside this ERA5-covered domain and common-year window.",
            "- Metrics are legacy sampled/case-control diagnostics for input-source comparison; primary deployment probability metrics remain in the calibrated full-grid tables.",
        ]
    )
    source_dir = output_dir / "shared_artifacts" / "source_markdown"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "era5_feature_build_log.md").write_text(log_text + "\n", encoding="utf-8")


def climate_feature_columns(columns: Sequence[str]) -> list[str]:
    return [col for col in columns if feature_group(col) == "weather_history"]


def drop_climate_variable_features(columns: Sequence[str], variables: Sequence[str]) -> list[str]:
    drop_prefixes = tuple(f"{variable}_" for variable in variables)
    drop_exact = set(variables)
    if not drop_prefixes and not drop_exact:
        return list(columns)
    return [col for col in columns if col not in drop_exact and not col.startswith(drop_prefixes)]


def run_suffix_for_drops(variables: Sequence[str]) -> tuple[str, str]:
    if not variables:
        return "", ""
    label = " (no " + ", ".join(variables) + ")"
    suffix = "_no_" + "_".join(re.sub(r"[^A-Za-z0-9]+", "_", variable).strip("_") for variable in variables)
    return label, suffix


def load_available_seas5_frame(
    features_path: Path,
    inventory: Era5Inventory,
    *,
    train_start_year: int,
    test_end_year: int | None,
) -> tuple[pd.DataFrame, int]:
    logging.info("Loading sampled feature table from %s", features_path)
    df = pd.read_parquet(features_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if df[DATE_COLUMN].isna().any():
        raise ValueError("Feature table contains unparseable datetimes.")

    max_year = max(inventory.common_years)
    if test_end_year is not None:
        max_year = min(max_year, int(test_end_year))
    mask = (
        df[DATE_COLUMN].dt.year.between(train_start_year, max_year)
        & df[LAT_COLUMN].between(inventory.lat_min, inventory.lat_max)
        & df[LON_COLUMN].between(inventory.lon_min, inventory.lon_max)
    )
    out = df.loc[mask].reset_index(drop=True)
    if out.empty:
        raise ValueError("No sampled feature rows overlap the ERA5-covered source-comparison domain.")
    logging.info(
        "ERA5-available sampled rows: %d; years %d-%d; lat %.2f..%.2f; lon %.2f..%.2f",
        len(out),
        train_start_year,
        max_year,
        inventory.lat_min,
        inventory.lat_max,
        inventory.lon_min,
        inventory.lon_max,
    )
    return out, max_year


def _date_value_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(pd.Timestamp(value).date())
    except Exception:
        return str(value)


def filter_full_era5_climate_coverage(
    frame: pd.DataFrame,
    *,
    era5_root: Path,
    variables: Sequence[str],
    n_days: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Keep only rows whose full ERA5 lookback window is covered for all variables."""

    if frame.empty:
        return frame.copy(), []

    target = frame[[DATE_COLUMN, LAT_COLUMN, LON_COLUMN]].copy()
    target["acq_date"] = target[DATE_COLUMN]
    target = target[["acq_date", LAT_COLUMN, LON_COLUMN]]

    keep = np.ones(len(frame), dtype=bool)
    coverage_rows: list[dict[str, Any]] = []
    for variable in variables:
        fragments = discover_climate_fragments(era5_root, variable)
        result = check_fragmented_dataset_bounds(fragments, target, n_days=n_days)
        assignments = result["assignments"]
        covered = assignments >= 0
        keep &= covered

        details = result.get("details", {})
        required = details.get("required", {})
        dataset_union = details.get("dataset_union", {})
        coverage_rows.append(
            {
                "variable": variable,
                "fragment_count": details.get("fragment_count", len(fragments)),
                "covered_rows": int(covered.sum()),
                "missing_rows": int((~covered).sum()),
                "missing_fraction": float((~covered).sum() / max(1, len(frame))),
                "required_time_min": _date_value_text((required.get("time") or (None, None))[0]),
                "required_time_max": _date_value_text((required.get("time") or (None, None))[1]),
                "dataset_time_min": _date_value_text((dataset_union.get("time") or (None, None))[0]),
                "dataset_time_max": _date_value_text((dataset_union.get("time") or (None, None))[1]),
                "required_lat_min": (required.get("latitude") or (None, None))[0],
                "required_lat_max": (required.get("latitude") or (None, None))[1],
                "required_lon_min": (required.get("longitude") or (None, None))[0],
                "required_lon_max": (required.get("longitude") or (None, None))[1],
            }
        )

    dropped = int((~keep).sum())
    if dropped:
        logging.warning(
            "Dropping %d/%d ERA5 source-comparison rows without complete %d-day climate coverage across %d variables.",
            dropped,
            len(frame),
            n_days,
            len(variables),
        )
    out = frame.loc[keep].reset_index(drop=True)
    if out.empty:
        raise ValueError("No sampled feature rows have complete ERA5 climate lookback coverage.")
    logging.info(
        "ERA5 full-lookback covered sampled rows: %d/%d (%.2f%%)",
        len(out),
        len(frame),
        100.0 * len(out) / max(1, len(frame)),
    )
    return out, coverage_rows


def split_masks_by_year(
    frame: pd.DataFrame,
    *,
    train_start_year: int,
    validation_start_year: int,
    test_start_year: int,
    test_end_year: int,
) -> dict[str, np.ndarray]:
    years = frame[DATE_COLUMN].dt.year
    masks = {
        "train": years.between(train_start_year, validation_start_year - 1).to_numpy(),
        "validation": years.between(validation_start_year, test_start_year - 1).to_numpy(),
        "test": years.between(test_start_year, test_end_year).to_numpy(),
    }
    for split, mask in masks.items():
        if not mask.any():
            raise ValueError(f"ERA5 source-comparison split {split!r} is empty.")
    return masks


def _safe_token(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def era5_cache_path(cache_dir: Path, frame: pd.DataFrame, inventory: Era5Inventory, test_end_year: int) -> Path:
    start_year = int(frame[DATE_COLUMN].dt.year.min())
    variable_token = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", variable).strip("_") for variable in inventory.variables)
    token = (
        f"era5_climate_features_{start_year}_{test_end_year}"
        f"_vars{variable_token}"
        f"_lat{_safe_token(f'{inventory.lat_min:g}')}_{_safe_token(f'{inventory.lat_max:g}')}"
        f"_lon{_safe_token(f'{inventory.lon_min:g}')}_{_safe_token(f'{inventory.lon_max:g}')}"
        f"_rows{len(frame)}.parquet"
    )
    return cache_dir / token


def load_or_build_era5_climate_features(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
    inventory: Era5Inventory,
    cache_dir: Path,
    *,
    era5_root: Path,
    force_rebuild: bool,
    test_end_year: int,
) -> tuple[pd.DataFrame, list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = era5_cache_path(cache_dir, frame, inventory, test_end_year)
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() and not force_rebuild:
        logging.info("Loading cached ERA5 climate features from %s", path)
        climate = pd.read_parquet(path)
        cols = json.loads(manifest_path.read_text(encoding="utf-8"))["climate_columns"] if manifest_path.exists() else list(climate.columns)
        return climate, cols

    climate_params = dict(feature_config["climate_data_params"])
    climate_params["climate_data_dir"] = str(era5_root)
    climate_params["climate_variables"] = inventory.variables
    generate = feature_config["generate_climate_params"]
    target = frame[[DATE_COLUMN, LAT_COLUMN, LON_COLUMN, TARGET_COLUMN]].copy()
    target["acq_date"] = target[DATE_COLUMN]
    target["id"] = np.arange(len(target), dtype=np.int64)

    logging.info("Generating ERA5 climate features for %d sampled rows", len(target))
    started = time.perf_counter()
    climate_df, climate_cols, _ = prepare_climate_data_func(
        climate_data_dir=climate_params["climate_data_dir"],
        climate_variables=climate_params["climate_variables"],
        target_df=target,
        n_days=climate_params["n_days"],
        test_mode=False,
        max_length_features=generate["max_length"],
        lags_features=generate["lags"],
        windows_features=generate["windows"],
        spans_features=generate["spans"],
        trend_window_features=generate.get("trend_window", [21, 90]),
        features_to_include_config=generate["features_to_include"],
        use_cached_files=True,
        cache_dir=str(cache_dir / "climate_matrix_cache"),
        return_features_df=True,
        location_batch_size=climate_params.get("location_batch_size"),
        max_time_span_days=climate_params.get("max_time_span_days"),
        persist_dataset=climate_params.get("persist_dataset", False),
        strict_climate_bounds=True,
    )
    climate = climate_df[climate_cols].astype("float32")
    climate.to_parquet(path, index=False)
    write_json(
        manifest_path,
        {
            "path": path,
            "rows": len(climate),
            "climate_columns": climate_cols,
            "climate_variables": inventory.variables,
            "elapsed_seconds": time.perf_counter() - started,
            "era5_years": inventory.common_years,
            "era5_bounds": {
                "lat_min": inventory.lat_min,
                "lat_max": inventory.lat_max,
                "lon_min": inventory.lon_min,
                "lon_max": inventory.lon_max,
            },
        },
    )
    logging.info("Saved ERA5 climate features to %s", path)
    return climate, climate_cols


def replace_weather_features(
    seas5: pd.DataFrame,
    era5_climate: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    weather_cols = climate_feature_columns(feature_columns)
    missing = sorted(set(weather_cols) - set(era5_climate.columns))
    if missing:
        raise ValueError(
            "ERA5 climate feature schema does not match model weather columns: "
            f"missing={missing[:5]}"
        )
    era5 = seas5.copy()
    era5.loc[:, weather_cols] = era5_climate.loc[:, weather_cols].to_numpy()
    return era5


def train_source_catboost(
    experiment_id: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    feature_config: dict[str, Any],
    args: Any,
    model_dir: Path,
) -> tuple[Any, list[str]]:
    validate_no_leakage_features(feature_columns)
    if CatBoostClassifier is None:
        raise ExperimentFailure(f"catboost import failed: {CATBOOST_IMPORT_ERROR}")
    categorical = catboost_categorical_features(train_frame, feature_columns, feature_config)
    numerical_cat = [c for c in feature_config.get("numerical_cat_features", []) if c in feature_columns]

    X_train = normalize_cat_columns(train_frame[feature_columns], categorical, numerical_cat)
    X_val = normalize_cat_columns(validation_frame[feature_columns], categorical, numerical_cat)
    y_train = positive_labels(train_frame[TARGET_COLUMN])
    y_val = positive_labels(validation_frame[TARGET_COLUMN])
    params = {
        "iterations": args.catboost_iterations,
        "depth": args.catboost_depth,
        "learning_rate": args.catboost_learning_rate,
        "l2_leaf_reg": 0.35,
        "min_data_in_leaf": 80,
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "class_weights": [1.0, 4.0],
        "random_seed": args.seed,
        "random_strength": 1.0,
        "verbose": args.catboost_verbose if args.catboost_verbose > 0 else False,
        "allow_writing_files": False,
    }
    if args.catboost_task_type:
        params["task_type"] = args.catboost_task_type

    logging.info(
        "Training %s on %d rows, validating on %d rows, %d features",
        experiment_id,
        len(train_frame),
        len(validation_frame),
        len(feature_columns),
    )
    model = CatBoostClassifier(**params)
    try:
        model.fit(
            catboost_pool(X_train, y_train, categorical),
            eval_set=catboost_pool(X_val, y_val, categorical),
            use_best_model=True,
            early_stopping_rounds=100,
        )
    except Exception:
        if params.get("task_type") == "GPU":
            logging.warning("CatBoost GPU failed for %s; retrying on CPU.", experiment_id)
            params.pop("task_type", None)
            model = CatBoostClassifier(**params)
            model.fit(
                catboost_pool(X_train, y_train, categorical),
                eval_set=catboost_pool(X_val, y_val, categorical),
                use_best_model=True,
                early_stopping_rounds=100,
            )
        else:
            raise

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / f"{experiment_id}.cbm")
    write_json(model_dir / f"{experiment_id}_features.json", {"features": feature_columns, "categorical_features": categorical})
    return model, categorical


def evaluate_source_model(
    experiment_id: str,
    experiment_name: str,
    interpretation: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    feature_config: dict[str, Any],
    regions: list[Any],
    args: Any,
    artifact_root: Path,
    notes: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model, cat_features = train_source_catboost(
        experiment_id,
        train_frame,
        validation_frame,
        feature_columns,
        feature_config,
        args,
        artifact_root / "models",
    )
    numerical_cat = feature_config.get("numerical_cat_features", [])
    X_val = normalize_cat_columns(validation_frame[feature_columns], cat_features, numerical_cat)
    X_test = normalize_cat_columns(test_frame[feature_columns], cat_features, numerical_cat)
    y_val = positive_labels(validation_frame[TARGET_COLUMN])
    y_test = positive_labels(test_frame[TARGET_COLUMN])
    val_prob = predict_catboost(model, X_val, cat_features)
    test_prob = predict_catboost(model, X_test, cat_features)
    threshold_info = choose_threshold_by_f1(y_val, val_prob)
    threshold = float(threshold_info["threshold"])

    save_predictions(artifact_root, experiment_id, "validation", validation_frame, y_val, val_prob, threshold)
    save_predictions(artifact_root, experiment_id, "test", test_frame, y_test, test_prob, threshold)

    metric_rows: list[dict[str, Any]] = []
    for split_name, frame, y, prob in [
        ("validation", validation_frame, y_val, val_prob),
        ("test", test_frame, y_test, test_prob),
    ]:
        metric_rows.extend(
            evaluate_predictions(
                experiment_id,
                "input_source_comparison",
                "CatBoost",
                experiment_name,
                split_name,
                frame,
                y,
                prob,
                threshold,
                regions,
                error_trials=args.random_error_trials,
                error_sample_size=args.random_error_sample_size,
                seed=args.seed,
            )
        )
    for row in metric_rows:
        row["experiment"] = experiment_name
        row["status"] = "completed_available_domain"
        row["interpretation"] = interpretation
        row["notes"] = notes
        row["threshold"] = threshold
        row["validation_f1_at_threshold"] = threshold_info.get("validation_f1")
    registry_row = {
        "experiment_id": experiment_id,
        "experiment": experiment_name,
        "status": "completed_available_domain",
        "interpretation": interpretation,
        "threshold": threshold,
        "validation_f1_at_threshold": threshold_info.get("validation_f1"),
        "notes": notes,
    }
    return [registry_row], metric_rows


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def best_neural_experiment_id(output_dir: Path, requested: str | None) -> str:
    requested = str(requested or "best_neural").strip()
    if requested and requested != "best_neural":
        return requested if requested.startswith("nn_") else f"nn_global_full_{requested}"

    raw_path = output_dir / "shared_artifacts" / "raw_tables_jsonl" / "embedding_fusion_ablation.jsonl.gz"
    if raw_path.is_file():
        rows = pd.read_json(raw_path, orient="records", lines=True, compression="gzip")
        if not rows.empty and "source_metrics" in rows.columns:
            candidates = rows[
                rows.get("region", pd.Series(index=rows.index, dtype=str)).astype(str).eq("global")
                & rows["source_metrics"].astype(str).str.contains("nn_global_full_", na=False)
            ].copy()
            if "period" in candidates.columns:
                period_mask = candidates["period"].astype(str).eq("2021-2025")
                if period_mask.any():
                    candidates = candidates.loc[period_mask].copy()
            if not candidates.empty and "average_precision" in candidates.columns:
                candidates["average_precision"] = pd.to_numeric(candidates["average_precision"], errors="coerce")
                candidates = candidates.dropna(subset=["average_precision"]).sort_values("average_precision", ascending=False)
                if not candidates.empty:
                    source_metrics = str(candidates.iloc[0]["source_metrics"])
                    exp_id = Path(source_metrics).parent.name
                    if exp_id.startswith("nn_"):
                        return exp_id

    metrics_dir = output_dir / "shared_artifacts" / "neural_model_metrics"
    candidates: list[tuple[float, str]] = []
    for metrics_path in sorted(metrics_dir.glob("nn_global_full_*_metrics.json")):
        payload = _read_json_if_exists(metrics_path)
        test_metrics = payload.get("test_metrics") or payload.get("metrics") or {}
        score = test_metrics.get("average_precision")
        if score is None:
            score = payload.get("average_precision")
        try:
            score_float = float(score)
        except (TypeError, ValueError):
            continue
        exp_id = metrics_path.name.removesuffix("_metrics.json")
        candidates.append((score_float, exp_id))
    if candidates:
        return sorted(candidates, reverse=True)[0][1]
    raise FileNotFoundError(
        "Could not identify the best neural model. Expected embedding_fusion_ablation raw table "
        f"or neural metrics under {metrics_dir}."
    )


def neural_prediction_path(output_dir: Path, experiment_id: str) -> Path:
    filename = f"{experiment_id}_test_legacy_predictions.parquet"
    candidates = [
        output_dir / "shared_artifacts" / "predictions" / filename,
        output_dir / "predictions" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Missing neural prediction metadata parquet. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def paired_ecmwf_neural_experiment_id(output_dir: Path, experiment_id: str) -> str | None:
    if experiment_id.endswith("_ecmwf"):
        return experiment_id
    candidate = f"{experiment_id}_ecmwf"
    try:
        neural_prediction_path(output_dir, candidate)
    except FileNotFoundError:
        return None
    metric_candidates = [
        output_dir / "shared_artifacts" / "neural_model_metrics" / f"{candidate}_metrics.json",
        output_dir / "neural_model_metrics" / f"{candidate}_metrics.json",
    ]
    return candidate if any(path.is_file() for path in metric_candidates) else None


def neural_source_feature_config(
    feature_config: dict[str, Any],
    *,
    climate_root: Path,
    variables: Sequence[str],
) -> dict[str, Any]:
    out = copy.deepcopy(feature_config)
    climate_params = dict(out.get("climate_data_params", {}))
    climate_params["climate_data_dir"] = str(climate_root)
    climate_params["climate_variables"] = list(variables)
    out["climate_data_params"] = climate_params
    return out


def evaluate_neural_source_model(
    experiment_id: str,
    experiment_name: str,
    interpretation: str,
    source_name: str,
    neural_experiment_id: str,
    prediction_path: Path,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_config: dict[str, Any],
    masked_variables: Sequence[str],
    regions: list[Any],
    args: Any,
    artifact_root: Path,
    notes: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    logging.info(
        "Scoring neural source experiment %s with checkpoint metadata %s on %s inputs",
        experiment_id,
        prediction_path,
        source_name,
    )
    predictor = load_dense_neural_predictor(
        results_dir=args.output_dir,
        prediction_path=prediction_path,
        training_features_path=Path(args.input_source_neural_training_features),
        model_path=None,
        batch_size=int(args.input_source_neural_batch_size),
        device=str(args.input_source_neural_device),
        feature_config_path=args.feature_config,
        feature_config=feature_config,
        masked_dynamic_variables=masked_variables,
    )
    y_val = positive_labels(validation_frame[TARGET_COLUMN])
    y_test = positive_labels(test_frame[TARGET_COLUMN])
    val_prob = predictor.predict(validation_frame)
    test_prob = predictor.predict(test_frame)
    threshold_info = choose_threshold_by_f1(y_val, val_prob)
    threshold = float(threshold_info["threshold"])

    save_predictions(artifact_root, experiment_id, "validation", validation_frame, y_val, val_prob, threshold)
    save_predictions(artifact_root, experiment_id, "test", test_frame, y_test, test_prob, threshold)

    masking_note = (
        "Dropped dynamic climate variables are set to the scaled training mean before scoring."
        if masked_variables
        else "The checkpoint is scored with the dynamic climate variables present in its training schema."
    )
    neural_notes = f"{notes} Neural source rows use checkpoint `{neural_experiment_id}`. {masking_note}"
    metric_rows: list[dict[str, Any]] = []
    for split_name, frame, y, prob in [
        ("validation", validation_frame, y_val, val_prob),
        ("test", test_frame, y_test, test_prob),
    ]:
        metric_rows.extend(
            evaluate_predictions(
                experiment_id,
                "input_source_comparison",
                "Neural",
                experiment_name,
                split_name,
                frame,
                y,
                prob,
                threshold,
                regions,
                error_trials=args.random_error_trials,
                error_sample_size=args.random_error_sample_size,
                seed=args.seed,
            )
        )
    for row in metric_rows:
        row["experiment"] = experiment_name
        row["status"] = "completed_available_domain"
        row["interpretation"] = interpretation
        row["notes"] = neural_notes
        row["threshold"] = threshold
        row["validation_f1_at_threshold"] = threshold_info.get("validation_f1")
        row["neural_experiment_id"] = neural_experiment_id
        row["source_name"] = source_name
    registry_row = {
        "experiment_id": experiment_id,
        "experiment": experiment_name,
        "status": "completed_available_domain",
        "interpretation": interpretation,
        "threshold": threshold,
        "validation_f1_at_threshold": threshold_info.get("validation_f1"),
        "notes": neural_notes,
        "neural_experiment_id": neural_experiment_id,
        "source_name": source_name,
    }
    return [registry_row], metric_rows


def source_tables_from_metrics(metric_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    common_cols = [
        "experiment",
        "status",
        "interpretation",
        "region",
        "region_display",
        "precision",
        "recall",
        "f1",
        "f1_error",
        "average_precision",
        "average_precision_error",
        "roc_auc",
        "brier_score",
        "notes",
    ]
    overall = metrics[metrics["split"].eq("test")].loc[:, common_cols].copy()

    yearly = metrics[metrics["split"].astype(str).str.fullmatch(r"test_\d{4}")].copy()
    if not yearly.empty:
        yearly["year"] = yearly["split"].str.replace("test_", "", regex=False).astype(int)
        yearly = yearly.loc[:, common_cols[:4] + ["year"] + common_cols[4:]].copy()
    return overall, yearly


def write_raw_table(output_dir: Path, csv_name: str, df: pd.DataFrame) -> None:
    raw_dir = output_dir / "shared_artifacts" / "raw_tables_jsonl"
    schema_dir = output_dir / "shared_artifacts" / "raw_table_schemas"
    raw_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_name).with_suffix("").name
    raw_path = raw_dir / f"{stem}.jsonl.gz"
    schema_path = schema_dir / f"{stem}.schema.json"
    csv_path = output_dir / csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_json(raw_path, orient="records", lines=True, compression="gzip")
    write_json(
        schema_path,
        {
            "source_csv": csv_name,
            "rows": int(len(df)),
            "columns": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
        },
    )


def read_raw_table(output_dir: Path, csv_name: str) -> pd.DataFrame:
    stem = Path(csv_name).with_suffix("").name
    raw_path = output_dir / "shared_artifacts" / "raw_tables_jsonl" / f"{stem}.jsonl.gz"
    if raw_path.exists():
        return pd.read_json(raw_path, orient="records", lines=True, compression="gzip")
    csv_path = output_dir / csv_name
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def merge_source_table(existing: pd.DataFrame, new: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    if existing.empty:
        return new.copy()
    if new.empty:
        return existing.copy()
    keys = [col for col in key_cols if col in existing.columns and col in new.columns]
    if not keys:
        return pd.concat([existing, new], ignore_index=True)
    existing_key = existing[keys].astype(str).agg("\0".join, axis=1)
    new_key = set(new[keys].astype(str).agg("\0".join, axis=1))
    kept = existing.loc[~existing_key.isin(new_key)].copy()
    return pd.concat([kept, new], ignore_index=True, sort=False)


def _round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out


def _select(df: pd.DataFrame, columns: list[tuple[str, str]], sort_by: list[str]) -> pd.DataFrame:
    selected = [src for src, _ in columns if src in df.columns]
    renamed = {src: dst for src, dst in columns if src in selected}
    out = df.loc[:, selected].rename(columns=renamed)
    actual_sort = [col for col in sort_by if col in out.columns]
    if actual_sort:
        out = out.sort_values(actual_sort)
    return _round_for_csv(out)


def _copy_plot_to_experiment(plot_path: Path, exp_dir: Path) -> None:
    for suffix, subdir in [(".png", "png"), (".pdf", "pdf")]:
        src = plot_path.with_suffix(suffix)
        if not src.exists():
            continue
        dst = exp_dir / "plots" / subdir / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def update_organized_input_source(output_dir: Path, overall: pd.DataFrame, yearly: pd.DataFrame) -> None:
    exp_dir = output_dir / "experiments" / "24_input_source_comparison_global"
    table_dir = exp_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    overall_global = overall[overall["region"].astype(str).eq("global")].copy() if "region" in overall.columns else overall
    (exp_dir / "description.md").write_text(
        "\n".join(
            [
                "# Input Source Comparison Global",
                "",
                "## Purpose",
                "Compares ERA5 and SEAS5/ECMWF source settings on the ERA5-covered sampled evaluation domain.",
                "",
                "## Source Tables",
                "- `input_source_comparison.csv`",
                "- `input_source_comparison_by_year.csv`",
                "- `era5_feature_schema.csv`",
                "- `era5_seas5_common_schema.csv`",
                "",
                "## Notes",
                "- Metrics are legacy sampled/case-control diagnostics for the available ERA5 domain and years.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (exp_dir / "analysis.md").write_text(
        "\n".join(
            [
                "# Analysis",
                "",
                "- ERA5 -> ERA5 is a retrospective upper-bound source setting on ERA5-available rows.",
                "- ERA5 -> SEAS5/ECMWF measures input-source domain shift.",
                "- ERA5 + SEAS5/ECMWF -> SEAS5/ECMWF tests mixed-source training for the operational source.",
                "- SEAS5/ECMWF -> SEAS5/ECMWF is the operationally matched source baseline on the same available rows.",
                "- Best neural source rows score the selected best spatial neural checkpoint with removed climate variables masked to their scaled training mean.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _select(
        overall_global,
        [
            ("experiment", "Experiment"),
            ("status", "Status"),
            ("f1", "F1"),
            ("f1_error", "F1 Error"),
            ("average_precision", "PR-AUC"),
            ("average_precision_error", "PR-AUC Error"),
        ],
        ["Experiment"],
    ).to_csv(table_dir / "input_source_metrics.csv", index=False)
    _select(
        overall_global,
        [
            ("experiment", "Experiment"),
            ("status", "Status"),
            ("interpretation", "Interpretation"),
            ("roc_auc", "ROC-AUC"),
            ("notes", "Notes"),
        ],
        ["Experiment"],
    ).to_csv(table_dir / "input_source_notes.csv", index=False)
    _select(
        yearly,
        [
            ("experiment", "Experiment"),
            ("status", "Status"),
            ("year", "Year"),
            ("region_display", "Region"),
            ("f1", "F1"),
            ("f1_error", "F1 Error"),
            ("average_precision", "PR-AUC"),
            ("average_precision_error", "PR-AUC Error"),
        ],
        ["Experiment", "Year", "Region"],
    ).to_csv(table_dir / "input_source_by_year_available.csv", index=False)

    stage = output_dir / "shared_artifacts" / "era5_source_comparison" / "_plot_stage"
    if stage.exists():
        shutil.rmtree(stage)
    plot_input_source(overall, stage)
    shared_plots = output_dir / "shared_artifacts" / "source_plots_mixed"
    shared_plots.mkdir(parents=True, exist_ok=True)
    for stem in ["input_source_comparison", "input_source_pr_auc", "input_source_f1"]:
        base = stage / "plots" / stem
        for suffix in [".png", ".pdf"]:
            src = base.with_suffix(suffix)
            if src.exists():
                dst = shared_plots / src.name
                shutil.copy2(src, dst)
        _copy_plot_to_experiment(shared_plots / stem, exp_dir)
    if stage.exists():
        shutil.rmtree(stage)

    artifact_dir = exp_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "raw_sources.json").write_text(
        json.dumps(
            {
                "source_tables": [
                    "shared_artifacts/raw_tables_jsonl/input_source_comparison.jsonl.gz",
                    "shared_artifacts/raw_tables_jsonl/input_source_comparison_by_year.jsonl.gz",
                ],
                "plots": [
                    "shared_artifacts/source_plots_mixed/input_source_comparison.png",
                    "shared_artifacts/source_plots_mixed/input_source_comparison.pdf",
                    "shared_artifacts/source_plots_mixed/input_source_pr_auc.png",
                    "shared_artifacts/source_plots_mixed/input_source_pr_auc.pdf",
                    "shared_artifacts/source_plots_mixed/input_source_f1.png",
                    "shared_artifacts/source_plots_mixed/input_source_f1.pdf",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def era5_source_limitations_text() -> str:
    return "\n".join(
        [
            "# Failures And Remaining Blockers",
            "",
            "No failed ERA5 input-source experiment remains after the available-domain rerun.",
            "",
            "## ERA5 Available-Domain Source Comparison",
            "",
            "- Status: completed for SEAS5/ECMWF -> SEAS5/ECMWF, ERA5 -> ERA5, ERA5 -> SEAS5/ECMWF, ERA5 + SEAS5/ECMWF -> SEAS5/ECMWF, and configured best-neural source rows.",
            "- Limitation: metrics are legacy sampled/case-control diagnostics on rows covered by processed ERA5 zarr data and common years; they are not primary calibrated deployment-grid probability metrics.",
            "- Coverage: processed ERA5 zarrs currently support the available-domain run through 2024. Exact full-domain/full-2025 parity remains a data-coverage limitation, not a model-code failure.",
        ]
    )


def write_failures_source_markdown(output_dir: Path) -> None:
    source_dir = output_dir / "shared_artifacts" / "source_markdown"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "failures.md").write_text(era5_source_limitations_text() + "\n", encoding="utf-8")


def update_failures_artifact(output_dir: Path) -> None:
    text = era5_source_limitations_text()
    write_failures_source_markdown(output_dir)

    exp_dir = output_dir / "experiments" / "30_failures_and_limitations"
    (exp_dir / "description.md").parent.mkdir(parents=True, exist_ok=True)
    (exp_dir / "description.md").write_text(
        "# Failures And Limitations\n\n## Purpose\nContains true remaining blockers and limitations after attempted rebuilds and reruns.\n",
        encoding="utf-8",
    )
    (exp_dir / "analysis.md").write_text(
        "# Analysis\n\n- ERA5 input-source comparison now has available-domain metrics; treat the ERA5 domain/year restriction as a reporting limitation.\n",
        encoding="utf-8",
    )
    artifact_dir = exp_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "failures_original.md").write_text(text + "\n", encoding="utf-8")
    table_dir = exp_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Item": "ERA5 input-source comparison",
                "Note": "Completed on processed ERA5 available domain/common years; see experiment 24 tables.",
            }
        ]
    ).to_csv(table_dir / "failure_notes.csv", index=False)


def run(args: Any) -> dict[str, Any]:
    setup_logging(args.output_dir)
    np.random.seed(args.seed)
    feature_config = load_yaml(args.feature_config)
    catboost_config = load_yaml(args.catboost_config)
    regions = load_regions(args.regions_file)
    variables = [str(variable) for variable in feature_config["climate_data_params"]["climate_variables"]]
    dropped_variables = set(args.drop_climate_variables)
    comparison_variables = [variable for variable in variables if variable not in dropped_variables]
    if not comparison_variables:
        raise ValueError("Input-source comparison cannot drop every configured climate variable.")
    if args.drop_climate_variables:
        logging.info(
            "Input-source comparison climate variables: %s; dropped variables: %s",
            comparison_variables,
            args.drop_climate_variables,
        )
    inventory = era5_inventory(args.era5_feature_root, comparison_variables)
    write_schema_tables(args.output_dir, inventory, args.era5_feature_root, args.ecmwf_feature_root)

    seas5, max_year = load_available_seas5_frame(
        args.features_path,
        inventory,
        train_start_year=args.train_start_year,
        test_end_year=args.test_end_year,
    )
    coverage_n_days = int(feature_config["climate_data_params"]["n_days"])
    rows_before_coverage_filter = len(seas5)
    seas5, coverage_rows = filter_full_era5_climate_coverage(
        seas5,
        era5_root=args.era5_feature_root,
        variables=comparison_variables,
        n_days=coverage_n_days,
    )
    write_raw_table(args.output_dir, "era5_climate_coverage_filter.csv", pd.DataFrame(coverage_rows))
    masks = split_masks_by_year(
        seas5,
        train_start_year=args.train_start_year,
        validation_start_year=args.validation_start_year,
        test_start_year=args.test_start_year,
        test_end_year=max_year,
    )

    ignored = (
        catboost_config.get("catboost_train", {})
        .get("features", {})
        .get("ignored", DEFAULT_IGNORED_FEATURES)
        if isinstance(catboost_config.get("catboost_train"), dict)
        else DEFAULT_IGNORED_FEATURES
    )
    feature_columns = model_feature_columns(
        seas5,
        ignored,
        use_lat_lon_features=args.use_lat_lon_features,
    )
    validate_no_leakage_features(feature_columns)
    feature_columns = build_feature_sets(feature_columns)["full"]["columns"]
    label_suffix, id_suffix = run_suffix_for_drops(args.drop_climate_variables)
    if args.drop_climate_variables:
        before = len(feature_columns)
        feature_columns = drop_climate_variable_features(feature_columns, args.drop_climate_variables)
        logging.info(
            "Dropped climate variables %s from model inputs: %d -> %d features",
            args.drop_climate_variables,
            before,
            len(feature_columns),
        )
    weather_cols = climate_feature_columns(feature_columns)
    logging.info("Using %d model features, including %d weather-history features", len(feature_columns), len(weather_cols))

    era5_climate, era5_cols = load_or_build_era5_climate_features(
        seas5,
        feature_config,
        inventory,
        args.cache_dir,
        era5_root=args.era5_feature_root,
        force_rebuild=args.force_rebuild_era5_features,
        test_end_year=max_year,
    )
    if len(era5_climate) != len(seas5):
        raise ValueError(f"ERA5 feature row count mismatch: {len(era5_climate)} vs {len(seas5)}")
    missing_era5_cols = sorted(set(weather_cols) - set(era5_cols))
    if missing_era5_cols:
        raise ValueError(
            "ERA5 generated climate columns do not match source-comparison weather columns: "
            f"missing={missing_era5_cols[:5]}"
        )
    era5 = replace_weather_features(seas5, era5_climate, feature_columns)

    dropped_note = ""
    if args.drop_climate_variables:
        dropped_note = " Dropped climate variables from model inputs: " + ", ".join(args.drop_climate_variables) + "."
    coord_note = " Direct latitude/longitude coordinate features were included." if args.use_lat_lon_features else " Direct latitude/longitude coordinate features were excluded."
    notes = (
        f"Evaluated on ERA5-covered sampled case-control rows only "
        f"(years {args.test_start_year}-{max_year}, lat {inventory.lat_min:g}-{inventory.lat_max:g}, "
        f"lon {inventory.lon_min:g}-{inventory.lon_max:g}). "
        f"Rows without complete {coverage_n_days}-day ERA5 lookback coverage for comparison climate variables "
        f"({', '.join(comparison_variables)}) were excluded "
        f"({rows_before_coverage_filter} -> {len(seas5)} rows). "
        "These are legacy source-comparison diagnostics, not primary calibrated full-grid probability metrics."
        f"{coord_note}"
        f"{dropped_note}"
    )
    artifact_root = args.output_dir / "shared_artifacts" / "era5_source_comparison"
    artifact_root.mkdir(parents=True, exist_ok=True)

    seas5_train, seas5_val, seas5_test = (seas5.loc[masks[name]].reset_index(drop=True) for name in ["train", "validation", "test"])
    era5_train, era5_val, era5_test = (era5.loc[masks[name]].reset_index(drop=True) for name in ["train", "validation", "test"])

    registry_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    specs_by_key = {
        "seas5_to_seas5": (
            f"source_seas5_to_seas5_available{id_suffix}",
            f"SEAS5/ECMWF -> SEAS5/ECMWF{label_suffix}",
            "Operationally matched setting on the ERA5-available row subset.",
            seas5_train,
            seas5_val,
            seas5_test,
        ),
        "era5_to_era5": (
            f"source_era5_to_era5_available{id_suffix}",
            f"ERA5 -> ERA5{label_suffix}",
            "Retrospective upper-bound source setting.",
            era5_train,
            era5_val,
            era5_test,
        ),
        "era5_to_seas5": (
            f"source_era5_to_seas5_available{id_suffix}",
            f"ERA5 -> SEAS5/ECMWF{label_suffix}",
            "Input-source domain-shift test.",
            era5_train,
            era5_val,
            seas5_test,
        ),
    }
    if not args.skip_mixed_source:
        mixed_train = pd.concat([seas5_train, era5_train], ignore_index=True)
        specs_by_key["mixed"] = (
            f"source_era5_plus_seas5_to_seas5_available{id_suffix}",
            f"ERA5 + SEAS5/ECMWF -> SEAS5/ECMWF{label_suffix}",
            "Mixed-source operational robustness test.",
            mixed_train,
            seas5_val,
            seas5_test,
        )
    if args.skip_mixed_source:
        raise ValueError("Input-source comparison must include the mixed ERA5 + SEAS5/ECMWF experiment.")
    if args.only_experiment:
        raise ValueError("Input-source comparison must run all source experiments; partial selections are not allowed.")

    selected = list(specs_by_key)
    specs = [specs_by_key[key] for key in selected if key in specs_by_key]
    if not specs:
        raise ValueError("No source-comparison experiments selected.")

    for experiment_id, experiment_name, interpretation, train_frame, val_frame, test_frame in specs:
        rows_reg, rows_metrics = evaluate_source_model(
            experiment_id,
            experiment_name,
            interpretation,
            train_frame,
            val_frame,
            test_frame,
            feature_columns,
            feature_config,
            regions,
            args,
            artifact_root,
            notes,
        )
        registry_rows.extend(rows_reg)
        metric_rows.extend(rows_metrics)

    neural_required_experiments: set[str] = set()
    neural_experiments: list[str] = []
    neural_masked_climate_variables: list[str] = []
    neural_climate_variables = list(comparison_variables)
    if args.input_source_include_best_neural:
        best_neural_id = best_neural_experiment_id(args.output_dir, args.input_source_neural_model)
        requested_neural = str(args.input_source_neural_model or "").strip()
        neural_masked_climate_variables = list(args.input_source_neural_masked_climate_variables)
        if not neural_masked_climate_variables and requested_neural in {"best_neural", "best_nn", "best_nns"}:
            neural_masked_climate_variables = list(args.drop_climate_variables)
        neural_climate_variables = list(variables if neural_masked_climate_variables else comparison_variables)
        best_prediction = neural_prediction_path(args.output_dir, best_neural_id)
        paired_ecmwf_id = paired_ecmwf_neural_experiment_id(args.output_dir, best_neural_id)
        logging.info(
            "Adding best neural source comparison with %s (paired ECMWF checkpoint: %s, climate variables: %s, masked: %s)",
            best_neural_id,
            paired_ecmwf_id or "not available",
            neural_climate_variables,
            neural_masked_climate_variables,
        )
        neural_specs = [
            (
                f"source_neural_era5_to_era5_available{id_suffix}",
                f"Best neural ERA5 -> ERA5{label_suffix}",
                "Best neural checkpoint scored on retrospective ERA5 inputs.",
                "ERA5",
                best_neural_id,
                best_prediction,
                neural_source_feature_config(
                    feature_config,
                    climate_root=args.era5_feature_root,
                    variables=neural_climate_variables,
                ),
                era5_val,
                era5_test,
            ),
            (
                f"source_neural_era5_to_seas5_available{id_suffix}",
                f"Best neural ERA5 -> SEAS5/ECMWF{label_suffix}",
                "Best neural checkpoint scored with SEAS5/ECMWF climate inputs to measure source shift.",
                "SEAS5/ECMWF",
                best_neural_id,
                best_prediction,
                neural_source_feature_config(
                    feature_config,
                    climate_root=args.ecmwf_feature_root,
                    variables=neural_climate_variables,
                ),
                seas5_val,
                seas5_test,
            ),
        ]
        if paired_ecmwf_id is not None:
            paired_prediction = neural_prediction_path(args.output_dir, paired_ecmwf_id)
            neural_specs.append(
                (
                    f"source_neural_seas5_to_seas5_available{id_suffix}",
                    f"Best neural SEAS5/ECMWF -> SEAS5/ECMWF{label_suffix}",
                    "Best neural architecture paired ECMWF checkpoint scored on operational SEAS5/ECMWF inputs.",
                    "SEAS5/ECMWF",
                    paired_ecmwf_id,
                    paired_prediction,
                    neural_source_feature_config(
                        feature_config,
                        climate_root=args.ecmwf_feature_root,
                        variables=neural_climate_variables,
                    ),
                    seas5_val,
                    seas5_test,
                )
            )

        for (
            neural_experiment_id,
            neural_experiment_name,
            neural_interpretation,
            source_name,
            checkpoint_experiment_id,
            prediction_path,
            neural_feature_config,
            val_frame,
            test_frame,
        ) in neural_specs:
            rows_reg, rows_metrics = evaluate_neural_source_model(
                neural_experiment_id,
                neural_experiment_name,
                neural_interpretation,
                source_name,
                checkpoint_experiment_id,
                prediction_path,
                val_frame,
                test_frame,
                neural_feature_config,
                neural_masked_climate_variables,
                regions,
                args,
                artifact_root,
                notes,
            )
            registry_rows.extend(rows_reg)
            metric_rows.extend(rows_metrics)
            neural_required_experiments.add(neural_experiment_name)
            neural_experiments.append(neural_experiment_name)

    registry = pd.DataFrame(registry_rows)
    metrics = pd.DataFrame(metric_rows)
    table_suffix = id_suffix or ""
    registry.drop(columns=["source_name"], errors="ignore").to_csv(
        artifact_root / f"era5_source_registry{table_suffix}.csv",
        index=False,
    )
    metrics.to_json(
        artifact_root / f"era5_source_metrics_wide{table_suffix}.jsonl.gz",
        orient="records",
        lines=True,
        compression="gzip",
    )

    overall, yearly = source_tables_from_metrics(metric_rows)
    if args.merge_existing_input_source:
        existing_overall = read_raw_table(args.output_dir, "input_source_comparison.csv")
        existing_yearly = read_raw_table(args.output_dir, "input_source_comparison_by_year.csv")
        overall = merge_source_table(existing_overall, overall, ["experiment", "region"])
        yearly = merge_source_table(existing_yearly, yearly, ["experiment", "year", "region"])
    required_experiments = {spec[1] for spec in specs_by_key.values()} | neural_required_experiments
    completed_global = overall[
        overall.get("region", pd.Series(index=overall.index, dtype=str)).astype(str).eq("global")
        & overall.get("status", pd.Series(index=overall.index, dtype=str)).astype(str).str.startswith("completed")
    ]
    completed_experiments = set(completed_global.get("experiment", pd.Series(dtype=str)).astype(str))
    missing = sorted(required_experiments - completed_experiments)
    if missing:
        raise RuntimeError(
            "Input-source comparison is incomplete; refusing to write a partial source-comparison table. "
            f"Missing completed global rows: {missing}"
        )
    write_raw_table(args.output_dir, "input_source_comparison.csv", overall)
    write_raw_table(args.output_dir, "input_source_comparison_by_year.csv", yearly)
    write_failures_source_markdown(args.output_dir)
    if args.update_organized_results:
        update_organized_input_source(args.output_dir, overall, yearly)
        update_failures_artifact(args.output_dir)

    manifest = {
        "rows_available": len(seas5),
        "rows_before_coverage_filter": rows_before_coverage_filter,
        "coverage_n_days": coverage_n_days,
        "train_rows": int(masks["train"].sum()),
        "validation_rows": int(masks["validation"].sum()),
        "test_rows": int(masks["test"].sum()),
        "test_years": f"{args.test_start_year}-{max_year}",
        "era5_bounds": {
            "lat_min": inventory.lat_min,
            "lat_max": inventory.lat_max,
            "lon_min": inventory.lon_min,
            "lon_max": inventory.lon_max,
        },
        "experiments": [item[1] for item in specs],
        "neural_experiments": neural_experiments,
        "comparison_climate_variables": comparison_variables,
        "neural_climate_variables": neural_climate_variables,
        "neural_masked_climate_variables": neural_masked_climate_variables,
        "drop_climate_variables": args.drop_climate_variables,
        "overall_rows": len(overall),
        "yearly_rows": len(yearly),
    }
    write_json(artifact_root / f"manifest{table_suffix}.json", manifest)
    logging.info("ERA5 source comparison complete: %s", manifest)
    return manifest


def main() -> int:
    run(default_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
