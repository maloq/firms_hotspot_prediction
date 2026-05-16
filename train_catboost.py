from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import catboost
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.revision_evaluation.calibration import (
    apply_calibrator,
    fit_calibrator,
    logit,
    save_calibrator,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


logger = logging.getLogger(__name__)


DEFAULT_COUNTRIES = [
    "Dem_Rep_Korea",
    "Russian_Federation",
    "Finland",
    "Norway",
    "Sweden",
    "Denmark",
    "Lithuania",
    "Latvia",
    "Estonia",
    "Poland",
    "Czech_Republic",
    "Germany",
    "Hungary",
    "Slovakia",
    "Belarus",
    "Ukraine",
    "Moldova",
    "Romania",
    "Bulgaria",
    "Albania",
    "Montenegro",
    "Macedonia_Former_Yugoslav_Republic_of",
    "Kosovo",
    "Serbia",
    "Croatia",
    "Bosnia_and_Herzegovina",
    "Slovenia",
    "Greece",
    "Turkey",
    "Georgia",
    "Azerbaijan",
    "Armenia",
    "Kazakhstan",
    "Kyrgyzstan",
    "Tajikistan",
    "Mongolia",
    "China",
    "Japan",
    "Republic_of_Korea",
]

DEFAULT_FEATURES_PATH = Path(
    "data/saved_features_boost/train_test_features_30d_strats.parquet"
)
DEFAULT_IGNORED_FEATURES = [
    "datetime",
    "day",
    "latitude",
    "longitude",
    "year",
    "lon_rounded",
    "soft_label",
    "negative_stratum",
    "sampling_probability",
    "sample_weight",
    "nearest_positive_distance_cells",
    "nearest_positive_delta_days",
]
DEFAULT_VALIDATION_START_DATE = "2021-01-01"
DEFAULT_TEST_START_DATE = "2023-01-01"
DEFAULT_FEATURE_WEIGHTS = {
    "ecoregion_name": 0.3,
}
DEFAULT_BAGGING_TEMPERATURE = 0.96
DEFAULT_PLOT_REGIONS = {
    "Eastern Europe": {"lon_range": (20.0, 50.0), "lat_range": (40.0, 60.0)},
    "Scandinavia": {"lon_range": (5.0, 30.0), "lat_range": (55.0, 75.0)},
    "Siberia West": {"lon_range": (60.0, 100.0), "lat_range": (50.0, 70.0)},
    "Siberia East": {"lon_range": (100.0, 140.0), "lat_range": (50.0, 70.0)},
    "Far East Russia": {"lon_range": (140.0, 180.0), "lat_range": (50.0, 70.0)},
    "Central Asia": {"lon_range": (50.0, 80.0), "lat_range": (35.0, 50.0)},
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the CatBoost wildfire model from a YAML config."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the CatBoost training YAML config.",
    )
    return parser.parse_args(argv)


def _section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected config section {key!r} to be a mapping.")
    return value


def _path_or_none(value: Any) -> Optional[Path]:
    return None if value is None else Path(value).expanduser()


def _as_list(value: Any, default: Sequence[Any] = ()) -> List[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_int_list(value: Any, default: Sequence[int] = ()) -> List[int]:
    return [int(item) for item in _as_list(value, default)]


def _first_defined(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _none_if_blank(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def build_args_from_config(config_path: Path) -> argparse.Namespace:
    run_config = load_config(config_path)
    cfg = _section(run_config, "catboost_train") if "catboost_train" in run_config else run_config
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {config_path} did not contain a YAML mapping.")

    run = _section(cfg, "run")
    data = _section(cfg, "data")
    target = _section(cfg, "target")
    split = _section(cfg, "split")
    model = _section(cfg, "model")
    features = _section(cfg, "features")
    thresholding = _section(cfg, "thresholding")
    calibration = _section(cfg, "calibration")
    plots = _section(cfg, "plots")
    importance = _section(cfg, "feature_importance_analysis")
    analysis = _section(cfg, "analysis")
    hardware = _section(cfg, "hardware")

    default_feature_config_path = (
        config_path
        if "cat_features" in cfg or "selected_feature_columns_path" in cfg
        else Path("configs/features_config_30d.yaml")
    )
    feature_config_path = _path_or_none(
        _first_defined(
            cfg.get("feature_config_path"),
            cfg.get("feature_config"),
            data.get("feature_config_path"),
            default=default_feature_config_path,
        )
    )
    if feature_config_path is None:
        raise ValueError("CatBoost training config must define feature_config_path.")

    return argparse.Namespace(
        run_config=config_path,
        config=feature_config_path,
        features_path=Path(data.get("features_path", DEFAULT_FEATURES_PATH)).expanduser(),
        rebuild_features=bool(data.get("rebuild_features", False)),
        per_country_features_dir=Path(
            data.get("per_country_features_dir", "data/saved_features")
        ).expanduser(),
        per_country_pattern=data.get(
            "per_country_pattern",
            "train_test_features_30d_{country}.parquet",
        ),
        countries=_as_list(data.get("countries"), DEFAULT_COUNTRIES),
        output_root=Path(run.get("output_root", "outputs")).expanduser(),
        run_prefix=run.get("run_prefix", "catboost_train"),
        model_path=_path_or_none(_first_defined(analysis.get("model_path"), cfg.get("model_path"))),
        analysis_only=bool(analysis.get("analysis_only", False)),
        target_column=target.get("column", data.get("target_column", "count")),
        positive_threshold=float(target.get("positive_threshold", 0.0)),
        soft_label_column=target.get("soft_label_column", "soft_label"),
        use_soft_labels_for_training=bool(target.get("use_soft_labels_for_training", False)),
        test_size=float(split.get("test_size", 0.15)),
        validation_size=float(split.get("validation_size", 0.15)),
        date_column=split.get("date_column", "datetime"),
        validation_start_date=_none_if_blank(
            split.get("validation_start_date", DEFAULT_VALIDATION_START_DATE)
        ),
        test_start_date=_none_if_blank(split.get("test_start_date", DEFAULT_TEST_START_DATE)),
        shuffle=bool(split.get("shuffle", False)),
        random_state=int(split.get("random_state", model.get("random_seed", 42))),
        iterations=int(model.get("iterations", 1000)),
        depth=int(model.get("depth", 6)),
        learning_rate=float(model.get("learning_rate", 0.03)),
        l2_leaf_reg=float(model.get("l2_leaf_reg", 0.35)),
        bagging_temperature=float(
            _first_defined(
                model.get("bagging_temperature"),
                default=DEFAULT_BAGGING_TEMPERATURE,
            )
        ),
        min_data_in_leaf=int(model.get("min_data_in_leaf", 80)),
        loss_function=model.get("loss_function", "Logloss"),
        eval_metric=model.get("eval_metric", "F1"),
        early_stopping_rounds=int(model.get("early_stopping_rounds", 100)),
        random_strength=float(model.get("random_strength", 1.0)),
        rsm=float(model.get("rsm", 1.0)),
        class_weight_negative=float(model.get("class_weight_negative", 1.0)),
        class_weight_positive=float(model.get("class_weight_positive", 4.0)),
        prediction_threshold=thresholding.get("prediction_threshold"),
        threshold_tuning_split=thresholding.get("tuning_split", "validation"),
        threshold_min_precision=thresholding.get("min_precision"),
        threshold_min_recall=thresholding.get("min_recall"),
        threshold_exclude_years=_as_int_list(thresholding.get("exclude_years")),
        calibration_enabled=bool(calibration.get("enabled", False)),
        calibration_method=calibration.get("method", "platt_month"),
        calibration_split=calibration.get("split", "validation"),
        calibration_exclude_years=_as_int_list(calibration.get("exclude_years"), [2021]),
        calibration_apply_to_metrics=bool(calibration.get("apply_to_metrics", True)),
        calibration_artifact_name=calibration.get("artifact_name", "probability_calibrator.joblib"),
        verbose=int(model.get("verbose", 50)),
        task_type=_first_defined(hardware.get("task_type"), model.get("task_type")),
        ignored_features=_as_list(features.get("ignored"), DEFAULT_IGNORED_FEATURES),
        selected_features_path=_path_or_none(features.get("selected_features_path")),
        no_selected_feature_filter=not bool(features.get("selected_feature_filter", True)),
        drop_feature=_as_list(features.get("drop_features")),
        drop_feature_prefix=_as_list(features.get("drop_feature_prefixes")),
        drop_feature_group=_as_list(features.get("drop_feature_groups")),
        feature_weight=features.get("feature_weights", {}),
        train_map_year=int(plots.get("train_map_year", 2011)),
        test_map_year=int(plots.get("test_map_year", 2025)),
        country_shapes_path=Path(
            plots.get("country_shapes_path", cfg.get("country_shapes_path", "data/countries"))
        ).expanduser(),
        no_map_plots=not bool(plots.get("map_plots", True)),
        no_region_map_plots=not bool(plots.get("region_map_plots", True)),
        region_map_year=plots.get("region_map_year"),
        save_eval_data=bool(data.get("save_eval_data", True)),
        no_full_eval_data=not bool(data.get("save_full_eval_data", True)),
        save_prediction_csv=bool(data.get("save_prediction_csv", True)),
        save_precision_recall_csv=bool(data.get("save_precision_recall_csv", True)),
        precision_recall_csv_max_points=data.get("precision_recall_csv_max_points"),
        no_feature_importance_analysis=not bool(importance.get("enabled", True)),
        importance_split=importance.get("split", "test"),
        importance_top_n=int(importance.get("top_n", 40)),
        importance_sample_size=int(importance.get("sample_size", 20000)),
        importance_random_state=int(importance.get("random_state", 42)),
        importance_permutation_repeats=int(importance.get("permutation_repeats", 3)),
        importance_permutation_top_n=int(importance.get("permutation_top_n", 40)),
        no_shap_importance=not bool(importance.get("include_shap", True)),
        no_permutation_importance=not bool(importance.get("include_permutation", True)),
        no_interaction_importance=not bool(importance.get("include_interactions", True)),
        limit_rows=data.get("limit_rows"),
        bootstrap_type=model.get("bootstrap_type"),
        subsample=model.get("subsample"),
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} did not contain a YAML mapping.")
    return config


def create_run_dir(output_root: Path, run_prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{run_prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return None if math.isnan(scalar) or math.isinf(scalar) else scalar
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(data), handle, indent=2)


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(sanitize_for_json(data), handle, sort_keys=False)


def parse_feature_weights(overrides: Iterable[str] | Dict[str, Any]) -> Dict[str, float]:
    feature_weights = dict(DEFAULT_FEATURE_WEIGHTS)
    if isinstance(overrides, dict):
        for feature, raw_weight in overrides.items():
            feature = str(feature).strip()
            if not feature:
                raise ValueError("Feature weight config contains an empty feature name.")
            feature_weights[feature] = float(raw_weight)
        return feature_weights

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid feature weight {override!r}. Expected FEATURE=WEIGHT.")
        feature, raw_weight = override.split("=", 1)
        feature = feature.strip()
        if not feature:
            raise ValueError(f"Invalid feature weight {override!r}: empty feature name.")
        feature_weights[feature] = float(raw_weight)
    return feature_weights


def load_or_build_features(
    features_path: Path,
    rebuild: bool,
    per_country_dir: Path,
    per_country_pattern: str,
    countries: Sequence[str],
) -> pd.DataFrame:
    if features_path.exists() and not rebuild:
        logger.info("Loading combined feature dataset from %s", features_path)
        return pd.read_parquet(features_path)

    if not countries:
        raise ValueError("No countries supplied for building the combined feature dataset.")

    logger.info("Building combined feature dataset at %s", features_path)
    missing_paths: List[Path] = []
    country_frames: List[pd.DataFrame] = []
    for country in countries:
        country_path = per_country_dir / per_country_pattern.format(country=country)
        if not country_path.exists():
            missing_paths.append(country_path)
            continue
        country_df = pd.read_parquet(country_path)
        if country == "Ukraine" and "datetime" in country_df.columns:
            country_df = country_df[pd.to_datetime(country_df["datetime"]) < "2022-02-24"]
        country_frames.append(country_df)

    if missing_paths:
        missing_preview = "\n".join(str(path) for path in missing_paths[:20])
        extra_count = len(missing_paths) - 20
        if extra_count > 0:
            missing_preview += f"\n... and {extra_count} more"
        raise FileNotFoundError(
            "Cannot build combined features because per-country files are missing:\n"
            f"{missing_preview}"
        )

    df = pd.concat(country_frames, ignore_index=True)
    if "datetime" in df.columns:
        df = df.sort_values("datetime")

    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path)
    logger.info("Saved combined feature dataset with shape %s to %s", df.shape, features_path)
    return df


def read_feature_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_selected_features(
    config: Dict[str, Any],
    override_path: Optional[Path] = None,
) -> Tuple[Optional[Path], List[str]]:
    raw_path = (
        override_path
        if override_path is not None
        else config.get("selected_feature_columns_path")
    )
    if not raw_path:
        return None, []
    selected_path = Path(raw_path)
    if not selected_path.exists():
        logger.warning("Selected feature columns file does not exist: %s", selected_path)
        return selected_path, []
    return selected_path, read_feature_list(selected_path)


def apply_selected_feature_filter(
    X: pd.DataFrame,
    selected_features: Sequence[str],
    enabled: bool,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    if not enabled or not selected_features:
        return X, [], sorted(set(selected_features) - set(X.columns)), []

    selected_existing = ordered_unique(
        feature for feature in selected_features if feature in X.columns
    )
    missing_selected = sorted(set(selected_features) - set(X.columns))
    if not selected_existing:
        raise ValueError(
            "Selected feature filtering was enabled, but none of the selected "
            "features are present in the dataframe."
        )

    dropped_by_filter = [
        feature for feature in X.columns if feature not in selected_existing
    ]
    return X.loc[:, selected_existing], selected_existing, missing_selected, dropped_by_filter


def apply_feature_exclusions(
    X: pd.DataFrame,
    drop_features: Sequence[str],
    drop_prefixes: Sequence[str],
    drop_groups: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
    drop_features_set = {feature for feature in drop_features if feature}
    drop_prefixes = [prefix for prefix in drop_prefixes if prefix]
    drop_groups_set = {group for group in drop_groups if group}

    exact_matches = sorted(drop_features_set & set(X.columns))
    prefix_matches = sorted(
        {
            feature
            for prefix in drop_prefixes
            for feature in X.columns
            if feature.startswith(prefix)
        }
    )
    group_matches = sorted(
        feature for feature in X.columns if infer_feature_group(feature) in drop_groups_set
    )
    dropped = sorted(set(exact_matches) | set(prefix_matches) | set(group_matches))
    if not dropped:
        return X, [], exact_matches, prefix_matches, group_matches
    return X.drop(columns=dropped), dropped, exact_matches, prefix_matches, group_matches


def prepare_dataframe(
    df: pd.DataFrame,
    config: Dict[str, Any],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str]]:
    df = df.copy()
    if target_column not in df.columns:
        raise KeyError(f"Target column {target_column!r} not found in dataframe.")

    if "population" in df.columns:
        df["population"] = df["population"].fillna(0).astype(int)

    configured_cat_features = list(config.get("cat_features", []) or [])
    numerical_cat_features = set(config.get("numerical_cat_features", []) or [])
    effective_cat_features: List[str] = []
    missing_cat_features: List[str] = []

    for col in configured_cat_features:
        if col not in df.columns:
            missing_cat_features.append(col)
            continue
        if col in numerical_cat_features:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(str)
        effective_cat_features.append(col)

    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y, configured_cat_features, effective_cat_features, missing_cat_features


def is_datetime_like_feature(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series):
        return False
    sample = series.dropna().head(20)
    return any(isinstance(value, (datetime, pd.Timestamp, np.datetime64)) for value in sample)


def prepare_catboost_input(
    X: pd.DataFrame,
    ignored_features: Sequence[str],
    cat_features: Sequence[str],
    feature_weights: Dict[str, float],
) -> Tuple[pd.DataFrame, List[str], Dict[str, float], List[str], List[str], List[str]]:
    dropped_features = [col for col in ignored_features if col in X.columns]
    X_model = X.drop(columns=dropped_features)
    datetime_features = [
        col for col in X_model.columns if is_datetime_like_feature(X_model[col])
    ]
    if datetime_features:
        dropped_features.extend(datetime_features)
        X_model = X_model.drop(columns=datetime_features)

    model_cat_features = [col for col in cat_features if col in X_model.columns]
    dropped_cat_features = [col for col in cat_features if col not in X_model.columns]

    model_feature_weights = {
        feature: weight for feature, weight in feature_weights.items() if feature in X_model.columns
    }
    dropped_feature_weights = [
        feature for feature in feature_weights if feature not in model_feature_weights
    ]

    return (
        X_model,
        model_cat_features,
        model_feature_weights,
        dropped_features,
        dropped_cat_features,
        dropped_feature_weights,
    )


def class_balance(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def extract_optional_soft_labels(
    df: pd.DataFrame,
    column: str,
    enabled: bool,
) -> Optional[pd.Series]:
    if not enabled:
        return None
    if not column:
        raise ValueError("Soft-label training is enabled, but target.soft_label_column is empty.")
    if column not in df.columns:
        raise ValueError(
            f"Soft-label training is enabled, but column {column!r} is missing from the "
            "feature table. Regenerate per-country features so target-derived columns are present."
        )
    soft = pd.to_numeric(df[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return soft.astype(float)


def expand_soft_labels_for_binary_training(
    X: pd.DataFrame,
    y_binary: pd.Series,
    soft_label: Optional[pd.Series],
) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": soft_label is not None,
        "input_rows": int(len(X)),
        "expanded_rows": int(len(X)),
        "soft_negative_rows": 0,
    }
    if soft_label is None:
        return X, y_binary.astype(int), None, info

    y = y_binary.astype(int)
    soft = soft_label.reindex(y.index).fillna(0.0).astype(float).clip(0.0, 1.0)
    soft = pd.Series(np.maximum(soft.to_numpy(dtype=float), y.to_numpy(dtype=float)), index=y.index)
    soft_negative_mask = (y == 0) & (soft > 0.0) & (soft < 1.0)
    info.update(
        {
            "soft_label_min": float(soft.min()) if len(soft) else None,
            "soft_label_max": float(soft.max()) if len(soft) else None,
            "soft_label_mean": float(soft.mean()) if len(soft) else None,
            "soft_negative_rows": int(soft_negative_mask.sum()),
        }
    )

    base_weight = np.ones(len(X), dtype=np.float32)
    base_weight[soft_negative_mask.to_numpy()] = (
        1.0 - soft.loc[soft_negative_mask].to_numpy(dtype=np.float32)
    )
    X_parts = [X]
    y_parts = [y]
    weight_parts = [base_weight]

    if soft_negative_mask.any():
        X_soft_positive = X.loc[soft_negative_mask]
        y_soft_positive = pd.Series(1, index=X_soft_positive.index, dtype=int)
        positive_weight = soft.loc[soft_negative_mask].to_numpy(dtype=np.float32)
        X_parts.append(X_soft_positive)
        y_parts.append(y_soft_positive)
        weight_parts.append(positive_weight)

    X_expanded = pd.concat(X_parts, axis=0)
    y_expanded = pd.concat(y_parts, axis=0).astype(int)
    weights = np.concatenate(weight_parts).astype(np.float32)
    info["expanded_rows"] = int(len(X_expanded))
    info["total_weight"] = float(weights.sum())
    info["positive_weight"] = float(weights[y_expanded.to_numpy(dtype=int) == 1].sum())
    info["negative_weight"] = float(weights[y_expanded.to_numpy(dtype=int) == 0].sum())
    return X_expanded, y_expanded, weights, info


def date_summary(
    frame: pd.DataFrame,
    date_column: str = "datetime",
) -> Dict[str, Optional[str]]:
    if date_column not in frame.columns:
        return {"min": None, "max": None, "start_year": None}
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.notna().sum() == 0:
        return {"min": None, "max": None, "start_year": None}
    min_date = dates.min()
    max_date = dates.max()
    return {
        "min": min_date.isoformat(),
        "max": max_date.isoformat(),
        "start_year": int(min_date.year),
    }


def parse_optional_timestamp(value: Optional[str], arg_name: str) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{arg_name} must be a valid date, got {value!r}.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{arg_name} must be a valid date, got {value!r}.")
    return timestamp


def validate_fraction(name: str, value: float, allow_zero: bool = False) -> None:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value >= 1.0:
        boundary = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{name} must be in {boundary}, got {value}.")


def validate_probability(name: str, value: Optional[float]) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def get_split_dates(X: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column not in X.columns:
        raise KeyError(f"Date column {date_column!r} not found in dataframe.")
    dates = pd.to_datetime(X[date_column], errors="coerce")
    if dates.isna().any():
        bad_count = int(dates.isna().sum())
        raise ValueError(
            f"Date column {date_column!r} contains {bad_count} values that cannot be parsed."
        )
    return dates


def sort_chronologically(
    X: pd.DataFrame,
    y: pd.Series,
    date_column: str,
    shuffle: bool,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    if shuffle or date_column not in X.columns:
        return X, y, False
    dates = get_split_dates(X, date_column)
    order = dates.sort_values(kind="stable").index
    if X.index.equals(order):
        return X, y, False
    return X.loc[order], y.loc[order], True


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> Tuple[
    pd.DataFrame,
    Optional[pd.DataFrame],
    pd.DataFrame,
    pd.Series,
    Optional[pd.Series],
    pd.Series,
    Dict[str, Any],
]:
    validate_fraction("--test-size", args.test_size)
    validate_fraction("--validation-size", args.validation_size, allow_zero=True)

    test_start = parse_optional_timestamp(args.test_start_date, "--test-start-date")
    validation_start = parse_optional_timestamp(
        args.validation_start_date,
        "--validation-start-date",
    )
    if validation_start is not None and test_start is not None and validation_start >= test_start:
        raise ValueError("--validation-start-date must be earlier than --test-start-date.")

    X, y, sorted_by_date = sort_chronologically(X, y, args.date_column, args.shuffle)
    split_info: Dict[str, Any] = {
        "date_column": args.date_column,
        "sorted_by_date": sorted_by_date,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "shuffle": args.shuffle,
        "random_state": args.random_state if args.shuffle else None,
        "test_start_date": test_start.isoformat() if test_start is not None else None,
        "validation_start_date": (
            validation_start.isoformat() if validation_start is not None else None
        ),
    }

    dates = (
        get_split_dates(X, args.date_column)
        if test_start is not None or validation_start is not None
        else None
    )
    if test_start is not None:
        assert dates is not None
        test_mask = dates >= test_start
        train_val_mask = ~test_mask
        if not test_mask.any() or not train_val_mask.any():
            raise ValueError(
                "--test-start-date produced an empty train or test split. "
                "Choose a date inside the dataset range."
            )
        X_train_val = X.loc[train_val_mask]
        y_train_val = y.loc[train_val_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]
        split_info["test_split_method"] = "date"
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            shuffle=args.shuffle,
            random_state=args.random_state if args.shuffle else None,
        )
        split_info["test_split_method"] = "fraction"

    if validation_start is not None:
        train_val_dates = get_split_dates(X_train_val, args.date_column)
        validation_mask = train_val_dates >= validation_start
        train_mask = ~validation_mask
        if not validation_mask.any() or not train_mask.any():
            raise ValueError(
                "--validation-start-date produced an empty train or validation split. "
                "Choose a date inside the pre-test range."
            )
        X_train = X_train_val.loc[train_mask]
        y_train = y_train_val.loc[train_mask]
        X_validation = X_train_val.loc[validation_mask]
        y_validation = y_train_val.loc[validation_mask]
        split_info["validation_split_method"] = "date"
    elif args.validation_size > 0.0:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_val,
            y_train_val,
            test_size=args.validation_size,
            shuffle=args.shuffle,
            random_state=args.random_state if args.shuffle else None,
        )
        split_info["validation_split_method"] = "fraction"
    else:
        X_train = X_train_val
        y_train = y_train_val
        X_validation = None
        y_validation = None
        split_info["validation_split_method"] = "disabled"

    split_info["rows"] = {
        "train": int(len(X_train)),
        "validation": int(len(X_validation)) if X_validation is not None else 0,
        "test": int(len(X_test)),
    }
    return X_train, X_validation, X_test, y_train, y_validation, y_test, split_info


def safe_metric(metric_name: str, func, *args) -> Optional[float]:
    try:
        value = func(*args)
    except ValueError as exc:
        logger.warning("Skipping %s: %s", metric_name, exc)
        return None
    scalar = float(value)
    if not math.isfinite(scalar):
        logger.warning("Skipping %s because it returned a non-finite value.", metric_name)
        return None
    return scalar


def evaluate_split(
    split_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    has_both_classes = len(np.unique(np.asarray(y_true))) == 2
    metrics = {
        "split": split_name,
        "rows": int(len(y_true)),
        "positives": int(np.asarray(y_true).sum()),
        "predicted_positives": int(np.asarray(y_pred).sum()),
        "threshold": threshold,
        "accuracy": safe_metric("accuracy", accuracy_score, y_true, y_pred),
        "precision": safe_metric(
            "precision",
            lambda yt, yp: precision_score(yt, yp, zero_division=0),
            y_true,
            y_pred,
        ),
        "recall": safe_metric(
            "recall",
            lambda yt, yp: recall_score(yt, yp, zero_division=0),
            y_true,
            y_pred,
        ),
        "f1": safe_metric(
            "f1",
            lambda yt, yp: f1_score(yt, yp, zero_division=0),
            y_true,
            y_pred,
        ),
        "roc_auc": (
            safe_metric("roc_auc", roc_auc_score, y_true, y_prob)
            if has_both_classes
            else None
        ),
        "average_precision": (
            safe_metric("average_precision", average_precision_score, y_true, y_prob)
            if has_both_classes
            else None
        ),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
            output_dict=True,
        ),
    }
    logger.info(
        "%s metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f AP=%s",
        split_name,
        metrics["accuracy"] if metrics["accuracy"] is not None else float("nan"),
        metrics["precision"] if metrics["precision"] is not None else float("nan"),
        metrics["recall"] if metrics["recall"] is not None else float("nan"),
        metrics["f1"] if metrics["f1"] is not None else float("nan"),
        (
            f"{metrics['average_precision']:.4f}"
            if metrics["average_precision"] is not None
            else "n/a"
        ),
    )
    return metrics


def save_metrics(metrics_by_split: Dict[str, Dict[str, Any]], eval_dir: Path) -> None:
    write_json(eval_dir / "metrics.json", metrics_by_split)
    rows = []
    for split_name, metrics in metrics_by_split.items():
        row = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float, str)) or value is None
        }
        row["split"] = split_name
        rows.append(row)
    pd.DataFrame(rows).to_csv(eval_dir / "metrics.csv", index=False)


def summarize_probability_metrics(
    split_name: str,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    year: Optional[int] = None,
) -> Dict[str, Any]:
    y_array = np.asarray(y_true).astype(int)
    prob_array = np.asarray(y_prob, dtype=float)
    pred_array = binary_from_probabilities(prob_array, threshold)
    has_both_classes = len(np.unique(y_array)) == 2
    rows = int(len(y_array))
    positives = int(y_array.sum())
    prevalence = float(np.mean(y_array)) if rows else None
    average_precision = (
        safe_metric("average_precision", average_precision_score, y_array, prob_array)
        if has_both_classes
        else None
    )
    normalized_ap = None
    ap_lift = None
    if average_precision is not None and prevalence is not None:
        if prevalence > 0:
            ap_lift = float(average_precision / prevalence)
        if prevalence < 1:
            normalized_ap = float((average_precision - prevalence) / (1.0 - prevalence))

    return {
        "split": split_name,
        "year": year,
        "rows": rows,
        "positives": positives,
        "prevalence": prevalence,
        "predicted_positives": int(pred_array.sum()),
        "predicted_positive_rate": float(np.mean(pred_array)) if rows else None,
        "probability_mean": float(np.mean(prob_array)) if rows else None,
        "probability_p95": float(np.quantile(prob_array, 0.95)) if rows else None,
        "threshold": threshold,
        "accuracy": safe_metric("accuracy", accuracy_score, y_array, pred_array),
        "precision": safe_metric(
            "precision",
            lambda yt, yp: precision_score(yt, yp, zero_division=0),
            y_array,
            pred_array,
        ),
        "recall": safe_metric(
            "recall",
            lambda yt, yp: recall_score(yt, yp, zero_division=0),
            y_array,
            pred_array,
        ),
        "f1": safe_metric(
            "f1",
            lambda yt, yp: f1_score(yt, yp, zero_division=0),
            y_array,
            pred_array,
        ),
        "roc_auc": (
            safe_metric("roc_auc", roc_auc_score, y_array, prob_array)
            if has_both_classes
            else None
        ),
        "average_precision": average_precision,
        "average_precision_lift": ap_lift,
        "normalized_average_precision": normalized_ap,
    }


def build_yearly_summary(
    split_name: str,
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    date_column: str = "datetime",
) -> List[Dict[str, Any]]:
    if date_column not in X.columns:
        logger.warning(
            "Skipping %s yearly summary because date column %s is missing.",
            split_name,
            date_column,
        )
        return []
    dates = pd.to_datetime(X[date_column], errors="coerce")
    if dates.isna().all():
        logger.warning(
            "Skipping %s yearly summary because date column %s could not be parsed.",
            split_name,
            date_column,
        )
        return []

    y_array = np.asarray(y_true).astype(int)
    prob_array = np.asarray(y_prob, dtype=float)
    rows: List[Dict[str, Any]] = []
    for year in sorted(int(value) for value in dates.dt.year.dropna().unique()):
        mask = dates.dt.year.to_numpy() == year
        if not mask.any():
            continue
        rows.append(
            summarize_probability_metrics(
                split_name=split_name,
                y_true=y_array[mask],
                y_prob=prob_array[mask],
                threshold=threshold,
                year=year,
            )
        )
    return rows


def save_yearly_summary(
    summaries_by_split: Dict[str, List[Dict[str, Any]]],
    eval_dir: Path,
) -> Optional[Path]:
    rows = [
        row
        for split_rows in summaries_by_split.values()
        for row in split_rows
    ]
    if not rows:
        return None
    output_path = eval_dir / "yearly_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    write_json(eval_dir / "yearly_metrics.json", {"rows": rows})
    return output_path


def log_yearly_summary(
    summaries_by_split: Dict[str, List[Dict[str, Any]]],
    title: str = "Yearly metrics",
) -> None:
    rows = [
        row
        for split_rows in summaries_by_split.values()
        for row in split_rows
    ]
    if not rows:
        return
    logger.info("%s:", title)
    for row in rows:
        logger.info(
            "  %-10s %s rows=%d pos=%d prev=%.4f AP=%s AUC=%s F1=%s P=%s R=%s pred_pos=%.4f",
            row.get("split"),
            row.get("year"),
            row.get("rows", 0),
            row.get("positives", 0),
            row.get("prevalence") if row.get("prevalence") is not None else float("nan"),
            (
                f"{row.get('average_precision'):.4f}"
                if row.get("average_precision") is not None
                else "n/a"
            ),
            f"{row.get('roc_auc'):.4f}" if row.get("roc_auc") is not None else "n/a",
            f"{row.get('f1'):.4f}" if row.get("f1") is not None else "n/a",
            f"{row.get('precision'):.4f}" if row.get("precision") is not None else "n/a",
            f"{row.get('recall'):.4f}" if row.get("recall") is not None else "n/a",
            (
                row.get("predicted_positive_rate")
                if row.get("predicted_positive_rate") is not None
                else float("nan")
            ),
        )


def build_eval_dataframe(
    split_name: str,
    X: pd.DataFrame,
    y_count: pd.Series,
    y_binary: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    full: bool,
) -> pd.DataFrame:
    if full:
        eval_df = X.copy()
    else:
        keep_cols = [
            col
            for col in ("datetime", "lat_rounded", "lon_rounded", "latitude", "longitude")
            if col in X.columns
        ]
        eval_df = X[keep_cols].copy()
    eval_df.insert(0, "eval_split", split_name)
    eval_df.insert(1, "eval_source_index", X.index.to_numpy())
    eval_df["eval_target_count"] = np.asarray(y_count)
    eval_df["eval_target_binary"] = np.asarray(y_binary)
    eval_df["eval_pred_binary"] = np.asarray(y_pred)
    eval_df["eval_pred_proba"] = np.asarray(y_prob)
    return eval_df


def save_predictions(
    split_name: str,
    X: pd.DataFrame,
    y_count: pd.Series,
    y_binary: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    eval_dir: Path,
    full: bool,
    save_evaluation_data: bool,
    save_compact_csv: bool,
) -> Optional[Path]:
    if not save_evaluation_data and not save_compact_csv:
        logger.info("Skipping saved prediction rows for %s split.", split_name)
        return None

    eval_df = build_eval_dataframe(
        split_name,
        X,
        y_count,
        y_binary,
        y_pred,
        y_prob,
        full if save_evaluation_data else False,
    )
    output_path: Optional[Path] = None
    if save_evaluation_data:
        output_path = eval_dir / f"{split_name}_evaluation_data.parquet"
        eval_df.to_parquet(output_path, index=False)

    if save_compact_csv:
        compact_cols = [
            col
            for col in (
                "eval_split",
                "eval_source_index",
                "datetime",
                "lat_rounded",
                "lon_rounded",
                "latitude",
                "longitude",
                "eval_target_count",
                "eval_target_binary",
                "eval_pred_binary",
                "eval_pred_proba",
            )
            if col in eval_df.columns
        ]
        eval_df[compact_cols].to_csv(
            eval_dir / f"{split_name}_predictions_compact.csv",
            index=False,
        )
    return output_path


def limit_curve_points(curve_df: pd.DataFrame, max_points: Optional[int]) -> pd.DataFrame:
    if max_points is None or max_points <= 0 or len(curve_df) <= max_points:
        return curve_df
    indices = np.linspace(0, len(curve_df) - 1, num=max_points, dtype=int)
    indices = np.unique(np.concatenate(([0, len(curve_df) - 1], indices)))
    return curve_df.iloc[indices].reset_index(drop=True)


def save_precision_recall(
    split_name: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
    eval_dir: Path,
    plots_dir: Path,
    save_curve_csv: bool,
    csv_max_points: Optional[int],
) -> Optional[Path]:
    unique_labels = np.unique(np.asarray(y_true))
    if len(unique_labels) < 2:
        logger.warning(
            "Skipping %s precision-recall curve because only one class is present.",
            split_name,
        )
        return None

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    curve_path: Optional[Path] = None
    if save_curve_csv:
        curve_df = pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": np.append(thresholds, np.nan),
            }
        )
        curve_df = limit_curve_points(curve_df, csv_max_points)
        curve_path = eval_dir / f"{split_name}_precision_recall_curve.csv"
        curve_df.to_csv(curve_path, index=False)

    avg_precision = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="blue", label=f"AP = {avg_precision:.4f}")
    ax.plot([0, 1], [baseline, baseline], "r--", label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{split_name.title()} Precision-Recall Curve")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc="lower left")
    fig.tight_layout()
    plot_path = plots_dir / f"{split_name}_precision_recall_curve.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return curve_path


def save_feature_importance(model: CatBoostClassifier, eval_dir: Path) -> Optional[Path]:
    try:
        importance = model.get_feature_importance()
        feature_names = model.feature_names_
    except Exception as exc:
        logger.warning("Could not compute feature importance: %s", exc)
        return None

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=False)
    output_path = eval_dir / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False)
    return output_path


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique_values: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def resolve_existing_model_path(
    args: argparse.Namespace,
    config: Dict[str, Any],
) -> Path:
    evaluation_config = config.get("evaluation")
    candidates: List[Any] = [args.model_path]
    if isinstance(evaluation_config, dict):
        candidates.append(evaluation_config.get("model_path"))
    candidates.append(config.get("model_path"))

    attempted: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        attempted.append(str(path))
        if path.exists():
            return path

    if attempted:
        raise FileNotFoundError(
            "Could not find an existing CatBoost model for analysis_only. "
            f"Checked: {attempted}"
        )
    raise ValueError(
        "analysis_only requires analysis.model_path or a model_path/evaluation.model_path "
        "entry in the config."
    )


def align_features_to_model(
    X_model: pd.DataFrame,
    model: CatBoostClassifier,
    split_name: str,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    model_feature_names = list(getattr(model, "feature_names_", []) or [])
    if not model_feature_names:
        return X_model, {"missing": [], "extra": []}

    missing = [feature for feature in model_feature_names if feature not in X_model.columns]
    if missing:
        preview = missing[:30]
        extra = len(missing) - len(preview)
        suffix = f" ... and {extra} more" if extra > 0 else ""
        raise KeyError(
            f"Model expects features missing from {split_name} data: {preview}{suffix}"
        )

    extra_features = [feature for feature in X_model.columns if feature not in model_feature_names]
    if extra_features:
        logger.info(
            "Dropping %d %s columns that are not used by the loaded/trained model.",
            len(extra_features),
            split_name,
        )
    return X_model[model_feature_names], {"missing": missing, "extra": extra_features}


def resolve_model_cat_features(
    model: CatBoostClassifier,
    configured_cat_features: Sequence[str],
    X_model: pd.DataFrame,
) -> List[str]:
    feature_names = list(getattr(model, "feature_names_", []) or X_model.columns)
    model_cat_names: List[str] = []
    try:
        model_cat_names = [
            feature_names[index]
            for index in model.get_cat_feature_indices()
            if index < len(feature_names)
        ]
    except Exception as exc:
        logger.debug("Could not read categorical feature indices from model: %s", exc)

    preferred = model_cat_names if model_cat_names else list(configured_cat_features)
    return [feature for feature in ordered_unique(preferred) if feature in X_model.columns]


def make_catboost_pool(
    X: pd.DataFrame,
    y: Optional[pd.Series | np.ndarray] = None,
    cat_features: Sequence[str] = (),
) -> Pool:
    pool_cat_features = [feature for feature in cat_features if feature in X.columns]
    if y is None:
        return Pool(X, cat_features=pool_cat_features) if pool_cat_features else Pool(X)
    return (
        Pool(X, label=np.asarray(y), cat_features=pool_cat_features)
        if pool_cat_features
        else Pool(X, label=np.asarray(y))
    )


def predict_probabilities(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    cat_features: Sequence[str],
) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(make_catboost_pool(X, cat_features=cat_features))
    )
    if probabilities.ndim == 2 and probabilities.shape[1] > 1:
        return probabilities[:, 1]
    return probabilities.reshape(-1)


def predict_binary(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    cat_features: Sequence[str],
) -> np.ndarray:
    predictions = np.asarray(
        model.predict(make_catboost_pool(X, cat_features=cat_features))
    )
    return predictions.astype(int).ravel()


def binary_from_probabilities(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_prob) >= threshold).astype(int)


def choose_threshold_by_f1(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> Dict[str, Any]:
    validate_probability("--threshold-min-precision", min_precision)
    validate_probability("--threshold-min-recall", min_recall)

    y_array = np.asarray(y_true).astype(int)
    prob_array = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_array)) < 2:
        return {
            "threshold": 0.5,
            "metric": "f1",
            "score": None,
            "precision": None,
            "recall": None,
            "reason": "only_one_class",
            "constraints_satisfied": False,
        }

    precision, recall, thresholds = precision_recall_curve(y_array, prob_array)
    if thresholds.size == 0:
        return {
            "threshold": 0.5,
            "metric": "f1",
            "score": None,
            "precision": None,
            "recall": None,
            "reason": "no_thresholds",
            "constraints_satisfied": False,
        }

    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    denominator = threshold_precision + threshold_recall
    f1 = np.divide(
        2.0 * threshold_precision * threshold_recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    valid = np.isfinite(f1)
    if min_precision is not None:
        valid &= threshold_precision >= min_precision
    if min_recall is not None:
        valid &= threshold_recall >= min_recall

    constraints_satisfied = bool(valid.any())
    if not constraints_satisfied:
        valid = np.isfinite(f1)
    if not valid.any():
        return {
            "threshold": 0.5,
            "metric": "f1",
            "score": None,
            "precision": None,
            "recall": None,
            "reason": "no_finite_scores",
            "constraints_satisfied": constraints_satisfied,
        }

    valid_indices = np.flatnonzero(valid)
    valid_scores = f1[valid_indices]
    best_score = np.nanmax(valid_scores)
    best_indices = valid_indices[np.isclose(valid_scores, best_score)]
    best_index = int(best_indices[-1])
    return {
        "threshold": float(thresholds[best_index]),
        "metric": "f1",
        "score": float(f1[best_index]),
        "precision": float(threshold_precision[best_index]),
        "recall": float(threshold_recall[best_index]),
        "reason": "best_f1",
        "constraints_satisfied": constraints_satisfied,
        "min_precision": min_precision,
        "min_recall": min_recall,
    }


def resolve_prediction_threshold(
    args: argparse.Namespace,
    split_probabilities: Dict[str, Tuple[pd.Series, np.ndarray]],
) -> Dict[str, Any]:
    validate_probability("--prediction-threshold", args.prediction_threshold)
    if args.prediction_threshold is not None:
        return {
            "threshold": float(args.prediction_threshold),
            "source": "fixed",
            "tuning_split": None,
        }

    if args.threshold_tuning_split == "none":
        return {
            "threshold": 0.5,
            "source": "default",
            "tuning_split": None,
        }

    tuning_data = split_probabilities.get(args.threshold_tuning_split)
    if tuning_data is None:
        logger.warning(
            "Threshold tuning split %s is not available; using 0.5.",
            args.threshold_tuning_split,
        )
        return {
            "threshold": 0.5,
            "source": "default_missing_tuning_split",
            "tuning_split": args.threshold_tuning_split,
        }

    y_true, y_prob = tuning_data
    tuning_result = choose_threshold_by_f1(
        y_true=y_true,
        y_prob=y_prob,
        min_precision=args.threshold_min_precision,
        min_recall=args.threshold_min_recall,
    )
    tuning_result["source"] = "tuned"
    tuning_result["tuning_split"] = args.threshold_tuning_split
    return tuning_result


def mask_excluding_years(
    X: pd.DataFrame,
    date_column: str,
    exclude_years: Sequence[int],
) -> np.ndarray:
    mask = np.ones(len(X), dtype=bool)
    years_to_exclude = {int(year) for year in exclude_years}
    if not years_to_exclude or date_column not in X.columns:
        return mask
    dates = pd.to_datetime(X[date_column], errors="coerce")
    years = dates.dt.year
    return (~years.isin(years_to_exclude)).fillna(False).to_numpy(dtype=bool)


def probability_calibration_frame(
    X: pd.DataFrame,
    y_true: Optional[pd.Series | np.ndarray],
    probabilities: np.ndarray,
    date_column: str,
    exclude_years: Sequence[int] = (),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    prob = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    mask = mask_excluding_years(X, date_column, exclude_years)
    frame = pd.DataFrame(
        {
            "prob_raw": prob[mask],
            "raw_score": logit(prob[mask]),
        }
    )
    if y_true is not None:
        frame["is_fire"] = np.asarray(y_true, dtype=int)[mask]
    if "month" in X.columns:
        frame["month"] = pd.to_numeric(X.loc[mask, "month"], errors="coerce").fillna(-1).astype(int).to_numpy()
    elif date_column in X.columns:
        frame["month"] = (
            pd.to_datetime(X.loc[mask, date_column], errors="coerce")
            .dt.month.fillna(-1)
            .astype(int)
            .to_numpy()
        )
    if "country" in X.columns:
        frame["country"] = X.loc[mask, "country"].fillna("missing").astype(str).to_numpy()

    info = {
        "input_rows": int(len(X)),
        "used_rows": int(mask.sum()),
        "excluded_rows": int((~mask).sum()),
        "exclude_years": [int(year) for year in exclude_years],
    }
    if "is_fire" in frame.columns and len(frame):
        info["used_positives"] = int(frame["is_fire"].sum())
        info["used_prevalence"] = float(frame["is_fire"].mean())
    return frame, info


def filter_probability_input_for_threshold(
    X: pd.DataFrame,
    y_true: pd.Series,
    probabilities: np.ndarray,
    date_column: str,
    exclude_years: Sequence[int],
) -> Tuple[pd.Series, np.ndarray, Dict[str, Any]]:
    mask = mask_excluding_years(X, date_column, exclude_years)
    info = {
        "input_rows": int(len(X)),
        "used_rows": int(mask.sum()),
        "excluded_rows": int((~mask).sum()),
        "exclude_years": [int(year) for year in exclude_years],
    }
    if not mask.any():
        logger.warning(
            "Threshold year filter removed every row; using unfiltered %s split.",
            date_column,
        )
        return y_true, probabilities, {**info, "fallback": "unfiltered_empty_after_year_filter"}
    return y_true.loc[mask], np.asarray(probabilities)[mask], info


def calibrate_split_probabilities(
    args: argparse.Namespace,
    model_dir: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_prob: np.ndarray,
    X_validation: Optional[pd.DataFrame],
    y_validation: Optional[pd.Series],
    validation_prob: Optional[np.ndarray],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_prob: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(args.calibration_enabled),
        "method": args.calibration_method,
        "split": args.calibration_split,
        "exclude_years": list(args.calibration_exclude_years),
        "apply_to_metrics": bool(args.calibration_apply_to_metrics),
        "applied": False,
    }
    if not args.calibration_enabled:
        return train_prob, validation_prob, test_prob, info
    if args.calibration_split != "validation":
        raise ValueError("Only calibration.split='validation' is supported for CatBoost training.")
    if X_validation is None or y_validation is None or validation_prob is None:
        logger.warning("Calibration requested but validation split is unavailable; using raw probabilities.")
        info["reason"] = "missing_validation_split"
        return train_prob, validation_prob, test_prob, info

    calibration_frame, frame_info = probability_calibration_frame(
        X_validation,
        y_validation,
        validation_prob,
        args.date_column,
        exclude_years=args.calibration_exclude_years,
    )
    info["frame"] = frame_info
    if len(calibration_frame) == 0 or calibration_frame["is_fire"].nunique() < 2:
        logger.warning(
            "Calibration skipped because filtered validation rows contain fewer than two classes."
        )
        info["reason"] = "insufficient_calibration_classes"
        return train_prob, validation_prob, test_prob, info

    logger.info(
        "Fitting %s calibrator on %d validation rows, excluding years %s.",
        args.calibration_method,
        len(calibration_frame),
        list(args.calibration_exclude_years),
    )
    calibrator = fit_calibrator(calibration_frame, args.calibration_method)
    calibrator_path = save_calibrator(calibrator, model_dir / args.calibration_artifact_name)

    train_frame, _ = probability_calibration_frame(
        X_train,
        y_train,
        train_prob,
        args.date_column,
    )
    validation_frame, _ = probability_calibration_frame(
        X_validation,
        y_validation,
        validation_prob,
        args.date_column,
    )
    test_frame, _ = probability_calibration_frame(
        X_test,
        y_test,
        test_prob,
        args.date_column,
    )
    calibrated_train = apply_calibrator(calibrator, train_frame)
    calibrated_validation = apply_calibrator(calibrator, validation_frame)
    calibrated_test = apply_calibrator(calibrator, test_frame)
    info.update(
        {
            "applied": bool(args.calibration_apply_to_metrics),
            "calibrator_path": str(calibrator_path),
            "calibrator_type": type(calibrator).__name__,
            "raw_probability_mean": {
                "train": float(np.mean(train_prob)),
                "validation": float(np.mean(validation_prob)),
                "test": float(np.mean(test_prob)),
            },
            "calibrated_probability_mean": {
                "train": float(np.mean(calibrated_train)),
                "validation": float(np.mean(calibrated_validation)),
                "test": float(np.mean(calibrated_test)),
            },
        }
    )
    if not args.calibration_apply_to_metrics:
        return train_prob, validation_prob, test_prob, info
    return calibrated_train, calibrated_validation, calibrated_test, info


def select_importance_splits(
    available_splits: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    requested_split: str,
) -> List[str]:
    if requested_split == "both":
        return [split for split in ("train", "test") if split in available_splits]
    return [requested_split] if requested_split in available_splits else []


def sample_analysis_rows(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    rows = len(X)
    if sample_size <= 0 or rows <= sample_size:
        return X, y, {
            "input_rows": rows,
            "sampled_rows": rows,
            "sampled": False,
            "sample_size_requested": sample_size,
            "stratified": False,
        }

    positions = np.arange(rows)
    y_values = np.asarray(y)
    stratified = False
    try:
        if len(np.unique(y_values)) > 1 and min(np.bincount(y_values.astype(int))) >= 2:
            sampled_positions, _ = train_test_split(
                positions,
                train_size=sample_size,
                stratify=y_values,
                random_state=random_state,
                shuffle=True,
            )
            stratified = True
        else:
            raise ValueError("not enough labels for stratified sampling")
    except Exception:
        rng = np.random.default_rng(random_state)
        sampled_positions = rng.choice(positions, size=sample_size, replace=False)

    sampled_positions = np.sort(sampled_positions)
    X_sample = X.iloc[sampled_positions].copy()
    y_sample = pd.Series(np.asarray(y)[sampled_positions], index=X_sample.index)
    return X_sample, y_sample, {
        "input_rows": rows,
        "sampled_rows": int(len(X_sample)),
        "sampled": True,
        "sample_size_requested": sample_size,
        "stratified": stratified,
    }


def infer_feature_group(feature: str) -> str:
    name = feature.lower()
    climate_groups = {
        "t2m": "climate_t2m",
        "d2m": "climate_d2m",
        "tp": "climate_tp",
        "stl1": "climate_stl1",
    }
    for token, group_name in climate_groups.items():
        if name == token or name.startswith(f"{token}_") or f"_{token}_" in name:
            return group_name
    if "ecoregion" in name or name in {"realm", "biome"}:
        return "ecoregion"
    if any(token in name for token in ("lat", "lon", "elevation", "topography", "slope", "aspect")):
        return "geospatial"
    if any(token in name for token in ("road", "distance_to_road")):
        return "infrastructure"
    if any(token in name for token in ("night", "light")):
        return "night_lights"
    if "population" in name:
        return "population"
    if any(token in name for token in ("forest", "vegetation", "soil", "landsea", "land_sea", "tvl", "tvh", "slt")):
        return "land_static"
    if "fire_index" in name:
        return "fire_index"
    if any(token in name for token in ("date", "time", "year", "month", "day")):
        return "time"
    return name.split("_", 1)[0] if "_" in name else "other"


def add_importance_columns(
    feature_names: Sequence[str],
    values: Sequence[float],
    method: str,
) -> pd.DataFrame:
    value_array = np.asarray(values, dtype=float).reshape(-1)
    if value_array.size < len(feature_names):
        value_array = np.pad(
            value_array,
            (0, len(feature_names) - value_array.size),
            constant_values=np.nan,
        )
    if value_array.size > len(feature_names):
        value_array = value_array[: len(feature_names)]

    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "group": [infer_feature_group(feature) for feature in feature_names],
            "method": method,
            "importance": value_array,
        }
    )
    df["importance_abs"] = df["importance"].abs()
    df["importance_positive"] = df["importance"].clip(lower=0)
    positive_sum = float(df["importance_positive"].sum(skipna=True))
    abs_sum = float(df["importance_abs"].sum(skipna=True))
    df["normalized_importance"] = (
        df["importance_positive"] / positive_sum if positive_sum > 0 else 0.0
    )
    df["normalized_abs_importance"] = (
        df["importance_abs"] / abs_sum if abs_sum > 0 else 0.0
    )
    df = df.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def shorten_label(value: str, max_len: int = 90) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def plot_top_bars(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    title: str,
    path: Path,
    top_n: int,
    xlabel: str = "Importance",
) -> Optional[Path]:
    if df.empty or value_col not in df.columns:
        return None
    plot_df = df.dropna(subset=[value_col]).sort_values(
        value_col, ascending=False, kind="mergesort"
    )
    plot_df = plot_df[plot_df[value_col].abs() > 1e-15]
    if top_n > 0:
        plot_df = plot_df.head(top_n)
    if plot_df.empty:
        return None

    plot_df = plot_df.iloc[::-1]
    labels = [shorten_label(str(value)) for value in plot_df[label_col]]
    height = max(4.0, min(18.0, 0.34 * len(plot_df) + 1.5))
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(labels, plot_df[value_col].to_numpy(dtype=float), color="#3b82f6")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_group_importance(
    importance_df: pd.DataFrame,
    method: str,
    output_dir: Path,
    plots_dir: Path,
    top_n: int,
) -> Dict[str, Optional[Path]]:
    if importance_df.empty:
        return {"csv": None, "plot": None}

    group_df = (
        importance_df.groupby("group", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            importance=("importance", "sum"),
            importance_positive=("importance_positive", "sum"),
            importance_abs=("importance_abs", "sum"),
            max_feature_importance=("importance", "max"),
        )
        .reset_index()
    )
    positive_sum = float(group_df["importance_positive"].sum(skipna=True))
    abs_sum = float(group_df["importance_abs"].sum(skipna=True))
    group_df["normalized_importance"] = (
        group_df["importance_positive"] / positive_sum if positive_sum > 0 else 0.0
    )
    group_df["normalized_abs_importance"] = (
        group_df["importance_abs"] / abs_sum if abs_sum > 0 else 0.0
    )
    sort_col = "normalized_importance" if positive_sum > 0 else "normalized_abs_importance"
    group_df = group_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    group_df["rank"] = np.arange(1, len(group_df) + 1)

    csv_path = output_dir / f"{method}_group_importance.csv"
    group_df.to_csv(csv_path, index=False)
    plot_path = plot_top_bars(
        group_df,
        value_col=sort_col,
        label_col="group",
        title=f"{method.replace('_', ' ').title()} By Feature Group",
        path=plots_dir / f"{method}_group_importance_top.png",
        top_n=top_n,
        xlabel="Normalized importance",
    )
    return {"csv": csv_path, "plot": plot_path}


def save_model_importance(
    model: CatBoostClassifier,
    feature_names: Sequence[str],
    method: str,
    importance_type: str,
    output_dir: Path,
    plots_dir: Path,
    top_n: int,
    data_pool: Optional[Pool] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Optional[Path]]]:
    try:
        if data_pool is not None:
            values = model.get_feature_importance(data=data_pool, type=importance_type)
        else:
            values = model.get_feature_importance(type=importance_type)
    except Exception as exc:
        logger.warning("Could not compute %s feature importance: %s", method, exc)
        return None, {"csv": None, "plot": None, "group_csv": None, "group_plot": None}

    importance_df = add_importance_columns(feature_names, values, method)
    csv_path = output_dir / f"{method}.csv"
    importance_df.to_csv(csv_path, index=False)
    plot_path = plot_top_bars(
        importance_df,
        value_col="importance",
        label_col="feature",
        title=method.replace("_", " ").title(),
        path=plots_dir / f"{method}_top.png",
        top_n=top_n,
    )
    group_artifacts = save_group_importance(importance_df, method, output_dir, plots_dir, top_n)
    return importance_df, {
        "csv": csv_path,
        "plot": plot_path,
        "group_csv": group_artifacts["csv"],
        "group_plot": group_artifacts["plot"],
    }


def probability_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Optional[float]]:
    y_array = np.asarray(y_true).astype(int)
    prob_array = np.asarray(y_prob, dtype=float)
    pred_array = (prob_array >= threshold).astype(int)
    has_both_classes = len(np.unique(y_array)) == 2
    return {
        "average_precision": (
            safe_metric("average_precision", average_precision_score, y_array, prob_array)
            if has_both_classes
            else None
        ),
        "roc_auc": (
            safe_metric("roc_auc", roc_auc_score, y_array, prob_array)
            if has_both_classes
            else None
        ),
        "f1": safe_metric(
            "f1",
            lambda yt, yp: f1_score(yt, yp, zero_division=0),
            y_array,
            pred_array,
        ),
        "log_loss": safe_metric(
            "log_loss",
            lambda yt, yp: log_loss(yt, yp, labels=[0, 1]),
            y_array,
            prob_array,
        ),
    }


def finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def metric_delta(base_value: Optional[float], permuted_value: Optional[float]) -> Optional[float]:
    if base_value is None or permuted_value is None:
        return None
    delta = float(base_value) - float(permuted_value)
    return delta if math.isfinite(delta) else None


def metric_increase(base_value: Optional[float], permuted_value: Optional[float]) -> Optional[float]:
    if base_value is None or permuted_value is None:
        return None
    delta = float(permuted_value) - float(base_value)
    return delta if math.isfinite(delta) else None


def mean_and_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    finite_values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite_values:
        return None, None
    if len(finite_values) == 1:
        return finite_values[0], 0.0
    return float(np.mean(finite_values)), float(np.std(finite_values, ddof=1))


def choose_permutation_score(row: Dict[str, Optional[float]]) -> float:
    for key in ("average_precision_drop_mean", "log_loss_increase_mean", "f1_drop_mean"):
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return 0.0


def save_permutation_importance(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: Sequence[str],
    candidate_features: Sequence[str],
    repeats: int,
    random_state: int,
    prediction_threshold: float,
    output_dir: Path,
    plots_dir: Path,
    top_n: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    if repeats <= 0:
        logger.info("Skipping permutation importance because repeats <= 0.")
        return None, {"csv": None, "plot": None, "group_csv": None, "group_plot": None}

    candidate_features = [feature for feature in ordered_unique(candidate_features) if feature in X.columns]
    if not candidate_features:
        logger.warning("Skipping permutation importance because no candidate features are available.")
        return None, {"csv": None, "plot": None, "group_csv": None, "group_plot": None}

    rng = np.random.default_rng(random_state)
    base_prob = predict_probabilities(model, X, cat_features)
    base_metrics = probability_metrics(y, base_prob, prediction_threshold)
    base_metrics_path = output_dir / "permutation_base_metrics.json"
    write_json(base_metrics_path, base_metrics)

    rows: List[Dict[str, Any]] = []
    logger.info(
        "Computing permutation importance for %d features x %d repeats.",
        len(candidate_features),
        repeats,
    )
    for feature_index, feature in enumerate(candidate_features, start=1):
        ap_drops: List[Optional[float]] = []
        auc_drops: List[Optional[float]] = []
        f1_drops: List[Optional[float]] = []
        loss_increases: List[Optional[float]] = []

        for _ in range(repeats):
            X_permuted = X.copy(deep=False)
            X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy(copy=True))
            permuted_prob = predict_probabilities(model, X_permuted, cat_features)
            permuted_metrics = probability_metrics(y, permuted_prob, prediction_threshold)

            ap_drops.append(
                metric_delta(base_metrics["average_precision"], permuted_metrics["average_precision"])
            )
            auc_drops.append(metric_delta(base_metrics["roc_auc"], permuted_metrics["roc_auc"]))
            f1_drops.append(metric_delta(base_metrics["f1"], permuted_metrics["f1"]))
            loss_increases.append(
                metric_increase(base_metrics["log_loss"], permuted_metrics["log_loss"])
            )

        ap_mean, ap_std = mean_and_std(ap_drops)
        auc_mean, auc_std = mean_and_std(auc_drops)
        f1_mean, f1_std = mean_and_std(f1_drops)
        loss_mean, loss_std = mean_and_std(loss_increases)
        row = {
            "feature": feature,
            "group": infer_feature_group(feature),
            "method": "permutation",
            "average_precision_drop_mean": finite_or_none(ap_mean),
            "average_precision_drop_std": finite_or_none(ap_std),
            "roc_auc_drop_mean": finite_or_none(auc_mean),
            "roc_auc_drop_std": finite_or_none(auc_std),
            "f1_drop_mean": finite_or_none(f1_mean),
            "f1_drop_std": finite_or_none(f1_std),
            "log_loss_increase_mean": finite_or_none(loss_mean),
            "log_loss_increase_std": finite_or_none(loss_std),
        }
        row["importance"] = choose_permutation_score(row)
        rows.append(row)

        if feature_index % 10 == 0 or feature_index == len(candidate_features):
            logger.info(
                "Permutation importance progress: %d/%d features.",
                feature_index,
                len(candidate_features),
            )

    importance_df = pd.DataFrame(rows)
    importance_df["importance_abs"] = importance_df["importance"].abs()
    importance_df["importance_positive"] = importance_df["importance"].clip(lower=0)
    positive_sum = float(importance_df["importance_positive"].sum(skipna=True))
    abs_sum = float(importance_df["importance_abs"].sum(skipna=True))
    importance_df["normalized_importance"] = (
        importance_df["importance_positive"] / positive_sum if positive_sum > 0 else 0.0
    )
    importance_df["normalized_abs_importance"] = (
        importance_df["importance_abs"] / abs_sum if abs_sum > 0 else 0.0
    )
    importance_df = importance_df.sort_values(
        "importance", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    csv_path = output_dir / "permutation_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    plot_path = plot_top_bars(
        importance_df,
        value_col="importance",
        label_col="feature",
        title="Permutation Importance",
        path=plots_dir / "permutation_importance_top.png",
        top_n=top_n,
        xlabel="Primary metric drop/increase",
    )
    group_artifacts = save_group_importance(
        importance_df,
        "permutation_importance",
        output_dir,
        plots_dir,
        top_n,
    )
    return importance_df, {
        "csv": csv_path,
        "plot": plot_path,
        "group_csv": group_artifacts["csv"],
        "group_plot": group_artifacts["plot"],
        "base_metrics_json": base_metrics_path,
        "base_metrics": base_metrics,
    }


def extract_shap_matrix(
    shap_values: np.ndarray,
    feature_count: int,
) -> np.ndarray:
    shap_array = np.asarray(shap_values, dtype=float)
    if shap_array.ndim == 2:
        return shap_array[:, :feature_count]
    if shap_array.ndim == 3:
        if shap_array.shape[0] >= 1 and shap_array.shape[-1] >= feature_count:
            if shap_array.shape[0] == feature_count + 1:
                return shap_array[:feature_count, :, -1].T
            if shap_array.shape[-1] == feature_count + 1:
                class_axis = 1 if shap_array.shape[1] <= 4 else 0
                if class_axis == 1:
                    class_index = min(1, shap_array.shape[1] - 1)
                    return shap_array[:, class_index, :feature_count]
                class_index = min(1, shap_array.shape[0] - 1)
                return shap_array[class_index, :, :feature_count]
    raise ValueError(f"Unsupported SHAP value shape: {shap_array.shape}")


def save_shap_summary_plot(
    shap_matrix: np.ndarray,
    X_sample: pd.DataFrame,
    importance_df: pd.DataFrame,
    feature_names: Sequence[str],
    plot_path: Path,
    top_n: int,
    random_state: int,
) -> Optional[Path]:
    if shap_matrix.size == 0 or importance_df.empty:
        return None

    top_features = [
        feature
        for feature in importance_df.sort_values("importance", ascending=False)["feature"].head(
            min(top_n, 20)
        )
        if feature in X_sample.columns and feature in feature_names
    ]
    if not top_features:
        return None

    rng = np.random.default_rng(random_state)
    plot_positions = np.arange(shap_matrix.shape[0])
    if plot_positions.size > 2500:
        plot_positions = np.sort(rng.choice(plot_positions, size=2500, replace=False))

    feature_to_index = {feature: index for index, feature in enumerate(feature_names)}
    fig_height = max(4.5, 0.38 * len(top_features) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    colored_scatter = None

    for y_position, feature in enumerate(reversed(top_features)):
        feature_index = feature_to_index[feature]
        shap_values = shap_matrix[plot_positions, feature_index]
        jitter = rng.normal(0, 0.07, size=len(plot_positions))
        y_values = np.full(len(plot_positions), y_position, dtype=float) + jitter

        numeric_values = pd.to_numeric(X_sample.iloc[plot_positions][feature], errors="coerce")
        if numeric_values.notna().sum() > 0:
            raw_values = numeric_values.to_numpy(dtype=float)
            finite_values = raw_values[np.isfinite(raw_values)]
            if finite_values.size > 0:
                low, high = np.nanpercentile(finite_values, [1, 99])
                if high > low:
                    color_values = np.clip((raw_values - low) / (high - low), 0, 1)
                else:
                    color_values = np.full_like(raw_values, 0.5, dtype=float)
                colored_scatter = ax.scatter(
                    shap_values,
                    y_values,
                    c=color_values,
                    cmap="coolwarm",
                    vmin=0,
                    vmax=1,
                    s=10,
                    alpha=0.55,
                    linewidths=0,
                )
                continue

        ax.scatter(
            shap_values,
            y_values,
            color="#64748b",
            s=10,
            alpha=0.45,
            linewidths=0,
        )

    ax.axvline(0, color="#334155", linewidth=1, alpha=0.75)
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels([shorten_label(feature) for feature in reversed(top_features)])
    ax.set_xlabel("SHAP contribution")
    ax.set_title("SHAP Summary")
    ax.grid(axis="x", alpha=0.25)
    if colored_scatter is not None:
        colorbar = fig.colorbar(colored_scatter, ax=ax, pad=0.01)
        colorbar.set_label("Feature value (clipped)")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_shap_importance(
    model: CatBoostClassifier,
    X_sample: pd.DataFrame,
    data_pool: Pool,
    feature_names: Sequence[str],
    output_dir: Path,
    plots_dir: Path,
    top_n: int,
    random_state: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Optional[Path]]]:
    try:
        shap_values = model.get_feature_importance(data=data_pool, type="ShapValues")
        shap_matrix = extract_shap_matrix(np.asarray(shap_values), len(feature_names))
    except Exception as exc:
        logger.warning("Could not compute SHAP feature importance: %s", exc)
        return None, {
            "csv": None,
            "plot": None,
            "summary_plot": None,
            "top_values_parquet": None,
            "group_csv": None,
            "group_plot": None,
        }

    usable_count = min(shap_matrix.shape[1], len(feature_names))
    shap_matrix = shap_matrix[:, :usable_count]
    feature_names = list(feature_names[:usable_count])

    mean_abs_shap = np.nanmean(np.abs(shap_matrix), axis=0)
    mean_shap = np.nanmean(shap_matrix, axis=0)
    std_shap = np.nanstd(shap_matrix, axis=0)
    importance_df = add_importance_columns(feature_names, mean_abs_shap, "shap_mean_abs")
    stats_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_shap": mean_shap,
            "std_shap": std_shap,
        }
    )
    importance_df = importance_df.merge(stats_df, on="feature", how="left")

    csv_path = output_dir / "shap_mean_abs_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    plot_path = plot_top_bars(
        importance_df,
        value_col="importance",
        label_col="feature",
        title="Mean Absolute SHAP Importance",
        path=plots_dir / "shap_mean_abs_importance_top.png",
        top_n=top_n,
    )
    summary_plot_path = save_shap_summary_plot(
        shap_matrix=shap_matrix,
        X_sample=X_sample,
        importance_df=importance_df,
        feature_names=feature_names,
        plot_path=plots_dir / "shap_summary_top.png",
        top_n=top_n,
        random_state=random_state,
    )

    top_features = importance_df["feature"].head(min(top_n, len(importance_df))).tolist()
    top_indices = [feature_names.index(feature) for feature in top_features if feature in feature_names]
    top_values_path: Optional[Path] = None
    if top_indices:
        shap_top_df = pd.DataFrame(
            shap_matrix[:, top_indices],
            columns=[feature_names[index] for index in top_indices],
        )
        shap_top_df.insert(0, "source_index", X_sample.index.to_numpy())
        top_values_path = output_dir / "shap_top_values.parquet"
        shap_top_df.to_parquet(top_values_path, index=False)

    group_artifacts = save_group_importance(
        importance_df,
        "shap_mean_abs",
        output_dir,
        plots_dir,
        top_n,
    )
    return importance_df, {
        "csv": csv_path,
        "plot": plot_path,
        "summary_plot": summary_plot_path,
        "top_values_parquet": top_values_path,
        "group_csv": group_artifacts["csv"],
        "group_plot": group_artifacts["plot"],
    }


def interaction_feature_name(value: Any, feature_names: Sequence[str]) -> str:
    try:
        index = int(value)
        if float(value) == float(index) and 0 <= index < len(feature_names):
            return feature_names[index]
    except Exception:
        pass
    return str(value)


def save_interaction_importance(
    model: CatBoostClassifier,
    feature_names: Sequence[str],
    output_dir: Path,
    plots_dir: Path,
    top_n: int,
) -> Dict[str, Optional[Path]]:
    try:
        raw_interactions = np.asarray(model.get_feature_importance(type="Interaction"))
    except Exception as exc:
        logger.warning("Could not compute CatBoost feature interactions: %s", exc)
        return {"csv": None, "plot": None}

    if raw_interactions.size == 0:
        return {"csv": None, "plot": None}
    if raw_interactions.ndim == 1 and raw_interactions.size % 3 == 0:
        raw_interactions = raw_interactions.reshape(-1, 3)
    if raw_interactions.ndim != 2 or raw_interactions.shape[1] < 3:
        logger.warning("Unexpected CatBoost interaction shape: %s", raw_interactions.shape)
        return {"csv": None, "plot": None}

    rows = []
    for row in raw_interactions:
        feature_1 = interaction_feature_name(row[0], feature_names)
        feature_2 = interaction_feature_name(row[1], feature_names)
        try:
            score = float(row[2])
        except Exception:
            continue
        rows.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "group_1": infer_feature_group(feature_1),
                "group_2": infer_feature_group(feature_2),
                "interaction": score,
                "pair": f"{feature_1} x {feature_2}",
            }
        )

    if not rows:
        return {"csv": None, "plot": None}

    interactions_df = pd.DataFrame(rows).sort_values(
        "interaction", ascending=False, kind="mergesort"
    )
    interactions_df["rank"] = np.arange(1, len(interactions_df) + 1)
    csv_path = output_dir / "feature_interactions.csv"
    interactions_df.to_csv(csv_path, index=False)
    plot_path = plot_top_bars(
        interactions_df,
        value_col="interaction",
        label_col="pair",
        title="CatBoost Feature Interactions",
        path=plots_dir / "feature_interactions_top.png",
        top_n=top_n,
        xlabel="Interaction score",
    )
    return {"csv": csv_path, "plot": plot_path}


def combine_importance_tables(method_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    feature_order: List[str] = []
    for table in method_tables.values():
        if table is None or table.empty or "feature" not in table.columns:
            continue
        feature_order.extend(str(feature) for feature in table["feature"].tolist())
    feature_order = ordered_unique(feature_order)
    if not feature_order:
        return pd.DataFrame()

    summary_df = pd.DataFrame({"feature": feature_order})
    summary_df["group"] = summary_df["feature"].map(infer_feature_group)
    rank_cols: List[str] = []
    norm_cols: List[str] = []

    for method, table in method_tables.items():
        if table is None or table.empty:
            continue
        method_df = table[
            ["feature", "importance", "rank", "normalized_importance", "normalized_abs_importance"]
        ].copy()
        method_df = method_df.rename(
            columns={
                "importance": f"{method}_importance",
                "rank": f"{method}_rank",
                "normalized_importance": f"{method}_normalized_importance",
                "normalized_abs_importance": f"{method}_normalized_abs_importance",
            }
        )
        summary_df = summary_df.merge(method_df, on="feature", how="left")
        rank_cols.append(f"{method}_rank")
        norm_cols.append(f"{method}_normalized_importance")

    summary_df["methods_present"] = summary_df[rank_cols].notna().sum(axis=1)
    summary_df["mean_rank"] = summary_df[rank_cols].mean(axis=1)
    summary_df["mean_normalized_importance"] = summary_df[norm_cols].mean(axis=1)
    summary_df = summary_df.sort_values(
        ["mean_rank", "mean_normalized_importance"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary_df["consensus_rank"] = np.arange(1, len(summary_df) + 1)
    return summary_df


def save_consensus_group_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
    plots_dir: Path,
    top_n: int,
) -> Dict[str, Optional[Path]]:
    if summary_df.empty:
        return {"csv": None, "plot": None}
    group_df = (
        summary_df.groupby("group", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            group_score=("mean_normalized_importance", "sum"),
            best_consensus_rank=("consensus_rank", "min"),
            mean_consensus_rank=("consensus_rank", "mean"),
        )
        .reset_index()
        .sort_values(["group_score", "best_consensus_rank"], ascending=[False, True])
    )
    score_sum = float(group_df["group_score"].sum(skipna=True))
    group_df["normalized_group_score"] = group_df["group_score"] / score_sum if score_sum > 0 else 0.0
    group_df["rank"] = np.arange(1, len(group_df) + 1)

    csv_path = output_dir / "feature_importance_group_summary.csv"
    group_df.to_csv(csv_path, index=False)
    plot_path = plot_top_bars(
        group_df,
        value_col="normalized_group_score",
        label_col="group",
        title="Consensus Feature Importance By Group",
        path=plots_dir / "feature_importance_group_summary_top.png",
        top_n=top_n,
        xlabel="Normalized consensus score",
    )
    return {"csv": csv_path, "plot": plot_path}


def run_feature_importance_analysis(
    model: CatBoostClassifier,
    split_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    cat_features: Sequence[str],
    eval_dir: Path,
    plots_dir: Path,
    requested_split: str,
    top_n: int,
    sample_size: int,
    random_state: int,
    permutation_repeats: int,
    permutation_top_n: int,
    include_shap: bool,
    include_permutation: bool,
    include_interactions: bool,
    prediction_threshold: float = 0.5,
) -> Dict[str, Any]:
    analysis_root = eval_dir / "feature_importance"
    plot_root = plots_dir / "feature_importance"
    analysis_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    selected_splits = select_importance_splits(split_data, requested_split)
    artifacts: Dict[str, Any] = {
        "root": analysis_root,
        "plots_root": plot_root,
        "requested_split": requested_split,
        "splits": {},
    }
    if not selected_splits:
        logger.warning("No splits selected for feature-importance analysis.")
        return artifacts

    for split_name in selected_splits:
        X_split, y_split = split_data[split_name]
        split_dir = analysis_root / split_name
        split_plot_dir = plot_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_plot_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running feature-importance analysis on %s split.", split_name)
        X_sample, y_sample, sample_info = sample_analysis_rows(
            X_split,
            y_split,
            sample_size=sample_size,
            random_state=random_state,
        )
        feature_names = list(getattr(model, "feature_names_", []) or X_sample.columns)
        feature_names = [feature for feature in feature_names if feature in X_sample.columns]
        if feature_names:
            X_sample = X_sample[feature_names]
        split_cat_features = [feature for feature in cat_features if feature in X_sample.columns]
        data_pool = make_catboost_pool(X_sample, y_sample, split_cat_features)

        method_tables: Dict[str, pd.DataFrame] = {}
        split_artifacts: Dict[str, Any] = {
            "sample": sample_info,
            "feature_count": len(feature_names),
            "cat_features": split_cat_features,
            "methods": {},
        }

        pvc_df, pvc_artifacts = save_model_importance(
            model=model,
            feature_names=feature_names,
            method="prediction_values_change",
            importance_type="PredictionValuesChange",
            output_dir=split_dir,
            plots_dir=split_plot_dir,
            top_n=top_n,
        )
        if pvc_df is not None:
            method_tables["prediction_values_change"] = pvc_df
        split_artifacts["methods"]["prediction_values_change"] = pvc_artifacts

        loss_df, loss_artifacts = save_model_importance(
            model=model,
            feature_names=feature_names,
            method="loss_function_change",
            importance_type="LossFunctionChange",
            output_dir=split_dir,
            plots_dir=split_plot_dir,
            top_n=top_n,
            data_pool=data_pool,
        )
        if loss_df is not None:
            method_tables["loss_function_change"] = loss_df
        split_artifacts["methods"]["loss_function_change"] = loss_artifacts

        if include_shap:
            shap_df, shap_artifacts = save_shap_importance(
                model=model,
                X_sample=X_sample,
                data_pool=data_pool,
                feature_names=feature_names,
                output_dir=split_dir,
                plots_dir=split_plot_dir,
                top_n=top_n,
                random_state=random_state,
            )
            if shap_df is not None:
                method_tables["shap_mean_abs"] = shap_df
            split_artifacts["methods"]["shap_mean_abs"] = shap_artifacts

        if include_permutation:
            if pvc_df is not None and not pvc_df.empty:
                candidate_features = pvc_df["feature"].tolist()
            else:
                candidate_features = feature_names
            if permutation_top_n > 0:
                candidate_features = candidate_features[:permutation_top_n]
            permutation_df, permutation_artifacts = save_permutation_importance(
                model=model,
                X=X_sample,
                y=y_sample,
                cat_features=split_cat_features,
                candidate_features=candidate_features,
                repeats=permutation_repeats,
                random_state=random_state,
                prediction_threshold=prediction_threshold,
                output_dir=split_dir,
                plots_dir=split_plot_dir,
                top_n=top_n,
            )
            if permutation_df is not None:
                method_tables["permutation"] = permutation_df
            split_artifacts["methods"]["permutation"] = permutation_artifacts

        if include_interactions:
            split_artifacts["methods"]["interactions"] = save_interaction_importance(
                model=model,
                feature_names=feature_names,
                output_dir=split_dir,
                plots_dir=split_plot_dir,
                top_n=top_n,
            )

        summary_df = combine_importance_tables(method_tables)
        if not summary_df.empty:
            summary_path = split_dir / "feature_importance_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            summary_plot_path = plot_top_bars(
                summary_df,
                value_col="mean_normalized_importance",
                label_col="feature",
                title="Consensus Feature Importance",
                path=split_plot_dir / "feature_importance_summary_top.png",
                top_n=top_n,
                xlabel="Mean normalized importance",
            )
            group_summary_artifacts = save_consensus_group_summary(
                summary_df,
                output_dir=split_dir,
                plots_dir=split_plot_dir,
                top_n=top_n,
            )
            split_artifacts["summary_csv"] = summary_path
            split_artifacts["summary_plot"] = summary_plot_path
            split_artifacts["group_summary_csv"] = group_summary_artifacts["csv"]
            split_artifacts["group_summary_plot"] = group_summary_artifacts["plot"]
        else:
            split_artifacts["summary_csv"] = None
            split_artifacts["summary_plot"] = None
            split_artifacts["group_summary_csv"] = None
            split_artifacts["group_summary_plot"] = None

        metadata_path = split_dir / "feature_importance_metadata.json"
        write_json(
            metadata_path,
            {
                "split": split_name,
                "sample": sample_info,
                "feature_count": len(feature_names),
                "cat_features": split_cat_features,
                "methods": list(split_artifacts["methods"].keys()),
                "top_n": top_n,
                "permutation_repeats": permutation_repeats,
                "permutation_top_n": permutation_top_n,
                "include_shap": include_shap,
                "include_permutation": include_permutation,
                "include_interactions": include_interactions,
            },
        )
        split_artifacts["metadata_json"] = metadata_path
        artifacts["splits"][split_name] = split_artifacts

    return artifacts


def primary_feature_importance_path(
    artifacts: Dict[str, Any],
    preferred_split: str = "test",
) -> Optional[Path]:
    splits = artifacts.get("splits")
    if not isinstance(splits, dict) or not splits:
        return None
    ordered_split_names = [preferred_split] + [
        split_name for split_name in splits if split_name != preferred_split
    ]
    for split_name in ordered_split_names:
        split_artifacts = splits.get(split_name)
        if not isinstance(split_artifacts, dict):
            continue
        summary_path = split_artifacts.get("summary_csv")
        if summary_path:
            return Path(summary_path)
        methods = split_artifacts.get("methods")
        if isinstance(methods, dict):
            prediction_values = methods.get("prediction_values_change")
            if isinstance(prediction_values, dict) and prediction_values.get("csv"):
                return Path(prediction_values["csv"])
    return None


def safe_filename(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "plot"


def load_world_background(country_shapes_path: Path) -> Optional[gpd.GeoDataFrame]:
    if not country_shapes_path.exists():
        logger.warning("Country shapes path does not exist: %s", country_shapes_path)
        return None
    try:
        return gpd.read_file(country_shapes_path)
    except Exception as exc:
        logger.warning("Could not read country background from %s: %s", country_shapes_path, exc)
        return None


def save_actual_vs_predicted_map(
    split_name: str,
    X: pd.DataFrame,
    y_true_binary: pd.Series,
    y_pred_binary: np.ndarray,
    year: int,
    world: Optional[gpd.GeoDataFrame],
    plots_dir: Path,
) -> Optional[Path]:
    required = {"datetime", "lat_rounded", "lon_rounded"}
    missing = sorted(required - set(X.columns))
    if missing:
        logger.warning(
            "Skipping %s map for %s because columns are missing: %s",
            split_name,
            year,
            missing,
        )
        return None

    dates = pd.to_datetime(X["datetime"], errors="coerce")
    year_mask = np.asarray(dates.dt.year == year)
    if year_mask.sum() == 0:
        logger.warning("Skipping %s map for %s because no rows match that year.", split_name, year)
        return None

    fig, ax = plt.subplots(figsize=(15, 10))
    if world is not None:
        world.plot(ax=ax, color="lightgrey", edgecolor="black")

    lat = X.loc[year_mask, "lat_rounded"]
    lon = X.loc[year_mask, "lon_rounded"]
    pred_count = np.asarray(y_pred_binary)[year_mask]
    actual_count = np.asarray(y_true_binary)[year_mask]

    ax.scatter(
        lon,
        lat,
        c="red",
        s=pred_count * 23,
        alpha=0.3,
        label="Predicted",
        edgecolors="none",
    )
    ax.scatter(
        lon,
        lat,
        c="blue",
        s=actual_count * 12,
        alpha=0.5,
        label="Actual",
        edgecolors="none",
    )
    ax.set_xlim(20, 170)
    ax.set_ylim(40, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Actual vs Predicted Fire Hotspots ({year}) - {split_name.title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = plots_dir / f"{split_name}_actual_vs_predicted_{year}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_plot_boosting_region_maps(
    split_name: str,
    X: pd.DataFrame,
    y_true_binary: pd.Series,
    y_pred_binary: np.ndarray,
    year: int,
    world: Optional[gpd.GeoDataFrame],
    plots_dir: Path,
    eval_dir: Path,
    regions: Dict[str, Dict[str, Tuple[float, float]]] = DEFAULT_PLOT_REGIONS,
) -> Tuple[Dict[str, Optional[Path]], Optional[Path], Optional[Path]]:
    required = {"datetime", "lat_rounded", "lon_rounded"}
    missing = sorted(required - set(X.columns))
    if missing:
        logger.warning(
            "Skipping %s plot_boosting region maps for %s because columns are missing: %s",
            split_name,
            year,
            missing,
        )
        return {}, None, None

    dates = pd.to_datetime(X["datetime"], errors="coerce")
    year_mask = np.asarray(dates.dt.year == year)
    if year_mask.sum() == 0:
        logger.warning(
            "Skipping %s plot_boosting region maps for %s because no rows match that year.",
            split_name,
            year,
        )
        return {}, None, None

    lat = X.loc[year_mask, "lat_rounded"].reset_index(drop=True)
    lon = X.loc[year_mask, "lon_rounded"].reset_index(drop=True)
    pred_count = pd.Series(np.asarray(y_pred_binary)[year_mask]).reset_index(drop=True)
    actual_count = pd.Series(np.asarray(y_true_binary)[year_mask]).reset_index(drop=True)

    region_dir = plots_dir / "plot_boosting_regions"
    region_dir.mkdir(parents=True, exist_ok=True)
    region_paths: Dict[str, Optional[Path]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for name, region_info in regions.items():
        lon_range = region_info["lon_range"]
        lat_range = region_info["lat_range"]
        region_mask = (
            (lon >= lon_range[0])
            & (lon <= lon_range[1])
            & (lat >= lat_range[0])
            & (lat <= lat_range[1])
        )

        rows = int(region_mask.sum())
        if rows == 0:
            logger.info("No %s rows for plot_boosting region %s in %s.", split_name, name, year)
            region_paths[name] = None
            summary_rows.append(
                {
                    "split": split_name,
                    "year": year,
                    "region": name,
                    "rows": 0,
                    "actual_positives": 0,
                    "predicted_positives": 0,
                    "plot_path": None,
                }
            )
            continue

        lon_region = lon.loc[region_mask]
        lat_region = lat.loc[region_mask]
        actual_region = actual_count.loc[region_mask]
        pred_region = pred_count.loc[region_mask]

        fig, ax = plt.subplots(figsize=(10, 8))
        if world is not None:
            world.plot(ax=ax, color="lightgrey", edgecolor="black")

        ax.scatter(
            lon_region,
            lat_region,
            c="red",
            s=np.clip(pred_region * 23, 0, 400),
            alpha=0.3,
            label="Predicted",
            edgecolors="none",
        )
        ax.scatter(
            lon_region,
            lat_region,
            c="blue",
            s=np.clip(actual_region * 12, 0, 400),
            alpha=0.5,
            label="Actual",
            edgecolors="none",
        )
        ax.set_xlim(lon_range[0], lon_range[1])
        ax.set_ylim(lat_range[0], lat_range[1])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"{name} ({lon_range[0]:g}-{lon_range[1]:g}E, "
            f"{lat_range[0]:g}-{lat_range[1]:g}N), Year {year}"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = region_dir / f"{split_name}_hotspots_{safe_filename(name)}_{year}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        region_paths[name] = plot_path
        summary_rows.append(
            {
                "split": split_name,
                "year": year,
                "region": name,
                "rows": rows,
                "actual_positives": int(actual_region.sum()),
                "predicted_positives": int(pred_region.sum()),
                "plot_path": plot_path,
            }
        )

    fig, ax = plt.subplots(figsize=(15, 10))
    if world is not None:
        world.plot(ax=ax, color="lightgrey", edgecolor="black")

    pred_sizes = np.clip(23 * pred_count, 0, 300)
    actual_sizes = np.clip(12 * actual_count, 0, 300)

    ax.scatter(
        lon,
        lat,
        c="red",
        s=pred_sizes,
        alpha=0.4,
        label="Predicted",
        edgecolors="none",
    )
    ax.scatter(
        lon,
        lat,
        c="blue",
        s=actual_sizes,
        alpha=0.6,
        label="Actual",
        edgecolors="none",
    )

    for name, region_info in regions.items():
        lon_range = region_info["lon_range"]
        lat_range = region_info["lat_range"]
        rect = plt.Rectangle(
            (lon_range[0], lat_range[0]),
            lon_range[1] - lon_range[0],
            lat_range[1] - lat_range[0],
            fill=False,
            linewidth=2,
            linestyle="--",
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            lon_range[0] + (lon_range[1] - lon_range[0]) / 2,
            lat_range[1] + 1,
            name,
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
        )

    ax.set_xlim(0, 180)
    ax.set_ylim(30, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Overview: Actual vs Predicted Fire Hotspots with Region Boundaries ({year})"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    overview_path = region_dir / f"{split_name}_overview_with_regions_{year}.png"
    fig.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_path = eval_dir / f"{split_name}_plot_boosting_region_summary_{year}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return region_paths, overview_path, summary_path


def collect_environment() -> Dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "catboost": catboost.__version__,
        "geopandas": gpd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def build_model_params(
    args: argparse.Namespace,
    cat_features: Sequence[str],
    feature_weights: Dict[str, float],
    run_dir: Path,
) -> Dict[str, Any]:
    verbose: Any = args.verbose
    if args.verbose <= 0:
        verbose = False

    params: Dict[str, Any] = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "min_data_in_leaf": args.min_data_in_leaf,
        "loss_function": args.loss_function,
        "eval_metric": args.eval_metric,
        "class_weights": [args.class_weight_negative, args.class_weight_positive],
        "random_seed": args.random_state,
        "random_strength": args.random_strength,
        "verbose": verbose,
        "cat_features": list(cat_features),
        "feature_weights": feature_weights,
        "train_dir": str(run_dir / "catboost_info"),
    }
    if args.bootstrap_type in (None, "Bayesian"):
        params["bagging_temperature"] = args.bagging_temperature
    elif args.bagging_temperature != DEFAULT_BAGGING_TEMPERATURE:
        logger.warning(
            "Ignoring bagging_temperature because it is only valid with Bayesian bootstrap."
        )
    if args.rsm < 1.0 and args.task_type == "GPU":
        logger.warning("Ignoring --rsm because CatBoost GPU does not support it here.")
    else:
        params["rsm"] = args.rsm
    if args.task_type:
        params["task_type"] = args.task_type
    if args.bootstrap_type:
        params["bootstrap_type"] = args.bootstrap_type
    if args.subsample is not None:
        params["subsample"] = args.subsample
    return params


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli_args = parse_args(argv)
    args = build_args_from_config(cli_args.config)
    run_dir = create_run_dir(args.output_root, args.run_prefix)
    setup_logging(run_dir)

    start_time = perf_counter()
    eval_dir = run_dir / "evaluation"
    plots_dir = run_dir / "plots"
    model_dir = run_dir / "model"
    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Created run directory: %s", run_dir.resolve())
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds must be >= 0.")
    if args.random_strength < 0:
        raise ValueError("--random-strength must be >= 0.")
    if not 0.0 < args.rsm <= 1.0:
        raise ValueError("--rsm must be in (0, 1].")
    if args.subsample is not None and not 0.0 < args.subsample <= 1.0:
        raise ValueError("--subsample must be in (0, 1].")
    validate_probability("--prediction-threshold", args.prediction_threshold)
    validate_probability("--threshold-min-precision", args.threshold_min_precision)
    validate_probability("--threshold-min-recall", args.threshold_min_recall)
    config = load_config(args.config)
    if args.run_config.is_file():
        shutil.copy2(args.run_config, run_dir / "run_config.yaml")
    else:
        logger.info("Skipping run config copy because %s is not a regular file.", args.run_config)
    shutil.copy2(args.config, run_dir / "feature_config.yaml")

    selected_features_path, selected_features = load_selected_features(
        config,
        override_path=args.selected_features_path,
    )
    feature_weights = parse_feature_weights(args.feature_weight)

    df = load_or_build_features(
        features_path=args.features_path,
        rebuild=args.rebuild_features,
        per_country_dir=args.per_country_features_dir,
        per_country_pattern=args.per_country_pattern,
        countries=args.countries,
    )
    if args.limit_rows is not None:
        logger.info("Limiting dataframe from %d to %d rows.", len(df), args.limit_rows)
        df = df.head(args.limit_rows)

    soft_label_all = extract_optional_soft_labels(
        df,
        column=args.soft_label_column,
        enabled=args.use_soft_labels_for_training,
    )
    X, y, configured_cat_features, cat_features, missing_cat_features = prepare_dataframe(
        df=df,
        config=config,
        target_column=args.target_column,
    )
    selected_feature_filter_enabled = (
        bool(selected_features) and not args.no_selected_feature_filter
    )
    (
        X,
        applied_selected_features,
        missing_selected_features,
        dropped_by_selected_feature_filter,
    ) = apply_selected_feature_filter(
        X=X,
        selected_features=selected_features,
        enabled=selected_feature_filter_enabled,
    )
    if selected_feature_filter_enabled:
        logger.info(
            "Applied selected feature filter: kept %d columns, dropped %d columns.",
            len(applied_selected_features),
            len(dropped_by_selected_feature_filter),
        )
    elif selected_features:
        logger.info(
            "Selected feature file was loaded but filtering is disabled; using all dataframe columns."
        )

    (
        X,
        dropped_by_feature_exclusions,
        dropped_exact_features,
        dropped_prefix_features,
        dropped_group_features,
    ) = apply_feature_exclusions(
        X=X,
        drop_features=args.drop_feature,
        drop_prefixes=args.drop_feature_prefix,
        drop_groups=args.drop_feature_group,
    )
    if dropped_by_feature_exclusions:
        logger.info(
            "Dropped %d columns by explicit feature exclusions.",
            len(dropped_by_feature_exclusions),
        )
    if not args.shuffle and args.date_column not in X.columns:
        raise ValueError(
            f"Date column {args.date_column!r} is required for the default "
            "chronological split, but it was removed by selected-feature "
            "filtering or feature exclusions."
        )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        split_info,
    ) = split_train_validation_test(X, y, args)
    y_train_binary = (y_train > args.positive_threshold).astype(int)
    y_validation_binary = (
        (y_validation > args.positive_threshold).astype(int)
        if y_validation is not None
        else None
    )
    y_test_binary = (y_test > args.positive_threshold).astype(int)
    soft_train = soft_label_all.loc[y_train.index] if soft_label_all is not None else None

    ignored_features = [col for col in args.ignored_features if col in X_train.columns]
    missing_ignored_features = sorted(set(args.ignored_features) - set(ignored_features))
    (
        X_train_model,
        model_cat_features,
        model_feature_weights,
        dropped_model_features,
        dropped_cat_features,
        dropped_feature_weights,
    ) = prepare_catboost_input(
        X=X_train,
        ignored_features=ignored_features,
        cat_features=cat_features,
        feature_weights=feature_weights,
    )
    X_validation_model = (
        X_validation[X_train_model.columns] if X_validation is not None else None
    )
    X_test_model = X_test[X_train_model.columns]

    logger.info(
        "Train rows: %d | Validation rows: %d | Test rows: %d",
        len(X_train),
        len(X_validation) if X_validation is not None else 0,
        len(X_test),
    )
    logger.info("CatBoost feature columns: %d", X_train_model.shape[1])
    if dropped_model_features:
        logger.info("Dropping non-model columns before CatBoost input: %s", dropped_model_features)
    if dropped_cat_features:
        logger.warning(
            "Categorical features dropped before CatBoost fit because they are ignored/missing: %s",
            dropped_cat_features,
        )
    if dropped_feature_weights:
        logger.info(
            "Feature weights ignored because features are not in CatBoost input: %s",
            dropped_feature_weights,
        )
    logger.info("Train date summary: %s", date_summary(X_train, args.date_column))
    if X_validation is not None:
        logger.info(
            "Validation date summary: %s",
            date_summary(X_validation, args.date_column),
        )
    logger.info("Test date summary: %s", date_summary(X_test, args.date_column))
    logger.info("Train class balance: %s", class_balance(y_train_binary))
    if y_validation_binary is not None:
        logger.info("Validation class balance: %s", class_balance(y_validation_binary))
    logger.info("Test class balance: %s", class_balance(y_test_binary))
    (
        X_train_fit,
        y_train_fit,
        train_sample_weight,
        soft_label_training_info,
    ) = expand_soft_labels_for_binary_training(
        X_train_model,
        y_train_binary,
        soft_train,
    )
    if soft_label_training_info.get("enabled"):
        logger.info(
            "Soft-label training expanded %d train rows to %d weighted rows (%d softened negatives).",
            soft_label_training_info.get("input_rows"),
            soft_label_training_info.get("expanded_rows"),
            soft_label_training_info.get("soft_negative_rows"),
        )

    existing_model_path = resolve_existing_model_path(args, config) if args.analysis_only else None
    model_params = (
        None
        if args.analysis_only
        else build_model_params(
            args=args,
            cat_features=model_cat_features,
            feature_weights=model_feature_weights,
            run_dir=run_dir,
        )
    )

    training_config: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "mode": "analysis_only" if args.analysis_only else "train",
        "argv": sys.argv[1:],
        "run_dir": run_dir,
        "paths": {
            "run_config": args.run_config,
            "config": args.config,
            "features_path": args.features_path,
            "output_root": args.output_root,
            "selected_feature_columns_path": selected_features_path,
            "requested_model_path": args.model_path,
            "loaded_model_path": existing_model_path,
        },
        "data": {
            "input_rows": int(len(df)),
            "input_columns": int(df.shape[1]),
            "target_column": args.target_column,
            "positive_threshold": args.positive_threshold,
            "soft_label_column": args.soft_label_column,
            "use_soft_labels_for_training": args.use_soft_labels_for_training,
            "save_eval_data": args.save_eval_data,
            "save_full_eval_data": not args.no_full_eval_data,
            "save_prediction_csv": args.save_prediction_csv,
            "save_precision_recall_csv": args.save_precision_recall_csv,
            "precision_recall_csv_max_points": args.precision_recall_csv_max_points,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "date_column": args.date_column,
            "validation_start_date": args.validation_start_date,
            "test_start_date": args.test_start_date,
            "shuffle": args.shuffle,
            "split_info": split_info,
            "countries": list(args.countries),
            "selected_features_count": len(selected_features),
            "selected_feature_filter_enabled": selected_feature_filter_enabled,
            "applied_selected_features_count": len(applied_selected_features),
            "missing_selected_features": missing_selected_features,
            "dropped_by_selected_feature_filter": dropped_by_selected_feature_filter,
            "feature_exclusions": {
                "drop_feature": list(args.drop_feature),
                "drop_feature_prefix": list(args.drop_feature_prefix),
                "drop_feature_group": list(args.drop_feature_group),
                "dropped": dropped_by_feature_exclusions,
                "dropped_exact_matches": dropped_exact_features,
                "dropped_prefix_matches": dropped_prefix_features,
                "dropped_group_matches": dropped_group_features,
            },
            "train": {
                "rows": int(len(X_train)),
                "date_summary": date_summary(X_train, args.date_column),
                "class_balance": class_balance(y_train_binary),
            },
            "validation": (
                {
                    "rows": int(len(X_validation)),
                    "date_summary": date_summary(X_validation, args.date_column),
                    "class_balance": class_balance(y_validation_binary),
                }
                if X_validation is not None and y_validation_binary is not None
                else None
            ),
            "test": {
                "rows": int(len(X_test)),
                "date_summary": date_summary(X_test, args.date_column),
                "class_balance": class_balance(y_test_binary),
            },
        },
        "preprocessing": {
            "configured_cat_features": configured_cat_features,
            "effective_cat_features": cat_features,
            "missing_cat_features": missing_cat_features,
            "requested_ignored_features": list(args.ignored_features),
            "effective_ignored_features": ignored_features,
            "missing_ignored_features": missing_ignored_features,
            "catboost_feature_count": int(X_train_model.shape[1]),
            "catboost_dropped_features": dropped_model_features,
            "catboost_cat_features": model_cat_features,
            "catboost_dropped_cat_features": dropped_cat_features,
            "catboost_feature_weights": model_feature_weights,
            "catboost_dropped_feature_weights": dropped_feature_weights,
            "soft_label_training": soft_label_training_info,
        },
        "plots": {
            "train_map_year": args.train_map_year,
            "test_map_year": args.test_map_year,
            "region_map_year": (
                args.region_map_year if args.region_map_year is not None else args.test_map_year
            ),
            "region_map_definitions": DEFAULT_PLOT_REGIONS,
            "map_plots_enabled": not args.no_map_plots,
            "region_map_plots_enabled": not args.no_map_plots and not args.no_region_map_plots,
        },
        "feature_importance_analysis": {
            "enabled": not args.no_feature_importance_analysis,
            "split": args.importance_split,
            "top_n": args.importance_top_n,
            "sample_size": args.importance_sample_size,
            "random_state": args.importance_random_state,
            "permutation_repeats": args.importance_permutation_repeats,
            "permutation_top_n": args.importance_permutation_top_n,
            "include_shap": not args.no_shap_importance,
            "include_permutation": not args.no_permutation_importance,
            "include_interactions": not args.no_interaction_importance,
        },
        "model_params": (
            model_params
            if model_params is not None
            else {"loaded_model_path": existing_model_path}
        ),
        "thresholding": {
            "prediction_threshold": args.prediction_threshold,
            "threshold_tuning_split": args.threshold_tuning_split,
            "threshold_min_precision": args.threshold_min_precision,
            "threshold_min_recall": args.threshold_min_recall,
            "threshold_exclude_years": list(args.threshold_exclude_years),
        },
        "calibration": {
            "enabled": args.calibration_enabled,
            "method": args.calibration_method,
            "split": args.calibration_split,
            "exclude_years": list(args.calibration_exclude_years),
            "apply_to_metrics": args.calibration_apply_to_metrics,
            "artifact_name": args.calibration_artifact_name,
        },
        "environment": collect_environment(),
    }
    write_yaml(run_dir / "training_config.yaml", training_config)
    write_json(run_dir / "training_config.json", training_config)

    if args.analysis_only:
        logger.info("Loading CatBoost model from %s", existing_model_path)
        model = CatBoostClassifier()
        model.load_model(existing_model_path)
        model_path = existing_model_path
    else:
        logger.info("Training CatBoost model.")
        if model_params is None:
            raise RuntimeError("Internal error: model params were not built for training mode.")
        model = CatBoostClassifier(**model_params)
        fit_kwargs: Dict[str, Any] = {}
        if X_validation_model is not None and y_validation_binary is not None:
            fit_kwargs["eval_set"] = (X_validation_model, y_validation_binary)
            fit_kwargs["use_best_model"] = True
            if args.early_stopping_rounds > 0:
                fit_kwargs["early_stopping_rounds"] = args.early_stopping_rounds
        elif args.early_stopping_rounds > 0:
            logger.warning(
                "Early stopping requested but validation is disabled; fitting all iterations."
            )
        if train_sample_weight is not None:
            fit_kwargs["sample_weight"] = train_sample_weight
        model.fit(X_train_fit, y_train_fit, **fit_kwargs)

        training_config["model_diagnostics"] = {
            "best_iteration": model.get_best_iteration(),
            "best_score": model.get_best_score(),
            "tree_count": getattr(model, "tree_count_", None),
            "validation_used": X_validation_model is not None,
            "early_stopping_rounds": (
                args.early_stopping_rounds
                if X_validation_model is not None and args.early_stopping_rounds > 0
                else None
            ),
        }

        model_path = model_dir / "catboost_model.cbm"
        model.save_model(model_path)
        logger.info("Saved model to %s", model_path)

    X_train_model, train_alignment = align_features_to_model(X_train_model, model, "train")
    validation_alignment = None
    if X_validation_model is not None:
        X_validation_model, validation_alignment = align_features_to_model(
            X_validation_model,
            model,
            "validation",
        )
    X_test_model, test_alignment = align_features_to_model(X_test_model, model, "test")
    prediction_cat_features = resolve_model_cat_features(model, model_cat_features, X_train_model)
    model_feature_names = list(getattr(model, "feature_names_", []) or X_train_model.columns)
    training_config["preprocessing"]["model_feature_names"] = model_feature_names
    training_config["preprocessing"]["prediction_cat_features"] = prediction_cat_features
    training_config["preprocessing"]["train_model_alignment"] = train_alignment
    training_config["preprocessing"]["validation_model_alignment"] = validation_alignment
    training_config["preprocessing"]["test_model_alignment"] = test_alignment
    training_config["paths"]["effective_model_path"] = model_path
    write_yaml(run_dir / "training_config.yaml", training_config)
    write_json(run_dir / "training_config.json", training_config)

    raw_train_prob = predict_probabilities(model, X_train_model, prediction_cat_features)
    raw_validation_prob = (
        predict_probabilities(model, X_validation_model, prediction_cat_features)
        if X_validation_model is not None
        else None
    )
    raw_test_prob = predict_probabilities(model, X_test_model, prediction_cat_features)
    train_prob, validation_prob, test_prob, calibration_info = calibrate_split_probabilities(
        args=args,
        model_dir=model_dir,
        X_train=X_train,
        y_train=y_train_binary,
        train_prob=raw_train_prob,
        X_validation=X_validation,
        y_validation=y_validation_binary,
        validation_prob=raw_validation_prob,
        X_test=X_test,
        y_test=y_test_binary,
        test_prob=raw_test_prob,
    )
    training_config["calibration"]["resolved"] = calibration_info

    threshold_inputs: Dict[str, Tuple[pd.Series, np.ndarray]] = {
        "train": (y_train_binary, train_prob),
        "test": (y_test_binary, test_prob),
    }
    threshold_filter_info = None
    if y_validation_binary is not None and validation_prob is not None:
        threshold_y, threshold_prob, threshold_filter_info = filter_probability_input_for_threshold(
            X_validation,
            y_validation_binary,
            validation_prob,
            args.date_column,
            args.threshold_exclude_years,
        )
        threshold_inputs["validation"] = (threshold_y, threshold_prob)
    threshold_info = resolve_prediction_threshold(args, threshold_inputs)
    if threshold_filter_info is not None:
        threshold_info["filter"] = threshold_filter_info
    prediction_threshold = float(threshold_info["threshold"])
    logger.info(
        "Using probability threshold %.4f for binary metrics/maps (%s).",
        prediction_threshold,
        threshold_info.get("source"),
    )

    train_pred = binary_from_probabilities(train_prob, prediction_threshold)
    validation_pred = (
        binary_from_probabilities(validation_prob, prediction_threshold)
        if validation_prob is not None
        else None
    )
    test_pred = binary_from_probabilities(test_prob, prediction_threshold)

    train_metrics = evaluate_split(
        "train",
        y_train_binary,
        train_pred,
        train_prob,
        prediction_threshold,
    )
    metrics_by_split = {"train": train_metrics}
    validation_metrics = None
    if (
        y_validation_binary is not None
        and validation_pred is not None
        and validation_prob is not None
    ):
        validation_metrics = evaluate_split(
            "validation",
            y_validation_binary,
            validation_pred,
            validation_prob,
            prediction_threshold,
        )
        metrics_by_split["validation"] = validation_metrics
    test_metrics = evaluate_split(
        "test",
        y_test_binary,
        test_pred,
        test_prob,
        prediction_threshold,
    )
    metrics_by_split["test"] = test_metrics
    training_config["thresholding"]["resolved"] = threshold_info
    save_metrics(metrics_by_split, eval_dir)
    yearly_summaries: Dict[str, List[Dict[str, Any]]] = {}
    if y_validation_binary is not None and validation_prob is not None:
        yearly_summaries["validation"] = build_yearly_summary(
            split_name="validation",
            X=X_validation,
            y_true=y_validation_binary,
            y_prob=validation_prob,
            threshold=prediction_threshold,
            date_column=args.date_column,
        )
    yearly_summaries["test"] = build_yearly_summary(
        split_name="test",
        X=X_test,
        y_true=y_test_binary,
        y_prob=test_prob,
        threshold=prediction_threshold,
        date_column=args.date_column,
    )
    yearly_metrics_path = save_yearly_summary(yearly_summaries, eval_dir)
    if yearly_metrics_path is not None:
        logger.info("Saved yearly validation/test metrics to %s", yearly_metrics_path)
        log_yearly_summary(yearly_summaries, title="Yearly validation/test metrics")

    full_eval_data = not args.no_full_eval_data
    train_eval_path = save_predictions(
        split_name="train",
        X=X_train,
        y_count=y_train,
        y_binary=y_train_binary,
        y_pred=train_pred,
        y_prob=train_prob,
        eval_dir=eval_dir,
        full=full_eval_data,
        save_evaluation_data=args.save_eval_data,
        save_compact_csv=args.save_prediction_csv,
    )
    validation_eval_path = None
    if (
        X_validation is not None
        and y_validation is not None
        and y_validation_binary is not None
        and validation_pred is not None
        and validation_prob is not None
    ):
        validation_eval_path = save_predictions(
            split_name="validation",
            X=X_validation,
            y_count=y_validation,
            y_binary=y_validation_binary,
            y_pred=validation_pred,
            y_prob=validation_prob,
            eval_dir=eval_dir,
            full=full_eval_data,
            save_evaluation_data=args.save_eval_data,
            save_compact_csv=args.save_prediction_csv,
        )
    test_eval_path = save_predictions(
        split_name="test",
        X=X_test,
        y_count=y_test,
        y_binary=y_test_binary,
        y_pred=test_pred,
        y_prob=test_prob,
        eval_dir=eval_dir,
        full=full_eval_data,
        save_evaluation_data=args.save_eval_data,
        save_compact_csv=args.save_prediction_csv,
    )

    precision_recall_curve_paths: Dict[str, Optional[Path]] = {}
    precision_recall_curve_paths["train"] = save_precision_recall(
        "train",
        y_train_binary,
        train_prob,
        eval_dir,
        plots_dir,
        args.save_precision_recall_csv,
        args.precision_recall_csv_max_points,
    )
    if y_validation_binary is not None and validation_prob is not None:
        precision_recall_curve_paths["validation"] = save_precision_recall(
            "validation",
            y_validation_binary,
            validation_prob,
            eval_dir,
            plots_dir,
            args.save_precision_recall_csv,
            args.precision_recall_csv_max_points,
        )
    precision_recall_curve_paths["test"] = save_precision_recall(
        "test",
        y_test_binary,
        test_prob,
        eval_dir,
        plots_dir,
        args.save_precision_recall_csv,
        args.precision_recall_csv_max_points,
    )
    feature_importance_artifacts: Dict[str, Any]
    if args.no_feature_importance_analysis:
        feature_importance_artifacts = {"enabled": False}
        feature_importance_path = None
    else:
        feature_importance_artifacts = run_feature_importance_analysis(
            model=model,
            split_data={
                "train": (X_train_model, y_train_binary),
                "test": (X_test_model, y_test_binary),
            },
            cat_features=prediction_cat_features,
            eval_dir=eval_dir,
            plots_dir=plots_dir,
            requested_split=args.importance_split,
            top_n=args.importance_top_n,
            sample_size=args.importance_sample_size,
            random_state=args.importance_random_state,
            permutation_repeats=args.importance_permutation_repeats,
            permutation_top_n=args.importance_permutation_top_n,
            include_shap=not args.no_shap_importance,
            include_permutation=not args.no_permutation_importance,
            include_interactions=not args.no_interaction_importance,
        )
        feature_importance_artifacts["enabled"] = True
        feature_importance_path = primary_feature_importance_path(
            feature_importance_artifacts,
            preferred_split="test",
        )

    map_paths: Dict[str, Optional[Path]] = {"train": None, "test": None}
    plot_boosting_region_paths: Dict[str, Optional[Path]] = {}
    plot_boosting_overview_path: Optional[Path] = None
    plot_boosting_summary_path: Optional[Path] = None
    if not args.no_map_plots:
        world_background = load_world_background(args.country_shapes_path)
        map_paths["train"] = save_actual_vs_predicted_map(
            split_name="train",
            X=X_train,
            y_true_binary=y_train_binary,
            y_pred_binary=train_pred,
            year=args.train_map_year,
            world=world_background,
            plots_dir=plots_dir,
        )
        map_paths["test"] = save_actual_vs_predicted_map(
            split_name="test",
            X=X_test,
            y_true_binary=y_test_binary,
            y_pred_binary=test_pred,
            year=args.test_map_year,
            world=world_background,
            plots_dir=plots_dir,
        )
        if not args.no_region_map_plots:
            region_map_year = (
                args.region_map_year if args.region_map_year is not None else args.test_map_year
            )
            (
                plot_boosting_region_paths,
                plot_boosting_overview_path,
                plot_boosting_summary_path,
            ) = save_plot_boosting_region_maps(
                split_name="test",
                X=X_test,
                y_true_binary=y_test_binary,
                y_pred_binary=test_pred,
                year=region_map_year,
                world=world_background,
                plots_dir=plots_dir,
                eval_dir=eval_dir,
            )

    training_config["artifacts"] = {
        "model": model_path,
        "metrics_json": eval_dir / "metrics.json",
        "metrics_csv": eval_dir / "metrics.csv",
        "yearly_metrics_json": (
            eval_dir / "yearly_metrics.json" if yearly_metrics_path is not None else None
        ),
        "yearly_metrics_csv": yearly_metrics_path,
        "train_evaluation_data": train_eval_path,
        "validation_evaluation_data": validation_eval_path,
        "test_evaluation_data": test_eval_path,
        "precision_recall_curve_csv": precision_recall_curve_paths,
        "feature_importance": feature_importance_path,
        "feature_importance_analysis": feature_importance_artifacts,
        "plots_dir": plots_dir,
        "map_plots": map_paths,
        "plot_boosting_region_maps": plot_boosting_region_paths,
        "plot_boosting_overview_map": plot_boosting_overview_path,
        "plot_boosting_region_summary": plot_boosting_summary_path,
    }
    training_config["metrics"] = metrics_by_split
    training_config["yearly_metrics"] = yearly_summaries
    training_config["duration_seconds"] = perf_counter() - start_time
    write_yaml(run_dir / "training_config.yaml", training_config)
    write_json(run_dir / "training_config.json", training_config)

    logger.info("Completed run in %.2fs", training_config["duration_seconds"])
    logger.info("Run artifacts are in %s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
