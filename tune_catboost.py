from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import train_catboost as tc


logger = logging.getLogger(__name__)


DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "iterations": [600, 900, 1200, 1800, 2400, 3200, 4200, 5500],
    "learning_rate": [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12],
    "depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "l2_leaf_reg": [0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
    "min_data_in_leaf": [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000],
    "random_strength": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "bootstrap_type": ["Bayesian", "Bernoulli", "No"],
    "bagging_temperature": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    "subsample": [0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 0.9, 1.0],
    "border_count": [32, 64, 128, 254],
    "leaf_estimation_iterations": [1, 3, 5, 10],
    "one_hot_max_size": [2, 5, 10, 20, 50],
    "max_ctr_complexity": [1, 2, 3, 4],
    "class_weight_positive": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0],
}


@dataclass
class TrialResult:
    trial: int
    status: str
    duration_sec: float
    objective: Optional[float]
    params: Dict[str, Any]
    feature_plan: Dict[str, Any]
    threshold: Optional[float]
    best_iteration: Optional[int]
    metrics: Dict[str, Dict[str, Optional[float]]]
    gaps: Dict[str, Optional[float]]
    yearly_metrics: Dict[str, List[Dict[str, Any]]]
    error: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "trial": self.trial,
            "status": self.status,
            "duration_sec": round(self.duration_sec, 2),
            "objective": self.objective,
            "threshold": self.threshold,
            "best_iteration": self.best_iteration,
            "feature_count": self.feature_plan.get("feature_count"),
            "feature_mode": self.feature_plan.get("mode"),
            "top_k": self.feature_plan.get("top_k"),
            "column_sample_fraction": self.feature_plan.get("column_sample_fraction"),
            "group_sample_fraction": self.feature_plan.get("group_sample_fraction"),
            "dropped_groups": ",".join(self.feature_plan.get("dropped_groups", [])),
            "sampled_groups": ",".join(self.feature_plan.get("sampled_groups", [])),
            "error": self.error or "",
        }
        for split, split_metrics in self.metrics.items():
            for name, value in split_metrics.items():
                row[f"{split}_{name}"] = value
        for name, value in self.gaps.items():
            row[f"gap_{name}"] = value
        for split_name, split_rows in self.yearly_metrics.items():
            for year_row in split_rows:
                year = year_row.get("year")
                if year is None:
                    continue
                prefix = f"{split_name}_{year}"
                for metric_name in ("average_precision", "f1", "precision", "recall", "prevalence"):
                    if metric_name in year_row:
                        row[f"{prefix}_{metric_name}"] = year_row[metric_name]
        for name, value in self.params.items():
            if name in {"cat_features", "feature_weights", "train_dir"}:
                continue
            row[f"param_{name}"] = value
        return row


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search CatBoost hyperparameters and feature subsets from a YAML config."
        )
    )
    parser.add_argument("config", type=Path, help="Path to the CatBoost tuning YAML config.")
    return parser.parse_args(argv)


def build_args_from_config(config_path: Path) -> argparse.Namespace:
    run_config = tc.load_config(config_path)
    cfg = tc._section(run_config, "catboost_tune") if "catboost_tune" in run_config else run_config
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {config_path} did not contain a YAML mapping.")

    run = tc._section(cfg, "run")
    data = tc._section(cfg, "data")
    target = tc._section(cfg, "target")
    split = tc._section(cfg, "split")
    features = tc._section(cfg, "features")
    search = tc._section(cfg, "search")
    model = tc._section(cfg, "model")
    hardware = tc._section(cfg, "hardware")
    sampling = tc._section(cfg, "sampling")
    feature_search = tc._section(cfg, "feature_search")
    objective = tc._section(cfg, "objective")
    thresholding = tc._section(cfg, "thresholding")
    persistence = tc._section(cfg, "persistence")

    default_feature_config_path = (
        config_path
        if "cat_features" in cfg or "selected_feature_columns_path" in cfg
        else Path("configs/features_config_30d.yaml")
    )
    feature_config_path = tc._path_or_none(
        tc._first_defined(
            cfg.get("feature_config_path"),
            cfg.get("feature_config"),
            data.get("feature_config_path"),
            default=default_feature_config_path,
        )
    )
    if feature_config_path is None:
        raise ValueError("CatBoost tuning config must define feature_config_path.")

    return argparse.Namespace(
        run_config=config_path,
        config=feature_config_path,
        features_path=Path(data.get("features_path", tc.DEFAULT_FEATURES_PATH)).expanduser(),
        rebuild_features=bool(data.get("rebuild_features", False)),
        per_country_features_dir=Path(
            data.get("per_country_features_dir", "data/saved_features")
        ).expanduser(),
        per_country_pattern=data.get(
            "per_country_pattern",
            "train_test_features_30d_{country}.parquet",
        ),
        countries=tc._as_list(data.get("countries"), tc.DEFAULT_COUNTRIES),
        output_root=Path(run.get("output_root", "outputs")).expanduser(),
        run_prefix=run.get("run_prefix", "catboost_tune"),
        target_column=target.get("column", data.get("target_column", "count")),
        positive_threshold=float(target.get("positive_threshold", 0.0)),
        date_column=split.get("date_column", "datetime"),
        validation_start_date=tc._none_if_blank(
            split.get("validation_start_date", tc.DEFAULT_VALIDATION_START_DATE)
        ),
        test_start_date=tc._none_if_blank(split.get("test_start_date", tc.DEFAULT_TEST_START_DATE)),
        test_size=float(split.get("test_size", 0.15)),
        validation_size=float(split.get("validation_size", 0.15)),
        shuffle=bool(split.get("shuffle", False)),
        random_state=int(split.get("random_state", 42)),
        limit_rows=data.get("limit_rows"),
        selected_features_path=tc._path_or_none(features.get("selected_features_path")),
        no_selected_feature_filter=not bool(features.get("selected_feature_filter", True)),
        ignored_features=tc._as_list(features.get("ignored"), tc.DEFAULT_IGNORED_FEATURES),
        drop_feature=tc._as_list(features.get("drop_features")),
        drop_feature_prefix=tc._as_list(features.get("drop_feature_prefixes")),
        drop_feature_group=tc._as_list(features.get("drop_feature_groups")),
        feature_weight=features.get("feature_weights", {}),
        search_space=tc._first_defined(search.get("space"), search.get("space_path")),
        strategy=search.get("strategy", "random"),
        max_trials=int(search.get("max_trials", 40)),
        seed=int(search.get("seed", 17)),
        fail_fast=bool(search.get("fail_fast", False)),
        task_type=hardware.get("task_type", "GPU"),
        devices=hardware.get("devices"),
        thread_count=hardware.get("thread_count"),
        gpu_ram_part=hardware.get("gpu_ram_part"),
        loss_function=model.get("loss_function", "Logloss"),
        eval_metric=model.get("eval_metric", "F1"),
        early_stopping_rounds=int(model.get("early_stopping_rounds", 150)),
        verbose=int(model.get("verbose", 0)),
        train_sample_frac=float(sampling.get("train_sample_frac", 1.0)),
        max_train_rows=sampling.get("max_train_rows"),
        feature_search=feature_search.get("mode", "both"),
        importance_csv=tc._path_or_none(feature_search.get("importance_csv")),
        top_k_choices=feature_search.get("top_k_choices", "40,60,80,120,160,220,all"),
        top_k_probability=float(feature_search.get("top_k_probability", 0.65)),
        max_drop_groups=int(feature_search.get("max_drop_groups", 3)),
        drop_group_probability=float(feature_search.get("drop_group_probability", 0.45)),
        group_sample_probability=float(feature_search.get("group_sample_probability", 0.20)),
        group_sample_fraction_choices=feature_search.get(
            "group_sample_fraction_choices",
            "0.45,0.60,0.75,0.90,1.0",
        ),
        column_sample_probability=float(feature_search.get("column_sample_probability", 0.35)),
        column_sample_fraction_choices=feature_search.get(
            "column_sample_fraction_choices",
            "0.35,0.50,0.65,0.80,0.90,1.0",
        ),
        min_features=int(feature_search.get("min_features", 25)),
        protected_feature=tc._as_list(feature_search.get("protected_features")),
        objective_split=objective.get("split", "test"),
        objective_metric=objective.get("metric", "average_precision"),
        train_test_gap_weight=float(objective.get("train_test_gap_weight", 0.0)),
        validation_test_gap_weight=float(objective.get("validation_test_gap_weight", 0.0)),
        train_validation_gap_weight=float(objective.get("train_validation_gap_weight", 0.0)),
        prediction_threshold=thresholding.get("prediction_threshold"),
        threshold_min_precision=thresholding.get("min_precision"),
        threshold_min_recall=thresholding.get("min_recall"),
        save_every_model=bool(persistence.get("save_every_model", False)),
    )


def setup_logging(run_dir: Path) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(tc.sanitize_for_json(data), handle, indent=2)


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(tc.sanitize_for_json(data), handle, sort_keys=False)


def load_search_space(
    path_or_mapping: Optional[Path | str | Dict[str, Any]],
) -> Dict[str, List[Any]]:
    if path_or_mapping is None:
        return {key: list(values) for key, values in DEFAULT_SEARCH_SPACE.items()}
    if isinstance(path_or_mapping, str) and path_or_mapping.strip().lower() in {
        "default",
        "wide",
        "default_wide",
    }:
        return {key: list(values) for key, values in DEFAULT_SEARCH_SPACE.items()}
    if isinstance(path_or_mapping, dict):
        data = path_or_mapping
    else:
        path = Path(path_or_mapping).expanduser()
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Search space must contain a YAML mapping.")
    parsed: Dict[str, List[Any]] = {}
    for key, values in data.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"Search space key {key!r} must contain a non-empty list.")
        parsed[str(key)] = list(values)
    return parsed


def iter_assignments(
    search_space: Dict[str, List[Any]],
    strategy: str,
    max_trials: int,
    rng: random.Random,
) -> Iterable[Dict[str, Any]]:
    if strategy == "grid":
        names = list(search_space)
        choices = [search_space[name] for name in names]
        for index, combination in enumerate(itertools.product(*choices), start=1):
            if index > max_trials:
                break
            yield {name: value for name, value in zip(names, combination)}
        return

    for _ in range(max_trials):
        yield {name: rng.choice(values) for name, values in search_space.items()}


def split_choice_values(raw: Any) -> List[Any]:
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def parse_top_k_choices(raw: Any) -> List[Optional[int]]:
    choices: List[Optional[int]] = []
    for part in split_choice_values(raw):
        value = str(part).strip().lower()
        if not value:
            continue
        if value in {"all", "none"}:
            choices.append(None)
        else:
            top_k = int(value)
            if top_k <= 0:
                raise ValueError("--top-k-choices values must be positive integers or 'all'.")
            choices.append(top_k)
    return choices or [None]


def parse_fraction_choices(raw: Any, name: str) -> List[float]:
    choices: List[float] = []
    for part in split_choice_values(raw):
        value = float(part)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} values must be in (0, 1], got {value}.")
        choices.append(value)
    return choices or [1.0]


def choose_importance_sort_column(df: pd.DataFrame) -> Tuple[str, bool]:
    rank_candidates = [
        "consensus_rank",
        "rank",
        "permutation_rank",
        "mean_rank",
        "prediction_values_change_rank",
    ]
    for col in rank_candidates:
        if col in df.columns:
            return col, True

    importance_candidates = [
        "mean_normalized_importance",
        "importance",
        "permutation_importance",
        "average_precision_drop_mean",
        "prediction_values_change_importance",
        "shap_mean_abs_importance",
    ]
    for col in importance_candidates:
        if col in df.columns:
            return col, False
    raise ValueError("Importance CSV must include a feature column and a rank/importance column.")


def load_importance_order(path: Optional[Path], all_features: Sequence[str]) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Importance CSV does not exist: {path}")

    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"Importance CSV {path} does not contain a 'feature' column.")
    sort_col, ascending = choose_importance_sort_column(df)
    ordered = (
        df.dropna(subset=["feature"])
        .sort_values(sort_col, ascending=ascending, kind="mergesort")["feature"]
        .astype(str)
        .tolist()
    )
    feature_set = set(all_features)
    ranked = tc.ordered_unique(feature for feature in ordered if feature in feature_set)
    ranked.extend(feature for feature in all_features if feature not in set(ranked))
    return ranked


def model_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Optional[float]]:
    y_array = np.asarray(y_true).astype(int)
    prob_array = np.asarray(y_prob, dtype=float)
    pred_array = (prob_array >= threshold).astype(int)
    has_both_classes = len(np.unique(y_array)) == 2
    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_array, pred_array)),
        "precision": float(precision_score(y_array, pred_array, zero_division=0)),
        "recall": float(recall_score(y_array, pred_array, zero_division=0)),
        "f1": float(f1_score(y_array, pred_array, zero_division=0)),
        "average_precision": None,
        "roc_auc": None,
        "log_loss": None,
        "positive_rate": float(np.mean(y_array)),
        "predicted_positive_rate": float(np.mean(pred_array)),
    }
    if has_both_classes:
        metrics["average_precision"] = float(average_precision_score(y_array, prob_array))
        metrics["roc_auc"] = float(roc_auc_score(y_array, prob_array))
        metrics["log_loss"] = float(log_loss(y_array, prob_array, labels=[0, 1]))
    return metrics


def build_yearly_summary_from_years(
    split_name: str,
    years: np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> List[Dict[str, Any]]:
    year_array = np.asarray(years)
    y_array = np.asarray(y_true).astype(int)
    prob_array = np.asarray(y_prob, dtype=float)
    rows: List[Dict[str, Any]] = []
    for year in sorted(int(value) for value in pd.Series(year_array).dropna().unique()):
        mask = year_array == year
        if not mask.any():
            continue
        rows.append(
            tc.summarize_probability_metrics(
                split_name=split_name,
                y_true=y_array[mask],
                y_prob=prob_array[mask],
                threshold=threshold,
                year=year,
            )
        )
    return rows


def write_yearly_metrics(path: Path, yearly_metrics: Dict[str, List[Dict[str, Any]]]) -> None:
    rows = [
        row
        for split_rows in yearly_metrics.values()
        for row in split_rows
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def finite_metric(metrics: Dict[str, Dict[str, Optional[float]]], split: str, metric: str) -> Optional[float]:
    value = metrics.get(split, {}).get(metric)
    if value is None:
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def signed_metric_value(value: Optional[float], metric: str) -> Optional[float]:
    if value is None:
        return None
    return -value if metric == "log_loss" else value


def compute_objective(
    metrics: Dict[str, Dict[str, Optional[float]]],
    objective_split: str,
    objective_metric: str,
    train_test_gap_weight: float,
    validation_test_gap_weight: float,
    train_validation_gap_weight: float,
) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    train = signed_metric_value(finite_metric(metrics, "train", objective_metric), objective_metric)
    validation = signed_metric_value(
        finite_metric(metrics, "validation", objective_metric),
        objective_metric,
    )
    test = signed_metric_value(finite_metric(metrics, "test", objective_metric), objective_metric)
    primary = validation if objective_split == "validation" else test
    if primary is None:
        return None, {}

    gaps = {
        "train_minus_test": None if train is None or test is None else max(train - test, 0.0),
        "validation_test_abs": (
            None if validation is None or test is None else abs(validation - test)
        ),
        "train_minus_validation": (
            None if train is None or validation is None else max(train - validation, 0.0)
        ),
    }
    objective = primary
    if gaps["train_minus_test"] is not None:
        objective -= train_test_gap_weight * gaps["train_minus_test"]
    if gaps["validation_test_abs"] is not None:
        objective -= validation_test_gap_weight * gaps["validation_test_abs"]
    if gaps["train_minus_validation"] is not None:
        objective -= train_validation_gap_weight * gaps["train_minus_validation"]
    return float(objective), gaps


def build_split_args(args: argparse.Namespace) -> argparse.Namespace:
    def optional_date(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if str(value).strip().lower() in {"", "none", "null"}:
            return None
        return value

    return argparse.Namespace(
        test_size=args.test_size,
        validation_size=args.validation_size,
        date_column=args.date_column,
        validation_start_date=optional_date(args.validation_start_date),
        test_start_date=optional_date(args.test_start_date),
        shuffle=args.shuffle,
        random_state=args.random_state,
    )


def sample_training_rows(
    X: pd.DataFrame,
    y: pd.Series,
    sample_frac: float,
    max_rows: Optional[int],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    if not 0.0 < sample_frac <= 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1].")

    requested = len(X)
    if sample_frac < 1.0:
        requested = min(requested, max(1, int(round(len(X) * sample_frac))))
    if max_rows is not None:
        requested = min(requested, max_rows)
    if requested >= len(X):
        return X, y, {"sampled": False, "rows": int(len(X)), "input_rows": int(len(X))}

    positions = np.arange(len(X))
    try:
        sampled_positions, _ = tc.train_test_split(
            positions,
            train_size=requested,
            stratify=np.asarray(y),
            random_state=random_state,
            shuffle=True,
        )
    except Exception:
        rng = np.random.default_rng(random_state)
        sampled_positions = rng.choice(positions, size=requested, replace=False)
    sampled_positions = np.sort(sampled_positions)
    X_sample = X.iloc[sampled_positions]
    y_sample = pd.Series(np.asarray(y)[sampled_positions], index=X_sample.index)
    return X_sample, y_sample, {
        "sampled": True,
        "rows": int(len(X_sample)),
        "input_rows": int(len(X)),
        "sample_frac": sample_frac,
        "max_rows": max_rows,
    }


def build_feature_plan(
    all_features: Sequence[str],
    importance_order: Sequence[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> Dict[str, Any]:
    all_features = list(all_features)
    selected = list(all_features)
    mode_parts: List[str] = []
    top_k: Optional[int] = None
    dropped_groups: List[str] = []
    sampled_groups: List[str] = []
    column_sample_fraction: Optional[float] = None
    group_sample_fraction: Optional[float] = None
    protected = [feature for feature in args.protected_feature if feature in all_features]
    feature_search_mode = str(args.feature_search or "both").lower()
    wide_feature_mode = feature_search_mode in {"all", "wide", "any"}

    can_use_importance = (
        (feature_search_mode in {"importance", "both"} or wide_feature_mode)
        and bool(importance_order)
    )
    if can_use_importance and rng.random() < args.top_k_probability:
        top_k = rng.choice(parse_top_k_choices(args.top_k_choices))
        if top_k is not None and top_k < len(all_features):
            selected = list(importance_order[:top_k])
            mode_parts.append("top_k")

    can_sample_groups = feature_search_mode in {"groups", "both"} or wide_feature_mode
    if can_sample_groups and rng.random() < args.group_sample_probability:
        groups = sorted({tc.infer_feature_group(feature) for feature in selected})
        if len(groups) > 1:
            sampled_group_fraction = rng.choice(
                parse_fraction_choices(
                    args.group_sample_fraction_choices,
                    "group_sample_fraction_choices",
                )
            )
            protected_groups = {tc.infer_feature_group(feature) for feature in protected}
            keep_count = min(
                len(groups),
                max(
                    1,
                    len(protected_groups),
                    int(round(len(groups) * sampled_group_fraction)),
                ),
            )
            optional_groups = [group for group in groups if group not in protected_groups]
            sampled_group_set = set(protected_groups)
            optional_keep_count = min(
                len(optional_groups),
                max(0, keep_count - len(sampled_group_set)),
            )
            if optional_keep_count:
                sampled_group_set.update(rng.sample(optional_groups, optional_keep_count))
            if len(sampled_group_set) < len(groups):
                selected = [
                    feature
                    for feature in selected
                    if tc.infer_feature_group(feature) in sampled_group_set
                ]
                sampled_groups = sorted(sampled_group_set)
                group_sample_fraction = sampled_group_fraction
                mode_parts.append("sample_groups")

    can_drop_groups = feature_search_mode in {"groups", "both"} or wide_feature_mode
    if can_drop_groups and rng.random() < args.drop_group_probability:
        groups = sorted({tc.infer_feature_group(feature) for feature in selected})
        if groups and args.max_drop_groups > 0:
            drop_count = rng.randint(0, min(args.max_drop_groups, len(groups)))
            dropped_groups = sorted(rng.sample(groups, drop_count)) if drop_count else []
            if dropped_groups:
                selected = [
                    feature
                    for feature in selected
                    if tc.infer_feature_group(feature) not in set(dropped_groups)
                ]
                mode_parts.append("drop_groups")

    can_sample_columns = feature_search_mode in {"columns", "column", "both"} or wide_feature_mode
    if can_sample_columns and rng.random() < args.column_sample_probability:
        if len(selected) > 1:
            sampled_column_fraction = rng.choice(
                parse_fraction_choices(
                    args.column_sample_fraction_choices,
                    "column_sample_fraction_choices",
                )
            )
            protected_set = set(protected)
            protected_in_selected = [feature for feature in selected if feature in protected_set]
            keep_count = min(
                len(selected),
                max(
                    1,
                    len(protected_in_selected),
                    int(round(len(selected) * sampled_column_fraction)),
                    min(args.min_features, len(selected)),
                ),
            )
            candidate_features = [feature for feature in selected if feature not in protected_set]
            candidate_set = set(candidate_features)
            ranked_candidates = [
                feature for feature in importance_order if feature in candidate_set
            ]
            slots_after_protected = max(0, keep_count - len(protected_in_selected))
            elite_count = min(
                len(ranked_candidates),
                slots_after_protected,
                max(0, int(round(keep_count * 0.25))),
            )
            elite_features = ranked_candidates[:elite_count]
            elite_set = set(elite_features)
            random_pool = [feature for feature in candidate_features if feature not in elite_set]
            random_keep_count = min(
                len(random_pool),
                max(0, keep_count - len(protected_in_selected) - len(elite_features)),
            )
            sampled_features = (
                rng.sample(random_pool, random_keep_count)
                if random_keep_count < len(random_pool)
                else list(random_pool)
            )
            selected_set = protected_set | elite_set | set(sampled_features)
            sampled_selected = [feature for feature in selected if feature in selected_set]
            if 0 < len(sampled_selected) < len(selected):
                selected = sampled_selected
                column_sample_fraction = sampled_column_fraction
                mode_parts.append("sample_columns")

    selected = tc.ordered_unique(list(selected) + protected)
    if not selected:
        selected = list(all_features)
        mode_parts = ["all_fallback"]
        top_k = None
        dropped_groups = []
        sampled_groups = []
        column_sample_fraction = None
        group_sample_fraction = None

    selected_set = set(selected)
    dropped_features = [feature for feature in all_features if feature not in selected_set]
    return {
        "mode": "+".join(mode_parts) if mode_parts else "all",
        "top_k": top_k,
        "column_sample_fraction": column_sample_fraction,
        "group_sample_fraction": group_sample_fraction,
        "feature_count": len(selected),
        "features": selected,
        "dropped_feature_count": len(dropped_features),
        "dropped_features": dropped_features,
        "dropped_groups": dropped_groups,
        "sampled_groups": sampled_groups,
    }


def build_model_params(
    assignments: Dict[str, Any],
    cat_features: Sequence[str],
    feature_weights: Dict[str, float],
    args: argparse.Namespace,
    trial_dir: Path,
    random_seed: int,
) -> Dict[str, Any]:
    verbose: Any = args.verbose if args.verbose > 0 else False
    defaults = {key: values[0] for key, values in DEFAULT_SEARCH_SPACE.items()}
    class_weight_negative = float(assignments.pop("class_weight_negative", 1.0))
    class_weight_positive = float(
        assignments.pop("class_weight_positive", defaults["class_weight_positive"])
    )
    params: Dict[str, Any] = {
        "iterations": int(assignments.pop("iterations", defaults["iterations"])),
        "learning_rate": float(assignments.pop("learning_rate", defaults["learning_rate"])),
        "depth": int(assignments.pop("depth", defaults["depth"])),
        "l2_leaf_reg": float(assignments.pop("l2_leaf_reg", defaults["l2_leaf_reg"])),
        "min_data_in_leaf": int(
            assignments.pop("min_data_in_leaf", defaults["min_data_in_leaf"])
        ),
        "random_strength": float(
            assignments.pop("random_strength", defaults["random_strength"])
        ),
        "loss_function": args.loss_function,
        "eval_metric": args.eval_metric,
        "class_weights": [class_weight_negative, class_weight_positive],
        "random_seed": random_seed,
        "verbose": verbose,
        "cat_features": list(cat_features),
        "feature_weights": feature_weights,
        "train_dir": str(trial_dir / "catboost_info"),
        "allow_writing_files": True,
    }
    bagging_temperature = assignments.pop(
        "bagging_temperature",
        defaults["bagging_temperature"],
    )
    bootstrap_type = assignments.pop("bootstrap_type", "Bayesian")
    if (
        isinstance(bootstrap_type, str)
        and bootstrap_type.strip().lower() in {"", "none", "null"}
    ):
        bootstrap_type = None
    if bootstrap_type:
        params["bootstrap_type"] = bootstrap_type
    if bootstrap_type in {None, "Bayesian"} and bagging_temperature is not None:
        params["bagging_temperature"] = float(bagging_temperature)
    subsample = assignments.pop("subsample", None)
    if bootstrap_type in {"Bernoulli", "MVS", "Poisson"} and subsample is not None:
        params["subsample"] = float(subsample)
    rsm = assignments.pop("rsm", None)
    if rsm is not None and args.task_type != "GPU":
        params["rsm"] = float(rsm)

    for name, value in list(assignments.items()):
        if value is not None:
            params[name] = value
    assignments.clear()

    if args.task_type:
        params["task_type"] = args.task_type
    if args.devices:
        params["devices"] = args.devices
    if args.thread_count is not None:
        params["thread_count"] = args.thread_count
    if args.gpu_ram_part is not None:
        params["gpu_ram_part"] = args.gpu_ram_part
    return params


def fit_and_score_trial(
    trial_index: int,
    assignments: Dict[str, Any],
    feature_plan: Dict[str, Any],
    data: Dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[TrialResult, Optional[CatBoostClassifier]]:
    start_time = perf_counter()
    trial_dir = run_dir / "trials" / f"trial_{trial_index:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    model: Optional[CatBoostClassifier] = None
    try:
        selected_features = feature_plan["features"]
        feature_plan = dict(feature_plan)
        feature_plan["train_catboost_selected_features"] = tc.ordered_unique(
            list(selected_features) + list(data["dropped_model_features"])
        )
        cat_features = [feature for feature in data["model_cat_features"] if feature in selected_features]
        feature_weights = {
            feature: weight
            for feature, weight in data["model_feature_weights"].items()
            if feature in selected_features
        }
        X_train = data["X_train_model"][selected_features]
        y_train = data["y_train_binary"]
        X_fit, y_fit, sample_info = sample_training_rows(
            X_train,
            y_train,
            sample_frac=args.train_sample_frac,
            max_rows=args.max_train_rows,
            random_state=args.seed + trial_index,
        )
        X_validation = data["X_validation_model"][selected_features]
        y_validation = data["y_validation_binary"]
        X_test = data["X_test_model"][selected_features]
        y_test = data["y_test_binary"]

        params = build_model_params(
            assignments=dict(assignments),
            cat_features=cat_features,
            feature_weights=feature_weights,
            args=args,
            trial_dir=trial_dir,
            random_seed=args.seed + trial_index,
        )
        write_json(
            trial_dir / "trial_config.json",
            {
                "trial": trial_index,
                "assignments": assignments,
                "params": params,
                "feature_plan": {
                    key: value for key, value in feature_plan.items() if key != "features"
                },
                "train_sample": sample_info,
            },
        )

        model = CatBoostClassifier(**params)
        fit_kwargs: Dict[str, Any] = {}
        if X_validation is not None and y_validation is not None:
            fit_kwargs["eval_set"] = (X_validation, y_validation)
            fit_kwargs["use_best_model"] = True
            if args.early_stopping_rounds > 0:
                fit_kwargs["early_stopping_rounds"] = args.early_stopping_rounds
        model.fit(X_fit, y_fit, **fit_kwargs)

        train_prob = tc.predict_probabilities(model, X_train, cat_features)
        validation_prob = tc.predict_probabilities(model, X_validation, cat_features)
        test_prob = tc.predict_probabilities(model, X_test, cat_features)

        if args.prediction_threshold is not None:
            threshold_info = {"threshold": float(args.prediction_threshold), "source": "fixed"}
        else:
            threshold_info = tc.choose_threshold_by_f1(
                y_validation,
                validation_prob,
                min_precision=args.threshold_min_precision,
                min_recall=args.threshold_min_recall,
            )
        threshold = float(threshold_info["threshold"])

        metrics = {
            "train": model_metrics(y_train, train_prob, threshold),
            "validation": model_metrics(y_validation, validation_prob, threshold),
            "test": model_metrics(y_test, test_prob, threshold),
        }
        yearly_metrics = {
            "validation": build_yearly_summary_from_years(
                split_name="validation",
                years=data["validation_years"],
                y_true=y_validation,
                y_prob=validation_prob,
                threshold=threshold,
            ),
            "test": build_yearly_summary_from_years(
                split_name="test",
                years=data["test_years"],
                y_true=y_test,
                y_prob=test_prob,
                threshold=threshold,
            ),
        }
        objective, gaps = compute_objective(
            metrics=metrics,
            objective_split=args.objective_split,
            objective_metric=args.objective_metric,
            train_test_gap_weight=args.train_test_gap_weight,
            validation_test_gap_weight=args.validation_test_gap_weight,
            train_validation_gap_weight=args.train_validation_gap_weight,
        )

        result = TrialResult(
            trial=trial_index,
            status="ok",
            duration_sec=perf_counter() - start_time,
            objective=objective,
            params=params,
            feature_plan=feature_plan,
            threshold=threshold,
            best_iteration=model.get_best_iteration(),
            metrics=metrics,
            gaps=gaps,
            yearly_metrics=yearly_metrics,
        )
        write_yearly_metrics(trial_dir / "yearly_metrics.csv", yearly_metrics)
        write_json(
            trial_dir / "metrics.json",
            {
                "result": result.to_row(),
                "metrics": metrics,
                "gaps": gaps,
                "yearly_metrics": yearly_metrics,
                "threshold": threshold_info,
                "feature_plan": feature_plan,
            },
        )
        if args.save_every_model:
            model.save_model(trial_dir / "catboost_model.cbm")
        return result, model
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        write_json(
            trial_dir / "error.json",
            {
                "trial": trial_index,
                "error": error,
                "traceback": traceback.format_exc(),
                "assignments": assignments,
                "feature_plan": feature_plan,
            },
        )
        return (
            TrialResult(
                trial=trial_index,
                status="failed",
                duration_sec=perf_counter() - start_time,
                objective=None,
                params={},
                feature_plan=feature_plan,
                threshold=None,
                best_iteration=None,
                metrics={},
                gaps={},
                yearly_metrics={},
                error=error,
            ),
            None,
        )


def save_leaderboard(run_dir: Path, results: Sequence[TrialResult]) -> Path:
    rows = [result.to_row() for result in results]
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty and "objective" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            ["objective", "trial"],
            ascending=[False, True],
            na_position="last",
            kind="mergesort",
        )
    path = run_dir / "leaderboard.csv"
    leaderboard.to_csv(path, index=False)
    return path


def best_row_by_metric(
    results: Sequence[TrialResult],
    split: str,
    metric: str,
    maximize: bool = True,
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[float, TrialResult]] = []
    for result in results:
        if result.status != "ok":
            continue
        value = result.metrics.get(split, {}).get(metric)
        if value is None:
            continue
        scalar = float(value)
        if math.isfinite(scalar):
            candidates.append((scalar, result))
    if not candidates:
        return None
    value, result = sorted(candidates, key=lambda item: item[0], reverse=maximize)[0]
    row = result.to_row()
    row[f"selected_by_{split}_{metric}"] = value
    return row


def compact_metric_summary(result: TrialResult) -> Dict[str, Any]:
    return {
        "trial": result.trial,
        "objective": result.objective,
        "threshold": result.threshold,
        "feature_count": result.feature_plan.get("feature_count"),
        "feature_mode": result.feature_plan.get("mode"),
        "validation_average_precision": result.metrics.get("validation", {}).get(
            "average_precision"
        ),
        "validation_f1": result.metrics.get("validation", {}).get("f1"),
        "test_average_precision": result.metrics.get("test", {}).get("average_precision"),
        "test_f1": result.metrics.get("test", {}).get("f1"),
        "test_precision": result.metrics.get("test", {}).get("precision"),
        "test_recall": result.metrics.get("test", {}).get("recall"),
    }


def write_best_copy_paste_artifacts(
    run_dir: Path,
    best_dir: Path,
    result: TrialResult,
    suggested_config: Dict[str, Any],
    suggested_config_path: Path,
    train_catboost_features_path: Path,
) -> None:
    catboost_train = suggested_config["catboost_train"]
    snippet = {
        "catboost_train": {
            "features": catboost_train["features"],
            "model": catboost_train["model"],
            "thresholding": catboost_train["thresholding"],
        }
    }
    write_yaml(best_dir / "best_train_snippet.yaml", snippet)
    write_yaml(
        best_dir / "selected_model_features.yaml",
        {"selected_feature_columns": result.feature_plan["features"]},
    )

    summary = compact_metric_summary(result)
    snippet_text = yaml.safe_dump(
        tc.sanitize_for_json(snippet),
        sort_keys=False,
    ).strip()
    selected_features_text = yaml.safe_dump(
        tc.sanitize_for_json({"selected_feature_columns": result.feature_plan["features"]}),
        sort_keys=False,
    ).strip()
    summary_text = yaml.safe_dump(tc.sanitize_for_json(summary), sort_keys=False).strip()
    command = f"python train_catboost.py {suggested_config_path.resolve()}"
    copy_paste_text = "\n\n".join(
        [
            "# Best CatBoost tuning result so far",
            f"# Updated: {datetime.now().isoformat(timespec='seconds')}",
            f"# Full suggested config: {suggested_config_path.resolve()}",
            f"# Selected feature file: {train_catboost_features_path.resolve()}",
            "# Run this to train the current best configuration",
            command,
            "# Metric summary",
            summary_text,
            "# Pasteable catboost_train snippet",
            snippet_text,
            "# Pasteable selected model feature columns",
            selected_features_text,
        ]
    )
    for path in (
        best_dir / "best_so_far_copy_paste.txt",
        run_dir / "best_so_far_copy_paste.txt",
    ):
        with path.open("w", encoding="utf-8") as handle:
            handle.write(copy_paste_text)
            handle.write("\n")


def copy_best_artifacts(
    run_dir: Path,
    result: TrialResult,
    model: CatBoostClassifier,
    args: argparse.Namespace,
) -> None:
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(best_dir / "catboost_model.cbm")
    write_json(
        best_dir / "best_trial.json",
        {
            "trial": result.trial,
            "objective": result.objective,
            "row": result.to_row(),
            "metrics": result.metrics,
            "gaps": result.gaps,
            "yearly_metrics": result.yearly_metrics,
            "params": result.params,
            "feature_plan": result.feature_plan,
        },
    )
    write_yearly_metrics(best_dir / "yearly_metrics.csv", result.yearly_metrics)
    write_json(best_dir / "yearly_metrics.json", {"rows": [
        row
        for split_rows in result.yearly_metrics.values()
        for row in split_rows
    ]})
    with (best_dir / "selected_model_features.txt").open("w", encoding="utf-8") as handle:
        for feature in result.feature_plan["features"]:
            handle.write(f"{feature}\n")
    train_catboost_features_path = best_dir / "selected_features_for_train_catboost.txt"
    with train_catboost_features_path.open("w", encoding="utf-8") as handle:
        for feature in result.feature_plan.get(
            "train_catboost_selected_features",
            result.feature_plan["features"],
        ):
            handle.write(f"{feature}\n")
    params = result.params
    suggested_config = {
        "catboost_train": {
            "feature_config_path": args.config,
            "run": {
                "output_root": args.output_root,
                "run_prefix": "catboost_train_from_tune",
            },
            "analysis": {"analysis_only": False, "model_path": None},
            "data": {
                "features_path": args.features_path,
                "rebuild_features": args.rebuild_features,
                "per_country_features_dir": args.per_country_features_dir,
                "per_country_pattern": args.per_country_pattern,
                "limit_rows": args.limit_rows,
                "save_full_eval_data": True,
                "countries": list(args.countries),
            },
            "target": {
                "column": args.target_column,
                "positive_threshold": args.positive_threshold,
            },
            "split": {
                "date_column": args.date_column,
                "validation_start_date": args.validation_start_date,
                "test_start_date": args.test_start_date,
                "validation_size": args.validation_size,
                "test_size": args.test_size,
                "shuffle": args.shuffle,
                "random_state": args.random_state,
            },
            "features": {
                "selected_features_path": train_catboost_features_path.resolve(),
                "selected_feature_filter": True,
                "ignored": list(args.ignored_features),
                "drop_features": list(args.drop_feature),
                "drop_feature_prefixes": list(args.drop_feature_prefix),
                "drop_feature_groups": list(args.drop_feature_group),
                "feature_weights": dict(args.feature_weight or {}),
            },
            "model": {
                "iterations": params.get("iterations"),
                "depth": params.get("depth"),
                "learning_rate": params.get("learning_rate"),
                "l2_leaf_reg": params.get("l2_leaf_reg"),
                "bagging_temperature": params.get("bagging_temperature"),
                "min_data_in_leaf": params.get("min_data_in_leaf"),
                "loss_function": params.get("loss_function", args.loss_function),
                "eval_metric": params.get("eval_metric", args.eval_metric),
                "early_stopping_rounds": args.early_stopping_rounds,
                "random_strength": params.get("random_strength"),
                "rsm": params.get("rsm", 1.0),
                "class_weight_negative": params.get("class_weights", [1.0, None])[0],
                "class_weight_positive": params.get("class_weights", [1.0, None])[1],
                "verbose": 50,
                "bootstrap_type": params.get("bootstrap_type"),
                "subsample": params.get("subsample"),
            },
            "hardware": {"task_type": params.get("task_type")},
            "thresholding": {
                "prediction_threshold": result.threshold,
                "tuning_split": "validation",
                "min_precision": args.threshold_min_precision,
                "min_recall": args.threshold_min_recall,
            },
            "plots": {
                "train_map_year": 2011,
                "test_map_year": 2025,
                "region_map_year": None,
                "country_shapes_path": "data/countries",
                "map_plots": True,
                "region_map_plots": True,
            },
            "feature_importance_analysis": {"enabled": False},
        }
    }
    suggested_config_path = best_dir / "suggested_train_config.yaml"
    write_yaml(suggested_config_path, suggested_config)
    with (best_dir / "train_catboost_command.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"python train_catboost.py {suggested_config_path.resolve()}\n")
    write_best_copy_paste_artifacts(
        run_dir=run_dir,
        best_dir=best_dir,
        result=result,
        suggested_config=suggested_config,
        suggested_config_path=suggested_config_path,
        train_catboost_features_path=train_catboost_features_path,
    )


def prepare_data(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    config = tc.load_config(args.config)
    shutil.copy2(args.config, run_dir / "feature_config.yaml")
    selected_features_path, selected_features = tc.load_selected_features(
        config,
        override_path=args.selected_features_path,
    )
    feature_weights = tc.parse_feature_weights(args.feature_weight)

    df = tc.load_or_build_features(
        features_path=args.features_path,
        rebuild=args.rebuild_features,
        per_country_dir=args.per_country_features_dir,
        per_country_pattern=args.per_country_pattern,
        countries=args.countries,
    )
    if args.limit_rows is not None:
        logger.info("Limiting dataframe from %d to %d rows.", len(df), args.limit_rows)
        df = df.head(args.limit_rows)

    X, y, configured_cat_features, cat_features, missing_cat_features = tc.prepare_dataframe(
        df=df,
        config=config,
        target_column=args.target_column,
    )
    selected_feature_filter_enabled = bool(selected_features) and not args.no_selected_feature_filter
    X, applied_selected_features, missing_selected_features, dropped_by_selected_filter = (
        tc.apply_selected_feature_filter(
            X=X,
            selected_features=selected_features,
            enabled=selected_feature_filter_enabled,
        )
    )
    X, dropped_by_exclusions, dropped_exact, dropped_prefix, dropped_group = (
        tc.apply_feature_exclusions(
            X=X,
            drop_features=args.drop_feature,
            drop_prefixes=args.drop_feature_prefix,
            drop_groups=args.drop_feature_group,
        )
    )
    if not args.shuffle and args.date_column not in X.columns:
        raise ValueError(
            f"Date column {args.date_column!r} is required for chronological tuning."
        )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        split_info,
    ) = tc.split_train_validation_test(X, y, build_split_args(args))
    if X_validation is None or y_validation is None:
        raise ValueError("This tuner requires a validation split for thresholding and early stopping.")

    y_train_binary = (y_train > args.positive_threshold).astype(int)
    y_validation_binary = (y_validation > args.positive_threshold).astype(int)
    y_test_binary = (y_test > args.positive_threshold).astype(int)

    X_train_model, model_cat_features, model_feature_weights, dropped_model_features, _, _ = (
        tc.prepare_catboost_input(
            X=X_train,
            ignored_features=args.ignored_features,
            cat_features=cat_features,
            feature_weights=feature_weights,
        )
    )
    X_validation_model = X_validation[X_train_model.columns]
    X_test_model = X_test[X_train_model.columns]

    logger.info(
        "Prepared data: train=%d validation=%d test=%d model_features=%d",
        len(X_train_model),
        len(X_validation_model),
        len(X_test_model),
        X_train_model.shape[1],
    )
    logger.info("Train balance: %s", tc.class_balance(y_train_binary))
    logger.info("Validation balance: %s", tc.class_balance(y_validation_binary))
    logger.info("Test balance: %s", tc.class_balance(y_test_binary))
    if dropped_by_exclusions:
        logger.info("Base feature exclusions removed %d columns.", len(dropped_by_exclusions))

    metadata = {
        "created_at": datetime.now().isoformat(),
        "argv": sys.argv[1:],
        "paths": {
            "run_config": args.run_config,
            "config": args.config,
            "features_path": args.features_path,
            "selected_features_path": selected_features_path,
            "importance_csv": args.importance_csv,
        },
        "data": {
            "input_rows": int(len(df)),
            "input_columns": int(df.shape[1]),
            "split_info": split_info,
            "countries": list(args.countries),
            "selected_feature_filter_enabled": selected_feature_filter_enabled,
            "applied_selected_features_count": len(applied_selected_features),
            "missing_selected_features": missing_selected_features,
            "dropped_by_selected_feature_filter": dropped_by_selected_filter,
            "feature_exclusions": {
                "drop_feature": args.drop_feature,
                "drop_feature_prefix": args.drop_feature_prefix,
                "drop_feature_group": args.drop_feature_group,
                "dropped": dropped_by_exclusions,
                "dropped_exact": dropped_exact,
                "dropped_prefix": dropped_prefix,
                "dropped_group": dropped_group,
            },
            "dropped_non_model_features": dropped_model_features,
            "train_balance": tc.class_balance(y_train_binary),
            "validation_balance": tc.class_balance(y_validation_binary),
            "test_balance": tc.class_balance(y_test_binary),
        },
        "preprocessing": {
            "configured_cat_features": configured_cat_features,
            "effective_cat_features": cat_features,
            "missing_cat_features": missing_cat_features,
            "model_cat_features": model_cat_features,
            "model_feature_weights": model_feature_weights,
            "model_features": list(X_train_model.columns),
        },
        "search": vars(args),
    }
    write_yaml(run_dir / "tuning_config.yaml", metadata)
    write_json(run_dir / "tuning_config.json", metadata)

    return {
        "X_train_model": X_train_model,
        "X_validation_model": X_validation_model,
        "X_test_model": X_test_model,
        "y_train_binary": y_train_binary,
        "y_validation_binary": y_validation_binary,
        "y_test_binary": y_test_binary,
        "validation_years": pd.to_datetime(X_validation[args.date_column]).dt.year.to_numpy(),
        "test_years": pd.to_datetime(X_test[args.date_column]).dt.year.to_numpy(),
        "model_cat_features": model_cat_features,
        "model_feature_weights": model_feature_weights,
        "dropped_model_features": dropped_model_features,
        "metadata": metadata,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli_args = parse_args(argv)
    args = build_args_from_config(cli_args.config)
    if args.max_trials <= 0:
        raise ValueError("--max-trials must be positive.")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds must be >= 0.")
    if not 0.0 <= args.top_k_probability <= 1.0:
        raise ValueError("--top-k-probability must be in [0, 1].")
    if not 0.0 <= args.drop_group_probability <= 1.0:
        raise ValueError("--drop-group-probability must be in [0, 1].")
    if not 0.0 <= args.group_sample_probability <= 1.0:
        raise ValueError("--group-sample-probability must be in [0, 1].")
    if not 0.0 <= args.column_sample_probability <= 1.0:
        raise ValueError("--column-sample-probability must be in [0, 1].")
    if args.max_drop_groups < 0:
        raise ValueError("--max-drop-groups must be >= 0.")
    if args.min_features <= 0:
        raise ValueError("--min-features must be positive.")
    parse_fraction_choices(args.group_sample_fraction_choices, "group_sample_fraction_choices")
    parse_fraction_choices(args.column_sample_fraction_choices, "column_sample_fraction_choices")
    parse_top_k_choices(args.top_k_choices)
    tc.validate_probability("--prediction-threshold", args.prediction_threshold)
    tc.validate_probability("--threshold-min-precision", args.threshold_min_precision)
    tc.validate_probability("--threshold-min-recall", args.threshold_min_recall)

    run_dir = tc.create_run_dir(args.output_root, args.run_prefix)
    setup_logging(run_dir)
    logger.info("Created tuning run directory: %s", run_dir.resolve())
    if args.run_config.is_file():
        shutil.copy2(args.run_config, run_dir / "run_config.yaml")
    else:
        logger.info("Skipping run config copy because %s is not a regular file.", args.run_config)

    search_space = load_search_space(args.search_space)
    data = prepare_data(args, run_dir)
    all_features = list(data["X_train_model"].columns)
    importance_order = load_importance_order(args.importance_csv, all_features)
    if args.feature_search in {"importance", "both"} and not importance_order:
        logger.info("No importance CSV supplied; importance-based feature search is disabled.")
    else:
        logger.info("Loaded importance ordering for %d features.", len(importance_order))
    gap_weights = (
        args.train_test_gap_weight,
        args.validation_test_gap_weight,
        args.train_validation_gap_weight,
    )
    gap_description = (
        "without gap penalties"
        if all(weight == 0.0 for weight in gap_weights)
        else (
            "with gap penalties "
            f"(train-test={args.train_test_gap_weight:.3f}, "
            f"validation-test={args.validation_test_gap_weight:.3f}, "
            f"train-validation={args.train_validation_gap_weight:.3f})"
        )
    )
    logger.info(
        "Optimizing objective: %s %s %s. CatBoost eval_metric for early stopping is %s.",
        args.objective_split,
        args.objective_metric,
        gap_description,
        args.eval_metric,
    )

    rng = random.Random(args.seed)
    results: List[TrialResult] = []
    best_result: Optional[TrialResult] = None
    best_model: Optional[CatBoostClassifier] = None
    started_at = perf_counter()

    for trial_index, assignments in enumerate(
        iter_assignments(search_space, args.strategy, args.max_trials, rng),
        start=1,
    ):
        feature_plan = build_feature_plan(all_features, importance_order, args, rng)
        logger.info(
            "Trial %d/%d: features=%d mode=%s params=%s",
            trial_index,
            args.max_trials,
            feature_plan["feature_count"],
            feature_plan["mode"],
            assignments,
        )
        result, model = fit_and_score_trial(
            trial_index=trial_index,
            assignments=assignments,
            feature_plan=feature_plan,
            data=data,
            args=args,
            run_dir=run_dir,
        )
        results.append(result)
        save_leaderboard(run_dir, results)

        if result.status == "ok":
            logger.info(
                "Trial %d objective=%.6f val_AP=%.4f test_AP=%.4f test_f1=%.4f",
                trial_index,
                result.objective if result.objective is not None else float("nan"),
                result.metrics["validation"]["average_precision"] or float("nan"),
                result.metrics["test"]["average_precision"] or float("nan"),
                result.metrics["test"]["f1"] or float("nan"),
            )
            if (
                model is not None
                and result.objective is not None
                and (best_result is None or result.objective > (best_result.objective or -math.inf))
            ):
                best_result = result
                best_model = model
                copy_best_artifacts(run_dir, best_result, best_model, args)
                logger.info("New best trial: %d objective=%.6f", result.trial, result.objective)
                logger.info(
                    "Best-so-far copy/paste summary: %s",
                    run_dir / "best_so_far_copy_paste.txt",
                )
                tc.log_yearly_summary(
                    result.yearly_metrics,
                    title=f"Yearly metrics for best trial {result.trial}",
                )
        else:
            logger.warning("Trial %d failed: %s", trial_index, result.error)
            if args.fail_fast:
                break

    leaderboard_path = save_leaderboard(run_dir, results)
    summary = {
        "run_dir": run_dir,
        "leaderboard": leaderboard_path,
        "duration_sec": perf_counter() - started_at,
        "completed_trials": len(results),
        "successful_trials": sum(result.status == "ok" for result in results),
        "best_trial": best_result.to_row() if best_result is not None else None,
        "best_test_f1_trial": best_row_by_metric(results, "test", "f1"),
        "best_test_average_precision_trial": best_row_by_metric(
            results,
            "test",
            "average_precision",
        ),
        "best_validation_f1_trial": best_row_by_metric(results, "validation", "f1"),
    }
    write_json(run_dir / "summary.json", summary)
    if best_result is None:
        logger.error("No successful trials. See %s for errors.", run_dir / "trials")
        return 1
    logger.info("Best trial: %d objective=%.6f", best_result.trial, best_result.objective)
    logger.info("Leaderboard: %s", leaderboard_path)
    logger.info("Best artifacts: %s", run_dir / "best")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
