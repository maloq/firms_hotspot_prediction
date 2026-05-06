from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_catboost import (
    align_features_to_model,
    apply_feature_exclusions,
    apply_selected_feature_filter,
    binary_from_probabilities,
    build_args_from_config,
    choose_threshold_by_f1,
    infer_feature_group,
    load_config,
    load_or_build_features,
    load_selected_features,
    parse_feature_weights,
    prepare_catboost_input,
    prepare_dataframe,
    predict_probabilities,
    resolve_model_cat_features,
    split_train_validation_test,
    summarize_probability_metrics,
)


EPS = 1e-9


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why CatBoost wildfire predictions differ by year and "
            "quantify easy/hard years."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="CatBoost training YAML config used for the run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing model/catboost_model.cbm. Defaults to latest run.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override model path. Defaults to <run-dir>/model/catboost_model.cbm.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic outputs. Defaults to <run-dir>/evaluation/year_difficulty.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=60,
        help="Number of important model features to use for PSI shift diagnostics.",
    )
    parser.add_argument(
        "--psi-bins",
        type=int,
        default=10,
        help="Quantile bins for numerical PSI diagnostics.",
    )
    parser.add_argument(
        "--historical-prior-alpha",
        type=float,
        default=50.0,
        help="Bayesian smoothing strength for historical ecoregion/month prior.",
    )
    return parser.parse_args(argv)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(data), handle, indent=2)


def latest_run_dir(output_root: Path, run_prefix: str) -> Path:
    candidates = [
        path
        for path in output_root.glob(f"{run_prefix}_*")
        if (path / "model" / "catboost_model.cbm").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run directories with model/catboost_model.cbm found under {output_root}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def safe_metric(func, y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    try:
        value = func(y_true, y_score)
    except ValueError:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int = 10,
) -> Optional[float]:
    if len(y_true) == 0:
        return None
    y_array = np.asarray(y_true, dtype=float)
    prob_array = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (prob_array >= left) & (prob_array <= right)
        else:
            mask = (prob_array >= left) & (prob_array < right)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(prob_array[mask].mean()) - float(y_array[mask].mean()))
    return ece


def probability_quantile(values: np.ndarray, q: float) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(np.quantile(values, q))


def logit(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    clipped = min(max(float(value), EPS), 1.0 - EPS)
    return float(math.log(clipped / (1.0 - clipped)))


def top_k_precision(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> Tuple[Optional[float], Optional[float]]:
    if k <= 0 or len(y_true) == 0:
        return None, None
    k = min(k, len(y_true))
    order = np.argsort(-np.asarray(y_prob, dtype=float), kind="mergesort")[:k]
    hits = int(np.asarray(y_true, dtype=int)[order].sum())
    positives = int(np.asarray(y_true, dtype=int).sum())
    precision = float(hits / k) if k else None
    recall = float(hits / positives) if positives else None
    return precision, recall


def probability_metrics_for_year(
    split_name: str,
    year: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    base = summarize_probability_metrics(
        split_name=split_name,
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        year=year,
    )
    y_array = np.asarray(y_true, dtype=int)
    prob_array = np.asarray(y_prob, dtype=float)
    pred_array = binary_from_probabilities(prob_array, threshold)
    cm = confusion_matrix(y_array, pred_array, labels=[0, 1])
    best = choose_threshold_by_f1(y_array, prob_array)
    best_threshold = best.get("threshold")
    best_pred = binary_from_probabilities(prob_array, float(best_threshold))

    positives = int(y_array.sum())
    precision_at_pos_count, recall_at_pos_count = top_k_precision(
        y_array,
        prob_array,
        positives,
    )
    pos_scores = prob_array[y_array == 1]
    neg_scores = prob_array[y_array == 0]
    normalized_ap = base.get("normalized_average_precision")
    roc_auc = base.get("roc_auc")
    global_f1 = base.get("f1")
    best_f1 = f1_score(y_array, best_pred, zero_division=0)

    base.update(
        {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
            "false_positive_rate": float(cm[0, 1] / max(cm[0].sum(), 1)),
            "false_negative_rate": float(cm[1, 0] / max(cm[1].sum(), 1)),
            "best_f1_threshold": float(best_threshold),
            "best_f1": float(best_f1),
            "best_f1_precision": precision_score(y_array, best_pred, zero_division=0),
            "best_f1_recall": recall_score(y_array, best_pred, zero_division=0),
            "f1_gap_to_year_best": (
                float(best_f1 - global_f1) if global_f1 is not None else None
            ),
            "threshold_logit_delta_from_global": (
                float(logit(best_threshold) - logit(threshold))
                if logit(best_threshold) is not None and logit(threshold) is not None
                else None
            ),
            "precision_at_actual_positive_count": precision_at_pos_count,
            "recall_at_actual_positive_count": recall_at_pos_count,
            "brier_score": brier_score(y_array, prob_array),
            "log_loss": safe_metric(
                lambda yt, yp: log_loss(yt, np.clip(yp, EPS, 1.0 - EPS), labels=[0, 1]),
                y_array,
                prob_array,
            ),
            "expected_calibration_error_10bin": expected_calibration_error(y_array, prob_array),
            "probability_mean_minus_prevalence": (
                float(base["probability_mean"] - base["prevalence"])
                if base.get("probability_mean") is not None and base.get("prevalence") is not None
                else None
            ),
            "predicted_positive_rate_minus_prevalence": (
                float(base["predicted_positive_rate"] - base["prevalence"])
                if base.get("predicted_positive_rate") is not None and base.get("prevalence") is not None
                else None
            ),
            "pos_probability_mean": float(pos_scores.mean()) if len(pos_scores) else None,
            "pos_probability_median": probability_quantile(pos_scores, 0.50),
            "pos_probability_p25": probability_quantile(pos_scores, 0.25),
            "pos_probability_p75": probability_quantile(pos_scores, 0.75),
            "neg_probability_mean": float(neg_scores.mean()) if len(neg_scores) else None,
            "neg_probability_median": probability_quantile(neg_scores, 0.50),
            "neg_probability_p75": probability_quantile(neg_scores, 0.75),
            "neg_probability_p90": probability_quantile(neg_scores, 0.90),
            "neg_probability_p95": probability_quantile(neg_scores, 0.95),
            "mean_score_gap_pos_minus_neg": (
                float(pos_scores.mean() - neg_scores.mean())
                if len(pos_scores) and len(neg_scores)
                else None
            ),
            "ranking_difficulty_0_100": (
                float(100.0 * (1.0 - min(max(normalized_ap, 0.0), 1.0)))
                if normalized_ap is not None
                else None
            ),
            "auc_difficulty_0_100": (
                float(100.0 * (1.0 - min(max(roc_auc, 0.5), 1.0)) / 0.5)
                if roc_auc is not None
                else None
            ),
            "operational_difficulty_0_100": (
                float(100.0 * (1.0 - global_f1))
                if global_f1 is not None
                else None
            ),
        }
    )
    return base


def make_split_predictions(
    split_name: str,
    X: pd.DataFrame,
    y_count: pd.Series,
    y_binary: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    date_column: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "split": split_name,
            "year": pd.to_datetime(X[date_column]).dt.year.to_numpy(dtype=int),
            "datetime": pd.to_datetime(X[date_column]).to_numpy(),
            "target_count": np.asarray(y_count),
            "target_binary": np.asarray(y_binary, dtype=int),
            "pred_probability": np.asarray(y_prob, dtype=float),
        },
        index=X.index,
    )
    frame["pred_binary"] = binary_from_probabilities(frame["pred_probability"].to_numpy(), threshold)
    for column in ("month", "lat_rounded", "lon_rounded", "ecoregion_name", "ecoregion_realm"):
        if column in X.columns:
            frame[column] = X[column].to_numpy()
    return frame


def normalized_ap_from_ap(ap: Optional[float], prevalence: float) -> Optional[float]:
    if ap is None or prevalence >= 1.0:
        return None
    return float((ap - prevalence) / (1.0 - prevalence))


def historical_prior_scores(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    keys: Sequence[str],
    y_col: str = "target_binary",
    alpha: float = 50.0,
) -> np.ndarray:
    global_rate = float(train_frame[y_col].mean())
    stats = (
        train_frame.groupby(list(keys), dropna=False)[y_col]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "prior_pos", "count": "prior_rows"})
    )
    stats["historical_prior"] = (
        stats["prior_pos"] + global_rate * alpha
    ) / (stats["prior_rows"] + alpha)
    merged = eval_frame[list(keys)].merge(stats[list(keys) + ["historical_prior"]], on=list(keys), how="left")
    return merged["historical_prior"].fillna(global_rate).to_numpy(dtype=float)


def add_historical_prior_metrics(
    yearly_rows: List[Dict[str, Any]],
    train_reference: pd.DataFrame,
    eval_predictions: pd.DataFrame,
    alpha: float,
) -> None:
    key_sets: Dict[str, Sequence[str]] = {}
    if {"ecoregion_name", "month"}.issubset(eval_predictions.columns):
        key_sets["ecoregion_month_prior"] = ["ecoregion_name", "month"]
    if {"lat_rounded", "lon_rounded", "month"}.issubset(eval_predictions.columns):
        key_sets["geo_month_prior"] = ["lat_rounded", "lon_rounded", "month"]
    if "month" in eval_predictions.columns:
        key_sets["month_prior"] = ["month"]

    for prefix, keys in key_sets.items():
        eval_predictions[f"{prefix}_score"] = historical_prior_scores(
            train_reference,
            eval_predictions,
            keys=keys,
            alpha=alpha,
        )

    row_lookup = {(row["split"], row["year"]): row for row in yearly_rows}
    for (split_name, year), group in eval_predictions.groupby(["split", "year"], sort=True):
        row = row_lookup.get((split_name, int(year)))
        if row is None:
            continue
        y_array = group["target_binary"].to_numpy(dtype=int)
        prevalence = float(y_array.mean()) if len(y_array) else 0.0
        for prefix in key_sets:
            scores = group[f"{prefix}_score"].to_numpy(dtype=float)
            has_both = len(np.unique(y_array)) == 2
            ap = safe_metric(average_precision_score, y_array, scores) if has_both else None
            auc = safe_metric(roc_auc_score, y_array, scores) if has_both else None
            pos_scores = scores[y_array == 1]
            neg_scores = scores[y_array == 0]
            row[f"{prefix}_average_precision"] = ap
            row[f"{prefix}_normalized_average_precision"] = normalized_ap_from_ap(ap, prevalence)
            row[f"{prefix}_roc_auc"] = auc
            row[f"{prefix}_pos_mean"] = float(pos_scores.mean()) if len(pos_scores) else None
            row[f"{prefix}_neg_mean"] = float(neg_scores.mean()) if len(neg_scores) else None


def distribution_from_series(series: pd.Series) -> pd.Series:
    counts = series.astype("object").fillna("<missing>").value_counts(dropna=False)
    total = float(counts.sum())
    return counts / total if total else counts.astype(float)


def js_divergence_bits(left: pd.Series, right: pd.Series) -> Optional[float]:
    if left.empty or right.empty:
        return None
    index = left.index.union(right.index)
    p = left.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    q = right.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    if p.sum() == 0 or q.sum() == 0:
        return None
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def spatial_bin_frame(frame: pd.DataFrame) -> pd.Series:
    lat_bin = np.floor(pd.to_numeric(frame["lat_rounded"], errors="coerce") / 2.5) * 2.5
    lon_bin = np.floor(pd.to_numeric(frame["lon_rounded"], errors="coerce") / 2.5) * 2.5
    return lat_bin.astype(str) + "|" + lon_bin.astype(str)


def composition_shift_rows(
    train_reference: pd.DataFrame,
    eval_predictions: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    train_all_month = distribution_from_series(train_reference["month"]) if "month" in train_reference else pd.Series(dtype=float)
    train_pos_month = (
        distribution_from_series(train_reference.loc[train_reference["target_binary"] == 1, "month"])
        if "month" in train_reference
        else pd.Series(dtype=float)
    )
    train_all_ecoregion = (
        distribution_from_series(train_reference["ecoregion_name"])
        if "ecoregion_name" in train_reference
        else pd.Series(dtype=float)
    )
    train_pos_ecoregion = (
        distribution_from_series(train_reference.loc[train_reference["target_binary"] == 1, "ecoregion_name"])
        if "ecoregion_name" in train_reference
        else pd.Series(dtype=float)
    )
    train_all_spatial = (
        distribution_from_series(spatial_bin_frame(train_reference))
        if {"lat_rounded", "lon_rounded"}.issubset(train_reference.columns)
        else pd.Series(dtype=float)
    )
    train_pos_spatial = (
        distribution_from_series(spatial_bin_frame(train_reference[train_reference["target_binary"] == 1]))
        if {"lat_rounded", "lon_rounded"}.issubset(train_reference.columns)
        else pd.Series(dtype=float)
    )

    for (split_name, year), group in eval_predictions.groupby(["split", "year"], sort=True):
        row: Dict[str, Any] = {"split": split_name, "year": int(year)}
        if "month" in group:
            row["month_all_jsd_vs_train"] = js_divergence_bits(
                distribution_from_series(group["month"]),
                train_all_month,
            )
            row["month_positive_jsd_vs_train_positive"] = js_divergence_bits(
                distribution_from_series(group.loc[group["target_binary"] == 1, "month"]),
                train_pos_month,
            )
        if "ecoregion_name" in group:
            row["ecoregion_all_jsd_vs_train"] = js_divergence_bits(
                distribution_from_series(group["ecoregion_name"]),
                train_all_ecoregion,
            )
            row["ecoregion_positive_jsd_vs_train_positive"] = js_divergence_bits(
                distribution_from_series(group.loc[group["target_binary"] == 1, "ecoregion_name"]),
                train_pos_ecoregion,
            )
        if {"lat_rounded", "lon_rounded"}.issubset(group.columns):
            row["spatial_all_jsd_vs_train"] = js_divergence_bits(
                distribution_from_series(spatial_bin_frame(group)),
                train_all_spatial,
            )
            row["spatial_positive_jsd_vs_train_positive"] = js_divergence_bits(
                distribution_from_series(spatial_bin_frame(group[group["target_binary"] == 1])),
                train_pos_spatial,
            )
        rows.append(row)
    return rows


def psi_from_proportions(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = np.maximum(left, EPS)
    right = np.maximum(right, EPS)
    left = left / left.sum()
    right = right / right.sum()
    return float(np.sum((right - left) * np.log(right / left)))


def numerical_psi(train: pd.Series, eval_: pd.Series, bins: int) -> Optional[float]:
    train_num = pd.to_numeric(train, errors="coerce")
    eval_num = pd.to_numeric(eval_, errors="coerce")
    finite_train = train_num[np.isfinite(train_num)]
    if finite_train.nunique(dropna=True) <= 1:
        return None
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.nanquantile(finite_train.to_numpy(dtype=float), quantiles))
    if len(edges) <= 2:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_bins = pd.cut(train_num, bins=edges, include_lowest=True)
    eval_bins = pd.cut(eval_num, bins=edges, include_lowest=True)
    train_counts = train_bins.value_counts(sort=False, dropna=False).to_numpy(dtype=float)
    eval_counts = eval_bins.value_counts(sort=False, dropna=False).to_numpy(dtype=float)
    return psi_from_proportions(train_counts, eval_counts)


def categorical_psi(train: pd.Series, eval_: pd.Series, max_categories: int = 30) -> float:
    train_values = train.astype("object").where(train.notna(), "<missing>")
    eval_values = eval_.astype("object").where(eval_.notna(), "<missing>")
    top_categories = set(train_values.value_counts().head(max_categories).index)
    train_bucket = train_values.where(train_values.isin(top_categories), "<other>")
    eval_bucket = eval_values.where(eval_values.isin(top_categories), "<other>")
    categories = pd.Index(train_bucket.unique()).union(pd.Index(eval_bucket.unique()))
    train_counts = train_bucket.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    eval_counts = eval_bucket.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    return psi_from_proportions(train_counts, eval_counts)


def feature_psi(
    train: pd.Series,
    eval_: pd.Series,
    feature: str,
    cat_features: Sequence[str],
    bins: int,
) -> Optional[float]:
    if feature in cat_features or train.dtype == object or eval_.dtype == object:
        return categorical_psi(train, eval_)
    value = numerical_psi(train, eval_, bins)
    if value is None:
        return categorical_psi(train, eval_)
    return value


def load_importance_weights(run_dir: Path, model_features: Sequence[str]) -> pd.DataFrame:
    summary_path = run_dir / "evaluation" / "feature_importance" / "test" / "feature_importance_summary.csv"
    if summary_path.exists():
        importance = pd.read_csv(summary_path)
        if {"feature", "mean_normalized_importance"}.issubset(importance.columns):
            importance = importance[["feature", "mean_normalized_importance"]].rename(
                columns={"mean_normalized_importance": "importance_weight"}
            )
        else:
            importance = importance[["feature"]].copy()
            importance["importance_weight"] = 1.0
    else:
        importance = pd.DataFrame({"feature": list(model_features), "importance_weight": 1.0})

    importance = importance[importance["feature"].isin(model_features)].copy()
    if importance.empty:
        importance = pd.DataFrame({"feature": list(model_features), "importance_weight": 1.0})
    importance["group"] = importance["feature"].map(infer_feature_group)
    total = float(importance["importance_weight"].fillna(0.0).clip(lower=0.0).sum())
    if total <= 0:
        importance["importance_weight"] = 1.0 / len(importance)
    else:
        importance["importance_weight"] = importance["importance_weight"].fillna(0.0).clip(lower=0.0) / total
    return importance


def feature_shift_diagnostics(
    X_train_model: pd.DataFrame,
    y_train_binary: pd.Series,
    X_eval_model: pd.DataFrame,
    eval_predictions: pd.DataFrame,
    cat_features: Sequence[str],
    importance: pd.DataFrame,
    top_features: int,
    bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = importance.head(top_features).copy()
    features = [feature for feature in selected["feature"].tolist() if feature in X_train_model.columns and feature in X_eval_model.columns]
    train_pos_mask = np.asarray(y_train_binary, dtype=int) == 1
    train_neg_mask = ~train_pos_mask
    rows: List[Dict[str, Any]] = []

    for (split_name, year), pred_group in eval_predictions.groupby(["split", "year"], sort=True):
        eval_index = pred_group.index
        X_year = X_eval_model.loc[eval_index]
        y_year = pred_group["target_binary"].to_numpy(dtype=int)
        eval_pos_mask = y_year == 1
        eval_neg_mask = ~eval_pos_mask
        for feature in features:
            weight = float(selected.loc[selected["feature"] == feature, "importance_weight"].iloc[0])
            psi_all = feature_psi(
                X_train_model[feature],
                X_year[feature],
                feature,
                cat_features,
                bins,
            )
            psi_pos = (
                feature_psi(
                    X_train_model.loc[train_pos_mask, feature],
                    X_year.loc[eval_pos_mask, feature],
                    feature,
                    cat_features,
                    bins,
                )
                if eval_pos_mask.sum() > 1 and train_pos_mask.sum() > 1
                else None
            )
            psi_neg = (
                feature_psi(
                    X_train_model.loc[train_neg_mask, feature],
                    X_year.loc[eval_neg_mask, feature],
                    feature,
                    cat_features,
                    bins,
                )
                if eval_neg_mask.sum() > 1 and train_neg_mask.sum() > 1
                else None
            )
            rows.append(
                {
                    "split": split_name,
                    "year": int(year),
                    "feature": feature,
                    "group": infer_feature_group(feature),
                    "importance_weight": weight,
                    "psi_all_vs_train": psi_all,
                    "psi_positive_vs_train_positive": psi_pos,
                    "psi_negative_vs_train_negative": psi_neg,
                    "weighted_psi_all_vs_train": None if psi_all is None else weight * psi_all,
                    "weighted_psi_positive_vs_train_positive": None if psi_pos is None else weight * psi_pos,
                    "weighted_psi_negative_vs_train_negative": None if psi_neg is None else weight * psi_neg,
                }
            )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["split", "year"], dropna=False)
        .agg(
            weighted_feature_shift=("weighted_psi_all_vs_train", "sum"),
            weighted_positive_feature_shift=("weighted_psi_positive_vs_train_positive", "sum"),
            weighted_negative_feature_shift=("weighted_psi_negative_vs_train_negative", "sum"),
            mean_feature_psi=("psi_all_vs_train", "mean"),
            max_feature_psi=("psi_all_vs_train", "max"),
            mean_positive_feature_psi=("psi_positive_vs_train_positive", "mean"),
            max_positive_feature_psi=("psi_positive_vs_train_positive", "max"),
        )
        .reset_index()
    )
    return detail, summary


def target_summary_all_years(
    X: pd.DataFrame,
    y_binary: pd.Series,
    split_labels: pd.Series,
    date_column: str,
) -> pd.DataFrame:
    dates = pd.to_datetime(X[date_column])
    frame = pd.DataFrame(
        {
            "split": split_labels.to_numpy(),
            "year": dates.dt.year.to_numpy(dtype=int),
            "datetime": dates.to_numpy(),
            "target_binary": np.asarray(y_binary, dtype=int),
        }
    )
    summary = (
        frame.groupby(["split", "year"], sort=True)
        .agg(
            rows=("target_binary", "size"),
            positives=("target_binary", "sum"),
            date_min=("datetime", "min"),
            date_max=("datetime", "max"),
            unique_days=("datetime", lambda value: pd.Series(value).dt.normalize().nunique()),
        )
        .reset_index()
    )
    summary["prevalence"] = summary["positives"] / summary["rows"]
    summary["months_covered"] = summary.apply(
        lambda row: int(
            frame.loc[
                (frame["split"] == row["split"]) & (frame["year"] == row["year"]),
                "datetime",
            ].dt.month.nunique()
        ),
        axis=1,
    )
    return summary


def error_slice_rows(eval_predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = [col for col in ["split", "year", "ecoregion_name", "month"] if col in eval_predictions.columns]
    if len(group_cols) < 3:
        return pd.DataFrame()
    frame = eval_predictions.copy()
    frame["tp"] = ((frame["target_binary"] == 1) & (frame["pred_binary"] == 1)).astype(int)
    frame["fp"] = ((frame["target_binary"] == 0) & (frame["pred_binary"] == 1)).astype(int)
    frame["fn"] = ((frame["target_binary"] == 1) & (frame["pred_binary"] == 0)).astype(int)
    frame["tn"] = ((frame["target_binary"] == 0) & (frame["pred_binary"] == 0)).astype(int)
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            rows=("target_binary", "size"),
            positives=("target_binary", "sum"),
            predicted_positives=("pred_binary", "sum"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tn=("tn", "sum"),
            mean_probability=("pred_probability", "mean"),
        )
        .reset_index()
    )
    grouped["prevalence"] = grouped["positives"] / grouped["rows"]
    grouped["precision"] = grouped["tp"] / (grouped["tp"] + grouped["fp"]).replace(0, np.nan)
    grouped["recall"] = grouped["tp"] / grouped["positives"].replace(0, np.nan)
    grouped["error_count"] = grouped["fp"] + grouped["fn"]
    grouped = grouped[grouped["rows"] >= 50].copy()
    grouped["error_rank"] = (
        grouped.groupby(["split", "year"])["error_count"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return grouped[grouped["error_rank"] <= 25].sort_values(["split", "year", "error_rank"])


def merge_yearly_rows(
    metrics: pd.DataFrame,
    feature_shift: pd.DataFrame,
    composition_shift: pd.DataFrame,
) -> pd.DataFrame:
    result = metrics.merge(feature_shift, on=["split", "year"], how="left")
    result = result.merge(composition_shift, on=["split", "year"], how="left")
    rank_source = result["ranking_difficulty_0_100"].fillna(result["operational_difficulty_0_100"])
    result["difficulty_rank_within_eval_years"] = rank_source.rank(method="dense", ascending=False).astype(int)
    return result.sort_values(["difficulty_rank_within_eval_years", "split", "year"])


def markdown_table(df: pd.DataFrame, columns: Sequence[str], float_digits: int = 3) -> str:
    table = df.loc[:, columns].copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}"
            )
    table = table.fillna("")
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def write_report(
    output_dir: Path,
    run_dir: Path,
    threshold: float,
    yearly: pd.DataFrame,
    target_summary: pd.DataFrame,
    top_shift: pd.DataFrame,
    error_slices: pd.DataFrame,
) -> None:
    ordered = yearly.sort_values("ranking_difficulty_0_100", ascending=False)
    eval_years = ordered[ordered["split"].isin(["validation", "test"])].copy()
    target_recent = target_summary[target_summary["year"] >= 2018].copy()
    top_shift_preview = top_shift.head(20).copy()
    error_preview = error_slices.head(20).copy() if not error_slices.empty else pd.DataFrame()

    lines = [
        "# Year Difficulty Diagnostics",
        "",
        f"Run directory: `{run_dir}`",
        f"Global probability threshold: `{threshold:.6f}`",
        "",
        "## Year Ranking",
        "",
        "Primary ranking difficulty is `100 * (1 - normalized_average_precision)`. ",
        "It is prevalence-normalized and threshold-free: 0 is easy/perfect, 100 is no better than random ranking for that year's prevalence.",
        "",
        markdown_table(
            eval_years,
            [
                "split",
                "year",
                "prevalence",
                "average_precision",
                "normalized_average_precision",
                "roc_auc",
                "f1",
                "best_f1",
                "ranking_difficulty_0_100",
                "operational_difficulty_0_100",
                "weighted_feature_shift",
                "weighted_positive_feature_shift",
            ],
        ),
        "",
        "## Target Prevalence Since 2018",
        "",
        markdown_table(
            target_recent,
            ["split", "year", "rows", "positives", "prevalence", "date_min", "date_max", "unique_days"],
        ),
        "",
        "## Largest Important-Feature Shifts",
        "",
        markdown_table(
            top_shift_preview,
            [
                "split",
                "year",
                "feature",
                "group",
                "importance_weight",
                "psi_all_vs_train",
                "psi_positive_vs_train_positive",
                "weighted_psi_all_vs_train",
            ],
        ),
    ]

    if not error_preview.empty:
        lines.extend(
            [
                "",
                "## Largest Error Slices",
                "",
                markdown_table(
                    error_preview,
                    [
                        "split",
                        "year",
                        "ecoregion_name",
                        "month",
                        "rows",
                        "positives",
                        "predicted_positives",
                        "fp",
                        "fn",
                        "precision",
                        "recall",
                    ],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `year_difficulty_metrics.csv`: merged year-level metrics and difficulty scores.",
            "- `feature_shift_by_year_feature.csv`: PSI by important feature and year.",
            "- `feature_shift_by_year_summary.csv`: weighted PSI summary by year.",
            "- `composition_shift_by_year.csv`: month/ecoregion/spatial distribution shift against train.",
            "- `top_error_slices.csv`: ecoregion/month slices with most false positives plus false negatives.",
            "- `target_summary_all_years.csv`: target prevalence and coverage for all years.",
        ]
    )
    (output_dir / "year_difficulty_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli_args = parse_args(argv)
    args = build_args_from_config(cli_args.config)
    run_dir = cli_args.run_dir or latest_run_dir(args.output_root, args.run_prefix)
    model_path = cli_args.model_path or (run_dir / "model" / "catboost_model.cbm")
    output_dir = cli_args.output_dir or (run_dir / "evaluation" / "year_difficulty")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    selected_features_path, selected_features = load_selected_features(
        config,
        override_path=args.selected_features_path,
    )
    feature_weights = parse_feature_weights(args.feature_weight)
    _ = feature_weights  # Kept to mirror training preprocessing and make config mistakes visible.

    df = load_or_build_features(
        features_path=args.features_path,
        rebuild=False,
        per_country_dir=args.per_country_features_dir,
        per_country_pattern=args.per_country_pattern,
        countries=args.countries,
    )
    if args.limit_rows is not None:
        df = df.head(args.limit_rows)

    X, y, _, cat_features, _ = prepare_dataframe(
        df=df,
        config=config,
        target_column=args.target_column,
    )
    selected_feature_filter_enabled = bool(selected_features) and not args.no_selected_feature_filter
    X, _, _, _ = apply_selected_feature_filter(
        X=X,
        selected_features=selected_features,
        enabled=selected_feature_filter_enabled,
    )
    X, _, _, _, _ = apply_feature_exclusions(
        X=X,
        drop_features=args.drop_feature,
        drop_prefixes=args.drop_feature_prefix,
        drop_groups=args.drop_feature_group,
    )

    X_train, X_validation, X_test, y_train, y_validation, y_test, _ = split_train_validation_test(X, y, args)
    if X_validation is None or y_validation is None:
        raise ValueError("This diagnostic expects a validation split.")

    y_binary = (y > args.positive_threshold).astype(int)
    y_train_binary = (y_train > args.positive_threshold).astype(int)
    y_validation_binary = (y_validation > args.positive_threshold).astype(int)
    y_test_binary = (y_test > args.positive_threshold).astype(int)

    ignored_features = [col for col in args.ignored_features if col in X_train.columns]
    X_train_model, model_cat_features, _, _, _, _ = prepare_catboost_input(
        X=X_train,
        ignored_features=ignored_features,
        cat_features=cat_features,
        feature_weights=parse_feature_weights(args.feature_weight),
    )

    model = CatBoostClassifier()
    model.load_model(model_path)
    X_train_model, _ = align_features_to_model(X_train_model, model, "train")
    X_validation_model = X_validation[X_train_model.columns]
    X_test_model = X_test[X_train_model.columns]
    X_validation_model, _ = align_features_to_model(X_validation_model, model, "validation")
    X_test_model, _ = align_features_to_model(X_test_model, model, "test")
    prediction_cat_features = resolve_model_cat_features(model, model_cat_features, X_train_model)

    validation_prob = predict_probabilities(model, X_validation_model, prediction_cat_features)
    test_prob = predict_probabilities(model, X_test_model, prediction_cat_features)

    threshold_path = run_dir / "evaluation" / "metrics.json"
    if threshold_path.exists():
        metrics_json = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = float(metrics_json.get("test", {}).get("threshold", 0.5))
    else:
        threshold = 0.5

    validation_predictions = make_split_predictions(
        "validation",
        X_validation,
        y_validation,
        y_validation_binary,
        validation_prob,
        threshold,
        args.date_column,
    )
    test_predictions = make_split_predictions(
        "test",
        X_test,
        y_test,
        y_test_binary,
        test_prob,
        threshold,
        args.date_column,
    )
    eval_predictions = pd.concat([validation_predictions, test_predictions], axis=0)
    eval_model = pd.concat([X_validation_model, X_test_model], axis=0)

    train_reference = pd.DataFrame(
        {
            "target_binary": np.asarray(y_train_binary, dtype=int),
            "month": X_train["month"].to_numpy() if "month" in X_train else pd.to_datetime(X_train[args.date_column]).dt.month.to_numpy(),
        },
        index=X_train.index,
    )
    for column in ("lat_rounded", "lon_rounded", "ecoregion_name", "ecoregion_realm"):
        if column in X_train.columns:
            train_reference[column] = X_train[column].to_numpy()

    yearly_rows: List[Dict[str, Any]] = []
    for (split_name, year), group in eval_predictions.groupby(["split", "year"], sort=True):
        yearly_rows.append(
            probability_metrics_for_year(
                split_name=split_name,
                year=int(year),
                y_true=group["target_binary"].to_numpy(dtype=int),
                y_prob=group["pred_probability"].to_numpy(dtype=float),
                threshold=threshold,
            )
        )
    add_historical_prior_metrics(
        yearly_rows=yearly_rows,
        train_reference=train_reference,
        eval_predictions=eval_predictions,
        alpha=cli_args.historical_prior_alpha,
    )
    yearly_metrics = pd.DataFrame(yearly_rows)

    importance = load_importance_weights(run_dir, list(X_train_model.columns))
    feature_shift_detail, feature_shift_summary = feature_shift_diagnostics(
        X_train_model=X_train_model,
        y_train_binary=y_train_binary,
        X_eval_model=eval_model,
        eval_predictions=eval_predictions,
        cat_features=prediction_cat_features,
        importance=importance,
        top_features=cli_args.top_features,
        bins=cli_args.psi_bins,
    )
    composition_shift = pd.DataFrame(composition_shift_rows(train_reference, eval_predictions))
    yearly = merge_yearly_rows(yearly_metrics, feature_shift_summary, composition_shift)

    split_labels = pd.Series(index=X.index, data="unused", dtype=object)
    split_labels.loc[X_train.index] = "train"
    split_labels.loc[X_validation.index] = "validation"
    split_labels.loc[X_test.index] = "test"
    target_summary = target_summary_all_years(X, y_binary, split_labels, args.date_column)

    error_slices = error_slice_rows(eval_predictions)
    top_shift = feature_shift_detail.sort_values(
        ["year", "weighted_psi_all_vs_train"],
        ascending=[True, False],
    )

    yearly.to_csv(output_dir / "year_difficulty_metrics.csv", index=False)
    feature_shift_detail.to_csv(output_dir / "feature_shift_by_year_feature.csv", index=False)
    feature_shift_summary.to_csv(output_dir / "feature_shift_by_year_summary.csv", index=False)
    composition_shift.to_csv(output_dir / "composition_shift_by_year.csv", index=False)
    target_summary.to_csv(output_dir / "target_summary_all_years.csv", index=False)
    error_slices.to_csv(output_dir / "top_error_slices.csv", index=False)
    top_shift.groupby(["split", "year"], sort=True).head(15).to_csv(
        output_dir / "top_feature_shifts_by_year.csv",
        index=False,
    )
    write_json(
        output_dir / "year_difficulty_metadata.json",
        {
            "config": cli_args.config,
            "run_dir": run_dir,
            "model_path": model_path,
            "output_dir": output_dir,
            "selected_features_path": selected_features_path,
            "threshold": threshold,
            "top_features_for_shift": cli_args.top_features,
            "psi_bins": cli_args.psi_bins,
            "historical_prior_alpha": cli_args.historical_prior_alpha,
        },
    )
    write_report(
        output_dir=output_dir,
        run_dir=run_dir,
        threshold=threshold,
        yearly=yearly,
        target_summary=target_summary,
        top_shift=top_shift.groupby(["split", "year"], sort=True).head(15),
        error_slices=error_slices,
    )
    print(f"Wrote year difficulty diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
