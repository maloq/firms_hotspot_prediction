from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


EPS = 1e-12


def _as_arrays(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    if sample_weight is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
    if not (len(y) == len(p) == len(w)):
        raise ValueError("y_true, y_prob, and sample_weight must have the same length.")
    finite = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    return y[finite].astype(int), np.clip(p[finite], EPS, 1.0 - EPS), w[finite]


def weighted_brier_score(y_true, y_prob, sample_weight=None) -> float | None:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0:
        return None
    return float(np.average((p - y) ** 2, weights=w))


def weighted_log_loss(y_true, y_prob, sample_weight=None) -> float | None:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0:
        return None
    try:
        return float(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    except Exception:
        return None


def weighted_prevalence(y_true, sample_weight=None) -> float | None:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    if sample_weight is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not finite.any():
        return None
    return float(np.average(y[finite], weights=w[finite]))


def expected_observed_count_ratio(y_true, y_prob, sample_weight=None) -> dict[str, float | None]:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0:
        return {
            "expected_fire_positive_grid_cells": None,
            "observed_fire_positive_grid_cells": None,
            "expected_observed_count_ratio": None,
        }
    expected = float(np.sum(p * w))
    observed = float(np.sum(y * w))
    ratio = expected / observed if observed > 0 else None
    return {
        "expected_fire_positive_grid_cells": expected,
        "observed_fire_positive_grid_cells": observed,
        "expected_observed_count_ratio": ratio,
    }


def _logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS)
    return np.log(prob / (1.0 - prob))


def calibration_slope_intercept(y_true, y_prob, sample_weight=None) -> dict[str, float | None]:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0 or np.unique(y).size < 2:
        return {"calibration_intercept": None, "calibration_slope": None}
    x = _logit(p).reshape(-1, 1)
    try:
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(x, y, sample_weight=w)
    except TypeError:
        model = LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(x, y, sample_weight=w)
    except Exception:
        return {"calibration_intercept": None, "calibration_slope": None}
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
    }


def weighted_average_precision(y_true, y_prob, sample_weight=None) -> float | None:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    try:
        return float(average_precision_score(y, p, sample_weight=w))
    except Exception:
        return None


def weighted_roc_auc(y_true, y_prob, sample_weight=None) -> float | None:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    try:
        return float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        return None


def weighted_binary_classification_metrics(
    y_true,
    y_prob,
    sample_weight=None,
    *,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0:
        return {
            "threshold": float(threshold),
            "precision": None,
            "recall": None,
            "f1": None,
            "predicted_positive_grid_cells": None,
        }
    pred = p >= float(threshold)
    tp = float(np.sum(w[pred & (y == 1)]))
    fp = float(np.sum(w[pred & (y == 0)]))
    fn = float(np.sum(w[(~pred) & (y == 1)]))
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_grid_cells": float(tp + fp),
    }


def max_weighted_f1(y_true, y_prob, sample_weight=None) -> dict[str, float | None]:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0 or np.unique(y).size < 2:
        return {
            "max_f1": None,
            "precision_at_max_f1": None,
            "recall_at_max_f1": None,
            "threshold_at_max_f1": None,
            "predicted_positive_grid_cells_at_max_f1": None,
        }
    order = np.argsort(-p, kind="mergesort")
    y_s = y[order]
    p_s = p[order]
    w_s = w[order]
    tp_cum = np.cumsum(w_s * (y_s == 1))
    pred_cum = np.cumsum(w_s)
    total_pos = float(np.sum(w_s * (y_s == 1)))
    if total_pos <= 0:
        return {
            "max_f1": None,
            "precision_at_max_f1": None,
            "recall_at_max_f1": None,
            "threshold_at_max_f1": None,
            "predicted_positive_grid_cells_at_max_f1": None,
        }
    threshold_ends = np.r_[np.flatnonzero(p_s[:-1] != p_s[1:]), len(p_s) - 1]
    tp = tp_cum[threshold_ends]
    predicted = pred_cum[threshold_ends]
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    recall = tp / total_pos
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    best_idx = int(np.nanargmax(f1))
    best_end = int(threshold_ends[best_idx])
    return {
        "max_f1": float(f1[best_idx]),
        "precision_at_max_f1": float(precision[best_idx]),
        "recall_at_max_f1": float(recall[best_idx]),
        "threshold_at_max_f1": float(p_s[best_end]),
        "predicted_positive_grid_cells_at_max_f1": float(predicted[best_idx]),
    }


def _weighted_quantile_bins(prob: np.ndarray, weight: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(prob)
    sorted_w = weight[order]
    cumulative = np.cumsum(sorted_w)
    total = cumulative[-1]
    targets = np.linspace(0.0, total, n_bins + 1)
    bins_sorted = np.searchsorted(cumulative, targets[1:-1], side="right")
    out = np.zeros(len(prob), dtype=int)
    start = 0
    for bin_idx, stop in enumerate(list(bins_sorted) + [len(prob)]):
        out[order[start:stop]] = bin_idx
        start = stop
    return out


def make_reliability_bins(
    y_true,
    y_prob,
    sample_weight=None,
    n_bins: int = 20,
    strategy: str = "equal_count",
) -> pd.DataFrame:
    y, p, w = _as_arrays(y_true, y_prob, sample_weight)
    if len(y) == 0:
        return pd.DataFrame(
            columns=[
                "bin",
                "n_unweighted",
                "n_weighted",
                "prob_min",
                "prob_max",
                "mean_predicted_probability",
                "observed_prevalence",
                "expected_fire_positive_grid_cells",
                "observed_fire_positive_grid_cells",
            ]
        )

    n_bins = max(1, int(n_bins))
    strategy_key = str(strategy or "equal_count").lower()
    if strategy_key == "equal_count":
        bin_id = _weighted_quantile_bins(p, w, n_bins)
    elif strategy_key in {"equal_width", "uniform"}:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    elif strategy_key in {"log", "log_spaced", "log-spaced"}:
        edges = np.r_[0.0, np.geomspace(1e-6, 1.0, n_bins)]
        bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown reliability binning strategy: {strategy}")

    rows: list[dict[str, Any]] = []
    for idx in range(n_bins):
        mask = bin_id == idx
        if not mask.any():
            continue
        y_b = y[mask]
        p_b = p[mask]
        w_b = w[mask]
        expected = float(np.sum(p_b * w_b))
        observed = float(np.sum(y_b * w_b))
        rows.append(
            {
                "bin": idx,
                "n_unweighted": int(mask.sum()),
                "n_weighted": float(w_b.sum()),
                "prob_min": float(p_b.min()),
                "prob_max": float(p_b.max()),
                "mean_predicted_probability": float(np.average(p_b, weights=w_b)),
                "observed_prevalence": float(np.average(y_b, weights=w_b)),
                "expected_fire_positive_grid_cells": expected,
                "observed_fire_positive_grid_cells": observed,
                "expected_observed_count_ratio": expected / observed if observed > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def daily_expected_observed_mae(
    frame: pd.DataFrame,
    *,
    date_col: str = "datetime",
    target_col: str = "is_fire",
    prob_col: str = "prob_calibrated",
    weight_col: str = "eval_weight",
) -> float | None:
    if frame.empty or date_col not in frame.columns:
        return None
    work = frame[[date_col, target_col, prob_col, weight_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.date
    work["expected"] = pd.to_numeric(work[prob_col], errors="coerce") * pd.to_numeric(
        work[weight_col], errors="coerce"
    )
    work["observed"] = pd.to_numeric(work[target_col], errors="coerce") * pd.to_numeric(
        work[weight_col], errors="coerce"
    )
    grouped = work.groupby(date_col, observed=True)[["expected", "observed"]].sum()
    if grouped.empty:
        return None
    return float(np.mean(np.abs(grouped["expected"] - grouped["observed"])))
