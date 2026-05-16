from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "val": 1,
    "test": 2,
}

SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


@dataclass(frozen=True)
class Region:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    threshold: float | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Region name cannot be empty")
        if math.isnan(self.lat_min) or math.isnan(self.lat_max):
            raise ValueError(f"Region '{self.name}' has NaN latitude bounds")
        if math.isnan(self.lon_min) or math.isnan(self.lon_max):
            raise ValueError(f"Region '{self.name}' has NaN longitude bounds")
        if self.lat_min > self.lat_max:
            raise ValueError(f"Region '{self.name}' has lat_min > lat_max")
        if self.lon_min > self.lon_max:
            raise ValueError(f"Region '{self.name}' has lon_min > lon_max")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Region '{self.name}' threshold must be within [0, 1]")

    def mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return (
            (lat >= self.lat_min)
            & (lat <= self.lat_max)
            & (lon >= self.lon_min)
            & (lon <= self.lon_max)
        )


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def load_regions(path: Path | None) -> list[Region]:
    if path is None:
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Regions file '{path}' is empty")
    if isinstance(payload, Mapping) and "regions" in payload:
        entries = payload["regions"]
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = [payload]
    regions: list[Region] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Unsupported region entry: {entry!r}")
        region = Region(
            name=str(entry["name"]),
            lat_min=float(entry["lat_min"]),
            lat_max=float(entry["lat_max"]),
            lon_min=float(entry["lon_min"]),
            lon_max=float(entry["lon_max"]),
            threshold=float(entry["threshold"]) if entry.get("threshold") is not None else None,
        )
        region.validate()
        regions.append(region)
    return regions


def _timestamps_from_prepared_dates(values: np.ndarray) -> pd.DatetimeIndex:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)
    if np.issubdtype(values.dtype, np.integer):
        return pd.to_datetime(values, unit="D", origin="unix")
    return pd.to_datetime(values, errors="coerce")


def month_to_season(month: pd.Series | np.ndarray) -> np.ndarray:
    months = pd.Series(month).astype("Int64")
    return months.map(lambda value: SEASON_BY_MONTH.get(int(value), "unknown") if not pd.isna(value) else "unknown").to_numpy(dtype=object)


def assign_regions(lat: np.ndarray, lon: np.ndarray, regions: Sequence[Region]) -> np.ndarray:
    labels = np.full(len(lat), "other", dtype=object)
    assigned = np.zeros(len(lat), dtype=bool)
    for region in regions:
        mask = (~assigned) & region.mask(lat, lon)
        labels[mask] = region.name
        assigned[mask] = True
    return labels


def read_split_context(data_path: Path, split_name: str, regions: Sequence[Region]) -> pd.DataFrame:
    split_code = SPLIT_CODES.get(split_name)
    if split_code is None:
        raise ValueError(f"Unsupported split '{split_name}'. Expected one of {sorted(SPLIT_CODES)}.")
    with np.load(data_path, allow_pickle=False) as data:
        split = np.asarray(data["split"])
        mask = split == split_code
        lat = np.asarray(data["lat"])[mask].astype(float)
        lon = np.asarray(data["lon"])[mask].astype(float)
        if "dates" in data.files:
            dates = _timestamps_from_prepared_dates(np.asarray(data["dates"])[mask])
        elif "years" in data.files:
            dates = pd.to_datetime(np.asarray(data["years"])[mask].astype(str) + "-01-01")
        else:
            dates = pd.DatetimeIndex([pd.NaT] * int(mask.sum()))
        if "years" in data.files:
            years = np.asarray(data["years"])[mask].astype(int)
        else:
            years = dates.year.to_numpy(dtype=int)

    month = dates.month.to_numpy(dtype=int)
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "year": years,
            "month": month,
            "season": month_to_season(month),
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
        }
    )
    frame["region"] = assign_regions(frame["lat"].to_numpy(), frame["lon"].to_numpy(), regions)
    return frame


def attach_context(
    predictions: pd.DataFrame,
    context: pd.DataFrame,
    *,
    split_name: str,
) -> pd.DataFrame:
    if len(predictions) != len(context):
        raise ValueError(
            f"{split_name} prediction/context row mismatch: "
            f"{len(predictions)} predictions vs {len(context)} prepared rows"
        )
    frame = predictions.reset_index(drop=True).copy()
    for column in ["datetime", "year", "month", "season", "lat", "lon", "region"]:
        frame[column] = context[column].to_numpy()
    if "split_name" not in frame.columns:
        frame["split_name"] = split_name
    return frame


def probability_column(frame: pd.DataFrame) -> str:
    for column in ["prob_raw", "pred_proba", "probability", "y_prob"]:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not find a probability column in {list(frame.columns)}")


def label_column(frame: pd.DataFrame) -> str:
    for column in ["is_fire", "target_binary", "y_true", "count"]:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not find a binary label column in {list(frame.columns)}")


def group_values(frame: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "global":
        return pd.Series("global", index=frame.index, dtype="object")
    if strategy == "season":
        return frame["season"].astype(str)
    if strategy == "region":
        return frame["region"].astype(str)
    if strategy == "region_season":
        return frame["region"].astype(str) + "|" + frame["season"].astype(str)
    raise ValueError(
        f"Unsupported threshold strategy '{strategy}'. "
        "Use one of: global, season, region, region_season."
    )


def choose_threshold_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    support = int(len(y_true))
    positives = int(y_true.sum())
    negatives = int(support - positives)
    base: dict[str, Any] = {
        "support": support,
        "positives": positives,
        "negatives": negatives,
    }
    if support == 0:
        return {**base, "threshold": 0.5, "validation_f1": None, "reason": "empty"}
    if positives == 0 or negatives == 0:
        return {**base, "threshold": 0.5, "validation_f1": None, "reason": "only_one_class"}

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return {**base, "threshold": 0.5, "validation_f1": None, "reason": "no_thresholds"}

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
    candidate = np.isfinite(f1)
    if min_precision is not None:
        candidate &= precision >= float(min_precision)
    if min_recall is not None:
        candidate &= recall >= float(min_recall)
    if not candidate.any():
        return {
            **base,
            "threshold": 0.5,
            "validation_f1": None,
            "reason": "constraints_unsatisfied",
        }

    candidate_idx = np.flatnonzero(candidate)
    best = int(candidate_idx[np.nanargmax(f1[candidate])])
    return {
        **base,
        "threshold": float(thresholds[best]),
        "validation_f1": float(f1[best]),
        "validation_precision": float(precision[best]),
        "validation_recall": float(recall[best]),
        "validation_predicted_positives": int((y_prob >= thresholds[best]).sum()),
        "reason": "validation_f1_max",
    }


def fit_threshold_policy(
    validation_frame: pd.DataFrame,
    *,
    strategy: str,
    min_val_rows: int = 1_000,
    min_val_positives: int = 20,
    min_val_negatives: int = 20,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> dict[str, Any]:
    prob_col = probability_column(validation_frame)
    y_col = label_column(validation_frame)
    global_info = choose_threshold_f1(
        validation_frame[y_col].to_numpy(),
        validation_frame[prob_col].to_numpy(),
        min_precision=min_precision,
        min_recall=min_recall,
    )
    if global_info.get("validation_f1") is None:
        global_info = {
            **global_info,
            "threshold": 0.5,
            "reason": f"{global_info.get('reason', 'unknown')}_global_default",
        }

    policy: dict[str, Any] = {
        "strategy": strategy,
        "global_threshold": float(global_info["threshold"]),
        "global": global_info,
        "groups": {},
        "min_val_rows": int(min_val_rows),
        "min_val_positives": int(min_val_positives),
        "min_val_negatives": int(min_val_negatives),
        "min_precision": min_precision,
        "min_recall": min_recall,
        "fallback": "global_validation_f1_threshold",
    }
    group_series = group_values(validation_frame, strategy)
    for group in sorted(group_series.unique()):
        mask = group_series == group
        y_true = validation_frame.loc[mask, y_col].to_numpy()
        y_prob = validation_frame.loc[mask, prob_col].to_numpy()
        support = int(len(y_true))
        positives = int(y_true.sum())
        negatives = int(support - positives)
        if strategy == "global":
            info = dict(global_info)
        elif support < min_val_rows:
            info = {
                "threshold": float(global_info["threshold"]),
                "validation_f1": None,
                "support": support,
                "positives": positives,
                "negatives": negatives,
                "reason": "fallback_min_val_rows",
            }
        elif positives < min_val_positives:
            info = {
                "threshold": float(global_info["threshold"]),
                "validation_f1": None,
                "support": support,
                "positives": positives,
                "negatives": negatives,
                "reason": "fallback_min_val_positives",
            }
        elif negatives < min_val_negatives:
            info = {
                "threshold": float(global_info["threshold"]),
                "validation_f1": None,
                "support": support,
                "positives": positives,
                "negatives": negatives,
                "reason": "fallback_min_val_negatives",
            }
        else:
            info = choose_threshold_f1(
                y_true,
                y_prob,
                min_precision=min_precision,
                min_recall=min_recall,
            )
            if info.get("validation_f1") is None:
                info = {
                    **info,
                    "threshold": float(global_info["threshold"]),
                    "reason": f"fallback_{info.get('reason', 'unknown')}",
                }
        if info["threshold"] == global_info["threshold"] and group != "global":
            info["fallback_to"] = "global"
        policy["groups"][str(group)] = info
    return policy


def thresholds_for_frame(frame: pd.DataFrame, policy: Mapping[str, Any]) -> np.ndarray:
    strategy = str(policy["strategy"])
    global_threshold = float(policy["global_threshold"])
    groups = policy.get("groups") or {}
    values = group_values(frame, strategy).astype(str)
    thresholds = np.empty(len(frame), dtype=np.float32)
    for group in values.unique():
        info = groups.get(str(group)) or {}
        threshold = float(info.get("threshold", global_threshold))
        thresholds[values == group] = threshold
    return thresholds


def safe_metric(func, *args: Any) -> float | None:
    try:
        value = float(func(*args))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    threshold_array = np.asarray(threshold, dtype=float)
    if threshold_array.ndim == 0:
        threshold_value: float | None = float(threshold_array)
        y_pred = (y_prob >= threshold_value).astype(int)
    else:
        if len(threshold_array) != len(y_prob):
            raise ValueError("Threshold array length must match prediction length")
        threshold_value = None
        y_pred = (y_prob >= threshold_array).astype(int)
    support = int(len(y_true))
    positives = int(y_true.sum()) if support else 0
    predicted_positives = int(y_pred.sum()) if support else 0
    has_both_classes = bool(support and positives > 0 and positives < support)
    return {
        "support": support,
        "positives": positives,
        "negatives": int(support - positives),
        "predicted_positives": predicted_positives,
        "positive_rate": float(positives / support) if support else None,
        "predicted_positive_rate": float(predicted_positives / support) if support else None,
        "threshold": threshold_value,
        "precision": safe_metric(lambda yt, yp: precision_score(yt, yp, zero_division=0), y_true, y_pred),
        "recall": safe_metric(lambda yt, yp: recall_score(yt, yp, zero_division=0), y_true, y_pred),
        "f1": safe_metric(lambda yt, yp: f1_score(yt, yp, zero_division=0), y_true, y_pred),
        "average_precision": safe_metric(average_precision_score, y_true, y_prob) if has_both_classes else None,
        "roc_auc": safe_metric(roc_auc_score, y_true, y_prob) if has_both_classes else None,
    }


def apply_threshold_policy(frame: pd.DataFrame, policy: Mapping[str, Any]) -> pd.DataFrame:
    prob_col = probability_column(frame)
    out = frame.copy()
    out["threshold_group"] = group_values(out, str(policy["strategy"])).to_numpy()
    out["threshold"] = thresholds_for_frame(out, policy)
    out["pred_binary"] = (out[prob_col].to_numpy() >= out["threshold"].to_numpy()).astype(np.int8)
    return out


def evaluate_threshold_policy(
    frame: pd.DataFrame,
    policy: Mapping[str, Any],
    *,
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    prob_col = probability_column(frame)
    y_col = label_column(frame)
    thresholds = thresholds_for_frame(frame, policy)
    overall = compute_metrics(frame[y_col].to_numpy(), frame[prob_col].to_numpy(), thresholds)
    overall.update(
        {
            "split": split_name,
            "strategy": str(policy["strategy"]),
            "group": "all",
            "threshold": None,
            "threshold_policy": "grouped",
        }
    )

    rows: list[dict[str, Any]] = []
    group_series = group_values(frame, str(policy["strategy"])).astype(str)
    for group in sorted(group_series.unique()):
        mask = group_series == group
        group_thresholds = thresholds[mask.to_numpy()]
        group_threshold = float(group_thresholds[0]) if len(group_thresholds) else float(policy["global_threshold"])
        row = compute_metrics(
            frame.loc[mask, y_col].to_numpy(),
            frame.loc[mask, prob_col].to_numpy(),
            group_threshold,
        )
        row.update(
            {
                "split": split_name,
                "strategy": str(policy["strategy"]),
                "group": str(group),
                "threshold_policy": "grouped",
            }
        )
        rows.append(row)
    return overall, pd.DataFrame(rows)


def load_metrics_payload(metrics_path: Path) -> dict[str, Any]:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def resolve_data_path(metrics_payload: Mapping[str, Any], metrics_path: Path, data_path: Path | None) -> Path:
    if data_path is not None:
        return data_path
    raw = metrics_payload.get("data_path")
    if raw is None:
        raise ValueError(f"{metrics_path} does not contain data_path; pass --data-path explicitly.")
    return Path(str(raw))


def load_prediction_frames(
    metrics_payload: Mapping[str, Any],
    data_path: Path,
    regions: Sequence[Region],
) -> dict[str, pd.DataFrame]:
    artifacts = metrics_payload.get("prediction_artifacts") or {}
    frames: dict[str, pd.DataFrame] = {}
    for split_name in ["validation", "test"]:
        if split_name not in artifacts:
            raise ValueError(f"metrics.json missing prediction_artifacts.{split_name}")
        predictions = pd.read_parquet(Path(str(artifacts[split_name])))
        context = read_split_context(data_path, split_name, regions)
        frames[split_name] = attach_context(predictions, context, split_name=split_name)
    return frames


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")

