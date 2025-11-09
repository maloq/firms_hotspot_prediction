"""Evaluation CLI for logistic regression wildfire models.

Loads saved preprocessing and logistic regression artifacts, applies them to a
feature dataset, and reports global/regional classification metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import joblib
except ImportError:  # pragma: no cover - fallback when joblib unavailable
    joblib = None


def load_artifact(path: Path) -> Any:
    if joblib is not None:
        return joblib.load(path)
    with path.open("rb") as handle:  # pragma: no cover - fallback branch
        return pickle.load(handle)


@dataclass(frozen=True)
class Region:
    """Lat/lon bounded region with optional probability threshold override."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    threshold: float | None = None

    def validate(self) -> None:
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


@dataclass(frozen=True)
class Metrics:
    region: str
    support: int
    positives: int
    predicted_positives: int
    threshold: float
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    average_precision: float | None
    true_negatives: int | None
    false_positives: int | None
    false_negatives: int | None
    true_positives: int | None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "region": self.region,
            "support": self.support,
            "positives": self.positives,
            "predicted_positives": self.predicted_positives,
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
        }


def _parse_region_spec(spec: str) -> Region:
    parts = spec.split(":")
    if len(parts) not in {5, 6}:
        raise ValueError(
            "Region spec must follow 'name:lat_min:lat_max:lon_min:lon_max[:threshold]'"
        )
    name = parts[0]
    lat_min, lat_max, lon_min, lon_max = map(float, parts[1:5])
    threshold = float(parts[5]) if len(parts) == 6 else None
    region = Region(name=name, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, threshold=threshold)
    region.validate()
    return region


def _load_regions_from_file(path: Path) -> list[Region]:
    data = yaml.safe_load(path.read_text())
    if data is None:
        raise ValueError(f"Regions file '{path}' is empty")
    if isinstance(data, dict) and "regions" in data:
        region_entries = data["regions"]
    elif isinstance(data, list):
        region_entries = data
    else:
        region_entries = [data]
    regions: list[Region] = []
    for entry in region_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Unsupported region entry: {entry!r}")
        region = Region(
            name=str(entry["name"]),
            lat_min=float(entry["lat_min"]),
            lat_max=float(entry["lat_max"]),
            lon_min=float(entry["lon_min"]),
            lon_max=float(entry["lon_max"]),
            threshold=float(entry["threshold"]) if "threshold" in entry else None,
        )
        region.validate()
        regions.append(region)
    return regions


def _prepare_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    if "population" in df.columns:
        df["population"] = df["population"].fillna(0).astype(int)
    cat_features: Sequence[str] = config.get("cat_features", []) or []
    numerical_cat_features = set(config.get("numerical_cat_features", []) or [])
    for col in cat_features:
        if col not in df.columns:
            continue
        if col in numerical_cat_features:
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = df[col].fillna("missing").astype(str)
    return df


def _safe_binary_metric(func, y_true, y_pred) -> float | None:
    try:
        return float(func(y_true, y_pred))
    except ValueError:
        return None


def _compute_metrics(
    region_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Metrics:
    support = int(len(y_true))
    if support == 0:
        return Metrics(
            region=region_name,
            support=0,
            positives=0,
            predicted_positives=0,
            threshold=threshold,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            roc_auc=None,
            average_precision=None,
            true_negatives=None,
            false_positives=None,
            false_negatives=None,
            true_positives=None,
        )

    y_pred = (y_prob >= threshold).astype(int)
    positives = int(y_true.sum())
    predicted_positives = int(y_pred.sum())

    accuracy = _safe_binary_metric(accuracy_score, y_true, y_pred)
    precision = _safe_binary_metric(lambda yt, yp: precision_score(yt, yp, zero_division=0), y_true, y_pred)
    recall = _safe_binary_metric(lambda yt, yp: recall_score(yt, yp, zero_division=0), y_true, y_pred)
    f1 = _safe_binary_metric(lambda yt, yp: f1_score(yt, yp, zero_division=0), y_true, y_pred)

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = None

    try:
        avg_precision = float(average_precision_score(y_true, y_prob))
    except ValueError:
        avg_precision = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return Metrics(
        region=region_name,
        support=support,
        positives=positives,
        predicted_positives=predicted_positives,
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        average_precision=avg_precision,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
    )


def _load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _resolve_path(candidates: Sequence[str | Path | None], label: str) -> Path:
    missing: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
        missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            f"None of the provided {label} paths exist: {missing}. "
            f"Override via CLI or update the configuration."
        )
    raise ValueError(f"No {label} specified. Provide a CLI argument or configuration entry.")


def _first_non_null(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _load_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_saved_target_encoding(
    features: pd.DataFrame,
    target_encoding: Dict[str, Any] | None,
) -> pd.DataFrame:
    if not target_encoding:
        return features
    encoded = features.copy()
    encoding_features = target_encoding.get("features", {}) if isinstance(target_encoding, dict) else {}
    for original_col, info in encoding_features.items():
        encoded_column = info.get("encoded_column")
        if not encoded_column:
            continue
        mapping = info.get("mapping", {})
        global_mean = float(info.get("global_mean", 0.0))
        if original_col in encoded.columns:
            column_as_str = encoded[original_col].astype(str)
            mapped = column_as_str.map(mapping)
            encoded_values = pd.to_numeric(mapped, errors="coerce").fillna(global_mean)
        else:
            encoded_values = pd.Series(global_mean, index=encoded.index)
        encoded[encoded_column] = encoded_values
        encoded = encoded.drop(columns=[original_col], errors="ignore")
    return encoded


def _ensure_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    low_cardinality: Iterable[str],
) -> pd.DataFrame:
    result = df.copy()
    low_card_set = set(low_cardinality)
    for column in required_columns:
        if column in result.columns:
            continue
        if column in low_card_set:
            result[column] = "missing"
        else:
            result[column] = np.nan
    return result[required_columns]


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    metadata_path = _resolve_path([args.metadata_path], label="metadata")
    metadata = _load_metadata(metadata_path)

    config_path = _resolve_path(
        [
            args.config_path,
            metadata.get("config_path"),
            Path("configs/features_config_30d.yaml"),
        ],
        label="config file",
    )
    config = _load_config(config_path)
    evaluation_section = config.get("evaluation")
    eval_cfg = evaluation_section if isinstance(evaluation_section, dict) else {}

    dataset_path = _resolve_path(
        [
            args.features_path,
            eval_cfg.get("features_path") if isinstance(eval_cfg, dict) else None,
            metadata.get("data_path"),
            config.get("prepared_prediction_features_path"),
            config.get("features_path"),
        ],
        label="feature dataset",
    )

    artifacts = metadata.get("artifacts", {}) if isinstance(metadata.get("artifacts"), dict) else {}
    output_dir = artifacts.get("output_dir")
    model_filename = artifacts.get("model_filename")
    preprocessor_filename = artifacts.get("preprocessor_filename")

    model_candidates: list[str | Path | None] = [
        args.model_path,
        artifacts.get("model_path"),
    ]
    if output_dir and model_filename:
        model_candidates.append(Path(output_dir) / model_filename)
    if metadata_path.parent and model_filename:
        model_candidates.append(metadata_path.parent / model_filename)
    model_candidates.append(Path("models/log_regression/log_regression_model.joblib"))
    model_path = _resolve_path(model_candidates, "logistic regression model")

    preprocessor_candidates: list[str | Path | None] = [
        args.preprocessor_path,
        artifacts.get("preprocessor_path"),
    ]
    if output_dir and preprocessor_filename:
        preprocessor_candidates.append(Path(output_dir) / preprocessor_filename)
    if metadata_path.parent and preprocessor_filename:
        preprocessor_candidates.append(metadata_path.parent / preprocessor_filename)
    preprocessor_candidates.append(Path("models/log_regression/preprocessor.joblib"))
    preprocessor_path = _resolve_path(preprocessor_candidates, "preprocessor")

    df = _load_dataset(dataset_path)
    df_prepared = _prepare_features(df, config)

    target_column = _first_non_null(
        args.target_column,
        metadata.get("target_column"),
        eval_cfg.get("target_column") if isinstance(eval_cfg, dict) else None,
        config.get("target_column"),
        config.get("target_col"),
        default="count",
    )
    if target_column not in df_prepared.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset")

    positive_threshold = float(
        _first_non_null(
            args.target_positive_threshold,
            metadata.get("positive_threshold"),
            eval_cfg.get("target_positive_threshold") if isinstance(eval_cfg, dict) else None,
            config.get("target_positive_threshold"),
            default=0.0,
        )
    )
    y_true_series = (df_prepared[target_column] > positive_threshold).astype(int)
    y_true = y_true_series.to_numpy()

    features = df_prepared.drop(columns=[target_column])
    features = _apply_saved_target_encoding(features, metadata.get("target_encoding"))

    ignored_features = metadata.get("ignored_features", [])
    if ignored_features:
        features = features.drop(columns=list(ignored_features), errors="ignore")

    feature_columns = metadata.get("features", {}).get("model_input_columns")
    if not feature_columns:
        raise KeyError("Metadata missing 'model_input_columns'; cannot align features with model inputs.")

    low_cardinality = metadata.get("features", {}).get("low_cardinality", [])
    features = _ensure_required_columns(features, feature_columns, low_cardinality)

    # Align column order strictly
    features = features.reindex(columns=feature_columns)

    preprocessor = load_artifact(preprocessor_path)
    model = load_artifact(model_path)

    transformed = preprocessor.transform(features)
    probabilities = model.predict_proba(transformed)[:, 1]

    default_threshold = float(
        _first_non_null(
            args.threshold,
            metadata.get("default_threshold"),
            eval_cfg.get("threshold") if isinstance(eval_cfg, dict) else None,
            config.get("threshold"),
            0.5,
        )
    )

    lat_col = _first_non_null(
        args.lat_column,
        eval_cfg.get("lat_column") if isinstance(eval_cfg, dict) else None,
        config.get("lat_column"),
        default="lat_rounded",
    )
    lon_col = _first_non_null(
        args.lon_column,
        eval_cfg.get("lon_column") if isinstance(eval_cfg, dict) else None,
        config.get("lon_column"),
        default="lon_rounded",
    )
    if lat_col not in df_prepared.columns or lon_col not in df_prepared.columns:
        raise KeyError(f"Latitude column '{lat_col}' or longitude column '{lon_col}' not found in dataset")

    results: list[Metrics] = []
    global_metrics = _compute_metrics("GLOBAL", y_true, probabilities, default_threshold)
    results.append(global_metrics)

    df_with_probs = df_prepared[[lat_col, lon_col]].copy()
    df_with_probs[target_column] = y_true
    df_with_probs["pred_proba"] = probabilities

    regions: list[Region] = []
    if args.regions_file:
        regions.extend(_load_regions_from_file(Path(args.regions_file)))
    if args.region:
        regions.extend(_parse_region_spec(spec) for spec in args.region)
    if not regions and isinstance(eval_cfg, dict):
        cfg_regions_path = eval_cfg.get("regions_file")
        if cfg_regions_path:
            regions.extend(_load_regions_from_file(Path(cfg_regions_path)))
    if not regions and isinstance(eval_cfg, dict):
        cfg_regions_entries = eval_cfg.get("regions")
        if cfg_regions_entries:
            regions.extend(
                Region(
                    name=str(entry["name"]),
                    lat_min=float(entry["lat_min"]),
                    lat_max=float(entry["lat_max"]),
                    lon_min=float(entry["lon_min"]),
                    lon_max=float(entry["lon_max"]),
                    threshold=float(entry["threshold"]) if "threshold" in entry else None,
                )
                for entry in cfg_regions_entries
            )

    for region in regions:
        mask = (
            (df_with_probs[lat_col] >= region.lat_min)
            & (df_with_probs[lat_col] <= region.lat_max)
            & (df_with_probs[lon_col] >= region.lon_min)
            & (df_with_probs[lon_col] <= region.lon_max)
        )
        region_threshold = region.threshold if region.threshold is not None else default_threshold
        metrics = _compute_metrics(
            region_name=region.name,
            y_true=df_with_probs[target_column].to_numpy()[mask],
            y_prob=df_with_probs["pred_proba"].to_numpy()[mask],
            threshold=region_threshold,
        )
        results.append(metrics)

    if len(results) == 1 and not regions:
        print("No regions provided; only GLOBAL metrics are returned.")

    result_df = pd.DataFrame([m.as_dict() for m in results]).set_index("region")

    if args.output_path:
        output_path = Path(args.output_path)
        if output_path.suffix == ".json":
            output_path.write_text(result_df.to_json(orient="index", indent=2))
        else:
            result_df.to_csv(output_path)

    return result_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", help="Path to feature dataset (parquet or csv)")
    parser.add_argument("--model-path", help="Path to logistic regression model (.joblib)")
    parser.add_argument("--preprocessor-path", help="Path to fitted preprocessor (.joblib)")
    parser.add_argument("--metadata-path", default="models/log_regression/metrics.json", help="Path to metadata produced during training")
    parser.add_argument("--config-path", help="Feature/config YAML used during training")
    parser.add_argument("--regions-file", help="YAML file describing evaluation regions")
    parser.add_argument(
        "--region",
        action="append",
        help="Inline region spec 'name:lat_min:lat_max:lon_min:lon_max[:threshold]' (repeatable)",
    )
    parser.add_argument("--lat-column", help="Latitude column name")
    parser.add_argument("--lon-column", help="Longitude column name")
    parser.add_argument("--target-column", help="Target column name in dataset")
    parser.add_argument(
        "--target-positive-threshold",
        type=float,
        help="Values strictly greater than this threshold are treated as positives",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Default probability threshold used for binarisation",
    )
    parser.add_argument("--output-path", help="Optional path to save metrics (csv or json)")
    parser.add_argument(
        "--quiet",
        dest="do_print",
        action="store_false",
        help="Suppress printing the metrics table to stdout",
    )
    parser.set_defaults(do_print=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result_df = evaluate(args)
    if args.do_print:
        with pd.option_context("display.max_columns", None, "display.max_rows", None):
            print(result_df)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
