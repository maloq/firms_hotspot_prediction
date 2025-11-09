from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import joblib
except ImportError:  # pragma: no cover - fallback for minimal environments
    joblib = None

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger(__name__)


def dump_artifact(obj: Any, path: Path) -> None:
    if joblib is not None:
        joblib.dump(obj, path)
    else:  # pragma: no cover - fallback for environments without joblib
        with path.open("wb") as handle:
            pickle.dump(obj, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic regression wildfire detector.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/features_config_30d.yaml"),
        help="Path to YAML config with feature metadata.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/saved_features_boost/train_test_features_30d_all_extended_north_14cb.parquet"),
        help="Path to parquet file with pre-computed features.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="count",
        help="Target column name in the features dataframe.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle data before splitting. Defaults to False for time-series behaviour.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for operations that rely on randomness.",
    )
    parser.add_argument(
        "--cardinality-threshold",
        type=int,
        default=60,
        help="Threshold for treating categorical columns as high-cardinality.",
    )
    parser.add_argument(
        "--target-smooth",
        type=float,
        default=10.0,
        help="Smoothing factor for target encoding high-cardinality categories.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="Threshold applied to the target to create binary labels.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=4.0,
        help="Relative weight for the positive class in logistic regression.",
    )
    parser.add_argument(
        "--ignored-features",
        type=str,
        nargs="*",
        default=("datetime", "day", "latitude", "longitude", "year"),
        help="Feature names excluded from modelling.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum iterations for logistic regression solver.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/log_regression"),
        help="Directory where trained model, preprocessor, and metrics will be stored.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="log_regression_model.joblib",
        help="Filename used when persisting the trained logistic regression model.",
    )
    parser.add_argument(
        "--preprocessor-name",
        type=str,
        default="preprocessor.joblib",
        help="Filename used when persisting the fitted preprocessing transformer.",
    )
    parser.add_argument(
        "--metrics-name",
        type=str,
        default="metrics.json",
        help="Filename used to store evaluation metrics.",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Set this flag to skip writing model artifacts to disk.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_features(data_path: Path) -> pd.DataFrame:
    start_time = perf_counter()
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows and %d columns from %s in %.2fs", len(df), df.shape[1], data_path, perf_counter() - start_time)
    return df


def prepare_dataframe(
    df: pd.DataFrame,
    config: Dict,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if "population" in df.columns:
        df["population"] = df["population"].fillna(0).astype(int)

    cat_features: Sequence[str] = config.get("cat_features", [])
    numerical_cat_features: Sequence[str] = config.get("numerical_cat_features", [])

    for col in cat_features:
        if col not in df.columns:
            continue
        if col in numerical_cat_features:
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = df[col].fillna("missing").astype(str)

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe.")

    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def split_categorical_by_cardinality(
    X_train: pd.DataFrame,
    categorical_features: Sequence[str],
    ignored_features: Iterable[str],
    threshold: int,
) -> Tuple[List[str], List[str]]:
    high_cardinality, low_cardinality = [], []
    ignored_set = set(ignored_features)
    for col in categorical_features:
        if col not in X_train.columns or col in ignored_set:
            continue
        unique_count = X_train[col].nunique(dropna=False)
        if unique_count > threshold:
            high_cardinality.append(col)
        else:
            low_cardinality.append(col)
        logger.debug("Feature %s has %d unique values.", col, unique_count)
    return high_cardinality, low_cardinality


def target_encode_single(
    train_col: pd.Series,
    test_col: pd.Series,
    target: pd.Series,
    smooth: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, float], float]:
    df_temp = pd.DataFrame({"feature": train_col, "target": target.values})
    global_mean = df_temp["target"].mean()
    stats = df_temp.groupby("feature")["target"].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)
    train_encoded = train_col.map(smoothed).fillna(global_mean)
    test_encoded = test_col.map(smoothed).fillna(global_mean)
    return train_encoded, test_encoded, smoothed.to_dict(), float(global_mean)


def apply_target_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_binary: pd.Series,
    high_cardinality_features: Sequence[str],
    smooth: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoding_maps: Dict[str, Dict[str, Any]] = {}
    for col in high_cardinality_features:
        encoded_train, encoded_test, mapping, global_mean = target_encode_single(X_train[col], X_test[col], y_train_binary, smooth)
        encoded_col_name = f"{col}_target_enc"
        X_train_encoded[encoded_col_name] = encoded_train
        X_test_encoded[encoded_col_name] = encoded_test
        X_train_encoded = X_train_encoded.drop(columns=[col])
        X_test_encoded = X_test_encoded.drop(columns=[col])
        encoding_maps[col] = {
            "mapping": {key: float(value) for key, value in mapping.items()},
            "global_mean": global_mean,
            "encoded_column": encoded_col_name,
        }
    return X_train_encoded, X_test_encoded, encoding_maps


def build_preprocessing_pipeline(
    numerical_features: Sequence[str],
    low_cardinality_features: Sequence[str],
) -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, list(numerical_features)),
            ("cat", categorical_pipeline, list(low_cardinality_features)),
        ]
    )


def train_logistic_regression(
    X_train_processed: np.ndarray,
    y_train_binary: pd.Series,
    positive_class_weight: float,
    max_iter: int,
    random_state: int,
) -> LogisticRegression:
    class_weight = {0: 1.0, 1: positive_class_weight}
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs",
    )
    start_time = perf_counter()
    model.fit(X_train_processed, y_train_binary)
    logger.info("Trained logistic regression in %.2fs", perf_counter() - start_time)
    return model


def safe_metric(name: str, func, *args) -> float:
    try:
        return func(*args)
    except ValueError as exc:
        logger.warning("Skipping %s metric: %s", name, exc)
        return float("nan")


def evaluate_split(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": safe_metric("roc_auc", roc_auc_score, y_true, y_proba),
        "average_precision": safe_metric("average_precision", average_precision_score, y_true, y_proba),
    }
    logger.info("%s metrics: accuracy=%.4f, roc_auc=%.4f, average_precision=%.4f", name, metrics["accuracy"], metrics["roc_auc"], metrics["average_precision"])
    logger.info("%s confusion matrix:\n%s", name, confusion_matrix(y_true, y_pred))
    logger.info("%s classification report:\n%s", name, classification_report(y_true, y_pred))
    return metrics


def normalize_metrics(metrics: Dict[str, float]) -> Dict[str, Optional[float]]:
    normalized: Dict[str, Optional[float]] = {}
    for key, value in metrics.items():
        if value is None:
            normalized[key] = None
            continue
        scalar = float(value)
        normalized[key] = None if math.isnan(scalar) else scalar
    return normalized


def class_balance_counts(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts().sort_index()
    return {str(int(idx)): int(count) for idx, count in counts.items()}


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return None if math.isnan(scalar) else scalar
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def build_metadata(
    args: argparse.Namespace,
    model: LogisticRegression,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    y_train_binary: pd.Series,
    y_test_binary: pd.Series,
    categorical_features: Sequence[str],
    high_cardinality: Sequence[str],
    low_cardinality: Sequence[str],
    numerical_features: Sequence[str],
    feature_columns: Sequence[str],
    target_encoding_maps: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "config_path": Path(args.config),
        "data_path": Path(args.data),
        "target_column": args.target_column,
        "positive_threshold": args.positive_threshold,
        "positive_class_weight": args.positive_class_weight,
        "cardinality_threshold": args.cardinality_threshold,
        "target_smooth": args.target_smooth,
        "ignored_features": list(args.ignored_features),
        "features": {
            "categorical": list(categorical_features),
            "high_cardinality": list(high_cardinality),
            "low_cardinality": list(low_cardinality),
            "numerical": list(numerical_features),
            "model_input_columns": list(feature_columns),
        },
        "dataset": {
            "train_rows": int(len(y_train_binary)),
            "test_rows": int(len(y_test_binary)),
            "train_class_balance": class_balance_counts(y_train_binary),
            "test_class_balance": class_balance_counts(y_test_binary),
        },
        "metrics": {
            "train": normalize_metrics(train_metrics),
            "test": normalize_metrics(test_metrics),
        },
        "target_encoding": {
            "smooth": args.target_smooth,
            "features": target_encoding_maps,
        },
        "model_params": {key: sanitize_for_json(value) for key, value in model.get_params().items()},
    }
    return metadata


def save_artifacts(
    output_dir: Path,
    model: LogisticRegression,
    preprocessor: ColumnTransformer,
    metadata: Dict[str, Any],
    model_filename: str,
    preprocessor_filename: str,
    metrics_filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_filename
    preprocessor_path = output_dir / preprocessor_filename
    metrics_path = output_dir / metrics_filename

    dump_artifact(model, model_path)
    dump_artifact(preprocessor, preprocessor_path)
    metadata.setdefault("artifacts", {})
    metadata["artifacts"].update(
        {
            "output_dir": output_dir,
            "model_filename": model_filename,
            "preprocessor_filename": preprocessor_filename,
            "metadata_filename": metrics_filename,
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "metadata_path": metrics_path,
        }
    )
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(sanitize_for_json(metadata), metrics_file, indent=2)

    logger.info(
        "Saved model artifacts: model=%s, preprocessor=%s, metrics=%s",
        model_path.resolve(),
        preprocessor_path.resolve(),
        metrics_path.resolve(),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Using configuration from %s", args.config.resolve())
    config = load_config(args.config)

    df = load_features(args.data)
    X, y = prepare_dataframe(df, config, args.target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        shuffle=args.shuffle,
        random_state=args.random_state if args.shuffle else None,
    )

    logger.info("Train rows: %d | Test rows: %d", len(X_train), len(X_test))
    if "datetime" in X_train.columns:
        logger.info("Train start year: %s", X_train["datetime"].min())
        logger.info("Test start year: %s", X_test["datetime"].min())

    y_train_binary = (y_train > args.positive_threshold).astype(int)
    y_test_binary = (y_test > args.positive_threshold).astype(int)
    logger.info("Train class balance:\n%s", y_train_binary.value_counts())
    logger.info("Test class balance:\n%s", y_test_binary.value_counts())

    categorical_features = config.get("cat_features", [])
    high_cardinality, low_cardinality = split_categorical_by_cardinality(
        X_train,
        categorical_features,
        args.ignored_features,
        args.cardinality_threshold,
    )
    logger.info("High-cardinality features (%d): %s", len(high_cardinality), high_cardinality)
    logger.info("Low-cardinality features (%d): %s", len(low_cardinality), low_cardinality)

    X_train_encoded, X_test_encoded, target_encoding_maps = apply_target_encoding(
        X_train,
        X_test,
        y_train_binary,
        high_cardinality,
        args.target_smooth,
    )

    X_train_encoded = X_train_encoded.drop(columns=list(args.ignored_features), errors="ignore")
    X_test_encoded = X_test_encoded.drop(columns=list(args.ignored_features), errors="ignore")

    feature_columns = X_train_encoded.columns.tolist()

    numerical_features = [
        col
        for col in X_train_encoded.columns
        if col not in low_cardinality and col not in args.ignored_features
    ]

    logger.info("Numerical features used: %d", len(numerical_features))
    logger.info("Low-cardinality categorical features used: %d", len(low_cardinality))

    preprocessor = build_preprocessing_pipeline(numerical_features, low_cardinality)

    preprocess_start = perf_counter()
    X_train_processed = preprocessor.fit_transform(X_train_encoded)
    X_test_processed = preprocessor.transform(X_test_encoded)
    logger.info("Preprocessing completed in %.2fs", perf_counter() - preprocess_start)
    logger.info("Processed train shape: %s | Processed test shape: %s", X_train_processed.shape, X_test_processed.shape)

    model = train_logistic_regression(
        X_train_processed,
        y_train_binary,
        args.positive_class_weight,
        args.max_iter,
        args.random_state,
    )

    train_pred = model.predict(X_train_processed)
    test_pred = model.predict(X_test_processed)
    train_proba = model.predict_proba(X_train_processed)[:, 1]
    test_proba = model.predict_proba(X_test_processed)[:, 1]

    train_metrics = evaluate_split("Train", y_train_binary, train_pred, train_proba)
    test_metrics = evaluate_split("Test", y_test_binary, test_pred, test_proba)

    if args.skip_save:
        logger.info("Skipping artifact persistence because --skip-save flag is set.")
        return

    metadata = build_metadata(
        args=args,
        model=model,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        y_train_binary=y_train_binary,
        y_test_binary=y_test_binary,
        categorical_features=categorical_features,
        high_cardinality=high_cardinality,
        low_cardinality=low_cardinality,
        numerical_features=numerical_features,
        feature_columns=feature_columns,
        target_encoding_maps=target_encoding_maps,
    )

    save_artifacts(
        output_dir=args.output_dir,
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        model_filename=args.model_name,
        preprocessor_filename=args.preprocessor_name,
        metrics_filename=args.metrics_name,
    )


if __name__ == "__main__":
    main()
