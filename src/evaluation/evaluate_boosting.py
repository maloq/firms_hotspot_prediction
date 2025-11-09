"""Evaluation CLI for CatBoost fire prediction models.

This script loads a feature dataset, applies a trained CatBoost model,
then reports classification metrics for user-defined geographic regions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
            name=entry["name"],
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
    """Pick the first existing path from candidates or raise a helpful error."""

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
    raise ValueError(
        f"No {label} specified. Provide a CLI argument or configure it in the YAML file."
    )


def _resolve_coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    lat_candidates = ("lat_rounded", "latitude", "lat")
    lon_candidates = ("lon_rounded", "longitude", "lon")
    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise KeyError(
            "Latitude/longitude columns not found. Expected one of 'lat_rounded', 'latitude', 'lat' and 'lon_rounded', 'longitude', 'lon'."
        )
    return lat_col, lon_col


def _load_metadata(metadata_dir: Path) -> dict:
    metadata_path = metadata_dir / "prepared_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file '{metadata_path}' not found. Expected prepared_metadata.json adjacent to prepared_data.npz."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=4)
def _load_coord_cache(metadata_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = _load_metadata(metadata_dir)
    coord_columns = (metadata.get("coordinates") or {}).get("columns") or []
    needed_keys = {entry.get("npz_key") for entry in coord_columns if entry.get("npz_key")}
    needed_keys.update({f"{split}_idx" for split in ("train", "val", "test")})

    npz_path = metadata_dir / "prepared_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dataset NPZ '{npz_path}' not found while loading coordinate metadata."
        )

    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for key in needed_keys:
            if key in data.files:
                arrays[key] = np.asarray(data[key])

    missing = [key for key in needed_keys if key not in arrays]
    if missing:
        raise KeyError(
            f"Dataset NPZ '{npz_path}' is missing required arrays {missing} for metadata reconstruction."
        )

    return metadata, arrays


def _load_coord_dataframe(metadata_dir: Path, split: str) -> pd.DataFrame:
    metadata, arrays = _load_coord_cache(metadata_dir)
    coord_columns = (metadata.get("coordinates") or {}).get("columns") or []
    if not coord_columns:
        raise ValueError("Metadata does not define coordinate columns.")

    idx_key = f"{split}_idx"
    if idx_key not in arrays:
        raise KeyError(f"Metadata missing indices for split '{split}'.")
    indices = arrays[idx_key].astype(int)

    columns: dict[str, np.ndarray] = {}
    for entry in coord_columns:
        name = entry.get("name")
        npz_key = entry.get("npz_key")
        dtype = entry.get("dtype")
        if not name or not npz_key:
            continue
        if npz_key not in arrays:
            raise KeyError(
                f"Coordinate array '{npz_key}' referenced in metadata not found in prepared_data.npz."
            )
        full_array = arrays[npz_key]
        selected = full_array[indices]
        if dtype == "datetime64[ns]":
            unit = entry.get("unit", "ns")
            selected = pd.to_datetime(selected.astype("int64"), unit=unit, errors="coerce")
        columns[name] = selected

    return pd.DataFrame(columns).reset_index(drop=True)


def _first_non_null(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _resolve_splits(requested, available_default: list[str]) -> list[str]:
    possible = ["train", "val", "test"]
    if requested:
        normalized = []
        for v in requested:
            v = v.lower()
            if v == "all":
                return available_default or possible
            normalized.append(v)
        return normalized
    return available_default or ["test"]


def _infer_date_column(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    candidates = [preferred] if preferred else []
    candidates += ["date", "datetime", "ds", "time", "timestamp", "day", "date_id"]
    for c in candidates:
        if c and c in df.columns:
            return c
    return None


def _choose_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float | None]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5, None
    precision = precision[:-1]
    recall = recall[:-1]
    if precision.size == 0 or recall.size == 0:
        return 0.5, None
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    if not np.isfinite(f1).any():
        return 0.5, None
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx]) if np.isfinite(f1[idx]) else None


def _regions_from_entries(entries: Sequence[dict]) -> list[Region]:
    regions: list[Region] = []
    for entry in entries:
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


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    config_path = Path(args.config_path)

    config = _load_config(config_path)
    evaluation_section = config.get("evaluation")
    eval_cfg = evaluation_section if isinstance(evaluation_section, dict) else {}

    dataset_path = _resolve_path(
        [
            args.features_path,
            eval_cfg.get("features_path") if isinstance(eval_cfg, dict) else None,
            config.get("prepared_prediction_features_path"),
            config.get("features_path"),
        ],
        label="feature dataset",
    )
    model_path = _resolve_path(
        [
            args.model_path,
            eval_cfg.get("model_path") if isinstance(eval_cfg, dict) else None,
            config.get("model_path"),
        ],
        label="model",
    )

    df = _load_dataset(dataset_path)
    df = _prepare_features(df, config)

    target_col = _first_non_null(
        args.target_column,
        eval_cfg.get("target_column") if eval_cfg else None,
        config.get("target_column"),
        config.get("target_col"),
        default="count",
    )
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    positive_threshold = _first_non_null(
        args.target_positive_threshold,
        eval_cfg.get("target_positive_threshold") if eval_cfg else None,
        config.get("target_positive_threshold"),
        default=0.0,
    )
    y_true_all = (df[target_col] > float(positive_threshold)).astype(int).to_numpy()
    features = df.drop(columns=[target_col])

    model = CatBoostClassifier()
    model.load_model(model_path)

    cat_features = [col for col in config.get("cat_features", []) if col in features.columns]

    feature_names = list(getattr(model, "feature_names_", []) or [])
    if feature_names:
        missing_in_data = set(feature_names) - set(features.columns)
        if missing_in_data:
            raise KeyError(
                f"Model expects features missing from dataset: {sorted(missing_in_data)}"
            )
        features = features[feature_names]

    pool = Pool(features, cat_features=cat_features) if cat_features else Pool(features)
    probabilities_all = model.predict_proba(pool)[:, 1]

    # Build a working dataframe with predictions and labels
    df_all = df.copy()
    df_all["y_true"] = y_true_all
    df_all["pred_proba"] = probabilities_all

    regions: list[Region] = []
    if args.regions_file:
        regions.extend(_load_regions_from_file(Path(args.regions_file)))
    if args.region:
        regions.extend(_parse_region_spec(spec) for spec in args.region)
    if not regions and eval_cfg:
        cfg_regions_path = eval_cfg.get("regions_file")
        if cfg_regions_path:
            regions.extend(_load_regions_from_file(Path(cfg_regions_path)))
    if not regions and eval_cfg:
        cfg_regions_entries = eval_cfg.get("regions")
        if cfg_regions_entries:
            regions.extend(_regions_from_entries(cfg_regions_entries))

    # Resolve coordinates for region masks
    lat_col = _first_non_null(
        args.lat_column,
        eval_cfg.get("lat_column") if eval_cfg else None,
        config.get("lat_column"),
    )
    lon_col = _first_non_null(
        args.lon_column,
        eval_cfg.get("lon_column") if eval_cfg else None,
        config.get("lon_column"),
    )
    if not lat_col or not lon_col:
        lat_col, lon_col = _resolve_coordinate_columns(df_all)
    if lat_col not in df_all.columns or lon_col not in df_all.columns:
        raise KeyError(f"Latitude column '{lat_col}' or longitude column '{lon_col}' not found")

    # Split handling
    metadata_dir = _first_non_null(
        args.metadata_dir,
        eval_cfg.get("metadata_dir") if eval_cfg else None,
        eval_cfg.get("nn_metadata_dir") if eval_cfg else None,
        config.get("nn_metadata_dir"),
        config.get("output_train_data_dir"),
    )

    date_col = _first_non_null(args.date_column, eval_cfg.get("date_column") if eval_cfg else None, config.get("date_column"))
    if date_col and date_col not in df_all.columns:
        date_col = None

    available_default = []
    coord_masks = {}
    if metadata_dir:
        md = Path(metadata_dir)
        try:
            for sp in ("train", "val", "test"):
                coord_df = _load_coord_dataframe(md, sp)
                coord_masks[sp] = coord_df
                available_default.append(sp)
        except Exception:
            # If metadata dir provided but unusable, fall back to date-based or global
            coord_masks = {}
            available_default = []

    splits = _resolve_splits(args.split, available_default)

    # Build per-split dataframes
    split_records: list[dict] = []
    if coord_masks:
        # Attempt to filter df_all to the same rows as coord masks by merging on lat/lon (+ date if present)
        for sp, coord_df in coord_masks.items():
            if sp not in splits:
                continue
            # Determine keys
            try:
                coord_lat, coord_lon = _resolve_coordinate_columns(coord_df)
            except KeyError:
                coord_lat, coord_lon = lat_col, lon_col

            join_keys = [(lat_col, coord_lat), (lon_col, coord_lon)]
            date_key_df = _infer_date_column(df_all, preferred=date_col)
            date_key_coord = _infer_date_column(coord_df, preferred=date_col)
            if date_key_df and date_key_coord:
                # Convert to datetime for robust merge
                try:
                    df_all[date_key_df] = pd.to_datetime(df_all[date_key_df])
                    coord_df[date_key_coord] = pd.to_datetime(coord_df[date_key_coord])
                    join_keys.append((date_key_df, date_key_coord))
                except Exception:
                    pass

            left_on = [k[0] for k in join_keys]
            right_on = [k[1] for k in join_keys]
            merged = df_all.merge(coord_df[right_on], left_on=left_on, right_on=right_on, how="inner")
            split_records.append({"split": sp, "df": merged})
    elif date_col and (config.get("train_end") or config.get("val_end") or args.train_end or args.val_end):
        # Date-based splits using same boundaries as NN config
        train_end = pd.to_datetime(_first_non_null(args.train_end, config.get("train_end"))) if _first_non_null(args.train_end, config.get("train_end")) else None
        val_end = pd.to_datetime(_first_non_null(args.val_end, config.get("val_end"))) if _first_non_null(args.val_end, config.get("val_end")) else None
        if date_col not in df_all.columns:
            raise KeyError(f"Date column '{date_col}' not found for splitting")
        dates = pd.to_datetime(df_all[date_col], errors="coerce")
        if "train" in splits and train_end is not None:
            split_records.append({"split": "train", "df": df_all[dates < train_end]})
        if "val" in splits and train_end is not None and val_end is not None:
            split_records.append({"split": "val", "df": df_all[(dates >= train_end) & (dates <= val_end)]})
        if "test" in splits:
            if val_end is not None:
                split_records.append({"split": "test", "df": df_all[dates > val_end]})
            elif train_end is not None:
                split_records.append({"split": "test", "df": df_all[dates >= train_end]})
    else:
        # Global fallback (no split info) — treat as a single split 'GLOBAL'
        split_name = splits[0] if splits else "all"
        split_records.append({"split": split_name, "df": df_all})

    # Determine threshold: use F1-optimal on validation split if not explicitly set
    default_threshold = float(
        _first_non_null(
            args.threshold,
            eval_cfg.get("threshold") if eval_cfg else None,
            config.get("threshold"),
            default=0.5,
        )
    )
    threshold_explicitly_set = any(
        value is not None for value in (args.threshold, (eval_cfg.get("threshold") if eval_cfg else None), config.get("threshold"))
    )

    effective_threshold = default_threshold
    if not threshold_explicitly_set:
        val_rec = next((r for r in split_records if r["split"] == "val"), None)
        if val_rec is not None and not val_rec["df"].empty:
            thr, best_f1 = _choose_threshold_f1(val_rec["df"]["y_true"].to_numpy(), val_rec["df"]["pred_proba"].to_numpy())
            if best_f1 is not None:
                effective_threshold = thr
                if args.do_print:
                    print(f"Using validation F1-optimal threshold: {effective_threshold:.4f} (val F1={best_f1:.4f})")

    # Compute metrics per split and per region
    results: list[dict] = []
    for rec in split_records:
        split_name = rec["split"].upper()
        sdf = rec["df"].copy()
        if sdf.empty:
            metrics = _compute_metrics("GLOBAL", np.array([], dtype=int), np.array([], dtype=float), effective_threshold)
            row = metrics.as_dict()
            row["split"] = rec["split"]
            results.append(row)
            continue

        # Global metrics for the split
        metrics = _compute_metrics("GLOBAL", sdf["y_true"].to_numpy(), sdf["pred_proba"].to_numpy(), effective_threshold)
        row = metrics.as_dict()
        row["split"] = rec["split"]
        results.append(row)

        # Region metrics
        for region in regions:
            mask = (
                (sdf[lat_col] >= region.lat_min)
                & (sdf[lat_col] <= region.lat_max)
                & (sdf[lon_col] >= region.lon_min)
                & (sdf[lon_col] <= region.lon_max)
            )
            region_threshold = region.threshold if region.threshold is not None else effective_threshold
            rmetrics = _compute_metrics(
                region_name=region.name,
                y_true=sdf.loc[mask, "y_true"].to_numpy(),
                y_prob=sdf.loc[mask, "pred_proba"].to_numpy(),
                threshold=region_threshold,
            )
            rrow = rmetrics.as_dict()
            rrow["split"] = rec["split"]
            results.append(rrow)

    if len(results) == 0:
        print("No results produced; check split configuration.")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    if "split" in result_df.columns:
        result_df = result_df.set_index(["split", "region"])
    else:
        result_df = result_df.set_index("region")

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
    parser.add_argument("--model-path", help="Path to CatBoost model (.cbm)")
    parser.add_argument("--config-path", default="configs/features_config_30d.yaml", help="Feature config YAML")
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test", "all"),
        help="Dataset split(s) to evaluate. Use metadata_dir or date boundaries.",
    )
    parser.add_argument("--regions-file", help="YAML file describing evaluation regions")
    parser.add_argument(
        "--region",
        action="append",
        help="Inline region spec 'name:lat_min:lat_max:lon_min:lon_max[:threshold]' (repeatable)",
    )
    parser.add_argument("--lat-column", help="Latitude column name")
    parser.add_argument("--lon-column", help="Longitude column name")
    parser.add_argument("--date-column", help="Date/time column name for date-based split")
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
    parser.add_argument(
        "--metadata-dir",
        help="Directory containing prepared_data.npz and prepared_metadata.json for NN-aligned splits.",
    )
    parser.add_argument("--train-end", help="Train end date (inclusive end of train) for date-based split")
    parser.add_argument("--val-end", help="Validation end date for date-based split")
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
