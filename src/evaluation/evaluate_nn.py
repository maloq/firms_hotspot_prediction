"""Evaluation CLI for neural-network fire prediction models.

This script loads the preprocessed NN training dataset (prepared_data.npz),
applies a trained Lightning checkpoint, and reports classification metrics
globally and for user-defined geographic regions.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
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

# Ensure the project root is importable when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.neural_net.models.lightning import SequenceStaticLightningModule


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
    region = Region(
        name=name,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        threshold=threshold,
    )
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


def _regions_from_entries(entries: Iterable[dict]) -> list[Region]:
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
    precision = _safe_binary_metric(
        lambda yt, yp: precision_score(yt, yp, zero_division=0),
        y_true,
        y_pred,
    )
    recall = _safe_binary_metric(
        lambda yt, yp: recall_score(yt, yp, zero_division=0),
        y_true,
        y_pred,
    )
    f1 = _safe_binary_metric(
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        y_true,
        y_pred,
    )

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


def _load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(
    candidates: Sequence[str | Path | None],
    label: str,
) -> Path:
    missing: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
        missing.append(str(path))

    if missing:
        raise FileNotFoundError(f"None of the provided {label} paths exist: {missing}")
    raise ValueError(f"No {label} specified. Provide a CLI argument or update the configuration.")


def _first_non_null(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _resolve_dataset_path(args: argparse.Namespace, config: dict, eval_cfg: dict) -> Path:
    base_dirs: list[str | Path | None] = [
        eval_cfg.get("metadata_dir") if eval_cfg else None,
        eval_cfg.get("nn_metadata_dir") if eval_cfg else None,
        config.get("nn_metadata_dir"),
        config.get("output_train_data_dir"),
    ]

    candidates: list[str | Path | None] = [
        args.data_path,
        (eval_cfg.get("prepared_data_path") if eval_cfg else None),
        (eval_cfg.get("nn_prepared_data_path") if eval_cfg else None),
        config.get("nn_prepared_data_path"),
        config.get("prepared_data_path"),
    ]

    for base in base_dirs:
        if not base:
            continue
        base_path = Path(base)
        if base_path.suffix == ".npz":
            candidates.append(base_path)
        else:
            candidates.append(base_path / "prepared_data.npz")

    candidates.append(Path("data/saved_features/nn_train_data/prepared_data.npz"))
    return _resolve_path(candidates, label="prepared NN dataset")


def _resolve_metadata_dir(
    args: argparse.Namespace,
    config: dict,
    eval_cfg: dict,
    dataset_path: Path,
) -> Path:
    candidates: list[str | Path | None] = [
        args.metadata_dir,
        eval_cfg.get("metadata_dir") if eval_cfg else None,
        eval_cfg.get("nn_metadata_dir") if eval_cfg else None,
        config.get("nn_metadata_dir"),
        config.get("output_train_data_dir"),
        dataset_path.parent,
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            path = path.parent
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate metadata directory containing prepared_metadata.json. "
        "Specify --metadata-dir explicitly."
    )


def _resolve_model_path(args: argparse.Namespace, config: dict, eval_cfg: dict) -> Path:
    candidates = [
        args.model_path,
        eval_cfg.get("model_path") if eval_cfg else None,
        eval_cfg.get("nn_model_path") if eval_cfg else None,
        config.get("nn_model_path"),
        config.get("model_path"),
    ]
    return _resolve_path(candidates, label="neural-net model checkpoint")


def _resolve_coordinate_columns(df: pd.DataFrame) -> tuple[str, str]:
    lat_candidates = ("lat_rounded", "latitude", "lat")
    lon_candidates = ("lon_rounded", "longitude", "lon")

    lat_col = next((col for col in lat_candidates if col in df.columns), None)
    lon_col = next((col for col in lon_candidates if col in df.columns), None)
    if lat_col is None or lon_col is None:
        raise KeyError(
            "Latitude/longitude columns not found in metadata. Expected one of "
            "'lat_rounded', 'latitude', 'lat' and 'lon_rounded', 'longitude', 'lon'."
        )
    return lat_col, lon_col


def _load_metadata(metadata_dir: Path) -> dict:
    metadata_path = metadata_dir / "prepared_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file '{metadata_path}' not found. Expected prepared_metadata.json next to prepared_data.npz."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_coord_dataframe(
    metadata: dict,
    npz_data: np.lib.npyio.NpzFile,
    split: str,
    expected_rows: int,
) -> pd.DataFrame:
    coord_columns = (metadata.get("coordinates") or {}).get("columns") or []
    if not coord_columns:
        raise ValueError("Metadata does not include coordinate column definitions.")

    idx_key = f"{split}_idx"
    if idx_key not in npz_data:
        raise KeyError(f"Dataset NPZ missing '{idx_key}' required to reconstruct split '{split}'.")
    split_indices = np.asarray(npz_data[idx_key], dtype=int)

    columns: dict[str, Iterable] = {}
    for entry in coord_columns:
        name = entry.get("name")
        npz_key = entry.get("npz_key")
        dtype = entry.get("dtype")
        if not name or not npz_key:
            continue
        if npz_key not in npz_data:
            raise KeyError(
                f"Coordinate array '{npz_key}' referenced in metadata not found in dataset NPZ."
            )
        full_array = np.asarray(npz_data[npz_key])
        selected = full_array[split_indices]
        if dtype == "datetime64[ns]":
            unit = entry.get("unit", "ns")
            selected = pd.to_datetime(selected.astype("int64"), unit=unit, errors="coerce")
        columns[name] = selected

    coord_df = pd.DataFrame(columns)
    coord_df = coord_df.reset_index(drop=True)
    if expected_rows >= 0 and len(coord_df) != expected_rows:
        raise ValueError(
            f"Coordinate mask length mismatch for split '{split}': expected {expected_rows}, got {len(coord_df)}"
        )
    return coord_df


def _predict_probabilities(
    model: SequenceStaticLightningModule,
    x_dyn: np.ndarray,
    x_stat: np.ndarray,
    x_cat: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if x_dyn.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    if x_cat is None:
        x_cat = np.zeros((x_dyn.shape[0], 0), dtype=np.int64)

    model.eval()
    probs: list[np.ndarray] = []
    n_samples = x_dyn.shape[0]

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            dyn_batch = torch.as_tensor(x_dyn[start:end], dtype=torch.float32, device=device)
            stat_batch = torch.as_tensor(x_stat[start:end], dtype=torch.float32, device=device)
            cat_batch = torch.as_tensor(x_cat[start:end], dtype=torch.long, device=device)
            logits = model(dyn_batch, stat_batch, cat_batch if cat_batch.numel() else None)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            probs.append(batch_probs)

    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def _load_nn_model(
    model_path: Path,
    device: torch.device,
    nn_model_cfg: dict | None,
) -> SequenceStaticLightningModule:
    def _torch_load_with_compat(weights_only: bool | None):
        kwargs = {"map_location": "cpu"}
        if weights_only is not None:
            kwargs["weights_only"] = weights_only
        return torch.load(model_path, **kwargs)

    try:
        checkpoint = _torch_load_with_compat(False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    except pickle.UnpicklingError:
        checkpoint = _torch_load_with_compat(None)

    hyperparams = checkpoint.get("hyper_parameters") or {}
    saved_arch = hyperparams.get("model_name", "lstm_mlp")
    saved_model_config = hyperparams.get("model_config") or {}

    if nn_model_cfg:
        requested_arch = nn_model_cfg.get("architecture")
        if requested_arch and requested_arch != saved_arch:
            raise ValueError(
                f"Configured architecture '{requested_arch}' does not match checkpoint architecture '{saved_arch}'."
            )

    lightning_defaults = {
        key: hyperparams.get(key)
        for key in ("learning_rate", "decay_rate", "decay_steps", "l2", "clip_gradient_norm")
        if key in hyperparams
    }
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Checkpoint '{model_path}' is missing 'state_dict'.")
    del checkpoint

    init_kwargs = {key: value for key, value in lightning_defaults.items() if value is not None}

    model = SequenceStaticLightningModule(
        model_name=saved_arch,
        model_config=saved_model_config,
        **init_kwargs,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def _resolve_splits(
    requested: Iterable[str] | None,
    available_keys: Iterable[str],
) -> list[str]:
    available = {key for key in available_keys}
    possible = ["train", "val", "test"]

    if requested:
        normalized: list[str] = []
        for value in requested:
            value = value.lower()
            if value == "all":
                return [split for split in possible if f"x_dyn_{split}" in available]
            normalized.append(value)
        return normalized

    for candidate in ("test", "val", "train"):
        if f"x_dyn_{candidate}" in available and f"y_{candidate}" in available:
            return [candidate]

    return [split for split in possible if f"x_dyn_{split}" in available and f"y_{split}" in available]


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    config_path = Path(args.config_path)
    config = _load_config(config_path)

    evaluation_section = config.get("evaluation_nn") or config.get("evaluation")
    eval_cfg = evaluation_section if isinstance(evaluation_section, dict) else {}

    dataset_path = _resolve_dataset_path(args, config, eval_cfg)
    metadata_dir = _resolve_metadata_dir(args, config, eval_cfg, dataset_path)
    metadata = _load_metadata(metadata_dir)
    model_path = _resolve_model_path(args, config, eval_cfg)

    nn_model_cfg = config.get("nn_model") if isinstance(config.get("nn_model"), dict) else {}

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_nn_model(model_path, device, nn_model_cfg)

    positive_threshold = float(
        _first_non_null(
            args.target_positive_threshold,
            eval_cfg.get("target_positive_threshold") if eval_cfg else None,
            config.get("target_positive_threshold"),
            default=0.0,
        )
    )

    default_threshold = float(
        _first_non_null(
            args.threshold,
            eval_cfg.get("threshold") if eval_cfg else None,
            config.get("threshold"),
            default=0.5,
        )
    )
    threshold_explicitly_set = any(
        value is not None
        for value in (
            args.threshold,
            (eval_cfg.get("threshold") if eval_cfg else None),
            config.get("threshold"),
        )
    )

    target_col_name = str(
        _first_non_null(
            args.target_column,
            eval_cfg.get("target_column") if eval_cfg else None,
            config.get("target_column"),
            config.get("target_col"),
            default="count",
        )
    )

    regions: list[Region] = []
    if args.regions_file:
        regions.extend(_load_regions_from_file(Path(args.regions_file)))
    if args.region:
        regions.extend(_parse_region_spec(spec) for spec in args.region)
    if not regions and eval_cfg:
        cfg_regions_file = eval_cfg.get("regions_file")
        if cfg_regions_file:
            regions.extend(_load_regions_from_file(Path(cfg_regions_file)))
    if not regions and eval_cfg:
        entries = eval_cfg.get("regions")
        if entries:
            if isinstance(entries, (str, Path)):
                regions.extend(_load_regions_from_file(Path(entries)))
            else:
                iterable = entries if isinstance(entries, list) else [entries]
                regions.extend(_regions_from_entries(iterable))
    if not regions:
        print("No regions provided; only GLOBAL metrics are returned.")

    results: list[dict] = []
    split_records: list[dict] = []

    with np.load(dataset_path, allow_pickle=False) as data:
        available_keys = data.files
        splits = _resolve_splits(args.split, available_keys)

        for split in splits:
            dyn_key = f"x_dyn_{split}"
            stat_key = f"x_stat_{split}"
            y_key = f"y_{split}"

            if dyn_key not in data or y_key not in data:
                raise KeyError(f"Dataset '{dataset_path}' is missing required keys for split '{split}'.")

            x_dyn = np.asarray(data[dyn_key], dtype=np.float32)
            x_stat = np.asarray(data[stat_key], dtype=np.float32) if stat_key in data else None
            y_raw = np.asarray(data[y_key], dtype=np.float32)

            n_samples = x_dyn.shape[0]
            if x_stat is None or x_stat.size == 0:
                x_stat = np.zeros((n_samples, 0), dtype=np.float32)

            cat_key = f"x_cat_{split}"
            if cat_key in data:
                x_cat = np.asarray(data[cat_key], dtype=np.int64)
            else:
                x_cat = np.zeros((n_samples, 0), dtype=np.int64)

            if y_raw.shape[0] != n_samples or x_stat.shape[0] != n_samples:
                raise ValueError(
                    f"Split '{split}' arrays have inconsistent lengths "
                    f"(x_dyn={n_samples}, x_stat={x_stat.shape[0]}, y={y_raw.shape[0]})."
                )
            if x_cat.shape[0] != n_samples:
                raise ValueError(
                    f"Split '{split}' categorical array has inconsistent length ({x_cat.shape[0]} vs {n_samples})."
                )

            y_true = (y_raw > positive_threshold).astype(np.int32)
            probabilities = _predict_probabilities(model, x_dyn, x_stat, x_cat, args.batch_size, device)

            if probabilities.shape[0] != n_samples:
                raise RuntimeError(
                    f"Predicted probability count mismatch for split '{split}': "
                    f"expected {n_samples}, got {probabilities.shape[0]}"
                )

            coord_df = _load_coord_dataframe(metadata, data, split, n_samples)
            lat_override = _first_non_null(args.lat_column, eval_cfg.get("lat_column") if eval_cfg else None, config.get("lat_column"))
            lon_override = _first_non_null(args.lon_column, eval_cfg.get("lon_column") if eval_cfg else None, config.get("lon_column"))

            if lat_override and lat_override not in coord_df.columns:
                raise KeyError(f"Latitude column '{lat_override}' not found in metadata for split '{split}'.")
            if lon_override and lon_override not in coord_df.columns:
                raise KeyError(f"Longitude column '{lon_override}' not found in metadata for split '{split}'.")

            if lat_override and lon_override:
                lat_col = lat_override
                lon_col = lon_override
            else:
                lat_col, lon_col = _resolve_coordinate_columns(coord_df)

            split_records.append(
                {
                    "split": split,
                    "y_true": y_true,
                    "probabilities": probabilities,
                    "coord_df": coord_df,
                    "lat_col": lat_col,
                    "lon_col": lon_col,
                }
            )

    effective_threshold = default_threshold
    if not threshold_explicitly_set:
        val_record = next((record for record in split_records if record["split"] == "val"), None)
        if val_record is not None:
            best_threshold, best_f1 = _choose_threshold_f1(val_record["y_true"], val_record["probabilities"])
            if best_f1 is not None:
                effective_threshold = best_threshold
                if args.do_print:
                    print(f"Using validation F1-optimal threshold: {effective_threshold:.4f} (val F1={best_f1:.4f})")

    for record in split_records:
        split = record["split"]
        y_true = record["y_true"]
        probabilities = record["probabilities"]
        coord_df = record["coord_df"]
        lat_col = record["lat_col"]
        lon_col = record["lon_col"]

        df_with_preds = coord_df.copy()
        df_with_preds[target_col_name] = y_true
        df_with_preds["pred_proba"] = probabilities

        global_metrics = _compute_metrics("GLOBAL", y_true, probabilities, effective_threshold)
        global_row = global_metrics.as_dict()
        global_row["split"] = split
        results.append(global_row)

        for region in regions:
            mask = (
                (df_with_preds[lat_col] >= region.lat_min)
                & (df_with_preds[lat_col] <= region.lat_max)
                & (df_with_preds[lon_col] >= region.lon_min)
                & (df_with_preds[lon_col] <= region.lon_max)
            )

            region_threshold = region.threshold if region.threshold is not None else effective_threshold
            region_metrics = _compute_metrics(
                region_name=region.name,
                y_true=df_with_preds[target_col_name].to_numpy()[mask],
                y_prob=df_with_preds["pred_proba"].to_numpy()[mask],
                threshold=region_threshold,
            )
            region_row = region_metrics.as_dict()
            region_row["split"] = split
            results.append(region_row)

    if len(results) == 0:
        raise RuntimeError("No evaluation results produced. Check chosen splits and dataset content.")

    result_df = pd.DataFrame(results)
    if "split" in result_df.columns:
        result_df = result_df.set_index(["split", "region"])
    else:
        result_df = result_df.set_index("region")

    output_path = args.output_path
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".json":
            output_path.write_text(result_df.to_json(orient="index", indent=2))
        else:
            result_df.to_csv(output_path)

    return result_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default="configs/features_config_30d_LSTM_early_fusion.yaml", help="Path to feature/config YAML.")
    parser.add_argument("--model-path", help="Path to trained Lightning checkpoint (.ckpt).")
    parser.add_argument("--data-path", help="Path to prepared_data.npz containing NN features.")
    parser.add_argument(
        "--metadata-dir",
        help="Directory containing prepared_data.npz and prepared_metadata.json.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test", "all"),
        help="Dataset split(s) to evaluate. Repeatable. Defaults to the first available split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size used for neural-network inference.",
    )
    parser.add_argument("--device", help="Torch device for inference (default: cuda if available else cpu).")
    parser.add_argument("--regions-file", default="configs/regions_example.yaml", help="YAML file describing evaluation regions.")
    parser.add_argument(
        "--region",
        action="append",
        help="Inline region spec 'name:lat_min:lat_max:lon_min:lon_max[:threshold]' (repeatable).",
    )
    parser.add_argument("--lat-column", help="Latitude column name in metadata.")
    parser.add_argument("--lon-column", help="Longitude column name in metadata.")
    parser.add_argument("--target-column", help="Target column name used in reported metrics.")
    parser.add_argument(
        "--target-positive-threshold",
        type=float,
        help="Values strictly greater than this threshold are treated as positives.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Default probability threshold used for binarisation.",
    )
    parser.add_argument("--output-path", help="Optional path to save metrics (csv or json).")
    parser.add_argument(
        "--quiet",
        dest="do_print",
        action="store_false",
        help="Suppress printing the metrics table to stdout.",
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
