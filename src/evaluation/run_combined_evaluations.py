"""Utility to run multiple evaluation pipelines and aggregate their metrics.

This script orchestrates the CatBoost, logistic regression, and neural-network
evaluation CLIs so that they run in sequence and produce a unified metrics
table. Neural networks can be evaluated with multiple configuration files,
and all of the resulting metrics are merged into a single DataFrame that is
printed to stdout and can optionally be written to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.evaluation import evaluate_boosting, evaluate_log_regression, evaluate_nn
else:
    from . import evaluate_boosting, evaluate_log_regression, evaluate_nn


DEFAULT_BOOSTING_CONFIGS: tuple[tuple[str, Path], ...] = (
    ("catboost_30d", Path("configs/features_config_30d.yaml")),
)

DEFAULT_LOGREG_METADATA: tuple[tuple[str, Path], ...] = (
    ("logreg_30d", Path("models/log_regression/metrics.json")),
)

DEFAULT_NN_CONFIGS: tuple[tuple[str, Path], ...] = (
    ("nn_lstm_early_fusion", Path("configs/features_config_30d_LSTM_early_fusion.yaml")),
    # ("nn_lstm", Path("configs/features_config_30d_LSTM.yaml")),
    # ("nn_mlp", Path("configs/features_config_30d_MLP.yaml")),
)

AGG_COLUMNS: tuple[str, ...] = (
    "model_family",
    "model_name",
    #"method_name",
    # "split",
    "region",
    # "support",
    # "positives",
    # "predicted_positives",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)

REGION_DISPLAY_COLUMNS: tuple[str, ...] = (
    "model_family",
    "model_name",
    #"method_name",
    # "split",
    # "support",
    # "positives",
    # "predicted_positives",
    # "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)

FLOAT_METRIC_COLUMNS: tuple[str, ...] = (
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)

FLOAT_METRIC_COLUMNS: tuple[str, ...] = (
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
)

RegionBounds = Tuple[float, float, float, float]
RegionBoundsMap = Dict[str, RegionBounds]


def _parse_named_paths(values: Sequence[str] | None, defaults: Sequence[tuple[str, Path]]) -> list[tuple[str, Path]]:
    if not values:
        return [(name, path) for name, path in defaults]

    pairs: list[tuple[str, Path]] = []
    for raw in values:
        if "=" in raw:
            label, path_str = raw.split("=", 1)
        elif ":" in raw:
            label, path_str = raw.split(":", 1)
        else:
            label, path_str = Path(raw).stem, raw

        path = Path(path_str).expanduser()
        pairs.append((label.strip(), path))
    return pairs


def _ensure_exists(entries: Iterable[tuple[str, Path]], description: str) -> list[tuple[str, Path]]:
    validated: list[tuple[str, Path]] = []
    for label, path in entries:
        if not path.exists():
            raise FileNotFoundError(f"{description} '{path}' (label '{label}') does not exist.")
        validated.append((label, path))
    return validated


def _load_yaml_dict(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data or {}


def _load_json_dict(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def _region_bounds_from_entries(entries: Iterable[dict]) -> RegionBoundsMap:
    bounds: RegionBoundsMap = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name"))
        if not name:
            continue
        try:
            lat_min = float(entry["lat_min"])
            lat_max = float(entry["lat_max"])
            lon_min = float(entry["lon_min"])
            lon_max = float(entry["lon_max"])
        except (KeyError, TypeError, ValueError):
            continue
        bounds[name] = (lat_min, lat_max, lon_min, lon_max)
    return bounds


def _region_bounds_from_file(path: Path) -> RegionBoundsMap:
    if not path:
        return {}
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if isinstance(data, dict) and "regions" in data:
        entries = data["regions"]
    elif isinstance(data, list):
        entries = data
    else:
        entries = [data]
    return _region_bounds_from_entries(entries)


def _collect_region_bounds_from_eval_cfg(eval_cfg: dict | None) -> RegionBoundsMap:
    if not isinstance(eval_cfg, dict):
        return {}
    # Prefer explicit regions_file when available.
    regions_file = eval_cfg.get("regions_file")
    if regions_file:
        file_bounds = _region_bounds_from_file(Path(regions_file))
        if file_bounds:
            return file_bounds

    regions_value = eval_cfg.get("regions")
    if not regions_value:
        return {}
    if isinstance(regions_value, (str, Path)):
        return _region_bounds_from_file(Path(regions_value))
    if not isinstance(regions_value, list):
        regions_value = [regions_value]
    return _region_bounds_from_entries(regions_value)


def _collect_boosting_region_bounds(config_path: Path, regions_file: Path | None) -> RegionBoundsMap:
    config = _load_yaml_dict(config_path)
    eval_cfg = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    if regions_file:
        file_bounds = _region_bounds_from_file(Path(regions_file))
        if file_bounds:
            return file_bounds
    return _collect_region_bounds_from_eval_cfg(eval_cfg)


def _collect_logistic_region_bounds(metadata_path: Path, regions_file: Path | None) -> RegionBoundsMap:
    metadata = _load_json_dict(metadata_path)
    config_candidates = [
        metadata.get("config_path"),
        Path("configs/features_config_30d.yaml"),
    ]
    config_path = _first_existing_path(config_candidates)
    config = _load_yaml_dict(config_path) if config_path else {}
    eval_cfg = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    if regions_file:
        file_bounds = _region_bounds_from_file(Path(regions_file))
        if file_bounds:
            return file_bounds
    return _collect_region_bounds_from_eval_cfg(eval_cfg)


def _collect_nn_region_bounds(config_path: Path, regions_file: Path | None) -> RegionBoundsMap:
    config = _load_yaml_dict(config_path)
    eval_cfg = config.get("evaluation_nn")
    if not isinstance(eval_cfg, dict):
        eval_cfg = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    if regions_file:
        file_bounds = _region_bounds_from_file(Path(regions_file))
        if file_bounds:
            return file_bounds
    return _collect_region_bounds_from_eval_cfg(eval_cfg)


def _first_existing_path(candidates: Iterable[str | Path | None]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


def _merge_region_bounds(target: RegionBoundsMap, new_bounds: RegionBoundsMap) -> None:
    for name, coords in new_bounds.items():
        target.setdefault(name, coords)


def _run_boosting(label: str, config_path: Path, regions_file: Path | None) -> tuple[pd.DataFrame, RegionBoundsMap]:
    parser = evaluate_boosting.build_parser()
    cli_args = ["--config-path", str(config_path), "--quiet"]
    if regions_file is not None:
        cli_args.extend(["--regions-file", str(regions_file)])
    args = parser.parse_args(cli_args)
    df = evaluate_boosting.evaluate(args).reset_index()
    df["model_family"] = "boosting"
    df["model_name"] = label
    df["method_name"] = "evaluate_boosting"
    df["split"] = "N/A"
    bounds = _collect_boosting_region_bounds(config_path, regions_file)
    return df, bounds


def _run_logistic(label: str, metadata_path: Path, regions_file: Path | None) -> tuple[pd.DataFrame, RegionBoundsMap]:
    parser = evaluate_log_regression.build_parser()
    cli_args = ["--metadata-path", str(metadata_path), "--quiet"]
    if regions_file is not None:
        cli_args.extend(["--regions-file", str(regions_file)])
    args = parser.parse_args(cli_args)
    df = evaluate_log_regression.evaluate(args).reset_index()
    df["model_family"] = "logistic_regression"
    df["model_name"] = label
    df["method_name"] = "evaluate_log_regression"
    df["split"] = "N/A"
    bounds = _collect_logistic_region_bounds(metadata_path, regions_file)
    return df, bounds


def _run_neural_network(
    label: str,
    config_path: Path,
    regions_file: Path | None,
    nn_splits: Sequence[str] | None,
) -> tuple[pd.DataFrame, RegionBoundsMap]:
    parser = evaluate_nn.build_parser()
    cli_args = ["--config-path", str(config_path), "--quiet"]
    if regions_file is not None:
        cli_args.extend(["--regions-file", str(regions_file)])
    for split in nn_splits or []:
        cli_args.extend(["--split", split])
    args = parser.parse_args(cli_args)
    df = evaluate_nn.evaluate(args).reset_index()
    df["model_family"] = "neural_network"
    df["model_name"] = label
    df["method_name"] = "evaluate_nn"
    if "split" not in df:
        df["split"] = "N/A"
    bounds = _collect_nn_region_bounds(config_path, regions_file)
    return df, bounds


def _merge_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True).reindex(columns=AGG_COLUMNS)
    sort_keys = [col for col in ("model_family", "model_name", "split", "region") if col in df.columns]
    if sort_keys:
        df.sort_values(sort_keys, inplace=True)
    return df


def _save_table(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix in {".csv", ""}:
        df.to_csv(output_path, index=False)
    elif suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format '{suffix}'. Use .csv, .json, or .parquet.")


def _print_region_tables(df: pd.DataFrame) -> None:
    if df.empty:
        print("No evaluation results to display.")
        return

    for region in sorted(df["region"].dropna().unique()):
        region_df = df[df["region"] == region].drop(columns=["region"])
        region_df = region_df[[col for col in REGION_DISPLAY_COLUMNS if col in region_df.columns]].reset_index(drop=True)
        print(f"\n======= Region: {region} =======")
        with pd.option_context("display.max_columns", None, "display.max_rows", None):
            print(region_df)


def _format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    metric_cols = [col for col in FLOAT_METRIC_COLUMNS if col in formatted.columns]
    for col in metric_cols:
        formatted[col] = formatted[col].astype(float).round(3)
    return formatted


def _with_region_labels(df: pd.DataFrame, bounds: RegionBoundsMap) -> pd.DataFrame:
    labelled = df.copy()
    labelled["region"] = labelled["region"].apply(lambda name: _format_region_label(name, bounds))
    return labelled


def _format_region_label(region: str, bounds: RegionBoundsMap) -> str:
    if not isinstance(region, str):
        return region
    info = bounds.get(region)
    if not info:
        return region
    lat_min, lat_max, lon_min, lon_max = info
    lat_range = _format_range(lat_min, lat_max, is_lat=True)
    lon_range = _format_range(lon_min, lon_max, is_lat=False)
    return f"{region} ({lat_range}, {lon_range})"


def _format_range(min_val: float, max_val: float, *, is_lat: bool) -> str:
    min_dir = _coord_direction(min_val, is_lat)
    max_dir = _coord_direction(max_val, is_lat)
    min_deg = _format_degree(abs(min_val))
    max_deg = _format_degree(abs(max_val))
    if min_dir == max_dir:
        return f"{min_deg}° - {max_deg}° {min_dir}"
    return f"{min_deg}° {min_dir} - {max_deg}° {max_dir}"


def _coord_direction(value: float, is_lat: bool) -> str:
    if is_lat:
        return "N" if value >= 0 else "S"
    return "E" if value >= 0 else "W"


def _format_degree(value: float) -> str:
    if float(value).is_integer():
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boosting-config",
        action="append",
        help="Optional label=path override for CatBoost configs. Repeatable.",
    )
    parser.add_argument(
        "--logreg-metadata",
        action="append",
        help="Optional label=path override for logistic regression metadata. Repeatable.",
    )
    parser.add_argument(
        "--nn-config",
        action="append",
        help="Optional label=path override for neural-network configs. Repeatable.",
    )
    parser.add_argument(
        "--nn-split",
        action="append",
        choices=("train", "val", "test", "all"),
        help="Restrict NN evaluation to specific splits. Repeatable. Default uses evaluate_nn heuristic.",
    )
    parser.add_argument(
        "--regions-file",
        type=Path,
        help="Regions YAML applied to every evaluation. Defaults depend on individual configs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/combined_metrics.csv"),
        help="Destination for the aggregated table (.csv, .json, .parquet). Defaults to outputs/evaluation/combined_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for per-region CSV files (mirrors the printed tables).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    boosting_entries = _ensure_exists(_parse_named_paths(args.boosting_config, DEFAULT_BOOSTING_CONFIGS), "Boosting config")
    logreg_entries = _ensure_exists(_parse_named_paths(args.logreg_metadata, DEFAULT_LOGREG_METADATA), "Logistic-regression metadata")
    nn_entries = _ensure_exists(_parse_named_paths(args.nn_config, DEFAULT_NN_CONFIGS), "Neural-network config")

    regions_file = args.regions_file
    nn_splits: Sequence[str] | None = tuple(args.nn_split) if args.nn_split else None

    frames: list[pd.DataFrame] = []
    region_bounds: RegionBoundsMap = {}

    for label, config_path in boosting_entries:
        df, bounds = _run_boosting(label, config_path, regions_file)
        frames.append(df)
        _merge_region_bounds(region_bounds, bounds)

    for label, metadata_path in logreg_entries:
        df, bounds = _run_logistic(label, metadata_path, regions_file)
        frames.append(df)
        _merge_region_bounds(region_bounds, bounds)

    for label, config_path in nn_entries:
        df, bounds = _run_neural_network(label, config_path, regions_file, nn_splits)
        frames.append(df)
        _merge_region_bounds(region_bounds, bounds)

    aggregated = _merge_frames(frames)
    formatted = _format_metrics(aggregated)
    labelled = _with_region_labels(formatted, region_bounds)

    _print_region_tables(labelled)

    if args.output:
        _save_table(labelled, args.output)

    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for region in sorted(labelled["region"].dropna().unique()):
            region_df = labelled[labelled["region"] == region].drop(columns=["region"])
            region_df = region_df[[col for col in REGION_DISPLAY_COLUMNS if col in region_df.columns]].reset_index(drop=True)
            safe_region = region.lower().replace(" ", "_").replace("/", "-")
            file_name = f"{safe_region}.csv"
            region_df.to_csv(output_dir / file_name, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
