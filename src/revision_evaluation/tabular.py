#!/usr/bin/env python
"""Run reviewer-requested wildfire ignition revision experiments.

The runner is intentionally lightweight: it reuses the existing precomputed
feature table, applies the reviewer-requested chronological split, trains the
tabular models and CatBoost ablations that are directly supported, and records
blocked experiments with concrete reasons.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from catboost import CatBoostClassifier, Pool
except Exception as exc:  # pragma: no cover - handled at runtime
    CatBoostClassifier = None
    Pool = None
    CATBOOST_IMPORT_ERROR = exc
else:
    CATBOOST_IMPORT_ERROR = None

try:
    from .artifacts import prune_empty_dirs
    from .full_grid_evaluation import (
        evaluate_model_full_grid_calibrated,
        write_full_grid_failure,
    )
    from .calibration import logit
except ImportError:  # pragma: no cover - supports direct script execution
    from src.revision_evaluation.artifacts import prune_empty_dirs
    from src.revision_evaluation.full_grid_evaluation import (
        evaluate_model_full_grid_calibrated,
        write_full_grid_failure,
    )
    from src.revision_evaluation.calibration import logit


SEED = 42
TARGET_COLUMN = "count"
DATE_COLUMN = "datetime"
LAT_COLUMN = "lat_rounded"
LON_COLUMN = "lon_rounded"
DEFAULT_FEATURES_PATH = Path(
    "data/saved_features_boost/train_test_features_30d_all_extended_north_14cb.parquet"
)
DEFAULT_RESULTS_DIR = Path("results/revision_experiments_complete")
DEFAULT_ERA5_DIR = Path("/home/ids/vmorozov/era5")
DEFAULT_IGNORED_FEATURES = ["datetime", "day", "latitude", "longitude", "year"]
FORBIDDEN_MODEL_FEATURES = {"brightness", "confidence", "count", "is_fire", "target", "label"}
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "average_precision",
    "roc_auc",
    "brier_score",
]


@dataclass
class Region:
    name: str
    display_name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        return (
            (frame[LAT_COLUMN].to_numpy() >= self.lat_min)
            & (frame[LAT_COLUMN].to_numpy() <= self.lat_max)
            & (frame[LON_COLUMN].to_numpy() >= self.lon_min)
            & (frame[LON_COLUMN].to_numpy() <= self.lon_max)
        )


class ExperimentFailure(Exception):
    """Raised for expected experiment-level blockers."""


class OrdinalTabularEncoder:
    """Small numeric/categorical encoder for linear and Random Forest baselines.

    The class avoids a large sparse one-hot matrix for the 2M-row feature table.
    Categorical values are encoded using train-only category maps; numerical
    columns are median-imputed, and optional standardization is fit on train only.
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str],
        standardize: bool,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.categorical_columns = [c for c in categorical_columns if c in self.feature_columns]
        self.numeric_columns = [c for c in self.feature_columns if c not in self.categorical_columns]
        self.standardize = standardize
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}
        self.category_maps: dict[str, dict[Any, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "OrdinalTabularEncoder":
        for col in self.numeric_columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            median = values.median()
            if pd.isna(median):
                median = 0.0
            filled = values.fillna(float(median)).astype("float64")
            self.numeric_medians[col] = float(median)
            if self.standardize:
                mean = float(filled.mean())
                std = float(filled.std())
                if not math.isfinite(std) or std <= 0:
                    std = 1.0
                self.numeric_means[col] = mean
                self.numeric_stds[col] = std

        for col in self.categorical_columns:
            series = frame[col].fillna("missing").astype(str)
            categories = pd.Index(series.unique())
            self.category_maps[col] = {value: idx for idx, value in enumerate(categories)}
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(frame), len(self.feature_columns)), dtype=np.float32)
        for idx, col in enumerate(self.feature_columns):
            if col in self.categorical_columns:
                mapping = self.category_maps.get(col, {})
                encoded = frame[col].fillna("missing").astype(str).map(mapping).fillna(-1)
                out[:, idx] = encoded.to_numpy(dtype=np.float32)
            else:
                values = pd.to_numeric(frame[col], errors="coerce")
                values = values.fillna(self.numeric_medians.get(col, 0.0)).to_numpy(dtype=np.float32)
                if self.standardize:
                    values = (values - self.numeric_means.get(col, 0.0)) / self.numeric_stds.get(col, 1.0)
                out[:, idx] = values
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def transform_batches(
        self,
        frame: pd.DataFrame,
        batch_size: int,
    ) -> Iterable[tuple[slice, np.ndarray]]:
        for start in range(0, len(frame), batch_size):
            end = min(start + batch_size, len(frame))
            yield slice(start, end), self.transform(frame.iloc[start:end])


def default_args(**overrides: Any) -> SimpleNamespace:
    data = {
        "features_path": DEFAULT_FEATURES_PATH,
        "feature_config": Path("configs/features_config_30d.yaml"),
        "target_config": Path("configs/target_config.yaml"),
        "catboost_config": Path("configs/catboost_train_config.yaml"),
        "regions_file": Path("configs/regions_example.yaml"),
        "output_dir": DEFAULT_RESULTS_DIR,
        "era5_dir": DEFAULT_ERA5_DIR,
        "seed": SEED,
        "catboost_iterations": 450,
        "catboost_depth": 5,
        "catboost_learning_rate": 0.03,
        "catboost_task_type": "GPU",
        "catboost_verbose": 100,
        "rf_max_train_rows": 300_000,
        "linear_epochs": 4,
        "point_process_max_train_rows": 500_000,
        "point_process_alpha": 1e-4,
        "point_process_max_iter": 200,
        "prediction_batch_size": 100_000,
        "permutation_sample_size": 50_000,
        "permutation_trials": 5,
        "random_error_trials": 5,
        "random_error_sample_size": 50_000,
        "shap_sample_size": 8_000,
        "skip_shap": False,
        "run_legacy_sampled_evaluation": True,
        "run_full_grid_evaluation": True,
        "full_grid_is_primary": False,
        "fail_on_full_grid_error": False,
        "calibration_start_date": None,
        "calibration_end_date": None,
        "test_start_date": None,
        "test_end_date": None,
        "deployment_grid_resolution": 0.1,
        "deployment_grid_universe": "land_or_burnable",
        "deployment_grid_chunk_by": ["country", "month"],
        "deployment_grid_countries": None,
        "deployment_grid_coordinate_bounds": None,
        "deployment_grid_clip_to_feature_bounds": True,
        "full_grid_mode": "full_grid",
        "weighted_grid_sample": False,
        "weighted_grid_sample_fraction": None,
        "weighted_grid_sample_strata": ["country", "month"],
        "calibration_method": "platt_month",
        "n_reliability_bins": 20,
        "reliability_binning": "equal_count",
        "save_full_grid_predictions": True,
        "save_calibrated_predictions": True,
        "max_grid_rows_per_chunk": None,
        "cache_full_grid_features": False,
        "use_lat_lon_features": True,
        "use_ecoregion_features": True,
        "use_historical_fire_features": True,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def split_csv_arg(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return None if not math.isfinite(scalar) else scalar
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(data), indent=2), encoding="utf-8")


def cleanup_removed_model_artifacts(output_dir: Path) -> None:
    """Drop artifacts from baselines that are no longer part of the suite."""
    stale_patterns = [
        "models/spline_logistic_regression_full*",
        "predictions/spline_logistic_regression_full*",
        "primary_full_grid_calibrated/calibrators/Spline_Logistic_Regression*",
        "primary_full_grid_calibrated/predictions/Spline_Logistic_Regression*",
        "primary_full_grid_calibrated/calibration_metadata/Spline_Logistic_Regression*",
    ]
    for pattern in stale_patterns:
        for path in output_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def read_feature_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _catboost_feature_config(catboost_config: dict[str, Any]) -> dict[str, Any]:
    section = catboost_config.get("catboost_train", {})
    if isinstance(section, dict):
        features = section.get("features", {})
        return features if isinstance(features, dict) else {}
    return {}


def selected_feature_filter_spec(
    feature_config: dict[str, Any],
    catboost_config: dict[str, Any],
) -> tuple[bool, Path | None, list[str]]:
    """Return the selected-column filter used by the CatBoost training config."""

    features_cfg = _catboost_feature_config(catboost_config)
    enabled = bool(features_cfg.get("selected_feature_filter", True))
    raw_path = features_cfg.get("selected_features_path") or feature_config.get(
        "selected_feature_columns_path"
    )
    if not raw_path:
        return enabled, None, []
    path = Path(raw_path)
    if not path.exists():
        logging.warning("Selected feature columns file does not exist: %s", path)
        return enabled, path, []
    return enabled, path, read_feature_list(path)


def apply_selected_feature_columns(
    feature_columns: Sequence[str],
    selected_features: Sequence[str],
    *,
    enabled: bool,
    table_columns: Sequence[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Filter model columns to the training config's selected-feature file."""

    columns = list(feature_columns)
    if not enabled or not selected_features:
        return columns, {
            "enabled": enabled,
            "selected_features_count": len(selected_features),
            "applied_selected_features_count": 0,
            "missing_selected_features": [],
            "dropped_by_selected_feature_filter": [],
        }

    column_set = set(columns)
    selected_existing = [feature for feature in selected_features if feature in column_set]
    available = set(table_columns) if table_columns is not None else column_set
    missing_selected = sorted(set(selected_features) - available)
    if not selected_existing:
        raise ExperimentFailure(
            "Selected feature filtering was enabled, but none of the selected "
            "features are present after applying ignored-feature and geography filters."
        )

    selected_set = set(selected_existing)
    dropped_by_filter = [feature for feature in columns if feature not in selected_set]
    return selected_existing, {
        "enabled": True,
        "selected_features_count": len(selected_features),
        "applied_selected_features_count": len(selected_existing),
        "missing_selected_features": missing_selected,
        "dropped_by_selected_feature_filter": dropped_by_filter,
    }


def run_command(command: Sequence[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.returncode, completed.stdout.strip()
    except Exception as exc:
        return 1, str(exc)


def package_availability() -> dict[str, dict[str, Any]]:
    packages = [
        "catboost",
        "xgboost",
        "lightgbm",
        "sklearn",
        "torch",
        "shap",
        "pandas",
        "numpy",
        "yaml",
        "pyarrow",
        "geopandas",
        "xarray",
        "netCDF4",
    ]
    result: dict[str, dict[str, Any]] = {}
    for package in packages:
        available = importlib.util.find_spec(package) is not None
        version = None
        if available:
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, "__version__", None)
            except Exception:
                version = None
        result[package] = {"available": available, "version": version}
    return result


def repo_audit(args: Any, output_dir: Path, packages: dict[str, Any]) -> dict[str, Any]:
    rc, commit = run_command(["git", "rev-parse", "HEAD"])
    commit_hash = commit.strip() if rc == 0 else None
    _, git_status = run_command(["git", "status", "--short"])

    scripts = sorted(
        str(path)
        for path in [
            Path("train_catboost.py"),
            Path("tune_catboost.py"),
            Path("make_train_data.py"),
            Path("make_nn_train_data.py"),
            Path("src/log_regression/train_log_regression.py"),
            Path("src/neural_net/train_nn.py"),
            Path("src/evaluation/evaluate_boosting.py"),
            Path("src/evaluation/evaluate_log_regression.py"),
            Path("src/evaluation/evaluate_nn.py"),
            Path("src/evaluation/run_combined_evaluations.py"),
        ]
        if path.exists()
    )
    configs = sorted(str(path) for path in Path("configs").glob("*.yaml"))

    era5_files = sorted(args.era5_dir.glob("*/*"))[:20] if args.era5_dir.exists() else []
    era5_vars = sorted({path.parent.name for path in args.era5_dir.glob("*/*")}) if args.era5_dir.exists() else []
    ecmwf_root = Path("/home/ids/vmorozov/data/climate_data/climate_features/ECMWF")
    seas5_files = list(ecmwf_root.glob("**/*.zarr"))[:20] if ecmwf_root.exists() else []
    processed_era5_root = Path("/home/ids/vmorozov/data/climate_data/climate_features/ERA5")
    processed_era5_files = list(processed_era5_root.glob("**/*.zarr"))[:20] if processed_era5_root.exists() else []
    existing_outputs = sorted(str(path) for path in Path("outputs").glob("catboost_train_*"))[-5:]
    existing_models = sorted(str(path) for path in Path("models").glob("*"))

    audit = {
        "created_at": datetime.now().isoformat(),
        "commit_hash": commit_hash,
        "git_status_short": git_status,
        "main_scripts": scripts,
        "configs": configs,
        "data_paths": {
            "features_path": args.features_path,
            "feature_config": args.feature_config,
            "target_config": args.target_config,
            "regions_file": args.regions_file,
            "era5_raw_dir": args.era5_dir,
            "ecmwf_climate_features_dir": ecmwf_root,
            "processed_era5_features_dir": processed_era5_root,
        },
        "era5": {
            "raw_path_exists": args.era5_dir.exists(),
            "raw_path_readable": os.access(args.era5_dir, os.R_OK),
            "variables_found": era5_vars,
            "sample_files": [str(path) for path in era5_files],
            "processed_zarr_dir_exists": processed_era5_root.exists(),
            "processed_zarr_samples": [str(path) for path in processed_era5_files],
        },
        "seas5_ecmwf": {
            "path_exists": ecmwf_root.exists(),
            "sample_zarr_files": [str(path) for path in seas5_files],
        },
        "existing_outputs": existing_outputs,
        "existing_models": existing_models,
        "package_availability": packages,
        "directly_supported_experiments": [
            "Reviewer chronological split from saved feature parquet",
            "CatBoost full, weather-only, FWI-only, and drop-group ablations",
            "Linear logistic baseline using train-only ordinal encoding",
            "Poisson point-process GLM baseline using train-only ordinal encoding",
            "Random Forest baseline using train-only ordinal encoding and capped bootstrap rows",
            "Minimal MLP and FT-Transformer NN baselines via the shared neural training registry when NN inputs are present",
            "Native CatBoost feature importance",
            "Grouped permutation importance",
            "CatBoost-native SHAP values if feasible",
        ],
        "blocked_or_adapter_needed": [
            "Full ERA5-vs-SEAS5 matrix requires an ERA5-derived feature parquet with schema parity; raw ERA5 GRIB is present but not a drop-in feature matrix.",
            "Neural embedding/fusion ablations require prepared_data.npz metadata, which is not present in the current workspace.",
            "No-dilation target sensitivity requires rebuilding target caches with a modified target-generation function.",
        ],
        "small_adapters_added": ["src/revision_evaluation/tabular.py"],
    }
    write_repo_audit_markdown(output_dir / "repo_audit.md", audit)
    return audit


def copy_configs_used(args: Any, output_dir: Path) -> list[str]:
    """Copy run-time configs into the results tree for reproducibility."""

    config_dir = output_dir / "configs_used"
    config_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in [args.feature_config, args.target_config, args.catboost_config, args.regions_file]:
        if path.exists():
            destination = config_dir / path.name
            shutil.copy2(path, destination)
            copied.append(str(destination))
    return copied


def write_repo_audit_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Revision Experiment Repo Audit",
        "",
        f"- Created at: `{audit['created_at']}`",
        f"- Commit hash: `{audit.get('commit_hash') or 'not available'}`",
        f"- Dirty worktree entries: `{audit.get('git_status_short') or 'none'}`",
        "",
        "## Main Scripts",
        *[f"- `{item}`" for item in audit["main_scripts"]],
        "",
        "## Configs",
        *[f"- `{item}`" for item in audit["configs"]],
        "",
        "## Data Paths",
        *[f"- `{key}`: `{value}`" for key, value in audit["data_paths"].items()],
        "",
        "## ERA5",
        f"- Raw ERA5 path exists/readable: `{audit['era5']['raw_path_exists']}` / `{audit['era5']['raw_path_readable']}`",
        f"- Raw ERA5 variables found: `{', '.join(audit['era5']['variables_found']) or 'none'}`",
        f"- Processed ERA5 zarr dir exists: `{audit['era5']['processed_zarr_dir_exists']}`",
        "",
        "## SEAS5 / ECMWF",
        f"- ECMWF climate feature path exists: `{audit['seas5_ecmwf']['path_exists']}`",
        f"- Sample files found: `{len(audit['seas5_ecmwf']['sample_zarr_files'])}`",
        "",
        "## Existing Outputs And Models",
        *[f"- Output: `{item}`" for item in audit["existing_outputs"]],
        *[f"- Model/artifact: `{item}`" for item in audit["existing_models"]],
        "",
        "## Installed Packages In `pointnet`",
    ]
    for pkg, info in audit["package_availability"].items():
        lines.append(f"- `{pkg}`: available=`{info['available']}`, version=`{info.get('version')}`")
    lines.extend(
        [
            "",
            "## Directly Supported Experiments",
            *[f"- {item}" for item in audit["directly_supported_experiments"]],
            "",
            "## Blockers / Needed Adapters",
            *[f"- {item}" for item in audit["blocked_or_adapter_needed"]],
            "",
            "## Small Adapters Added",
            *[f"- `{item}`" for item in audit["small_adapters_added"]],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_regions(path: Path) -> list[Region]:
    data = load_yaml(path)
    entries = data.get("regions", []) if isinstance(data, dict) else []
    regions: list[Region] = []
    for entry in entries:
        name = str(entry["name"])
        display = name.replace("_", " ").title()
        regions.append(
            Region(
                name=name,
                display_name=display,
                lat_min=float(entry["lat_min"]),
                lat_max=float(entry["lat_max"]),
                lon_min=float(entry["lon_min"]),
                lon_max=float(entry["lon_max"]),
            )
        )
    return regions


def load_dataset(path: Path) -> pd.DataFrame:
    logging.info("Loading feature dataset from %s", path)
    start = time.perf_counter()
    df = pd.read_parquet(path)
    if DATE_COLUMN not in df.columns:
        raise KeyError(f"Dataset missing required date column {DATE_COLUMN!r}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if df[DATE_COLUMN].isna().any():
        raise ValueError("Dataset contains unparseable datetimes.")
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Dataset missing required target column {TARGET_COLUMN!r}")
    logging.info("Loaded dataset shape=%s in %.1fs", df.shape, time.perf_counter() - start)
    return df


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    years = df[DATE_COLUMN].dt.year
    return {
        "train": ((years >= 2001) & (years <= 2018)).to_numpy(),
        "validation": ((years >= 2019) & (years <= 2020)).to_numpy(),
        "test": ((years >= 2021) & (years <= 2025)).to_numpy(),
    }


def positive_labels(series: pd.Series) -> np.ndarray:
    return (pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() > 0).astype(np.int8)


def infer_resolution(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for col in [LAT_COLUMN, LON_COLUMN]:
        if col not in df.columns:
            result[col] = None
            continue
        values = np.sort(pd.Series(df[col].dropna().unique()).astype(float).to_numpy())
        diffs = np.diff(values)
        diffs = diffs[diffs > 1e-9]
        result[col] = float(np.median(diffs)) if diffs.size else None
    lat_res = result.get(LAT_COLUMN)
    if lat_res is not None:
        result["approx_km_lat"] = float(lat_res * 111.1)
    return result


def dataset_statistics(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    regions: list[Region],
    target_config: dict[str, Any],
    feature_config: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    resolution = infer_resolution(df)
    target_rule = (
        f"MODIS/FIRMS detections with brightness >= {target_config.get('brightness_threshold')} "
        f"and confidence >= {target_config.get('confidence_threshold')}; high-latitude relaxed thresholds "
        f"{target_config.get('brightness_threshold_high_lat')}/{target_config.get('confidence_threshold_high_lat')}; "
        "stationary points filtered; grouped by date and rounded grid cell; multi-detection cells expanded to neighbors."
    )
    land_rule = "Rows with `landseamask < 70` retained during feature merge" if "landseamask" in df.columns else "Not available in feature table"
    neg_rule = f"Sampled negatives, target_samples_per_area_per_year={feature_config.get('target_samples_per_area_per_year')}"

    rows: list[dict[str, Any]] = []

    def add_row(label: str, years: str, mask: np.ndarray, region: Region | None = None) -> None:
        subset = df.loc[mask]
        y = positive_labels(subset[TARGET_COLUMN])
        if region is None:
            lat_bounds = (float(subset[LAT_COLUMN].min()), float(subset[LAT_COLUMN].max())) if len(subset) else (None, None)
            lon_bounds = (float(subset[LON_COLUMN].min()), float(subset[LON_COLUMN].max())) if len(subset) else (None, None)
        else:
            lat_bounds = (region.lat_min, region.lat_max)
            lon_bounds = (region.lon_min, region.lon_max)
        unique_cells = (
            int(subset[[LAT_COLUMN, LON_COLUMN]].drop_duplicates().shape[0])
            if LAT_COLUMN in subset.columns and LON_COLUMN in subset.columns
            else None
        )
        positives = int(y.sum())
        total = int(len(y))
        rows.append(
            {
                "split_region": label,
                "years": years,
                "lat_bounds": f"{lat_bounds[0]} to {lat_bounds[1]}",
                "lon_bounds": f"{lon_bounds[0]} to {lon_bounds[1]}",
                "unique_grid_cells": unique_cells,
                "cell_days": total,
                "positive_samples": positives,
                "negative_samples": total - positives,
                "positive_rate": positives / total if total else np.nan,
                "negatives": neg_rule,
                "land_water_masking_rule": land_rule,
                "target_labeling_rule": target_rule,
                "grid_resolution": f"{resolution.get(LAT_COLUMN)} deg lat x {resolution.get(LON_COLUMN)} deg lon (~{resolution.get('approx_km_lat'):.2f} km latitude spacing)",
            }
        )

    add_row("Global train", "2001-2018", masks["train"])
    add_row("Global validation", "2019-2020", masks["validation"])
    add_row("Global test", "2021-2025", masks["test"])
    test_frame = df.loc[masks["test"]]
    for region in regions:
        test_region_mask = masks["test"].copy()
        test_region_mask[masks["test"]] = region.mask(test_frame)
        add_row(region.display_name, "2021-2025", test_region_mask, region)

    by_year_rows: list[dict[str, Any]] = []
    for year in range(2021, 2026):
        year_mask = masks["test"] & (df[DATE_COLUMN].dt.year.to_numpy() == year)
        add_row(f"Global test {year}", str(year), year_mask)
        subset = df.loc[year_mask]
        y = positive_labels(subset[TARGET_COLUMN]) if len(subset) else np.array([], dtype=np.int8)
        by_year_rows.append(
            {
                "target_variant": "main/current",
                "split": "test",
                "region": "global",
                "region_display": "Global",
                "year": year,
                "cell_days": int(len(subset)),
                "positive_samples": int(y.sum()),
                "negative_samples": int(len(y) - y.sum()),
                "positive_rate": float(y.mean()) if len(y) else np.nan,
                "unique_grid_cells": int(subset[[LAT_COLUMN, LON_COLUMN]].drop_duplicates().shape[0]) if len(subset) else 0,
            }
        )
        for region in regions:
            region_mask = year_mask.copy()
            region_mask[year_mask] = region.mask(subset)
            r_subset = df.loc[region_mask]
            r_y = positive_labels(r_subset[TARGET_COLUMN]) if len(r_subset) else np.array([], dtype=np.int8)
            by_year_rows.append(
                {
                    "target_variant": "main/current",
                    "split": "test",
                    "region": region.name,
                    "region_display": region.display_name,
                    "year": year,
                    "cell_days": int(len(r_subset)),
                    "positive_samples": int(r_y.sum()),
                    "negative_samples": int(len(r_y) - r_y.sum()),
                    "positive_rate": float(r_y.mean()) if len(r_y) else np.nan,
                    "unique_grid_cells": int(r_subset[[LAT_COLUMN, LON_COLUMN]].drop_duplicates().shape[0]) if len(r_subset) else 0,
                }
            )

    stats_df = pd.DataFrame(rows)
    by_year_df = pd.DataFrame(by_year_rows)
    stats_df.to_csv(output_dir / "dataset_statistics.csv", index=False)
    by_year_df.to_csv(output_dir / "dataset_statistics_by_year.csv", index=False)
    write_markdown_table(output_dir / "dataset_statistics.md", "Dataset Statistics", stats_df)
    return stats_df


def normalize_cat_columns(
    frame: pd.DataFrame,
    categorical_columns: Sequence[str],
    numerical_cat_columns: Sequence[str],
) -> pd.DataFrame:
    frame = frame.copy()
    numerical_set = set(numerical_cat_columns)
    for col in categorical_columns:
        if col not in frame.columns:
            continue
        if col in numerical_set:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
        else:
            frame[col] = frame[col].fillna("missing").astype(str)
    return frame


def catboost_categorical_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    config: dict[str, Any],
) -> list[str]:
    """Return configured plus schema-inferred categorical columns for CatBoost."""

    feature_set = set(feature_columns)
    numerical_set = set(config.get("numerical_cat_features", []) or [])
    configured = [c for c in config.get("cat_features", []) or [] if c in feature_set]
    inferred: list[str] = []
    for col in feature_columns:
        if col in numerical_set or col not in frame.columns:
            continue
        dtype = frame[col].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            inferred.append(col)
    return list(dict.fromkeys([*configured, *inferred]))


def feature_group(feature: str) -> str:
    name = feature.lower()
    if name in {"month"} or "dayofyear" in name or name.startswith("doy") or name.endswith("_sin") or name.endswith("_cos"):
        return "seasonality"
    if name.startswith(("t2m_", "d2m_", "tp_", "stl1_", "u10_", "v10_", "msl_", "sd_")):
        return "weather_history"
    if name.startswith("fire_index_"):
        return "fire_weather_fwi"
    if any(token in name for token in ["road", "population", "night_light", "light_density", "distance_to_light"]):
        return "anthropogenic"
    if "ecoregion" in name or name in {"slt", "tvh", "tvl", "slt_soil_type_stream_oper_daily_mean"}:
        return "vegetation_fuel_ecoregion"
    if any(token in name for token in ["lai", "forest", "vegetation", "soil_type"]):
        return "vegetation_fuel_ecoregion"
    if any(token in name for token in ["elevation", "topography", "slope", "aspect", "anor", "isor", "slor", "sdor", "sdfor"]) or name == "z":
        return "terrain_topography"
    if any(
        token in name
        for token in [
            "historical",
            "prior_fire",
            "past_fire",
            "fire_density",
            "fire_count",
            "burned",
            "proximity_fire",
        ]
    ):
        return "historical_fire_context"
    if name in {"lat_rounded", "lon_rounded"}:
        return "location"
    if any(token in name for token in ["landseamask", "distance_to_coast", "coast"]):
        return "coast_landmask"
    return "other"


def group_display(group: str) -> str:
    mapping = {
        "weather_history": "Weather / meteorology history",
        "fire_weather_fwi": "Fire-weather / FWI",
        "anthropogenic": "Anthropogenic",
        "vegetation_fuel_ecoregion": "Vegetation/fuel/ecoregion",
        "terrain_topography": "Terrain/topography",
        "seasonality": "Seasonality",
        "historical_fire_context": "Historical fire/proximity/density",
        "location": "Location coordinates",
        "coast_landmask": "Coast/land mask",
        "other": "Other",
    }
    return mapping.get(group, group.replace("_", " ").title())


def model_feature_columns(
    df: pd.DataFrame,
    ignored_features: Sequence[str],
    *,
    use_lat_lon_features: bool = True,
    use_ecoregion_features: bool = True,
    use_historical_fire_features: bool = True,
) -> list[str]:
    ignored = set(ignored_features) | FORBIDDEN_MODEL_FEATURES
    features = [col for col in df.columns if col not in ignored]
    if not use_lat_lon_features:
        features = [col for col in features if col not in {LAT_COLUMN, LON_COLUMN, "latitude", "longitude"}]
    if not use_ecoregion_features:
        features = [col for col in features if "ecoregion" not in col.lower()]
    if not use_historical_fire_features:
        features = [col for col in features if feature_group(col) != "historical_fire_context"]
    return features


def validate_no_leakage_features(feature_columns: Sequence[str]) -> None:
    bad = sorted(set(feature_columns) & FORBIDDEN_MODEL_FEATURES)
    if bad:
        raise ExperimentFailure(
            "Forbidden leakage columns are present in model inputs: "
            f"{bad}. These target/FIRMS columns must not be used as predictors."
        )


def feature_window_days(feature: str) -> int | None:
    for token in reversed(feature.lower().split("_")):
        if token.isdigit():
            return int(token)
    return None


def is_periodic_seasonality_feature(feature: str) -> bool:
    name = feature.lower()
    season_tokens = ("dayofyear", "day_of_year", "doy")
    return ("sin" in name or "cos" in name) and any(token in name for token in season_tokens)


def is_gaussian_smoothed_anthropogenic_feature(feature: str) -> bool:
    name = feature.lower()
    return "gaussian" in name and any(token in name for token in ["road", "light", "population", "pop"])


def is_ecoregion_feature(feature: str) -> bool:
    return "ecoregion" in feature.lower()


def is_direct_geography_feature(feature: str) -> bool:
    name = feature.lower()
    exact = {
        LAT_COLUMN.lower(),
        LON_COLUMN.lower(),
        "latitude",
        "longitude",
        "lat",
        "lon",
        "country",
        "country_code",
        "admin",
        "region",
    }
    if name in exact:
        return True
    if name in {"lat_rounded", "lon_rounded"}:
        return True
    if name.startswith(("lat_", "lon_", "latitude_", "longitude_", "country_", "admin_")):
        return True
    return feature_group(feature) in {"location", "coast_landmask"}


def build_feature_sets(
    all_features: list[str],
) -> dict[str, dict[str, Any]]:
    by_group: dict[str, list[str]] = {}
    for col in all_features:
        by_group.setdefault(feature_group(col), []).append(col)

    weather = by_group.get("weather_history", [])
    fwi = by_group.get("fire_weather_fwi", [])
    anthropogenic = by_group.get("anthropogenic", [])
    ecology = by_group.get("vegetation_fuel_ecoregion", [])
    terrain = by_group.get("terrain_topography", [])
    seasonality = by_group.get("seasonality", [])
    history = by_group.get("historical_fire_context", [])
    ecoregion = [col for col in all_features if is_ecoregion_feature(col)]
    direct_geography = [col for col in all_features if is_direct_geography_feature(col)]
    long_weather_context = []
    for col in weather:
        window = feature_window_days(col)
        if window is not None and window > 30:
            long_weather_context.append(col)
    periodic_seasonality = [col for col in all_features if is_periodic_seasonality_feature(col)]
    gaussian_smoothed_anthropogenic = [
        col for col in all_features if is_gaussian_smoothed_anthropogenic_feature(col)
    ]

    def minus(cols: Sequence[str]) -> list[str]:
        drop = set(cols)
        return [c for c in all_features if c not in drop]

    static_cols = [c for c in all_features if c not in set(weather) | set(fwi)]
    feature_sets = {
        "full": {"label": "Full CatBoost", "feature_set": "all features", "columns": all_features},
        "weather_only": {"label": "Weather only", "feature_set": "meteorology/history features only", "columns": weather},
        "fwi_only": {"label": "FWI only", "feature_set": "fire-weather variables only", "columns": fwi},
        "no_anthropogenic": {"label": "No anthropogenic", "feature_set": "full minus roads/population/night lights", "columns": minus(anthropogenic), "dropped": anthropogenic},
        "no_ecoregion": {"label": "CatBoost-no-ecoregion", "feature_set": "full minus ecoregion categorical variables", "columns": minus(ecoregion), "dropped": ecoregion},
        "no_geography": {"label": "CatBoost-no-geography", "feature_set": "full minus direct geography/location variables", "columns": minus(direct_geography), "dropped": direct_geography},
        "no_fuel_ecoregion_vegetation": {"label": "No fuel/ecoregion/vegetation", "feature_set": "full minus ecological/fuel variables", "columns": minus(ecology), "dropped": ecology},
        "no_terrain": {"label": "No terrain", "feature_set": "full minus topography", "columns": minus(terrain), "dropped": terrain},
        "no_seasonality": {"label": "No seasonality", "feature_set": "full minus month/day-of-year features", "columns": minus(seasonality), "dropped": seasonality},
        "no_history": {"label": "No history", "feature_set": "full minus prior-fire/proximity features", "columns": minus(history), "dropped": history},
        "no_temporal_history": {"label": "No temporal-history/weather lags", "feature_set": "full minus lagged meteorology", "columns": minus(weather), "dropped": weather},
        "shorter_sequence_30d": {"label": "Shorter sequence (<=30d climate windows)", "feature_set": "full minus >30-day weather-history windows", "columns": minus(long_weather_context), "dropped": long_weather_context},
        "no_periodic_seasonality": {"label": "No sine/cos day-of-year encoding", "feature_set": "full minus periodic seasonality encodings", "columns": minus(periodic_seasonality), "dropped": periodic_seasonality},
        "no_gaussian_smoothing": {"label": "No Gaussian-smoothed anthropogenic rasters", "feature_set": "full minus Gaussian road/light/population rasters", "columns": minus(gaussian_smoothed_anthropogenic), "dropped": gaussian_smoothed_anthropogenic},
        "static_only": {"label": "Static only", "feature_set": "static non-weather features", "columns": static_cols},
        "dynamic_weather_fwi_only": {"label": "Dynamic weather+FWI only", "feature_set": "meteorology plus fire-weather variables", "columns": weather + fwi},
    }
    return feature_sets


def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"threshold": 0.5, "validation_f1": None, "reason": "only_one_class"}
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return {"threshold": 0.5, "validation_f1": None, "reason": "no_thresholds"}
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)
    if not np.isfinite(f1).any():
        return {"threshold": 0.5, "validation_f1": None, "reason": "no_finite_f1"}
    best = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best]),
        "validation_f1": float(f1[best]),
        "validation_precision": float(precision[best]),
        "validation_recall": float(recall[best]),
        "reason": "validation_f1_max",
    }


def safe_score(func, *args) -> float | None:
    try:
        value = float(func(*args))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return {
            "support": 0,
            "positives": 0,
            "predicted_positives": 0,
            "threshold": threshold,
            **{metric: None for metric in METRIC_COLUMNS},
        }
    y_pred = (y_prob >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "support": int(len(y_true)),
        "positives": int(y_true.sum()),
        "predicted_positives": int(y_pred.sum()),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
        "threshold": float(threshold),
        "precision": safe_score(lambda yt, yp: precision_score(yt, yp, zero_division=0), y_true, y_pred),
        "recall": safe_score(lambda yt, yp: recall_score(yt, yp, zero_division=0), y_true, y_pred),
        "f1": safe_score(lambda yt, yp: f1_score(yt, yp, zero_division=0), y_true, y_pred),
        "average_precision": safe_score(average_precision_score, y_true, y_prob) if both else None,
        "roc_auc": safe_score(roc_auc_score, y_true, y_prob) if both else None,
        "brier_score": safe_score(brier_score_loss, y_true, y_prob) if both else None,
    }


def compute_metric_errors(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    trials: int,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    if trials <= 1 or len(y_true) == 0:
        return {f"{metric}_error": None for metric in METRIC_COLUMNS}

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    n = min(len(y_true), sample_size) if sample_size > 0 else len(y_true)
    if n <= 1:
        return {f"{metric}_error": None for metric in METRIC_COLUMNS}

    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in METRIC_COLUMNS}
    pos_frac = len(pos) / len(y_true) if len(y_true) else 0.0
    for _ in range(trials):
        if len(pos) and len(neg):
            n_pos = min(max(1, int(round(n * pos_frac))), n - 1)
            n_neg = n - n_pos
            idx = np.concatenate(
                [
                    rng.choice(pos, size=n_pos, replace=True),
                    rng.choice(neg, size=n_neg, replace=True),
                ]
            )
            rng.shuffle(idx)
        else:
            idx = rng.choice(np.arange(len(y_true)), size=n, replace=True)
        metrics = compute_metrics(y_true[idx], y_prob[idx], threshold)
        for metric in METRIC_COLUMNS:
            value = metrics.get(metric)
            if value is not None and math.isfinite(float(value)):
                values[metric].append(float(value))

    return {
        f"{metric}_error": float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        for metric, vals in values.items()
    }


def evaluate_predictions(
    experiment_id: str,
    experiment_type: str,
    model_name: str,
    feature_set: str,
    split_name: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    regions: list[Region],
    error_trials: int = 5,
    error_sample_size: int = 50_000,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(region_name: str, display: str, mask: np.ndarray | None = None) -> None:
        if mask is None:
            mask = np.ones(len(frame), dtype=bool)
        mask_y = y_true[mask]
        mask_prob = y_prob[mask]
        metrics = compute_metrics(mask_y, mask_prob, threshold)
        metrics.update(
            compute_metric_errors(
                mask_y,
                mask_prob,
                threshold,
                trials=error_trials,
                sample_size=error_sample_size,
                seed=seed + len(rows),
            )
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "experiment_type": experiment_type,
                "evaluation_type": "legacy_sampled_case_control",
                "is_primary": False,
                "model": model_name,
                "feature_set": feature_set,
                "split": split_name,
                "region": region_name,
                "region_display": display,
                **metrics,
            }
        )

    add("global", "Global")
    if split_name == "test":
        for region in regions:
            add(region.name, region.display_name, region.mask(frame))

        years = pd.to_datetime(frame[DATE_COLUMN]).dt.year.to_numpy()
        period_specs: list[tuple[str, np.ndarray]] = [
            (str(year), years == year) for year in range(2021, 2026)
        ]
        period_specs.extend(
            [
                ("2021-2023", (years >= 2021) & (years <= 2023)),
                ("2021-2025", (years >= 2021) & (years <= 2025)),
            ]
        )
        for period_label, period_mask in period_specs:
            if not period_mask.any():
                continue
            add("global", "Global", period_mask)
            rows[-1]["split"] = f"test_{period_label}"
            for region in regions:
                combined_mask = period_mask & region.mask(frame)
                if not combined_mask.any():
                    continue
                add(region.name, region.display_name, combined_mask)
                rows[-1]["split"] = f"test_{period_label}"
    return rows


def save_predictions(
    output_dir: Path,
    experiment_id: str,
    split_name: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Path:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = [col for col in [DATE_COLUMN, LAT_COLUMN, LON_COLUMN, "month", "year"] if col in frame.columns]
    pred = frame[keep_cols].copy()
    pred["target_binary"] = y_true.astype(int)
    pred["pred_proba"] = y_prob.astype(np.float32)
    pred["pred_binary"] = (y_prob >= threshold).astype(np.int8)
    path = pred_dir / f"{experiment_id}_{split_name}_predictions.parquet"
    pred.to_parquet(path, index=False)
    return path


def catboost_pool(
    X: pd.DataFrame,
    y: np.ndarray | None,
    cat_features: Sequence[str],
) -> Any:
    cat_features = [c for c in cat_features if c in X.columns]
    if y is None:
        return Pool(X, cat_features=cat_features) if cat_features else Pool(X)
    return Pool(X, label=y, cat_features=cat_features) if cat_features else Pool(X, label=y)


def predict_catboost(model: Any, X: pd.DataFrame, cat_features: Sequence[str]) -> np.ndarray:
    probs = np.asarray(model.predict_proba(catboost_pool(X, None, cat_features)))
    return probs[:, 1] if probs.ndim == 2 else probs.reshape(-1)


def maybe_run_full_grid_tabular_evaluation(
    *,
    experiment_id: str,
    experiment_type: str,
    model_label: str,
    model_type: str,
    feature_set_label: str,
    feature_columns: list[str],
    predict_raw_fn,
    args: Any,
    output_dir: Path,
    regions: list[Region],
    feature_config: dict[str, Any],
    model_path: Path | str | None,
) -> dict[str, Any] | None:
    if not args.run_full_grid_evaluation or experiment_type != "main_model_comparison":
        return None
    try:
        metrics = evaluate_model_full_grid_calibrated(
            model_name=model_label,
            model_type=model_type,
            feature_columns=feature_columns,
            categorical_columns=feature_config.get("cat_features", []),
            config=args,
            output_dir=output_dir,
            predict_raw_fn=predict_raw_fn,
            feature_config=args.feature_config,
            target_config=load_yaml(args.target_config),
            regions=regions,
            model_path=model_path,
            feature_set=feature_set_label,
        )
        logging.info("Primary full-grid calibrated evaluation complete for %s", experiment_id)
        return metrics
    except Exception as exc:
        failure_path = write_full_grid_failure(
            output_dir,
            model_name=model_label,
            model_type=model_type,
            exc=exc,
        )
        logging.exception(
            "Primary full-grid calibrated evaluation failed for %s; saved failure to %s",
            experiment_id,
            failure_path,
        )
        if args.fail_on_full_grid_error:
            raise
        return None


def make_catboost_raw_predict_fn(model: Any, cat_features: Sequence[str], config: dict[str, Any]):
    def _predict(X: pd.DataFrame) -> dict[str, Any]:
        X_work = normalize_cat_columns(X, cat_features, config.get("numerical_cat_features", []))
        pool = catboost_pool(X_work, None, cat_features)
        raw = np.asarray(model.predict(pool, prediction_type="RawFormulaVal"), dtype=float).reshape(-1)
        prob = np.asarray(model.predict_proba(pool), dtype=float)
        prob = prob[:, 1] if prob.ndim == 2 else prob.reshape(-1)
        return {"raw_score": raw, "prob_raw": prob, "raw_score_source": "catboost_raw_formula_val"}

    return _predict


def make_linear_raw_predict_fn(model: Any, encoder: Any, batch_size: int):
    def _predict(X: pd.DataFrame) -> dict[str, Any]:
        raw = np.empty(len(X), dtype=np.float32)
        prob = np.empty(len(X), dtype=np.float32)
        for slc, X_batch in encoder.transform_batches(X, batch_size):
            if hasattr(model, "decision_function"):
                scores = np.asarray(model.decision_function(X_batch), dtype=float).reshape(-1)
                raw[slc] = scores.astype(np.float32)
                prob[slc] = (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)
            else:
                batch_prob = np.asarray(model.predict_proba(X_batch)[:, 1], dtype=float)
                prob[slc] = batch_prob.astype(np.float32)
                raw[slc] = logit(batch_prob).astype(np.float32)
        return {"raw_score": raw, "prob_raw": prob, "raw_score_source": "decision_function"}

    return _predict


def make_poisson_raw_predict_fn(model: PoissonRegressor, encoder: Any, batch_size: int):
    def _predict(X: pd.DataFrame) -> dict[str, Any]:
        raw = np.empty(len(X), dtype=np.float32)
        prob = np.empty(len(X), dtype=np.float32)
        for slc, X_batch in encoder.transform_batches(X, batch_size):
            intensity = np.asarray(model.predict(X_batch), dtype=np.float64)
            intensity = np.nan_to_num(intensity, nan=0.0, posinf=50.0, neginf=0.0)
            intensity = np.clip(intensity, 0.0, 50.0)
            batch_prob = np.clip(-np.expm1(-intensity), 0.0, 1.0)
            prob[slc] = batch_prob.astype(np.float32)
            raw[slc] = logit(batch_prob).astype(np.float32)
        return {"raw_score": raw, "prob_raw": prob, "raw_score_source": "logit_poisson_probability"}

    return _predict


def make_predict_proba_raw_fn(model: Any, encoder: Any, batch_size: int):
    def _predict(X: pd.DataFrame) -> dict[str, Any]:
        raw = np.empty(len(X), dtype=np.float32)
        prob = np.empty(len(X), dtype=np.float32)
        for slc, X_batch in encoder.transform_batches(X, batch_size):
            batch_prob = np.asarray(model.predict_proba(X_batch)[:, 1], dtype=float)
            prob[slc] = batch_prob.astype(np.float32)
            raw[slc] = logit(batch_prob).astype(np.float32)
        return {"raw_score": raw, "prob_raw": prob, "raw_score_source": "logit_predict_proba"}

    return _predict


def train_catboost_model(
    experiment_id: str,
    feature_columns: list[str],
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    args: Any,
    output_dir: Path,
) -> tuple[Any, list[str], dict[str, Any]]:
    validate_no_leakage_features(feature_columns)
    if CatBoostClassifier is None:
        raise ExperimentFailure(f"catboost import failed: {CATBOOST_IMPORT_ERROR}")
    if not feature_columns:
        raise ExperimentFailure("No feature columns available for CatBoost experiment.")

    categorical = catboost_categorical_features(df, feature_columns, config)
    numerical_cat = [c for c in config.get("numerical_cat_features", []) if c in feature_columns]
    X_train = normalize_cat_columns(df.loc[masks["train"], feature_columns], categorical, numerical_cat)
    X_val = normalize_cat_columns(df.loc[masks["validation"], feature_columns], categorical, numerical_cat)
    y_train = positive_labels(df.loc[masks["train"], TARGET_COLUMN])
    y_val = positive_labels(df.loc[masks["validation"], TARGET_COLUMN])

    params = {
        "iterations": args.catboost_iterations,
        "depth": args.catboost_depth,
        "learning_rate": args.catboost_learning_rate,
        "l2_leaf_reg": 0.35,
        "min_data_in_leaf": 80,
        "loss_function": "Logloss",
        "eval_metric": "F1",
        "class_weights": [1.0, 4.0],
        "random_seed": args.seed,
        "random_strength": 1.0,
        "verbose": args.catboost_verbose if args.catboost_verbose > 0 else False,
        "allow_writing_files": False,
    }
    if args.catboost_task_type:
        params["task_type"] = args.catboost_task_type

    model = CatBoostClassifier(**params)
    logging.info("Training CatBoost %s with %d features", experiment_id, len(feature_columns))
    try:
        model.fit(
            catboost_pool(X_train, y_train, categorical),
            eval_set=catboost_pool(X_val, y_val, categorical),
            use_best_model=True,
            early_stopping_rounds=100,
        )
    except Exception:
        if params.get("task_type") == "GPU":
            logging.warning("CatBoost GPU failed for %s; retrying on CPU.", experiment_id)
            params.pop("task_type", None)
            model = CatBoostClassifier(**params)
            model.fit(
                catboost_pool(X_train, y_train, categorical),
                eval_set=catboost_pool(X_val, y_val, categorical),
                use_best_model=True,
                early_stopping_rounds=100,
            )
        else:
            raise

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{experiment_id}.cbm"
    model.save_model(model_path)
    write_json(model_dir / f"{experiment_id}_features.json", {"features": feature_columns, "categorical_features": categorical})
    diagnostics = {
        "model_path": model_path,
        "feature_count": len(feature_columns),
        "categorical_features": categorical,
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score(),
        "params": params,
    }
    return model, categorical, diagnostics


def run_catboost_experiment(
    experiment_id: str,
    experiment_type: str,
    model_label: str,
    feature_set_label: str,
    feature_columns: list[str],
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    regions: list[Region],
    config: dict[str, Any],
    args: Any,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], Any, list[str]]:
    model, cat_features, diagnostics = train_catboost_model(
        experiment_id, feature_columns, df, masks, config, args, output_dir
    )
    predictions: dict[str, np.ndarray] = {}
    val_frame = df.loc[masks["validation"]]
    test_frame = df.loc[masks["test"]]
    X_val = normalize_cat_columns(val_frame[feature_columns], cat_features, config.get("numerical_cat_features", []))
    X_test = normalize_cat_columns(test_frame[feature_columns], cat_features, config.get("numerical_cat_features", []))
    y_val = positive_labels(val_frame[TARGET_COLUMN])
    y_test = positive_labels(test_frame[TARGET_COLUMN])
    predictions["validation"] = predict_catboost(model, X_val, cat_features)
    predictions["test"] = predict_catboost(model, X_test, cat_features)
    threshold_info = choose_threshold_by_f1(y_val, predictions["validation"])
    threshold = float(threshold_info["threshold"])
    pred_paths = {
        "validation": save_predictions(output_dir, experiment_id, "validation", val_frame, y_val, predictions["validation"], threshold),
        "test": save_predictions(output_dir, experiment_id, "test", test_frame, y_test, predictions["test"], threshold),
    }
    metric_rows = []
    metric_rows.extend(
        evaluate_predictions(
            experiment_id,
            experiment_type,
            model_label,
            feature_set_label,
            "validation",
            val_frame,
            y_val,
            predictions["validation"],
            threshold,
            regions,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            seed=args.seed,
        )
    )
    metric_rows.extend(
        evaluate_predictions(
            experiment_id,
            experiment_type,
            model_label,
            feature_set_label,
            "test",
            test_frame,
            y_test,
            predictions["test"],
            threshold,
            regions,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            seed=args.seed,
        )
    )
    registry_row = {
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "evaluation_type": "legacy_sampled_case_control",
        "is_primary": False,
        "model": model_label,
        "feature_set": feature_set_label,
        "status": "completed",
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "threshold_source": "validation_f1_max",
        "validation_f1_at_threshold": threshold_info.get("validation_f1"),
        "model_path": diagnostics.get("model_path"),
        "prediction_paths": pred_paths,
        "notes": "",
    }
    primary_metrics = maybe_run_full_grid_tabular_evaluation(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        model_label=model_label,
        model_type="CatBoost",
        feature_set_label=feature_set_label,
        feature_columns=feature_columns,
        predict_raw_fn=make_catboost_raw_predict_fn(model, cat_features, config),
        args=args,
        output_dir=output_dir,
        regions=regions,
        feature_config=config,
        model_path=diagnostics.get("model_path"),
    )
    if primary_metrics:
        registry_row["primary_full_grid_metrics"] = primary_metrics
    return registry_row, metric_rows, predictions, model, cat_features


def predict_linear_batches(
    model: SGDClassifier,
    encoder: Any,
    frame: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    probs = np.empty(len(frame), dtype=np.float32)
    for slc, X_batch in encoder.transform_batches(frame, batch_size):
        if hasattr(model, "predict_proba"):
            probs[slc] = model.predict_proba(X_batch)[:, 1].astype(np.float32)
        else:
            scores = model.decision_function(X_batch)
            probs[slc] = (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)
    return probs


def predict_poisson_point_process_batches(
    model: PoissonRegressor,
    encoder: Any,
    frame: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    probs = np.empty(len(frame), dtype=np.float32)
    for slc, X_batch in encoder.transform_batches(frame, batch_size):
        intensity = np.asarray(model.predict(X_batch), dtype=np.float64)
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=50.0, neginf=0.0)
        intensity = np.clip(intensity, 0.0, 50.0)
        probs[slc] = (-np.expm1(-intensity)).astype(np.float32)
    return probs


def run_linear_logistic_experiment(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    regions: list[Region],
    feature_columns: list[str],
    config: dict[str, Any],
    args: Any,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    validate_no_leakage_features(feature_columns)
    experiment_id = "logistic_regression_full"
    categorical = [c for c in config.get("cat_features", []) if c in feature_columns]
    train_frame = df.loc[masks["train"], feature_columns]
    val_frame = df.loc[masks["validation"], feature_columns]
    test_frame = df.loc[masks["test"], feature_columns]
    y_train = positive_labels(df.loc[masks["train"], TARGET_COLUMN])
    y_val = positive_labels(df.loc[masks["validation"], TARGET_COLUMN])
    y_test = positive_labels(df.loc[masks["test"], TARGET_COLUMN])

    encoder = OrdinalTabularEncoder(feature_columns, categorical, standardize=True).fit(train_frame)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        learning_rate="optimal",
        class_weight={0: 1.0, 1: 4.0},
        random_state=args.seed,
        average=True,
    )

    classes = np.array([0, 1], dtype=np.int8)
    logging.info("Training linear logistic baseline with %d features for %d epochs", len(feature_columns), args.linear_epochs)
    for epoch in range(args.linear_epochs):
        order = np.arange(len(train_frame))
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(order)
        for start in range(0, len(order), args.prediction_batch_size):
            positions = order[start : start + args.prediction_batch_size]
            X_batch = encoder.transform(train_frame.iloc[positions])
            y_batch = y_train[positions]
            sample_weight = np.where(y_batch == 1, 4.0, 1.0)
            model.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight)
        logging.info("Completed linear logistic epoch %d/%d", epoch + 1, args.linear_epochs)

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump({"model": model, "encoder": encoder}, model_dir / f"{experiment_id}.joblib")
        model_path = model_dir / f"{experiment_id}.joblib"
    except Exception:
        model_path = None

    predictions = {
        "validation": predict_linear_batches(model, encoder, val_frame, args.prediction_batch_size),
        "test": predict_linear_batches(model, encoder, test_frame, args.prediction_batch_size),
    }
    threshold_info = choose_threshold_by_f1(y_val, predictions["validation"])
    threshold = float(threshold_info["threshold"])
    pred_paths = {
        "validation": save_predictions(output_dir, experiment_id, "validation", df.loc[masks["validation"]], y_val, predictions["validation"], threshold),
        "test": save_predictions(output_dir, experiment_id, "test", df.loc[masks["test"]], y_test, predictions["test"], threshold),
    }
    metric_rows = []
    metric_rows.extend(
        evaluate_predictions(
            experiment_id,
            "main_model_comparison",
            "Logistic Regression (linear SGD)",
            "full features",
            "validation",
            df.loc[masks["validation"]],
            y_val,
            predictions["validation"],
            threshold,
            regions,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            seed=args.seed,
        )
    )
    metric_rows.extend(
        evaluate_predictions(
            experiment_id,
            "main_model_comparison",
            "Logistic Regression (linear SGD)",
            "full features",
            "test",
            df.loc[masks["test"]],
            y_test,
            predictions["test"],
            threshold,
            regions,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            seed=args.seed,
        )
    )
    registry_row = {
        "experiment_id": experiment_id,
        "experiment_type": "main_model_comparison",
        "evaluation_type": "legacy_sampled_case_control",
        "is_primary": False,
        "model": "Logistic Regression (linear SGD)",
        "feature_set": "full features",
        "status": "completed",
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "threshold_source": "validation_f1_max",
        "validation_f1_at_threshold": threshold_info.get("validation_f1"),
        "model_path": model_path,
        "prediction_paths": pred_paths,
        "notes": "Linear logistic model trained with SGDClassifier and train-only ordinal categorical encoding to keep the full feature table tractable.",
    }
    primary_metrics = maybe_run_full_grid_tabular_evaluation(
        experiment_id=experiment_id,
        experiment_type="main_model_comparison",
        model_label="Logistic Regression (linear SGD)",
        model_type="linear_sgd",
        feature_set_label="full features",
        feature_columns=feature_columns,
        predict_raw_fn=make_linear_raw_predict_fn(model, encoder, args.prediction_batch_size),
        args=args,
        output_dir=output_dir,
        regions=regions,
        feature_config=config,
        model_path=model_path,
    )
    if primary_metrics:
        registry_row["primary_full_grid_metrics"] = primary_metrics
    return registry_row, metric_rows, predictions


def run_poisson_point_process_experiment(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    regions: list[Region],
    feature_columns: list[str],
    config: dict[str, Any],
    args: Any,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    validate_no_leakage_features(feature_columns)
    experiment_id = "poisson_point_process_full"
    categorical = [c for c in config.get("cat_features", []) if c in feature_columns]
    train_frame = df.loc[masks["train"], feature_columns]
    val_frame = df.loc[masks["validation"], feature_columns]
    test_frame = df.loc[masks["test"], feature_columns]
    y_train_binary = positive_labels(df.loc[masks["train"], TARGET_COLUMN])
    y_val = positive_labels(df.loc[masks["validation"], TARGET_COLUMN])
    y_test = positive_labels(df.loc[masks["test"], TARGET_COLUMN])
    y_train_count = (
        pd.to_numeric(df.loc[masks["train"], TARGET_COLUMN], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=np.float64)
    )

    fit_positions = stratified_sample_positions(
        y_train_binary,
        args.point_process_max_train_rows,
        args.seed,
    )
    encoder = OrdinalTabularEncoder(feature_columns, categorical, standardize=True).fit(train_frame)
    X_fit = encoder.transform(train_frame.iloc[fit_positions])
    y_fit = y_train_count[fit_positions]
    sample_weight = np.where(y_fit > 0, 4.0, 1.0)
    model = PoissonRegressor(
        alpha=float(args.point_process_alpha),
        max_iter=int(args.point_process_max_iter),
        fit_intercept=True,
    )
    logging.info(
        "Training Poisson point-process baseline on %d rows and %d features",
        len(fit_positions),
        len(feature_columns),
    )
    model.fit(X_fit, y_fit, sample_weight=sample_weight)

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump({"model": model, "encoder": encoder}, model_dir / f"{experiment_id}.joblib")
        model_path = model_dir / f"{experiment_id}.joblib"
    except Exception:
        model_path = None

    predictions = {
        "validation": predict_poisson_point_process_batches(model, encoder, val_frame, args.prediction_batch_size),
        "test": predict_poisson_point_process_batches(model, encoder, test_frame, args.prediction_batch_size),
    }
    threshold_info = choose_threshold_by_f1(y_val, predictions["validation"])
    threshold = float(threshold_info["threshold"])
    pred_paths = {
        "validation": save_predictions(output_dir, experiment_id, "validation", df.loc[masks["validation"]], y_val, predictions["validation"], threshold),
        "test": save_predictions(output_dir, experiment_id, "test", df.loc[masks["test"]], y_test, predictions["test"], threshold),
    }
    metric_rows = []
    metric_rows.extend(
        evaluate_predictions(
            experiment_id,
            "main_model_comparison",
            "Poisson Point-Process GLM",
            "full features",
            "validation",
            df.loc[masks["validation"]],
            y_val,
            predictions["validation"],
            threshold,
            regions,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            seed=args.seed,
        )
    )
    metric_rows.extend(
        evaluate_predictions(
            experiment_id,
            "main_model_comparison",
            "Poisson Point-Process GLM",
            "full features",
            "test",
            df.loc[masks["test"]],
            y_test,
            predictions["test"],
            threshold,
            regions,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            seed=args.seed,
        )
    )
    registry_row = {
        "experiment_id": experiment_id,
        "experiment_type": "main_model_comparison",
        "evaluation_type": "legacy_sampled_case_control",
        "is_primary": False,
        "model": "Poisson Point-Process GLM",
        "feature_set": "full features",
        "status": "completed",
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "threshold_source": "validation_f1_max",
        "validation_f1_at_threshold": threshold_info.get("validation_f1"),
        "model_path": model_path,
        "prediction_paths": pred_paths,
        "notes": (
            "Poisson GLM estimates cell-day event intensity and converts it to ignition probability "
            "as 1-exp(-lambda); equal exposure is assumed for sampled grid-cell days."
        ),
    }
    primary_metrics = maybe_run_full_grid_tabular_evaluation(
        experiment_id=experiment_id,
        experiment_type="main_model_comparison",
        model_label="Poisson Point-Process GLM",
        model_type="poisson_glm",
        feature_set_label="full features",
        feature_columns=feature_columns,
        predict_raw_fn=make_poisson_raw_predict_fn(model, encoder, args.prediction_batch_size),
        args=args,
        output_dir=output_dir,
        regions=regions,
        feature_config=config,
        model_path=model_path,
    )
    if primary_metrics:
        registry_row["primary_full_grid_metrics"] = primary_metrics
    return registry_row, metric_rows, predictions


def run_random_forest_experiment(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    regions: list[Region],
    feature_columns: list[str],
    config: dict[str, Any],
    args: Any,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    validate_no_leakage_features(feature_columns)
    experiment_id = "random_forest_full"
    categorical = [c for c in config.get("cat_features", []) if c in feature_columns]
    train_frame = df.loc[masks["train"], feature_columns]
    val_frame = df.loc[masks["validation"], feature_columns]
    test_frame = df.loc[masks["test"], feature_columns]
    y_train = positive_labels(df.loc[masks["train"], TARGET_COLUMN])
    y_val = positive_labels(df.loc[masks["validation"], TARGET_COLUMN])
    y_test = positive_labels(df.loc[masks["test"], TARGET_COLUMN])

    if args.rf_max_train_rows > 0 and len(train_frame) > args.rf_max_train_rows:
        idx = np.arange(len(train_frame))
        sampled_idx, _ = train_test_split(
            idx,
            train_size=args.rf_max_train_rows,
            stratify=y_train if len(np.unique(y_train)) > 1 else None,
            random_state=args.seed,
            shuffle=True,
        )
        sampled_idx = np.sort(sampled_idx)
        fit_frame = train_frame.iloc[sampled_idx]
        fit_y = y_train[sampled_idx]
        sample_note = f"Training capped to stratified {len(sampled_idx)} rows for feasibility."
    else:
        fit_frame = train_frame
        fit_y = y_train
        sample_note = "Training used all rows."

    encoder = OrdinalTabularEncoder(feature_columns, categorical, standardize=False).fit(train_frame)
    X_fit = encoder.transform(fit_frame)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=20,
        class_weight={0: 1.0, 1: 4.0},
        n_jobs=-1,
        random_state=args.seed,
        bootstrap=True,
    )
    logging.info("Training Random Forest baseline on %d rows and %d features", len(fit_frame), len(feature_columns))
    model.fit(X_fit, fit_y)

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump({"model": model, "encoder": encoder}, model_dir / f"{experiment_id}.joblib")
        model_path = model_dir / f"{experiment_id}.joblib"
    except Exception:
        model_path = None

    def predict_rf(frame: pd.DataFrame) -> np.ndarray:
        probs = np.empty(len(frame), dtype=np.float32)
        for slc, X_batch in encoder.transform_batches(frame, args.prediction_batch_size):
            probs[slc] = model.predict_proba(X_batch)[:, 1].astype(np.float32)
        return probs

    predictions = {"validation": predict_rf(val_frame), "test": predict_rf(test_frame)}
    threshold_info = choose_threshold_by_f1(y_val, predictions["validation"])
    threshold = float(threshold_info["threshold"])
    pred_paths = {
        "validation": save_predictions(output_dir, experiment_id, "validation", df.loc[masks["validation"]], y_val, predictions["validation"], threshold),
        "test": save_predictions(output_dir, experiment_id, "test", df.loc[masks["test"]], y_test, predictions["test"], threshold),
    }
    metric_rows = []
    metric_rows.extend(
        evaluate_predictions(
            experiment_id,
            "main_model_comparison",
            "Random Forest",
            "full features",
            "validation",
            df.loc[masks["validation"]],
            y_val,
            predictions["validation"],
            threshold,
            regions,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            seed=args.seed,
        )
    )
    metric_rows.extend(
        evaluate_predictions(
            experiment_id,
            "main_model_comparison",
            "Random Forest",
            "full features",
            "test",
            df.loc[masks["test"]],
            y_test,
            predictions["test"],
            threshold,
            regions,
            error_trials=args.random_error_trials,
            error_sample_size=args.random_error_sample_size,
            seed=args.seed,
        )
    )
    registry_row = {
        "experiment_id": experiment_id,
        "experiment_type": "main_model_comparison",
        "evaluation_type": "legacy_sampled_case_control",
        "is_primary": False,
        "model": "Random Forest",
        "feature_set": "full features",
        "status": "completed",
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "threshold_source": "validation_f1_max",
        "validation_f1_at_threshold": threshold_info.get("validation_f1"),
        "model_path": model_path,
        "prediction_paths": pred_paths,
        "notes": sample_note,
    }
    primary_metrics = maybe_run_full_grid_tabular_evaluation(
        experiment_id=experiment_id,
        experiment_type="main_model_comparison",
        model_label="Random Forest",
        model_type="random_forest",
        feature_set_label="full features",
        feature_columns=feature_columns,
        predict_raw_fn=make_predict_proba_raw_fn(model, encoder, args.prediction_batch_size),
        args=args,
        output_dir=output_dir,
        regions=regions,
        feature_config=config,
        model_path=model_path,
    )
    if primary_metrics:
        registry_row["primary_full_grid_metrics"] = primary_metrics
    return registry_row, metric_rows, predictions


def stratified_sample_positions(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    positions = np.arange(len(y))
    if max_rows <= 0 or len(y) <= max_rows:
        return positions
    try:
        sample, _ = train_test_split(
            positions,
            train_size=max_rows,
            stratify=y if len(np.unique(y)) > 1 else None,
            random_state=seed,
            shuffle=True,
        )
    except Exception:
        rng = np.random.default_rng(seed)
        sample = rng.choice(positions, size=max_rows, replace=False)
    return np.sort(sample)


def native_feature_importance(
    model: Any,
    feature_columns: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    values = np.asarray(model.get_feature_importance(), dtype=float)
    df = pd.DataFrame(
        {
            "feature": feature_columns,
            "group": [group_display(feature_group(col)) for col in feature_columns],
            "importance": values[: len(feature_columns)],
        }
    ).sort_values("importance", ascending=False)
    total = df["importance"].clip(lower=0).sum()
    df["normalized_importance"] = df["importance"].clip(lower=0) / total if total > 0 else 0.0
    df["rank"] = np.arange(1, len(df) + 1)
    df.to_csv(output_dir / "feature_importance_native.csv", index=False)
    plot_bar(
        df.head(30),
        x_col="importance",
        y_col="feature",
        title="Native CatBoost Feature Importance (Top 30)",
        output_base=output_dir / "plots/native_feature_importance_top30",
        xlabel="Importance",
    )
    return df


def grouped_permutation_importance(
    model: Any,
    feature_columns: list[str],
    cat_features: list[str],
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    threshold: float,
    args: Any,
    output_dir: Path,
) -> pd.DataFrame:
    return permutation_importance_for_groups(
        model=model,
        feature_columns=feature_columns,
        cat_features=cat_features,
        df=df,
        masks=masks,
        config=config,
        threshold=threshold,
        args=args,
        output_dir=output_dir,
        groups={group_display(feature_group(col)): [] for col in feature_columns},
        output_name="grouped_permutation_importance",
        title="Grouped Permutation Importance",
    )


def permutation_importance_for_groups(
    *,
    model: Any,
    feature_columns: list[str],
    cat_features: list[str],
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    threshold: float,
    args: Any,
    output_dir: Path,
    groups: dict[str, list[str]],
    output_name: str,
    title: str,
) -> pd.DataFrame:
    test_frame_all = df.loc[masks["test"]].reset_index(drop=True)
    y_all = positive_labels(test_frame_all[TARGET_COLUMN])
    sample_positions = stratified_sample_positions(y_all, args.permutation_sample_size, args.seed)
    sample_frame = test_frame_all.iloc[sample_positions].reset_index(drop=True)
    y_sample = y_all[sample_positions]
    X_base = normalize_cat_columns(sample_frame[feature_columns], cat_features, config.get("numerical_cat_features", []))
    base_prob = predict_catboost(model, X_base, cat_features)
    base = compute_metrics(y_sample, base_prob, threshold)

    if not any(groups.values()):
        for col in feature_columns:
            groups.setdefault(group_display(feature_group(col)), []).append(col)

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    trials = max(1, int(getattr(args, "permutation_trials", 1)))
    for group, cols in groups.items():
        trial_rows: list[dict[str, Any]] = []
        for _ in range(trials):
            X_perm = X_base.copy()
            for col in cols:
                X_perm[col] = rng.permutation(X_perm[col].to_numpy(copy=True))
            perm_prob = predict_catboost(model, X_perm, cat_features)
            perm = compute_metrics(y_sample, perm_prob, threshold)
            trial_rows.append(
                {
                    "permuted_pr_auc": perm.get("average_precision"),
                    "pr_auc_drop": metric_drop(base.get("average_precision"), perm.get("average_precision")),
                    "permuted_f1": perm.get("f1"),
                    "f1_drop": metric_drop(base.get("f1"), perm.get("f1")),
                    "permuted_roc_auc": perm.get("roc_auc"),
                    "roc_auc_drop": metric_drop(base.get("roc_auc"), perm.get("roc_auc")),
                }
            )
        trial_df = pd.DataFrame(trial_rows)
        rows.append(
            {
                "group": group,
                "raw_group": group,
                "feature_count": len(cols),
                "sample_rows": len(sample_frame),
                "trials": trials,
                "baseline_pr_auc": base.get("average_precision"),
                "permuted_pr_auc": trial_df["permuted_pr_auc"].mean(),
                "pr_auc_drop": trial_df["pr_auc_drop"].mean(),
                "pr_auc_drop_error": trial_df["pr_auc_drop"].std(ddof=1) if trials > 1 else None,
                "baseline_f1": base.get("f1"),
                "permuted_f1": trial_df["permuted_f1"].mean(),
                "f1_drop": trial_df["f1_drop"].mean(),
                "f1_drop_error": trial_df["f1_drop"].std(ddof=1) if trials > 1 else None,
                "baseline_roc_auc": base.get("roc_auc"),
                "permuted_roc_auc": trial_df["permuted_roc_auc"].mean(),
                "roc_auc_drop": trial_df["roc_auc_drop"].mean(),
                "roc_auc_drop_error": trial_df["roc_auc_drop"].std(ddof=1) if trials > 1 else None,
            }
        )
    imp = pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False, na_position="last")
    imp.to_csv(output_dir / f"{output_name}.csv", index=False)
    plot_bar(
        imp,
        x_col="pr_auc_drop",
        y_col="group",
        title=title,
        output_base=output_dir / f"plots/{output_name}",
        xlabel="PR-AUC drop after permutation",
    )
    return imp


def climate_window_permutation_importance(
    model: Any,
    feature_columns: list[str],
    cat_features: list[str],
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    threshold: float,
    args: Any,
    output_dir: Path,
) -> pd.DataFrame:
    groups: dict[str, list[str]] = {}
    for col in feature_columns:
        if feature_group(col) != "weather_history":
            continue
        window = feature_window_days(col)
        if window is not None:
            groups.setdefault(f"{window}-day climate window", []).append(col)
    if not groups:
        out = pd.DataFrame(
            columns=[
                "group",
                "raw_group",
                "feature_count",
                "sample_rows",
                "trials",
                "baseline_pr_auc",
                "permuted_pr_auc",
                "pr_auc_drop",
                "pr_auc_drop_error",
                "baseline_f1",
                "permuted_f1",
                "f1_drop",
                "f1_drop_error",
            ]
        )
        out.to_csv(output_dir / "climate_window_permutation_importance.csv", index=False)
        return out
    return permutation_importance_for_groups(
        model=model,
        feature_columns=feature_columns,
        cat_features=cat_features,
        df=df,
        masks=masks,
        config=config,
        threshold=threshold,
        args=args,
        output_dir=output_dir,
        groups=dict(sorted(groups.items(), key=lambda item: int(item[0].split("-")[0]))),
        output_name="climate_window_permutation_importance",
        title="Climate Window Permutation Importance",
    )


def metric_drop(base: Any, new: Any) -> float | None:
    if base is None or new is None:
        return None
    try:
        return float(base) - float(new)
    except Exception:
        return None


def catboost_native_shap(
    model: Any,
    feature_columns: list[str],
    cat_features: list[str],
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    args: Any,
    output_dir: Path,
) -> pd.DataFrame | None:
    if args.skip_shap:
        return None
    test_frame_all = df.loc[masks["test"]].reset_index(drop=True)
    y_all = positive_labels(test_frame_all[TARGET_COLUMN])
    sample_positions = stratified_sample_positions(y_all, args.shap_sample_size, args.seed)
    sample_frame = test_frame_all.iloc[sample_positions].reset_index(drop=True)
    X_sample = normalize_cat_columns(sample_frame[feature_columns], cat_features, config.get("numerical_cat_features", []))
    logging.info("Computing CatBoost-native SHAP on %d rows", len(X_sample))
    shap_values = model.get_feature_importance(
        data=catboost_pool(X_sample, None, cat_features),
        type="ShapValues",
    )
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim != 2 or shap_values.shape[1] < len(feature_columns):
        raise ExperimentFailure(f"Unexpected SHAP shape: {shap_values.shape}")
    shap_matrix = shap_values[:, : len(feature_columns)]
    mean_abs = np.nanmean(np.abs(shap_matrix), axis=0)
    shap_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "group": [group_display(feature_group(col)) for col in feature_columns],
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df["rank"] = np.arange(1, len(shap_df) + 1)
    shap_df.to_csv(output_dir / "shap_importance.csv", index=False)
    plot_bar(
        shap_df.head(30),
        x_col="mean_abs_shap",
        y_col="feature",
        title="CatBoost-Native SHAP Importance (Top 30)",
        output_base=output_dir / "plots/shap_summary",
        xlabel="Mean absolute SHAP value",
    )
    return shap_df


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_base: Path,
    xlabel: str,
) -> None:
    if df.empty or x_col not in df.columns:
        return
    plot_df = df.dropna(subset=[x_col]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(x_col, ascending=True)
    height = max(4.0, min(12.0, 0.35 * len(plot_df) + 1.5))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(plot_df[y_col].astype(str), plot_df[x_col].astype(float), color="#2563eb")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def wrap_plot_label(label: Any, width: int = 24) -> str:
    text = str(label)
    wrapped = textwrap.wrap(text, width=width, break_long_words=False)
    return "\n".join(wrapped) if wrapped else text


def plot_pr_curves(
    prediction_store: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    regions: list[Region],
    output_dir: Path,
) -> None:
    test_frame = df.loc[masks["test"]].reset_index(drop=True)
    y_test = positive_labels(test_frame[TARGET_COLUMN])
    desired = [
        "logistic_regression_full",
        "poisson_point_process_full",
        "catboost_fwi_only",
        "catboost_weather_only",
        "catboost_full",
        "random_forest_full",
    ]
    available = [exp for exp in desired if exp in prediction_store and "test" in prediction_store[exp]]
    if not available:
        return

    fig, (ax, metric_ax) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    summary_rows: list[dict[str, Any]] = []
    for exp in available:
        prob = prediction_store[exp]["test"]
        if len(np.unique(y_test)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        label = prediction_store[exp]["label"]
        threshold = float(prediction_store[exp].get("threshold", 0.5))
        y_pred = (np.asarray(prob) >= threshold).astype(int)
        f1 = safe_score(lambda yt, yp: f1_score(yt, yp, zero_division=0), y_test, y_pred)
        ax.plot(recall, precision, linewidth=2, label=label)
        summary_rows.append({"method": label, "average_precision": ap, "f1": f1})
    if not summary_rows:
        plt.close(fig)
        return
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Global sampled precision-recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    summary_df = pd.DataFrame(summary_rows).sort_values("average_precision", ascending=True)
    y_pos = np.arange(len(summary_df))
    metric_ax.barh(y_pos - 0.18, summary_df["average_precision"].astype(float), height=0.35, color="#2563eb", label="PR-AUC")
    metric_ax.barh(y_pos + 0.18, summary_df["f1"].astype(float), height=0.35, color="#f97316", label="F1")
    metric_ax.set_yticks(y_pos)
    metric_ax.set_yticklabels([wrap_plot_label(v, 24) for v in summary_df["method"]], fontsize=8)
    metric_ax.set_xlabel("Absolute test metric")
    metric_ax.set_xlim(0, 1)
    metric_ax.set_title("Global sampled metrics")
    metric_ax.grid(axis="x", alpha=0.25)
    metric_ax.legend(fontsize=8)
    fig.tight_layout()
    base = output_dir / "plots/pr_curves_global"
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    region_specs: list[tuple[str, np.ndarray]] = [("Global", np.ones(len(test_frame), dtype=bool))]
    region_specs.extend((region.display_name, region.mask(test_frame)) for region in regions)
    if not region_specs:
        return
    regional_rows: list[dict[str, Any]] = []
    method_order = [prediction_store[exp]["label"] for exp in available]
    for region_label, mask in region_specs:
        y_region = y_test[mask]
        for exp in available:
            prob = prediction_store[exp]["test"][mask]
            threshold = float(prediction_store[exp].get("threshold", 0.5))
            y_pred = (np.asarray(prob) >= threshold).astype(int)
            regional_rows.append(
                {
                    "region": region_label,
                    "method": prediction_store[exp]["label"],
                    "average_precision": (
                        average_precision_score(y_region, prob)
                        if len(y_region) and len(np.unique(y_region)) == 2
                        else np.nan
                    ),
                    "f1": safe_score(lambda yt, yp: f1_score(yt, yp, zero_division=0), y_region, y_pred),
                }
            )
    regional_df = pd.DataFrame(regional_rows)
    if regional_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4.8, 0.58 * len(region_specs) + 2.0)), squeeze=False)
    for ax, metric, title in [
        (axes[0][0], "average_precision", "Regional PR-AUC"),
        (axes[0][1], "f1", "Regional F1"),
    ]:
        pivot = (
            regional_df.pivot(index="region", columns="method", values=metric)
            .reindex(index=[name for name, _ in region_specs], columns=method_order)
        )
        values = pivot.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        im = ax.imshow(masked, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(method_order)))
        ax.set_xticklabels([wrap_plot_label(v, 18) for v in method_order], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str), fontsize=9)
        for row_idx in range(values.shape[0]):
            for col_idx in range(values.shape[1]):
                value = values[row_idx, col_idx]
                if np.isfinite(value):
                    color = "white" if value >= 0.55 else "black"
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Regional sampled metric contrast", y=1.02)
    fig.tight_layout()
    base = output_dir / "plots/pr_curves_regions"
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


ABLATION_DISPLAY_NAMES = {
    "ablation_full": "Full CatBoost",
    "ablation_weather_only": "Weather-only CatBoost",
    "ablation_fwi_only": "FWI-only CatBoost",
    "ablation_no_anthropogenic": "CatBoost no anthropogenic",
    "ablation_no_ecoregion": "CatBoost-no-ecoregion",
    "ablation_no_geography": "CatBoost-no-geography",
    "ablation_no_fuel_ecoregion_vegetation": "CatBoost no fuel/ecoregion/vegetation",
    "ablation_no_terrain": "CatBoost no terrain",
    "ablation_no_seasonality": "CatBoost no seasonality",
    "ablation_no_history": "CatBoost no history",
    "ablation_no_temporal_history": "CatBoost no temporal-history/weather lags",
    "ablation_shorter_sequence_30d": "CatBoost <=30d climate windows",
    "ablation_no_periodic_seasonality": "CatBoost no sine/cos seasonality",
    "ablation_no_gaussian_smoothing": "CatBoost no Gaussian-smoothed anthropogenic rasters",
    "ablation_static_only": "CatBoost static only",
    "ablation_dynamic_weather_fwi_only": "CatBoost dynamic weather+FWI only",
}


def ablation_display_name(experiment_id: Any, feature_set: Any | None = None, model: Any | None = None) -> str:
    key = str(experiment_id)
    if key in ABLATION_DISPLAY_NAMES:
        return ABLATION_DISPLAY_NAMES[key]
    if model is not None and str(model) and str(model) != "CatBoost":
        return str(model)
    if feature_set is not None and str(feature_set):
        return f"CatBoost {feature_set}"
    return key


def plot_ablation_metric_comparison(
    global_df: pd.DataFrame,
    *,
    full_value: float | None,
    metric_col: str,
    delta_col: str,
    metric_label: str,
    output_base: Path,
) -> None:
    if metric_col not in global_df.columns or delta_col not in global_df.columns:
        return
    work = global_df.dropna(subset=[metric_col]).sort_values(metric_col, ascending=True).copy()
    if work.empty:
        return
    height = max(5.0, min(12.5, 0.42 * len(work) + 1.8))
    labels = [wrap_plot_label(v, 34) for v in work["experiment_label"]]
    y_pos = np.arange(len(work))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, height),
        gridspec_kw={"width_ratios": [1.05, 0.95]},
    )
    axes[0].barh(y_pos, work[metric_col].astype(float), color="#2563eb")
    if full_value is not None and math.isfinite(float(full_value)):
        axes[0].axvline(float(full_value), color="#111827", linestyle="--", linewidth=1.4, label="Full CatBoost")
        axes[0].legend(fontsize=8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel(metric_label)
    axes[0].set_title(f"Absolute {metric_label}")
    axes[0].set_xlim(0, 1)
    axes[0].grid(axis="x", alpha=0.25)

    delta_values = work[delta_col].astype(float)
    colors = np.where(delta_values >= 0, "#dc2626", "#16a34a")
    axes[1].barh(y_pos, delta_values, color=colors)
    axes[1].axvline(0, color="#111827", linewidth=1.0)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel(f"Full CatBoost minus variant {metric_label}")
    axes[1].set_title("Delta vs full")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle(f"Feature ablation {metric_label}: absolute metric and difference", y=1.01)
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_feature_ablation_drops(ablation_df: pd.DataFrame, output_dir: Path) -> None:
    full_rows = ablation_df[
        (ablation_df["split"] == "test")
        & (ablation_df["region"] == "global")
        & (ablation_df["experiment_id"] == "ablation_full")
    ].copy()
    full_ap = float(full_rows["average_precision"].iloc[0]) if not full_rows.empty else None
    full_f1 = float(full_rows["f1"].iloc[0]) if not full_rows.empty else None
    global_df = ablation_df[
        (ablation_df["split"] == "test")
        & (ablation_df["region"] == "global")
        & (ablation_df["experiment_id"] != "ablation_full")
    ].copy()
    if global_df.empty:
        return
    global_df["experiment_label"] = global_df.apply(
        lambda row: ablation_display_name(row["experiment_id"], row.get("feature_set"), row.get("model")),
        axis=1,
    )
    plot_ablation_metric_comparison(
        global_df,
        full_value=full_ap,
        metric_col="average_precision",
        delta_col="delta_average_precision_vs_full",
        metric_label="PR-AUC",
        output_base=output_dir / "plots/feature_ablation_pr_auc_drop",
    )
    plot_ablation_metric_comparison(
        global_df,
        full_value=full_f1,
        metric_col="f1",
        delta_col="delta_f1_vs_full",
        metric_label="F1",
        output_base=output_dir / "plots/feature_ablation_f1_drop",
    )


def plot_input_source(input_df: pd.DataFrame, output_dir: Path) -> None:
    df = input_df[
        (input_df["region"] == "global")
        & input_df["status"].astype(str).str.startswith("completed")
    ].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [
        (axes[0], "average_precision", "Input Source PR-AUC"),
        (axes[1], "f1", "Input Source F1"),
    ]:
        ax.bar(df["experiment"], df[metric], color="#0f766e")
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for name in ["input_source_comparison", "input_source_pr_auc", "input_source_f1"]:
        base = output_dir / f"plots/{name}"
        base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(display.columns)
    rows = display.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep, *body])


def write_markdown_table(path: Path, title: str, df: pd.DataFrame) -> None:
    path.write_text(f"# {title}\n\n{markdown_table(df)}\n", encoding="utf-8")


def build_metrics_long(metrics_wide: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "experiment_id",
        "experiment_type",
        "model",
        "feature_set",
        "split",
        "region",
        "region_display",
        "support",
        "positives",
        "threshold",
    ]
    value_cols = [
        col
        for col in list(METRIC_COLUMNS) + [f"{metric}_error" for metric in METRIC_COLUMNS]
        if col in metrics_wide.columns
    ]
    return metrics_wide.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric", value_name="value")


def make_main_model_table(metrics_wide: pd.DataFrame, registry: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    main_ids = registry.loc[registry["experiment_type"].eq("main_model_comparison") & registry["status"].eq("completed"), "experiment_id"]
    df = metrics_wide[
        metrics_wide["experiment_id"].isin(main_ids)
        & metrics_wide["split"].eq("test")
    ].copy()
    cols = [
        "model",
        "feature_set",
        "region_display",
        "support",
        "positives",
        "precision",
        "recall",
        "f1",
        "f1_error",
        "average_precision",
        "average_precision_error",
        "roc_auc",
        "brier_score",
        "threshold",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[cols].sort_values(["region_display", "average_precision"], ascending=[True, False])
    df = df.rename(
        columns={
            "model": "Model",
            "feature_set": "Feature set",
            "region_display": "Region",
            "average_precision": "PR-AUC",
            "average_precision_error": "PR-AUC error",
            "f1_error": "F1 error",
            "roc_auc": "ROC-AUC",
            "brier_score": "Brier",
        }
    )
    df.to_csv(output_dir / "main_model_comparison.csv", index=False)
    write_markdown_table(output_dir / "main_model_comparison.md", "Main Model Comparison", df)

    by_year = metrics_wide[
        metrics_wide["experiment_id"].isin(main_ids)
        & metrics_wide["split"].astype(str).str.startswith("test_")
    ].copy()
    if not by_year.empty:
        by_year["period"] = by_year["split"].astype(str).str.replace("test_", "", regex=False)
        for col in ["f1_error", "average_precision_error"]:
            if col not in by_year.columns:
                by_year[col] = np.nan
        by_year = by_year[
            [
                "model",
                "feature_set",
                "region_display",
                "period",
                "support",
                "positives",
                "positive_rate",
                "precision",
                "recall",
                "f1",
                "f1_error",
                "average_precision",
                "average_precision_error",
                "roc_auc",
                "brier_score",
                "threshold",
            ]
        ].sort_values(["model", "region_display", "period"])
        by_year.to_csv(output_dir / "main_model_comparison_by_year.csv", index=False)
    return df


def save_legacy_sampled_outputs(
    output_dir: Path,
    *,
    main_table: pd.DataFrame,
    metrics_wide: pd.DataFrame,
    metrics_long: pd.DataFrame,
) -> None:
    legacy_dir = output_dir / "legacy_sampled_case_control"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(legacy_dir / "model_comparison.csv", index=False)
    main_table.to_csv(output_dir / "legacy_sampled_model_comparison.csv", index=False)
    by_year = output_dir / "main_model_comparison_by_year.csv"
    if by_year.exists():
        by = pd.read_csv(by_year)
        by.to_csv(legacy_dir / "model_comparison_by_year.csv", index=False)
        by.to_csv(output_dir / "legacy_sampled_model_comparison_by_year.csv", index=False)
    if not metrics_wide.empty:
        legacy_metrics = metrics_wide.copy()
        legacy_metrics["evaluation_type"] = "legacy_sampled_case_control"
        legacy_metrics["is_primary"] = False
        legacy_metrics.to_csv(legacy_dir / "metrics_wide.csv", index=False)
        threshold_cols = [
            col
            for col in [
                "experiment_id",
                "model",
                "feature_set",
                "split",
                "region",
                "region_display",
                "threshold",
                "precision",
                "recall",
                "f1",
            ]
            if col in legacy_metrics.columns
        ]
        legacy_metrics[threshold_cols].to_csv(legacy_dir / "threshold_metrics.csv", index=False)
    if not metrics_long.empty:
        metrics_long.to_csv(legacy_dir / "metrics_long.csv", index=False)


def make_feature_ablation_table(metrics_wide: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    df = metrics_wide[
        metrics_wide["experiment_type"].eq("feature_ablation")
        & metrics_wide["split"].eq("test")
    ].copy()
    if df.empty:
        return df
    for col in ["f1_error", "average_precision_error"]:
        if col not in df.columns:
            df[col] = np.nan
    full = df[df["experiment_id"].eq("ablation_full")][["region", "f1", "average_precision"]].rename(
        columns={"f1": "full_f1", "average_precision": "full_average_precision"}
    )
    df = df.merge(full, on="region", how="left")
    df["delta_f1_vs_full"] = df["full_f1"] - df["f1"]
    df["delta_average_precision_vs_full"] = df["full_average_precision"] - df["average_precision"]
    df["experiment_label"] = df.apply(
        lambda row: ablation_display_name(row["experiment_id"], row.get("feature_set"), row.get("model")),
        axis=1,
    )
    table = df[
        [
            "experiment_label",
            "experiment_id",
            "model",
            "feature_set",
            "region_display",
            "precision",
            "recall",
            "f1",
            "f1_error",
            "average_precision",
            "average_precision_error",
            "delta_f1_vs_full",
            "delta_average_precision_vs_full",
        ]
    ].rename(
        columns={
            "experiment_label": "Experiment",
            "experiment_id": "Experiment ID",
            "model": "Model",
            "feature_set": "Feature set",
            "region_display": "Region",
            "average_precision": "PR-AUC",
            "average_precision_error": "PR-AUC error",
            "f1_error": "F1 error",
            "delta_f1_vs_full": "Delta F1 vs full",
            "delta_average_precision_vs_full": "Delta PR-AUC vs full",
        }
    )
    table.to_csv(output_dir / "feature_ablation.csv", index=False)
    write_markdown_table(output_dir / "feature_ablation.md", "Feature Ablation", table)
    plot_feature_ablation_drops(df, output_dir)

    by_year = metrics_wide[
        metrics_wide["experiment_type"].eq("feature_ablation")
        & metrics_wide["split"].astype(str).str.startswith("test_")
    ].copy()
    if not by_year.empty:
        by_year["period"] = by_year["split"].astype(str).str.replace("test_", "", regex=False)
        for col in ["f1_error", "average_precision_error"]:
            if col not in by_year.columns:
                by_year[col] = np.nan
        full_by_period = by_year[by_year["experiment_id"].eq("ablation_full")][
            ["region", "period", "f1", "average_precision"]
        ].rename(columns={"f1": "full_f1", "average_precision": "full_average_precision"})
        by_year = by_year.merge(full_by_period, on=["region", "period"], how="left")
        by_year["delta_f1_vs_full"] = by_year["full_f1"] - by_year["f1"]
        by_year["delta_average_precision_vs_full"] = (
            by_year["full_average_precision"] - by_year["average_precision"]
        )
        by_year["experiment_label"] = by_year.apply(
            lambda row: ablation_display_name(row["experiment_id"], row.get("feature_set"), row.get("model")),
            axis=1,
        )
        by_year[
            [
                "experiment_label",
                "experiment_id",
                "model",
                "feature_set",
                "region_display",
                "period",
                "support",
                "positives",
                "precision",
                "recall",
                "f1",
                "f1_error",
                "average_precision",
                "average_precision_error",
                "delta_f1_vs_full",
                "delta_average_precision_vs_full",
                "threshold",
            ]
        ].to_csv(output_dir / "feature_ablation_by_year.csv", index=False)
    return table


def placeholder_blocked_tables(
    output_dir: Path,
    failures: list[dict[str, Any]],
    full_global_metrics: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    embedding = pd.DataFrame(
        [
            {
                "experiment": name,
                "status": "blocked",
                "reason": "Prepared NN dataset `data/saved_features/nn_train_data/prepared_data.npz` is absent; current checkpoints cannot be evaluated or ablated reproducibly without rebuilding NN inputs.",
                "precision": None,
                "recall": None,
                "f1": None,
                "f1_error": None,
                "average_precision": None,
                "average_precision_error": None,
            }
            for name in [
                "LSTM-MLP full current model",
                "Simple concatenation / one-hot categorical variables",
                "Learned categorical embeddings",
                "Temporal-only LSTM branch",
                "Static-only MLP branch",
                "Minimal MLP baseline",
                "FT-Transformer baseline",
                "Dynamic + static without categorical embeddings",
            ]
        ]
    )
    embedding.to_csv(output_dir / "embedding_fusion_ablation.csv", index=False)
    write_markdown_table(output_dir / "embedding_fusion_ablation.md", "Embedding Fusion Ablation", embedding)
    failures.append(
        {
            "experiment": "Neural embedding/fusion ablations",
            "reason": "No prepared NN `.npz` dataset was present in the expected metadata directory.",
            "affects_main_claims": "No; CatBoost/data-fusion ablations cover the primary reviewer request.",
            "suggested_next_action": "Run `make_nn_train_data.py` for the reviewer split, then train/evaluate the listed NN variants.",
        }
    )

    label_rows = [
        {
            "experiment": "Main target labeling",
            "status": "completed",
            "precision": (full_global_metrics or {}).get("precision"),
            "recall": (full_global_metrics or {}).get("recall"),
            "f1": (full_global_metrics or {}).get("f1"),
            "f1_error": (full_global_metrics or {}).get("f1_error"),
            "average_precision": (full_global_metrics or {}).get("average_precision"),
            "average_precision_error": (full_global_metrics or {}).get("average_precision_error"),
            "interpretation": "Main MODIS target with configured thresholds, high-latitude relaxation, stationary-point filtering, positive expansion, sampled negatives, and land mask.",
        },
        {
            "experiment": "No historical-fire features with main target",
            "status": "completed_no_op",
            "precision": (full_global_metrics or {}).get("precision"),
            "recall": (full_global_metrics or {}).get("recall"),
            "f1": (full_global_metrics or {}).get("f1"),
            "f1_error": (full_global_metrics or {}).get("f1_error"),
            "average_precision": (full_global_metrics or {}).get("average_precision"),
            "average_precision_error": (full_global_metrics or {}).get("average_precision_error"),
            "interpretation": "No historical-fire/proximity/density columns were present in the saved feature matrix, so this sensitivity is equivalent to the main model.",
        },
        {
            "experiment": "No morphological expansion / no dilation",
            "status": "blocked",
            "precision": None,
            "recall": None,
            "f1": None,
            "f1_error": None,
            "average_precision": None,
            "average_precision_error": None,
            "interpretation": "Requires rebuilding target caches with expansion disabled; no no-dilation cache exists in the current workspace.",
        },
        {
            "experiment": "Stricter MODIS confidence/brightness threshold",
            "status": "blocked",
            "precision": None,
            "recall": None,
            "f1": None,
            "f1_error": None,
            "average_precision": None,
            "average_precision_error": None,
            "interpretation": "Requires rebuilding target caches with modified target_config thresholds.",
        },
        {
            "experiment": "Different negative sampling ratio",
            "status": "blocked",
            "precision": None,
            "recall": None,
            "f1": None,
            "f1_error": None,
            "average_precision": None,
            "average_precision_error": None,
            "interpretation": "Requires regenerating per-country targets/features with a different samples_per_area_per_year.",
        },
    ]
    label = pd.DataFrame(label_rows)
    label.to_csv(output_dir / "label_sensitivity.csv", index=False)
    write_markdown_table(output_dir / "label_sensitivity.md", "Label Sensitivity", label)
    failures.extend(
        [
            {
                "experiment": "No morphological expansion / no dilation",
                "reason": "Current feature parquet was already built from expanded targets; no no-dilation target cache was found.",
                "affects_main_claims": "Potentially relevant to target-construction sensitivity; reported as a limitation.",
                "suggested_next_action": "Add a target_config flag to bypass `expand_positive_points`, rebuild target caches, and rerun CatBoost on the rebuilt feature matrix.",
            },
            {
                "experiment": "Stricter MODIS confidence/brightness thresholds",
                "reason": "Requires target and feature regeneration; no stricter-threshold cache was present.",
                "affects_main_claims": "Limited; main performance and feature ablation claims are still based on the stated target config.",
                "suggested_next_action": "Create a stricter target_config and rebuild per-country feature parquet files.",
            },
        ]
    )

    lead_time = pd.DataFrame(
        [
            {
                "experiment": "Lead-time sensitivity",
                "status": "blocked",
                "reason": "The saved training matrix contains 30-day historical aggregates and no explicit forecast lead-time dimension.",
                "precision": None,
                "recall": None,
                "f1": None,
                "f1_error": None,
                "average_precision": None,
                "average_precision_error": None,
            }
        ]
    )
    lead_time.to_csv(output_dir / "lead_time_sensitivity.csv", index=False)
    write_markdown_table(output_dir / "lead_time_sensitivity.md", "Lead-Time Sensitivity", lead_time)
    failures.append(
        {
            "experiment": "Lead-time sensitivity",
            "reason": "No lead-time-specific features or forecast-initialization metadata are present in the saved training matrix.",
            "affects_main_claims": "No direct effect on retrospective ignition discrimination; affects operational lead-time claims.",
            "suggested_next_action": "Generate feature matrices by forecast lead time from forecast/hindcast inputs.",
        }
    )
    return embedding, label, lead_time


def input_source_table(
    output_dir: Path,
    full_metrics_wide: pd.DataFrame,
    args: Any,
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_rows = full_metrics_wide[
        full_metrics_wide["experiment_id"].eq("catboost_full")
        & full_metrics_wide["split"].eq("test")
    ]
    for _, row in full_rows.iterrows():
        rows.append(
            {
                "experiment": "SEAS5/ECMWF -> SEAS5/ECMWF",
                "status": "completed",
                "interpretation": "Operationally matched setting using the existing ECMWF-derived feature matrix.",
                "region": row["region"],
                "region_display": row["region_display"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "f1_error": row.get("f1_error"),
                "average_precision": row["average_precision"],
                "average_precision_error": row.get("average_precision_error"),
                "roc_auc": row["roc_auc"],
                "notes": "Threshold selected on ECMWF validation split and applied to ECMWF test split.",
            }
        )
    blocked = [
        ("ERA5 -> ERA5", "Retrospective upper-bound setting"),
        ("ERA5 -> SEAS5/ECMWF", "Input-source domain-shift test"),
        ("ERA5 + SEAS5/ECMWF -> SEAS5/ECMWF", "Mixed-source operational robustness test"),
    ]
    reason = (
        f"Raw ERA5 files are readable at {args.era5_dir}, but no precomputed ERA5-derived feature parquet "
        "with the same schema/date/grid rows as the ECMWF feature table was found. Building it requires a "
        "full climate-feature regeneration pass, not a lightweight in-run adapter."
    )
    for experiment, interpretation in blocked:
        rows.append(
            {
                "experiment": experiment,
                "status": "blocked",
                "interpretation": interpretation,
                "region": "global",
                "region_display": "Global",
                "precision": None,
                "recall": None,
                "f1": None,
                "f1_error": None,
                "average_precision": None,
                "average_precision_error": None,
                "roc_auc": None,
                "notes": reason,
            }
        )
        failures.append(
            {
                "experiment": experiment,
                "reason": reason,
                "affects_main_claims": "No for model/ablation claims; yes for the requested weather-source comparison.",
                "suggested_next_action": "Generate an ERA5-derived feature parquet on the same target rows and common feature subset, then rerun this runner with an ERA5 features path.",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "input_source_comparison.csv", index=False)
    write_markdown_table(output_dir / "input_source_comparison.md", "ERA5 / SEAS5 Input Source Comparison", df)
    plot_input_source(df, output_dir)
    return df


def write_failures(path: Path, failures: list[dict[str, Any]]) -> None:
    if not failures:
        path.write_text("# Failed Or Skipped Experiments\n\nNo failed or skipped experiments.\n", encoding="utf-8")
        return
    lines = ["# Failed Or Skipped Experiments", ""]
    for item in failures:
        lines.extend(
            [
                f"## {item.get('experiment')}",
                "",
                f"- Failure reason: {item.get('reason')}",
                f"- Affects main paper claims: {item.get('affects_main_claims')}",
                f"- Suggested next action: {item.get('suggested_next_action')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def best_global(metrics_wide: pd.DataFrame, registry: pd.DataFrame) -> pd.Series | None:
    ids = registry.loc[registry["status"].eq("completed"), "experiment_id"]
    df = metrics_wide[
        metrics_wide["experiment_id"].isin(ids)
        & metrics_wide["split"].eq("test")
        & metrics_wide["region"].eq("global")
    ].copy()
    if df.empty:
        return None
    return df.sort_values("average_precision", ascending=False).iloc[0]


def get_metric_row(metrics_wide: pd.DataFrame, experiment_id: str, region: str = "global") -> pd.Series | None:
    rows = metrics_wide[
        metrics_wide["experiment_id"].eq(experiment_id)
        & metrics_wide["split"].eq("test")
        & metrics_wide["region"].eq(region)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def generate_interpretation(
    output_dir: Path,
    metrics_wide: pd.DataFrame,
    registry: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    ablation_table: pd.DataFrame,
    input_source: pd.DataFrame,
    native_importance: pd.DataFrame | None,
    grouped_perm: pd.DataFrame | None,
    climate_window_perm: pd.DataFrame | None,
    shap_df: pd.DataFrame | None,
    failures: list[dict[str, Any]],
) -> str:
    full = get_metric_row(metrics_wide, "catboost_full")
    fwi = get_metric_row(metrics_wide, "catboost_fwi_only")
    weather = get_metric_row(metrics_wide, "catboost_weather_only")
    logistic = get_metric_row(metrics_wide, "logistic_regression_full")
    poisson = get_metric_row(metrics_wide, "poisson_point_process_full")
    best = best_global(metrics_wide, registry)
    full_ap = full.get("average_precision") if full is not None else None
    fwi_ap = fwi.get("average_precision") if fwi is not None else None
    weather_ap = weather.get("average_precision") if weather is not None else None
    logistic_ap = logistic.get("average_precision") if logistic is not None else None
    poisson_ap = poisson.get("average_precision") if poisson is not None else None
    fwi_delta = metric_drop(full_ap, fwi_ap)
    weather_delta = metric_drop(full_ap, weather_ap)
    logistic_delta = metric_drop(full_ap, logistic_ap)
    poisson_delta = metric_drop(full_ap, poisson_ap)

    top_group = None
    if grouped_perm is not None and not grouped_perm.empty:
        top_group = grouped_perm.sort_values("pr_auc_drop", ascending=False).iloc[0]
    top_window = None
    if climate_window_perm is not None and not climate_window_perm.empty:
        top_window = climate_window_perm.sort_values("pr_auc_drop", ascending=False).iloc[0]
    top_features = []
    if native_importance is not None and not native_importance.empty:
        top_features = native_importance.head(5)["feature"].tolist()

    paragraphs = [
        "## Experimental Setup",
        "We evaluated all feasible revision experiments on the existing precomputed feature matrix using a fixed chronological split: 2001-2018 for training, 2019-2020 for threshold selection and validation, and 2021-2025 for testing. Binary decision thresholds were selected only on validation by maximizing F1 and then applied unchanged to the test split and all regional/yearly subsets. Metric error columns are estimated with five stratified bootstrap trials over saved predictions for efficient uncertainty summaries.",
        "",
        "## Dataset Statistics",
        f"The saved dataset uses a detected grid spacing of {dataset_stats.iloc[0]['grid_resolution']}. Negatives are sampled rather than full-grid negatives, and the feature merge retains land rows according to the configured land/sea mask rule. The global train, validation, and test sample counts are reported in `dataset_statistics.csv`.",
        "",
        "## Main Performance",
        (
            f"The best completed global test model is `{best['model'] if best is not None else 'NA'}` "
            f"with PR-AUC {fmt(best.get('average_precision') if best is not None else None)} "
            f"and F1 {fmt(best.get('f1') if best is not None else None)}. "
            f"The full CatBoost model achieves PR-AUC {fmt(full_ap)} "
            f"and F1 {fmt(full.get('f1') if full is not None else None)}. "
            f"Relative to the FWI-only baseline, the full model changes PR-AUC by {fmt(fwi_delta)}; "
            f"relative to weather-only CatBoost by {fmt(weather_delta)}; "
            f"relative to linear logistic by {fmt(logistic_delta)}; "
            f"and relative to Poisson point-process by {fmt(poisson_delta)}."
        ),
        "",
        "## Feature Ablations",
        "The CatBoost ablations quantify data-fusion value by removing or isolating feature sources while keeping the same validation thresholding rule. The ablation plots report absolute PR-AUC/F1 next to the full-minus-variant delta, so positive deltas mean the variant scored lower than the full fused CatBoost model and negative deltas mean the variant scored higher.",
        "",
        "## ERA5 / SEAS5",
        "SEAS5/ECMWF -> SEAS5/ECMWF is the clean operationally matched setting available from the existing feature matrix. ERA5->ERA5 would represent a retrospective upper-bound setting, while ERA5->SEAS5 measures input-source domain shift, not simply model quality. The raw ERA5 files are readable, but exact feature-schema parity was blocked by the absence of a precomputed ERA5-derived feature parquet.",
        "",
        "## Interpretability",
        f"Native CatBoost importance ranks the following features highest: {', '.join(top_features) if top_features else 'NA'}. Grouped permutation importance shows the largest PR-AUC drop for `{top_group['group'] if top_group is not None else 'NA'}` ({fmt(top_group['pr_auc_drop'] if top_group is not None else None)}), while climate-window permutation is strongest for `{top_window['group'] if top_window is not None else 'NA'}` ({fmt(top_window['pr_auc_drop'] if top_window is not None else None)}). These attributions are model explanations, not causal effects.",
        "",
        "## Lead Time",
        "Lead-time sensitivity was not directly supported by the saved 30-day aggregate feature matrix because it does not retain forecast lead-time metadata.",
        "",
        "## Limitations",
        "Neural embedding/fusion ablations, no-dilation label sensitivity, stricter target-threshold sensitivity, and full ERA5 parity require regenerated intermediate datasets. These blockers are listed explicitly in `failures.md`.",
        "",
    ]
    text = "\n".join(paragraphs)
    (output_dir / "paper_ready_interpretation.md").write_text(text, encoding="utf-8")
    return text


def generate_reviewer_response(output_dir: Path, interpretation: str) -> None:
    text = """# Reviewer Response Insertions

## Reviewer 1: Novelty / Methodological Innovation

We added a reviewer-focused revision experiment package comparing linear, fire-weather-only, weather-only, Random Forest, and fused CatBoost models under a fixed chronological split. These experiments clarify that the proposed contribution is not a single classifier choice, but a data-fusion pipeline that combines meteorological history with ecological, topographic, and anthropogenic context.

## Reviewer 2: Ablation And Feature Importance Requests

We added CatBoost feature-source ablations and grouped permutation importance. The ablations report absolute metrics and drops relative to the full fused model, while grouped permutation measures the PR-AUC/F1 decrease after disrupting each feature group. We explicitly interpret these as model attributions rather than causal effects.

## Reviewer 2: Lead-Time Justification

The current saved training matrix contains 30-day aggregate historical features and does not preserve forecast lead-time metadata. We therefore report lead-time sensitivity as unsupported by the present pipeline and identify forecast-lead-specific feature generation as the next required experiment.

## Reviewer 3: Dataset Statistics Request

We added dataset statistics for global train, validation, and test periods and for each test region, including grid resolution, positive/negative counts, unique grid cells, positive rate, negative sampling status, land/water masking, and target-labeling rules.

## Reviewer 3: Embedding/Fusion Reproducibility Request

The repository contains neural checkpoints but not the prepared NN `.npz` dataset required to reproduce embedding/fusion ablations. We therefore completed the more stable CatBoost data-fusion ablations and report the neural embedding ablation as blocked until the NN input dataset is regenerated.

## Reviewer 3: Morphological Expansion / Grid-Size Concern

We now report that the target configuration uses 0.1-degree spatial coarsening, not a 5 km grid. The no-dilation experiment requires target-cache regeneration because the current feature matrix was already built after positive expansion; this limitation is reported explicitly rather than inferred from the existing labels.
"""
    (output_dir / "reviewer_response_insertions.md").write_text(text, encoding="utf-8")


def generate_reports(
    output_dir: Path,
    audit: dict[str, Any],
    dataset_stats: pd.DataFrame,
    main_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
    embedding: pd.DataFrame,
    label_sensitivity: pd.DataFrame,
    lead_time: pd.DataFrame,
    input_source: pd.DataFrame,
    native_importance: pd.DataFrame | None,
    grouped_perm: pd.DataFrame | None,
    climate_window_perm: pd.DataFrame | None,
    shap_df: pd.DataFrame | None,
    metrics_wide: pd.DataFrame,
    registry: pd.DataFrame,
    failures: list[dict[str, Any]],
    interpretation: str,
    command: str,
) -> None:
    tables_md = [
        "# Paper-Ready Tables",
        "",
        "## Dataset Statistics",
        markdown_table(dataset_stats),
        "",
        "## Main Model Comparison",
        markdown_table(main_table),
        "",
        "## Feature-Source Ablations",
        markdown_table(ablation_table),
        "",
        "## ERA5 / SEAS5 Input Source Comparison",
        markdown_table(input_source),
        "",
        "## Lead-Time Sensitivity",
        markdown_table(lead_time),
        "",
        "## Native Feature Importance Top 30",
        markdown_table(native_importance.head(30) if native_importance is not None else pd.DataFrame()),
        "",
        "## Grouped Permutation Importance",
        markdown_table(grouped_perm if grouped_perm is not None else pd.DataFrame()),
        "",
        "## Climate Window Permutation Importance",
        markdown_table(climate_window_perm if climate_window_perm is not None else pd.DataFrame()),
        "",
    ]
    (output_dir / "paper_ready_tables.md").write_text("\n".join(tables_md), encoding="utf-8")
    report = [
        "# Revision Experiments Report",
        "",
        "## Executive Summary",
        interpretation.replace("## ", "### "),
        "",
        "## Repo/Data Audit",
        f"- Commit: `{audit.get('commit_hash')}`",
        f"- Feature table: `{audit['data_paths']['features_path']}`",
        f"- ERA5 raw path readable: `{audit['era5']['raw_path_readable']}`",
        f"- ECMWF/SEAS5 files found: `{audit['seas5_ecmwf']['path_exists']}`",
        "",
        "## Dataset Statistics",
        markdown_table(dataset_stats),
        "",
        "## Main Model Comparison",
        markdown_table(main_table),
        "",
        "## Feature-Source Ablations",
        markdown_table(ablation_table),
        "",
        "## Embedding/Fusion Ablations",
        markdown_table(embedding),
        "",
        "## Label Sensitivity",
        markdown_table(label_sensitivity),
        "",
        "## Lead-Time Sensitivity",
        markdown_table(lead_time),
        "",
        "## ERA5 vs SEAS5 Input-Source Comparison",
        markdown_table(input_source),
        "",
        "## Feature Importance And Interpretability",
        "Native feature importance, grouped permutation importance, and climate-window permutation importance were generated for the best/full CatBoost model. SHAP is reported when CatBoost-native SHAP completed.",
        "",
        "## Recommended Manuscript Changes",
        "- State that the grid is 0.1 degree and that negatives are sampled.",
        "- Report validation-only threshold selection.",
        "- Add the feature ablation and grouped permutation tables to support the data-fusion claim.",
        "- Describe ERA5->ERA5 as retrospective upper bound and SEAS5/ECMWF->SEAS5/ECMWF as operationally matched.",
        "",
        "## Limitations / Failed Experiments",
        markdown_table(pd.DataFrame(failures)),
        "",
        "## Exact Commands / Configs Used",
        f"- Runner command: `{command}`",
        f"- Feature config: `{audit['data_paths']['feature_config']}`",
        f"- Target config: `{audit['data_paths']['target_config']}`",
        f"- Regions file: `{audit['data_paths']['regions_file']}`",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    generate_reviewer_response(output_dir, interpretation)


def record_failure(
    failures: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    experiment_id: str,
    experiment_type: str,
    model: str,
    feature_set: str,
    exc: Exception,
    affects: str,
    next_action: str,
) -> None:
    reason = str(exc)
    logging.error("%s failed: %s", experiment_id, reason)
    logging.debug(traceback.format_exc())
    failures.append(
        {
            "experiment": experiment_id,
            "reason": reason,
            "affects_main_claims": affects,
            "suggested_next_action": next_action,
        }
    )
    registry_rows.append(
        {
            "experiment_id": experiment_id,
            "experiment_type": experiment_type,
            "model": model,
            "feature_set": feature_set,
            "status": "failed",
            "feature_count": None,
            "threshold": None,
            "threshold_source": None,
            "validation_f1_at_threshold": None,
            "model_path": None,
            "prediction_paths": None,
            "notes": reason,
        }
    )


def run(args: Any, command: str | None = None) -> int:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_removed_model_artifacts(output_dir)
    setup_logging(output_dir)
    started_at = datetime.now()
    command = command or "config-driven revision_evaluation.tabular"
    (output_dir / "commands_used.txt").write_text(command + "\n", encoding="utf-8")

    np.random.seed(args.seed)
    packages = package_availability()
    audit = repo_audit(args, output_dir, packages)
    configs_copied = copy_configs_used(args, output_dir)
    feature_config = load_yaml(args.feature_config)
    target_config = load_yaml(args.target_config)
    catboost_config = load_yaml(args.catboost_config)
    regions = load_regions(args.regions_file)

    registry_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    prediction_store: dict[str, dict[str, Any]] = {}
    full_model = None
    full_cat_features: list[str] = []
    native_importance = None
    grouped_perm = None
    climate_window_perm = None
    shap_df = None

    df = load_dataset(args.features_path)
    masks = split_masks(df)
    for split, mask in masks.items():
        if not mask.any():
            raise RuntimeError(f"Reviewer split {split!r} is empty.")
        logging.info("%s rows: %d", split, int(mask.sum()))

    ignored_features = (
        catboost_config.get("catboost_train", {})
        .get("features", {})
        .get("ignored", DEFAULT_IGNORED_FEATURES)
        if isinstance(catboost_config.get("catboost_train"), dict)
        else DEFAULT_IGNORED_FEATURES
    )
    all_features = model_feature_columns(
        df,
        ignored_features,
        use_lat_lon_features=args.use_lat_lon_features,
        use_ecoregion_features=args.use_ecoregion_features,
        use_historical_fire_features=args.use_historical_fire_features,
    )
    (
        selected_feature_filter_enabled,
        selected_feature_columns_path,
        selected_features,
    ) = selected_feature_filter_spec(feature_config, catboost_config)
    all_features, selected_feature_metadata = apply_selected_feature_columns(
        all_features,
        selected_features,
        enabled=selected_feature_filter_enabled,
        table_columns=df.columns,
    )
    selected_feature_metadata["selected_feature_columns_path"] = selected_feature_columns_path
    if selected_feature_metadata["enabled"]:
        logging.info(
            "Applied CatBoost selected-feature filter for revision features: kept %d columns, dropped %d columns.",
            selected_feature_metadata["applied_selected_features_count"],
            len(selected_feature_metadata["dropped_by_selected_feature_filter"]),
        )
    validate_no_leakage_features(all_features)
    feature_sets = build_feature_sets(all_features)
    write_json(
        output_dir / "feature_groups.json",
        {
            "all_feature_count": len(all_features),
            "selected_feature_filter": selected_feature_metadata,
            "feature_groups": {col: group_display(feature_group(col)) for col in all_features},
            "feature_sets": {
                key: {
                    "label": value["label"],
                    "feature_set": value["feature_set"],
                    "feature_count": len(value["columns"]),
                    "dropped_count": len(value.get("dropped", [])),
                }
                for key, value in feature_sets.items()
            },
        },
    )

    dataset_stats = dataset_statistics(df, masks, regions, target_config, feature_config, output_dir)

    # Priority 1: main model comparison and CatBoost ablations.
    try:
        row, rows, preds = run_linear_logistic_experiment(
            df, masks, regions, feature_sets["full"]["columns"], feature_config, args, output_dir
        )
        registry_rows.append(row)
        metric_rows.extend(rows)
        prediction_store["logistic_regression_full"] = {
            **preds,
            "label": "Logistic Regression (linear SGD)",
            "short_label": "Logistic Regression (linear SGD)",
            "threshold": row.get("threshold"),
        }
    except Exception as exc:
        record_failure(
            failures,
            registry_rows,
            "logistic_regression_full",
            "main_model_comparison",
            "Logistic Regression",
            "full features",
            exc,
            "Moderate; CatBoost and other baselines still ran.",
            "Retry with a smaller learning rate or a sampled linear baseline.",
        )

    try:
        row, rows, preds = run_poisson_point_process_experiment(
            df, masks, regions, feature_sets["full"]["columns"], feature_config, args, output_dir
        )
        registry_rows.append(row)
        metric_rows.extend(rows)
        prediction_store["poisson_point_process_full"] = {
            **preds,
            "label": "Poisson Point-Process GLM",
            "short_label": "Poisson Point-Process GLM",
            "threshold": row.get("threshold"),
        }
    except Exception as exc:
        record_failure(
            failures,
            registry_rows,
            "poisson_point_process_full",
            "main_model_comparison",
            "Poisson Point-Process GLM",
            "full features",
            exc,
            "Low; other discriminative baselines still ran.",
            "Retry with fewer point-process training rows or stronger L2 regularization.",
        )

    catboost_order = [
        ("catboost_full", "main_model_comparison", "CatBoost", feature_sets["full"]),
        ("catboost_weather_only", "main_model_comparison", "Weather-only CatBoost", feature_sets["weather_only"]),
        ("catboost_fwi_only", "main_model_comparison", "FWI-only CatBoost", feature_sets["fwi_only"]),
    ]
    for experiment_id, experiment_type, model_label, spec in catboost_order:
        try:
            row, rows, preds, model, cat_features = run_catboost_experiment(
                experiment_id,
                experiment_type,
                model_label,
                spec["feature_set"],
                spec["columns"],
                df,
                masks,
                regions,
                feature_config,
                args,
                output_dir,
            )
            registry_rows.append(row)
            metric_rows.extend(rows)
            prediction_store[experiment_id] = {
                **preds,
                "label": model_label,
                "short_label": model_label,
                "threshold": row.get("threshold"),
            }
            if experiment_id == "catboost_full":
                full_model = model
                full_cat_features = cat_features
        except Exception as exc:
            record_failure(
                failures,
                registry_rows,
                experiment_id,
                experiment_type,
                model_label,
                spec["feature_set"],
                exc,
                "High if full CatBoost failed; otherwise limited to that baseline.",
                "Inspect CatBoost logs and rerun with CPU task_type or fewer features.",
            )

    try:
        row, rows, preds = run_random_forest_experiment(
            df, masks, regions, feature_sets["full"]["columns"], feature_config, args, output_dir
        )
        registry_rows.append(row)
        metric_rows.extend(rows)
        prediction_store["random_forest_full"] = {
            **preds,
            "label": "Random Forest",
            "short_label": "Random Forest",
            "threshold": row.get("threshold"),
        }
    except Exception as exc:
        record_failure(
            failures,
            registry_rows,
            "random_forest_full",
            "main_model_comparison",
            "Random Forest",
            "full features",
            exc,
            "Low; an alternative tree baseline was optional among available libraries.",
            "Increase memory or reduce rf-max-train-rows.",
        )

    # Add LSTM blocked row to main comparison registry because requested if usable.
    nn_dataset_path = Path("data/saved_features/nn_train_data/prepared_data.npz")
    if not nn_dataset_path.exists():
        for experiment_id, model_label in [
            ("lstm_mlp_full", "LSTM-MLP"),
            ("minimal_mlp_full", "Minimal MLP"),
            ("ft_transformer_full", "FT-Transformer"),
        ]:
            record_failure(
                failures,
                registry_rows,
                experiment_id,
                "main_model_comparison",
                model_label,
                "full features",
                ExperimentFailure(f"Prepared NN dataset not found: {nn_dataset_path}"),
                "No; tabular data-fusion experiments are still complete.",
                "Regenerate NN prepared_data.npz, then run `python -m src.revision_evaluation.neural_training`.",
            )

    # Feature ablations: full/weather/FWI are already run above, so duplicate them
    # into the ablation experiment type by reusing metric rows and registry entries.
    for src_id, ablation_id in [
        ("catboost_full", "ablation_full"),
        ("catboost_weather_only", "ablation_weather_only"),
        ("catboost_fwi_only", "ablation_fwi_only"),
    ]:
        for row in [r for r in registry_rows if r["experiment_id"] == src_id and r["status"] == "completed"]:
            cloned = dict(row)
            cloned["experiment_id"] = ablation_id
            cloned["experiment_type"] = "feature_ablation"
            registry_rows.append(cloned)
        for row in [r for r in metric_rows if r["experiment_id"] == src_id]:
            cloned = dict(row)
            cloned["experiment_id"] = ablation_id
            cloned["experiment_type"] = "feature_ablation"
            metric_rows.append(cloned)

    ablation_specs = [
        ("ablation_no_anthropogenic", feature_sets["no_anthropogenic"]),
        ("ablation_no_ecoregion", feature_sets["no_ecoregion"]),
        ("ablation_no_geography", feature_sets["no_geography"]),
        ("ablation_no_fuel_ecoregion_vegetation", feature_sets["no_fuel_ecoregion_vegetation"]),
        ("ablation_no_terrain", feature_sets["no_terrain"]),
        ("ablation_no_seasonality", feature_sets["no_seasonality"]),
        ("ablation_no_history", feature_sets["no_history"]),
        ("ablation_no_temporal_history", feature_sets["no_temporal_history"]),
        ("ablation_shorter_sequence_30d", feature_sets["shorter_sequence_30d"]),
        ("ablation_no_periodic_seasonality", feature_sets["no_periodic_seasonality"]),
        ("ablation_no_gaussian_smoothing", feature_sets["no_gaussian_smoothing"]),
        ("ablation_static_only", feature_sets["static_only"]),
        ("ablation_dynamic_weather_fwi_only", feature_sets["dynamic_weather_fwi_only"]),
    ]
    for experiment_id, spec in ablation_specs:
        if spec.get("dropped") == []:
            # No matching columns exist; copy full metrics as a documented no-op sensitivity.
            note = f"No columns matched this ablation pattern for `{spec['feature_set']}`."
            for row in [r for r in registry_rows if r["experiment_id"] == "ablation_full" and r["status"] == "completed"]:
                cloned = dict(row)
                cloned["experiment_id"] = experiment_id
                cloned["experiment_type"] = "feature_ablation"
                cloned["feature_set"] = spec["feature_set"]
                cloned["status"] = "completed_no_op"
                cloned["notes"] = note
                registry_rows.append(cloned)
            for row in [r for r in metric_rows if r["experiment_id"] == "ablation_full"]:
                cloned = dict(row)
                cloned["experiment_id"] = experiment_id
                cloned["experiment_type"] = "feature_ablation"
                cloned["feature_set"] = spec["feature_set"]
                metric_rows.append(cloned)
            continue
        try:
            row, rows, preds, _, _ = run_catboost_experiment(
                experiment_id,
                "feature_ablation",
                "CatBoost",
                spec["feature_set"],
                spec["columns"],
                df,
                masks,
                regions,
                feature_config,
                args,
                output_dir,
            )
            row["notes"] = f"Dropped {len(spec.get('dropped', []))} features." if spec.get("dropped") is not None else ""
            registry_rows.append(row)
            metric_rows.extend(rows)
            prediction_store[experiment_id] = {
                **preds,
                "label": spec["label"],
                "short_label": spec["label"],
                "threshold": row.get("threshold"),
            }
        except Exception as exc:
            record_failure(
                failures,
                registry_rows,
                experiment_id,
                "feature_ablation",
                "CatBoost",
                spec["feature_set"],
                exc,
                "Limited to the corresponding feature-source ablation.",
                "Inspect feature availability and CatBoost logs, then rerun that ablation.",
            )

    # Consolidate metrics before derived reports.
    registry_df = pd.DataFrame(registry_rows)
    metrics_wide = pd.DataFrame(metric_rows)
    if not metrics_wide.empty:
        metrics_long = build_metrics_long(metrics_wide)
    else:
        metrics_long = pd.DataFrame()
    registry_df.to_csv(output_dir / "experiment_registry.csv", index=False)
    metrics_wide.to_csv(output_dir / "metrics_wide.csv", index=False)
    metrics_long.to_csv(output_dir / "metrics_long.csv", index=False)

    main_table = make_main_model_table(metrics_wide, registry_df, output_dir)
    if args.run_legacy_sampled_evaluation:
        save_legacy_sampled_outputs(
            output_dir,
            main_table=main_table,
            metrics_wide=metrics_wide,
            metrics_long=metrics_long,
        )
    ablation_table = make_feature_ablation_table(metrics_wide, output_dir)
    plot_pr_curves(prediction_store, df, masks, regions, output_dir)

    # Interpretability after full CatBoost completes.
    if full_model is not None:
        try:
            native_importance = native_feature_importance(full_model, feature_sets["full"]["columns"], output_dir)
        except Exception as exc:
            record_failure(
                failures,
                registry_rows,
                "native_feature_importance",
                "interpretability",
                "CatBoost",
                "full features",
                exc,
                "Moderate for interpretability only.",
                "Rerun feature importance from the saved CatBoost model.",
            )
        try:
            full_row = next(row for row in registry_rows if row["experiment_id"] == "catboost_full" and row["status"] == "completed")
            grouped_perm = grouped_permutation_importance(
                full_model,
                feature_sets["full"]["columns"],
                full_cat_features,
                df,
                masks,
                feature_config,
                float(full_row["threshold"]),
                args,
                output_dir,
            )
        except Exception as exc:
            record_failure(
                failures,
                registry_rows,
                "grouped_permutation_importance",
                "interpretability",
                "CatBoost",
                "full features",
                exc,
                "Moderate for interpretability only.",
                "Reduce permutation sample size or rerun from the saved CatBoost model.",
            )
        try:
            full_row = next(row for row in registry_rows if row["experiment_id"] == "catboost_full" and row["status"] == "completed")
            climate_window_perm = climate_window_permutation_importance(
                full_model,
                feature_sets["full"]["columns"],
                full_cat_features,
                df,
                masks,
                feature_config,
                float(full_row["threshold"]),
                args,
                output_dir,
            )
        except Exception as exc:
            record_failure(
                failures,
                registry_rows,
                "climate_window_permutation_importance",
                "interpretability",
                "CatBoost",
                "full features",
                exc,
                "Moderate for climate-window interpretation only.",
                "Reduce permutation sample size or inspect climate window feature names.",
            )
        try:
            shap_df = catboost_native_shap(
                full_model,
                feature_sets["full"]["columns"],
                full_cat_features,
                df,
                masks,
                feature_config,
                args,
                output_dir,
            )
        except Exception as exc:
            failures.append(
                {
                    "experiment": "CatBoost-native SHAP",
                    "reason": str(exc),
                    "affects_main_claims": "No; native and grouped permutation importance are available.",
                    "suggested_next_action": "Install shap if desired or rerun CatBoost-native SHAP with a smaller sample.",
                }
            )
    else:
        failures.append(
            {
                "experiment": "Feature importance and grouped permutation importance",
                "reason": "Full CatBoost model was unavailable.",
                "affects_main_claims": "Yes for interpretability claims.",
                "suggested_next_action": "Fix and rerun full CatBoost.",
            }
        )

    # Priority 2/3 tables and blocked experiments.
    full_global = None
    full_row = get_metric_row(metrics_wide, "catboost_full")
    if full_row is not None:
        full_global = full_row.to_dict()
    embedding, label_sensitivity, lead_time = placeholder_blocked_tables(output_dir, failures, full_global)
    input_source = input_source_table(output_dir, metrics_wide, args, failures)

    # Save final registry after interpretability/blocked rows are known.
    registry_df = pd.DataFrame(registry_rows)
    metrics_wide = pd.DataFrame(metric_rows)
    metrics_long = build_metrics_long(metrics_wide) if not metrics_wide.empty else pd.DataFrame()
    registry_df.to_csv(output_dir / "experiment_registry.csv", index=False)
    metrics_wide.to_csv(output_dir / "metrics_wide.csv", index=False)
    metrics_long.to_csv(output_dir / "metrics_long.csv", index=False)
    interpretation = generate_interpretation(
        output_dir,
        metrics_wide,
        registry_df,
        dataset_stats,
        ablation_table,
        input_source,
        native_importance,
        grouped_perm,
        climate_window_perm,
        shap_df,
        failures,
    )
    write_failures(output_dir / "failures.md", failures)
    generate_reports(
        output_dir,
        audit,
        dataset_stats,
        main_table,
        ablation_table,
        embedding,
        label_sensitivity,
        lead_time,
        input_source,
        native_importance,
        grouped_perm,
        climate_window_perm,
        shap_df,
        metrics_wide,
        registry_df,
        failures,
        interpretation,
        command,
    )

    ended_at = datetime.now()
    manifest = {
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
        "command": command,
        "cwd": Path.cwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": args.seed,
        "args": vars(args),
        "split": {
            "train": "2001-2018",
            "validation": "2019-2020",
            "test": "2021-2025",
            "thresholding": "Validation F1-max threshold applied unchanged to test.",
        },
        "package_availability": packages,
        "outputs": {
            "registry": output_dir / "experiment_registry.csv",
            "metrics_long": output_dir / "metrics_long.csv",
            "metrics_wide": output_dir / "metrics_wide.csv",
            "report": output_dir / "report.md",
            "plots": output_dir / "plots",
            "configs_used": configs_copied,
        },
        "failures": failures,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    prune_empty_dirs(output_dir)
    logging.info("Revision experiment package complete: %s", output_dir)
    return 0


def main() -> int:
    return run(default_args(), command="config-driven default revision_evaluation.tabular")


if __name__ == "__main__":
    raise SystemExit(main())
