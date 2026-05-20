"""Config-driven full-grid probability overlay plots for revision evaluation.

The runner reads `revision_evaluation` full-grid calibrated test prediction
parquet files, scores each test period inside each configured region, chooses
the best period, and writes paper-ready maps with observed fire-positive
grid-cell days overlaid on predicted probabilities.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from catboost import CatBoostClassifier

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.feature_generation.make_features import make_features_from_target_df  # noqa: E402
from src.target_generation.create_grid_target import country_mapping  # noqa: E402
from src.utils.prediction_adjustments import (  # noqa: E402
    DEFAULT_DEPLOY_PRIOR,
    DEFAULT_TRAIN_PRIOR,
    adjust_probabilities_for_prior,
)


DATE_COL = "datetime"
LAT_COL = "lat_rounded"
LON_COL = "lon_rounded"
TARGET_COL = "is_fire"
WEIGHT_COL = "eval_weight"
PROB_COL = "predicted_probability"
DEFAULT_PROB_COL = "auto"


def load_prepared_nn_script_module():
    module_name = "_revision_eval_build_prepared_nn_data"
    if module_name in sys.modules:
        return sys.modules[module_name]
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "build_prepared_nn_data.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prepared NN data helpers from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

PROBABILITY_COLUMNS = ["prob_calibrated", "pred_proba", "prob_raw"]
TARGET_COLUMNS = ["is_fire", "target_binary", "count"]
MODEL_ALIASES = {
    "catboost": ["catboost_full", "CatBoost"],
    "catboost_full": ["catboost_full", "CatBoost"],
    "weather-only_catboost": ["catboost_weather_only", "Weather-only_CatBoost"],
    "fwi-only_catboost": ["catboost_fwi_only", "FWI-only_CatBoost"],
    "random_forest": ["random_forest_full", "Random_Forest"],
    "logistic_regression": ["logistic_regression_full", "Logistic_Regression_linear_SGD"],
    "spline_logistic_regression": ["spline_logistic_regression_full", "Spline_Logistic_Regression"],
    "poisson_point-process_glm": ["poisson_point_process_full", "Poisson_Point-Process_GLM"],
}

DEFAULT_NEURAL_DYNAMIC_COLUMNS = [
    f"{name}_{window}"
    for window in (7, 14, 30, 90, 120)
    for name in ("t2m_lag", "d2m_lag", "t2m_mean", "d2m_mean", "tp_mean", "stl1_mean")
]
DEFAULT_NEURAL_STATIC_COLUMNS = [
    "month",
    "elevation_point_stddev",
    "elevation_point_max",
    "elevation_point_min",
    "elevation_min_0.1deg",
    "elevation_max_0.1deg",
    "elevation_mean_0.1deg",
    "elevation_std_0.1deg",
    "elevation_gradient_min_0.1deg",
    "elevation_gradient_max_0.1deg",
    "elevation_gradient_mean_0.1deg",
    "elevation_gradient_std_0.1deg",
    "elevation_min_0.2deg",
    "elevation_max_0.2deg",
    "elevation_mean_0.2deg",
    "elevation_std_0.2deg",
    "elevation_gradient_min_0.2deg",
    "elevation_gradient_max_0.2deg",
    "elevation_gradient_mean_0.2deg",
    "elevation_gradient_std_0.2deg",
    "road_presence_1km",
    "road_density_gaussian_5km",
    "road_density_gaussian_10km",
    "road_density_gaussian_25km",
    "distance_to_road_meters",
    "night_light_radiance_2024",
    "night_light_presence_1km",
    "light_density_gaussian_5km",
    "light_density_gaussian_10km",
    "light_density_gaussian_25km",
    "distance_to_light_source_meters",
    "fire_index_drtcode_max",
    "fire_index_drtcode_max_trend",
    "fire_index_drtcode_mean",
    "fire_index_drtcode_mean_trend",
    "fire_index_drtcode_median",
    "fire_index_drtcode_median_trend",
    "fire_index_drtcode_min",
    "fire_index_drtcode_std",
    "fire_index_fbupinx_max",
    "fire_index_fbupinx_max_trend",
    "fire_index_fbupinx_mean",
    "fire_index_fbupinx_mean_trend",
    "fire_index_fbupinx_median",
    "fire_index_fbupinx_median_trend",
    "fire_index_fbupinx_min",
    "fire_index_fbupinx_std",
    "fire_index_fdsrte_max",
    "fire_index_fdsrte_max_trend",
    "fire_index_fdsrte_mean",
    "fire_index_fdsrte_mean_trend",
    "fire_index_fdsrte_median",
    "fire_index_fdsrte_median_trend",
    "fire_index_fdsrte_min",
    "fire_index_fdsrte_std",
    "fire_index_ffmcode_max",
    "fire_index_ffmcode_max_trend",
    "fire_index_ffmcode_mean",
    "fire_index_ffmcode_mean_trend",
    "fire_index_ffmcode_median",
    "fire_index_ffmcode_median_trend",
    "fire_index_ffmcode_min",
    "fire_index_ffmcode_std",
    "fire_index_fwinx_max",
    "fire_index_fwinx_max_trend",
    "fire_index_fwinx_mean",
    "fire_index_fwinx_mean_trend",
    "fire_index_fwinx_median",
    "fire_index_fwinx_median_trend",
    "fire_index_fwinx_min",
    "fire_index_fwinx_std",
    "lai_hv",
    "lai_lv",
    "landseamask",
    "population",
    "anor",
    "isor",
    "z",
    "lsm",
    "slor",
    "sdfor",
    "sdor",
    "distance_to_coast_km",
    "distance_to_coast_dilated_km",
]
DEFAULT_NEURAL_CATEGORICAL_COLUMNS = [
    "ecoregion_name",
    "ecoregion_realm",
    "slt",
    "tvl",
    "tvh",
    "slt_soil_type_stream_oper_daily_mean",
]

METRIC_DIRECTIONS = {
    "average_precision": "max",
    "spatial_tolerant_average_precision": "max",
    "roc_auc": "max",
    "spatial_tolerant_roc_auc": "max",
    "weighted_brier_score": "min",
    "spatial_tolerant_weighted_brier_score": "min",
    "count_abs_error": "min",
    "count_ratio_abs_log_error": "min",
}
DISCRIMINATION_METRICS = {
    "average_precision",
    "roc_auc",
    "spatial_tolerant_average_precision",
    "spatial_tolerant_roc_auc",
}
_DENSE_NEURAL_CACHE: dict[tuple[str, ...], "DenseNeuralPredictor"] = {}


@dataclass(frozen=True)
class ProbabilityOverlayConfig:
    results_dir: Path
    regions_file: Path = Path("configs/regions_example.yaml")
    feature_config: Path = Path("configs/features_config_30d.yaml")
    target_config: Path = Path("configs/target_config.yaml")
    source: str = "legacy"
    model: str = "best_neural"
    prob_col: str = DEFAULT_PROB_COL
    selection_metric: str = "average_precision"
    min_wildfires: int = 7
    spatial_tolerance_degrees: float = 0.0
    window_days: int = 3
    top_periods: int = 1
    max_period_end: str | None = None
    allow_overlapping_periods: bool = False
    regions: list[str] | None = None
    include_global: bool = True
    allow_partial_periods: bool = False
    map_summary: str = "sum"
    surface_source: str = "dense-neural"
    dense_model_path: Path | None = None
    dense_neural_model_path: Path | None = None
    dense_neural_training_features: Path = Path(
        "data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet"
    )
    dense_neural_batch_size: int = 8192
    dense_neural_device: str = "auto"
    overwrite_dense: bool = False
    grid_resolution: float | None = None
    interpolation_factor: int = 5
    prior_correction: bool = True
    train_prior: float = DEFAULT_TRAIN_PRIOR
    deploy_prior: float = DEFAULT_DEPLOY_PRIOR
    colormap: str = "YlOrRd"
    color_floor: float | None = None
    color_vmax: float | None = None
    verbose_feature_generation: bool = False
    country_shapes: Path = Path("data/countries")
    output_dir: Path | None = None
    source_label: str | None = None
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 320
    keep_existing_plots: bool = False


@dataclass(frozen=True)
class Region:
    name: str
    display_name: str
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        if self.name == "global":
            return np.ones(len(frame), dtype=bool)
        return (
            (frame[LAT_COL].to_numpy(dtype=float) >= float(self.lat_min))
            & (frame[LAT_COL].to_numpy(dtype=float) <= float(self.lat_max))
            & (frame[LON_COL].to_numpy(dtype=float) >= float(self.lon_min))
            & (frame[LON_COL].to_numpy(dtype=float) <= float(self.lon_max))
        )

    def extent(self, frame: pd.DataFrame) -> tuple[float, float, float, float]:
        if self.name != "global":
            return (
                float(self.lon_min),
                float(self.lon_max),
                float(self.lat_min),
                float(self.lat_max),
            )
        return (
            float(frame[LON_COL].min()),
            float(frame[LON_COL].max()),
            float(frame[LAT_COL].min()),
            float(frame[LAT_COL].max()),
        )


def safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "value"


def display_name(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def load_regions(path: Path, *, include_global: bool, only: set[str] | None) -> list[Region]:
    regions: list[Region] = []
    if include_global:
        regions.append(Region(name="global", display_name="Global"))

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entries = payload.get("regions", []) if isinstance(payload, dict) else []
    for entry in entries:
        name = str(entry["name"])
        if only is not None and name not in only:
            continue
        regions.append(
            Region(
                name=name,
                display_name=str(entry.get("display_name") or display_name(name)),
                lat_min=float(entry["lat_min"]),
                lat_max=float(entry["lat_max"]),
                lon_min=float(entry["lon_min"]),
                lon_max=float(entry["lon_max"]),
            )
        )

    if only is not None:
        loaded = {region.name for region in regions}
        missing = sorted(only - loaded)
        if missing:
            raise ValueError(f"Requested regions were not found in {path}: {', '.join(missing)}")
    if not regions:
        raise ValueError(f"No regions loaded from {path}.")
    return regions


def prediction_dirs(results_dir: Path, source: str) -> list[Path]:
    legacy_dirs = [
        results_dir / "shared_artifacts" / "predictions",
        results_dir / "predictions",
    ]
    primary_dirs = [
        results_dir / "shared_artifacts" / "primary_full_grid_calibrated" / "predictions",
        results_dir / "primary_full_grid_calibrated" / "predictions",
    ]
    if source == "legacy":
        return legacy_dirs
    if source == "primary":
        return primary_dirs
    return legacy_dirs + primary_dirs


def find_all_prediction_files(results_dir: Path, source: str) -> list[Path]:
    files: list[Path] = []
    for directory in prediction_dirs(results_dir, source):
        if directory.exists():
            files.extend(sorted(directory.glob("*_test_predictions.parquet")))
            files.extend(sorted(directory.glob("*_test_legacy_predictions.parquet")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def prediction_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_test_predictions", "_test_legacy_predictions"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def model_search_keys(model: str) -> set[str]:
    raw = str(model)
    keys = {raw, safe_slug(raw)}
    alias_key = safe_slug(raw).lower()
    keys.update(MODEL_ALIASES.get(alias_key, []))
    keys.update(MODEL_ALIASES.get(raw.lower(), []))
    return {key.lower() for key in keys}


def pretty_model_name(path: Path) -> str:
    stem = prediction_stem(path)
    known = {
        "catboost_full": "CatBoost",
        "catboost_weather_only": "Weather-only CatBoost",
        "catboost_fwi_only": "FWI-only CatBoost",
        "random_forest_full": "Random Forest",
        "logistic_regression_full": "Logistic Regression",
        "spline_logistic_regression_full": "Spline Logistic Regression",
        "poisson_point_process_full": "Poisson Point-Process GLM",
        "nn_global_full_ft_transformer": "FT-Transformer (global full)",
        "nn_global_full_lstm_attention": "LSTM attention (global full)",
        "nn_global_full_lstm_gated_moe": "LSTM gated MoE (global full)",
        "nn_global_full_lstm_static_concat": "LSTM static concat (global full)",
        "nn_global_full_spatial_tsn": "Spatial climate TSN-MLP (global full)",
        "nn_global_full_spatial_tsn_ecmwf": "Spatial climate TSN-MLP (ECMWF global full)",
        "nn_global_full_minimal_mlp": "Minimal MLP (global full)",
        "nn_global_full_tsn": "TemporalConvNet / TSN-MLP (global full)",
    }
    return known.get(stem, display_name(stem))


def find_prediction_file(results_dir: Path, model: str, source: str) -> Path:
    model_path = Path(model)
    if model_path.exists():
        return model_path

    candidates = find_all_prediction_files(results_dir, source)
    if not candidates:
        searched = "\n".join(str(path) for path in prediction_dirs(results_dir, source))
        raise FileNotFoundError(f"No *_test_predictions.parquet files found. Searched:\n{searched}")

    wanted = model_search_keys(model)
    for path in candidates:
        stem = prediction_stem(path).lower()
        if stem in wanted:
            return path

    available = "\n".join(f"  - {prediction_stem(path)} ({path})" for path in candidates)
    raise FileNotFoundError(f"Could not find test predictions for model {model!r}. Available:\n{available}")


def find_prediction_files_for_model(results_dir: Path, model: str, source: str) -> list[Path]:
    normalized = safe_slug(model).lower()
    if normalized in {"best", "best_all"}:
        files = find_all_prediction_files(results_dir, source)
    elif normalized in {"best_neural", "best_nn", "best_nns"}:
        files = [
            path
            for path in find_all_prediction_files(results_dir, source)
            if prediction_stem(path).startswith("nn_global_full_")
        ]
    else:
        files = [find_prediction_file(results_dir, model, source)]
    if not files:
        raise FileNotFoundError(f"No prediction files found for configured overlay model {model!r}.")
    return files


def default_output_dir(results_dir: Path) -> Path:
    return results_dir / "shared_artifacts" / "probability_overlays"


def choose_column(columns: set[str], requested: str, candidates: list[str], label: str, path: Path) -> str:
    if requested and requested != "auto":
        if requested not in columns:
            raise ValueError(f"{path} is missing requested {label} column {requested!r}.")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"{path} is missing a usable {label} column. Tried: {', '.join(candidates)}")


def results_dir_from_prediction_path(path: Path) -> Path | None:
    resolved = path.resolve()
    for parent in [resolved.parent, *resolved.parents]:
        if parent.name == "shared_artifacts":
            return parent.parent
        if parent.name == "predictions" and (parent.parent / "neural_model_metrics").is_dir():
            return parent.parent
        if (parent / "neural_model_metrics").is_dir() and (parent / "predictions").is_dir():
            return parent
    return None


def split_code_from_prediction_path(path: Path) -> int:
    stem = path.stem
    if stem.endswith("_validation_predictions") or stem.endswith("_validation_legacy_predictions"):
        return 1
    return 2


def neural_data_paths(path: Path) -> list[Path]:
    results_dir = results_dir_from_prediction_path(path)
    candidates: list[Path] = []
    if results_dir is not None:
        candidates.append(results_dir / "shared_artifacts" / "neural_data" / "prepared_data.npz")
        metric_candidates = [
            results_dir / "neural_model_metrics" / f"{prediction_stem(path)}_metrics.json",
            results_dir / "shared_artifacts" / "neural_model_metrics" / f"{prediction_stem(path)}_metrics.json",
        ]
        for metrics_path in metric_candidates:
            if not metrics_path.is_file():
                continue
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            data_path = payload.get("data_path")
            if data_path:
                candidates.append(Path(data_path))
    stem = prediction_stem(path)
    if stem in {"nn_global_full_spatial_tsn", "nn_global_full_spatial_tsn_no_tp"}:
        candidates.extend(
            [
                Path("data/saved_features/nn_train_data_daily_spatial_3x3_no_tp/prepared_data.npz"),
                Path("/home/ids/vmorozov/data/saved_features/nn_train_data_daily_spatial_3x3_no_tp/prepared_data.npz"),
            ]
        )
    direct_metrics = path.parent.parent / "metrics.json"
    if direct_metrics.is_file():
        with direct_metrics.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        data_path = payload.get("data_path")
        if data_path:
            candidates.append(Path(data_path))
    candidates.append(Path("data/saved_features/nn_train_data/prepared_data.npz"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def readable_neural_data_paths(path: Path, required_keys: Iterable[str] = ()) -> list[Path]:
    required = set(required_keys)
    readable: list[Path] = []
    for data_path in neural_data_paths(path):
        if not data_path.is_file():
            continue
        try:
            with np.load(data_path) as data:
                if required and not required.issubset(set(data.files)):
                    continue
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
        readable.append(data_path)
    return readable


def neural_coordinate_frame(path: Path, row_count: int) -> pd.DataFrame | None:
    split_code = split_code_from_prediction_path(path)
    required_keys = ["split", "lat", "lon", "dates"]
    for data_path in readable_neural_data_paths(path, required_keys):
        frame = neural_coordinate_frame_from_data(data_path, split_code, row_count)
        if frame is not None:
            return frame
    return None


def neural_coordinate_frame_from_data(data_path: Path, split_code: int, row_count: int) -> pd.DataFrame | None:
    with np.load(data_path) as data:
        keys = set(data.files)
        required = {"split", "lat", "lon", "dates"}
        if not required.issubset(keys):
            return None
        split = np.asarray(data["split"], dtype=np.int8)
        mask = split == split_code
        if int(mask.sum()) != int(row_count):
            return None
        return pd.DataFrame(
            {
                DATE_COL: pd.to_datetime(np.asarray(data["dates"])[mask], unit="D", errors="coerce"),
                LAT_COL: np.asarray(data["lat"])[mask].astype("float32"),
                LON_COL: np.asarray(data["lon"])[mask].astype("float32"),
            }
        )


def read_prediction_columns(path: Path, prob_col: str) -> pd.DataFrame:
    schema_names = set(pq.read_schema(path).names)
    source_prob_col = choose_column(schema_names, prob_col, PROBABILITY_COLUMNS, "probability", path)
    source_target_col = choose_column(schema_names, "auto", TARGET_COLUMNS, "target", path)
    coordinate_cols = {DATE_COL, LAT_COL, LON_COL}
    required = coordinate_cols | {source_target_col, source_prob_col}
    missing = sorted(required - schema_names)
    has_coordinates = coordinate_cols.issubset(schema_names)
    if missing and has_coordinates:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if missing and not coordinate_cols.issubset(set(missing)):
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    optional = [WEIGHT_COL, "model_name", "model_type", "country", "split_name"]
    base_columns = [source_target_col, source_prob_col]
    if has_coordinates:
        base_columns = [DATE_COL, LAT_COL, LON_COL] + base_columns
    columns = base_columns + [
        col for col in optional if col in schema_names and col not in {source_target_col, source_prob_col}
    ]
    frame = pd.read_parquet(path, columns=columns)
    if not has_coordinates:
        coordinates = neural_coordinate_frame(path, len(frame))
        if coordinates is None:
            raise ValueError(
                f"{path} is missing {DATE_COL}/{LAT_COL}/{LON_COL}, and matching neural "
                "coordinate metadata was not found."
            )
        frame = pd.concat([coordinates.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL]).dt.normalize()
    frame[LAT_COL] = pd.to_numeric(frame[LAT_COL], errors="coerce").astype("float32")
    frame[LON_COL] = pd.to_numeric(frame[LON_COL], errors="coerce").astype("float32")
    target_values = pd.to_numeric(frame[source_target_col], errors="coerce").fillna(0)
    frame[TARGET_COL] = (target_values > 0).astype("int8")
    frame[PROB_COL] = pd.to_numeric(frame[source_prob_col], errors="coerce").clip(lower=0.0, upper=1.0)
    if WEIGHT_COL not in frame.columns:
        frame[WEIGHT_COL] = 1.0
    frame[WEIGHT_COL] = pd.to_numeric(frame[WEIGHT_COL], errors="coerce").fillna(1.0).astype("float64")
    if "model_name" not in frame.columns:
        frame["model_name"] = pretty_model_name(path)
    if "model_type" not in frame.columns:
        frame["model_type"] = frame["model_name"]
    frame["source_probability_col"] = source_prob_col
    frame["source_target_col"] = source_target_col
    frame = frame.dropna(subset=[DATE_COL, LAT_COL, LON_COL, PROB_COL, WEIGHT_COL])
    frame = frame[frame[WEIGHT_COL] > 0].copy()
    keep = [
        DATE_COL,
        LAT_COL,
        LON_COL,
        TARGET_COL,
        WEIGHT_COL,
        PROB_COL,
        "model_name",
        "model_type",
        "source_probability_col",
        "source_target_col",
    ]
    if "country" in frame.columns:
        keep.append("country")
    return frame[keep]


@dataclass
class DenseNeuralPredictor:
    model: object
    dynamic_columns: list[str]
    static_columns: list[str]
    categorical_columns: list[str]
    spatial_coordinate_columns: list[str]
    dynamic_shape: tuple[int, ...]
    dynamic_mode: str
    dyn_fill: np.ndarray
    dyn_mean: np.ndarray
    dyn_std: np.ndarray
    stat_fill: np.ndarray
    stat_mean: np.ndarray
    stat_std: np.ndarray
    category_maps: dict[str, dict[str, int]]
    batch_size: int
    device: str
    dynamic_metadata: dict[str, object] = field(default_factory=dict)
    feature_config: dict[str, object] = field(default_factory=dict)
    masked_dynamic_variables: tuple[str, ...] = field(default_factory=tuple)

    def _numeric_matrix(self, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
        values = []
        for col in columns:
            if col in frame.columns:
                values.append(pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32))
            else:
                values.append(np.full(len(frame), np.nan, dtype=np.float32))
        return np.column_stack(values).astype(np.float32) if values else np.zeros((len(frame), 0), dtype=np.float32)

    @staticmethod
    def _scale(x: np.ndarray, fill: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        x_filled = np.where(np.isnan(x), fill, x).astype(np.float32)
        return ((x_filled - mean) / std).astype(np.float32)

    def _apply_mask_to_scaled_dynamic(self, dyn_scaled: np.ndarray) -> np.ndarray:
        if not self.masked_dynamic_variables or not self.dynamic_columns:
            return dyn_scaled
        mask_columns = [
            any(col == variable or col.startswith(f"{variable}_") for variable in self.masked_dynamic_variables)
            for col in self.dynamic_columns
        ]
        if not any(mask_columns):
            return dyn_scaled
        out = dyn_scaled.copy()
        out[:, np.asarray(mask_columns, dtype=bool)] = 0.0
        return out

    def _dense_climate_params(self) -> tuple[Path, list[str], int, int]:
        climate_params = self.feature_config.get("climate_data_params", {}) if self.feature_config else {}
        variables = (
            climate_params.get("climate_variables")
            or self.dynamic_metadata.get("daily_climate_variables")
            or ["t2m", "d2m", "tp", "stl1"]
        )
        variables = [str(variable) for variable in variables]
        n_days = int(climate_params.get("n_days") or self.dynamic_shape[0])
        if self.dynamic_mode == "daily_spatial":
            patch_size = int(
                self.dynamic_metadata.get("daily_spatial_patch_size")
                or (self.dynamic_shape[1] if len(self.dynamic_shape) >= 3 else 1)
            )
        else:
            patch_size = 1
        climate_data_dir = (
            climate_params.get("climate_data_dir")
            or self.dynamic_metadata.get("climate_data_dir")
        )
        if not climate_data_dir:
            raise ValueError(
                "Dense neural prediction needs climate_data_params.climate_data_dir "
                f"for dynamic_mode={self.dynamic_mode!r}."
            )
        expected_channels = int(self.dynamic_shape[-1])
        if len(variables) != expected_channels:
            raise ValueError(
                f"Dense neural climate variable mismatch: model expects {expected_channels} "
                f"channels, feature config provides {len(variables)} ({variables})."
            )
        return Path(str(climate_data_dir)), variables, n_days, patch_size

    def _scale_daily_block(self, raw: np.ndarray, variable: str) -> np.ndarray:
        stats_by_variable = self.dynamic_metadata.get("daily_dynamic_stats") or {}
        if not isinstance(stats_by_variable, dict) or variable not in stats_by_variable:
            raise ValueError(
                f"Dense neural prediction cannot scale daily climate variable {variable!r}; "
                "the trained model metadata is missing daily_dynamic_stats."
            )
        stats = stats_by_variable[variable]
        fill = np.asarray(stats.get("fill", 0.0), dtype=np.float32)
        if raw.ndim == 4 and fill.ndim == 1:
            fill = fill[:, None, None]
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0))
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0
        filled = np.where(np.isfinite(raw), raw, fill).astype(np.float32)
        return ((filled - mean) / std).astype(np.float32)

    def _extract_dense_daily_patches(
        self,
        frame: pd.DataFrame,
        *,
        climate_data_dir: Path,
        variable: str,
        n_days: int,
        patch_size: int,
    ) -> np.ndarray:
        import polars as pl
        from src.feature_generation.load_climate_data import load_climate_variable_mf
        from src.feature_generation.prepare_climate_data import discover_climate_fragments

        prepared_nn = load_prepared_nn_script_module()
        expanded_spatial_range = prepared_nn.expanded_spatial_range
        extract_spatial_climate_timeseries = prepared_nn.extract_spatial_climate_timeseries
        extract_spatial_climate_timeseries_fragmented = prepared_nn.extract_spatial_climate_timeseries_fragmented

        target_pd = frame[[DATE_COL, LAT_COL, LON_COL]].copy()
        target_pd = target_pd.rename(columns={DATE_COL: "acq_date"})
        target_pd["acq_date"] = pd.to_datetime(target_pd["acq_date"], errors="coerce")
        target_df_pl = pl.from_pandas(target_pd[["acq_date", LAT_COL, LON_COL]])

        fragments = discover_climate_fragments(str(climate_data_dir), variable)
        if len(fragments) > 1:
            return extract_spatial_climate_timeseries_fragmented(
                fragments,
                variable,
                target_df_pl,
                n_days=n_days,
                patch_size=patch_size,
            )

        fragment = fragments[0]
        radius = patch_size // 2
        lat_range = expanded_spatial_range(target_pd[LAT_COL], fragment.lat_step, radius)
        lon_range = expanded_spatial_range(target_pd[LON_COL], fragment.lon_step, radius)
        time_range = (
            target_pd["acq_date"].min() - pd.DateOffset(days=n_days - 1),
            target_pd["acq_date"].max(),
        )
        ds = load_climate_variable_mf(
            str(climate_data_dir),
            variable,
            time_range=time_range,
            lat_range=lat_range,
            lon_range=lon_range,
            test_mode=False,
        )
        try:
            return extract_spatial_climate_timeseries(
                ds,
                variable,
                target_df_pl,
                n_days=n_days,
                patch_size=patch_size,
            )
        finally:
            ds.close()

    def _dense_daily_dynamic_tensor(self, frame: pd.DataFrame) -> np.ndarray:
        if not {DATE_COL, LAT_COL, LON_COL}.issubset(frame.columns):
            raise ValueError(
                f"Dense neural prediction for dynamic_mode={self.dynamic_mode!r} requires "
                f"{DATE_COL!r}, {LAT_COL!r}, and {LON_COL!r} columns."
            )
        climate_data_dir, variables, n_days, patch_size = self._dense_climate_params()
        if self.dynamic_mode == "daily":
            x_dyn = np.empty((len(frame), n_days, len(variables)), dtype=np.float32)
        else:
            x_dyn = np.empty((len(frame), n_days, patch_size, patch_size, len(variables)), dtype=np.float32)
        print(
            f"  building dense {self.dynamic_mode} tensor from {climate_data_dir} "
            f"({len(frame):,} rows, {n_days} days, variables={variables})"
        )
        for var_idx, variable in enumerate(variables):
            if variable in self.masked_dynamic_variables:
                print(f"  masking dense neural dynamic variable {variable!r} to scaled mean")
                if self.dynamic_mode == "daily":
                    x_dyn[:, :, var_idx] = 0.0
                else:
                    x_dyn[..., var_idx] = 0.0
                continue
            raw = self._extract_dense_daily_patches(
                frame,
                climate_data_dir=climate_data_dir,
                variable=variable,
                n_days=n_days,
                patch_size=patch_size,
            )
            scaled = self._scale_daily_block(raw, variable)
            if self.dynamic_mode == "daily":
                x_dyn[:, :, var_idx] = scaled[:, :, 0, 0]
            else:
                x_dyn[..., var_idx] = scaled
        return x_dyn

    def tensors_from_features(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        present_dynamic = [col for col in self.dynamic_columns if col in frame.columns]
        if len(present_dynamic) != len(self.dynamic_columns):
            missing_count = len(self.dynamic_columns) - len(present_dynamic)
            if self.dynamic_mode in {"daily", "daily_spatial"}:
                x_dyn = self._dense_daily_dynamic_tensor(frame)
            elif not present_dynamic:
                raise ValueError(
                    "Dense neural prediction cannot build dynamic tensor for "
                    f"dynamic_mode={self.dynamic_mode!r}: feature frame is missing "
                    f"{missing_count}/{len(self.dynamic_columns)} dynamic source columns. "
                    "Use a summary-feature neural model for dense overlays, or generate "
                    "dense daily/daily-spatial NN features first."
                )
            else:
                dyn_flat = self._numeric_matrix(frame, self.dynamic_columns)
                dyn_scaled = self._scale(dyn_flat, self.dyn_fill, self.dyn_mean, self.dyn_std)
                dyn_scaled = self._apply_mask_to_scaled_dynamic(dyn_scaled)
                x_dyn = dyn_scaled.reshape((len(frame), *self.dynamic_shape))
        else:
            dyn_flat = self._numeric_matrix(frame, self.dynamic_columns)
            dyn_scaled = self._scale(dyn_flat, self.dyn_fill, self.dyn_mean, self.dyn_std)
            dyn_scaled = self._apply_mask_to_scaled_dynamic(dyn_scaled)
            x_dyn = dyn_scaled.reshape((len(frame), *self.dynamic_shape))

        stat_raw = self._numeric_matrix(frame, self.static_columns)
        x_stat = self._scale(stat_raw, self.stat_fill, self.stat_mean, self.stat_std)
        if self.spatial_coordinate_columns:
            coords = build_spatial_coordinate_features(frame[LAT_COL].to_numpy(), frame[LON_COL].to_numpy())
            x_stat = np.concatenate([x_stat, coords], axis=1).astype(np.float32)

        x_cat = np.zeros((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for idx, col in enumerate(self.categorical_columns):
            mapping = self.category_maps.get(col, {})
            if col in frame.columns:
                values = frame[col].fillna("__missing__").astype(str)
            else:
                values = pd.Series(["__missing__"] * len(frame), index=frame.index)
            x_cat[:, idx] = values.map(mapping).fillna(0).astype(np.int64).to_numpy()
        return x_dyn, x_stat, x_cat

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        import torch

        x_dyn, x_stat, x_cat = self.tensors_from_features(frame)
        probs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(frame), self.batch_size):
                end = min(start + self.batch_size, len(frame))
                dyn = torch.as_tensor(x_dyn[start:end], dtype=torch.float32, device=self.device)
                stat = torch.as_tensor(x_stat[start:end], dtype=torch.float32, device=self.device)
                cat = torch.as_tensor(x_cat[start:end], dtype=torch.long, device=self.device)
                logits = self.model(dyn, stat, cat if cat.numel() else None)
                probs.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
        return np.concatenate(probs).astype(np.float32) if probs else np.zeros((0,), dtype=np.float32)


def parse_neural_schema(results_dir: Path) -> dict | None:
    candidates = [
        results_dir / "shared_artifacts" / "source_markdown" / "neural_data_schema.md",
        results_dir / "neural_data_schema.md",
    ]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
    return None


def parse_neural_metadata(data_path: Path) -> dict | None:
    metadata_path = data_path.with_name("prepared_metadata.json")
    if not metadata_path.is_file():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def load_neural_encoders(data_path: Path) -> dict:
    encoders_path = data_path.with_name("encoders_meta.joblib")
    if not encoders_path.is_file():
        return {}
    import joblib

    payload = joblib.load(encoders_path)
    return payload if isinstance(payload, dict) else {}


def infer_neural_schema_from_current_dataset(prediction_path: Path, training_features_path: Path) -> dict:
    data_path = next(iter(readable_neural_data_paths(prediction_path)), None)
    if data_path is None:
        raise FileNotFoundError(
            f"Could not infer neural feature schema because no prepared_data.npz was found for {prediction_path}"
        )
    if not training_features_path.is_file():
        raise FileNotFoundError(
            f"Could not infer neural feature schema because training features are missing: {training_features_path}"
        )

    feature_columns = set(pq.read_schema(training_features_path).names)
    with np.load(data_path) as data:
        if "x_dyn" in data:
            dynamic_shape = tuple(int(value) for value in data["x_dyn"].shape[1:])
        elif "x_dyn_train" in data:
            dynamic_shape = tuple(int(value) for value in data["x_dyn_train"].shape[1:])
        else:
            raise KeyError(f"{data_path} is missing x_dyn/x_dyn_train arrays.")

        if "x_static" in data:
            static_dim = int(data["x_static"].shape[1])
        elif "x_stat" in data:
            static_dim = int(data["x_stat"].shape[1])
        elif "x_stat_train" in data:
            static_dim = int(data["x_stat_train"].shape[1])
        else:
            raise KeyError(f"{data_path} is missing x_static/x_stat arrays.")

        if "x_cat" in data:
            categorical_dim = int(data["x_cat"].shape[1])
        elif "x_cat_train" in data:
            categorical_dim = int(data["x_cat_train"].shape[1])
        else:
            categorical_dim = 0

    dynamic_columns = [col for col in DEFAULT_NEURAL_DYNAMIC_COLUMNS if col in feature_columns]
    if len(dynamic_shape) != 2 or len(dynamic_columns) != int(np.prod(dynamic_shape)):
        raise FileNotFoundError(
            "Could not infer neural_data_schema.md for this prepared_data.npz shape. "
            f"Expected {int(np.prod(dynamic_shape))} dynamic columns, found {len(dynamic_columns)}."
        )

    static_columns = [col for col in DEFAULT_NEURAL_STATIC_COLUMNS if col in feature_columns]
    if len(static_columns) != static_dim:
        raise FileNotFoundError(
            "Could not infer neural_data_schema.md for this static feature shape. "
            f"Expected {static_dim} static columns, found {len(static_columns)}."
        )

    categorical_columns = [col for col in DEFAULT_NEURAL_CATEGORICAL_COLUMNS if col in feature_columns]
    if len(categorical_columns) != categorical_dim:
        raise FileNotFoundError(
            "Could not infer neural_data_schema.md for this categorical feature shape. "
            f"Expected {categorical_dim} categorical columns, found {len(categorical_columns)}."
        )

    return {
        "dynamic_source_columns": dynamic_columns,
        "static_columns": static_columns,
        "categorical_columns": categorical_columns,
        "dynamic_shape": [None, *dynamic_shape],
        "source": "inferred_from_prepared_data_and_training_features",
    }


def load_neural_schema(results_dir: Path, prediction_path: Path, training_features_path: Path) -> dict:
    for data_path in neural_data_paths(prediction_path):
        if not data_path.is_file():
            continue
        schema = parse_neural_metadata(data_path)
        if schema is not None:
            return schema
    schema = parse_neural_schema(results_dir)
    if schema is not None:
        return schema
    return infer_neural_schema_from_current_dataset(prediction_path, training_features_path)


def neural_metrics_payload(results_dir: Path, prediction_path: Path) -> dict:
    exp_id = prediction_stem(prediction_path)
    candidates = [
        results_dir / "neural_model_metrics" / f"{exp_id}_metrics.json",
        results_dir / "shared_artifacts" / "neural_model_metrics" / f"{exp_id}_metrics.json",
        prediction_path.parent.parent / "metrics.json",
    ]
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    return {}


def spatial_coordinate_columns(payload: dict[str, object]) -> list[str]:
    coordinate_info = ((payload.get("spatial_training") or {}).get("coordinate_features") or {})
    if not isinstance(coordinate_info, dict) or not coordinate_info.get("enabled", False):
        return []
    return [str(col) for col in coordinate_info.get("columns") or []]


def build_spatial_coordinate_features(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float32).reshape(-1)
    lon = np.asarray(lon, dtype=np.float32).reshape(-1)
    if lat.shape[0] != lon.shape[0]:
        raise ValueError(f"Latitude/longitude row mismatch: {lat.shape[0]} vs {lon.shape[0]}")
    lat_clean = np.nan_to_num(lat, nan=0.0, posinf=90.0, neginf=-90.0)
    lon_clean = np.nan_to_num(lon, nan=0.0, posinf=180.0, neginf=-180.0)
    lat_rad = np.deg2rad(lat_clean).astype(np.float32)
    lon_rad = np.deg2rad(lon_clean).astype(np.float32)
    return np.column_stack(
        [
            np.clip(lat_clean / 90.0, -1.0, 1.0),
            np.clip(lon_clean / 180.0, -1.0, 1.0),
            np.sin(lat_rad),
            np.cos(lat_rad),
            np.sin(lon_rad),
            np.cos(lon_rad),
        ]
    ).astype(np.float32)


def training_scaler_stats(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fill = np.nanmedian(train, axis=0)
    fill = np.nan_to_num(fill, nan=0.0).astype(np.float32)
    filled = np.where(np.isnan(train), fill, train).astype(np.float32)
    mean = filled.mean(axis=0).astype(np.float32)
    std = filled.std(axis=0).astype(np.float32)
    std = np.where(std <= 0, 1.0, std).astype(np.float32)
    return fill, mean, std


def read_training_rows(path: Path, columns: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(columns + ["year"]))
    try:
        return pd.read_parquet(path, columns=columns, filters=[("year", ">=", 2001), ("year", "<=", 2018)])
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        return frame[(pd.to_numeric(frame["year"], errors="coerce") >= 2001) & (pd.to_numeric(frame["year"], errors="coerce") <= 2018)].copy()


def infer_categorical_maps_from_prepared_data(
    prediction_path: Path,
    training_features_path: Path,
    categorical_columns: list[str],
) -> dict[str, dict[str, int]]:
    if not categorical_columns or not training_features_path.is_file():
        return {}
    data_path = next(iter(readable_neural_data_paths(prediction_path, ["x_cat"])), None)
    if data_path is None:
        return {}
    with np.load(data_path) as data:
        if "x_cat" in data:
            x_cat = np.asarray(data["x_cat"], dtype=np.int64)
        else:
            return {}
    if x_cat.ndim != 2 or x_cat.shape[1] != len(categorical_columns):
        return {}

    frame = pd.read_parquet(training_features_path, columns=categorical_columns)
    if len(frame) != x_cat.shape[0]:
        return {}

    maps: dict[str, dict[str, int]] = {}
    for idx, col in enumerate(categorical_columns):
        values = frame[col].fillna("__missing__").astype(str)
        codes = x_cat[:, idx]
        pairs = pd.DataFrame({"value": values, "code": codes})
        nunique = pairs.groupby("value", observed=True)["code"].nunique()
        if not nunique.empty and int(nunique.max()) > 1:
            continue
        maps[col] = {
            str(row.value): int(row.code)
            for row in pairs.drop_duplicates(["value", "code"]).itertuples(index=False)
        }
    return maps


def infer_dense_neural_model_path(results_dir: Path, prediction_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    exp_id = prediction_stem(prediction_path)
    metric_paths = [
        results_dir / "neural_model_metrics" / f"{exp_id}_metrics.json",
        results_dir / "shared_artifacts" / "neural_model_metrics" / f"{exp_id}_metrics.json",
        prediction_path.parent.parent / "metrics.json",
    ]
    for metrics_path in metric_paths:
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_path = Path(str(payload.get("model_path") or ""))
        if model_path.exists():
            return model_path
    for model_dir in [results_dir / "models", results_dir / "shared_artifacts" / "models", Path("models")]:
        matches = sorted(model_dir.glob(f"{exp_id}*.ckpt"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not infer neural checkpoint for {exp_id}")


def load_dense_neural_predictor(
    *,
    results_dir: Path,
    prediction_path: Path,
    training_features_path: Path,
    model_path: Path | None,
    batch_size: int,
    device: str,
    feature_config_path: Path | None = None,
    feature_config: dict[str, object] | None = None,
    masked_dynamic_variables: Sequence[str] | None = None,
) -> DenseNeuralPredictor:
    import torch
    from src.neural_net.models.lightning import SequenceStaticLightningModule

    resolved_model = infer_dense_neural_model_path(results_dir, prediction_path, model_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    masked_dynamic_variables = tuple(
        dict.fromkeys(str(variable).strip() for variable in (masked_dynamic_variables or []) if str(variable).strip())
    )
    feature_config_fingerprint = json.dumps(
        (feature_config or {}).get("climate_data_params", {}),
        sort_keys=True,
        default=str,
    )
    cache_key = (
        str(results_dir.resolve()),
        str(prediction_path.resolve()),
        str(training_features_path.resolve()),
        str(resolved_model.resolve()),
        str(feature_config_path.resolve()) if feature_config_path is not None else "",
        feature_config_fingerprint,
        ",".join(masked_dynamic_variables),
        str(batch_size),
        device,
    )
    if cache_key in _DENSE_NEURAL_CACHE:
        return _DENSE_NEURAL_CACHE[cache_key]

    data_path = next((path for path in neural_data_paths(prediction_path) if path.is_file()), None)
    encoders = load_neural_encoders(data_path) if data_path is not None else {}
    schema = load_neural_schema(results_dir, prediction_path, training_features_path)
    metrics_payload = neural_metrics_payload(results_dir, prediction_path)
    dynamic_columns = [str(col) for col in schema["dynamic_source_columns"]]
    static_columns = [str(col) for col in schema["static_columns"]]
    categorical_columns = [str(col) for col in schema.get("categorical_columns", [])]
    coordinate_columns = [col for col in spatial_coordinate_columns(metrics_payload) if col not in static_columns]
    dynamic_shape = tuple(int(value) for value in schema["dynamic_shape"][1:])
    dynamic_mode = str(encoders.get("dynamic_mode") or schema.get("dynamic_mode") or "summary")
    dynamic_metadata = {
        "daily_dynamic_stats": encoders.get("daily_dynamic_stats") or schema.get("daily_dynamic_stats"),
        "daily_climate_variables": encoders.get("daily_climate_variables") or schema.get("daily_climate_variables"),
        "daily_spatial_patch_size": encoders.get("daily_spatial_patch_size") or schema.get("daily_spatial_patch_size"),
        "climate_data_dir": encoders.get("climate_data_dir") or schema.get("climate_data_dir"),
    }

    dyn_fill = encoders.get("dynamic_fill")
    dyn_mean = encoders.get("dynamic_mean")
    dyn_std = encoders.get("dynamic_std")
    stat_fill = encoders.get("static_fill")
    stat_mean = encoders.get("static_mean")
    stat_std = encoders.get("static_std")
    if dynamic_mode in {"daily", "daily_spatial"} and any(value is None for value in [dyn_fill, dyn_mean, dyn_std]):
        # Dense overlays currently consume tabular feature frames. For daily and
        # spatial-daily NNs, the prediction call below will fail fast unless the
        # caller provides the explicit dense dynamic tensor columns.
        dyn_fill = np.zeros(len(dynamic_columns), dtype=np.float32)
        dyn_mean = np.zeros(len(dynamic_columns), dtype=np.float32)
        dyn_std = np.ones(len(dynamic_columns), dtype=np.float32)
    category_maps: dict[str, dict[str, int]] = {
        str(col): {str(key): int(value) for key, value in mapping.items()}
        for col, mapping in (encoders.get("categorical_maps") or {}).items()
        if isinstance(mapping, dict)
    }

    train = None
    needs_training_stats = any(value is None for value in [dyn_fill, dyn_mean, dyn_std, stat_fill, stat_mean, stat_std])
    needs_category_fallback = bool(categorical_columns) and len(category_maps) < len(categorical_columns)
    if needs_training_stats or needs_category_fallback:
        train_columns = dynamic_columns + static_columns + categorical_columns
        train = read_training_rows(training_features_path, train_columns)

        def numeric_frame(columns: list[str]) -> np.ndarray:
            arrays = []
            for col in columns:
                if col in train.columns:
                    arrays.append(pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=np.float32))
                else:
                    arrays.append(np.full(len(train), np.nan, dtype=np.float32))
            return np.column_stack(arrays).astype(np.float32) if arrays else np.zeros((len(train), 0), dtype=np.float32)

        if any(value is None for value in [dyn_fill, dyn_mean, dyn_std]):
            dyn_fill, dyn_mean, dyn_std = training_scaler_stats(numeric_frame(dynamic_columns))
        if any(value is None for value in [stat_fill, stat_mean, stat_std]):
            stat_fill, stat_mean, stat_std = training_scaler_stats(numeric_frame(static_columns))

    dyn_fill = np.asarray(dyn_fill, dtype=np.float32)
    dyn_mean = np.asarray(dyn_mean, dtype=np.float32)
    dyn_std = np.asarray(dyn_std, dtype=np.float32)
    stat_fill = np.asarray(stat_fill, dtype=np.float32)
    stat_mean = np.asarray(stat_mean, dtype=np.float32)
    stat_std = np.asarray(stat_std, dtype=np.float32)
    if dyn_fill.size != len(dynamic_columns) or dyn_mean.size != len(dynamic_columns) or dyn_std.size != len(dynamic_columns):
        raise ValueError(
            f"Dense neural scaler shape mismatch for {prediction_stem(prediction_path)}: "
            f"dynamic columns={len(dynamic_columns)}, fill={dyn_fill.shape}, mean={dyn_mean.shape}, std={dyn_std.shape}."
        )
    if stat_fill.size != len(static_columns) or stat_mean.size != len(static_columns) or stat_std.size != len(static_columns):
        raise ValueError(
            f"Dense neural scaler shape mismatch for {prediction_stem(prediction_path)}: "
            f"static columns={len(static_columns)}, fill={stat_fill.shape}, mean={stat_mean.shape}, std={stat_std.shape}."
        )

    schema_category_maps = {
        str(col): {str(key): int(value) for key, value in mapping.items()}
        for col, mapping in (schema.get("category_maps") or {}).items()
        if isinstance(mapping, dict)
    }
    category_maps.update({col: mapping for col, mapping in schema_category_maps.items() if col not in category_maps})
    inferred_maps = infer_categorical_maps_from_prepared_data(
        prediction_path,
        training_features_path,
        categorical_columns,
    )
    for col in categorical_columns:
        if col in category_maps:
            continue
        if col in inferred_maps:
            category_maps[col] = inferred_maps[col]
            continue
        if train is not None and col in train.columns:
            values = train[col].fillna("__missing__").astype(str)
        else:
            values = pd.Series(["__missing__"])
        cats = pd.Index(values.unique())
        category_maps[col] = {str(value): idx + 1 for idx, value in enumerate(cats)}
    if train is not None:
        del train

    model = SequenceStaticLightningModule.load_from_checkpoint(str(resolved_model))
    model.to(device)
    model.eval()
    predictor = DenseNeuralPredictor(
        model=model,
        dynamic_columns=dynamic_columns,
        static_columns=static_columns,
        categorical_columns=categorical_columns,
        spatial_coordinate_columns=coordinate_columns,
        dynamic_shape=dynamic_shape,
        dynamic_mode=dynamic_mode,
        dyn_fill=dyn_fill,
        dyn_mean=dyn_mean,
        dyn_std=dyn_std,
        stat_fill=stat_fill,
        stat_mean=stat_mean,
        stat_std=stat_std,
        category_maps=category_maps,
        batch_size=int(batch_size),
        device=device,
        dynamic_metadata={key: value for key, value in dynamic_metadata.items() if value is not None},
        feature_config=dict(feature_config or {}),
        masked_dynamic_variables=masked_dynamic_variables,
    )
    _DENSE_NEURAL_CACHE[cache_key] = predictor
    return predictor


def first_value(frame: pd.DataFrame, column: str, fallback: str) -> str:
    if column in frame.columns and not frame[column].dropna().empty:
        return str(frame[column].dropna().iloc[0])
    return fallback


def finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else math.nan


def weighted_brier(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    return float(np.average((p - y) ** 2, weights=w))


def spatial_tolerant_binary_labels(period: pd.DataFrame, radius_degrees: float) -> np.ndarray:
    """Mark sampled cells near an observed fire as positive for visualization scoring."""
    y = period[TARGET_COL].to_numpy(dtype=int)
    if radius_degrees <= 0 or not np.any(y > 0):
        return y

    positive = period.loc[period[TARGET_COL] > 0, [LAT_COL, LON_COL]].drop_duplicates()
    if positive.empty:
        return y

    lat = period[LAT_COL].to_numpy(dtype=float)
    lon = period[LON_COL].to_numpy(dtype=float)
    pos_lat = positive[LAT_COL].to_numpy(dtype=float)
    pos_lon = positive[LON_COL].to_numpy(dtype=float)
    cos_lat = max(math.cos(math.radians(float(np.nanmean(pos_lat)))), 0.2)
    tolerant = y > 0
    radius_sq = float(radius_degrees) ** 2

    for start in range(0, len(pos_lat), 256):
        end = min(start + 256, len(pos_lat))
        dlat = lat[:, None] - pos_lat[None, start:end]
        dlon = (lon[:, None] - pos_lon[None, start:end]) * cos_lat
        tolerant |= np.any((dlat * dlat + dlon * dlon) <= radius_sq, axis=1)
    return tolerant.astype(int)


def binary_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    unique_y = np.unique(y)
    if unique_y.size >= 2:
        ap = finite_or_nan(average_precision_score(y, p, sample_weight=w))
        roc_auc = finite_or_nan(roc_auc_score(y, p, sample_weight=w))
    else:
        ap = math.nan
        roc_auc = math.nan
    return ap, roc_auc, weighted_brier(y, p, w)


def period_metrics_for_model(
    frame: pd.DataFrame,
    *,
    prediction_path: Path,
    regions: Iterable[Region],
    window_days: int,
    require_full_periods: bool,
    spatial_tolerance_degrees: float = 0.0,
) -> pd.DataFrame:
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1; received {window_days}.")

    model_name = first_value(frame, "model_name", prediction_stem(prediction_path))
    model_type = first_value(frame, "model_type", model_name)
    source_probability_col = first_value(frame, "source_probability_col", PROB_COL)
    source_target_col = first_value(frame, "source_target_col", TARGET_COL)
    min_date = frame[DATE_COL].min()
    max_date = frame[DATE_COL].max()
    rows: list[dict[str, object]] = []

    for region in regions:
        region_mask = region.mask(frame)
        region_frame = frame.loc[region_mask]
        if region_frame.empty:
            continue

        first_start = pd.Timestamp(region_frame[DATE_COL].min()).normalize()
        last_start = pd.Timestamp(region_frame[DATE_COL].max()).normalize()
        if require_full_periods:
            last_start = min(last_start, pd.Timestamp(max_date).normalize() - pd.Timedelta(days=window_days - 1))
        if last_start < first_start:
            continue

        for period_start in pd.date_range(first_start, last_start, freq="D"):
            period_end = period_start + pd.Timedelta(days=window_days - 1)
            if require_full_periods and (period_start < min_date or period_end > max_date):
                continue
            period = region_frame[(region_frame[DATE_COL] >= period_start) & (region_frame[DATE_COL] <= period_end)]
            if period.empty:
                continue

            y = period[TARGET_COL].to_numpy(dtype=int)
            p = period[PROB_COL].to_numpy(dtype=float)
            w = period[WEIGHT_COL].to_numpy(dtype=float)
            y_spatial = spatial_tolerant_binary_labels(period, float(spatial_tolerance_degrees))
            expected = float(np.sum(p * w))
            observed = float(np.sum(y * w))
            observed_locations = int(period.loc[period[TARGET_COL] > 0, [LAT_COL, LON_COL]].drop_duplicates().shape[0])
            spatial_tolerant_locations = int(
                period.loc[y_spatial > 0, [LAT_COL, LON_COL]].drop_duplicates().shape[0]
            )
            count_abs_error = abs(expected - observed)
            if expected > 0 and observed > 0:
                count_ratio_abs_log_error = abs(math.log(expected / observed))
                expected_observed_count_ratio = expected / observed
            else:
                count_ratio_abs_log_error = math.nan
                expected_observed_count_ratio = math.nan

            ap, roc_auc, brier = binary_metrics(y, p, w)
            spatial_ap, spatial_roc_auc, spatial_brier = binary_metrics(y_spatial, p, w)

            rows.append(
                {
                    "model_name": model_name,
                    "model_type": model_type,
                    "source_probability_col": source_probability_col,
                    "source_target_col": source_target_col,
                    "prediction_path": str(prediction_path),
                    "region": region.name,
                    "region_display": region.display_name,
                    "period_start": period_start.date().isoformat(),
                    "period_end": period_end.date().isoformat(),
                    "window_days": int(window_days),
                    "support": int(len(period)),
                    "weighted_support": float(np.sum(w)),
                    "positive_rows": int(np.sum(y)),
                    "observed_positive_locations": observed_locations,
                    "spatial_tolerance_degrees": float(spatial_tolerance_degrees),
                    "spatial_tolerant_positive_rows": int(np.sum(y_spatial)),
                    "spatial_tolerant_positive_locations": spatial_tolerant_locations,
                    "observed_fire_positive_grid_cells": observed,
                    "expected_fire_positive_grid_cells": expected,
                    "expected_observed_count_ratio": expected_observed_count_ratio,
                    "count_abs_error": count_abs_error,
                    "count_ratio_abs_log_error": count_ratio_abs_log_error,
                    "mean_calibrated_predicted_probability": float(np.average(p, weights=w)),
                    "max_calibrated_predicted_probability": float(np.max(p)) if len(p) else math.nan,
                    "average_precision": ap,
                    "roc_auc": roc_auc,
                    "weighted_brier_score": brier,
                    "spatial_tolerant_average_precision": spatial_ap,
                    "spatial_tolerant_roc_auc": spatial_roc_auc,
                    "spatial_tolerant_weighted_brier_score": spatial_brier,
                }
            )

    return pd.DataFrame(rows)


def select_best_periods(
    metrics: pd.DataFrame,
    *,
    metric: str,
    min_wildfires: int,
    top_periods: int,
    allow_overlapping_periods: bool,
) -> pd.DataFrame:
    if metric not in METRIC_DIRECTIONS:
        raise ValueError(f"Unsupported selection metric: {metric}")
    if metrics.empty:
        raise ValueError("No period metrics were computed.")
    if top_periods < 1:
        raise ValueError(f"top_periods must be >= 1; received {top_periods}.")

    selections: list[pd.Series] = []
    for region, group in metrics.groupby("region", observed=True, sort=False):
        metric_col = metric
        direction = METRIC_DIRECTIONS[metric_col]
        candidates = group[group["observed_positive_locations"] >= int(min_wildfires)].copy()

        if candidates.empty:
            raise ValueError(
                f"No probability-overlay period for region {region!r} has at least "
                f"{int(min_wildfires)} observed positive locations."
            )
        candidates = candidates[np.isfinite(pd.to_numeric(candidates[metric_col], errors="coerce"))]

        if candidates.empty:
            raise ValueError(
                f"No finite {metric} probability-overlay period for region {region!r} "
                f"after applying min_wildfires={int(min_wildfires)}."
            )

        ascending = direction == "min"
        sort_cols = [metric_col, "count_abs_error", "weighted_brier_score", "period_start"]
        sort_ascending = [ascending, True, True, True]
        ranked = candidates.sort_values(sort_cols, ascending=sort_ascending, na_position="last").copy()

        picked: list[pd.Series] = []
        picked_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for _, candidate in ranked.iterrows():
            period_start = pd.Timestamp(candidate["period_start"])
            period_end = pd.Timestamp(candidate["period_end"])
            overlaps = any(period_start <= end and period_end >= start for start, end in picked_intervals)
            if allow_overlapping_periods or not overlaps:
                picked.append(candidate.copy())
                picked_intervals.append((period_start, period_end))
            if len(picked) >= top_periods:
                break

        if len(picked) < top_periods:
            raise ValueError(
                f"Only {len(picked)} non-overlapping probability-overlay period(s) were available for "
                f"region {region!r}; requested {top_periods}."
            )

        for rank, selected in enumerate(picked, start=1):
            selected["selection_metric"] = metric_col
            selected["selection_direction"] = direction
            selected["selection_fallback_reason"] = ""
            selected["period_rank"] = rank
            selections.append(selected)

    return pd.DataFrame(selections).reset_index(drop=True)


def load_world_boundaries(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        import geopandas as gpd

        return gpd.read_file(path)
    except Exception as exc:  # pragma: no cover - optional visual enhancement
        print(f"Warning: could not load country boundaries from {path}: {exc}")
        return None


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}.")
    return payload


def prediction_countries(feature_config: dict) -> list[str]:
    raw = feature_config.get("prediction_countries") or feature_config.get("modis_countries") or []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item) for item in raw]


def plot_boundaries(ax, world, extent: tuple[float, float, float, float]) -> None:
    if world is None:
        return
    lon_min, lon_max, lat_min, lat_max = extent
    try:
        subset = world.cx[lon_min:lon_max, lat_min:lat_max]
        if subset.empty:
            subset = world
        subset.boundary.plot(ax=ax, color="#4a4a4a", linewidth=0.45, alpha=0.75, zorder=5)
    except Exception:
        return


def probability_formatter(value: float, _pos: int) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.1e}"
    if abs(value) < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def metric_formatter(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    if abs(numeric) == 0:
        return "0"
    if abs(numeric) < 0.001:
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def region_grid_frame(
    region: Region,
    *,
    resolution: float,
    date: pd.Timestamp,
    world,
    countries: list[str],
    coordinate_bounds: list[float] | None = None,
) -> pd.DataFrame:
    country_shapes = country_shapes_for_grid(world, countries)
    country_label_col = "__prediction_country"
    if country_shapes is not None and not country_shapes.empty:
        label_lookup = {str(country): str(country) for country in countries}
        label_lookup.update({str(country_mapping.get(country, country)): str(country) for country in countries})
        name_cols = [col for col in ["SOVEREIGNT", "ADMIN", "NAME", "NAME_EN"] if col in country_shapes.columns]

        def prediction_country(row: pd.Series) -> str:
            for col in name_cols:
                value = str(row.get(col))
                if value in label_lookup:
                    return label_lookup[value]
            return str(row.get(name_cols[0], "Unknown")) if name_cols else "Unknown"

        country_shapes = country_shapes.copy()
        country_shapes[country_label_col] = country_shapes.apply(prediction_country, axis=1)

    if region.name == "global" and country_shapes is not None and not country_shapes.empty:
        lon_min, lat_min, lon_max, lat_max = (float(value) for value in country_shapes.total_bounds)
    else:
        lon_min, lon_max, lat_min, lat_max = region.extent(pd.DataFrame({LAT_COL: [], LON_COL: []}))

    if coordinate_bounds:
        if len(coordinate_bounds) != 4:
            raise ValueError(f"coordinate_bounds must have four values [lat_min, lon_min, lat_max, lon_max]: {coordinate_bounds}")
        cfg_lat_min, cfg_lon_min, cfg_lat_max, cfg_lon_max = (float(value) for value in coordinate_bounds)
        lat_min = max(float(lat_min), cfg_lat_min)
        lat_max = min(float(lat_max), cfg_lat_max)
        lon_min = max(float(lon_min), cfg_lon_min)
        lon_max = min(float(lon_max), cfg_lon_max)

    bounds = np.asarray([lon_min, lon_max, lat_min, lat_max], dtype=float)
    if not np.isfinite(bounds).all() or lon_min > lon_max or lat_min > lat_max:
        raise ValueError(
            f"Cannot build dense grid for region {region.name!r}; invalid bounds "
            f"lon={lon_min}..{lon_max}, lat={lat_min}..{lat_max}."
        )
    precision = precision_for_resolution(resolution)
    lat_coords = np.round(np.arange(lat_min, lat_max + resolution / 2.0, resolution), precision)
    lon_coords = np.round(np.arange(lon_min, lon_max + resolution / 2.0, resolution), precision)
    lat_mesh, lon_mesh = np.meshgrid(lat_coords, lon_coords, indexing="ij")
    grid = pd.DataFrame(
        {
            LAT_COL: lat_mesh.ravel().astype("float32"),
            LON_COL: lon_mesh.ravel().astype("float32"),
        }
    )

    if country_shapes is not None and not country_shapes.empty:
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            points = [Point(lon, lat) for lon, lat in zip(grid[LON_COL], grid[LAT_COL])]
            points_gdf = gpd.GeoDataFrame(grid, geometry=points, crs=country_shapes.crs)
            joined = gpd.sjoin(
                points_gdf,
                country_shapes[[country_label_col, "geometry"]],
                how="inner",
                predicate="within",
            )
            grid = (
                joined[[LAT_COL, LON_COL, country_label_col]]
                .rename(columns={country_label_col: "country"})
                .drop_duplicates([LAT_COL, LON_COL])
                .reset_index(drop=True)
            )
        except Exception as exc:
            print(f"Warning: country grid filtering failed for {region.display_name}: {exc}")

    grid[DATE_COL] = pd.Timestamp(date).normalize()
    grid["acq_date"] = grid[DATE_COL]
    grid["month"] = grid[DATE_COL].dt.month.astype("int16")
    grid["day"] = grid[DATE_COL].dt.day.astype("int16")
    grid["year"] = grid[DATE_COL].dt.year.astype("int16")
    grid["count"] = 0
    return grid


def filter_grid_to_climate_coverage(grid: pd.DataFrame, feature_config: dict[str, object]) -> pd.DataFrame:
    if grid.empty:
        return grid
    climate_params = feature_config.get("climate_data_params", {})
    if not isinstance(climate_params, dict):
        return grid
    climate_data_dir = climate_params.get("climate_data_dir")
    variables = [str(variable) for variable in climate_params.get("climate_variables", [])]
    n_days = int(climate_params.get("n_days", 0) or 0)
    if not climate_data_dir or not variables or n_days <= 0:
        return grid

    from src.feature_generation.prepare_climate_data import (
        check_fragmented_dataset_bounds,
        discover_climate_fragments,
    )

    target = grid[["acq_date", LAT_COL, LON_COL]].copy()
    keep = np.ones(len(target), dtype=bool)
    dropped_by_variable: dict[str, int] = {}
    for variable in variables:
        fragments = discover_climate_fragments(str(climate_data_dir), variable)
        coverage = check_fragmented_dataset_bounds(fragments, target, n_days=n_days)
        covered = np.asarray(coverage["assignments"]) >= 0
        dropped_by_variable[variable] = int((~covered).sum())
        keep &= covered
    dropped = int((~keep).sum())
    if dropped:
        print(
            "  dropped "
            f"{dropped:,}/{len(grid):,} dense grid points outside full {n_days}-day climate coverage "
            f"for {climate_data_dir}; per-variable misses={dropped_by_variable}"
        )
    return grid.loc[keep].reset_index(drop=True)


def country_shapes_for_grid(world, countries: list[str]):
    if world is None or not countries:
        return None
    name_cols = [col for col in ["SOVEREIGNT", "ADMIN", "NAME", "NAME_EN"] if col in world.columns]
    if not name_cols:
        return None
    mapped = set(countries)
    mapped.update(country_mapping.get(country, country) for country in countries)
    mask = np.zeros(len(world), dtype=bool)
    for col in name_cols:
        mask |= world[col].astype(str).isin(mapped).to_numpy()
    return world.loc[mask]


def fill_prediction_features(model: CatBoostClassifier, features: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(
        {
            col: features[col] if col in features.columns else np.nan
            for col in model.feature_names_
        },
        index=features.index,
    )

    cat_feature_names = {model.feature_names_[idx] for idx in model.get_cat_feature_indices()}
    for col in X.columns:
        if col in cat_feature_names:
            if X[col].dtype == "object" or pd.api.types.is_string_dtype(X[col]):
                X[col] = X[col].fillna("Unknown").astype(str)
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(99).astype(int)
        else:
            values = pd.to_numeric(X[col], errors="coerce")
            fill_value = values.mean()
            if pd.isna(fill_value):
                fill_value = 0.0
            X[col] = values.fillna(fill_value)
    return X


def dense_prediction_cache_path(
    output_dir: Path,
    *,
    model_name: str,
    region_name: str,
    period_start: pd.Timestamp,
    window_days: int,
    map_summary: str,
    resolution: float,
    probability_mode: str,
) -> Path:
    return (
        output_dir
        / "artifacts"
        / "dense_predictions"
        / (
            f"dense_{safe_slug(model_name)}_{safe_slug(region_name)}_{period_start:%Y%m%d}_"
            f"{int(window_days)}d_"
            f"{safe_slug(map_summary)}_{safe_slug(f'{resolution:g}deg')}_{safe_slug(probability_mode)}.parquet"
        )
    )


def dense_neural_prediction_cache_path(
    output_dir: Path,
    *,
    model_name: str,
    region_name: str,
    period_start: pd.Timestamp,
    window_days: int,
    map_summary: str,
    resolution: float,
    probability_mode: str,
) -> Path:
    return (
        output_dir
        / "artifacts"
        / "dense_neural_predictions"
        / (
            f"dense_neural_{safe_slug(model_name)}_{safe_slug(region_name)}_{period_start:%Y%m%d}_"
            f"{int(window_days)}d_{safe_slug(map_summary)}_{safe_slug(f'{resolution:g}deg')}_"
            f"{safe_slug(probability_mode)}.parquet"
        )
    )


def dense_period_predictions(
    *,
    output_dir: Path,
    selection: pd.Series,
    region: Region,
    model_path: Path,
    feature_config_path: Path,
    target_resolution: float,
    map_summary: str,
    world,
    overwrite: bool,
    prior_correction: bool,
    train_prior: float,
    deploy_prior: float,
    verbose_feature_generation: bool,
) -> pd.DataFrame:
    period_start = pd.Timestamp(selection["period_start"])
    window_days = int(selection.get("window_days", 3))
    probability_mode = "prior_corrected" if prior_correction else "raw"
    cache_path = dense_prediction_cache_path(
        output_dir,
        model_name=str(selection["model_name"]),
        region_name=region.name,
        period_start=period_start,
        window_days=window_days,
        map_summary=map_summary,
        resolution=target_resolution,
        probability_mode=probability_mode,
    )
    if cache_path.exists() and not overwrite:
        return pd.read_parquet(cache_path)

    feature_config = load_yaml(feature_config_path)
    countries = prediction_countries(feature_config)
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    daily_frames: list[pd.DataFrame] = []
    for date in pd.date_range(period_start, period_start + pd.Timedelta(days=window_days - 1), freq="D"):
        start = time.perf_counter()
        target_grid = region_grid_frame(
            region,
            resolution=target_resolution,
            date=date,
            world=world,
            countries=countries,
            coordinate_bounds=feature_config.get("coordinate_bounds"),
        )
        print(
            f"Dense prediction {selection['model_name']} | {region.display_name} | "
            f"{date:%Y-%m-%d}: {len(target_grid)} grid points"
        )
        if target_grid.empty:
            continue
        log_path = (
            output_dir
            / "artifacts"
            / "dense_feature_logs"
            / f"{safe_slug(region.name)}_{date:%Y%m%d}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose_feature_generation:
            features = make_features_from_target_df(
                feature_config,
                target_grid,
                test_mode=True,
                use_cached_files=True,
                cache_dir=str(output_dir / "artifacts" / "dense_feature_cache"),
            )
        else:
            with log_path.open("w", encoding="utf-8") as log_handle:
                with contextlib.redirect_stdout(log_handle):
                    features = make_features_from_target_df(
                        feature_config,
                        target_grid,
                        test_mode=True,
                        use_cached_files=True,
                        cache_dir=str(output_dir / "artifacts" / "dense_feature_cache"),
                    )
            print(f"  feature-generation log: {log_path}")
        if features.empty:
            continue
        X = fill_prediction_features(model, features)
        pred = model.predict_proba(X)[:, 1]
        if prior_correction:
            pred = adjust_probabilities_for_prior(
                pred,
                train_prior=train_prior,
                deploy_prior=deploy_prior,
                assume_logits=False,
            )
        daily_frames.append(
            pd.DataFrame(
                {
                    LAT_COL: features[LAT_COL].to_numpy(dtype="float32"),
                    LON_COL: features[LON_COL].to_numpy(dtype="float32"),
                    "prediction": pred.astype("float32"),
                }
            )
        )
        print(f"  finished in {time.perf_counter() - start:.1f}s")

    if not daily_frames:
        raise RuntimeError(f"No dense predictions were generated for {region.display_name} {period_start:%Y-%m-%d}.")

    daily = pd.concat(daily_frames, ignore_index=True)
    spatial = (
        daily.groupby([LAT_COL, LON_COL], observed=True)["prediction"]
        .agg(prob_period_sum="sum", prob_period_mean="mean", prob_period_max="max")
        .reset_index()
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    spatial.to_parquet(cache_path, index=False)
    return spatial


def dense_period_neural_predictions(
    *,
    output_dir: Path,
    results_dir: Path,
    selection: pd.Series,
    region: Region,
    feature_config_path: Path,
    training_features_path: Path,
    neural_model_path: Path | None,
    target_resolution: float,
    map_summary: str,
    world,
    overwrite: bool,
    batch_size: int,
    device: str,
    prior_correction: bool,
    train_prior: float,
    deploy_prior: float,
    verbose_feature_generation: bool,
) -> pd.DataFrame:
    period_start = pd.Timestamp(selection["period_start"])
    window_days = int(selection.get("window_days", 3))
    probability_mode = "prior_corrected" if prior_correction else "raw"
    cache_path = dense_neural_prediction_cache_path(
        output_dir,
        model_name=str(selection["model_name"]),
        region_name=region.name,
        period_start=period_start,
        window_days=window_days,
        map_summary=map_summary,
        resolution=target_resolution,
        probability_mode=probability_mode,
    )
    if cache_path.exists() and not overwrite:
        return pd.read_parquet(cache_path)

    prediction_path = Path(str(selection["prediction_path"]))
    feature_config = load_yaml(feature_config_path)
    predictor = load_dense_neural_predictor(
        results_dir=results_dir,
        prediction_path=prediction_path,
        training_features_path=training_features_path,
        model_path=neural_model_path,
        batch_size=batch_size,
        device=device,
        feature_config_path=feature_config_path,
        feature_config=feature_config,
    )
    countries = prediction_countries(feature_config)

    daily_frames: list[pd.DataFrame] = []
    for date in pd.date_range(period_start, period_start + pd.Timedelta(days=window_days - 1), freq="D"):
        start = time.perf_counter()
        target_grid = region_grid_frame(
            region,
            resolution=target_resolution,
            date=date,
            world=world,
            countries=countries,
            coordinate_bounds=feature_config.get("coordinate_bounds"),
        )
        target_grid = filter_grid_to_climate_coverage(target_grid, feature_config)
        print(
            f"Dense neural prediction {selection['model_name']} | {region.display_name} | "
            f"{date:%Y-%m-%d}: {len(target_grid)} grid points"
        )
        if target_grid.empty:
            continue
        log_path = (
            output_dir
            / "artifacts"
            / "dense_feature_logs"
            / f"neural_{safe_slug(region.name)}_{date:%Y%m%d}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose_feature_generation:
            features = make_features_from_target_df(
                feature_config,
                target_grid,
                test_mode=True,
                use_cached_files=True,
                cache_dir=str(output_dir / "artifacts" / "dense_feature_cache"),
            )
        else:
            with log_path.open("w", encoding="utf-8") as log_handle:
                with contextlib.redirect_stdout(log_handle):
                    features = make_features_from_target_df(
                        feature_config,
                        target_grid,
                        test_mode=True,
                        use_cached_files=True,
                        cache_dir=str(output_dir / "artifacts" / "dense_feature_cache"),
                    )
            print(f"  feature-generation log: {log_path}")
        if features.empty:
            continue
        pred = predictor.predict(features)
        if prior_correction:
            pred = adjust_probabilities_for_prior(
                pred,
                train_prior=train_prior,
                deploy_prior=deploy_prior,
                assume_logits=False,
            )
        daily_frames.append(
            pd.DataFrame(
                {
                    LAT_COL: features[LAT_COL].to_numpy(dtype="float32"),
                    LON_COL: features[LON_COL].to_numpy(dtype="float32"),
                    "prediction": pred.astype("float32"),
                }
            )
        )
        print(f"  finished in {time.perf_counter() - start:.1f}s")

    if not daily_frames:
        raise RuntimeError(f"No dense neural predictions were generated for {region.display_name} {period_start:%Y-%m-%d}.")

    daily = pd.concat(daily_frames, ignore_index=True)
    spatial = (
        daily.groupby([LAT_COL, LON_COL], observed=True)["prediction"]
        .agg(prob_period_sum="sum", prob_period_mean="mean", prob_period_max="max")
        .reset_index()
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    spatial.to_parquet(cache_path, index=False)
    return spatial


def precision_for_resolution(resolution: float) -> int:
    if resolution <= 0:
        return 6
    return max(0, int(math.ceil(-math.log10(resolution))) + 2)


def spatial_grid(
    spatial: pd.DataFrame,
    *,
    value_col: str,
    extent: tuple[float, float, float, float],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = extent
    precision = precision_for_resolution(resolution)
    lat_start = math.floor(lat_min / resolution) * resolution
    lat_stop = math.ceil(lat_max / resolution) * resolution
    lon_start = math.floor(lon_min / resolution) * resolution
    lon_stop = math.ceil(lon_max / resolution) * resolution
    lat_centers = np.round(np.arange(lat_start, lat_stop + resolution / 2.0, resolution), precision)
    lon_centers = np.round(np.arange(lon_start, lon_stop + resolution / 2.0, resolution), precision)
    grid = np.full((len(lat_centers), len(lon_centers)), np.nan, dtype=float)

    lat_idx = np.rint((spatial[LAT_COL].to_numpy(dtype=float) - lat_start) / resolution).astype(int)
    lon_idx = np.rint((spatial[LON_COL].to_numpy(dtype=float) - lon_start) / resolution).astype(int)
    valid = (
        (lat_idx >= 0)
        & (lat_idx < len(lat_centers))
        & (lon_idx >= 0)
        & (lon_idx < len(lon_centers))
    )
    grid[lat_idx[valid], lon_idx[valid]] = spatial[value_col].to_numpy(dtype=float)[valid]

    return lon_centers, lat_centers, grid


def interpolated_prediction_surface(
    lon_coords: np.ndarray,
    lat_coords: np.ndarray,
    prediction_grid: np.ndarray,
    *,
    interpolation_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match prediction_pipeline_boosting's no-gap interpolation strategy."""

    valid_mask = np.isfinite(prediction_grid)
    if not np.any(valid_mask):
        return lon_coords, lat_coords, prediction_grid

    lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)
    points = np.column_stack((lon_mesh[valid_mask], lat_mesh[valid_mask]))
    values = prediction_grid[valid_mask]

    factor = max(1, int(interpolation_factor))
    interp_lon = np.linspace(float(lon_coords.min()), float(lon_coords.max()), len(lon_coords) * factor)
    interp_lat = np.linspace(float(lat_coords.min()), float(lat_coords.max()), len(lat_coords) * factor)
    interp_lon_mesh, interp_lat_mesh = np.meshgrid(interp_lon, interp_lat)

    if len(values) >= 3:
        interp_linear = griddata(points, values, (interp_lon_mesh, interp_lat_mesh), method="linear", fill_value=np.nan)
    else:
        interp_linear = np.full_like(interp_lon_mesh, np.nan, dtype=float)
    interp_nearest = griddata(points, values, (interp_lon_mesh, interp_lat_mesh), method="nearest")
    interp_values = np.where(np.isnan(interp_linear), interp_nearest, interp_linear)
    return interp_lon, interp_lat, interp_values


def probability_colormap(name: str):
    cmap = plt.get_cmap(name).copy()
    transparent = (1.0, 1.0, 1.0, 0.0)
    prediction_base = cmap(0.0)
    cmap.set_bad(transparent)
    cmap.set_under(prediction_base)
    return cmap


def color_scale(values: np.ndarray, *, color_floor: float | None, color_vmax: float | None) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if color_vmax is None:
        vmax = float(np.nanpercentile(positive, 99.5)) if len(positive) else 1.0
    else:
        vmax = float(color_vmax)
    if not math.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    floor = 0.0 if color_floor is None else float(color_floor)
    if not math.isfinite(floor) or floor < 0:
        floor = 0.0
    if floor >= vmax:
        floor = max(0.0, vmax * 0.25)
    return floor, vmax


def plot_selected_period(
    frame: pd.DataFrame,
    *,
    selection: pd.Series,
    region: Region,
    output_dir: Path,
    results_dir: Path,
    formats: Iterable[str],
    dpi: int,
    map_summary: str,
    grid_resolution: float,
    interpolation_factor: int,
    surface_source: str,
    dense_model_path: Path,
    dense_neural_model_path: Path | None,
    dense_neural_training_features: Path,
    dense_neural_batch_size: int,
    dense_neural_device: str,
    feature_config_path: Path,
    overwrite_dense: bool,
    prior_correction: bool,
    train_prior: float,
    deploy_prior: float,
    colormap: str,
    color_floor: float | None,
    color_vmax: float | None,
    source_label: str | None,
    verbose_feature_generation: bool,
    world,
) -> list[Path]:
    period_start = pd.Timestamp(selection["period_start"])
    period_end = pd.Timestamp(selection["period_end"])
    window_days = int(selection.get("window_days", (period_end - period_start).days + 1))
    period_mask = (frame[DATE_COL] >= period_start) & (frame[DATE_COL] <= period_end)
    plot_frame = frame.loc[period_mask & region.mask(frame)].copy()
    if plot_frame.empty:
        return []

    plot_frame["expected_weighted"] = plot_frame[PROB_COL] * plot_frame[WEIGHT_COL]
    plot_frame["observed_weighted"] = plot_frame[TARGET_COL] * plot_frame[WEIGHT_COL]

    observed_spatial = (
        plot_frame.groupby([LAT_COL, LON_COL], observed=True)[TARGET_COL]
        .sum()
        .rename("observed_days")
        .reset_index()
    )
    if surface_source == "dense":
        spatial = dense_period_predictions(
            output_dir=output_dir,
            selection=selection,
            region=region,
            model_path=dense_model_path,
            feature_config_path=feature_config_path,
            target_resolution=grid_resolution,
            map_summary=map_summary,
            world=world,
            overwrite=overwrite_dense,
            prior_correction=prior_correction,
            train_prior=train_prior,
            deploy_prior=deploy_prior,
            verbose_feature_generation=verbose_feature_generation,
        )
        spatial = spatial.merge(observed_spatial, on=[LAT_COL, LON_COL], how="left")
        spatial["observed_days"] = spatial["observed_days"].fillna(0)
    elif surface_source == "dense-neural":
        spatial = dense_period_neural_predictions(
            output_dir=output_dir,
            results_dir=results_dir,
            selection=selection,
            region=region,
            feature_config_path=feature_config_path,
            training_features_path=dense_neural_training_features,
            neural_model_path=dense_neural_model_path,
            target_resolution=grid_resolution,
            map_summary=map_summary,
            world=world,
            overwrite=overwrite_dense,
            batch_size=dense_neural_batch_size,
            device=dense_neural_device,
            prior_correction=prior_correction,
            train_prior=train_prior,
            deploy_prior=deploy_prior,
            verbose_feature_generation=verbose_feature_generation,
        )
        spatial = spatial.merge(observed_spatial, on=[LAT_COL, LON_COL], how="left")
        spatial["observed_days"] = spatial["observed_days"].fillna(0)
    else:
        spatial = (
            plot_frame.groupby([LAT_COL, LON_COL], observed=True)
            .agg(
                prob_period_sum=(PROB_COL, "sum"),
                prob_period_mean=(PROB_COL, "mean"),
                prob_period_max=(PROB_COL, "max"),
                observed_days=(TARGET_COL, "sum"),
            )
            .reset_index()
        )
    if map_summary == "sum":
        color_col = "prob_period_sum"
        color_label = f"{window_days}-day\nprob. sum"
    elif map_summary == "mean":
        color_col = "prob_period_mean"
        color_label = "Mean daily\nprobability"
    else:
        color_col = "prob_period_max"
        color_label = "Max daily\nprobability"

    extent = region.extent(plot_frame)
    lon_min, lon_max, lat_min, lat_max = extent
    lon_span = max(lon_max - lon_min, 1e-6)
    lat_span = max(lat_max - lat_min, 1e-6)
    mid_lat = (lat_min + lat_max) / 2.0
    display_aspect = (lat_span / lon_span) / max(math.cos(math.radians(mid_lat)), 0.2)
    figure_width = 6.7
    figure_height = min(6.2, max(3.6, 0.85 + 5.4 * display_aspect))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax_map = plt.subplots(figsize=(figure_width, figure_height))
    values = spatial[color_col].to_numpy(dtype=float)
    display_floor, vmax = color_scale(values, color_floor=color_floor, color_vmax=color_vmax)
    lon_coords, lat_coords, probability_grid = spatial_grid(
        spatial,
        value_col=color_col,
        extent=extent,
        resolution=float(grid_resolution),
    )
    if surface_source in {"dense", "dense-neural"}:
        mesh_lon, mesh_lat = lon_coords, lat_coords
        mesh_values = probability_grid
    else:
        mesh_lon, mesh_lat, mesh_values = interpolated_prediction_surface(
            lon_coords,
            lat_coords,
            probability_grid,
            interpolation_factor=interpolation_factor,
        )
    mesh_values = np.asarray(mesh_values, dtype=float)
    finite_mesh_values = mesh_values[np.isfinite(mesh_values)]
    has_under_color = bool(display_floor > 0 and len(finite_mesh_values) and np.nanmin(finite_mesh_values) < display_floor)
    has_over_color = bool(len(finite_mesh_values) and np.nanmax(finite_mesh_values) > vmax)
    mesh_values = np.ma.masked_invalid(mesh_values)
    norm = mcolors.Normalize(vmin=display_floor, vmax=vmax, clip=False)
    mesh = ax_map.pcolormesh(
        mesh_lon,
        mesh_lat,
        mesh_values,
        cmap=probability_colormap(colormap),
        norm=norm,
        rasterized=True,
        shading="auto",
        zorder=2,
    )
    plot_boundaries(ax_map, world, extent)

    observed = observed_spatial[observed_spatial["observed_days"] > 0]
    if not observed.empty:
        n_observed = len(observed)
        if n_observed > 500:
            base_size, size_step, size_cap = 4.0, 0.8, 7.0
            line_width, marker_alpha = 0.0, 0.74
        elif n_observed > 100:
            base_size, size_step, size_cap = 9.0, 1.5, 16.0
            line_width, marker_alpha = 0.25, 0.82
        else:
            base_size, size_step, size_cap = 16.0, 2.5, 28.0
            line_width, marker_alpha = 0.35, 0.92
        marker_size = np.clip(
            base_size + observed["observed_days"].to_numpy(dtype=float) * size_step,
            base_size,
            size_cap,
        )
        ax_map.scatter(
            observed[LON_COL],
            observed[LAT_COL],
            s=marker_size,
            marker="o",
            facecolors="#00c8ff",
            edgecolors="#031a3a" if line_width > 0 else "none",
            linewidths=line_width,
            alpha=marker_alpha,
            label="Observed fire",
            zorder=6,
            rasterized=True,
        )
        ax_map.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            borderpad=0.25,
            handlelength=1.1,
            markerscale=0.9,
        )

    lon_pad = max((lon_max - lon_min) * 0.025, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.025, 0.1)
    ax_map.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax_map.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
    ax_map.set_aspect(1.0 / max(math.cos(math.radians(mid_lat)), 0.2), adjustable="box")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(color="#d0d0d0", linewidth=0.35, alpha=0.5)
    for spine in ax_map.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.6)

    if has_under_color and has_over_color:
        colorbar_extend = "both"
    elif has_under_color:
        colorbar_extend = "min"
    elif has_over_color:
        colorbar_extend = "max"
    else:
        colorbar_extend = "neither"
    colorbar = fig.colorbar(mesh, ax=ax_map, fraction=0.045, pad=0.025, extend=colorbar_extend)
    label = color_label.replace("\n", " ")
    if display_floor > 0:
        label = f"{label} (yellow < {probability_formatter(display_floor, None)})"
    colorbar.set_label(label)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(probability_formatter))

    fig.subplots_adjust(left=0.10, right=0.92, top=0.97, bottom=0.10)

    png_dir = output_dir / "plots" / "png"
    pdf_dir = output_dir / "plots" / "pdf"
    for directory in [png_dir, pdf_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    source_prefix = f"{safe_slug(source_label)}_" if source_label else ""
    stem = (
        f"{source_prefix}probability_overlay_{safe_slug(selection['model_name'])}_"
        f"{safe_slug(selection['region'])}_rank{int(selection.get('period_rank', 1)):02d}_"
        f"{period_start:%Y%m%d}_{window_days}d"
    )
    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt == "png":
            out = png_dir / f"{stem}.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        elif fmt == "pdf":
            out = pdf_dir / f"{stem}.pdf"
            fig.savefig(out, bbox_inches="tight")
        else:
            out = output_dir / f"{stem}.{fmt}"
            fig.savefig(out, dpi=dpi if fmt in {"jpg", "jpeg", "tif", "tiff"} else None, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def write_markdown_summary(
    output_dir: Path,
    selected: pd.DataFrame,
    *,
    source: str,
    source_label: str | None,
    surface_source: str,
    grid_resolution: float,
    prior_correction: bool,
    train_prior: float,
    deploy_prior: float,
) -> None:
    lines = [
        "# Probability Period Overlays",
        "",
        "Selected top test-set periods for each region from revision evaluation predictions.",
        f"",
        f"Prediction source: `{source}`.",
        "",
        f"Feature source: `{source_label or 'default'}`.",
        "",
        f"Map surface: `{surface_source}`.",
        "",
        f"Dense grid resolution: `{grid_resolution:g}` degrees.",
        "",
        f"Prior correction: `{'enabled' if prior_correction else 'disabled'}`"
        f" (train prior `{train_prior:g}`, deployment prior `{deploy_prior:g}`).",
        "",
        "| Region | Rank | Model | Period | Days | Selection | Fire locations | AP | Brier | Probability column |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for _, row in selected.iterrows():
        period = f"{row['period_start']} to {row['period_end']}"
        selection = str(row["selection_metric"])
        if row.get("selection_fallback_reason"):
            selection += " (fallback)"
        lines.append(
            "| {region} | {rank} | {model} | {period} | {days} | {selection} | {locations} | {ap} | {brier} | {prob_col} |".format(
                region=row["region_display"],
                rank=int(row.get("period_rank", 1)),
                model=row["model_name"],
                period=period,
                days=int(row.get("window_days", 0)),
                selection=selection,
                locations=int(row.get("observed_positive_locations", 0)),
                ap=metric_formatter(row.get("average_precision")),
                brier=metric_formatter(row.get("weighted_brier_score")),
                prob_col=row.get("source_probability_col", "n/a"),
            )
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_old_plot_files(output_dir: Path) -> None:
    for subdir in [output_dir / "plots" / "png", output_dir / "plots" / "pdf"]:
        if not subdir.exists():
            continue
        for path in subdir.glob("*probability_overlay_*"):
            if path.is_file():
                path.unlink()


def compact_overlay_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    return out


def write_wide_jsonl(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(path, orient="records", lines=True, compression="gzip")


def run_probability_overlays(config: ProbabilityOverlayConfig) -> dict[str, object]:
    output_dir = config.output_dir or default_output_dir(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    for stale_table in [
        "weekly_probability_metrics.csv",
        "selected_probability_weeks.csv",
        "period_probability_metrics.csv",
        "selected_probability_periods.csv",
    ]:
        stale_path = output_dir / "tables" / stale_table
        if stale_path.exists():
            stale_path.unlink()
    if config.grid_resolution is None:
        target_config = load_yaml(config.target_config)
        grid_resolution = float(target_config.get("spatial_coarseness", 0.1))
    else:
        grid_resolution = float(config.grid_resolution)
    window_days = int(config.window_days)
    if window_days < 1:
        raise ValueError("probability_overlay_window_days must be >= 1")
    top_periods = int(config.top_periods)
    if top_periods < 1:
        raise ValueError("probability_overlay_top_periods must be >= 1")

    only = set(config.regions) if config.regions else None
    regions = load_regions(config.regions_file, include_global=config.include_global, only=only)

    prediction_files = find_prediction_files_for_model(config.results_dir, config.model, config.source)

    print(f"Scoring {len(prediction_files)} test prediction file(s) across {len(regions)} region(s).")
    metric_frames: list[pd.DataFrame] = []
    for path in prediction_files:
        print(f"  - {path}")
        frame = read_prediction_columns(path, config.prob_col)
        metrics = period_metrics_for_model(
            frame,
            prediction_path=path,
            regions=regions,
            window_days=window_days,
            require_full_periods=not config.allow_partial_periods,
            spatial_tolerance_degrees=config.spatial_tolerance_degrees,
        )
        metric_frames.append(metrics)
        del frame

    period_metrics = pd.concat(metric_frames, ignore_index=True)
    if config.max_period_end:
        max_period_end = pd.to_datetime(config.max_period_end).normalize()
        period_end = pd.to_datetime(period_metrics["period_end"], errors="coerce")
        period_metrics = period_metrics.loc[period_end <= max_period_end].copy()
        if period_metrics.empty:
            raise ValueError(f"No probability overlay periods remain on or before {max_period_end:%Y-%m-%d}.")
    wide_dir = output_dir / "artifacts" / "wide_tables"
    write_wide_jsonl(wide_dir / "period_probability_metrics.jsonl.gz", period_metrics)
    period_metrics_path = output_dir / "tables" / "period_probability_metrics.csv"
    compact_overlay_table(
        period_metrics.sort_values("average_precision", ascending=False).head(200),
        [
            "region_display",
            "model_name",
            "period_start",
            "period_end",
            "observed_positive_locations",
            "average_precision",
            "spatial_tolerant_average_precision",
            "spatial_tolerant_weighted_brier_score",
        ],
    ).to_csv(period_metrics_path, index=False)

    selected = select_best_periods(
        period_metrics,
        metric=config.selection_metric,
        min_wildfires=config.min_wildfires,
        top_periods=top_periods,
        allow_overlapping_periods=config.allow_overlapping_periods,
    )
    write_wide_jsonl(wide_dir / "selected_probability_periods.jsonl.gz", selected)
    selected_path = output_dir / "tables" / "selected_probability_periods.csv"
    compact_overlay_table(
        selected,
        [
            "region_display",
            "model_name",
            "period_start",
            "period_end",
            "window_days",
            "average_precision",
            "spatial_tolerant_average_precision",
            "spatial_tolerant_weighted_brier_score",
        ],
    ).to_csv(selected_path, index=False)
    write_markdown_summary(
        output_dir,
        selected,
        source=config.source,
        source_label=config.source_label,
        surface_source=config.surface_source,
        grid_resolution=grid_resolution,
        prior_correction=config.prior_correction,
        train_prior=config.train_prior,
        deploy_prior=config.deploy_prior,
    )

    world = load_world_boundaries(config.country_shapes)
    region_by_name = {region.name: region for region in regions}
    formats = [fmt.strip() for fmt in config.formats if fmt.strip()]
    if not config.keep_existing_plots:
        cleanup_old_plot_files(output_dir)
    written: list[Path] = []
    for prediction_path, group in selected.groupby("prediction_path", sort=False):
        frame = read_prediction_columns(Path(prediction_path), config.prob_col)
        for _, selection in group.iterrows():
            region = region_by_name[str(selection["region"])]
            written.extend(
                plot_selected_period(
                    frame,
                    selection=selection,
                    region=region,
                    output_dir=output_dir,
                    results_dir=config.results_dir,
                    formats=formats,
                    dpi=config.dpi,
                    map_summary=config.map_summary,
                    grid_resolution=grid_resolution,
                    interpolation_factor=config.interpolation_factor,
                    surface_source=config.surface_source,
                    dense_model_path=config.dense_model_path
                    or (config.results_dir / "shared_artifacts" / "models" / "catboost_full.cbm"),
                    dense_neural_model_path=config.dense_neural_model_path,
                    dense_neural_training_features=config.dense_neural_training_features,
                    dense_neural_batch_size=config.dense_neural_batch_size,
                    dense_neural_device=config.dense_neural_device,
                    feature_config_path=config.feature_config,
                    overwrite_dense=config.overwrite_dense,
                    prior_correction=config.prior_correction,
                    train_prior=config.train_prior,
                    deploy_prior=config.deploy_prior,
                    colormap=config.colormap,
                    color_floor=config.color_floor,
                    color_vmax=config.color_vmax,
                    source_label=config.source_label,
                    verbose_feature_generation=config.verbose_feature_generation,
                    world=world,
                )
            )
        del frame

    print(f"Wrote period metrics: {period_metrics_path}")
    print(f"Wrote selected periods: {selected_path}")
    print(f"Wrote {len(written)} plot file(s) under {output_dir / 'plots'}")
    fallbacks = selected[selected["selection_fallback_reason"].astype(str).ne("")]
    if not fallbacks.empty:
        print("Selection fallbacks:")
        for _, row in fallbacks.iterrows():
            print(f"  - {row['region_display']}: {row['selection_fallback_reason']}")
    manifest = {
        "output_dir": str(output_dir),
        "period_metrics": str(period_metrics_path),
        "selected_periods": str(selected_path),
        "plots": [str(path) for path in written],
        "plot_count": len(written),
        "model": config.model,
        "source": config.source,
        "source_label": config.source_label,
        "surface_source": config.surface_source,
        "selection_metric": config.selection_metric,
        "spatial_tolerance_degrees": config.spatial_tolerance_degrees,
        "max_period_end": config.max_period_end,
        "colormap": config.colormap,
        "color_floor": config.color_floor,
        "color_vmax": config.color_vmax,
        "regions": [region.name for region in regions],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
