"""Prediction time-series and spatial error diagnostic plots.

This module reads saved test prediction parquet files, aggregates predicted
fire-positive grid-cell days against observed labels, and writes:

- time-series plots by region;
- spatial error maps averaged over test years;
- spatial error maps for each individual test year.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from .artifacts import ensure_dir, write_json
from .config import NN_LABELS
from .probability_overlays import (  # noqa: E402
    DATE_COL,
    LAT_COL,
    LON_COL,
    PROB_COL,
    TARGET_COL,
    WEIGHT_COL,
    find_prediction_files_for_model,
    load_regions,
    load_world_boundaries,
    plot_boundaries,
    read_prediction_columns,
    safe_slug,
    spatial_grid,
)


@dataclass(frozen=True)
class PredictionDiagnosticConfig:
    results_dir: Path
    regions_file: Path = Path("configs/regions_example.yaml")
    source: str = "legacy"
    model: str = "best_neural"
    prob_col: str = "auto"
    regions: list[str] | None = None
    include_global: bool = True
    time_frequency: str = "month"
    output_dir: Path | None = None
    formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 260
    grid_resolution: float | None = None
    plot_interpolation_resolution: float | None = None
    country_shapes: Path = Path("data/countries")
    error_colormap: str = "risk_residual"
    ground_truth_smoothing_sigma_cells: float = 0.0
    recalibrate_on_test: bool = True
    require_full_grid_predictions: bool = True
    generate_full_grid_predictions: bool = False
    feature_config: Path = Path("configs/features_config_30d.yaml")
    target_config: Path = Path("configs/target_config.yaml")
    catboost_config: Path = Path("configs/catboost_train_config.yaml")
    full_grid_model_path: Path | None = None
    full_grid_feature_schema_path: Path | None = None
    deployment_grid_countries: list[str] | None = None
    deployment_grid_coordinate_bounds: list[float] | None = None
    deployment_grid_clip_to_feature_bounds: bool = True
    test_start_date: str | None = None
    test_end_date: str | None = None
    max_grid_rows_per_chunk: int | None = None
    cache_full_grid_features: bool = False
    sample_prediction_days_per_month: int | None = None
    months_per_feature_chunk: int = 6
    seed: int = 17


def default_output_dir(results_dir: Path) -> Path:
    return results_dir / "shared_artifacts" / "prediction_diagnostics"


def _metric_column(frame: pd.DataFrame) -> str | None:
    for column in ["average_precision", "PR-AUC", "PR_AUC", "ap"]:
        if column in frame.columns:
            return column
    return None


def _model_column(frame: pd.DataFrame) -> str | None:
    for column in ["model_name", "model", "Model", "experiment"]:
        if column in frame.columns:
            return column
    return None


def _global_rows(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ["region", "Region", "region_display"]:
        if column in frame.columns:
            return frame[frame[column].astype(str).str.lower().eq("global")].copy()
    return frame.copy()


def _read_metric_table(results_dir: Path, source: str) -> pd.DataFrame:
    source = source.lower()
    if source == "primary":
        candidates = [
            results_dir / "primary_full_grid_calibrated" / "model_comparison.csv",
            results_dir / "shared_artifacts" / "primary_full_grid_calibrated" / "model_comparison.csv",
            results_dir
            / "shared_artifacts"
            / "raw_tables_jsonl"
            / "primary_full_grid_calibrated_model_comparison.jsonl.gz",
        ]
    else:
        candidates = [
            results_dir / "main_model_comparison.csv",
            results_dir / "legacy_sampled_case_control" / "model_comparison.csv",
            results_dir / "shared_artifacts" / "legacy_sampled_case_control" / "model_comparison.csv",
            results_dir / "shared_artifacts" / "raw_tables_jsonl" / "main_model_comparison.jsonl.gz",
            results_dir
            / "shared_artifacts"
            / "raw_tables_jsonl"
            / "legacy_sampled_case_control_model_comparison.jsonl.gz",
        ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".gz":
            return pd.read_json(path, orient="records", lines=True, compression="gzip")
        return pd.read_csv(path)
    return pd.DataFrame()


def _prediction_key_from_label(value: object) -> str:
    text = str(value)
    if text.startswith("nn_global_full_"):
        return text
    for key, label in NN_LABELS.items():
        if text == label:
            return f"nn_global_full_{key}"
    return text


def resolve_model(results_dir: Path, requested: str, source: str) -> str:
    normalized = safe_slug(requested).lower()
    if normalized not in {"best", "best_all", "best_neural", "best_nn", "best_nns"}:
        return requested

    table = _read_metric_table(results_dir, source)
    if table.empty:
        return requested

    model_col = _model_column(table)
    metric_col = _metric_column(table)
    if model_col is None or metric_col is None:
        return requested

    work = _global_rows(table)
    if normalized in {"best_neural", "best_nn", "best_nns"}:
        model_text = work[model_col].astype(str)
        neural_labels = set(NN_LABELS.values())
        work = work[
            model_text.str.startswith("nn_global_full_")
            | model_text.isin(neural_labels)
            | model_text.str.contains("MLP|Transformer|TSN|LSTM|TemporalConvNet", case=False, regex=True)
        ].copy()
    if work.empty:
        return requested

    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col])
    if work.empty:
        return requested

    best = work.sort_values(metric_col, ascending=False).iloc[0]
    return _prediction_key_from_label(best[model_col])


def _weighted_counts(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["expected_fire_positive_grid_cells"] = (
        pd.to_numeric(work[PROB_COL], errors="coerce").fillna(0.0).astype(float)
        * pd.to_numeric(work[WEIGHT_COL], errors="coerce").fillna(1.0).astype(float)
    )
    work["observed_fire_positive_grid_cells"] = (
        pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0.0).astype(float)
        * pd.to_numeric(work[WEIGHT_COL], errors="coerce").fillna(1.0).astype(float)
    )
    work["year"] = pd.to_datetime(work[DATE_COL]).dt.year.astype(int)
    return work


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def recalibrate_probabilities_on_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit a diagnostic-only test-set calibrator and replace PROB_COL.

    This is intentionally not a production evaluation path. It is useful for
    residual maps where we want spatial over/under-estimation after matching the
    test-set base rate as closely as possible.
    """

    y = pd.to_numeric(frame[TARGET_COL], errors="coerce").fillna(0).astype(int).to_numpy()
    p = pd.to_numeric(frame[PROB_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w = pd.to_numeric(frame[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    metadata: dict[str, object] = {
        "enabled": True,
        "method": "test_platt_logit",
        "rows": int(finite.sum()),
        "mean_probability_before": float(np.average(np.clip(p[finite], 0.0, 1.0), weights=w[finite])) if finite.any() else None,
        "observed_weighted_prevalence": (
            float(np.average(y[finite], weights=w[finite])) if finite.any() else None
        ),
    }
    if int(finite.sum()) < 2 or np.unique(y[finite]).size < 2:
        metadata["status"] = "skipped_single_class"
        return frame.copy(), metadata

    x = _logit(p[finite]).reshape(-1, 1)
    try:
        calibrator = LogisticRegression(solver="lbfgs", max_iter=300)
        calibrator.fit(x, y[finite], sample_weight=w[finite])
        calibrated = p.copy()
        calibrated[finite] = calibrator.predict_proba(x)[:, 1]
        mean_after_platt = float(np.average(calibrated[finite], weights=w[finite]))
        observed_prevalence = float(np.average(y[finite], weights=w[finite]))
        prevalence_scale = observed_prevalence / mean_after_platt if mean_after_platt > 0 else 1.0
        calibrated = np.clip(calibrated * prevalence_scale, 0.0, 1.0)
        out = frame.copy()
        out["diagnostic_probability_before_test_calibration"] = p
        out[PROB_COL] = np.clip(calibrated, 0.0, 1.0)
        metadata.update(
            {
                "status": "fit",
                "intercept": float(calibrator.intercept_[0]),
                "slope": float(calibrator.coef_[0][0]),
                "mean_probability_after_platt": mean_after_platt,
                "post_platt_prevalence_scale": prevalence_scale,
                "mean_probability_after": float(np.average(out[PROB_COL].to_numpy(dtype=float)[finite], weights=w[finite])),
            }
        )
        return out, metadata
    except Exception as exc:
        expected = float(np.sum(p[finite] * w[finite]))
        observed = float(np.sum(y[finite] * w[finite]))
        ratio = observed / expected if expected > 0 else 1.0
        out = frame.copy()
        out["diagnostic_probability_before_test_calibration"] = p
        out[PROB_COL] = np.clip(p * ratio, 0.0, 1.0)
        metadata.update(
            {
                "status": "fallback_count_ratio",
                "reason": str(exc),
                "count_ratio": ratio,
                "mean_probability_after": float(np.average(out[PROB_COL].to_numpy(dtype=float)[finite], weights=w[finite])),
            }
        )
        return out, metadata


def _is_full_grid_prediction_frame(frame: pd.DataFrame) -> bool:
    if WEIGHT_COL not in frame.columns:
        return True
    weights = pd.to_numeric(frame[WEIGHT_COL], errors="coerce").dropna()
    if weights.empty:
        return True
    return bool(np.allclose(weights.to_numpy(dtype=float), 1.0, rtol=1e-6, atol=1e-6))


def _assert_full_grid_predictions(frame: pd.DataFrame, prediction_files: list[Path]) -> None:
    if _is_full_grid_prediction_frame(frame):
        return
    max_weight = float(pd.to_numeric(frame[WEIGHT_COL], errors="coerce").max())
    files = "\n".join(f"  - {path}" for path in prediction_files)
    raise RuntimeError(
        "Prediction diagnostics require true full-grid predictions, but the selected prediction file "
        f"contains sampling weights up to {max_weight:g}. Refusing to plot/interpolate sparse weighted data.\n"
        "Regenerate primary predictions with full_grid_mode='full_grid', weighted_grid_sample=false, and "
        "weighted_grid_sample_fraction unset, then rerun this diagnostic.\n"
        f"Selected file(s):\n{files}"
    )


def _unique_paths(paths: Iterable[Path | None]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _catboost_artifact_keys(model: str) -> list[str]:
    normalized = safe_slug(model).lower()
    keys = [safe_slug(model), normalized]
    if normalized in {"catboost", "catboost_full", "full_catboost"} or str(model).lower() == "catboost":
        keys.extend(["catboost_full", "CatBoost"])
    return list(dict.fromkeys(key for key in keys if key))


def _resolve_existing_artifact(candidates: Iterable[Path], label: str) -> Path:
    checked = _unique_paths(candidates)
    for path in checked:
        if path.exists():
            return path
    searched = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(f"Could not find {label}. Searched:\n{searched}")


def _resolve_catboost_artifacts(config: PredictionDiagnosticConfig, model: str) -> tuple[Path, Path]:
    keys = _catboost_artifact_keys(model)
    model_dirs = [
        config.results_dir / "shared_artifacts" / "models",
        config.results_dir / "models",
        Path("models"),
    ]
    model_candidates = [config.full_grid_model_path]
    feature_candidates = [config.full_grid_feature_schema_path]
    for directory in model_dirs:
        for key in keys:
            model_candidates.append(directory / f"{key}.cbm")
            feature_candidates.append(directory / f"{key}_features.json")
    return (
        _resolve_existing_artifact(model_candidates, "CatBoost full-grid model"),
        _resolve_existing_artifact(feature_candidates, "CatBoost feature schema"),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _load_feature_schema(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = [str(value) for value in payload.get("features", []) if str(value)]
    categorical = [str(value) for value in payload.get("categorical_features", []) if str(value)]
    if not features:
        raise ValueError(f"{path} does not contain a non-empty `features` list.")
    return features, categorical


def _region_union_bounds(regions: Iterable[object]) -> list[float] | None:
    bounds: list[tuple[float, float, float, float]] = []
    for region in regions:
        if getattr(region, "name", None) == "global":
            return None
        lat_min = getattr(region, "lat_min", None)
        lat_max = getattr(region, "lat_max", None)
        lon_min = getattr(region, "lon_min", None)
        lon_max = getattr(region, "lon_max", None)
        if None in {lat_min, lat_max, lon_min, lon_max}:
            return None
        bounds.append((float(lat_min), float(lon_min), float(lat_max), float(lon_max)))
    if not bounds:
        return None
    return [
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    ]


def _diagnostic_deployment_config(
    config: PredictionDiagnosticConfig,
    *,
    regions: Iterable[object],
    coordinate_bounds: list[float] | None = None,
) -> SimpleNamespace:
    if coordinate_bounds is None:
        coordinate_bounds = config.deployment_grid_coordinate_bounds
    if coordinate_bounds is None:
        coordinate_bounds = _region_union_bounds(regions)
    return SimpleNamespace(
        seed=config.seed,
        test_start_date=config.test_start_date,
        test_end_date=config.test_end_date,
        deployment_grid_resolution=float(config.grid_resolution or 0.1),
        deployment_grid_countries=config.deployment_grid_countries,
        deployment_grid_coordinate_bounds=coordinate_bounds,
        deployment_grid_clip_to_feature_bounds=config.deployment_grid_clip_to_feature_bounds,
        full_grid_mode="full_grid",
        weighted_grid_sample=False,
        weighted_grid_sample_fraction=None,
        weighted_grid_sample_include_all_positives=True,
        max_grid_rows_per_chunk=config.max_grid_rows_per_chunk,
        deployment_grid_months_per_chunk=max(1, int(config.months_per_feature_chunk or 1)),
        cache_full_grid_features=config.cache_full_grid_features,
        feature_config=config.feature_config,
        target_config=config.target_config,
    )


def _full_grid_generation_scopes(
    config: PredictionDiagnosticConfig,
    regions: Iterable[object],
) -> list[tuple[str, list[object], list[float] | None]]:
    region_list = list(regions)
    if config.deployment_grid_coordinate_bounds is not None or any(
        getattr(region, "name", None) == "global" for region in region_list
    ):
        return [("all_regions", region_list, config.deployment_grid_coordinate_bounds)]

    scopes: list[tuple[str, list[object], list[float] | None]] = []
    for region in region_list:
        bounds = _region_union_bounds([region])
        scopes.append((str(getattr(region, "name", "region")), [region], bounds))
    return scopes or [("all_regions", region_list, None)]


def _compact_sum_parts(parts: list[pd.DataFrame], key_cols: list[str]) -> list[pd.DataFrame]:
    if not parts:
        return []
    merged = pd.concat(parts, ignore_index=True)
    value_cols = [col for col in merged.columns if col not in key_cols]
    compacted = merged.groupby(key_cols, observed=True, dropna=False)[value_cols].sum().reset_index()
    return [compacted]


def _full_grid_aggregate_cache_paths(
    output_dir: Path,
    *,
    scope_name: str,
    approximation_tag: str,
    resolved_model: str,
    test_start_date: str | None,
    test_end_date: str | None,
    country: str | None,
    period: str,
) -> dict[str, Path]:
    tag = "_".join(
        safe_slug(part)
        for part in [
            scope_name,
            approximation_tag,
            resolved_model,
            test_start_date or "default_start",
            test_end_date or "default_end",
            country or "all",
            period,
        ]
    )
    root = output_dir / "full_grid_aggregate_cache"
    return {
        "calibration": root / f"{tag}_calibration.parquet",
        "timeseries": root / f"{tag}_timeseries.parquet",
        "spatial": root / f"{tag}_spatial.parquet",
    }


def _read_full_grid_aggregate_cache(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not all(path.exists() for path in paths.values()):
        return None
    return (
        pd.read_parquet(paths["calibration"]),
        pd.read_parquet(paths["timeseries"]),
        pd.read_parquet(paths["spatial"]),
    )


def _write_full_grid_aggregate_cache(
    paths: dict[str, Path],
    calibration: pd.DataFrame,
    timeseries: pd.DataFrame,
    spatial: pd.DataFrame,
) -> None:
    for path, frame in [
        (paths["calibration"], calibration),
        (paths["timeseries"], timeseries),
        (paths["spatial"], spatial),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def _prediction_day_sample(
    rows: pd.DataFrame,
    *,
    days_per_month: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if days_per_month is None or int(days_per_month) <= 0 or rows.empty:
        return rows, {"enabled": False}

    work = rows.copy()
    dates = pd.to_datetime(work[DATE_COL]).dt.normalize()
    unique_dates = pd.Index(sorted(dates.dropna().unique()))
    requested = max(1, int(days_per_month))
    if len(unique_dates) == 0:
        return rows.iloc[0:0].copy(), {
            "enabled": True,
            "requested_days_per_month": requested,
            "source_days": 0,
            "sampled_days": 0,
            "scale": 1.0,
            "months": [],
        }

    selected_keys: set[str] = set()
    scale_by_date: dict[str, float] = {}
    month_metadata: list[dict[str, object]] = []
    unique_series = pd.Series(unique_dates)
    for period, values in unique_series.groupby(unique_series.dt.to_period("M")):
        month_dates = pd.Index(sorted(pd.to_datetime(values).dt.normalize().unique()))
        if len(month_dates) <= requested:
            selected = month_dates
        else:
            positions = np.linspace(0, len(month_dates) - 1, requested)
            selected = month_dates[np.unique(np.rint(positions).astype(int))]
        scale = float(len(month_dates)) / float(len(selected)) if len(selected) else 1.0
        selected_for_month = [pd.Timestamp(value).date().isoformat() for value in selected]
        for key in selected_for_month:
            selected_keys.add(key)
            scale_by_date[key] = scale
        month_metadata.append(
            {
                "month": str(period),
                "source_days": int(len(month_dates)),
                "sampled_days": int(len(selected)),
                "scale": scale,
                "sampled_dates": selected_for_month,
            }
        )

    date_keys = dates.dt.date.astype(str)
    selected_mask = date_keys.isin(selected_keys)
    sampled = work.loc[selected_mask].copy()
    sampled_date_keys = pd.to_datetime(sampled[DATE_COL]).dt.normalize().dt.date.astype(str)
    sample_scales = sampled_date_keys.map(scale_by_date).fillna(1.0).to_numpy(dtype=float)
    sampled[WEIGHT_COL] = pd.to_numeric(sampled[WEIGHT_COL], errors="coerce").fillna(1.0) * sample_scales
    return sampled.reset_index(drop=True), {
        "enabled": True,
        "requested_days_per_month": requested,
        "source_days": int(len(unique_dates)),
        "sampled_days": int(len(selected_keys)),
        "scale": float(np.mean([item["scale"] for item in month_metadata])) if month_metadata else 1.0,
        "months": month_metadata,
    }


def _deployment_prediction_frame(rows: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    keep = [col for col in [DATE_COL, LAT_COL, LON_COL, TARGET_COL, WEIGHT_COL, "country"] if col in rows.columns]
    frame = rows[keep].copy()
    if TARGET_COL not in frame.columns:
        frame[TARGET_COL] = np.int8(0)
    if WEIGHT_COL not in frame.columns:
        frame[WEIGHT_COL] = 1.0
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL]).dt.normalize()
    frame[PROB_COL] = 0.0
    frame["model_name"] = model_name
    return frame


def _filter_rows_to_feature_cells(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    *,
    resolution: float,
) -> pd.DataFrame:
    if rows.empty or features.empty or not {LAT_COL, LON_COL}.issubset(features.columns):
        return rows.iloc[0:0].copy()
    scale = 1.0 / float(resolution)
    feature_keys = pd.MultiIndex.from_arrays(
        [
            np.rint(pd.to_numeric(features[LAT_COL], errors="coerce").to_numpy(dtype=float) * scale).astype(np.int64),
            np.rint(pd.to_numeric(features[LON_COL], errors="coerce").to_numpy(dtype=float) * scale).astype(np.int64),
        ]
    ).unique()
    row_keys = pd.MultiIndex.from_arrays(
        [
            np.rint(pd.to_numeric(rows[LAT_COL], errors="coerce").to_numpy(dtype=float) * scale).astype(np.int64),
            np.rint(pd.to_numeric(rows[LON_COL], errors="coerce").to_numpy(dtype=float) * scale).astype(np.int64),
        ]
    )
    return rows.loc[row_keys.isin(feature_keys)].copy()


def _combine_prediction_observed_aggregates(
    prediction: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    key_cols: list[str],
) -> pd.DataFrame:
    if observed.empty:
        return prediction.copy()
    expected = prediction[key_cols + ["expected_fire_positive_grid_cells_raw"]].copy()
    expected = (
        expected.groupby(key_cols, observed=True, dropna=False)
        .agg(expected_fire_positive_grid_cells_raw=("expected_fire_positive_grid_cells_raw", "sum"))
        .reset_index()
    )
    out = observed.drop(columns=["expected_fire_positive_grid_cells_raw"], errors="ignore").merge(
        expected,
        on=key_cols,
        how="left",
    )
    out["expected_fire_positive_grid_cells_raw"] = pd.to_numeric(
        out["expected_fire_positive_grid_cells_raw"],
        errors="coerce",
    ).fillna(0.0)
    return out


def _aggregate_full_grid_chunk(
    frame: pd.DataFrame,
    *,
    regions: Iterable[object],
    frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = _weighted_counts(frame)
    work["month"] = pd.to_datetime(work[DATE_COL]).dt.month.astype(np.int16)
    work["rows"] = 1
    start, end = _period_bounds(work[DATE_COL], frequency)
    work["period_start"] = start
    work["period_end"] = end

    calibration = (
        work.groupby(["month"], observed=True, dropna=False)
        .agg(
            rows=("rows", "sum"),
            weighted_support=(WEIGHT_COL, "sum"),
            expected_fire_positive_grid_cells_raw=("expected_fire_positive_grid_cells", "sum"),
            observed_fire_positive_grid_cells=("observed_fire_positive_grid_cells", "sum"),
        )
        .reset_index()
    )

    timeseries_parts: list[pd.DataFrame] = []
    spatial_parts: list[pd.DataFrame] = []
    for region in regions:
        region_frame = work.loc[region.mask(work)]
        if region_frame.empty:
            continue
        timeseries = (
            region_frame.groupby(
                ["model_name", "period_start", "period_end", "month"],
                observed=True,
                dropna=False,
            )
            .agg(
                weighted_support=(WEIGHT_COL, "sum"),
                expected_fire_positive_grid_cells_raw=("expected_fire_positive_grid_cells", "sum"),
                observed_fire_positive_grid_cells=("observed_fire_positive_grid_cells", "sum"),
            )
            .reset_index()
        )
        timeseries.insert(1, "region", region.name)
        timeseries_parts.append(timeseries)

        spatial = (
            region_frame.groupby(
                ["model_name", "year", "month", LAT_COL, LON_COL],
                observed=True,
                dropna=False,
            )
            .agg(
                weighted_support=(WEIGHT_COL, "sum"),
                expected_fire_positive_grid_cells_raw=("expected_fire_positive_grid_cells", "sum"),
                observed_fire_positive_grid_cells=("observed_fire_positive_grid_cells", "sum"),
            )
            .reset_index()
        )
        spatial.insert(1, "region", region.name)
        spatial_parts.append(spatial)

    timeseries_out = pd.concat(timeseries_parts, ignore_index=True) if timeseries_parts else pd.DataFrame()
    spatial_out = pd.concat(spatial_parts, ignore_index=True) if spatial_parts else pd.DataFrame()
    return calibration, timeseries_out, spatial_out


def _calibration_scales_from_monthly(
    calibration: pd.DataFrame,
    *,
    enabled: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if calibration.empty:
        metadata = {"enabled": bool(enabled), "status": "skipped_no_rows", "method": "test_monthly_count_ratio_streaming"}
        return pd.DataFrame(columns=["month", "probability_scale"]), metadata

    out = calibration.copy()
    raw_expected = pd.to_numeric(out["expected_fire_positive_grid_cells_raw"], errors="coerce").fillna(0.0)
    observed = pd.to_numeric(out["observed_fire_positive_grid_cells"], errors="coerce").fillna(0.0)
    support = pd.to_numeric(out["weighted_support"], errors="coerce").fillna(0.0)
    total_expected = float(raw_expected.sum())
    total_observed = float(observed.sum())
    total_support = float(support.sum())
    global_scale = total_observed / total_expected if total_expected > 0 else 1.0

    if enabled:
        month_scale = np.divide(
            observed.to_numpy(dtype=float),
            raw_expected.to_numpy(dtype=float),
            out=np.full(len(out), global_scale, dtype=float),
            where=raw_expected.to_numpy(dtype=float) > 0,
        )
    else:
        month_scale = np.ones(len(out), dtype=float)
    out["probability_scale"] = month_scale
    out["mean_probability_before"] = np.divide(
        raw_expected.to_numpy(dtype=float),
        support.to_numpy(dtype=float),
        out=np.zeros(len(out), dtype=float),
        where=support.to_numpy(dtype=float) > 0,
    )
    out["mean_probability_after"] = np.minimum(out["mean_probability_before"] * out["probability_scale"], 1.0)
    out["observed_prevalence"] = np.divide(
        observed.to_numpy(dtype=float),
        support.to_numpy(dtype=float),
        out=np.zeros(len(out), dtype=float),
        where=support.to_numpy(dtype=float) > 0,
    )
    metadata = {
        "enabled": bool(enabled),
        "method": "test_monthly_count_ratio_streaming",
        "status": "fit" if enabled else "disabled",
        "rows": int(pd.to_numeric(out.get("rows", 0), errors="coerce").fillna(0).sum()),
        "weighted_support": total_support,
        "observed_fire_positive_grid_cells": total_observed,
        "expected_fire_positive_grid_cells_before": total_expected,
        "global_count_ratio": global_scale,
        "mean_probability_before": total_expected / total_support if total_support > 0 else None,
        "observed_weighted_prevalence": total_observed / total_support if total_support > 0 else None,
    }
    scaled_expected = float(np.minimum(raw_expected.to_numpy(dtype=float) * month_scale, support.to_numpy(dtype=float)).sum())
    metadata["expected_fire_positive_grid_cells_after"] = scaled_expected
    metadata["mean_probability_after"] = scaled_expected / total_support if total_support > 0 else None
    return out, metadata


def _apply_monthly_scales(
    frame: pd.DataFrame,
    scales: pd.DataFrame,
    *,
    raw_expected_col: str = "expected_fire_positive_grid_cells_raw",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.merge(scales[["month", "probability_scale"]], on="month", how="left")
    work["probability_scale"] = pd.to_numeric(work["probability_scale"], errors="coerce").fillna(1.0)
    work["expected_fire_positive_grid_cells"] = (
        pd.to_numeric(work[raw_expected_col], errors="coerce").fillna(0.0) * work["probability_scale"]
    )
    if "weighted_support" in work.columns:
        support = pd.to_numeric(work["weighted_support"], errors="coerce").fillna(np.inf)
    else:
        support = pd.Series(np.full(len(work), np.inf), index=work.index)
    work["expected_fire_positive_grid_cells"] = np.minimum(
        work["expected_fire_positive_grid_cells"].to_numpy(dtype=float),
        support.to_numpy(dtype=float),
    )
    return work


def _finalize_full_grid_timeseries(raw_timeseries: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    if raw_timeseries.empty:
        return pd.DataFrame()
    work = _apply_monthly_scales(raw_timeseries, scales)
    grouped = (
        work.groupby(["model_name", "region", "period_start", "period_end"], observed=True, dropna=False)
        .agg(
            expected_fire_positive_grid_cells=("expected_fire_positive_grid_cells", "sum"),
            observed_fire_positive_grid_cells=("observed_fire_positive_grid_cells", "sum"),
        )
        .reset_index()
    )
    grouped["error_fire_positive_grid_cells"] = (
        grouped["expected_fire_positive_grid_cells"] - grouped["observed_fire_positive_grid_cells"]
    )
    grouped["absolute_error_fire_positive_grid_cells"] = grouped["error_fire_positive_grid_cells"].abs()
    grouped["period_start"] = pd.to_datetime(grouped["period_start"]).dt.date.astype(str)
    grouped["period_end"] = pd.to_datetime(grouped["period_end"]).dt.date.astype(str)
    return grouped[
        [
            "model_name",
            "region",
            "period_start",
            "period_end",
            "expected_fire_positive_grid_cells",
            "observed_fire_positive_grid_cells",
            "error_fire_positive_grid_cells",
            "absolute_error_fire_positive_grid_cells",
        ]
    ].copy()


def _finish_spatial_period(
    spatial: pd.DataFrame,
    *,
    region: object,
    period: str,
    grid_resolution: float,
    ground_truth_smoothing_sigma_cells: float,
) -> pd.DataFrame:
    if spatial.empty:
        return spatial
    support = pd.to_numeric(spatial["weighted_support"], errors="coerce").to_numpy(dtype=float)
    out = spatial.copy()
    out.insert(2, "period", period)
    out["predicted_risk"] = np.divide(
        out["expected_fire_positive_grid_cells"].to_numpy(dtype=float),
        support,
        out=np.zeros(len(out), dtype=float),
        where=support > 0,
    )
    out["observed_risk"] = np.divide(
        out["observed_fire_positive_grid_cells"].to_numpy(dtype=float),
        support,
        out=np.zeros(len(out), dtype=float),
        where=support > 0,
    )
    out = _add_smoothed_observed_risk(
        out,
        region=region,
        grid_resolution=grid_resolution,
        sigma_cells=ground_truth_smoothing_sigma_cells,
    )
    out["error_fire_positive_grid_cells"] = (
        out["expected_fire_positive_grid_cells"] - out["observed_fire_positive_grid_cells"]
    )
    out["risk_error_smoothed_observed"] = out["predicted_risk"] - out["smoothed_observed_risk"]
    return out


def _finalize_full_grid_spatial(
    raw_spatial: pd.DataFrame,
    scales: pd.DataFrame,
    *,
    regions: Iterable[object],
    grid_resolution: float,
    ground_truth_smoothing_sigma_cells: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_spatial.empty:
        return pd.DataFrame(), pd.DataFrame()
    region_by_name = {region.name: region for region in regions}
    work = _apply_monthly_scales(raw_spatial, scales)
    yearly = (
        work.groupby(["model_name", "region", "year", LAT_COL, LON_COL], observed=True, dropna=False)
        .agg(
            weighted_support=("weighted_support", "sum"),
            expected_fire_positive_grid_cells=("expected_fire_positive_grid_cells", "sum"),
            observed_fire_positive_grid_cells=("observed_fire_positive_grid_cells", "sum"),
        )
        .reset_index()
    )

    cell_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for (model_name, region_name), group in yearly.groupby(["model_name", "region"], observed=True, dropna=False):
        region = region_by_name.get(str(region_name))
        if region is None:
            continue
        years = sorted(int(year) for year in group["year"].dropna().unique())
        if not years:
            continue
        mean = (
            group.groupby(["model_name", "region", LAT_COL, LON_COL], observed=True, dropna=False)
            .agg(
                weighted_support=("weighted_support", "sum"),
                expected_fire_positive_grid_cells=("expected_fire_positive_grid_cells", "sum"),
                observed_fire_positive_grid_cells=("observed_fire_positive_grid_cells", "sum"),
            )
            .reset_index()
        )
        divisor = float(len(years))
        for col in [
            "weighted_support",
            "expected_fire_positive_grid_cells",
            "observed_fire_positive_grid_cells",
        ]:
            mean[col] = pd.to_numeric(mean[col], errors="coerce").fillna(0.0) / divisor

        period_frames: list[tuple[str, pd.DataFrame]] = [(f"{min(years)}-{max(years)} mean", mean)]
        for year in years:
            per_year = group.loc[group["year"].eq(year)].drop(columns=["year"]).copy()
            period_frames.append((str(year), per_year))

        for period, period_frame in period_frames:
            finished = _finish_spatial_period(
                period_frame.reset_index(drop=True),
                region=region,
                period=period,
                grid_resolution=grid_resolution,
                ground_truth_smoothing_sigma_cells=ground_truth_smoothing_sigma_cells,
            )
            if finished.empty:
                continue
            cell_rows.append(
                finished[
                    [
                        "model_name",
                        "region",
                        "period",
                        LAT_COL,
                        LON_COL,
                        "weighted_support",
                        "predicted_risk",
                        "observed_risk",
                        "smoothed_observed_risk",
                        "risk_error_smoothed_observed",
                        "expected_fire_positive_grid_cells",
                        "observed_fire_positive_grid_cells",
                        "error_fire_positive_grid_cells",
                    ]
                ].copy()
            )
            errors = finished["error_fire_positive_grid_cells"].to_numpy(dtype=float)
            risk_errors = finished["risk_error_smoothed_observed"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "model_name": str(model_name),
                    "region": str(region_name),
                    "period": period,
                    "cells": int(len(finished)),
                    "mean_predicted_risk": float(np.nanmean(finished["predicted_risk"])),
                    "mean_smoothed_observed_risk": float(np.nanmean(finished["smoothed_observed_risk"])),
                    "mean_risk_error": float(np.nanmean(risk_errors)) if len(risk_errors) else math.nan,
                    "mean_abs_risk_error": float(np.nanmean(np.abs(risk_errors))) if len(risk_errors) else math.nan,
                    "expected_fire_positive_grid_cells": float(finished["expected_fire_positive_grid_cells"].sum()),
                    "observed_fire_positive_grid_cells": float(finished["observed_fire_positive_grid_cells"].sum()),
                    "bias_fire_positive_grid_cells": float(np.sum(errors)),
                    "mean_abs_error_per_cell": float(np.mean(np.abs(errors))) if len(errors) else math.nan,
                }
            )

    cells = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return cells, summary


def generate_full_grid_prediction_diagnostic_tables(
    config: PredictionDiagnosticConfig,
    *,
    regions: Iterable[object],
    resolved_model: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    normalized = safe_slug(resolved_model).lower()
    if normalized not in {"catboost", "catboost_full"} and str(resolved_model).lower() != "catboost":
        raise ValueError(
            "Streaming full-grid diagnostic generation currently supports the saved CatBoost model only. "
            f"Requested: {resolved_model!r}."
        )

    from .deployment_grid import iter_deployment_grid_chunks
    from .full_grid_evaluation import _features_for_chunk, _prediction_frame_from_features
    from .tabular import CATBOOST_IMPORT_ERROR, CatBoostClassifier, make_catboost_raw_predict_fn

    if CatBoostClassifier is None:
        raise ImportError(f"CatBoost is required for full-grid diagnostic generation: {CATBOOST_IMPORT_ERROR}")

    model_path, schema_path = _resolve_catboost_artifacts(config, resolved_model)
    feature_columns, categorical_columns = _load_feature_schema(schema_path)
    feature_payload = _load_yaml_mapping(config.feature_config)
    target_payload = _load_yaml_mapping(config.target_config)

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    predict_raw_fn = make_catboost_raw_predict_fn(model, categorical_columns, feature_payload)
    region_list = list(regions)
    diagnostic_model_name = "CatBoost" if str(resolved_model).lower() == "catboost" else resolved_model

    calibration_parts: list[pd.DataFrame] = []
    timeseries_parts: list[pd.DataFrame] = []
    spatial_parts: list[pd.DataFrame] = []
    chunk_count = 0
    cached_chunk_count = 0
    generated_rows = 0
    feature_rows = 0
    compact_every = 20
    scopes = _full_grid_generation_scopes(config, region_list)
    months_tag = f"months{max(1, int(config.months_per_feature_chunk or 1))}"
    approximation_tag = (
        f"days{int(config.sample_prediction_days_per_month)}_{months_tag}"
        if config.sample_prediction_days_per_month and int(config.sample_prediction_days_per_month) > 0
        else f"exact_days_{months_tag}"
    )
    sample_metadata: list[dict[str, object]] = []

    for scope_name, scope_regions, scope_bounds in scopes:
        deployment_config = _diagnostic_deployment_config(
            config,
            regions=scope_regions,
            coordinate_bounds=scope_bounds,
        )
        feature_cache_split = f"test_prediction_diagnostics_{safe_slug(scope_name)}_{safe_slug(approximation_tag)}"
        for chunk in iter_deployment_grid_chunks(
            config=deployment_config,
            feature_config=feature_payload,
            target_config=target_payload,
            split_name="test",
        ):
            if chunk.rows.empty:
                continue
            chunk_count += 1
            generated_rows += int(len(chunk.rows))
            aggregate_cache_paths = _full_grid_aggregate_cache_paths(
                output_dir,
                scope_name=scope_name,
                approximation_tag=approximation_tag,
                resolved_model=resolved_model,
                test_start_date=config.test_start_date,
                test_end_date=config.test_end_date,
                country=chunk.country,
                period=chunk.period,
            )
            cached_aggregates = _read_full_grid_aggregate_cache(aggregate_cache_paths)
            if cached_aggregates is not None:
                cached_chunk_count += 1
                calibration, timeseries, spatial = cached_aggregates
                calibration_parts.append(calibration)
                if not timeseries.empty:
                    timeseries_parts.append(timeseries)
                if not spatial.empty:
                    spatial_parts.append(spatial)
                logging.info(
                    "Loaded cached prediction-diagnostic aggregates for %s %s/%s",
                    scope_name,
                    chunk.country,
                    chunk.period,
                )
                continue
            logging.info(
                "Prediction diagnostics full-grid chunk %s %s/%s rows=%d",
                scope_name,
                chunk.country,
                chunk.period,
                len(chunk.rows),
            )
            prediction_rows, day_sample = _prediction_day_sample(
                chunk.rows,
                days_per_month=config.sample_prediction_days_per_month,
            )
            sample_metadata.append(
                {
                    "scope": scope_name,
                    "country": chunk.country,
                    "period": chunk.period,
                    "input_rows": int(len(chunk.rows)),
                    "prediction_rows": int(len(prediction_rows)),
                    **day_sample,
                }
            )
            features = _features_for_chunk(
                prediction_rows,
                config=deployment_config,
                output_dir=output_dir,
                split_name=feature_cache_split,
                country=chunk.country,
                period=chunk.period,
                feature_config=config.feature_config,
                feature_columns=feature_columns,
                test_mode=True,
            )
            if features.empty:
                continue
            feature_rows += int(len(features))
            pred_frame = _prediction_frame_from_features(
                features,
                model_name=diagnostic_model_name,
                model_type="CatBoost",
                split_name="test",
                feature_columns=feature_columns,
                predict_raw_fn=predict_raw_fn,
                feature_config_path=config.feature_config,
                target_config_path=config.target_config,
                model_path=model_path,
                calibration_method="test_monthly_count_ratio",
            )
            pred_frame[DATE_COL] = pd.to_datetime(pred_frame[DATE_COL]).dt.normalize()
            pred_frame[PROB_COL] = pd.to_numeric(pred_frame["prob_raw"], errors="coerce").fillna(0.0)
            if day_sample.get("enabled"):
                pred_for_expected = pred_frame.copy()
                pred_for_expected[TARGET_COL] = np.int8(0)
                pred_calibration, pred_timeseries, pred_spatial = _aggregate_full_grid_chunk(
                    pred_for_expected,
                    regions=scope_regions,
                    frequency=config.time_frequency,
                )
                observed_rows = _filter_rows_to_feature_cells(
                    chunk.rows,
                    features,
                    resolution=float(config.grid_resolution or 0.1),
                )
                observed_frame = _deployment_prediction_frame(observed_rows, model_name=diagnostic_model_name)
                obs_calibration, obs_timeseries, obs_spatial = _aggregate_full_grid_chunk(
                    observed_frame,
                    regions=scope_regions,
                    frequency=config.time_frequency,
                )
                calibration = _combine_prediction_observed_aggregates(
                    pred_calibration,
                    obs_calibration,
                    key_cols=["month"],
                )
                timeseries = _combine_prediction_observed_aggregates(
                    pred_timeseries,
                    obs_timeseries,
                    key_cols=["model_name", "region", "period_start", "period_end", "month"],
                )
                spatial = _combine_prediction_observed_aggregates(
                    pred_spatial,
                    obs_spatial,
                    key_cols=["model_name", "region", "year", "month", LAT_COL, LON_COL],
                )
            else:
                calibration, timeseries, spatial = _aggregate_full_grid_chunk(
                    pred_frame,
                    regions=scope_regions,
                    frequency=config.time_frequency,
                )
            _write_full_grid_aggregate_cache(aggregate_cache_paths, calibration, timeseries, spatial)
            calibration_parts.append(calibration)
            if not timeseries.empty:
                timeseries_parts.append(timeseries)
            if not spatial.empty:
                spatial_parts.append(spatial)
            if len(timeseries_parts) >= compact_every:
                timeseries_parts = _compact_sum_parts(
                    timeseries_parts,
                    ["model_name", "region", "period_start", "period_end", "month"],
                )
            if len(spatial_parts) >= compact_every:
                spatial_parts = _compact_sum_parts(
                    spatial_parts,
                    ["model_name", "region", "year", "month", LAT_COL, LON_COL],
                )

    if not calibration_parts:
        raise RuntimeError("No full-grid diagnostic prediction chunks were generated.")

    calibration_raw = _compact_sum_parts(calibration_parts, ["month"])[0]
    timeseries_raw = (
        _compact_sum_parts(
            timeseries_parts,
            ["model_name", "region", "period_start", "period_end", "month"],
        )[0]
        if timeseries_parts
        else pd.DataFrame()
    )
    spatial_raw = (
        _compact_sum_parts(
            spatial_parts,
            ["model_name", "region", "year", "month", LAT_COL, LON_COL],
        )[0]
        if spatial_parts
        else pd.DataFrame()
    )

    calibration, calibration_metadata = _calibration_scales_from_monthly(
        calibration_raw,
        enabled=config.recalibrate_on_test,
    )
    timeseries = _finalize_full_grid_timeseries(timeseries_raw, calibration)
    spatial, spatial_summary = _finalize_full_grid_spatial(
        spatial_raw,
        calibration,
        regions=regions,
        grid_resolution=float(config.grid_resolution or 0.1),
        ground_truth_smoothing_sigma_cells=config.ground_truth_smoothing_sigma_cells,
    )
    generation_metadata = {
        "enabled": True,
        "model_path": str(model_path),
        "feature_schema_path": str(schema_path),
        "feature_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "chunks": chunk_count,
        "cached_aggregate_chunks": cached_chunk_count,
        "generated_grid_rows": generated_rows,
        "scored_feature_rows": feature_rows,
        "row_level_prediction_file": None,
        "aggregate_cache_dir": str(output_dir / "full_grid_aggregate_cache"),
        "approximation": {
            "tag": approximation_tag,
            "sample_prediction_days_per_month": config.sample_prediction_days_per_month,
            "months_per_feature_chunk": max(1, int(config.months_per_feature_chunk or 1)),
            "exact_observed_labels": bool(
                config.sample_prediction_days_per_month and int(config.sample_prediction_days_per_month) > 0
            ),
            "sampled_chunks": sample_metadata,
        },
        "generation_scopes": [
            {
                "name": scope_name,
                "regions": [str(getattr(region, "name", region)) for region in scope_regions],
                "coordinate_bounds": scope_bounds,
            }
            for scope_name, scope_regions, scope_bounds in scopes
        ],
        "notes": "Generated from true land-grid chunks and aggregated directly; sparse prediction interpolation is not used.",
    }
    return timeseries, spatial, spatial_summary, calibration, {**calibration_metadata, "generation": generation_metadata}


def _period_bounds(dates: pd.Series, frequency: str) -> tuple[pd.Series, pd.Series]:
    normalized = pd.to_datetime(dates).dt.normalize()
    frequency = frequency.lower()
    if frequency in {"day", "daily", "d"}:
        return normalized, normalized
    if frequency in {"week", "weekly", "w"}:
        period = normalized.dt.to_period("W-SUN")
    elif frequency in {"month", "monthly", "m"}:
        period = normalized.dt.to_period("M")
    else:
        raise ValueError("prediction_diagnostics_time_frequency must be day, week, or month.")
    start = period.apply(lambda value: value.start_time).dt.normalize()
    end = period.apply(lambda value: value.end_time).dt.normalize()
    return start, end


def _infer_grid_resolution(frame: pd.DataFrame, fallback: float = 0.1) -> float:
    values: list[float] = []
    for column in [LAT_COL, LON_COL]:
        unique = np.sort(pd.to_numeric(frame[column], errors="coerce").dropna().unique())
        if len(unique) < 2:
            continue
        diffs = np.diff(unique)
        diffs = diffs[diffs > 1e-9]
        if len(diffs):
            values.append(float(np.nanmedian(diffs)))
    if not values:
        return fallback
    resolution = min(values)
    if not math.isfinite(resolution) or resolution <= 0:
        return fallback
    return float(resolution)


def build_timeseries_table(
    frame: pd.DataFrame,
    *,
    regions: Iterable[object],
    frequency: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    work = _weighted_counts(frame)
    start, end = _period_bounds(work[DATE_COL], frequency)
    work["period_start"] = start
    work["period_end"] = end

    for region in regions:
        mask = region.mask(work)
        region_frame = work.loc[mask]
        if region_frame.empty:
            continue
        grouped = (
            region_frame.groupby(["model_name", "period_start", "period_end"], observed=True, dropna=False)
            .agg(
                expected_fire_positive_grid_cells=("expected_fire_positive_grid_cells", "sum"),
                observed_fire_positive_grid_cells=("observed_fire_positive_grid_cells", "sum"),
            )
            .reset_index()
        )
        grouped.insert(1, "region", region.name)
        grouped["error_fire_positive_grid_cells"] = (
            grouped["expected_fire_positive_grid_cells"] - grouped["observed_fire_positive_grid_cells"]
        )
        grouped["absolute_error_fire_positive_grid_cells"] = grouped["error_fire_positive_grid_cells"].abs()
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["period_start"] = pd.to_datetime(out["period_start"]).dt.date.astype(str)
    out["period_end"] = pd.to_datetime(out["period_end"]).dt.date.astype(str)
    return out[
        [
            "model_name",
            "region",
            "period_start",
            "period_end",
            "expected_fire_positive_grid_cells",
            "observed_fire_positive_grid_cells",
            "error_fire_positive_grid_cells",
            "absolute_error_fire_positive_grid_cells",
        ]
    ].copy()


def build_spatial_error_tables(
    frame: pd.DataFrame,
    *,
    regions: Iterable[object],
    grid_resolution: float,
    ground_truth_smoothing_sigma_cells: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    work = _weighted_counts(frame)

    for region in regions:
        region_frame = work.loc[region.mask(work)].copy()
        if region_frame.empty:
            continue
        years = sorted(int(year) for year in region_frame["year"].dropna().unique())
        if not years:
            continue

        groups: list[tuple[str, pd.DataFrame, int]] = [
            (f"{min(years)}-{max(years)} mean", region_frame, len(years))
        ]
        groups.extend((str(year), region_frame[region_frame["year"].eq(year)], 1) for year in years)

        for period, period_frame, divisor in groups:
            if period_frame.empty:
                continue
            spatial = (
                period_frame.groupby(["model_name", LAT_COL, LON_COL], observed=True, dropna=False)
                .agg(
                    weighted_support=(WEIGHT_COL, "sum"),
                    expected_fire_positive_grid_cells=("expected_fire_positive_grid_cells", "sum"),
                    observed_fire_positive_grid_cells=("observed_fire_positive_grid_cells", "sum"),
                )
                .reset_index()
            )
            if divisor > 1:
                spatial["weighted_support"] /= float(divisor)
                spatial["expected_fire_positive_grid_cells"] /= float(divisor)
                spatial["observed_fire_positive_grid_cells"] /= float(divisor)
            spatial.insert(1, "region", region.name)
            spatial.insert(2, "period", period)
            support = pd.to_numeric(spatial["weighted_support"], errors="coerce").to_numpy(dtype=float)
            spatial["predicted_risk"] = np.divide(
                spatial["expected_fire_positive_grid_cells"].to_numpy(dtype=float),
                support,
                out=np.zeros(len(spatial), dtype=float),
                where=support > 0,
            )
            spatial["observed_risk"] = np.divide(
                spatial["observed_fire_positive_grid_cells"].to_numpy(dtype=float),
                support,
                out=np.zeros(len(spatial), dtype=float),
                where=support > 0,
            )
            spatial = _add_smoothed_observed_risk(
                spatial,
                region=region,
                grid_resolution=grid_resolution,
                sigma_cells=ground_truth_smoothing_sigma_cells,
            )
            spatial["error_fire_positive_grid_cells"] = (
                spatial["expected_fire_positive_grid_cells"] - spatial["observed_fire_positive_grid_cells"]
            )
            spatial["risk_error_smoothed_observed"] = spatial["predicted_risk"] - spatial["smoothed_observed_risk"]
            cell_rows.append(
                spatial[
                    [
                        "model_name",
                        "region",
                        "period",
                        LAT_COL,
                        LON_COL,
                        "weighted_support",
                        "predicted_risk",
                        "observed_risk",
                        "smoothed_observed_risk",
                        "risk_error_smoothed_observed",
                        "expected_fire_positive_grid_cells",
                        "observed_fire_positive_grid_cells",
                        "error_fire_positive_grid_cells",
                    ]
                ].copy()
            )
            errors = spatial["error_fire_positive_grid_cells"].to_numpy(dtype=float)
            risk_errors = spatial["risk_error_smoothed_observed"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "model_name": str(spatial["model_name"].iloc[0]),
                    "region": region.name,
                    "period": period,
                    "cells": int(len(spatial)),
                    "mean_predicted_risk": float(np.nanmean(spatial["predicted_risk"])),
                    "mean_smoothed_observed_risk": float(np.nanmean(spatial["smoothed_observed_risk"])),
                    "mean_risk_error": float(np.nanmean(risk_errors)) if len(risk_errors) else math.nan,
                    "mean_abs_risk_error": float(np.nanmean(np.abs(risk_errors))) if len(risk_errors) else math.nan,
                    "expected_fire_positive_grid_cells": float(spatial["expected_fire_positive_grid_cells"].sum()),
                    "observed_fire_positive_grid_cells": float(spatial["observed_fire_positive_grid_cells"].sum()),
                    "bias_fire_positive_grid_cells": float(np.sum(errors)),
                    "mean_abs_error_per_cell": float(np.mean(np.abs(errors))) if len(errors) else math.nan,
                }
            )

    cells = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return cells, summary


def _nan_gaussian_filter(values: np.ndarray, *, sigma: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if sigma <= 0:
        return arr.copy()
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.full_like(arr, np.nan, dtype=float)
    filled = np.where(valid, arr, 0.0)
    weights = valid.astype(float)
    smooth_values = gaussian_filter(filled, sigma=float(sigma), mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=float(sigma), mode="nearest")
    out = np.divide(
        smooth_values,
        smooth_weights,
        out=np.full_like(smooth_values, np.nan, dtype=float),
        where=smooth_weights > 1e-9,
    )
    return out


def _add_smoothed_observed_risk(
    spatial: pd.DataFrame,
    *,
    region: object,
    grid_resolution: float,
    sigma_cells: float,
) -> pd.DataFrame:
    if spatial.empty:
        spatial["smoothed_observed_risk"] = []
        return spatial
    extent = region.extent(spatial)
    lon_coords, lat_coords, observed_grid = spatial_grid(
        spatial,
        value_col="observed_risk",
        extent=extent,
        resolution=grid_resolution,
    )
    smoothed = _nan_gaussian_filter(observed_grid, sigma=float(sigma_cells))
    lat_idx = np.rint(
        (pd.to_numeric(spatial[LAT_COL], errors="coerce").to_numpy(dtype=float) - float(lat_coords[0]))
        / float(grid_resolution)
    ).astype(int)
    lon_idx = np.rint(
        (pd.to_numeric(spatial[LON_COL], errors="coerce").to_numpy(dtype=float) - float(lon_coords[0]))
        / float(grid_resolution)
    ).astype(int)
    values = np.full(len(spatial), np.nan, dtype=float)
    valid = (
        (lat_idx >= 0)
        & (lat_idx < smoothed.shape[0])
        & (lon_idx >= 0)
        & (lon_idx < smoothed.shape[1])
    )
    values[valid] = smoothed[lat_idx[valid], lon_idx[valid]]
    out = spatial.copy()
    out["smoothed_observed_risk"] = values
    return out


def _error_formatter(value: float, _pos: int) -> str:
    if not math.isfinite(float(value)):
        return ""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _risk_formatter(value: float, _pos: int) -> str:
    if not math.isfinite(float(value)):
        return ""
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _risk_residual_colormap(name: str):
    if str(name).lower() in {"risk_residual", "diagnostic_risk_residual"}:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "risk_residual",
            [
                (0.00, "#08306b"),
                (0.32, "#4292c6"),
                (0.50, "#b8e186"),
                (0.68, "#fdae61"),
                (1.00, "#7f0026"),
            ],
        )
    else:
        cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#9ca3af")
    return cmap


def _bilinear_interpolate_grid(
    lon_coords: np.ndarray,
    lat_coords: np.ndarray,
    values: np.ndarray,
    *,
    target_resolution: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    if target_resolution is None:
        return lon_coords, lat_coords, values, None
    target = float(target_resolution)
    if not math.isfinite(target) or target <= 0:
        return lon_coords, lat_coords, values, None
    if len(lon_coords) < 2 or len(lat_coords) < 2:
        return lon_coords, lat_coords, values, None

    lon_min = float(np.nanmin(lon_coords))
    lon_max = float(np.nanmax(lon_coords))
    lat_min = float(np.nanmin(lat_coords))
    lat_max = float(np.nanmax(lat_coords))
    if not all(math.isfinite(value) for value in [lon_min, lon_max, lat_min, lat_max]):
        return lon_coords, lat_coords, values, None

    precision = max(0, int(math.ceil(-math.log10(target))) + 2) if target < 1 else 2
    interp_lon = np.round(np.arange(lon_min, lon_max + target / 2.0, target), precision)
    interp_lat = np.round(np.arange(lat_min, lat_max + target / 2.0, target), precision)
    if len(interp_lon) == len(lon_coords) and len(interp_lat) == len(lat_coords):
        if np.allclose(interp_lon, lon_coords) and np.allclose(interp_lat, lat_coords):
            return lon_coords, lat_coords, values, None

    interpolator = RegularGridInterpolator(
        (np.asarray(lat_coords, dtype=float), np.asarray(lon_coords, dtype=float)),
        np.asarray(values, dtype=float),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    interp_lon_mesh, interp_lat_mesh = np.meshgrid(interp_lon, interp_lat)
    points = np.column_stack([interp_lat_mesh.ravel(), interp_lon_mesh.ravel()])
    interp_values = interpolator(points).reshape(interp_lat_mesh.shape)
    return interp_lon, interp_lat, interp_values, target


def _period_label(period: str) -> str:
    text = str(period)
    if "mean" in text:
        return text.replace(" mean", " average")
    return text


def _plot_timeseries(
    table: pd.DataFrame,
    *,
    region: object,
    model_name: str,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    data = table[(table["region"].eq(region.name)) & (table["model_name"].eq(model_name))].copy()
    if data.empty:
        return []
    data["period_start"] = pd.to_datetime(data["period_start"])
    data = data.sort_values("period_start")

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = data["period_start"].to_numpy()
    observed = data["observed_fire_positive_grid_cells"].to_numpy(dtype=float)
    expected = data["expected_fire_positive_grid_cells"].to_numpy(dtype=float)
    ax.plot(
        x,
        observed,
        color="#111827",
        linewidth=1.7,
        label="Ground truth",
    )
    ax.plot(
        x,
        expected,
        color="#2563eb",
        linewidth=1.7,
        label="Predicted",
    )
    ax.fill_between(
        x,
        observed,
        expected,
        color="#94a3b8",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_title(f"{region.display_name}: Predicted vs Ground Truth", loc="left", fontsize=11, pad=8)
    ax.text(
        0.0,
        1.01,
        f"{model_name}; monthly weighted fire-positive cell-days",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
        va="bottom",
    )
    ax.set_ylabel("Weighted fire-positive cell-days")
    ax.set_xlabel("Month")
    ax.grid(color="#d1d5db", linewidth=0.45, alpha=0.75)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    written: list[Path] = []
    stem = f"timeseries_{safe_slug(model_name)}_{safe_slug(region.name)}"
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        subdir = "png" if fmt == "png" else "pdf" if fmt == "pdf" else fmt
        out = output_dir / "plots" / subdir / f"{stem}.{fmt}"
        ensure_dir(out.parent)
        fig.savefig(out, dpi=dpi if fmt != "pdf" else None, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def _plot_error_map(
    spatial: pd.DataFrame,
    *,
    region: object,
    model_name: str,
    period: str,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    grid_resolution: float,
    plot_interpolation_resolution: float | None,
    ground_truth_smoothing_sigma_cells: float,
    colormap: str,
    world: object,
) -> list[Path]:
    data = spatial[
        (spatial["region"].eq(region.name))
        & (spatial["model_name"].eq(model_name))
        & (spatial["period"].eq(period))
    ].copy()
    if data.empty:
        return []

    extent = region.extent(data)
    lon_min, lon_max, lat_min, lat_max = extent
    lon_span = max(lon_max - lon_min, 1e-6)
    lat_span = max(lat_max - lat_min, 1e-6)
    mid_lat = (lat_min + lat_max) / 2.0
    display_aspect = (lat_span / lon_span) / max(math.cos(math.radians(mid_lat)), 0.2)
    figure_width = 6.7
    figure_height = min(6.4, max(3.8, 0.85 + 5.4 * display_aspect))

    lon_coords, lat_coords, predicted_grid = spatial_grid(
        data,
        value_col="predicted_risk",
        extent=extent,
        resolution=grid_resolution,
    )
    _, _, observed_grid = spatial_grid(
        data,
        value_col="observed_risk",
        extent=extent,
        resolution=grid_resolution,
    )
    smoothed_observed = _nan_gaussian_filter(
        observed_grid,
        sigma=float(ground_truth_smoothing_sigma_cells),
    )
    source_lon_coords = lon_coords
    source_lat_coords = lat_coords
    lon_coords, lat_coords, predicted_grid, interpolation_resolution = _bilinear_interpolate_grid(
        source_lon_coords,
        source_lat_coords,
        predicted_grid,
        target_resolution=plot_interpolation_resolution,
    )
    _, _, smoothed_observed, _ = _bilinear_interpolate_grid(
        source_lon_coords,
        source_lat_coords,
        smoothed_observed,
        target_resolution=plot_interpolation_resolution,
    )
    plot_resolution = float(interpolation_resolution or grid_resolution)
    error_grid = predicted_grid - smoothed_observed
    finite = error_grid[np.isfinite(error_grid)]
    if len(finite):
        limit = float(np.nanpercentile(np.abs(finite), 99.0))
    else:
        limit = 0.01
    if not math.isfinite(limit) or limit <= 0:
        limit = 0.01

    cmap = _risk_residual_colormap(colormap)
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image_extent = [
        float(lon_coords[0]) - plot_resolution / 2.0,
        float(lon_coords[-1]) + plot_resolution / 2.0,
        float(lat_coords[0]) - plot_resolution / 2.0,
        float(lat_coords[-1]) + plot_resolution / 2.0,
    ]
    mesh = ax.imshow(
        np.ma.masked_invalid(error_grid),
        extent=image_extent,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="none",
        rasterized=True,
        zorder=2,
    )
    ax.set_facecolor("#9ca3af")
    plot_boundaries(ax, world, extent)

    lon_pad = max((lon_max - lon_min) * 0.025, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.025, 0.1)
    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
    ax.set_aspect(1.0 / max(math.cos(math.radians(mid_lat)), 0.2), adjustable="box")
    ax.set_title(f"{region.display_name}: Risk residual, {_period_label(period)}", loc="left", fontsize=11, pad=7)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.tick_params(labelsize=8)
    ax.grid(color="#d1d5db", linewidth=0.35, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.6)

    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.025, extend="both")
    colorbar.set_label("Predicted risk minus observed", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(_risk_formatter))
    fig.tight_layout()

    period_slug = "mean_test_years" if "mean" in period else safe_slug(period)
    stem = f"error_map_{safe_slug(model_name)}_{safe_slug(region.name)}_{period_slug}"
    written: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        subdir = "png" if fmt == "png" else "pdf" if fmt == "pdf" else fmt
        out = output_dir / "plots" / subdir / f"{stem}.{fmt}"
        ensure_dir(out.parent)
        fig.savefig(out, dpi=dpi if fmt != "pdf" else None, bbox_inches="tight")
        written.append(out)
    plt.close(fig)
    return written


def _round_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].round(4)
    return out


def _write_tables(output_dir: Path, timeseries: pd.DataFrame, spatial: pd.DataFrame, summary: pd.DataFrame) -> dict[str, str]:
    table_dir = output_dir / "tables"
    ensure_dir(table_dir)
    paths = {
        "timeseries_counts": table_dir / "timeseries_counts.csv",
        "spatial_error_cells": table_dir / "spatial_error_cells.csv",
        "spatial_error_summary": table_dir / "spatial_error_summary.csv",
    }
    spatial_cols = [
        "model_name",
        "region",
        "period",
        LAT_COL,
        LON_COL,
        "predicted_risk",
        "smoothed_observed_risk",
        "risk_error_smoothed_observed",
    ]
    summary_cols = [
        "model_name",
        "region",
        "period",
        "cells",
        "mean_predicted_risk",
        "mean_smoothed_observed_risk",
        "mean_risk_error",
        "mean_abs_risk_error",
    ]
    _round_table(timeseries).to_csv(paths["timeseries_counts"], index=False)
    _round_table(spatial.loc[:, [col for col in spatial_cols if col in spatial.columns]]).to_csv(
        paths["spatial_error_cells"],
        index=False,
    )
    _round_table(summary.loc[:, [col for col in summary_cols if col in summary.columns]]).to_csv(
        paths["spatial_error_summary"],
        index=False,
    )
    return {key: str(path) for key, path in paths.items()}


def _cleanup_old_outputs(output_dir: Path) -> None:
    for table in [
        "timeseries_counts.csv",
        "spatial_error_cells.csv",
        "spatial_error_summary.csv",
    ]:
        path = output_dir / "tables" / table
        if path.exists():
            path.unlink()
    for plot_dir in [output_dir / "plots" / "png", output_dir / "plots" / "pdf"]:
        if not plot_dir.exists():
            continue
        for pattern in ["timeseries_*", "error_map_*"]:
            for path in plot_dir.glob(pattern):
                if path.is_file():
                    path.unlink()


def run_prediction_diagnostics(config: PredictionDiagnosticConfig) -> dict[str, object]:
    output_dir = config.output_dir or default_output_dir(config.results_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "tables")
    _cleanup_old_outputs(output_dir)

    only = set(config.regions) if config.regions else None
    regions = load_regions(config.regions_file, include_global=config.include_global, only=only)
    source = config.source.lower()
    resolved_model = resolve_model(config.results_dir, config.model, source)
    grid_resolution = float(config.grid_resolution or 0.1)
    formats = [fmt.strip().lower().lstrip(".") for fmt in config.formats if fmt.strip()]
    world = load_world_boundaries(config.country_shapes)

    calibration_table = pd.DataFrame()
    generation_metadata: dict[str, object] = {"enabled": False}
    prediction_files: list[Path] = []
    predictions: pd.DataFrame | None = None

    if config.generate_full_grid_predictions:
        timeseries, spatial, spatial_summary, calibration_table, calibration_metadata = (
            generate_full_grid_prediction_diagnostic_tables(
                config,
                regions=regions,
                resolved_model=resolved_model,
                output_dir=output_dir,
            )
        )
        generation_metadata = dict(calibration_metadata.pop("generation", {"enabled": True}))
    else:
        prediction_files = find_prediction_files_for_model(config.results_dir, resolved_model, source)
        frames: list[pd.DataFrame] = []
        for path in prediction_files:
            frame = read_prediction_columns(path, config.prob_col)
            frame["prediction_path"] = str(path)
            frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"No prediction files found for diagnostic model {resolved_model!r}.")
        predictions = pd.concat(frames, ignore_index=True)
        if config.require_full_grid_predictions:
            _assert_full_grid_predictions(predictions, prediction_files)

        grid_resolution = float(config.grid_resolution or _infer_grid_resolution(predictions))
        calibration_metadata: dict[str, object] = {"enabled": False}
        if config.recalibrate_on_test:
            predictions, calibration_metadata = recalibrate_probabilities_on_test(predictions)

        timeseries = build_timeseries_table(predictions, regions=regions, frequency=config.time_frequency)
        spatial, spatial_summary = build_spatial_error_tables(
            predictions,
            regions=regions,
            grid_resolution=float(grid_resolution),
            ground_truth_smoothing_sigma_cells=config.ground_truth_smoothing_sigma_cells,
        )
    table_paths = _write_tables(output_dir, timeseries, spatial, spatial_summary)
    if not calibration_table.empty:
        calibration_path = output_dir / "tables" / "full_grid_test_calibration_by_month.csv"
        _round_table(calibration_table).to_csv(calibration_path, index=False)
        table_paths["full_grid_test_calibration_by_month"] = str(calibration_path)

    written: list[Path] = []
    model_names = sorted(
        {
            *[str(name) for name in timeseries.get("model_name", pd.Series(dtype=object)).dropna().unique()],
            *[str(name) for name in spatial.get("model_name", pd.Series(dtype=object)).dropna().unique()],
        }
    )
    for model_name in model_names:
        for region in regions:
            written.extend(
                _plot_timeseries(
                    timeseries,
                    region=region,
                    model_name=model_name,
                    output_dir=output_dir,
                    formats=formats,
                    dpi=config.dpi,
                )
            )
            periods = spatial.loc[
                (spatial["model_name"].eq(model_name)) & (spatial["region"].eq(region.name)),
                "period",
            ].drop_duplicates()
            for period in periods:
                written.extend(
                    _plot_error_map(
                        spatial,
                        region=region,
                        model_name=model_name,
                        period=str(period),
                        output_dir=output_dir,
                        formats=formats,
                        dpi=config.dpi,
                        grid_resolution=float(grid_resolution),
                        plot_interpolation_resolution=config.plot_interpolation_resolution,
                        ground_truth_smoothing_sigma_cells=config.ground_truth_smoothing_sigma_cells,
                        colormap=config.error_colormap,
                        world=world,
                    )
                )

    manifest = {
        "output_dir": str(output_dir),
        "prediction_files": [str(path) for path in prediction_files],
        "requested_model": config.model,
        "resolved_model": resolved_model,
        "source": source,
        "regions": [region.name for region in regions],
        "time_frequency": config.time_frequency,
        "grid_resolution": float(grid_resolution),
        "plot_interpolation_resolution": (
            float(config.plot_interpolation_resolution)
            if config.plot_interpolation_resolution is not None
            else None
        ),
        "ground_truth_smoothing_sigma_cells": float(config.ground_truth_smoothing_sigma_cells),
        "require_full_grid_predictions": bool(config.require_full_grid_predictions),
        "generated_full_grid_predictions": generation_metadata,
        "test_recalibration": calibration_metadata,
        "tables": table_paths,
        "plots": [str(path) for path in written],
        "plot_count": len(written),
    }
    write_json(output_dir / "manifest.json", manifest)
    smoothing_label = (
        "disabled"
        if float(config.ground_truth_smoothing_sigma_cells) <= 0
        else f"{config.ground_truth_smoothing_sigma_cells:g} grid cells"
    )
    (output_dir / "README.md").write_text(
        "# Prediction Diagnostics\n\n"
        "Aggregates saved test predictions into regional time series and spatial error maps.\n\n"
        f"- Source: `{source}`\n"
        f"- Requested model: `{config.model}`\n"
        f"- Resolved model: `{resolved_model}`\n"
        f"- Time frequency: `{config.time_frequency}`\n"
        f"- Plot interpolation resolution: `{config.plot_interpolation_resolution if config.plot_interpolation_resolution is not None else 'disabled'}`\n"
        f"- Ground-truth risk smoothing: `{smoothing_label}`\n"
        f"- Test-set risk recalibration: `{'enabled' if config.recalibrate_on_test else 'disabled'}`\n"
        f"- Full-grid prediction generation: `{'enabled' if config.generate_full_grid_predictions else 'disabled'}`\n"
        f"- Missing grid-cell fill/interpolation: `disabled`; plot-only bilinear interpolation may be configured\n"
        f"- Plot files: `{len(written)}`\n",
        encoding="utf-8",
    )
    return manifest
