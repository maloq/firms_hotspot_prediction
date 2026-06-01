from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression

try:
    import eccodes
except Exception as exc:  # pragma: no cover - exercised only without eccodes
    eccodes = None
    ECCODES_IMPORT_ERROR = exc
else:
    ECCODES_IMPORT_ERROR = None

from .config import EvaluationConfig
from .deployment_grid import (
    _filter_frame_to_bounds,
    _iter_cell_date_row_blocks,
    _sample_cross_cells_dates,
    build_fire_label_frame,
    coordinate_bounds_for_grid,
    countries_for_grid,
    generate_grid_cells_for_geometry,
    load_country_geometries,
)
from .full_grid_evaluation import (
    RISK_TOP_FRACTIONS,
    _bootstrap_array_metric_errors,
    _neighborhood_positive_labels,
    _risk_concentration_values,
    _spatial_metric_row,
    _stable_seed,
    safe_slug,
)
from .probability_metrics import (
    max_weighted_f1,
    weighted_average_precision,
    weighted_prevalence,
    weighted_roc_auc,
)
from .tabular import load_regions, load_yaml


EVALUATION_TYPE = "fire_weather_index_full_grid_ranking"
DEFAULT_OUTPUT_SUBDIR = "fire_weather_index_ranking"
LOGISTIC_MODEL_TYPE = "FWI logistic regression"
LOGISTIC_SCORE_TRANSFORM = "weighted_logistic_regression_probability"
PREDICTION_COLUMNS = [
    "datetime",
    "lat_rounded",
    "lon_rounded",
    "country",
    "month",
    "year",
    "is_fire",
    "eval_weight",
    "raw_score",
]


def _logistic_model_name(train_year: int) -> str:
    return f"FWI Logistic Regression (train {int(train_year)})"


@dataclass(frozen=True)
class FWIInventory:
    table: pd.DataFrame

    @property
    def variables(self) -> list[str]:
        if self.table.empty:
            return []
        return sorted(self.table["variable"].astype(str).unique())

    def available_dates(
        self,
        variables: Iterable[str] | None = None,
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        table = self.table
        if variables is not None:
            wanted = {str(value) for value in variables}
            table = table.loc[table["variable"].astype(str).isin(wanted)]
        if start_date is not None:
            table = table.loc[table["date"] >= pd.to_datetime(start_date).normalize()]
        if end_date is not None:
            table = table.loc[table["date"] <= pd.to_datetime(end_date).normalize()]
        return pd.DatetimeIndex(sorted(table["date"].drop_duplicates()))

    def complete_case_dates(
        self,
        variables: Iterable[str],
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        date_sets: list[set[pd.Timestamp]] = []
        for variable in variables:
            dates = self.available_dates([variable], start_date=start_date, end_date=end_date)
            date_sets.append({pd.Timestamp(value).normalize() for value in dates})
        if not date_sets:
            return pd.DatetimeIndex([])
        common = set.intersection(*date_sets)
        return pd.DatetimeIndex(sorted(common))

    def variables_available_in_year(
        self,
        variables: Iterable[str],
        year: int,
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> list[str]:
        year_start = pd.Timestamp(year=int(year), month=1, day=1)
        year_end = pd.Timestamp(year=int(year), month=12, day=31)
        if start_date is not None:
            year_start = max(year_start, pd.to_datetime(start_date).normalize())
        if end_date is not None:
            year_end = min(year_end, pd.to_datetime(end_date).normalize())
        if year_end < year_start:
            return []
        selected = {str(variable) for variable in variables}
        table = self.table.loc[
            self.table["variable"].astype(str).isin(selected)
            & self.table["date"].between(year_start, year_end)
        ]
        return sorted(table["variable"].astype(str).unique())

    def date_to_path(self, variables: Iterable[str]) -> dict[str, dict[pd.Timestamp, Path]]:
        wanted = {str(value) for value in variables}
        table = self.table.loc[self.table["variable"].astype(str).isin(wanted)].copy()
        mapping: dict[str, dict[pd.Timestamp, Path]] = {variable: {} for variable in wanted}
        for row in table.sort_values(["variable", "date", "path"]).itertuples(index=False):
            variable = str(row.variable)
            date = pd.Timestamp(row.date).normalize()
            mapping.setdefault(variable, {}).setdefault(date, Path(row.path))
        return mapping

    def availability_summary(
        self,
        variables: Iterable[str],
        *,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for variable in variables:
            table = self.table.loc[self.table["variable"].astype(str).eq(str(variable))]
            if start_date is not None:
                table = table.loc[table["date"] >= pd.to_datetime(start_date).normalize()]
            if end_date is not None:
                table = table.loc[table["date"] <= pd.to_datetime(end_date).normalize()]
            if table.empty:
                continue
            for year, group in table.groupby(table["date"].dt.year, observed=True):
                dates = pd.DatetimeIndex(sorted(group["date"].drop_duplicates()))
                months = sorted(dates.strftime("%Y-%m").unique())
                rows.append(
                    {
                        "index": variable,
                        "model_name": f"FWI {variable}",
                        "year": int(year),
                        "available_start_date": dates.min().date().isoformat(),
                        "available_end_date": dates.max().date().isoformat(),
                        "available_days": int(len(dates)),
                        "available_months": ",".join(months),
                        "source_files": ",".join(sorted({Path(value).name for value in group["path"]})),
                        "grid_types": ",".join(sorted(group["grid_type"].astype(str).unique())),
                    }
                )
        return pd.DataFrame(rows)


class FWIDataset:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.dataset = xr.open_dataset(
                self.path,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )
        self.variables = set(str(name) for name in self.dataset.data_vars)
        self.time = pd.DatetimeIndex(pd.to_datetime(self.dataset["time"].values)).normalize()
        self.time_lookup = {pd.Timestamp(value).normalize(): idx for idx, value in enumerate(self.time)}
        self._tree: cKDTree | None = None
        self._tree_points: np.ndarray | None = None

    def close(self) -> None:
        self.dataset.close()

    @staticmethod
    def _unit_sphere(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_rad = np.deg2rad(np.asarray(lat, dtype=float))
        lon_rad = np.deg2rad(np.mod(np.asarray(lon, dtype=float), 360.0))
        cos_lat = np.cos(lat_rad)
        return np.column_stack(
            [
                cos_lat * np.cos(lon_rad),
                cos_lat * np.sin(lon_rad),
                np.sin(lat_rad),
            ]
        )

    def _ensure_tree(self) -> cKDTree:
        if self._tree is None:
            lat = np.asarray(self.dataset["latitude"].values, dtype=float).reshape(-1)
            lon = np.asarray(self.dataset["longitude"].values, dtype=float).reshape(-1)
            self._tree_points = self._unit_sphere(lat, lon)
            self._tree = cKDTree(self._tree_points)
        return self._tree

    def nearest_flat_indices(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        if "values" in self.dataset.dims:
            _, idx = self._ensure_tree().query(self._unit_sphere(lat, lon), k=1)
            return np.asarray(idx, dtype=np.int64)

        lats = np.asarray(self.dataset["latitude"].values, dtype=float)
        lons = np.asarray(self.dataset["longitude"].values, dtype=float)
        if len(lats) < 2 or len(lons) < 2:
            raise ValueError(f"Cannot infer regular grid spacing for {self.path}.")
        lat_step = float(abs(np.median(np.diff(lats))))
        lon_step = float(abs(np.median(np.diff(lons))))
        if lats[0] > lats[-1]:
            lat_idx = np.rint((float(lats[0]) - np.asarray(lat, dtype=float)) / lat_step)
        else:
            lat_idx = np.rint((np.asarray(lat, dtype=float) - float(lats[0])) / lat_step)
        lon_mod = np.mod(np.asarray(lon, dtype=float), 360.0)
        lon_idx = np.rint((lon_mod - float(lons[0])) / lon_step)
        lat_idx = np.clip(lat_idx.astype(np.int64), 0, len(lats) - 1)
        lon_idx = np.mod(lon_idx.astype(np.int64), len(lons))
        return lat_idx * len(lons) + lon_idx

    def point_values(self, variable: str, date: pd.Timestamp, flat_idx: np.ndarray) -> np.ndarray:
        if variable not in self.variables:
            return np.full(len(flat_idx), np.nan, dtype=np.float32)
        time_idx = self.time_lookup.get(pd.Timestamp(date).normalize())
        if time_idx is None:
            return np.full(len(flat_idx), np.nan, dtype=np.float32)
        values = np.asarray(self.dataset[variable].isel(time=time_idx).values, dtype=np.float32).reshape(-1)
        out = values[np.asarray(flat_idx, dtype=np.int64)]
        return np.asarray(out, dtype=np.float32)


class FWIDatasetCache:
    def __init__(self, max_open: int = 2) -> None:
        self.max_open = max(1, int(max_open))
        self._cache: OrderedDict[Path, FWIDataset] = OrderedDict()

    def get(self, path: Path) -> FWIDataset:
        key = Path(path)
        if key in self._cache:
            dataset = self._cache.pop(key)
            self._cache[key] = dataset
            return dataset
        while len(self._cache) >= self.max_open:
            _, old = self._cache.popitem(last=False)
            old.close()
        dataset = FWIDataset(key)
        self._cache[key] = dataset
        return dataset

    def close(self) -> None:
        for dataset in self._cache.values():
            dataset.close()
        self._cache.clear()


class PredictionPartStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.counts: dict[str, int] = {}
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, variable: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        slug = safe_slug(variable)
        part_dir = self.root / slug
        part_dir.mkdir(parents=True, exist_ok=True)
        part_idx = self.counts.get(variable, 0)
        self.counts[variable] = part_idx + 1
        frame.to_parquet(part_dir / f"part-{part_idx:06d}.parquet", index=False)

    def read(self, variable: str) -> pd.DataFrame:
        files = sorted((self.root / safe_slug(variable)).glob("part-*.parquet"))
        if not files:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)

    def persist_to(self, destination: Path) -> None:
        destination = Path(destination)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.root, destination)

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)


def scan_fwi_inventory(fwi_dir: Path) -> FWIInventory:
    if eccodes is None:
        raise ImportError(f"eccodes is required to scan GRIB metadata: {ECCODES_IMPORT_ERROR}")
    paths = sorted(Path(fwi_dir).glob("*.grib"))
    if not paths:
        raise FileNotFoundError(f"No .grib files found under {fwi_dir}.")

    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("rb") as handle:
            while True:
                gid = eccodes.codes_grib_new_from_file(handle)
                if gid is None:
                    break
                try:
                    variable = str(eccodes.codes_get(gid, "shortName"))
                    date = pd.to_datetime(str(int(eccodes.codes_get(gid, "dataDate"))), format="%Y%m%d")
                    grid_type = str(eccodes.codes_get(gid, "gridType"))
                finally:
                    eccodes.codes_release(gid)
                rows.append(
                    {
                        "path": str(path),
                        "file": path.name,
                        "variable": variable,
                        "date": date.normalize(),
                        "year": int(date.year),
                        "month": int(date.month),
                        "grid_type": grid_type,
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.drop_duplicates(["path", "variable", "date"]).sort_values(
            ["variable", "date", "path"]
        )
    return FWIInventory(table=table.reset_index(drop=True))


def _selected_variables(inventory: FWIInventory, configured: Iterable[str] | None) -> list[str]:
    if configured is None:
        return inventory.variables
    selected = [str(value).strip() for value in configured if str(value).strip()]
    available = set(inventory.variables)
    missing = [value for value in selected if value not in available]
    if missing:
        raise ValueError(f"Requested FWI variables are not available: {missing}. Available: {sorted(available)}")
    return selected


def _date_bounds(config: Any) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = getattr(config, "test_start_date", None)
    end = getattr(config, "test_end_date", None)
    return (
        pd.to_datetime(start).normalize() if start else None,
        pd.to_datetime(end).normalize() if end else None,
    )


def iter_available_deployment_chunks(
    *,
    config: Any,
    feature_config: dict[str, Any],
    target_config: dict[str, Any],
    available_dates: pd.DatetimeIndex,
) -> Iterable[Any]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(available_dates).normalize().unique()))
    if len(dates) == 0:
        return
    start_date = pd.Timestamp(dates.min()).normalize()
    end_date = pd.Timestamp(dates.max()).normalize()
    countries = countries_for_grid(config, feature_config, target_config)
    resolution = float(getattr(config, "deployment_grid_resolution", None) or 0.1)
    shapes_path = feature_config.get("country_shapes_path") or target_config.get("country_shapes_path") or "data/countries"
    coordinate_bounds = coordinate_bounds_for_grid(config, feature_config, target_config)
    labels = build_fire_label_frame(
        feature_config=feature_config,
        target_config=target_config,
        countries=countries,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
    )
    labels = _filter_frame_to_bounds(labels, coordinate_bounds, resolution=resolution)
    geometries = load_country_geometries(shapes_path, countries)
    max_rows = getattr(config, "max_grid_rows_per_chunk", None)
    max_rows = int(max_rows) if max_rows not in {None, ""} else None
    weighted = bool(getattr(config, "weighted_grid_sample", False)) or str(
        getattr(config, "full_grid_mode", "full_grid")
    ).lower() == "weighted_grid_sample"

    date_series = pd.Series(dates, index=dates)
    for country in countries:
        cells = generate_grid_cells_for_geometry(geometries[country], country=country, resolution=resolution)
        cells = _filter_frame_to_bounds(cells, coordinate_bounds, resolution=resolution)
        for period, period_values in date_series.groupby(date_series.dt.to_period("M")):
            month_dates = pd.DatetimeIndex(period_values.to_numpy())
            if weighted and getattr(config, "weighted_grid_sample_fraction", None) is not None:
                rows = _sample_cross_cells_dates(
                    cells,
                    month_dates,
                    labels,
                    country,
                    config=config,
                    split_name="test",
                    period=str(period),
                    resolution=resolution,
                )
                yield SimpleNamespace(split_name="test", country=country, period=str(period), rows=rows.reset_index(drop=True))
                continue
            for suffix, rows in _iter_cell_date_row_blocks(
                cells,
                month_dates,
                labels,
                country,
                max_rows,
                resolution=resolution,
            ):
                yield SimpleNamespace(
                    split_name="test",
                    country=country,
                    period=f"{period}{suffix}",
                    rows=rows.reset_index(drop=True),
                )


def _score_chunk(
    rows: pd.DataFrame,
    *,
    variables: list[str],
    date_to_path: dict[str, dict[pd.Timestamp, Path]],
    cache: FWIDatasetCache,
) -> dict[str, np.ndarray]:
    scores = {variable: np.full(len(rows), np.nan, dtype=np.float32) for variable in variables}
    if rows.empty:
        return scores

    dates = pd.to_datetime(rows["datetime"]).dt.normalize()
    grouped_indices = pd.Series(np.arange(len(rows), dtype=np.int64)).groupby(dates).indices
    flat_index_cache: dict[Path, np.ndarray] = {}
    lat = pd.to_numeric(rows["lat_rounded"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(rows["lon_rounded"], errors="coerce").to_numpy(dtype=float)

    for date, row_indices in grouped_indices.items():
        date_ts = pd.Timestamp(date).normalize()
        variables_by_path: dict[Path, list[str]] = {}
        for variable in variables:
            path = date_to_path.get(variable, {}).get(date_ts)
            if path is not None:
                variables_by_path.setdefault(path, []).append(variable)

        idx = np.asarray(row_indices, dtype=np.int64)
        for path, path_variables in variables_by_path.items():
            dataset = cache.get(path)
            if path not in flat_index_cache:
                flat_index_cache[path] = dataset.nearest_flat_indices(lat, lon)
            flat_idx = flat_index_cache[path][idx]
            for variable in path_variables:
                scores[variable][idx] = dataset.point_values(variable, date_ts, flat_idx)
    return scores


def _prediction_base(rows: pd.DataFrame) -> pd.DataFrame:
    base = rows[[col for col in PREDICTION_COLUMNS if col != "raw_score"]].copy()
    base["datetime"] = pd.to_datetime(base["datetime"]).dt.normalize()
    return base


def _write_scored_chunk(store: PredictionPartStore, rows: pd.DataFrame, scores: dict[str, np.ndarray]) -> None:
    base = _prediction_base(rows)
    for variable, score in scores.items():
        finite = np.isfinite(score)
        if not np.any(finite):
            continue
        frame = base.loc[finite].copy()
        frame["raw_score"] = np.asarray(score[finite], dtype=np.float32)
        store.add(variable, frame[PREDICTION_COLUMNS])


def _write_wide_scored_chunk(
    store: PredictionPartStore,
    rows: pd.DataFrame,
    scores: dict[str, np.ndarray],
    *,
    variables: list[str],
    key: str,
) -> None:
    if not variables or rows.empty:
        return
    mask = np.ones(len(rows), dtype=bool)
    for variable in variables:
        mask &= np.isfinite(scores.get(variable, np.full(len(rows), np.nan, dtype=np.float32)))
    if not np.any(mask):
        return
    frame = _prediction_base(rows).loc[mask].copy()
    for variable in variables:
        frame[variable] = np.asarray(scores[variable][mask], dtype=np.float32)
    store.add(key, frame)


def _normalize_scores(raw_score: pd.Series) -> tuple[np.ndarray, dict[str, float | None]]:
    raw = pd.to_numeric(raw_score, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(raw)
    if not finite.any():
        return np.full(len(raw), np.nan, dtype=float), {
            "raw_score_min": None,
            "raw_score_max": None,
            "raw_score_mean": None,
        }
    raw_min = float(np.nanmin(raw[finite]))
    raw_max = float(np.nanmax(raw[finite]))
    if raw_max > raw_min:
        score = (raw - raw_min) / (raw_max - raw_min)
    else:
        score = np.full(len(raw), 0.5, dtype=float)
    score[~finite] = np.nan
    return np.clip(score, 0.0, 1.0), {
        "raw_score_min": raw_min,
        "raw_score_max": raw_max,
        "raw_score_mean": float(np.nanmean(raw[finite])),
    }


def _probability_scores(raw_score: pd.Series) -> tuple[np.ndarray, dict[str, float | None]]:
    raw = pd.to_numeric(raw_score, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(raw)
    if not finite.any():
        return np.full(len(raw), np.nan, dtype=float), {
            "raw_score_min": None,
            "raw_score_max": None,
            "raw_score_mean": None,
        }
    score = np.clip(raw, 0.0, 1.0)
    score[~finite] = np.nan
    return score, {
        "raw_score_min": float(np.nanmin(raw[finite])),
        "raw_score_max": float(np.nanmax(raw[finite])),
        "raw_score_mean": float(np.nanmean(raw[finite])),
    }


def _available_months(frame: pd.DataFrame) -> str:
    months = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m").drop_duplicates().sort_values()
    return ",".join(months.astype(str))


def _ranking_metric_row(
    frame: pd.DataFrame,
    *,
    variable: str,
    model_name: str,
    model_type: str,
    feature_set: str,
    score_transform: str,
    region: str,
    region_display: str,
    split_label: str,
    test_period: str,
    score: np.ndarray,
    score_stats: dict[str, float | None],
    config: Any,
    seed: int,
) -> dict[str, Any]:
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    w = pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    finite = np.isfinite(score) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    y = y[finite]
    p = np.asarray(score, dtype=float)[finite]
    w = w[finite]
    f1 = max_weighted_f1(y, p, w)
    row = {
        "Model": model_name,
        "model": model_name,
        "model_name": model_name,
        "model_type": model_type,
        "index": variable,
        "Feature set": feature_set,
        "feature_set": feature_set,
        "Region": region_display,
        "region": region,
        "region_display": region_display,
        "split": split_label,
        "evaluation_type": EVALUATION_TYPE,
        "is_primary": False,
        "score_transform": score_transform,
        "test_universe": str(getattr(config, "deployment_grid_universe", "land_or_burnable")),
        "test_period": test_period,
        "available_months": _available_months(frame),
        "available_days": int(pd.to_datetime(frame["datetime"]).dt.normalize().nunique()),
        "support": int(len(y)),
        "weighted_support": float(np.sum(w)),
        "positives": float(np.sum(y * w)),
        "observed_prevalence": weighted_prevalence(y, w),
        "mean_normalized_score": float(np.average(p, weights=w)) if len(y) else None,
        "average_precision": weighted_average_precision(y, p, w),
        "roc_auc": weighted_roc_auc(y, p, w),
        **f1,
        **score_stats,
    }
    threshold = row.get("threshold_at_max_f1")
    raw_min = score_stats.get("raw_score_min")
    raw_max = score_stats.get("raw_score_max")
    if str(score_transform).startswith("minmax"):
        row["raw_threshold_at_max_f1"] = (
            float(raw_min) + float(threshold) * (float(raw_max) - float(raw_min))
            if threshold is not None and raw_min is not None and raw_max is not None
            else None
        )
    else:
        row["raw_threshold_at_max_f1"] = threshold
    row.update(
        _bootstrap_array_metric_errors(
            y,
            p,
            w,
            config=config,
            seed=seed,
            metric_names=["observed_prevalence", "average_precision", "roc_auc", "max_f1"],
        )
    )
    return row


def _regional_metric_rows(
    frame: pd.DataFrame,
    *,
    variable: str,
    model_name: str,
    model_type: str,
    feature_set: str,
    score_transform: str,
    regions: Iterable[Any] | None,
    score: np.ndarray,
    score_stats: dict[str, float | None],
    config: Any,
) -> pd.DataFrame:
    start = pd.to_datetime(frame["datetime"]).min().date().isoformat()
    end = pd.to_datetime(frame["datetime"]).max().date().isoformat()
    period = f"{start} to {end} (available FWI dates only)"
    rows = [
        _ranking_metric_row(
            frame,
            variable=variable,
            model_name=model_name,
            model_type=model_type,
            feature_set=feature_set,
            score_transform=score_transform,
            region="global",
            region_display="Global",
            split_label="test",
            test_period=period,
            score=score,
            score_stats=score_stats,
            config=config,
            seed=_stable_seed(model_name, 10),
        )
    ]
    if regions is not None and {"lat_rounded", "lon_rounded"}.issubset(frame.columns):
        for idx, region in enumerate(regions):
            mask = np.asarray(region.mask(frame), dtype=bool)
            if not mask.any():
                continue
            rows.append(
                _ranking_metric_row(
                    frame.loc[mask].reset_index(drop=True),
                    variable=variable,
                    model_name=model_name,
                    model_type=model_type,
                    feature_set=feature_set,
                    score_transform=score_transform,
                    region=region.name,
                    region_display=region.display_name,
                    split_label="test",
                    test_period=period,
                    score=score[mask],
                    score_stats=score_stats,
                    config=config,
                    seed=_stable_seed(f"{model_name}:{region.name}", 20 + idx),
                )
            )
    return pd.DataFrame(rows)


def _by_year_rows(
    frame: pd.DataFrame,
    *,
    variable: str,
    model_name: str,
    model_type: str,
    feature_set: str,
    score_transform: str,
    score: np.ndarray,
    score_stats: dict[str, float | None],
    config: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = pd.to_datetime(frame["datetime"]).dt.year
    for year in sorted(years.drop_duplicates()):
        mask = years.eq(year).to_numpy()
        group = frame.loc[mask].reset_index(drop=True)
        row = _ranking_metric_row(
            group,
            variable=variable,
            model_name=model_name,
            model_type=model_type,
            feature_set=feature_set,
            score_transform=score_transform,
            region="global",
            region_display="Global",
            split_label=f"test_{int(year)}",
            test_period=f"{int(year)} (available FWI dates only)",
            score=score[mask],
            score_stats=score_stats,
            config=config,
            seed=_stable_seed(f"{model_name}:{int(year)}", 100),
        )
        row["period"] = str(int(year))
        rows.append(row)
    return pd.DataFrame(rows)


def _risk_table(
    frame: pd.DataFrame,
    *,
    variable: str,
    model_name: str,
    model_type: str,
    score_transform: str,
    score: np.ndarray,
    config: Any,
) -> pd.DataFrame:
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    w = pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for q_idx, fraction in enumerate(RISK_TOP_FRACTIONS):
        row = {
            "model_name": model_name,
            "model_type": model_type,
            "index": variable,
            "evaluation_type": EVALUATION_TYPE,
            "score_transform": score_transform,
            "q_label": f"top_{100.0 * fraction:g}pct",
            **_risk_concentration_values(y, score, w, fraction),
        }
        row.update(
            _bootstrap_array_metric_errors(
                y,
                score,
                w,
                config=config,
                seed=_stable_seed(f"{model_name}:risk:{fraction}", 500 + q_idx),
                metric_names=["average_precision", "max_f1"],
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _coarse_spatial_ranking_frame(frame: pd.DataFrame, score: np.ndarray, *, resolution: float) -> pd.DataFrame:
    work = frame.copy()
    work["score"] = np.asarray(score, dtype=float)
    work["date_key"] = pd.to_datetime(work["datetime"]).values.astype("datetime64[D]").astype("int64")
    work["lat_bin"] = np.floor(pd.to_numeric(work["lat_rounded"], errors="coerce").astype(float) / resolution) * resolution
    work["lon_bin"] = np.floor(pd.to_numeric(work["lon_rounded"], errors="coerce").astype(float) / resolution) * resolution
    work["weight"] = pd.to_numeric(work["eval_weight"], errors="coerce").fillna(1.0).astype(float)
    work["observed"] = pd.to_numeric(work["is_fire"], errors="coerce").fillna(0).astype(float) * work["weight"]
    work["weighted_score"] = work["score"] * work["weight"]
    grouped = (
        work.groupby(["date_key", "lat_bin", "lon_bin"], observed=True, dropna=False)
        .agg(
            support_units=("is_fire", "size"),
            weighted_support=("weight", "sum"),
            observed_fire_positive_grid_cells=("observed", "sum"),
            weighted_score=("weighted_score", "sum"),
        )
        .reset_index()
    )
    grouped["label"] = (grouped["observed_fire_positive_grid_cells"] > 0).astype(int)
    grouped["score"] = np.divide(
        grouped["weighted_score"].to_numpy(dtype=float),
        grouped["weighted_support"].to_numpy(dtype=float),
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["weighted_support"].to_numpy(dtype=float) > 0,
    )
    return grouped


def _spatial_table(
    frame: pd.DataFrame,
    *,
    variable: str,
    model_name: str,
    model_type: str,
    score_transform: str,
    score: np.ndarray,
    config: Any,
) -> pd.DataFrame:
    y = pd.to_numeric(frame["is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    w = pd.to_numeric(frame["eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    base_resolution = float(getattr(config, "deployment_grid_resolution", 0.1) or 0.1)
    rows = [
        _spatial_metric_row(
            y,
            score,
            w,
            model_name=model_name,
            model_type=model_type,
            scale="exact_0.1_degree_cell_day",
            spatial_resolution_degrees=base_resolution,
            config=config,
            seed=_stable_seed(model_name, 600),
        )
    ]
    if {"datetime", "lat_rounded", "lon_rounded"}.issubset(frame.columns):
        neigh_y = _neighborhood_positive_labels(frame, resolution=base_resolution, radius_cells=1)
        rows.append(
            _spatial_metric_row(
                neigh_y,
                score,
                w,
                model_name=model_name,
                model_type=model_type,
                scale="neighborhood_3x3_cell_day",
                spatial_resolution_degrees=base_resolution * 3.0,
                config=config,
                seed=_stable_seed(model_name, 700),
            )
        )
        for idx, coarse_resolution in enumerate((0.5, 1.0)):
            coarse = _coarse_spatial_ranking_frame(frame, score, resolution=coarse_resolution)
            rows.append(
                _spatial_metric_row(
                    coarse["label"].to_numpy(dtype=int),
                    coarse["score"].to_numpy(dtype=float),
                    coarse["weighted_support"].to_numpy(dtype=float),
                    model_name=model_name,
                    model_type=model_type,
                    scale=f"coarse_{coarse_resolution:g}_degree_cell_day",
                    spatial_resolution_degrees=float(coarse_resolution),
                    config=config,
                    seed=_stable_seed(model_name, 800 + idx),
                    observed_fire_positive_grid_cells=float(coarse["observed_fire_positive_grid_cells"].sum()),
                    expected_fire_positive_grid_cells=float(coarse["weighted_score"].sum()),
                )
            )
    out = pd.DataFrame(rows)
    out.insert(2, "index", variable)
    out.insert(3, "evaluation_type", EVALUATION_TYPE)
    out.insert(4, "score_transform", score_transform)
    if "expected_fire_positive_grid_cells" in out.columns:
        out = out.rename(columns={"expected_fire_positive_grid_cells": "weighted_score_sum"})
    if "expected_observed_count_ratio" in out.columns:
        out = out.rename(columns={"expected_observed_count_ratio": "weighted_score_observed_ratio"})
    out = out.drop(columns=["weighted_brier_score", "weighted_logloss"], errors="ignore")
    return out


def write_variable_metric_tables(
    frame: pd.DataFrame,
    *,
    variable: str,
    output_dir: Path,
    regions: Iterable[Any] | None,
    config: Any,
    model_name: str | None = None,
    model_type: str = "Fire weather index",
    feature_set: str = "FWI GRIB index",
    score_transform: str = "minmax_per_index_evaluation_rows",
    score_mode: str = "minmax",
) -> dict[str, Any] | None:
    if frame.empty:
        return None
    frame = frame.dropna(subset=["raw_score"]).reset_index(drop=True)
    if frame.empty:
        return None
    if score_mode == "probability":
        score, score_stats = _probability_scores(frame["raw_score"])
    elif score_mode == "minmax":
        score, score_stats = _normalize_scores(frame["raw_score"])
    else:
        raise ValueError(f"Unknown score_mode={score_mode!r}; expected 'minmax' or 'probability'.")
    finite = np.isfinite(score)
    frame = frame.loc[finite].reset_index(drop=True)
    score = score[finite]
    if frame.empty:
        return None

    model_name = model_name or f"FWI {variable}"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = _regional_metric_rows(
        frame,
        variable=variable,
        model_name=model_name,
        model_type=model_type,
        feature_set=feature_set,
        score_transform=score_transform,
        regions=regions,
        score=score,
        score_stats=score_stats,
        config=config,
    )
    by_year = _by_year_rows(
        frame,
        variable=variable,
        model_name=model_name,
        model_type=model_type,
        feature_set=feature_set,
        score_transform=score_transform,
        score=score,
        score_stats=score_stats,
        config=config,
    )
    risk = _risk_table(
        frame,
        variable=variable,
        model_name=model_name,
        model_type=model_type,
        score_transform=score_transform,
        score=score,
        config=config,
    )
    spatial = _spatial_table(
        frame,
        variable=variable,
        model_name=model_name,
        model_type=model_type,
        score_transform=score_transform,
        score=score,
        config=config,
    )
    return {
        "comparison": comparison,
        "by_year": by_year,
        "risk": risk,
        "spatial": spatial,
        "global": comparison.loc[comparison["region"].eq("global")].iloc[0].to_dict(),
    }


def _upsert_table(path: Path, rows: pd.DataFrame, key_cols: list[str]) -> None:
    if rows.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if not old.empty and all(col in old.columns for col in key_cols) and all(col in rows.columns for col in key_cols):
            old_key = old[key_cols].astype(str).agg("\x1f".join, axis=1)
            new_key = rows[key_cols].astype(str).agg("\x1f".join, axis=1)
            old = old.loc[~old_key.isin(set(new_key))]
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_csv(path, index=False)


def _weighted_standardizer(X: np.ndarray, sample_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(sample_weight, dtype=float)
    weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
    if weight.sum() <= 0:
        weight = np.ones(len(X), dtype=float)
    mean = np.average(X, axis=0, weights=weight)
    var = np.average((X - mean) ** 2, axis=0, weights=weight)
    scale = np.sqrt(np.maximum(var, 1e-12))
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return mean.astype(float), scale.astype(float)


def _fit_predict_fwi_logistic(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    train_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = [
        "datetime",
        "lat_rounded",
        "lon_rounded",
        "country",
        "month",
        "year",
        "is_fire",
        "eval_weight",
        *feature_columns,
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"FWI logistic frame is missing required columns: {missing}")

    work = frame[required].copy()
    for col in feature_columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    finite = np.ones(len(work), dtype=bool)
    for col in feature_columns:
        finite &= np.isfinite(work[col].to_numpy(dtype=float))
    finite &= np.isfinite(pd.to_numeric(work["eval_weight"], errors="coerce").to_numpy(dtype=float))
    work = work.loc[finite].reset_index(drop=True)
    if work.empty:
        raise RuntimeError("No complete FWI rows are available for logistic regression.")

    years = pd.to_numeric(work["year"], errors="coerce").astype(int)
    train_mask = years.eq(int(train_year)).to_numpy()
    test_mask = ~train_mask
    if not train_mask.any():
        raise RuntimeError(f"No FWI rows are available for logistic training year {train_year}.")
    if not test_mask.any():
        raise RuntimeError(f"No non-{train_year} FWI rows are available for logistic scoring.")

    y_train = pd.to_numeric(work.loc[train_mask, "is_fire"], errors="coerce").fillna(0).astype(int).to_numpy()
    if np.unique(y_train).size < 2:
        raise RuntimeError(f"FWI logistic training year {train_year} does not contain both classes.")
    w_train = pd.to_numeric(work.loc[train_mask, "eval_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    X_train_raw = work.loc[train_mask, feature_columns].to_numpy(dtype=float)
    mean, scale = _weighted_standardizer(X_train_raw, w_train)
    X_train = (X_train_raw - mean) / scale
    fit_weight = w_train / float(np.mean(w_train)) if np.mean(w_train) > 0 else w_train

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X_train, y_train, sample_weight=fit_weight)

    eval_frame = work.loc[test_mask, [col for col in PREDICTION_COLUMNS if col != "raw_score"]].copy()
    X_test = (work.loc[test_mask, feature_columns].to_numpy(dtype=float) - mean) / scale
    eval_frame["raw_score"] = model.predict_proba(X_test)[:, 1].astype(np.float32)

    coef_scaled = np.asarray(model.coef_[0], dtype=float)
    coef_original = coef_scaled / scale
    intercept_original = float(model.intercept_[0] - np.sum(coef_scaled * mean / scale))
    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient_scaled": coef_scaled,
            "coefficient_original_units": coef_original,
            "training_weighted_mean": mean,
            "training_weighted_scale": scale,
        }
    )
    coefficients = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature": ["intercept"],
                    "coefficient_scaled": [float(model.intercept_[0])],
                    "coefficient_original_units": [intercept_original],
                    "training_weighted_mean": [np.nan],
                    "training_weighted_scale": [np.nan],
                }
            ),
            coefficients,
        ],
        ignore_index=True,
    )
    metadata = {
        "model_name": _logistic_model_name(train_year),
        "model_type": LOGISTIC_MODEL_TYPE,
        "train_year": int(train_year),
        "feature_columns": feature_columns,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_weighted_support": float(np.sum(w_train)),
        "train_positives": float(np.sum(y_train * w_train)),
        "train_observed_prevalence": float(np.average(y_train, weights=w_train)),
        "test_years": sorted(int(value) for value in years.loc[test_mask].drop_duplicates()),
    }
    return eval_frame, coefficients, metadata


def write_fwi_logistic_regression_tables(
    feature_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    train_year: int,
    output_dir: Path,
    regions: Iterable[Any] | None,
    config: Any,
    append_to_main: bool = True,
) -> dict[str, Any]:
    eval_frame, coefficients, metadata = _fit_predict_fwi_logistic(
        feature_frame,
        feature_columns=feature_columns,
        train_year=train_year,
    )
    feature_set = f"FWI indexes trained on {train_year}: {', '.join(feature_columns)}"
    metrics = write_variable_metric_tables(
        eval_frame,
        variable=f"logistic_train_{train_year}",
        output_dir=output_dir,
        regions=regions,
        config=config,
        model_name=_logistic_model_name(train_year),
        model_type=LOGISTIC_MODEL_TYPE,
        feature_set=feature_set,
        score_transform=LOGISTIC_SCORE_TRANSFORM,
        score_mode="probability",
    )
    if metrics is None:
        raise RuntimeError("FWI logistic regression produced no evaluation rows.")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics["comparison"].to_csv(output_dir / "logistic_regression_model_comparison.csv", index=False)
    metrics["by_year"].to_csv(output_dir / "logistic_regression_model_comparison_by_year.csv", index=False)
    metrics["risk"].to_csv(output_dir / "logistic_regression_risk_concentration.csv", index=False)
    metrics["spatial"].to_csv(output_dir / "logistic_regression_spatial_scale_evaluation.csv", index=False)
    coefficients.to_csv(output_dir / "logistic_regression_coefficients.csv", index=False)

    metadata = {
        **metadata,
        "evaluation_type": EVALUATION_TYPE,
        "score_transform": LOGISTIC_SCORE_TRANSFORM,
        "global_metrics": metrics["global"],
    }
    (output_dir / "logistic_regression_manifest.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    if append_to_main:
        _upsert_table(output_dir / "model_comparison.csv", metrics["comparison"], ["model_name", "region", "split"])
        _upsert_table(output_dir / "model_comparison_by_year.csv", metrics["by_year"], ["model_name", "region", "period"])
        _upsert_table(output_dir / "risk_concentration.csv", metrics["risk"], ["model_name", "q_label"])
        _upsert_table(output_dir / "spatial_scale_evaluation.csv", metrics["spatial"], ["model_name", "scale"])

    return metadata


def _write_readme(output_dir: Path) -> None:
    text = """# Fire Weather Index Full-Grid Ranking Evaluation

These tables evaluate raw fire-weather index GRIB values as risk-ranking scores on the same deployment-grid target rows used by the revision full-grid evaluation.

Raw index scores are min-max normalized per index only so ranking metrics can reuse the same metric helpers. The optional FWI logistic-regression baseline uses weighted probabilities from a model trained on the configured training year. These FWI baselines are not post-hoc calibrated like the primary CatBoost full-grid study, so reliability and expected-count calibration tables are intentionally omitted.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run_fire_weather_index_evaluation(
    config: EvaluationConfig,
    *,
    fwi_dir: Path | None = None,
    output_dir: Path | None = None,
    variables: Iterable[str] | None = None,
    save_predictions: bool | None = None,
    max_open_gribs: int = 2,
) -> dict[str, Any]:
    fwi_dir = Path(fwi_dir or getattr(config, "fire_weather_index_dir", "fire_weather_indexes"))
    output_dir = Path(
        output_dir
        or getattr(config, "fire_weather_index_output_dir", None)
        or (config.output_dir / DEFAULT_OUTPUT_SUBDIR)
    )
    save_predictions = bool(
        getattr(config, "fire_weather_index_save_predictions", False)
        if save_predictions is None
        else save_predictions
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_readme(output_dir)

    logging.info("Scanning FWI GRIB inventory under %s", fwi_dir)
    inventory = scan_fwi_inventory(fwi_dir)
    configured_variables = variables if variables is not None else getattr(config, "fire_weather_index_variables", None)
    selected = _selected_variables(inventory, configured_variables)
    start, end = _date_bounds(config)
    logistic_train_year = int(getattr(config, "fire_weather_index_logistic_train_year", 2022) or 2022)
    run_logistic = bool(getattr(config, "fire_weather_index_run_logistic_regression", True))
    logistic_variables = (
        inventory.variables_available_in_year(selected, logistic_train_year, start_date=start, end_date=end)
        if run_logistic
        else []
    )
    available_dates = inventory.available_dates(selected, start_date=start, end_date=end)
    if len(available_dates) == 0:
        raise RuntimeError("No FWI dates overlap the configured test period.")

    inventory.table.to_csv(output_dir / "inventory.csv", index=False)
    availability = inventory.availability_summary(selected, start_date=start, end_date=end)
    availability.to_csv(output_dir / "availability.csv", index=False)
    date_to_path = inventory.date_to_path(selected)

    feature_config = load_yaml(config.feature_config)
    target_config = load_yaml(config.target_config)
    regions = load_regions(config.regions_file)
    part_store = PredictionPartStore(output_dir / "_prediction_parts")
    logistic_store_key = f"fwi_logistic_train_{logistic_train_year}_features"
    cache = FWIDatasetCache(max_open=max_open_gribs)
    try:
        for chunk_idx, chunk in enumerate(
            iter_available_deployment_chunks(
                config=config,
                feature_config=feature_config,
                target_config=target_config,
                available_dates=available_dates,
            ),
            start=1,
        ):
            if chunk.rows.empty:
                continue
            logging.info(
                "Scoring FWI chunk %s country=%s period=%s rows=%d",
                chunk_idx,
                chunk.country,
                chunk.period,
                len(chunk.rows),
            )
            scores = _score_chunk(
                chunk.rows,
                variables=selected,
                date_to_path=date_to_path,
                cache=cache,
            )
            _write_scored_chunk(part_store, chunk.rows, scores)
            _write_wide_scored_chunk(
                part_store,
                chunk.rows,
                scores,
                variables=logistic_variables,
                key=logistic_store_key,
            )
    finally:
        cache.close()

    comparison_tables: list[pd.DataFrame] = []
    by_year_tables: list[pd.DataFrame] = []
    risk_tables: list[pd.DataFrame] = []
    spatial_tables: list[pd.DataFrame] = []
    global_rows: list[dict[str, Any]] = []
    for variable in selected:
        frame = part_store.read(variable)
        logging.info("Aggregating FWI metrics for %s rows=%d", variable, len(frame))
        metrics = write_variable_metric_tables(
            frame,
            variable=variable,
            output_dir=output_dir,
            regions=regions,
            config=config,
        )
        if metrics is None:
            logging.warning("No scored rows for FWI variable %s", variable)
            continue
        comparison_tables.append(metrics["comparison"])
        by_year_tables.append(metrics["by_year"])
        risk_tables.append(metrics["risk"])
        spatial_tables.append(metrics["spatial"])
        global_rows.append(metrics["global"])

    logistic_metadata: dict[str, Any] | None = None
    if run_logistic:
        if not logistic_variables:
            logging.warning(
                "Skipping FWI logistic regression because no selected FWI variables are available in %s.",
                logistic_train_year,
            )
        else:
            logistic_frame = part_store.read(logistic_store_key)
            logging.info(
                "Training FWI logistic regression on %s variables from %s; complete rows=%d",
                len(logistic_variables),
                logistic_train_year,
                len(logistic_frame),
            )
            logistic_metadata = write_fwi_logistic_regression_tables(
                logistic_frame,
                feature_columns=logistic_variables,
                train_year=logistic_train_year,
                output_dir=output_dir,
                regions=regions,
                config=config,
                append_to_main=False,
            )
            comparison_tables.append(pd.read_csv(output_dir / "logistic_regression_model_comparison.csv"))
            by_year_tables.append(pd.read_csv(output_dir / "logistic_regression_model_comparison_by_year.csv"))
            risk_tables.append(pd.read_csv(output_dir / "logistic_regression_risk_concentration.csv"))
            spatial_tables.append(pd.read_csv(output_dir / "logistic_regression_spatial_scale_evaluation.csv"))
            global_rows.append(logistic_metadata["global_metrics"])

    if comparison_tables:
        pd.concat(comparison_tables, ignore_index=True).to_csv(output_dir / "model_comparison.csv", index=False)
    if by_year_tables:
        pd.concat(by_year_tables, ignore_index=True).to_csv(output_dir / "model_comparison_by_year.csv", index=False)
    if risk_tables:
        pd.concat(risk_tables, ignore_index=True).to_csv(output_dir / "risk_concentration.csv", index=False)
    if spatial_tables:
        pd.concat(spatial_tables, ignore_index=True).to_csv(output_dir / "spatial_scale_evaluation.csv", index=False)
    if save_predictions:
        part_store.persist_to(output_dir / "predictions")
    part_store.cleanup()

    manifest = {
        "evaluation_type": EVALUATION_TYPE,
        "fwi_dir": str(fwi_dir),
        "output_dir": str(output_dir),
        "variables": selected,
        "logistic_regression": logistic_metadata,
        "available_start_date": pd.Timestamp(available_dates.min()).date().isoformat(),
        "available_end_date": pd.Timestamp(available_dates.max()).date().isoformat(),
        "available_days_union": int(len(available_dates)),
        "test_start_date": str(getattr(config, "test_start_date", None)),
        "test_end_date": str(getattr(config, "test_end_date", None)),
        "deployment_grid_resolution": float(getattr(config, "deployment_grid_resolution", 0.1) or 0.1),
        "full_grid_mode": str(getattr(config, "full_grid_mode", "full_grid")),
        "weighted_grid_sample_fraction": getattr(config, "weighted_grid_sample_fraction", None),
        "save_predictions": save_predictions,
        "global_metrics": global_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def run_fire_weather_index_logistic_regression(
    config: EvaluationConfig,
    *,
    fwi_dir: Path | None = None,
    output_dir: Path | None = None,
    variables: Iterable[str] | None = None,
    train_year: int | None = None,
    max_open_gribs: int = 2,
) -> dict[str, Any]:
    fwi_dir = Path(fwi_dir or getattr(config, "fire_weather_index_dir", "fire_weather_indexes"))
    output_dir = Path(
        output_dir
        or getattr(config, "fire_weather_index_output_dir", None)
        or (config.output_dir / DEFAULT_OUTPUT_SUBDIR)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_readme(output_dir)

    logging.info("Scanning FWI GRIB inventory under %s", fwi_dir)
    inventory = scan_fwi_inventory(fwi_dir)
    configured_variables = variables if variables is not None else getattr(config, "fire_weather_index_variables", None)
    selected = _selected_variables(inventory, configured_variables)
    start, end = _date_bounds(config)
    train_year = int(train_year or getattr(config, "fire_weather_index_logistic_train_year", 2022) or 2022)
    feature_columns = inventory.variables_available_in_year(selected, train_year, start_date=start, end_date=end)
    if not feature_columns:
        raise RuntimeError(f"No selected FWI variables are available in logistic training year {train_year}.")

    available_dates = inventory.complete_case_dates(feature_columns, start_date=start, end_date=end)
    train_dates = available_dates[available_dates.year == train_year]
    eval_dates = available_dates[available_dates.year != train_year]
    if len(train_dates) == 0:
        raise RuntimeError(f"No complete-case FWI dates are available for training year {train_year}.")
    if len(eval_dates) == 0:
        raise RuntimeError(f"No complete-case FWI dates are available outside training year {train_year}.")

    inventory.table.to_csv(output_dir / "inventory.csv", index=False)
    inventory.availability_summary(feature_columns, start_date=start, end_date=end).to_csv(
        output_dir / "logistic_regression_availability.csv",
        index=False,
    )
    date_to_path = inventory.date_to_path(feature_columns)
    feature_config = load_yaml(config.feature_config)
    target_config = load_yaml(config.target_config)
    regions = load_regions(config.regions_file)

    part_store = PredictionPartStore(output_dir / "_logistic_prediction_parts")
    store_key = f"fwi_logistic_train_{train_year}_features"
    cache = FWIDatasetCache(max_open=max_open_gribs)
    try:
        for chunk_idx, chunk in enumerate(
            iter_available_deployment_chunks(
                config=config,
                feature_config=feature_config,
                target_config=target_config,
                available_dates=available_dates,
            ),
            start=1,
        ):
            if chunk.rows.empty:
                continue
            logging.info(
                "Scoring FWI logistic chunk %s country=%s period=%s rows=%d",
                chunk_idx,
                chunk.country,
                chunk.period,
                len(chunk.rows),
            )
            scores = _score_chunk(
                chunk.rows,
                variables=feature_columns,
                date_to_path=date_to_path,
                cache=cache,
            )
            _write_wide_scored_chunk(
                part_store,
                chunk.rows,
                scores,
                variables=feature_columns,
                key=store_key,
            )
    finally:
        cache.close()

    try:
        feature_frame = part_store.read(store_key)
        logging.info(
            "Training FWI logistic regression on %s variables from %s; complete rows=%d",
            len(feature_columns),
            train_year,
            len(feature_frame),
        )
        metadata = write_fwi_logistic_regression_tables(
            feature_frame,
            feature_columns=feature_columns,
            train_year=train_year,
            output_dir=output_dir,
            regions=regions,
            config=config,
            append_to_main=True,
        )
    finally:
        part_store.cleanup()

    metadata.update(
        {
            "fwi_dir": str(fwi_dir),
            "output_dir": str(output_dir),
            "available_start_date": pd.Timestamp(available_dates.min()).date().isoformat(),
            "available_end_date": pd.Timestamp(available_dates.max()).date().isoformat(),
            "available_days_complete_case": int(len(available_dates)),
            "training_days": int(len(train_dates)),
            "scored_days": int(len(eval_dates)),
        }
    )
    (output_dir / "logistic_regression_manifest.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["logistic_regression"] = metadata
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Fire Weather Index GRIBs as full-grid fire-risk rankings.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/revision_evaluation_all_models_with_nns.yaml",
        help="Revision evaluation YAML config.",
    )
    parser.add_argument("--fwi-dir", type=Path, default=None, help="Directory containing FWI .grib files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for FWI ranking tables.")
    parser.add_argument(
        "--variables",
        default=None,
        help="Comma-separated FWI shortNames to evaluate. Defaults to all variables in the GRIB inventory.",
    )
    parser.add_argument("--save-predictions", action="store_true", help="Persist scored grid rows by index.")
    parser.add_argument(
        "--only-logistic-regression",
        action="store_true",
        help="Only run the FWI logistic-regression baseline and append it to existing FWI tables.",
    )
    parser.add_argument(
        "--skip-logistic-regression",
        action="store_true",
        help="Skip the FWI logistic-regression baseline during the full raw-index run.",
    )
    parser.add_argument("--logistic-train-year", type=int, default=None, help="Training year for FWI logistic regression.")
    parser.add_argument("--max-open-gribs", type=int, default=2, help="Maximum number of open GRIB datasets.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = EvaluationConfig.from_yaml(Path(args.config))
    variables = [item.strip() for item in args.variables.split(",") if item.strip()] if args.variables else None
    if args.logistic_train_year is not None:
        config.fire_weather_index_logistic_train_year = int(args.logistic_train_year)
    if args.skip_logistic_regression:
        config.fire_weather_index_run_logistic_regression = False
    if args.only_logistic_regression:
        run_fire_weather_index_logistic_regression(
            config,
            fwi_dir=args.fwi_dir,
            output_dir=args.output_dir,
            variables=variables,
            train_year=args.logistic_train_year,
            max_open_gribs=args.max_open_gribs,
        )
        return 0
    run_fire_weather_index_evaluation(
        config,
        fwi_dir=args.fwi_dir,
        output_dir=args.output_dir,
        variables=variables,
        save_predictions=args.save_predictions,
        max_open_gribs=args.max_open_gribs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
