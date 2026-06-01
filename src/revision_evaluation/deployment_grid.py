from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely import vectorized as shapely_vectorized
except Exception as exc:  # pragma: no cover - optional until full-grid eval runs
    gpd = None
    shapely_vectorized = None
    GEOSPATIAL_IMPORT_ERROR = exc
else:
    GEOSPATIAL_IMPORT_ERROR = None

from src.target_generation.prepare_target_new import (
    SPATIAL_COARSENESS,
    country_mapping,
    expand_positive_points,
    load_modis_data,
    _initial_positive_counts,
)


REQUIRED_DEPLOYMENT_COLUMNS = [
    "datetime",
    "lat_rounded",
    "lon_rounded",
    "country",
    "month",
    "year",
    "is_fire",
    "eval_weight",
]


@dataclass
class DeploymentGridChunk:
    split_name: str
    country: str | None
    period: str
    rows: pd.DataFrame


def stable_seed(*parts: Any, global_seed: int = 42) -> int:
    text = "|".join(str(part) for part in (global_seed, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def split_csv(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def date_range_for_split(config: Any, split_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if split_name == "calibration":
        start = getattr(config, "calibration_start_date", None) or "2021-01-01"
        end = getattr(config, "calibration_end_date", None) or "2021-12-31"
    elif split_name == "test":
        start = getattr(config, "test_start_date", None) or "2022-01-01"
        end = getattr(config, "test_end_date", None) or "2025-10-02"
    else:
        raise ValueError(f"Unknown deployment split: {split_name}")
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if end_ts < start_ts:
        raise ValueError(f"{split_name} end date {end_ts.date()} is before start date {start_ts.date()}.")
    return start_ts, end_ts


def countries_for_grid(config: Any, feature_config: dict[str, Any], target_config: dict[str, Any]) -> list[str]:
    configured = getattr(config, "deployment_grid_countries", None)
    if configured:
        return split_csv(configured)
    for key in ["modis_countries", "prediction_countries"]:
        value = feature_config.get(key) or target_config.get(key)
        if value:
            return split_csv(value)
    return ["Russian_Federation"]


def parse_coordinate_bounds(value: Any) -> tuple[float, float, float, float] | None:
    """Return coordinate bounds as (min_lat, min_lon, max_lat, max_lon)."""

    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        parts = split_csv(value)
    else:
        parts = list(value)
    if len(parts) != 4:
        raise ValueError(
            "deployment-grid coordinate bounds must have four values: "
            "[min_lat, min_lon, max_lat, max_lon]."
        )
    min_lat, min_lon, max_lat, max_lon = (float(part) for part in parts)
    if max_lat < min_lat or max_lon < min_lon:
        raise ValueError(
            f"Invalid coordinate bounds {(min_lat, min_lon, max_lat, max_lon)}; "
            "expected min values before max values."
        )
    return min_lat, min_lon, max_lat, max_lon


def coordinate_bounds_for_grid(
    config: Any,
    feature_config: dict[str, Any],
    target_config: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    configured = parse_coordinate_bounds(getattr(config, "deployment_grid_coordinate_bounds", None))
    if configured is not None:
        return configured
    if not bool(getattr(config, "deployment_grid_clip_to_feature_bounds", True)):
        return None
    return parse_coordinate_bounds(feature_config.get("coordinate_bounds") or target_config.get("coordinate_bounds"))


def _filter_frame_to_bounds(
    frame: pd.DataFrame,
    bounds: tuple[float, float, float, float] | None,
    *,
    resolution: float,
) -> pd.DataFrame:
    if bounds is None or frame.empty:
        return frame
    min_lat, min_lon, max_lat, max_lon = bounds
    eps = max(float(resolution), 1e-6) * 1e-6
    mask = (
        frame["lat_rounded"].between(min_lat - eps, max_lat + eps)
        & frame["lon_rounded"].between(min_lon - eps, max_lon + eps)
    )
    return frame.loc[mask].reset_index(drop=True)


def _coordinate_key_frame(frame: pd.DataFrame, resolution: float) -> pd.DataFrame:
    scale = 1.0 / float(resolution)
    return pd.DataFrame(
        {
            "lat_key": np.rint(pd.to_numeric(frame["lat_rounded"], errors="coerce").to_numpy(dtype=float) * scale).astype(
                np.int64
            ),
            "lon_key": np.rint(pd.to_numeric(frame["lon_rounded"], errors="coerce").to_numpy(dtype=float) * scale).astype(
                np.int64
            ),
        },
        index=frame.index,
    )


def _geometry_name_columns(world: Any) -> list[str]:
    return [col for col in ["SOVEREIGNT", "ADMIN", "NAME", "NAME_EN"] if col in world.columns]


def load_country_geometries(country_shapes_path: str | Path, countries: list[str]) -> dict[str, Any]:
    if gpd is None:
        raise ImportError(f"geopandas/shapely are required for full-grid evaluation: {GEOSPATIAL_IMPORT_ERROR}")
    world = gpd.read_file(country_shapes_path)
    name_cols = _geometry_name_columns(world)
    if not name_cols:
        raise ValueError(f"No usable country-name column found in {country_shapes_path}.")

    geometries: dict[str, Any] = {}
    for country in countries:
        mapped = country_mapping.get(country, country)
        mask = np.zeros(len(world), dtype=bool)
        for col in name_cols:
            mask |= world[col].astype(str).eq(mapped).to_numpy()
            mask |= world[col].astype(str).eq(country).to_numpy()
        matches = world.loc[mask]
        if matches.empty:
            raise ValueError(f"Country geometry not found for {country!r} (mapped to {mapped!r}).")
        geom = matches.geometry.unary_union
        if not geom.is_valid:
            geom = geom.buffer(0)
        geometries[country] = geom
    return geometries


def generate_grid_cells_for_geometry(
    geometry: Any,
    *,
    country: str,
    resolution: float,
) -> pd.DataFrame:
    if shapely_vectorized is None:
        raise ImportError(f"shapely.vectorized is required for full-grid evaluation: {GEOSPATIAL_IMPORT_ERROR}")
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    lat = np.round(
        np.arange(
            np.floor(min_lat / resolution) * resolution,
            np.ceil(max_lat / resolution) * resolution + resolution / 2.0,
            resolution,
        ),
        10,
    )
    lon = np.round(
        np.arange(
            np.floor(min_lon / resolution) * resolution,
            np.ceil(max_lon / resolution) * resolution + resolution / 2.0,
            resolution,
        ),
        10,
    )
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask = shapely_vectorized.contains(geometry, lon_grid, lat_grid)
    cells = pd.DataFrame(
        {
            "lat_rounded": lat_grid[mask].astype(np.float32),
            "lon_rounded": lon_grid[mask].astype(np.float32),
            "country": country,
        }
    )
    return cells.drop_duplicates(["lat_rounded", "lon_rounded"]).reset_index(drop=True)


def _positive_label_frame(raw: pd.DataFrame, resolution: float) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["datetime", "lat_rounded", "lon_rounded", "country", "is_fire"])
    rounding_precision = (
        max(0, int(np.ceil(-np.log10(resolution))))
        if resolution > 0
        else int(-np.log10(SPATIAL_COARSENESS))
    )
    data = raw.copy()
    data["lat_rounded"] = data["latitude"].round(rounding_precision)
    data["lon_rounded"] = data["longitude"].round(rounding_precision)
    data["acq_date"] = pd.to_datetime(data["acq_date"]).dt.date
    data["month"] = pd.to_datetime(data["acq_date"]).dt.month
    data["year"] = pd.to_datetime(data["acq_date"]).dt.year
    data["day"] = pd.to_datetime(data["acq_date"]).dt.day
    data["count"] = _initial_positive_counts(data["latitude"], data["longitude"])
    grouped = data.groupby(["lat_rounded", "lon_rounded", "acq_date"], observed=True).agg(
        brightness=("brightness", "mean"),
        confidence=("confidence", "mean"),
        count=("count", "sum"),
        month=("month", "first"),
        year=("year", "first"),
        country=("country", "first"),
        day=("day", "first"),
    ).reset_index()
    expanded = expand_positive_points(
        grouped,
        spatial_coarseness=resolution,
        lat_col="lat_rounded",
        lon_col="lon_rounded",
        count_col="count",
    )
    expanded["datetime"] = pd.to_datetime(expanded["acq_date"])
    labels = expanded[["datetime", "lat_rounded", "lon_rounded", "country"]].drop_duplicates()
    labels["is_fire"] = 1
    return labels


def build_fire_label_frame(
    *,
    feature_config: dict[str, Any],
    target_config: dict[str, Any],
    countries: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    resolution: float,
) -> pd.DataFrame:
    modis_path = feature_config.get("modis_data_path") or target_config.get("modis_data_path") or "data/modis"
    raw = load_modis_data(
        data_dir=modis_path,
        countries=countries,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    labels = _positive_label_frame(raw, resolution)
    labels["datetime"] = pd.to_datetime(labels["datetime"])
    return labels


def _apply_weighted_sample(
    rows: pd.DataFrame,
    *,
    config: Any,
    split_name: str,
    country: str | None,
    period: str,
) -> pd.DataFrame:
    fraction = getattr(config, "weighted_grid_sample_fraction", None)
    if fraction is None:
        return rows
    fraction = float(fraction)
    if fraction <= 0 or fraction >= 1:
        return rows
    rng = np.random.default_rng(stable_seed(split_name, country or "all", period, global_seed=getattr(config, "seed", 42)))
    take = max(1, int(round(len(rows) * fraction)))
    idx = rng.choice(np.arange(len(rows)), size=take, replace=False)
    sampled = rows.iloc[np.sort(idx)].copy()
    sampled["eval_weight"] = sampled["eval_weight"].astype(float) * (len(rows) / len(sampled))
    return sampled.reset_index(drop=True)


def _cross_cells_dates(
    cells: pd.DataFrame,
    dates: pd.DatetimeIndex,
    labels: pd.DataFrame,
    country: str,
    *,
    resolution: float = SPATIAL_COARSENESS,
) -> pd.DataFrame:
    if cells.empty or len(dates) == 0:
        return pd.DataFrame(columns=REQUIRED_DEPLOYMENT_COLUMNS)
    dates_frame = pd.DataFrame({"datetime": dates})
    cells = cells[["lat_rounded", "lon_rounded", "country"]].copy()
    cells["_join_key"] = 1
    dates_frame["_join_key"] = 1
    rows = cells.merge(dates_frame, on="_join_key", how="inner").drop(columns="_join_key")
    rows["datetime"] = pd.to_datetime(rows["datetime"])
    # The revision feature pipeline historically uses both names:
    # `datetime` for target anchors and `acq_date` for climate slicing.
    rows["acq_date"] = rows["datetime"].dt.normalize()
    rows["month"] = rows["datetime"].dt.month.astype(np.int16)
    rows["year"] = rows["datetime"].dt.year.astype(np.int16)
    rows["day"] = rows["datetime"].dt.day.astype(np.int16)
    rows["count"] = 0
    key_cols = ["datetime", "lat_rounded", "lon_rounded"]
    if not labels.empty:
        row_keys = _coordinate_key_frame(rows, resolution)
        rows["_lat_key"] = row_keys["lat_key"].to_numpy()
        rows["_lon_key"] = row_keys["lon_key"].to_numpy()
        label_subset = labels[["datetime", "lat_rounded", "lon_rounded", "is_fire"]].copy()
        label_subset["datetime"] = pd.to_datetime(label_subset["datetime"]).dt.normalize()
        label_keys = _coordinate_key_frame(label_subset, resolution)
        label_subset["_lat_key"] = label_keys["lat_key"].to_numpy()
        label_subset["_lon_key"] = label_keys["lon_key"].to_numpy()
        label_subset = label_subset[["datetime", "_lat_key", "_lon_key", "is_fire"]].drop_duplicates(
            ["datetime", "_lat_key", "_lon_key"]
        )
        rows = rows.merge(label_subset, on=["datetime", "_lat_key", "_lon_key"], how="left")
        rows = rows.drop(columns=["_lat_key", "_lon_key"])
        rows["is_fire"] = rows["is_fire"].fillna(0).astype(np.int8)
    else:
        rows["is_fire"] = 0
    rows.loc[rows["is_fire"] > 0, "count"] = 1
    rows["eval_weight"] = 1.0
    rows["country"] = rows["country"].fillna(country)
    return rows[REQUIRED_DEPLOYMENT_COLUMNS + ["acq_date", "day", "count"]]


def _rows_from_flat_indices(
    cells: pd.DataFrame,
    dates: pd.DatetimeIndex,
    flat_idx: np.ndarray,
    *,
    country: str,
    eval_weight: float,
) -> pd.DataFrame:
    if len(flat_idx) == 0:
        return pd.DataFrame(columns=REQUIRED_DEPLOYMENT_COLUMNS + ["acq_date", "day", "count"])
    date_count = len(dates)
    cell_idx = flat_idx // date_count
    date_idx = flat_idx % date_count
    sampled_cells = cells.iloc[cell_idx][["lat_rounded", "lon_rounded", "country"]].reset_index(drop=True)
    rows = sampled_cells.copy()
    rows["datetime"] = pd.to_datetime(np.asarray(dates)[date_idx])
    rows["acq_date"] = rows["datetime"].dt.normalize()
    rows["month"] = rows["datetime"].dt.month.astype(np.int16)
    rows["year"] = rows["datetime"].dt.year.astype(np.int16)
    rows["day"] = rows["datetime"].dt.day.astype(np.int16)
    rows["is_fire"] = np.int8(0)
    rows["count"] = 0
    rows["eval_weight"] = float(eval_weight)
    rows["country"] = rows["country"].fillna(country)
    return rows[REQUIRED_DEPLOYMENT_COLUMNS + ["acq_date", "day", "count"]]


def _positive_flat_indices(
    cells: pd.DataFrame,
    dates: pd.DatetimeIndex,
    labels: pd.DataFrame,
    country: str,
    *,
    resolution: float = SPATIAL_COARSENESS,
) -> np.ndarray:
    if labels.empty or cells.empty or len(dates) == 0:
        return np.array([], dtype=np.int64)

    label_subset = labels.copy()
    if "country" in label_subset.columns:
        label_subset = label_subset.loc[label_subset["country"].fillna(country).astype(str).eq(str(country))]
    label_subset["datetime"] = pd.to_datetime(label_subset["datetime"]).dt.normalize()
    date_index = pd.Index(pd.to_datetime(dates).normalize())
    label_subset = label_subset.loc[label_subset["datetime"].isin(date_index)]
    if label_subset.empty:
        return np.array([], dtype=np.int64)

    label_subset = label_subset[["datetime", "lat_rounded", "lon_rounded"]].drop_duplicates()
    cell_keys = _coordinate_key_frame(cells, resolution)
    label_keys = _coordinate_key_frame(label_subset, resolution)
    cell_index = pd.MultiIndex.from_frame(cell_keys[["lat_key", "lon_key"]])
    label_cell_index = pd.MultiIndex.from_frame(label_keys[["lat_key", "lon_key"]])
    cell_idx = cell_index.get_indexer(label_cell_index)
    date_idx = date_index.get_indexer(label_subset["datetime"])
    valid = (cell_idx >= 0) & (date_idx >= 0)
    if not np.any(valid):
        return np.array([], dtype=np.int64)
    flat = cell_idx[valid].astype(np.int64) * np.int64(len(dates)) + date_idx[valid].astype(np.int64)
    return np.unique(flat)


def _sample_flat_indices(
    total: int,
    take: int,
    rng: np.random.Generator,
    *,
    excluded: np.ndarray | None = None,
) -> np.ndarray:
    excluded_arr = np.unique(np.asarray(excluded if excluded is not None else [], dtype=np.int64))
    available = max(0, int(total) - len(excluded_arr))
    take = min(max(0, int(take)), available)
    if take == 0:
        return np.array([], dtype=np.int64)
    if len(excluded_arr) == 0:
        return np.sort(rng.choice(total, size=take, replace=False).astype(np.int64))

    candidate_size = min(int(total), take + len(excluded_arr))
    candidates = rng.choice(total, size=candidate_size, replace=False).astype(np.int64)
    selected = candidates[~np.isin(candidates, excluded_arr, assume_unique=False)]
    if len(selected) >= take:
        return np.sort(selected[:take])

    banned = np.unique(np.concatenate([excluded_arr, selected]))
    while len(selected) < take:
        remaining = take - len(selected)
        batch_size = min(int(total), remaining + len(banned))
        extra = rng.choice(total, size=batch_size, replace=False).astype(np.int64)
        extra = extra[~np.isin(extra, banned, assume_unique=False)]
        if len(extra) == 0:
            break
        selected = np.concatenate([selected, extra[:remaining]])
        banned = np.unique(np.concatenate([banned, extra[:remaining]]))
    return np.sort(selected[:take])


def _sample_cross_cells_dates(
    cells: pd.DataFrame,
    dates: pd.DatetimeIndex,
    labels: pd.DataFrame,
    country: str,
    *,
    config: Any,
    split_name: str,
    period: str,
    resolution: float = SPATIAL_COARSENESS,
) -> pd.DataFrame:
    fraction = getattr(config, "weighted_grid_sample_fraction", None)
    if fraction is None:
        return _cross_cells_dates(cells, dates, labels, country, resolution=resolution)
    fraction = float(fraction)
    if fraction <= 0 or fraction >= 1:
        return _cross_cells_dates(cells, dates, labels, country, resolution=resolution)
    if cells.empty or len(dates) == 0:
        return pd.DataFrame(columns=REQUIRED_DEPLOYMENT_COLUMNS)

    total = int(len(cells) * len(dates))
    rng = np.random.default_rng(stable_seed(split_name, country, period, global_seed=getattr(config, "seed", 42)))
    include_all_positives = bool(getattr(config, "weighted_grid_sample_include_all_positives", True))
    if include_all_positives:
        positive_flat = _positive_flat_indices(cells, dates, labels, country, resolution=resolution)
        total_negative = total - len(positive_flat)
        take_negative = max(1, int(round(total_negative * fraction))) if total_negative > 0 else 0
        negative_flat = _sample_flat_indices(total, take_negative, rng, excluded=positive_flat)
        negative_weight = float(total_negative) / float(len(negative_flat)) if len(negative_flat) else 1.0
        positive_rows = _rows_from_flat_indices(cells, dates, positive_flat, country=country, eval_weight=1.0)
        if not positive_rows.empty:
            positive_rows["is_fire"] = np.int8(1)
            positive_rows["count"] = 1
        negative_rows = _rows_from_flat_indices(cells, dates, negative_flat, country=country, eval_weight=negative_weight)
        rows = pd.concat([positive_rows, negative_rows], ignore_index=True)
        if not rows.empty:
            rows = rows.sort_values(["datetime", "lat_rounded", "lon_rounded"]).reset_index(drop=True)
        return rows[REQUIRED_DEPLOYMENT_COLUMNS + ["acq_date", "day", "count"]]

    take = max(1, int(round(total * fraction)))
    flat_idx = _sample_flat_indices(total, take, rng)
    rows = _rows_from_flat_indices(cells, dates, flat_idx, country=country, eval_weight=float(total) / float(take))
    key_cols = ["datetime", "lat_rounded", "lon_rounded"]
    if not labels.empty:
        label_subset = labels[key_cols + ["is_fire"]].drop_duplicates(key_cols)
        rows = rows.merge(label_subset, on=key_cols, how="left", suffixes=("", "_label"))
        rows["is_fire"] = rows["is_fire_label"].fillna(rows["is_fire"]).astype(np.int8)
        rows = rows.drop(columns=["is_fire_label"])
    rows.loc[rows["is_fire"] > 0, "count"] = 1
    return rows[REQUIRED_DEPLOYMENT_COLUMNS + ["acq_date", "day", "count"]]


def _iter_cell_date_row_blocks(
    cells: pd.DataFrame,
    dates: pd.DatetimeIndex,
    labels: pd.DataFrame,
    country: str,
    max_rows: int | None,
    resolution: float = SPATIAL_COARSENESS,
) -> Iterable[tuple[str, pd.DataFrame]]:
    if max_rows is None or cells.empty or len(dates) == 0 or len(cells) * len(dates) <= max_rows:
        yield "", _cross_cells_dates(cells, dates, labels, country, resolution=resolution)
        return

    cells_per_block = max(1, int(max_rows) // max(1, len(dates)))
    for part_idx, start in enumerate(range(0, len(cells), cells_per_block)):
        part_cells = cells.iloc[start : start + cells_per_block].reset_index(drop=True)
        yield f"_part{part_idx:04d}", _cross_cells_dates(part_cells, dates, labels, country, resolution=resolution)


def _month_date_batches(
    dates: pd.DatetimeIndex,
    months_per_chunk: int,
) -> Iterable[tuple[str, pd.DatetimeIndex]]:
    grouped = [
        (period, pd.DatetimeIndex(values))
        for period, values in dates.to_series().groupby(dates.to_period("M"))
    ]
    chunk_size = max(1, int(months_per_chunk))
    for start in range(0, len(grouped), chunk_size):
        batch = grouped[start : start + chunk_size]
        periods = [period for period, _ in batch]
        if len(periods) == 1:
            label = str(periods[0])
        else:
            label = f"{periods[0]}_to_{periods[-1]}"
        date_values = np.concatenate([month_dates.to_numpy() for _, month_dates in batch])
        yield label, pd.DatetimeIndex(date_values)


def iter_deployment_grid_chunks(
    *,
    config: Any,
    feature_config: dict[str, Any],
    target_config: dict[str, Any],
    split_name: str,
) -> Iterable[DeploymentGridChunk]:
    start_date, end_date = date_range_for_split(config, split_name)
    countries = countries_for_grid(config, feature_config, target_config)
    resolution = float(getattr(config, "deployment_grid_resolution", None) or SPATIAL_COARSENESS)
    shapes_path = feature_config.get("country_shapes_path") or target_config.get("country_shapes_path") or "data/countries"
    logging.info(
        "Building %s deployment grid from %s to %s for countries=%s",
        split_name,
        start_date.date(),
        end_date.date(),
        countries,
    )
    coordinate_bounds = coordinate_bounds_for_grid(config, feature_config, target_config)
    if coordinate_bounds is not None:
        logging.info(
            "Clipping deployment grid to coordinate bounds [min_lat=%s, min_lon=%s, max_lat=%s, max_lon=%s]",
            *coordinate_bounds,
        )
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
    months_per_chunk = getattr(config, "deployment_grid_months_per_chunk", 1)
    months_per_chunk = max(1, int(months_per_chunk)) if months_per_chunk not in {None, ""} else 1

    for country in countries:
        cells = generate_grid_cells_for_geometry(geometries[country], country=country, resolution=resolution)
        before_clip = len(cells)
        cells = _filter_frame_to_bounds(cells, coordinate_bounds, resolution=resolution)
        if coordinate_bounds is not None:
            logging.info(
                "Deployment grid cells for %s clipped from %d to %d by feature coordinate bounds.",
                country,
                before_clip,
                len(cells),
            )
        country_dates = pd.date_range(start_date, end_date, freq="D")
        for period, dates in _month_date_batches(country_dates, months_per_chunk):
            if weighted and getattr(config, "weighted_grid_sample_fraction", None) is not None:
                rows = _sample_cross_cells_dates(
                    cells,
                    pd.DatetimeIndex(dates),
                    labels,
                    country,
                    config=config,
                    split_name=split_name,
                    period=str(period),
                    resolution=resolution,
                )
                yield DeploymentGridChunk(split_name, country, str(period), rows.reset_index(drop=True))
                continue
            for suffix, rows in _iter_cell_date_row_blocks(
                cells,
                pd.DatetimeIndex(dates),
                labels,
                country,
                max_rows,
                resolution=resolution,
            ):
                if weighted:
                    rows = _apply_weighted_sample(
                        rows,
                        config=config,
                        split_name=split_name,
                        country=country,
                        period=f"{period}{suffix}",
                    )
                yield DeploymentGridChunk(split_name, country, f"{period}{suffix}", rows.reset_index(drop=True))
