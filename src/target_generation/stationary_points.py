"""Utilities for identifying and filtering stationary MODIS detections.

The MODIS archive occasionally contains artefacts where the same latitude/
longitude pair fires repeatedly for many months. Those points should be
flagged so downstream target generation can exclude them. This module
provides helpers to build that catalogue and to drop the flagged detections
from raw MODIS data frames.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


STATIONARY_FILE_PREFIX = "stationary_points_"
DEFAULT_OUTPUT_DIR = Path("data/modis_stationary_points")
DEFAULT_MIN_MONTHS = 4  # "More than three months" => 4 months or more


@dataclass(frozen=True)
class StationaryPoint:
    """Aggregated information about a stationary MODIS detection."""

    latitude: float
    longitude: float
    unique_months: int
    first_month: str
    last_month: str
    months: str


def _parse_modis_filename(path: Path) -> Tuple[int, str]:
    """Extract the year and country fragment from a MODIS CSV filename."""

    stem = path.stem
    if not stem.startswith("modis_"):
        raise ValueError(f"Unrecognised MODIS filename format: {path.name}")

    remainder = stem[len("modis_") :]
    try:
        year_str, country_fragment = remainder.split("_", 1)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Could not split MODIS filename into year/country: {path.name}") from exc

    return int(year_str), country_fragment


def _month_code_from_datetime(series: pd.Series) -> pd.Series:
    """Return a YYYYMM integer code for each timestamp in *series*."""

    return series.dt.year.astype(int) * 100 + series.dt.month.astype(int)


def _month_code_to_str(month_code: int) -> str:
    """Render a YYYYMM integer back into a "YYYY-MM" string."""

    year = month_code // 100
    month = month_code % 100
    return f"{year:04d}-{month:02d}"


def iter_modis_files(
    modis_dir: Path | str,
    countries: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, Path]]:
    """Yield ``(country, path)`` pairs for MODIS CSV files under *modis_dir*.

    Parameters
    ----------
    modis_dir
        Root directory containing per-year sub-folders with MODIS CSV files.
    countries
        Optional whitelist of country identifiers (matching the filename
        suffix, e.g. ``"Russian_Federation"``).
    """

    root = Path(modis_dir)
    if not root.exists():  # pragma: no cover - configuration issue
        raise FileNotFoundError(f"MODIS directory not found: {root}")

    allowed = set(countries) if countries else None

    for year_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv_path in sorted(year_dir.glob("modis_*_*.csv")):
            try:
                _, country = _parse_modis_filename(csv_path)
            except ValueError:
                continue

            if allowed and country not in allowed:
                continue

            yield country, csv_path


def collect_stationary_points(
    modis_dir: Path | str,
    min_months: int = DEFAULT_MIN_MONTHS,
    countries: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Scan MODIS files and identify stationary latitude/longitude pairs.

    Returns a mapping from country identifier to a dataframe containing the
    stationary points for that country.
    """

    if min_months < 1:
        raise ValueError("min_months must be at least 1")

    coord_months: Dict[Tuple[str, float, float], set[int]] = defaultdict(set)

    if show_progress:
        file_list = list(iter_modis_files(modis_dir, countries=countries))
        iterable: Iterable[Tuple[str, Path]] = tqdm(
            file_list, desc="Scanning MODIS files", unit="file"
        )
    else:
        iterable = iter_modis_files(modis_dir, countries=countries)

    for country, csv_path in iterable:
        df = pd.read_csv(
            csv_path,
            usecols=["latitude", "longitude", "acq_date"],
            dtype={"latitude": float, "longitude": float, "acq_date": "string"},
        )
        df = df.dropna(subset=["latitude", "longitude", "acq_date"])
        if df.empty:
            continue

        acq_dates = pd.to_datetime(df["acq_date"], errors="coerce", utc=False)
        df = df.loc[acq_dates.notna()].copy()
        if df.empty:
            continue

        month_codes = _month_code_from_datetime(acq_dates.loc[df.index])
        df["month_code"] = month_codes.astype(int)
        df = df.drop_duplicates(subset=["latitude", "longitude", "month_code"])

        for row in df.itertuples(index=False):
            key = (country, float(row.latitude), float(row.longitude))
            coord_months[key].add(int(row.month_code))

    records: Dict[str, List[StationaryPoint]] = defaultdict(list)

    for (country, lat, lon), months in coord_months.items():
        if len(months) < min_months:
            continue

        sorted_months = sorted(months)
        month_strings = [_month_code_to_str(code) for code in sorted_months]
        records[country].append(
            StationaryPoint(
                latitude=lat,
                longitude=lon,
                unique_months=len(sorted_months),
                first_month=month_strings[0],
                last_month=month_strings[-1],
                months=";".join(month_strings),
            )
        )

    dataframes: Dict[str, pd.DataFrame] = {}
    for country, points in records.items():
        if not points:
            continue

        df_country = pd.DataFrame([point.__dict__ for point in points])
        df_country.insert(0, "country", country)
        df_country = df_country.sort_values(
            by=["unique_months", "first_month"], ascending=[False, True]
        ).reset_index(drop=True)
        dataframes[country] = df_country

    return dataframes


def save_stationary_points(
    stationary_points: Dict[str, pd.DataFrame],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> List[Path]:
    """Persist stationary-point dataframes to ``output_dir``.

    Returns the list of files written.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: List[Path] = []

    for country, df_country in stationary_points.items():
        if df_country.empty:
            continue

        file_path = output_path / f"{STATIONARY_FILE_PREFIX}{country}.csv"
        df_country.to_csv(file_path, index=False)
        written_files.append(file_path)

    return written_files


def load_stationary_points(
    directory: Path | str = DEFAULT_OUTPUT_DIR,
    countries: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load previously saved stationary-point CSVs from *directory*."""

    base = Path(directory)
    if not base.exists():
        return {}

    allowed = set(countries) if countries else None
    loaded: Dict[str, pd.DataFrame] = {}

    for csv_path in base.glob(f"{STATIONARY_FILE_PREFIX}*.csv"):
        country = csv_path.stem[len(STATIONARY_FILE_PREFIX) :]
        if allowed and country not in allowed:
            continue

        df = pd.read_csv(csv_path)
        loaded[country] = df

    return loaded


def drop_stationary_points(
    df: pd.DataFrame,
    stationary_dir: Path | str = DEFAULT_OUTPUT_DIR,
    country_col: str = "country",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> Tuple[pd.DataFrame, int]:
    """Remove rows whose coordinates match pre-computed stationary points.

    Returns a tuple ``(filtered_df, removed_count)``.
    """

    if df.empty:
        return df.copy(), 0

    base = Path(stationary_dir)
    if not base.exists():
        return df.copy(), 0

    filtered = df.copy()
    total_removed = 0

    if country_col not in filtered.columns:
        raise KeyError(f"Expected column '{country_col}' to filter stationary points.")

    unique_countries = filtered[country_col].dropna().unique().tolist()

    for country in unique_countries:
        csv_path = base / f"{STATIONARY_FILE_PREFIX}{country}.csv"
        if not csv_path.exists():
            continue

        try:
            stationary_df = pd.read_csv(csv_path, usecols=["latitude", "longitude"])
        except ValueError:  # Columns missing or malformed
            stationary_df = pd.read_csv(csv_path)

        stationary_df = stationary_df.dropna(subset=["latitude", "longitude"]).drop_duplicates()
        if stationary_df.empty:
            continue

        merged = (
            filtered.loc[filtered[country_col] == country, [lat_col, lon_col]]
            .reset_index()
            .merge(
                stationary_df,
                how="left",
                on=[lat_col, lon_col],
                indicator=True,
            )
        )

        to_drop = merged.loc[merged["_merge"] == "both", "index"]
        if to_drop.empty:
            continue

        filtered = filtered.drop(index=to_drop)
        total_removed += len(to_drop)

    return filtered.reset_index(drop=True), total_removed


__all__ = [
    "DEFAULT_MIN_MONTHS",
    "DEFAULT_OUTPUT_DIR",
    "StationaryPoint",
    "collect_stationary_points",
    "drop_stationary_points",
    "iter_modis_files",
    "load_stationary_points",
    "save_stationary_points",
]
