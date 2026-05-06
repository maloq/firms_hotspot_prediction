#!/usr/bin/env python3
"""Download ERA5 daily statistics needed for full training-domain parity.

The revision failure note records that processed ERA5 zarr files only cover
2009-2024 and a narrower domain. This script fills the raw ERA5 daily-stat files
needed to rebuild those zarrs over the full model window/domain.

By default it only prints the missing/insufficient files. Add ``--execute`` to
submit CDS requests. Planned downloads are ordered by repair priority: truly
missing files first, spatially narrow files second, and other insufficient files
after that.

Example:
    python scripts/download_missing_era5_daily.py --execute

The default request matches the ECMWF/SEAS5 climate-feature domain:
    latitude 35..75, longitude 1..179, years 2000..2025

The 2000 lead-in year is included because 2001 feature rows need climate
history windows.
"""

from __future__ import annotations

import argparse
import calendar
import dataclasses
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Iterable

import xarray as xr


DATASET = "derived-era5-single-levels-daily-statistics"
DEFAULT_RAW_ROOT = Path("/home/ids/vmorozov/era5")
DEFAULT_AREA = (75.0, 1.0, 35.0, 179.0)  # north, west, south, east
DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2025
DEFAULT_TIME_ZONE = "utc+00:00"
DEFAULT_FREQUENCY = "1_hourly"
COMMON_VARIABLES = ("t2m", "d2m", "tp", "stl1")


@dataclasses.dataclass(frozen=True)
class Era5Variable:
    key: str
    cds_name: str
    short_name: str
    daily_statistic: str

    @property
    def output_dir_name(self) -> str:
        return self.cds_name


VARIABLES: dict[str, Era5Variable] = {
    "t2m": Era5Variable(
        key="t2m",
        cds_name="2m_temperature",
        short_name="t2m",
        daily_statistic="daily_mean",
    ),
    "d2m": Era5Variable(
        key="d2m",
        cds_name="2m_dewpoint_temperature",
        short_name="d2m",
        daily_statistic="daily_mean",
    ),
    "tp": Era5Variable(
        key="tp",
        cds_name="total_precipitation",
        short_name="tp",
        daily_statistic="daily_sum",
    ),
    "stl1": Era5Variable(
        key="stl1",
        cds_name="soil_temperature_level_1",
        short_name="stl1",
        daily_statistic="daily_mean",
    ),
    "msl": Era5Variable(
        key="msl",
        cds_name="mean_sea_level_pressure",
        short_name="msl",
        daily_statistic="daily_mean",
    ),
}


@dataclasses.dataclass(frozen=True)
class Coverage:
    path: Path
    valid: bool
    reason: str
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    n_times: int | None = None


@dataclasses.dataclass(frozen=True)
class DownloadTask:
    variable: Era5Variable
    year: int
    output_path: Path
    reason: str

    def request(self, area: tuple[float, float, float, float], use_nocache: bool) -> dict[str, object]:
        request: dict[str, object] = {
            "product_type": "reanalysis",
            "variable": [self.variable.cds_name],
            "year": str(self.year),
            "month": [f"{month:02d}" for month in range(1, 13)],
            "day": [f"{day:02d}" for day in range(1, 32)],
            "daily_statistic": self.variable.daily_statistic,
            "time_zone": DEFAULT_TIME_ZONE,
            "frequency": DEFAULT_FREQUENCY,
            "area": list(area),
        }
        if use_nocache:
            request["nocache"] = str(time.time_ns())
        return request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or download missing ERA5 daily-statistic NetCDF files needed "
            "to rebuild full-domain ERA5 zarr features."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Existing ERA5 raw root used for coverage checks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Where new files are written. Defaults to --raw-root. Files are "
            "placed under <output-root>/<cds_variable>/."
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="First year to require. Default includes a 2000 lead-in for 2001 features.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Last year to require.",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=DEFAULT_AREA,
        help="CDS area in north west south east order.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=sorted(VARIABLES),
        default=list(COMMON_VARIABLES),
        help="ERA5 variables to require. Defaults to the common ERA5/SEAS5 schema.",
    )
    parser.add_argument(
        "--include-msl",
        action="store_true",
        help="Also require mean sea-level pressure, which is present locally but not in the common schema.",
    )
    parser.add_argument(
        "--extension",
        default=".nc",
        choices=(".nc", ".grib"),
        help="Extension for newly written files. The daily-stat product is NetCDF; .grib is legacy naming only.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit CDS requests. Without this flag the script is a dry-run planner.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the planned output path if it already exists but failed coverage checks.",
    )
    parser.add_argument(
        "--use-cds-cache",
        action="store_true",
        help=(
            "Do not add a unique nocache key. The default avoids stale CDS "
            "area-extraction cache after the February 2026 server-side change."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path for the planned/downloaded tasks.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def validate_area(area: tuple[float, float, float, float]) -> None:
    north, west, south, east = area
    if south > north:
        raise ValueError(f"Area must be north west south east; got south {south} > north {north}.")
    if west > east:
        raise ValueError(f"Area must not cross the antimeridian; got west {west} > east {east}.")


def selected_variables(names: Iterable[str], include_msl: bool) -> list[Era5Variable]:
    keys = list(dict.fromkeys(names))
    if include_msl and "msl" not in keys:
        keys.append("msl")
    return [VARIABLES[key] for key in keys]


def expected_days(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def candidate_paths(raw_root: Path, output_root: Path, variable: Era5Variable, year: int, extension: str) -> list[Path]:
    variable_dirs = [
        raw_root / variable.output_dir_name,
        raw_root / variable.key,
        output_root / variable.output_dir_name,
        output_root / variable.key,
    ]
    names = [
        f"{variable.cds_name}_{year}{extension}",
        f"{variable.cds_name}_{year}.nc",
        f"{variable.cds_name}_{year}.grib",
        f"{variable.key}_{year}{extension}",
        f"{variable.key}_{year}.nc",
        f"{variable.key}_{year}.grib",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for directory in variable_dirs:
        for name in names:
            path = directory / name
            if path not in seen:
                paths.append(path)
                seen.add(path)
        if directory.exists():
            for path in sorted(directory.glob(f"*{year}*.nc")) + sorted(directory.glob(f"*{year}*.grib")):
                if path not in seen:
                    paths.append(path)
                    seen.add(path)
    return paths


def planned_output_path(output_root: Path, variable: Era5Variable, year: int, extension: str) -> Path:
    return output_root / variable.output_dir_name / f"{variable.cds_name}_{year}{extension}"


def inspect_coverage(path: Path, year: int, area: tuple[float, float, float, float]) -> Coverage:
    if not path.exists():
        return Coverage(path=path, valid=False, reason="missing")
    if path.stat().st_size == 0:
        return Coverage(path=path, valid=False, reason="empty file")

    north, west, south, east = area
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception as exc:
        return Coverage(path=path, valid=False, reason=f"not readable as NetCDF: {exc}")

    try:
        lat_name = "latitude" if "latitude" in ds.coords else "lat" if "lat" in ds.coords else None
        lon_name = "longitude" if "longitude" in ds.coords else "lon" if "lon" in ds.coords else None
        time_name = (
            "valid_time"
            if "valid_time" in ds.coords
            else "time"
            if "time" in ds.coords
            else None
        )
        if lat_name is None or lon_name is None or time_name is None:
            return Coverage(path=path, valid=False, reason="missing latitude/longitude/time coordinates")

        lat_min = float(ds[lat_name].min())
        lat_max = float(ds[lat_name].max())
        lon_min = float(ds[lon_name].min())
        lon_max = float(ds[lon_name].max())
        n_times = int(ds[time_name].size)

        if lat_min > south or lat_max < north or lon_min > west or lon_max < east:
            reason = (
                f"domain too narrow: lat {lat_min:g}..{lat_max:g}, "
                f"lon {lon_min:g}..{lon_max:g}"
            )
            return Coverage(path, False, reason, lat_min, lat_max, lon_min, lon_max, n_times)
        if n_times < expected_days(year):
            reason = f"time coverage too short: {n_times} < {expected_days(year)} days"
            return Coverage(path, False, reason, lat_min, lat_max, lon_min, lon_max, n_times)

        return Coverage(path, True, "ok", lat_min, lat_max, lon_min, lon_max, n_times)
    finally:
        ds.close()


def best_existing_coverage(paths: list[Path], year: int, area: tuple[float, float, float, float]) -> Coverage:
    failures: list[Coverage] = []
    for path in paths:
        coverage = inspect_coverage(path, year, area)
        if coverage.valid:
            return coverage
        if coverage.reason != "missing":
            failures.append(coverage)
    if failures:
        return failures[0]
    return Coverage(path=paths[0], valid=False, reason="missing")


def repair_priority(reason: str) -> int:
    if reason == "missing":
        return 0
    if reason.startswith("domain too narrow"):
        return 1
    return 2


def build_plan(args: argparse.Namespace) -> list[DownloadTask]:
    output_root = args.output_root or args.raw_root
    area = tuple(float(value) for value in args.area)
    validate_area(area)

    variables = selected_variables(args.variables, args.include_msl)
    variable_rank = {variable.key: index for index, variable in enumerate(variables)}
    tasks: list[DownloadTask] = []
    for variable in variables:
        for year in range(args.start_year, args.end_year + 1):
            paths = candidate_paths(args.raw_root, output_root, variable, year, args.extension)
            coverage = best_existing_coverage(paths, year, area)
            output_path = planned_output_path(output_root, variable, year, args.extension)
            if coverage.valid and not args.overwrite:
                logging.debug("OK %s %s via %s", variable.key, year, coverage.path)
                continue
            if output_path.exists() and not args.overwrite:
                logging.warning(
                    "Planned output exists but is insufficient; skipping without --overwrite: %s (%s)",
                    output_path,
                    coverage.reason,
                )
                continue
            tasks.append(DownloadTask(variable=variable, year=year, output_path=output_path, reason=coverage.reason))
    tasks.sort(key=lambda task: (repair_priority(task.reason), variable_rank[task.variable.key], task.year))
    return tasks


def retrieve_to_file(client: object, task: DownloadTask, request: dict[str, object], target: Path) -> None:
    result = client.retrieve(DATASET, request)
    if hasattr(result, "download"):
        result.download(str(target))
    elif not target.exists():
        raise RuntimeError("CDS retrieve did not create an output file and returned no downloadable result")


def materialize_download(temp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(temp_path):
        with zipfile.ZipFile(temp_path) as archive:
            nc_names = [name for name in archive.namelist() if name.endswith(".nc")]
            if not nc_names:
                raise RuntimeError(f"No .nc file found inside downloaded archive {temp_path}")
            if len(nc_names) > 1:
                logging.warning("Archive has %d NetCDF files; using %s", len(nc_names), nc_names[0])
            with archive.open(nc_names[0]) as src, final_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        temp_path.unlink(missing_ok=True)
    else:
        os.replace(temp_path, final_path)


def download_tasks(tasks: list[DownloadTask], area: tuple[float, float, float, float], use_nocache: bool) -> None:
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("cdsapi is required for --execute. Install requirements.txt first.") from exc

    client = cdsapi.Client()
    for index, task in enumerate(tasks, start=1):
        logging.info(
            "[%d/%d] Downloading %s %s -> %s (%s)",
            index,
            len(tasks),
            task.variable.key,
            task.year,
            task.output_path,
            task.reason,
        )
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        request = task.request(area, use_nocache=use_nocache)
        with tempfile.NamedTemporaryFile(
            prefix=f".{task.output_path.name}.",
            suffix=".download",
            dir=task.output_path.parent,
            delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)
        try:
            retrieve_to_file(client, task, request, temp_path)
            materialize_download(temp_path, task.output_path)
            coverage = inspect_coverage(task.output_path, task.year, area)
            if not coverage.valid:
                raise RuntimeError(f"downloaded file failed coverage check: {coverage.reason}")
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise


def write_manifest(path: Path, tasks: list[DownloadTask], area: tuple[float, float, float, float], executed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": DATASET,
        "area_north_west_south_east": list(area),
        "executed": executed,
        "task_count": len(tasks),
        "tasks": [
            {
                "variable": task.variable.key,
                "cds_variable": task.variable.cds_name,
                "daily_statistic": task.variable.daily_statistic,
                "year": task.year,
                "output_path": str(task.output_path),
                "reason": task.reason,
            }
            for task in tasks
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    area = tuple(float(value) for value in args.area)

    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    tasks = build_plan(args)
    mode = "download" if args.execute else "dry-run"
    logging.info("%s plan has %d task(s)", mode, len(tasks))

    for task in tasks:
        print(
            f"{task.variable.key:4s} {task.year} -> {task.output_path} "
            f"({task.variable.daily_statistic}; {task.reason})"
        )

    if args.manifest:
        write_manifest(args.manifest, tasks, area, executed=args.execute)
        logging.info("Wrote manifest: %s", args.manifest)

    if not args.execute:
        if tasks:
            print("\nDry run only. Re-run with --execute to submit CDS requests.")
        return 0

    if not tasks:
        logging.info("Nothing to download.")
        return 0

    download_tasks(tasks, area, use_nocache=not args.use_cds_cache)
    logging.info("Finished %d download(s).", len(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
