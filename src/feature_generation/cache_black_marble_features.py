"""Precompute Black Marble point samples, then optionally delete cached raw HDF5 years."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from src.data_download.download_black_marble_vnp46a4 import EURASIA_BBOX, _tile_range_for_bbox
from src.feature_generation.prepare_night_light_features import (
    DEFAULT_BLACK_MARBLE_FILTERED_FEATURE_NAME,
    DEFAULT_BLACK_MARBLE_OBSERVATIONS_FEATURE_NAME,
    DEFAULT_BLACK_MARBLE_QUALITY_FEATURE_NAME,
    DEFAULT_BLACK_MARBLE_RADIANCE_FEATURE_NAME,
    RECENT_CACHE_KEY_COLUMNS,
    _available_black_marble_sources,
    _black_marble_request_frame,
    _nearest_available_years,
    _sample_black_marble_features,
)


DEFAULT_FEATURES_GLOBS = (
    "data/saved_features_boost/train_test_features_30d_all_extended_north_14cb_laggedfire.parquet",
    "data/saved_features_boost/train_test_features_30d_all_extended_north_14cb.parquet",
    "data/saved_features/target_cache/target_data_*.parquet",
)


def _expand_feature_paths(paths: list[Path], globs: list[str]) -> list[Path]:
    expanded: list[Path] = []
    expanded.extend(paths)
    for pattern in globs:
        expanded.extend(Path(path) for path in glob.glob(pattern))
    unique = sorted({path for path in expanded if path.exists()})
    if not unique:
        raise FileNotFoundError("No feature parquet files matched --features-path/--features-glob.")
    return unique


def _load_unique_coordinate_years(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = ["lat_rounded", "lon_rounded", "year"]
    for path in paths:
        print(f"Loading coordinate/year columns from {path}")
        frame = pd.read_parquet(path, columns=columns)
        frames.append(frame)
    coords = pd.concat(frames, ignore_index=True)
    coords = coords.dropna(subset=columns)
    coords["year"] = pd.to_numeric(coords["year"], errors="coerce")
    coords = coords.dropna(subset=["year"])
    coords["year"] = coords["year"].round().astype(np.int16)
    coords = coords.drop_duplicates(columns, keep="first").reset_index(drop=True)
    print(f"Loaded {len(coords):,} unique coordinate/year rows.")
    return coords


def _complete_downloaded_years(
    source_dir: Path,
    expected_tiles: set[tuple[int, int]],
) -> dict[int, list[Path]]:
    files_by_year: dict[int, list[Path]] = {}
    try:
        sources = _available_black_marble_sources(source_dir)
    except FileNotFoundError as exc:
        print(str(exc))
        return {}
    for (year, h, v), path in sources.items():
        if (h, v) in expected_tiles:
            files_by_year.setdefault(int(year), []).append(path)
    complete: dict[int, list[Path]] = {}
    for year, files in sorted(files_by_year.items()):
        present_tiles = {
            (int(path.name.split(".h", 1)[1][:2]), int(path.name.split("v", 1)[1][:2]))
            for path in files
        }
        partials = list((source_dir / str(year)).glob("*.partial"))
        if len(present_tiles) == len(expected_tiles) and not partials:
            complete[year] = sorted(files)
    return complete


def _cache_missing_count(cache_path: Path, request: pd.DataFrame, radiance_feature_name: str) -> int:
    if not cache_path.exists():
        return len(request)
    cache = pd.read_parquet(cache_path, columns=[*RECENT_CACHE_KEY_COLUMNS, radiance_feature_name])
    cache = cache.drop_duplicates(RECENT_CACHE_KEY_COLUMNS, keep="last")
    resolved = request[RECENT_CACHE_KEY_COLUMNS].merge(
        cache,
        on=RECENT_CACHE_KEY_COLUMNS,
        how="left",
        sort=False,
    )
    return int(resolved[radiance_feature_name].isna().sum())


def _delete_files(files: list[Path]) -> int:
    freed = 0
    for path in files:
        size = path.stat().st_size
        path.unlink()
        freed += size
    return freed


def cache_black_marble_features(
    *,
    feature_paths: list[Path],
    source_dir: Path,
    cache_path: Path,
    years: list[int] | None,
    available_years: list[int],
    bbox: tuple[float, float, float, float],
    delete_raw: bool,
    dry_run: bool,
) -> None:
    expected_tiles = set(_tile_range_for_bbox(*bbox))
    complete_years = _complete_downloaded_years(source_dir, expected_tiles)
    if years is not None:
        process_years = [year for year in years if year in complete_years]
        skipped = sorted(set(years) - set(process_years))
        if skipped:
            print(f"Skipping incomplete/unavailable years: {skipped}")
    else:
        process_years = sorted(complete_years)
    if not process_years:
        print("No complete downloaded years are ready to cache.")
        return

    coords = _load_unique_coordinate_years(feature_paths)
    chosen_source_years = _nearest_available_years(coords["year"].to_numpy(), available_years)
    coords = coords.assign(_black_marble_source_year=chosen_source_years)

    for source_year in process_years:
        year_coords = coords.loc[coords["_black_marble_source_year"] == source_year].copy()
        raw_files = complete_years[source_year]
        raw_gib = sum(path.stat().st_size for path in raw_files) / 1024**3
        if year_coords.empty:
            print(f"{source_year}: no requested training coordinates; raw files kept.")
            continue

        sample_coords = year_coords[["lat_rounded", "lon_rounded"]].to_numpy(dtype=np.float64).T
        sample_years = np.full(sample_coords.shape[1], source_year, dtype=np.int16)
        request = _black_marble_request_frame(sample_coords[0], sample_coords[1], sample_years)
        request = request.drop_duplicates(RECENT_CACHE_KEY_COLUMNS, keep="first").reset_index(drop=True)

        missing_before = _cache_missing_count(cache_path, request, DEFAULT_BLACK_MARBLE_RADIANCE_FEATURE_NAME)
        print(
            f"{source_year}: {len(request):,} unique point samples requested; "
            f"{missing_before:,} missing from cache; raw size {raw_gib:.2f} GiB."
        )
        if dry_run:
            continue

        if missing_before:
            _sample_black_marble_features(
                coords=sample_coords,
                years=sample_years,
                source_dir=source_dir,
                radiance_sds="NearNadir_Composite_Snow_Free",
                quality_sds="NearNadir_Composite_Snow_Free_Quality",
                observations_sds="NearNadir_Composite_Snow_Free_Num",
                radiance_feature_name=DEFAULT_BLACK_MARBLE_RADIANCE_FEATURE_NAME,
                quality_feature_name=DEFAULT_BLACK_MARBLE_QUALITY_FEATURE_NAME,
                observations_feature_name=DEFAULT_BLACK_MARBLE_OBSERVATIONS_FEATURE_NAME,
                filtered_feature_name=DEFAULT_BLACK_MARBLE_FILTERED_FEATURE_NAME,
                quality_keep_values=(0,),
                cache_path=cache_path,
            )

        missing_after = _cache_missing_count(cache_path, request, DEFAULT_BLACK_MARBLE_RADIANCE_FEATURE_NAME)
        if missing_after:
            raise RuntimeError(
                f"{source_year}: cache verification failed; {missing_after:,} samples are still missing. "
                "Raw files were not deleted."
            )
        print(f"{source_year}: cache verified.")

        if delete_raw:
            freed = _delete_files(raw_files)
            print(f"{source_year}: deleted {len(raw_files)} raw HDF5 files, freed {freed / 1024**3:.2f} GiB.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, action="append", default=[])
    parser.add_argument("--features-glob", action="append", default=list(DEFAULT_FEATURES_GLOBS))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/night_lights/black_marble_vnp46a4"),
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/night_lights/black_marble_vnp46a4_point_cache.parquet"),
    )
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--available-years", nargs="+", type=int, default=list(range(2012, 2025)))
    parser.add_argument("--bbox", nargs=4, type=float, default=EURASIA_BBOX)
    parser.add_argument("--delete-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feature_paths = _expand_feature_paths(args.features_path, args.features_glob)
    cache_black_marble_features(
        feature_paths=feature_paths,
        source_dir=args.source_dir,
        cache_path=args.cache_path,
        years=args.years,
        available_years=sorted(set(args.available_years)),
        bbox=tuple(args.bbox),
        delete_raw=args.delete_raw,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
