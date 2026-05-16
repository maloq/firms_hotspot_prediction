"""Add nearest-available annual night-light radiance to an existing feature parquet."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.getcwd())

from src.feature_generation.prepare_night_light_features import (
    DEFAULT_RECENT_FEATURE_NAME,
    get_recent_night_light_radiance_for_coords,
)


def _resolve_output_path(features_path: Path, output_path: Path | None, overwrite: bool) -> Path:
    if output_path is not None:
        return output_path
    if overwrite:
        return features_path
    return features_path.with_name(f"{features_path.stem}_with_recent_night_light{features_path.suffix}")


def add_recent_night_light_feature(
    features_path: Path,
    output_path: Path | None,
    *,
    annual_source_dir: Path,
    feature_name: str,
    source_glob: str,
    recent_cache_path: Path | None,
    overwrite: bool,
) -> Path:
    resolved_output = _resolve_output_path(features_path, output_path, overwrite)
    if not features_path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {features_path}")
    if resolved_output.exists() and resolved_output != features_path and not overwrite:
        raise FileExistsError(f"{resolved_output} already exists; pass --overwrite to replace it")

    coordinate_columns = ["lat_rounded", "lon_rounded", "year"]
    print(f"Loading coordinates and years from {features_path}")
    coords_years = pd.read_parquet(features_path, columns=coordinate_columns)
    coords = coords_years[["lat_rounded", "lon_rounded"]].to_numpy().T
    recent = get_recent_night_light_radiance_for_coords(
        coords=coords,
        years=coords_years["year"].to_numpy(),
        annual_source_dir=annual_source_dir,
        recent_feature_name=feature_name,
        source_glob=source_glob,
        cache_path=recent_cache_path,
    )

    print(f"Loading full feature table from {features_path}")
    full = pd.read_parquet(features_path)
    full[feature_name] = recent[feature_name].astype("float32").to_numpy()

    if resolved_output == features_path:
        tmp_path = features_path.with_suffix(features_path.suffix + ".tmp")
        print(f"Writing temporary parquet to {tmp_path}")
        full.to_parquet(tmp_path)
        tmp_path.replace(features_path)
        print(f"Updated {features_path}")
    else:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing updated parquet to {resolved_output}")
        full.to_parquet(resolved_output)
    return resolved_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/saved_features_boost/train_test_features_30d_all_extended_north_14cb.parquet"),
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--annual-source-dir", type=Path, default=Path("data/night_lights/raw"))
    parser.add_argument("--feature-name", default=DEFAULT_RECENT_FEATURE_NAME)
    parser.add_argument("--source-glob", default="*.tif")
    parser.add_argument(
        "--recent-cache-path",
        type=Path,
        default=Path("data/night_lights/recent_radiance_point_cache.parquet"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Update --features-path in place when --output-path is not set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    add_recent_night_light_feature(
        args.features_path,
        args.output_path,
        annual_source_dir=args.annual_source_dir,
        feature_name=args.feature_name,
        source_glob=args.source_glob,
        recent_cache_path=args.recent_cache_path,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
