"""CLI entry-point to catalogue stationary MODIS detections.

Usage example:

    python -m src.target_generation.find_stationary_modis_points \
        --modis-dir data/modis \
        --output-dir data/modis_stationary_points
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stationary_points import (
    DEFAULT_MIN_MONTHS,
    DEFAULT_OUTPUT_DIR,
    collect_stationary_points,
    save_stationary_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify stationary MODIS detections (same lat/lon active for"
            " multiple months) and export the coordinates to CSV files."
        )
    )
    parser.add_argument(
        "--modis-dir",
        default="data/modis",
        help="Root directory containing per-year MODIS CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where stationary-point CSV files will be written.",
    )
    parser.add_argument(
        "--min-months",
        default=DEFAULT_MIN_MONTHS,
        type=int,
        help="Minimum number of distinct months required to flag a point.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        help=(
            "Optional list of country identifiers (matching the MODIS file"
            " suffix, e.g. 'Russian_Federation') to scan."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stationary = collect_stationary_points(
        modis_dir=Path(args.modis_dir),
        min_months=args.min_months,
        countries=args.countries,
        show_progress=not args.no_progress,
    )

    if not stationary:
        print(
            "No stationary detections met the threshold. Adjust --min-months"
            " or ensure the MODIS archive is populated."
        )
        return

    written_files = save_stationary_points(stationary, output_dir=Path(args.output_dir))
    total_points = sum(len(df) for df in stationary.values())

    print(
        f"Identified {total_points} stationary points across {len(stationary)}"
        f" countries (threshold: {args.min_months} months)."
    )

    for file_path in written_files:
        print(f"Saved stationary catalogue to {file_path}")


if __name__ == "__main__":
    main()

