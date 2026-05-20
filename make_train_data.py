import argparse
import copy
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from src.feature_generation.make_features import (
    generate_target_data,
    make_features_and_save,
    make_features_from_target_df,
)


DEFAULT_COUNTRIES_TRAIN = [
    "Dem_Rep_Korea",
    "Russian_Federation",
    "Finland",
    "Norway",
    "Sweden",
    "Denmark",
    "Lithuania",
    "Latvia",
    "Estonia",
    "Poland",
    "Czech_Republic",
    "Germany",
    "Hungary",
    "Slovakia",
    "Belarus",
    "Ukraine",
    "Moldova",
    "Romania",
    "Bulgaria",
    "Albania",
    "Montenegro",
    "Macedonia_Former_Yugoslav_Republic_of",
    "Kosovo",
    "Serbia",
    "Croatia",
    "Bosnia_and_Herzegovina",
    "Slovenia",
    "Greece",
    "Turkey",
    "Georgia",
    "Azerbaijan",
    "Armenia",
    "Kazakhstan",
    "Kyrgyzstan",
    "Tajikistan",
    "Mongolia",
    "China",
    "Japan",
    "Republic_of_Korea",
]

TARGET_CACHE_VERSION = "target_v6_city_hard_negatives_no_north_boost"


def _target_config_cache_token() -> str:
    target_config_path = Path("configs/target_config.yaml")
    if not target_config_path.exists():
        return ""
    return target_config_path.read_text(encoding="utf-8")


def _parquet_row_count(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def _country_config(base_config: dict, country: str, target_cache_dir: Path) -> dict:
    config = copy.deepcopy(base_config)
    config["modis_countries"] = [country]
    config["prediction_countries"] = [country]
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = "|".join(
        [
            country,
            str(config.get("target_start_date")),
            str(config.get("target_end_date")),
            str(config.get("target_samples_per_area_per_year")),
            str(config.get("coordinate_bounds")),
            TARGET_CACHE_VERSION,
            _target_config_cache_token(),
        ]
    )
    cache_digest = hashlib.blake2b(cache_key.encode("utf-8"), digest_size=8).hexdigest()
    config["target_cache_path"] = str(target_cache_dir / f"target_data_{country}_{cache_digest}.parquet")
    return config


def _make_country_features(
    base_config: dict,
    country: str,
    output_dir: str,
    target_cache_dir: str,
    climate_cache_dir: str,
    test_mode: bool,
    use_cache: bool,
    force: bool,
) -> tuple[str, str]:
    output_file = Path(output_dir) / f"train_test_features_30d_{country}.parquet"
    if output_file.exists() and not force:
        rows = _parquet_row_count(output_file)
        row_msg = f" ({rows} rows)" if rows is not None else ""
        return country, f"skipped existing {output_file}{row_msg}"

    config = _country_config(base_config, country, Path(target_cache_dir))
    make_features_and_save(
        config,
        str(output_file),
        test_mode=test_mode,
        use_cached_files=use_cache,
        use_cached_target=use_cache,
        cache_dir=climate_cache_dir,
    )

    rows = _parquet_row_count(output_file)
    row_msg = f" with {rows} rows" if rows is not None else ""
    return country, f"saved {output_file}{row_msg}"


def _output_path(output_dir: str, country: str) -> Path:
    return Path(output_dir) / f"train_test_features_30d_{country}.parquet"


def _target_date_range(config: dict, test_mode: bool) -> tuple[str, str]:
    if test_mode:
        return (
            config.get("test_start_date", "2020-01-01"),
            config.get("test_end_date", "2020-12-31"),
        )
    return config["target_start_date"], config["target_end_date"]


def _load_country_target(
    base_config: dict,
    country: str,
    target_cache_dir: str,
    test_mode: bool,
    use_cache: bool,
) -> pd.DataFrame:
    config = _country_config(base_config, country, Path(target_cache_dir))
    start_date, end_date = _target_date_range(config, test_mode)
    df_target = generate_target_data(
        modis_data_path=config["modis_data_path"],
        modis_countries=config["modis_countries"],
        target_samples_per_area_per_year=config["target_samples_per_area_per_year"],
        coordinate_bounds=tuple(config["coordinate_bounds"]),
        start_date=start_date,
        end_date=end_date,
        use_cached=use_cache,
        cache_path=config["target_cache_path"],
        feature_config=config,
    )
    df_target["_source_country"] = country
    return df_target


def _run_single_pass(
    base_config: dict,
    countries: list[str],
    output_dir: str,
    target_cache_dir: str,
    climate_cache_dir: str,
    test_mode: bool,
    use_cache: bool,
    force: bool,
) -> None:
    pending_countries = []
    for country in countries:
        output_file = _output_path(output_dir, country)
        if output_file.exists() and not force:
            rows = _parquet_row_count(output_file)
            row_msg = f" ({rows} rows)" if rows is not None else ""
            print(f"[{country}] skipped existing {output_file}{row_msg}")
        else:
            pending_countries.append(country)

    if not pending_countries:
        print("All requested country outputs already exist.")
        return

    target_frames = []
    try:
        for country in pending_countries:
            print(f"[{country}] preparing target for single-pass feature generation")
            target_frames.append(
                _load_country_target(
                    base_config,
                    country,
                    target_cache_dir,
                    test_mode,
                    use_cache,
                )
            )

        combined_target = pd.concat(target_frames, ignore_index=True)
        combined_target = combined_target.sort_values(
            by=["datetime", "_source_country", "lat_rounded", "lon_rounded"],
            kind="stable",
        ).reset_index(drop=True)
        print(
            f"Single-pass target shape: {combined_target.shape} "
            f"for {len(pending_countries)} countries"
        )

        final_df = make_features_from_target_df(
            base_config,
            combined_target,
            test_mode=test_mode,
            use_cached_files=use_cache,
            cache_dir=climate_cache_dir,
            extra_anchor_cols=["_source_country"],
        )

        if "_source_country" not in final_df.columns:
            raise ValueError("Single-pass features are missing '_source_country'; cannot split output by country.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for country in pending_countries:
            output_file = _output_path(output_dir, country)
            country_df = final_df[final_df["_source_country"] == country].drop(columns=["_source_country"])
            if country_df.empty:
                print(f"[{country}] warning: no rows after feature generation; not saving {output_file}")
                continue
            country_df.to_parquet(output_file)
            print(f"[{country}] saved {output_file} with {len(country_df)} rows")
    except KeyboardInterrupt:
        _stop_after_keyboard_interrupt()


def _stop_after_keyboard_interrupt() -> None:
    print("\nInterrupted by user. Stopping without starting more countries.")
    raise SystemExit(130)


def _run_sequential(worker_args: list[tuple]) -> None:
    try:
        for item in worker_args:
            country, message = _make_country_features(*item)
            print(f"[{country}] {message}")
    except KeyboardInterrupt:
        _stop_after_keyboard_interrupt()


def _run_parallel(worker_args: list[tuple], workers: int) -> None:
    executor = ProcessPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [executor.submit(_make_country_features, *item) for item in worker_args]
        for future in as_completed(futures):
            country, message = future.result()
            print(f"[{country}] {message}")
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        _stop_after_keyboard_interrupt()
    else:
        executor.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-country training feature parquet files.")
    parser.add_argument("--config", default="configs/features_config_30d.yaml")
    parser.add_argument("--output-dir", default="data/saved_features")
    parser.add_argument("--target-cache-dir", default="data/saved_features/target_cache")
    parser.add_argument("--climate-cache-dir", default="data/saved_features/climate_features_cache")
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MAKE_TRAIN_DATA_WORKERS", "1")))
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if parquet files already exist.")
    parser.add_argument("--no-cache", action="store_true", help="Disable target and raw climate matrix caches.")
    parser.add_argument(
        "--per-country-pass",
        action="store_true",
        help="Disable multi-country single-pass feature generation and process countries one by one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as config_file:
        base_config = yaml.safe_load(config_file)

    countries = args.countries or DEFAULT_COUNTRIES_TRAIN
    use_cache = not args.no_cache
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Generating features for {len(countries)} countries")
    print(f"Cache enabled: {use_cache}")
    print(f"Country workers: {args.workers}")

    worker_args = [
        (
            base_config,
            country,
            args.output_dir,
            args.target_cache_dir,
            args.climate_cache_dir,
            args.test_mode,
            use_cache,
            args.force,
        )
        for country in countries
    ]

    if args.workers <= 1 and not args.per_country_pass and len(countries) > 1:
        _run_single_pass(
            base_config,
            countries,
            args.output_dir,
            args.target_cache_dir,
            args.climate_cache_dir,
            args.test_mode,
            use_cache,
            args.force,
        )
        return

    if args.workers <= 1:
        _run_sequential(worker_args)
        return

    _run_parallel(worker_args, args.workers)


if __name__ == "__main__":
    main()
