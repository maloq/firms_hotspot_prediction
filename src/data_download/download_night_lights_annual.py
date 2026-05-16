"""Download annual VIIRS night-light rasters from the Zenodo time series."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


ZENODO_RECORD_ID = "17294744"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ANNUAL_VIIRS_PATTERN = re.compile(
    r"nightlights\.average_viirs\..*?_s_(?P<start_year>\d{4})0101_"
    r"(?P<end_year>\d{4})1231_.*?\.tif$"
)


def _load_record() -> dict:
    with urllib.request.urlopen(ZENODO_API_URL, timeout=60) as response:
        return json.load(response)


def _annual_viirs_files(record: dict) -> dict[int, dict]:
    annual_files: dict[int, dict] = {}
    for item in record.get("files", []):
        key = item.get("key", "")
        match = ANNUAL_VIIRS_PATTERN.match(key)
        if not match:
            continue
        start_year = int(match.group("start_year"))
        end_year = int(match.group("end_year"))
        if start_year != end_year:
            continue
        annual_files[start_year] = item
    if not annual_files:
        raise RuntimeError(f"No annual VIIRS GeoTIFFs found in Zenodo record {ZENODO_RECORD_ID}")
    return annual_files


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_is_complete(path: Path, item: dict, verify_checksum: bool) -> bool:
    expected_size = item.get("size")
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != int(expected_size):
        return False
    checksum = str(item.get("checksum", ""))
    if verify_checksum and checksum.startswith("md5:"):
        return _md5(path) == checksum.split(":", 1)[1]
    return True


def _download_file(
    item: dict,
    output_dir: Path,
    verify_checksum: bool,
    *,
    retries: int,
    retry_sleep_seconds: float,
) -> Path:
    filename = item["key"]
    output_path = output_dir / filename
    if _file_is_complete(output_path, item, verify_checksum):
        print(f"Already present: {output_path}", flush=True)
        return output_path

    url = item["links"]["self"]
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    for attempt in range(1, retries + 1):
        partial_path.unlink(missing_ok=True)
        try:
            print(f"Downloading {filename} (attempt {attempt}/{retries})", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            break
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            partial_path.unlink(missing_ok=True)
            if attempt >= retries:
                raise
            sleep_for = retry_sleep_seconds * attempt
            print(
                f"Download failed for {filename}: {exc}. Retrying in {sleep_for:g}s.",
                flush=True,
            )
            time.sleep(sleep_for)

    partial_path.replace(output_path)
    if not _file_is_complete(output_path, item, verify_checksum=True):
        raise RuntimeError(f"Downloaded file failed size/checksum validation: {output_path}")
    return output_path


def download_annual_viirs(
    output_dir: Path,
    *,
    years: list[int] | None,
    verify_checksum: bool,
    dry_run: bool,
    retries: int,
    retry_sleep_seconds: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = _load_record()
    annual_files = _annual_viirs_files(record)
    selected_years = sorted(annual_files) if years is None else sorted(set(years))
    missing_years = [year for year in selected_years if year not in annual_files]
    if missing_years:
        raise ValueError(
            f"Zenodo record {ZENODO_RECORD_ID} does not contain requested year(s): {missing_years}"
        )

    downloaded = []
    for year in selected_years:
        item = annual_files[year]
        output_path = output_dir / item["key"]
        if dry_run:
            status = "present" if output_path.exists() else "missing"
            print(f"{year}: {status} {output_path}")
            continue
        downloaded_path = _download_file(
            item,
            output_dir,
            verify_checksum,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        downloaded.append(
            {
                "year": year,
                "path": str(downloaded_path),
                "size": int(item.get("size", downloaded_path.stat().st_size)),
                "checksum": item.get("checksum"),
                "source_url": item["links"]["self"],
            }
        )

    manifest = {
        "format": "night_light_annual_viirs_download_manifest_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zenodo_record_id": ZENODO_RECORD_ID,
        "zenodo_doi": record.get("doi"),
        "source_title": record.get("metadata", {}).get("title"),
        "available_years": sorted(annual_files),
        "selected_years": selected_years,
        "latest_available_year": max(annual_files),
        "files": downloaded,
    }
    if not dry_run:
        manifest_path = output_dir / "zenodo_annual_viirs_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"Saved manifest to {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/night_lights/raw"))
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Annual raster years to download. Defaults to all years in the Zenodo record.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip checksum verification for files that are already present.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    download_annual_viirs(
        args.output_dir,
        years=args.years,
        verify_checksum=not args.skip_checksum,
        dry_run=args.dry_run,
        retries=args.retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
