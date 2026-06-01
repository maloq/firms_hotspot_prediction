"""Download NASA Black Marble VNP46A4 annual HDF5 tiles from LAADS."""

from __future__ import annotations

import argparse
import concurrent.futures
import html.parser
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


LAADS_ARCHIVE_SET = "5200"
PRODUCT = "VNP46A4"
DAY_OF_YEAR = "001"
DEFAULT_YEARS = list(range(2012, 2025))
DEFAULT_BBOX = (35.0, 6.0, 75.0, 179.0)  # min_lat, min_lon, max_lat, max_lon
# VNP46A4 listings do not consistently include the polar v00 row (80-90N).
# This covers the available northern rows, including 70-80N for our 75N target cap.
NORTHERN_HEMISPHERE_BBOX = (0.0001, -180.0, 80.0, 180.0)
# Eurasia plus all mainland/island Europe west to Iceland, avoiding the missing 80-90N row.
EURASIA_BBOX = (34.0, -25.0, 80.0, 180.0)
OLD_ZENODO_PATTERN = "nightlights.average_viirs.v21_m_500m_s_*_go_epsg4326_v20250904.tif"
VNP46A4_FILE_PATTERN = re.compile(
    r"VNP46A4\.A(?P<year>\d{4})001\.h(?P<h>\d{2})v(?P<v>\d{2})\."
    r"(?P<collection>\d{3})\..*?\.h5$"
)


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def _auth_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _archive_listing_url(year: int) -> str:
    return (
        "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives/"
        f"allData/{LAADS_ARCHIVE_SET}/{PRODUCT}/{year}/{DAY_OF_YEAR}/"
    )


def _download_url(filename: str) -> str:
    return f"https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads/{PRODUCT}/{filename}"


def _read_url(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise PermissionError(
                "LAADS denied access. Export EARTHDATA_TOKEN or LAADS_TOKEN with a valid "
                "Earthdata token before downloading."
            ) from exc
        raise


def _list_year_files(year: int) -> list[str]:
    body = _read_url(_archive_listing_url(year), {"User-Agent": "Mozilla/5.0"}).decode(
        "utf-8",
        errors="ignore",
    )
    parser = LinkParser()
    parser.feed(body)
    names: list[str] = []
    for link in parser.links:
        name = Path(urllib.parse.urlparse(link).path).name
        if VNP46A4_FILE_PATTERN.match(name):
            names.append(name)
    if not names:
        raise FileNotFoundError(f"No {PRODUCT} files found in LAADS listing for {year}")
    return sorted(set(names))


def _tile_range_for_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> list[tuple[int, int]]:
    h_min = int((min_lon + 180.0) // 10.0)
    h_max = int((max_lon + 180.0) // 10.0)
    v_min = int((90.0 - max_lat) // 10.0)
    v_max = int((90.0 - min_lat) // 10.0)
    h_min = max(0, min(35, h_min))
    h_max = max(0, min(35, h_max))
    v_min = max(0, min(17, v_min))
    v_max = max(0, min(17, v_max))
    return [(h, v) for h in range(h_min, h_max + 1) for v in range(v_min, v_max + 1)]


def _download_file(
    url: str,
    output_path: Path,
    headers: dict[str, str],
    retries: int,
    timeout: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already present: {output_path}", flush=True)
        return output_path

    partial_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.partial")
    for attempt in range(1, retries + 1):
        partial_path.unlink(missing_ok=True)
        try:
            print(f"Downloading {output_path.name} (attempt {attempt}/{retries})", flush=True)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response, partial_path.open("wb") as handle:
                final_url = response.geturl()
                if "urs.earthdata.nasa.gov" in final_url:
                    raise PermissionError(
                        "LAADS redirected to Earthdata Login. The token was missing, invalid, "
                        "or not authorized for LAADS."
                    )
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            break
        except PermissionError:
            partial_path.unlink(missing_ok=True)
            raise
        except Exception:
            partial_path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

    partial_path.replace(output_path)
    return output_path


def _probe_download_access(url: str, headers: dict[str, str]) -> None:
    probe_headers = {**headers, "Range": "bytes=0-15"}
    request = urllib.request.Request(url, headers=probe_headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            final_url = response.geturl()
            chunk = response.read(16)
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise PermissionError(
                "LAADS requires an Earthdata Download token for VNP46A4 HDF5 downloads. "
                "Export EARTHDATA_TOKEN or LAADS_TOKEN and rerun."
            ) from exc
        raise

    if "urs.earthdata.nasa.gov" in final_url or not chunk.startswith(b"\x89HDF"):
        raise PermissionError(
            "LAADS requires an Earthdata Download token for VNP46A4 HDF5 downloads. "
            "Export EARTHDATA_TOKEN or LAADS_TOKEN and rerun."
        )


def _delete_old_zenodo_data(old_dir: Path) -> list[str]:
    deleted: list[str] = []
    for path in sorted(old_dir.glob(OLD_ZENODO_PATTERN)):
        path.unlink()
        deleted.append(str(path))
    for path in [
        old_dir / "zenodo_annual_viirs_manifest.json",
        old_dir.parent / "recent_radiance_point_cache.parquet",
    ]:
        if path.exists():
            path.unlink()
            deleted.append(str(path))
    return deleted


def download_black_marble_vnp46a4(
    *,
    output_dir: Path,
    years: list[int],
    bbox: tuple[float, float, float, float],
    token: str | None,
    retries: int,
    workers: int,
    download_timeout: int,
    delete_old_dir: Path | None,
) -> dict:
    tiles = _tile_range_for_bbox(*bbox)
    wanted_tiles = {f"h{h:02d}v{v:02d}" for h, v in tiles}
    headers = _auth_headers(token)
    download_jobs: list[tuple[int, int, int, str, str, Path]] = []

    for year in years:
        names = _list_year_files(year)
        selected = [
            name
            for name in names
            if any(tile_id in name for tile_id in wanted_tiles)
        ]
        missing_tiles = sorted(
            tile_id for tile_id in wanted_tiles if not any(tile_id in name for name in selected)
        )
        if missing_tiles:
            raise FileNotFoundError(f"{year} LAADS listing is missing tiles: {missing_tiles}")

        for name in selected:
            match = VNP46A4_FILE_PATTERN.match(name)
            assert match is not None
            h = int(match.group("h"))
            v = int(match.group("v"))
            path = output_dir / str(year) / name
            download_jobs.append((year, h, v, _download_url(name), name, path))

    if download_jobs:
        first_missing_url = next(
            (url for _year, _h, _v, url, _name, path in download_jobs if not path.exists()),
            None,
        )
        if first_missing_url is not None:
            _probe_download_access(first_missing_url, headers)

    def _run_job(job: tuple[int, int, int, str, str, Path]) -> dict:
        year, h, v, url, _name, path = job
        _download_file(url, path, headers, retries, download_timeout)
        return {"year": year, "h": h, "v": v, "path": str(path)}

    downloaded: list[dict] = []
    if workers <= 1:
        for job in download_jobs:
            downloaded.append(_run_job(job))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(_run_job, job): job for job in download_jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                downloaded.append(future.result())
    downloaded.sort(key=lambda item: (item["year"], item["h"], item["v"], item["path"]))

    expected_count = len(years) * len(tiles)
    if len(downloaded) < expected_count:
        raise RuntimeError(
            f"Refusing cleanup because only {len(downloaded)} of {expected_count} expected "
            "Black Marble files were downloaded."
        )
    deleted: list[str] = []
    if delete_old_dir is not None:
        deleted = _delete_old_zenodo_data(delete_old_dir)

    manifest = {
        "format": "black_marble_vnp46a4_download_manifest_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "product": PRODUCT,
        "archive_set": LAADS_ARCHIVE_SET,
        "years": years,
        "bbox": list(bbox),
        "tiles": [f"h{h:02d}v{v:02d}" for h, v in tiles],
        "workers": int(workers),
        "used_bearer_token": bool(token),
        "download_endpoint": "earthdata_cloud_https",
        "files": downloaded,
        "deleted_old_files": deleted,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "black_marble_vnp46a4_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved manifest to {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/night_lights/black_marble_vnp46a4"))
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        default=DEFAULT_BBOX,
    )
    parser.add_argument(
        "--northern-hemisphere",
        action="store_true",
        help=(
            "Download available tiles north of the equator through 80N. "
            "Overrides --bbox with "
            f"{NORTHERN_HEMISPHERE_BBOX}."
        ),
    )
    parser.add_argument(
        "--eurasia",
        action="store_true",
        help=(
            "Download Eurasia including all Europe west to Iceland, through 80N. "
            f"Overrides --bbox with {EURASIA_BBOX}."
        ),
    )
    parser.add_argument("--token", default=os.environ.get("LAADS_TOKEN") or os.environ.get("EARTHDATA_TOKEN"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--download-timeout", type=int, default=600)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel file downloads. Use 1 for serial downloads.",
    )
    parser.add_argument(
        "--delete-old-dir",
        type=Path,
        default=None,
        help="Delete old Zenodo average_viirs files from this directory after replacement verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.northern_hemisphere and args.eurasia:
        raise SystemExit("--northern-hemisphere and --eurasia are mutually exclusive.")
    if args.eurasia:
        bbox = EURASIA_BBOX
    elif args.northern_hemisphere:
        bbox = NORTHERN_HEMISPHERE_BBOX
    else:
        bbox = tuple(args.bbox)
    try:
        download_black_marble_vnp46a4(
            output_dir=args.output_dir,
            years=sorted(set(args.years)),
            bbox=bbox,
            token=args.token,
            retries=args.retries,
            workers=max(1, int(args.workers)),
            download_timeout=max(1, int(args.download_timeout)),
            delete_old_dir=args.delete_old_dir,
        )
    except PermissionError as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
