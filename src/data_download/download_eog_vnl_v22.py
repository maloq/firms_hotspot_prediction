"""Download authenticated EOG Annual VNL V2.2 median_masked/cf_cvg GeoTIFFs.

The EOG archive requires an authenticated session. Pass a browser cookie header
or a bearer token; this script deliberately refuses to delete older data unless
the requested replacement years are present.
"""

from __future__ import annotations

import argparse
import gzip
import html.parser
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://eogdata.mines.edu/nighttime_light/annual/v22/"
PRODUCT_TOKENS = ("median_masked", "cf_cvg")
VNL_YEAR_PATTERN = re.compile(r".*?VNL.*?_(?P<year>\d{4})_.*?\.(?P<product>median_masked|cf_cvg)\.dat\.tif(?:\.gz)?$")
OLD_ZENODO_PATTERN = "nightlights.average_viirs.v21_m_500m_s_*_go_epsg4326_v20250904.tif"


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


def _request_headers(cookie: str | None, bearer_token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    if cookie:
        headers["Cookie"] = cookie
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    return headers


def _read_url(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise PermissionError(
                "EOG denied access. Export EOG_COOKIE from an authenticated browser "
                "session or pass --bearer-token/--cookie."
            ) from exc
        raise


def _list_links(base_url: str, headers: dict[str, str]) -> list[str]:
    body = _read_url(base_url, headers).decode("utf-8", errors="ignore")
    if "login" in body.lower() and "password" in body.lower():
        raise PermissionError(
            "EOG returned a login page. Provide an authenticated cookie or bearer token."
        )
    parser = LinkParser()
    parser.feed(body)
    return [urllib.parse.urljoin(base_url, link) for link in parser.links]


def _discover_files(
    base_url: str,
    headers: dict[str, str],
    years: list[int],
) -> dict[tuple[int, str], str]:
    links = _list_links(base_url, headers)
    discovered: dict[tuple[int, str], str] = {}
    for link in links:
        filename = Path(urllib.parse.urlparse(link).path).name
        match = VNL_YEAR_PATTERN.match(filename)
        if not match:
            continue
        year = int(match.group("year"))
        product = match.group("product")
        if year in years:
            discovered[(year, product)] = link
    return discovered


def _download(url: str, output_path: Path, headers: dict[str, str], retries: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_path = output_path.with_suffix("") if output_path.suffix == ".gz" else output_path
    if final_path.exists() and final_path.stat().st_size > 0:
        print(f"Already present: {final_path}", flush=True)
        return final_path

    partial_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.partial")
    for attempt in range(1, retries + 1):
        partial_path.unlink(missing_ok=True)
        try:
            print(f"Downloading {url} (attempt {attempt}/{retries})", flush=True)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response, partial_path.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            break
        except Exception:
            partial_path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)

    if output_path.suffix == ".gz":
        with gzip.open(partial_path, "rb") as src, final_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        partial_path.unlink(missing_ok=True)
        return final_path

    partial_path.replace(final_path)
    return final_path


def _verify_replacement(output_dir: Path, years: list[int]) -> None:
    missing: list[str] = []
    for year in years:
        for token in PRODUCT_TOKENS:
            if not list(output_dir.glob(f"*{year}*{token}*.tif")):
                missing.append(f"{year}:{token}")
    if missing:
        raise FileNotFoundError(
            "Refusing cleanup because replacement files are missing: "
            + ", ".join(missing)
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


def download_eog_vnl_v22(
    *,
    base_url: str,
    output_dir: Path,
    years: list[int],
    cookie: str | None,
    bearer_token: str | None,
    retries: int,
    delete_old_dir: Path | None,
) -> dict:
    headers = _request_headers(cookie, bearer_token)
    discovered = _discover_files(base_url, headers, years)
    missing = [
        (year, product)
        for year in years
        for product in PRODUCT_TOKENS
        if (year, product) not in discovered
    ]
    if missing:
        raise FileNotFoundError(f"EOG listing did not contain requested files: {missing}")

    files: list[dict] = []
    for year in years:
        for product in PRODUCT_TOKENS:
            url = discovered[(year, product)]
            filename = Path(urllib.parse.urlparse(url).path).name
            path = _download(url, output_dir / filename, headers, retries)
            files.append({"year": year, "product": product, "path": str(path), "source_url": url})

    _verify_replacement(output_dir, years)
    deleted: list[str] = []
    if delete_old_dir is not None:
        deleted = _delete_old_zenodo_data(delete_old_dir)

    manifest = {
        "format": "eog_vnl_v22_night_light_download_manifest_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "years": years,
        "products": list(PRODUCT_TOKENS),
        "files": files,
        "deleted_old_files": deleted,
    }
    manifest_path = output_dir / "eog_vnl_v22_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved manifest to {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/night_lights/eog_v22"))
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2012, 2025)))
    parser.add_argument("--cookie", default=os.environ.get("EOG_COOKIE"))
    parser.add_argument("--bearer-token", default=os.environ.get("EOG_TOKEN"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--delete-old-dir",
        type=Path,
        default=None,
        help="Delete old Zenodo average_viirs files from this directory after replacement verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        download_eog_vnl_v22(
            base_url=args.base_url,
            output_dir=args.output_dir,
            years=sorted(set(args.years)),
            cookie=args.cookie,
            bearer_token=args.bearer_token,
            retries=args.retries,
            delete_old_dir=args.delete_old_dir,
        )
    except PermissionError as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
