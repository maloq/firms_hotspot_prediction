#!/usr/bin/env python3
"""
yadisk_dl.py – Download files from Yandex Disk.

Examples
--------
# 1) Public share link
python yadisk_dl.py --public https://disk.yandex.ru/d/abc123XYZ -o /tmp

# 2) Private file (needs OAuth token)
export YD_TOKEN="ya29.A0AR...YourToken..."
python yadisk_dl.py --path /Backups/db.dump -o .

OAuth token notes
-----------------
Create an app in https://oauth.yandex.com, give it “Yandex.Disk REST API”
access, then open:
  https://oauth.yandex.com/authorize?response_type=token&client_id=YOUR_CLIENT_ID
Copy the token part from the redirected URL.
"""
import argparse, os, sys, requests, urllib.parse as ul
PUBLIC_API  = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
PRIVATE_API = "https://cloud-api.yandex.net/v1/disk/resources/download"

def _stream_save(url: str, outfile: str):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(outfile, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk: f.write(chunk)

def download_public(public_url: str, outdir: str = "."):
    meta = requests.get(PUBLIC_API + ul.quote_plus(public_url)).json()
    href = meta["href"]                                       # real download URL
    fname = ul.parse_qs(ul.urlparse(href).query).get("filename",["file"])[0]
    path = os.path.join(outdir, fname)
    _stream_save(href, path)
    print(f"✔ Saved → {path}")

def download_private(path_on_disk: str, token: str, outdir: str = "."):
    hdr = {"Authorization": f"OAuth {token}"}
    meta = requests.get(PRIVATE_API, headers=hdr, params={"path": path_on_disk}).json()
    href = meta["href"]
    fname = os.path.basename(path_on_disk.rstrip("/"))
    path = os.path.join(outdir, fname)
    _stream_save(href, path)
    print(f"✔ Saved → {path}")

def cli():
    ap = argparse.ArgumentParser(description="Download from Yandex Disk")
    g  = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--public", help="public share link URL")
    g.add_argument("--path",   help="path on your own disk (needs token)")
    ap.add_argument("-t", "--token", help="OAuth token (or set YD_TOKEN env var)")
    ap.add_argument("-o", "--out", default=".", help="output directory")
    args = ap.parse_args()

    if args.public:
        download_public(args.public, args.out)
    else:
        token = args.token or os.getenv("YD_TOKEN")
        if not token:
            ap.error("OAuth token required for --path downloads")
        download_private(args.path, token, args.out)

if __name__ == "__main__":
    try:
        cli()
    except (KeyboardInterrupt, SystemExit):
        sys.exit(1)