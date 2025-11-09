# -*- coding: utf-8 -*-
"""
Extract NASA FIRMS MODIS fire points from the supplied shapefiles (2024 & 2025)
and export them to country–level CSV files that look like

    modis_<year>_<Country>.csv   (e.g. modis_2024_Azerbaijan.csv)

Usage
-----
python -m src.data_download.extract_FIRMS_from_shx \
       --years 2024 2025 \
       --countries Azerbaijan Georgia Armenia

Optional flags:
    --fire-root   Where the fire shapefiles live   (default: data)
    --out-root    Where the CSVs will be written   (default: data/modis)
    --country-shp World countries polygon file     (default: data/countries/ne_110m_admin_0_countries.shp)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
DEFAULT_FIRE_ROOT = Path("data")                              # data/modis2024, data/modis2025, …
DEFAULT_OUT_ROOT = DEFAULT_FIRE_ROOT / "modis"                # data/modis/<year>
DEFAULT_COUNTRY_SHP = DEFAULT_FIRE_ROOT / "countries" / "ne_110m_admin_0_countries.shp"

FIRE_COLUMNS_ORDER = [
    "latitude", "longitude", "brightness", "scan", "track",
    "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "version", "bright_t31", "frp", "daynight", "type"
]

country_mapping = {
        'Russian_Federation': 'Russia',
        'United_Kingdom': 'United Kingdom',
        'Czech_Republic': 'Czechia',
        'Bosnia_and_Herzegovina': 'Bosnia and Herzegovina',
        'Serbia': 'Republic of Serbia',
        'Dem_Rep_Korea': 'North Korea',
        'Republic_of_Korea': 'South Korea',
        'Macedonia_Former_Yugoslav_Republic_of': 'North Macedonia',
        'Bosnia_and_Herzegovina': 'Bosnia and Herzegovina',
    }


def read_fire_points(year_dir: Path) -> gpd.GeoDataFrame:
    """
    Read all shapefiles inside *year_dir* (archive &/or nrt) and concatenate them.
    Assumes WGS-84 geographic coordinates (EPSG:4326) as supplied by FIRMS.
    """
    shp_files = sorted(year_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No *.shp files found in {year_dir}")

    frames: list[gpd.GeoDataFrame] = []
    for shp in shp_files:
        gdf = gpd.read_file(shp, low_memory=True)
        # Ensure CRS is lon/lat
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        frames.append(gdf)

    fires = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(fires, geometry="geometry", crs="EPSG:4326")


def standardise_columns(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Lower-case all column names and ensure 'latitude' & 'longitude' exist.
    """
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "latitude" not in df.columns or "longitude" not in df.columns:
        df["latitude"] = df.geometry.y
        df["longitude"] = df.geometry.x
    return df


def export_country_csv(fires: gpd.GeoDataFrame,
                       world: gpd.GeoDataFrame,
                       year: int,
                       out_root: Path) -> None:
    """
    Do a spatial join → write one CSV per country.
    """
    fires = standardise_columns(fires)

    # Spatial join (within) to tag each point with its country name
    fires_country = gpd.sjoin(
        fires,
        world[["NAME", "geometry"]],
        how="inner",
        predicate="within"
    ).rename(columns={"NAME": "country"}).drop(columns="index_right")

    out_root_year = out_root / str(year)
    out_root_year.mkdir(parents=True, exist_ok=True)

    for country, sub in fires_country.groupby("country"):
        csv_name = f"modis_{year}_{country_mapping.get(country, country).replace(' ', '_')}.csv"
        out_f = out_root_year / csv_name

        cols_available = [c for c in FIRE_COLUMNS_ORDER if c in sub.columns]
        # Keep original order + any extra columns
        cols = cols_available + [c for c in sub.columns
                                 if c not in cols_available + ["geometry", "country"]]

        sub[cols].to_csv(out_f, index=False)
        print(f"✓ {country:<20} {len(sub):>7} rows  → {out_f}")


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def run(years: list[int],
        countries: list[str],
        fire_root: Path = DEFAULT_FIRE_ROOT,
        out_root: Path = DEFAULT_OUT_ROOT,
        country_shp: Path = DEFAULT_COUNTRY_SHP) -> None:

    # Load world polygons and convert the *requested* country names
    # (often given in FIRMS / ISO-like form) to the canonical names that
    # appear in the Natural-Earth ``NAME`` field.
    world = gpd.read_file(country_shp)
    norm_countries = [country_mapping.get(c, c).replace("_", " ") for c in countries]
    world = world[world["NAME"].isin(norm_countries)].to_crs(4326)
    if world.empty:
        raise ValueError("No countries matched in the provided shapefile.")

    for year in years:
        year_dir = fire_root
        if not year_dir.exists():
            raise FileNotFoundError(f"Directory {year_dir} does not exist.")
        print(f"\n==== {year} ====")
        fires = read_fire_points(year_dir)
        export_country_csv(fires, world, year, out_root)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract FIRMS MODIS shapefiles to per-country CSV.")
    p.add_argument("--years", nargs="+", type=int, required=True,
                   help="Years to process (expected sub-dirs data/modis<year>/).")
    p.add_argument("--countries", nargs="+", required=True,
                   help="Country names as they appear in Natural Earth 'NAME' field.")
    p.add_argument("--fire-root", type=Path, default=DEFAULT_FIRE_ROOT,
                   help=f"Root dir containing modis<year>/ folders (default: {DEFAULT_FIRE_ROOT})")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help=f"Where to write CSV files   (default: {DEFAULT_OUT_ROOT})")
    p.add_argument("--country-shp", type=Path, default=DEFAULT_COUNTRY_SHP,
                   help=f"Path to world country polygons (default: {DEFAULT_COUNTRY_SHP})")
    return p


if __name__ == "__main__":
    #args = _build_argparser().parse_args()
    # run(
    #     years=args.years,
    #     countries=args.countries,
    #     fire_root=args.fire_root,
    #     out_root=args.out_root,
    #     country_shp=args.country_shp,
    # )
    run(
        years=[2024],
        countries = ["Dem_Rep_Korea",
                       "Russian_Federation",
                       "Belarus",
                       "Lithuania",
                       "Latvia",
                       "Estonia",
                       "Poland",
                       "Czechia",
                       "Germany",
                       "Hungary",
                       "Slovakia",
                       "Finland",
                       "Norway",
                       "Sweden",
                       "Denmark",
                       "Ukraine",
                       "Moldova",
                       "Romania",
                       "Bulgaria",
                       "Albania",
                       "Montenegro",
                       "North_Macedonia",
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
                       "South_Korea"],
        fire_root=Path("data/modis2024"),
        out_root=Path("data/modis"),
        country_shp=Path("data/countries/ne_110m_admin_0_countries.shp"),
    )
