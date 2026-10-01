#!/usr/bin/env python3
"""
Download the on-street parking spots of Paris Open Data where free-floating
bikes may park and aggregate the capacity to a grid.

Source: "Stationnement sur voie publique - emplacements"
  https://opendata.paris.fr/explore/dataset/stationnement-voie-publique-emplacements/
  licence: ODbL. Filter: stationnement_autorises_aux_velos_en_libre_service = "oui"
  (all of them are regpar = "Vélos"). Capacity field: placal (computed places).

Writes
  data/reference/paris_bike_parking_libre_service.csv   raw export (one row per spot)
  reports/parking_capacity_grid.csv                      capacity per grid cell (same grid as analyze_snapshots.py)
  reports/parking_capacity_arrondissement.csv

Usage: python fetch_parking_zones.py [--grid-m 500]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import requests

from analyze_snapshots import BASE_DIR, M_PER_DEG_LAT, M_PER_DEG_LON

DATASET = "stationnement-voie-publique-emplacements"
URL = f"https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/{DATASET}/exports/csv"
FIELDS = "id,regpri,regpar,typsta,arrond,placal,plarel,lon,surface_calculee,typemob,nummob,numiris,geo_point_2d"
UA = "ColumbiaIEOR4004-LimeRebalancingProject/1.0 (academic research; contact: leopoldv.scholz@gmail.com)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-m", type=float, default=500)
    a = p.parse_args(argv)
    ref = BASE_DIR / "data" / "reference"
    rep = BASE_DIR / "reports"
    ref.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    r = requests.get(URL, params={
        "where": 'stationnement_autorises_aux_velos_en_libre_service="oui"',
        "select": FIELDS, "delimiter": ",",
    }, headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")
    raw = ref / "paris_bike_parking_libre_service.csv"
    raw.write_text(text)
    rows = list(csv.DictReader(io.StringIO(text)))
    print(f"{len(rows):,} spots -> {raw.relative_to(BASE_DIR)}")

    dlat = a.grid_m / M_PER_DEG_LAT
    dlon = a.grid_m / M_PER_DEG_LON
    grid: dict[tuple, dict] = {}
    arr: dict[str, dict] = {}
    for row in rows:
        try:
            lat, lon = (float(x) for x in row["geo_point_2d"].split(","))
            cap = int(float(row["placal"] or 0))
        except (ValueError, KeyError):
            continue
        key = (round((lat // dlat) * dlat + dlat / 2, 6), round((lon // dlon) * dlon + dlon / 2, 6))
        g = grid.setdefault(key, {"cell_lat": key[0], "cell_lon": key[1], "spots": 0, "capacity": 0})
        g["spots"] += 1
        g["capacity"] += cap
        ar = arr.setdefault(row["arrond"], {"arrond": row["arrond"], "spots": 0, "capacity": 0})
        ar["spots"] += 1
        ar["capacity"] += cap

    for name, data, cols in (("parking_capacity_grid.csv", grid.values(), ["cell_lat", "cell_lon", "spots", "capacity"]),
                             ("parking_capacity_arrondissement.csv", arr.values(), ["arrond", "spots", "capacity"])):
        with (rep / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(sorted(data, key=lambda d: -d["capacity"]))
    total = sum(g["capacity"] for g in grid.values())
    print(f"total capacity {total:,} places in {len(grid)} grid cells of {a.grid_m:.0f} m -> reports/parking_capacity_grid.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
