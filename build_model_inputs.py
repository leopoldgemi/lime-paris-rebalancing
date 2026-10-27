#!/usr/bin/env python3
"""
Build the input data for the LP rebalancing model (lp_model.py).

What it does
------------
Paris is cut into square cells of 300 m x 300 m. Only cells whose centre lies
inside Paris intra-muros are kept. For every "operating day" k it counts, per
cell i and hour t (t = 1..18):

    r[i,t,k]   bikes returned in the cell (trip ends)
    d[i,t,k]   rental demand, corrected for minutes in which the cell was empty
    s[i]       bikes in the cell at night, before the day starts (mean over days)

and it writes the straight-line distance dist[i,j] between cell centres for all
pairs up to --R-max km (the model keeps the pairs with dist <= R, called A).

Operating day k
---------------
    weekday (Mon-Fri)   07:00 - 01:00 next day   night snapshot at 01:00
    weekend (Sat-Sun)   09:00 - 03:00 next day   night snapshot at 03:00
Both are 18 hours long. Trips after midnight belong to the day before.
Only complete days are used (see is_complete).

Rentals and returns use exactly the counting rules and filters of
build_demand.py (blip filter, flicker filter, van pick-ups, 200 m minimum move,
id rotation pairs skipped). "Available" = not reserved, no range threshold.

Demand correction
-----------------
A cell without any available bike cannot be rented from, so observed rentals
understate demand. For each cell and hour:
    d = rentals x 60 / (minutes with at least one available bike)
The factor 60 / minutes is capped at 5. If the cell was empty the whole hour,
d = (average daily demand of the cell) x (share of this hour in the city-wide
hourly profile). Demand is never set below the rentals actually observed.

Reads
-----
    daily snapshot files lime_paris_YYYY-MM-DD.csv.gz (release "daily-data")
    data/reference/quartiers_paris.geojson (Paris boundary and arrondissements)

Writes (into --out, default model_input_lp/, ignored by git)
------
    cells.csv          cell id, centre (km and lat/lon), arrondissement
    days.csv           every operating day found, its type and whether it is complete
    s.csv              s[i] per day type
    hourly.csv.gz      r, rentals, d per day type, day, cell and hour
    arcs.csv.gz        all pairs (i, j), i != j, with dist[i,j] <= --R-max km
    summary.md         the key numbers

Usage
-----
    python build_model_inputs.py --data-dir ../lime-data-backup/daily
"""

import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, shape
from shapely.ops import transform

from analyze_snapshots import BASE_DIR, M_PER_DEG_LAT, M_PER_DEG_LON, PARIS_TZ
from build_demand import (BLIP_RADIUS_M, BLIP_RANGE_M, BLIP_WINDOW_MIN, EBIKE_TYPE, MIN_MOVE_M,
                          QUARTIERS_PATH, USECOLS, BlipIndex, Zoning, resolve_pair, snapshot_record)

# pandas 2.3 with numpy 2.4 prints a harmless warning on every Timedelta; hide it.
warnings.filterwarnings("ignore", message=".*'generic' unit for NumPy timedelta.*")

# ---- Parameters -------------------------------------------------------------

GRID_M = 300                     # cell size in metres
HOURS = 18                       # hours per operating day, t = 1..18
DAY_START = {"weekday": 7, "weekend": 9}    # local hour at which day k starts
NIGHT_HOUR = {"weekday": 1, "weekend": 3}   # local hour of the night snapshot (before day k)
FACTOR_CAP = 5.0                 # demand correction factor 60 / minutes is capped here
OPS_RADIUS_M = 100.0             # van pick-up filter, as in build_demand.py
OPS_MIN_BIKES = 4
BUFFER_MIN = 20                  # extra minutes loaded around a day so the blip filter sees both sides
MAX_GAP_MIN = 10                 # a complete day has no gap between snapshots longer than this
MIN_SNAPSHOT_SHARE = 0.9         # ... and at least 90 % of the 540 possible 2-minute snapshots
MIN_HOUR_MINUTES = 30            # ... and at least 30 usable minutes in every hour


# ---- Cells ------------------------------------------------------------------

def paris_cells(quartiers_path):
    """All 300 m cells whose centre is inside Paris, with their arrondissement."""
    features = json.loads(Path(quartiers_path).read_text())["features"]

    # Convert lon/lat to metres so that the grid matches build_demand.py exactly.
    def to_metres(x, y, z=None):
        return np.asarray(x) * M_PER_DEG_LON, np.asarray(y) * M_PER_DEG_LAT

    quartiers = [transform(to_metres, shape(f["geometry"])) for f in features]
    arrondissement = [int(f["properties"]["c_ar"]) for f in features]
    paris = shapely.union_all(quartiers)

    rows = []
    minx, miny, maxx, maxy = paris.bounds
    for x in range(int(minx // GRID_M), int(maxx // GRID_M) + 1):
        for y in range(int(miny // GRID_M), int(maxy // GRID_M) + 1):
            centre = Point((x + 0.5) * GRID_M, (y + 0.5) * GRID_M)
            if not paris.contains(centre):
                continue
            # Arrondissement = the one of the quartier closest to the centre (distance 0 = inside).
            arr = arrondissement[int(np.argmin([q.distance(centre) for q in quartiers]))]
            rows.append({"cell": f"{x}_{y}",
                         "x_km": centre.x / 1000, "y_km": centre.y / 1000,
                         "lat": centre.y / M_PER_DEG_LAT, "lon": centre.x / M_PER_DEG_LON,
                         "arrondissement": arr})
    return pd.DataFrame(rows)


def all_arcs(cells, R_max):
    """All pairs (i, j), i != j, whose centres are at most R_max km apart."""
    xy = cells[["x_km", "y_km"]].to_numpy()
    rows = []
    for i in range(len(cells)):
        dist = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        for j in np.flatnonzero((dist <= R_max) & (dist > 0)):
            rows.append((cells["cell"].iat[i], cells["cell"].iat[j], round(float(dist[j]), 4)))
    return pd.DataFrame(rows, columns=["i", "j", "dist"])


# ---- Loading snapshots ------------------------------------------------------

def load_file(path):
    """One daily file: e-bikes only, timestamps parsed."""
    df = pd.read_csv(path, usecols=USECOLS, dtype={"bike_id": "string", "vehicle_type_id": "string"})
    df = df[df["vehicle_type_id"] == EBIKE_TYPE].copy()
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)
    print(f"  loaded {Path(path).name}", flush=True)
    return df


def snapshots_between(files, cache, t0, t1):
    """All snapshot rows with t0 <= time < t1, read from the daily files that cover it."""
    parts = []
    for date in pd.date_range(t0.tz_convert("UTC").date(), t1.tz_convert("UTC").date()):
        key = date.strftime("%Y-%m-%d")
        if key not in files:
            continue
        if key not in cache:
            cache[key] = load_file(files[key])
        df = cache[key]
        parts.append(df[(df["snapshot_utc"] >= t0) & (df["snapshot_utc"] < t1)])
    if not parts:
        return pd.DataFrame(columns=USECOLS)
    return pd.concat(parts, ignore_index=True)


# ---- One operating day ------------------------------------------------------

def process_day(df, start, zoning, cell_index):
    """Count rentals, returns and minutes with an available bike per cell and hour."""
    end = start + pd.Timedelta(hours=HOURS)
    n = len(cell_index)
    rentals = np.zeros((n, HOURS))
    returns = np.zeros((n, HOURS))
    avail_minutes = np.zeros((n, HOURS))    # minutes with >= 1 available bike in the cell
    pair_minutes = np.zeros(HOURS)          # usable minutes in the hour (rotation pairs skipped)

    # One record per snapshot, built by the same function as in build_demand.py.
    recs = [snapshot_record(t, g, zoning, 60, 0.0) for t, g in df.groupby("snapshot_utc", sort=True)]

    # Flicker filter: an id seen in exactly one snapshot is neither a rental nor a return.
    for k, rec in enumerate(recs):
        if 0 < k < len(recs) - 1:
            rec["flicker"] = rec["ids"] - recs[k - 1]["ids"] - recs[k + 1]["ids"]
        else:
            rec["flicker"] = set()
    blips = BlipIndex(recs, BLIP_WINDOW_MIN, BLIP_RADIUS_M, BLIP_RANGE_M)

    # Walk through consecutive snapshot pairs (a, b). The pair belongs to the hour of a.
    for a, b in zip(recs, recs[1:]):
        if a["t"] < start or a["t"] >= end:
            continue
        out = resolve_pair(a, b, blips, OPS_RADIUS_M, OPS_MIN_BIKES, MIN_MOVE_M)
        if out is None:
            continue                        # gap too long or id rotation: pair not usable
        flows, info = out
        t = int((a["t"] - start) / pd.Timedelta(hours=1))
        minutes = info["minutes"]
        pair_minutes[t] += minutes

        # Cells with at least one available bike at the start of the pair.
        for cell, bikes in a["inv"].items():
            if cell in cell_index and bikes >= 1:
                avail_minutes[cell_index[cell], t] += minutes

        # Rentals (departures) and returns (arrivals) of this pair.
        for zone, dep, arr in zip(flows["zone"], flows["departures"], flows["arrivals"]):
            if zone in cell_index:
                rentals[cell_index[zone], t] += dep
                returns[cell_index[zone], t] += arr

    return rentals, returns, avail_minutes, pair_minutes, recs


def is_complete(recs, start):
    """True if the 18 hours of the day are well covered by snapshots."""
    end = start + pd.Timedelta(hours=HOURS)
    times = pd.Series([rec["t"] for rec in recs if start <= rec["t"] < end])
    if len(times) < MIN_SNAPSHOT_SHARE * HOURS * 30:
        return False, len(times), np.nan
    edges = pd.concat([pd.Series([start]), times, pd.Series([end])])
    max_gap = edges.diff().max() / pd.Timedelta(minutes=1)
    return max_gap <= MAX_GAP_MIN, len(times), max_gap


def night_inventory(df, night, zoning, cell_index):
    """Available bikes per cell at the last snapshot at or before the night time."""
    before = df[(df["snapshot_utc"] <= night) & (df["snapshot_utc"] > night - pd.Timedelta(minutes=MAX_GAP_MIN))]
    if before.empty:
        return None
    t = before["snapshot_utc"].max()
    rec = snapshot_record(t, before[before["snapshot_utc"] == t], zoning, 60, 0.0)
    s = np.zeros(len(cell_index))
    for cell, bikes in rec["inv"].items():
        if cell in cell_index:
            s[cell_index[cell]] = bikes
    return s


# ---- Demand correction ------------------------------------------------------

def corrected_demand(h):
    """Add columns d (corrected demand) and case to the hourly table of one day type."""
    # Rentals scaled up to the full hour: id-rotation pairs are skipped, and the
    # rotation clock does not depend on demand, so this is a fair scale-up.
    h["rentals"] = h["rentals_obs"] * 60 / h["pair_minutes"]
    h["r"] = h["returns_obs"] * 60 / h["pair_minutes"]

    # Normal case: d = rentals x 60 / minutes with a bike, factor capped at 5.
    factor = h["pair_minutes"] / h["avail_minutes"].where(h["avail_minutes"] > 0)
    h["case"] = np.where(h["avail_minutes"] == 0, "empty_hour",
                         np.where(factor > FACTOR_CAP, "capped", "normal"))
    h["d"] = h["rentals"] * factor.clip(upper=FACTOR_CAP)

    # City-wide hourly profile: share of each hour t in the daily rentals.
    profile = h.groupby("t")["rentals"].sum()
    share = h["t"].map(profile / profile.sum())

    # Average daily demand of each cell, estimated from the hours in which the
    # cell had a bike: total demand seen / the share of the day those hours cover.
    seen = h["case"] != "empty_hour"
    daily = (h["d"].where(seen, 0).groupby(h["cell"]).sum()
             / share.where(seen, 0).groupby(h["cell"]).sum())
    daily = daily.fillna(0)

    # Empty-hour case: d = average daily demand of the cell x share of the hour.
    fallback = h["cell"].map(daily) * share
    h["d"] = np.where(seen, h["d"], fallback)
    h["d"] = np.maximum(h["d"], h["rentals"])   # never below what was actually rented
    return h


# ---- Main -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="folder with lime_paris_YYYY-MM-DD.csv.gz")
    p.add_argument("--out", default="model_input_lp")
    p.add_argument("--R-max", type=float, default=3.0, help="keep pairs up to this distance in km")
    a = p.parse_args()

    data_dir = Path(a.data_dir).expanduser()
    out = Path(a.out).expanduser()
    out = out if out.is_absolute() else BASE_DIR / out   # relative paths start at the repo
    out.mkdir(parents=True, exist_ok=True)

    files = {Path(f).name[len("lime_paris_"):len("lime_paris_YYYY-MM-DD")]: f
             for f in sorted(glob.glob(str(data_dir / "lime_paris_*.csv.gz")))}
    print(f"{len(files)} daily files in {data_dir}")

    # Cells and distances.
    cells = paris_cells(QUARTIERS_PATH)
    cell_index = {c: k for k, c in enumerate(cells["cell"])}
    cells.to_csv(out / "cells.csv", index=False)
    arcs = all_arcs(cells, a.R_max)
    arcs.to_csv(out / "arcs.csv.gz", index=False)
    print(f"{len(cells)} cells, {len(arcs):,} pairs within {a.R_max} km")

    zoning = Zoning("grid", GRID_M)
    cache = {}
    days, s_rows, hourly = [], [], []
    for date in pd.date_range(min(files), max(files)):
        day_type = "weekend" if date.dayofweek >= 5 else "weekday"
        start = pd.Timestamp(date.year, date.month, date.day, DAY_START[day_type], tz=PARIS_TZ)
        night = pd.Timestamp(date.year, date.month, date.day, NIGHT_HOUR[day_type], tz=PARIS_TZ)
        buffer = pd.Timedelta(minutes=BUFFER_MIN)

        # Night snapshot (state before the day) and the snapshots of the day itself.
        night_df = snapshots_between(files, cache, night - pd.Timedelta(minutes=MAX_GAP_MIN), night + pd.Timedelta(seconds=1))
        s = night_inventory(night_df, night, zoning, cell_index)
        df = snapshots_between(files, cache, start - buffer, start + pd.Timedelta(hours=HOURS) + buffer)
        if df.empty:
            days.append({"day": date.date(), "day_type": day_type, "complete": False, "reason": "no data"})
            continue
        rentals, returns, avail_minutes, pair_minutes, recs = process_day(df, start, zoning, cell_index)
        complete, n_snaps, max_gap = is_complete(recs, start)
        reason = ""
        if s is None:
            complete, reason = False, "no night snapshot"
        elif pair_minutes.min() < MIN_HOUR_MINUTES:
            complete, reason = False, "hour with < 30 usable minutes"
        elif not complete:
            reason = "too few snapshots or gap"
        days.append({"day": date.date(), "day_type": day_type, "complete": complete, "reason": reason,
                     "snapshots": n_snaps, "max_gap_min": max_gap,
                     "s_total": None if s is None else s.sum(), "rentals_obs": rentals.sum()})
        print(f"{date.date()} {day_type}: complete={complete} {reason} snapshots={n_snaps} "
              f"rentals={rentals.sum():,.0f}", flush=True)
        if not complete:
            continue

        s_rows.append(pd.DataFrame({"day_type": day_type, "day": date.date(), "cell": cells["cell"], "s": s}))
        hourly.append(pd.DataFrame({
            "day_type": day_type, "day": date.date(),
            "cell": np.repeat(cells["cell"].to_numpy(), HOURS),
            "t": np.tile(np.arange(1, HOURS + 1), len(cells)),
            "rentals_obs": rentals.ravel(), "returns_obs": returns.ravel(),
            "avail_minutes": avail_minutes.ravel(),
            "pair_minutes": np.tile(pair_minutes, len(cells))}))

        # Keep only the files still needed for the next days in memory.
        for key in [k for k in cache if k < (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]:
            del cache[key]

    days = pd.DataFrame(days)
    days.to_csv(out / "days.csv", index=False)

    # s[i]: mean night inventory over the days of each type.
    s_all = pd.concat(s_rows, ignore_index=True)
    s_mean = s_all.groupby(["day_type", "cell"], as_index=False)["s"].mean()
    s_mean.to_csv(out / "s.csv", index=False)

    # r and corrected d, per day type.
    h = pd.concat([corrected_demand(g.copy()) for _, g in pd.concat(hourly).groupby("day_type")], ignore_index=True)
    h = h[["day_type", "day", "cell", "t", "rentals_obs", "returns_obs", "avail_minutes", "pair_minutes",
           "rentals", "r", "d", "case"]]
    h.to_csv(out / "hourly.csv.gz", index=False, float_format="%.4f")

    write_summary(out, cells, arcs, days, s_mean, s_all, h)


def write_summary(out, cells, arcs, days, s_mean, s_all, h):
    lines = ["# Model inputs (build_model_inputs.py)", "",
             f"- cells: {len(cells)} ({GRID_M} m, centre inside Paris intra-muros)", ""]
    lines += ["| day type | complete days | sum(s) | rentals/day | corrected demand/day | correction | returns/day |",
              "|---|---|---|---|---|---|---|"]
    for day_type, g in h.groupby("day_type"):
        n_days = g["day"].nunique()
        rentals = g["rentals"].sum() / n_days
        demand = g["d"].sum() / n_days
        ret = g["r"].sum() / n_days
        s_sum = s_mean.loc[s_mean["day_type"] == day_type, "s"].sum()
        lines.append(f"| {day_type} | {n_days} | {s_sum:,.0f} | {rentals:,.0f} | {demand:,.0f} | "
                     f"+{demand / rentals - 1:.1%} | {ret:,.0f} |")
    lines += ["", "sum(s) per day (min / max): " + ", ".join(
        f"{t} {g.groupby('day')['s'].sum().min():,.0f} / {g.groupby('day')['s'].sum().max():,.0f}"
        for t, g in s_all.groupby("day_type")), ""]

    lines += ["Demand correction cases (cell-hours):", "",
              "| day type | cell-hours | normal | capped at 5 | empty whole hour | empty hour with d > 0 |",
              "|---|---|---|---|---|---|"]
    for day_type, g in h.groupby("day_type"):
        c = g["case"].value_counts()
        empty_pos = ((g["case"] == "empty_hour") & (g["d"] > 0)).sum()
        lines.append(f"| {day_type} | {len(g):,} | {c.get('normal', 0):,} | {c.get('capped', 0):,} "
                     f"({c.get('capped', 0) / len(g):.1%}) | {c.get('empty_hour', 0):,} "
                     f"({c.get('empty_hour', 0) / len(g):.1%}) | {empty_pos:,} |")
    lines += ["", "Demand added by each case (share of corrected demand):", ""]
    for day_type, g in h.groupby("day_type"):
        extra = (g["d"] - g["rentals"]).groupby(g["case"]).sum() / g["d"].sum()
        lines.append(f"- {day_type}: " + ", ".join(f"{k} {v:+.1%}" for k, v in extra.items()))

    lines += ["", "| R (km) | 1.0 | 1.5 | 2.0 |", "|---|---|---|---|",
              "| pairs in A | " + " | ".join(f"{(arcs['dist'] <= R).sum():,}" for R in (1.0, 1.5, 2.0)) + " |",
              "", "Days found:", "", "```", days.to_string(index=False), "```"]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
