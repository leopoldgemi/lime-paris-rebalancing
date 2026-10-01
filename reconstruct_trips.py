#!/usr/bin/env python3
"""
Derive events per bike_id from consecutive snapshots.

For each bike, consecutive observations (t_prev at A, t_next at B) are
classified. Thresholds are the PARAMETERS below (also overridable via CLI
flags, so a sensitivity analysis can loop over them).

  trip     bike reappears > MIN_TRIP_DIST_M away, absence MIN_TRIP_MIN..MAX_TRIP_MIN,
           range unchanged or lower (range_delta <= RANGE_TOL_M)
  ops      range rose by > RANGE_INCREASE_OPS_M (battery swap, with or without
           relocation), OR the bike reappears as part of a batch: >= OPS_BATCH_MIN_BIKES
           bikes reappearing in the same snapshot within OPS_BATCH_RADIUS_M (van drop-off)
  unclear  absence > MAX_TRIP_MIN with relocation; relocation with absence < MIN_TRIP_MIN;
           moderate range increase (RANGE_TOL_M < delta <= RANGE_INCREASE_OPS_M);
           same position but long absence (> MAX_TRIP_MIN); anything else that is
           not simply "still parked"
  (ignored) same position (< STATIONARY_DIST_M) and short gap, or small
           displacement (< MIN_TRIP_DIST_M) treated as GPS jitter

Id rotation
  Lime rotates all bike_ids about every 15 min. linking.py re-links ids across a
  rotation via the (lat, lon, range) fingerprint of parked bikes and provides a
  chain id "uid"; events are built on uid. A bike that rides across a rotation
  boundary cannot be linked: its old chain ends, a new one starts (no event).
  Expect roughly (polling interval / 15 min) of trips to be lost this way.

Caveats
  * Duration = gap between last sighting at A and first sighting at B, so it
    overstates the ride by up to one polling interval (and includes any time the
    bike sat at B before the next snapshot).
  * With 2-5 min snapshots a bike may complete a ride between two snapshots
    without ever "vanishing"; this is handled identically (gap == one interval).
  * Lime only shows available bikes: a bike in a ride, reserved for long, or in
    a van is absent. Absent-at-end-of-data bikes produce no event.

Output: trips.csv with (bike_id = chain id "uid", see linking.py)
  bike_id, start_time, end_time, duration_min, start_lat, start_lon, end_lat, end_lon,
  distance_m, range_start, range_end, range_delta, event_type, reason, batch_size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from analyze_snapshots import BASE_DIR, M_PER_DEG_LAT, M_PER_DEG_LON, load_snapshots
from linking import link_ids

# --------------------------------------------------------------------------- #
# PARAMETERS (defaults; override with CLI flags for sensitivity analysis)
# --------------------------------------------------------------------------- #
MIN_TRIP_DIST_M = 200        # relocation smaller than this is GPS jitter / not a trip
MIN_TRIP_MIN = 2             # shortest plausible ride (gap between sightings)
MAX_TRIP_MIN = 90            # longest plausible ride; longer absence -> unclear
RANGE_TOL_M = 500            # range may "increase" by up to this (sensor noise) and still be a trip
RANGE_INCREASE_OPS_M = 5_000 # range increase above this = battery swap (ops)
OPS_BATCH_MIN_BIKES = 4      # >= this many bikes reappearing together = van drop-off (ops)
OPS_BATCH_RADIUS_M = 100     # ... within this distance of each other (grid cell size)
STATIONARY_DIST_M = 50       # below this the bike is considered not to have moved
MAX_STILL_GAP_MIN = 90       # same position but absent longer than this -> unclear (e.g. taken and returned)


def classify(df: pd.DataFrame, P: dict) -> pd.DataFrame:
    d = df.sort_values(["uid", "snapshot_utc"])[
        ["uid", "bike_id", "snapshot_utc", "lat", "lon", "current_range_meters"]
    ].rename(columns={"current_range_meters": "range"})
    prev = d.groupby("uid").shift(1)
    d = d.assign(prev_t=prev["snapshot_utc"], prev_lat=prev["lat"], prev_lon=prev["lon"], prev_range=prev["range"])
    d = d.dropna(subset=["prev_t"]).copy()

    dy = (d["lat"] - d["prev_lat"]) * M_PER_DEG_LAT
    dx = (d["lon"] - d["prev_lon"]) * M_PER_DEG_LON
    d["dist_m"] = (dx**2 + dy**2) ** 0.5
    d["gap_min"] = (d["snapshot_utc"] - d["prev_t"]).dt.total_seconds() / 60
    d["range_delta"] = d["range"] - d["prev_range"]

    # typical polling interval -> "was absent" means gap clearly longer than one interval
    interval_min = d.groupby("snapshot_utc").size().index.to_series().diff().dt.total_seconds().div(60).median()
    interval_min = float(interval_min) if pd.notna(interval_min) else 2.0
    d["absent"] = d["gap_min"] > 1.5 * interval_min

    moved = d["dist_m"] >= P["MIN_TRIP_DIST_M"]
    jitter = (d["dist_m"] >= P["STATIONARY_DIST_M"]) & ~moved
    still = d["dist_m"] < P["STATIONARY_DIST_M"]
    swap = d["range_delta"] > P["RANGE_INCREASE_OPS_M"]
    range_ok = d["range_delta"] <= P["RANGE_TOL_M"]
    dur_ok = (d["gap_min"] >= P["MIN_TRIP_MIN"]) & (d["gap_min"] <= P["MAX_TRIP_MIN"])

    # batch drop-off: bikes that were absent and reappear in the same snapshot in the same ~100 m cell
    cell = P["OPS_BATCH_RADIUS_M"]
    d["cell"] = (
        (d["lat"] * M_PER_DEG_LAT // cell).astype(int).astype(str) + "_" + (d["lon"] * M_PER_DEG_LON // cell).astype(int).astype(str)
    )
    cand = d[d["absent"] & moved]
    batch_size = cand.groupby(["snapshot_utc", "cell"])["uid"].transform("size")
    d["batch_size"] = 0
    d.loc[cand.index, "batch_size"] = batch_size
    batch = d["batch_size"] >= P["OPS_BATCH_MIN_BIKES"]

    d["event_type"] = "ignore"
    d["reason"] = ""

    def set_(mask, etype, reason):
        m = mask & (d["event_type"] == "ignore")
        d.loc[m, "event_type"] = etype
        d.loc[m, "reason"] = reason

    # order matters: first match wins
    set_(swap & still, "ops", "battery swap in place")
    set_(swap & moved, "ops", "relocated with battery swap")
    set_(moved & batch, "ops", "batch drop-off")
    set_(moved & dur_ok & range_ok, "trip", "trip")
    set_(moved & (d["gap_min"] > P["MAX_TRIP_MIN"]), "unclear", "relocated after long absence")
    set_(moved & (d["gap_min"] < P["MIN_TRIP_MIN"]), "unclear", "relocated within < MIN_TRIP_MIN")
    set_(moved & ~range_ok, "unclear", "relocated, moderate range increase")
    set_(still & (d["gap_min"] > P["MAX_STILL_GAP_MIN"]), "unclear", "same position after long absence")
    set_(jitter & d["absent"], "unclear", "small displacement after absence")
    # remaining: still parked / jitter without absence -> ignore

    out = d[d["event_type"] != "ignore"].copy()
    out = out.drop(columns="bike_id").rename(columns={
        "uid": "bike_id", "prev_t": "start_time", "snapshot_utc": "end_time", "prev_lat": "start_lat", "prev_lon": "start_lon",
        "lat": "end_lat", "lon": "end_lon", "dist_m": "distance_m", "prev_range": "range_start", "range": "range_end",
        "gap_min": "duration_min",
    })
    cols = ["bike_id", "start_time", "end_time", "duration_min", "start_lat", "start_lon", "end_lat", "end_lon",
            "distance_m", "range_start", "range_end", "range_delta", "event_type", "reason", "batch_size"]
    out = out[cols].sort_values(["start_time", "bike_id"]).reset_index(drop=True)
    out["duration_min"] = out["duration_min"].round(1)
    out["distance_m"] = out["distance_m"].round(0)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="reports/trips.csv")
    for name in ["MIN_TRIP_DIST_M", "MIN_TRIP_MIN", "MAX_TRIP_MIN", "RANGE_TOL_M", "RANGE_INCREASE_OPS_M",
                 "OPS_BATCH_MIN_BIKES", "OPS_BATCH_RADIUS_M", "STATIONARY_DIST_M", "MAX_STILL_GAP_MIN"]:
        p.add_argument(f"--{name.lower().replace('_', '-')}", dest=name, type=float, default=globals()[name])
    a = p.parse_args(argv)
    P = {k: getattr(a, k) for k in vars(a) if k.isupper()}

    data_dir = Path(a.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else BASE_DIR / data_dir
    out = Path(a.out)
    out = out if out.is_absolute() else BASE_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)

    df, boundaries = link_ids(load_snapshots(data_dir))
    print(f"id rotations detected: {len(boundaries)}" + (
        f", fingerprint link rate {boundaries['link_rate'].mean():.1%}" if len(boundaries) else ""))
    events = classify(df, P)
    events.to_csv(out, index=False)

    print(f"\nparameters: {P}")
    print(f"events written: {len(events):,} -> {out}")
    if len(events):
        print(events["event_type"].value_counts().to_string())
        print()
        print(events.groupby(["event_type", "reason"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
