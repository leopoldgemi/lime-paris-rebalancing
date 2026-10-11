#!/usr/bin/env python3
"""
Turn raw snapshots into the parameter tables the optimisation model needs.

For every zone (square grid, default 1 km) and every period of the day
(default 20 minutes), separately for weekdays and weekends, this estimates:

    departures_per_h   bikes leaving the zone      -> demand d[i,t]
    arrivals_per_h     bikes entering the zone     -> arrival shares w[i,t]
    mean_inventory     available e-bikes present   -> s[i,0]
    empty_share        share of time with 0 bikes  -> censoring indicator
    capacity           legal parking places        -> C[i] (Paris Open Data)

Counting rule
-------------
A departure of zone i between snapshots t and t+1 is a bike that was in zone i
at t and, at t+1, is either in a different zone or absent from the feed (Lime
hides a bike while it is rented or in a van). Arrivals are the mirror image.

Feed blips are removed. Many disappearances are a bike dropping out of the feed
for a few minutes and coming back almost where it was, with almost the same
range; those are not trips. Every disappearance and every appearance in the file
(including those at id rotations) is an event. A counted departure is a blip if
some appearance follows within --blip-window-min (16) minutes, at most
--blip-radius-m (50) metres away and with at most --blip-range-m (300) metres
range difference; a counted arrival is a blip if such a disappearance precedes
it. The rule is symmetric, so each blip removes one departure and one arrival.
Matching by position and range survives Lime's 15-minute id rotation, which an
id-based test does not. A parked neighbour can match by chance (about 2 % of
departures in a control run with ranges shifted by 5 km).

A bike that stays visible but lands in another zone counts only if it moved more
than --min-move-m (default 200 m); shorter hops are GPS jitter at a boundary.

Snapshot pairs that span an id rotation are skipped entirely (ids are not
comparable across them). Rates are normalised by the minutes actually observed;
the rotation clock is independent of demand, so this is unbiased.

Only e-bikes (vehicle_type_id = 3) count. "Available" means not reserved.

Censoring: departures observed while a zone is empty understate true demand.
`empty_share` reports how much of the time a zone had no bike at all; with
--censoring-correction the observed rate is divided by (1 - empty_share).

Usage:
    python build_demand.py --data-dir ../lime-data/data
    python build_demand.py --data-dir . --period-minutes 20 --grid-m 1000
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from analyze_snapshots import BASE_DIR, M_PER_DEG_LAT, M_PER_DEG_LON, PARIS_TZ

QUARTIERS_PATH = BASE_DIR / "data" / "reference" / "quartiers_paris.geojson"
QUARTIERS_URL = ("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
                 "quartier_paris/exports/geojson")   # 80 quartiers administratifs, Paris Open Data
OUTSIDE = "OUTSIDE"          # pseudo-zone for anything beyond Paris intra-muros

EBIKE_TYPE = "3"
ROTATION_OVERLAP = 0.5      # id overlap below this = rotation, pair unusable
MAX_PAIR_GAP_MIN = 6.0      # ignore snapshot pairs further apart than this
MIN_MOVE_M = 200.0          # a visible bike changing zone counts only if it moved further than this
FP_DECIMALS = 5             # position rounding (~1 m) for the zone lookup cache
BLIP_WINDOW_MIN = 16.0      # a disappearance and an appearance this close in time ...
BLIP_RADIUS_M = 50.0        # ... and in space ...
BLIP_RANGE_M = 300.0        # ... and in current_range_meters are one bike blinking, not a trip
USECOLS = ["snapshot_utc", "bike_id", "lat", "lon", "vehicle_type_id", "is_reserved",
           "current_range_meters"]


def grid_zone_of(lat: np.ndarray, lon: np.ndarray, size_m: float) -> np.ndarray:
    x = np.floor(lon * M_PER_DEG_LON / size_m).astype(np.int64)
    y = np.floor(lat * M_PER_DEG_LAT / size_m).astype(np.int64)
    return np.char.add(np.char.add(x.astype(str), "_"), y.astype(str))


def grid_centre(zone: str, size_m: float) -> tuple[float, float]:
    x, y = (int(v) for v in zone.split("_"))
    return ((y + 0.5) * size_m / M_PER_DEG_LAT, (x + 0.5) * size_m / M_PER_DEG_LON)


class Zoning:
    """Maps coordinates to zones: the 80 quartiers administratifs, or a square grid.

    Anything outside Paris intra-muros becomes the pseudo-zone OUTSIDE. Flows across
    the city boundary are still counted (a bike leaving Paris is a departure from its
    quartier), but OUTSIDE itself is dropped from the model input.
    """

    def __init__(self, kind: str, size_m: float = 1000.0):
        self.kind = kind
        self.size_m = size_m
        self._cache: dict[tuple, str] = {}
        if kind == "quartier":
            from shapely.geometry import shape                  # noqa: PLC0415
            from shapely.strtree import STRtree                  # noqa: PLC0415
            if not QUARTIERS_PATH.exists():
                import requests                                  # noqa: PLC0415
                print(f"downloading quartier polygons -> {QUARTIERS_PATH}")
                r = requests.get(QUARTIERS_URL, timeout=60)
                r.raise_for_status()
                QUARTIERS_PATH.parent.mkdir(parents=True, exist_ok=True)
                QUARTIERS_PATH.write_bytes(r.content)
            gj = json.loads(QUARTIERS_PATH.read_text())
            self.polys = [shape(f["geometry"]) for f in gj["features"]]
            self.props = [f["properties"] for f in gj["features"]]
            self.names = [f"{p['c_qu']}_{p['l_qu']}".replace(" ", "-") for p in self.props]
            self.tree = STRtree(self.polys)

    def assign(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        if self.kind == "grid":
            return grid_zone_of(lat, lon, self.size_m)
        from shapely.geometry import Point                       # noqa: PLC0415
        keys = list(zip(np.round(lat, FP_DECIMALS), np.round(lon, FP_DECIMALS)))
        unknown = sorted({k for k in keys if k not in self._cache})
        if unknown:
            pts = [Point(lo, la) for la, lo in unknown]
            hit = self.tree.query(pts, predicate="within")
            found = {}
            for pi, gi in zip(hit[0], hit[1]):
                found[int(pi)] = self.names[int(gi)]
            for k, key in enumerate(unknown):
                self._cache[key] = found.get(k, OUTSIDE)
        return np.array([self._cache[k] for k in keys], dtype=object)

    def centre(self, zone: str) -> tuple[float, float]:
        if self.kind == "grid":
            return grid_centre(zone, self.size_m)
        i = self.names.index(zone)
        g = self.props[i]["geom_x_y"]
        return (g["lat"], g["lon"])

    def meta(self, zone: str) -> dict:
        if self.kind == "grid":
            return {"name": zone, "arrondissement": None, "area_km2": (self.size_m / 1000) ** 2}
        p = self.props[self.names.index(zone)]
        return {"name": p["l_qu"], "arrondissement": p["c_ar"], "area_km2": p["surface"] / 1e6}


def snapshot_record(t, g: pd.DataFrame, zoning: Zoning, period_min: int, min_range_m: float) -> dict:
    bike = g["bike_id"].to_numpy()
    _, first = np.unique(bike, return_index=True)
    first.sort()
    g = g.iloc[first]
    zones = zoning.assign(g["lat"].to_numpy(), g["lon"].to_numpy())
    local = t.tz_convert(PARIS_TZ)
    z = pd.Series(zones, index=g["bike_id"].to_numpy())
    lat5 = g["lat"].round(FP_DECIMALS).to_numpy()
    lon5 = g["lon"].round(FP_DECIMALS).to_numpy()
    # "available" follows the brief: not reserved and enough range to be rentable
    ok = (g["is_reserved"].to_numpy() == 0) & (g["current_range_meters"].to_numpy() >= min_range_m)
    inv = pd.Series(zones[ok]).value_counts()
    return {"t": t, "ids": set(z.index), "zone": z, "inv": inv,
            "rng": pd.Series(g["current_range_meters"].to_numpy(dtype=float), index=g["bike_id"].to_numpy()),
            "lat": pd.Series(lat5, index=g["bike_id"].to_numpy()),
            "lon": pd.Series(lon5, index=g["bike_id"].to_numpy()),
            "period": (local.hour * 60 + local.minute) // period_min,
            "weekend": local.dayofweek >= 5}


def ops_cluster_mask(a: dict, bikes: list, radius_m: float, min_bikes: int) -> set:
    """Bikes that vanished together in a tight cluster: a van pick-up, not rentals."""
    if min_bikes <= 0 or len(bikes) < min_bikes:
        return set()
    lat = a["lat"].loc[bikes].to_numpy() * M_PER_DEG_LAT
    lon = a["lon"].loc[bikes].to_numpy() * M_PER_DEG_LON
    cell = pd.Series([f"{int(y // radius_m)}_{int(x // radius_m)}" for y, x in zip(lat, lon)], index=bikes)
    big = cell.value_counts()
    big = set(big[big >= min_bikes].index)
    return set(cell[cell.isin(big)].index)


class BlipIndex:
    """All disappearances / appearances of one file, searchable by time, position and range."""

    def __init__(self, recs: list[dict], window_min: float, radius_m: float, range_m: float,
                 range_shift: float = 0.0):
        self.window_s, self.radius, self.range_m, self.shift = window_min * 60, radius_m, range_m, range_shift
        dis, app = [], []
        for a, b in zip(recs, recs[1:]):
            for src, ids, out in ((a, a["ids"] - b["ids"], dis), (b, b["ids"] - a["ids"], app)):
                if ids:
                    ids = list(ids)
                    out.append(pd.DataFrame({"t": src["t"].timestamp(), "y": src["lat"].loc[ids].to_numpy() * M_PER_DEG_LAT,
                                             "x": src["lon"].loc[ids].to_numpy() * M_PER_DEG_LON,
                                             "r": src["rng"].loc[ids].to_numpy()}))
        self.dis = self._index(dis)
        self.app = self._index(app)

    def _index(self, parts):
        d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["t", "x", "y", "r"])
        d = d.dropna()
        pts = np.column_stack([d["x"], d["y"], d["r"] * self.radius / max(self.range_m, 1e-9)])
        return d[["t", "x", "y", "r"]].to_numpy(dtype=float), (cKDTree(pts) if len(d) else None)

    def mask(self, rec: dict, ids: list, forward: bool) -> np.ndarray:
        """True for each bike of `rec` that has a matching appearance after (forward) or
        disappearance before (not forward) within the blip window."""
        arr, tree = self.app if forward else self.dis
        res = np.zeros(len(ids), dtype=bool)
        if tree is None or not ids:
            return res
        y = rec["lat"].loc[ids].to_numpy() * M_PER_DEG_LAT
        x = rec["lon"].loc[ids].to_numpy() * M_PER_DEG_LON
        r = rec["rng"].loc[ids].to_numpy() + self.shift
        t0 = rec["t"].timestamp()
        q = np.column_stack([x, y, r * self.radius / max(self.range_m, 1e-9)])
        for i, hits in enumerate(tree.query_ball_point(q, r=self.radius, p=np.inf)):
            if not hits:
                continue
            h = arr[hits]
            dt = h[:, 0] - t0 if forward else t0 - h[:, 0]
            ok = (dt > 0) & (dt <= self.window_s) & (np.abs(h[:, 3] - r[i]) <= self.range_m) \
                & (np.hypot(h[:, 1] - x[i], h[:, 2] - y[i]) <= self.radius)
            res[i] = ok.any()
        return res


def resolve_pair(a: dict, b: dict, blips: BlipIndex,
                 ops_radius_m: float, ops_min_bikes: int, min_move_m: float = MIN_MOVE_M):
    """Departures/arrivals per zone for the pair (a, b), feed blips removed."""
    gap = (b["t"] - a["t"]).total_seconds() / 60
    if gap > MAX_PAIR_GAP_MIN:
        return None
    if len(a["ids"] & b["ids"]) / max(1, len(b["ids"])) < ROTATION_OVERLAP:
        return None

    common = a["ids"] & b["ids"]
    common = pd.Index(sorted(common))
    za, zb = a["zone"].loc[common], b["zone"].loc[common]
    moved = za[za.to_numpy() != zb.to_numpy()]
    if min_move_m > 0 and len(moved):
        # GPS jitter of a parked bike next to a quartier boundary is not a rental
        idx = moved.index
        dy = (b["lat"].loc[idx].to_numpy() - a["lat"].loc[idx].to_numpy()) * M_PER_DEG_LAT
        dx = (b["lon"].loc[idx].to_numpy() - a["lon"].loc[idx].to_numpy()) * M_PER_DEG_LON
        moved = moved[np.hypot(dx, dy) > min_move_m]
    dep_zones = [za.loc[moved.index]]
    arr_zones = [zb.loc[moved.index]]

    # bikes that left the feed; drop feed blips and van pick-ups
    gone = sorted(a["ids"] - b["ids"])
    n_ops = n_blip_dep = n_blip_arr = 0
    if gone:
        blip = blips.mask(a, gone, forward=True)
        n_blip_dep = int(blip.sum())
        real_gone = [bid for bid, bl in zip(gone, blip) if not bl]
        ops = ops_cluster_mask(a, real_gone, ops_radius_m, ops_min_bikes)
        n_ops = len(ops)
        real_gone = [bid for bid in real_gone if bid not in ops]
        dep_zones.append(a["zone"].loc[real_gone])

    # bikes that entered the feed; drop the reappearance half of a blip
    appeared = sorted(b["ids"] - a["ids"])
    if appeared:
        blip = blips.mask(b, appeared, forward=False)
        n_blip_arr = int(blip.sum())
        real_new = [bid for bid, bl in zip(appeared, blip) if not bl]
        arr_zones.append(b["zone"].loc[real_new])

    dep = pd.concat(dep_zones).value_counts() if dep_zones else pd.Series(dtype=int)
    arr = pd.concat(arr_zones).value_counts() if arr_zones else pd.Series(dtype=int)
    zones = dep.index.union(arr.index)
    flows = pd.DataFrame({"zone": zones, "period": b["period"], "weekend": b["weekend"],
                          "departures": dep.reindex(zones, fill_value=0).to_numpy(),
                          "arrivals": arr.reindex(zones, fill_value=0).to_numpy()})
    return flows, {"period": b["period"], "weekend": b["weekend"], "minutes": gap,
                   "ops_excluded": n_ops, "blip_departures": n_blip_dep, "blip_arrivals": n_blip_arr}


def process_file(path: str, zoning: Zoning, period_min: int, blip_params: tuple,
                 min_range_m: float, ops_radius_m: float, ops_min_bikes: int,
                 min_move_m: float = MIN_MOVE_M):
    """Return (flows, presence, pairs, snaps) for one file of snapshots."""
    df = pd.read_csv(path, usecols=USECOLS,
                     dtype={"bike_id": "string", "vehicle_type_id": "string"},
                     compression="gzip" if path.endswith(".gz") else None)
    df = df[df["vehicle_type_id"] == EBIKE_TYPE]
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)

    recs = [snapshot_record(t, g, zoning, period_min, min_range_m)
            for t, g in df.groupby("snapshot_utc", sort=True)]
    blips = BlipIndex(recs, *blip_params)

    flows, pairs = [], []
    for a, b in zip(recs, recs[1:]):
        out = resolve_pair(a, b, blips, ops_radius_m, ops_min_bikes, min_move_m)
        if out is not None:
            flows.append(out[0])
            pairs.append(out[1])
    presence = [pd.DataFrame({"zone": r["inv"].index, "period": r["period"], "weekend": r["weekend"],
                              "bikes": r["inv"].to_numpy()}) for r in recs]
    snaps = [{"period": r["period"], "weekend": r["weekend"]} for r in recs]

    empty = pd.DataFrame(columns=["zone", "period", "weekend", "departures", "arrivals"])
    return (pd.concat(flows, ignore_index=True) if flows else empty,
            pd.concat(presence, ignore_index=True) if presence else pd.DataFrame(columns=["zone", "period", "weekend", "bikes"]),
            pd.DataFrame(pairs, columns=["period", "weekend", "minutes", "ops_excluded",
                                         "blip_departures", "blip_arrivals"]),
            pd.DataFrame(snaps, columns=["period", "weekend"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="model_input")
    p.add_argument("--zones", default="quartier", choices=["quartier", "grid"],
                   help="80 quartiers administratifs (default) or a square grid")
    p.add_argument("--grid-m", type=float, default=1000, help="cell size when --zones grid")
    p.add_argument("--min-range-m", type=float, default=5000,
                   help="a bike with less range than this is not rentable (default 5000)")
    p.add_argument("--ops-cluster-min", type=int, default=4,
                   help="this many bikes vanishing together in one cluster count as a van "
                        "pick-up, not as rentals (0 disables)")
    p.add_argument("--ops-cluster-radius", type=float, default=100.0)
    p.add_argument("--min-move-m", type=float, default=MIN_MOVE_M,
                   help="a visible bike that changes zone counts as departure/arrival only if it moved "
                        "more than this many metres between the two snapshots (default 200, 0 disables)")
    p.add_argument("--period-minutes", type=int, default=20,
                   help="length of one model period in minutes; must divide 1440 (default 20)")
    p.add_argument("--blip-window-min", type=float, default=BLIP_WINDOW_MIN,
                   help="max minutes between a disappearance and a reappearance of a blip (default 16)")
    p.add_argument("--blip-radius-m", type=float, default=BLIP_RADIUS_M,
                   help="max distance between disappearance and reappearance of a blip (default 50)")
    p.add_argument("--blip-range-m", type=float, default=BLIP_RANGE_M,
                   help="max current_range_meters difference of a blip (default 300)")
    p.add_argument("--censoring-correction", action="store_true")
    p.add_argument("--min-departures", type=float, default=0.5,
                   help="drop zones whose busiest period has fewer departures/h than this")
    a = p.parse_args(argv)
    if 1440 % a.period_minutes:
        sys.exit("--period-minutes must divide 1440")
    n_periods = 1440 // a.period_minutes

    data_dir = Path(a.data_dir); data_dir = data_dir if data_dir.is_absolute() else BASE_DIR / data_dir
    out = Path(a.out); out = out if out.is_absolute() else BASE_DIR / out
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(data_dir / "lime_paris_*.csv.gz"))) \
        or sorted(glob.glob(str(data_dir / "lime_paris_*.csv"))) \
        or sorted(glob.glob(str(data_dir / "*" / "*.csv.gz")))
    if not files:
        sys.exit(f"no snapshot files under {data_dir}")

    zoning = Zoning(a.zones, a.grid_m)
    all_flows, all_pres, all_pairs, all_snaps = [], [], [], []
    for f in files:
        fl, pr, pa, sn = process_file(f, zoning, a.period_minutes,
                                      (a.blip_window_min, a.blip_radius_m, a.blip_range_m),
                                      a.min_range_m, a.ops_cluster_radius, a.ops_cluster_min,
                                      a.min_move_m)
        all_flows.append(fl); all_pres.append(pr); all_pairs.append(pa); all_snaps.append(sn)
        print(f"  {Path(f).name}: {len(sn):,} snapshots, {len(pa):,} usable pairs "
              f"({pa['minutes'].sum() / 60:.1f} h), {fl['departures'].sum():,} departures, "
              f"{pa['ops_excluded'].sum():,} excluded as operations")
    flows = pd.concat(all_flows, ignore_index=True)
    pres = pd.concat(all_pres, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    snaps = pd.concat(all_snaps, ignore_index=True)

    key = ["zone", "period", "weekend"]
    bucket = ["period", "weekend"]
    minutes = pairs.groupby(bucket, as_index=False)["minutes"].sum().rename(
        columns={"minutes": "minutes_observed"})
    n_snaps = snaps.groupby(bucket, as_index=False).size().rename(columns={"size": "snapshots"})

    f = flows.groupby(key, as_index=False).agg(departures=("departures", "sum"),
                                               arrivals=("arrivals", "sum"))
    f = f.merge(minutes, on=bucket, how="left")
    f["departures_per_h"] = f["departures"] / f["minutes_observed"] * 60
    f["arrivals_per_h"] = f["arrivals"] / f["minutes_observed"] * 60

    inv = pres.groupby(key, as_index=False).agg(bike_snapshots=("bikes", "sum"),
                                                nonempty_snaps=("bikes", "size"))
    inv = inv.merge(n_snaps, on=bucket, how="left")
    inv["mean_inventory"] = inv["bike_snapshots"] / inv["snapshots"]
    inv["empty_share"] = (1 - inv["nonempty_snaps"] / inv["snapshots"]).clip(0, 1)

    tab = f.merge(inv[key + ["mean_inventory", "empty_share"]], on=key, how="outer").fillna(
        {"departures_per_h": 0, "arrivals_per_h": 0, "mean_inventory": 0, "empty_share": 1.0})

    if a.censoring_correction:
        tab["departures_per_h_observed"] = tab["departures_per_h"]
        tab["departures_per_h"] = tab["departures_per_h"] / (1 - tab["empty_share"]).clip(lower=0.05)

    outside = tab[tab["zone"] == OUTSIDE]
    out_share = outside["departures"].sum() / max(1, tab["departures"].sum())
    tab = tab[tab["zone"] != OUTSIDE].copy()

    busiest = tab.groupby("zone")["departures_per_h"].max()
    keep = busiest[busiest >= a.min_departures].index
    dropped = tab["zone"].nunique() - len(keep)
    tab = tab[tab["zone"].isin(keep)].copy()

    centres = {z: zoning.centre(z) for z in tab["zone"].unique()}
    tab["lat"] = tab["zone"].map(lambda z: centres[z][0])
    tab["lon"] = tab["zone"].map(lambda z: centres[z][1])
    tab["period_minutes"] = a.period_minutes
    tab = tab.sort_values(["zone", "weekend", "period"])
    tab.to_csv(out / "zone_period_demand.csv", index=False)

    cap_path = BASE_DIR / "data" / "reference" / "paris_bike_parking_libre_service.csv"
    zones = pd.DataFrame({"zone": sorted(tab["zone"].unique())})
    zones["lat"] = zones["zone"].map(lambda z: centres[z][0])
    zones["lon"] = zones["zone"].map(lambda z: centres[z][1])
    meta = pd.DataFrame([zoning.meta(z) for z in zones["zone"]])
    zones = pd.concat([zones, meta], axis=1)
    if cap_path.exists():
        spots = pd.read_csv(cap_path)
        ll = spots["geo_point_2d"].str.split(",", expand=True).astype(float)
        spots["zone"] = zoning.assign(ll[0].to_numpy(), ll[1].to_numpy())
        zones = zones.merge(spots.groupby("zone")["placal"].sum().rename("capacity"), on="zone", how="left")
        zones["capacity"] = zones["capacity"].fillna(0)
    else:
        print("warning: parking reference missing, capacity set to infinity")
        zones["capacity"] = np.inf
    peak_inv = tab.groupby("zone")["mean_inventory"].max()
    zones["peak_mean_inventory"] = zones["zone"].map(peak_inv).fillna(0)
    zones["capacity"] = np.maximum(zones["capacity"], np.ceil(zones["peak_mean_inventory"] * 1.5))
    zones.to_csv(out / "zones.csv", index=False)

    # ---- Little's Law check: fleet size and mean trip duration -----------------
    agg = tab.groupby(["weekend", "period"]).agg(dep=("departures_per_h", "sum"),
                                                 feed=("mean_inventory", "sum")).reset_index()
    A = np.column_stack([np.ones(len(agg)), -agg["dep"].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, agg["feed"].to_numpy(), rcond=None)
    pred = A @ coef
    r2 = 1 - ((agg["feed"] - pred) ** 2).sum() / ((agg["feed"] - agg["feed"].mean()) ** 2).sum()
    fleet_est, tau_h = coef[0], coef[1]

    wd = tab[~tab["weekend"]]; we = tab[tab["weekend"]]
    per_day = lambda g: g.groupby("period")["departures_per_h"].sum().mean() * 24  # noqa: E731
    lines = [
        "# Model input built", "",
        f"- files {len(files)}, zoning {a.zones}" + (f" ({a.grid_m:.0f} m)" if a.zones == "grid" else
                                                      " (80 quartiers administratifs, Paris intra-muros)")
        + f", {n_periods} periods of {a.period_minutes} min",
        f"- zones kept {len(zones)} (dropped {dropped} below {a.min_departures} departures/h)",
        f"- censoring correction: {'on' if a.censoring_correction else 'off'}",
        f"- blip filter: disappearance + appearance within {a.blip_window_min:.0f} min, <= {a.blip_radius_m:.0f} m, "
        f"<= {a.blip_range_m:.0f} m range difference ({pairs['blip_departures'].sum():,} departures and "
        f"{pairs['blip_arrivals'].sum():,} arrivals removed)",
        f"- operations filter: >= {a.ops_cluster_min} bikes vanishing within {a.ops_cluster_radius:.0f} m of each "
        f"other in one transition ({pairs['ops_excluded'].sum():,} bikes excluded, "
        f"{pairs['ops_excluded'].sum() / max(1, pairs['ops_excluded'].sum() + flows['departures'].sum()):.1%} "
        "of all disappearances)",
        f"- zone change of a visible bike counts only if it moved > {a.min_move_m:.0f} m",
        f"- availability: not reserved and range >= {a.min_range_m / 1000:.0f} km",
        f"- outside Paris intra-muros: {out_share:.1%} of departures, dropped",
        "",
        "| | rentals/day | returns/day | rentals per bike/day | peak period | peak rate |",
        "|---|---|---|---|---|---|",
    ]
    for lab, g in (("weekday", wd), ("weekend", we)):
        if not len(g):
            continue
        s = g.groupby("period")["departures_per_h"].sum()
        pk = s.idxmax()
        ret = g.groupby("period")["arrivals_per_h"].sum().mean() * 24
        lines.append(f"| {lab} | {per_day(g):,.0f} | {ret:,.0f} | {per_day(g) / fleet_est:.1f} | "
                     f"{pk * a.period_minutes // 60:02d}:{pk * a.period_minutes % 60:02d} | {s.max():,.0f}/h |")
    lines += [
        "",
        "Little's Law check  (bikes visible in feed) = fleet - mean trip duration x departure rate:",
        f"- total e-bike fleet estimated at {fleet_est:,.0f}",
        f"- mean trip duration {tau_h * 60:.1f} min, R^2 {r2:.3f}",
        f"- implies {per_day(wd) / fleet_est:.1f} rentals per bike per weekday",
        "",
        f"- parking capacity {zones['capacity'].sum():,.0f} places over {len(zones)} zones "
        f"(median {zones['capacity'].median():.0f})",
        f"- zone-periods fully empty: {(tab['empty_share'] > 0.99).mean():.1%}",
        "",
        "Files: model_input/zone_period_demand.csv, model_input/zones.csv",
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
