#!/usr/bin/env python3
"""
Censored demand: how often are zones empty, so that demand cannot be observed?

Zones: square grid 500 m and 1 km over Paris (metre projection around 48.86 N).
Only e-bikes (vehicle_type_id == 3). Zones with no e-bike in the whole period
are ignored (Seine, parks, outside the service area).

Variants (all computed):
  availability  A: not reserved AND range >= RANGE_MIN_M      B: not reserved (no battery filter)
  empty         0: available == 0                              1: available <= 1

Outputs (reports/censored_demand/):
  summary.md, datasets.csv, zone_stats_{500,1000}.csv, hourly_{500,1000}.csv,
  top15_{500,1000}.csv, departures_{500,1000}.csv, heatmap_empty_500m.png

Departures per zone-hour: a bike chain (uid, see linking.py) present in a zone
at snapshot t and not in the same zone at the next snapshot. Snapshot pairs
that span an id rotation are skipped (unlinked bikes would look like departures).

Usage: python censored_demand.py [--data-dir data] [--out reports/censored_demand]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_snapshots import BASE_DIR, HAVE_MPL, M_PER_DEG_LAT, M_PER_DEG_LON, PARIS_TZ, load_snapshots
from linking import link_ids

if HAVE_MPL:
    import matplotlib.pyplot as plt

RANGE_MIN_M = 5_000
GAP_CAP_MIN = 10.0          # time weight of a snapshot = gap to next snapshot, capped here
GRIDS = (500, 1000)
EBIKE_TYPE = "3"
MIN_EVENTS_STABLE = 30      # events per zone and bucket for a "stable" rate estimate (rule of thumb)
DEPARTURE_MIN_DIST_M = 200  # next sighting at least this far away = the bike left (same as MIN_TRIP_DIST_M)


def add_zone(df: pd.DataFrame, size_m: int) -> pd.Series:
    x = np.floor(df["lon"] * M_PER_DEG_LON / size_m).astype(int)
    y = np.floor(df["lat"] * M_PER_DEG_LAT / size_m).astype(int)
    return x.astype(str) + "_" + y.astype(str)


def zone_centroid(zone: str, size_m: int) -> tuple[float, float]:
    x, y = (int(v) for v in zone.split("_"))
    return ((y + 0.5) * size_m / M_PER_DEG_LAT, (x + 0.5) * size_m / M_PER_DEG_LON)


def load_arrondissement_lookup(data_dir: Path):
    """Nearest-parking-spot arrondissement (from fetch_parking_zones.py export), None if missing."""
    p = data_dir / "reference" / "paris_bike_parking_libre_service.csv"
    if not p.exists():
        return None
    spots = pd.read_csv(p)
    ll = spots["geo_point_2d"].str.split(",", expand=True).astype(float)
    pts = np.column_stack([ll[0] * M_PER_DEG_LAT, ll[1] * M_PER_DEG_LON])
    arr = spots["arrond"].to_numpy()

    def lookup(lat, lon):
        d = (pts[:, 0] - lat * M_PER_DEG_LAT) ** 2 + (pts[:, 1] - lon * M_PER_DEG_LON) ** 2
        i = int(d.argmin())
        return int(arr[i]) if math.sqrt(d[i]) < 700 else None
    return lookup


def snapshot_weights(snaps: pd.DatetimeIndex) -> pd.Series:
    gaps = pd.Series(snaps).diff().shift(-1).dt.total_seconds().div(60)
    gaps = gaps.fillna(gaps.median()).clip(upper=GAP_CAP_MIN)
    return pd.Series(gaps.values, index=snaps)


def analyse_grid(df: pd.DataFrame, size_m: int, boundaries: pd.DataFrame, arr_lookup, out: Path) -> dict:
    d = df.copy()
    d["zone"] = add_zone(d, size_m)
    snaps = pd.DatetimeIndex(sorted(d["snapshot_utc"].unique()))
    w = snapshot_weights(snaps)
    active = sorted(d["zone"].unique())
    res = {"size_m": size_m, "n_zones": len(active)}

    # available counts per zone x snapshot for both availability variants
    counts = {}
    for key, mask in (("A", (d["is_reserved"] == 0) & (d["current_range_meters"] >= RANGE_MIN_M)),
                      ("B", d["is_reserved"] == 0)):
        c = d[mask].groupby(["zone", "snapshot_utc"]).size().unstack("snapshot_utc", fill_value=0)
        counts[key] = c.reindex(index=active, columns=snaps, fill_value=0)

    local = snaps.tz_convert(PARIS_TZ)
    hour_of_day = pd.Series(local.hour, index=snaps)
    hour_bucket = pd.Series(local.floor("h"), index=snaps)
    weekend = pd.Series(local.dayofweek >= 5, index=snaps)

    hourly_rows, zone_rows = [], []
    for avail in ("A", "B"):
        for thr in (0, 1):
            empty = counts[avail] <= thr                      # zones x snapshots (bool)
            key = f"{avail}{thr}"
            # share of zone-time empty (time-weighted)
            share_time = float((empty * w.values).sum().sum() / (w.sum() * len(active)))
            # share of zone-hours with >= 1 empty snapshot
            zh = empty.T.groupby(hour_bucket.values).any()    # hour_bucket x zones
            share_zh = float(zh.values.mean())
            res[f"share_time_{key}"] = share_time
            res[f"share_zonehours_{key}"] = share_zh
            # by hour of day / weekend
            for h in range(24):
                cols = snaps[hour_of_day.values == h]
                if len(cols) == 0:
                    continue
                e = empty[cols]
                hourly_rows.append({"variant": key, "hour_paris": h, "n_snapshots": len(cols),
                                    "share_time_empty": float((e * w[cols].values).sum().sum() / (w[cols].sum() * len(active)))})
            for wk, lab in ((False, "weekday"), (True, "weekend")):
                cols = snaps[weekend.values == wk]
                if len(cols):
                    e = empty[cols]
                    res[f"share_time_{key}_{lab}"] = float((e * w[cols].values).sum().sum() / (w[cols].sum() * len(active)))
            # per zone
            zs = (empty * w.values).sum(axis=1) / w.sum()
            if avail == "A" and thr == 0:
                zone_share_main = zs
            zone_rows.append(zs.rename(f"empty_share_{key}"))

    zones = pd.concat(zone_rows, axis=1)
    zones["mean_available_A"] = counts["A"].mean(axis=1)
    zones["mean_ebikes"] = d.groupby(["zone", "snapshot_utc"]).size().unstack(fill_value=0).reindex(index=active, columns=snaps, fill_value=0).mean(axis=1)
    cent = [zone_centroid(z, size_m) for z in zones.index]
    zones["lat"] = [c[0] for c in cent]
    zones["lon"] = [c[1] for c in cent]
    zones["arrondissement"] = [arr_lookup(c[0], c[1]) if arr_lookup else None for c in cent]

    # departures per zone-hour.
    # A departure of a chain from zone Z at sighting t means: the chain's next sighting is
    # >= DEPARTURE_MIN_DIST_M away (ride or relocation), or the chain is never seen again
    # although >= 3 more snapshots follow (vanished = rented or collected). Feed blips
    # (bike missing for one snapshot, back at the same spot) and cell-border GPS jitter
    # are therefore not departures. Sightings right before an id rotation are skipped
    # because an unlinked chain end there is an artefact.
    rot_to = set(pd.to_datetime(boundaries["to"])) if len(boundaries) else set()
    before_rot = {snaps[i] for i in range(len(snaps) - 1) if snaps[i + 1] in rot_to}
    dd = d.sort_values(["uid", "snapshot_utc"])[["uid", "snapshot_utc", "lat", "lon", "zone"]]
    nxt = dd.groupby("uid").shift(-1)
    dist = np.sqrt(((dd["lat"] - nxt["lat"]) * M_PER_DEG_LAT) ** 2 + ((dd["lon"] - nxt["lon"]) * M_PER_DEG_LON) ** 2)
    snap_pos = {t: i for i, t in enumerate(snaps)}
    idx = dd["snapshot_utc"].map(snap_pos)
    vanished = nxt["snapshot_utc"].isna() & (idx <= len(snaps) - 4)
    relocated = dist >= DEPARTURE_MIN_DIST_M
    valid = ~dd["snapshot_utc"].isin(before_rot)
    dep = dd[(vanished | relocated) & valid]
    departures = dep.assign(hour_bucket=dep["snapshot_utc"].map(hour_bucket)).groupby(["zone", "hour_bucket"]).size().rename("departures").reset_index()
    # observed minutes per hour bucket (sightings usable for departure detection)
    bucket_minutes: dict = {}
    observed_min = 0.0
    for t0, t1 in zip(snaps, snaps[1:]):
        if t1 in rot_to or (t1 - t0).total_seconds() / 60 > GAP_CAP_MIN:
            continue
        g = (t1 - t0).total_seconds() / 60
        observed_min += g
        bucket_minutes[hour_bucket[t0]] = bucket_minutes.get(hour_bucket[t0], 0) + g
    full = departures.copy()
    full["minutes_observed"] = full["hour_bucket"].map(bucket_minutes)
    full = full.dropna(subset=["minutes_observed"])
    full["departures_per_hour"] = full["departures"] / full["minutes_observed"] * 60
    all_pairs = pd.MultiIndex.from_product([active, list(bucket_minutes)], names=["zone", "hour_bucket"])
    rate = full.set_index(["zone", "hour_bucket"])["departures_per_hour"].reindex(all_pairs, fill_value=0.0)
    zone_rate = rate.groupby("zone").mean()
    zones["departures_per_hour"] = zone_rate
    res["observed_minutes_for_departures"] = observed_min
    res["departures_total"] = int(full["departures"].sum()) if len(full) else 0
    res["departures_relocated"] = int((relocated & valid).sum())   # next sighting >= 200 m away (confirmed move)
    res["departures_vanished"] = int((vanished & ~relocated & valid).sum())  # never seen again (rented, collected, or feed dropout)
    res["fleet_departures_per_hour"] = res["departures_total"] / observed_min * 60 if observed_min else 0.0
    res["dep_per_zone_hour_median"] = float(zone_rate.median())
    res["dep_per_zone_hour_mean"] = float(zone_rate.mean())
    res["dep_per_zone_hour_p90"] = float(zone_rate.quantile(0.9))
    res["share_zones_with_departure"] = float((zone_rate > 0).mean())
    med, mean = zone_rate.median(), zone_rate.mean()
    res["days_for_stable_hourly_rate_median_zone"] = float(MIN_EVENTS_STABLE / med) if med > 0 else float("inf")
    res["days_for_stable_hourly_rate_mean_zone"] = float(MIN_EVENTS_STABLE / mean) if mean > 0 else float("inf")

    zones = zones.sort_values("empty_share_A0", ascending=False)
    zones.to_csv(out / f"zone_stats_{size_m}.csv")
    zones.head(15).to_csv(out / f"top15_{size_m}.csv")
    pd.DataFrame(hourly_rows).to_csv(out / f"hourly_{size_m}.csv", index=False)
    full.to_csv(out / f"departures_{size_m}.csv", index=False)
    res["zones"] = zones
    res["hourly"] = pd.DataFrame(hourly_rows)
    return res


def heatmap(zones: pd.DataFrame, size_m: int, out: Path) -> str | None:
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(8, 7.5))
    sc = ax.scatter(zones["lon"], zones["lat"], c=zones["empty_share_A0"] * 100, cmap="Reds", vmin=0, vmax=100,
                    s=28, marker="s", edgecolors="grey", linewidths=0.2)
    ax.set_aspect(M_PER_DEG_LAT / M_PER_DEG_LON)
    ax.set_xlim(zones["lon"].quantile(0.005) - 0.01, zones["lon"].quantile(0.995) + 0.01)
    ax.set_ylim(zones["lat"].quantile(0.005) - 0.005, zones["lat"].quantile(0.995) + 0.005)
    ax.set_title(f"Share of time with 0 available e-bikes per {size_m} m zone\n(available = not reserved, range >= {RANGE_MIN_M/1000:.0f} km)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(sc, ax=ax, label="% of observed time empty")
    fig.tight_layout()
    name = f"heatmap_empty_{size_m}m.png"
    fig.savefig(out / name, dpi=140)
    plt.close(fig)
    return name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="reports/censored_demand")
    a = p.parse_args(argv)
    data_dir = Path(a.data_dir); data_dir = data_dir if data_dir.is_absolute() else BASE_DIR / data_dir
    out = Path(a.out); out = out if out.is_absolute() else BASE_DIR / out
    out.mkdir(parents=True, exist_ok=True)

    raw = load_snapshots(data_dir)
    raw = raw[raw["vehicle_type_id"].astype(str) == EBIKE_TYPE]
    df, boundaries = link_ids(raw)

    # 1. data basis
    snaps = pd.DatetimeIndex(sorted(df["snapshot_utc"].unique()))
    gaps = pd.Series(snaps).diff().dt.total_seconds().div(60).dropna()
    big_gaps = [(snaps[i], snaps[i + 1], g) for i, g in enumerate(gaps) if g > 10]
    t0, t1 = snaps[0], snaps[-1]
    span_h = (t1 - t0).total_seconds() / 3600
    basis = {
        "from_utc": t0, "to_utc": t1, "span_hours": round(span_h, 2), "n_snapshots": len(snaps),
        "median_interval_min": round(float(gaps.median()), 2), "mean_interval_min": round(float(gaps.mean()), 2),
        "gaps_over_10min": len(big_gaps), "id_rotations": len(boundaries),
        "days_covered": df["snapshot_utc"].dt.tz_convert(PARIS_TZ).dt.date.nunique(),
    }
    pd.DataFrame([basis]).to_csv(out / "datasets.csv", index=False)

    arr_lookup = load_arrondissement_lookup(data_dir)
    results = {g: analyse_grid(df, g, boundaries, arr_lookup, out) for g in GRIDS}
    hm = heatmap(results[500]["zones"], 500, out)

    # ---- summary -----------------------------------------------------------
    L = []
    L.append("# Censored demand - empty zones")
    L.append("")
    L.append(f"Data: {t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M} UTC ({span_h:.1f} h, {basis['days_covered']} calendar day(s)), "
             f"{len(snaps)} snapshots, median interval {basis['median_interval_min']:.1f} min, "
             f"{len(big_gaps)} gap(s) > 10 min" + (": " + ", ".join(f"{a:%H:%M}-{b:%H:%M} ({g:.0f} min)" for a, b, g in big_gaps) if big_gaps else "")
             + f", {len(boundaries)} id rotations.")
    L.append(f"E-bikes only (type 3): {df['uid'].nunique():,} chains, mean {df.groupby('snapshot_utc').size().mean():.0f} per snapshot.")
    L.append("")
    L.append("| grid | active zones | zone-hours with >=1 empty snapshot | zone-time empty | ... no battery filter | ... <=1 bike | ... <=1, no battery filter |")
    L.append("|---|---|---|---|---|---|---|")
    for g in GRIDS:
        r = results[g]
        L.append(f"| {g} m | {r['n_zones']} | {r['share_zonehours_A0']:.1%} | {r['share_time_A0']:.1%} | {r['share_time_B0']:.1%} | {r['share_time_A1']:.1%} | {r['share_time_B1']:.1%} |")
    L.append("")
    for g in GRIDS:
        r = results[g]
        h = r["hourly"]; h = h[h["variant"] == "A0"]
        L.append(f"Hour of day (Paris), {g} m, share of zone-time empty (variant A0): " +
                 ", ".join(f"{int(row.hour_paris):02d}h {row.share_time_empty:.0%}" for row in h.itertuples()))
        wk = [k for k in ("share_time_A0_weekday", "share_time_A0_weekend") if k in r]
        L.append("  weekday/weekend: " + ", ".join(f"{k.split('_')[-1]} {r[k]:.1%}" for k in wk) if len(wk) == 2 else "  weekday vs weekend: not enough days yet")
    L.append("")
    for g in GRIDS:
        r = results[g]
        L.append(f"Top 15 most often empty {g} m zones (variant A0):")
        L.append("")
        L.append("| arr. | lat | lon | % time empty | mean e-bikes | mean available | departures/h |")
        L.append("|---|---|---|---|---|---|---|")
        for z in r["zones"].head(15).itertuples():
            L.append(f"| {int(z.arrondissement) if pd.notna(z.arrondissement) else 'suburb'} | {z.lat:.4f} | {z.lon:.4f} | {z.empty_share_A0:.0%} | {z.mean_ebikes:.1f} | {z.mean_available_A:.1f} | {z.departures_per_hour:.1f} |")
        L.append("")
    L.append("Departures per zone and hour (chains leaving the zone; rotation-spanning pairs skipped):")
    L.append("")
    L.append("| grid | departures observed | observed minutes | fleet-wide per hour | zones with >=1 departure | median zone /h | mean zone /h | p90 zone /h | days until 30 departures per hour-of-day bucket (median zone / mean zone) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for g in GRIDS:
        r = results[g]
        dm, da = r["days_for_stable_hourly_rate_median_zone"], r["days_for_stable_hourly_rate_mean_zone"]
        L.append(f"| {g} m | {r['departures_total']:,} ({r['departures_relocated']} relocated + {r['departures_vanished']} vanished) | {r['observed_minutes_for_departures']:.0f} | {r['fleet_departures_per_hour']:.0f} | "
                 f"{r['share_zones_with_departure']:.0%} | {r['dep_per_zone_hour_median']:.2f} | {r['dep_per_zone_hour_mean']:.2f} | "
                 f"{r['dep_per_zone_hour_p90']:.2f} | {dm:.0f} / {da:.0f} |")
    L.append("")
    if hm:
        L.append(f"Heatmap: reports/censored_demand/{hm}")
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
