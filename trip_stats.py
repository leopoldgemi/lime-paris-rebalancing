#!/usr/bin/env python3
"""
First statistics on reconstructed events (reports/trips.csv) + battery state
from the snapshots. Writes tables and PNG plots to reports/.

  trips_per_hour.csv / .png        trips per hour of day (Paris time), mean per observed day
  event_share.csv / .png           trip / ops / unclear share
  trip_duration_hist.png           distribution of trip duration (gap between sightings)
  trip_distance_hist.png           distribution of straight-line trip distance
  low_range_share.csv / .png       share of vehicles with range < LOW_RANGE_KM per snapshot
  trip_stats.md                    text summary

Usage: python trip_stats.py [--trips reports/trips.csv] [--data-dir data] [--out reports]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from analyze_snapshots import BASE_DIR, HAVE_MPL, PARIS_TZ, load_snapshots

if HAVE_MPL:
    import matplotlib.pyplot as plt

LOW_RANGE_KM = 10.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trips", default="reports/trips.csv")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="reports")
    p.add_argument("--low-range-km", type=float, default=LOW_RANGE_KM)
    a = p.parse_args(argv)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else BASE_DIR / s  # noqa: E731
    trips_path, data_dir, out = resolve(a.trips), resolve(a.data_dir), resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(trips_path, parse_dates=["start_time", "end_time"])
    for c in ("start_time", "end_time"):
        ev[c] = pd.to_datetime(ev[c], utc=True)
    ev["hour"] = ev["start_time"].dt.tz_convert(PARIS_TZ).dt.hour
    ev["day"] = ev["start_time"].dt.tz_convert(PARIS_TZ).dt.date
    trips = ev[ev["event_type"] == "trip"]
    lines = ["# Event statistics", ""]

    # --- share -------------------------------------------------------------
    share = ev["event_type"].value_counts()
    share_df = pd.DataFrame({"count": share, "share": (share / share.sum()).round(3)})
    share_df.to_csv(out / "event_share.csv")
    lines += [f"- events: {len(ev):,}  (trip {share.get('trip',0):,} / ops {share.get('ops',0):,} / unclear {share.get('unclear',0):,})",
              "", share_df.to_string(), ""]

    # --- trips per hour of day ----------------------------------------------
    n_days = max(1, ev["day"].nunique())
    hours_observed = ev["start_time"].dt.tz_convert(PARIS_TZ).dt.floor("h").nunique()
    per_hour = trips.groupby("hour").size().reindex(range(24), fill_value=0)
    # normalise by how many times each hour-of-day was actually observed
    obs = pd.Series(ev["start_time"].dt.tz_convert(PARIS_TZ).dt.floor("h").unique())
    obs_per_hour = pd.Series(obs.dt.hour).value_counts().reindex(range(24), fill_value=0)
    tph = pd.DataFrame({"trips": per_hour, "hours_observed": obs_per_hour})
    tph["trips_per_observed_hour"] = (tph["trips"] / tph["hours_observed"].where(tph["hours_observed"] > 0)).round(1)
    tph.index.name = "hour_paris"
    tph.to_csv(out / "trips_per_hour.csv")
    lines += [f"- trips: {len(trips):,} over {hours_observed} observed hours on {n_days} day(s)"
              f" = {len(trips)/max(hours_observed,1):.0f} trips / h", ""]

    # --- duration / distance --------------------------------------------------
    if len(trips):
        lines += ["## Trip duration (min, gap between sightings; overstates by <= 1 interval)",
                  trips["duration_min"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string(), "",
                  "## Trip straight-line distance (m)",
                  trips["distance_m"].describe(percentiles=[.1, .25, .5, .75, .9]).round(0).to_string(), ""]
        ops = ev[ev["event_type"] == "ops"]
        if len(ops):
            lines += ["## Ops events by reason", ops["reason"].value_counts().to_string(), ""]
        unc = ev[ev["event_type"] == "unclear"]
        if len(unc):
            lines += ["## Unclear events by reason", unc["reason"].value_counts().to_string(), ""]

    # --- battery ----------------------------------------------------------------
    df = load_snapshots(data_dir)
    low = df.assign(low=(df["current_range_meters"] < a.low_range_km * 1000)).groupby("snapshot_utc")["low"].agg(["sum", "mean", "size"])
    low.columns = ["low_range_vehicles", "low_range_share", "fleet"]
    low.to_csv(out / "low_range_share.csv")
    lines += [f"## Vehicles with range < {a.low_range_km:.0f} km",
              f"- mean {low['low_range_vehicles'].mean():.0f} vehicles = {low['low_range_share'].mean():.1%} of fleet"
              f" (min {low['low_range_share'].min():.1%}, max {low['low_range_share'].max():.1%})", ""]

    # --- plots -------------------------------------------------------------------
    plots = []
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(tph.index, tph["trips_per_observed_hour"].fillna(0))
        ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hour of day (Paris)"); ax.set_ylabel("trips per observed hour")
        ax.set_title("Trips per hour of day"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(out / "trips_per_hour.png", dpi=130); plt.close(fig); plots.append("trips_per_hour.png")

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.pie(share_df["count"], labels=share_df.index, autopct="%1.0f%%", startangle=90)
        ax.set_title("Event types"); fig.tight_layout(); fig.savefig(out / "event_share.png", dpi=130); plt.close(fig)
        plots.append("event_share.png")

        if len(trips):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(trips["duration_min"], bins=range(0, int(trips["duration_min"].max()) + 5, 2))
            ax.set_xlabel("duration (min)"); ax.set_ylabel("trips"); ax.set_title("Trip duration")
            fig.tight_layout(); fig.savefig(out / "trip_duration_hist.png", dpi=130); plt.close(fig); plots.append("trip_duration_hist.png")

            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(trips["distance_m"] / 1000, bins=40)
            ax.set_xlabel("straight-line distance (km)"); ax.set_ylabel("trips"); ax.set_title("Trip distance")
            fig.tight_layout(); fig.savefig(out / "trip_distance_hist.png", dpi=130); plt.close(fig); plots.append("trip_distance_hist.png")

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(low.index.tz_convert(PARIS_TZ), low["low_range_share"] * 100)
        ax.set_ylabel(f"% fleet with range < {a.low_range_km:.0f} km"); ax.set_title("Low-battery share per snapshot")
        ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(out / "low_range_share.png", dpi=130); plt.close(fig); plots.append("low_range_share.png")
    else:
        lines.append("(matplotlib not installed - no plots)")
    lines += ["## Files", ""] + [f"- reports/{f}" for f in
                                 ["trips_per_hour.csv", "event_share.csv", "low_range_share.csv"] + plots]
    text = "\n".join(lines) + "\n"
    (out / "trip_stats.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
