#!/usr/bin/env python3
"""
Is bike_id stable over time, or does Lime rotate ids (e.g. daily)?

Checks, over all collected snapshots:
  * share of ids from the first snapshot still present in the last snapshot
  * per snapshot: number of ids never seen before ("new") and ids seen for the
    last time ("gone"); a rotation would show as a spike of both at the same moment
  * same thing aggregated per Paris calendar day (daily rotation check)
  * distribution of how long an id is observed (first seen -> last seen)

Usage: python check_id_stability.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from analyze_snapshots import BASE_DIR, PARIS_TZ, load_snapshots
from linking import link_ids


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)
    data_dir = Path(a.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else BASE_DIR / data_dir

    raw = load_snapshots(data_dir)
    linked, boundaries = link_ids(raw)
    df = raw[["snapshot_utc", "bike_id"]]
    snaps = sorted(df["snapshot_utc"].unique())
    first, last = snaps[0], snaps[-1]
    ids_first = set(df.loc[df["snapshot_utc"] == first, "bike_id"])
    ids_last = set(df.loc[df["snapshot_utc"] == last, "bike_id"])
    span_h = (last - first).total_seconds() / 3600
    all_ids = df["bike_id"].nunique()

    print(f"snapshots: {len(snaps)}   span: {span_h:.1f} h   ({first:%Y-%m-%d %H:%M} -> {last:%Y-%m-%d %H:%M} UTC)")
    print(f"ids in first snapshot: {len(ids_first):,}   in last: {len(ids_last):,}   distinct overall: {all_ids:,}")
    inter = len(ids_first & ids_last)
    print(f"first ∩ last: {inter:,}  = {inter/len(ids_first):.1%} of first, {inter/len(ids_last):.1%} of last")
    print(f"distinct ids / mean fleet size: {all_ids / df.groupby('snapshot_utc').size().mean():.2f}"
          "  (≈1.0 means stable ids; grows with a rotation or churn)")

    # first / last appearance per id
    life = df.groupby("bike_id")["snapshot_utc"].agg(["min", "max"])
    life["hours_seen"] = (life["max"] - life["min"]).dt.total_seconds() / 3600
    print("\nobserved lifetime per id (first seen -> last seen), hours:")
    print(life["hours_seen"].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())

    # new / gone per snapshot
    per_snap = pd.DataFrame({
        "new_ids": life.groupby("min").size(),
        "gone_ids": life.groupby("max").size(),
    }).reindex(snaps).fillna(0).astype(int)
    per_snap.index.name = "snapshot_utc"
    per_snap["fleet"] = df.groupby("snapshot_utc").size()
    # first and last snapshot are trivially all-new / all-gone
    inner = per_snap.iloc[1:-1]
    if len(inner):
        print("\nnew / gone ids per snapshot (excluding first and last snapshot):")
        print(inner[["new_ids", "gone_ids"]].describe().round(1).loc[["mean", "50%", "max"]].to_string())
        top = inner.sort_values("new_ids", ascending=False).head(5)
        print("\nsnapshots with most new ids (a rotation would show thousands at once):")
        print(top.to_string())

    # rotations + fingerprint linking
    print(f"\nid rotations detected (consecutive-snapshot id overlap < 50%): {len(boundaries)}")
    if len(boundaries):
        b = boundaries.copy()
        b["from"] = b["from"].dt.strftime("%m-%d %H:%M:%S"); b["to"] = b["to"].dt.strftime("%m-%d %H:%M:%S")
        print(b.to_string(index=False))
        gaps = boundaries["to"].diff().dt.total_seconds().div(60).dropna()
        if len(gaps):
            print(f"time between rotations: median {gaps.median():.1f} min, min {gaps.min():.1f}, max {gaps.max():.1f}")
        chains = linked["uid"].nunique()
        print(f"after fingerprint linking: {chains:,} chains (uid) instead of {all_ids:,} raw ids; "
              f"chains / mean fleet = {chains / df.groupby('snapshot_utc').size().mean():.2f}")
        life2 = linked.groupby("uid")["snapshot_utc"].agg(["min", "max"])
        print(f"median observed lifetime per chain: {((life2['max']-life2['min']).dt.total_seconds()/3600).median():.2f} h")

    # per Paris day
    day = df["snapshot_utc"].dt.tz_convert(PARIS_TZ).dt.date
    ids_by_day = df.assign(day=day).groupby("day")["bike_id"].apply(set)
    if len(ids_by_day) > 1:
        print("\nid overlap between consecutive Paris days:")
        days = list(ids_by_day.index)
        for d0, d1 in zip(days, days[1:]):
            s0, s1 = ids_by_day[d0], ids_by_day[d1]
            print(f"  {d0} -> {d1}: {len(s0 & s1):,} shared = {len(s0 & s1)/len(s1):.1%} of {d1}'s ids")
    else:
        print("\nonly one calendar day collected so far - rerun after >= 2 days for the daily-rotation check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
