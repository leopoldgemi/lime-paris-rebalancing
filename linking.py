#!/usr/bin/env python3
"""
Re-link bike ids across Lime's id rotations.

Observation (2026-09-30): Lime Paris replaces *every* bike_id in the feed at
once, roughly every 15 minutes (quarter-hour boundaries). Within an epoch ids
are stable and follow the bike through a ride. Across a rotation a parked bike
keeps its exact position and range, so it can be re-identified by the
fingerprint (lat, lon, current_range_meters). Bikes that moved during the
rotation interval cannot be linked and start a new chain.

link_ids(df) adds a column "uid" (chain id = bike_id of the first sighting) and
returns the list of detected rotation boundaries.
"""

from __future__ import annotations

import pandas as pd

ROTATION_THRESHOLD = 0.5   # id overlap between consecutive snapshots below this = rotation
FP_DECIMALS = 6            # lat/lon rounding for the fingerprint


def link_ids(df: pd.DataFrame, fp_decimals: int = FP_DECIMALS, rotation_threshold: float = ROTATION_THRESHOLD):
    """Return (df with 'uid', boundaries DataFrame). df needs snapshot_utc, bike_id, lat, lon, current_range_meters."""
    df = df.copy()
    df["_fp"] = list(zip(df["lat"].round(fp_decimals), df["lon"].round(fp_decimals), df["current_range_meters"]))
    snaps = sorted(df["snapshot_utc"].unique())
    uid_of: dict[str, str] = {}
    boundaries = []
    prev = None
    for t in snaps:
        cur = df[df["snapshot_utc"] == t]
        cur_ids = set(cur["bike_id"])
        if prev is not None:
            prev_ids = set(prev["bike_id"])
            overlap = len(cur_ids & prev_ids) / max(1, len(cur_ids))
            new_ids = cur_ids - prev_ids
            vanished = prev_ids - cur_ids
            if new_ids and vanished:
                a = prev[prev["bike_id"].isin(vanished)][["bike_id", "_fp"]].drop_duplicates("_fp", keep=False)
                b = cur[cur["bike_id"].isin(new_ids)][["bike_id", "_fp"]].drop_duplicates("_fp", keep=False)
                m = b.merge(a, on="_fp", suffixes=("_new", "_old"))
                for new, old in zip(m["bike_id_new"], m["bike_id_old"]):
                    uid_of[new] = uid_of.get(old, old)
                linked = len(m)
            else:
                linked = 0
            if overlap < rotation_threshold:
                boundaries.append({"from": prev["snapshot_utc"].iloc[0], "to": t, "id_overlap": round(overlap, 4),
                                   "new_ids": len(new_ids), "linked_by_fingerprint": linked,
                                   "link_rate": round(linked / max(1, len(new_ids)), 4)})
        for bid in cur_ids:
            uid_of.setdefault(bid, bid)
        prev = cur
    df["uid"] = df["bike_id"].map(uid_of)
    df = df.drop(columns="_fp")
    return df, pd.DataFrame(boundaries)
