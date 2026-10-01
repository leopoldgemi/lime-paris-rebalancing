#!/usr/bin/env python3
"""
Why do ~900 bikes/hour "vanish" from the feed at night? Is it the id rotation?

1. Per snapshot transition: vanished ids, new ids, rotation markers (plot).
2. Pair check: vanished bikes with a new bike < PAIR_RADIUS_M away in the same
   transition; distribution of distances and range differences.
3. Optimal matching (scipy linear_sum_assignment per connected component of the
   candidate graph) between vanished and new bikes with cost = distance, allowed
   only if distance < X and range difference within [-Y, +RANGE_UP_TOL_M].
   Sensitivity over X and Y: how many real departures per hour remain.
4. With the chosen (X, Y): departures and arrivals per hour fleet-wide and per
   1 km zone.
Extra: last_reported age of vanished vs staying bikes (staleness filter test).

Usage: python vanish_diagnosis.py [--data-dir data] [--out reports/vanish] [--x 30] [--y 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from analyze_snapshots import BASE_DIR, HAVE_MPL, M_PER_DEG_LAT, M_PER_DEG_LON, PARIS_TZ, load_snapshots
from censored_demand import add_zone

if HAVE_MPL:
    import matplotlib.pyplot as plt

PAIR_RADIUS_M = 30
ROTATION_THRESHOLD = 0.5
RANGE_UP_TOL_M = 100          # range may rise by at most this while parked (noise); larger = swap, not a match
X_GRID = (5, 15, 30, 60, 100)
Y_GRID = (0, 200, 500, 1000, 3000)
GAP_CAP_MIN = 10


def xy(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df["lon"].to_numpy() * M_PER_DEG_LON, df["lat"].to_numpy() * M_PER_DEG_LAT])


def match(vanished: pd.DataFrame, new: pd.DataFrame, x_m: float, y_m: float):
    """Optimal 1:1 matching, returns array of (idx_vanished, idx_new, dist, drange)."""
    if len(vanished) == 0 or len(new) == 0:
        return np.empty((0, 4))
    pv, pn = xy(vanished), xy(new)
    rv, rn = vanished["current_range_meters"].to_numpy(), new["current_range_meters"].to_numpy()
    tree = cKDTree(pn)
    cand = tree.query_ball_point(pv, r=x_m)
    edges = []
    for i, js in enumerate(cand):
        for j in js:
            dr = rn[j] - rv[i]
            if -y_m <= dr <= RANGE_UP_TOL_M:
                edges.append((i, j, float(np.hypot(*(pn[j] - pv[i]))), float(dr)))
    if not edges:
        return np.empty((0, 4))
    e = pd.DataFrame(edges, columns=["i", "j", "d", "dr"])
    # connected components via union-find on bipartite graph
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(e["i"], e["j"]):
        parent[find(("v", i))] = find(("n", j))
    e["comp"] = [find(("v", i)) for i in e["i"]]
    out = []
    for _, g in e.groupby("comp"):
        vi = sorted(g["i"].unique()); nj = sorted(g["j"].unique())
        if len(vi) == 1 and len(nj) == 1:
            r = g.iloc[g["d"].argmin()]
            out.append((r["i"], r["j"], r["d"], r["dr"]))
            continue
        cost = np.full((len(vi), len(nj)), 1e6)
        vi_idx = {v: k for k, v in enumerate(vi)}; nj_idx = {n: k for k, n in enumerate(nj)}
        for r in g.itertuples():
            cost[vi_idx[r.i], nj_idx[r.j]] = r.d
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            if cost[a, b] < 1e6:
                row = g[(g["i"] == vi[a]) & (g["j"] == nj[b])].iloc[0]
                out.append((vi[a], nj[b], row["d"], row["dr"]))
    return np.array(out) if out else np.empty((0, 4))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="reports/vanish")
    p.add_argument("--x", type=float, default=30, help="max matching distance (m) for the final run")
    p.add_argument("--y", type=float, default=500, help="max range decrease (m) for the final run")
    a = p.parse_args(argv)
    data_dir = Path(a.data_dir); data_dir = data_dir if data_dir.is_absolute() else BASE_DIR / data_dir
    out = Path(a.out); out = out if out.is_absolute() else BASE_DIR / out
    out.mkdir(parents=True, exist_ok=True)

    df = load_snapshots(data_dir)
    df = df[df["vehicle_type_id"].astype(str) == "3"].copy()
    snaps = sorted(df["snapshot_utc"].unique())
    by_t = {t: df[df["snapshot_utc"] == t].set_index("bike_id") for t in snaps}
    L = ["# Vanishing bikes - diagnosis", ""]

    # ---- 1. per transition --------------------------------------------------------
    rows = []
    trans = []
    for t0, t1 in zip(snaps, snaps[1:]):
        a0, a1 = by_t[t0], by_t[t1]
        ids0, ids1 = set(a0.index), set(a1.index)
        van = a0.loc[sorted(ids0 - ids1)]
        new = a1.loc[sorted(ids1 - ids0)]
        overlap = len(ids0 & ids1) / max(1, len(ids1))
        gap = (t1 - t0).total_seconds() / 60
        rot = overlap < ROTATION_THRESHOLD
        # staleness test: last_reported age at t0
        age_van = (t0.timestamp() - van["last_reported"]).median() / 60 if len(van) else np.nan
        stay = a0.loc[sorted(ids0 & ids1)]
        age_stay = (t0.timestamp() - stay["last_reported"]).median() / 60 if len(stay) else np.nan
        rows.append({"t0": t0, "t1": t1, "gap_min": gap, "rotation": rot, "fleet_t1": len(ids1),
                     "vanished": len(van), "new": len(new), "vanished_per_h": len(van) / gap * 60,
                     "new_per_h": len(new) / gap * 60, "age_vanished_min": age_van, "age_staying_min": age_stay})
        trans.append((t0, t1, van, new, rot, gap))
    tr = pd.DataFrame(rows)
    tr.to_csv(out / "transitions.csv", index=False)
    nr = tr[~tr["rotation"] & (tr["gap_min"] <= GAP_CAP_MIN)]
    L.append(f"Transitions: {len(tr)} ({tr['rotation'].sum()} rotations). Non-rotation transitions (<= {GAP_CAP_MIN} min): "
             f"vanished {nr['vanished_per_h'].mean():.0f}/h (median {nr['vanished_per_h'].median():.0f}), "
             f"new {nr['new_per_h'].mean():.0f}/h. Rotation transitions: all ids vanish/new by definition.")
    L.append(f"last_reported age at last sighting: vanished bikes median {nr['age_vanished_min'].median():.1f} min, "
             f"staying bikes median {nr['age_staying_min'].median():.1f} min.")

    # ---- 2. pair check (< 30 m) -------------------------------------------------------
    pair_rows = []
    for t0, t1, van, new, rot, gap in trans:
        if len(van) == 0 or len(new) == 0:
            continue
        tree = cKDTree(xy(new))
        d, j = tree.query(xy(van), k=1)
        close = d < PAIR_RADIUS_M
        dr = new["current_range_meters"].to_numpy()[j] - van["current_range_meters"].to_numpy()
        pair_rows.append(pd.DataFrame({"t0": t0, "rotation": rot, "dist_m": d, "drange_m": dr, "close": close}))
    pairs = pd.concat(pair_rows)
    pairs.to_csv(out / "pairs.csv", index=False)
    for rot, lab in ((True, "rotation"), (False, "non-rotation")):
        g = pairs[pairs["rotation"] == rot]
        if len(g) == 0:
            continue
        c = g[g["close"]]
        L.append(f"Pair check {lab} transitions: {len(g):,} vanished, {c.shape[0]:,} ({c.shape[0]/len(g):.1%}) have a new bike < {PAIR_RADIUS_M} m; "
                 f"of those |drange| == 0: {(c['drange_m']==0).mean():.1%}, |drange| <= 500 m: {(c['drange_m'].abs()<=500).mean():.1%}; "
                 f"nearest-new distance median {g['dist_m'].median():.0f} m, p90 {g['dist_m'].quantile(.9):.0f} m.")

    # ---- 3. sensitivity -------------------------------------------------------------------
    sens = []
    for x in X_GRID:
        for y in Y_GRID:
            unm_rot = unm_non = 0; min_rot = min_non = 0.0; n_rot = n_non = 0
            for t0, t1, van, new, rot, gap in trans:
                if gap > GAP_CAP_MIN and not rot:
                    continue
                m = match(van, new, x, y)
                unmatched = len(van) - len(m)
                if rot:
                    unm_rot += unmatched; min_rot += gap; n_rot += len(van)
                else:
                    unm_non += unmatched; min_non += gap; n_non += len(van)
            sens.append({"X_m": x, "Y_m": y,
                         "rot_vanished": n_rot, "rot_unmatched": unm_rot, "rot_unmatched_per_h": unm_rot / min_rot * 60 if min_rot else np.nan,
                         "non_vanished": n_non, "non_unmatched": unm_non, "non_unmatched_per_h": unm_non / min_non * 60 if min_non else np.nan})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity.csv", index=False)
    L += ["", "Sensitivity: unmatched vanished bikes per hour (= candidate real departures) after optimal matching", "",
          "| X (m) | Y (m) | rotation transitions: unmatched/h (of vanished) | non-rotation: unmatched/h (of vanished) |", "|---|---|---|---|"]
    for r in sens.itertuples():
        L.append(f"| {r.X_m} | {r.Y_m} | {r.rot_unmatched_per_h:.0f} ({r.rot_unmatched}/{r.rot_vanished}) | {r.non_unmatched_per_h:.0f} ({r.non_unmatched}/{r.non_vanished}) |")

    # ---- 4. chains with chosen X, Y -> departures / arrivals -------------------------------
    uid_of: dict[str, str] = {}
    for bid in by_t[snaps[0]].index:
        uid_of[bid] = bid
    dep_rows, arr_rows = [], []
    for t0, t1, van, new, rot, gap in trans:
        if gap > GAP_CAP_MIN and not rot:
            for bid in new.index:
                uid_of.setdefault(bid, bid)
            continue
        m = match(van, new, a.x, a.y)
        matched_v = {int(i) for i in m[:, 0]} if len(m) else set()
        matched_n = {int(j) for j in m[:, 1]} if len(m) else set()
        for i, j in zip(m[:, 0], m[:, 1]) if len(m) else []:
            uid_of[new.index[int(j)]] = uid_of.get(van.index[int(i)], van.index[int(i)])
        for k, bid in enumerate(new.index):
            if k not in matched_n:
                uid_of.setdefault(bid, bid)
                arr_rows.append({"t": t1, "lat": new["lat"].iloc[k], "lon": new["lon"].iloc[k], "gap_min": gap, "rotation": rot})
        for k, bid in enumerate(van.index):
            if k not in matched_v:
                dep_rows.append({"t": t0, "lat": van["lat"].iloc[k], "lon": van["lon"].iloc[k], "gap_min": gap, "rotation": rot})
        # relocations of surviving ids (>= 200 m) are departures + arrivals too
        a0, a1 = by_t[t0], by_t[t1]
        common = sorted(set(a0.index) & set(a1.index))
        if common:
            d = np.hypot((a1.loc[common, "lon"].to_numpy() - a0.loc[common, "lon"].to_numpy()) * M_PER_DEG_LON,
                         (a1.loc[common, "lat"].to_numpy() - a0.loc[common, "lat"].to_numpy()) * M_PER_DEG_LAT)
            for bid, dd in zip(common, d):
                if dd >= 200:
                    dep_rows.append({"t": t0, "lat": a0.at[bid, "lat"], "lon": a0.at[bid, "lon"], "gap_min": gap, "rotation": rot})
                    arr_rows.append({"t": t1, "lat": a1.at[bid, "lat"], "lon": a1.at[bid, "lon"], "gap_min": gap, "rotation": rot})
    dep = pd.DataFrame(dep_rows); arr = pd.DataFrame(arr_rows)
    obs_min = sum(gap for *_, rot, gap in trans if gap <= GAP_CAP_MIN or rot)
    for name, ev in (("departures", dep), ("arrivals", arr)):
        ev["zone_1km"] = add_zone(ev, 1000) if len(ev) else []
        ev.to_csv(out / f"{name}.csv", index=False)
    L += ["", f"With X={a.x:.0f} m, Y={a.y:.0f} m over {obs_min:.0f} observed minutes:",
          f"- departures: {len(dep):,} = {len(dep)/obs_min*60:.0f}/h fleet-wide ({(dep['rotation']).sum() if len(dep) else 0} at rotation transitions)",
          f"- arrivals:   {len(arr):,} = {len(arr)/obs_min*60:.0f}/h fleet-wide ({(arr['rotation']).sum() if len(arr) else 0} at rotation transitions)"]
    if len(dep):
        zones_active = df.assign(z=add_zone(df, 1000))["z"].nunique()
        per_zone = dep.groupby("zone_1km").size() / obs_min * 60
        per_zone = per_zone.reindex(df.assign(z=add_zone(df, 1000))["z"].unique(), fill_value=0)
        L.append(f"- per 1 km zone ({zones_active} active): departures/h median {per_zone.median():.2f}, mean {per_zone.mean():.2f}, "
                 f"p90 {per_zone.quantile(.9):.2f}, zones with 0: {(per_zone==0).mean():.0%}")
        per_zone.sort_values(ascending=False).rename("departures_per_h").to_csv(out / "departures_per_zone_1km.csv")

    # ---- plots -----------------------------------------------------------------------------------
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(11, 4))
        x = tr["t1"].dt.tz_convert(PARIS_TZ)
        ax.plot(x, tr["vanished"], marker="o", ms=3, label="vanished ids")
        ax.plot(x, tr["new"], marker="o", ms=3, label="new ids")
        for t in tr.loc[tr["rotation"], "t1"].dt.tz_convert(PARIS_TZ):
            ax.axvline(t, color="red", alpha=0.4, ls="--")
        ax.set_yscale("log"); ax.set_ylabel("ids per transition (log)"); ax.set_title("Vanished / new ids per snapshot transition (red = id rotation)")
        ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(out / "vanished_new_over_time.png", dpi=130); plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        for rot, lab, c in ((False, "non-rotation", "C0"), (True, "rotation", "C3")):
            g = pairs[pairs["rotation"] == rot]
            if len(g):
                axs[0].hist(g["dist_m"].clip(upper=500), bins=50, alpha=0.6, label=lab, color=c)
                cl = g[g["close"]]
                axs[1].hist(cl["drange_m"].clip(-3000, 3000), bins=60, alpha=0.6, label=f"{lab} (< {PAIR_RADIUS_M} m)", color=c)
        axs[0].set_xlabel("distance vanished -> nearest new bike (m, clipped 500)"); axs[0].set_yscale("log"); axs[0].legend()
        axs[1].set_xlabel("range difference new - vanished (m, clipped)"); axs[1].set_yscale("log"); axs[1].legend()
        fig.tight_layout(); fig.savefig(out / "pair_distances.png", dpi=130); plt.close(fig)

        piv = sens.pivot(index="Y_m", columns="X_m", values="non_unmatched_per_h")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        im = ax.imshow(piv.values, cmap="viridis", origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns); ax.set_xlabel("X (m)")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index); ax.set_ylabel("Y (m)")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center", color="w", fontsize=8)
        ax.set_title("unmatched vanished per hour, non-rotation transitions")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "sensitivity.png", dpi=130); plt.close(fig)
        L += ["", "Plots: vanished_new_over_time.png, pair_distances.png, sensitivity.png"]

    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
