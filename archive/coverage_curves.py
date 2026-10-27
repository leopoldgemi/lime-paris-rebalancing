#!/usr/bin/env python3
"""
How much of a quartier's demand is within walking distance of a bike?

The optimisation decides how many bikes stand in each quartier, but whether a
rider is served depends on distance, not on the quartier count. This script
measures the link between the two: for every quartier and every snapshot it
computes

    n          available e-bikes standing in the quartier
    coverage   share of that quartier's demand that has an available bike
               within --radius metres

Demand is located on a 200 m lattice, weighted by where rentals actually start.
Only bikes inside the quartier count, so a rider served by a bike just across
the boundary is treated as unserved: the curves are mildly pessimistic, and the
model never has to reason about where inside a quartier a bike stands.

The measured points are averaged per bike count and turned into a concave,
increasing, piecewise-linear curve with --breakpoints segments, which is what
makes the optimisation a linear program: served demand in quartier i is then

    r[i,t] <= d[i,t] * (alpha[i,k] + beta[i,k] * s[i,t])    for every segment k

Output: model_input/coverage_curves.csv (one row per quartier and segment) and
coverage_curves.png.

Usage: python coverage_curves.py --data-dir ../lime-data/data
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
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from analyze_snapshots import BASE_DIR, HAVE_MPL, M_PER_DEG_LAT, M_PER_DEG_LON

if HAVE_MPL:
    import matplotlib.pyplot as plt

QUARTIERS = BASE_DIR / "data" / "reference" / "quartiers_paris.geojson"
COLS = ["snapshot_utc", "bike_id", "lat", "lon", "vehicle_type_id", "is_reserved", "current_range_meters"]
CELL = 200.0


def concave_envelope(n: np.ndarray, cov: np.ndarray, k: int) -> list[tuple[float, float]]:
    """Breakpoints of a concave, increasing, piecewise-linear upper bound on cov(n)."""
    cov = np.maximum.accumulate(cov)                      # enforce monotonicity
    xs = np.linspace(n.min(), n.max(), k + 1)
    ys = np.interp(xs, n, cov)
    # make the sequence of slopes non-increasing (concavity) by a simple pass
    for _ in range(50):
        sl = np.diff(ys) / np.diff(xs)
        bad = np.where(np.diff(sl) > 1e-12)[0]
        if not len(bad):
            break
        for b in bad:                                     # lift the middle point
            ys[b + 1] = min(1.0, (ys[b] + ys[b + 2]) / 2)
    return list(zip(xs, np.clip(ys, 0, 1)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="model_input")
    p.add_argument("--radius", type=float, default=300.0, help="walking radius in metres (default 300)")
    p.add_argument("--breakpoints", type=int, default=5)
    p.add_argument("--min-range-m", type=float, default=5000)
    p.add_argument("--every", type=int, default=5, help="use every Nth snapshot")
    a = p.parse_args(argv)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else BASE_DIR / s  # noqa: E731
    data_dir, out = resolve(a.data_dir), resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)

    gj = json.loads(QUARTIERS.read_text())
    polys = [shape(f["geometry"]) for f in gj["features"]]
    names = [f"{f['properties']['c_qu']}_{f['properties']['l_qu']}".replace(" ", "-") for f in gj["features"]]
    tree_q = STRtree(polys)

    def quartier_of(lat, lon):
        hit = tree_q.query([Point(lo, la) for la, lo in zip(lat, lon)], predicate="within")
        q = np.full(len(lat), -1)
        q[hit[0]] = hit[1]
        return q

    files = sorted(glob.glob(str(data_dir / "lime_paris_*.csv.gz"))) or \
        sorted(glob.glob(str(data_dir / "*" / "*.csv.gz")))
    if not files:
        sys.exit(f"no snapshot files under {data_dir}")

    # ---- 1. demand lattice from observed rental origins -----------------------
    dem: dict[tuple, float] = {}
    frames = []
    for path in files:
        df = pd.read_csv(path, usecols=COLS, dtype={"bike_id": "string", "vehicle_type_id": "string"})
        df = df[df.vehicle_type_id == "3"]
        df["snapshot_utc"] = pd.to_datetime(df.snapshot_utc, utc=True)
        snaps = sorted(df.snapshot_utc.unique())
        by = {t: g.set_index("bike_id") for t, g in df.groupby("snapshot_utc")}
        by = {t: g[~g.index.duplicated()] for t, g in by.items()}
        prev = prev_t = None
        for t in snaps:
            cur = by[t]
            if prev is not None:
                gap = (t - prev_t).total_seconds() / 60
                if gap <= 6 and len(prev.index.intersection(cur.index)) / max(1, len(cur)) >= 0.5:
                    common = prev.index.intersection(cur.index)
                    dist = np.hypot((cur.loc[common, "lon"].to_numpy() - prev.loc[common, "lon"].to_numpy()) * M_PER_DEG_LON,
                                    (cur.loc[common, "lat"].to_numpy() - prev.loc[common, "lat"].to_numpy()) * M_PER_DEG_LAT)
                    origins = prev.loc[common[dist >= 200].union(prev.index.difference(cur.index))]
                    gx = np.floor(origins.lon.to_numpy() * M_PER_DEG_LON / CELL).astype(int)
                    gy = np.floor(origins.lat.to_numpy() * M_PER_DEG_LAT / CELL).astype(int)
                    for key in zip(gx, gy):
                        dem[key] = dem.get(key, 0.0) + 1.0
            prev, prev_t = cur, t
        frames.append((path, snaps, by))

    keys = np.array(list(dem.keys()), float)
    wts = np.array(list(dem.values()), float)
    cx = (keys[:, 0] + .5) * CELL
    cy = (keys[:, 1] + .5) * CELL
    qi = quartier_of(cy / M_PER_DEG_LAT, cx / M_PER_DEG_LON)
    ok = qi >= 0
    cx, cy, wts, qi = cx[ok], cy[ok], wts[ok], qi[ok]
    print(f"demand lattice: {len(cx):,} cells of {CELL:.0f} m in Paris, {wts.sum():,.0f} rental origins")

    cells_by_q = {q: (np.column_stack([cx[qi == q], cy[qi == q]]), wts[qi == q])
                  for q in range(len(names)) if (qi == q).any()}

    # ---- 2. coverage measurements --------------------------------------------
    rows = []
    for path, snaps, by in frames:
        for k, t in enumerate(snaps):
            if k % a.every:
                continue
            g = by[t]
            g = g[(g.is_reserved == 0) & (g.current_range_meters >= a.min_range_m)]
            if g.empty:
                continue
            bq = quartier_of(g.lat.to_numpy(), g.lon.to_numpy())
            bx = g.lon.to_numpy() * M_PER_DEG_LON
            byy = g.lat.to_numpy() * M_PER_DEG_LAT
            for q, (pts, w) in cells_by_q.items():
                sel = bq == q
                n = int(sel.sum())
                if n == 0:
                    rows.append({"quartier": names[q], "n": 0, "coverage": 0.0, "demand": w.sum()})
                    continue
                tr = cKDTree(np.column_stack([bx[sel], byy[sel]]))
                near = np.array(tr.query_ball_point(pts, a.radius, return_length=True))
                rows.append({"quartier": names[q], "n": n,
                             "coverage": float(w[near > 0].sum() / w.sum()), "demand": w.sum()})
    obs = pd.DataFrame(rows)
    obs.to_csv(out / "coverage_observations.csv", index=False)
    print(f"{len(obs):,} quartier-snapshot observations")

    # ---- 3. concave piecewise-linear curves ----------------------------------
    recs = []
    for q, g in obs.groupby("quartier"):
        agg = g.groupby("n")["coverage"].mean().sort_index()
        if len(agg) < 3:
            continue
        bp = concave_envelope(agg.index.to_numpy(float), agg.to_numpy(), a.breakpoints)
        for k in range(len(bp) - 1):
            (x0, y0), (x1, y1) = bp[k], bp[k + 1]
            beta = (y1 - y0) / max(x1 - x0, 1e-9)
            recs.append({"quartier": q, "segment": k, "n_from": x0, "n_to": x1,
                         "alpha": y0 - beta * x0, "beta": beta})
    cur = pd.DataFrame(recs)
    cur.to_csv(out / "coverage_curves.csv", index=False)

    med = obs.groupby("quartier")["coverage"].median()
    print(f"\nradius {a.radius:.0f} m, {a.breakpoints} segments per quartier")
    print(f"median coverage across quartier-snapshots: {obs['coverage'].median():.1%}")
    print(f"quartiers with median coverage below 80%: {(med < 0.8).sum()} of {len(med)}")
    print("worst quartiers (median coverage):")
    print(med.nsmallest(8).to_string())

    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        big = obs.groupby("quartier")["demand"].first().nlargest(6).index
        for q in big:
            g = obs[obs.quartier == q].groupby("n")["coverage"].mean().sort_index()
            ax.plot(g.index, g.to_numpy(), lw=1, alpha=.5)
            c = cur[cur.quartier == q]
            xs = np.r_[c.n_from.to_numpy(), c.n_to.to_numpy()[-1:]]
            ys = [float(c[(c.n_from <= x) & (c.n_to >= x)].iloc[0].alpha +
                        c[(c.n_from <= x) & (c.n_to >= x)].iloc[0].beta * x) for x in xs]
            ax.plot(xs, ys, lw=2, label=q.split("_", 1)[1][:18])
        ax.set_xlabel(f"available bikes in the quartier"); ax.set_ylabel(f"demand within {a.radius:.0f} m")
        ax.set_title("Coverage curves (thin = measured, thick = concave approximation)")
        ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_ylim(0, 1.02)
        fig.tight_layout(); fig.savefig(out / "coverage_curves.png", dpi=130); plt.close(fig)
        print(f"\nplot: {(out / 'coverage_curves.png').relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
