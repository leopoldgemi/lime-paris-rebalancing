#!/usr/bin/env python3
"""
Report figures for the Data section, built on build_demand.py output.

    rentals_by_hour.png / .csv    rentals per hour of day, all quartiers, weekday vs weekend
                                  (from a quartier run of build_demand.py)
    grid_rentals_per_cell_hour.csv / .md
                                  rentals per 500 m cell and hour (median, mean, p90), active
                                  cells only, weekday vs weekend (from a --zones grid run)
    stockout_map_500m.png / .csv  per 500 m cell: share of hours 07-23 with at least one
                                  snapshot without an available e-bike; quartier borders drawn

Active cell = centre inside Paris intra-muros and on average >= --active-min available
e-bikes. Available = e-bike, not reserved, range >= --min-range-m (as in build_demand.py).

Usage:
    python build_demand.py --data-dir D --out model_input
    python build_demand.py --data-dir D --out model_input_g500 --zones grid --grid-m 500
    python demand_report.py --data-dir D --quartier-dir model_input --grid-dir model_input_g500
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_snapshots import BASE_DIR, M_PER_DEG_LAT, M_PER_DEG_LON, PARIS_TZ
from build_demand import EBIKE_TYPE, QUARTIERS_PATH, Zoning, grid_zone_of

BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(ax, plt):
    ax.figure.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors=INK2)


def rentals_by_hour(qdir: Path, out: Path, plt) -> None:
    d = pd.read_csv(qdir / "zone_period_demand.csv")
    pm = int(d["period_minutes"].iloc[0])
    d["hour"] = d["period"] * pm // 60
    h = (d.groupby(["weekend", "hour", "period"])[["departures_per_h", "arrivals_per_h"]].sum()
         .groupby(["weekend", "hour"]).mean())
    h.rename(columns={"departures_per_h": "rentals_per_h", "arrivals_per_h": "returns_per_h"}).round(1) \
        .to_csv(out / "rentals_by_hour.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    style(ax, plt)
    for we, col, lab in ((False, BLUE, "Weekday"), (True, ORANGE, "Weekend")):
        x = h.loc[we]["departures_per_h"]
        ax.plot(x.index, x.values, color=col, lw=2, marker="o", ms=4, label=lab)
        ax.annotate(lab, (x.index[-1], x.values[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 25.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Hour of day (Paris local time)", color=INK2)
    ax.set_ylabel("Rentals per hour (all 80 quartiers)", color=INK2)
    ax.set_title("Lime Paris e-bike rentals per hour, weekday vs weekend", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "rentals_by_hour.png")
    plt.close(fig)


def paris_cells(size_m: float):
    import shapely                                             # noqa: PLC0415
    from shapely.geometry import shape                         # noqa: PLC0415
    from shapely.ops import transform                          # noqa: PLC0415
    gj = json.loads(QUARTIERS_PATH.read_text())
    to_m = lambda x, y, z=None: (np.asarray(x) * M_PER_DEG_LON, np.asarray(y) * M_PER_DEG_LAT)  # noqa: E731
    polys = [transform(to_m, shape(f["geometry"])) for f in gj["features"]]
    paris = shapely.union_all(polys)
    minx, miny, maxx, maxy = paris.bounds
    cells = [f"{x}_{y}" for x in range(int(minx // size_m), int(maxx // size_m) + 1)
             for y in range(int(miny // size_m), int(maxy // size_m) + 1)
             if paris.contains(shapely.Point((x + .5) * size_m, (y + .5) * size_m))]
    return cells, polys


def grid_inventory(data_dir: Path, cells: list[str], size_m: float, min_range_m: float) -> pd.DataFrame:
    """Available e-bikes per cell and snapshot (zeros included), with the Paris hour."""
    files = sorted(glob.glob(str(data_dir / "lime_paris_*.csv.gz"))) or sorted(glob.glob(str(data_dir / "lime_paris_*.csv")))
    rows = []
    for f in files:
        df = pd.read_csv(f, usecols=["snapshot_utc", "bike_id", "lat", "lon", "vehicle_type_id", "is_reserved",
                                     "current_range_meters"], dtype={"bike_id": "string", "vehicle_type_id": "string"})
        df = df[(df["vehicle_type_id"] == EBIKE_TYPE)].drop_duplicates(["snapshot_utc", "bike_id"])
        df = df[(df["is_reserved"] == 0) & (df["current_range_meters"] >= min_range_m)]
        df["cell"] = grid_zone_of(df["lat"].to_numpy(), df["lon"].to_numpy(), size_m)
        c = df[df["cell"].isin(cells)].groupby(["snapshot_utc", "cell"]).size()
        snaps = df["snapshot_utc"].unique()
        full = c.reindex(pd.MultiIndex.from_product([snaps, cells], names=["snapshot_utc", "cell"]), fill_value=0)
        rows.append(full.rename("inv").reset_index())
        print(f"  {Path(f).name}: {len(snaps)} snapshots", flush=True)
    inv = pd.concat(rows, ignore_index=True)
    inv["hour"] = pd.to_datetime(inv["snapshot_utc"], utc=True).dt.tz_convert(PARIS_TZ).dt.floor("h")
    return inv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with daily lime_paris_YYYY-MM-DD.csv.gz files")
    p.add_argument("--quartier-dir", default="model_input")
    p.add_argument("--grid-dir", default="model_input_g500", help="build_demand.py output with --zones grid")
    p.add_argument("--out", default="reports")
    p.add_argument("--grid-m", type=float, default=500)
    p.add_argument("--min-range-m", type=float, default=5000)
    p.add_argument("--active-min", type=float, default=2.0, help="mean available bikes for a cell to count as active")
    p.add_argument("--hours", default="7-23", help="hours of day for the stockout map, end exclusive (default 7-23)")
    p.add_argument("--min-snapshots", type=int, default=15, help="ignore cell-hours with fewer snapshots")
    a = p.parse_args(argv)
    import matplotlib                                          # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                            # noqa: PLC0415
    from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: PLC0415

    res = lambda x: Path(x) if Path(x).is_absolute() else BASE_DIR / x  # noqa: E731
    out = res(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Zoning("quartier")                                         # downloads the polygons if missing

    rentals_by_hour(res(a.quartier_dir), out, plt)

    cells, polys = paris_cells(a.grid_m)
    inv = grid_inventory(res(a.data_dir), cells, a.grid_m, a.min_range_m)
    mean_inv = inv.groupby("cell")["inv"].mean()
    active = set(mean_inv[mean_inv >= a.active_min].index)

    # rentals per active cell and hour
    g = pd.read_csv(res(a.grid_dir) / "zone_period_demand.csv")
    g = g[g["zone"].isin(active)].copy()
    g["hour"] = g["period"] * int(g["period_minutes"].iloc[0]) // 60
    ch = g.groupby(["weekend", "zone", "hour"])["departures_per_h"].mean().unstack("hour").fillna(0).stack()
    ch = ch.rename("rentals_per_h").reset_index()
    ch.to_csv(out / "grid_rentals_per_cell_hour.csv", index=False)
    lines = [f"# Rentals per {a.grid_m:.0f} m cell and hour ({len(active)} active cells of {len(cells)})", "",
             "| | median | mean | p90 | max |", "|---|---|---|---|---|"]
    for we, lab in ((False, "weekday"), (True, "weekend")):
        x = ch[ch["weekend"] == we]["rentals_per_h"]
        lines.append(f"| {lab} | {x.median():.1f} | {x.mean():.1f} | {x.quantile(.9):.1f} | {x.max():.1f} |")

    # stockout map
    h0, h1 = (int(v) for v in a.hours.split("-"))
    day = inv[inv["hour"].dt.hour.between(h0, h1 - 1)]
    hh = day.groupby(["cell", "hour"])["inv"].agg(["min", "size"])
    hh = hh[hh["size"] >= a.min_snapshots]
    share = (hh["min"] == 0).groupby("cell").mean().reindex(cells)
    pd.DataFrame({"empty_hour_share": share.round(4), "mean_available": mean_inv.reindex(cells).round(2),
                  "active": [c in active for c in cells]}).rename_axis("cell").to_csv(out / "stockout_map_500m.csv")
    act = share[share.index.isin(active)]
    lines += ["", f"Hours {h0:02d}-{h1:02d} with >= 1 empty snapshot, active cells: "
              f"median {act.median():.1%}, mean {act.mean():.1%}, p90 {act.quantile(.9):.1%}; "
              f"cells with > 25 %: {(act > .25).sum()}"]
    (out / "grid_rentals_per_cell_hour.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    bounds = [0, .02, .05, .10, .20, .30, .50, 1.0001]
    cmap, norm = ListedColormap(SEQ), BoundaryNorm(bounds, len(SEQ))
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
    style(ax, plt)
    for c in cells:
        x, y = (int(v) for v in c.split("_"))
        if c in active:
            fc, hatch, ec = cmap(norm(share[c])) if pd.notna(share[c]) else "#f0efec", None, SURFACE
        else:
            fc, hatch, ec = "#f0efec", "////", "#d6d5d0"
        ax.add_patch(plt.Rectangle((x * a.grid_m, y * a.grid_m), a.grid_m, a.grid_m, facecolor=fc,
                                   edgecolor=ec, lw=0.6, hatch=hatch))
    for poly in polys:
        for geom in getattr(poly, "geoms", [poly]):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=INK2, lw=0.4)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, ticks=bounds[:-1] + [1.0], format=lambda v, _: f"{v:.0%}")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.set_label(f"share of hours {h0:02d}-{h1:02d} with >= 1 empty snapshot", color=INK2)
    ax.set_title(f"Stockouts per {a.grid_m:.0f} m cell (hatched: inactive, < {a.active_min:.0f} bikes on average)",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "stockout_map_500m.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
