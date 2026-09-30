#!/usr/bin/env python3
"""
First-look analysis of the collected Lime Paris snapshots.

Reads every snapshot written by collect_lime.py (both formats):
    data/lime_paris_YYYY-MM-DD.csv          (daily CSV, local collector)
    data/YYYY-MM-DD/HHMMSS.csv.gz           (one file per snapshot, GitHub Actions)

and writes tables + plots to reports/:

    summary.md                 human-readable overview (also printed to stdout)
    fleet_per_snapshot.csv     vehicles per snapshot: total / available / reserved / disabled,
                               split by form factor (e-bike vs scooter), low-battery share
    hourly_profile.csv         mean fleet metrics by hour of day (Europe/Paris)
    grid_density.csv           mean number of vehicles per ~500 m grid cell
    moves_per_hour.csv         position changes between consecutive snapshots (trip proxy)
    *.png                      plots (only if matplotlib is installed)

Dependencies: pandas (required), matplotlib (optional, for plots).

Usage:
    python analyze_snapshots.py                      # data/ -> reports/
    python analyze_snapshots.py --data-dir data --out reports
    python analyze_snapshots.py --grid-m 250 --move-threshold-m 100
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:  # plots are optional
    HAVE_MPL = False

BASE_DIR = Path(__file__).resolve().parent
PARIS_TZ = "Europe/Paris"
LOW_BATTERY_M = 5_000  # vehicles below this range are treated as "needs battery swap"

# Paris centre-ish; used for the metre <-> degree conversion of the grid
LAT0 = 48.8566
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_snapshots(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "lime_paris_*.csv")))
    files += sorted(glob.glob(str(data_dir / "*" / "*.csv.gz")))
    if not files:
        sys.exit(f"no snapshot files found under {data_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(
            f,
            dtype={"bike_id": "string", "vehicle_type_id": "string", "vehicle_type": "string"},
            compression="gzip" if f.endswith(".gz") else None,
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)
    df["snapshot_paris"] = df["snapshot_utc"].dt.tz_convert(PARIS_TZ)
    df["is_reserved"] = df["is_reserved"].fillna(0).astype(int)
    df["is_disabled"] = df["is_disabled"].fillna(0).astype(int)
    df["available"] = ((df["is_reserved"] == 0) & (df["is_disabled"] == 0)).astype(int)
    # drop exact duplicate rows (same snapshot + bike), e.g. after a re-run
    df = df.drop_duplicates(subset=["snapshot_utc", "bike_id"])
    print(f"loaded {len(files)} file(s), {df['snapshot_utc'].nunique()} snapshots, {len(df):,} rows")
    return df


def load_vehicle_types(data_dir: Path) -> pd.DataFrame:
    """vehicle_type_id -> form_factor / propulsion / max_range_meters (from reference JSON)."""
    p = data_dir / "reference" / "vehicle_types.json"
    if not p.exists():
        print("warning: reference/vehicle_types.json missing - form factor inferred from vehicle_type column")
        return pd.DataFrame(columns=["vehicle_type_id", "form_factor", "propulsion_type", "max_range_meters"])
    doc = json.loads(p.read_text())
    vt = pd.DataFrame(doc["data"]["vehicle_types"])
    vt["vehicle_type_id"] = vt["vehicle_type_id"].astype("string")
    return vt


def add_form_factor(df: pd.DataFrame, vt: pd.DataFrame) -> pd.DataFrame:
    if len(vt):
        df = df.merge(vt[["vehicle_type_id", "form_factor", "max_range_meters"]], on="vehicle_type_id", how="left")
    else:
        df["form_factor"] = None
        df["max_range_meters"] = None
    # fallback: Lime's non-standard "vehicle_type" column ("e-bike", "scooter", ...)
    if "vehicle_type" in df.columns:
        guess = df["vehicle_type"].str.lower().map(
            lambda s: "bicycle" if isinstance(s, str) and "bike" in s
            else ("scooter" if isinstance(s, str) and "scoot" in s else None)
        )
        df["form_factor"] = df["form_factor"].fillna(guess)
    df["form_factor"] = df["form_factor"].fillna("unknown")
    df["kind"] = df["form_factor"].map({"bicycle": "e-bike", "scooter": "scooter"}).fillna("other")
    df["soc"] = df["current_range_meters"] / df["max_range_meters"]  # state of charge proxy, may be NaN
    df["low_battery"] = (df["current_range_meters"] < LOW_BATTERY_M).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #


def fleet_per_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("snapshot_utc")
    out = pd.DataFrame({
        "total": g.size(),
        "available": g["available"].sum(),
        "reserved": g["is_reserved"].sum(),
        "disabled": g["is_disabled"].sum(),
        "low_battery": g["low_battery"].sum(),
        "median_range_m": g["current_range_meters"].median(),
    })
    kinds = df.pivot_table(index="snapshot_utc", columns="kind", values="bike_id", aggfunc="count", fill_value=0)
    for k in ("e-bike", "scooter", "other"):
        out[k] = kinds[k] if k in kinds.columns else 0
    out["low_battery_share"] = out["low_battery"] / out["total"]
    out.index.name = "snapshot_utc"
    return out.sort_index()


def hourly_profile(fleet: pd.DataFrame) -> pd.DataFrame:
    idx = fleet.index.tz_convert(PARIS_TZ)
    tmp = fleet.copy()
    tmp["hour"] = idx.hour
    tmp["weekday"] = idx.dayofweek < 5
    cols = ["total", "available", "reserved", "low_battery_share", "e-bike", "scooter"]
    return tmp.groupby("hour")[cols].mean().round(2)


def grid_density(df: pd.DataFrame, cell_m: float) -> pd.DataFrame:
    dlat = cell_m / M_PER_DEG_LAT
    dlon = cell_m / M_PER_DEG_LON
    tmp = df[["snapshot_utc", "lat", "lon", "available", "low_battery"]].copy()
    tmp["cell_lat"] = (tmp["lat"] // dlat) * dlat + dlat / 2
    tmp["cell_lon"] = (tmp["lon"] // dlon) * dlon + dlon / 2
    n_snap = df["snapshot_utc"].nunique()
    g = tmp.groupby(["cell_lat", "cell_lon"])
    out = pd.DataFrame({
        "mean_vehicles": g.size() / n_snap,
        "mean_available": g["available"].sum() / n_snap,
        "mean_low_battery": g["low_battery"].sum() / n_snap,
    }).reset_index()
    out["cell_lat"] = out["cell_lat"].round(6)
    out["cell_lon"] = out["cell_lon"].round(6)
    return out.sort_values("mean_vehicles", ascending=False)


def movements(df: pd.DataFrame, threshold_m: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect position changes of the same bike between consecutive snapshots.

    Returns (moves, per_hour). A "move" = same bike_id, consecutive snapshots in
    which it appears, displacement > threshold_m. Bikes that vanish between
    snapshots (in a ride, or picked up by Lime) and reappear elsewhere also count
    as one move; the gap length is recorded in gap_min.
    """
    d = df.sort_values(["bike_id", "snapshot_utc"])[["bike_id", "snapshot_utc", "lat", "lon", "current_range_meters"]]
    prev = d.groupby("bike_id").shift(1)
    d = d.assign(
        prev_lat=prev["lat"], prev_lon=prev["lon"], prev_t=prev["snapshot_utc"], prev_range=prev["current_range_meters"]
    ).dropna(subset=["prev_lat"])
    dy = (d["lat"] - d["prev_lat"]) * M_PER_DEG_LAT
    dx = (d["lon"] - d["prev_lon"]) * M_PER_DEG_LON
    d["dist_m"] = (dx**2 + dy**2) ** 0.5
    d["gap_min"] = (d["snapshot_utc"] - d["prev_t"]).dt.total_seconds() / 60
    d["range_delta_m"] = d["current_range_meters"] - d["prev_range"]
    moves = d[d["dist_m"] > threshold_m].copy()
    # range went UP by a lot while moving -> most likely a battery swap / Lime relocation, not a ride
    moves["likely_operator_move"] = moves["range_delta_m"] > 10_000
    moves["hour_paris"] = moves["snapshot_utc"].dt.tz_convert(PARIS_TZ).dt.hour
    per_hour = moves.groupby("hour_paris").agg(
        moves=("bike_id", "size"),
        median_dist_m=("dist_m", "median"),
        likely_operator_moves=("likely_operator_move", "sum"),
    )
    return moves, per_hour


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_all(fleet: pd.DataFrame, hourly: pd.DataFrame, grid: pd.DataFrame, per_hour: pd.DataFrame,
             df: pd.DataFrame, out: Path) -> list[str]:
    if not HAVE_MPL:
        print("matplotlib not installed - skipping plots (pip install matplotlib)")
        return []
    made = []
    x = fleet.index.tz_convert(PARIS_TZ)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, fleet["total"], label="total in feed")
    ax.plot(x, fleet["available"], label="available (not reserved / disabled)")
    ax.plot(x, fleet["reserved"], label="reserved")
    ax.plot(x, fleet["low_battery"], label=f"range < {LOW_BATTERY_M/1000:.0f} km")
    ax.set_title("Lime Paris - vehicles per snapshot (Europe/Paris time)")
    ax.set_ylabel("vehicles")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fleet_over_time.png", dpi=130)
    plt.close(fig)
    made.append("fleet_over_time.png")

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.stackplot(x, fleet["e-bike"], fleet["scooter"], fleet["other"], labels=["e-bike", "scooter", "other"])
    ax.set_title("Fleet split: e-bike vs scooter")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fleet_split.png", dpi=130)
    plt.close(fig)
    made.append("fleet_split.png")

    if len(hourly) > 1:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(hourly.index, hourly["available"], marker="o", label="available")
        ax.plot(hourly.index, hourly["reserved"], marker="o", label="reserved")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("hour of day (Paris)")
        ax.set_title("Mean fleet by hour of day")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "hourly_profile.png", dpi=130)
        plt.close(fig)
        made.append("hourly_profile.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["current_range_meters"].dropna() / 1000, bins=40)
    ax.set_xlabel("current_range_meters (km)")
    ax.set_ylabel("vehicle-snapshots")
    ax.set_title("Battery range distribution (all snapshots)")
    fig.tight_layout()
    fig.savefig(out / "battery_hist.png", dpi=130)
    plt.close(fig)
    made.append("battery_hist.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(grid["cell_lon"], grid["cell_lat"], c=grid["mean_vehicles"], s=12, cmap="viridis", marker="s")
    ax.set_aspect(M_PER_DEG_LAT / M_PER_DEG_LON)  # metres per degree differ for lat/lon
    # ignore far-out stragglers so the map shows Paris, not the whole Ile-de-France
    ax.set_xlim(df["lon"].quantile(0.005) - 0.01, df["lon"].quantile(0.995) + 0.01)
    ax.set_ylim(df["lat"].quantile(0.005) - 0.005, df["lat"].quantile(0.995) + 0.005)
    ax.set_title("Mean vehicles per grid cell")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.colorbar(sc, ax=ax, label="mean vehicles / snapshot")
    fig.tight_layout()
    fig.savefig(out / "grid_density.png", dpi=130)
    plt.close(fig)
    made.append("grid_density.png")

    if len(per_hour):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(per_hour.index, per_hour["moves"])
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("hour of day (Paris)")
        ax.set_ylabel("position changes")
        ax.set_title("Detected position changes between consecutive snapshots")
        fig.tight_layout()
        fig.savefig(out / "moves_per_hour.png", dpi=130)
        plt.close(fig)
        made.append("moves_per_hour.png")
    return made


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def write_summary(df, fleet, hourly, grid, moves, per_hour, plots, out: Path, grid_m: float, thr: float) -> str:
    t0, t1 = df["snapshot_utc"].min(), df["snapshot_utc"].max()
    span_h = (t1 - t0).total_seconds() / 3600
    n_snap = len(fleet)
    gaps = fleet.index.to_series().diff().dt.total_seconds().div(60).dropna()
    kinds = df.groupby("kind")["bike_id"].nunique()
    lines = [
        "# Lime Paris snapshots - first overview",
        "",
        f"- Period (UTC): {t0:%Y-%m-%d %H:%M} to {t1:%Y-%m-%d %H:%M}  ({span_h:.1f} h)",
        f"- Snapshots: {n_snap}  |  rows: {len(df):,}  |  distinct bike_ids: {df['bike_id'].nunique():,}",
    ]
    if len(gaps):
        lines.append(f"- Interval between snapshots: median {gaps.median():.1f} min, max {gaps.max():.1f} min")
    lines += [
        "",
        "## Fleet size per snapshot",
        f"- total: mean {fleet['total'].mean():.0f}, min {fleet['total'].min()}, max {fleet['total'].max()}",
        f"- available: mean {fleet['available'].mean():.0f}  |  reserved: mean {fleet['reserved'].mean():.0f}"
        f"  |  disabled: mean {fleet['disabled'].mean():.0f}",
        f"- range < {LOW_BATTERY_M/1000:.0f} km: mean {fleet['low_battery'].mean():.0f} vehicles"
        f" ({fleet['low_battery_share'].mean():.1%})  |  median range {fleet['median_range_m'].median()/1000:.1f} km",
        "",
        "## E-bike vs scooter (distinct vehicles seen)",
    ]
    for k, v in kinds.items():
        per_snap = f"  (mean per snapshot: {fleet[k].mean():.0f})" if k in fleet.columns else ""
        lines.append(f"- {k}: {v:,}{per_snap}")
    lines += ["", f"## Spatial density (grid {grid_m:.0f} m, {len(grid)} occupied cells)", "Top 10 cells by mean vehicles:", ""]
    lines.append(grid.head(10).to_string(index=False))
    lines += ["", f"## Position changes between consecutive snapshots (> {thr:.0f} m)",
              f"- moves detected: {len(moves):,}  ({len(moves)/max(span_h,1e-9):.0f} per hour of observation)"]
    if len(moves):
        lines.append(f"- median displacement: {moves['dist_m'].median():.0f} m  |  "
                     f"likely operator moves (range jumped > 10 km): {int(moves['likely_operator_move'].sum()):,}")
        lines.append(f"- distinct bikes that moved: {moves['bike_id'].nunique():,}")
    if len(hourly) > 1:
        lines += ["", "## Hourly profile (Paris time, mean per hour)", "", hourly.to_string()]
    lines += ["", "## Files", ""]
    for f in ["fleet_per_snapshot.csv", "hourly_profile.csv", "grid_density.csv", "moves_per_hour.csv", "moves.csv"] + plots:
        lines.append(f"- reports/{f}")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="snapshot directory, relative to this script")
    p.add_argument("--out", default="reports", help="output directory, relative to this script")
    p.add_argument("--grid-m", type=float, default=500, help="grid cell size in metres (default 500)")
    p.add_argument("--move-threshold-m", type=float, default=50, help="min displacement counted as a move (default 50)")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else BASE_DIR / data_dir
    out = Path(args.out)
    out = out if out.is_absolute() else BASE_DIR / out
    out.mkdir(parents=True, exist_ok=True)

    df = load_snapshots(data_dir)
    df = add_form_factor(df, load_vehicle_types(data_dir))

    fleet = fleet_per_snapshot(df)
    hourly = hourly_profile(fleet)
    grid = grid_density(df, args.grid_m)
    moves, per_hour = movements(df, args.move_threshold_m)

    fleet.to_csv(out / "fleet_per_snapshot.csv")
    hourly.to_csv(out / "hourly_profile.csv")
    grid.to_csv(out / "grid_density.csv", index=False)
    per_hour.to_csv(out / "moves_per_hour.csv")
    moves[["bike_id", "prev_t", "snapshot_utc", "prev_lat", "prev_lon", "lat", "lon", "dist_m", "gap_min",
           "range_delta_m", "likely_operator_move"]].to_csv(out / "moves.csv", index=False)

    plots = plot_all(fleet, hourly, grid, per_hour, df, out)
    print(write_summary(df, fleet, hourly, grid, moves, per_hour, plots, out, args.grid_m, args.move_threshold_m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
