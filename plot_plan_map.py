#!/usr/bin/env python3
"""
Map of the night relocation plan of one lp_model.py run.

What it does
------------
Colours every 300 m cell by its net inflow = bikes received - bikes sent away
at night. Blue cells receive bikes, red cells give bikes away, grey cells are
not touched. Arrondissement borders are drawn as thin lines.

Reads
-----
    reports/lp/<run>/cells.csv              per cell: sent, received, net_inflow (from lp_model.py)
    model_input_lp/base/cells.csv           cell centres in km (from build_model_inputs.py)
    data/reference/quartiers_paris.geojson  quartier polygons, merged into arrondissements

Writes
------
    reports/lp/<run>/net_flow_map.png

Usage
-----
    python plot_plan_map.py --run reports/lp/weekday_p3.5_R1.5_alpha0.5
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shapely
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import shape
from shapely.ops import transform

from analyze_snapshots import BASE_DIR, M_PER_DEG_LAT, M_PER_DEG_LON
from build_demand import QUARTIERS_PATH

GRID_KM = 0.3

# Colour classes: (lower bound, upper bound, colour, legend text).
# Blue = receives bikes, red = gives bikes away, grey = (almost) no change.
CLASSES = [
    (-1e9, -15, "#8a1c16", "gives 15 or more bikes"),
    (-15, -5, "#e34948", "gives 5 to 15 bikes"),
    (-5, -0.5, "#f2a8a3", "gives 0.5 to 5 bikes"),
    (-0.5, 0.5, "#f0efec", "no change"),
    (0.5, 5, "#a9c8ef", "receives 0.5 to 5 bikes"),
    (5, 15, "#2a78d6", "receives 5 to 15 bikes"),
    (15, 1e9, "#123f7a", "receives 15 or more bikes"),
]
BORDER = "#52514e"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"

# ---- Parameters -------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run", default="reports/lp/weekday_p3.5_R1.5_alpha0.5", help="folder of an lp_model.py run")
parser.add_argument("--input", default="model_input_lp/base", help="folder written by build_model_inputs.py")
args = parser.parse_args()
run_dir = BASE_DIR / args.run

# ---- Load data --------------------------------------------------------------

result = pd.read_csv(run_dir / "cells.csv")
centres = pd.read_csv(BASE_DIR / args.input / "cells.csv")[["cell", "x_km", "y_km"]]
result = result.merge(centres, on="cell")
summary = json.loads((run_dir / "summary.json").read_text())

# Arrondissement shapes: merge the 80 quartiers by arrondissement, in km.
features = json.loads(QUARTIERS_PATH.read_text())["features"]


def to_km(x, y, z=None):
    return x * M_PER_DEG_LON / 1000, y * M_PER_DEG_LAT / 1000


arrondissements = {}
for f in features:
    number = int(f["properties"]["c_ar"])
    arrondissements.setdefault(number, []).append(transform(to_km, shape(f["geometry"])))
arrondissements = {n: shapely.union_all(parts) for n, parts in arrondissements.items()}

# ---- Draw -------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

# One square per cell, coloured by its class of net inflow.
for row in result.itertuples():
    colour = next(c for lo, hi, c, _ in CLASSES if lo <= row.net_inflow < hi)
    ax.add_patch(Rectangle((row.x_km - GRID_KM / 2, row.y_km - GRID_KM / 2), GRID_KM, GRID_KM,
                           facecolor=colour, edgecolor=SURFACE, linewidth=0.3))

# Thin arrondissement borders and their numbers.
for number, shape_km in arrondissements.items():
    polygons = shape_km.geoms if shape_km.geom_type == "MultiPolygon" else [shape_km]
    for polygon in polygons:
        xs, ys = polygon.exterior.xy
        ax.plot(xs, ys, color=BORDER, linewidth=0.6)
    centre = shape_km.representative_point()
    ax.text(centre.x, centre.y, str(number), ha="center", va="center", fontsize=8, color=INK, alpha=0.7)

ax.set_aspect("equal")
ax.autoscale_view()
ax.axis("off")

# Legend with the number of cells per class.
handles = []
for lo, hi, colour, text in CLASSES:
    n = ((result["net_inflow"] >= lo) & (result["net_inflow"] < hi)).sum()
    handles.append(Patch(facecolor=colour, edgecolor=BORDER, linewidth=0.4, label=f"{text} ({n} cells)"))
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9, title="Net inflow per night",
          title_fontsize=9, labelcolor=INK)

ax.set_title(f"Night relocation plan, {summary['day_type']}s, p = {summary['p']:g}, R = {summary['R']:g} km: "
             f"{summary['bikes_moved']:,.0f} bikes moved, mean {summary['mean_distance_km']:.2f} km",
             fontsize=11, color=INK, loc="left")
fig.tight_layout()
fig.savefig(run_dir / "net_flow_map.png", dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
print(f"saved {run_dir / 'net_flow_map.png'}")
