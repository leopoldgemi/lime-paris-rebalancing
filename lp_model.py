#!/usr/bin/env python3
"""
LP rebalancing model for Lime e-bikes in Paris, solved with Gurobi (gurobipy).

What it does
------------
At night, a van moves x[i,j] bikes from cell i to a nearby cell j. The same
night plan is used for every day k of the chosen day type. During the day,
riders take bikes where they are (served trips y) or, if the cell is empty,
the trip is lost (l). The model chooses the night plan that minimises

    relocation cost  +  average cost of lost trips per day

    min  sum_{(i,j) in A} c[i,j] x[i,j]  +  (1/|K|) sum_{i,t,k} p l[i,t,k]
    c[i,j] = h + v * dist[i,j] / Q

    h  handling cost per bike (pick up + drop off)
    v  van cost per km, shared by the Q bikes it carries
    p  lost profit of one lost trip

Reads (written by build_model_inputs.py, default folder model_input_lp/base)
-----
    cells.csv       cells with arrondissement
    s.csv           s[i]: bikes in the cell at night, per day type
    hourly.csv.gz   r[i,t,k] returns and d[i,t,k] demand, per day type
    arcs.csv.gz     dist[i,j] between cell centres

Writes (into --out, default reports/lp/<run name>/, ignored by git)
------
    summary.json    the key results of this run
    x.csv           the night plan: all pairs (i, j) with x[i,j] > 0
    cells.csv       per cell: s, bikes given away, bikes received, net inflow, lost trips per day

Usage
-----
    python lp_model.py --area 18                  # small test, 18th arrondissement
    python lp_model.py                            # all Paris, weekdays
    python lp_model.py --baseline                 # no relocation (x = 0)
    python lp_model.py --day-type weekend --p 20 --R 1.0 --M 2000
"""

import argparse
import json
import time
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

BASE_DIR = Path(__file__).resolve().parent

# ---- Parameters -------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", default="model_input_lp/base", help="folder written by build_model_inputs.py")
parser.add_argument("--day-type", default="weekday", choices=["weekday", "weekend"])
parser.add_argument("--p", type=float, default=3.50, help="lost profit per lost trip (EUR)")
parser.add_argument("--h", type=float, default=0.67, help="handling cost per relocated bike (EUR)")
parser.add_argument("--v", type=float, default=1.30, help="van cost per km (EUR)")
parser.add_argument("--Q", type=float, default=15, help="bikes per van load")
parser.add_argument("--R", type=float, default=1.5, help="max relocation distance (km)")
parser.add_argument("--M", type=float, default=None, help="max bikes relocated per night (default: no limit)")
parser.add_argument("--area", type=int, default=None, help="only this arrondissement (1-20)")
parser.add_argument("--baseline", action="store_true", help="no relocation: fix all x = 0")
parser.add_argument("--out", default=None, help="output folder (default reports/lp/<run name>)")
args = parser.parse_args()

p, h, v, Q, R, M = args.p, args.h, args.v, args.Q, args.R, args.M

run_name = f"{args.day_type}_p{p:g}_R{R:g}" + (f"_M{M:g}" if M else "") \
    + (f"_area{args.area}" if args.area else "") + ("_baseline" if args.baseline else "")
out = Path(args.out) if args.out else BASE_DIR / "reports" / "lp" / run_name
out.mkdir(parents=True, exist_ok=True)
input_dir = BASE_DIR / args.input

# ---- Load data --------------------------------------------------------------

t_start = time.perf_counter()

# Cells I, optionally only one arrondissement.
cells = pd.read_csv(input_dir / "cells.csv")
if args.area:
    cells = cells[cells["arrondissement"] == args.area]
I = list(cells["cell"])
arrondissement = dict(zip(cells["cell"], cells["arrondissement"]))

# s[i]: bikes in cell i at night (mean over the days of this type).
s_table = pd.read_csv(input_dir / "s.csv")
s_table = s_table[(s_table["day_type"] == args.day_type) & s_table["cell"].isin(I)]
s = dict(zip(s_table["cell"], s_table["s"]))

# d[i,t,k] demand and r[i,t,k] returns, for every cell, hour t = 1..18 and day k.
hourly = pd.read_csv(input_dir / "hourly.csv.gz", usecols=["day_type", "day", "cell", "t", "r", "d"])
hourly = hourly[(hourly["day_type"] == args.day_type) & hourly["cell"].isin(I)]
K = sorted(hourly["day"].unique())
T = sorted(hourly["t"].unique())
keys = list(zip(hourly["cell"], hourly["t"], hourly["day"]))
d = dict(zip(keys, hourly["d"]))
r = dict(zip(keys, hourly["r"]))

# A: pairs (i, j), i != j, at most R km apart, with distance dist[i,j] and cost c[i,j].
arcs = pd.read_csv(input_dir / "arcs.csv.gz")
arcs = arcs[(arcs["dist"] <= R) & arcs["i"].isin(I) & arcs["j"].isin(I)]
A = list(zip(arcs["i"], arcs["j"]))
dist = dict(zip(A, arcs["dist"]))
c = {(i, j): h + v * dist[i, j] / Q for (i, j) in A}

print(f"{len(I)} cells, {len(T)} hours, {len(K)} days ({args.day_type}: {', '.join(K)}), |A| = {len(A):,}")
print(f"sum(s) = {sum(s.values()):,.0f}, demand per day = {sum(d.values()) / len(K):,.0f}")

# ---- Build the model --------------------------------------------------------

model = gp.Model("lime_rebalancing")

# Variable x[i,j]: bikes moved at night from cell i to cell j (same plan every day).
x = model.addVars(A, lb=0, name="x")

# Variable b[i,t,k]: bikes in cell i at the start of hour t on day k.
b = model.addVars(keys, lb=0, name="b")

# Variable y[i,t,k]: trips served in cell i in hour t on day k.
y = model.addVars(keys, lb=0, name="y")

# Variable l[i,t,k]: trips lost in cell i in hour t on day k (rider found no bike).
l = model.addVars(keys, lb=0, name="l")

# Baseline: no relocation at all.
if args.baseline:
    for arc in A:
        x[arc].UB = 0

# Outgoing and incoming relocations of every cell, used in (1) and (2).
out_arcs = {i: [] for i in I}
in_arcs = {i: [] for i in I}
for (i, j) in A:
    out_arcs[i].append((i, j))
    in_arcs[j].append((i, j))
sent = {i: gp.quicksum(x[arc] for arc in out_arcs[i]) for i in I}
received = {i: gp.quicksum(x[arc] for arc in in_arcs[i]) for i in I}

# Objective: relocation cost + average cost of lost trips per day.
relocation_cost = gp.quicksum(c[arc] * x[arc] for arc in A)
lost_cost = (1 / len(K)) * gp.quicksum(p * l[key] for key in keys)
model.setObjective(relocation_cost + lost_cost, GRB.MINIMIZE)

# Constraint (1): supply.
# A cell cannot give away more bikes than it has at night.
model.addConstrs((sent[i] <= s[i] for i in I), name="supply")

# Constraint (2): morning inventory.
# Bikes at the start of hour 1 = night bikes - bikes sent away + bikes received.
model.addConstrs((b[i, 1, k] == s[i] - sent[i] + received[i] for i in I for k in K), name="start")

# Constraint (3): hourly inventory balance.
# Bikes next hour = bikes now - trips served + bikes returned.
model.addConstrs((b[i, t + 1, k] == b[i, t, k] - y[i, t, k] + r[i, t, k]
                  for (i, t, k) in keys if t < len(T)), name="balance")

# Constraint (4): a cell can only serve trips with the bikes it has.
model.addConstrs((y[key] <= b[key] for key in keys), name="serve")

# Constraint (5): every trip asked for is either served or lost.
model.addConstrs((y[key] + l[key] == d[key] for key in keys), name="demand")

# Constraint (6), optional: at most M bikes relocated per night.
if M is not None:
    model.addConstr(x.sum() <= M, name="max_relocation")

model.update()
t_built = time.perf_counter()
print(f"model: {model.NumVars:,} variables, {model.NumConstrs:,} constraints, "
      f"built in {t_built - t_start:.0f} s")

# ---- Solve ------------------------------------------------------------------

model.optimize()
t_solved = time.perf_counter()
if model.Status != GRB.OPTIMAL:
    raise SystemExit(f"no optimal solution, Gurobi status {model.Status}")

# ---- Results ----------------------------------------------------------------

plan = pd.DataFrame([(i, j, dist[i, j], x[i, j].X) for (i, j) in A if x[i, j].X > 1e-6],
                    columns=["i", "j", "dist", "x"])
plan.to_csv(out / "x.csv", index=False)

lost_per_day = sum(l[key].X for key in keys) / len(K)
demand_per_day = sum(d.values()) / len(K)
moved = plan["x"].sum()
mean_dist = (plan["x"] * plan["dist"]).sum() / moved if moved > 0 else 0.0

per_cell = pd.DataFrame({"cell": I, "arrondissement": [arrondissement[i] for i in I],
                         "s": [s[i] for i in I],
                         "sent": [sent[i].getValue() for i in I],
                         "received": [received[i].getValue() for i in I]})
per_cell["net_inflow"] = per_cell["received"] - per_cell["sent"]
lost_by_cell = pd.Series({i: 0.0 for i in I})
for (i, t, k) in keys:
    lost_by_cell[i] += l[i, t, k].X / len(K)
per_cell["lost_per_day"] = per_cell["cell"].map(lost_by_cell)
per_cell.to_csv(out / "cells.csv", index=False)

summary = {
    "run": run_name, "day_type": args.day_type, "days": K, "area": args.area, "baseline": args.baseline,
    "p": p, "h": h, "v": v, "Q": Q, "R": R, "M": M,
    "cells": len(I), "arcs": len(A), "variables": model.NumVars, "constraints": model.NumConstrs,
    "build_seconds": round(t_built - t_start, 1), "solve_seconds": round(t_solved - t_built, 1),
    "objective": model.ObjVal,
    "relocation_cost": relocation_cost.getValue(), "lost_cost_per_day": lost_cost.getValue(),
    "demand_per_day": demand_per_day, "lost_per_day": lost_per_day,
    "lost_share": lost_per_day / demand_per_day,
    "bikes_moved": moved, "mean_distance_km": mean_dist,
    # Sanity check: no cell gives away more than it has (constraint (1)).
    "max_sent_minus_s": float((per_cell["sent"] - per_cell["s"]).max()),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))

print(f"\n=== {run_name} ===")
print(f"solve time {summary['solve_seconds']} s, objective {model.ObjVal:,.2f} EUR per day")
print(f"lost trips per day   {lost_per_day:,.0f} of {demand_per_day:,.0f} ({summary['lost_share']:.1%})")
print(f"bikes relocated      {moved:,.0f}, mean distance {mean_dist:.2f} km")
print(f"relocation cost      {summary['relocation_cost']:,.2f} EUR")
print(f"lost trips cost/day  {summary['lost_cost_per_day']:,.2f} EUR")
print(f"max (sent - s)       {summary['max_sent_minus_s']:.6f}  (must be <= 0)")
if moved > 0:
    print("\nTop 10 net inflow (receive bikes):")
    print(per_cell.nlargest(10, "net_inflow").round(1).to_string(index=False))
    print("\nTop 10 net outflow (give bikes):")
    print(per_cell.nsmallest(10, "net_inflow").round(1).to_string(index=False))
print(f"\nresults in {out}")
