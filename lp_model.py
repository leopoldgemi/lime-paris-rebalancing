#!/usr/bin/env python3
"""
LP rebalancing model for Lime e-bikes in Paris, solved with Gurobi (gurobipy).

What it does
------------
At night, a van moves x[i,j] bikes from cell i to a nearby cell j. The same
night plan is used for every day k of the chosen day type. During the day,
riders take bikes where they are (served trips y) or, if the cell is empty,
the trip is lost (l). Every served trip brings its bike back within the same
hour, spread over the cells like the observed returns of that hour, so the
fleet stays constant (--fixed-returns uses the observed returns r instead).
The model chooses the night plan that minimises

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
    python lp_model.py --alpha 0.25               # less same-hour reuse of returned bikes
    python lp_model.py --fixed-returns            # first model version: returns r are fixed data
    python lp_model.py --input model_input_lp/disruption \
        --plan reports/lp/weekday_p3.5_R1.5_alpha0.5/x.csv   # test a fixed night plan on other days
"""

import argparse
import json
import time
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

BASE_DIR = Path(__file__).resolve().parent


# ---- Helper functions -------------------------------------------------------


def fix_plan(x, A, I, s, plan_path):
    """Set x to the night plan of an earlier run instead of optimising it.

    If a cell has fewer bikes at night than the plan wants to take away, all of
    its moves are scaled down by the same factor, so that constraint (1) still
    holds. Returns the number of planned bikes that could not be moved.
    """
    old_plan = pd.read_csv(plan_path)
    old_plan = old_plan[old_plan["i"].isin(I) & old_plan["j"].isin(I)]
    planned_sent = old_plan.groupby("i")["x"].sum()
    factor = {i: min(1.0, s[i] / planned_sent[i]) for i in planned_sent.index}

    # First no move at all, then exactly the (scaled) planned moves.
    for arc in A:
        x[arc].UB = 0
    dropped = 0.0
    for i, j, planned in zip(old_plan["i"], old_plan["j"], old_plan["x"]):
        x[i, j].LB = planned * factor[i]
        x[i, j].UB = planned * factor[i]
        dropped += planned * (1 - factor[i])
    print(f"fixed plan from {plan_path}: {old_plan['x'].sum():,.0f} bikes planned, "
          f"{dropped:,.1f} dropped because the origin cell had fewer bikes")
    return dropped


def write_results(out, run_name, args, model, cells, s, d, keys, A, dist, x, b, y, l, seconds, plan_dropped):
    """Save the night plan, per-cell results and key numbers, and print them.

    out, run_name, args   where to write and which run this is
    model                 the solved Gurobi model
    cells, s, d, keys     input data (cells table, night bikes, demand, all (i, t, k))
    A, dist               allowed pairs and their distance
    x, b, y, l            the model variables
    seconds               (build time, solve time)
    plan_dropped          bikes of a fixed plan that could not be moved (0 if no --plan)
    """
    I = list(cells["cell"])
    K = sorted({k for (i, t, k) in keys})
    T = sorted({t for (i, t, k) in keys})

    # Night plan: all pairs with x > 0.
    plan = pd.DataFrame([(i, j, dist[i, j], x[i, j].X) for (i, j) in A if x[i, j].X > 1e-6],
                        columns=["i", "j", "dist", "x"])
    plan.to_csv(out / "x.csv", index=False)
    moved = plan["x"].sum()
    mean_dist = (plan["x"] * plan["dist"]).sum() / moved if moved > 0 else 0.0

    # Trips per day, and the two parts of the cost.
    demand_per_day = sum(d.values()) / len(K)
    lost_per_day = sum(l[key].X for key in keys) / len(K)
    served_per_day = sum(y[key].X for key in keys) / len(K)
    lost_cost = args.p * lost_per_day
    relocation_cost = model.ObjVal - lost_cost

    # Fleet check: bikes in the city at the start of every hour (should equal sum(s)
    # for every t with endogenous returns; with --fixed-returns it drifts).
    fleet = pd.Series({(t, k): sum(b[i, t, k].X for i in I) for t in T for k in K})
    fleet_deviation = float((fleet - sum(s.values())).abs().max())

    # Per cell: bikes sent away, received, net inflow and lost trips per day.
    sent = {i: 0.0 for i in I}
    received = {i: 0.0 for i in I}
    for (i, j) in A:
        sent[i] += x[i, j].X
        received[j] += x[i, j].X
    lost_by_cell = {i: 0.0 for i in I}
    for (i, t, k) in keys:
        lost_by_cell[i] += l[i, t, k].X / len(K)
    per_cell = pd.DataFrame({"cell": I, "arrondissement": list(cells["arrondissement"]),
                             "s": [s[i] for i in I],
                             "sent": [sent[i] for i in I],
                             "received": [received[i] for i in I]})
    per_cell["net_inflow"] = per_cell["received"] - per_cell["sent"]
    per_cell["lost_per_day"] = [lost_by_cell[i] for i in I]
    per_cell.to_csv(out / "cells.csv", index=False)

    summary = {
        "run": run_name, "day_type": args.day_type, "days": K, "area": args.area, "baseline": args.baseline,
        "plan": args.plan, "plan_bikes_dropped": plan_dropped,
        "p": args.p, "h": args.h, "v": args.v, "Q": args.Q, "R": args.R, "M": args.M,
        "alpha": None if args.fixed_returns else args.alpha, "fixed_returns": args.fixed_returns,
        "cells": len(I), "arcs": len(A), "variables": model.NumVars, "constraints": model.NumConstrs,
        "build_seconds": round(seconds[0], 1), "solve_seconds": round(seconds[1], 1),
        "objective": model.ObjVal,
        "relocation_cost": relocation_cost, "lost_cost_per_day": lost_cost,
        "demand_per_day": demand_per_day, "served_per_day": served_per_day, "lost_per_day": lost_per_day,
        "lost_share": lost_per_day / demand_per_day,
        "bikes_moved": moved, "mean_distance_km": mean_dist,
        # Sanity check: no cell gives away more than it has (constraint (1)).
        "max_sent_minus_s": float((per_cell["sent"] - per_cell["s"]).max()),
        # Fleet check: largest gap between bikes in the city and sum(s), over all hours and days.
        "fleet_deviation": fleet_deviation,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {run_name} ===")
    print(f"solve time {summary['solve_seconds']} s, objective {model.ObjVal:,.2f} EUR per day")
    print(f"lost trips per day   {lost_per_day:,.0f} of {demand_per_day:,.0f} ({summary['lost_share']:.1%})")
    print(f"bikes relocated      {moved:,.0f}, mean distance {mean_dist:.2f} km")
    print(f"relocation cost      {relocation_cost:,.2f} EUR")
    print(f"lost trips cost/day  {lost_cost:,.2f} EUR")
    print(f"served trips per day {served_per_day:,.0f}")
    print(f"max (sent - s)       {summary['max_sent_minus_s']:.6f}  (must be <= 0)")
    print(f"fleet deviation      {fleet_deviation:.6f} bikes  (max |sum_i b[i,t,k] - sum(s)|, 0 with endogenous returns)")
    if moved > 0:
        print("\nTop 10 net inflow (receive bikes):")
        print(per_cell.nlargest(10, "net_inflow").round(1).to_string(index=False))
        print("\nTop 10 net outflow (give bikes):")
        print(per_cell.nsmallest(10, "net_inflow").round(1).to_string(index=False))
    print(f"\nresults in {out}")


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
parser.add_argument("--alpha", type=float, default=None,
                    help="share of the bikes returned in an hour that can be rented again in the same hour "
                         "(default: calibrated value, 0.5 weekday, 0.75 weekend)")
parser.add_argument("--fixed-returns", action="store_true",
                    help="first model version: returns r are fixed data, constraints (4) and (5) without Y")
parser.add_argument("--plan", default=None,
                    help="x.csv of an earlier run: use this night plan instead of optimising x (a test, not an optimisation)")
parser.add_argument("--out", default=None, help="output folder (default reports/lp/<run name>)")
args = parser.parse_args()

# Calibrated alpha: with x = 0 the model then serves as many trips as were observed
# (weekday 0.5: -0.4 %, weekend 0.75: +0.1 %). The value used is stored in args.
if args.alpha is None:
    args.alpha = {"weekday": 0.5, "weekend": 0.75}[args.day_type]

p, h, v, Q, R, M, alpha = args.p, args.h, args.v, args.Q, args.R, args.M, args.alpha

# Run name: one part per setting, so that every run gets its own folder.
dataset = Path(args.input).name
run_name = ""
if dataset != "base":
    run_name += dataset + "_"                             # data set, e.g. disruption_
run_name += f"{args.day_type}_p{p:g}_R{R:g}"             # day type, lost profit p, radius R
if M:
    run_name += f"_M{M:g}"                                # limit on relocated bikes
if args.fixed_returns:
    run_name += "_fixed-returns"                          # first model version
else:
    run_name += f"_alpha{alpha:g}"                        # same-hour reuse share
if args.area:
    run_name += f"_area{args.area}"                       # one arrondissement only
if args.baseline:
    run_name += "_baseline"                               # x = 0
if args.plan:
    run_name += "_plan-" + Path(args.plan).parent.name    # fixed plan of another run

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

# pi[j,t,k]: share of all returns in hour t of day k that end in cell j (sums to 1 over j).
returns_per_hour = hourly.groupby(["t", "day"])["r"].sum()
pi = {(j, t, k): (r[j, t, k] / returns_per_hour[t, k] if returns_per_hour[t, k] > 0 else 0.0)
      for (j, t, k) in keys}

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
# Constraint (8): non-negativity (lb=0), here and for all other variables.
x = model.addVars(A, lb=0, name="x")

# Variable b[i,t,k]: bikes in cell i at the start of hour t on day k. (8): b >= 0.
b = model.addVars(keys, lb=0, name="b")

# Variable y[i,t,k]: trips served in cell i in hour t on day k. (8): y >= 0.
y = model.addVars(keys, lb=0, name="y")

# Variable l[i,t,k]: trips lost in cell i in hour t on day k (rider found no bike). (8): l >= 0.
l = model.addVars(keys, lb=0, name="l")

# Baseline: no relocation at all.
if args.baseline:
    for arc in A:
        x[arc].UB = 0

# Fixed plan from an earlier run (a test, nothing is optimised in x).
plan_dropped = 0.0
if args.plan:
    plan_dropped = fix_plan(x, A, I, s, args.plan)

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

if args.fixed_returns:
    # Constraint (4), first version: hourly inventory balance with the observed returns.
    # Bikes next hour = bikes now - trips served + bikes returned (data r).
    model.addConstrs((b[i, t + 1, k] == b[i, t, k] - y[i, t, k] + r[i, t, k]
                      for (i, t, k) in keys if t < len(T)), name="balance")

    # Constraint (5), first version: a cell can only serve trips with the bikes it has.
    model.addConstrs((y[key] <= b[key] for key in keys), name="serve")
else:
    # Variable Y[t,k]: all trips served in the city in hour t on day k. (8): Y >= 0.
    Y = model.addVars(T, K, lb=0, name="Y")

    # Constraint (3): Y is the sum of the served trips over all cells.
    model.addConstrs((Y[t, k] == gp.quicksum(y[i, t, k] for i in I) for t in T for k in K),
                     name="total_served")

    # Constraint (4): hourly inventory balance with returns that follow the served trips.
    # Every served bike comes back in the same hour, spread over the cells like the
    # observed returns of that hour (share pi). So the fleet stays exactly constant.
    # Bikes next hour = bikes now - trips served + this cell's share of all served trips.
    model.addConstrs((b[j, t + 1, k] == b[j, t, k] - y[j, t, k] + pi[j, t, k] * Y[t, k]
                      for (j, t, k) in keys if t < len(T)), name="balance")

    # Constraint (5): same-hour reuse.
    # A cell can serve the bikes it has at the start of the hour plus a share alpha
    # of the bikes returned to it during the hour (they can be rented again at once).
    model.addConstrs((y[i, t, k] <= b[i, t, k] + alpha * pi[i, t, k] * Y[t, k]
                      for (i, t, k) in keys), name="serve")

# Constraint (6): every trip asked for is either served or lost.
model.addConstrs((y[key] + l[key] == d[key] for key in keys), name="demand")

# Constraint (7), optional: at most M bikes relocated per night.
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

write_results(out, run_name, args, model, cells, s, d, keys, A, dist, x, b, y, l,
              (t_built - t_start, t_solved - t_built), plan_dropped)
