#!/usr/bin/env python3
"""
Rebalancing model for Lime e-bikes in Paris (IEOR 4004).

A deterministic multi-period minimum-cost flow problem on a zone network: decide
how many bikes an operator should relocate between zones in each period so that
as little demand as possible is lost, at the lowest relocation cost.

FORMULATION
-----------
Sets
    i, j in Z   zones (grid cells with measurable demand)
    t in T      periods of a representative day (default 72 periods of 20 min)
    A           allowed relocation pairs (i, j) with dist(i, j) <= --max-move-km
    Tm          periods in which relocation is allowed (--move-hours)

Parameters, all estimated from the collected snapshots by build_demand.py
    d[i,t]      demand: rentals wanted in zone i during period t
    w[i,t]      share of period-t arrivals landing in zone i, sum_i w[i,t] = 1
    C[i]        parking capacity of zone i (Paris Open Data, legal bike racks)
    s0[i]       bikes in zone i at the start of the day
    rev         contribution of one rental (--revenue-per-trip), excluding the
                relocation cost, which the objective charges separately
    f, c        relocation cost per bike and per bike-km, derived from the van
                economics: a van holds --van-capacity bikes, costs
                --van-cost-per-hour, drives at --van-speed-kmh and needs
                --handling-min-per-bike to load and unload each bike. Handling
                falls on the single bike, driving time is shared by the van load,
                so f = handling * cost/min and c = (60/speed) * cost/min / capacity.
    M           optional cap on bikes relocated per period (--moves-per-hour)
    theta       share of a period a bike returning mid-period is still rentable

Decision variables
    s[i,t] >= 0     bikes in zone i at the start of period t
    r[i,t] >= 0     rentals served in zone i during period t
    u[i,t] >= 0     demand lost in zone i during period t
    x[i,j,t] >= 0   bikes relocated from i to j during period t

Objective
    max  rev * sum_{i,t} r[i,t]  -  sum_{(i,j) in A, t in Tm} (f + c * dist(i,j)) * x[i,j,t]

    Maximising profit and minimising (lost revenue + relocation cost) are the same
    problem, because r[i,t] + u[i,t] = d[i,t] makes rev * sum r a constant minus
    rev * sum u. Profit is reported because it is the quantity the operator cares
    about and it makes the relocation decision self-limiting: a bike is only worth
    moving if it earns back the van cost.

Constraints
    (1) r[i,t] + u[i,t] = d[i,t]                          demand is served or lost
    (2) r[i,t] <= d[i,t] * cov_i(s[i,t] + theta * w[i,t] * R[t])   coverage
    (3) sum_j x[i,j,t] <= s[i,t] - r[i,t] + w[i,t] * R[t]  only idle bikes can be moved
    (4) s[i,t+1] = s[i,t] - r[i,t] + w[i,t] * R[t]
                   - sum_j x[i,j,t] + sum_j x[j,i,t]      inventory balance
    (5) s[i,t] <= C[i]                                    parking capacity
    (6) sum_{(i,j) in A} x[i,j,t] <= M  for t in Tm        optional throughput cap
    (7) s[i,0] = s0[i], and with --cyclic also s[i,|T|] = s[i,0]

    where R[t] = sum_i r[i,t] is the rentals served in period t.

Two modelling choices are worth defending.

Arrivals. A bike rented in zone i returns somewhere, and the places it can
return to are distributed like the arrivals actually observed in that period.
Writing arrivals as w[i,t] * R[t] keeps the model linear (R[t] is a sum of
decision variables) and conserves the fleet: every served rental produces
exactly one arrival. Fixing arrivals at their observed level instead
(--arrival-model exogenous) creates bikes out of nothing whenever demand is lost.

Coverage. A rider is served only if an available bike stands within the walking
radius (300 m, from the literature on shared-bike access distance). The model
decides bikes per quartier, not their exact position, so coverage_curves.py
measures empirically how the share of a quartier's demand within 300 m of a bike
rises with the number of bikes standing in that quartier. That curve is concave
and increasing, so approximating it by segments keeps the problem linear:
constraint (2) is one inequality per segment. Only bikes inside the quartier
count towards its coverage, which makes the curves mildly pessimistic near
boundaries. With --no-coverage the model falls back to the cruder rule that any
bike anywhere in the quartier serves the whole quartier.

Turnover. Measured mean trip duration is about 17 minutes, so with a period of
20 minutes a bike that is returned mid-period can be rented again before the
period ends. Constraint (2) allows that for a fraction theta (--theta, default
0.5) of the arrivals. Without it the model would cap each bike at one rental per
period and overstate lost demand badly.

The LP relaxation is solved by default with CBC, whose dual on constraint (2)
is the marginal value of one more bike in zone i at period t and is what makes
the rebalancing decision interpretable. --integer switches to the MIP, solved by
HiGHS (no duals, but far faster on integers). --solver overrides the choice.

Usage
    python build_demand.py --data-dir ../lime-data/data      # once
    python rebalancing_model.py                              # weekday, night relocation
    python rebalancing_model.py --weekend --move-hours all
    python rebalancing_model.py --moves-per-hour 400 --penalty 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from analyze_snapshots import BASE_DIR, HAVE_MPL, M_PER_DEG_LAT, M_PER_DEG_LON

if HAVE_MPL:
    import matplotlib.pyplot as plt


def load_curves(inp: Path, Z: list[str]):
    """Piecewise-linear concave coverage curves per zone, from coverage_curves.py."""
    f = inp / "coverage_curves.csv"
    if not f.exists():
        return None
    cur = pd.read_csv(f)
    segs = {z: [(r.alpha, r.beta) for r in g.itertuples()] for z, g in cur.groupby("quartier")}
    missing = [z for z in Z if z not in segs]
    if missing:
        print(f"warning: no coverage curve for {len(missing)} zones, they fall back to count availability")
    return segs


def load_input(inp: Path, weekend: bool):
    tab = pd.read_csv(inp / "zone_period_demand.csv")
    zones = pd.read_csv(inp / "zones.csv")
    period_min = int(tab["period_minutes"].iloc[0])
    n_t = 1440 // period_min
    tab = tab[tab["weekend"] == weekend]
    if tab.empty:
        sys.exit(f"no {'weekend' if weekend else 'weekday'} data in {inp}")
    Z = sorted(zones["zone"])
    idx = pd.MultiIndex.from_product([Z, range(n_t)], names=["zone", "period"])
    g = tab.set_index(["zone", "period"]).reindex(idx).fillna(0.0)
    rate = g["departures_per_h"].unstack("period").loc[Z].to_numpy()
    arr = g["arrivals_per_h"].unstack("period").loc[Z].to_numpy()
    inv = g["mean_inventory"].unstack("period").loc[Z].to_numpy()
    d = rate * period_min / 60.0                       # rentals wanted per period
    tot = arr.sum(axis=0)
    w = np.divide(arr, tot, out=np.full_like(arr, 1.0 / len(Z)), where=tot > 0)
    zz = zones.set_index("zone").loc[Z]
    return (Z, d, w, inv, zz["capacity"].to_numpy(float), zz["lat"].to_numpy(),
            zz["lon"].to_numpy(), period_min, n_t)


def distances(lat, lon):
    y, x = lat * M_PER_DEG_LAT, lon * M_PER_DEG_LON
    return np.sqrt((y[:, None] - y[None, :]) ** 2 + (x[:, None] - x[None, :]) ** 2) / 1000.0


def make_solver(name: str, integer: bool, msg: bool):
    """CBC returns duals, HiGHS does not but is much faster on the MIP."""
    if name == "auto":
        name = "highs" if integer else "cbc"
    if name == "highs":
        return pulp.HiGHS(msg=msg), False
    return pulp.PULP_CBC_CMD(msg=msg), not integer


def build_and_solve(Z, d, w, s0, cap, dist, n_t, *, move_periods, max_move_km, moves_per_period,
                    revenue, cost_fix, cost_per_km, theta, cyclic, integer, solver="auto",
                    curves=None, allow_moves=True, msg=False):
    n = len(Z)
    T = range(n_t)
    pairs = [(i, j) for i in range(n) for j in range(n)
             if i != j and dist[i, j] <= max_move_km] if allow_moves else []
    m = pulp.LpProblem("lime_rebalancing", pulp.LpMaximize)
    cat = "Integer" if integer else "Continuous"

    s = {(i, t): pulp.LpVariable(f"s_{i}_{t}", 0, cap[i]) for i in range(n) for t in range(n_t + 1)}
    r = {(i, t): pulp.LpVariable(f"r_{i}_{t}", 0) for i in range(n) for t in T}
    u = {(i, t): pulp.LpVariable(f"u_{i}_{t}", 0) for i in range(n) for t in T}
    x = {(i, j, t): pulp.LpVariable(f"x_{i}_{j}_{t}", 0, None, cat=cat)
         for (i, j) in pairs for t in move_periods}

    m += (revenue * pulp.lpSum(r.values())
          - pulp.lpSum((cost_fix + cost_per_km * dist[i, j]) * x[i, j, t]
                       for (i, j) in pairs for t in move_periods))

    served = {t: pulp.lpSum(r[i, t] for i in range(n)) for t in T}
    for i in range(n):
        m += s[i, 0] == s0[i], f"init_{i}"
        for t in T:
            m += r[i, t] + u[i, t] == d[i, t], f"demand_{i}_{t}"                       # (1)
            # bikes that provide service during the period: those present at the start
            # plus the share theta of those returned mid-period
            present = s[i, t] + theta * w[i, t] * served[t]
            segs = curves.get(Z[i]) if curves else None
            if segs:
                # (2) r <= d * coverage(present), coverage concave piecewise linear
                for k, (al, be) in enumerate(segs):
                    m += r[i, t] <= d[i, t] * (al + be * present), f"cov_{i}_{t}_{k}"
                m += r[i, t] <= present, f"avail_{i}_{t}"
            else:
                m += r[i, t] - theta * w[i, t] * served[t] <= s[i, t], f"avail_{i}_{t}"  # (2)
            out = pulp.lpSum(x[i, j, t] for j in range(n) if (i, j, t) in x)
            inn = pulp.lpSum(x[j, i, t] for j in range(n) if (j, i, t) in x)
            if t in move_periods and pairs:
                # bikes physically idle in the zone at the end of the period: start
                # inventory, less what was rented out, plus what was returned. With
                # out = 0 this is implied by s >= 0, so it does not tighten the
                # no-relocation case.
                m += out <= s[i, t] - r[i, t] + w[i, t] * served[t], f"movable_{i}_{t}"  # (3)
            m += s[i, t + 1] == s[i, t] - r[i, t] + w[i, t] * served[t] - out + inn, f"bal_{i}_{t}"  # (4)
        if cyclic:
            m += s[i, n_t] == s[i, 0], f"cyclic_{i}"
    if moves_per_period is not None:
        for t in move_periods:
            if pairs:
                m += pulp.lpSum(x[i, j, t] for (i, j) in pairs) <= moves_per_period, f"fleet_{t}"  # (6)

    solver_obj, has_duals = make_solver(solver, integer, msg)
    m.solve(solver_obj)
    status = pulp.LpStatus[m.status]
    if status != "Optimal":
        return {"status": status}
    U = np.array([[u[i, t].value() or 0.0 for t in T] for i in range(n)])
    R = np.array([[r[i, t].value() or 0.0 for t in T] for i in range(n)])
    S = np.array([[s[i, t].value() or 0.0 for t in range(n_t + 1)] for i in range(n)])
    moves = [{"from": Z[i], "to": Z[j], "period": t, "bikes": x[i, j, t].value() or 0.0, "km": dist[i, j]}
             for (i, j) in pairs for t in move_periods if (x[i, j, t].value() or 0) > 1e-6]
    duals = None
    if has_duals:
        # CBC reports the dual of a <= constraint in a minimisation as a non-positive
        # number; the economic shadow price of one more bike is its magnitude.
        duals = -np.array([[m.constraints[f"avail_{i}_{t}"].pi or 0.0 for t in T] for i in range(n)])
        duals = np.maximum(duals, 0.0)
    reloc_cost = sum((cost_fix + cost_per_km * dist[i, j]) * (x[i, j, t].value() or 0.0)
                     for (i, j) in pairs for t in move_periods)
    return {"status": status, "objective": pulp.value(m.objective), "unmet": U, "served": R,
            "revenue": revenue * R.sum(), "reloc_cost": reloc_cost,
            "inventory": S, "moves": pd.DataFrame(moves), "duals": duals,
            "n_vars": len(m.variables()), "n_cons": len(m.constraints)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="model_input")
    p.add_argument("--out", default="reports/model")
    p.add_argument("--weekend", action="store_true")
    p.add_argument("--move-hours", default="23,0,1,2,3,4,5",
                   help='hours in which relocation is allowed, or "all" (default: night)')
    p.add_argument("--max-move-km", type=float, default=2.0)
    p.add_argument("--moves-per-hour", type=float, default=None,
                   help="optional cap on bikes relocated per hour; by default the van "
                        "economics alone decide how much relocation is worth it")
    p.add_argument("--revenue-per-trip", type=float, default=2.50,
                   help="contribution of one rental in euros, EXCLUDING relocation cost, which the "
                        "model charges separately. Gross revenue per rental in Paris is about "
                        "2.50-3.50 after passes and subscriptions; Lime's reported adjusted EBITDA "
                        "of 1.00-1.75 per ride is net of all field costs including relocation, so "
                        "using it here would double-count the vans. 2.50 is the base case, "
                        "1.75 and 3.50 bracket it.")
    p.add_argument("--van-capacity", type=float, default=15, help="e-bikes per van load (10-20)")
    p.add_argument("--van-cost-per-hour", type=float, default=45.0, help="driver plus vehicle, euros/h")
    p.add_argument("--handling-min-per-bike", type=float, default=2.0, help="load plus unload, minutes")
    p.add_argument("--van-speed-kmh", type=float, default=20.0, help="average night speed in Paris")
    p.add_argument("--theta", type=float, default=0.5,
                   help="share of mid-period arrivals that can be rented again in the same period")
    p.add_argument("--cyclic", action="store_true")
    p.add_argument("--integer", action="store_true")
    p.add_argument("--no-coverage", action="store_true",
                   help="ignore the walking-radius coverage curves and only require a bike in the zone")
    p.add_argument("--solver", default="auto", choices=["auto", "cbc", "highs"],
                   help="auto = CBC for the LP (gives shadow prices), HiGHS for the MIP")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    resolve = lambda s: Path(s) if Path(s).is_absolute() else BASE_DIR / s  # noqa: E731
    inp, out = resolve(a.input), resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)

    Z, d, w, inv, cap, lat, lon, period_min, n_t = load_input(inp, a.weekend)
    curves = None if a.no_coverage else load_curves(inp, Z)
    per_hour = 60 // period_min
    hours = list(range(24)) if a.move_hours == "all" else [int(h) for h in a.move_hours.split(",")]
    move_periods = [h * per_hour + k for h in hours for k in range(per_hour)]
    dist = distances(lat, lon)
    s0 = inv[:, 0].copy()
    day = "weekend" if a.weekend else "weekday"
    label = lambda t: f"{t * period_min // 60:02d}:{t * period_min % 60:02d}"  # noqa: E731
    print(f"{len(Z)} zones, {n_t} periods of {period_min} min, {day}; "
          f"fleet at 00:00 {s0.sum():.0f}, demand {d.sum():,.0f} rentals/day; "
          f"service rule: {'coverage curves (walking radius)' if curves else 'any bike in the zone'}")

    # Van economics, converted to a cost per relocated bike. A van runs full, so the
    # cost of one trip is shared by --van-capacity bikes: handling is charged per bike,
    # driving time per bike-km.
    eur_per_min = a.van_cost_per_hour / 60.0
    cost_fix = a.handling_min_per_bike * eur_per_min
    cost_per_km = (60.0 / a.van_speed_kmh) * eur_per_min / a.van_capacity
    print(f"van economics: {cost_fix:.2f} EUR handling per bike + {cost_per_km:.3f} EUR per bike-km "
          f"({a.van_capacity:.0f} bikes per van, {a.van_cost_per_hour:.0f} EUR/h, {a.van_speed_kmh:.0f} km/h); "
          f"revenue {a.revenue_per_trip:.2f} EUR per rental")
    common = dict(move_periods=move_periods, max_move_km=a.max_move_km,
                  moves_per_period=(a.moves_per_hour * period_min / 60.0
                                    if a.moves_per_hour else None),
                  revenue=a.revenue_per_trip, cost_fix=cost_fix, cost_per_km=cost_per_km,
                  theta=a.theta, cyclic=a.cyclic, integer=a.integer,
                  solver=a.solver, curves=curves, msg=a.verbose)
    print("solving baseline (no relocation) ...")
    base = build_and_solve(Z, d, w, s0, cap, dist, n_t, allow_moves=False, **common)
    print("solving with relocation ...")
    opt = build_and_solve(Z, d, w, s0, cap, dist, n_t, allow_moves=True, **common)
    for name, res in (("baseline", base), ("optimised", opt)):
        if res["status"] != "Optimal":
            sys.exit(f"{name} problem not solved: {res['status']}")

    lost_b, lost_o, total_d = base["unmet"].sum(), opt["unmet"].sum(), d.sum()
    moves = opt["moves"]
    moved = moves["bikes"].sum() if len(moves) else 0.0
    bike_km = (moves["bikes"] * moves["km"]).sum() if len(moves) else 0.0

    pd.DataFrame({"zone": Z, "lat": lat, "lon": lon, "capacity": cap,
                  "demand_day": d.sum(axis=1), "lost_baseline": base["unmet"].sum(axis=1),
                  "lost_optimised": opt["unmet"].sum(axis=1), "start_inventory": s0}) \
        .sort_values("lost_baseline", ascending=False).to_csv(out / f"zones_{day}.csv", index=False)
    if len(moves):
        mv = moves.copy(); mv["time"] = mv["period"].map(label)
        mv.sort_values(["period", "bikes"], ascending=[True, False]).to_csv(out / f"moves_{day}.csv", index=False)
    pd.DataFrame({"period": range(n_t), "time": [label(t) for t in range(n_t)],
                  "demand": d.sum(axis=0), "lost_baseline": base["unmet"].sum(axis=0),
                  "lost_optimised": opt["unmet"].sum(axis=0)}).to_csv(out / f"periods_{day}.csv", index=False)
    if opt["duals"] is not None:
        pd.DataFrame(opt["duals"], index=Z, columns=[label(t) for t in range(n_t)]) \
            .to_csv(out / f"shadow_prices_{day}.csv")

    L = [f"# Rebalancing model - {day}", "",
         f"- {len(Z)} zones, {n_t} periods of {period_min} min; relocation allowed in hours {hours} "
         f"between zones up to {a.max_move_km:.1f} km apart",
         f"- problem size {opt['n_vars']:,} variables, {opt['n_cons']:,} constraints "
         f"({'MIP' if a.integer else 'LP relaxation'}, HiGHS)",
         f"- revenue {a.revenue_per_trip:.2f} EUR per rental; relocation {cost_fix:.2f} EUR per bike "
         f"+ {cost_per_km:.3f} EUR per bike-km ({a.van_capacity:.0f} bikes/van, "
         f"{a.van_cost_per_hour:.0f} EUR/h, {a.van_speed_kmh:.0f} km/h)",
         f"- theta {a.theta}, cyclic {a.cyclic}, "
         + (f"cap {a.moves_per_hour:.0f} relocations/h" if a.moves_per_hour else "no cap on relocations"), "",
         "| | rentals served | lost | revenue EUR | relocation cost EUR | profit EUR | bikes moved | bike-km |",
         "|---|---|---|---|---|---|---|---|",
         f"| no relocation | {total_d - lost_b:,.0f} | {lost_b:,.0f} | {base['revenue']:,.0f} | 0 | "
         f"{base['objective']:,.0f} | 0 | 0 |",
         f"| optimised | {total_d - lost_o:,.0f} | {lost_o:,.0f} | {opt['revenue']:,.0f} | "
         f"{opt['reloc_cost']:,.0f} | {opt['objective']:,.0f} | {moved:,.0f} | {bike_km:,.0f} |", ""]
    gain = opt["objective"] - base["objective"]
    L += [f"Relocation adds {gain:,.0f} EUR of profit per {day} "
          f"({gain / max(moved, 1e-9):.2f} EUR per bike moved, "
          f"{(lost_b - lost_o) / max(moved, 1e-9):.2f} extra rentals per bike moved).", ""]
    if lost_b > 0:
        L.append(f"That recovers {lost_b - lost_o:,.0f} of the {lost_b:,.0f} lost rentals "
                 f"({(lost_b - lost_o) / lost_b:.1%}).")
    else:
        L.append("The fleet is large enough that no demand is lost even without relocation.")
    L.append("")

    if len(moves):
        bh = moves.assign(hour=moves["period"] * period_min // 60).groupby("hour")["bikes"].sum()
        L += ["Relocations by hour: " + ", ".join(f"{h}h {v:.0f}" for h, v in bh.items()), "",
              "Biggest single relocations:", "", "| time | from | to | bikes | km |", "|---|---|---|---|---|"]
        for mr in moves.sort_values("bikes", ascending=False).head(10).itertuples():
            L.append(f"| {label(mr.period)} | {getattr(mr, '_1')} | {mr.to} | {mr.bikes:.0f} | {mr.km:.1f} |")
        L.append("")

    worst = pd.DataFrame({"lost": base["unmet"].sum(axis=1), "demand": d.sum(axis=1),
                          "lat": lat, "lon": lon}).sort_values("lost", ascending=False).head(10)
    L += ["Zones losing the most demand without relocation:", "",
          "| zone centre | demand/day | lost/day |", "|---|---|---|"]
    for zr in worst.itertuples():
        L.append(f"| {zr.lat:.4f}, {zr.lon:.4f} | {zr.demand:,.0f} | {zr.lost:,.0f} |")
    if opt["duals"] is not None:
        dv = opt["duals"]
        # a bike can serve at most a couple of rentals in one period, so a shadow
        # price far above the penalty means the LP is degenerate there and CBC
        # returned a meaningless dual; report those separately rather than as a price.
        bogus = dv > 10 * a.revenue_per_trip
        clean = np.where(bogus, 0.0, dv)
        binding = clean > 1e-6
        L += ["", f"Shadow price of one more bike, EUR (dual of the availability constraint): "
                  f"max {clean.max():.2f}, mean where binding "
                  f"{clean[binding].mean() if binding.any() else 0:.2f}, binding in "
                  f"{binding.mean():.1%} of zone-periods."]
        if bogus.any():
            L.append(f"  ({bogus.sum():,} zone-periods, {bogus.mean():.1%}, returned a degenerate dual "
                     f"above {10 * a.revenue_per_trip:.0f} and were excluded; the LP has alternative optima there.)")

    if HAVE_MPL:
        tt = np.arange(n_t) * period_min / 60
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(tt, d.sum(axis=0) * per_hour, color="grey", ls="--", label="demand")
        ax.plot(tt, base["unmet"].sum(axis=0) * per_hour, color="C3", label="lost without relocation")
        ax.plot(tt, opt["unmet"].sum(axis=0) * per_hour, color="C2", label="lost with relocation")
        ax.set_xticks(range(0, 25, 2)); ax.set_xlabel("hour (Paris)"); ax.set_ylabel("rentals per hour")
        ax.set_title(f"Demand and lost demand, {day}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"lost_demand_{day}.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 8))
        lost_zone = base["unmet"].sum(axis=1)
        qpath = BASE_DIR / "data" / "reference" / "quartiers_paris.geojson"
        drawn = False
        if qpath.exists():
            import json as _json
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Polygon as MplPolygon
            gj = _json.loads(qpath.read_text())
            by_name = {f"{f['properties']['c_qu']}_{f['properties']['l_qu']}".replace(" ", "-"): f
                       for f in gj["features"]}
            patches, vals = [], []
            for k, z in enumerate(Z):
                feat = by_name.get(z)
                if feat is None:
                    continue
                geom = feat["geometry"]
                rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else \
                        [poly[0] for poly in geom["coordinates"]]
                for ring in rings:
                    patches.append(MplPolygon(np.array(ring), closed=True))
                    vals.append(lost_zone[k])
            if patches:
                pc = PatchCollection(patches, cmap="Reds", edgecolor="white", linewidths=0.5)
                pc.set_array(np.array(vals))
                ax.add_collection(pc)
                sc = pc
                drawn = True
        if not drawn:
            sc = ax.scatter(lon, lat, c=lost_zone, s=34, cmap="Reds", marker="s",
                            edgecolors="grey", linewidths=0.2)
        if len(moves):
            agg = moves.groupby(["from", "to"])["bikes"].sum().reset_index()
            pos = {z: (lon[k], lat[k]) for k, z in enumerate(Z)}
            for mr in agg.sort_values("bikes", ascending=False).head(60).itertuples():
                x0, y0 = pos[getattr(mr, "_1")]; x1, y1 = pos[mr.to]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color="C0", alpha=0.6,
                                            lw=0.4 + 2.5 * mr.bikes / agg["bikes"].max()))
        ax.set_aspect(M_PER_DEG_LAT / M_PER_DEG_LON)
        ax.set_xlim(lon.min() - 0.012, lon.max() + 0.012)
        ax.set_ylim(lat.min() - 0.008, lat.max() + 0.008)
        if len(moves):
            top = agg.nlargest(12, "bikes")
            labelled = set()
            for mr in top.itertuples():
                for z in (getattr(mr, "_1"), mr.to):
                    if z in labelled:
                        continue
                    labelled.add(z)
                    k = Z.index(z)
                    ax.text(lon[k], lat[k], z.split("_", 1)[1], fontsize=6, ha="center",
                            va="center", color="black",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65))
        ax.set_title(f"Lost demand per quartier and overnight relocation flows ({day})")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        fig.colorbar(sc, ax=ax, label="lost rentals per day (no relocation)", shrink=0.8)
        fig.tight_layout(); fig.savefig(out / f"moves_map_{day}.png", dpi=140); plt.close(fig)
        L += ["", f"Plots: lost_demand_{day}.png, moves_map_{day}.png"]

    text = "\n".join(L) + "\n"
    (out / f"summary_{day}.md").write_text(text)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
