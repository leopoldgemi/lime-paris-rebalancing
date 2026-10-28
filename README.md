# Lime Paris e-bike rebalancing – data collection

Columbia IEOR 4004 (Optimization) project. Lime publishes no historical vehicle
positions, so we record them ourselves from the public GBFS feed.

- Source: Lime Paris GBFS v2.2, discovery feed
  `https://data.lime.bike/api/partners/v2/gbfs/paris/gbfs.json`
- Licence: [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/) (attribution: Lime)

## What the feed actually contains (checked 2026-09-30)

| Feed | Content |
|---|---|
| `free_bike_status` | ~6 900 vehicles, `ttl` 60 s. Every vehicle has `bike_id` (UUID), `lat`, `lon`, `vehicle_type_id`, `is_reserved`, `is_disabled`, `current_range_meters` (filled for 100 %, 0–102 km), `last_reported`, plus a non-standard `vehicle_type` ("e-bike"). |
| `vehicle_types` | 4 types: `1`/`2` electric scooters, `3` e-bike (`max_range_meters` 85 000), `4` human-powered bike. In practice ≈ all vehicles are type 3 (e-bike); scooters were banned in Paris in 2023. |
| `system_information` | `lime_paris`, timezone Europe/Paris |
| `station_information` / `station_status` | one dummy "station" covering all of Paris – no real docks. Useless. |
| `geofencing_zones` | **not offered** (not listed in discovery, direct URL gives 404). |

**`bike_id` is NOT stable.** Lime replaces every id in the feed at once, about
every 15 minutes (observed boundaries at ~23:45 and 00:00 UTC on 2026-09-30;
0 % id overlap across the boundary, 99 % within an epoch). This follows the GBFS
recommendation to rotate ids for privacy. Consequences and the workaround are in
"Id rotation" below.

## Files

| File | Purpose |
|---|---|
| `collect_lime.py` | collector (requests + stdlib only). Reads feed URLs from the discovery feed, polls `free_bike_status` every 2 min, appends to `data/lime_paris_YYYY-MM-DD.csv`, saves static feeds once to `data/reference/*.json`. Network errors are logged, loop continues. |
| `analyze_snapshots.py` | first overview (pandas, matplotlib optional): fleet per snapshot, e-bike/scooter split, reserved/disabled/low-battery shares, hourly profile, 500 m grid density, position changes (trip proxy). Output in `reports/`. |
| `check_id_stability.py` | id overlap first vs last snapshot, new/gone ids per snapshot, rotation boundaries, day-to-day overlap. |
| `linking.py` | re-links ids across rotations via the (lat, lon, range) fingerprint of parked bikes, gives a chain id `uid`. |
| `reconstruct_trips.py` | classifies consecutive sightings of a chain into trip / ops / unclear, writes `reports/trips.csv`. Thresholds are parameters (top of file + CLI flags). |
| `trip_stats.py` | trips per hour, event shares, duration/distance distributions, low-range share; PNG plots in `reports/`. |
| `fetch_parking_zones.py` | downloads Paris Open Data on-street bike parking where free-floating bikes may park, aggregates capacity per grid cell / arrondissement. |
| `.github/workflows/collect-loop.yml` | **active collector**: self-chaining ~5.5 h jobs, real 2-min polling, commits every 10 min to the `data` branch. |
| `.github/workflows/collect.yml` | manual single snapshot (the cron schedule never fired; superseded by the loop). |
| `.github/workflows/daily-release.yml` | daily 03:40 UTC: merges yesterday's snapshots into `lime_paris_YYYY-MM-DD.csv.gz` on the release `daily-data`. |
| `.github/workflows/keepalive.yml` | weekly: empty commit if `main` is older than 45 days (GitHub disables schedules after 60 idle days). |
| `build_model_inputs.py` | LP inputs on a 300 m grid: night inventory s, returns r, empty-minute-corrected demand d per cell/hour/day, distances (see "Model"). |
| `lp_model.py` | LP rebalancing model in gurobipy (see "Model"). |
| `plot_plan_map.py` | map of a night plan: cells coloured by net inflow, arrondissement borders. |
| `merge_day.py` | merges one day of per-snapshot files into a single daily csv.gz (stdlib). |
| `deploy/` | systemd unit + setup script for a small Linux VM; bootstrap script for the `data` branch. |

All paths are relative to the script location, so the repo runs unchanged on any machine.

## Run locally

```bash
pip install -r requirements.txt
python collect_lime.py              # runs until Ctrl-C, one snapshot every 120 s
python collect_lime.py --once       # single snapshot
python analyze_snapshots.py         # data/ -> reports/summary.md + csv + png
```

CSV columns: `snapshot_utc, feed_last_updated, bike_id, lat, lon, vehicle_type_id,
vehicle_type, is_reserved, is_disabled, current_range_meters, last_reported`
(booleans stored as 0/1). One snapshot ≈ 6 900 rows ≈ 850 KB raw, ≈ 230 KB gzip.

## Running 24/7 – options

| | GitHub Actions (`collect.yml`) | Small VM (Hetzner CX22 ≈ €4/month, or Oracle Cloud "Always Free") |
|---|---|---|
| Cost | free (repo must be **public**; a private repo burns the 2 000 free minutes in ~10 days) | ≈ €4/month or free (Oracle) |
| Cadence | `*/5` is the minimum; runs are delayed 5–20 min at busy times, so expect **~200–250 snapshots/day**, irregular spacing | exact 2 min, 720 snapshots/day |
| Storage | git commits on `data` branch, ~60 MB/day, ~1.5–2 GB/month (GitHub recommends < 5 GB) | disk, ~600 MB/day raw; gzip daily files, `rsync`/`scp` to laptop |
| Ops | zero – but scheduled workflows are switched off after 60 days without repo activity | one-time SSH setup (`deploy/setup_vm.sh`), then `systemctl`; someone must keep the VM alive |
| Failure modes | GitHub outages, missed runs (gaps), workflow silently disabled | VM reboot (systemd restarts it), disk full |

### GitHub Actions: the collection loop

GitHub's cron scheduler never fired for this repo (0 runs in 50 min), so the
collector runs as a chain of long jobs instead: `collect-loop.yml` polls every
2 min for 330 min (job limit is 6 h), commits every 10 min, and queues its own
successor at start via `workflow_dispatch`. The concurrency group holds the
successor until the current run ends or crashes, so there is no gap. An hourly
cron only re-seeds the chain if it ever breaks.

```bash
gh workflow run collect-loop.yml                 # start (or re-seed) the chain
gh run list --workflow collect-loop --limit 3    # one in_progress + one pending = healthy
```

The successor is dispatched twice: at start (covers crashes) and again right
before the job ends (the start-queued run once stayed "pending" after its
predecessor finished, causing a 73-min gap on 2026-10-01; a fresh dispatch at
the end starts within seconds). GitHub keeps only one pending run per
concurrency group, so a newer dispatch (or the hourly cron) simply replaces
the older pending one. Handover gap is ~3 min (one missed snapshot).

To stop: cancel the running and the queued run in the Actions tab.

### GitHub Actions setup (one-time)

```bash
gh repo create lime-paris-rebalancing --public --source . --push
bash deploy/bootstrap_data_branch.sh   # creates orphan "data" branch + reference JSONs
gh workflow run collect-lime-paris      # first manual run; then every 5 min
```

Get the data for analysis – easiest: the merged daily files from the release
(one file per UTC day, plus `reference/*.json`):

```bash
gh release download daily-data -D data/ --clobber      # or download from the Releases page
python analyze_snapshots.py                              # reads data/lime_paris_*.csv.gz
```

Raw per-snapshot files (including today's, not yet merged):

```bash
git clone -b data --single-branch https://github.com/leopoldvscholz-star/lime-paris-rebalancing lime-data
python analyze_snapshots.py --data-dir ../lime-data/data
```

**Known limitations** (GitHub docs): the schedule is best-effort, minimum
interval 5 minutes, high delay at the top of the hour; scheduled workflows are
disabled after 60 days of no commits by a human. `analyze_snapshots.py`
reports the real median/max gap between snapshots.

### VM setup

```bash
# on a fresh Ubuntu 24.04 VM, as root
REPO=leopoldvscholz-star/lime-paris-rebalancing bash deploy/setup_vm.sh
journalctl -u lime-collector -f
# fetch data to your laptop
rsync -avz lime@<vm-ip>:lime-paris-rebalancing/data/ ./data/
```

## Analysis pipeline

```bash
python check_id_stability.py          # is bike_id stable? rotation boundaries, link rate
python reconstruct_trips.py           # -> reports/trips.csv   (flags: --min-trip-dist-m 300 ...)
python trip_stats.py                  # -> reports/trip_stats.md + png
python fetch_parking_zones.py         # -> reports/parking_capacity_grid.csv
```

### Id rotation and linking (`linking.py`)

- A rotation is detected when the id overlap between two consecutive snapshots
  is below `ROTATION_THRESHOLD = 0.5`.
- Across a rotation, a new id is linked to a vanished id if both have the same
  fingerprint `(round(lat, 6), round(lon, 6), current_range_meters)` and the
  fingerprint is unique on both sides. Parked bikes keep this fingerprint
  exactly. Observed link rate: 91 % across a 15-min gap, 98 % across 2 min.
- The chain id `uid` is the bike_id of the first sighting. A bike that is riding
  (or in a van) during a rotation cannot be linked: its chain ends, a new chain
  starts, and that trip is lost. Expected loss ≈ polling interval / 15 min
  (≈ 13 % of trips at 2-min polling, ≈ 33 % at 5-min polling). Trips are
  therefore undercounted; the hourly profile shape is unaffected if the loss is
  time-independent.
- Within an epoch ids follow the bike through a ride (verified: same id, new
  position, lower range).

### Event classification (`reconstruct_trips.py`)

An event is a pair of consecutive sightings of one chain (A at `t_prev`, B at `t_next`).

| Parameter | Default | Meaning |
|---|---|---|
| `MIN_TRIP_DIST_M` | 200 | displacement below this is not a relocation (GPS jitter) |
| `STATIONARY_DIST_M` | 50 | below this the bike has not moved at all |
| `MIN_TRIP_MIN` / `MAX_TRIP_MIN` | 2 / 90 | plausible gap between sightings for a ride |
| `RANGE_TOL_M` | 500 | range may rise by up to this (sensor noise) and still count as a ride |
| `RANGE_INCREASE_OPS_M` | 5 000 | range rise above this = battery swap (ops) |
| `OPS_BATCH_MIN_BIKES` / `OPS_BATCH_RADIUS_M` | 4 / 100 | ≥ 4 bikes reappearing in the same snapshot in one 100 m cell = van drop-off (ops) |
| `MAX_STILL_GAP_MIN` | 90 | same position but absent longer than this = unclear |

Rules, first match wins:

1. `ops` – range rose by > `RANGE_INCREASE_OPS_M` (in place or relocated)
2. `ops` – relocated and part of a batch drop-off
3. `trip` – relocated ≥ `MIN_TRIP_DIST_M`, gap within [`MIN_TRIP_MIN`, `MAX_TRIP_MIN`], range delta ≤ `RANGE_TOL_M`
4. `unclear` – relocated but gap > `MAX_TRIP_MIN` (long absence), gap < `MIN_TRIP_MIN`, or moderate range rise
5. `unclear` – same position but gap > `MAX_STILL_GAP_MIN`; small displacement (50–200 m) after an absence
6. ignored – same position, short gap (still parked); jitter without absence

Assumptions behind this:

- Lime hides a bike from the feed while it is rented (and in a van). Absence
  therefore means "in use or in operations", presence means "available".
- Duration = gap between last sighting at A and first sighting at B, so it
  overstates the ride by up to one polling interval plus idle time at B.
- Distance is straight-line between A and B, not the ridden route.
- A battery swap raises `current_range_meters` by tens of km; rides only lower it.
- Van pick-ups are invisible as such (the bike just vanishes); only drop-offs
  are detectable, via the batch rule or the range jump.
- Reserved bikes stay in the feed (`is_reserved = 1`); they are not treated specially yet.

### Parking capacity (Paris Open Data)

Dataset `stationnement-voie-publique-emplacements` (licence ODbL) has 65 833
on-street parking spots with a field
`stationnement_autorises_aux_velos_en_libre_service`. 12 997 spots are "oui"
(all `regpar = "Vélos"`, i.e. bike racks), with `placal` = computed places:
**137 864 places** in total, 3 300–10 300 per arrondissement. This is the
legal parking capacity for free-floating bikes per zone (Paris requires
free-floating bikes to be parked on these racks or in dedicated bays).
`fetch_parking_zones.py` aggregates it on the same 500 m grid as the
snapshot density, so capacity − mean vehicles gives free parking per cell.
Caveats: capacity is shared with private bikes and other operators (Vélib'
stations are separate records, `regpar = "Vélib'"`); the city does not
guarantee completeness.

## Model

The rebalancing model is a linear program solved with Gurobi. A van moves
bikes between nearby 300 m cells at night (one plan, used every day); during
the day riders take bikes where they are, and a trip is lost when the cell is
empty. Every served trip brings its bike back in the same hour, spread over
the cells like the observed returns of that hour, so the fleet stays constant.
The LP chooses the night plan with the lowest cost of relocation plus lost trips.

    min  sum_{(i,j) in A} c_ij x_ij + (1/|K|) sum_{i,t,k} p l_itk,   c_ij = h + v dist_ij / Q
    (1) sum_j x_ij <= s_i                            supply at night
    (2) b_i1k = s_i - sum_j x_ij + sum_j x_ji        inventory at the start of the day
    (3) Y_tk = sum_i y_itk                           all trips served in hour t
    (4) b_j,t+1,k = b_jtk - y_jtk + pi_jtk Y_tk      hourly balance, returns follow served trips
    (5) y_itk <= b_itk + alpha pi_itk Y_tk           bikes present + same-hour reuse of returns
    (6) y_itk + l_itk = d_itk                        every trip asked for is served or lost
    (7) sum x_ij <= M                                optional limit per night
    (8) x, b, y, l, Y >= 0                           non-negativity

| Symbol | Meaning | Default |
|---|---|---|
| i, j | 300 m cells with centre inside Paris (1 178) | |
| t | hour of the operating day, 1..18 (weekday 07:00-01:00, weekend 09:00-03:00) | |
| k | operating day of one type (weekday or weekend) | |
| s_i | bikes in the cell at night (01:00 weekday, 03:00 weekend), mean over days | data |
| r_itk | returns (trip ends) | data |
| pi_jtk | r_jtk / sum_j r_jtk: share of the returns of hour t, day k that end in cell j | data |
| alpha | share of the bikes returned in an hour that can be rented again in the same hour | 0.5 weekday, 0.75 weekend |
| d_itk | demand = rentals corrected for minutes without any available bike | data |
| A | pairs i != j with dist_ij <= R | R = 1.5 km |
| p, h, v, Q, M | lost profit per trip, handling cost per bike, van cost per km, bikes per van, limit | 3.50, 0.67, 1.30, 15, off |

alpha = 0.5 is calibrated: with x = 0 the model then serves 56,263 trips per
weekday against 56,464 observed rentals (-0.4 %; alpha 0 / 0.25 / 0.75 / 1 give
-7.5 / -3.2 / +1.4 / +2.6 %). On weekends alpha = 0.75 fits best (57,320 vs
57,248, +0.1 %; alpha 0 / 0.25 / 0.5 / 1 give -9.9 / -5.0 / -1.9 / +1.4 %). `--fixed-returns` runs the first model version, in
which (4) uses the observed returns r instead of pi_jtk Y_tk and (5) is y_itk <= b_itk; served bikes then
leave the model, so relocation helps much less (kept for comparison in the report).

### Setup (once)

1. Python 3.10 or newer, then `pip install -r requirements.txt` (includes `gurobipy`).
2. Gurobi needs a licence for models above 2 000 variables. The **Gurobi Academic
   License is free** for students: register with the university e-mail at
   <https://www.gurobi.com/academia/academic-program-and-licenses/>, create a
   "WLS Academic" licence in the Gurobi portal and save the `gurobi.lic` file it
   gives you in your home folder (`~/gurobi.lic`).
3. Check: `python -c "import gurobipy; gurobipy.Model()"` must print
   "Academic license ...", not "Restricted license".
4. Snapshot data: download the daily files from the `daily-data` release, e.g.
   `gh release download daily-data -R leopoldvscholz-star/lime-paris-rebalancing -D ../lime-data/daily`.

### Step 1: model inputs (`build_model_inputs.py`, ~10 s per day)

```bash
# base case: 02.-16.10., without 06.10. (demonstration day, +18 % rentals)
python build_model_inputs.py --data-dir ../lime-data/daily \
    --first-day 2026-10-02 --last-day 2026-10-16 --exclude 2026-10-06 --out model_input_lp/base
# what-if data set "disruption": 17.-26.10. (unrest, curfew, storms)
python build_model_inputs.py --data-dir ../lime-data/daily \
    --first-day 2026-10-17 --last-day 2026-10-26 --out model_input_lp/disruption
```

Writes `cells.csv`, `days.csv`, `s.csv`, `hourly.csv.gz` (r and d), `arcs.csv.gz`
(distances) and `summary.md` into the `--out` folder (`model_input_lp/` is not in git).

### Step 2: solve (`lp_model.py`)

```bash
python lp_model.py --area 18               # quick test: 18th arrondissement only, < 1 s
python lp_model.py                          # all Paris, weekdays, base case
```

All parameters are flags (`--day-type --p --h --v --Q --R --M --alpha --area --baseline
--fixed-returns --plan --input`). All Paris on weekdays: ~717k variables, 30 s to 2.5 min
on a laptop. The Gurobi log, run time and key numbers are printed; results go to
`reports/lp/<run name>/` (`summary.json`, `x.csv` = night plan, `cells.csv` = per cell).
The run name holds the data set (if not `base`), day type, p, R, alpha and flags,
e.g. `reports/lp/disruption_weekday_p3.5_R1.5_alpha0.5_baseline/`.

### Step 3: reproduce the D2 results

Run Step 1 first (both data sets). Each block below gives one table of the report;
the expected numbers follow. Lost trips per day, bikes moved etc. are printed at the
end of each run and stored in `summary.json`.

```bash
# (a) alpha calibration: baseline (x = 0) for 5 values of alpha, weekdays and weekends.
#     Compare "served trips per day" with the observed rentals/day in
#     model_input_lp/base/summary.md (weekday 56,464, weekend 57,248).
for a in 0 0.25 0.5 0.75 1; do python lp_model.py --baseline --alpha $a; done
for a in 0 0.25 0.5 0.75 1; do python lp_model.py --day-type weekend --baseline --alpha $a; done

# (b) baseline and optimum, weekdays (alpha 0.5 by default), lost-profit variants
python lp_model.py --baseline
for p in 3.5 0.1 7 20; do python lp_model.py --p $p; done

# (c) weekends (alpha 0.75 by default): baseline and optimum
python lp_model.py --day-type weekend --baseline
python lp_model.py --day-type weekend

# (d) radius test, weekdays, p = 3.50
for R in 1.0 1.5 2.0 3.0; do python lp_model.py --R $R; done

# (e) stress test: base-case weekday plan (x fixed) on the 6 disruption weekdays,
#     compared with no relocation on the same days
python lp_model.py --input model_input_lp/disruption --baseline
python lp_model.py --input model_input_lp/disruption --plan reports/lp/weekday_p3.5_R1.5_alpha0.5/x.csv

# (f) first model version (returns fixed) for comparison
python lp_model.py --fixed-returns --baseline
python lp_model.py --fixed-returns

# (g) map of the base-case weekday plan -> reports/lp/weekday_p3.5_R1.5_alpha0.5/net_flow_map.png
python plot_plan_map.py --run reports/lp/weekday_p3.5_R1.5_alpha0.5
```

"Avoided per bike" in the report = (lost trips of the baseline - lost trips of the
run) / bikes moved, both from `summary.json`.

Expected results (Gurobi 13.0.3; another version can return a different night plan
with the same cost, so bikes moved and distances may differ slightly, lost trips
and costs should not):

| Run | Lost trips/day | Bikes moved | Mean dist. | Cost/day (EUR) |
|---|---|---|---|---|
| (b) weekday baseline | 12,082 | 0 | - | 42,289 |
| (b) weekday p = 3.50 | 10,024 | 2,090 | 1.12 km | 36,685 |
| (b) weekday p = 0.10 | 12,082 | 0 | - | 1,208 |
| (b) weekday p = 7 | 9,947 | 2,555 | 1.19 km | 71,607 |
| (b) weekday p = 20 | 9,908 | 3,103 | 1.24 km | 200,576 |
| (c) weekend baseline | 11,427 | 0 | - | 39,996 |
| (c) weekend p = 3.50 | 8,854 | 2,458 | 0.95 km | 32,840 |
| (d) R = 1.0 / 1.5 / 2.0 / 3.0 km | 10,165 / 10,024 / 9,979 / 9,938 | 2,222 / 2,090 / 2,014 / 2,014 | 0.79 / 1.12 / 1.35 / 1.65 km | 37,220 / 36,685 / 36,514 / 36,421 |
| (e) disruption baseline | 9,953 | 0 | - | 34,835 |
| (e) disruption with base plan | 9,367 | 1,860 | 1.12 km | 34,211 |
| (f) fixed returns, baseline / optimum | 14,850 / 14,445 | 0 / 774 | - / 0.61 km | 51,975 / 51,116 |

(a) served trips per day for alpha 0 / 0.25 / 0.5 / 0.75 / 1: weekdays 52,218 /
54,657 / 56,263 / 57,255 / 57,907, weekends 51,554 / 54,372 / 56,182 / 57,320 / 58,034.
In (e), 230 of the 2,090 planned bikes are not moved because their origin cell has
fewer bikes at night on the disruption days (the run prints this).

## Politeness

User-Agent identifies the project and a contact e-mail. Polling interval
(120 s) is twice the feed's `ttl` (60 s); static feeds are fetched once.
