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
git clone -b data --single-branch https://github.com/leopoldgemi/lime-paris-rebalancing lime-data
python analyze_snapshots.py --data-dir ../lime-data/data
```

**Known limitations** (GitHub docs): the schedule is best-effort, minimum
interval 5 minutes, high delay at the top of the hour; scheduled workflows are
disabled after 60 days of no commits by a human. `analyze_snapshots.py`
reports the real median/max gap between snapshots.

### VM setup

```bash
# on a fresh Ubuntu 24.04 VM, as root
REPO=leopoldgemi/lime-paris-rebalancing bash deploy/setup_vm.sh
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

## Politeness

User-Agent identifies the project and a contact e-mail. Polling interval
(120 s) is twice the feed's `ttl` (60 s); static feeds are fetched once.
