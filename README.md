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

`bike_id` appears stable across snapshots, so trips can be reconstructed as
position changes of the same id between consecutive snapshots.

## Files

| File | Purpose |
|---|---|
| `collect_lime.py` | collector (requests + stdlib only). Reads feed URLs from the discovery feed, polls `free_bike_status` every 2 min, appends to `data/lime_paris_YYYY-MM-DD.csv`, saves static feeds once to `data/reference/*.json`. Network errors are logged, loop continues. |
| `analyze_snapshots.py` | first overview (pandas, matplotlib optional): fleet per snapshot, e-bike/scooter split, reserved/disabled/low-battery shares, hourly profile, 500 m grid density, position changes (trip proxy). Output in `reports/`. |
| `.github/workflows/collect.yml` | GitHub Actions: one snapshot every 5 min, committed to the `data` branch. |
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

### GitHub Actions setup

```bash
gh repo create lime-paris-rebalancing --public --source . --push
bash deploy/bootstrap_data_branch.sh   # creates orphan "data" branch + reference JSONs
gh workflow run collect-lime-paris      # first manual run; then every 5 min
```

Get the data for analysis:

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

## Politeness

User-Agent identifies the project and a contact e-mail. Polling interval
(120 s) is twice the feed's `ttl` (60 s); static feeds are fetched once.
