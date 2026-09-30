# data branch

Snapshots collected by `.github/workflows/collect.yml` (see `main` branch).

- `data/reference/*.json`      static GBFS feeds, fetched once
- `data/YYYY-MM-DD/HHMMSS.csv.gz`  one free_bike_status snapshot per file (UTC)

Analyse with:  `git clone -b data --single-branch <repo> lime-data && python analyze_snapshots.py --data-dir ../lime-data/data`
Licence: Licence Ouverte 2.0 (Lime Paris GBFS).
