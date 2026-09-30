#!/usr/bin/env bash
# Creates the orphan "data" branch that the GitHub Actions workflow commits into.
# Run once from the repo root after the remote exists:  bash deploy/bootstrap_data_branch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if git ls-remote --exit-code --heads origin data >/dev/null 2>&1; then
  echo "remote branch 'data' already exists - nothing to do"; exit 0
fi
tmp=$(mktemp -d)
git worktree add --quiet --detach "$tmp"
pushd "$tmp" >/dev/null
git checkout --quiet --orphan data
git rm -rfq --cached . >/dev/null 2>&1 || true
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
mkdir -p data
python3 "$OLDPWD/collect_lime.py" --once --gzip-snapshots --no-log-file --data-dir "$tmp/data"
cat > README.md <<'MD'
# data branch

Snapshots collected by `.github/workflows/collect.yml` (see `main` branch).

- `data/reference/*.json`      static GBFS feeds, fetched once
- `data/YYYY-MM-DD/HHMMSS.csv.gz`  one free_bike_status snapshot per file (UTC)

Analyse with:  `git clone -b data --single-branch <repo> lime-data && python analyze_snapshots.py --data-dir ../lime-data/data`
Licence: Licence Ouverte 2.0 (Lime Paris GBFS).
MD
git add -A
git commit --quiet -m "bootstrap data branch"
git push --quiet -u origin data
popd >/dev/null
git worktree remove --force "$tmp"
echo "data branch created and pushed"
