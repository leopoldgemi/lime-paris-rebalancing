#!/usr/bin/env python3
"""
Merge the per-snapshot files of one day (data/YYYY-MM-DD/HHMMSS.csv.gz, as
written by GitHub Actions) into a single daily file lime_paris_YYYY-MM-DD.csv.gz
with one header row. Standard library only.

    python merge_day.py 2026-10-01                       # -> data/lime_paris_2026-10-01.csv.gz
    python merge_day.py 2026-10-01 --data-dir ../lime-data/data --out merged/
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def merge_day(day_dir: Path, target: Path) -> int:
    files = sorted(day_dir.glob("*.csv.gz"))
    if not files:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    header = None
    n = 0
    with gzip.open(target, "wt", encoding="utf-8", newline="") as out:
        for f in files:
            with gzip.open(f, "rt", encoding="utf-8", newline="") as fh:
                first = fh.readline()
                if header is None:
                    header = first
                    out.write(header)
                elif first != header:
                    print(f"warning: header mismatch in {f.name}, skipped", file=sys.stderr)
                    continue
                for line in fh:
                    out.write(line)
                n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("day", help="YYYY-MM-DD (UTC)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=None, help="output directory (default: same as --data-dir)")
    a = p.parse_args(argv)
    data_dir = Path(a.data_dir)
    data_dir = data_dir if data_dir.is_absolute() else BASE_DIR / data_dir
    out_dir = Path(a.out) if a.out else data_dir
    out_dir = out_dir if out_dir.is_absolute() else BASE_DIR / out_dir
    target = out_dir / f"lime_paris_{a.day}.csv.gz"
    n = merge_day(data_dir / a.day, target)
    if n == 0:
        print(f"no snapshots found in {data_dir / a.day}", file=sys.stderr)
        return 1
    print(f"{n} snapshots -> {target} ({target.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
