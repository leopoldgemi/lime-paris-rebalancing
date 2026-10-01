#!/usr/bin/env python3
"""
Lime Paris GBFS collector (IEOR 4004 rebalancing project).

Polls the free_bike_status feed of Lime's public GBFS v2.2 feed for Paris and
appends every snapshot to a daily CSV. Static feeds (system_information,
vehicle_types, geofencing_zones if present, station_*) are saved once as JSON.

Dependencies: requests + Python standard library (Python >= 3.9).

Usage
-----
    python collect_lime.py                # loop forever, one snapshot every 2 min
    python collect_lime.py --once         # one snapshot, then exit (used by GitHub Actions)
    python collect_lime.py --interval 300 # custom interval in seconds
    python collect_lime.py --once --gzip-snapshots
                                          # write data/YYYY-MM-DD/HHMMSS.csv.gz instead of
                                          # appending to the daily CSV (keeps git history small)

All paths are relative to this file's directory, so the script runs unchanged
on any machine or CI runner. Data licence: Licence Ouverte 2.0 (Lime).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import logging.handlers
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DISCOVERY_URL = "https://data.lime.bike/api/partners/v2/gbfs/paris/gbfs.json"
USER_AGENT = (
    "ColumbiaIEOR4004-LimeRebalancingProject/1.0 "
    "(academic research; contact: leopoldv.scholz@gmail.com)"
)
DEFAULT_INTERVAL_S = 120
HTTP_TIMEOUT = (10, 60)  # (connect, read) seconds
DISCOVERY_REFRESH_S = 6 * 3600  # re-read gbfs.json every 6 h in case URLs change
MAX_CONSECUTIVE_FAILURES_BEFORE_REDISCOVER = 5

# Feeds saved once as JSON (if the discovery feed lists them).
STATIC_FEEDS = (
    "system_information",
    "vehicle_types",
    "geofencing_zones",
    "station_information",
    "station_status",
)

CSV_COLUMNS = [
    "snapshot_utc",          # ISO-8601 time the request was made (UTC)
    "feed_last_updated",     # unix ts reported by the feed
    "bike_id",
    "lat",
    "lon",
    "vehicle_type_id",
    "vehicle_type",          # non-standard extra field Lime sends ("e-bike", ...)
    "is_reserved",           # 0/1
    "is_disabled",           # 0/1
    "current_range_meters",
    "last_reported",         # unix ts per vehicle (may be missing)
]

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("lime_collector")


def rel(path: Path) -> str:
    """Path relative to the project dir for log output (falls back to absolute)."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def setup_logging(log_to_file: bool) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            LOG_DIR / "collector.log", maxBytes=5_000_000, backupCount=3
        )
        fh.setFormatter(fmt)
        log.addHandler(fh)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def fetch_json(session: requests.Session, url: str) -> dict:
    r = session.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def discover_feeds(session: requests.Session) -> dict[str, str]:
    """Read gbfs.json and return {feed_name: url} for the first language block."""
    doc = fetch_json(session, DISCOVERY_URL)
    languages = doc.get("data", {})
    if not languages:
        raise ValueError("discovery feed has no 'data' block")
    lang = "en" if "en" in languages else next(iter(languages))
    feeds = {f["name"]: f["url"] for f in languages[lang].get("feeds", [])}
    if "free_bike_status" not in feeds:
        raise ValueError(f"free_bike_status not in discovery feed: {sorted(feeds)}")
    log.info("discovery ok (lang=%s): %s", lang, ", ".join(sorted(feeds)))
    return feeds


def save_static_feeds(session: requests.Session, feeds: dict[str, str], out_dir: Path) -> None:
    """Save the static feeds once. Skips files that already exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in STATIC_FEEDS:
        url = feeds.get(name)
        target = out_dir / f"{name}.json"
        if url is None:
            if name == "geofencing_zones":
                log.info("geofencing_zones not offered by this feed (skipped)")
            continue
        if target.exists():
            continue
        try:
            doc = fetch_json(session, url)
            doc["_fetched_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            doc["_source_url"] = url
            target.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
            log.info("saved %s", rel(target))
        except (requests.RequestException, ValueError) as exc:
            log.warning("could not save static feed %s: %s", name, exc)


def bool01(v) -> int:
    return 1 if v else 0


def rows_from_feed(doc: dict, snapshot_utc: datetime) -> list[list]:
    bikes = doc.get("data", {}).get("bikes", [])
    ts = snapshot_utc.isoformat(timespec="seconds")
    lu = doc.get("last_updated")
    rows = []
    for b in bikes:
        rows.append([
            ts,
            lu,
            b.get("bike_id"),
            b.get("lat"),
            b.get("lon"),
            b.get("vehicle_type_id"),
            b.get("vehicle_type"),
            bool01(b.get("is_reserved")),
            bool01(b.get("is_disabled")),
            b.get("current_range_meters"),
            b.get("last_reported"),
        ])
    return rows


def write_snapshot(rows: list[list], snapshot_utc: datetime, data_dir: Path, gzip_snapshots: bool) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    day = snapshot_utc.strftime("%Y-%m-%d")
    if gzip_snapshots:
        day_dir = data_dir / day
        day_dir.mkdir(parents=True, exist_ok=True)
        target = day_dir / f"{snapshot_utc.strftime('%H%M%S')}.csv.gz"
        with gzip.open(target, "wt", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_COLUMNS)
            w.writerows(rows)
        return target
    target = data_dir / f"lime_paris_{day}.csv"
    new_file = not target.exists() or target.stat().st_size == 0
    with target.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(CSV_COLUMNS)
        w.writerows(rows)
    return target


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


class Collector:
    def __init__(self, data_dir: Path, interval_s: int, gzip_snapshots: bool):
        self.data_dir = data_dir
        self.interval_s = interval_s
        self.gzip_snapshots = gzip_snapshots
        self.session = make_session()
        self.feeds: dict[str, str] = {}
        self.discovered_at = 0.0
        self.last_feed_ts: int | None = None
        self.consecutive_failures = 0
        self.stop = False

    def ensure_discovery(self, force: bool = False) -> None:
        stale = time.monotonic() - self.discovered_at > DISCOVERY_REFRESH_S
        if force or not self.feeds or stale:
            self.feeds = discover_feeds(self.session)
            self.discovered_at = time.monotonic()
            save_static_feeds(self.session, self.feeds, self.data_dir / "reference")

    def poll_once(self) -> bool:
        """Fetch one snapshot. Returns True if a new snapshot was written."""
        self.ensure_discovery()
        snapshot_utc = datetime.now(timezone.utc)
        doc = fetch_json(self.session, self.feeds["free_bike_status"])
        feed_ts = doc.get("last_updated")
        ttl = doc.get("ttl")
        if feed_ts is not None and feed_ts == self.last_feed_ts:
            log.warning("feed last_updated unchanged (%s) - stale snapshot skipped", feed_ts)
            return False
        rows = rows_from_feed(doc, snapshot_utc)
        target = write_snapshot(rows, snapshot_utc, self.data_dir, self.gzip_snapshots)
        self.last_feed_ts = feed_ts
        log.info("snapshot %s: %d vehicles -> %s (feed ttl=%s)",
                 snapshot_utc.strftime("%H:%M:%SZ"), len(rows), rel(target), ttl)
        if isinstance(ttl, int) and ttl > self.interval_s:
            log.info("feed ttl %ss > interval %ss; polling more often is pointless", ttl, self.interval_s)
        return True

    def run_forever(self, duration_s: float | None = None) -> None:
        log.info("starting loop: every %ss, data_dir=%s, duration=%s", self.interval_s, self.data_dir,
                 f"{duration_s/60:.0f} min" if duration_s else "unlimited")
        deadline = time.monotonic() + duration_s if duration_s else None
        while not self.stop:
            if deadline and time.monotonic() >= deadline:
                log.info("duration reached")
                break
            t0 = time.monotonic()
            try:
                self.poll_once()
                self.consecutive_failures = 0
            except (requests.RequestException, ValueError, KeyError, OSError) as exc:
                self.consecutive_failures += 1
                log.error("poll failed (%d in a row): %s", self.consecutive_failures, exc)
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_REDISCOVER:
                    try:
                        self.ensure_discovery(force=True)
                    except Exception as exc2:  # noqa: BLE001
                        log.error("re-discovery failed: %s", exc2)
            except Exception:  # noqa: BLE001 - never die inside the loop
                log.exception("unexpected error in poll loop")
                self.consecutive_failures += 1
            # sleep the remainder of the interval, in small steps so Ctrl-C works
            elapsed = time.monotonic() - t0
            remaining = max(0.0, self.interval_s - elapsed)
            end = time.monotonic() + remaining
            while not self.stop and time.monotonic() < end:
                time.sleep(min(1.0, end - time.monotonic()))
        log.info("stopped")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S, help="seconds between polls (default 120)")
    p.add_argument("--once", action="store_true", help="take a single snapshot and exit")
    p.add_argument("--duration-minutes", type=float, default=None,
                   help="stop the loop after this many minutes (used by the GitHub Actions loop job)")
    p.add_argument("--data-dir", default="data", help="output directory, relative to this script (default: data)")
    p.add_argument("--gzip-snapshots", action="store_true",
                   help="one gzip file per snapshot (data/YYYY-MM-DD/HHMMSS.csv.gz) instead of daily CSV")
    p.add_argument("--no-log-file", action="store_true", help="log only to stderr")
    args = p.parse_args(argv)

    setup_logging(log_to_file=not args.no_log_file)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir

    c = Collector(data_dir=data_dir, interval_s=max(60, args.interval), gzip_snapshots=args.gzip_snapshots)

    def _stop(signum, frame):  # noqa: ARG001
        log.info("signal %s received, finishing current cycle", signum)
        c.stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if args.once:
        try:
            c.poll_once()
            return 0
        except Exception as exc:  # noqa: BLE001
            log.error("single poll failed: %s", exc)
            return 1
    c.run_forever(duration_s=args.duration_minutes * 60 if args.duration_minutes else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
