"""Long-running Ambient Weather ingest service."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

from utils.ambient import AmbientClient
from utils.db import DatabaseManager
from utils.derived import compute_all_derived

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


CONFIG_PATHS = [Path("config.toml"), Path("homesky/config.toml")]


def load_config(path: Path | None = None) -> Dict:
    candidates = [path] if path else CONFIG_PATHS
    for candidate in candidates:
        if candidate and candidate.exists():
            with candidate.open("rb") as fh:
                return tomllib.load(fh)
    raise FileNotFoundError(
        "config.toml not found. Run tools/ensure_config.ps1 or copy homesky/config.example.toml to homesky/config.toml and populate your credentials."
    )


def setup_logging(config: Dict) -> None:
    log_path = Path(config.get("ingest", {}).get("log_path", "./data/logs/ingest.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rotation_days = config.get("ingest", {}).get("log_rotate_days", 14)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        log_path,
        rotation=f"{rotation_days} days",
        retention=f"{rotation_days * 2} days",
        level="DEBUG",
        backtrace=True,
        enqueue=True,
    )
    logger.info("Logging initialized at %s", log_path)


def flatten_observations(records: List[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for device in records:
        last = device.get("lastData", {})
        if not last:
            continue
        row = {**last}
        row["mac"] = device.get("macAddress") or device.get("mac")
        if "dateutc" in row:
            dateutc = row["dateutc"]
            if isinstance(dateutc, (int, float)):
                row["obs_time_utc"] = pd.to_datetime(dateutc, unit="ms", errors="coerce", utc=True)
            else:
                row["obs_time_utc"] = pd.to_datetime(dateutc, errors="coerce", utc=True)
        if "date" in row:
            row["obs_time_local"] = pd.to_datetime(row["date"], errors="coerce")
        row["raw"] = last
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "obs_time_utc" in frame:
        frame = frame.sort_values("obs_time_utc")
        frame = frame.set_index("obs_time_utc")
    return frame


def deduplicate(df: pd.DataFrame, db: DatabaseManager, mac: str | None, enabled: bool) -> pd.DataFrame:
    if df.empty or not enabled:
        return df
    latest = db.fetch_last_timestamp(mac)
    if not latest:
        return df
    mask = df.index > pd.to_datetime(latest, utc=True)
    return df.loc[mask]


def detect_cadence_seconds(df: pd.DataFrame) -> int:
    if df.empty or "epoch" not in df:
        return 60
    epochs = pd.to_numeric(df["epoch"], errors="coerce").dropna()
    if epochs.empty:
        return 60
    diffs = epochs.diff().dropna()
    if diffs.empty:
        return 60
    cadence = int(diffs.median())
    return max(cadence, 60)


def drop_implausible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if "tempf" in df:
        mask &= df["tempf"].between(-80, 150)
    if "windspeedmph" in df:
        mask &= df["windspeedmph"].between(0, 200)
    if "humidity" in df:
        mask &= df["humidity"].between(0, 100)
    return df.loc[mask]


def ingest_once(config: Dict, db: DatabaseManager) -> int:
    client = AmbientClient(
        api_key=config["ambient"]["api_key"],
        application_key=config["ambient"]["application_key"],
        mac=config["ambient"].get("mac") or None,
        retries=config.get("ingest", {}).get("max_retries", 3),
        backoff=config.get("ingest", {}).get("retry_backoff_sec", 10),
    )
    records = client.get_device_data(limit=288)
    frame = flatten_observations(records)
    if config.get("ingest", {}).get("drop_implausible_values", True):
        frame = drop_implausible(frame)
    frame = deduplicate(
        frame,
        db,
        mac=config["ambient"].get("mac") or None,
        enabled=config.get("ingest", {}).get("deduplicate_by_timestamp", True),
    )
    if frame.empty:
        logger.info("No new observations to ingest.")
        return 0
    records_for_db = frame.reset_index().to_dict(orient="records")
    inserted = db.insert_observations(records_for_db)
    enriched = compute_all_derived(frame.drop(columns=["raw"], errors="ignore"), config)
    db.append_parquet(enriched)
    logger.info("Ingested %d new observation rows", inserted)
    return inserted


def run_loop(config: Dict, db: DatabaseManager) -> None:
    poll_seconds = 60
    while True:
        try:
            inserted = ingest_once(config, db)
        except KeyboardInterrupt:
            logger.info("Ingest loop interrupted by user")
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("Ingest failed: %s", exc)
        else:
            if inserted:
                poll_seconds = detect_cadence_seconds(db.read_dataframe(limit=10))
        logger.info("Sleeping for %s seconds", poll_seconds)
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HomeSky ingest service")
    parser.add_argument("--once", action="store_true", help="Run a single ingest cycle and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    setup_logging(config)
    logger.info("Starting HomeSky ingest service")
    storage = config.get("storage", {})
    db = DatabaseManager(
        sqlite_path=Path(storage.get("sqlite_path", "./data/homesky.sqlite")),
        parquet_path=Path(storage.get("parquet_path", "./data/homesky.parquet")),
    )
    try:
        if args.once:
            ingest_once(config, db)
        else:
            run_loop(config, db)
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting.")


if __name__ == "__main__":
    main()
