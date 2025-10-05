"""Database and storage helpers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from loguru import logger


OBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mac TEXT NOT NULL,
    obs_time_utc TEXT NOT NULL,
    obs_time_local TEXT,
    epoch INTEGER,
    epoch_ms INTEGER,
    data JSON NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(mac, obs_time_utc)
);
"""


@dataclass(slots=True)
class DatabaseManager:
    """Manage SQLite and Parquet persistence."""

    sqlite_path: Path
    parquet_path: Path

    def __post_init__(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_schema(self.sqlite_path)

    # -- SQLite helpers -------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def insert_observations(self, rows: Iterable[Dict]) -> int:
        """Insert observation rows, returning the count of new records."""

        rows = list(rows)
        if not rows:
            return 0
        inserted = 0
        with self._connect() as conn:
            cursor = conn.cursor()
            for row in rows:
                payload_source = row.get("lastData") or row.get("raw") or row
                mac = (
                    row.get("mac")
                    or row.get("macAddress")
                    or payload_source.get("mac")
                    or "unknown"
                )
                obs_time_utc = (
                    row.get("obs_time_utc")
                    or payload_source.get("obs_time_utc")
                    or payload_source.get("dateutc")
                )
                obs_time_local = (
                    row.get("obs_time_local")
                    or payload_source.get("obs_time_local")
                    or payload_source.get("date")
                )
                epoch = row.get("epoch") or payload_source.get("epoch")
                epoch_ms = row.get("epoch_ms") or payload_source.get("epoch_ms")
                if not epoch_ms and epoch:
                    try:
                        epoch_ms = int(float(epoch) * 1000)
                    except (TypeError, ValueError):
                        epoch_ms = None
                if not epoch_ms and obs_time_utc:
                    try:
                        parsed_epoch = pd.to_datetime(obs_time_utc, utc=True, errors="coerce")
                    except Exception:  # pragma: no cover - defensive
                        parsed_epoch = None
                    if parsed_epoch is not None and not pd.isna(parsed_epoch):
                        epoch_ms = int(parsed_epoch.value // 1_000_000)
                        epoch = epoch or int(parsed_epoch.timestamp())
                payload_source["epoch_ms"] = epoch_ms
                payload = json.dumps(payload_source)
                try:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO observations(mac, obs_time_utc, obs_time_local, epoch, epoch_ms, data)
                        VALUES(?,?,?,?,?,?)
                        """,
                        (
                            mac,
                            str(obs_time_utc),
                            str(obs_time_local),
                            epoch,
                            epoch_ms,
                            payload,
                        ),
                    )
                    inserted += cursor.rowcount
                except sqlite3.DatabaseError as exc:  # pragma: no cover
                    logger.exception("Failed to insert row: {}", exc)
            conn.commit()
        logger.debug("Inserted {} new rows into SQLite", inserted)
        return inserted

    def fetch_last_timestamp(self, mac: Optional[str] = None) -> Optional[str]:
        query = "SELECT obs_time_utc FROM observations"
        params: tuple = ()
        if mac:
            query += " WHERE mac = ?"
            params = (mac,)
        query += " ORDER BY obs_time_utc DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return row[0] if row else None

    def fetch_last_epoch_ms(self, mac: Optional[str] = None) -> Optional[int]:
        query = "SELECT epoch_ms FROM observations WHERE epoch_ms IS NOT NULL"
        params: tuple = ()
        if mac:
            query += " AND mac = ?"
            params = (mac,)
        query += " ORDER BY epoch_ms DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def read_dataframe(
        self, mac: Optional[str] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        query = "SELECT mac, obs_time_utc, obs_time_local, epoch, epoch_ms, data FROM observations"
        params: tuple = ()
        if mac:
            query += " WHERE mac = ?"
            params = (mac,)
        query += " ORDER BY obs_time_utc"
        if limit:
            query += " LIMIT ?"
            params = params + (limit,)
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            return df
        expanded = pd.json_normalize(df["data"].apply(json.loads))
        expanded.index = pd.to_datetime(df["obs_time_utc"], errors="coerce", utc=True)
        expanded.insert(0, "mac", df["mac"].values)
        expanded.insert(1, "obs_time_utc", pd.to_datetime(df["obs_time_utc"], errors="coerce", utc=True))
        expanded.insert(2, "obs_time_local", pd.to_datetime(df["obs_time_local"], errors="coerce"))
        expanded.insert(3, "epoch", df["epoch"].values)
        expanded.insert(4, "epoch_ms", df["epoch_ms"].values)
        return expanded

    # -- Parquet helpers ------------------------------------------------
    def append_parquet(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df_to_write = df.copy()
        df_to_write.reset_index(drop=True, inplace=True)
        write_kwargs = {
            "engine": "pyarrow",
            "compression": "snappy",
            "index": False,
        }
        if self.parquet_path.exists():
            existing = pd.read_parquet(self.parquet_path)
            combined = pd.concat([existing, df_to_write], ignore_index=True)
            combined.to_parquet(self.parquet_path, **write_kwargs)
        else:
            df_to_write.to_parquet(self.parquet_path, **write_kwargs)
        logger.debug("Appended {} rows to Parquet lake", len(df_to_write))


def ensure_schema(sqlite_path: Path | str) -> None:
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(OBS_TABLE_SQL)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(observations)")
        }
        if "epoch_ms" not in columns:
            conn.execute("ALTER TABLE observations ADD COLUMN epoch_ms INTEGER")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_observations_mac_epoch
            ON observations(mac, epoch_ms)
            WHERE epoch_ms IS NOT NULL
            """
        )
        conn.commit()


__all__ = ["DatabaseManager", "ensure_schema"]
