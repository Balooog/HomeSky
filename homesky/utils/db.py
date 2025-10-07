"""Database and storage helpers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from homesky.utils.config import get_station_tz
from homesky.utils.logging_setup import get_logger

log = get_logger("db")


def _json_default(value: object) -> Optional[object]:
    """Return a JSON-serializable representation for database payloads."""

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        ts = value
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        ts = value
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        return ts.isoformat().replace("+00:00", "Z")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - fallback to string
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - fallback to string
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fallback to string
            pass
    if value is None:
        return None
    return str(value)


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


def parse_obs_times(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize observation timestamps and drop duplicate local readings."""

    if "obs_time_local" not in df.columns:
        return df

    zone = get_station_tz()
    working = df.copy()
    try:
        ts = pd.to_datetime(working["obs_time_local"], errors="coerce", utc=False)

        tz_attr = getattr(ts.dt, "tz", None)
        if tz_attr is None:
            ts = ts.dt.tz_localize(
                zone,
                nonexistent="shift_forward",
                ambiguous="NaT",
            )
        else:
            ts = ts.dt.tz_convert(zone)

        working["obs_time_local"] = ts
        working = working.drop_duplicates(subset=["obs_time_local"])
    except Exception as exc:  # pragma: no cover - defensive logging only
        log.exception("parse_obs_times failed: %s", exc)
    return working


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
                payload = json.dumps(payload_source, default=_json_default)
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
                    log.exception("Failed to insert row: %s", exc)
            conn.commit()
        log.debug("Inserted %s new rows into SQLite", inserted)
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
        self,
        mac: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        local_tz: str | None = None,
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
        df = parse_obs_times(df)
        expanded = pd.json_normalize(df["data"].apply(json.loads))
        epoch_ms_series = pd.to_numeric(df["epoch_ms"], errors="coerce")
        observed_at = pd.to_datetime(epoch_ms_series, unit="ms", errors="coerce", utc=True)
        valid_mask = observed_at.notna()
        if not bool(valid_mask.any()):
            return expanded.iloc[0:0]
        expanded = expanded.loc[valid_mask.values].copy()
        observed_at = observed_at.loc[valid_mask]
        epoch_ms_int = epoch_ms_series.loc[valid_mask].round().astype("int64")
        epoch_series = pd.to_numeric(df["epoch"], errors="coerce").loc[valid_mask]
        obs_time_local = pd.to_datetime(
            df["obs_time_local"], errors="coerce"
        ).loc[valid_mask]

        def _ensure_column(name: str, data: pd.Series, position: int | None = None) -> None:
            if name in expanded.columns:
                expanded[name] = data
                return
            if position is None:
                expanded[name] = data
            else:
                if name not in expanded.columns:
                    expanded.insert(position, name, data)

        _ensure_column("mac", df["mac"].loc[valid_mask], position=0)
        _ensure_column("observed_at", observed_at, position=1)
        _ensure_column("obs_time_utc", observed_at, position=2)
        _ensure_column("obs_time_local", obs_time_local, position=3)
        _ensure_column("epoch", epoch_series, position=4)
        _ensure_column("epoch_ms", epoch_ms_int, position=5)
        if "s_time_local" not in expanded.columns:
            expanded.insert(3, "s_time_local", obs_time_local)
        else:
            expanded["s_time_local"] = obs_time_local
        expanded["epoch_ms"] = expanded["epoch_ms"].astype("int64")
        expanded.index = observed_at
        expanded.index.name = "s_time_utc"

        tz_name = local_tz or "UTC"
        try:
            zone = ZoneInfo(tz_name)
        except Exception:  # pragma: no cover - fallback to UTC
            zone = ZoneInfo("UTC")

        if "s_time_local" in expanded.columns:
            local_series = pd.to_datetime(expanded["s_time_local"], errors="coerce")
        else:
            local_series = pd.to_datetime(expanded.index, errors="coerce")
        if getattr(local_series.dtype, "tz", None) is None:
            localized = local_series.dt.tz_localize(zone, ambiguous="NaT", nonexistent="shift_forward")
            if localized.isna().any():
                fallback = pd.Series(expanded.index, index=expanded.index).dt.tz_convert(zone)
                localized = localized.where(~localized.isna(), fallback)
            local_series = localized
        else:
            try:
                local_series = local_series.dt.tz_convert(zone)
            except Exception:
                local_series = pd.Series(expanded.index, index=expanded.index).dt.tz_convert(zone)
        expanded["s_time_local"] = local_series.values
        expanded["s_time_utc"] = pd.Series(expanded.index, index=expanded.index)

        if "observed_at" in expanded.columns:
            expanded = expanded.drop(columns=["observed_at"])

        if "s_time_local" in expanded.columns and "s_time_local" in expanded.index.names:
            expanded = expanded.reset_index(drop=False)
        if "s_time_local" not in expanded.columns:
            expanded = expanded.reset_index(drop=False)
            if "s_time_local" not in expanded.columns and "epoch_ms" in expanded.columns:
                timestamps = pd.to_datetime(expanded["epoch_ms"], unit="ms", errors="coerce", utc=True)
                expanded["s_time_local"] = timestamps.dt.tz_convert(zone)
        expanded = expanded.drop_duplicates(subset=["s_time_local", "mac"], keep="last")
        expanded = expanded.sort_values("s_time_local").set_index("s_time_local")
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
        log.debug("Appended %s rows to Parquet lake", len(df_to_write))


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
