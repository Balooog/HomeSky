"""Shared persistence helpers for HomeSky data flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from loguru import logger

from utils.db import DatabaseManager
from utils.derived import compute_all_derived


EXCLUDED_KEYS: frozenset[str] = frozenset({
    "lastData",
    "raw",
    "macAddress",
    "macaddress",
})


@dataclass(slots=True)
class StorageResult:
    """Result metadata returned after persisting observations."""

    inserted: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]


def _coerce_utc(value: object) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:  # pragma: no cover - defensive guard
        return None
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _coerce_local(value: object) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:  # pragma: no cover - defensive guard
        return None
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    return ts


def _resolve_mac(candidate: dict, mac_hint: Optional[str]) -> Optional[str]:
    for key in ("station_mac", "mac", "macAddress", "macaddress", "device_mac"):
        value = candidate.get(key)
        if value:
            return str(value)
    return mac_hint


def _extract_timestamp_payload(payload: dict) -> Optional[pd.Timestamp]:
    for key in (
        "timestamp_utc",
        "obs_time_utc",
        "dateutc",
        "time_utc",
        "timestamp",
        "datetime",
    ):
        if key in payload and payload.get(key) not in (None, ""):
            ts = _coerce_utc(payload.get(key))
            if ts is not None:
                return ts
    epoch_ms = payload.get("epoch_ms")
    if epoch_ms:
        try:
            ts = pd.to_datetime(int(epoch_ms), unit="ms", utc=True)
            return ts
        except Exception:  # pragma: no cover - defensive guard
            return None
    epoch = payload.get("epoch")
    if epoch:
        try:
            ts = pd.to_datetime(int(epoch), unit="s", utc=True)
            return ts
        except Exception:  # pragma: no cover - defensive guard
            return None
    return None


def _extract_local_timestamp(payload: dict) -> Optional[pd.Timestamp]:
    for key in ("timestamp_local", "obs_time_local", "date", "datetime_local"):
        if key in payload and payload.get(key) not in (None, ""):
            local = _coerce_local(payload.get(key))
            if local is not None:
                return local
    return None


def canonicalize_records(
    records: Sequence[dict],
    *,
    mac_hint: Optional[str] = None,
) -> pd.DataFrame:
    """Return a canonical DataFrame suitable for persistence."""

    canonical_rows: List[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_payload = record.get("raw")
        if not isinstance(raw_payload, dict):
            raw_payload = {k: v for k, v in record.items() if k not in EXCLUDED_KEYS}
        combined: dict = {**raw_payload, **record}
        for key in EXCLUDED_KEYS:
            combined.pop(key, None)
        timestamp_utc = _extract_timestamp_payload({**raw_payload, **record})
        mac_value = _resolve_mac({**raw_payload, **record}, mac_hint)
        if timestamp_utc is None or not mac_value:
            continue
        timestamp_local = _extract_local_timestamp({**raw_payload, **record})
        epoch_ms = record.get("epoch_ms") or raw_payload.get("epoch_ms")
        if not epoch_ms:
            epoch = record.get("epoch") or raw_payload.get("epoch")
            if epoch:
                try:
                    epoch_ms = int(float(epoch) * 1000)
                except (TypeError, ValueError):
                    epoch_ms = None
        if not epoch_ms:
            epoch_ms = int(timestamp_utc.value // 1_000_000)
        epoch_seconds = record.get("epoch") or raw_payload.get("epoch")
        if not epoch_seconds:
            epoch_seconds = int(epoch_ms // 1000)
        canonical = dict(combined)
        canonical["station_mac"] = mac_value
        canonical["mac"] = mac_value
        canonical["timestamp_utc"] = timestamp_utc
        canonical["timestamp_local"] = timestamp_local
        canonical["epoch"] = epoch_seconds
        canonical["epoch_ms"] = epoch_ms
        canonical["raw"] = raw_payload
        canonical_rows.append(canonical)

    if not canonical_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(canonical_rows)
    frame = frame.dropna(subset=["timestamp_utc", "station_mac"])
    if frame.empty:
        return frame
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"])
    if frame.empty:
        return frame
    frame = frame.sort_values("timestamp_utc")
    frame = frame.drop_duplicates(subset=["station_mac", "epoch_ms"], keep="last")
    frame = frame.set_index("timestamp_utc")
    frame.index.name = "timestamp_utc"
    return frame


class StorageManager:
    """Coordinate persistence of weather observations."""

    def __init__(
        self,
        sqlite_path: Path | str,
        parquet_path: Path | str,
        *,
        config: Optional[dict] = None,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.parquet_path = Path(parquet_path)
        self.config = config or {}
        self._db = DatabaseManager(self.sqlite_path, self.parquet_path)

    @classmethod
    def from_config(cls, config: dict) -> "StorageManager":
        storage_cfg = config.get("storage", {})
        return cls(
            storage_cfg.get("sqlite_path", "./data/homesky.sqlite"),
            storage_cfg.get("parquet_path", "./data/homesky.parquet"),
            config=config,
        )

    @property
    def database(self) -> DatabaseManager:
        return self._db

    def upsert_canonical(
        self,
        canonical: pd.DataFrame,
        *,
        compute_derived: bool = True,
    ) -> StorageResult:
        if canonical.empty:
            return StorageResult(0, None, None)
        rows = canonical.reset_index().to_dict(orient="records")
        inserted = self._db.insert_observations(rows)
        if inserted and compute_derived:
            base = canonical.drop(columns=["raw"], errors="ignore")
            try:
                enriched = compute_all_derived(base, self.config)
            except Exception as exc:  # pragma: no cover - do not block persistence
                logger.warning("Failed to compute derived metrics: {}", exc)
            else:
                self._db.append_parquet(enriched)
        start = canonical.index.min() if inserted else None
        end = canonical.index.max() if inserted else None
        return StorageResult(inserted, start, end)

    def upsert_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        mac_hint: Optional[str] = None,
        compute_derived: bool = True,
    ) -> StorageResult:
        if frame.empty:
            return StorageResult(0, None, None)
        canonical = canonicalize_records(frame.to_dict(orient="records"), mac_hint=mac_hint)
        if canonical.empty:
            return StorageResult(0, None, None)
        return self.upsert_canonical(canonical, compute_derived=compute_derived)


__all__ = [
    "StorageManager",
    "StorageResult",
    "canonicalize_records",
]

