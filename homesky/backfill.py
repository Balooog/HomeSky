"""Historical Ambient Weather backfill utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from storage import StorageManager, StorageResult, canonicalize_records
from utils.ambient import AmbientClient


@dataclass(slots=True)
class BackfillState:
    start: pd.Timestamp
    end: pd.Timestamp
    cursor: pd.Timestamp
    mac: str


def _checkpoint_path(config: Dict) -> Path:
    storage_cfg = config.get("storage", {})
    root = Path(storage_cfg.get("root_dir", "./data"))
    return root / "state" / "backfill.json"


def _load_checkpoint(path: Path) -> Optional[BackfillState]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive guard
        logger.warning("Backfill checkpoint at %s is corrupt; ignoring", path)
        return None
    try:
        start = pd.to_datetime(payload["start"], utc=True)
        end = pd.to_datetime(payload["end"], utc=True)
        cursor = pd.to_datetime(payload["cursor"], utc=True)
        mac = str(payload["mac"])
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Backfill checkpoint missing fields: %s", exc)
        return None
    return BackfillState(start=start, end=end, cursor=cursor, mac=mac)


def _save_checkpoint(path: Path, state: BackfillState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "start": state.start.isoformat(),
        "end": state.end.isoformat(),
        "cursor": state.cursor.isoformat(),
        "mac": state.mac,
    }
    path.write_text(json.dumps(payload, indent=2))


def _clear_checkpoint(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:  # pragma: no cover - best-effort cleanup
        logger.warning("Unable to remove backfill checkpoint: %s", exc)


def _apply_filters(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if df.empty:
        return df
    ingest_cfg = config.get("ingest", {})
    if not ingest_cfg.get("drop_implausible_values", True):
        return df
    mask = pd.Series(True, index=df.index)
    if "tempf" in df:
        mask &= df["tempf"].between(-80, 150)
    if "windspeedmph" in df:
        mask &= df["windspeedmph"].between(0, 200)
    if "humidity" in df:
        mask &= df["humidity"].between(0, 100)
    return df.loc[mask]


def backfill_range(
    *,
    config: Dict,
    storage: StorageManager,
    start_dt: datetime | pd.Timestamp,
    end_dt: datetime | pd.Timestamp,
    mac: str,
    window_minutes: int = 24 * 60,
    limit_per_call: int = 288,
) -> StorageResult:
    if limit_per_call <= 0:
        raise ValueError("limit_per_call must be positive")
    start_ts = pd.to_datetime(start_dt, utc=True)
    end_ts = pd.to_datetime(end_dt, utc=True)
    if start_ts >= end_ts:
        return StorageResult(0, None, None)
    checkpoint_file = _checkpoint_path(config)
    checkpoint = _load_checkpoint(checkpoint_file)
    if checkpoint and checkpoint.mac == mac:
        start_ts = max(start_ts, checkpoint.start)
        end_ts = min(end_ts, checkpoint.end)
        cursor = checkpoint.cursor
    else:
        cursor = end_ts
    client = AmbientClient(
        api_key=config["ambient"]["api_key"],
        application_key=config["ambient"]["application_key"],
        mac=mac,
    )
    total_inserted = 0
    overall_start: Optional[pd.Timestamp] = None
    overall_end: Optional[pd.Timestamp] = None
    window_delta = pd.Timedelta(minutes=max(window_minutes, 1))

    while cursor > start_ts:
        window_end = cursor
        window_start = max(start_ts, cursor - window_delta)
        logger.info(
            "[Backfill] Requesting history for %s between %s and %s",
            mac,
            window_start,
            window_end,
        )
        records = client.get_device_data(mac=mac, end_dt=window_end.to_pydatetime(), limit=limit_per_call)
        if not records:
            cursor = window_start
            _save_checkpoint(
                checkpoint_file,
                BackfillState(start=start_ts, end=end_ts, cursor=cursor, mac=mac),
            )
            if window_start == start_ts:
                break
            continue
        canonical = canonicalize_records(records, mac_hint=mac)
        if canonical.empty:
            cursor = window_start
            continue
        canonical = canonical.loc[(canonical.index >= window_start) & (canonical.index <= window_end)]
        canonical = _apply_filters(canonical, config)
        if canonical.empty:
            cursor = window_start
            continue
        result = storage.upsert_canonical(canonical)
        total_inserted += result.inserted
        if result.start is not None:
            overall_start = result.start if overall_start is None else min(overall_start, result.start)
        if result.end is not None:
            overall_end = result.end if overall_end is None else max(overall_end, result.end)
        logger.info(
            "[Backfill] Inserted %s rows covering %s – %s",
            result.inserted,
            result.start,
            result.end,
        )
        oldest = canonical.index.min()
        if oldest <= start_ts:
            cursor = start_ts
        else:
            cursor = oldest - pd.Timedelta(seconds=1)
        _save_checkpoint(
            checkpoint_file,
            BackfillState(start=start_ts, end=end_ts, cursor=cursor, mac=mac),
        )

    if cursor <= start_ts:
        _clear_checkpoint(checkpoint_file)

    return StorageResult(total_inserted, overall_start, overall_end)


__all__ = ["backfill_range"]

