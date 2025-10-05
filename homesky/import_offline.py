"""Offline import utilities for Ambient Weather CSV/XLSX exports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from loguru import logger

import ingest
from storage import StorageManager, canonicalize_records

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

CANONICAL_RENAMES = {
    "temperaturef": "tempf",
    "temperature_f": "tempf",
    "temp_f": "tempf",
    "temperature": "tempf",
    "humidity_percent": "humidity",
    "relative_humidity": "humidity",
    "wind_speed_mph": "windspeedmph",
    "wind_speed": "windspeedmph",
    "windgustmph": "windgustmph",
    "wind_gust_mph": "windgustmph",
    "wind_gust": "windgustmph",
    "winddir": "winddir",
    "wind_direction": "winddir",
    "pressurein": "baromin",
    "baromin": "baromin",
    "barometer_in": "baromin",
    "precipitationin": "rainin",
    "precip_in": "rainin",
    "rainfall_in": "rainin",
    "rain": "rainin",
    "dewpointf": "dewptf",
    "dew_point_f": "dewptf",
    "solar_radiation": "solarradiation",
    "uv": "uv",
}

TIMESTAMP_COLUMNS = (
    "dateutc",
    "timestamp_utc",
    "datetime_utc",
    "time_utc",
    "timestamp",
    "datetime",
)

LOCAL_TIME_COLUMNS = ("date", "time", "timestamp_local", "datetime_local")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path}")
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=0)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        lower = str(column).strip().lower()
        renamed[column] = CANONICAL_RENAMES.get(lower, lower)
    normalized = df.rename(columns=renamed)
    normalized.columns = [str(col).strip() for col in normalized.columns]
    return normalized


def _combine_date_time(df: pd.DataFrame) -> Optional[pd.Series]:
    if "date" in df.columns and "time" in df.columns:
        combined = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
        return pd.to_datetime(combined, utc=True, errors="coerce")
    return None


def _extract_timestamp_series(df: pd.DataFrame) -> pd.Series:
    for column in TIMESTAMP_COLUMNS:
        if column in df.columns:
            series = pd.to_datetime(df[column], utc=True, errors="coerce")
            if series.notna().any():
                return series
    combined = _combine_date_time(df)
    if combined is not None and combined.notna().any():
        return combined
    raise ValueError("Could not determine timestamp column in offline import")


def _extract_local_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for column in LOCAL_TIME_COLUMNS:
        if column in df.columns:
            series = pd.to_datetime(df[column], errors="coerce")
            if series.notna().any():
                return series
    return None


def _frame_to_records(
    df: pd.DataFrame,
    *,
    mac: Optional[str],
) -> List[dict]:
    normalized = _normalize_columns(df)
    timestamps = _extract_timestamp_series(normalized)
    local_series = _extract_local_series(normalized)
    records: List[dict] = []
    for idx, row in normalized.iterrows():
        timestamp = timestamps.iloc[idx] if idx < len(timestamps) else pd.NaT
        if pd.isna(timestamp):
            continue
        payload: Dict[str, object] = {}
        for column, value in row.items():
            if pd.isna(value):
                continue
            payload[column] = value
        payload["dateutc"] = timestamp.tz_convert(timezone.utc) if timestamp.tzinfo else timestamp.tz_localize(timezone.utc)
        if local_series is not None and idx < len(local_series):
            local_value = local_series.iloc[idx]
            if pd.notna(local_value):
                payload["date"] = local_value
        if mac:
            payload.setdefault("mac", mac)
        records.append(payload)
    return records


def _write_report(config: Dict, report: Dict) -> Path:
    storage_cfg = config.get("storage", {})
    root = Path(storage_cfg.get("root_dir", "./data"))
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir / f"import_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def import_files(
    paths: Iterable[Path],
    *,
    config: Dict,
    storage: StorageManager,
) -> Dict:
    mac = config.get("ambient", {}).get("mac")
    summaries: List[Dict] = []
    total_inserted = 0
    total_rows = 0
    canonical_rows = 0
    overall_start: Optional[pd.Timestamp] = None
    overall_end: Optional[pd.Timestamp] = None

    for path in paths:
        df = _read_table(path)
        records = _frame_to_records(df, mac=mac)
        total_rows += len(df)
        canonical = canonicalize_records(records, mac_hint=mac)
        canonical_rows += len(canonical)
        result = storage.upsert_canonical(canonical)
        total_inserted += result.inserted
        if result.start is not None:
            overall_start = result.start if overall_start is None else min(overall_start, result.start)
        if result.end is not None:
            overall_end = result.end if overall_end is None else max(overall_end, result.end)
        summaries.append(
            {
                "path": str(path),
                "rows_read": len(df),
                "rows_normalized": len(canonical),
                "rows_inserted": result.inserted,
                "time_start": result.start.isoformat() if result.start is not None else None,
                "time_end": result.end.isoformat() if result.end is not None else None,
            }
        )
        logger.info(
            "Imported %s rows from %s (inserted %s new rows)",
            len(df),
            path,
            result.inserted,
        )

    report = {
        "files": summaries,
        "total_rows": total_rows,
        "total_normalized": canonical_rows,
        "total_inserted": total_inserted,
        "time_start": overall_start.isoformat() if overall_start is not None else None,
        "time_end": overall_end.isoformat() if overall_end is not None else None,
    }
    report_path = _write_report(config, report)
    report["report_path"] = str(report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import offline Ambient/Weather Underground exports")
    parser.add_argument("paths", nargs="+", type=Path, help="CSV/XLSX files to import")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ingest.load_config()
    storage = ingest.get_storage_manager(config)
    report = import_files(args.paths, config=config, storage=storage)
    logger.info("Offline import complete: %s rows inserted", report["total_inserted"])
    logger.info("Import report saved to %s", report["report_path"])


if __name__ == "__main__":
    main()

