"""Offline import utilities for Ambient Weather CSV/XLSX exports."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger

import ingest
from storage import StorageManager
from utils.timeparse import OfflineTimestampError, parse_offline_timestamp

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

CANONICAL_RENAMES = {
    "temperaturef": "temp_f",
    "temperature_f": "temp_f",
    "tempf": "temp_f",
    "temperature": "temp_f",
    "humidity_percent": "humidity",
    "relative_humidity": "humidity",
    "wind_speed_mph": "wind_speed_mph",
    "wind_speed": "wind_speed_mph",
    "windspeedmph": "wind_speed_mph",
    "wind_gust_mph": "wind_gust_mph",
    "windgustmph": "wind_gust_mph",
    "wind_gust": "wind_gust_mph",
    "winddir": "wind_dir",
    "wind_direction": "wind_dir",
    "pressurein": "barom_in",
    "barometer_in": "barom_in",
    "precipitationin": "rain_in",
    "precip_in": "rain_in",
    "rainfall_in": "rain_in",
    "rain": "rain_in",
    "dewpointf": "dew_point_f",
    "dew_point_f": "dew_point_f",
    "dewptf": "dew_point_f",
    "solar_radiation": "solar_radiation",
    "uv": "uv",
    "station mac": "station_mac",
    "stationmac": "station_mac",
    "mac address": "mac",
    "macaddress": "mac",
    "device mac": "device_mac",
    "epochms": "epoch_ms",
}

MAC_COLUMN_NAMES = ("station_mac", "mac", "macaddress", "device_mac")
LOCAL_TIME_CANDIDATES = (
    "timestamp_local",
    "obs_time_local",
    "datetime_local",
    "date_local",
)


def _isoformat_utc(value: object) -> Optional[str]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    epoch_ms = int(dt.timestamp() * 1000)
    return datetime.utcfromtimestamp(epoch_ms / 1000).isoformat() + "Z"


def _normalize_name(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _match_columns(columns: Iterable[object], names: Iterable[str]) -> List[str]:
    normalized: Dict[str, List[str]] = {}
    for original in columns:
        normalized.setdefault(_normalize_name(original), []).append(str(original))
    resolved: List[str] = []
    for name in names:
        key = _normalize_name(name)
        for column in normalized.get(key, []):
            if column not in resolved:
                resolved.append(column)
    return resolved


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path}")
    try:
        if suffix == ".csv":
            return pd.read_csv(path, encoding="utf-8-sig")
        engines: List[str] = ["openpyxl"]
        if suffix == ".xls":
            engines.append("xlrd")
        last_error: Optional[Exception] = None
        for engine in engines:
            try:
                return pd.read_excel(path, engine=engine)
            except ImportError as exc:
                last_error = exc
                continue
            except ValueError as exc:
                last_error = exc
                if "Excel file format cannot be determined" in str(exc) and engine == "openpyxl" and suffix == ".xls":
                    continue
                break
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = exc
                break
        if last_error:
            if isinstance(last_error, ImportError):
                raise RuntimeError(
                    "Missing Excel reader. Install 'openpyxl>=3.1' (and 'xlrd' for legacy .xls) to import Excel files."
                ) from last_error
            raise RuntimeError(f"Failed to read {path.name}: {last_error}") from last_error
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Failed to read {path.name}: {exc}") from exc
    return pd.read_excel(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())
        normalized = re.sub(r"__+", "_", normalized).strip("_")
        normalized = CANONICAL_RENAMES.get(normalized, normalized)
        renamed[column] = normalized
    normalized = df.rename(columns=renamed)
    normalized.columns = [str(col).strip() for col in normalized.columns]
    return normalized


def _normalize_string_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.where(normalized != "", pd.NA)


def _resolve_station_series(df: pd.DataFrame, mac_hint: Optional[str]) -> Optional[pd.Series]:
    candidate_columns = _match_columns(df.columns, MAC_COLUMN_NAMES)
    for column in candidate_columns:
        series = _normalize_string_series(df[column])
        if series.notna().any():
            if mac_hint:
                series = series.fillna(mac_hint)
            return series.astype("string")
    if mac_hint:
        return pd.Series([mac_hint] * len(df), index=df.index, dtype="string")
    return None


def _extract_local_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for column in LOCAL_TIME_CANDIDATES:
        matches = _match_columns(df.columns, [column])
        if not matches:
            continue
        series = pd.to_datetime(df[matches[0]], errors="coerce")
        if series.notna().any():
            return series
    return None


def _prepare_dataframe(
    raw: pd.DataFrame,
    *,
    mac_hint: Optional[str],
    config: Dict,
    interactive: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    normalized = _normalize_columns(raw)
    try:
        timestamp_result = parse_offline_timestamp(normalized, config, interactive=interactive)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    except OfflineTimestampError as exc:
        detail_lines = "\n".join(f"- {line}" for line in exc.details) if exc.details else "(no candidate columns)"
        raise RuntimeError(
            "Unable to determine a timestamp column.\n" + detail_lines
        ) from exc

    observed_at = timestamp_result.timestamp_utc.reindex(normalized.index)
    epoch_ms = timestamp_result.epoch_ms.reindex(normalized.index)
    epoch_ms_int = epoch_ms.astype("Int64")

    normalized["observed_at"] = observed_at
    normalized["timestamp_utc"] = observed_at
    normalized["dateutc"] = observed_at
    normalized["obs_time_utc"] = observed_at
    normalized["epoch_ms"] = epoch_ms_int
    normalized["epoch"] = (epoch_ms_int // 1000).astype("Int64")

    for line in timestamp_result.details:
        logger.debug("[offline] %s", line)

    station_series = _resolve_station_series(normalized, mac_hint)
    if station_series is None:
        raise RuntimeError(
            "No station MAC detected. Add [ambient].mac to config.toml or include a station_mac column in the file."
        )
    normalized["station_mac"] = station_series

    if "mac" in normalized.columns:
        mac_series = _normalize_string_series(normalized["mac"]).fillna(station_series)
    else:
        mac_series = station_series
    normalized["mac"] = mac_series.astype("string")

    local_series = _extract_local_series(normalized)
    if local_series is not None:
        normalized["timestamp_local"] = local_series
    else:
        tz_name = str(config.get("timezone", {}).get("local_tz") or "UTC")
        try:
            normalized["timestamp_local"] = observed_at.dt.tz_convert(tz_name)
        except Exception as exc:  # pragma: no cover - timezone fallback
            logger.debug("Unable to convert timestamps to %s: %s", tz_name, exc)

    valid = normalized.dropna(subset=["epoch_ms", "timestamp_utc", "station_mac"])
    valid = valid.reset_index(drop=True)
    return valid, timestamp_result.details


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
    interactive: bool = False,
) -> Dict:
    mac = config.get("ambient", {}).get("mac")
    summaries: List[Dict] = []
    total_inserted = 0
    total_rows = 0
    total_after_dedup = 0
    total_duplicates = 0
    overall_start: Optional[pd.Timestamp] = None
    overall_end: Optional[pd.Timestamp] = None

    for path in paths:
        try:
            df = _read_table(path)
        except Exception as exc:
            raise RuntimeError(f"{path.name}: {exc}") from exc
        try:
            prepared, timestamp_details = _prepare_dataframe(
                df,
                mac_hint=mac,
                config=config,
                interactive=interactive,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{path.name}: {exc}") from exc
        total_rows += len(df)
        if prepared.empty:
            summaries.append(
                {
                    "path": str(path),
                    "rows_read": len(df),
                    "rows_valid": 0,
                    "rows_after_dedup": 0,
                    "rows_inserted": 0,
                    "rows_duplicates": 0,
                    "timestamp_details": timestamp_details,
                }
            )
            logger.warning("No valid rows detected in %s", path.name)
            continue

        deduped = prepared.drop_duplicates(subset=["station_mac", "epoch_ms"])
        dropped_duplicates = len(prepared) - len(deduped)
        result = storage.upsert_dataframe(deduped, mac_hint=mac)
        inserted = result.inserted
        duplicates_in_db = len(deduped) - inserted

        total_after_dedup += len(deduped)
        total_inserted += inserted
        total_duplicates += dropped_duplicates + duplicates_in_db

        if result.start is not None:
            overall_start = result.start if overall_start is None else min(overall_start, result.start)
        if result.end is not None:
            overall_end = result.end if overall_end is None else max(overall_end, result.end)

        summaries.append(
            {
                "path": str(path),
                "rows_read": len(df),
                "rows_valid": len(prepared),
                "rows_after_dedup": len(deduped),
                "rows_inserted": inserted,
                "rows_duplicates": dropped_duplicates + duplicates_in_db,
                "timestamp_details": timestamp_details,
                "time_start": _isoformat_utc(result.start),
                "time_end": _isoformat_utc(result.end),
            }
        )

        logger.info(
            "Imported %s rows from %s (%s new, %s duplicates)",
            len(df),
            path.name,
            inserted,
            dropped_duplicates + duplicates_in_db,
        )

    report = {
        "files": summaries,
        "total_rows": total_rows,
        "total_after_dedup": total_after_dedup,
        "total_inserted": total_inserted,
        "total_duplicates": total_duplicates,
        "time_start": _isoformat_utc(overall_start),
        "time_end": _isoformat_utc(overall_end),
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
    report = import_files(args.paths, config=config, storage=storage, interactive=False)
    logger.info("Offline import complete: %s rows inserted", report["total_inserted"])
    logger.info("Import report saved to %s", report["report_path"])


if __name__ == "__main__":
    main()
