"""Offline import utilities for Ambient Weather CSV/XLSX exports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger

import ingest
from storage import StorageManager
from utils.timeparse import normalize_columns, to_epoch_ms

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

HEADER_ALIASES = {
    "roof_ws_2000_temperature": "temp_f",
    "temperature": "temp_f",
    "temperature_f": "temp_f",
    "temperaturef": "temp_f",
    "tempf": "temp_f",
    "feels_like": "feelslike_f",
    "feelslike": "feelslike_f",
    "dew_point": "dew_point_f",
    "dew_point_f": "dew_point_f",
    "dewpoint": "dew_point_f",
    "dewptf": "dew_point_f",
    "wind_speed": "wind_speed_mph",
    "wind_speed_mph": "wind_speed_mph",
    "windspeed": "wind_speed_mph",
    "windspeedmph": "wind_speed_mph",
    "wind_gust": "wind_gust_mph",
    "wind_gust_mph": "wind_gust_mph",
    "windgust": "wind_gust_mph",
    "windgustmph": "wind_gust_mph",
    "max_daily_gust": "max_gust_mph",
    "wind_direction": "wind_dir_deg",
    "wind_dir": "wind_dir_deg",
    "winddir": "wind_dir_deg",
    "avg_wind_direction_10_mins": "avg_wind_dir_deg",
    "avg_wind_speed_10_mins": "avg_wind_mph",
    "rain_rate": "rain_rate_in_hr",
    "rainrate": "rain_rate_in_hr",
    "event_rain": "rain_event_in",
    "daily_rain": "rain_day_in",
    "weekly_rain": "rain_week_in",
    "monthly_rain": "rain_month_in",
    "yearly_rain": "rain_year_in",
    "relative_pressure": "rel_pressure_inhg",
    "absolute_pressure": "abs_pressure_inhg",
    "pressure": "rel_pressure_inhg",
    "pressurein": "rel_pressure_inhg",
    "barometer_in": "rel_pressure_inhg",
    "ultra_violet_radiation_index": "uv_index",
    "uv": "uv_index",
    "solar_radiation": "solar_wm2",
    "indoor_temperature": "indoor_temp_f",
    "indoor_humidity": "indoor_humidity",
    "pm2_5_outdoor": "pm25_ugm3",
    "pm2_5_outdoor_24_hour_average": "pm25_24h_avg_ugm3",
    "pm2_5": "pm25_ugm3",
    "lightning_strikes_per_day": "lightning_day",
    "lightning_strikes_per_hour": "lightning_hour",
}

NUMERIC_COLUMNS = {
    "temp_f",
    "feelslike_f",
    "dew_point_f",
    "wind_speed_mph",
    "wind_gust_mph",
    "max_gust_mph",
    "wind_dir_deg",
    "avg_wind_dir_deg",
    "avg_wind_mph",
    "rain_rate_in_hr",
    "rain_event_in",
    "rain_day_in",
    "rain_week_in",
    "rain_month_in",
    "rain_year_in",
    "rel_pressure_inhg",
    "abs_pressure_inhg",
    "humidity",
    "indoor_humidity",
    "indoor_temp_f",
    "uv_index",
    "solar_wm2",
    "pm25_ugm3",
    "pm25_24h_avg_ugm3",
    "lightning_day",
    "lightning_hour",
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
    if isinstance(value, (int, float)) and not pd.isna(value):
        ts = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    else:
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


def _epoch_to_iso(epoch_ms: int) -> str:
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
    if suffix == ".csv":
        last_error: Optional[Exception] = None
        for encoding in ("utf-8-sig", "latin1"):
            try:
                return pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    encoding=encoding,
                    on_bad_lines="skip",
                )
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                if encoding == "latin1":
                    break
        if last_error:
            raise RuntimeError(f"Failed to read {path.name}: {last_error}") from last_error
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
    raise RuntimeError(f"Failed to read {path.name}: unknown error")


def _apply_header_aliases(df: pd.DataFrame) -> None:
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        alias = HEADER_ALIASES.get(column)
        if not alias or alias == column:
            continue
        if alias in df.columns:
            continue
        rename_map[column] = alias
    if rename_map:
        df.rename(columns=rename_map, inplace=True)


def _coerce_numeric_columns(df: pd.DataFrame) -> None:
    for column in NUMERIC_COLUMNS.intersection(df.columns):
        df[column] = pd.to_numeric(df[column], errors="coerce")


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


TIMESTAMP_ERROR_MESSAGE = (
    "No valid timestamps found. Make sure your file contains a 'dateutc', 'epoch', or 'date'+'time' column."
)


def _write_error_sample(config: Dict, raw: pd.DataFrame) -> Path:
    storage_cfg = config.get("storage", {})
    root = Path(storage_cfg.get("root_dir", "./data"))
    errors_dir = root / "import_errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    sample_path = errors_dir / "last_failed_sample.csv"
    try:
        raw.head(50).to_csv(sample_path, index=False)
    except Exception as exc:  # pragma: no cover - diagnostics helper
        logger.debug("[offline] Unable to write error sample: %s", exc)
    return sample_path


def _raise_timestamp_error(raw: pd.DataFrame, config: Dict) -> None:
    sample_path = _write_error_sample(config, raw)
    columns = ", ".join(str(col).strip() for col in raw.columns if str(col).strip())
    columns = columns or "(none)"
    logger.warning("[offline] No valid timestamps found. Columns detected: %s", columns)
    message = (
        f"{TIMESTAMP_ERROR_MESSAGE}\n\n"
        "Examples:\n- dateutc\n- epoch\n- date + time\n\n"
        f"Columns detected: {columns}\n"
        f"Sample saved to {sample_path.resolve()}"
    )
    raise ValueError(message)


def _prepare_dataframe(
    raw: pd.DataFrame,
    *,
    mac_hint: Optional[str],
    config: Dict,
    interactive: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    working = raw.copy()
    normalize_columns(working)
    _apply_header_aliases(working)
    try:
        epoch_ms_series = to_epoch_ms(working)
    except ValueError:
        _raise_timestamp_error(raw, config)
    if epoch_ms_series.empty:
        _raise_timestamp_error(raw, config)

    valid = working.loc[epoch_ms_series.index].copy()
    _apply_header_aliases(valid)
    _coerce_numeric_columns(valid)

    epoch_ms_int = epoch_ms_series.astype("int64")
    observed_at = pd.to_datetime(epoch_ms_int, unit="ms", utc=True)

    valid["observed_at"] = observed_at
    valid["timestamp_utc"] = observed_at
    valid["dateutc"] = observed_at
    valid["obs_time_utc"] = observed_at
    valid["epoch_ms"] = epoch_ms_int
    valid["epoch"] = (epoch_ms_int // 1000).astype("int64")

    source = epoch_ms_series.attrs.get("source", "timestamp")
    logger.info("[offline] Parsed %d timestamp row(s) using %s", len(epoch_ms_int), source)

    details: List[str] = []
    total_rows = len(raw)
    dropped = total_rows - len(valid)
    if dropped > 0:
        details.append(f"Dropped {dropped} row(s) without valid timestamps.")
    details.append(
        f"Range: {_epoch_to_iso(int(epoch_ms_int.min()))} – {_epoch_to_iso(int(epoch_ms_int.max()))}"
    )
    columns_used = epoch_ms_series.attrs.get("columns") or []
    if columns_used:
        details.append(f"Using columns: {', '.join(columns_used)}")
    for line in details:
        logger.debug("[offline] %s", line)

    station_series = _resolve_station_series(valid, mac_hint)
    if station_series is None:
        raise RuntimeError(
            "No station MAC detected. Add [ambient].mac to config.toml or include a station_mac column in the file."
        )
    valid["station_mac"] = station_series

    if "mac" in valid.columns:
        mac_series = _normalize_string_series(valid["mac"]).fillna(station_series)
    else:
        mac_series = station_series
    valid["mac"] = mac_series.astype("string")

    local_series = _extract_local_series(valid)
    if local_series is not None:
        valid["timestamp_local"] = local_series
    else:
        tz_name = str(config.get("timezone", {}).get("local_tz") or "UTC")
        try:
            valid["timestamp_local"] = observed_at.dt.tz_convert(tz_name)
        except Exception as exc:  # pragma: no cover - timezone fallback
            logger.debug("Unable to convert timestamps to %s: %s", tz_name, exc)

    valid = valid.dropna(subset=["epoch_ms", "timestamp_utc", "station_mac"]).reset_index(drop=True)
    return valid, details


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
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
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
