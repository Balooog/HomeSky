"""Helpers for parsing offline timestamp columns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from loguru import logger

__all__ = [
    "OfflineTimestampError",
    "OfflineTimestampResult",
    "parse_offline_timestamp",
    "to_epoch_ms",
]


class OfflineTimestampError(RuntimeError):
    """Raised when offline timestamp detection fails."""

    def __init__(self, message: str, details: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.details = list(details or [])


@dataclass(slots=True)
class OfflineTimestampResult:
    """Result describing resolved timestamps for offline imports."""

    epoch_ms: pd.Series
    timestamp_utc: pd.Series
    source: str
    columns: Tuple[str, ...]
    non_null: int
    details: List[str]


_COLUMN_RENAMES: Dict[str, str] = {
    "date_time": "datetime",
    "date_utc": "dateutc",
    "epochms": "epoch_ms",
    "macaddress": "mac",
    "station mac": "station_mac",
    "stationmac": "station_mac",
    "device mac": "device_mac",
    "tempf": "temp_f",
    "dewptf": "dew_point_f",
    "windspeedmph": "wind_speed_mph",
    "windgustmph": "wind_gust_mph",
    "winddir": "wind_dir",
    "solarradiation": "solar_radiation",
    "rainin": "rain_in",
    "dailyrainin": "daily_rain_in",
    "hourlyrainin": "hourly_rain_in",
}

_CANDIDATES_DEFAULT = [
    "epoch_ms",
    "epoch",
    "dateutc",
    "datetime",
    "observed_at",
    "time",
    "date",
]


def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower())
        normalized = re.sub(r"__+", "_", normalized).strip("_")
        normalized = _COLUMN_RENAMES.get(normalized, normalized)
        rename_map[str(column)] = normalized
    df.rename(columns=rename_map, inplace=True)
    return rename_map


def _finalize_epoch_series(values: pd.Series, *, source: str, columns: Tuple[str, ...]) -> pd.Series | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.empty:
        return None
    numeric = numeric.where(numeric > 0)
    numeric = numeric.dropna()
    if numeric.empty:
        return None
    epoch = numeric.round().astype("int64")
    epoch.attrs["source"] = source
    epoch.attrs["columns"] = list(columns)
    return epoch


def _finalize_from_datetime(series: pd.Series, *, source: str, columns: Tuple[str, ...]) -> pd.Series | None:
    if series.empty:
        return None
    if not pd.api.types.is_datetime64_any_dtype(series):
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    if getattr(valid.dt, "tz", None) is None:
        valid = valid.dt.tz_localize("UTC")
    else:
        valid = valid.dt.tz_convert("UTC")
    epoch = (valid.astype("int64") // 1_000_000).astype("int64")
    epoch.attrs["source"] = source
    epoch.attrs["columns"] = list(columns)
    return epoch


def _excel_serial_to_datetime(series: pd.Series) -> pd.Series | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return None
    dt = pd.to_datetime(
        numeric.astype("float64"),
        unit="D",
        origin="1899-12-30",
        errors="coerce",
        utc=True,
    )
    if dt.dropna().empty:
        return None
    return dt


def _combine_date_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    date_series = df[date_col].astype("string").str.strip()
    time_series = df[time_col].astype("string").str.strip()
    combined = date_series.str.cat(time_series, sep=" ", na_rep=None)
    combined = combined.where(date_series.notna() & time_series.notna())
    return combined


def _parse_time_only(series: pd.Series) -> pd.Series | None:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dropna().empty:
        return None
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        return None
    valid = parsed.dropna()
    if valid.empty:
        return None
    normalized_days = valid.dt.normalize().nunique()
    if normalized_days > 1:
        return None
    localized = valid.dt.tz_localize("UTC")
    return localized


def to_epoch_ms(df: pd.DataFrame, *, candidate_cols: List[str] | None = None) -> pd.Series:
    """Coerce a dataframe's timestamp columns into epoch milliseconds."""

    _normalize_columns(df)
    candidates = candidate_cols or list(_CANDIDATES_DEFAULT)

    if "epoch_ms" in df.columns and "epoch_ms" in candidates:
        epoch = _finalize_epoch_series(df["epoch_ms"], source="epoch_ms", columns=("epoch_ms",))
        if epoch is not None:
            return epoch

    if "epoch" in df.columns and "epoch" in candidates:
        numeric = pd.to_numeric(df["epoch"], errors="coerce")
        if not numeric.dropna().empty:
            scaled = numeric * 1000.0
            epoch = _finalize_epoch_series(scaled, source="epoch", columns=("epoch",))
            if epoch is not None:
                return epoch

    for name in ("dateutc", "datetime", "observed_at"):
        if name not in df.columns or name not in candidates:
            continue
        dt = pd.to_datetime(df[name], errors="coerce", utc=True)
        epoch = _finalize_from_datetime(dt, source=name, columns=(name,))
        if epoch is not None:
            return epoch

    for name in ("dateutc", "datetime", "date"):
        if name not in df.columns:
            continue
        dt = _excel_serial_to_datetime(df[name])
        if dt is None:
            continue
        epoch = _finalize_from_datetime(dt, source=f"{name} (excel)", columns=(name,))
        if epoch is not None:
            return epoch

    date_columns = [col for col in ("date", "dateutc") if col in df.columns and col in candidates]
    time_columns = [col for col in ("time", "time_utc") if col in df.columns]
    for date_col in date_columns:
        for time_col in time_columns or ["time"]:
            if time_col not in df.columns:
                continue
            combined = _combine_date_time(df, date_col, time_col)
            dt = pd.to_datetime(combined, errors="coerce", utc=True)
            epoch = _finalize_from_datetime(dt, source=f"{date_col}+{time_col}", columns=(date_col, time_col))
            if epoch is not None:
                return epoch

    if "time" in df.columns and "time" in candidates:
        dt = _parse_time_only(df["time"])
        if dt is not None:
            epoch = _finalize_from_datetime(dt, source="time", columns=("time",))
            if epoch is not None:
                return epoch

    raise OfflineTimestampError("Could not determine timestamp column", details=list(df.columns))


def parse_offline_timestamp(
    df: pd.DataFrame,
    config: dict,
    *,
    interactive: bool = False,
) -> OfflineTimestampResult:
    """Detect and normalize timestamp columns for offline imports."""

    del config, interactive  # unused but kept for signature compatibility

    try:
        epoch_series = to_epoch_ms(df)
    except OfflineTimestampError as exc:
        available = ", ".join(str(col) for col in df.columns)
        if available:
            logger.debug("[offline] No timestamp match. Available columns: %s", available)
        raise ValueError(
            "No valid timestamps found. Make sure your file contains a 'dateutc', 'epoch', or 'date'+'time' column.\n\n"
            "Examples:\n- dateutc\n- epoch\n- date + time"
        ) from exc
    valid_rows = len(epoch_series)
    total_rows = len(df)
    dropped = max(total_rows - valid_rows, 0)

    timestamp_utc = pd.to_datetime(epoch_series, unit="ms", utc=True)
    timestamp_utc.name = "observed_at"

    source = epoch_series.attrs.get("source") or "timestamp"
    columns_attr = epoch_series.attrs.get("columns") or [source]
    columns = tuple(str(col) for col in columns_attr)

    details: List[str] = []
    if dropped:
        details.append(f"Dropped {dropped} row(s) without valid timestamps.")

    epoch_min = int(epoch_series.min())
    epoch_max = int(epoch_series.max())
    start_iso = datetime.utcfromtimestamp(epoch_min / 1000).isoformat() + "Z"
    end_iso = datetime.utcfromtimestamp(epoch_max / 1000).isoformat() + "Z"
    details.append(f"Range: {start_iso} – {end_iso}")
    details.append(f"Using columns: {', '.join(columns)}")

    logger.info(
        "[offline] Parsed %d timestamp rows using %s (dropped %d)",
        valid_rows,
        source,
        dropped,
    )
    logger.debug("[offline] Timestamp range %s – %s", start_iso, end_iso)

    return OfflineTimestampResult(
        epoch_ms=epoch_series,
        timestamp_utc=timestamp_utc,
        source=source,
        columns=columns,
        non_null=valid_rows,
        details=details,
    )
