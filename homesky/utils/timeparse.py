"""Utilities for normalizing headers and parsing offline timestamps."""

from __future__ import annotations

from datetime import datetime
import re
import unicodedata
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from pandas.api import types as ptypes
from loguru import logger

__all__ = ["normalize_columns", "to_epoch_ms"]


DEFAULT_CANDIDATES: List[str] = [
    "epoch_ms",
    "epoch",
    "dateutc",
    "date_utc",
    "datetime",
    "date_time",
    "observed_at",
    "time",
    "date",
]

HEADER_SYNONYMS: Dict[str, str] = {
    "temperature": "temp_f",
    "temperaturef": "temp_f",
    "tempf": "temp_f",
    "roof_ws_2000_temperature": "temp_f",
    "feels_like": "feelslike_f",
    "feelslike": "feelslike_f",
    "dew_point": "dew_point_f",
    "dewpoint": "dew_point_f",
    "dewptf": "dew_point_f",
    "windspeed": "wind_speed_mph",
    "windspeedmph": "wind_speed_mph",
    "wind_speed": "wind_speed_mph",
    "wind_speed_mph": "wind_speed_mph",
    "wind_gust": "wind_gust_mph",
    "windgust": "wind_gust_mph",
    "windgustmph": "wind_gust_mph",
    "max_daily_gust": "max_gust_mph",
    "wind_direction": "wind_dir_deg",
    "winddir": "wind_dir_deg",
    "wind_dir": "wind_dir_deg",
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
    "humidity_percent": "humidity",
    "relative_humidity": "humidity",
    "ultra_violet_radiation_index": "uv_index",
    "uv": "uv_index",
    "solar_radiation": "solar_wm2",
    "indoor_temperature": "indoor_temp_f",
    "indoor_humidity": "indoor_humidity",
    "pm2_5_outdoor": "pm25_ugm3",
    "pm2_5_outdoor_24_hour_average": "pm25_24h_avg_ugm3",
    "lightning_strikes_per_day": "lightning_day",
    "lightning_strikes_per_hour": "lightning_hour",
    "epochms": "epoch_ms",
    "date_time": "datetime",
    "dateutc": "dateutc",
    "date_utc": "date_utc",
    "observed": "observed_at",
}


def _strip_units(name: str) -> str:
    return re.sub(r"\([^)]*\)", "", name)


def _normalize_header(raw: object) -> str:
    text = str(raw).replace("\ufeff", "")
    text = unicodedata.normalize("NFKD", text)
    text = _strip_units(text)
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace(".", " ")
    text = re.sub(r"[^a-z0-9_ ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Normalize dataframe column names in-place."""

    rename_map: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for original in df.columns:
        normalized = _normalize_header(original)
        normalized = HEADER_SYNONYMS.get(normalized, normalized) or "column"
        count = counts.get(normalized, 0)
        final_name = normalized if count == 0 else f"{normalized}_{count + 1}"
        counts[normalized] = count + 1
        rename_map[str(original)] = final_name
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return rename_map


def _finalize_numeric(
    series: pd.Series,
    *,
    source: str,
    columns: Tuple[str, ...],
    scale: float = 1.0,
) -> pd.Series | None:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.empty:
        return None
    numeric = numeric.astype("float64") * scale
    valid = numeric[numeric.notna() & (numeric > 0)]
    if valid.empty:
        return None
    epoch = valid.round().astype("int64")
    epoch.attrs["source"] = source
    epoch.attrs["columns"] = list(columns)
    return epoch


def _finalize_from_datetime(
    series: pd.Series,
    *,
    source: str,
    columns: Tuple[str, ...],
) -> pd.Series | None:
    if series is None or series.empty:
        return None
    if not ptypes.is_datetime64_any_dtype(series):
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    if ptypes.is_datetime64tz_dtype(valid):
        utc_values = valid.dt.tz_convert("UTC")
    else:
        utc_values = valid.dt.tz_localize("UTC")
    epoch = (utc_values.view("int64") // 1_000_000).astype("int64")
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
    return combined.where(date_series.notna() & time_series.notna())


def _time_only_to_datetime(series: pd.Series) -> pd.Series | None:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dropna().empty:
        return None
    if not ptypes.is_datetime64_any_dtype(parsed):
        return None
    valid = parsed.dropna()
    if valid.empty:
        return None
    normalized_days = valid.dt.normalize().nunique()
    if normalized_days > 1:
        return None
    today = datetime.utcnow().date()
    formatted = valid.dt.strftime("%H:%M:%S.%f")
    combined_strings = formatted.apply(
        lambda value: f"{today.isoformat()} {value.rstrip('0').rstrip('.') if '.' in value else value}"
    )
    combined = pd.to_datetime(combined_strings, errors="coerce", utc=True)
    combined.index = valid.index
    return combined


def _format_iso(epoch_ms: int) -> str:
    return datetime.utcfromtimestamp(epoch_ms / 1000).isoformat() + "Z"


def _coerce_epoch_ms(
    df: pd.DataFrame,
    *,
    candidate_cols: Iterable[str] | None = None,
) -> pd.Series:
    """Coerce timestamp-related columns into epoch milliseconds."""

    normalize_columns(df)
    candidates = list(candidate_cols) if candidate_cols is not None else list(DEFAULT_CANDIDATES)
    candidate_set = {_normalize_header(name) for name in candidates}

    if "epoch_ms" in df.columns and "epoch_ms" in candidate_set:
        epoch = _finalize_numeric(df["epoch_ms"], source="epoch_ms", columns=("epoch_ms",))
        if epoch is not None:
            logger.debug("[offline] Using epoch_ms column for timestamps")
            return epoch

    if "epoch" in df.columns and "epoch" in candidate_set:
        epoch = _finalize_numeric(df["epoch"], source="epoch", columns=("epoch",), scale=1000.0)
        if epoch is not None:
            logger.debug("[offline] Using epoch column for timestamps")
            return epoch

    for name in ("dateutc", "date_utc", "datetime", "date_time", "observed_at"):
        if name not in df.columns or name not in candidate_set:
            continue
        dt = pd.to_datetime(df[name], errors="coerce", utc=True)
        epoch = _finalize_from_datetime(dt, source=name, columns=(name,))
        if epoch is not None:
            logger.debug("[offline] Using %s column for timestamps", name)
            return epoch

    for name in ("dateutc", "date_utc", "datetime", "date", "simple_date"):
        if name not in df.columns:
            continue
        dt = _excel_serial_to_datetime(df[name])
        if dt is None:
            continue
        epoch = _finalize_from_datetime(dt, source=f"{name} (excel)", columns=(name,))
        if epoch is not None:
            logger.debug("[offline] Using %s excel serials for timestamps", name)
            return epoch

    date_candidates = [
        col
        for col in ("date", "date_utc", "dateutc", "simple_date")
        if col in df.columns
    ]
    time_candidates = [col for col in ("time", "time_utc") if col in df.columns]
    for date_col in date_candidates:
        if "date" not in candidate_set and date_col not in candidate_set:
            continue
        for time_col in time_candidates:
            if "time" not in candidate_set and time_col not in candidate_set:
                continue
            combined = _combine_date_time(df, date_col, time_col)
            dt = pd.to_datetime(combined, errors="coerce", utc=True)
            epoch = _finalize_from_datetime(
                dt, source=f"{date_col}+{time_col}", columns=(date_col, time_col)
            )
            if epoch is not None:
                logger.debug(
                    "[offline] Combining %s and %s columns for timestamps", date_col, time_col
                )
                return epoch

    if "time" in df.columns and "time" in candidate_set:
        dt = _time_only_to_datetime(df["time"])
        if dt is not None:
            epoch = _finalize_from_datetime(dt, source="time", columns=("time",))
            if epoch is not None:
                logger.debug("[offline] Using time-only column with today's date for timestamps")
                return epoch

    raise ValueError("No valid timestamps found")


def _log_epoch_summary(epoch: pd.Series) -> None:
    dropped = epoch.attrs.get("_rows_total")
    if isinstance(dropped, tuple):
        total_rows, valid_rows = dropped
        dropped_rows = total_rows - valid_rows
    else:
        total_rows = valid_rows = None
        dropped_rows = None
    if dropped_rows is not None:
        logger.debug("[offline] Dropped %d row(s) without valid timestamps", dropped_rows)
    if len(epoch) == 0:
        return
    start_iso = _format_iso(int(epoch.min()))
    end_iso = _format_iso(int(epoch.max()))
    logger.debug("[offline] Timestamp range %s – %s", start_iso, end_iso)


def _attach_totals(epoch: pd.Series, *, total_rows: int) -> None:
    epoch.attrs["_rows_total"] = (total_rows, len(epoch))
    _log_epoch_summary(epoch)


def to_epoch_ms(
    df: pd.DataFrame,
    *,
    candidate_cols: Iterable[str] | None = None,
) -> pd.Series:
    epoch = _coerce_epoch_ms(df, candidate_cols=candidate_cols)
    _attach_totals(epoch, total_rows=len(df))
    return epoch

