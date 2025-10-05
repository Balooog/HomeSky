"""Utilities for normalizing headers and parsing offline timestamps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger
from pandas.api import types as ptypes
from zoneinfo import ZoneInfo


def _is_tz_aware(series: pd.Series) -> bool:
    dtype = getattr(series, "dtype", None)
    tz = getattr(dtype, "tz", None)
    return tz is not None

__all__ = ["normalize_columns", "to_epoch_ms", "TimestampOverride"]


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
    "simple_date",
]

ISO_CANDIDATES: Tuple[str, ...] = (
    "date",
    "datetime",
    "timestamp",
    "dateutc",
    "date_utc",
    "obs_time_utc",
    "observed_at",
    "time_utc",
)

LOCAL_CANDIDATES: Tuple[str, ...] = (
    "simple_date",
    "simpledate",
    "date_local",
    "obs_time_local",
)

NUMERIC_CANDIDATES: Tuple[str, ...] = (
    "epoch_ms",
    "epoch",
    "dateutc",
    "date_utc",
    "time_utc",
)

PAIR_DATE_CANDIDATES: Tuple[str, ...] = (
    "date",
    "dateutc",
    "date_utc",
    "simple_date",
    "simpledate",
    "date_local",
)

TIME_CANDIDATES: Tuple[str, ...] = (
    "time",
    "time_utc",
    "time_local",
)

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


@dataclass(slots=True)
class TimestampOverride:
    """Describe a manual timestamp mapping selected by the user."""

    column: str
    kind: str
    timezone: Optional[str] = None
    time_column: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        payload = asdict(self)
        # ``asdict`` preserves Optional[None]; the GUI serializer expects lowercase keys.
        return payload


def _strip_units(name: str) -> str:
    return name.replace("Â", "").replace("º", "°")


def _normalize_header(raw: object) -> str:
    text = str(raw).replace("\ufeff", "")
    text = _strip_units(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace(".", " ")
    text = text.strip()
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = " ".join(part for part in text.split() if part)
    text = text.replace(" ", "_")
    text = text.replace("__", "_")
    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Normalize dataframe column names in-place and return the rename map."""

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


def _format_iso(epoch_ms: int) -> str:
    dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _finalize_numeric(
    series: pd.Series,
    *,
    source: str,
    columns: Sequence[str],
    scale_hint: Optional[str] = None,
) -> Optional[pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return None
    values = numeric.astype("float64")
    if scale_hint == "ms":
        epoch = values
    elif scale_hint == "s":
        epoch = values * 1000.0
    else:
        median = float(values.median()) if not values.empty else 0.0
        epoch = values if median >= 1e11 else values * 1000.0
    epoch = epoch.round()
    epoch = epoch[epoch > 0]
    if epoch.empty:
        return None
    coerced = epoch.astype("int64")
    coerced.attrs["source"] = source
    coerced.attrs["columns"] = list(columns)
    return coerced


def _finalize_from_datetime(
    series: pd.Series,
    *,
    source: str,
    columns: Sequence[str],
) -> Optional[pd.Series]:
    if series is None or not isinstance(series, pd.Series):
        return None
    if not ptypes.is_datetime64_any_dtype(series):
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    if _is_tz_aware(valid):
        utc_values = valid.dt.tz_convert("UTC")
    else:
        utc_values = valid.dt.tz_localize("UTC")
    values = pd.Series(
        (utc_values.astype("int64", copy=False) // 1_000_000).astype("int64"),
        index=utc_values.index,
    )
    values.attrs["source"] = source
    values.attrs["columns"] = list(columns)
    return values


def _excel_serial_to_datetime(series: pd.Series) -> Optional[pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        return None
    converted = pd.to_datetime(
        numeric.astype("float64"),
        unit="D",
        origin="1899-12-30",
        errors="coerce",
        utc=True,
    )
    if converted.dropna().empty:
        return None
    converted.index = numeric.index
    return converted


def _combine_date_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    date_series = df[date_col].astype("string").str.strip()
    time_series = df[time_col].astype("string").str.strip()
    combined = date_series.str.cat(time_series, sep=" ")
    combined = combined.where(date_series.notna() & time_series.notna())
    combined.index = df.index
    return combined


def _time_only_to_datetime(series: pd.Series, tz_hint: str) -> Optional[pd.Series]:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dropna().empty or not ptypes.is_datetime64_any_dtype(parsed):
        return None
    valid = parsed.dropna()
    if valid.dt.normalize().nunique() > 1:
        return None
    today = datetime.now(tz=timezone.utc).date()
    formatted = valid.dt.strftime("%H:%M:%S.%f")
    combined = formatted.apply(
        lambda value: f"{today.isoformat()} {value.rstrip('0').rstrip('.') if '.' in value else value}"
    )
    dt = pd.to_datetime(combined, errors="coerce")
    if dt.dropna().empty:
        return None
    try:
        tz = ZoneInfo(tz_hint)
    except Exception:
        tz = timezone.utc
    localized = dt.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
    localized.index = valid.index
    return localized


def _localize_to_utc(series: pd.Series, tz_hint: str) -> Optional[pd.Series]:
    if series is None or not isinstance(series, pd.Series):
        return None
    if not ptypes.is_datetime64_any_dtype(series):
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    try:
        tz = ZoneInfo(tz_hint)
    except Exception:
        tz = timezone.utc
    if _is_tz_aware(valid):
        localized = valid.dt.tz_convert("UTC")
    else:
        try:
            localized = valid.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            localized = valid.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dropna()
        if localized.empty:
            return None
        localized = localized.dt.tz_convert("UTC")
    localized.index = valid.index
    return localized


def _coerce_epoch_ms(
    df: pd.DataFrame,
    *,
    candidate_cols: Iterable[str] | None,
    tz_hint: str,
    override: Optional[TimestampOverride],
) -> pd.Series:
    candidates = list(candidate_cols) if candidate_cols is not None else list(DEFAULT_CANDIDATES)
    candidate_set = {_normalize_header(name) for name in candidates}

    if override:
        column = override.column
        time_column = override.time_column
        kind = override.kind.lower()
        source_label = f"override:{kind}:{column}" if time_column is None else f"override:{kind}:{column}+{time_column}"
        if column in df.columns:
            if kind == "iso":
                dt = pd.to_datetime(df[column], errors="coerce", utc=True)
                result = _finalize_from_datetime(dt, source=source_label, columns=[column])
                if result is not None:
                    return result
            elif kind in {"epoch_ms", "epoch"}:
                scale = "ms" if kind == "epoch_ms" else "s"
                result = _finalize_numeric(df[column], source=source_label, columns=[column], scale_hint=scale)
                if result is not None:
                    return result
            elif kind == "local":
                dt = pd.to_datetime(df[column], errors="coerce")
                localized = _localize_to_utc(dt, override.timezone or tz_hint)
                if localized is not None:
                    result = _finalize_from_datetime(
                        localized, source=source_label, columns=[column]
                    )
                    if result is not None:
                        return result
            elif kind == "excel":
                dt = _excel_serial_to_datetime(df[column])
                if dt is not None:
                    result = _finalize_from_datetime(dt, source=source_label, columns=[column])
                    if result is not None:
                        return result
            elif kind == "pair" and time_column and time_column in df.columns:
                combined = _combine_date_time(df, column, time_column)
                dt = pd.to_datetime(combined, errors="coerce")
                localized = _localize_to_utc(dt, override.timezone or tz_hint)
                if localized is not None:
                    result = _finalize_from_datetime(
                        localized,
                        source=source_label,
                        columns=[column, time_column],
                    )
                    if result is not None:
                        return result
        logger.debug("[offline] Timestamp override for %s failed; falling back to auto-detection", column)

    if "epoch_ms" in df.columns and (not candidate_set or "epoch_ms" in candidate_set):
        epoch = _finalize_numeric(df["epoch_ms"], source="epoch_ms", columns=("epoch_ms",), scale_hint="ms")
        if epoch is not None:
            logger.debug("[offline] Using epoch_ms column for timestamps")
            return epoch

    if "epoch" in df.columns and (not candidate_set or "epoch" in candidate_set):
        epoch = _finalize_numeric(df["epoch"], source="epoch", columns=("epoch",), scale_hint="s")
        if epoch is not None:
            logger.debug("[offline] Using epoch column for timestamps")
            return epoch

    for name in NUMERIC_CANDIDATES:
        if name in {"epoch_ms", "epoch"}:
            continue
        if name not in df.columns or (candidate_set and name not in candidate_set):
            continue
        epoch = _finalize_numeric(df[name], source=name, columns=(name,))
        if epoch is not None:
            logger.debug("[offline] Using %s column for numeric timestamps", name)
            return epoch

    for name in ISO_CANDIDATES:
        if name == "date" and "time" in df.columns:
            continue
        if name not in df.columns or (candidate_set and name not in candidate_set):
            continue
        dt = pd.to_datetime(df[name], errors="coerce", utc=True)
        epoch = _finalize_from_datetime(dt, source=name, columns=(name,))
        if epoch is not None:
            logger.debug("[offline] Using %s column for ISO timestamps", name)
            return epoch

    for name in LOCAL_CANDIDATES:
        if name not in df.columns or (candidate_set and name not in candidate_set):
            continue
        dt = pd.to_datetime(df[name], errors="coerce")
        localized = _localize_to_utc(dt, tz_hint)
        if localized is None:
            continue
        epoch = _finalize_from_datetime(
            localized,
            source=f"{name} (local {tz_hint})",
            columns=(name,),
        )
        if epoch is not None:
            logger.debug("[offline] Using %s column localized to %s", name, tz_hint)
            return epoch

    for name in ISO_CANDIDATES + LOCAL_CANDIDATES:
        if name not in df.columns or (candidate_set and name not in candidate_set):
            continue
        dt = _excel_serial_to_datetime(df[name])
        if dt is None:
            continue
        epoch = _finalize_from_datetime(dt, source=f"{name} (excel)", columns=(name,))
        if epoch is not None:
            logger.debug("[offline] Using %s Excel serials for timestamps", name)
            return epoch

    for date_col in PAIR_DATE_CANDIDATES:
        if date_col not in df.columns:
            continue
        if candidate_set and date_col not in candidate_set and "date" not in candidate_set:
            continue
        for time_col in TIME_CANDIDATES:
            if time_col not in df.columns:
                continue
            if candidate_set and time_col not in candidate_set and "time" not in candidate_set:
                continue
            combined = _combine_date_time(df, date_col, time_col)
            dt = pd.to_datetime(combined, errors="coerce")
            localized = _localize_to_utc(dt, tz_hint)
            if localized is None:
                continue
            epoch = _finalize_from_datetime(
                localized,
                source=f"{date_col}+{time_col}",
                columns=(date_col, time_col),
            )
            if epoch is not None:
                logger.debug(
                    "[offline] Combining %s and %s columns for timestamps",
                    date_col,
                    time_col,
                )
                return epoch

    if "time" in df.columns and (not candidate_set or "time" in candidate_set):
        localized = _time_only_to_datetime(df["time"], tz_hint)
        if localized is not None:
            epoch = _finalize_from_datetime(localized, source="time", columns=("time",))
            if epoch is not None:
                logger.debug("[offline] Using time-only column with today's date for timestamps")
                return epoch

    raise ValueError("No valid timestamps found")


def _attach_totals(epoch: pd.Series, *, total_rows: int) -> None:
    epoch.attrs["_rows_total"] = (total_rows, len(epoch))
    dropped_rows = total_rows - len(epoch)
    if dropped_rows > 0:
        logger.debug("[offline] Dropped %d row(s) without valid timestamps", dropped_rows)
    if len(epoch) == 0:
        return
    start_iso = _format_iso(int(epoch.min()))
    end_iso = _format_iso(int(epoch.max()))
    logger.debug("[offline] Timestamp range %s – %s", start_iso, end_iso)


def to_epoch_ms(
    df: pd.DataFrame,
    *,
    candidate_cols: Iterable[str] | None = None,
    tz_hint: str = "UTC",
    override: Optional[TimestampOverride] = None,
) -> pd.Series:
    epoch = _coerce_epoch_ms(
        df,
        candidate_cols=candidate_cols,
        tz_hint=tz_hint,
        override=override,
    )
    _attach_totals(epoch, total_rows=len(df))
    return epoch

