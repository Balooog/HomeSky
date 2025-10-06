"""Streamlit dashboard for HomeSky."""

from __future__ import annotations

from calendar import monthrange
from datetime import timedelta
from io import BytesIO
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
from pandas.api import types as ptypes
import streamlit as st

import ingest
from utils.derived import compute_all_derived
from utils.theming import get_theme, load_typography


MetricEntry = Tuple[str, Tuple[str, ...]]

# Friendly metric labels mapped to canonical columns (and synonyms for legacy
# datasets). Only the metrics present in the loaded DataFrame will be shown to
# the user.
METRIC_REGISTRY: Sequence[MetricEntry] = (
    ("Outdoor Temperature (°F)", ("temp_f", "tempf", "temperature")),
    ("Feels Like (°F)", ("feelslike_f", "feels_like_f")),
    ("Dew Point (°F)", ("dew_point_f", "dewptf")),
    ("Humidity (%)", ("humidity", "humidity_pct")),
    ("Wind Speed (mph)", ("wind_speed_mph", "windspeedmph")),
    ("Wind Gust (mph)", ("wind_gust_mph", "windgustmph")),
    ("Barometric Pressure (inHg)", ("rel_pressure_inhg", "pressure_inhg")),
    ("Solar Radiation (W/m²)", ("solar_wm2", "solar_radiation_wm2")),
    ("Rain Rate (in/hr)", ("rain_rate_in_hr",)),
    ("Daily Rain (in)", ("rain_day_in", "daily_rain_in")),
    ("24h Rain (in)", ("rain_24h_in",)),
    ("PM2.5 Outdoor (µg/m³)", ("pm25_ugm3", "pm25_out_ugm3")),
    ("PM2.5 24h Avg (µg/m³)", ("pm25_24h_avg_ugm3",)),
    ("UV Index", ("uv_index", "uv")),
)


RESAMPLE_UI: Sequence[Tuple[str, Optional[str], str]] = (
    ("raw", None, "Raw"),
    ("5min", "5min", "5 minutes"),
    ("15min", "15min", "15 minutes"),
    ("H", "H", "Hourly"),
    ("D", "D", "Daily"),
    ("W", "W", "Weekly"),
    ("M", "M", "Monthly"),
)

AGGREGATE_OPTIONS: Sequence[str] = ("mean", "max", "min", "sum", "last")


def _get_zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except Exception:  # pragma: no cover - fallback to UTC
        return ZoneInfo("UTC")


def ensure_time_index(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    """Return a copy of *df* indexed by local time without column/index clashes."""

    if df.empty:
        return df.copy()

    working = df.copy()
    zone = _get_zone(tz_name)

    utc_source: Optional[pd.Series] = None

    if "obs_time_utc" in working.columns:
        utc_source = pd.to_datetime(
            working["obs_time_utc"], errors="coerce", utc=True
        )
    elif "epoch_ms" in working.columns:
        epoch_series = pd.to_numeric(working["epoch_ms"], errors="coerce")
        utc_source = pd.to_datetime(epoch_series, unit="ms", errors="coerce", utc=True)
    elif "s_time_local" in working.columns:
        local_series = pd.to_datetime(working["s_time_local"], errors="coerce")
        if getattr(local_series.dt, "tz", None) is None:
            localized = local_series.dt.tz_localize(
                zone, ambiguous="NaT", nonexistent="shift_forward"
            )
        else:
            localized = local_series.dt.tz_convert(zone)
        utc_source = localized.dt.tz_convert("UTC")
    elif isinstance(working.index, pd.DatetimeIndex):
        index_values = pd.Series(working.index, index=working.index)
        if index_values.dt.tz is None:
            localized = index_values.dt.tz_localize(
                zone, ambiguous="NaT", nonexistent="shift_forward"
            )
        else:
            localized = index_values.dt.tz_convert(zone)
        utc_source = localized.dt.tz_convert("UTC")
    else:
        raise ValueError(
            "No recognizable time column: need obs_time_utc or epoch_ms or s_time_local"
        )

    if utc_source is None:
        raise ValueError(
            "No recognizable time column: need obs_time_utc or epoch_ms or s_time_local"
        )

    if not isinstance(utc_source, pd.Series):
        utc_source = pd.Series(utc_source, index=working.index)

    mask = utc_source.notna()
    if not bool(mask.any()):
        empty = working.iloc[0:0].copy()
        empty.index = pd.DatetimeIndex([], tz=zone, name="s_time_local")
        return empty

    working = working.loc[mask].copy()
    utc_valid = utc_source.loc[mask]
    local_index = utc_valid.dt.tz_convert(zone)

    working = working.drop(columns=["s_time_local"], errors="ignore")
    working.index = pd.DatetimeIndex(local_index.array)
    working.index.name = "s_time_local"
    working = working.sort_index(kind="mergesort")
    return working


def ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if df.index.name == "s_time_local":
        out = df.reset_index()
        if "s_time_local" in out.columns[1:]:
            out = out.loc[:, ~out.columns.duplicated()]
        return out
    return df.copy()


def _safe_localize_day(value: pd.Timestamp | str, zone: ZoneInfo) -> pd.Timestamp:
    """Return a timezone-aware timestamp for the start of *value*'s day."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(zone)
        return ts.normalize()

    ts = ts.normalize()
    localized = ts.tz_localize(zone, ambiguous="NaT", nonexistent="shift_forward")
    if pd.isna(localized):
        for offset_hours in (1, -1):
            try:
                nudged = (ts + pd.Timedelta(hours=offset_hours)).tz_localize(
                    zone, ambiguous="NaT", nonexistent="shift_forward"
                )
            except Exception:  # pragma: no cover - defensive fallback
                nudged = pd.NaT
            if pd.isna(nudged):
                continue
            localized = nudged - pd.Timedelta(hours=offset_hours)
            break
        if pd.isna(localized):
            localized = ts.tz_localize(zone, ambiguous=True, nonexistent="shift_forward")
    return localized


def _format_timestamp(ts: Optional[pd.Timestamp], tz_name: str) -> str:
    if ts is None or pd.isna(ts):
        return "n/a"
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    zone = _get_zone(tz_name)
    try:
        localized = ts.tz_convert(zone)
    except Exception:
        localized = ts.tz_convert("UTC")
    abbr = localized.tzname() or ""
    display_abbr = "ET" if abbr in {"EDT", "EST"} else abbr
    timestamp_str = localized.strftime("%Y-%m-%d %H:%M")
    return f"{timestamp_str} {display_abbr}".strip()


def _cache_token(sqlite_path: str, parquet_path: str) -> str:
    parts: List[str] = []
    for raw_path in (sqlite_path, parquet_path):
        path = Path(raw_path).expanduser()
        try:
            stat = path.stat()
            parts.append(f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}")
        except FileNotFoundError:
            resolved = path.resolve() if path.exists() else path
            parts.append(f"{resolved}:{0}:{0}")
    digest_source = "|".join(parts)
    return hashlib.sha1(digest_source.encode("utf-8")).hexdigest()


@st.cache_data(ttl=60)
def load_data(sqlite_path: str, parquet_path: str, cache_token: str) -> pd.DataFrame:
    del cache_token  # included for cache invalidation via hash key
    storage_config: Dict[str, Dict[str, str]] = {
        "storage": {
            "sqlite_path": sqlite_path,
            "parquet_path": parquet_path,
        }
    }
    return ingest.load_dataset(storage_config)


def _ensure_logging(config: Dict) -> None:
    if st.session_state.get("_homesky_logging_configured"):
        return
    ingest.setup_logging(config)
    st.session_state["_homesky_logging_configured"] = True


def _resolve_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _available_metrics(df: pd.DataFrame) -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    for label, synonyms in METRIC_REGISTRY:
        column = _resolve_column(df, *synonyms)
        if column and ptypes.is_numeric_dtype(df[column]):
            options.append((label, column))
    if options:
        return options
    numeric_columns = [col for col in df.columns if ptypes.is_numeric_dtype(df[col])]
    return [(col, col) for col in sorted(numeric_columns)]


def _prepare_time_columns(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    working = ensure_time_index(df, tz_name)

    utc_index = working.index.tz_convert("UTC")
    if "s_time_utc" in working.columns:
        s_time_utc = pd.to_datetime(working["s_time_utc"], errors="coerce", utc=True)
        mask = s_time_utc.notna()
        if mask.any():
            utc_index = utc_index.where(~mask, s_time_utc)
    working["s_time_utc"] = utc_index

    if "epoch_ms" in working.columns:
        epoch_numeric = pd.to_numeric(working["epoch_ms"], errors="coerce")
        working["epoch_ms"] = epoch_numeric.astype("Int64")
    else:
        working["epoch_ms"] = (utc_index.view("int64") // 1_000_000).astype("int64")

    return working


def _latest_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.iloc[-1])


def _format_temperature(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{int(round(float(value)))}°"


def _format_inches(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1f} in"


def _format_speed(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1f} mph"


def _is_temperature_column(column: str) -> bool:
    lowered = column.lower()
    return "temp" in lowered or "feels" in lowered


def _is_rain_column(column: str) -> bool:
    lowered = column.lower()
    return "rain" in lowered


def _format_stat_value(value: float, column: str) -> str:
    if pd.isna(value):
        return "n/a"
    if _is_temperature_column(column):
        return _format_temperature(value)
    if _is_rain_column(column):
        return _format_inches(value)
    return f"{value:.2f}"


def _coerce_month_number(value: object) -> Optional[int]:
    try:
        month_int = int(value)
        if 1 <= month_int <= 12:
            return month_int
    except (TypeError, ValueError):
        pass
    try:
        parsed = pd.to_datetime(str(value), errors="coerce")
    except Exception:  # pragma: no cover - defensive guard
        parsed = pd.NaT
    if pd.isna(parsed):
        return None
    return int(parsed.month)


def _monthly_normals_from_config(config: Dict) -> Tuple[Dict[int, float], Optional[str]]:
    noaa_cfg = config.get("noaa", {})
    normals_path = noaa_cfg.get("normals_csv")
    if not normals_path:
        return {}, None
    path = Path(normals_path).expanduser()
    if not path.exists():
        return {}, f"Normals CSV not found at {path}"
    try:
        normals_df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - surface to UI
        return {}, f"Unable to read NOAA normals: {exc}"
    if normals_df.empty:
        return {}, "Normals CSV is empty"
    month_column = None
    for candidate in normals_df.columns:
        name = str(candidate).lower()
        if name in {"month", "mon"}:
            month_column = candidate
            break
    if month_column is None:
        month_column = normals_df.columns[0]
    value_candidates = [col for col in normals_df.columns if col != month_column]
    if not value_candidates:
        return {}, "Normals CSV is missing value columns"
    preferred = None
    for candidate in value_candidates:
        lowered = str(candidate).lower()
        if any(token in lowered for token in ("in", "inch", "rain")):
            preferred = candidate
            break
    value_column = preferred or value_candidates[0]
    values = pd.to_numeric(normals_df[value_column], errors="coerce")
    months_raw = normals_df[month_column]
    mapping: Dict[int, float] = {}
    for month_raw, value in zip(months_raw, values):
        month_number = _coerce_month_number(month_raw)
        if month_number is None or pd.isna(value):
            continue
        mapping[int(month_number)] = float(value)
    if not mapping:
        return {}, "Normals CSV does not contain usable month totals"
    unit_hint = str(value_column).lower()
    if "mm" in unit_hint:
        mapping = {month: amount / 25.4 for month, amount in mapping.items()}
    else:
        if max(mapping.values()) > 50:  # likely provided in millimetres
            mapping = {month: amount / 25.4 for month, amount in mapping.items()}
    return mapping, None


def _daily_normals_for_year(
    monthly_normals: Dict[int, float], year: int, zone: ZoneInfo
) -> pd.Series:
    if not monthly_normals:
        return pd.Series(dtype="float64")
    start = pd.Timestamp(year=year, month=1, day=1, tz=zone)
    end = pd.Timestamp(year=year, month=12, day=31, tz=zone)
    dates = pd.date_range(start=start, end=end, freq="D", tz=zone)
    values: List[float] = []
    for day in dates:
        month_total = float(monthly_normals.get(day.month, 0.0))
        days_in_month = monthrange(day.year, day.month)[1]
        daily_value = month_total / days_in_month if days_in_month else 0.0
        values.append(daily_value)
    series = pd.Series(values, index=dates, dtype="float64")
    series.name = "normal_in"
    return series


def _daily_rainfall(df: pd.DataFrame) -> Tuple[pd.Series, Optional[str]]:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(dtype="float64"), None
    column = _resolve_column(df, "daily_rain_in", "rain_day_in")
    if column:
        series = pd.to_numeric(df[column], errors="coerce")
        daily = series.resample("D").max().fillna(0.0)
        daily.name = column
        return daily, column
    column = _resolve_column(df, "event_rain_in", "rain_event_in")
    if column:
        series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        daily = series.resample("D").sum(min_count=1).fillna(0.0)
        daily.name = column
        return daily, column
    return pd.Series(dtype="float64"), None


def _top_rain_events(
    df: pd.DataFrame, column: Optional[str], limit: int = 5
) -> pd.DataFrame:
    if not column or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["s_time_local", "amount"])
    series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if series.empty:
        return pd.DataFrame(columns=["s_time_local", "amount"])
    daily = series.resample("D").max().dropna()
    daily = daily[daily > 0]
    if daily.empty:
        return pd.DataFrame(columns=["s_time_local", "amount"])
    top = daily.sort_values(ascending=False).head(limit)
    result = top.reset_index()
    result.columns = ["s_time_local", "amount"]
    return result


def _rain_rate_histogram(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    column = _resolve_column(df, "rain_rate_in_hr", "rainrate_in_hr")
    if not column:
        return pd.DataFrame(columns=["bucket", "count"]), None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame(columns=["bucket", "count"]), column
    bins = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf")]
    labels = ["0–0.1", "0.1–0.25", "0.25–0.5", "0.5–1", "1–2", ">2"]
    categorized = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=False)
    counts = categorized.value_counts().sort_index()
    histogram = counts.reset_index()
    histogram.columns = ["bucket", "count"]
    return histogram, column


def _direction_to_cardinal(degrees: float) -> Optional[str]:
    if pd.isna(degrees):
        return None
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    idx = int((degrees % 360) / 22.5 + 0.5) % 16
    return directions[idx]


def _rain_24h(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    diffs = numeric.diff().clip(lower=0)
    return float(diffs.sum(skipna=True))


def sanitize_for_arrow(
    df: pd.DataFrame,
    *,
    tz_name: str,
    value_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sanitized = ensure_time_index(df, tz_name)

    keep: List[str] = []
    for name in ("mac", "epoch_ms", "s_time_utc", "dateutc"):
        if name in sanitized.columns:
            keep.append(name)
    if value_columns is not None:
        keep.extend(col for col in value_columns if col in sanitized.columns)
    keep = list(dict.fromkeys(keep))
    if keep:
        sanitized = sanitized.loc[:, keep].copy()
    else:
        sanitized = sanitized.copy()

    zone = _get_zone(tz_name)
    local_index = sanitized.index.tz_convert(zone)
    sanitized = sanitized.sort_index(kind="mergesort")
    sanitized_reset = sanitized.reset_index()

    local_series = pd.to_datetime(
        sanitized_reset["s_time_local"], errors="coerce"
    )
    if getattr(local_series.dt, "tz", None) is None:
        local_series = local_series.dt.tz_localize(
            zone, ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        local_series = local_series.dt.tz_convert(zone)
    sanitized_reset["s_time_local"] = local_series
    sanitized_reset["s_time_utc"] = local_series.dt.tz_convert("UTC")

    if "dateutc" in sanitized_reset.columns:
        sanitized_reset["dateutc"] = pd.to_datetime(
            sanitized_reset["dateutc"], errors="coerce", utc=True
        )

    if "epoch_ms" in sanitized_reset.columns:
        sanitized_reset["epoch_ms"] = pd.to_numeric(
            sanitized_reset["epoch_ms"], errors="coerce"
        ).astype("Int64")
    else:
        sanitized_reset["epoch_ms"] = (
            sanitized_reset["s_time_utc"].view("int64") // 1_000_000
        ).astype("int64")

    drop_candidates = [
        "observed_at",
        "obs_time_utc",
        "obs_time_local",
        "timestamp_local",
        "timestamp_utc",
    ]
    sanitized_reset = sanitized_reset.drop(
        columns=[c for c in drop_candidates if c in sanitized_reset.columns],
        errors="ignore",
    )

    for column in sanitized_reset.select_dtypes(include="object").columns:
        converted = pd.to_numeric(sanitized_reset[column], errors="ignore")
        sanitized_reset[column] = converted
        if sanitized_reset[column].dtype == "object":
            sanitized_reset[column] = sanitized_reset[column].astype("string")

    if "mac" in sanitized_reset.columns:
        sanitized_reset["mac"] = sanitized_reset["mac"].astype("string")

    sanitized_reset = sanitized_reset.dropna(subset=["s_time_local"]).reset_index(drop=True)
    return sanitized_reset


def _compute_rainfall(window: pd.DataFrame) -> Tuple[float, Optional[str]]:
    rain_column = _resolve_column(
        window,
        "rain_24h_in",
        "rain_day_in",
        "daily_rain_in",
        "rain_event_in",
        "rain_hour_in",
        "rain_rate_in_hr",
    )
    if not rain_column:
        return float("nan"), None
    if rain_column == "rain_rate_in_hr":
        rainfall = float(pd.to_numeric(window[rain_column], errors="coerce").fillna(0).sum())
    else:
        rainfall = _rain_24h(window[rain_column])
    return rainfall, rain_column


def _metric_series(
    df: pd.DataFrame,
    metric_key: str,
    tz_name: str,
    rule: Optional[str],
    agg: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[metric_key])

    working = ensure_time_index(df, tz_name)

    aliases = {
        "tempf": "temp_f",
        "dewpoint_f": "dew_point_f",
        "winddir": "wind_dir_deg",
        "windspeed_mph": "wind_speed_mph",
        "gust_mph": "wind_gust_mph",
        "rain_rate_in_hr": "rain_rate_in_hr",
    }
    column = aliases.get(metric_key, metric_key)
    if column not in working.columns:
        return pd.DataFrame(columns=[column])

    working[column] = pd.to_numeric(working[column], errors="coerce")

    agg = (agg or "mean").lower()
    if rule:
        resampler = working[column].resample(rule)
        if agg == "last":
            working = resampler.last().to_frame(name=column)
        elif agg == "sum":
            working = resampler.sum(min_count=1).to_frame(name=column)
        elif hasattr(resampler, agg):
            working = getattr(resampler, agg)().to_frame(name=column)
        else:
            working = resampler.mean().to_frame(name=column)
    else:
        working = working[[column]]

    working = working.dropna(how="all")
    working.index.name = "s_time_local"
    return working


@st.cache_data(show_spinner=False)
def _metric_series_cached(
    cache_key: str,
    df: pd.DataFrame,
    metric_key: str,
    tz_name: str,
    rule: Optional[str],
    agg: str,
) -> pd.DataFrame:
    del cache_key
    return _metric_series(df, metric_key, tz_name, rule, agg)


def main() -> None:
    st.set_page_config(page_title="HomeSky", layout="wide")
    try:
        config = ingest.load_config()
    except FileNotFoundError as exc:
        st.error(
            "HomeSky configuration not found. The dashboard now uses the same loader as the GUI."
        )
        expected_path = ingest.get_config_path()
        st.info(
            f"Ensure a valid config exists at `{expected_path}` or run tools/ensure_config.ps1 to generate one."
        )
        st.exception(exc)
        st.stop()
    except Exception as exc:  # pragma: no cover - defensive guard for UI feedback
        st.error("Unable to load HomeSky configuration.")
        st.exception(exc)
        st.stop()

    _ensure_logging(config)

    config_path = getattr(ingest.load_config, "last_path", None)
    if config_path is not None:
        st.caption(f"Config: {config_path}")
    else:
        st.caption("Config: using default search path")

    theme_choice = config.get("visualization", {}).get("theme", "dark")
    theme = get_theme(theme_choice)
    typography = load_typography()

    storage_cfg = config.get("storage", {})
    sqlite_path = storage_cfg.get("sqlite_path", "./data/homesky.sqlite")
    parquet_path = storage_cfg.get("parquet_path", "./data/homesky.parquet")
    token = _cache_token(sqlite_path, parquet_path)
    try:
        df = load_data(sqlite_path, parquet_path, token)
    except Exception as exc:  # pragma: no cover - surfaced via Streamlit UI
        st.error("Unable to load stored observations.")
        st.exception(exc)
        st.stop()

    if df.empty:
        st.info("No observations stored yet. Run ingest.py to begin capturing data.")
        st.stop()

    df = compute_all_derived(df, config)

    tz_name = str(config.get("timezone", {}).get("local_tz") or "UTC")
    df = _prepare_time_columns(df, tz_name)
    df_time = ensure_time_index(df, tz_name)

    if df_time.empty:
        st.info("No observations available to display yet.")
        st.stop()

    latest_ts = df_time.index.max() if not df_time.empty else None

    st.markdown(
        f"<style>body {{ background-color: {theme.background}; color: {theme.text}; }}"
        f".stApp {{ font-family: {typography['font_family']}; }}</style>",
        unsafe_allow_html=True,
    )

    st.title("HomeSky Dashboard")

    st.markdown(
        f"<div style='display:inline-block;padding:0.35rem 0.75rem;border-radius:999px;"
        f"background-color:{theme.primary};color:white;font-weight:600;'>"
        f"Latest: {_format_timestamp(latest_ts, tz_name)} • {len(df):,} rows"  # noqa: E501
        "</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("Controls")
    if st.sidebar.button("Rebuild dashboard cache"):
        load_data.clear()
        arrow_dir_setting = storage_cfg.get("arrow_cache_dir")
        if arrow_dir_setting:
            arrow_cache_dir = Path(arrow_dir_setting).expanduser()
        else:
            root_dir = Path(storage_cfg.get("root_dir", "./data")).expanduser()
            arrow_cache_dir = root_dir / "arrow_cache"
        if arrow_cache_dir.exists():
            for feather_file in arrow_cache_dir.glob("*.feather"):
                try:
                    feather_file.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass
        st.experimental_rerun()

    metric_options = _available_metrics(df)
    if not metric_options:
        st.error("No numeric columns available to plot.")
        st.stop()
    metric_labels = [label for label, _ in metric_options]
    metric_lookup = {label: column for label, column in metric_options}

    default_metric = config.get("visualization", {}).get("default_metric")
    default_index = 0
    found_metric = False
    if default_metric is not None:
        for idx, (_, column) in enumerate(metric_options):
            if column == default_metric:
                default_index = idx
                found_metric = True
                break
    if not found_metric:
        for fallback in ("temp_f", "tempf", "temperature", "feels_like_f", "feelslike_f"):
            for idx, (_, column) in enumerate(metric_options):
                if column == fallback:
                    default_index = idx
                    found_metric = True
                    break
            if found_metric:
                break
    if not found_metric:
        for fallback in ("temp_f", "tempf", "temperature", "feels_like_f", "feelslike_f"):
            for idx, (_, column) in enumerate(metric_options):
                if column == fallback:
                    default_index = idx
                    found_metric = True
                    break
            if found_metric:
                break

    metric_state_key = "homesky_metric_label"
    if metric_state_key in st.session_state and st.session_state[metric_state_key] in metric_labels:
        default_index = metric_labels.index(st.session_state[metric_state_key])
    else:
        st.session_state[metric_state_key] = metric_labels[default_index]

    metric_state_key = "homesky_metric_label"
    if metric_state_key in st.session_state and st.session_state[metric_state_key] in metric_labels:
        default_index = metric_labels.index(st.session_state[metric_state_key])
    else:
        st.session_state[metric_state_key] = metric_labels[default_index]

    metric_state_key = "homesky_metric_label"
    if metric_state_key in st.session_state and st.session_state[metric_state_key] in metric_labels:
        default_index = metric_labels.index(st.session_state[metric_state_key])
    else:
        st.session_state[metric_state_key] = metric_labels[default_index]

    metric_label = st.sidebar.selectbox(
        "Metric", metric_labels, index=default_index, key=metric_state_key
    )
    metric_column = metric_lookup[metric_label]

    resample_keys = [option[0] for option in RESAMPLE_UI]
    resample_labels = {key: label for key, _, label in RESAMPLE_UI}
    configured_resample = config.get("visualization", {}).get("default_resample", "raw")
    if isinstance(configured_resample, str):
        configured_resample = configured_resample.strip().lower()
    default_resample_key = configured_resample if configured_resample in resample_keys else "raw"
    resample_index = resample_keys.index(default_resample_key)
    resample_state_key = "homesky_resample"
    if resample_state_key in st.session_state and st.session_state[resample_state_key] in resample_keys:
        resample_index = resample_keys.index(st.session_state[resample_state_key])
    else:
        st.session_state[resample_state_key] = resample_keys[resample_index]
    selected_resample_key = st.sidebar.selectbox(
        "Resample",
        options=resample_keys,
        index=resample_index,
        format_func=lambda key: resample_labels.get(key, key),
        key=resample_state_key,
    )
    resample_value = next(value for key, value, _ in RESAMPLE_UI if key == selected_resample_key)

    default_aggregate = config.get("visualization", {}).get("default_aggregate", "mean")
    aggregate_index = (
        AGGREGATE_OPTIONS.index(default_aggregate)
        if default_aggregate in AGGREGATE_OPTIONS
        else 0
    )
    aggregate_state_key = "homesky_aggregate"
    if (
        aggregate_state_key in st.session_state
        and st.session_state[aggregate_state_key] in AGGREGATE_OPTIONS
    ):
        aggregate_index = AGGREGATE_OPTIONS.index(st.session_state[aggregate_state_key])
    else:
        st.session_state[aggregate_state_key] = AGGREGATE_OPTIONS[aggregate_index]
    aggregate = st.sidebar.selectbox(
        "Aggregate", AGGREGATE_OPTIONS, index=aggregate_index, key=aggregate_state_key
    )

    fill_state_key = "homesky_fill_under_line"
    fill_default = _is_rain_column(metric_column)
    if st.session_state.get("_homesky_fill_metric") != metric_column:
        st.session_state["_homesky_fill_metric"] = metric_column
        st.session_state[fill_state_key] = fill_default
    fill_under_line = st.sidebar.checkbox("Fill under line", key=fill_state_key)

    min_ts = df_time.index.min()
    max_ts = df_time.index.max()
    default_end = max_ts
    default_start = max(default_end - timedelta(days=30), min_ts)
    date_state_key = "homesky_date_range"
    default_dates = (default_start.date(), default_end.date())
    if date_state_key not in st.session_state:
        st.session_state[date_state_key] = default_dates
    date_range = st.sidebar.date_input(
        "Date range",
        value=st.session_state[date_state_key],
        min_value=min_ts.date(),
        max_value=max_ts.date(),
        key=date_state_key,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    st.session_state[date_state_key] = (start_date, end_date)

    zone = _get_zone(tz_name)
    requested_start = _safe_localize_day(start_date, zone)
    requested_end = _safe_localize_day(end_date, zone) + timedelta(days=1) - timedelta(milliseconds=1)

    if requested_start > max_ts:
        st.warning("Selected window is in the future; no points yet.")
        st.toast("Selected window has no data points.")
        st.stop()
    if requested_end < min_ts:
        st.warning("Selected window is before available data.")
        st.toast("Selected window has no data points.")
        st.stop()

    start_ts = max(requested_start, min_ts)
    end_ts = min(requested_end, max_ts)

    mask = (df_time.index >= start_ts) & (df_time.index <= end_ts)
    filtered_time = df_time.loc[mask].copy()
    filtered = ensure_time_column(filtered_time)

    if filtered_time.empty:
        st.toast("No data points in the chosen range.")
        st.info("No data available for the selected range/metric.")
        st.stop()

    chart_cache_key = "|".join(
        [
            metric_column,
            resample_value or "raw",
            aggregate.lower(),
            start_ts.isoformat(),
            end_ts.isoformat(),
        ]
    )
    prepared = _metric_series_cached(
        chart_cache_key,
        filtered_time,
        metric_column,
        tz_name,
        resample_value,
        aggregate,
    )

    if prepared.empty:
        st.toast("No data points after resampling/aggregation.")
        st.info("No data available after applying resample/aggregate.")
        st.stop()

    column_name = prepared.columns[0]
    stats_series = prepared[column_name]

    rain_total, rain_column = _compute_rainfall(filtered)
    rain_display = _format_inches(rain_total)

    stats_cols = st.columns(5)
    stats_cols[0].metric("Min", _format_stat_value(stats_series.min(), column_name))
    stats_cols[1].metric("Mean", _format_stat_value(stats_series.mean(), column_name))
    stats_cols[2].metric("Max", _format_stat_value(stats_series.max(), column_name))
    stats_cols[3].metric("Rain Total", rain_display)
    if rain_column:
        stats_cols[3].markdown(
            f"<small>Rain metric: <code>{rain_column}</code></small>",
            unsafe_allow_html=True,
        )
    else:
        stats_cols[3].markdown(
            "<small>Rain metric: n/a</small>",
            unsafe_allow_html=True,
        )
    stats_cols[4].metric(
        "Last Observation", _format_timestamp(filtered_time.index.max(), tz_name)
    )

    tz_abbr = tz_name
    if isinstance(filtered_time.index, pd.DatetimeIndex) and len(filtered_time.index):
        midpoint = filtered_time.index[int(len(filtered_time.index) * 0.5)]
        try:
            localized_mid = midpoint.tz_convert(zone) if midpoint.tzinfo else midpoint.tz_localize(zone)
        except Exception:
            localized_mid = pd.Timestamp(midpoint).tz_localize(zone, ambiguous="NaT", nonexistent="shift_forward")
            if pd.isna(localized_mid):
                localized_mid = pd.Timestamp(midpoint).tz_localize(zone, ambiguous=True, nonexistent="shift_forward")
        tz_candidate = localized_mid.tzname() if localized_mid is not None else None
        if tz_candidate:
            tz_abbr = tz_candidate
    axis_title = f"Time ({tz_abbr})"

    prepared_reset = prepared.reset_index()
    axis_kwargs: Dict[str, object] = {"title": axis_title}
    if resample_value in {"D", "W"}:
        axis_kwargs["format"] = "%b %d"
        axis_kwargs["tickCount"] = 10
    elif resample_value == "M":
        axis_kwargs["format"] = "%b"
        axis_kwargs["tickCount"] = 12

    base_chart = alt.Chart(prepared_reset).encode(
        x=alt.X("s_time_local:T", axis=alt.Axis(**axis_kwargs)),
        y=alt.Y(f"{column_name}:Q", title=metric_label, scale=alt.Scale(zero=False, nice=True)),
        tooltip=[
            alt.Tooltip("s_time_local:T", title="Time"),
            alt.Tooltip(f"{column_name}:Q", title=metric_label),
        ],
    )
    line_chart = base_chart.mark_line(point=False)
    if fill_under_line:
        area_chart = base_chart.mark_area(opacity=0.25)
        chart = (area_chart + line_chart).properties(height=320).interactive()
    else:
        chart = line_chart.properties(height=320).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Metric column: `{metric_column}`")

    st.subheader("Rain — Year to Date vs Normal")
    daily_rain, daily_rain_column = _daily_rainfall(df_time)
    event_column = _resolve_column(df, "event_rain_in", "rain_event_in")
    normals_monthly, normals_error = _monthly_normals_from_config(config)
    if normals_error:
        st.warning(normals_error)
    if daily_rain.empty:
        st.info("No rain totals available. Add rain metrics to see cumulative comparisons.")
    else:
        ytd_end = filtered_time.index.max()
        start_of_year = pd.Timestamp(year=ytd_end.year, month=1, day=1, tz=zone)
        ytd_mask = (daily_rain.index >= start_of_year) & (
            daily_rain.index <= ytd_end.normalize()
        )
        ytd_daily = daily_rain.loc[ytd_mask]
        if ytd_daily.empty:
            st.info("No rainfall recorded for the selected year yet.")
        else:
            actual_total = float(ytd_daily.sum())
            normals_series = (
                _daily_normals_for_year(normals_monthly, ytd_end.year, zone)
                if normals_monthly
                else pd.Series(dtype="float64")
            )
            normal_total = float("nan")
            normal_cumulative = None
            if not normals_series.empty:
                normals_to_date = normals_series.loc[: ytd_end.normalize()]
                normal_total = float(normals_to_date.sum())
                normal_cumulative = normals_to_date.reindex(ytd_daily.index, fill_value=0).cumsum()
            actual_cumulative = ytd_daily.cumsum()

            rain_cards = st.columns(3)
            rain_cards[0].metric("YTD total", _format_inches(actual_total))
            if normal_cumulative is not None:
                rain_cards[1].metric("Normal to date", _format_inches(normal_total))
                departure = actual_total - normal_total
                departure_color = "#2e8540" if departure >= 0 else "#b31b1b"
                rain_cards[2].markdown(
                    "<div style='padding:0.5rem;border-radius:0.5rem;text-align:center;"
                    f"background-color:{departure_color};color:white;font-weight:600;'>"
                    f"Departure {departure:+.1f} in"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                rain_cards[1].info("Add NOAA normals to compare (see Settings)")
                rain_cards[2].empty()

            ytd_records: List[Dict[str, object]] = []
            for date, value in actual_cumulative.items():
                ytd_records.append({"date": date, "Series": "Actual", "value": value})
            if normal_cumulative is not None:
                for date, value in normal_cumulative.items():
                    ytd_records.append({"date": date, "Series": "Normal", "value": value})
            ytd_chart_df = pd.DataFrame(ytd_records)
            if not ytd_chart_df.empty:
                ytd_chart = (
                    alt.Chart(ytd_chart_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", axis=alt.Axis(title="Date")),
                        y=alt.Y(
                            "value:Q",
                            title="Cumulative rain (in)",
                            scale=alt.Scale(nice=True),
                        ),
                        color=alt.Color("Series:N", title="Series"),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date"),
                            alt.Tooltip("value:Q", title="Rain (in)"),
                            alt.Tooltip("Series:N", title="Series"),
                        ],
                    )
                    .properties(height=320)
                )
                events_df = _top_rain_events(df_time.loc[start_of_year:end_ts], event_column)
                if not events_df.empty:
                    events_df["date"] = events_df["s_time_local"].dt.floor("D")
                    events_df["cumulative"] = (
                        actual_cumulative.reindex(events_df["date"], method="ffill").to_numpy()
                    )
                    events_layer = (
                        alt.Chart(events_df)
                        .mark_point(size=80, color=theme.accent)
                        .encode(
                            x="date:T",
                            y="cumulative:Q",
                            tooltip=[
                                alt.Tooltip("date:T", title="Event"),
                                alt.Tooltip("amount:Q", title="Rain (in)"),
                            ],
                        )
                    )
                    ytd_chart = ytd_chart + events_layer
                st.altair_chart(ytd_chart, use_container_width=True)
            rain_caption_source = daily_rain_column or event_column or "n/a"
            if rain_caption_source == "n/a":
                st.caption("Rain column: n/a")
            else:
                st.caption(f"Rain column: `{rain_caption_source}`")

            year_options = sorted(daily_rain.index.year.unique().tolist())
            rain_year_key = "homesky_rain_year"
            default_year = int(ytd_end.year)
            if (
                rain_year_key not in st.session_state
                or st.session_state[rain_year_key] not in year_options
            ):
                fallback_year = default_year if default_year in year_options else year_options[-1]
                st.session_state[rain_year_key] = fallback_year
            year_index = year_options.index(st.session_state[rain_year_key])
            selected_year = st.selectbox(
                "Rain year",
                year_options,
                index=year_index,
                key=rain_year_key,
            )
            year_start = pd.Timestamp(year=selected_year, month=1, day=1, tz=zone)
            year_stop = pd.Timestamp(year=selected_year, month=12, day=31, tz=zone)
            yearly_rain = daily_rain.loc[
                (daily_rain.index >= year_start) & (daily_rain.index <= year_stop)
            ]
            yearly_df = df_time.loc[(df_time.index >= year_start) & (df_time.index <= year_stop)]

            rain_cols = st.columns(2)
            with rain_cols[0]:
                st.markdown("**Monthly totals**")
                if yearly_rain.empty:
                    st.info("No rainfall recorded for the selected year.")
                else:
                    event_daily = pd.Series(False, index=yearly_rain.index)
                    if event_column:
                        event_series = (
                            pd.to_numeric(df_time[event_column], errors="coerce").fillna(0.0)
                        )
                        event_daily_series = event_series.resample("D").max()
                        event_daily = event_daily_series.reindex(yearly_rain.index, fill_value=0) > 0
                    monthly_frame = pd.DataFrame(
                        {
                            "date": yearly_rain.index,
                            "rain": yearly_rain.values,
                            "category": [
                                "Event day" if flag else "Other day" for flag in event_daily
                            ],
                        }
                    )
                    monthly_frame["month"] = (
                        monthly_frame["date"].dt.to_period("M").dt.to_timestamp()
                    )
                    monthly_totals = (
                        monthly_frame.groupby(["month", "category"], as_index=False)["rain"].sum()
                    )
                    month_chart = (
                        alt.Chart(monthly_totals)
                        .mark_bar()
                        .encode(
                            x=alt.X("month:T", axis=alt.Axis(title="Month", format="%b")),
                            y=alt.Y("rain:Q", axis=alt.Axis(title="Rain (in)")),
                            color=alt.Color("category:N", title="Day type"),
                            tooltip=[
                                alt.Tooltip("month:T", title="Month"),
                                alt.Tooltip("rain:Q", title="Rain (in)"),
                                alt.Tooltip("category:N", title="Day type"),
                            ],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(month_chart, use_container_width=True)

            with rain_cols[1]:
                st.markdown("**Hourly intensity**")
                hist_df, hist_column = _rain_rate_histogram(yearly_df)
                if hist_df.empty:
                    st.info("No rain rate observations for this year.")
                else:
                    hist_chart = (
                        alt.Chart(hist_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "bucket:N",
                                title="Rain rate (in/hr)",
                                sort=list(hist_df["bucket"]),
                            ),
                            y=alt.Y("count:Q", title="Hours"),
                            tooltip=[
                                alt.Tooltip("bucket:N", title="Rain rate"),
                                alt.Tooltip("count:Q", title="Hours"),
                            ],
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(hist_chart, use_container_width=True)
                    if hist_column:
                        st.caption(f"Intensity column: `{hist_column}`")

            st.markdown("**Biggest rain days**")
            if yearly_rain.empty:
                st.info("No rain days to summarise for the selected year.")
            else:
                temp_column = _resolve_column(df_time, "temp_f", "tempf", "temperature")
                feels_column = _resolve_column(df_time, "feels_like_f", "feelslike_f")
                top_days = yearly_rain.sort_values(ascending=False).head(10)
                table_rows: List[Dict[str, str]] = []
                temp_min = temp_max = temp_median = None
                if temp_column:
                    temp_series = pd.to_numeric(yearly_df[temp_column], errors="coerce")
                    temp_min = temp_series.resample("D").min()
                    temp_max = temp_series.resample("D").max()
                    temp_median = temp_series.resample("D").median()
                if feels_column:
                    feels_series = pd.to_numeric(yearly_df[feels_column], errors="coerce")
                    temp_median = feels_series.resample("D").median()
                for date, amount in top_days.items():
                    min_val = temp_min.loc[date] if temp_min is not None and date in temp_min.index else float("nan")
                    max_val = temp_max.loc[date] if temp_max is not None and date in temp_max.index else float("nan")
                    median_val = (
                        temp_median.loc[date]
                        if temp_median is not None and date in temp_median.index
                        else float("nan")
                    )
                    table_rows.append(
                        {
                            "Date": date.strftime("%Y-%m-%d"),
                            "Rain (in)": _format_inches(amount),
                            "Min temp": _format_temperature(min_val),
                            "Median temp": _format_temperature(median_val),
                            "Max temp": _format_temperature(max_val),
                        }
                    )
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    st.subheader("Daily temperature bands")
    band_window_key = "homesky_temp_band_window"
    window_options = [7, 14, 30]
    if band_window_key not in st.session_state:
        st.session_state[band_window_key] = window_options[0]
    band_days = st.selectbox(
        "Window",
        options=window_options,
        format_func=lambda days: f"{days} days",
        key=band_window_key,
    )
    band_end = filtered_time.index.max()
    band_start = band_end - pd.Timedelta(days=band_days - 1)
    band_mask = (filtered_time.index >= band_start) & (filtered_time.index <= band_end)
    band_df_time = filtered_time.loc[band_mask]
    band_df = ensure_time_column(band_df_time)
    temp_column = _resolve_column(band_df, "temp_f", "tempf", "temperature")
    feels_column = _resolve_column(band_df, "feels_like_f", "feelslike_f")
    if not temp_column:
        st.info("No temperature column available for band view.")
    else:
        temp_series = pd.to_numeric(band_df_time[temp_column], errors="coerce")
        daily_min = temp_series.resample("D").min()
        daily_max = temp_series.resample("D").max()
        if feels_column:
            feels_series = pd.to_numeric(band_df_time[feels_column], errors="coerce")
            daily_mean = feels_series.resample("D").mean()
        else:
            daily_mean = temp_series.resample("D").mean()
        bands = pd.DataFrame(
            {
                "date": daily_min.index,
                "temp_min": daily_min,
                "temp_max": daily_max,
                "temp_mean": daily_mean,
            }
        ).dropna()
        if bands.empty:
            st.info("Not enough temperature data for the selected window.")
        else:
            band_source = bands.reset_index(drop=True)
            band_chart = (
                alt.Chart(band_source)
                .mark_rule(color=theme.accent, size=6)
                .encode(
                    x=alt.X("date:T", axis=alt.Axis(title="Date")),
                    y=alt.Y("temp_min:Q", title="Temperature (°F)"),
                    y2="temp_max:Q",
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("temp_min:Q", title="Min (°F)"),
                        alt.Tooltip("temp_mean:Q", title="Mean (°F)"),
                        alt.Tooltip("temp_max:Q", title="Max (°F)"),
                    ],
                )
            )
            mean_points = (
                alt.Chart(band_source)
                .mark_point(color=theme.primary, size=90)
                .encode(
                    x="date:T",
                    y="temp_mean:Q",
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("temp_mean:Q", title="Mean (°F)"),
                    ],
                )
            )
            st.altair_chart((band_chart + mean_points).properties(height=320), use_container_width=True)
            st.caption(
                f"Temperature columns: `{temp_column}`"
                + (f", feels like `{feels_column}`" if feels_column else "")
            )

    explorer_df = prepared
    st.subheader("Data Explorer")
    display_df = explorer_df.reset_index()
    st.dataframe(display_df, use_container_width=True)

    csv_buffer = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv_buffer,
        file_name="homesky_filtered.csv",
        mime="text/csv",
    )

    value_columns: List[str] = [column_name]
    if rain_column:
        value_columns.append(rain_column)

    sanitized = sanitize_for_arrow(filtered, tz_name=tz_name, value_columns=value_columns)
    parquet_buffer = BytesIO()
    sanitized.to_parquet(parquet_buffer, engine="pyarrow")
    st.download_button(
        "Download Parquet",
        parquet_buffer.getvalue(),
        file_name="homesky.parquet",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
