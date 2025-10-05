"""Streamlit dashboard for HomeSky."""

from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.api import types as ptypes
import streamlit as st

import ingest
from utils.charts import build_metric_chart, prepare_timeseries
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
    return localized.strftime("%Y-%m-%d %H:%M %Z")


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
    zone = _get_zone(tz_name)
    working = df.copy()

    if isinstance(working.index, pd.DatetimeIndex):
        idx = working.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        working["s_time_utc"] = idx
    else:
        working["s_time_utc"] = pd.NaT

    for candidate in ("s_time_utc", "observed_at", "obs_time_utc", "timestamp_utc", "dateutc"):
        if candidate in working.columns:
            utc_series = pd.to_datetime(working[candidate], errors="coerce", utc=True)
            mask = utc_series.notna()
            working.loc[mask, "s_time_utc"] = utc_series.loc[mask]

    if working["s_time_utc"].isna().any() and "epoch_ms" in working.columns:
        epoch_dt = pd.to_datetime(pd.to_numeric(working["epoch_ms"], errors="coerce"), unit="ms", errors="coerce", utc=True)
        mask = epoch_dt.notna() & working["s_time_utc"].isna()
        working.loc[mask, "s_time_utc"] = epoch_dt.loc[mask]

    working = working.dropna(subset=["s_time_utc"]).copy()

    local_candidates = None
    for name in ("s_time_local", "timestamp_local", "obs_time_local"):
        if name in working.columns:
            local_candidates = pd.to_datetime(working[name], errors="coerce")
            if local_candidates.notna().any():
                break
    if local_candidates is not None:
        if getattr(local_candidates.dtype, "tz", None) is None:
            try:
                local_series = local_candidates.dt.tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                local_series = working["s_time_utc"].dt.tz_convert(zone)
        else:
            try:
                local_series = local_candidates.dt.tz_convert(zone)
            except Exception:
                local_series = working["s_time_utc"].dt.tz_convert(zone)
    else:
        local_series = working["s_time_utc"].dt.tz_convert(zone)

    working["s_time_local"] = local_series
    working = working.sort_values("s_time_local")
    working = working.set_index("s_time_local", drop=False)
    return working


def _latest_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.iloc[-1])


def _format_temperature(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1f} °F"


def _format_inches(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f} in"


def _format_speed(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1f} mph"


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
    keep: List[str] = []
    for name in ("mac", "epoch_ms", "s_time_local", "s_time_utc", "dateutc"):
        if name in df.columns:
            keep.append(name)
    if value_columns is not None:
        keep.extend(col for col in value_columns if col in df.columns)
    keep = list(dict.fromkeys(keep))  # preserve order, drop duplicates
    sanitized = df.loc[:, keep].copy() if keep else df.copy()

    zone = _get_zone(tz_name)
    if "s_time_local" in sanitized.columns:
        local_series = pd.to_datetime(sanitized["s_time_local"], errors="coerce")
    else:
        local_series = pd.to_datetime(df.index, errors="coerce")
    if getattr(local_series.dtype, "tz", None) is None:
        try:
            local_series = local_series.dt.tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            local_series = local_series.dt.tz_localize(zone, nonexistent="shift_forward")
    else:
        try:
            local_series = local_series.dt.tz_convert(zone)
        except Exception:
            local_series = local_series.dt.tz_localize(zone, nonexistent="shift_forward")
    sanitized["s_time_local"] = local_series

    if "s_time_utc" in sanitized.columns:
        utc_series = pd.to_datetime(sanitized["s_time_utc"], errors="coerce", utc=True)
    else:
        utc_series = sanitized["s_time_local"].dt.tz_convert("UTC")
    sanitized["s_time_utc"] = utc_series

    if "dateutc" in sanitized.columns:
        sanitized["dateutc"] = pd.to_datetime(sanitized["dateutc"], errors="coerce", utc=True)

    if "epoch_ms" in sanitized.columns:
        sanitized["epoch_ms"] = pd.to_numeric(sanitized["epoch_ms"], errors="coerce").astype("Int64")

    drop_candidates = [
        "observed_at",
        "obs_time_utc",
        "obs_time_local",
        "timestamp_local",
        "timestamp_utc",
    ]
    sanitized = sanitized.drop(columns=[c for c in drop_candidates if c in sanitized.columns], errors="ignore")

    for column in sanitized.select_dtypes(include="object").columns:
        converted = pd.to_numeric(sanitized[column], errors="ignore")
        sanitized[column] = converted
        if sanitized[column].dtype == "object":
            sanitized[column] = sanitized[column].astype("string")

    if "mac" in sanitized.columns:
        sanitized["mac"] = sanitized["mac"].astype("string")

    sanitized = sanitized.dropna(subset=["s_time_local"]).sort_values("s_time_local")
    sanitized.reset_index(drop=True, inplace=True)
    return sanitized


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

    latest_ts = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None

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
        st.experimental_rerun()

    metric_options = _available_metrics(df)
    if not metric_options:
        st.error("No numeric columns available to plot.")
        st.stop()
    metric_labels = [label for label, _ in metric_options]
    metric_lookup = {label: column for label, column in metric_options}

    default_metric = config.get("visualization", {}).get("default_metric")
    default_index = 0
    if default_metric is not None:
        for idx, (_, column) in enumerate(metric_options):
            if column == default_metric:
                default_index = idx
                break

    metric_label = st.sidebar.selectbox("Metric", metric_labels, index=default_index)
    metric_column = metric_lookup[metric_label]

    resample_keys = [option[0] for option in RESAMPLE_UI]
    resample_labels = {key: label for key, _, label in RESAMPLE_UI}
    configured_resample = config.get("visualization", {}).get("default_resample", "raw")
    if isinstance(configured_resample, str):
        configured_resample = configured_resample.strip().lower()
    default_resample_key = configured_resample if configured_resample in resample_keys else "raw"
    resample_index = resample_keys.index(default_resample_key)
    selected_resample_key = st.sidebar.selectbox(
        "Resample",
        options=resample_keys,
        index=resample_index,
        format_func=lambda key: resample_labels.get(key, key),
    )
    resample_value = next(value for key, value, _ in RESAMPLE_UI if key == selected_resample_key)

    default_aggregate = config.get("visualization", {}).get("default_aggregate", "mean")
    aggregate_index = (
        AGGREGATE_OPTIONS.index(default_aggregate)
        if default_aggregate in AGGREGATE_OPTIONS
        else 0
    )
    aggregate = st.sidebar.selectbox("Aggregate", AGGREGATE_OPTIONS, index=aggregate_index)

    min_ts = df.index.min()
    max_ts = df.index.max()
    default_end = max_ts
    default_start = max(default_end - timedelta(days=30), min_ts)
    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start.date(), default_end.date()),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    zone = _get_zone(tz_name)
    start_ts = pd.Timestamp(start_date).tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
    end_ts = pd.Timestamp(end_date).tz_localize(zone, ambiguous="infer", nonexistent="shift_forward") + timedelta(days=1)
    start_ts = max(start_ts, min_ts)
    end_ts = min(end_ts, max_ts + timedelta(seconds=1))

    mask = (df.index >= start_ts) & (df.index <= end_ts)
    filtered = df.loc[mask].copy()

    if filtered.empty:
        st.warning("No data available for the selected range/metric.")
        st.stop()

    chart_source = filtered.set_index("s_time_local")
    prepared = prepare_timeseries(chart_source, metric_column, resample_value, aggregate)

    if prepared.empty:
        st.warning("No data available after applying resample/aggregate.")
        st.stop()

    last_24h_start = filtered.index.max() - timedelta(hours=24)
    window_24h = filtered[filtered.index >= last_24h_start]

    temp_column = _resolve_column(filtered, "temp_f", "tempf", "temperature")
    current_temp = _format_temperature(_latest_numeric(filtered[temp_column])) if temp_column else "n/a"
    high_24h = _format_temperature(float(pd.to_numeric(window_24h[temp_column], errors="coerce").max())) if temp_column else "n/a"
    low_24h = _format_temperature(float(pd.to_numeric(window_24h[temp_column], errors="coerce").min())) if temp_column else "n/a"

    rain_total, rain_column = _compute_rainfall(window_24h)
    rain_display = _format_inches(rain_total)

    wind_speed_column = _resolve_column(filtered, "wind_speed_mph", "windspeedmph")
    wind_gust_column = _resolve_column(filtered, "wind_gust_mph", "windgustmph")
    wind_dir_column = _resolve_column(filtered, "wind_dir_deg", "winddir")
    wind_speed = _format_speed(_latest_numeric(filtered[wind_speed_column])) if wind_speed_column else "n/a"
    gust_value = _latest_numeric(filtered[wind_gust_column]) if wind_gust_column else float("nan")
    wind_gust = _format_speed(gust_value)
    wind_dir = _direction_to_cardinal(_latest_numeric(filtered[wind_dir_column])) if wind_dir_column else None
    wind_delta_parts = []
    if wind_gust != "n/a":
        wind_delta_parts.append(f"Gust {wind_gust}")
    if wind_dir:
        wind_delta_parts.append(wind_dir)
    wind_delta = " • ".join(wind_delta_parts) if wind_delta_parts else ""

    station_ids = (
        filtered.get("mac").dropna().astype("string").unique().tolist()
        if "mac" in filtered.columns
        else []
    )
    if not station_ids:
        source_label = ""
    elif len(station_ids) <= 2:
        source_label = ", ".join(station_ids)
    else:
        source_label = ", ".join(station_ids[:2]) + "…"

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Outdoor Temp", current_temp)
    kpi_cols[1].metric("24h High", high_24h)
    kpi_cols[2].metric("24h Low", low_24h)
    kpi_cols[3].metric("Rain (24h)", rain_display, delta=rain_column or "")
    kpi_cols[4].metric("Wind", wind_speed, delta=wind_delta or None)
    kpi_cols[5].metric("Last Update", _format_timestamp(filtered.index.max(), tz_name), delta=source_label or None)

    st.altair_chart(
        build_metric_chart(
            prepared,
            metric_column,
            theme,
            title=metric_label,
            timezone_label=tz_name,
        ),
        use_container_width=True,
    )

    explorer_df = prepared if resample_value else filtered[[c for c in filtered.columns if c != "raw"]]
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

    value_columns = {metric_column}
    if temp_column:
        value_columns.add(temp_column)
    if rain_column:
        value_columns.add(rain_column)
    if wind_speed_column:
        value_columns.add(wind_speed_column)
    if wind_gust_column:
        value_columns.add(wind_gust_column)
    if wind_dir_column:
        value_columns.add(wind_dir_column)

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
