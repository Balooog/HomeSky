"""Streamlit dashboard for HomeSky."""

from __future__ import annotations

import datetime
from datetime import date, timedelta
from io import BytesIO
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
from pandas.api import types as ptypes
import streamlit as st
import traceback

try:  # Package-aware import shim for Streamlit's execution context
    if __package__:
        from . import ingest  # type: ignore
        from .utils.db import parse_obs_times  # type: ignore
        from .utils.logging_setup import get_logger  # type: ignore
    else:  # pragma: no cover - Streamlit sets __package__ to None
        raise ImportError
except Exception:  # pragma: no cover - fallback for direct script execution
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from homesky import ingest  # type: ignore
    from homesky.utils.db import parse_obs_times  # type: ignore
    from homesky.utils.logging_setup import get_logger  # type: ignore

from homesky.rain_dashboard import compute_rainfall, render_rain_dashboard
from homesky.utils.config import get_station_tz
from homesky.utils.derived import compute_all_derived
from homesky.utils.logging import setup_streamlit_logging
from homesky.utils.theming import Theme, get_theme, load_typography


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
    ("H", "h", "Hourly"),
    ("D", "d", "Daily"),
    ("W", "w", "Weekly"),
    ("M", "m", "Monthly"),
)

AGGREGATE_OPTIONS: Sequence[str] = ("mean", "max", "min", "sum", "last")


STREAMLIT_LOG_PATH = Path("data/logs/streamlit_error.log")
RERUN_SENTINEL_KEY = "homesky_rerun_pending"


log = get_logger("streamlit")


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
    index_names: List[Optional[str]]
    if isinstance(working.index, pd.MultiIndex):
        index_names = list(working.index.names)
    else:
        index_names = [working.index.name]
    if "s_time_local" in working.columns and "s_time_local" in index_names:
        working = working.drop(columns=["s_time_local"], errors="ignore")

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
        local_series = pd.to_datetime(
            working["s_time_local"], errors="coerce", utc=False
        )
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


def _as_date(value: object, fallback: date) -> date:
    if isinstance(value, date):
        return value
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return fallback
    if pd.isna(ts):
        return fallback
    return ts.date()


def get_date_range(
    default_start: pd.Timestamp, default_end: pd.Timestamp, key: str = "homesky_date_range"
) -> Tuple[date, date]:
    """Initialize and return the stored date range for the dashboard controls."""

    default_tuple = (default_start.date(), default_end.date())
    if key not in st.session_state:
        st.session_state[key] = default_tuple
        return default_tuple

    stored = st.session_state.get(key, default_tuple)
    if isinstance(stored, (list, tuple)) and len(stored) == 2:
        start_raw, end_raw = stored
    else:
        start_raw = stored
        end_raw = stored

    start_date = _as_date(start_raw, default_tuple[0])
    end_date = _as_date(end_raw, default_tuple[1])
    return start_date, end_date


def _normalize_date_pair(start: date, end: date) -> Tuple[date, date]:
    return (start, end) if start <= end else (end, start)


def _record_streamlit_error(exc: BaseException) -> None:
    """Append structured crash details to the Streamlit error log."""

    STREAMLIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    with STREAMLIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {type(exc).__name__}: {exc}\n")
        handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        handle.write("\n")


def _trigger_streamlit_rerun() -> None:
    """Trigger a rerun without entering repeated reset loops."""

    if st.session_state.get(RERUN_SENTINEL_KEY):
        log.debug("Rerun already pending; skipping duplicate trigger")
        return
    st.session_state[RERUN_SENTINEL_KEY] = True
    try:
        st.rerun()
    except AttributeError as exc:
        log.warning("st.rerun() unavailable; attempting experimental fallback: %s", exc)
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:  # pragma: no cover - defensive guard for unexpected Streamlit API changes
            raise


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


def _default_window_last_10_days(tz_name: str) -> Tuple[date, date]:
    try:
        zone = ZoneInfo(tz_name)
    except Exception:
        zone = ZoneInfo("UTC")
    now = datetime.datetime.now(zone)
    start = (now - timedelta(days=10)).date()
    end = now.date()
    return start, end


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


def _is_pressure_column(column: str) -> bool:
    return "press" in column.lower()


def _init_widget_state(key: str, options: Sequence[Any], default_value: Any) -> Any:
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = default_value
    return st.session_state[key]


def _render_stat_card(
    container: "st.delta_generator.DeltaGenerator",
    *,
    icon: str,
    label: str,
    value: str,
    theme: Theme,
    footnote: Optional[str] = None,
) -> None:
    footnote_html = (
        f"<div style='font-size:0.72rem;color:{theme.muted_text};margin-top:0.35rem'>{footnote}</div>"
        if footnote
        else ""
    )
    container.markdown(
        "<div style='display:flex;align-items:center;background:"
        f"{theme.surface};border-radius:0.75rem;padding:0.85rem 1rem;box-shadow:0 8px 18px rgba(0,0,0,0.12);'>"
        f"<div style='font-size:1.5rem;margin-right:0.85rem'>{icon}</div>"
        "<div style='display:flex;flex-direction:column;'>"
        f"<span style='font-size:0.85rem;text-transform:uppercase;color:{theme.muted_text};letter-spacing:0.08em'>{label}</span>"
        f"<span style='font-size:1.55rem;font-weight:600;color:{theme.text};margin-top:0.2rem'>{value}</span>"
        f"{footnote_html}"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _metric_visual_style(metric_column: str, theme: Theme) -> Tuple[str, Optional[alt.Gradient]]:
    if _is_temperature_column(metric_column):
        gradient = alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#2563eb", offset=0.0),
                alt.GradientStop(color="#f8fafc", offset=0.5),
                alt.GradientStop(color="#dc2626", offset=1.0),
            ],
            x1=0,
            x2=0,
            y1=1,
            y2=0,
        )
        return "#f97316", gradient
    if _is_rain_column(metric_column):
        gradient = alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#0f172a", offset=0.0),
                alt.GradientStop(color="#1d4ed8", offset=0.4),
                alt.GradientStop(color="#60a5fa", offset=1.0),
            ],
            x1=0,
            x2=0,
            y1=1,
            y2=0,
        )
        return "#38bdf8", gradient
    if _is_pressure_column(metric_column):
        gradient = alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#0e7490", offset=0.0),
                alt.GradientStop(color="#cffafe", offset=0.7),
                alt.GradientStop(color="#0891b2", offset=1.0),
            ],
            x1=0,
            x2=0,
            y1=1,
            y2=0,
        )
        return "#06b6d4", gradient
    return theme.primary, None


def _infer_tz_abbreviation(index: pd.DatetimeIndex, zone: ZoneInfo) -> str:
    """Return a friendly timezone abbreviation for *index* clamping lookups safely."""

    def _fallback() -> str:
        zone_name = str(zone)
        return zone_name.split("/")[-1] if "/" in zone_name else zone_name

    def _normalize_abbr(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return "ET" if value in {"EDT", "EST"} else value

    def _tzname_for(timestamp: pd.Timestamp) -> Optional[str]:
        if pd.isna(timestamp):
            return None
        tzinfo = getattr(timestamp, "tz", None) or getattr(timestamp, "tzinfo", None)
        try:
            if tzinfo is not None:
                name = tzinfo.tzname(timestamp)
                if name:
                    return name
            localized = timestamp.tz_localize(
                zone, ambiguous="NaT", nonexistent="shift_forward"
            )
        except Exception:
            try:
                localized = timestamp.tz_convert(zone)
            except Exception:
                return None
        if pd.isna(localized):
            return None
        return localized.tzname()

    try:
        size = len(index)
        if size == 0:
            now = pd.Timestamp.now(tz=zone)
            return _normalize_abbr(now.tzname()) or _fallback()
        if size == 1:
            name = _tzname_for(index[0])
            return _normalize_abbr(name) or _fallback()

        samples: List[pd.Timestamp] = []
        for frac in (0.0, 0.5, 1.0):
            pos = int((size - 1) * frac)
            pos = max(0, min(size - 1, pos))
            samples.append(index[pos])
        for ts in samples:
            name = _tzname_for(ts)
            normalized = _normalize_abbr(name)
            if normalized:
                return normalized
    except Exception:
        pass
    return _fallback()


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
        sanitized_reset["s_time_local"], errors="coerce", utc=True
    ).dt.tz_convert(zone)
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
        try:
            converted = pd.to_numeric(sanitized_reset[column])
        except (TypeError, ValueError):
            sanitized_reset[column] = sanitized_reset[column].astype("string")
            continue
        sanitized_reset[column] = converted
        if sanitized_reset[column].dtype == "object":
            sanitized_reset[column] = sanitized_reset[column].astype("string")

    if "mac" in sanitized_reset.columns:
        sanitized_reset["mac"] = sanitized_reset["mac"].astype("string")

    sanitized_reset = sanitized_reset.dropna(subset=["s_time_local"]).reset_index(drop=True)
    return sanitized_reset


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


def _run_dashboard() -> None:
    st.set_page_config(page_title="HomeSky", layout="wide")
    if RERUN_SENTINEL_KEY not in st.session_state:
        st.session_state[RERUN_SENTINEL_KEY] = False
    elif st.session_state[RERUN_SENTINEL_KEY]:
        st.session_state[RERUN_SENTINEL_KEY] = False

    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
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

    storage_manager = ingest.get_storage_manager(config)
    ingest_cfg = config.get("ingest", {})
    auto_sync_enabled = bool(ingest_cfg.get("auto_sync_on_launch", True))
    auto_sync_done_key = "_homesky_auto_sync_done"
    if auto_sync_enabled and not st.session_state.get(auto_sync_done_key, False):
        with st.spinner("Checking for new observations…"):
            try:
                result = ingest.sync_new(config=config, storage=storage_manager)
            except Exception as exc:  # pragma: no cover - surfaced to UI
                st.session_state[auto_sync_done_key] = True
                st.session_state["_homesky_auto_sync_error"] = str(exc)
                log.warning("Auto-sync on launch failed: %s", exc)
            else:
                st.session_state[auto_sync_done_key] = True
                st.session_state["_homesky_auto_sync_result"] = result
                added = int(result.get("added", 0) or 0)
                if added > 0:
                    st.toast(f"Synced {added} new observations", icon="✅")
                else:
                    log.info("Auto-sync on launch: database already up-to-date")
    elif not auto_sync_enabled:
        st.session_state.setdefault(auto_sync_done_key, True)

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
    tz_name = get_station_tz(default="UTC")

    df_time: pd.DataFrame = pd.DataFrame()
    with st.status("Loading data…", expanded=False) as status:
        progress = st.progress(0)
        status.update(label="Reading stored observations…")
        try:
            df = load_data(sqlite_path, parquet_path, token)
        except Exception as exc:  # pragma: no cover - surfaced via Streamlit UI
            progress.progress(100)
            status.update(label="Unable to load stored observations", state="error")
            st.error("Unable to load stored observations.")
            st.exception(exc)
            st.stop()

        progress.progress(30)
        if df.empty:
            status.update(label="No observations stored yet", state="error")
            progress.progress(100)
            st.info("No observations stored yet. Run ingest.py to begin capturing data.")
            st.stop()

        status.update(label="Computing derived metrics…")
        try:
            df = compute_all_derived(df, config)
        except Exception as exc:  # pragma: no cover - surfaced via Streamlit UI
            progress.progress(100)
            status.update(label="Unable to compute derived metrics", state="error")
            st.error("Unable to compute derived metrics.")
            st.exception(exc)
            st.stop()

        progress.progress(60)
        status.update(label="Normalizing timestamps…")
        try:
            df = _prepare_time_columns(df, tz_name)
        except Exception as exc:  # pragma: no cover - surfaced via Streamlit UI
            progress.progress(100)
            status.update(label="Unable to normalize timestamps", state="error")
            st.error("Unable to normalize timestamps.")
            st.exception(exc)
            st.stop()

        progress.progress(85)
        df_time = ensure_time_index(df, tz_name)
        if df_time.empty:
            status.update(label="No observations available", state="error")
            progress.progress(100)
            st.info("No observations available to display yet.")
            st.stop()

        progress.progress(100)
        status.update(label="Data ready", state="complete")

    latest_ts = df_time.index.max() if not df_time.empty else None

    st.markdown(
        f"<style>body {{ background-color: {theme.background}; color: {theme.text}; }}"
        f".stApp {{ font-family: {typography['font_family']}; }}</style>",
        unsafe_allow_html=True,
    )

    st.title("HomeSky Dashboard")

    auto_sync_error = st.session_state.get("_homesky_auto_sync_error")
    auto_sync_result = st.session_state.get("_homesky_auto_sync_result")
    if auto_sync_error:
        st.warning(f"Auto-sync failed: {auto_sync_error}")
    elif auto_sync_result:
        added = int(auto_sync_result.get("added", 0) or 0)
        skipped = int(auto_sync_result.get("skipped", 0) or 0)
        since_raw = auto_sync_result.get("since")
        until_raw = auto_sync_result.get("until")
        since_ts = (
            pd.to_datetime(since_raw, utc=True, errors="coerce")
            if since_raw
            else None
        )
        until_ts = (
            pd.to_datetime(until_raw, utc=True, errors="coerce")
            if until_raw
            else None
        )
        summary_parts: List[str] = []
        if added > 0:
            summary_parts.append(f"added {added:,} rows")
        else:
            summary_parts.append("database already up-to-date")
        if skipped > 0:
            summary_parts.append(f"skipped {skipped:,}")
        if since_ts is not None and not pd.isna(since_ts):
            summary_parts.append(f"since {_format_timestamp(since_ts, tz_name)}")
        if until_ts is not None and not pd.isna(until_ts):
            summary_parts.append(f"latest {_format_timestamp(until_ts, tz_name)}")
        if summary_parts:
            st.caption("Auto-sync: " + " • ".join(summary_parts))

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
        _trigger_streamlit_rerun()
        return

    with st.sidebar.expander("Advanced ingestion", expanded=False):
        st.caption(
            "Backfill operations use the Ambient history API and may take several minutes."
        )
        show_backfill_controls = st.checkbox(
            "Show backfill controls",
            value=False,
            key="homesky_show_backfill_controls",
        )
        if show_backfill_controls:
            if st.button("Backfill last 24 hours", key="homesky_backfill_24h"):
                with st.spinner("Backfilling last 24 hours…"):
                    try:
                        inserted = ingest.backfill(
                            config,
                            24,
                            storage=storage_manager,
                            tz=tz_name,
                        )
                    except Exception as exc:  # pragma: no cover - surfaced to UI
                        st.error(f"Backfill failed: {exc}")
                        log.warning("Backfill (24h) failed: %s", exc)
                    else:
                        if inserted:
                            st.success(
                                f"Inserted {inserted} rows from the last 24 hours."
                            )
                            st.toast(f"Backfill added {inserted} rows", icon="✅")
                        else:
                            st.info("No new observations found in the last 24 hours.")
                        load_data.clear()
                        _trigger_streamlit_rerun()

        enable_custom_backfill = st.checkbox(
            "Enable custom backfill",
            value=False,
            key="homesky_enable_custom_backfill",
        )
        if enable_custom_backfill:
            default_start = date.today() - timedelta(days=7)
            default_end = date.today()
            custom_start = st.date_input(
                "Start date",
                value=default_start,
                key="homesky_custom_backfill_start",
            )
            custom_end = st.date_input(
                "End date",
                value=default_end,
                key="homesky_custom_backfill_end",
            )
            if st.button("Run custom backfill", key="homesky_custom_backfill_run"):
                start_date = _as_date(custom_start, default_start)
                end_date = _as_date(custom_end, default_end)
                start_date, end_date = _normalize_date_pair(start_date, end_date)
                mac_value = config.get("ambient", {}).get("mac") or ""
                if not mac_value:
                    st.error("Backfill requires a configured station MAC address.")
                else:
                    try:
                        from homesky.backfill import backfill_range
                    except ImportError as exc:  # pragma: no cover - compatibility
                        st.error("Backfill utilities are unavailable in this build.")
                        log.warning("Custom backfill unavailable: %s", exc)
                    else:
                        zone = _get_zone(tz_name)
                        start_ts = _safe_localize_day(pd.Timestamp(start_date), zone)
                        end_ts = (
                            _safe_localize_day(pd.Timestamp(end_date), zone)
                            + pd.Timedelta(days=1)
                            - pd.Timedelta(seconds=1)
                        )
                        with st.spinner("Running custom backfill…"):
                            try:
                                result = backfill_range(
                                    config=config,
                                    storage=storage_manager,
                                    start_dt=start_ts.tz_convert("UTC"),
                                    end_dt=end_ts.tz_convert("UTC"),
                                    mac=mac_value,
                                    limit_per_call=int(
                                        ingest_cfg.get("backfill_limit", 288) or 288
                                    ),
                                )
                            except Exception as exc:  # pragma: no cover - UI feedback
                                st.error(f"Custom backfill failed: {exc}")
                                log.warning("Custom backfill failed: %s", exc)
                            else:
                                inserted = int(result.inserted)
                                if inserted > 0:
                                    st.success(
                                        "Inserted {0:,} rows covering {1} – {2}.".format(
                                            inserted,
                                            _format_timestamp(result.start, tz_name),
                                            _format_timestamp(result.end, tz_name),
                                        )
                                    )
                                    st.toast(
                                        f"Backfill added {inserted} rows", icon="✅"
                                    )
                                    load_data.clear()
                                    _trigger_streamlit_rerun()
                                else:
                                    st.info(
                                        "No new data found for the selected backfill range."
                                    )

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
    selected_metric_label = _init_widget_state(
        metric_state_key, metric_labels, metric_labels[default_index]
    )
    metric_label = st.sidebar.selectbox(
        "Metric",
        metric_labels,
        index=metric_labels.index(selected_metric_label),
        key=metric_state_key,
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
    resample_default = resample_keys[resample_index]
    current_resample = _init_widget_state(resample_state_key, resample_keys, resample_default)
    selected_resample_key = st.sidebar.selectbox(
        "Resample",
        options=resample_keys,
        index=resample_keys.index(current_resample),
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
    aggregate_default = AGGREGATE_OPTIONS[aggregate_index]
    current_aggregate = _init_widget_state(
        aggregate_state_key, AGGREGATE_OPTIONS, aggregate_default
    )
    aggregate = st.sidebar.selectbox(
        "Aggregate",
        AGGREGATE_OPTIONS,
        index=AGGREGATE_OPTIONS.index(current_aggregate),
        key=aggregate_state_key,
    )

    fill_state_key = "homesky_fill_under_line"
    fill_default = any(
        (
            _is_rain_column(metric_column),
            _is_temperature_column(metric_column),
            _is_pressure_column(metric_column),
        )
    )
    if st.session_state.get("_homesky_fill_metric") != metric_column:
        st.session_state["_homesky_fill_metric"] = metric_column
        st.session_state[fill_state_key] = fill_default
    fill_under_line = st.sidebar.checkbox("Fill under line", key=fill_state_key)

    min_ts = df_time.index.min()
    max_ts = df_time.index.max()
    min_date = min_ts.date()
    max_date = max_ts.date()
    hint_start, hint_end = _default_window_last_10_days(tz_name)
    default_start_date = max(hint_start, min_date)
    default_end_date = min(hint_end, max_date)
    if default_start_date > default_end_date:
        default_start_date, default_end_date = min_date, max_date

    date_state_key = "homesky_date_range"
    default_dates = get_date_range(
        pd.Timestamp(default_start_date), pd.Timestamp(default_end_date), date_state_key
    )

    from_key = "homesky_date_from"
    to_key = "homesky_date_to"
    st.session_state.setdefault(from_key, default_dates[0])
    st.session_state.setdefault(to_key, default_dates[1])

    st.caption("Select date range")
    col_from, col_to = st.columns(2)
    with col_from:
        date_from = st.date_input(
            "From",
            value=st.session_state[from_key],
            min_value=min_date,
            max_value=max_date,
            key=from_key,
        )
    with col_to:
        date_to = st.date_input(
            "To",
            value=st.session_state[to_key],
            min_value=min_date,
            max_value=max_date,
            key=to_key,
        )

    start_date = _as_date(date_from, default_dates[0])
    end_date = _as_date(date_to, default_dates[1])
    start_date, end_date = _normalize_date_pair(start_date, end_date)
    if start_date != date_from:
        st.session_state[from_key] = start_date
    if end_date != date_to:
        st.session_state[to_key] = end_date
    st.session_state[date_state_key] = (start_date, end_date)
    normalized_key = "_homesky_date_range_selected"
    st.session_state[normalized_key] = (start_date, end_date)

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

    rain_total, rain_column = compute_rainfall(filtered)
    rain_display = _format_inches(rain_total)

    stats_cols = st.columns(5)
    _render_stat_card(
        stats_cols[0],
        icon="📉",
        label="Min",
        value=_format_stat_value(stats_series.min(), column_name),
        theme=theme,
    )
    _render_stat_card(
        stats_cols[1],
        icon="📊",
        label="Mean",
        value=_format_stat_value(stats_series.mean(), column_name),
        theme=theme,
    )
    _render_stat_card(
        stats_cols[2],
        icon="📈",
        label="Max",
        value=_format_stat_value(stats_series.max(), column_name),
        theme=theme,
    )
    rain_footnote = f"Source: <code>{rain_column}</code>" if rain_column else "Source: n/a"
    _render_stat_card(
        stats_cols[3],
        icon="🌧️",
        label="Rain Total",
        value=rain_display,
        theme=theme,
        footnote=rain_footnote,
    )
    _render_stat_card(
        stats_cols[4],
        icon="🕒",
        label="Last Observation",
        value=_format_timestamp(filtered_time.index.max(), tz_name),
        theme=theme,
    )

    tz_abbr = _infer_tz_abbreviation(filtered_time.index, zone)
    axis_title = f"Time ({tz_abbr})"

    prepared_reset = prepared.reset_index()
    axis_kwargs: Dict[str, object] = {"title": axis_title}
    if resample_value in {"D", "W"}:
        axis_kwargs["format"] = "%b %d"
        axis_kwargs["tickCount"] = 10
    elif resample_value == "M":
        axis_kwargs["format"] = "%b"
        axis_kwargs["tickCount"] = 12

    line_color, area_gradient = _metric_visual_style(metric_column, theme)
    y_min = float(stats_series.min(skipna=True))
    y_max = float(stats_series.max(skipna=True))
    scale_kwargs: Dict[str, object] = {"zero": False, "nice": True}
    if math.isfinite(y_min) and math.isfinite(y_max):
        if y_min == y_max:
            padding = max(abs(y_min) * 0.05, 1.0)
            scale_kwargs["domain"] = [y_min - padding, y_max + padding]
        else:
            padding = max((y_max - y_min) * 0.08, 0.5)
            scale_kwargs["domain"] = [y_min - padding, y_max + padding]

    base_chart = alt.Chart(prepared_reset).encode(
        x=alt.X("s_time_local:T", axis=alt.Axis(**axis_kwargs)),
        y=alt.Y(f"{column_name}:Q", title=metric_label, scale=alt.Scale(**scale_kwargs)),
        tooltip=[
            alt.Tooltip("s_time_local:T", title="Time"),
            alt.Tooltip(f"{column_name}:Q", title=metric_label),
        ],
    )
    line_chart = base_chart.mark_line(color=line_color, strokeWidth=2.5, point=False)
    if fill_under_line:
        area_color = area_gradient if area_gradient is not None else line_color
        area_chart = base_chart.mark_area(color=area_color, opacity=0.55)
        chart = (area_chart + line_chart).properties(height=320).interactive()
    else:
        chart = line_chart.properties(height=320).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"Metric column: `{metric_column}`")

    render_rain_dashboard(
        df_full=df,
        df_time=df_time,
        filtered_time=filtered_time,
        end_ts=end_ts,
        config=config,
        zone=zone,
        theme=theme,
        format_inches=_format_inches,
        format_temperature=_format_temperature,
        rain_metric=rain_column,
    )

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


def main() -> None:
    setup_streamlit_logging(str(STREAMLIT_LOG_PATH))
    try:
        _run_dashboard()
    except AttributeError as exc:
        if "experimental_rerun" in str(exc):
            log.warning("Streamlit rerun API changed; retrying with st.rerun()")
            try:
                st.rerun()
            except Exception as rerun_exc:  # pragma: no cover - defensive guard
                log.error("st.rerun() failed: %s", rerun_exc)
                _record_streamlit_error(rerun_exc)
                raise
            return
        log.exception("Streamlit AttributeError: %s", exc)
        _record_streamlit_error(exc)
        raise
    except Exception as exc:  # pragma: no cover - surfaced to UI
        log.exception("Unhandled exception in Streamlit main: %s", exc)
        _record_streamlit_error(exc)
        raise


if __name__ == "__main__":
    main()
