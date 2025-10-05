"""Streamlit dashboard for HomeSky."""

from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
from typing import Dict

import pandas as pd
from pandas.api import types as ptypes
import streamlit as st

import ingest
from utils.charts import build_metric_chart, describe_metric, prepare_timeseries
from utils.derived import compute_all_derived
from utils.theming import get_theme, load_typography


def _format_timestamp(ts: pd.Timestamp | None) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "n/a"
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0]
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[-1]
    if not isinstance(ts, pd.Timestamp):
        return str(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def _cache_token(sqlite_path: str, parquet_path: str) -> str:
    parts = []
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

    latest_ts = df.index.max() if not df.empty else None
    st.markdown(
        f"<div style='display:inline-block;padding:0.35rem 0.75rem;border-radius:999px;background-color:#1f77b4;color:white;font-weight:600;'>"
        f"Latest: {_format_timestamp(latest_ts)} • {len(df):,} rows"  # noqa: E501
        "</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("Controls")
    numeric_columns = [col for col in df.columns if ptypes.is_numeric_dtype(df[col])]
    if not numeric_columns:
        st.error("No numeric columns available to plot.")
        st.stop()
    metrics = sorted(numeric_columns)
    default_metric = config.get("visualization", {}).get("default_metric", metrics[0])
    metric_index = metrics.index(default_metric) if default_metric in metrics else 0
    metric = st.sidebar.selectbox("Metric", metrics, index=metric_index)

    resample_options = ["", "15min", "h", "d", "w"]
    configured_resample = config.get("visualization", {}).get("default_resample", "h")
    if isinstance(configured_resample, str):
        configured_resample = configured_resample.lower()
    default_resample = configured_resample if configured_resample in resample_options else ""
    resample_index = resample_options.index(default_resample)
    resample = st.sidebar.selectbox(
        "Resample",
        options=resample_options,
        index=resample_index,
        format_func=lambda x: "Raw" if x == "" else x,
    )

    aggregate_options = ["mean", "min", "max", "median"]
    default_aggregate = config.get("visualization", {}).get("default_aggregate", "mean")
    aggregate_index = (
        aggregate_options.index(default_aggregate)
        if default_aggregate in aggregate_options
        else 0
    )
    aggregate = st.sidebar.selectbox(
        "Aggregate",
        options=aggregate_options,
        index=aggregate_index,
    )
    date_range = st.sidebar.date_input(
        "Date range",
        value=(df.index.min().date(), df.index.max().date()),
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        mask = (df.index.date >= start) & (df.index.date <= end)
        df = df.loc[mask]

    st.markdown(
        f"""
        <style>
        body {{ background-color: {theme.background}; color: {theme.text}; }}
        .stApp {{ font-family: {typography['font_family']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    prepared = prepare_timeseries(df, metric, resample, aggregate)

    if prepared.empty:
        st.warning("No data available for the selected range/metric.")
        st.stop()

    st.title("HomeSky Dashboard")
    st.altair_chart(build_metric_chart(prepared, metric, theme, title=metric.title()), use_container_width=True)

    stats = describe_metric(prepared, metric)
    cols = st.columns(3)
    cols[0].metric("Min", f"{stats['min']:.2f}")
    cols[1].metric("Mean", f"{stats['mean']:.2f}")
    cols[2].metric("Max", f"{stats['max']:.2f}")

    st.subheader("Data Explorer")
    st.dataframe(df.tail(500))

    csv = df.to_csv().encode("utf-8")
    st.download_button("Download CSV", csv, file_name="homesky.csv", mime="text/csv")
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, engine="pyarrow")
    st.download_button(
        "Download Parquet",
        parquet_buffer.getvalue(),
        file_name="homesky.parquet",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
