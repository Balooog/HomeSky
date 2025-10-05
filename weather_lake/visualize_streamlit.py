"""Streamlit dashboard for Weather Lake."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas.api import types as ptypes
import streamlit as st

from utils.charts import build_metric_chart, describe_metric, prepare_timeseries
from utils.db import DatabaseManager
from utils.derived import compute_all_derived
from utils.theming import get_theme, load_typography

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

CONFIG_PATH = Path("config.toml")


@st.cache_data(ttl=60)
def load_config(path: Path = CONFIG_PATH) -> Dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


@st.cache_data(ttl=60)
def load_data(sqlite_path: str, parquet_path: str) -> pd.DataFrame:
    db = DatabaseManager(Path(sqlite_path), Path(parquet_path))
    df = db.read_dataframe()
    return df


def main() -> None:
    st.set_page_config(page_title="Weather Lake", layout="wide")
    try:
        config = load_config()
    except FileNotFoundError:
        st.error("Missing config.toml. Copy config.example.toml and update your credentials.")
        st.stop()

    theme_choice = config.get("visualization", {}).get("theme", "dark")
    theme = get_theme(theme_choice)
    typography = load_typography()

    storage_cfg = config.get("storage", {})
    df = load_data(storage_cfg.get("sqlite_path", "./data/weather.sqlite"), storage_cfg.get("parquet_path", "./data/lake.parquet"))

    if df.empty:
        st.info("No observations stored yet. Run ingest.py to begin capturing data.")
        st.stop()

    df = compute_all_derived(df, config)

    st.sidebar.header("Controls")
    numeric_columns = [col for col in df.columns if ptypes.is_numeric_dtype(df[col])]
    if not numeric_columns:
        st.error("No numeric columns available to plot.")
        st.stop()
    metrics = sorted(numeric_columns)
    default_metric = config.get("visualization", {}).get("default_metric", metrics[0])
    metric_index = metrics.index(default_metric) if default_metric in metrics else 0
    metric = st.sidebar.selectbox("Metric", metrics, index=metric_index)

    resample_options = ["", "15min", "H", "D", "W"]
    default_resample = config.get("visualization", {}).get("default_resample", "H")
    resample_index = resample_options.index(default_resample) if default_resample in resample_options else 0
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

    st.title("Weather Lake Dashboard")
    st.altair_chart(build_metric_chart(prepared, metric, theme, title=metric.title()), use_container_width=True)

    stats = describe_metric(prepared, metric)
    cols = st.columns(3)
    cols[0].metric("Min", f"{stats['min']:.2f}")
    cols[1].metric("Mean", f"{stats['mean']:.2f}")
    cols[2].metric("Max", f"{stats['max']:.2f}")

    st.subheader("Data Explorer")
    st.dataframe(df.tail(500))

    csv = df.to_csv().encode("utf-8")
    st.download_button("Download CSV", csv, file_name="weather_lake.csv", mime="text/csv")
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, engine="pyarrow")
    st.download_button(
        "Download Parquet",
        parquet_buffer.getvalue(),
        file_name="weather_lake.parquet",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
