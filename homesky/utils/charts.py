"""Chart helpers for Streamlit."""

from __future__ import annotations

from typing import Dict, Tuple

import altair as alt
import pandas as pd

from .theming import Theme


def prepare_timeseries(
    df: pd.DataFrame,
    metric: str,
    resample: str,
    aggregate: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    ts = df.copy()
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index, errors="coerce")
    ts = ts.sort_index()
    metric_series = pd.to_numeric(ts[metric], errors="coerce")
    if resample:
        metric_series = getattr(metric_series.resample(resample), aggregate)()
    return metric_series.to_frame(name=metric)


def build_metric_chart(
    df: pd.DataFrame,
    metric: str,
    theme: Theme,
    title: str | None = None,
) -> alt.Chart:
    chart = (
        alt.Chart(df.reset_index())
        .mark_area(line={"color": theme.primary}, color=theme.primary + "33")
        .encode(
            x=alt.X("index:T", title="Time"),
            y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
            tooltip=["index:T", alt.Tooltip(f"{metric}:Q", format=".2f")],
        )
    )
    if title:
        chart = chart.properties(title=title)
    return chart.configure(
        background=theme.background,
        padding=10,
    ).configure_title(color=theme.text).configure_axis(
        labelColor=theme.muted_text,
        titleColor=theme.text,
        gridColor=theme.muted_text + "33",
    )


def describe_metric(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    if df.empty or metric not in df:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    series = pd.to_numeric(df[metric], errors="coerce")
    return {
        "min": float(series.min(skipna=True)),
        "max": float(series.max(skipna=True)),
        "mean": float(series.mean(skipna=True)),
    }


__all__ = ["prepare_timeseries", "build_metric_chart", "describe_metric"]
