"""Chart helpers for Streamlit."""

from __future__ import annotations

from typing import Dict, Optional

import altair as alt
import pandas as pd

from .theming import Theme

# Canonical resample aliases exposed in the UI. Using uppercase single-letter
# aliases avoids pandas deprecation warnings and keeps the labels predictable.
RESAMPLE_ALIASES: Dict[str, Optional[str]] = {
    "": None,
    "none": None,
    "raw": None,
    "5min": "5T",
    "5": "5T",
    "5t": "5T",
    "15min": "15T",
    "15": "15T",
    "15t": "15T",
    "h": "H",
    "H": "H",
    "hour": "H",
    "d": "D",
    "D": "D",
    "day": "D",
    "w": "W",
    "W": "W",
    "week": "W",
    "m": "M",
    "M": "M",
    "month": "M",
}


def _normalize_resample(resample: Optional[str]) -> Optional[str]:
    if resample is None:
        return None
    if isinstance(resample, str):
        key = resample.strip()
        if key in RESAMPLE_ALIASES:
            return RESAMPLE_ALIASES[key]
        key_lower = key.lower()
        return RESAMPLE_ALIASES.get(key_lower, key)
    return None


def prepare_timeseries(
    df: pd.DataFrame,
    metric: str,
    resample: Optional[str],
    aggregate: str,
) -> pd.DataFrame:
    if df.empty or metric not in df:
        return pd.DataFrame(columns=[metric])
    ts = df.copy()
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index, errors="coerce")
    ts = ts.sort_index()
    metric_series = pd.to_numeric(ts[metric], errors="coerce")
    metric_series = metric_series.dropna()

    rule = _normalize_resample(resample)
    if rule:
        grouper = metric_series.resample(rule)
        agg = aggregate.lower()
        if agg == "sum":
            metric_series = grouper.sum(min_count=1)
        elif agg == "last":
            metric_series = grouper.last()
        elif hasattr(grouper, agg):
            metric_series = getattr(grouper, agg)()
        else:
            metric_series = grouper.mean()
    result = metric_series.to_frame(name=metric)
    if result.index.name is None:
        result.index.name = "s_time_local"
    return result


def build_metric_chart(
    df: pd.DataFrame,
    metric: str,
    theme: Theme,
    *,
    title: str | None = None,
    timezone_label: str | None = None,
) -> alt.Chart:
    display = df.reset_index().rename(columns={df.index.name or "index": "s_time_local"})
    x_title = "Time"
    if timezone_label:
        x_title = f"Time ({timezone_label})"
    chart = (
        alt.Chart(display)
        .mark_line(color=theme.primary, interpolate="monotone")
        .encode(
            x=alt.X("s_time_local:T", title=x_title),
            y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
            tooltip=[
                alt.Tooltip("s_time_local:T", title="Timestamp"),
                alt.Tooltip(f"{metric}:Q", format=".2f", title="Value"),
            ],
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


__all__ = [
    "prepare_timeseries",
    "build_metric_chart",
    "describe_metric",
    "RESAMPLE_ALIASES",
]
