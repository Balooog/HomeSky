"""Rainfall visualization helpers for the HomeSky dashboard."""

from __future__ import annotations

from calendar import monthrange
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

from utils.theming import Theme


RAIN_INTENSITY_BINS: Sequence[float] = (
    0.0,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    float("inf"),
)
RAIN_INTENSITY_LABELS: Sequence[str] = (
    "0–0.05",
    "0.05–0.10",
    "0.10–0.25",
    "0.25–0.50",
    "0.50–0.75",
    "0.75–1.00",
    "1.00–1.50",
    "1.50–2.00",
    ">2.00",
)

RAIN_GRADIENT = alt.Gradient(
    gradient="linear",
    stops=[
        alt.GradientStop(color="#0f172a", offset=0.0),
        alt.GradientStop(color="#1d4ed8", offset=0.45),
        alt.GradientStop(color="#38bdf8", offset=1.0),
    ],
    x1=0,
    x2=0,
    y1=1,
    y2=0,
)


def _resolve_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


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
    categorized = pd.cut(
        series,
        bins=RAIN_INTENSITY_BINS,
        labels=RAIN_INTENSITY_LABELS,
        include_lowest=True,
        right=False,
    )
    counts = categorized.value_counts().sort_index()
    histogram = counts.reset_index()
    histogram.columns = ["bucket", "count"]
    return histogram, column


def _rain_24h(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    diffs = numeric.diff().clip(lower=0)
    return float(diffs.sum(skipna=True))


def compute_rainfall(window: pd.DataFrame) -> Tuple[float, Optional[str]]:
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


def render_rain_dashboard(
    *,
    df_full: pd.DataFrame,
    df_time: pd.DataFrame,
    filtered_time: pd.DataFrame,
    end_ts: pd.Timestamp,
    config: Dict,
    zone: ZoneInfo,
    theme: Theme,
    format_inches: Callable[[float], str],
    format_temperature: Callable[[float], str],
    rain_metric: Optional[str],
) -> None:
    st.subheader("Rain — Year to Date vs Normal")

    daily_rain, daily_rain_column = _daily_rainfall(df_time)
    event_column = _resolve_column(df_full, "event_rain_in", "rain_event_in")
    normals_monthly, normals_error = _monthly_normals_from_config(config)
    if normals_error:
        st.warning(normals_error)

    if daily_rain.empty:
        st.info("No rain totals available. Add rain metrics to see cumulative comparisons.")
        return

    ytd_end = filtered_time.index.max()
    start_of_year = pd.Timestamp(year=ytd_end.year, month=1, day=1, tz=zone)
    ytd_mask = (daily_rain.index >= start_of_year) & (
        daily_rain.index <= ytd_end.normalize()
    )
    ytd_daily = daily_rain.loc[ytd_mask]
    if ytd_daily.empty:
        st.info("No rainfall recorded for the selected year yet.")
        return

    actual_total = float(ytd_daily.sum())
    actual_cumulative = ytd_daily.cumsum()

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

    cards = st.columns(3)
    cards[0].markdown(
        "<div style='background:{};color:{};padding:0.8rem 1rem;border-radius:0.75rem;font-weight:600;font-size:1.4rem;text-align:center;'>".format(
            theme.surface,
            theme.text,
        )
        + f"YTD total<br><span style='font-size:1.8rem'>{format_inches(actual_total)}</span>"
        + "</div>",
        unsafe_allow_html=True,
    )
    if normal_cumulative is not None:
        cards[1].markdown(
            "<div style='background:{};color:{};padding:0.8rem 1rem;border-radius:0.75rem;font-weight:600;font-size:1.4rem;text-align:center;'>".format(
                theme.surface,
                theme.text,
            )
            + f"Normal to date<br><span style='font-size:1.8rem'>{format_inches(normal_total)}</span>"
            + "</div>",
            unsafe_allow_html=True,
        )
        departure = actual_total - normal_total
        departure_color = "#1f9d55" if departure >= 0 else "#dc2626"
        cards[2].markdown(
            "<div style='background:{};color:white;padding:0.8rem 1rem;border-radius:0.75rem;font-weight:600;font-size:1.4rem;text-align:center;'>".format(
                departure_color
            )
            + f"Departure<br><span style='font-size:1.8rem'>{departure:+.1f} in</span>"
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        cards[1].info("Add NOAA normals to compare (see Settings)")
        cards[2].empty()

    ytd_frames = [
        actual_cumulative.rename("value").to_frame().assign(Series="Actual"),
    ]
    if normal_cumulative is not None:
        ytd_frames.append(normal_cumulative.rename("value").to_frame().assign(Series="Normal"))
    ytd_chart_df = (
        pd.concat(ytd_frames)
        .reset_index()
        .rename(columns={"s_time_local": "date"})
    )
    if not ytd_chart_df.empty:
        base = alt.Chart(ytd_chart_df).encode(
            x=alt.X("date:T", axis=alt.Axis(title="Date")),
            y=alt.Y("value:Q", title="Cumulative rain (in)", scale=alt.Scale(nice=True)),
            color=alt.Color(
                "Series:N",
                title="Series",
                scale=alt.Scale(domain=["Actual", "Normal"], range=["#38bdf8", "#facc15"]),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("value:Q", title="Rain (in)"),
                alt.Tooltip("Series:N", title="Series"),
            ],
        )
        actual_layer = base.transform_filter(alt.datum.Series == "Actual").mark_area(
            color=RAIN_GRADIENT, opacity=0.65
        )
        actual_line = base.transform_filter(alt.datum.Series == "Actual").mark_line(
            color="#0ea5e9", strokeWidth=2.5
        )
        chart = actual_layer + actual_line
        if normal_cumulative is not None:
            normal_layer = base.transform_filter(alt.datum.Series == "Normal").mark_line(
                color="#facc15", strokeDash=[6, 4], strokeWidth=2
            )
            chart = chart + normal_layer
        events_df = _top_rain_events(df_time.loc[start_of_year:end_ts], event_column)
        if not events_df.empty:
            events_df = events_df.copy()
            events_df["date"] = events_df["s_time_local"].dt.floor("D")
            events_df["cumulative"] = (
                actual_cumulative.reindex(events_df["date"], method="ffill").to_numpy()
            )
            events_layer = (
                alt.Chart(events_df)
                .mark_point(size=90, color=theme.accent)
                .encode(
                    x="date:T",
                    y="cumulative:Q",
                    tooltip=[
                        alt.Tooltip("date:T", title="Event"),
                        alt.Tooltip("amount:Q", title="Rain (in)"),
                    ],
                )
            )
            chart = chart + events_layer
        st.altair_chart(chart.properties(height=320), use_container_width=True)

    rain_caption_source = rain_metric or daily_rain_column or event_column or "n/a"
    st.caption(f"Rain column: `{rain_caption_source}`")

    year_options = sorted(daily_rain.index.year.unique().tolist())
    rain_year_key = "homesky_rain_year"
    default_year = int(ytd_end.year)
    if rain_year_key not in st.session_state or st.session_state[rain_year_key] not in year_options:
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
    yearly_rain = daily_rain.loc[(daily_rain.index >= year_start) & (daily_rain.index <= year_stop)]
    yearly_df = df_time.loc[(df_time.index >= year_start) & (df_time.index <= year_stop)]

    rain_cols = st.columns(2)
    with rain_cols[0]:
        st.markdown("**Monthly totals**")
        if yearly_rain.empty:
            st.info("No rainfall recorded for the selected year.")
        else:
            event_daily = pd.Series(False, index=yearly_rain.index)
            if event_column:
                event_series = pd.to_numeric(df_time[event_column], errors="coerce").fillna(0.0)
                event_daily_series = event_series.resample("D").max()
                event_daily = event_daily_series.reindex(yearly_rain.index, fill_value=0) > 0
            monthly_frame = pd.DataFrame(
                {
                    "date": yearly_rain.index,
                    "rain": yearly_rain.values,
                    "category": ["Event day" if flag else "Other day" for flag in event_daily],
                }
            )
            monthly_frame["month"] = monthly_frame["date"].dt.to_period("M").dt.to_timestamp()
            monthly_totals = (
                monthly_frame.groupby(["month", "category"], as_index=False)["rain"].sum()
            )
            chart = (
                alt.Chart(monthly_totals)
                .mark_bar()
                .encode(
                    x=alt.X("month:T", axis=alt.Axis(title="Month")),
                    y=alt.Y("rain:Q", title="Monthly rain (in)", stack=None),
                    color=alt.Color(
                        "category:N",
                        title="Day type",
                        scale=alt.Scale(range=["#38bdf8", "#94a3b8"]),
                    ),
                    tooltip=[
                        alt.Tooltip("month:T", title="Month"),
                        alt.Tooltip("rain:Q", title="Rain (in)"),
                        alt.Tooltip("category:N", title="Day"),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
    with rain_cols[1]:
        st.markdown("**Rain intensity histogram**")
        histogram, hist_column = _rain_rate_histogram(yearly_df)
        if histogram.empty:
            st.info("No rain intensity data for the selected year.")
        else:
            hist_chart = (
                alt.Chart(histogram)
                .mark_bar(color="#1d4ed8")
                .encode(
                    x=alt.X("bucket:N", title="Rain rate (in/hr)"),
                    y=alt.Y("count:Q", title="Hours"),
                    tooltip=[
                        alt.Tooltip("bucket:N", title="Rate (in/hr)"),
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
                    "Rain (in)": format_inches(amount),
                    "Min temp": format_temperature(min_val),
                    "Median temp": format_temperature(median_val),
                    "Max temp": format_temperature(max_val),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)


__all__ = ["compute_rainfall", "render_rain_dashboard"]
