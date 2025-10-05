"""Helpers for parsing offline timestamp columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger
from zoneinfo import ZoneInfo

__all__ = [
    "OfflineTimestampError",
    "OfflineTimestampResult",
    "parse_offline_timestamp",
]


class OfflineTimestampError(RuntimeError):
    """Raised when offline timestamp detection fails."""

    def __init__(self, message: str, details: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.details = list(details or [])


@dataclass(slots=True)
class OfflineTimestampResult:
    """Result produced after resolving an offline timestamp column."""

    epoch_ms: pd.Series
    timestamp_utc: pd.Series
    source: str
    columns: Tuple[str, ...]
    non_null: int
    details: List[str]


_CANDIDATE_PRIORITIES = {
    "epoch_ms": 0,
    "epoch": 1,
    "dateutc": 2,
    "datetime": 3,
    "timestamp": 4,
    "created": 5,
    "date+time": 6,
    "date": 7,
    "time": 8,
}

_CANDIDATE_SYNONYMS = {
    "epoch_ms": ("epochms",),
    "epoch": ("epoch",),
    "dateutc": ("dateutc", "datetimeutc", "timestamputc", "obstimeutc", "timeutc"),
    "datetime": ("datetime",),
    "timestamp": ("timestamp",),
    "created": ("created",),
    "date": ("date",),
    "time": ("time",),
}


@dataclass(slots=True)
class _TimestampCandidate:
    key: str
    columns: Tuple[str, ...]
    timestamp_utc: pd.Series
    epoch_ms: pd.Series
    valid_count: int
    label: str
    description: str


def _normalize(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _match_columns(columns: Iterable[object], names: Sequence[str]) -> List[str]:
    normalized = {}
    for original in columns:
        key = _normalize(original)
        normalized.setdefault(key, []).append(str(original))
    resolved: List[str] = []
    for name in names:
        key = _normalize(name)
        for candidate in normalized.get(key, []):
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved


def _candidate_columns(columns: Iterable[object], key: str) -> List[str]:
    synonyms = _CANDIDATE_SYNONYMS.get(key, (key,))
    return _match_columns(columns, synonyms)


def _get_local_timezone(config: dict) -> str:
    timezone_cfg = config.get("timezone", {}) if isinstance(config, dict) else {}
    tz_name = timezone_cfg.get("local_tz") or "UTC"
    return str(tz_name)


def _ensure_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:  # pragma: no cover - fallback for invalid tz
        logger.warning("Unknown timezone '%s'; defaulting to UTC for offline import", name)
        return ZoneInfo("UTC")


def _ensure_utc(series: pd.Series, tz_name: str) -> Optional[pd.Series]:
    if series.empty:
        return series
    tzinfo = getattr(series.dt, "tz", None)
    try:
        if tzinfo is None:
            localized = series.dt.tz_localize(
                _ensure_zone(tz_name), nonexistent="shift_forward", ambiguous="NaT"
            )
        else:
            localized = series
        return localized.dt.tz_convert("UTC")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to normalize timezone for offline timestamps: %s", exc)
        return None


def _epoch_ms_from_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], index=series.index, dtype="Int64")
    values = series.view("int64")
    epoch_values = values // 1_000_000
    epoch_series = pd.Series(epoch_values, index=series.index, dtype="Int64")
    epoch_series = epoch_series.mask(series.isna(), pd.NA)
    return epoch_series.astype("Int64")


def _build_candidate(
    key: str,
    columns: Sequence[str],
    dt_series: pd.Series,
    tz_name: str,
    label: str,
) -> Optional[_TimestampCandidate]:
    utc_series = _ensure_utc(dt_series, tz_name)
    if utc_series is None or utc_series.empty:
        return None
    valid_count = int(utc_series.notna().sum())
    if valid_count == 0:
        return None
    epoch_ms = _epoch_ms_from_datetime(utc_series)
    if int(epoch_ms.notna().sum()) == 0:
        return None
    description = f"{label} → {valid_count} valid rows"
    return _TimestampCandidate(
        key=key,
        columns=tuple(columns),
        timestamp_utc=utc_series,
        epoch_ms=epoch_ms,
        valid_count=valid_count,
        label=label,
        description=description,
    )


def _parse_epoch_column(
    series: pd.Series,
    column_name: str,
    tz_name: str,
    *,
    assume_seconds: Optional[bool] = None,
) -> Tuple[Optional[_TimestampCandidate], Optional[str]]:
    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) == 0:
        return None, f"{column_name} → no numeric values"
    unit = "ms"
    if assume_seconds is None:
        cleaned = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
        cleaned = cleaned[cleaned.notna()]
        if not cleaned.empty:
            median_len = cleaned.str.len().median()
            if median_len is not None and median_len <= 10:
                unit = "s"
        median_value = numeric.dropna().abs().median()
        if pd.notna(median_value) and median_value < 1e11:
            unit = "s"
    elif assume_seconds:
        unit = "s"
    dt_series = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    if int(dt_series.notna().sum()) == 0:
        return None, f"{column_name} → conversion produced all NaT"
    label_suffix = " (seconds → ms)" if unit == "s" else ""
    candidate = _build_candidate(
        "epoch" if unit == "s" else "epoch_ms",
        (column_name,),
        dt_series,
        tz_name,
        f"{column_name}{label_suffix}",
    )
    if candidate is None:
        return None, f"{column_name} → failed to normalize timezone"
    return candidate, None


def _combine_date_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    date_series = df[date_col].astype("string").str.strip()
    time_series = df[time_col].astype("string").str.strip()
    combined = date_series.str.cat(time_series, sep=" ", na_rep=None)
    combined = combined.str.strip()
    mask = date_series.isna() | time_series.isna()
    combined = combined.mask(mask)
    return combined


def _prompt_for_candidate(candidates: Sequence[_TimestampCandidate]) -> Optional[_TimestampCandidate]:
    try:
        import PySimpleGUI as sg  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency missing
        logger.debug("PySimpleGUI unavailable for timestamp chooser: %s", exc)
        return None

    option_labels = [
        f"{cand.label} — {cand.valid_count} row(s) ({', '.join(cand.columns)})"
        for cand in candidates
    ]
    if not option_labels:
        return None
    width = max(len(label) for label in option_labels)
    layout = [
        [sg.Text("Select the timestamp column to import:")],
        [
            sg.Listbox(
                values=option_labels,
                default_values=[option_labels[0]],
                size=(min(width + 4, 80), min(len(option_labels), 8)),
                key="-choice-",
            )
        ],
        [sg.Button("Use selection"), sg.Button("Cancel")],
    ]
    try:
        window = sg.Window(
            "Choose timestamp column",
            layout,
            modal=True,
            keep_on_top=True,
            finalize=True,
        )
    except Exception as exc:  # pragma: no cover - GUI fallback
        logger.debug("Unable to open timestamp chooser window: %s", exc)
        return None
    event, values = window.read()
    window.close()
    if event == "Use selection" and values and values.get("-choice-"):
        selected = values["-choice-"][0]
        try:
            index = option_labels.index(selected)
        except ValueError:  # pragma: no cover - defensive
            return None
        return candidates[index]
    return None


def parse_offline_timestamp(
    df: pd.DataFrame,
    config: dict,
    *,
    interactive: bool = False,
) -> OfflineTimestampResult:
    """Detect and normalize a timestamp column for offline imports."""

    tz_name = _get_local_timezone(config)
    details: List[str] = []
    candidates: List[_TimestampCandidate] = []

    # Epoch-based columns first.
    for column in _candidate_columns(df.columns, "epoch_ms"):
        candidate, failure = _parse_epoch_column(df[column], column, tz_name, assume_seconds=False)
        if candidate:
            candidates.append(candidate)
            details.append(candidate.description)
        elif failure:
            details.append(failure)
    for column in _candidate_columns(df.columns, "epoch"):
        candidate, failure = _parse_epoch_column(df[column], column, tz_name, assume_seconds=None)
        if candidate:
            candidates.append(candidate)
            details.append(candidate.description)
        elif failure:
            details.append(failure)

    # UTC-like textual columns.
    for column in _candidate_columns(df.columns, "dateutc"):
        series = pd.to_datetime(df[column], errors="coerce", utc=True)
        if int(series.notna().sum()) == 0:
            details.append(f"{column} → no parseable UTC timestamps")
            continue
        candidate = _build_candidate("dateutc", (column,), series, tz_name, column)
        if candidate:
            candidates.append(candidate)
            details.append(candidate.description)
        else:
            details.append(f"{column} → failed to normalize timezone")

    # Local or ambiguous textual columns.
    for key in ("datetime", "timestamp", "created", "date", "time"):
        for column in _candidate_columns(df.columns, key):
            series = pd.to_datetime(df[column], errors="coerce", utc=False)
            if int(series.notna().sum()) == 0:
                details.append(f"{column} → no parseable values")
                continue
            candidate = _build_candidate(key, (column,), series, tz_name, column)
            if candidate:
                candidates.append(candidate)
                details.append(candidate.description)
            else:
                details.append(f"{column} → failed to normalize timezone")

    # Combine separate date/time columns if available.
    date_columns = _candidate_columns(df.columns, "date")
    time_columns = _candidate_columns(df.columns, "time")
    for date_col in date_columns:
        for time_col in time_columns:
            combined = _combine_date_time(df, date_col, time_col)
            if int(combined.dropna().shape[0]) == 0:
                details.append(f"{date_col}+{time_col} → insufficient data to combine")
                continue
            series = pd.to_datetime(combined, errors="coerce", utc=False)
            if int(series.notna().sum()) == 0:
                details.append(f"{date_col}+{time_col} → parsing produced only NaT")
                continue
            candidate = _build_candidate(
                "date+time",
                (date_col, time_col),
                series,
                tz_name,
                f"{date_col} + {time_col}",
            )
            if candidate:
                candidates.append(candidate)
                details.append(candidate.description)
            else:
                details.append(f"{date_col}+{time_col} → failed to normalize timezone")

    # Remove duplicate candidate descriptions for readability.
    seen_descriptions = set()
    unique_candidates: List[_TimestampCandidate] = []
    unique_details: List[str] = []
    for candidate in candidates:
        if candidate.description in seen_descriptions:
            continue
        seen_descriptions.add(candidate.description)
        unique_candidates.append(candidate)
    for detail in details:
        if detail not in unique_details:
            unique_details.append(detail)

    candidates = unique_candidates
    details = unique_details

    if not candidates:
        available = ", ".join(str(col) for col in df.columns)
        message_lines = [
            "Could not determine a usable timestamp column for offline import.",
            "Inspected columns:",
        ]
        if details:
            message_lines.extend(f"- {line}" for line in details)
        if available:
            message_lines.append(f"Available columns: {available}")
        raise OfflineTimestampError("\n".join(message_lines), details)

    candidates.sort(
        key=lambda cand: (
            _CANDIDATE_PRIORITIES.get(cand.key, 99),
            -cand.valid_count,
        )
    )

    chosen: Optional[_TimestampCandidate] = None
    if interactive and len(candidates) > 1:
        chosen = _prompt_for_candidate(candidates)
    if chosen is None:
        chosen = candidates[0]
        if len(candidates) > 1:
            logger.info(
                "Multiple timestamp columns detected (%s); defaulting to %s",
                ", ".join(candidate.label for candidate in candidates),
                chosen.label,
            )
    details.append(f"Selected {chosen.label} ({chosen.valid_count} valid rows)")

    return OfflineTimestampResult(
        epoch_ms=chosen.epoch_ms,
        timestamp_utc=chosen.timestamp_utc,
        source=chosen.label,
        columns=chosen.columns,
        non_null=chosen.valid_count,
        details=details,
    )
