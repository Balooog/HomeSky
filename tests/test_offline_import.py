import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "homesky"
for candidate in (ROOT, MODULE_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from homesky import import_offline
from homesky.storage import canonicalize_records
from homesky.utils.db import _json_default
from homesky.utils.timeparse import normalize_columns, to_epoch_ms


def test_to_epoch_ms_iso_date():
    df = pd.DataFrame(
        {
            "Date": ["2025-10-05T10:25:00-04:00"],
            "TempF": [65.1],
        }
    )
    normalize_columns(df)
    series = to_epoch_ms(df, tz_hint="America/New_York")
    expected = int(pd.Timestamp("2025-10-05T14:25:00Z").value // 1_000_000)
    assert series.dtype == "int64"
    assert series.tolist() == [expected]


def test_to_epoch_ms_simple_date_localized():
    df = pd.DataFrame(
        {
            "Simple Date": ["10/5/2025 10:25"],
        }
    )
    normalize_columns(df)
    series = to_epoch_ms(df, tz_hint="America/New_York")
    expected = int(pd.Timestamp("2025-10-05T14:25:00Z").value // 1_000_000)
    assert series.dtype == "int64"
    assert series.tolist() == [expected]


def test_to_epoch_ms_pair_date_time():
    df = pd.DataFrame(
        {
            "Date": ["2025-10-05"],
            "Time": ["10:25"],
        }
    )
    normalize_columns(df)
    series = to_epoch_ms(df, tz_hint="America/New_York")
    expected = int(pd.Timestamp("2025-10-05T14:25:00Z").value // 1_000_000)
    assert series.dtype == "int64"
    assert series.tolist() == [expected]


def test_prepare_dataframe_emits_iso_strings():
    raw = pd.DataFrame(
        {
            "Date": ["2025-10-05T10:25:00-04:00"],
            "Simple Date": ["10/5/2025 10:25"],
            "TempF": [65.1],
        }
    )
    config = {"timezone": {"local_tz": "America/New_York"}}
    prepared, details = import_offline._prepare_dataframe(
        raw,
        mac_hint="AA:BB:CC",
        config=config,
        tz_hint="America/New_York",
        source_path=Path("sample.csv"),
        override=None,
    )
    assert not prepared.empty
    assert prepared["epoch_ms"].dtype == "int64"
    assert prepared["observed_at"].dt.tz is not None
    assert prepared["timestamp_utc"].iloc[0] == "2025-10-05T14:25:00Z"
    assert prepared["dateutc"].iloc[0] == "2025-10-05T14:25:00Z"
    assert prepared["timestamp_local"].iloc[0].endswith("-0400")
    row_payload = prepared.iloc[0].to_dict()
    # Ensure JSON serialization works for timestamp-bearing payloads.
    json.dumps(row_payload, default=_json_default)
    canonical = canonicalize_records(prepared.to_dict(orient="records"), mac_hint="AA:BB:CC")
    assert not canonical.empty
    assert canonical.index.tz is not None
    assert canonical["epoch_ms"].dtype == "int64"
