"""Tests for Streamlit helpers."""

from zoneinfo import ZoneInfo

import pandas as pd

from homesky.visualize_streamlit import _safe_localize_day


def test_safe_localize_day_handles_spring_forward() -> None:
    zone = ZoneInfo("America/New_York")
    localized = _safe_localize_day(pd.Timestamp("2025-03-09"), zone)

    assert localized.tzinfo == zone
    assert localized.hour == 0
    # EST offset still applies at midnight before the jump forward
    assert localized.utcoffset().total_seconds() == -5 * 3600


def test_safe_localize_day_handles_fall_back_from_utc() -> None:
    zone = ZoneInfo("America/New_York")
    aware = pd.Timestamp("2025-11-02T05:00:00Z")
    localized = _safe_localize_day(aware, zone)

    assert localized.tzinfo == zone
    assert localized.hour == 0
    # Midnight remains valid even on the fallback day
    assert localized.day == 2
