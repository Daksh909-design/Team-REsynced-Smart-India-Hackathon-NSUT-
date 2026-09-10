"""Time helpers. All persisted timestamps are timezone-aware UTC."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


def as_utc(value: object) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp or NaT for a missing value."""
    if value is None or value == "":
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed if isinstance(parsed, pd.Timestamp) else pd.NaT


def iso_utc_now() -> str:
    return datetime.now(UTC).isoformat()


def minutes_between(later: object, earlier: object) -> float:
    later_ts = as_utc(later)
    earlier_ts = as_utc(earlier)
    if pd.isna(later_ts) or pd.isna(earlier_ts):
        return float("nan")
    return (later_ts - earlier_ts).total_seconds() / 60.0
