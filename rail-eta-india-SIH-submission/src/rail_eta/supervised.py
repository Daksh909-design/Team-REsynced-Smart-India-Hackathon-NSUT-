"""Turn immutable live snapshots into leakage-safe supervised ETA examples.

One training row means: *given only the information in this snapshot, predict
the eventual arrival delay at this still-upcoming stop*.  The label is read
from a later snapshot of the same journey, never from an ETA published at the
time of prediction.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rail_eta.contracts import live_data
from rail_eta.timeutils import as_utc, minutes_between

TARGET = "arrival_delay_minutes"
KEY_COLUMNS = ["train_number", "journey_date", "target_station_code", "target_sequence"]

NUMERIC_FEATURES = [
    "current_delay_minutes",
    "remaining_distance_km",
    "scheduled_remaining_minutes",
    "remaining_halts",
    "target_sequence",
    "total_halts",
    "segment_progress",
    "current_speed_kmh",
    "current_hour_sin",
    "current_hour_cos",
    "current_day_of_week",
    "station_snapshot_congestion",
    "historical_station_delay_median_28d",
    "historical_train_station_delay_median_28d",
    "historical_station_observations_28d",
    "weather_temperature_c",
    "weather_precipitation_mm",
    "weather_rain_mm",
    "weather_wind_kmh",
    "weather_code",
]
CATEGORICAL_FEATURES = [
    "train_type",
    "train_category",
    "source_station_code",
    "current_station_code",
    "target_station_code",
    "is_weekend",
    "is_diverted",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _number(value: Any) -> float:
    return float(value) if value is not None and value != "" else float("nan")


def _stop_time(stop: dict[str, Any]) -> pd.Timestamp:
    """The best observed timestamp at a passed stop."""
    return (
        as_utc(stop.get("actualDeparture"))
        if stop.get("actualDeparture")
        else as_utc(stop.get("actualArrival"))
    )


def _observed_at(envelope: dict[str, Any]) -> pd.Timestamp:
    data = live_data(envelope["payload"])
    return as_utc(
        envelope.get("provider_last_updated_at")
        or data.get("lastUpdatedAt")
        or envelope.get("captured_at")
    )


def _data_and_route(envelope: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = live_data(envelope["payload"])
    route = [item for item in data["route"] if isinstance(item, dict)]
    return data, route


def observation_rows(envelope: dict[str, Any], snapshot_id: str = "") -> list[dict[str, Any]]:
    """Create feature candidates for all future halting stops in one snapshot."""
    data, route = _data_and_route(envelope)
    observed_at = _observed_at(envelope)
    if pd.isna(observed_at):
        return []

    train = data.get("train") or {}
    current_location = data.get("currentLocation") or {}
    passed = [
        stop for stop in route if not pd.isna(_stop_time(stop)) and _stop_time(stop) <= observed_at
    ]
    latest = max(passed, key=_stop_time) if passed else None
    passed_by_time = sorted(passed, key=_stop_time)
    current_station = str(
        (latest or {}).get("stationCode") or current_location.get("stationCode") or "UNKNOWN"
    )
    current_sequence = int(
        (latest or {}).get("sequence") or current_location.get("sequence") or 0
    )
    current_distance = _number((latest or {}).get("distance"))
    if np.isnan(current_distance):
        current_distance = _number(current_location.get("distance"))
    if np.isnan(current_distance):
        current_distance = 0.0

    current_delay = _number(data.get("delayMinutes"))
    if np.isnan(current_delay) and latest:
        current_delay = _number(latest.get("delayDeparture") or latest.get("delayArrival"))
    previous_delay = float("nan")
    if len(passed_by_time) >= 2:
        previous_stop = passed_by_time[-2]
        previous_delay = _number(
            previous_stop.get("delayDeparture") or previous_stop.get("delayArrival")
        )
    total_halts = int(sum(bool(stop.get("isHalt", True)) for stop in route))
    exceptions = data.get("exceptions") or []
    weather = envelope.get("weather_current") or {}
    india_time = observed_at.tz_convert("Asia/Kolkata")
    clock_minutes = india_time.hour * 60 + india_time.minute
    source_station = str(
        (route[0] if route else {}).get("stationCode")
        or train.get("source", {}).get("code")
        or "UNKNOWN"
    )

    rows: list[dict[str, Any]] = []
    for stop in route:
        scheduled_arrival = as_utc(stop.get("scheduledArrival"))
        # A prediction target has to be a future passenger halt with a real planned time.
        if (
            not stop.get("isHalt", True)
            or pd.isna(scheduled_arrival)
            or scheduled_arrival <= observed_at
        ):
            continue
        target_distance = _number(stop.get("distance"))
        remaining_distance = (
            max(0.0, target_distance - current_distance)
            if not np.isnan(target_distance)
            else float("nan")
        )
        remaining_halts = sum(
            bool(candidate.get("isHalt", True))
            and int(candidate.get("sequence", 0)) >= int(stop.get("sequence", 0))
            for candidate in route
        )
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "observed_at": observed_at,
                "train_number": str(data.get("trainNumber")),
                "journey_date": str(data.get("startDate") or "unknown"),
                "source_station_code": source_station,
                "current_station_code": current_station,
                "current_sequence": current_sequence,
                "target_station_code": str(stop.get("stationCode") or "UNKNOWN"),
                "target_sequence": int(stop.get("sequence") or 0),
                "target_latitude": _number(stop.get("lat")),
                "target_longitude": _number(stop.get("lng")),
                "scheduled_arrival_at": scheduled_arrival,
                "current_delay_minutes": current_delay,
                "previous_delay_minutes": previous_delay,
                "delay_trend_minutes": current_delay - previous_delay
                if np.isfinite(current_delay) and np.isfinite(previous_delay)
                else float("nan"),
                "remaining_distance_km": remaining_distance,
                "scheduled_remaining_minutes": minutes_between(scheduled_arrival, observed_at),
                "remaining_halts": remaining_halts,
                "total_halts": total_halts,
                "segment_progress": _number(current_location.get("segmentProgress")),
                "current_speed_kmh": _number(current_location.get("speedKmh")),
                "current_hour_sin": np.sin(2 * np.pi * clock_minutes / 1440),
                "current_hour_cos": np.cos(2 * np.pi * clock_minutes / 1440),
                "current_day_of_week": india_time.dayofweek,
                "train_type": str(train.get("type") or "UNKNOWN"),
                "train_category": str(train.get("category") or "UNKNOWN"),
                "is_weekend": "yes" if india_time.dayofweek >= 5 else "no",
                "is_diverted": "yes"
                if any(
                    item.get("type") == "DIVERTED" for item in exceptions if isinstance(item, dict)
                )
                else "no",
                "weather_temperature_c": _number(weather.get("temperature_2m")),
                "weather_precipitation_mm": _number(weather.get("precipitation")),
                "weather_rain_mm": _number(weather.get("rain")),
                "weather_wind_kmh": _number(weather.get("wind_speed_10m")),
                "weather_code": _number(weather.get("weather_code")),
            }
        )
    return rows


def outcome_rows(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract eventual, actually observed arrivals present in this snapshot."""
    data, route = _data_and_route(envelope)
    available_at = _observed_at(envelope)
    rows: list[dict[str, Any]] = []
    for stop in route:
        scheduled = as_utc(stop.get("scheduledArrival"))
        actual = as_utc(stop.get("actualArrival"))
        if pd.isna(scheduled) or pd.isna(actual):
            continue
        rows.append(
            {
                "train_number": str(data.get("trainNumber")),
                "journey_date": str(data.get("startDate") or "unknown"),
                "target_station_code": str(stop.get("stationCode") or "UNKNOWN"),
                "target_sequence": int(stop.get("sequence") or 0),
                "scheduled_arrival_at": scheduled,
                "actual_arrival_at": actual,
                "available_at": available_at,
                TARGET: minutes_between(actual, scheduled),
            }
        )
    return rows


def read_envelopes(snapshot_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read only append-only JSON snapshots, ignoring incomplete temporary files."""
    records: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(snapshot_dir.rglob("*.json")):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            live_data(envelope["payload"])
            records.append((str(path), envelope))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            # Bad external input is not silently used. The caller gets the count in its audit.
            continue
    return records


def _deduplicate_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return outcomes
    # The first snapshot that reports the actual is when it became available to this system.
    outcomes = outcomes.sort_values("available_at")
    return outcomes.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)


def _add_snapshot_congestion(observations: pd.DataFrame) -> pd.DataFrame:
    observations = observations.copy()
    if observations.empty:
        observations["station_snapshot_congestion"] = pd.Series(dtype=float)
        return observations
    observations["_congestion_bin"] = observations["observed_at"].dt.floor("15min")
    observations["station_snapshot_congestion"] = observations.groupby(
        ["_congestion_bin", "current_station_code"], dropna=False
    )["snapshot_id"].transform("nunique")
    return observations.drop(columns="_congestion_bin")


def _add_point_in_time_history(examples: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Add 28-day historical medians that were available *before* each snapshot."""
    examples = examples.sort_values("observed_at").copy()
    if examples.empty:
        for column in (
            "historical_station_delay_median_28d",
            "historical_train_station_delay_median_28d",
            "historical_station_observations_28d",
        ):
            examples[column] = pd.Series(dtype=float)
        return examples

    known = outcomes.sort_values("available_at").to_dict("records") if not outcomes.empty else []
    station_history: dict[str, deque[tuple[pd.Timestamp, float]]] = defaultdict(deque)
    train_station_history: dict[tuple[str, str], deque[tuple[pd.Timestamp, float]]] = defaultdict(
        deque
    )
    index = 0
    station_median: list[float] = []
    train_station_median: list[float] = []
    station_count: list[int] = []
    window = pd.Timedelta(days=28)

    for row in examples.itertuples(index=False):
        while index < len(known) and known[index]["available_at"] <= row.observed_at:
            event = known[index]
            if not pd.isna(event["actual_arrival_at"]):
                station = event["target_station_code"]
                pair = (event["train_number"], station)
                value = float(event[TARGET])
                station_history[station].append((event["actual_arrival_at"], value))
                train_station_history[pair].append((event["actual_arrival_at"], value))
            index += 1
        cutoff = row.observed_at - window
        station_values = station_history[row.target_station_code]
        while station_values and station_values[0][0] < cutoff:
            station_values.popleft()
        pair_values = train_station_history[(row.train_number, row.target_station_code)]
        while pair_values and pair_values[0][0] < cutoff:
            pair_values.popleft()
        values = [value for _, value in station_values]
        train_values = [value for _, value in pair_values]
        station_median.append(float(np.median(values)) if values else float("nan"))
        train_station_median.append(
            float(np.median(train_values)) if train_values else float("nan")
        )
        station_count.append(len(values))

    examples["historical_station_delay_median_28d"] = station_median
    examples["historical_train_station_delay_median_28d"] = train_station_median
    examples["historical_station_observations_28d"] = station_count
    return examples


def build_training_frame(snapshot_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build labels and features from captured real snapshots with a full audit."""
    snapshot_files_found = len(list(snapshot_dir.rglob("*.json")))
    envelopes = read_envelopes(snapshot_dir)
    observations = [row for path, envelope in envelopes for row in observation_rows(envelope, path)]
    outcomes = [row for _, envelope in envelopes for row in outcome_rows(envelope)]
    observation_frame = pd.DataFrame(observations)
    outcome_frame = _deduplicate_outcomes(pd.DataFrame(outcomes))
    audit = {
        "snapshot_files_found": snapshot_files_found,
        "snapshot_files_read": len(envelopes),
        "snapshot_files_rejected": snapshot_files_found - len(envelopes),
        "future_stop_candidates": len(observation_frame),
        "deduplicated_actual_arrivals": len(outcome_frame),
        "labelled_examples": 0,
    }
    if observation_frame.empty or outcome_frame.empty:
        return pd.DataFrame(), audit

    observation_frame["observed_at"] = pd.to_datetime(observation_frame["observed_at"], utc=True)
    outcome_frame["actual_arrival_at"] = pd.to_datetime(
        outcome_frame["actual_arrival_at"], utc=True
    )
    outcome_frame["available_at"] = pd.to_datetime(outcome_frame["available_at"], utc=True)
    examples = observation_frame.merge(
        outcome_frame[KEY_COLUMNS + ["actual_arrival_at", "available_at", TARGET]],
        on=KEY_COLUMNS,
        how="inner",
    )
    # Prevent a completed/known arrival from ever being a target for an earlier task.
    examples = examples.loc[examples["actual_arrival_at"] > examples["observed_at"]].copy()
    examples = _add_snapshot_congestion(examples)
    examples = _add_point_in_time_history(examples, outcome_frame)
    audit["labelled_examples"] = len(examples)
    return examples.reset_index(drop=True), audit


def make_online_feature_rows(
    envelope: dict[str, Any], history_profile: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Create serving rows and join a train-split-only station history profile."""
    frame = pd.DataFrame(observation_rows(envelope, snapshot_id="online"))
    if frame.empty:
        return frame
    frame = _add_snapshot_congestion(frame)
    for column in (
        "historical_station_delay_median_28d",
        "historical_train_station_delay_median_28d",
        "historical_station_observations_28d",
    ):
        frame[column] = np.nan
    if history_profile is not None and not history_profile.empty:
        merge_columns = ["target_station_code"]
        available = [column for column in history_profile.columns if column not in merge_columns]
        frame = frame.drop(
            columns=[column for column in available if column in frame], errors="ignore"
        )
        frame = frame.merge(history_profile, on=merge_columns, how="left")
    return frame


def save_training_frame(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def load_training_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("observed_at", "scheduled_arrival_at", "actual_arrival_at", "available_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame
