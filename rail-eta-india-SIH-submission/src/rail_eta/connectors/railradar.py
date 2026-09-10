"""Connector for a licensed RailRadar live-status feed.

The connector stores append-only raw snapshots. It never mutates a previous
snapshot, which is essential for reconstructing the information available at
prediction time and for detecting feed revisions.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

from rail_eta.connectors.weather import OpenMeteoClient
from rail_eta.contracts import live_data
from rail_eta.timeutils import iso_utc_now


class RailRadarClient:
    """Minimal async client for the documented live-train endpoint."""

    def __init__(self, api_key: str, base_url: str, timeout_seconds: float = 20.0) -> None:
        if not api_key:
            raise ValueError("RAILRADAR_API_KEY is required to collect live snapshots")
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def __aenter__(self) -> RailRadarClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.aclose()

    async def live_status(
        self, train_number: str, journey_date: str | None = None
    ) -> dict[str, Any]:
        train_number = str(train_number).strip()
        if not re.fullmatch(r"\d{5}", train_number):
            raise ValueError(f"train number must contain five digits, got {train_number!r}")
        params: dict[str, str] = {"haltsOnly": "false", "includeCoordinates": "true"}
        if journey_date:
            params["date"] = journey_date
        response = await self._client.get(f"/trains/{train_number}/live", params=params)
        response.raise_for_status()
        payload = response.json()
        live_data(payload)
        return payload

def snapshot_envelope(payload: dict[str, Any], train_number: str) -> dict[str, Any]:
    """Preserve raw provider data with ingestion provenance."""
    data = live_data(payload)
    return {
        "schema_version": 1,
        "captured_at": iso_utc_now(),
        "source": "railradar.live_status",
        "train_number_requested": str(train_number),
        "provider_last_updated_at": data.get("lastUpdatedAt"),
        "payload": payload,
    }


def persist_snapshot(envelope: dict[str, Any], snapshot_dir: Path) -> Path:
    """Atomically persist one immutable snapshot, returning its path."""
    data = live_data(envelope["payload"])
    journey_date = str(data.get("startDate") or "unknown-date")
    safe_train = re.sub(r"[^0-9]", "", str(data["trainNumber"])) or "unknown-train"
    stamp = str(envelope["captured_at"]).replace(":", "-").replace("+", "_")
    destination = snapshot_dir / f"journey_date={journey_date}" / f"train={safe_train}"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{stamp}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(envelope, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
    return path


async def collect_once(
    train_numbers: Iterable[str],
    *,
    api_key: str,
    base_url: str,
    snapshot_dir: Path,
    journey_date: str | None = None,
    concurrency: int = 4,
    weather_enabled: bool = True,
) -> tuple[list[Path], dict[str, str]]:
    """Collect a bounded batch; failures are returned per train, never hidden."""
    semaphore = asyncio.Semaphore(concurrency)
    written: list[Path] = []
    failures: dict[str, str] = {}

    async with RailRadarClient(api_key, base_url) as client:

        async def one(number: str) -> None:
            async with semaphore:
                try:
                    payload = await client.live_status(number, journey_date)
                    envelope = snapshot_envelope(payload, number)
                    if weather_enabled:
                        await attach_current_weather(envelope)
                    written.append(persist_snapshot(envelope, snapshot_dir))
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
                    failures[str(number)] = str(error)

        await asyncio.gather(
            *(one(str(number).strip()) for number in train_numbers if str(number).strip())
        )
    return written, failures


async def attach_current_weather(envelope: dict[str, Any]) -> None:
    """Best-effort collection-time weather; a weather outage never loses a train event."""
    data = live_data(envelope["payload"])
    current_code = str((data.get("currentLocation") or {}).get("stationCode") or "")
    matching_stop = next(
        (
            stop
            for stop in data.get("route", [])
            if isinstance(stop, dict) and stop.get("stationCode") == current_code
        ),
        None,
    )
    if not matching_stop or matching_stop.get("lat") is None or matching_stop.get("lng") is None:
        return
    try:
        async with OpenMeteoClient() as weather:
            envelope["weather_current"] = await weather.current(
                float(matching_stop["lat"]), float(matching_stop["lng"])
            )
    except (httpx.HTTPError, ValueError, TypeError):
        envelope["weather_current_error"] = "weather_unavailable"
