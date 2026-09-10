"""Weather enrichment using the documented Open-Meteo forecast API.

Only forecast/current values observed at collection time belong in production
features. Historical reanalysis must not be substituted during backtests: it
would tell the model weather that was not forecast at the time.
"""

from __future__ import annotations

from typing import Any

import httpx


class OpenMeteoClient:
    def __init__(self, timeout_seconds: float = 12.0) -> None:
        self._client = httpx.AsyncClient(
            base_url="https://api.open-meteo.com/v1", timeout=timeout_seconds
        )

    async def __aenter__(self) -> OpenMeteoClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.aclose()

    async def current(self, latitude: float, longitude: float) -> dict[str, Any]:
        response = await self._client.get(
            "/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,precipitation,rain,wind_speed_10m,weather_code",
                "timezone": "Asia/Kolkata",
            },
        )
        response.raise_for_status()
        body = response.json()
        return body.get("current", {})
