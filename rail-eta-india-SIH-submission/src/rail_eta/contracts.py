"""Small, explicit contracts for external live-data payloads.

The provider's raw JSON is retained unchanged. These checks reject malformed
responses before they enter the immutable snapshot log.
"""

from __future__ import annotations

from typing import Any


class PayloadError(ValueError):
    """A live provider response cannot be used as a train snapshot."""


def live_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the minimal live-status contract and return its data section."""
    if not isinstance(payload, dict):
        raise PayloadError("live response must be a JSON object")
    if payload.get("success") is False:
        detail = payload.get("error", {}).get("message", "provider returned an error")
        raise PayloadError(str(detail))
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise PayloadError("live response has no data object")
    if not str(data.get("trainNumber", "")).strip():
        raise PayloadError("live response has no trainNumber")
    route = data.get("route")
    if not isinstance(route, list) or not route:
        raise PayloadError("live response has no route stops")
    return data
