"""FastAPI surface for passenger apps, station systems, and control rooms."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rail_eta.bootstrap import live_bootstrap_features
from rail_eta.config import Settings
from rail_eta.connectors.railradar import (
    RailRadarClient,
    attach_current_weather,
    persist_snapshot,
    snapshot_envelope,
)
from rail_eta.supervised import make_online_feature_rows
from rail_eta.training import ModelBundle, load_bundle


class StopEta(BaseModel):
    station_code: str
    sequence: int
    scheduled_arrival: str
    predicted_arrival: str
    lower_arrival: str
    upper_arrival: str
    predicted_delay_minutes: float
    lower_delay_minutes: float
    upper_delay_minutes: float
    remaining_distance_km: float | None
    prediction_mode: str
    confidence: str


class EtaResponse(BaseModel):
    train_number: str
    journey_date: str
    source_updated_at: str | None
    model_name: str
    trained_through: str
    etas: list[StopEta]
    current_station_code: str | None = None
    current_delay_minutes: float | None = None
    current_speed_kmh: float | None = None
    source_freshness_seconds: float | None = None
    response_generated_at: str
    journey_status: str = "running"
    status_message: str | None = None


def _timestamp_string(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if pd.notna(number) else None


def create_app(
    model_path: Path = Path("models/eta_model.joblib"), settings: Settings | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if model_path.exists():
            state["bundle"] = load_bundle(model_path)
        else:
            fallback = model_path.with_name("bootstrap_prior.joblib")
            if not fallback.exists():
                raise FileNotFoundError(f"neither {model_path} nor {fallback} exists")
            state["bundle"] = joblib.load(fallback)
        yield

    app = FastAPI(
        title="Indian Railways Coaching Train ETA API",
        version="0.1.0",
        description="Point-in-time ETA predictions with calibrated 90% arrival intervals.",
        lifespan=lifespan,
    )
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    if frontend_dir.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dir), name="assets")

        @app.get("/", include_in_schema=False)
        async def frontend() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        bundle = state["bundle"]
        return {
            "status": "ok",
            "model": bundle.model_name if isinstance(bundle, ModelBundle) else "bootstrap_prior",
            "trained_through": getattr(bundle, "trained_through", "public table; no timestamp"),
            "live_connector": bool(settings.railradar_api_key),
            "transition_model": bool(getattr(bundle, "transition_model", None)),
        }

    @app.get("/v1/eta/{train_number}", response_model=EtaResponse)
    async def eta(
        train_number: str,
        journey_date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    ) -> EtaResponse:
        bundle = state["bundle"]
        try:
            async with RailRadarClient(
                settings.railradar_api_key, settings.railradar_base_url
            ) as client:
                payload = await client.live_status(train_number, journey_date)
            envelope = snapshot_envelope(payload, train_number)
            data = payload["data"] if "data" in payload else payload
            provider_status = str(data.get("status") or "").strip().lower()
            journey_finished = provider_status in {
                "completed",
                "cancelled",
                "canceled",
            }
            journey_live = data.get("isLive") is not False and not journey_finished
            if settings.weather_enabled:
                await attach_current_weather(envelope)
            # Serving traffic is valuable future training evidence, so persist it.
            persist_snapshot(envelope, settings.snapshot_dir)
            current_location = data.get("currentLocation") or {}
            if not journey_live:
                return EtaResponse(
                    train_number=str(data["trainNumber"]),
                    journey_date=str(data.get("startDate") or journey_date or "unknown"),
                    source_updated_at=data.get("lastUpdatedAt"),
                    model_name="not_applicable",
                    trained_through="no future stops; journey is not live",
                    etas=[],
                    current_station_code=current_location.get("stationCode"),
                    current_delay_minutes=None,
                    current_speed_kmh=None,
                    source_freshness_seconds=None,
                    response_generated_at=_timestamp_string(envelope["captured_at"]),
                    journey_status="finished" if journey_finished else "not_live",
                    status_message=(
                        "Journey finished; there is no future ETA to predict."
                        if journey_finished
                        else "RailRadar has not marked this journey as live."
                    ),
                )
            profile = bundle.history_profile if isinstance(bundle, ModelBundle) else None
            features = make_online_feature_rows(envelope, profile)
            if features.empty:
                raise HTTPException(
                    status_code=409, detail="no future halting stop is available for this journey"
                )
            if isinstance(bundle, ModelBundle):
                point, lower, upper = bundle.predict_delay(features)
                model_name = bundle.model_name
                trained_through = bundle.trained_through
            else:
                live_features = live_bootstrap_features(features)
                point, lower, upper = bundle.predict_live(live_features)
                model_name = (
                    "bootstrap_prior+transition"
                    if getattr(bundle, "transition_model", None) is not None
                    else "bootstrap_prior"
                )
                trained_through = "public station-level table; no timestamp"
        except HTTPException:
            raise
        except Exception as error:  # The public API should not expose provider internals.
            raise HTTPException(
                status_code=502, detail=f"unable to obtain a live ETA: {error}"
            ) from error

        etas: list[StopEta] = []
        for row, delay, low, high in zip(
            features.to_dict("records"), point, lower, upper, strict=True
        ):
            scheduled = pd.Timestamp(row["scheduled_arrival_at"])
            etas.append(
                StopEta(
                    station_code=row["target_station_code"],
                    sequence=int(row["target_sequence"]),
                    scheduled_arrival=_timestamp_string(scheduled),
                    predicted_arrival=_timestamp_string(
                        scheduled + pd.Timedelta(minutes=float(delay))
                    ),
                    lower_arrival=_timestamp_string(scheduled + pd.Timedelta(minutes=float(low))),
                    upper_arrival=_timestamp_string(scheduled + pd.Timedelta(minutes=float(high))),
                    predicted_delay_minutes=round(float(delay), 1),
                    lower_delay_minutes=round(float(low), 1),
                    upper_delay_minutes=round(float(high), 1),
                    remaining_distance_km=_finite_or_none(row["remaining_distance_km"]),
                    prediction_mode=(
                        "transition"
                        if not isinstance(bundle, ModelBundle)
                        and pd.notna(row.get("current_delay_minutes"))
                        and 1
                        <= int(row.get("target_sequence", 0))
                        - int(row.get("current_sequence", 0))
                        <= getattr(bundle, "transition_max_horizon_stops", 0)
                        else "dynamic_model"
                        if isinstance(bundle, ModelBundle)
                        else "static_prior"
                    ),
                    confidence=(
                        "high"
                        if not isinstance(bundle, ModelBundle)
                        and pd.notna(row.get("delay_trend_minutes"))
                        and float(row.get("current_delay_minutes"))
                        <= float(
                            (getattr(bundle, "transition_selective_rule", None) or {}).get(
                                "previous_delay_max_minutes", -1
                            )
                        )
                        and abs(float(row.get("delay_trend_minutes")))
                        <= float(
                            (getattr(bundle, "transition_selective_rule", None) or {}).get(
                                "previous_segment_change_max_minutes", -1
                            )
                        )
                        else "standard"
                    ),
                )
            )
        source_updated = data.get("lastUpdatedAt")
        freshness_seconds: float | None = None
        if source_updated:
            try:
                freshness_seconds = max(
                    0.0,
                    float(
                        (
                            pd.Timestamp(envelope["captured_at"])
                            - pd.Timestamp(source_updated)
                        ).total_seconds()
                    ),
                )
            except (TypeError, ValueError):
                freshness_seconds = None
        return EtaResponse(
            train_number=str(data["trainNumber"]),
            journey_date=str(data.get("startDate") or journey_date or "unknown"),
            source_updated_at=data.get("lastUpdatedAt"),
            model_name=model_name,
            trained_through=trained_through,
            etas=etas,
            current_station_code=current_location.get("stationCode"),
            current_delay_minutes=_finite_or_none(data.get("delayMinutes")),
            current_speed_kmh=_finite_or_none(current_location.get("speedKmh")),
            source_freshness_seconds=freshness_seconds,
            response_generated_at=_timestamp_string(envelope["captured_at"]),
            journey_status="running",
            status_message=None,
        )

    return app
