"""Configuration loaded only from environment variables or explicit arguments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    railradar_api_key: str
    railradar_base_url: str
    snapshot_dir: Path
    weather_enabled: bool

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            railradar_api_key=os.getenv("RAILRADAR_API_KEY", "").strip(),
            railradar_base_url=os.getenv(
                "RAILRADAR_BASE_URL", "https://api.railradar.in/v1"
            ).rstrip("/"),
            snapshot_dir=Path(os.getenv("SNAPSHOT_DIR", "data/raw/snapshots")),
            weather_enabled=os.getenv("WEATHER_ENABLED", "true").lower() in {"1", "true", "yes"},
        )
