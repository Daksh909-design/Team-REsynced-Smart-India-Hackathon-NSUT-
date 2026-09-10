from __future__ import annotations

import json

from rail_eta.supervised import build_training_frame


def _envelope(
    captured_at: str, *, b_actual: str | None = None, c_actual: str | None = None
) -> dict:
    return {
        "captured_at": captured_at,
        "provider_last_updated_at": captured_at,
        "payload": {
            "success": True,
            "data": {
                "trainNumber": "12919",
                "startDate": "2026-01-01",
                "lastUpdatedAt": captured_at,
                "delayMinutes": 8,
                "train": {"type": "Superfast Express", "category": "Superfast"},
                "currentLocation": {"stationCode": "AAA", "segmentProgress": 0.4, "speedKmh": 55},
                "route": [
                    {
                        "sequence": 1,
                        "stationCode": "AAA",
                        "isHalt": True,
                        "distance": 0,
                        "scheduledDeparture": "2026-01-01T00:00:00+05:30",
                        "actualDeparture": "2026-01-01T00:08:00+05:30",
                    },
                    {
                        "sequence": 2,
                        "stationCode": "BBB",
                        "isHalt": True,
                        "distance": 50,
                        "scheduledArrival": "2026-01-01T01:00:00+05:30",
                        "actualArrival": b_actual,
                    },
                    {
                        "sequence": 3,
                        "stationCode": "CCC",
                        "isHalt": True,
                        "distance": 100,
                        "scheduledArrival": "2026-01-01T02:00:00+05:30",
                        "actualArrival": c_actual,
                    },
                ],
            },
        },
    }


def test_build_only_labels_future_actual_arrivals(tmp_path):
    early = _envelope("2025-12-31T18:45:00Z")
    late = _envelope(
        "2025-12-31T21:00:00Z", b_actual="2025-12-31T19:45:00Z", c_actual="2025-12-31T20:45:00Z"
    )
    root = tmp_path / "journey_date=2026-01-01" / "train=12919"
    root.mkdir(parents=True)
    (root / "early.json").write_text(json.dumps(early), encoding="utf-8")
    (root / "late.json").write_text(json.dumps(late), encoding="utf-8")

    frame, audit = build_training_frame(tmp_path)

    assert audit["snapshot_files_read"] == 2
    assert set(frame["target_station_code"]) == {"BBB", "CCC"}
    assert (frame["actual_arrival_at"] > frame["observed_at"]).all()
    # The actual is first visible in the later snapshot, never in the early input.
    assert (frame["available_at"] > frame["observed_at"]).all()
