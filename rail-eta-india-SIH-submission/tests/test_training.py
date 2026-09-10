from __future__ import annotations

import numpy as np
import pandas as pd

from rail_eta.supervised import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from rail_eta.training import train


def test_training_uses_chronological_holdout_and_predicts():
    rng = np.random.default_rng(7)
    rows = 600
    observed = pd.date_range("2026-01-01", periods=30, freq="12h", tz="UTC").repeat(20)
    frame = pd.DataFrame(
        {"observed_at": observed, "scheduled_arrival_at": observed + pd.Timedelta(hours=2)}
    )
    for feature in NUMERIC_FEATURES:
        frame[feature] = rng.normal(10, 3, rows)
    frame["current_delay_minutes"] = rng.uniform(0, 50, rows)
    frame["remaining_distance_km"] = rng.uniform(5, 800, rows)
    frame["scheduled_remaining_minutes"] = rng.uniform(10, 900, rows)
    for feature in CATEGORICAL_FEATURES:
        frame[feature] = "A"
    frame["target_station_code"] = np.where(np.arange(rows) % 2, "BBB", "CCC")
    frame[TARGET] = 0.65 * frame["current_delay_minutes"] + rng.normal(0, 4, rows)

    bundle, report = train(frame)
    point, lower, upper = bundle.predict_delay(frame.iloc[:4])

    assert report["time_ranges"]["test_start"] > report["time_ranges"]["train_end"]
    assert len(point) == 4
    assert np.all(lower <= point)
    assert np.all(point <= upper)
