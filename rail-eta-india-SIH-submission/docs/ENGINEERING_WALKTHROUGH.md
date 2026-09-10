# Engineering walkthrough

## 1. What the current score means

The reported 51.3% is **within 15 minutes on a train-ID group holdout**. It is
not classification accuracy and it is not a promise about every live ETA. The
public bootstrap table has no timestamps, so it cannot tell us what was known
at 09:05 before a train reached a station. I therefore kept it separate from
the dynamic ETA evaluation.

The corrected run uses 70% of train IDs for fitting, 15% for validation, and
15% for a final untouched test. The current artifact reports:

| Metric | Global median | LightGBM |
| --- | ---: | ---: |
| MAE | 42.28 min | 35.85 min |
| Within 5 min | 14.9% | 21.9% |
| Within 10 min | 28.9% | 39.2% |
| Within 15 min | 43.7% | 51.3% |

A 95% prediction interval covers 92.4% of the untouched test rows, but its
half-width is about 107 minutes. That is the honest distinction: 90%+ coverage
is possible by returning a wide uncertainty band; 90% exact ETA accuracy is
not supported by this data.

There is also an explicitly online transition path. For rows after the first
station, the preceding station's observed delay is an upstream feature, so it
does not leak the target. A residual LightGBM model predicts the next-station
delay change from that delay, recent trend, route position, station, origin,
train class, and a fit-only station residual prior:

| Transition metric (untouched train-ID test) | Result |
| --- | ---: |
| Carry-forward baseline, within 15 min | 84.3% |
| Transition LightGBM, within 15 min | 87.5% |
| Transition LightGBM, MAE | 7.53 min |
| Validation-selected high-confidence gate, within 15 min | **91.0%** |
| High-confidence gate coverage | 80.3% |

The gate is selected on validation before the final test and marks stable
recent-delay regimes. It is not a claim that every ETA is 91% accurate. The
API exposes `prediction_mode` (`transition` or `static_prior`) and `confidence`
(`high` or `standard`) so a demo can show this distinction honestly.

## 2. How it is coded

`connectors/railradar.py` calls the live endpoint, validates the response, and
writes immutable JSON envelopes. Each envelope retains the provider payload,
capture time, provider update time, and source name. `connectors/weather.py`
adds collection-time Open-Meteo weather without making historical reanalysis a
future feature.

`supervised.py` converts an envelope into one row per future halting stop. It
extracts current delay, speed, segment progress, remaining distance, planned
time, remaining halts, station/train categories, time-of-week, weather, and a
15-minute local snapshot congestion proxy.

The label is not the provider's ETA. It is the actual arrival from a later
snapshot minus the scheduled arrival. A row is kept only when the actual is
strictly after the feature snapshot. Historical station medians are updated
only when their `available_at` timestamp is before the feature timestamp.

`training.py` fits a scikit-learn `HistGradientBoostingRegressor` baseline and
a regularised LightGBM L1 regressor. The live pipeline uses chronological
70/15/15 splitting, early stopping on validation, minimum leaf sizes, row and
column subsampling, L1/L2 regularisation, and an interval calibrated from
validation residuals.

`bootstrap.py` adapts the public station-level CSV into a cold-start prior. Its
features are station, zone, origin station, train class, route position, route
length, and coordinates when available. One-hot categorical encoding handles
unseen stations. It also builds an adjacent-station residual model whose target
is `delay_at_target - delay_at_previous_station`. The holdout is grouped by
train ID so rows from one route do not leak into both fit and test.

`api.py` first loads `models/eta_model.joblib` when a timestamped live model
exists. Otherwise it loads `models/bootstrap_prior.joblib`, maps live route
stops into the prior's feature contract, and uses the transition model for the
first three upcoming stops when current and previous delay observations are
available. Longer horizons and incomplete payloads fall back to the static
prior. Every live prediction is persisted as a future training snapshot.

## 3. Why 90% exact accuracy is not a legitimate tuning target yet

The public table is a historical station delay profile without date, weather,
current location, upstream delay, congestion, signal, or disruption state. A
model cannot infer an unscheduled block or a preceding train from station code
and route position. Attempting to force 90% would mean leakage, a random row
split, a target definition such as “late/not late,” or a very broad ETA range.

For a real-time model, collect at least 3--6 months of 5-minute snapshots for
many train classes, zones, seasons, and long/short horizons. Then train on
the residual `future_arrival_delay - current_delay` so the model learns
recovery and propagation, not just the current delay level. The transition
result is a useful demo bridge; it should be replaced or recalibrated with
timestamped snapshot outcomes before production claims.

## 4. Highest-impact improvements

1. **Real labels at scale.** Capture every active service through completion;
   target at least 50k journeys and hundreds of thousands of snapshot/stop
   examples. This is the biggest improvement, not a hyperparameter tweak.
2. **Operational features.** Add permitted section running-time history,
   preceding-train delays, headway/congestion, signal aspects, speed
   restrictions, maintenance blocks, weather forecast at target time, rake and
   crew continuity, and station dwell history. Each needs an availability
   timestamp.
3. **Residual and horizon models.** Fit separate or interacting models for
   0--30, 30--120, and 120+ minute horizons; predict delay change from the
   current state and blend with a robust route/station prior.
4. **Hierarchical calibration.** Calibrate intervals by zone, train category,
   horizon, monsoon/fog season, and disruption regime. Monitor coverage rather
   than assuming one national interval works everywhere.
5. **Model selection.** Compare LightGBM, CatBoost-style categorical boosting,
   quantile LightGBM, and a residual ensemble only on rolling-origin validation.
   Keep the final future window untouched.
6. **Data quality.** Track feed freshness, corrected timestamps, cancellations,
   diversions, duplicate snapshots, and station-code changes. Bad labels can
   dominate any model choice.

## 5. Reproduce the current run

```powershell
.\.venv\Scripts\rail-eta bootstrap --csv data\raw\Master_3M_Delay.csv
.\.venv\Scripts\pytest -q
.\.venv\Scripts\ruff check src tests
```

For the dynamic model, configure a permitted live key and run `collect` every
five minutes, then `build`, `train`, and `serve` after enough completed
journeys exist.
