# Model card: Rail ETA India

## Intended use

Predict coaching-train ETA at upcoming halting stations for passenger display,
station operations, crew/resource planning, and downstream logistics.

## Prediction target

`actual_arrival_at - scheduled_arrival_at` in minutes. The API adds the
predicted delay to the scheduled arrival to produce ETA. Negative values are
kept: early arrival is operationally meaningful.

## Features available at prediction time

- current delay, current route position/progress, speed, remaining distance,
  planned time remaining, stops remaining;
- train type/category, stations, time/day/weekend;
- diversion flag and locally measured snapshot congestion;
- rolling station and train-station delay profiles gated by when an actual was
  captured;
- current weather collected with the snapshot.

Platform prediction, future actual times, post-snapshot cancellation outcome,
and revised future ETA are not features. Signal/section/block/rake feeds are
optional only after permitted integration with availability timestamps.

## Training and selection

The first 70% of snapshot times train the models, the next 15% chooses between
scikit-learn `HistGradientBoostingRegressor` and a LightGBM L1 regressor, and
the last 15% is held out for final reporting. No random split is used. LightGBM
has 24 leaves, a 120-row minimum child size, row/column subsampling, L1/L2
regularisation, early stopping, and a fixed seed. These are conservative
starting settings, not a promise of universal optimality.

A 90% split-conformal half-width is calculated from validation residuals and
applied to final ETA. Monitor actual coverage: undercoverage requires
recalibration or retraining.

The cold-start artifact also contains a transition residual model. It predicts
the delay change to the next station from upstream delay observations and is
used for up to three upcoming stops when the live payload contains enough route
history. On the untouched train-ID holdout it achieved 87.5% within 15 minutes
overall; a validation-selected confidence gate achieved 91.0% within 15
minutes at 80.3% coverage. The latter is selective accuracy, not an all-trip
guarantee.

## Known limitations

- The base feed may have reporting latency or corrected arrivals.
- Snapshot congestion is a proxy, not track-circuit occupancy.
- Rare disruptions, diversions, and long horizons have wider irreducible error.
- A station code new to training is represented as unknown and should be
monitored separately.
- Weather is currently a current-location feature; route/horizon weather needs
  an archived forecast join for full fidelity.
- The included public bootstrap table has no timestamps. Its transition result
  is a route-order proxy and must not be described as a nationwide live
  backtest. Timestamped provider snapshots remain the production training
  source of truth.

## Promotion gates

Promote only if the candidate improves or is non-inferior to the current model
on a future rolling window, has 90% interval coverage near its target, and does
not materially regress on any major zone, train category, horizon bucket,
monsoon/fog period, or high-delay cohort. Keep static schedule and
current-delay-forward baselines in every report.

Retrain on a fixed cadence (e.g. weekly) or when population drift, source
freshness, or error alerts trigger. Retain model version, data cutoff, feature
schema, source revisions, metrics, and the approval decision for every release.
