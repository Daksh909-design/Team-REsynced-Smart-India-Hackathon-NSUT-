# Rail ETA India

A production-oriented ETA forecasting project for Indian Railways coaching
trains. It predicts the eventual arrival time at every upcoming halting station
from a live running-status snapshot, rather than forwarding the published
schedule or the current delay unchanged.

The project is intentionally honest about data: no simulated delay records are
included and no accuracy number is claimed before real labelled journeys have
been collected. The pipeline persists live observations, observes their final
actual arrivals later, creates strictly point-in-time training examples, and
then trains only on those real outcomes.

## What is implemented

```text
licensed live running feed ──> immutable JSON snapshots ──> labelled examples
  location / speed / route          actual arrival later       point-in-time history
  delay / diversion / platform              │                         │
  schedule / station coordinates       Open-Meteo forecast           ▼
                         └──────────────────────> temporal validation ──> ETA API
                                                         sklearn + LightGBM
```

- A live ingestion connector for the documented RailRadar endpoint, which has
  per-stop planned/actual times, current progress and speed, diversions,
  cancellations, and station coordinates.
- Immutable snapshots and reconstruction of *what was known at prediction
  time*. A later actual arrival is the label; published ETA is never the label.
- Current weather at collection time through Open-Meteo. Missing weather is
  explicit and safely imputed rather than fabricated.
- Leakage-safe historical station/train-station features, local snapshot-based
  congestion, schedule geometry, current delay, route progress, speed, train
  category, and time-of-week features.
- A regularised scikit-learn `HistGradientBoostingRegressor` baseline and a
  regularised LightGBM L1 model. They compete on a chronological validation
  period; the future test period is used exactly once for the report.
- Split-conformal 90% delay/arrival intervals, not just point estimates.
- Residual adjacent-station transition model for the first few upcoming stops,
  with validation-selected confidence gating and explicit `prediction_mode`/
  `confidence` fields in each API stop result.
- FastAPI endpoint for mobile apps, station boards, and dashboards. API calls
  are themselves stored as future training evidence.

## Data decision and provenance

There is a substantial distinction between real **timetable** data and real
**actual-running** data. The public Indian Railways timetable catalogue exposes
route, distance, source/destination, and planned arrival/departure fields, but
does not expose a nationwide downloadable history of actual arrivals. A recent
nationwide research dataset similarly says its operation records were collected
from runningstatus.in and covers 3,892 long-distance trains and 4,735 stations
for September 2024; it is research-release material, not bundled here. The
older academic Indian-delay dataset is request-only. See
[data-source notes](docs/DATA_SOURCES.md) for the exact source roles, license
checks, and collection plan.

This project therefore uses a defensible route to real training data:

1. Obtain a permitted live-status feed (the included connector expects a
   RailRadar production key) and schedule snapshots every 5 minutes while the
   target service is running.
2. Keep capturing through final arrival. The same feed later supplies actual
   arrival timestamps for passed stops; those are the outcome labels.
3. Train only after at least 500 labelled examples across at least 10 distinct
   snapshot times; in practice, target 3--6 months and a stratified sample of
   routes, zones, train classes, monsoon/fog seasons, and congested junctions.
4. Treat signal aspects, block occupancy, TSRs, maintenance blocks, crew/rake
   state, and true control-office congestion as restricted operational feeds.
   Join them only under an approved Indian Railways data agreement, with an
   `available_at` timestamp. The base model works without claiming to have
   those unavailable fields.

Do not substitute a Kaggle "Indian Railway Delay" file for this pipeline: many
such listings explicitly say they are manually prepared or simulated. That
would make an impressive-looking but untrustworthy ETA model.

## Setup

Requires Python 3.11--3.14. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `RAILRADAR_API_KEY` in `.env`; it is never committed. The connector uses
the documented `Authorization: Bearer` flow.

## Live browser demo

The service includes a polished RailPulse web client at `/`. It keeps the API
key on the server, requests a fresh RailRadar snapshot for the selected train,
and renders the current station, live delay, model signal, ETA windows, and
the complete upcoming-stop response. Completed journeys are identified from the
provider status and shown as finished rather than being assigned a misleading
zero-minute running delay.

```powershell
.\.venv\Scripts\rail-eta serve --host 127.0.0.1 --port 8000
Start-Process http://127.0.0.1:8000/
```

Use a five-digit train number and journey date, then press **Forecast ETA**.
The UI is a thin client over the same `/v1/eta/{train_number}` endpoint used by
mobile apps and station dashboards; it does not invent offline values when the
live provider is unavailable.

## End-to-end workflow

Capture a bounded real batch (use a scheduler to repeat this roughly every five
minutes while each service runs):

```powershell
.\.venv\Scripts\rail-eta collect --trains 12919,12002
```

After journeys finish and later snapshots contain actual arrivals, materialise
the leakage-safe supervised table:

```powershell
.\.venv\Scripts\rail-eta build
```

Train and write a model plus a reproducible report:

```powershell
.\.venv\Scripts\rail-eta train
```

The project also includes a real public station-level delay table at
`data/raw/Master_3M_Delay.csv` (142,327 rows). It contains no timestamp, so it
is not silently mixed into the dynamic backtest. Train a broad cold-start prior
from it with a train-ID group holdout:

```powershell
.\.venv\Scripts\rail-eta bootstrap --csv data/raw/Master_3M_Delay.csv
```

This writes `models/bootstrap_prior.joblib` and
`reports/bootstrap_report.json`. The prior is useful for unseen live routes
until enough point-in-time snapshots have accumulated; it is not a replacement
for the dynamic model.

The bootstrap report also evaluates the online transition path: 87.5% within
15 minutes overall on the untouched train-ID holdout, and 91.0% within 15
minutes on 80.3% of eligible cases after a validation-selected confidence
gate. The latter is selective accuracy, not an all-trip guarantee.

Start the service:

```powershell
.\.venv\Scripts\rail-eta serve --host 0.0.0.0 --port 8000
```

Then query `GET /v1/eta/{train_number}?journey_date=YYYY-MM-DD`. The response
contains one ETA and a lower/upper arrival bound for every future halting stop.
`GET /docs` exposes the OpenAPI UI.

## Evaluation policy

The report at `reports/model_report.json` includes MAE, RMSE, SMAPE, and
within-5/10/15-minute accuracy for three references: static timetable, current
delay carried forward, and the selected model. It also reports empirical
coverage of its nominal 90% interval.

The safeguards are deliberate:

- no random row split (near-duplicate observations of one trip cannot land on
  both sides);
- the model selection window precedes a final untouched future test window;
- historical statistics use only outcomes that were already captured before a
  snapshot;
- collection-time forecast/current weather only; ERA5/reanalysis is prohibited
  in a live-style backtest;
- categorical station/train fields use unknown-safe encoding;
- minimum leaf sizes, L1 loss, subsampling, and regularisation are fixed
  conservative defaults, not endlessly tuned to the held-out test.

Use rolling-origin retraining and compare the new candidate with the deployed
model by route, zone, train category, horizon, monsoon/fog period, and delay
severity before promoting it. Details are in [the model card](docs/MODEL_CARD.md).

## Project layout

```text
src/rail_eta/connectors/  permitted live and weather data clients
src/rail_eta/supervised.py point-in-time examples and labels
src/rail_eta/training.py  baselines, LightGBM, intervals, report
src/rail_eta/api.py       FastAPI integration surface
tests/                    no-leakage and training smoke tests
docs/                     source, model, and operations documentation
```

## Operational notes

The free RailRadar sandbox is capped; it is suitable for integration testing,
not nationwide five-minute polling. For thousands of trains, use an approved
high-volume provider/CRIS agreement, partition collection by train or station
board, queue ingestion, retain raw payloads in object storage, and move the
derived feature/outcome tables into a warehouse. The online API remains
stateless except for its model; a production deployment should load snapshots
and real-time congestion from that shared store.

This is a decision-support system. It should show source freshness and interval
width to passengers/staff, suppress predictions for cancelled/diverted services
where route certainty is inadequate, and never be used as a safety or train
control system.
