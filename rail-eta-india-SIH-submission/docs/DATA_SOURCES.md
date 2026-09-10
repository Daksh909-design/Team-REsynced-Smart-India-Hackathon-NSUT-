# Data sources and collection policy

## Included live connector: RailRadar

The included connector targets RailRadar's documented live-train endpoint.
Its response contains current telemetry progress, speed, delay, next halt,
diversion/cancellation state, a full route, per-stop planned and actual arrival
and departure timestamps, and station coordinates. It requires a bearer key;
the free sandbox's stated 1,000 requests/month is not a national production
quota. Confirm your plan, retention rights, and permitted use with the provider
before collecting.

- [Live train endpoint documentation](https://railradar.in/docs/live-train-status)
- [Provider overview and sandbox statement](https://railradar.in/)

## Schedule and geography backfill

The [Government of India timetable catalogue](https://www.data.gov.in/catalog/indian-railways-train-time-table)
is an appropriate scheduled-route reference. For a current, reproducible
timetable acquisition workflow, the open-source
[railpull project](https://github.com/shwetankg07/railpull) documents a
rate-limited/resumable NTES-based collector and OSM station matching. It warns
that bulk collection/redistribution may be subject to operator terms; this
project does not run it automatically or bundle its output.

Station coordinates and physical route geometry can be derived from
[OpenStreetMap](https://www.openstreetmap.org/copyright) under ODbL, retaining
the required attribution/share-alike obligations.

## Weather

[Open-Meteo's forecast API](https://open-meteo.com/) supplies current/forecast
weather without an API key. Its
[historical weather documentation](https://open-meteo.com/en/docs/historical-weather-api)
states that ERA5/ERA5-Land reanalysis is retrospective. Use it for exploratory
analysis only, not as a feature in a pseudo-real-time backtest. The collector
stores current weather at snapshot time; a production setup should also retain
the forecast issued at each snapshot for every target-horizon weather join.

## Actual-running history: what is and is not available

### Included bootstrap table

The repository includes a downloaded copy of the public
`Master_3M_Delay.csv` table from the Apache-2.0 licensed
[Indian-Railway-Delay-Visualization repository](https://github.com/adityaazad79/Indian-Railway-Delay-Visualization).
It has 142,327 station-level rows across approximately 8,800 train IDs and
fields for train, station, zone, route order, delay, and (where present)
coordinates. Its schema has no date or capture time. The bootstrap command
therefore evaluates by held-out train IDs and labels its artifact as a prior;
it is never passed as if it were a point-in-time live snapshot.

The raw source file is [Master_3M_Delay.csv](https://raw.githubusercontent.com/adityaazad79/Indian-Railway-Delay-Visualization/main/Master_3M_Delay.csv).
Retain the upstream attribution and re-check its redistribution terms before
publishing a derived dataset.

The nationwide RSTGCN study reports 3,892 long-distance passenger trains,
4,735 stations, and operation records during September 2024; its paper says
the data are gathered from runningstatus.in and released for research purposes.
It demonstrates scale but is not treated as an open, redistributable dependency
until the authors' release terms are verified.

- [RSTGCN paper/dataset description](https://cse.iitkgp.ac.in/~abhijnan/papers/chowdhury_rail_network_delay_prediction.pdf)
- [Earlier two-year Indian train-delay research and request path](https://github.com/R-Gaurav/train-delay-estimation)

## Data contract

Raw snapshots have `captured_at` (when this system saw the response),
`provider_last_updated_at`, provider name, request train number, and the raw
payload. Derived labels are keyed by `(train_number, journey_date, station_code,
sequence)` and carry both `actual_arrival_at` and `available_at`. The latter is
the critical anti-leakage timestamp.

For restricted operational feeds, require: source identifier, event time,
availability time, geographic/section key, revision ID, quality flag, and a
written retention/use basis. Never make a feature available to a model snapshot
earlier than its recorded availability time.
