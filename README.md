# Team-REsynced-Smart-India-Hackathon-NSUT-
This is the official repository for the project work of Team REsynced from NSUT for SIH, our PS is #28 Dynamic Forecast of Expected Time of Arrival (ETA) for Coaching Trains from the ministry of railways.
# RailPulse — Dynamic ETA Forecasting for Coaching Trains

We built RailPulse for SIH Problem Statement #28: Dynamic Forecast of Expected Time of Arrival for Indian Railways coaching trains.

RailPulse combines live train-running data, route schedules, delay history, weather context, and machine-learning models to forecast arrival times at upcoming stations.

## Key Features

- Live train-status integration through RailRadar
- FastAPI backend for mobile apps and dashboards
- Responsive RailPulse web interface
- ETA predictions for upcoming halting stations
- Current delay and source-freshness display
- Scikit-learn baseline and LightGBM forecasting model
- Short-horizon transition model for live delay propagation
- Prediction intervals showing uncertainty
- Finished, cancelled, diverted, and unavailable journey handling
- Snapshot collection for future point-in-time training
- API key protection through server-side environment variables

## Architecture

┌─────────────────────────────────────────────────────────────┐
│     RailRadar Live Running Data + Open-Meteo Weather Data   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Live Data Connector                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Immutable JSON Snapshots                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                Point-in-Time Feature Builder                │
├─────────────────────────────────────────────────────────────┤
│  ├── Route and schedule features                            │
│  ├── Current delay and progress                             │
│  ├── Historical station/train delay                         │
│  ├── Weather features                                       │
│  └── Time and spatial features                              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Machine-Learning Forecasting Layer              │
├─────────────────────────────────────────────────────────────┤
│  ├── Scikit-learn baseline                                  │
│  ├── LightGBM delay model                                   │
│  ├── Transition delay model                                 │
│  └── Prediction interval calibration                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI ETA Service                     │
├─────────────────────────────────────────────────────────────┤
│  ├── GET /health                                            │
│  ├── GET /docs                                              │
│  └── GET /v1/eta/{train_number}                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   RailPulse Web Frontend                    │
└─────────────────────────────────────────────────────────────┘
