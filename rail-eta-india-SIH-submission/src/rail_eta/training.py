"""Temporal evaluation and regularised scikit-learn/LightGBM ETA models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from rail_eta.supervised import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


@dataclass
class ModelBundle:
    """Everything serving needs, including the fitted feature transformation."""

    model_name: str
    preprocessor: ColumnTransformer
    model: Any
    interval_half_width_minutes: float
    history_profile: pd.DataFrame
    feature_names: list[str]
    trained_through: str

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(_feature_frame(frame))

    def predict_delay(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = np.asarray(self.model.predict(self.transform(frame)), dtype=float)
        radius = float(self.interval_half_width_minutes)
        return predictions, predictions - radius, predictions + radius


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"feature frame is missing columns: {sorted(missing)}")
    result = frame[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype("string").fillna("UNKNOWN")
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )


def _temporal_splits(frame: pd.DataFrame) -> tuple[pd.Index, pd.Index, pd.Index]:
    """70/15/15 chronological split with no random mixing of future traffic."""
    ordered_times = pd.Index(sorted(pd.to_datetime(frame["observed_at"], utc=True).unique()))
    if len(ordered_times) < 10:
        raise ValueError(
            "need at least 10 distinct snapshot times for a temporal train/validation/test split"
        )
    train_end = max(1, int(len(ordered_times) * 0.70))
    validation_end = max(train_end + 1, int(len(ordered_times) * 0.85))
    train_times = ordered_times[:train_end]
    validation_times = ordered_times[train_end:validation_end]
    test_times = ordered_times[validation_end:]
    stamps = pd.to_datetime(frame["observed_at"], utc=True)
    return (
        frame.index[stamps.isin(train_times)],
        frame.index[stamps.isin(validation_times)],
        frame.index[stamps.isin(test_times)],
    )


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    absolute = np.abs(y_true - prediction)
    denominator = np.abs(y_true) + np.abs(prediction) + 1.0
    return {
        "mae_minutes": float(mean_absolute_error(y_true, prediction)),
        "rmse_minutes": float(mean_squared_error(y_true, prediction) ** 0.5),
        "smape_percent": float(200 * np.mean(absolute / denominator)),
        "within_5_minutes": float(np.mean(absolute <= 5)),
        "within_10_minutes": float(np.mean(absolute <= 10)),
        "within_15_minutes": float(np.mean(absolute <= 15)),
    }


def _fit_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    categorical_indices: list[int],
) -> LGBMRegressor:
    # Conservative capacity and strong regularisation are intentional. The number
    # of leaves/minimum child size should be revisited only after an untouched
    # rolling-origin evaluation, not tuned against the final holdout.
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=24,
        max_depth=-1,
        min_child_samples=120,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.4,
        reg_lambda=2.0,
        random_state=20260909,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        categorical_feature=categorical_indices,
        eval_set=[(x_validation, y_validation)],
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
    )
    return model


def _history_profile(train: pd.DataFrame) -> pd.DataFrame:
    """Serving fallback, learned strictly from the first chronological split."""
    if train.empty:
        return pd.DataFrame(columns=["target_station_code"])
    cutoff = train["observed_at"].max()
    # A long journey observed before the cutoff can arrive after it. Its label is
    # valid for supervised training but was not yet available for an online
    # station-history profile at model cutover.
    if "available_at" in train.columns:
        known_by_cutoff = train.loc[train["available_at"] <= cutoff]
    else:
        # Unit-test and externally materialised tables may omit the audit-only
        # availability column; the live builder always supplies it.
        known_by_cutoff = train
    grouped = (
        known_by_cutoff.groupby("target_station_code", dropna=False)[TARGET]
        .agg(["median", "count"])
        .reset_index()
    )
    return grouped.rename(
        columns={
            "median": "historical_station_delay_median_28d",
            "count": "historical_station_observations_28d",
        }
    )


def train(frame: pd.DataFrame) -> tuple[ModelBundle, dict[str, Any]]:
    """Fit baselines and a LightGBM model, choosing with validation only."""
    required = {TARGET, "observed_at", "scheduled_arrival_at", "current_delay_minutes", *FEATURES}
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"training data is missing required columns: {sorted(absent)}")
    cleaned = frame.loc[frame[TARGET].notna() & frame["observed_at"].notna()].copy()
    if len(cleaned) < 500:
        raise ValueError(
            "need at least 500 labelled examples before training a production ETA model"
        )
    train_idx, validation_idx, test_idx = _temporal_splits(cleaned)
    train_frame, validation_frame, test_frame = (
        cleaned.loc[train_idx],
        cleaned.loc[validation_idx],
        cleaned.loc[test_idx],
    )
    if min(len(train_frame), len(validation_frame), len(test_frame)) < 50:
        raise ValueError("each chronological split needs at least 50 rows")

    preprocessor = _preprocessor()
    x_train = preprocessor.fit_transform(_feature_frame(train_frame))
    x_validation = preprocessor.transform(_feature_frame(validation_frame))
    x_test = preprocessor.transform(_feature_frame(test_frame))
    y_train = train_frame[TARGET].to_numpy(dtype=float)
    y_validation = validation_frame[TARGET].to_numpy(dtype=float)
    y_test = test_frame[TARGET].to_numpy(dtype=float)
    categorical_indices = list(
        range(x_train.shape[1] - len(CATEGORICAL_FEATURES), x_train.shape[1])
    )

    sklearn_baseline = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=24,
        min_samples_leaf=120,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=20260909,
    ).fit(x_train, y_train)
    lightgbm = _fit_lightgbm(x_train, y_train, x_validation, y_validation, categorical_indices)

    candidates: dict[str, Any] = {
        "sklearn_hist_gradient_boosting": sklearn_baseline,
        "lightgbm_l1": lightgbm,
    }
    validation_scores = {
        name: _metrics(y_validation, model.predict(x_validation))
        for name, model in candidates.items()
    }
    champion_name = min(validation_scores, key=lambda name: validation_scores[name]["mae_minutes"])
    champion = candidates[champion_name]
    validation_residual = np.abs(y_validation - champion.predict(x_validation))
    # Split conformal interval: an honest uncertainty band calibrated on the
    # middle time window, then reported only once on the future holdout.
    interval_half_width = float(np.quantile(validation_residual, 0.90, method="higher"))
    test_prediction = np.asarray(champion.predict(x_test), dtype=float)
    test_metrics = _metrics(y_test, test_prediction)
    test_metrics["interval_90_coverage"] = float(
        np.mean(np.abs(y_test - test_prediction) <= interval_half_width)
    )
    test_metrics["interval_half_width_minutes"] = interval_half_width

    static_schedule = np.zeros_like(y_test)
    current_delay = test_frame["current_delay_minutes"].fillna(0).to_numpy(dtype=float)
    report: dict[str, Any] = {
        "rows": {
            "train": len(train_frame),
            "validation": len(validation_frame),
            "test": len(test_frame),
        },
        "time_ranges": {
            "train_end": str(train_frame["observed_at"].max()),
            "validation_end": str(validation_frame["observed_at"].max()),
            "test_start": str(test_frame["observed_at"].min()),
            "test_end": str(test_frame["observed_at"].max()),
        },
        "validation": validation_scores,
        "selected_model": champion_name,
        "test": {
            "static_schedule": _metrics(y_test, static_schedule),
            "current_delay_propagation": _metrics(y_test, current_delay),
            "champion": test_metrics,
        },
        "feature_count": len(FEATURES),
        "leakage_controls": [
            "chronological train/validation/test split",
            "label only from a later actual-arrival observation",
            "historical aggregates gated by snapshot availability time",
            "weather is collection-time forecast/current data, never future reanalysis",
            "model selection uses validation, not the final test window",
        ],
    }
    bundle = ModelBundle(
        model_name=champion_name,
        preprocessor=preprocessor,
        model=champion,
        interval_half_width_minutes=interval_half_width,
        history_profile=_history_profile(train_frame),
        feature_names=list(FEATURES),
        trained_through=str(train_frame["observed_at"].max()),
    )
    return bundle, report


def save_bundle(
    bundle: ModelBundle, report: dict[str, Any], model_path: Path, report_path: Path
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def load_bundle(model_path: Path) -> ModelBundle:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("model artifact is not a rail_eta ModelBundle")
    return bundle
