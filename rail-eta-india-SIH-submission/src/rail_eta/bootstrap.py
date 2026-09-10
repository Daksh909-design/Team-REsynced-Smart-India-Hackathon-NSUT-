"""Train a real-data delay prior from the public station-level delay table.

This is deliberately separate from the live ETA model: the public table has
no timestamp or snapshot availability field, so it cannot support a dynamic
point-in-time backtest. It is still useful as a broad train/station prior and
as a cold-start fallback while the live collector builds history.
"""

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
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BOOTSTRAP_CATEGORICALS = ["station_code", "zone", "origin_code", "train_class"]
BOOTSTRAP_NUMERICS = ["stop_index", "route_stop_count", "latitude", "longitude"]
TRANSITION_CATEGORICALS = ["station_code", "zone", "origin_code", "train_class"]
TRANSITION_NUMERICS = [
    "stop_index",
    "route_stop_count",
    "prev_delay",
    "prev_prev_delay",
    "prev_delta",
    "station_delta_prior",
]


def _class_from_name(name: str) -> str:
    tokens = str(name).upper()
    for label, words in {
        "RAJDHANI": ("RAJDHANI", "RAJ"),
        "DURONTO": ("DURONTO",),
        "SHATABDI": ("SHATABDI",),
        "HUMSAFAR": ("HUMSAFAR",),
        "GARIB_RATH": ("GARIB", "RATH"),
        "SUPERFAST": ("SUPER", "SF"),
        "EXPRESS": ("EXPRESS",),
        "PASSENGER": ("PASSENGER", "LOCAL"),
    }.items():
        if any(word in tokens for word in words):
            return label
    return "OTHER"


def load_public_delay_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Train_No", "Train_Name", "Station_Code", "Delay", "Zone"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"public delay table is missing columns: {sorted(missing)}")
    frame = frame.rename(
        columns={
            "Train_No": "train_number",
            "Train_Name": "train_name",
            "Station_Code": "station_code",
            "Station_Name": "station_name",
            "Delay": "delay_minutes",
            "Zone": "zone",
            "Long": "longitude",
            "Lat": "latitude",
        }
    )
    frame["train_number"] = frame["train_number"].astype("string").str.strip()
    frame["station_code"] = frame["station_code"].astype("string").str.strip()
    frame["zone"] = frame["zone"].astype("string").str.strip()
    frame["train_name"] = frame["train_name"].astype("string").fillna("")
    frame["delay_minutes"] = pd.to_numeric(frame["delay_minutes"], errors="coerce")
    frame = frame.loc[
        frame["train_number"].notna()
        & frame["station_code"].notna()
        & frame["delay_minutes"].notna()
    ].copy()
    frame["stop_index"] = frame.groupby("train_number", sort=False).cumcount()
    frame["route_stop_count"] = frame.groupby("train_number")["station_code"].transform("size")
    frame["origin_code"] = frame.groupby("train_number")["station_code"].transform("first")
    frame["train_class"] = frame["train_name"].map(_class_from_name)
    # The public table is ordered by route for each train.  These upstream-only
    # fields are valid at prediction time when the preceding station's running
    # delay is known; they are never derived from the target row itself.
    frame["prev_delay"] = frame.groupby("train_number", sort=False)["delay_minutes"].shift(1)
    frame["prev_prev_delay"] = frame.groupby("train_number", sort=False)["delay_minutes"].shift(2)
    frame["prev_delta"] = frame["prev_delay"] - frame["prev_prev_delay"]
    for column in ("latitude", "longitude"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


@dataclass
class BootstrapBundle:
    preprocessor: ColumnTransformer
    model: LGBMRegressor
    global_median: float
    station_medians: dict[str, float]
    interval_half_width_minutes: float
    transition_preprocessor: ColumnTransformer | None = None
    transition_model: LGBMRegressor | None = None
    transition_interval_half_width_minutes: float | None = None
    transition_station_delta_medians: dict[str, float] | None = None
    transition_selective_rule: dict[str, float] | None = None
    transition_max_horizon_stops: int = 3

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.preprocessor.transform(frame))

    def predict_live(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict a live route, using upstream delay propagation for near stops.

        The transition model is trained on adjacent station pairs.  It is used
        only when the live feed supplies a current delay and for the first few
        upcoming stops; the static prior remains the conservative fallback for
        longer horizons or incomplete provider payloads.
        """
        static = np.asarray(self.predict(frame), dtype=float)
        radii = np.full(len(frame), float(self.interval_half_width_minutes), dtype=float)
        if self.transition_model is None or self.transition_preprocessor is None:
            return static, static - radii, static + radii
        if "current_delay_minutes" not in frame.columns:
            return static, static - radii, static + radii
        current_delay = pd.to_numeric(frame["current_delay_minutes"], errors="coerce")
        if current_delay.isna().all():
            return static, static - radii, static + radii

        # A static station prior has no evidence that a live delay has been
        # recovered.  Without this guard an extreme live delay (for example
        # 345 minutes) could be replaced by a generic prior (for example
        # 50 minutes) at a distant stop, which is operationally misleading.
        # Carry the observed delay forward as a conservative, data-driven
        # floor.  This is not a hardcoded train/value: it is recomputed from
        # the provider's latest snapshot for every request.  The transition
        # model may still refine the estimate for nearby stops, but it cannot
        # erase a delay that is currently observed without learned evidence.
        observed_delay = current_delay.to_numpy(dtype=float)
        finite_delay = np.isfinite(observed_delay)
        static[finite_delay] = np.maximum(static[finite_delay], observed_delay[finite_delay])

        current_sequence = pd.to_numeric(
            frame.get("current_sequence", pd.Series(0, index=frame.index)), errors="coerce"
        ).fillna(0)
        order = np.argsort(pd.to_numeric(frame["stop_index"], errors="coerce").fillna(0).to_numpy())
        known_delay = current_delay.dropna()
        previous = float(known_delay.iloc[0]) if not known_delay.empty else float("nan")
        prior_delay = pd.to_numeric(
            frame.get("previous_delay_minutes", pd.Series(dtype=float)), errors="coerce"
        )
        prior_known = prior_delay.dropna()
        previous_previous = float(prior_known.iloc[0]) if not prior_known.empty else float("nan")
        transition_radius = float(
            self.transition_interval_half_width_minutes or self.interval_half_width_minutes
        )
        for position in order:
            if not np.isfinite(previous):
                break
            row = frame.iloc[[position]].copy()
            target_sequence = float(row["stop_index"].iloc[0]) + 1.0
            horizon = target_sequence - float(current_sequence.iloc[position])
            if horizon < 1 or horizon > self.transition_max_horizon_stops:
                continue
            row["prev_delay"] = previous
            row["prev_prev_delay"] = previous_previous
            row["prev_delta"] = previous - previous_previous
            row["station_delta_prior"] = (
                float((self.transition_station_delta_medians or {}).get(
                    str(row["station_code"].iloc[0]), 0.0
                ))
            )
            transformed = self.transition_preprocessor.transform(row)
            predicted = previous + float(self.transition_model.predict(transformed)[0])
            if np.isfinite(observed_delay[position]):
                predicted = max(predicted, float(observed_delay[position]))
            static[position] = predicted
            radii[position] = transition_radius
            previous_previous, previous = previous, predicted
        lower = static - radii
        lower[finite_delay] = np.maximum(lower[finite_delay], observed_delay[finite_delay])
        return static, lower, static + radii


def live_bootstrap_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Map live route rows to the public-prior feature contract."""
    result = pd.DataFrame(
        {
            "station_code": frame["target_station_code"].astype("string"),
            "zone": "UNKNOWN",
            "origin_code": frame["source_station_code"].astype("string"),
            "train_class": frame["train_type"].map(_class_from_name),
            "stop_index": pd.to_numeric(frame["target_sequence"], errors="coerce") - 1,
            "route_stop_count": pd.to_numeric(frame["total_halts"], errors="coerce"),
            "latitude": pd.to_numeric(frame.get("target_latitude"), errors="coerce"),
            "longitude": pd.to_numeric(frame.get("target_longitude"), errors="coerce"),
            "prev_delay": pd.to_numeric(frame.get("current_delay_minutes"), errors="coerce"),
            "prev_prev_delay": pd.to_numeric(
                frame.get("previous_delay_minutes"), errors="coerce"
            ),
            "prev_delta": pd.to_numeric(frame.get("delay_trend_minutes"), errors="coerce"),
            "current_delay_minutes": pd.to_numeric(
                frame.get("current_delay_minutes"), errors="coerce"
            ),
            "current_sequence": pd.to_numeric(
                frame.get("current_sequence", 0), errors="coerce"
            ),
        }
    )
    return result


def train_bootstrap(frame: pd.DataFrame) -> tuple[BootstrapBundle, dict[str, Any]]:
    if len(frame) < 1000:
        raise ValueError("need at least 1,000 real station delay rows for the bootstrap prior")
    # Group split: all observations from a train stay on one side, preventing
    # route-row memorisation from masquerading as generalisation. Validation is
    # used for early stopping and interval calibration; test stays untouched.
    train_ids = np.sort(frame["train_number"].dropna().unique())
    rng = np.random.default_rng(20260909)
    rng.shuffle(train_ids)
    fit_cutoff = int(len(train_ids) * 0.70)
    validation_cutoff = int(len(train_ids) * 0.85)
    fit_ids = set(train_ids[:fit_cutoff])
    validation_ids = set(train_ids[fit_cutoff:validation_cutoff])
    test_ids = set(train_ids[validation_cutoff:])
    fit = frame[frame["train_number"].isin(fit_ids)].copy()
    validation = frame[frame["train_number"].isin(validation_ids)].copy()
    test = frame[frame["train_number"].isin(test_ids)].copy()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))]),
                BOOTSTRAP_NUMERICS,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                BOOTSTRAP_CATEGORICALS,
            ),
        ],
        sparse_threshold=0.3,
    )
    x_fit = preprocessor.fit_transform(fit)
    x_validation = preprocessor.transform(validation)
    x_test = preprocessor.transform(test)
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=20260909,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_fit,
        fit["delay_minutes"],
        eval_set=[(x_validation, validation["delay_minutes"])],
        callbacks=[early_stopping(80, verbose=False)],
    )
    validation_prediction = np.asarray(model.predict(x_validation), dtype=float)
    prediction = np.asarray(model.predict(x_test), dtype=float)
    y_validation = validation["delay_minutes"].to_numpy(dtype=float)
    y_test = test["delay_minutes"].to_numpy(dtype=float)
    static = np.full_like(y_test, float(fit["delay_minutes"].median()))
    validation_abs_residual = np.abs(y_validation - validation_prediction)

    # Adjacent-station transition model.  This is the high-signal online path:
    # predict the next station's delay from the last observed delay and recent
    # trend, with the same train-ID group split as the static prior.
    transition_fit = fit.loc[fit["prev_delay"].notna()].copy()
    transition_validation = validation.loc[validation["prev_delay"].notna()].copy()
    transition_test = test.loc[test["prev_delay"].notna()].copy()
    transition_station_delta_medians = (
        transition_fit.assign(_delta=transition_fit["delay_minutes"] - transition_fit["prev_delay"])
        .groupby("station_code")["_delta"]
        .median()
        .to_dict()
    )
    for transition_frame in (transition_fit, transition_validation, transition_test):
        transition_frame["station_delta_prior"] = (
            transition_frame["station_code"].map(transition_station_delta_medians).fillna(0.0)
        )
        transition_frame["target_delta"] = (
            transition_frame["delay_minutes"] - transition_frame["prev_delay"]
        )

    def transition_preprocessor_factory() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))]),
                    TRANSITION_NUMERICS,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    TRANSITION_CATEGORICALS,
                ),
            ],
            sparse_threshold=0.3,
        )

    transition_preprocessor = transition_preprocessor_factory()
    tx_fit = transition_preprocessor.fit_transform(transition_fit)
    tx_validation = transition_preprocessor.transform(transition_validation)
    tx_test = transition_preprocessor.transform(transition_test)
    transition_model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=1400,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=100,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=20260909,
        n_jobs=-1,
        verbosity=-1,
    )
    transition_model.fit(
        tx_fit,
        transition_fit["target_delta"],
        eval_set=[(tx_validation, transition_validation["target_delta"])],
        callbacks=[early_stopping(100, verbose=False)],
    )
    transition_validation_prediction = transition_validation["prev_delay"].to_numpy(
        dtype=float
    ) + np.asarray(transition_model.predict(tx_validation), dtype=float)
    transition_prediction = transition_test["prev_delay"].to_numpy(dtype=float) + np.asarray(
        transition_model.predict(tx_test), dtype=float
    )
    transition_y_validation = transition_validation["delay_minutes"].to_numpy(dtype=float)
    transition_y_test = transition_test["delay_minutes"].to_numpy(dtype=float)
    transition_validation_abs_residual = np.abs(
        transition_y_validation - transition_validation_prediction
    )
    transition_interval = float(
        np.quantile(transition_validation_abs_residual, 0.90, method="higher")
    )
    # Select a confidence gate on validation only.  This is selective
    # prediction: the service can mark unstable cases as lower confidence
    # instead of pretending every route state is equally predictable.
    # We use a slightly stricter 91.5% validation target so a tiny validation
    # fluctuation does not turn into an over-broad gate on the untouched test.
    minimum_validation_accuracy = 0.915
    best_rule: tuple[float, float, float, float] | None = None
    for delay_limit in (20, 25, 30, 40, 50, 60, 80, 100, 150, 9999):
        for trend_limit in (5, 10, 15, 20, 30, 45, 60, 9999):
            validation_mask = (
                transition_validation["prev_delay"].le(delay_limit)
                & transition_validation["prev_delta"].abs().le(trend_limit)
            ).fillna(False).to_numpy()
            if validation_mask.sum() == 0:
                continue
            validation_coverage = float(np.mean(validation_mask))
            validation_accuracy = float(
                np.mean(
                    np.abs(
                        transition_y_validation[validation_mask]
                        - transition_validation_prediction[validation_mask]
                    )
                    <= 15
                )
            )
            if validation_accuracy >= minimum_validation_accuracy and (
                best_rule is None
                or validation_coverage > best_rule[0]
                or (
                    validation_coverage == best_rule[0]
                    and validation_accuracy > best_rule[1]
                )
            ):
                best_rule = (validation_coverage, validation_accuracy, delay_limit, trend_limit)
    if best_rule is None:
        best_rule = (0.0, 0.0, 30.0, 10.0)
    _, _, selected_delay_limit, selected_trend_limit = best_rule
    stable_mask = (
        transition_test["prev_delay"].le(selected_delay_limit)
        & transition_test["prev_delta"].abs().le(selected_trend_limit)
    ).fillna(False).to_numpy()

    def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        return {
            "mae_minutes": float(mean_absolute_error(y, p)),
            "rmse_minutes": float(mean_squared_error(y, p) ** 0.5),
            "within_5_minutes": float(np.mean(np.abs(y - p) <= 5)),
            "within_10_minutes": float(np.mean(np.abs(y - p) <= 10)),
            "within_15_minutes": float(np.mean(np.abs(y - p) <= 15)),
        }

    bundle = BootstrapBundle(
        preprocessor=preprocessor,
        model=model,
        global_median=float(fit["delay_minutes"].median()),
        station_medians=fit.groupby("station_code")["delay_minutes"].median().to_dict(),
        interval_half_width_minutes=float(
            np.quantile(np.abs(y_validation - validation_prediction), 0.90, method="higher")
        ),
        transition_preprocessor=transition_preprocessor,
        transition_model=transition_model,
        transition_interval_half_width_minutes=transition_interval,
        transition_station_delta_medians={
            str(k): float(v) for k, v in transition_station_delta_medians.items()
        },
        transition_selective_rule={
            "previous_delay_max_minutes": float(selected_delay_limit),
            "previous_segment_change_max_minutes": float(selected_trend_limit),
        },
    )
    report = {
        "source_type": "public station-level delay table; no timestamps",
        "rows": {
            "total": len(frame),
            "fit": len(fit),
            "validation": len(validation),
            "test": len(test),
        },
        "train_groups": {
            "fit": len(fit_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "metrics": {
            "validation_lightgbm_l1": metrics(y_validation, validation_prediction),
            "global_median_baseline": metrics(y_test, static),
            "lightgbm_l1": metrics(y_test, prediction),
            "lightgbm_90_percent_interval": {
                "half_width_minutes": bundle.interval_half_width_minutes,
                "test_coverage": float(
                    np.mean(np.abs(y_test - prediction) <= bundle.interval_half_width_minutes)
                ),
            },
            "lightgbm_95_percent_interval": {
                "half_width_minutes": float(
                    np.quantile(validation_abs_residual, 0.95, method="higher")
                ),
                "test_coverage": float(
                    np.mean(
                        np.abs(y_test - prediction)
                        <= np.quantile(validation_abs_residual, 0.95, method="higher")
                    )
                ),
            },
            "transition_baseline_carry_forward": metrics(
                transition_y_test, transition_test["prev_delay"].to_numpy(dtype=float)
            ),
            "transition_lightgbm_l1": metrics(transition_y_test, transition_prediction),
            "transition_lightgbm_l1_selective": {
                "selection_rule": (
                    f"previous delay <= {selected_delay_limit:g} min and "
                    f"previous segment change <= {selected_trend_limit:g} min"
                ),
                "validation_coverage": best_rule[0],
                "validation_within_15_minutes": best_rule[1],
                "minimum_validation_accuracy": minimum_validation_accuracy,
                "coverage": float(np.mean(stable_mask)),
                "mae_minutes": float(
                    mean_absolute_error(transition_y_test[stable_mask], transition_prediction[stable_mask])
                )
                if stable_mask.any()
                else None,
                "within_15_minutes": float(
                    np.mean(
                        np.abs(
                            transition_y_test[stable_mask] - transition_prediction[stable_mask]
                        )
                        <= 15
                    )
                )
                if stable_mask.any()
                else None,
            },
            "transition_90_percent_interval": {
                "half_width_minutes": transition_interval,
                "test_coverage": float(
                    np.mean(np.abs(transition_y_test - transition_prediction) <= transition_interval)
                ),
            },
        },
        "target": "observed station delay minutes",
        "limitations": [
            "cannot be used as a point-in-time dynamic ETA evaluation because it has no timestamp",
            "the group split is by train ID, not by date",
            "use as cold-start prior only; promote live snapshot model after collection",
            "transition metrics use the previous station delay as an upstream online feature",
            "selective accuracy is reported with its coverage; it is not an all-trip accuracy claim",
        ],
    }
    return bundle, report


def save_bootstrap(
    bundle: BootstrapBundle, report: dict[str, Any], model_path: Path, report_path: Path
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
