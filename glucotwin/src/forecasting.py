"""Glucose forecasting: data split, model families, split-conformal prediction intervals.

Models predict the CHANGE in glucose over the horizon (target_delta_h); the forecast is
g_now + predicted change. Two families are compared for every feature configuration:
  * ridge - a simple, reliable, linear model (the "simple first" baseline family)
  * gbm   - gradient-boosted trees (XGBoost if installed, otherwise sklearn HistGradientBoosting)
The family used by the dashboard is chosen on the CALIBRATION set, never on the test set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import MODEL_CONFIGS, available_features

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:  # pragma: no cover - depends on environment
    HAS_XGBOOST = False

GBM_BACKEND = "XGBoost" if HAS_XGBOOST else "sklearn HistGradientBoosting"
FAMILIES = ("ridge", "gbm")
FAMILY_LABELS = {"ridge": "Ridge regression", "gbm": f"Gradient boosting ({GBM_BACKEND})"}


# ----------------------------------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------------------------------
@dataclass
class SplitInfo:
    """Leakage-safe split with PER-PATIENT chronological cut-offs (real patients are recorded on different dates).

    patient_time (larger cohorts):
      fit   : training patients, rows whose 2 h target lies before their cut-off (embargo)
      calib : training patients, rows after their cut-off (model selection, intervals, calibration)
      test  : held-out patients, rows after their cut-off (reported metrics only)
    time_only (small cohorts, e.g. OhioT1DM): every patient's own timeline is split
      fit | embargo | calib | embargo | test   - this measures personalised performance on future data,
      not generalisation to unseen people.
    """
    mode: str
    train_patients: list
    test_patients: list
    calib_cutoffs: dict
    test_cutoffs: dict
    embargo_minutes: int = C.HORIZON_MINUTES

    def cutoff_for(self, patient_id: str):
        return self.test_cutoffs.get(patient_id)

    def is_test_patient(self, patient_id: str) -> bool:
        return patient_id in self.test_patients

    def masks(self, df: pd.DataFrame):
        emb = pd.Timedelta(minutes=self.embargo_minutes)
        ts, pid = df["timestamp"], df["patient_id"]
        cal_c = pid.map(self.calib_cutoffs)
        test_c = pid.map(self.test_cutoffs)
        if self.mode == "time_only":
            fit = ts + emb < cal_c
            calib = (ts >= cal_c) & (ts + emb < test_c)
            test = ts >= test_c
        else:
            train, held = pid.isin(self.train_patients), pid.isin(self.test_patients)
            fit = train & (ts + emb < cal_c)
            calib = train & (ts >= cal_c)
            test = held & (ts >= test_c)
        return fit.fillna(False).astype(bool), calib.fillna(False).astype(bool), test.fillna(False).astype(bool)

    def describe(self) -> str:
        if self.mode == "time_only":
            return (f"Within-patient chronological split for all {len(self.train_patients)} patients: first "
                    f"{C.TIME_ONLY_CALIB_FRACTION:.0%} of each timeline to fit, next "
                    f"{C.TIME_ONLY_TEST_FRACTION - C.TIME_ONLY_CALIB_FRACTION:.0%} to calibrate, last "
                    f"{1 - C.TIME_ONLY_TEST_FRACTION:.0%} to test ({self.embargo_minutes}-min embargo at each boundary). "
                    "Measures personalised performance on future data, not generalisation to new patients.")
        return (f"{len(self.train_patients)} training / {len(self.test_patients)} held-out patients (stratified by "
                f"diabetes type), per-patient time cut-off at {C.TIME_CUTOFF_FRACTION:.0%} of each timeline with a "
                f"{self.embargo_minutes}-min embargo. Test patients are never used for training, selection or calibration.")


def make_split(df: pd.DataFrame, patients: pd.DataFrame, seed: int = C.SEED, mode: str = C.SPLIT_MODE,
               test_fraction: float = C.TEST_PATIENT_FRACTION,
               cutoff_fraction: float = C.TIME_CUTOFF_FRACTION) -> SplitInfo:
    ids = sorted(df["patient_id"].unique())
    if mode == "auto":
        mode = "time_only" if len(ids) < C.MIN_PATIENTS_FOR_PATIENT_SPLIT else "patient_time"
    bounds = df.groupby("patient_id")["timestamp"].agg(["min", "max"])
    freq = f"{C.SAMPLE_MINUTES}min"

    def at(frac):
        return {pid: (r["min"] + (r["max"] - r["min"]) * frac).floor(freq) for pid, r in bounds.iterrows()}

    if mode == "time_only":
        return SplitInfo(mode=mode, train_patients=ids, test_patients=ids,
                         calib_cutoffs=at(C.TIME_ONLY_CALIB_FRACTION), test_cutoffs=at(C.TIME_ONLY_TEST_FRACTION))
    if mode != "patient_time":
        raise ValueError(f"Unknown split mode {mode}")
    rng = np.random.default_rng(seed)
    pats = patients[patients["patient_id"].isin(ids)].copy()
    pats["diabetes_type"] = pats["diabetes_type"].fillna("unknown")
    test = []
    for _, grp in pats.groupby("diabetes_type", sort=True):   # stratify by diabetes type
        g_ids = sorted(grp["patient_id"])
        n_test = max(1, int(round(test_fraction * len(g_ids)))) if len(g_ids) > 1 else 0
        test += list(rng.choice(g_ids, size=n_test, replace=False))
    cut = at(cutoff_fraction)
    return SplitInfo(mode=mode, train_patients=sorted(set(ids) - set(test)), test_patients=sorted(test),
                     calib_cutoffs=cut, test_cutoffs=cut)


# ----------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------
def make_regressor(family: str, seed: int = C.SEED):
    if family == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=1.0))
    if family == "gbm":
        if HAS_XGBOOST:
            return XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                                n_jobs=-1, random_state=seed)
        return HistGradientBoostingRegressor(max_iter=400, max_depth=5, learning_rate=0.05, random_state=seed)
    raise ValueError(f"Unknown model family: {family}")


def conformal_quantile(abs_residuals: np.ndarray, level: float) -> float:
    """Split-conformal quantile with the finite-sample correction ceil((n+1)level)/n."""
    r = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(r)
    if n == 0:
        return float("nan")
    k = min(n - 1, int(np.ceil((n + 1) * level)) - 1)
    return float(r[k])


@dataclass
class ForecastBundle:
    horizons: tuple
    features: dict = field(default_factory=dict)        # config -> list[str]
    models: dict = field(default_factory=dict)          # (config, family, h) -> fitted model
    interval_q: dict = field(default_factory=dict)      # (config, family, h) -> conformal half-width
    calib_mae: dict = field(default_factory=dict)       # (config, family, h) -> MAE on calibration set
    selected_family: dict = field(default_factory=dict)  # config -> family
    medians: dict = field(default_factory=dict)         # config -> training medians (for explanations)
    n_fit: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame, config: str, h: int, family: str | None = None) -> np.ndarray:
        family = family or self.selected_family[config]
        model = self.models[(config, family, h)]
        delta = model.predict(X[self.features[config]])
        return np.clip(X["g_now"].to_numpy(dtype=float) + delta, 40, 400)

    def interval(self, config: str, h: int, family: str | None = None) -> float:
        return self.interval_q[(config, family or self.selected_family[config], h)]


def train_forecasters(df: pd.DataFrame, split: SplitInfo, horizons=C.TRAJECTORY_HORIZONS,
                      progress=None) -> ForecastBundle:
    fit_mask, calib_mask, _ = split.masks(df)
    has_now = df["g_now"].notna()
    bundle = ForecastBundle(horizons=tuple(horizons))
    total = len(MODEL_CONFIGS) * len(horizons) * len(FAMILIES)
    done = 0
    for config in MODEL_CONFIGS:
        feats = available_features(df.loc[fit_mask], config)
        bundle.features[config] = feats
        bundle.medians[config] = df.loc[fit_mask, feats].median()
        for h in horizons:
            target = f"target_delta_{h}"
            fit = df.loc[fit_mask & has_now & df[target].notna()].iloc[::C.TRAIN_STRIDE]
            cal = df.loc[calib_mask & has_now & df[target].notna()]
            for family in FAMILIES:
                model = make_regressor(family)
                model.fit(fit[feats], fit[target])
                resid = np.abs(cal[target].to_numpy() - model.predict(cal[feats]))
                key = (config, family, h)
                bundle.models[key] = model
                bundle.interval_q[key] = conformal_quantile(resid, C.PREDICTION_INTERVAL)
                bundle.calib_mae[key] = float(resid.mean())
                bundle.n_fit[key] = len(fit)
                done += 1
                if progress:
                    progress(done / total, f"Forecast model {done}/{total}: {config}, {family}, {h} min")
        h_sel = max(horizons)
        bundle.selected_family[config] = min(FAMILIES, key=lambda fam: bundle.calib_mae[(config, fam, h_sel)])
    return bundle
