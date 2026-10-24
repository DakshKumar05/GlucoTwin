"""Two-hour threshold-crossing risk models (hypo: < HYPO_THRESHOLD, hyper: > HYPER_THRESHOLD).

Label (see features.add_targets): 1 if glucose crosses the threshold at any time in (t, t + 2 h].
Only rows that are not already beyond the threshold are used ("new crossing" task).

Probabilities are calibrated with isotonic regression on the calibration split. If there are too
few positive examples, the model is marked unavailable and NO probability is shown.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

from . import config as C
from .features import MODEL_CONFIGS

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    HAS_XGBOOST = False

EVENTS = {
    "hypo": dict(label="label_hypo", eligible="eligible_hypo",
                 name=f"Glucose below {C.HYPO_THRESHOLD:.0f} mg/dL within the next 2 h"),
    "hyper": dict(label="label_hyper", eligible="eligible_hyper",
                  name=f"Glucose above {C.HYPER_THRESHOLD:.0f} mg/dL within the next 2 h"),
}

RISK_DEFINITIONS = {
    "Low": f"Calibrated probability < {C.RISK_BANDS['low']:.0%}",
    "Elevated": f"Calibrated probability {C.RISK_BANDS['low']:.0%} to < {C.RISK_BANDS['high']:.0%}",
    "High": f"Calibrated probability >= {C.RISK_BANDS['high']:.0%}",
}


def risk_category(p: float | None) -> str:
    if p is None or np.isnan(p):
        return "Unavailable"
    if p < C.RISK_BANDS["low"]:
        return "Low"
    if p < C.RISK_BANDS["high"]:
        return "Elevated"
    return "High"


def make_classifier(seed: int = C.SEED):
    if HAS_XGBOOST:
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                             eval_metric="logloss", n_jobs=-1, random_state=seed)
    return HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=seed)


@dataclass
class RiskModel:
    event: str
    config: str
    features: list
    status: str = "unavailable"          # "ok" | "uncalibrated" | "unavailable"
    message: str = ""
    model: object = None
    calibrator: object = None
    threshold: float = 0.5
    stats: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status in ("ok", "uncalibrated") and self.model is not None

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.available:
            return np.full(len(X), np.nan)
        p = self.raw_proba(X)
        return np.clip(self.calibrator.predict(p), 0, 1) if self.calibrator is not None else p


def train_risk_models(df: pd.DataFrame, split, features_by_config: dict, progress=None) -> dict:
    fit_mask, calib_mask, _ = split.masks(df)
    models = {}
    total = len(EVENTS) * len(MODEL_CONFIGS)
    done = 0
    for event, spec in EVENTS.items():
        usable = df[spec["eligible"]] & df[spec["label"]].notna()
        fit = df.loc[fit_mask & usable]
        cal = df.loc[calib_mask & usable]
        for config in MODEL_CONFIGS:
            feats = features_by_config[config]
            rm = RiskModel(event=event, config=config, features=feats)
            y_fit, y_cal = fit[spec["label"]].astype(int), cal[spec["label"]].astype(int)
            rm.stats = dict(n_fit=len(fit), pos_fit=int(y_fit.sum()), n_calib=len(cal), pos_calib=int(y_cal.sum()))
            done += 1
            if progress:
                progress(done / total, f"Risk model {done}/{total}: {event}, {config}")
            if rm.stats["pos_fit"] < C.MIN_TRAIN_EVENTS or rm.stats["pos_fit"] == len(y_fit):
                rm.message = (f"Only {rm.stats['pos_fit']} positive training windows "
                              f"(minimum {C.MIN_TRAIN_EVENTS}); risk probability not estimated.")
                models[(event, config)] = rm
                continue
            rm.model = make_classifier()
            rm.model.fit(fit[feats], y_fit)
            raw_cal = rm.raw_proba(cal)
            if rm.stats["pos_calib"] >= C.MIN_CALIB_EVENTS:
                rm.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, y_cal)
                rm.status = "ok"
                p_cal = rm.calibrator.predict(raw_cal)
                prec, rec, thr = precision_recall_curve(y_cal, p_cal)
                f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
                rm.threshold = float(thr[int(np.argmax(f1))]) if len(thr) else 0.5
                rm.message = "Isotonic-calibrated; alert threshold maximises F1 on the calibration split."
            else:
                rm.status = "uncalibrated"
                rm.message = (f"Only {rm.stats['pos_calib']} positive calibration windows; probabilities are "
                              "uncalibrated model scores and should be read as a ranking only.")
            models[(event, config)] = rm
    return models
