"""End-to-end orchestration: data -> features -> split -> models -> evaluation -> saved artifacts."""
from __future__ import annotations

import hashlib
import time

import joblib
import pandas as pd

from . import config as C
from .evaluation import evaluate
from .explainability import global_importance
from .features import add_targets, build_features, static_features
from .forecasting import GBM_BACKEND, make_split, train_forecasters
from .preprocessing import preprocess
from .risk_prediction import train_risk_models

ARTIFACT_VERSION = 2


def build_dataset(tables: dict) -> dict:
    grid, quality, issues = preprocess(tables["timeseries"], tables["patients"])
    static = static_features(tables)
    features = add_targets(build_features(grid, static))
    return dict(grid=grid, quality=quality, issues=issues, static=static, features=features)


def fingerprint(tables: dict) -> str:
    ts = tables["timeseries"]
    parts = [
        str(ARTIFACT_VERSION), str(ts.shape), f"{ts['glucose_mgdl'].sum():.1f}",
        ",".join(sorted(tables["patients"]["patient_id"])),
        str((C.HORIZON_MINUTES, C.TRAJECTORY_HORIZONS, C.HYPO_THRESHOLD, C.HYPER_THRESHOLD,
             C.SAMPLE_MINUTES, C.TEST_PATIENT_FRACTION, C.TIME_CUTOFF_FRACTION, C.SEED, C.SPLIT_MODE,
             C.TIME_ONLY_CALIB_FRACTION, C.TIME_ONLY_TEST_FRACTION)),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def train_all(tables: dict, dataset: dict | None = None, progress=None, source: str = "synthetic",
              split_mode: str = C.SPLIT_MODE, artifact_path=None) -> dict:
    """Train every model from scratch on `tables` (synthetic or real) and save the artifacts."""
    t0 = time.time()
    dataset = dataset or build_dataset(tables)
    df = dataset["features"]
    split = make_split(df, tables["patients"], mode=split_mode)
    fit_mask, calib_mask, test_mask = split.masks(df)
    for name, m in (("training", fit_mask), ("calibration", calib_mask), ("test", test_mask)):
        n = int((m & df["g_now"].notna() & df[f"target_glucose_{C.HORIZON_MINUTES}"].notna()).sum())
        if n < 200:
            raise ValueError(f"Only {n} usable {name} rows (split mode '{split.mode}'). The dataset is too small or "
                             "too gappy; add more days/patients or try --split time_only.")

    def sub(lo, hi):
        return (lambda frac, msg: progress(lo + (hi - lo) * frac, msg)) if progress else None

    forecast = train_forecasters(df, split, progress=sub(0.0, 0.7))
    risk = train_risk_models(df, split, forecast.features, progress=sub(0.7, 0.95))
    if progress:
        progress(0.96, "Evaluating on test rows")
    evaluation = evaluate(df, split, forecast, risk)

    _, _, test_mask = split.masks(df)
    test = df.loc[test_mask & df["g_now"].notna()]
    sample = test.sample(n=min(2000, len(test)), random_state=C.SEED)
    importance = {}
    for config in forecast.features:
        fam = forecast.selected_family[config]
        model = forecast.models[(config, fam, C.HORIZON_MINUTES)]
        importance[config] = global_importance(model, sample[forecast.features[config]], forecast.medians[config])

    artifacts = dict(
        forecast=forecast, risk=risk, evaluation=evaluation, split=split, importance=importance,
        meta=dict(fingerprint=fingerprint(tables) + split_mode, gbm_backend=GBM_BACKEND,
                  trained_at=pd.Timestamp.now().isoformat(), train_seconds=round(time.time() - t0, 1),
                  n_rows=len(df), n_patients=int(df["patient_id"].nunique()), version=ARTIFACT_VERSION,
                  source=source, split_mode=split.mode),
    )
    path = artifact_path or C.artifact_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)
    if progress:
        progress(1.0, "Done")
    return artifacts


def load_or_train(tables: dict, dataset: dict | None = None, retrain: bool = False, progress=None,
                  source: str = "synthetic", split_mode: str = C.SPLIT_MODE) -> dict:
    """Load saved models if they were trained on exactly this data + config; otherwise retrain.
    Adding or changing data therefore triggers retraining automatically."""
    path = C.artifact_path(source)
    if path.exists() and not retrain:
        try:
            artifacts = joblib.load(path)
            if artifacts.get("meta", {}).get("fingerprint") == fingerprint(tables) + split_mode:
                return artifacts
        except Exception:
            pass  # stale/incompatible artifact -> retrain
    return train_all(tables, dataset=dataset, progress=progress, source=source, split_mode=split_mode)
