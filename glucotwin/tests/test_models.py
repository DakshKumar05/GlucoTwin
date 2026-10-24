"""Feature, horizon, split, missing-data and digital-twin tests."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.digital_twin import PatientTwin
from src.evaluation import evaluate
from src.features import EHR_FEATURES, available_features, build_features, dynamic_features
from src.forecasting import conformal_quantile, make_split, train_forecasters
from src.risk_prediction import train_risk_models


def test_target_is_observed_glucose_exactly_two_hours_ahead(small_dataset):
    df = small_dataset["features"]
    h = C.HORIZON_MINUTES
    n = C.steps(h)
    assert n == 24  # 120 min at 5-min sampling
    p = df[df["patient_id"] == df["patient_id"].iloc[0]].reset_index(drop=True)
    for i in (300, 500, 700):
        future = p.loc[i + n]
        assert future["timestamp"] - p.loc[i, "timestamp"] == pd.Timedelta(minutes=h)
        expected = future["glucose"] if future["glucose_observed"] else np.nan
        assert (np.isnan(expected) and np.isnan(p.loc[i, f"target_glucose_{h}"])) or \
            p.loc[i, f"target_glucose_{h}"] == pytest.approx(expected)


def test_targets_never_cross_patients(small_dataset):
    df = small_dataset["features"]
    last_rows = df.groupby("patient_id").tail(C.steps(C.HORIZON_MINUTES))
    assert last_rows[f"target_glucose_{C.HORIZON_MINUTES}"].isna().all()


def test_risk_label_matches_future_window(small_dataset):
    df = small_dataset["features"]
    p = df[df["patient_id"] == df["patient_id"].iloc[0]].reset_index(drop=True)
    n = C.steps(C.HORIZON_MINUTES)
    for i in range(300, 900, 37):
        window = p.loc[i + 1:i + n, "glucose"]
        if window.notna().mean() >= C.MIN_FUTURE_COVERAGE:
            assert p.loc[i, "label_hypo"] == float(window.min() < C.HYPO_THRESHOLD)
            assert p.loc[i, "label_hyper"] == float(window.max() > C.HYPER_THRESHOLD)


def test_features_are_causal(small_dataset):
    """Features at time t must be identical whether computed on the full series or only on data <= t."""
    grid = small_dataset["grid"]
    p = grid[grid["patient_id"] == grid["patient_id"].iloc[0]].reset_index(drop=True)
    full = dynamic_features(p)
    t = 800
    sliced = dynamic_features(p.iloc[t - C.TWIN_HISTORY_STEPS + 1:t + 1].reset_index(drop=True)).iloc[-1]
    np.testing.assert_allclose(full.iloc[t].to_numpy(float), sliced.to_numpy(float), rtol=1e-6, atol=0.05,
                               equal_nan=True)


def test_patient_split_has_no_patient_leakage_and_respects_embargo(small_dataset, small_tables):
    df = small_dataset["features"]
    split = make_split(df, small_tables["patients"], seed=1, mode="patient_time")
    assert not set(split.train_patients) & set(split.test_patients)
    assert set(split.train_patients) | set(split.test_patients) == set(small_tables["patients"]["patient_id"])
    fit, cal, test = split.masks(df)
    assert not (fit & test).any() and not (fit & cal).any()
    cut = df["patient_id"].map(split.test_cutoffs)
    assert (df.loc[fit, "timestamp"] + pd.Timedelta(minutes=C.HORIZON_MINUTES) < cut[fit]).all()
    assert set(df.loc[test, "patient_id"]) <= set(split.test_patients)
    types = small_tables["patients"].set_index("patient_id").loc[split.test_patients, "diabetes_type"]
    assert types.nunique() == small_tables["patients"]["diabetes_type"].nunique()


def test_time_only_split_is_chronological_per_patient(small_dataset, small_tables):
    df = small_dataset["features"].copy()
    # Shift one patient to a different calendar period, as happens in real datasets.
    other = df["patient_id"] == df["patient_id"].iloc[0]
    df.loc[other, "timestamp"] += pd.Timedelta(days=400)
    split = make_split(df, small_tables["patients"], mode="auto")
    assert split.mode == "time_only"   # 8 patients < MIN_PATIENTS_FOR_PATIENT_SPLIT
    fit, cal, test = split.masks(df)
    assert not (fit & cal).any() and not (cal & test).any() and not (fit & test).any()
    emb = pd.Timedelta(minutes=C.HORIZON_MINUTES)
    for pid, p in df.groupby("patient_id"):
        assert test[p.index].any() and fit[p.index].any(), pid
        assert (p.loc[fit[p.index], "timestamp"] + emb < split.calib_cutoffs[pid]).all()
        assert (p.loc[cal[p.index], "timestamp"] + emb < split.test_cutoffs[pid]).all()
        assert (p.loc[test[p.index], "timestamp"] >= split.test_cutoffs[pid]).all()


def test_available_features_drops_absent_columns(small_dataset):
    df = small_dataset["features"].drop(columns=["hr_now", "steps_30m"])
    feats = available_features(df, "full_twin")
    assert "hr_now" not in feats and "steps_30m" not in feats
    assert set(EHR_FEATURES) <= set(feats)


def test_conformal_quantile():
    r = np.arange(1, 101, dtype=float)
    assert conformal_quantile(r, 0.8) == 81.0
    assert np.isnan(conformal_quantile(np.array([]), 0.8))


@pytest.fixture(scope="module")
def trained(small_dataset, small_tables):
    df = small_dataset["features"]
    split = make_split(df, small_tables["patients"], seed=1)
    fc = train_forecasters(df, split, horizons=(30, 120))
    risk = train_risk_models(df, split, fc.features)
    return df, split, fc, risk


def test_models_predict_with_missing_wearable_data(trained):
    df, _, fc, _ = trained
    X = df[df["g_now"].notna()].tail(20).copy()
    for col in ("hr_now", "hr_mean_30m", "steps_30m", "steps_2h", "sleep_24h_h"):
        X[col] = np.nan
    for family in ("ridge", "gbm"):
        pred = fc.predict(X, "full_twin", 120, family)
        assert np.isfinite(pred).all() and ((pred >= 40) & (pred <= 400)).all()


def test_evaluation_metrics_are_computed(trained):
    df, split, fc, risk = trained
    ev = evaluate(df, split, fc, risk)
    fm = ev["forecast_metrics"]
    assert {"persistence", "glucose_only", "glucose_wearable", "full_twin"} <= set(fm["config"])
    assert (fm["MAE"] > 0).all() and (fm["RMSE"] >= fm["MAE"]).all()
    assert len(ev["ablation"]) == 4
    assert (ev["ablation"]["ci_low"] <= ev["ablation"]["ci_high"]).all()


def test_risk_model_refuses_with_too_few_events(trained, small_dataset):
    df, split, fc, _ = trained
    df2 = df.copy()
    df2["label_hypo"] = np.where(df2["label_hypo"].notna(), 0.0, np.nan)
    risk = train_risk_models(df2, split, fc.features)
    rm = risk[("hypo", "full_twin")]
    assert not rm.available
    assert np.isnan(rm.predict_proba(df2.head(3))).all()


def test_twin_updates_and_withholds_stale_forecasts(trained, small_dataset, small_tables):
    df, split, fc, risk = trained
    artifacts = dict(forecast=fc, risk=risk)
    pid = split.test_patients[0]
    twin = PatientTwin(pid, small_dataset["grid"], small_dataset["static"], small_tables, artifacts)
    t0 = twin.first_time + pd.Timedelta(hours=30)
    twin.observe_until(t0)
    s1 = twin.state()
    twin.observe_until(t0 + pd.Timedelta(minutes=30))
    s2 = twin.state()
    assert s2.cursor > s1.cursor
    assert twin.observed()["timestamp"].max() == s2.cursor  # twin never sees beyond the clock
    if s1.can_forecast and s2.can_forecast:
        assert s1.forecast["predicted"].iloc[-1] != s2.forecast["predicted"].iloc[-1]

    # Simulate a sensor dropout: blank the last 25 minutes of real CGM -> forecast must be withheld.
    grid = small_dataset["grid"].copy()
    m = (grid["patient_id"] == pid) & (grid["timestamp"] > t0 - pd.Timedelta(minutes=25)) & (grid["timestamp"] <= t0)
    grid.loc[m, ["glucose", "glucose_raw"]] = np.nan
    grid.loc[m, "glucose_observed"] = False
    grid.loc[m, "glucose_imputed"] = False
    twin2 = PatientTwin(pid, grid, small_dataset["static"], small_tables, artifacts)
    twin2.observe_until(t0)
    s3 = twin2.state()
    assert not s3.can_forecast and s3.forecast is None
