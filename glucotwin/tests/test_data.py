"""Data generation and preprocessing tests."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_loader import load_from_sqlite, save_to_sqlite, to_fhir_bundle
from src.preprocessing import DataValidationError, fill_short_gaps, preprocess, validate_timeseries


def test_patient_ids_are_coherent_across_tables(small_tables):
    ids = set(small_tables["patients"]["patient_id"])
    assert len(ids) == 8
    for name in ("conditions", "medications", "labs", "timeseries", "simulation_truth"):
        assert set(small_tables[name]["patient_id"]) <= ids, name
    assert set(small_tables["timeseries"]["patient_id"]) == ids
    assert (small_tables["patients"]["provenance"] == "simulated").all()
    assert (small_tables["timeseries"]["provenance"] == "simulated").all()


def test_timeseries_sampling_and_ranges(small_tables):
    ts = small_tables["timeseries"]
    for _, p in ts.groupby("patient_id"):
        diffs = p["timestamp"].diff().dropna().unique()
        assert list(diffs) == [pd.Timedelta(minutes=C.SAMPLE_MINUTES)]
        assert len(p) == 4 * 24 * 60 // C.SAMPLE_MINUTES
    g = ts["glucose_mgdl"].dropna()
    assert g.between(40, 400).all()
    assert ts["glucose_mgdl"].isna().any(), "missingness should be injected"
    assert (ts["carbs_logged_g"] >= 0).all() and (ts["insulin_logged_u"] >= 0).all()


def test_generation_is_deterministic():
    from src.synthetic_data import generate_cohort
    a = generate_cohort(n_patients=2, n_days=1, seed=3)["timeseries"]
    b = generate_cohort(n_patients=2, n_days=1, seed=3)["timeseries"]
    pd.testing.assert_frame_equal(a, b)


def test_sqlite_roundtrip(tmp_path, small_tables):
    db = tmp_path / "t.db"
    save_to_sqlite(small_tables, db)
    loaded = load_from_sqlite(db)
    assert loaded["timeseries"].shape == small_tables["timeseries"].shape
    assert pd.api.types.is_datetime64_any_dtype(loaded["timeseries"]["timestamp"])


def test_fhir_bundle_is_tagged_synthetic(small_tables):
    pid = small_tables["patients"]["patient_id"].iloc[0]
    bundle = to_fhir_bundle(pid, small_tables)
    assert bundle["resourceType"] == "Bundle"
    assert all(e["resource"]["meta"]["tag"][0]["code"] == "synthetic" for e in bundle["entry"])
    assert any(e["resource"]["resourceType"] == "Observation" for e in bundle["entry"])


def test_fill_short_gaps_only_fills_short_interior_gaps():
    s = pd.Series([100, np.nan, np.nan, 130] + [140] + [np.nan] * 8 + [200] + [np.nan])
    filled, imputed = fill_short_gaps(s, max_gap_steps=6)
    assert filled.iloc[1] == pytest.approx(110) and filled.iloc[2] == pytest.approx(120)
    assert imputed.iloc[1] and imputed.iloc[2]
    assert filled.iloc[5:13].isna().all(), "8-step gap exceeds the limit and must stay missing"
    assert np.isnan(filled.iloc[-1]), "trailing gap is not interior"
    assert imputed.sum() == 2


def test_validation_fixes_and_reports_problems(small_tables):
    ts = small_tables["timeseries"].head(50).copy()
    ts.loc[0, "glucose_mgdl"] = 900
    ts.loc[1, "heart_rate_bpm"] = -5
    ts = pd.concat([ts, ts.iloc[[2]]])  # duplicate row
    clean, issues = validate_timeseries(ts, small_tables["patients"])
    assert np.isnan(clean.loc[0, "glucose_mgdl"])
    assert any("glucose_mgdl" in i for i in issues)
    assert any("duplicate" in i for i in issues)
    with pytest.raises(DataValidationError):
        validate_timeseries(ts.drop(columns=["glucose_mgdl"]))


def test_preprocess_regular_grid_and_flags(small_dataset):
    grid = small_dataset["grid"]
    for _, p in grid.groupby("patient_id"):
        assert (p["timestamp"].diff().dropna() == pd.Timedelta(minutes=C.SAMPLE_MINUTES)).all()
    assert not (grid["glucose_imputed"] & grid["glucose_observed"]).any()
    assert (grid.loc[grid["glucose_imputed"], "glucose"].notna()).all()
    assert (grid["carbs_logged_g"] >= 0).all()
    q = small_dataset["quality"]
    assert {"pct_glucose_imputed", "longest_glucose_gap_min"} <= set(q.columns)


def test_preprocess_handles_missing_optional_columns(small_tables):
    ts = small_tables["timeseries"].drop(columns=["heart_rate_bpm", "steps", "is_asleep"])
    grid, _, _ = preprocess(ts, small_tables["patients"])
    assert "heart_rate_bpm" not in grid.columns
    assert grid["glucose"].notna().mean() > 0.9
