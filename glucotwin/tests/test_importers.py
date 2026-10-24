"""Real-data path: OhioT1DM-format XML and generic CSV importers, then end-to-end training.

No real data is used: synthetic patients are written out in the OhioT1DM XML layout and in a
Dexcom-like CSV layout (mmol/L), on DIFFERENT calendar dates, to exercise the importers.
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_loader import load_from_sqlite, merge_tables, save_to_sqlite
from src.digital_twin import PatientTwin
from src.importers import dataset_report, import_csv, import_ohiot1dm
from src.pipeline import build_dataset, train_all
from src.synthetic_data import generate_cohort

OHIO_FMT = "%d-%m-%Y %H:%M:%S"


def _ev(ts, **attrs):
    a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f'<event ts="{ts.strftime(OHIO_FMT)}" {a}/>'


def write_ohio_xml(path, ohio_id, p: pd.DataFrame, shift_days: int, with_band: bool = True):
    p = p.copy()
    # real CGM timestamps are not aligned to 5-min boundaries
    p["timestamp"] = p["timestamp"] + pd.Timedelta(days=shift_days) + pd.Timedelta(minutes=2, seconds=13)
    g = [_ev(r.timestamp, value=int(r.glucose_mgdl)) for r in p.itertuples() if not np.isnan(r.glucose_mgdl)]
    hr = [_ev(r.timestamp, value=int(r.heart_rate_bpm)) for r in p.itertuples() if not np.isnan(r.heart_rate_bpm)]
    st = [_ev(r.timestamp, value=int(r.steps)) for r in p.itertuples() if not np.isnan(r.steps)]
    meals = [_ev(r.timestamp, type="Lunch", carbs=int(r.carbs_logged_g)) for r in p.itertuples() if r.carbs_logged_g > 0]
    bol = [f'<event ts_begin="{r.timestamp.strftime(OHIO_FMT)}" ts_end="{r.timestamp.strftime(OHIO_FMT)}" '
           f'type="normal" dose="{r.insulin_logged_u}" bwz_carb_input="0"/>'
           for r in p.itertuples() if r.insulin_logged_u > 0]
    asleep = p["is_asleep"].fillna(0).to_numpy() > 0
    starts = np.flatnonzero(asleep & ~np.r_[False, asleep[:-1]])
    ends = np.flatnonzero(asleep & ~np.r_[asleep[1:], False])
    sleep = [f'<event ts_begin="{p["timestamp"].iloc[e].strftime(OHIO_FMT)}" '   # Ohio sometimes lists begin > end
             f'ts_end="{p["timestamp"].iloc[s].strftime(OHIO_FMT)}" quality="2"/>' for s, e in zip(starts, ends)]
    band = (f"<basis_heart_rate>{''.join(hr)}</basis_heart_rate><basis_steps>{''.join(st)}</basis_steps>"
            if with_band else "")
    xml = (f'<patient id="{ohio_id}" weight="99" insulin_type="Humalog"><glucose_level>{"".join(g)}</glucose_level>'
           f"<finger_stick/><basal/><temp_basal/><bolus>{''.join(bol)}</bolus><meal>{''.join(meals)}</meal>"
           f"<sleep>{''.join(sleep)}</sleep><work/><stressors/><hypo_event/><illness/><exercise/>{band}</patient>")
    path.write_text(xml, encoding="utf-8")


@pytest.fixture(scope="module")
def ohio_folder(tmp_path_factory):
    cohort = generate_cohort(n_patients=4, n_days=10, seed=11)
    ts = cohort["timeseries"]
    folder = tmp_path_factory.mktemp("ohio")
    for i, (pid, p) in enumerate(ts.groupby("patient_id")):
        p = p.reset_index(drop=True)
        cut = int(len(p) * 0.8)   # mimic separate training / testing files
        oid = str(540 + i)
        write_ohio_xml(folder / f"{oid}-ws-training.xml", oid, p.iloc[:cut], shift_days=200 * i, with_band=i != 3)
        write_ohio_xml(folder / f"{oid}-ws-testing.xml", oid, p.iloc[cut:], shift_days=200 * i, with_band=i != 3)
    return folder


def test_ohio_import_structure(ohio_folder):
    tables, notes = import_ohiot1dm(ohio_folder)
    ts = tables["timeseries"]
    assert sorted(ts["patient_id"].unique()) == ["OHIO-540", "OHIO-541", "OHIO-542", "OHIO-543"]
    assert (ts["timestamp"].dt.minute % C.SAMPLE_MINUTES == 0).all() and (ts["timestamp"].dt.second == 0).all()
    assert not ts.duplicated(["patient_id", "timestamp"]).any()
    p0 = ts[ts["patient_id"] == "OHIO-540"]
    assert p0["glucose_mgdl"].notna().sum() > 2000          # training + testing files concatenated
    assert p0["is_asleep"].eq(1).any()                        # begin > end intervals handled
    assert ts.loc[ts["patient_id"] == "OHIO-543", "heart_rate_bpm"].isna().all()  # no band -> missing, not zero
    assert (tables["patients"]["diabetes_type"] == "T1D").all()
    assert tables["patients"]["age"].isna().all()             # not in the XML -> unknown, not invented
    assert any("unavailable" in n for n in notes)


def test_ohio_end_to_end_training_and_twin(ohio_folder, tmp_path):
    tables, _ = import_ohiot1dm(ohio_folder)
    db = tmp_path / "real.db"
    save_to_sqlite(tables, db)
    tables = load_from_sqlite(db)
    dataset = build_dataset(tables)
    art = train_all(tables, dataset=dataset, source="real", split_mode="auto", artifact_path=tmp_path / "a.joblib")
    assert art["split"].mode == "time_only"
    ev = art["evaluation"]
    fm = ev["forecast_metrics"]
    assert fm["MAE"].notna().all()
    assert ev["split"]["n_test_patients"] == 4
    feats = art["forecast"].features["full_twin"]
    assert "age" not in feats and "bmi" not in feats          # unavailable EHR fields dropped automatically
    assert "hr_now" in feats
    pid = "OHIO-543"                                         # patient without wearable data still works
    twin = PatientTwin(pid, dataset["grid"], dataset["static"], tables, art)
    twin.observe_until(art["split"].cutoff_for(pid) + pd.Timedelta(hours=3))
    s = twin.state()
    assert s.glucose_now is not None
    assert "hr_now" in s.missing_features or not s.can_forecast


def test_csv_import_aliases_mmol_and_ehr(tmp_path):
    cohort = generate_cohort(n_patients=2, n_days=2, seed=5)
    ts = cohort["timeseries"]
    raw = pd.DataFrame({
        "Subject": ts["patient_id"],
        "Timestamp (YYYY-MM-DDThh:mm:ss)": ts["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "Glucose Value (mmol/L)": (ts["glucose_mgdl"] / 18.0).round(1),
        "HR": ts["heart_rate_bpm"], "Carbs": ts["carbs_logged_g"],
    })
    raw.to_csv(tmp_path / "timeseries.csv", index=False)
    pd.DataFrame({"patient": ts["patient_id"].unique(), "Age": [30, 61], "Gender": ["F", "M"],
                  "HbA1c": [7.1, 8.4], "type": ["T1D", "T2D"]}).to_csv(tmp_path / "patients.csv", index=False)
    tables, notes = import_csv(tmp_path)
    t = tables["timeseries"]
    assert t["glucose_mgdl"].dropna().between(40, 420).all()   # converted from mmol/L
    assert abs(t["glucose_mgdl"].mean() - ts["glucose_mgdl"].mean()) < 2
    assert "heart_rate_bpm" in t and t["heart_rate_bpm"].notna().any()
    assert set(tables["labs"]["test"]) == {"HbA1c"}
    assert tables["patients"]["age"].notna().all()
    assert any("mmol" in n for n in notes)
    report = dataset_report(tables)
    assert "Patients: 2" in report


def test_csv_import_requires_glucose_and_supports_explicit_map(tmp_path):
    df = pd.DataFrame({"when": pd.date_range("2025-01-01", periods=10, freq="5min").astype(str),
                       "reading": np.linspace(100, 150, 10)})
    df.to_csv(tmp_path / "p1.csv", index=False)
    with pytest.raises(ValueError):
        import_csv(tmp_path / "p1.csv")
    tables, _ = import_csv(tmp_path / "p1.csv", {"timestamp": "when", "glucose_mgdl": "reading"})
    assert tables["timeseries"]["patient_id"].iloc[0] == "p1"


def test_append_merges_without_duplicates(ohio_folder):
    tables, _ = import_ohiot1dm(ohio_folder)
    merged = merge_tables(tables, tables)
    assert len(merged["timeseries"]) == len(tables["timeseries"])
    assert len(merged["patients"]) == len(tables["patients"])
