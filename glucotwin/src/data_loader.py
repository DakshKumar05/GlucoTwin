"""Data storage and exchange: SQLite persistence, CSV export, FHIR-style export.

SQLite is used through the standard-library `sqlite3` driver and plain SQL-compatible tables, so the
same schema can be moved to PostgreSQL by swapping `get_connection()` for a SQLAlchemy engine
(e.g. `create_engine("postgresql+psycopg://...")`) - `pandas.read_sql` / `to_sql` accept both.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .synthetic_data import generate_cohort

TABLES = ("patients", "conditions", "medications", "labs", "timeseries", "simulation_truth", "simulation_params")
DATETIME_COLUMNS = {"timeseries": ["timestamp"], "simulation_truth": ["timestamp"]}

# Minimum columns every table must have. Real datasets often lack whole tables (e.g. no labs) or
# fields (e.g. no BMI); missing ones are created empty/NaN so downstream code never crashes and the
# corresponding features are simply dropped as "not available".
SCHEMA = {
    "patients": ["patient_id", "display_name", "age", "sex", "diabetes_type", "years_since_diagnosis", "bmi",
                 "insulin_delivery", "prior_hypo_events_12m", "provenance"],
    "conditions": ["patient_id", "code_system", "code", "display", "onset_date"],
    "medications": ["patient_id", "medication", "drug_class", "start_date", "status"],
    "labs": ["patient_id", "date", "test", "loinc", "value", "unit"],
    "timeseries": ["patient_id", "timestamp", "glucose_mgdl", "heart_rate_bpm", "steps", "is_asleep",
                   "carbs_logged_g", "insulin_logged_u", "provenance"],
    "simulation_truth": ["patient_id", "timestamp"],
    "simulation_params": ["patient_id"],
}
TEXT_COLUMNS = {"patient_id", "display_name", "sex", "diabetes_type", "insulin_delivery", "provenance", "code_system",
                "code", "display", "onset_date", "medication", "drug_class", "start_date", "status", "date", "test",
                "loinc", "unit"}


def normalize_tables(tables: dict, provenance: str | None = None) -> dict:
    """Ensure every table/column in SCHEMA exists; fill sensible defaults for missing patient fields."""
    out = {}
    for name, cols in SCHEMA.items():
        df = tables.get(name)
        df = pd.DataFrame(columns=cols) if df is None else df.copy()
        for col in cols:
            if col not in df.columns:
                df[col] = "" if (col in TEXT_COLUMNS and name != "patients") else np.nan
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype(str)
        out[name] = df
    ts, pats = out["timeseries"], out["patients"]
    known = set(pats["patient_id"])
    extra = sorted(set(ts["patient_id"]) - known)
    if extra:  # time-series patients without an EHR row get an empty profile
        out["patients"] = pats = pd.concat([pats, pd.DataFrame({"patient_id": extra})], ignore_index=True)
    pats["display_name"] = pats["display_name"].fillna("Patient " + pats["patient_id"])
    pats["diabetes_type"] = pats["diabetes_type"].fillna("unknown")
    pats["insulin_delivery"] = pats["insulin_delivery"].fillna("unknown")
    if provenance:
        pats["provenance"] = pats["provenance"].fillna(provenance)
        ts["provenance"] = ts["provenance"].fillna(provenance)
    out["patients"] = pats.reset_index(drop=True)
    return out


def is_synthetic(tables: dict) -> bool:
    prov = tables["patients"]["provenance"].dropna()
    return len(prov) > 0 and (prov == "simulated").all()


def get_connection(db_path: Path = C.DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def save_to_sqlite(tables: dict, db_path: Path = C.DB_PATH) -> None:
    with get_connection(db_path) as conn:
        for name, df in tables.items():
            out = df.copy()
            for col in DATETIME_COLUMNS.get(name, []):
                out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
            out.to_sql(name, conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_ts_pid_time ON timeseries(patient_id, timestamp)")


def load_from_sqlite(db_path: Path = C.DB_PATH) -> dict:
    tables = {}
    with get_connection(db_path) as conn:
        present = set(pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"])
        for name in TABLES:
            if name not in present:
                continue
            df = pd.read_sql(f"SELECT * FROM {name}", conn)
            for col in DATETIME_COLUMNS.get(name, []):
                df[col] = pd.to_datetime(df[col])
            tables[name] = df
    return normalize_tables(tables)


def ensure_data(regenerate: bool = False, n_patients: int = C.N_PATIENTS, n_days: int = C.N_DAYS,
                seed: int = C.SEED, db_path: Path | None = None, source: str = "synthetic") -> dict:
    """Load a data source from SQLite.

    synthetic: generated on first use (offline fallback).
    real:      must have been created with `python import_data.py ...` first.
    """
    db_path = db_path or C.db_path(source)
    if source == "real":
        if not db_path.exists():
            raise FileNotFoundError(f"No imported real dataset at {db_path}. Run: python import_data.py --format "
                                    "ohio|csv --input <folder>")
        return load_from_sqlite(db_path)
    if regenerate or not db_path.exists():
        tables = generate_cohort(n_patients=n_patients, n_days=n_days, seed=seed)
        save_to_sqlite(tables, db_path)
    return load_from_sqlite(db_path)


def available_sources() -> list[str]:
    return ["synthetic"] + (["real"] if C.db_path("real").exists() else [])


def merge_tables(old: dict, new: dict) -> dict:
    """Append newly imported data to an existing dataset; newer rows win on duplicate keys."""
    keys = {"patients": ["patient_id"], "timeseries": ["patient_id", "timestamp"],
            "labs": ["patient_id", "date", "test"], "conditions": ["patient_id", "display"],
            "medications": ["patient_id", "medication"]}
    out = {}
    for name in SCHEMA:
        both = pd.concat([old.get(name, pd.DataFrame()), new.get(name, pd.DataFrame())], ignore_index=True)
        if name in keys and len(both):
            both = both.drop_duplicates(subset=keys[name], keep="last")
        out[name] = both
    out["timeseries"] = out["timeseries"].sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    return normalize_tables(out)


def export_csv(tables: dict, out_dir: Path = C.CSV_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def to_fhir_bundle(patient_id: str, tables: dict) -> dict:
    """Export one synthetic patient's static record as a FHIR R4-style collection Bundle.

    Only codes we are confident about are included (SNOMED-CT diabetes/hypertension codes,
    LOINC 4548-4 for HbA1c); other items are exported as text. Every resource is tagged as synthetic.
    This is an interoperability illustration, not a validated FHIR implementation.
    """
    synthetic = is_synthetic(tables)
    tag = [{"system": "urn:glucotwin:provenance", "code": "synthetic" if synthetic else "imported",
            "display": "Synthetic data" if synthetic else "Imported dataset"}]
    pat = tables["patients"].set_index("patient_id").loc[patient_id]
    entries = [{"resource": {
        "resourceType": "Patient", "id": patient_id, "meta": {"tag": tag},
    }}]
    if isinstance(pat["sex"], str) and pat["sex"]:
        entries[0]["resource"]["gender"] = pat["sex"]
    if pd.notna(pat["age"]):
        entries[0]["resource"]["extension"] = [{"url": "urn:glucotwin:age-years", "valueInteger": int(pat["age"])}]
    for i, c in enumerate(tables["conditions"].query("patient_id == @patient_id").itertuples()):
        code = {"text": c.display}
        if c.code_system == "SNOMED-CT" and c.code:
            code["coding"] = [{"system": "http://snomed.info/sct", "code": c.code, "display": c.display}]
        entries.append({"resource": {"resourceType": "Condition", "id": f"{patient_id}-cond-{i}", "meta": {"tag": tag},
                                     "subject": {"reference": f"Patient/{patient_id}"}, "code": code,
                                     "onsetDateTime": c.onset_date}})
    for i, m in enumerate(tables["medications"].query("patient_id == @patient_id").itertuples()):
        entries.append({"resource": {"resourceType": "MedicationStatement", "id": f"{patient_id}-med-{i}",
                                     "meta": {"tag": tag}, "status": m.status,
                                     "subject": {"reference": f"Patient/{patient_id}"},
                                     "medication": {"concept": {"text": m.medication}}}})
    for i, lab in enumerate(tables["labs"].query("patient_id == @patient_id").itertuples()):
        code = {"text": lab.test}
        if isinstance(lab.loinc, str) and lab.loinc:
            code["coding"] = [{"system": "http://loinc.org", "code": lab.loinc, "display": lab.test}]
        entries.append({"resource": {"resourceType": "Observation", "id": f"{patient_id}-lab-{i}", "meta": {"tag": tag},
                                     "status": "final", "subject": {"reference": f"Patient/{patient_id}"},
                                     "code": code, "effectiveDateTime": lab.date,
                                     "valueQuantity": {"value": float(lab.value), "unit": str(lab.unit)}}})
    return {"resourceType": "Bundle", "type": "collection", "meta": {"tag": tag}, "entry": entries}


def fhir_json(patient_id: str, tables: dict) -> str:
    return json.dumps(to_fhir_bundle(patient_id, tables), indent=2)
