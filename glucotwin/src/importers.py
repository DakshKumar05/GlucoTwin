"""Importers for REAL datasets -> GlucoTwin tables.

Supported:
  * OhioT1DM XML files (one file per patient and split, e.g. 559-ws-training.xml / 559-ws-testing.xml).
    Access requires the dataset's data-use agreement; nothing is bundled here.
  * Generic CSV: a time-series file (or one file per patient) with flexible column names, plus optional
    patients.csv / conditions.csv / medications.csv / labs.csv.

Importers never invent links between people: every row keeps the patient identifier found in the source.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data_loader import normalize_tables

BIN = f"{C.SAMPLE_MINUTES}min"
EHR_FILES = ("patients", "conditions", "medications", "labs")

# ----------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------
def _bin(df: pd.DataFrame, col: str, how: str) -> pd.Series:
    """Aggregate one event column onto the 5-min grid (bins are floored timestamps)."""
    if df.empty:
        return pd.Series(dtype=float, name=col)
    d = df.assign(timestamp=df["timestamp"].dt.floor(BIN))
    return d.groupby("timestamp")[col].agg(how).rename(col)


def _intervals_to_bins(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    idx = []
    for begin, end in intervals:
        if pd.isna(begin) or pd.isna(end):
            continue
        if begin > end:
            begin, end = end, begin
        if end - begin > pd.Timedelta(hours=16):   # implausible single sleep episode
            continue
        idx.append(pd.date_range(begin.floor(BIN), end.floor(BIN), freq=BIN))
    if not idx:
        return pd.Series(dtype=float, name="is_asleep")
    allbins = idx[0].append(idx[1:]) if len(idx) > 1 else idx[0]
    return pd.Series(1.0, index=allbins.unique(), name="is_asleep")


def _to_mgdl(values: pd.Series, column_name: str = "") -> tuple[pd.Series, str]:
    v = pd.to_numeric(values, errors="coerce")
    if "mmol" in column_name.lower() or (v.notna().any() and v.median() < C.MMOL_DETECTION_MEDIAN):
        return v * C.MMOL_TO_MGDL, "mmol/L (converted to mg/dL)"
    return v, "mg/dL"


# ----------------------------------------------------------------------------------------------
# OhioT1DM
# ----------------------------------------------------------------------------------------------
OHIO_TS_FORMAT = "%d-%m-%Y %H:%M:%S"


def _ohio_events(root: ET.Element, tag: str, attrs: tuple[str, ...]) -> pd.DataFrame:
    node = root.find(tag)
    rows = [] if node is None else [{a: ev.get(a) for a in attrs} for ev in node.findall("event")]
    df = pd.DataFrame(rows, columns=list(attrs))
    for a in attrs:
        if a.startswith("ts"):
            df[a] = pd.to_datetime(df[a], format=OHIO_TS_FORMAT, errors="coerce")
    return df


def parse_ohio_file(path: Path) -> tuple[str, dict, pd.DataFrame, list]:
    """Parse one OhioT1DM XML file -> (patient id, patient attributes, 5-min frame, sleep intervals)."""
    root = ET.parse(path).getroot()
    pid = root.get("id") or Path(path).stem.split("-")[0]
    attrs = dict(root.attrib)

    glucose = _ohio_events(root, "glucose_level", ("ts", "value")).rename(columns={"ts": "timestamp"})
    glucose["glucose_mgdl"] = pd.to_numeric(glucose["value"], errors="coerce")
    hr = _ohio_events(root, "basis_heart_rate", ("ts", "value")).rename(columns={"ts": "timestamp"})
    hr["heart_rate_bpm"] = pd.to_numeric(hr["value"], errors="coerce")
    steps = _ohio_events(root, "basis_steps", ("ts", "value")).rename(columns={"ts": "timestamp"})
    steps["steps"] = pd.to_numeric(steps["value"], errors="coerce")
    meals = _ohio_events(root, "meal", ("ts", "carbs")).rename(columns={"ts": "timestamp"})
    meals["carbs_logged_g"] = pd.to_numeric(meals["carbs"], errors="coerce")
    bolus = _ohio_events(root, "bolus", ("ts_begin", "dose")).rename(columns={"ts_begin": "timestamp"})
    bolus["insulin_logged_u"] = pd.to_numeric(bolus["dose"], errors="coerce")

    parts = [
        _bin(glucose.dropna(subset=["timestamp"]), "glucose_mgdl", "mean"),
        _bin(hr.dropna(subset=["timestamp"]), "heart_rate_bpm", "mean"),
        _bin(steps.dropna(subset=["timestamp"]), "steps", "sum"),
        _bin(meals.dropna(subset=["timestamp"]), "carbs_logged_g", "sum"),
        _bin(bolus.dropna(subset=["timestamp"]), "insulin_logged_u", "sum"),
    ]
    frame = pd.concat([p for p in parts if len(p)], axis=1, sort=True) if any(len(p) for p in parts) else pd.DataFrame()
    for col in ("glucose_mgdl", "heart_rate_bpm", "steps", "carbs_logged_g", "insulin_logged_u"):
        if col not in frame:
            frame[col] = np.nan
    # Drop wearable columns entirely if this file has none of them (e.g. cohorts without a Basis band).
    if hr.empty:
        frame = frame.drop(columns=["heart_rate_bpm"])
    if steps.empty:
        frame = frame.drop(columns=["steps"])

    sleep = []
    for tag in ("sleep", "basis_sleep"):
        s = _ohio_events(root, tag, ("ts_begin", "ts_end"))
        sleep += list(zip(s["ts_begin"], s["ts_end"]))
    return pid, attrs, frame, sleep


def import_ohiot1dm(input_path: str | Path) -> tuple[dict, list[str]]:
    """Import all OhioT1DM XML files under a folder (or a single file). Training and testing files of the
    same patient are concatenated chronologically."""
    path = Path(input_path)
    files = sorted(path.rglob("*.xml")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No .xml files found under {path}")
    notes, frames, sleeps, attrs = [], {}, {}, {}
    for f in files:
        pid, a, frame, sleep = parse_ohio_file(f)
        frames.setdefault(pid, []).append(frame)
        sleeps.setdefault(pid, []).extend(sleep)
        attrs.setdefault(pid, {}).update(a)
        notes.append(f"{f.name}: patient {pid}, {frame['glucose_mgdl'].notna().sum():,} CGM readings")

    ts_parts, patients, meds, conds = [], [], [], []
    for pid, fl in frames.items():
        frame = pd.concat(fl)
        frame = frame.groupby(level=0).agg({c: ("sum" if c in ("steps", "carbs_logged_g", "insulin_logged_u") else "mean")
                                             for c in frame.columns}).sort_index()
        for c in ("carbs_logged_g", "insulin_logged_u"):
            frame[c] = frame[c].replace(0, np.nan) if c in frame else np.nan
        asleep = _intervals_to_bins(sleeps[pid])
        if len(asleep):
            frame["is_asleep"] = 0.0
            frame.loc[frame.index.isin(asleep.index), "is_asleep"] = 1.0
        frame.index.name = "timestamp"
        frame = frame.reset_index()
        frame.insert(0, "patient_id", f"OHIO-{pid}")
        ts_parts.append(frame)
        insulin = attrs[pid].get("insulin_type") or "rapid-acting insulin"
        patients.append(dict(patient_id=f"OHIO-{pid}", display_name=f"OhioT1DM {pid}", diabetes_type="T1D",
                             insulin_delivery="pump (CSII)", provenance="OhioT1DM"))
        meds.append(dict(patient_id=f"OHIO-{pid}", medication=f"{insulin} via pump (CSII)", drug_class="insulin",
                         start_date="", status="active"))
        conds.append(dict(patient_id=f"OHIO-{pid}", code_system="SNOMED-CT", code="46635009",
                          display="Diabetes mellitus type 1", onset_date=""))
    tables = dict(timeseries=pd.concat(ts_parts, ignore_index=True), patients=pd.DataFrame(patients),
                  medications=pd.DataFrame(meds), conditions=pd.DataFrame(conds))
    notes.append("OhioT1DM provides no age/BMI/HbA1c in the XML: those EHR features will be unavailable. "
                 "Sleep = self-reported + band sleep intervals; basal insulin is not used.")
    return normalize_tables(tables, provenance="OhioT1DM"), notes


# ----------------------------------------------------------------------------------------------
# Generic CSV
# ----------------------------------------------------------------------------------------------
ALIASES = {
    "patient_id": ["patient_id", "patient", "subject", "subject_id", "participant", "pid", "id", "ptid"],
    "timestamp": ["timestamp", "time", "datetime", "date_time", "ts", "device_timestamp", "displaytime",
                  "timestamp_yyyy_mm_ddthh_mm_ss", "date"],
    "glucose_mgdl": ["glucose_mgdl", "glucose", "glucose_value", "glucose_value_mg_dl", "bg", "sgv", "cgm",
                     "historic_glucose_mg_dl", "historic_glucose_mmol_l", "glucose_mmol", "glucose_mmol_l", "sensor_glucose",
                     "glucose_value_mmol_l", "sensor_glucose_mg_dl", "glucose_level"],
    "heart_rate_bpm": ["heart_rate_bpm", "heart_rate", "hr", "bpm", "heartrate"],
    "steps": ["steps", "step_count", "stepcount"],
    "is_asleep": ["is_asleep", "asleep", "sleep", "sleeping"],
    "carbs_logged_g": ["carbs_logged_g", "carbs", "carbohydrates", "carb_input", "meal_carbs", "carbs_g"],
    "insulin_logged_u": ["insulin_logged_u", "insulin", "bolus", "bolus_insulin", "insulin_units", "insulin_value_u",
                         "bolus_u"],
}
PATIENT_ALIASES = {
    "patient_id": ALIASES["patient_id"], "age": ["age", "age_years"], "sex": ["sex", "gender"],
    "diabetes_type": ["diabetes_type", "type", "dm_type"], "bmi": ["bmi"],
    "years_since_diagnosis": ["years_since_diagnosis", "diabetes_duration", "duration_years"],
    "prior_hypo_events_12m": ["prior_hypo_events_12m", "hypo_events", "hypo_events_12m"],
    "insulin_delivery": ["insulin_delivery", "insulin_method"], "hba1c": ["hba1c", "a1c", "hba1c_percent"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _map_columns(df: pd.DataFrame, aliases: dict, overrides: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Rename source columns to GlucoTwin names. Returns (renamed df, {target: source column})."""
    overrides = overrides or {}
    by_norm = {_norm(c): c for c in df.columns}
    mapping = {}
    for target, names in aliases.items():
        if target in overrides:
            if overrides[target] not in df.columns:
                raise KeyError(f"--map {target}={overrides[target]}: column not found. Columns: {list(df.columns)}")
            mapping[target] = overrides[target]
            continue
        for n in names:
            if _norm(n) in by_norm and by_norm[_norm(n)] not in mapping.values():
                mapping[target] = by_norm[_norm(n)]
                break
    return df.rename(columns={v: k for k, v in mapping.items()})[list(mapping)], mapping


def _read_timeseries_csv(path: Path, overrides: dict | None, notes: list) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df, mapping = _map_columns(raw, ALIASES, overrides)
    if "glucose_mgdl" not in df or "timestamp" not in df:
        raise ValueError(f"{path.name}: need a timestamp and a glucose column. Found {list(raw.columns)}. "
                         "Use --map timestamp=<col> --map glucose_mgdl=<col>.")
    if "patient_id" not in df:
        df["patient_id"] = path.stem
        notes.append(f"{path.name}: no patient column - using file name '{path.stem}' as patient_id")
    df["glucose_mgdl"], unit = _to_mgdl(df["glucose_mgdl"], mapping.get("glucose_mgdl", ""))
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=False, format="mixed")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    df["timestamp"] = ts
    notes.append(f"{path.name}: mapped {mapping}; glucose unit {unit}; "
                 f"{df['timestamp'].isna().sum()} unparseable timestamps dropped")
    return df[df["timestamp"].notna()]


def import_csv(input_path: str | Path, overrides: dict | None = None) -> tuple[dict, list[str]]:
    """Import a CSV file, or a folder containing timeseries.csv (or one CSV per patient) plus optional
    patients.csv / conditions.csv / medications.csv / labs.csv."""
    path = Path(input_path)
    notes: list[str] = []
    if path.is_file():
        ts_files, folder = [path], path.parent
    else:
        folder = path
        ts_files = [folder / "timeseries.csv"] if (folder / "timeseries.csv").exists() else \
            [f for f in sorted(folder.glob("*.csv")) if f.stem.lower() not in EHR_FILES]
    if not ts_files:
        raise FileNotFoundError(f"No time-series CSV found in {path}")
    ts = pd.concat([_read_timeseries_csv(f, overrides, notes) for f in ts_files], ignore_index=True)
    ts["patient_id"] = ts["patient_id"].astype(str)
    tables = {"timeseries": ts}

    pfile = folder / "patients.csv"
    if pfile.exists() and pfile not in ts_files:
        pats, pmap = _map_columns(pd.read_csv(pfile), PATIENT_ALIASES)
        pats["patient_id"] = pats["patient_id"].astype(str)
        notes.append(f"patients.csv: mapped {pmap}")
        if "hba1c" in pats:  # a single HbA1c column becomes a lab result
            first = ts.groupby("patient_id")["timestamp"].min()
            labs = pats[["patient_id", "hba1c"]].dropna().rename(columns={"hba1c": "value"})
            labs["date"] = labs["patient_id"].map(first).dt.date.astype(str)
            labs = labs.assign(test="HbA1c", loinc="4548-4", unit="%")
            tables["labs"] = labs
            pats = pats.drop(columns=["hba1c"])
        tables["patients"] = pats
    for name in ("conditions", "medications", "labs"):
        f = folder / f"{name}.csv"
        if f.exists() and f not in ts_files:
            df = pd.read_csv(f, dtype={"patient_id": str})
            tables[name] = pd.concat([tables.get(name, pd.DataFrame()), df], ignore_index=True)
            notes.append(f"{name}.csv: {len(df)} rows")
    return normalize_tables(tables, provenance="imported"), notes


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def dataset_report(tables: dict) -> str:
    ts = tables["timeseries"].copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    lines = [f"Patients: {ts['patient_id'].nunique()}   rows: {len(ts):,}"]
    rows = []
    for pid, p in ts.groupby("patient_id"):
        g = p["glucose_mgdl"]
        step = p.loc[g.notna(), "timestamp"].sort_values().diff().median()
        rows.append(dict(
            patient=pid, days=round((p["timestamp"].max() - p["timestamp"].min()).total_seconds() / 86400, 1),
            cgm_readings=int(g.notna().sum()), median_interval=str(step).replace("0 days ", ""),
            mean_glucose=round(float(g.mean()), 1), pct_below_70=round(float((g < 70).mean() * 100), 2),
            pct_above_250=round(float((g > 250).mean() * 100), 2),
            hr=f"{p['heart_rate_bpm'].notna().mean() * 100:.0f}%" if "heart_rate_bpm" in p else "-",
            steps=f"{p['steps'].notna().mean() * 100:.0f}%" if "steps" in p else "-",
            carbs_logs=int((p["carbs_logged_g"].fillna(0) > 0).sum()) if "carbs_logged_g" in p else 0,
            insulin_logs=int((p["insulin_logged_u"].fillna(0) > 0).sum()) if "insulin_logged_u" in p else 0,
        ))
    lines.append(pd.DataFrame(rows).to_string(index=False))
    pats = tables["patients"]
    ehr = [c for c in ("age", "sex", "bmi", "years_since_diagnosis", "prior_hypo_events_12m") if pats[c].notna().any()]
    lab_tests = sorted(tables["labs"]["test"].dropna().unique()) if len(tables["labs"]) else []
    lines.append(f"EHR fields available: {ehr or 'none'}; labs: {lab_tests or 'none'}; "
                 f"medications: {len(tables['medications'])}; conditions: {len(tables['conditions'])}")
    n = ts["patient_id"].nunique()
    mode = "time_only" if n < C.MIN_PATIENTS_FOR_PATIENT_SPLIT else "patient_time"
    lines.append(f"Split that 'auto' will use: {mode} "
                 f"({'within-patient chronological' if mode == 'time_only' else 'held-out patients + time cut-off'})")
    if ts["timestamp"].max() - ts["timestamp"].min() < pd.Timedelta(days=3):
        lines.append("WARNING: less than 3 days of data - too little for meaningful training.")
    return "\n".join(lines)
