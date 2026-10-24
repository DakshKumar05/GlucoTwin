"""Feature engineering for the digital twin.

Three feature groups power the ablation study:
  * glucose  - what a CGM alone provides (recent glucose history + time of day)
  * wearable - smartwatch signals and logged lifestyle/therapy events
  * ehr      - static/historical clinical context from the EHR

All dynamic features are CAUSAL: the value at time t only uses samples at or before t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config as C

GLUCOSE_FEATURES = [
    "g_now", "g_lag_15", "g_lag_30", "g_lag_60", "g_delta_15", "g_delta_30", "g_delta_60", "g_accel",
    "g_mean_1h", "g_std_1h", "g_mean_3h", "g_std_3h", "g_min_3h", "g_max_3h", "g_mean_24h", "g_cv_24h",
    "g_pct_below_70_24h", "g_pct_above_180_24h", "g_imputed_frac_1h", "tod_sin", "tod_cos",
]
WEARABLE_FEATURES = [
    "hr_now", "hr_mean_30m", "hr_mean_2h", "hr_above_rest", "steps_30m", "steps_2h", "steps_6h",
    "activity_load", "asleep_now", "sleep_24h_h", "carbs_30m", "carbs_2h", "carbs_on_board",
    "insulin_2h", "insulin_on_board", "min_since_carbs",
]
EHR_FEATURES = [
    "age", "sex_male", "is_t1d", "years_since_diagnosis", "bmi", "hba1c_latest", "hba1c_change",
    "egfr_latest", "on_insulin", "on_bolus_insulin", "on_metformin", "on_glp1", "on_sulfonylurea",
    "has_hypertension", "has_ckd", "hypo_unawareness", "prior_hypo_events_12m",
]
FEATURE_GROUPS = {"glucose": GLUCOSE_FEATURES, "wearable": WEARABLE_FEATURES, "ehr": EHR_FEATURES}
GROUP_LABELS = {"glucose": "CGM glucose history", "wearable": "Wearable & logged lifestyle", "ehr": "EHR clinical context"}

MODEL_CONFIGS = {
    "glucose_only": ("glucose",),
    "glucose_wearable": ("glucose", "wearable"),
    "full_twin": ("glucose", "wearable", "ehr"),
}
CONFIG_LABELS = {
    "glucose_only": "1 - Glucose only",
    "glucose_wearable": "2 - Glucose + wearable",
    "full_twin": "3 - Full digital twin (+ EHR)",
}

FEATURE_DESCRIPTIONS = {
    "g_now": "Current glucose (mg/dL)", "g_lag_15": "Glucose 15 min ago", "g_lag_30": "Glucose 30 min ago",
    "g_lag_60": "Glucose 60 min ago", "g_delta_15": "Glucose change over 15 min", "g_delta_30": "Glucose change over 30 min",
    "g_delta_60": "Glucose change over 60 min", "g_accel": "Glucose acceleration (15-min change of change)",
    "g_mean_1h": "Mean glucose, last 1 h", "g_std_1h": "Glucose variability (SD), last 1 h",
    "g_mean_3h": "Mean glucose, last 3 h", "g_std_3h": "Glucose variability (SD), last 3 h",
    "g_min_3h": "Lowest glucose, last 3 h", "g_max_3h": "Highest glucose, last 3 h",
    "g_mean_24h": "Mean glucose, last 24 h", "g_cv_24h": "Coefficient of variation, last 24 h (%)",
    "g_pct_below_70_24h": "% time < 70 mg/dL, last 24 h", "g_pct_above_180_24h": "% time > 180 mg/dL, last 24 h",
    "g_imputed_frac_1h": "Fraction of imputed glucose, last 1 h", "tod_sin": "Time of day (sine)",
    "tod_cos": "Time of day (cosine)",
    "hr_now": "Heart rate now (bpm)", "hr_mean_30m": "Mean heart rate, last 30 min", "hr_mean_2h": "Mean heart rate, last 2 h",
    "hr_above_rest": "Heart rate above personal resting level", "steps_30m": "Steps, last 30 min",
    "steps_2h": "Steps, last 2 h", "steps_6h": "Steps, last 6 h", "activity_load": "Decaying activity load (3 h half-life)",
    "asleep_now": "Asleep now (wearable)", "sleep_24h_h": "Sleep in last 24 h (h)", "carbs_30m": "Logged carbs, last 30 min (g)",
    "carbs_2h": "Logged carbs, last 2 h (g)", "carbs_on_board": "Logged carbs on board (decayed, g)",
    "insulin_2h": "Logged insulin, last 2 h (U)", "insulin_on_board": "Logged insulin on board (decayed, U)",
    "min_since_carbs": "Minutes since last logged carbs",
    "age": "Age (years)", "sex_male": "Sex: male", "is_t1d": "Type 1 diabetes", "years_since_diagnosis": "Years since diagnosis",
    "bmi": "BMI", "hba1c_latest": "Latest HbA1c (%)", "hba1c_change": "HbA1c change vs previous test",
    "egfr_latest": "Latest eGFR", "on_insulin": "On insulin", "on_bolus_insulin": "On mealtime insulin",
    "on_metformin": "On metformin", "on_glp1": "On GLP-1 RA", "on_sulfonylurea": "On sulfonylurea",
    "has_hypertension": "Hypertension recorded", "has_ckd": "CKD recorded", "hypo_unawareness": "Hypoglycaemia unawareness recorded",
    "prior_hypo_events_12m": "Prior hypo events (12 months)",
}


def feature_group(name: str) -> str:
    base = name.replace(" (missing)", "")
    for group, feats in FEATURE_GROUPS.items():
        if base in feats:
            return group
    return "other"


def describe(name: str) -> str:
    base = name.replace(" (missing)", "")
    text = FEATURE_DESCRIPTIONS.get(base, base)
    return f"{text} - missing indicator" if name.endswith("(missing)") else text


# ----------------------------------------------------------------------------------------------
# Static (EHR) features
# ----------------------------------------------------------------------------------------------
def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def static_features(tables: dict) -> pd.DataFrame:
    """One row per patient, derived only from records dated before monitoring starts."""
    patients = tables["patients"].set_index("patient_id")
    conds = tables["conditions"]
    meds = tables["medications"]
    labs = tables["labs"].copy()
    labs["date"] = pd.to_datetime(labs["date"], errors="coerce")
    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    first_ts = tables["timeseries"].groupby("patient_id")["timestamp"].min()

    rows = []
    for pid, pat in patients.iterrows():
        start = first_ts.get(pid, pd.Timestamp.max)
        pl = labs[(labs["patient_id"] == pid) & (labs["date"] <= start)].sort_values("date")
        a1c = pl[pl["test"] == "HbA1c"]["value"].to_numpy()
        egfr = pl[pl["test"] == "eGFR"]["value"].to_numpy()
        pc = conds[conds["patient_id"] == pid]["display"].fillna("").astype(str).str.lower()
        pm = meds[meds["patient_id"] == pid]
        names = pm["medication"].fillna("").astype(str).str.lower()
        sex = str(pat.get("sex", "")).lower()
        dtype = str(pat.get("diabetes_type", "")).upper()
        has_meds = len(pm) > 0
        rows.append(dict(
            patient_id=pid,
            age=_num(pat.get("age")),
            sex_male=1.0 if sex in ("male", "m") else 0.0 if sex in ("female", "f") else np.nan,
            is_t1d=1.0 if dtype in ("T1D", "TYPE 1", "1") else 0.0 if dtype in ("T2D", "TYPE 2", "2") else np.nan,
            years_since_diagnosis=_num(pat.get("years_since_diagnosis")),
            bmi=_num(pat.get("bmi")),
            hba1c_latest=float(a1c[-1]) if len(a1c) else np.nan,
            hba1c_change=float(a1c[-1] - a1c[-2]) if len(a1c) > 1 else np.nan,
            egfr_latest=float(egfr[-1]) if len(egfr) else np.nan,
            # Medication/condition flags are only known when the record lists any medications/conditions.
            on_insulin=float((pm["drug_class"] == "insulin").any() or names.str.contains("insulin").any()) if has_meds else np.nan,
            on_bolus_insulin=float(names.str.contains("mealtime|pump|bolus|lispro|aspart|humalog|novolog").any()) if has_meds else np.nan,
            on_metformin=float((pm["drug_class"] == "biguanide").any() or names.str.contains("metformin").any()) if has_meds else np.nan,
            on_glp1=float((pm["drug_class"] == "GLP-1 receptor agonist").any()
                          or names.str.contains("semaglutide|liraglutide|dulaglutide|exenatide").any()) if has_meds else np.nan,
            on_sulfonylurea=float((pm["drug_class"] == "sulfonylurea").any()
                                  or names.str.contains("gliclazide|glipizide|glimepiride|glyburide").any()) if has_meds else np.nan,
            has_hypertension=float(pc.str.contains("hypertens").any()) if len(pc) else np.nan,
            has_ckd=float(pc.str.contains("kidney").any()) if len(pc) else np.nan,
            hypo_unawareness=float(pc.str.contains("awareness of hypo").any()) if len(pc) else np.nan,
            prior_hypo_events_12m=_num(pat.get("prior_hypo_events_12m")),
        ))
    return pd.DataFrame(rows).set_index("patient_id")


# ----------------------------------------------------------------------------------------------
# Dynamic features
# ----------------------------------------------------------------------------------------------
def _decayed_sum(x: pd.Series, half_life_steps: float) -> np.ndarray:
    d = 0.5 ** (1.0 / half_life_steps)
    return lfilter([1.0], [1.0, -d], np.nan_to_num(x.to_numpy(dtype=float), nan=0.0))


def dynamic_features(p: pd.DataFrame) -> pd.DataFrame:
    """Causal features for ONE patient on a regular grid (output of preprocessing)."""
    s = C.steps
    g = p["glucose"].astype(float)
    f = pd.DataFrame(index=p.index)
    f["g_now"] = g
    for m in (15, 30, 60):
        f[f"g_lag_{m}"] = g.shift(s(m))
        f[f"g_delta_{m}"] = g - f[f"g_lag_{m}"]
    f["g_accel"] = f["g_delta_15"] - (f["g_lag_15"] - f["g_lag_30"])
    for label, minutes in (("1h", 60), ("3h", 180)):
        r = g.rolling(s(minutes), min_periods=max(2, int(0.66 * s(minutes))))
        f[f"g_mean_{label}"] = r.mean()
        f[f"g_std_{label}"] = r.std()
    r3 = g.rolling(s(180), min_periods=int(0.66 * s(180)))
    f["g_min_3h"] = r3.min()
    f["g_max_3h"] = r3.max()
    r24 = g.rolling(s(1440), min_periods=s(720))
    f["g_mean_24h"] = r24.mean()
    f["g_cv_24h"] = 100 * r24.std() / f["g_mean_24h"]
    present = g.notna()
    f["g_pct_below_70_24h"] = 100 * (g < 70).astype(float).where(present).rolling(s(1440), min_periods=s(720)).mean()
    f["g_pct_above_180_24h"] = 100 * (g > 180).astype(float).where(present).rolling(s(1440), min_periods=s(720)).mean()
    f["g_imputed_frac_1h"] = p["glucose_imputed"].astype(float).rolling(s(60), min_periods=1).mean()
    minutes = p["timestamp"].dt.hour * 60 + p["timestamp"].dt.minute
    f["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)

    if "heart_rate_bpm" in p:
        hr = p["heart_rate_bpm"].astype(float)
        f["hr_now"] = hr
        f["hr_mean_30m"] = hr.rolling(s(30), min_periods=3).mean()
        f["hr_mean_2h"] = hr.rolling(s(120), min_periods=12).mean()
        f["hr_above_rest"] = f["hr_mean_30m"] - hr.rolling(s(1440), min_periods=s(360)).quantile(0.05)
    if "steps" in p:
        st = p["steps"].astype(float)
        f["steps_30m"] = st.rolling(s(30), min_periods=3).sum()
        f["steps_2h"] = st.rolling(s(120), min_periods=12).sum()
        f["steps_6h"] = st.rolling(s(360), min_periods=36).sum()
        f["activity_load"] = _decayed_sum(st / 600.0, s(180))
    if "is_asleep" in p:
        sl = p["is_asleep"].astype(float)
        f["asleep_now"] = sl
        f["sleep_24h_h"] = sl.rolling(s(1440), min_periods=int(0.7 * s(1440))).sum() * C.SAMPLE_MINUTES / 60
    if "carbs_logged_g" in p:
        cb = p["carbs_logged_g"].astype(float)
        f["carbs_30m"] = cb.rolling(s(30), min_periods=1).sum()
        f["carbs_2h"] = cb.rolling(s(120), min_periods=1).sum()
        f["carbs_on_board"] = _decayed_sum(cb, s(60))
        idx = np.arange(len(cb))
        last = pd.Series(np.where(cb.to_numpy() > 0, idx, np.nan)).ffill().to_numpy()
        f["min_since_carbs"] = np.where(np.isnan(last), 360.0, np.minimum((idx - last) * C.SAMPLE_MINUTES, 360.0))
    if "insulin_logged_u" in p:
        ins = p["insulin_logged_u"].astype(float)
        f["insulin_2h"] = ins.rolling(s(120), min_periods=1).sum()
        f["insulin_on_board"] = _decayed_sum(ins, s(90))
    return f


def build_features(grid: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Dynamic features for every patient, joined with their static EHR features."""
    parts = []
    for pid, p in grid.groupby("patient_id", sort=True):
        p = p.reset_index(drop=True)
        f = dynamic_features(p)
        base = p[["patient_id", "timestamp", "glucose", "glucose_observed", "glucose_imputed"]]
        parts.append(pd.concat([base, f], axis=1))
    out = pd.concat(parts, ignore_index=True)
    return out.merge(static, left_on="patient_id", right_index=True, how="left")


def add_targets(df: pd.DataFrame, horizons=C.TRAJECTORY_HORIZONS,
                hypo: float = C.HYPO_THRESHOLD, hyper: float = C.HYPER_THRESHOLD) -> pd.DataFrame:
    """Add forecast targets and 2-hour threshold-crossing labels, per patient.

    * target_glucose_{h}: OBSERVED (not imputed) CGM value exactly h minutes ahead.
    * label_hypo / label_hyper: whether glucose crosses the threshold at any point in (t, t+2h],
      requiring >= MIN_FUTURE_COVERAGE of that window to be available.
    * eligible_*: only rows not already beyond the threshold are eligible ("new crossing" task).
    """
    window = C.steps(C.HORIZON_MINUTES)
    min_cov = int(np.ceil(C.MIN_FUTURE_COVERAGE * window))
    out = []
    for _, p in df.groupby("patient_id", sort=True):
        p = p.copy()
        g_obs = p["glucose"].where(p["glucose_observed"])
        for h in horizons:
            p[f"target_glucose_{h}"] = g_obs.shift(-C.steps(h))
            p[f"target_delta_{h}"] = p[f"target_glucose_{h}"] - p["g_now"]
        rev = p["glucose"].iloc[::-1]
        fut_min = rev.rolling(window, min_periods=min_cov).min().iloc[::-1].shift(-1)
        fut_max = rev.rolling(window, min_periods=min_cov).max().iloc[::-1].shift(-1)
        p["future_min_2h"] = fut_min
        p["future_max_2h"] = fut_max
        p["label_hypo"] = (fut_min < hypo).astype(float).where(fut_min.notna())
        p["label_hyper"] = (fut_max > hyper).astype(float).where(fut_max.notna())
        p["eligible_hypo"] = p["g_now"].notna() & (p["g_now"] >= hypo)
        p["eligible_hyper"] = p["g_now"].notna() & (p["g_now"] <= hyper)
        out.append(p)
    return pd.concat(out, ignore_index=True)


def available_features(df: pd.DataFrame, config: str) -> list[str]:
    """Features of a config that actually exist and are not entirely missing in `df`."""
    feats = [f for group in MODEL_CONFIGS[config] for f in FEATURE_GROUPS[group]]
    return [f for f in feats if f in df.columns and df[f].notna().any()]
