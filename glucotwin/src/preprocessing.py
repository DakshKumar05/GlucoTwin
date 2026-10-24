"""Validation, regularisation to a fixed time grid, and explicit gap handling.

Conventions (documented so they can be explained to judges):
* Glucose gaps <= MAX_IMPUTE_GAP_MINUTES are linearly interpolated and flagged `glucose_imputed`.
  Longer gaps stay missing; they are never used as forecast targets.
* Heart-rate gaps <= HR_MAX_IMPUTE_GAP_MINUTES are interpolated; longer gaps stay missing.
* Steps / sleep that are missing stay missing (we do NOT assume "no activity").
* Carbohydrate and insulin columns are event logs: no entry means "nothing logged" -> 0.
  This is not the same as "nothing happened" (unlogged meals exist in the synthetic data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

REQUIRED_COLUMNS = {"patient_id", "timestamp", "glucose_mgdl"}
OPTIONAL_COLUMNS = ["heart_rate_bpm", "steps", "is_asleep", "carbs_logged_g", "insulin_logged_u"]
VALID_RANGES = {
    "glucose_mgdl": (40, 400),
    "heart_rate_bpm": (30, 220),
    "steps": (0, 3000),
    "carbs_logged_g": (0, 300),
    "insulin_logged_u": (0, 50),
}


class DataValidationError(ValueError):
    """Raised when time-series data cannot be used at all."""


def validate_timeseries(ts: pd.DataFrame, patients: pd.DataFrame | None = None):
    """Return (clean_df, issues). Fixes recoverable problems and records each one."""
    missing = REQUIRED_COLUMNS - set(ts.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")
    issues: list[str] = []
    df = ts.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = int(df["timestamp"].isna().sum())
    if n_bad:
        issues.append(f"{n_bad} rows with unparseable timestamps dropped")
        df = df[df["timestamp"].notna()]
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.any():
            issues.append(f"{int(bad.sum())} {col} values outside [{lo}, {hi}] set to missing")
            df.loc[bad, col] = np.nan
    if "is_asleep" in df.columns:
        df["is_asleep"] = pd.to_numeric(df["is_asleep"], errors="coerce")
    if patients is not None:
        unknown = sorted(set(df["patient_id"]) - set(patients["patient_id"].astype(str)))
        if unknown:
            issues.append(f"Rows for unknown patient IDs dropped: {unknown[:5]}")
            df = df[~df["patient_id"].isin(unknown)]
    n_dup = int(df.duplicated(["patient_id", "timestamp"]).sum())
    if n_dup:
        issues.append(f"{n_dup} duplicate (patient, timestamp) rows averaged")
    if df.empty:
        raise DataValidationError("No usable time-series rows after validation")
    return df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True), issues


def fill_short_gaps(s: pd.Series, max_gap_steps: int):
    """Linearly interpolate only interior NaN runs of length <= max_gap_steps.

    Returns (filled_series, imputed_mask). Longer gaps are left entirely untouched.
    """
    s = s.astype(float)
    na = s.isna()
    if not na.any():
        return s.copy(), pd.Series(False, index=s.index)
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    interp = s.interpolate(method="linear", limit_area="inside")
    fillable = na & (run_len <= max_gap_steps) & interp.notna()
    return s.where(~fillable, interp), fillable


def _regularize_patient(p: pd.DataFrame) -> pd.DataFrame:
    rule = f"{C.SAMPLE_MINUTES}min"
    p = p.set_index("timestamp").sort_index()
    out = pd.DataFrame({"glucose_raw": p["glucose_mgdl"].resample(rule).mean()})
    if "heart_rate_bpm" in p:
        out["heart_rate_bpm"] = p["heart_rate_bpm"].resample(rule).mean()
    if "steps" in p:
        out["steps"] = p["steps"].resample(rule).sum(min_count=1)
    if "is_asleep" in p:
        out["is_asleep"] = p["is_asleep"].resample(rule).max()
    for col in ("carbs_logged_g", "insulin_logged_u"):
        if col in p:
            out[col] = p[col].resample(rule).sum(min_count=1).fillna(0.0)

    out["glucose_observed"] = out["glucose_raw"].notna()
    out["glucose"], out["glucose_imputed"] = fill_short_gaps(out["glucose_raw"], C.steps(C.MAX_IMPUTE_GAP_MINUTES))
    if "heart_rate_bpm" in out:
        out["heart_rate_bpm"], out["hr_imputed"] = fill_short_gaps(
            out["heart_rate_bpm"], C.steps(C.HR_MAX_IMPUTE_GAP_MINUTES))
    return out.reset_index()


def _longest_run(mask: pd.Series) -> int:
    if not mask.any():
        return 0
    run_id = (mask != mask.shift()).cumsum()
    return int(mask.groupby(run_id).sum().max())


def preprocess(ts: pd.DataFrame, patients: pd.DataFrame | None = None):
    """Validate and regularise all patients. Returns (grid_df, quality_df, issues)."""
    df, issues = validate_timeseries(ts, patients)
    grids, quality = [], []
    for pid, p in df.groupby("patient_id", sort=True):
        g = _regularize_patient(p)
        g.insert(0, "patient_id", pid)
        grids.append(g)
        quality.append(dict(
            patient_id=pid,
            n_samples=len(g),
            first_timestamp=g["timestamp"].iloc[0],
            last_timestamp=g["timestamp"].iloc[-1],
            pct_glucose_missing_raw=100 * (~g["glucose_observed"]).mean(),
            pct_glucose_imputed=100 * g["glucose_imputed"].mean(),
            pct_glucose_missing_after=100 * g["glucose"].isna().mean(),
            longest_glucose_gap_min=_longest_run(~g["glucose_observed"]) * C.SAMPLE_MINUTES,
            pct_hr_missing=100 * g["heart_rate_bpm"].isna().mean() if "heart_rate_bpm" in g else 100.0,
            pct_steps_missing=100 * g["steps"].isna().mean() if "steps" in g else 100.0,
        ))
    grid = pd.concat(grids, ignore_index=True)
    return grid, pd.DataFrame(quality), issues
