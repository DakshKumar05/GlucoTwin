"""Loader for the external GlucoFM-Bench CGM forecasting benchmark (Hugging Face: glucofmbench/GlucoFM-Bench).

The raw splits are downloaded once with `python download_glucofm.py` into data/raw/glucofm/{train,test}.parquet.
Each row is one subject from one of 11 open CGM cohorts: `timestamp` (Unix seconds) and `BGvalue` (mg/dL) are
equal-length lists, already resampled to 5-minute readings. Gaps over 1 hour inside a list are sequence boundaries.
Note: although the dataset card describes a participant-level split, the downloaded data is a chronological split
WITHIN each subject (every subject appears in both splits; test = the last ~20% of their readings, after train ends).
Licence: CC BY-NC-SA 4.0 (non-commercial) plus each source cohort's licence.

This data is kept separate from the synthetic cohort and from `data/real/`; nothing here touches SQLite.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from . import config as C

HF_DATASET = "glucofmbench/GlucoFM-Bench"
GLUCOFM_DIR = C.DATA_DIR / "raw" / "glucofm"
SPLITS = ("train", "test")
# The dataset card uses both spellings for the glucose list column; accept either.
GLUCOSE_COLUMNS = ("BGvalue", "BGvalues")
LONG_COLUMNS = ["dataset", "subject_id", "timestamp", "glucose_mg_dl"]
PLAUSIBLE_MGDL = (20.0, 600.0)  # outside this a CGM value is treated as an artefact (e.g. 7 readings of 1.8 mg/dL)


def split_path(split: str) -> Path:
    """Return the local Parquet path for a split ("train" or "test")."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return GLUCOFM_DIR / f"{split}.parquet"


def glucose_column(df: pd.DataFrame) -> str:
    """Name of the per-row glucose list column (BGvalue or BGvalues) present in `df`."""
    for col in GLUCOSE_COLUMNS:
        if col in df.columns:
            return col
    raise KeyError(f"No glucose column found; expected one of {GLUCOSE_COLUMNS}, got {list(df.columns)}")


def _as_list(cohorts: str | Iterable[str] | None) -> list[str] | None:
    if cohorts is None:
        return None
    return [cohorts] if isinstance(cohorts, str) else list(cohorts)


def load_split(split: str, cohorts: str | Iterable[str] | None = None) -> pd.DataFrame:
    """Read one split from the local Parquet file (one row per subject, list-valued columns).

    Args:
        split: "train" or "test".
        cohorts: optional source cohort name(s) to keep, matched against the `dataset` column.

    Raises:
        FileNotFoundError: if the split has not been downloaded yet.
        ValueError: if a requested cohort is not present in the split.
    """
    path = split_path(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python download_glucofm.py")
    df = pd.read_parquet(path)
    return filter_cohorts(df, cohorts)


def filter_cohorts(df: pd.DataFrame, cohorts: str | Iterable[str] | None) -> pd.DataFrame:
    """Keep only rows whose `dataset` (source cohort) is in `cohorts`; `None` keeps everything."""
    wanted = _as_list(cohorts)
    if wanted is None:
        return df
    unknown = sorted(set(wanted) - set(df["dataset"].unique()))
    if unknown:
        raise ValueError(f"Unknown cohort(s) {unknown}; available: {sorted(df['dataset'].unique())}")
    return df[df["dataset"].isin(wanted)].reset_index(drop=True)


def to_long(df: pd.DataFrame, cohorts: str | Iterable[str] | None = None,
            mask_implausible: bool = False) -> pd.DataFrame:
    """Explode list columns into one row per reading: [dataset, subject_id, timestamp, glucose_mg_dl].

    Args:
        df: a split as returned by `load_split`.
        cohorts: optional source cohort name(s) to keep (see `filter_cohorts`).
        mask_implausible: set glucose outside PLAUSIBLE_MGDL to NaN (the reading is kept, so timestamps stay aligned).

    `timestamp` keeps the dataset's raw float values (Unix seconds; no conversion is applied). Rows whose timestamp
    and glucose lists differ in length raise a ValueError rather than being silently misaligned.
    """
    df = filter_cohorts(df, cohorts)
    gcol = glucose_column(df)
    lengths_t = df["timestamp"].map(len)
    lengths_g = df[gcol].map(len)
    bad = lengths_t != lengths_g
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have timestamp/glucose lists of different lengths")
    out = (df[["dataset", "subject_id", "timestamp", gcol]]
           .rename(columns={gcol: "glucose_mg_dl"})
           .explode(["timestamp", "glucose_mg_dl"], ignore_index=True))
    out["timestamp"] = out["timestamp"].astype("float64")
    out["glucose_mg_dl"] = out["glucose_mg_dl"].astype("float64")
    if mask_implausible:
        lo, hi = PLAUSIBLE_MGDL
        out["glucose_mg_dl"] = out["glucose_mg_dl"].where(out["glucose_mg_dl"].between(lo, hi))
    return out[LONG_COLUMNS]
