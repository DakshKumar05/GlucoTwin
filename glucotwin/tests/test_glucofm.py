"""Sanity checks for the downloaded GlucoFM-Bench data. Skipped until `python download_glucofm.py` has been run."""
import pytest

from src.glucofm import LONG_COLUMNS, PLAUSIBLE_MGDL, glucose_column, load_split, split_path, to_long

pytestmark = pytest.mark.skipif(not split_path("train").exists(),
                                reason="GlucoFM-Bench not downloaded (run: python download_glucofm.py)")


@pytest.fixture(scope="module")
def train():
    return load_split("train")


def test_expected_columns(train):
    assert {"dataset", "subject_id", "timestamp"} <= set(train.columns)
    glucose_column(train)  # raises if neither BGvalue nor BGvalues exists


def test_list_lengths_match(train):
    gcol = glucose_column(train)
    assert (train["timestamp"].map(len) == train[gcol].map(len)).all()


def test_glucose_in_plausible_range(train):
    lo, hi = PLAUSIBLE_MGDL
    raw = to_long(train)["glucose_mg_dl"]
    assert raw.notna().mean() > 0.99
    # The source has a handful of artefacts (7 readings of 1.8 mg/dL in one T1DM-UOM subject); fail if that grows.
    assert (~raw.between(lo, hi)).mean() < 1e-4, f"min={raw.min()}, max={raw.max()}"
    masked = to_long(train, mask_implausible=True)["glucose_mg_dl"].dropna()
    assert masked.between(lo, hi).all(), f"min={masked.min()}, max={masked.max()}"


def test_to_long_and_cohort_filter(train):
    cohort = train["dataset"].iloc[0]
    long = to_long(train, cohorts=cohort)
    assert list(long.columns) == LONG_COLUMNS
    assert (long["dataset"] == cohort).all()
    gcol = glucose_column(train)
    assert len(long) == train.loc[train["dataset"] == cohort, gcol].map(len).sum()
