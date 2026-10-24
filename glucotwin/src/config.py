"""Central configuration for GlucoTwin.

Every threshold below is a configurable research/demo setting, NOT universal medical advice.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "glucotwin.db"
CSV_DIR = DATA_DIR / "csv"
ARTIFACT_PATH = DATA_DIR / "artifacts" / "glucotwin_models.joblib"

# Data sources: the synthetic demo cohort and an imported real dataset live side by side.
SOURCE_DIRS = {"synthetic": DATA_DIR, "real": DATA_DIR / "real"}


def db_path(source: str = "synthetic") -> Path:
    return SOURCE_DIRS[source] / "glucotwin.db"


def artifact_path(source: str = "synthetic") -> Path:
    return SOURCE_DIRS[source] / "artifacts" / "glucotwin_models.joblib"

# --- Synthetic cohort -------------------------------------------------------------------------
SEED = 42
N_PATIENTS = 40
N_DAYS = 14
SAMPLE_MINUTES = 5                 # CGM sampling interval after regularisation
START_TIME = "2026-01-05 00:00"    # synthetic calendar start (not a real date of care)

# --- Prediction task --------------------------------------------------------------------------
HORIZON_MINUTES = 120                      # primary forecast horizon (2 h)
TRAJECTORY_HORIZONS = (30, 60, 90, 120)    # horizons used to draw the forecast trajectory
HYPO_THRESHOLD = 70.0                      # mg/dL, "low" threshold for the risk label
HYPER_THRESHOLD = 250.0                    # mg/dL, "high" threshold for the risk label
TARGET_RANGE = (70.0, 180.0)               # mg/dL, used for time-in-range summaries
MIN_FUTURE_COVERAGE = 0.8                  # fraction of the 2 h window that must have glucose to label it

# --- Data quality -----------------------------------------------------------------------------
MAX_IMPUTE_GAP_MINUTES = 30        # glucose gaps up to this length are linearly interpolated
HR_MAX_IMPUTE_GAP_MINUTES = 15     # heart-rate gaps up to this length are interpolated
STALE_AFTER_MINUTES = 15           # no forecast if the latest real CGM reading is older than this
TWIN_HISTORY_STEPS = 24 * 60 // SAMPLE_MINUTES + 36   # history the twin re-reads at every update

# --- Modelling / evaluation -------------------------------------------------------------------
PREDICTION_INTERVAL = 0.80         # split-conformal interval level
RISK_BANDS = {"low": 0.15, "high": 0.40}   # p < low: Low; low <= p < high: Elevated; p >= high: High
SPLIT_MODE = "auto"                # "patient_time" | "time_only" | "auto"
MIN_PATIENTS_FOR_PATIENT_SPLIT = 10  # auto -> time_only below this (e.g. OhioT1DM has 6-12 patients)
TEST_PATIENT_FRACTION = 0.25       # patient_time: patients held out entirely for testing
TIME_CUTOFF_FRACTION = 0.70        # patient_time: per-patient chronological cut-off
TIME_ONLY_CALIB_FRACTION = 0.65    # time_only: each patient's timeline -> fit | calib | test
TIME_ONLY_TEST_FRACTION = 0.80
MMOL_DETECTION_MEDIAN = 35.0       # median glucose below this is treated as mmol/L and converted
MMOL_TO_MGDL = 18.0
TRAIN_STRIDE = 3                   # use every 3rd 5-min sample for regression fitting (autocorrelation)
MIN_TRAIN_EVENTS = 30              # minimum positive training labels to fit a risk model
MIN_CALIB_EVENTS = 10              # minimum positive calibration labels to calibrate it
BOOTSTRAP_RESAMPLES = 1000


def steps(minutes: float) -> int:
    """Convert minutes to number of samples at SAMPLE_MINUTES resolution."""
    return int(round(minutes / SAMPLE_MINUTES))
