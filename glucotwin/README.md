# GlucoTwin — an AI patient digital twin for 2-hour glucose risk

> **Research & educational prototype on synthetic data.** Not a medical device, not clinically validated,
> and it never gives treatment or insulin-dosing advice.

GlucoTwin builds a virtual copy of each synthetic patient by combining their static EHR record (diagnoses,
medications, labs, history) with a continuously updated stream from a glucose sensor (CGM) and a wearable. From that
combined state it forecasts glucose for the next 2 hours, estimates the risk of crossing a low (< 70 mg/dL) or
high (> 250 mg/dL) threshold, and explains each prediction in a doctor-facing Streamlit dashboard.

It also runs an experiment the challenge asks for: **does adding wearable data and EHR context actually improve
predictions compared with glucose history alone?**

---

## 1. Quick start (Windows)

```powershell
cd C:\Users\daksh\glucotwin
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python train.py --regenerate     # ~1 min: generates synthetic data, trains 30 models, prints evaluation
streamlit run app.py             # opens http://localhost:8501
pytest -q                        # 30 tests, ~20 s
```

`train.py` is optional: if no trained models exist, the dashboard generates the data and trains automatically on
first launch, showing a progress bar. Nothing needs to be downloaded; the app works fully offline.

If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call the venv
Python directly: `.venv\Scripts\python -m streamlit run app.py`.

### Google Colab (optional)
```python
!git clone <your-repo-url> glucotwin   # or upload the folder
%cd glucotwin
!pip install -q -r requirements.txt
!python train.py --regenerate
!pytest -q
# Dashboard (needs a tunnel; less reliable than running locally):
!streamlit run app.py --server.port 8501 &>/content/st.log &
!npx --yes localtunnel --port 8501
```
For a live presentation, run locally. Colab tunnels can drop mid-demo.

---

## 2. Architecture

```
synthetic_data.py ──► data_loader.py (SQLite) ──► preprocessing.py ──► features.py ──┐
   (one coherent ID per patient:                 (validate, 5-min grid,   (causal       │
    EHR + labs + meds + CGM + wearable)           gap imputation ≤30 min)  features,     │
                                                                            targets)     ▼
             app.py (Streamlit) ◄── digital_twin.py ◄── forecasting.py / risk_prediction.py
             dashboard + replay      PatientTwin:          ridge + XGBoost per horizon,
                                     re-reads data ≤ clock  conformal intervals,
                                     every update           isotonic-calibrated risk
                                            ▲
                         explainability.py (TreeSHAP / linear)   evaluation.py (held-out patients,
                                                                  block bootstrap ablation)
```

| File | Responsibility |
|---|---|
| `src/config.py` | All settings: horizons, thresholds, split fractions, interval level |
| `src/synthetic_data.py` | Synthetic cohort generator + list of every simulation assumption |
| `src/data_loader.py` | SQLite persistence (PostgreSQL-ready), CSV export, FHIR R4-style export |
| `src/preprocessing.py` | Validation, regular 5-min grid, explicit observed / imputed / missing flags |
| `src/features.py` | Glucose, wearable and EHR feature groups; targets and 2-h labels |
| `src/forecasting.py` | Patient + time split, ridge vs gradient boosting, split-conformal intervals |
| `src/risk_prediction.py` | Threshold-crossing classifiers, isotonic calibration, refusal when data is thin |
| `src/explainability.py` | Per-prediction contributions (TreeSHAP via XGBoost, exact linear for ridge) |
| `src/evaluation.py` | MAE/RMSE, precision/recall/F1/PR-AUC/Brier/ECE, ablation with block bootstrap |
| `src/digital_twin.py` | `PatientTwin`: the evolving virtual patient |
| `src/pipeline.py` | Orchestration, artifact caching (retrains automatically if data/config changes) |
| `app.py` | Dashboard: 8 sections (incl. the embedded Patient Journey) + twin clock with replay |
| `.streamlit/config.toml`, `static/` | Dashboard theme (shared design system: dark ink canvas, Aspekta + Roboto Mono, one lime accent) and its bundled fonts (OFL) |
| `src/journey.py` | Embeds `../demo/patient-journey.html` as a self-sizing component |
| `train.py` | CLI training/evaluation |
| `src/importers.py` | Real-data importers: OhioT1DM XML, generic CSV (column aliases, mmol/L), dataset report |
| `import_data.py` | CLI to import / append a real dataset into `data/real/` |
| `src/glucofm.py` | Loader for the external GlucoFM-Bench benchmark (`load_split`, `to_long`, cohort filter) |
| `download_glucofm.py` | CLI to download GlucoFM-Bench into `data/raw/glucofm/` |
| `tests/` | 30 tests: data, preprocessing, horizons, causality, leakage, both split modes, missing data, importers, twin behaviour |

### How the twin differs from an ordinary prediction model
An ordinary model takes a feature table and returns numbers. The `PatientTwin` is a **stateful, patient-specific
object**:
1. It holds the patient's **EHR layer** and **sensor layer** under one identity.
2. It has a **clock**. Every update re-reads only the observations available up to that time and recomputes all
   features. A test (`test_features_are_causal`) checks that this matches the offline feature table exactly, so
   nothing from the future leaks in.
3. Its state includes data quality: data freshness, imputation, missing inputs. It **withholds** forecasts when the
   CGM is stale for more than 15 min, rather than guessing.
4. It runs **forward**: a forecast trajectory with prediction intervals, risk probabilities, and a
   model-sensitivity tool for hypothetical carbs or activity.
5. Its output goes back to a **human**: explanations and review prompts, never automatic actions.

This follows the digital-twin definition (physical entity, virtual model, two-way information flow) used in the
healthcare-DT literature. It is a *statistical* twin; it does not include a validated mechanistic physiology model.

---

## 3. Data strategy and simulated linkage

* **Default (used here): fully synthetic.** `src/synthetic_data.py` creates 40 patients (about 40% T1D, 60% T2D)
  × 14 days at 5-minute resolution, which is 161,280 sensor rows. **Each patient's EHR and time series share one
  `patient_id`.** The link is simulated, but it is coherent by construction: we never join unrelated people.
* **Synthea / OhioT1DM:** not bundled. Synthea patients and OhioT1DM subjects are different people, so joining them
  would invent a false link. OhioT1DM also requires a data-use agreement. The loader accepts any table with
  `patient_id, timestamp, glucose_mgdl` plus any of the optional columns, so legitimately obtained data can be
  plugged in later. Optional columns are handled explicitly when absent (see the tests).
* **Provenance labels:** every row carries `provenance = "simulated"`. Glucose samples are flagged
  `glucose_observed` or `glucose_imputed`. The hidden "true" glucose and true carbs are stored separately in
  `simulation_truth` and are **never used by any model**.

### Simulation assumptions (also shown in the dashboard)
Baselines are calibrated so that mean CGM ≈ the ADAG estimated average glucose for each patient's HbA1c. Glucose
follows a mean-reverting difference equation driven by:
* meal carbohydrates (gamma kernel, peak about 50 min)
* rapid-acting insulin (peak about 75 min)
* exercise (an immediate effect plus a 3-hour sensitivity boost)
* a dawn phenomenon
* short-sleep effects
* sulfonylurea / GLP-1 effects
* random noise

A hidden "hypoglycaemia propensity" drives both the EHR count of prior hypos and systematic insulin
over-estimation. Patients self-treat lows with rescue carbs, and heart rate rises during hypoglycaemia unless
unawareness is recorded. Missingness is injected on purpose: CGM dropouts, watch charging, nights without the watch,
unlogged meals, and noisy carb estimates. **All effect sizes were chosen by us for plausibility; none are estimated
from real data.** The full list is in `SIMULATION_ASSUMPTIONS`.

---

## 4. Modelling choices

| Item | Choice | Why |
|---|---|---|
| Horizon | 120 min = 24 steps at 5 min (`config.steps()` adapts to other rates) | Challenge requirement |
| Target | *Observed* CGM exactly 2 h ahead; models predict the change from now | Imputed values are never targets; predicting deltas is easier for trees |
| Trajectory | Separate direct models at 30/60/90/120 min | No error compounding |
| Model families | Ridge (simple, reliable) vs XGBoost; chosen by **calibration-set** MAE | "Simple first; add complexity only if it helps" |
| Configurations | ① glucose only ② + wearable/lifestyle ③ + EHR (full twin), plus a persistence baseline | Ablation |
| Uncertainty | Split-conformal 80% intervals from calibration residuals | Distribution-free; coverage reported on test |
| Risk label | Any crossing of < 70 or > 250 mg/dL in the next 2 h (≥ 80% of the window observed), scored only if not already beyond the threshold | A *new*-crossing task, not a trivial one |
| Calibration | Isotonic on calibration split; alert threshold = F1-max on calibration split | Never tuned on test |
| Refusal | Fewer than 30 training (or 10 calibration) positive windows → no probability shown | No invented probabilities |
| Explanations | TreeSHAP from XGBoost (`pred_contribs`), exact linear contributions for ridge | Labelled as associations, not causes |

### Evaluation design
* **Patient-level split:** 30 training / 10 held-out patients, stratified by diabetes type.
* **Chronological split:** models are fitted on data before the cut-off (70% of the calendar), with a 2-hour
  embargo so no training target falls after the cut-off. Calibration uses training patients after the cut-off;
  **reported metrics use only held-out patients after the cut-off.**
* Medians, scalers and calibrators are fitted on training data only.
* Ablation significance uses a **block bootstrap over patient-days** (1,000 resamples). That respects
  autocorrelation, unlike naive per-sample tests.

---

## 5. Results from our run (seed 42, synthetic data)

Produced by `python train.py --regenerate`. Every number is computed from predictions at run time; nothing is
hard-coded. Rerun to reproduce.

**2-hour forecast, held-out patients (n = 11,625 test points)**

| Configuration | Model used | MAE (mg/dL) | RMSE | 80% interval coverage |
|---|---|---|---|---|
| 0 · Persistence | last value | 31.85 | 42.67 | — |
| 1 · Glucose only | XGBoost | 22.05 | 31.74 | 0.80 |
| 2 · + Wearable | XGBoost | 20.76 | 29.13 | 0.77 |
| 3 · Full twin (+ EHR) | XGBoost | 20.26 | 28.57 | 0.77 |

**Ablation (Δ MAE, 95% block-bootstrap CI)**
* Learned glucose model vs persistence: −9.80 [−12.13, −7.70]. Clear improvement.
* \+ wearable & lifestyle: −1.29 [−2.21, −0.34]. Clear improvement.
* \+ EHR on top of wearable: −0.50 [−1.18, 0.11]. **No clear difference.**
* Full twin vs glucose only: −1.79 [−2.74, −0.83]. Clear improvement.

**Risk (held-out patients)**

The hyper model (> 250, base rate 7.6%) reaches PR-AUC 0.50–0.55. The hypo model (< 70, base rate 1.1%, only 125
overlapping positive windows) reaches PR-AUC 0.34–0.46. For hypo, *adding* features made results worse, which is
consistent with overfitting on static patient-level features when only 30 patients are available for training.

**How to read this honestly**
* Wearable signals help in this simulation. EHR context adds little once wearable data is present, and the
  simulator *was designed* to let EHR attributes matter. On real data the effect could be larger or smaller.
* Interval coverage is slightly under nominal (0.77 vs 0.80). That is expected, because time series violate the
  exchangeability that conformal guarantees rely on.
* With 10 test patients, all estimates are noisy. **None of this establishes clinical performance.**

---

## 6. Dashboard tour

| Section | What it shows |
|---|---|
| **Overview** | Current glucose and trend arrow, 2-h forecast, time in range / CV / GMI, risk status, profile, latest observations with "last seen" times, data-quality and provenance panel |
| **Glucose Forecast** | Observed vs imputed CGM, forecast trajectory with 80% band, thresholds, optional "reveal what actually happened" (hidden from the model) |
| **Clinical Context** | Diagnoses (SNOMED-CT where known), medications, labs, HbA1c history, 24-h heart rate / steps / sleep / carbs / insulin, explanation of the fusion |
| **Risk & Explainability** | Calibrated hypo/hyper risk with definitions, top drivers (TreeSHAP), contribution by data source, uncertainty and missing-data warnings, informational review prompts, scenario-sensitivity tool |
| **Model Performance** | Split design, metrics for every configuration and family, MAE by horizon, ablation with CIs, risk metrics, calibration curve, predicted-vs-actual, global importance |
| **Twin Timeline** | Live state with deltas, forecasts the twin made earlier plotted against what then happened, risk over time, update log |
| **About & Limitations** | Digital-twin rationale, all simulation assumptions, limitations, cohort data quality, FHIR / CSV download |

**Twin clock (sidebar):** a slider, +5m / +30m / +2h buttons, **▶ Play** (streams one step per tick), and
**⚑ Next alert** (jumps to just before the next low-glucose alert, or high-glucose if there is none). The default patient is the held-out patient with the
most post-cut-off hypo windows. It was chosen for demonstration value, and we disclose that.

---

## 7. Testing & troubleshooting

`pytest -q` covers:
* coherent IDs across tables
* sampling interval and ranges
* determinism and the SQLite round trip
* FHIR tagging
* gap imputation that fills only gaps of ≤ 30 min
* validation errors
* the regular grid
* missing optional columns
* the exact 24-step target
* no targets crossing patients
* labels matching the future window
* **feature causality**
* **no patient leakage and the embargo**
* missing-feature handling
* the conformal quantile
* predictions with missing wearable data
* metrics computed
* risk-model refusal
* **the twin updating with time and withholding stale forecasts**

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` when running tests | Run `pytest` from the project root (`tests/conftest.py` adds the path) |
| XGBoost fails to install/import | The code falls back to sklearn `HistGradientBoosting` automatically; explanations switch to occlusion sensitivity |
| Dashboard is slow on first launch | It is training (~25 s). Run `python train.py` beforehand |
| Models seem out of date after changing `config.py` | Artifacts carry a fingerprint and retrain automatically; force it with `python train.py` |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Want fresh data | `python train.py --regenerate --patients 60 --days 21` |

**PostgreSQL path:** replace `get_connection()` in `data_loader.py` with a SQLAlchemy engine
(`create_engine("postgresql+psycopg://user:pw@host/db")`). `pandas.read_sql` / `to_sql` work unchanged.

---

## 8. Limitations
* Synthetic data from our own simulator: models can learn the simulator's rules. The EHR→glucose links are our
  assumptions. Real-world performance is unknown.
* Small cohort (40 patients, 14 days, 10 held-out). Hypoglycaemia windows are rare and heavily overlapping.
* Static EHR features can act as patient identifiers. That is mitigated by held-out-patient evaluation, but it
  limits what 30 training patients can teach a model.
* Conformal coverage is approximate for time series.
* There is no external validation, no prospective study, and no fairness audit across demographic groups (the
  synthetic cohort is too small).
* Not a medical device. There are no treatment, triage or dosing recommendations.

See `docs/PRESENTATION.md` for the pitch, Mermaid diagrams, demo script and judge Q&A.

---

## 9. Using a real dataset

Models trained on the synthetic cohort should **not** be applied to real patients. Instead, GlucoTwin
**retrains from scratch** on whatever real data you import. Real data is stored separately in `data/real/`, so the
synthetic demo keeps working.

```powershell
# 1. Import (pick one)
python import_data.py --format ohio --input "C:\data\OhioT1DM"     # folder of OhioT1DM *.xml files
python import_data.py --format csv  --input "C:\data\my_study"     # folder with timeseries.csv (+ optional EHR CSVs)
python import_data.py --format csv  --input export.csv --map timestamp="Device Time" --map glucose_mgdl="Sensor Glucose"

# 2. Train and evaluate on it (prints the same metrics, ablation and risk tables)
python train.py --source real

# 3. Explore it: the dashboard sidebar now shows a "Data source" switch (Synthetic / Imported real)
streamlit run app.py

# Later: add more recordings. The fingerprint changes, so models retrain automatically.
python import_data.py --format csv --input next_month.csv --append
python import_data.py --report
```

**CSV format.** The only required columns are a patient ID (optional for one-patient files, where the file name is
used), a timestamp and glucose. Common names are recognised automatically (e.g. `Glucose Value (mg/dL)`, `sgv`,
`bg`, `Historic Glucose mmol/L`, `Subject`, `Timestamp (YYYY-MM-DDThh:mm:ss)`); otherwise use `--map`.
* **Units:** mmol/L is detected (column name or median < 35) and converted to mg/dL.
* **Optional signals:** heart rate, steps, sleep, carbs, insulin.
* **Optional EHR files:** `patients.csv` (age, sex, diabetes type, BMI, diabetes duration, prior hypo events,
  HbA1c), `conditions.csv`, `medications.csv`, `labs.csv`.

**What adapts automatically**
* **Missing signals or EHR fields:** they become missing values, never zeros, and features that are unavailable
  for the whole dataset are dropped from the models. OhioT1DM, for example, has no age, BMI or HbA1c.
* **Different calendar dates per patient:** every patient gets their own chronological cut-off.
* **Small cohorts:** with fewer than 10 patients, `auto` switches to a **within-patient time split** (first 65% to
  fit, next 15% to calibrate, last 20% to test, with a 2-hour embargo at each boundary). This measures
  personalised forecasting on future data, not generalisation to new patients, and the dashboard says so. Force a
  mode with `--split patient_time|time_only`.
* **Too little data:** training stops with a clear message if any split has fewer than 200 usable rows.
* **Rare events:** risk models refuse to output probabilities when there are too few events to learn from.

**Tested without real data.** The importer tests (`tests/test_importers.py`) write synthetic patients in the
OhioT1DM XML layout (different dates, two patients without a wristband, misaligned timestamps, sleep intervals
with begin > end) and in a Dexcom-style mmol/L CSV, then run import → train → twin end to end. The OhioT1DM parser
follows the published XML element names (`glucose_level`, `basis_heart_rate`, `basis_steps`, `meal`, `bolus`,
`sleep`, `basis_sleep`). If your copy differs, the CSV route always works.

**Honesty notes for real data.** Retrospective metrics on a small dataset are not clinical validation. Report the
split mode alongside every number, and do not compare time-split results with held-out-patient results.

---

## 10. Data: GlucoFM-Bench (external CGM benchmark)

[GlucoFM-Bench](https://huggingface.co/datasets/glucofmbench/GlucoFM-Bench) is a public CGM blood-glucose
forecasting benchmark that pools **11 open cohorts** (AZT1D, BIG IDEA Lab, Bris-T1D Open, CGMacros, Colas2019,
D1NAMO, HUPA-UCM, Hall2018, ShanghaiT1DM, ShanghaiT2DM, T1DM-UOM): 529 subjects, about 2.9 million readings. Each row
is one subject with `dataset`, `subject_id`, `timestamp` (list of Unix seconds) and `BGvalue` (list, mg/dL), already
resampled to 5 minutes. Gaps of more than 1 hour inside a list are sequence boundaries.

```powershell
python download_glucofm.py            # ~18 MB, no login needed; saves data/raw/glucofm/{train,test}.parquet
python download_glucofm.py --force    # re-download
```

```python
from src.glucofm import load_split, to_long
train = load_split("train", cohorts=["Hall2018", "Colas2019"])   # cohorts= is optional
long = to_long(train, mask_implausible=True)  # columns: dataset, subject_id, timestamp, glucose_mg_dl
```

* **Split:** the downloaded `train`/`test` is a **chronological split within each subject** (test = the last ~20% of
  every subject's readings), even though the dataset card describes a split by participant. Results on it measure
  forecasting into each person's future, not generalisation to new people.
* **Known artefact:** 7 training readings of 1.8 mg/dL (one T1DM-UOM subject). `mask_implausible=True` sets values
  outside 20–600 mg/dL to NaN.
* The data lives in `data/raw/`, which is git-ignored, and is separate from the synthetic cohort and `data/real/`.
  `tests/test_glucofm.py` runs sanity checks once it has been downloaded (otherwise those tests are skipped).
* **Licence:** CC BY-NC-SA 4.0, **non-commercial use only**, plus the licence of each source cohort. If you use it,
  cite GlucoFM-Bench **and the original cohort publications** listed on the dataset page.
