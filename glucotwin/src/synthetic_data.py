"""Synthetic cohort generator for GlucoTwin.

EVERYTHING produced here is SIMULATED. Each synthetic patient has ONE patient_id shared by their
EHR profile, labs, medications, conditions and time-series, so the EHR <-> sensor link is coherent
by construction (it is not a join of unrelated real people).

The glucose simulator is a deliberately simple, interpretable "minimal-model-style" difference
equation, not a validated physiological model (e.g. it is NOT the UVA/Padova simulator).
Its purpose is to produce plausible-looking, internally consistent data so the end-to-end
digital-twin pipeline can be demonstrated without restricted datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config as C

STEPS_PER_DAY = 24 * 60 // C.SAMPLE_MINUTES

# Human-readable list of every modelling assumption, surfaced in the README and the dashboard.
SIMULATION_ASSUMPTIONS = [
    "All patients, records, and sensor streams are synthetic; no real person is represented.",
    "Each synthetic patient's EHR and time-series share the same patient_id (linkage is simulated but coherent).",
    "Each patient's baseline is calibrated (two simulate-measure-shift passes with the same random stream) so the "
    "simulated mean CGM approximately matches the ADAG estimated average glucose for their HbA1c "
    "(eAG = 28.7 x HbA1c - 46.7).",
    "Glucose follows a mean-reverting difference equation: carbohydrate appearance raises it, "
    "insulin action and physical activity lower it, plus random process noise.",
    "Meal carbohydrate absorption and rapid-acting insulin action use gamma-shaped kernels "
    "(peaks ~50 min and ~75 min respectively); these shapes are illustrative, not fitted to data.",
    "Type 1 patients mean-revert more slowly and have larger carbohydrate sensitivity than type 2 patients.",
    "A hidden 'hypoglycaemia propensity' drives both the EHR count of prior hypo events and a "
    "systematic over-estimation of mealtime insulin doses, so that EHR history is informative by design.",
    "Sulfonylurea use adds a small glucose-lowering drift; GLP-1 RA use reduces carbohydrate sensitivity by 15%.",
    "Exercise lowers glucose immediately and raises insulin sensitivity for several hours afterwards.",
    "Short sleep (< 6 h) raises next-day baseline glucose and carbohydrate sensitivity.",
    "A dawn phenomenon raises the baseline between ~03:00 and ~08:00.",
    "When true glucose < 70 mg/dL, patients may self-treat with 15-25 g fast carbs (less often if "
    "hypoglycaemia unawareness is recorded); insulin users may take correction doses when > 250 mg/dL.",
    "Heart rate rises with activity and, unless hypoglycaemia unawareness is recorded, during hypoglycaemia.",
    "CGM readings = true glucose + slow calibration drift + white noise, clipped to 40-400 mg/dL.",
    "Missingness is injected on purpose: CGM dropouts (10-120 min), daily watch charging gaps, "
    "occasional nights without a watch, unlogged meals, and noisy carbohydrate estimates.",
    "Effect sizes were chosen by the authors for plausibility; they are not estimated from any dataset.",
]


def _gamma_kernel(peak_min: float, duration_min: float, shape: float = 3.0) -> np.ndarray:
    """Discrete gamma-shaped kernel (sums to 1) with its mode at `peak_min`."""
    dt = C.SAMPLE_MINUTES
    t = np.arange(dt, duration_min + dt, dt, dtype=float)
    scale = peak_min / (shape - 1.0)
    k = t ** (shape - 1.0) * np.exp(-t / scale)
    return k / k.sum()


MEAL_KERNEL = _gamma_kernel(50, 240)
RESCUE_KERNEL = _gamma_kernel(15, 90)
INSULIN_KERNEL = _gamma_kernel(75, 300)


# ----------------------------------------------------------------------------------------------
# Static EHR profile
# ----------------------------------------------------------------------------------------------
def _make_profile(rng: np.random.Generator, idx: int, start: pd.Timestamp):
    pid = f"SYN-{idx:03d}"
    is_t1d = bool(rng.random() < 0.4)
    age = int(rng.integers(18, 61) if is_t1d else rng.integers(35, 79))
    sex = str(rng.choice(["female", "male"]))
    bmi = round(float(np.clip(rng.normal(25.0, 3.5) if is_t1d else rng.normal(31.0, 5.0), 18.5, 48.0)), 1)
    years_dx = int(np.clip(rng.gamma(2.0, 6.0 if is_t1d else 4.0), 1, age - 2))
    hba1c = round(float(np.clip(rng.normal(7.8 if is_t1d else 7.4, 1.0), 5.8, 11.5)), 1)
    hypo_latent = float(rng.normal())

    meds: list[tuple[str, str]] = []
    if is_t1d:
        pump = bool(rng.random() < 0.4)
        on_insulin = on_bolus = True
        on_metformin = on_glp1 = on_su = False
        if pump:
            meds.append(("Rapid-acting insulin via pump (CSII)", "insulin"))
        else:
            meds += [("Insulin glargine (basal)", "insulin"), ("Insulin lispro (mealtime)", "insulin")]
        delivery = "pump (CSII)" if pump else "pen (MDI)"
    else:
        pump = False
        on_metformin = bool(rng.random() < 0.85)
        on_glp1 = bool(rng.random() < 0.30)
        on_su = bool(rng.random() < 0.20)
        on_insulin = bool(rng.random() < (0.6 if hba1c > 8.5 else 0.3))
        on_bolus = bool(on_insulin and rng.random() < 0.35)
        if on_metformin:
            meds.append(("Metformin", "biguanide"))
        if on_glp1:
            meds.append(("Semaglutide", "GLP-1 receptor agonist"))
        if on_su:
            meds.append(("Gliclazide", "sulfonylurea"))
        if on_insulin:
            meds.append(("Insulin glargine (basal)", "insulin"))
        if on_bolus:
            meds.append(("Insulin aspart (mealtime)", "insulin"))
        delivery = "pen (basal + bolus)" if on_bolus else ("pen (basal only)" if on_insulin else "none")

    has_htn = bool(rng.random() < (0.2 if is_t1d else 0.6))
    has_ckd = bool(rng.random() < (0.05 if is_t1d else float(np.clip(0.05 + 0.004 * (age - 40), 0.05, 0.25))))
    unaware = bool(rng.random() < (0.15 if is_t1d else (0.05 if on_insulin else 0.0)))

    base = 3.0 if is_t1d else (1.2 if on_insulin else (0.8 if on_su else 0.2))
    lam = base * np.exp(0.6 * hypo_latent) * (1 + 0.4 * max(0.0, 7.5 - hba1c)) * (1.8 if unaware else 1.0)
    prior_hypo = int(rng.poisson(lam))

    patient = dict(
        patient_id=pid,
        display_name=f"Synthetic Patient {idx:03d}",
        age=age,
        sex=sex,
        diabetes_type="T1D" if is_t1d else "T2D",
        years_since_diagnosis=years_dx,
        bmi=bmi,
        insulin_delivery=delivery,
        prior_hypo_events_12m=prior_hypo,
        provenance="simulated",
    )

    def onset(years_ago: float) -> str:
        return (start - pd.Timedelta(days=int(years_ago * 365.25))).date().isoformat()

    conditions = [
        dict(patient_id=pid, code_system="SNOMED-CT",
             code="46635009" if is_t1d else "44054006",
             display="Diabetes mellitus type 1" if is_t1d else "Diabetes mellitus type 2",
             onset_date=onset(years_dx))
    ]
    if has_htn:
        conditions.append(dict(patient_id=pid, code_system="SNOMED-CT", code="38341003",
                               display="Hypertensive disorder", onset_date=onset(rng.uniform(1, 15))))
    if has_ckd:
        conditions.append(dict(patient_id=pid, code_system="text", code="",
                               display="Chronic kidney disease, stage 3", onset_date=onset(rng.uniform(0.5, 6))))
    if unaware:
        conditions.append(dict(patient_id=pid, code_system="text", code="",
                               display="Impaired awareness of hypoglycaemia", onset_date=onset(rng.uniform(0.5, 5))))
    if bmi >= 30:
        conditions.append(dict(patient_id=pid, code_system="text", code="",
                               display="Obesity (BMI >= 30)", onset_date=onset(rng.uniform(1, 10))))

    medications = [
        dict(patient_id=pid, medication=name, drug_class=cls,
             start_date=onset(rng.uniform(0.3, max(0.5, years_dx))), status="active")
        for name, cls in meds
    ]

    labs = []
    trend = rng.normal(0.0, 0.3)  # HbA1c change per 6 months
    for months_before in (18, 12, 6, 1):
        value = hba1c if months_before == 1 else hba1c - trend * ((months_before - 1) / 6) + rng.normal(0, 0.15)
        labs.append(dict(patient_id=pid, date=(start - pd.DateOffset(months=months_before)).date().isoformat(),
                         test="HbA1c", loinc="4548-4", value=round(float(np.clip(value, 5.0, 13.0)), 1), unit="%"))
    egfr = rng.uniform(35, 59) if has_ckd else float(np.clip(rng.uniform(80, 115) - 0.4 * (age - 40), 62, 120))
    labs.append(dict(patient_id=pid, date=(start - pd.DateOffset(months=2)).date().isoformat(),
                     test="eGFR", loinc="", value=round(float(egfr)), unit="mL/min/1.73m2"))
    labs.append(dict(patient_id=pid, date=(start - pd.DateOffset(months=2)).date().isoformat(),
                     test="LDL cholesterol", loinc="", value=round(float(np.clip(rng.normal(100, 25), 40, 200))),
                     unit="mg/dL"))

    eag = 28.7 * hba1c - 46.7
    if is_t1d:
        k = rng.uniform(0.012, 0.020)
        cf = rng.uniform(3.0, 4.5)
    else:
        k = rng.uniform(0.018, 0.028) if on_insulin else rng.uniform(0.028, 0.045)
        cf = rng.uniform(1.8, 3.0) * (1 + 0.015 * (bmi - 25))
    if on_glp1:
        cf *= 0.85
    params = dict(
        patient_id=pid, is_t1d=is_t1d, on_bolus=on_bolus, pump=pump, unaware=unaware,
        eag=eag, gb=eag, k=k, cf=cf,
        icr=rng.uniform(8, 15),
        bolus_bias=float(np.clip(0.06 + 0.10 * hypo_latent, -0.15, 0.40)),
        miss_bolus_p=rng.uniform(0.03, 0.10),
        e_sens=rng.uniform(0.8, 1.4) if is_t1d else rng.uniform(0.4, 0.9),
        dawn=rng.uniform(5, 25),
        noise_sd=rng.uniform(1.2, 2.0) if is_t1d else rng.uniform(0.7, 1.2),
        su_drift=0.25 if on_su else 0.0,
        hr_rest=float(rng.normal(66, 6) + 0.4 * (bmi - 25) + (0 if is_t1d else 3)),
        base_steps=rng.uniform(25, 45) if is_t1d else rng.uniform(15, 35),
        p_rescue=0.12 if unaware else 0.35,
        log_meal_p=0.9 if on_bolus else 0.6,
        log_insulin_p=1.0 if pump else 0.85,
        hypo_latent=hypo_latent,
    )
    return patient, conditions, medications, labs, params


# ----------------------------------------------------------------------------------------------
# Dynamic time-series
# ----------------------------------------------------------------------------------------------
def _simulate_timeseries(rng: np.random.Generator, p: dict, start: pd.Timestamp, n_days: int):
    n = n_days * STEPS_PER_DAY
    dt = C.SAMPLE_MINUTES
    idx = np.arange(n)
    minute_of_day = (idx % STEPS_PER_DAY) * dt

    # --- sleep schedule ---
    asleep = np.zeros(n, dtype=bool)
    cf_mult = np.ones(n)
    gb_shift = np.zeros(n)
    prev_bed = -int(rng.integers(60, 96))  # fell asleep before the window started
    for d in range(n_days):
        wake = d * STEPS_PER_DAY + int(round((390 + rng.normal(0, 40)) / dt))
        asleep[max(prev_bed, 0):wake] = True
        if (wake - prev_bed) * dt / 60 < 6.0:          # short night -> worse next-day control
            cf_mult[wake:wake + STEPS_PER_DAY] = 1.15
            gb_shift[wake:wake + STEPS_PER_DAY] = 8.0
        bed_min = 1380 + rng.normal(0, 50) + (rng.uniform(90, 150) if rng.random() < 0.15 else 0)
        prev_bed = d * STEPS_PER_DAY + int(round(bed_min / dt))
    asleep[min(prev_bed, n):] = True

    # --- activity ---
    steps_true = np.where(asleep, 0.0, rng.gamma(1.5, p["base_steps"] / 1.5, size=n))
    for d in range(n_days):
        if rng.random() < (0.5 if p["is_t1d"] else 0.4):
            s0 = d * STEPS_PER_DAY + int(rng.integers(7 * 12, 20 * 12))
            dur = int(rng.integers(6, 16))
            steps_true[s0:s0 + dur] = rng.uniform(450, 650, size=len(steps_true[s0:s0 + dur]))
    steps_true = np.round(steps_true)

    # --- meals, logs and boluses ---
    carbs_true = np.zeros(n)
    carbs_logged = np.zeros(n)
    bolus = np.zeros(n)
    insulin_logged = np.zeros(n)
    meal_plan = [(0.85, 450, 40, 40, 12), (0.92, 765, 50, 60, 18), (0.95, 1140, 50, 70, 22)]
    for d in range(n_days):
        meals = [(m, sd, g, gsd) for prob, m, sd, g, gsd in meal_plan if rng.random() < prob]
        meals += [(rng.uniform(540, 1320), 0, rng.uniform(15, 30), 3) for _ in range(rng.poisson(1.0))]
        for mean_min, sd_min, g_mean, g_sd in meals:
            t = d * STEPS_PER_DAY + int(np.clip(round((mean_min + rng.normal(0, sd_min)) / dt), 0, STEPS_PER_DAY - 1))
            grams = max(10.0, rng.normal(g_mean, g_sd))
            carbs_true[t] += grams
            if rng.random() < p["log_meal_p"]:
                carbs_logged[t] += 5 * round(grams * rng.lognormal(0, 0.25) / 5)
            if p["on_bolus"] and rng.random() > p["miss_bolus_p"]:
                units = round(2 * grams * rng.lognormal(0, 0.3) / p["icr"]) / 2
                tb = min(n - 1, t + int(rng.integers(0, 4)))
                bolus[tb] += units
                if rng.random() < p["log_insulin_p"]:
                    insulin_logged[tb] += units

    isf = p["cf"] * p["icr"] * (1 + p["bolus_bias"])            # mg/dL drop per unit (total)
    meal_ra = np.convolve(carbs_true * cf_mult, MEAL_KERNEL)[:n] * p["cf"]
    ins_eff = np.convolve(bolus, INSULIN_KERNEL)[:n] * isf
    dawn_curve = np.where((minute_of_day >= 180) & (minute_of_day <= 480),
                          np.sin(np.pi * (minute_of_day - 180) / 300), 0.0)

    # --- state-dependent simulation loop ---
    pad = len(INSULIN_KERNEL) + 2
    extra = np.zeros(n + pad)
    true_g = np.zeros(n)
    rescue = np.zeros(n)
    g = p["gb"] + rng.normal(0, 10)
    act = 0.0
    decay = 0.5 ** (1 / 36)  # activity memory half-life 3 h
    last_rescue, last_corr = -999, -999
    for t in range(n):
        s = steps_true[t] / 600.0
        act = act * decay + s
        sens = min(1.3, 1 + 0.02 * act)
        gb_t = p["gb"] + gb_shift[t] + p["dawn"] * dawn_curve[t]
        dg = (-p["k"] * (g - gb_t) + meal_ra[t] - ins_eff[t] * sens + extra[t]
              - p["e_sens"] * (1.5 * s + 0.03 * act) - p["su_drift"] + rng.normal(0, p["noise_sd"]))
        g = float(np.clip(g + dg, 39.0, 450.0))
        true_g[t] = g
        if g < 70 and t - last_rescue >= 4 and rng.random() < p["p_rescue"]:
            grams = rng.uniform(15, 25)
            last_rescue = t
            extra[t + 1:t + 1 + len(RESCUE_KERNEL)] += grams * p["cf"] * RESCUE_KERNEL
            rescue[t] += grams
            if rng.random() < 0.5:
                carbs_logged[t] += round(grams)
        if p["on_bolus"] and g > 250 and t - last_corr >= 36 and rng.random() < 0.05:
            units = max(0.5, round(2 * (g - 150) / isf) / 2)
            last_corr = t
            extra[t + 1:t + 1 + len(INSULIN_KERNEL)] -= units * isf * INSULIN_KERNEL
            bolus[t] += units
            if rng.random() < p["log_insulin_p"]:
                insulin_logged[t] += units

    # --- wearable heart rate ---
    act_mem = lfilter([1.0], [1.0, -decay], steps_true / 600.0)
    hr_noise = lfilter([1.0], [1.0, -0.8], rng.normal(0, 2.0, size=n))
    hr = (p["hr_rest"] + 28 * np.minimum(steps_true / 600.0, 1.2) + 6 * np.minimum(act_mem / 12, 1.0)
          - 7 * asleep + 3 * (cf_mult > 1) + hr_noise)
    if not p["unaware"]:
        hr = hr + np.where(true_g < 70, 8 + 0.3 * (70 - true_g), 0.0)
    hr = np.round(hr)

    # --- CGM measurement ---
    drift = lfilter([1.0], [1.0, -0.995], rng.normal(0, 0.4, size=n))
    cgm = np.clip(np.round(true_g + drift + rng.normal(0, 5, size=n)), 40, 400)

    # --- injected missingness ---
    cgm = cgm.astype(float)
    hr = hr.astype(float)
    steps_obs = steps_true.astype(float)
    asleep_obs = asleep.astype(float)
    for d in range(n_days):
        for _ in range(rng.poisson(0.8)):
            length = int(rng.integers(2, 7) if rng.random() < 0.7 else rng.integers(7, 25))
            s0 = d * STEPS_PER_DAY + int(rng.integers(0, STEPS_PER_DAY))
            cgm[s0:s0 + length] = np.nan
        if rng.random() < 0.85:  # watch charging in the evening
            s0 = d * STEPS_PER_DAY + int(rng.integers(18 * 12, 21 * 12 + 6))
            length = int(rng.integers(9, 16))
            for arr in (hr, steps_obs, asleep_obs):
                arr[s0:s0 + length] = np.nan
        if rng.random() < 0.05:  # watch not worn overnight
            s0 = d * STEPS_PER_DAY
            for arr in (hr, steps_obs, asleep_obs):
                arr[s0:s0 + 7 * 12] = np.nan

    ts = start + pd.to_timedelta(idx * dt, unit="min")
    series = pd.DataFrame(dict(
        patient_id=p["patient_id"], timestamp=ts, glucose_mgdl=cgm, heart_rate_bpm=hr, steps=steps_obs,
        is_asleep=asleep_obs, carbs_logged_g=carbs_logged, insulin_logged_u=insulin_logged,
        provenance="simulated",
    ))
    truth = pd.DataFrame(dict(
        patient_id=p["patient_id"], timestamp=ts, true_glucose=np.round(true_g, 1),
        true_carbs_g=np.round(carbs_true + rescue, 1), rescue_carbs_g=np.round(rescue, 1),
        true_insulin_u=bolus, true_steps=steps_true,
    ))
    return series, truth


def generate_cohort(n_patients: int = C.N_PATIENTS, n_days: int = C.N_DAYS, seed: int = C.SEED) -> dict:
    """Generate a coherent synthetic cohort. Returns a dict of DataFrames keyed by table name."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(C.START_TIME)
    patients, conditions, medications, labs, params, series, truth = [], [], [], [], [], [], []
    for i in range(1, n_patients + 1):
        pat, cond, meds, lab, par = _make_profile(rng, i, start)
        child_seed = int(rng.integers(0, 2**32 - 1))
        # Calibration pass: same random stream, then shift the baseline so mean CGM ~= eAG.
        for _ in range(2):
            ts, _ = _simulate_timeseries(np.random.default_rng(child_seed), par, start, n_days)
            par["gb"] += par["eag"] - float(np.nanmean(ts["glucose_mgdl"]))
        ts, tr = _simulate_timeseries(np.random.default_rng(child_seed), par, start, n_days)
        patients.append(pat)
        conditions += cond
        medications += meds
        labs += lab
        params.append(par)
        series.append(ts)
        truth.append(tr)
    return dict(
        patients=pd.DataFrame(patients),
        conditions=pd.DataFrame(conditions),
        medications=pd.DataFrame(medications, columns=["patient_id", "medication", "drug_class", "start_date", "status"]),
        labs=pd.DataFrame(labs),
        timeseries=pd.concat(series, ignore_index=True),
        simulation_truth=pd.concat(truth, ignore_index=True),
        simulation_params=pd.DataFrame(params),
    )
