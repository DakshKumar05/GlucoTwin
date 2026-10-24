"""Evaluation on held-out patients after the chronological cut-off.

Every number here is computed from model predictions at run time - nothing is hard-coded.
Synthetic-data results do NOT establish real-world clinical performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score

from . import config as C
from .features import CONFIG_LABELS, MODEL_CONFIGS
from .forecasting import FAMILIES, FAMILY_LABELS, ForecastBundle, SplitInfo
from .risk_prediction import EVENTS


def regression_metrics(y_true, y_pred) -> dict:
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2))), n=int(len(err)))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


def classification_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    yhat = (p >= threshold).astype(int)
    has_both = 0 < y.sum() < len(y)
    return dict(
        n=int(len(y)), positives=int(y.sum()), base_rate=float(y.mean()) if len(y) else np.nan,
        threshold=float(threshold),
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall=float(recall_score(y, yhat, zero_division=0)),
        F1=float(f1_score(y, yhat, zero_division=0)),
        PR_AUC=float(average_precision_score(y, p)) if has_both else np.nan,
        Brier=float(brier_score_loss(y, p)) if len(y) else np.nan,
        ECE=expected_calibration_error(y, p) if len(y) else np.nan,
    )


def block_bootstrap_mae_diff(err_a, err_b, groups, n_resamples: int = C.BOOTSTRAP_RESAMPLES, seed: int = C.SEED):
    """Paired bootstrap of MAE(a) - MAE(b), resampling whole (patient, day) blocks to respect
    autocorrelation. Negative values mean model a has lower error."""
    d = np.abs(np.asarray(err_a, float)) - np.abs(np.asarray(err_b, float))
    frame = pd.DataFrame({"d": d, "g": np.asarray(groups)})
    agg = frame.groupby("g")["d"].agg(["sum", "size"])
    sums, sizes = agg["sum"].to_numpy(), agg["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_resamples, len(sums)))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def evaluate(df: pd.DataFrame, split: SplitInfo, forecast: ForecastBundle, risk_models: dict) -> dict:
    _, _, test_mask = split.masks(df)
    test = df.loc[test_mask & df["g_now"].notna()]
    results: dict = {}

    # --- forecasting: identical rows for every model at a given horizon ---
    rows, preds_store = [], {}
    for h in forecast.horizons:
        t = test[test[f"target_glucose_{h}"].notna()]
        y = t[f"target_glucose_{h}"].to_numpy()
        rows.append(dict(config="persistence", config_label="0 - Persistence (last value)", family="naive",
                         family_label="Naive", horizon_min=h, selected=False,
                         **regression_metrics(y, t["g_now"].to_numpy())))
        preds_store[("persistence", h)] = t["g_now"].to_numpy()
        for config in MODEL_CONFIGS:
            for family in FAMILIES:
                pred = forecast.predict(t, config, h, family)
                q = forecast.interval(config, h, family)
                coverage = float(np.mean(np.abs(y - pred) <= q))
                sel = forecast.selected_family[config] == family
                rows.append(dict(config=config, config_label=CONFIG_LABELS[config], family=family,
                                 family_label=FAMILY_LABELS[family], horizon_min=h, selected=sel,
                                 interval_halfwidth=q, interval_coverage=coverage,
                                 calib_MAE=forecast.calib_mae[(config, family, h)],
                                 **regression_metrics(y, pred)))
                if sel:
                    preds_store[(config, h)] = pred
    results["forecast_metrics"] = pd.DataFrame(rows)

    # --- ablation at the primary horizon with block bootstrap CIs ---
    h = C.HORIZON_MINUTES
    t = test[test[f"target_glucose_{h}"].notna()]
    y = t[f"target_glucose_{h}"].to_numpy()
    groups = t["patient_id"].astype(str) + "|" + t["timestamp"].dt.date.astype(str)
    comparisons = [
        ("glucose_only", "persistence", "Learned glucose model vs persistence"),
        ("glucose_wearable", "glucose_only", "+ Wearable & lifestyle signals"),
        ("full_twin", "glucose_wearable", "+ EHR clinical context"),
        ("full_twin", "glucose_only", "Full twin vs glucose only"),
    ]
    abl = []
    for a, b, label in comparisons:
        diff, lo, hi = block_bootstrap_mae_diff(preds_store[(a, h)] - y, preds_store[(b, h)] - y, groups)
        abl.append(dict(comparison=label, model=a, reference=b, mae_difference=diff, ci_low=lo, ci_high=hi,
                        conclusion=("lower error (CI excludes 0)" if hi < 0 else
                                    "higher error (CI excludes 0)" if lo > 0 else "no clear difference (CI includes 0)")))
    results["ablation"] = pd.DataFrame(abl)
    sample = t.sample(n=min(3000, len(t)), random_state=C.SEED).index
    pos = t.index.get_indexer(sample)
    results["scatter"] = pd.DataFrame({
        "actual": y[pos], "full_twin": preds_store[("full_twin", h)][pos],
        "glucose_only": preds_store[("glucose_only", h)][pos], "patient_id": t.loc[sample, "patient_id"].to_numpy(),
    })

    # --- risk classification ---
    crow, calib = [], {}
    for (event, config), rm in risk_models.items():
        spec = EVENTS[event]
        te = test[test[spec["eligible"]] & test[spec["label"]].notna()]
        base = dict(event=event, event_name=spec["name"], config=config, config_label=CONFIG_LABELS[config],
                    status=rm.status, message=rm.message, **{f"train_{k}": v for k, v in rm.stats.items()})
        if not rm.available or len(te) == 0:
            crow.append(base)
            continue
        yy = te[spec["label"]].astype(int).to_numpy()
        pp = rm.predict_proba(te)
        m = classification_metrics(yy, pp, rm.threshold)
        if m["positives"] < C.MIN_CALIB_EVENTS:
            base["message"] = rm.message + f" Only {m['positives']} positive test windows: metrics are unreliable."
        crow.append({**base, **m})
        calib[(event, config)] = calibration_table(yy, pp)
    results["risk_metrics"] = pd.DataFrame(crow)
    results["calibration"] = calib

    results["split"] = dict(
        mode=split.mode, description=split.describe(),
        train_patients=split.train_patients, test_patients=split.test_patients,
        test_cutoffs=split.test_cutoffs, embargo_minutes=split.embargo_minutes,
        n_test_rows=int(len(test)), n_test_patients=int(test["patient_id"].nunique()),
    )
    return results
