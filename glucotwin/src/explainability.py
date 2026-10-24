"""Per-prediction explanations.

Methods (labelled in the UI so they are not over-interpreted):
  * XGBoost models  -> TreeSHAP contributions computed by XGBoost itself (`pred_contribs=True`).
  * Ridge models    -> exact linear contributions: coefficient x standardised feature value
                       (relative to the training mean).
  * Other models    -> occlusion sensitivity: change in output when a feature is set to its
                       training median (not additive).
Contributions describe how the MODEL used the inputs. They are associations, not causes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import GROUP_LABELS, describe, feature_group


def _xgb_contribs(model, X: pd.DataFrame) -> pd.DataFrame:
    import xgboost as xgb
    booster = model.get_booster()
    values = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(values[:, :-1], columns=list(X.columns), index=X.index)


def _ridge_contribs(pipe, X: pd.DataFrame) -> pd.DataFrame:
    imputer, scaler, ridge = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1]
    Z = scaler.transform(imputer.transform(X))
    names = list(X.columns)
    if getattr(imputer, "indicator_", None) is not None:
        names += [f"{X.columns[i]} (missing)" for i in imputer.indicator_.features_]
    return pd.DataFrame(Z * ridge.coef_, columns=names, index=X.index)


def _occlusion(predict_fn, X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    base = predict_fn(X)
    out = {}
    for col in X.columns:
        Xo = X.copy()
        Xo[col] = medians.get(col, np.nan)
        out[col] = base - predict_fn(Xo)
    return pd.DataFrame(out, index=X.index)


def contributions(model, X: pd.DataFrame, medians: pd.Series | None = None, classifier: bool = False):
    """Return (contributions DataFrame, method label). Units: mg/dL of predicted change for
    regressors; log-odds for XGBoost classifiers; probability for occlusion on classifiers."""
    if hasattr(model, "get_booster"):
        unit = "log-odds" if classifier else "mg/dL"
        return _xgb_contribs(model, X), f"TreeSHAP (XGBoost pred_contribs), {unit}"
    if hasattr(model, "steps") and model.steps[-1][0] == "ridge":
        return _ridge_contribs(model, X), "Linear contribution (coef x standardised value), mg/dL"
    medians = medians if medians is not None else X.median()
    fn = (lambda Z: model.predict_proba(Z)[:, 1]) if classifier else model.predict
    return _occlusion(fn, X, medians), "Occlusion sensitivity vs training median (not additive)"


def top_contributors(contrib_row: pd.Series, X_row: pd.Series, n: int = 10) -> pd.DataFrame:
    order = contrib_row.abs().sort_values(ascending=False).index[:n]
    rows = []
    for name in order:
        base = name.replace(" (missing)", "")
        value = X_row.get(base, np.nan)
        rows.append(dict(
            feature=name, description=describe(name), group=GROUP_LABELS.get(feature_group(name), "Other"),
            value=("missing" if name.endswith("(missing)") or pd.isna(value) else round(float(value), 2)),
            contribution=float(contrib_row[name]),
        ))
    return pd.DataFrame(rows)


def group_totals(contrib_row: pd.Series) -> pd.Series:
    groups = contrib_row.groupby(lambda c: GROUP_LABELS.get(feature_group(c), "Other"))
    return groups.sum()


def global_importance(model, X: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, str]:
    contrib, method = contributions(model, X, medians)
    imp = contrib.abs().mean().sort_values(ascending=False)
    df = pd.DataFrame({"feature": imp.index, "mean_abs_contribution": imp.values})
    df["description"] = df["feature"].map(describe)
    df["group"] = df["feature"].map(lambda c: GROUP_LABELS.get(feature_group(c), "Other"))
    return df, method
