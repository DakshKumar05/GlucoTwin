"""The patient digital twin.

A PatientTwin is a virtual representation of ONE synthetic patient that combines:
  * a static/historical EHR layer (profile, conditions, medications, labs -> static features), and
  * a dynamic layer rebuilt from the sensor stream every time the twin clock advances.

Each call to `state()` re-reads only the observations available up to the twin clock (`cursor`),
recomputes features, and re-runs the forecast and risk models. This is what makes the twin "evolve":
new observations change the state, the forecast, the risk estimate and the explanation.
Data after the cursor is never visible to the models; `revealed_future()` exists only so the
replay can show, afterwards, what actually happened.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .explainability import contributions, group_totals, top_contributors
from .features import dynamic_features
from .risk_prediction import EVENTS, risk_category


def trend_arrow(rate: float | None) -> str:
    """CGM-style trend arrow from the 15-min rate of change (mg/dL per minute)."""
    if rate is None or np.isnan(rate):
        return "?"
    for bound, arrow in ((3, "⇈"), (2, "↑"), (1, "↗"), (-1, "→"), (-2, "↘"), (-3, "↓")):
        if rate >= bound:
            return arrow
    return "⇊"


@dataclass
class TwinState:
    patient_id: str
    cursor: pd.Timestamp
    config: str
    glucose_now: float | None = None
    glucose_now_imputed: bool = False
    latest_obs_time: pd.Timestamp | None = None
    minutes_since_obs: float | None = None
    trend_rate: float | None = None
    trend: str = "?"
    can_forecast: bool = False
    reason: str = ""
    forecast: pd.DataFrame | None = None
    risk: dict = field(default_factory=dict)
    contributions: pd.DataFrame | None = None
    group_contributions: pd.Series | None = None
    explain_method: str = ""
    risk_contributions: pd.DataFrame | None = None
    risk_explain_event: str | None = None
    risk_explain_method: str = ""
    summary_24h: dict = field(default_factory=dict)
    wearable: dict = field(default_factory=dict)
    missing_features: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    features: pd.Series | None = None


class PatientTwin:
    def __init__(self, patient_id: str, grid: pd.DataFrame, static: pd.DataFrame, tables: dict, artifacts: dict):
        self.patient_id = patient_id
        self.grid = grid[grid["patient_id"] == patient_id].sort_values("timestamp").reset_index(drop=True)
        if self.grid.empty:
            raise ValueError(f"No time-series data for {patient_id}")
        self.static_row = static.loc[patient_id]
        self.profile = tables["patients"].set_index("patient_id").loc[patient_id]
        self.conditions = tables["conditions"].query("patient_id == @patient_id")
        self.medications = tables["medications"].query("patient_id == @patient_id")
        self.labs = tables["labs"].query("patient_id == @patient_id").sort_values("date")
        self.forecaster = artifacts["forecast"]
        self.risk_models = artifacts["risk"]
        self.cursor = self.grid["timestamp"].iloc[0]

    # ---- clock -------------------------------------------------------------------------------
    @property
    def first_time(self) -> pd.Timestamp:
        return self.grid["timestamp"].iloc[0]

    @property
    def last_time(self) -> pd.Timestamp:
        return self.grid["timestamp"].iloc[-1]

    def observe_until(self, t) -> None:
        """Advance (or rewind) the twin clock; only data at or before `t` is visible."""
        self.cursor = min(max(pd.Timestamp(t), self.first_time), self.last_time)

    def observed(self, hours: float | None = None) -> pd.DataFrame:
        m = self.grid["timestamp"] <= self.cursor
        if hours is not None:
            m &= self.grid["timestamp"] > self.cursor - pd.Timedelta(hours=hours)
        return self.grid.loc[m]

    def revealed_future(self, hours: float = 2) -> pd.DataFrame:
        """Ground truth AFTER the cursor - for retrospective replay display only, never for models."""
        m = (self.grid["timestamp"] > self.cursor) & (self.grid["timestamp"] <= self.cursor + pd.Timedelta(hours=hours))
        return self.grid.loc[m]

    # ---- features ----------------------------------------------------------------------------
    def feature_row(self, scenario: dict | None = None) -> pd.DataFrame:
        hist = self.grid[self.grid["timestamp"] <= self.cursor].tail(C.TWIN_HISTORY_STEPS).reset_index(drop=True).copy()
        if scenario:
            if scenario.get("carbs_g") and "carbs_logged_g" in hist:
                hist.loc[hist.index[-1], "carbs_logged_g"] += float(scenario["carbs_g"])
            if scenario.get("walk_steps_per_5min") and "steps" in hist:
                last6 = hist.index[-C.steps(30):]
                hist.loc[last6, "steps"] = hist.loc[last6, "steps"].fillna(0) + float(scenario["walk_steps_per_5min"])
        row = dynamic_features(hist).iloc[[-1]].copy()
        for col, val in self.static_row.items():
            row[col] = val
        return row

    # ---- state -------------------------------------------------------------------------------
    def state(self, config: str = "full_twin", scenario: dict | None = None, explain: bool = True) -> TwinState:
        st = TwinState(patient_id=self.patient_id, cursor=self.cursor, config=config)
        obs = self.observed()
        real = obs[obs["glucose_observed"]]
        if not real.empty:
            st.latest_obs_time = real["timestamp"].iloc[-1]
            st.minutes_since_obs = (self.cursor - st.latest_obs_time).total_seconds() / 60
        X = self.feature_row(scenario)
        xs = X.iloc[0]
        st.features = xs
        g_now = xs.get("g_now", np.nan)
        st.glucose_now = None if pd.isna(g_now) else float(g_now)
        st.glucose_now_imputed = bool(obs["glucose_imputed"].iloc[-1]) if len(obs) else False
        st.trend_rate = None if pd.isna(xs.get("g_delta_15", np.nan)) else float(xs["g_delta_15"]) / 15
        st.trend = trend_arrow(st.trend_rate)

        feats = self.forecaster.features[config]
        st.missing_features = [f for f in feats if pd.isna(xs.get(f, np.nan))]
        st.summary_24h = self._summary(self.observed(24))
        st.wearable = {k: (None if pd.isna(xs.get(k, np.nan)) else float(xs[k])) for k in
                       ("hr_now", "hr_mean_30m", "steps_30m", "steps_2h", "sleep_24h_h", "carbs_2h",
                        "carbs_on_board", "insulin_2h", "insulin_on_board", "asleep_now")}
        last24 = self.observed(24)
        st.quality = dict(
            pct_glucose_imputed_24h=100 * last24["glucose_imputed"].mean() if len(last24) else np.nan,
            pct_glucose_missing_24h=100 * last24["glucose"].isna().mean() if len(last24) else np.nan,
            pct_hr_missing_24h=100 * last24["heart_rate_bpm"].isna().mean() if "heart_rate_bpm" in last24 and len(last24) else np.nan,
        )

        if st.glucose_now is None:
            st.reason = "No current glucose value (sensor gap longer than the imputation limit)."
            return st
        if st.minutes_since_obs is None or st.minutes_since_obs > C.STALE_AFTER_MINUTES:
            st.reason = (f"Latest real CGM reading is {st.minutes_since_obs:.0f} min old "
                         f"(> {C.STALE_AFTER_MINUTES} min): forecast withheld." if st.minutes_since_obs is not None
                         else "No real CGM reading yet.")
            return st
        st.can_forecast = True

        rows = [dict(horizon_min=0, time=self.cursor, predicted=st.glucose_now, lower=st.glucose_now, upper=st.glucose_now)]
        for h in self.forecaster.horizons:
            pred = float(self.forecaster.predict(X, config, h)[0])
            q = self.forecaster.interval(config, h)
            rows.append(dict(horizon_min=h, time=self.cursor + pd.Timedelta(minutes=h), predicted=pred,
                             lower=max(40.0, pred - q), upper=min(400.0, pred + q)))
        st.forecast = pd.DataFrame(rows)

        for event, spec in EVENTS.items():
            rm = self.risk_models[(event, config)]
            beyond = (st.glucose_now < C.HYPO_THRESHOLD) if event == "hypo" else (st.glucose_now > C.HYPER_THRESHOLD)
            info = dict(name=spec["name"], status=rm.status, message=rm.message, threshold=rm.threshold,
                        probability=None, category="Unavailable", alert=False, currently_beyond=bool(beyond))
            if beyond:
                info["category"] = "Currently beyond threshold"
            elif rm.available:
                p = float(rm.predict_proba(X)[0])
                info.update(probability=p, category=risk_category(p), alert=p >= rm.threshold)
            st.risk[event] = info

        if explain:
            fam = self.forecaster.selected_family[config]
            model = self.forecaster.models[(config, fam, C.HORIZON_MINUTES)]
            contrib, st.explain_method = contributions(model, X[feats], self.forecaster.medians[config])
            st.contributions = top_contributors(contrib.iloc[0], xs, n=10)
            st.group_contributions = group_totals(contrib.iloc[0])
            candidates = [(e, i["probability"]) for e, i in st.risk.items() if i["probability"] is not None]
            if candidates:
                event = max(candidates, key=lambda c: c[1])[0]
                rm = self.risk_models[(event, config)]
                rc, st.risk_explain_method = contributions(rm.model, X[rm.features], classifier=True)
                st.risk_contributions = top_contributors(rc.iloc[0], xs, n=8)
                st.risk_explain_event = event
        return st

    @staticmethod
    def _summary(window: pd.DataFrame) -> dict:
        g = window["glucose"].dropna()
        if len(g) < 12:
            return {}
        mean = float(g.mean())
        lo, hi = C.TARGET_RANGE
        return dict(
            mean=mean, sd=float(g.std()), cv=float(100 * g.std() / mean),
            tir=float(100 * ((g >= lo) & (g <= hi)).mean()), tbr=float(100 * (g < lo).mean()),
            tar=float(100 * (g > hi).mean()), gmi=3.31 + 0.02392 * mean, n=int(len(g)),
        )
