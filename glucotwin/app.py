"""GlucoTwin - doctor-facing digital-twin dashboard (research & educational prototype).

Run:  streamlit run app.py
All data shown is SYNTHETIC. Not a medical device. No treatment or dosing recommendations.
"""
from __future__ import annotations

import html
import re
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from src import config as C
from src.data_loader import available_sources, ensure_data, fhir_json, is_synthetic
from src.digital_twin import PatientTwin, TwinState
from src.home import home_available, home_component_dir
from src.journey import JOURNEY_ROOT, journey_available, journey_component_dir
from src.features import CONFIG_LABELS, FEATURE_GROUPS, GROUP_LABELS, MODEL_CONFIGS
from src.forecasting import FAMILY_LABELS
from src.pipeline import build_dataset, load_or_train
from src.risk_prediction import EVENTS, RISK_DEFINITIONS
from src.synthetic_data import SIMULATION_ASSUMPTIONS

st.set_page_config(page_title="GlucoTwin", page_icon=":material/monitor_heart:", layout="wide", initial_sidebar_state="expanded")

# Design tokens (mirror ../styles/tokens.css): dark green-black canvas, single-weight type, hairlines, one lime signal.
INK, PAPER, LICHEN, GRAPHITE, LIME = "#222f30", "#ffffff", "#c9cbbe", "#4d5757", "#cef79e"
ALERT = "#e5484d"    # the one clinical-alert token: hypo threshold, high risk
SERIOUS = "#ec835a"  # status "serious": hyper threshold, elevated risk, warnings
SANS, MONO = "Aspekta, sans-serif", "Roboto Mono, monospace"
# Chart roles: system inks for single series; the data-source groups use a 3-slot categorical palette validated on
# INK (dataviz validator: all-pairs CVD dE >= 9.4, contrast >= 3:1). Graphite is 1.86:1 on INK, so hairlines only.
COL = dict(obs=PAPER, imputed=LICHEN, forecast=LICHEN, band="rgba(201,203,190,0.12)",
           actual=PAPER, hypo=ALERT, hyper=SERIOUS, target="rgba(255,255,255,0.05)",
           glucose="#3987e5", wearable="#d95926", ehr="#199e70", other=LICHEN)
GROUP_COLORS = {GROUP_LABELS["glucose"]: COL["glucose"], GROUP_LABELS["wearable"]: COL["wearable"],
                GROUP_LABELS["ehr"]: COL["ehr"], "Other": COL["other"]}
CATEGORY_STYLE = {"Low": (LIME, GRAPHITE), "Elevated": (SERIOUS, GRAPHITE), "High": (ALERT, ALERT),  # (dot, border)
                  "Currently beyond threshold": (ALERT, ALERT), "Unavailable": (LICHEN, GRAPHITE)}
PAGES = ["Home", "Overview", "Glucose Forecast", "Clinical Context", "Risk & Explainability", "Model Performance",
         "Twin Timeline", "Patient Journey", "About & Limitations"]
# Short labels for the section pills (the page keys above stay the same internally).
NAV_LABELS = {"Home": "Home", "Overview": "Overview", "Glucose Forecast": "Forecast", "Clinical Context": "Clinical",
              "Risk & Explainability": "Risk & XAI", "Model Performance": "Performance", "Twin Timeline": "Timeline",
              "Patient Journey": "Journey", "About & Limitations": "About"}
# Every dashboard section opens with its full name and a one- or two-line description for a first-time reviewer.
INTROS = {
    "Overview": "Current state of the selected patient's digital twin: latest glucose, the 2-hour forecast, risk and data "
                "quality at a glance.",
    "Glucose Forecast": "Predicts glucose for the next 2 hours with an 80% prediction interval, and flags whether the path "
                        "crosses the low (70 mg/dL) or high (250 mg/dL) threshold.",
    "Clinical Context": "The health-record and wearable context the twin uses: diagnoses, medications, labs, heart rate, "
                        "steps, sleep, meals and insulin.",
    "Risk & Explainability": "Chance of a low or a high in the next 2 hours, and which inputs pushed the forecast up or down.",
    "Model Performance": "How accurate the models are on patients they never saw: error by horizon, alert quality, "
                         "calibration and the most important inputs.",
    "Twin Timeline": "Replays the twin over time so you can compare earlier forecasts with what actually happened.",
    "Patient Journey": "An animated teaching simulation of one ward patient, from admission to the nurse's response.",
    "About & Limitations": "What makes this a digital twin, the simulation assumptions, known limitations, and data export.",
}

st.markdown(f"""
<style>
.block-container {{padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px;}}
/* Streamlit's header bar takes no space and shows nothing; it stays click-through so the button that re-opens a
   collapsed sidebar (which lives inside it) keeps working. */
[data-testid="stHeader"] {{background: transparent; height: 0; min-height: 0; pointer-events: none;}}
[data-testid="stToolbar"], [data-testid="stDecoration"] {{display: none !important;}}
[data-testid="stExpandSidebarButton"] {{pointer-events: auto;}}
/* sidebar starts at the top; its collapse button floats in the corner instead of taking a 60px row */
[data-testid="stSidebarHeader"] {{height: 0; min-height: 0; padding: 0; overflow: visible; position: relative; z-index: 2;}}
[data-testid="stSidebarCollapseButton"] {{position: absolute; top: 10px; right: 10px;}}
[data-testid="stSidebarUserContent"] {{padding-top: 1rem;}}
/* app bar and patient strip */
.gt-appbar {{display: flex; justify-content: space-between; align-items: center; gap: 12px 24px; flex-wrap: wrap;
            padding: 12px 0; border-bottom: 1px solid {GRAPHITE};}}
.gt-appbar .brand {{display: flex; align-items: baseline; gap: 14px; min-width: 0;}}
.gt-appbar .name {{font-size: 28px; line-height: 1; letter-spacing: -0.02em; color: {PAPER}; white-space: nowrap;}}
.gt-appbar .tagline {{font-family: {MONO}; font-size: 12px; letter-spacing: -0.02em; text-transform: uppercase;
                      color: {LICHEN}; white-space: nowrap;}}
.gt-appbar .right {{display: flex; align-items: center; gap: 10px; white-space: nowrap; margin-left: auto; flex: none;}}
@media (min-width: 1400px) {{  /* the full bar needs ~1,405px of window width; narrower windows wrap to two rows */
  .gt-appbar {{flex-wrap: nowrap;}}
  .gt-appbar .brand {{overflow: hidden;}}
  .gt-appbar .tagline {{overflow: hidden; text-overflow: ellipsis;}}
}}
.gt-appbar .clock {{font-family: {MONO}; font-size: 14px; letter-spacing: -0.008em; color: {PAPER}; margin-right: 6px;}}
.gt-appbar .clock span {{font-size: 12px; color: {LICHEN}; text-transform: uppercase; margin-right: 8px;}}
.gt-chip {{display: inline-flex; align-items: center; gap: 6px; font-family: {MONO}; font-size: 12px; line-height: 1;
          letter-spacing: -0.02em; text-transform: uppercase; color: {LICHEN}; border: 1px solid {GRAPHITE};
          border-radius: 9999px; padding: 5px 10px; white-space: nowrap;}}
.gt-chip i {{width: 6px; height: 6px; border-radius: 50%; flex: none;}}
.gt-patient {{display: flex; align-items: center; gap: 8px 10px; flex-wrap: wrap; padding: 4px 0 0;}}
.gt-patient .name {{font-size: 28px; line-height: 1.1; letter-spacing: -0.02em; color: {PAPER}; white-space: nowrap; margin-right: 4px;}}
.gt-patient .pid {{font-family: {MONO}; font-size: 13px; color: {PAPER}; background: #2b3738; border: 1px solid {GRAPHITE};
                  border-radius: 6px; padding: 3px 8px; margin-right: 6px;}}
/* section intro, block headings, hairlines */
.gt-intro {{margin: 0;}}
.gt-intro .t {{font-size: 24px; line-height: 1.2; letter-spacing: -0.006em; color: {PAPER};}}
.gt-intro .d {{font-size: 15px; line-height: 1.4; color: {LICHEN}; max-width: 90ch; margin-top: 6px;}}
.gt-block .t {{font-size: 20px; line-height: 1.2; letter-spacing: -0.006em; color: {PAPER};}}
.gt-block .c {{font-family: {MONO}; font-size: 13px; line-height: 1.45; letter-spacing: -0.02em; color: {LICHEN}; margin-top: 4px;}}
.gt-hr {{border: 0; border-top: 1px solid {GRAPHITE}; margin: 16px 0;}}
.gt-block.prose .c {{font-family: {SANS}; font-size: 15px; letter-spacing: -0.001em;}}
[data-testid="stColumn"] [data-testid="stElementContainer"]:has(.gt-block.prose):not(:first-child) {{margin-top: 8px;}}  /* 16 + 8 = 24px */
.gt-mini {{display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); grid-auto-rows: 1fr; gap: 12px; height: 100%;}}
.gt-mini > div {{border: 1px solid {GRAPHITE}; border-radius: 12px; padding: 14px; display: flex; flex-direction: column;
                justify-content: space-between;}}
.gt-table td.key {{white-space: nowrap;}}
/* Overview: the right column stretches to the left column's height and its last block (the 2x2 cards) takes up the rest,
   so both columns end on the same line instead of leaving an empty block under the shorter one. */
[data-testid="stHorizontalBlock"]:has(.gt-mini) {{align-items: stretch;}}
[data-testid="stColumn"]:has(.gt-mini) {{display: flex; flex-direction: column;}}
[data-testid="stColumn"]:has(.gt-mini) > [data-testid="stVerticalBlock"] {{flex: 1 1 auto;}}
[data-testid="stElementContainer"]:has(.gt-mini) {{flex: 1 1 auto; display: flex; flex-direction: column;}}
[data-testid="stElementContainer"]:has(.gt-mini) *:has(.gt-mini) {{flex: 1 1 auto; display: flex; flex-direction: column;}}
.gt-mini .label {{font-family: {MONO}; font-size: 12px; line-height: 1.3; letter-spacing: -0.02em; text-transform: uppercase; color: {LICHEN};}}
.gt-mini .value {{font-size: 28px; line-height: 1.1; letter-spacing: -0.02em; color: {PAPER}; margin-top: 6px;
                  font-variant-numeric: tabular-nums;}}
/* Streamlit pulls every markdown block up by -1rem; cancel it for our components so every gap is the same 16px */
[data-testid="stMarkdownContainer"]:has(> .gt-intro, > .gt-block, > .gt-grid, > .gt-table-wrap, > .gt-hr, > .gt-stack, > .gt-mini,
  > .gt-appbar, > .gt-patient, > .gt-sgroup, > .gt-snote) {{margin-bottom: 0 !important;}}
/* equal-height card grids */
.gt-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); grid-auto-rows: 1fr; gap: 16px;}}
.gt-grid.two {{grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));}}
.gt-grid.three {{grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));}}
.gt-stack {{display: grid; gap: 12px;}}
.gt-card {{display: flex; flex-direction: column;}}
.gt-card .label {{white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
.gt-card .what {{margin-top: auto; padding-top: 12px; font-size: 13px; line-height: 1.35; color: {LICHEN};}}
/* tables: full width, wrapped text, one row height, scroll inside the box when long */
.gt-table-wrap {{width: 100%; overflow: auto; border: 1px solid {GRAPHITE}; border-radius: 8px;}}
.gt-table {{width: 100%; border-collapse: collapse; font-size: 14px; line-height: 1.35;}}
.gt-table-wrap .gt-table {{margin: 0 !important;}}  /* Streamlit gives markdown tables a 1rem bottom margin */
.gt-table th {{position: sticky; top: 0; background: {INK}; text-align: left; font-family: {MONO}; font-size: 12px;
              letter-spacing: -0.02em; text-transform: uppercase; color: {LICHEN}; padding: 10px 12px;
              border-bottom: 1px solid {GRAPHITE}; vertical-align: bottom;}}
.gt-table td {{padding: 8px 12px; border-bottom: 1px solid {GRAPHITE}; vertical-align: top; color: {PAPER};}}
.gt-table tr:last-child td {{border-bottom: 0;}}
.gt-table .num {{text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;}}
/* sidebar control panel */
.gt-sgroup {{margin-top: 24px;}}
.gt-sgroup.first {{margin-top: 0;}}
.gt-sgroup .gt-label {{font-size: 12px; color: {PAPER};}}
.gt-sgroup .cap {{font-size: 13px; line-height: 1.35; color: {LICHEN}; margin-top: 4px;}}
.gt-snote {{font-family: {MONO}; font-size: 12px; line-height: 1.45; letter-spacing: -0.02em; color: {LICHEN};
           border-top: 1px solid {GRAPHITE}; padding-top: 12px; margin-top: 12px;}}
h1, h2, h3 {{letter-spacing: -0.012em;}}
h4, h5 {{letter-spacing: -0.006em;}}
/* mono voice: labels, captions, metric labels, buttons, section pills, badges */
.gt-label, .gt-card .label, [data-testid="stMetricLabel"] p, [data-testid="stCaptionContainer"],
.stButton button p, .stDownloadButton button p, [data-testid="stButtonGroup"] button {{font-family: {MONO}; letter-spacing: -0.02em;}}
.gt-label, .gt-card .label, [data-testid="stMetricLabel"] p {{font-size: 13px; text-transform: uppercase; color: {LICHEN};}}
[data-testid="stCaptionContainer"] {{color: {LICHEN}; font-size: 13px;}}
.stButton button p, .stDownloadButton button p {{font-size: 14px; text-transform: uppercase;}}
/* filled action on a dark canvas = Paper with Ink text (lime never fills a button) */
[data-testid="stBaseButton-primary"] {{background: {PAPER}; border-color: {PAPER}; color: {INK};}}
[data-testid="stBaseButton-primary"]:hover {{background: {LICHEN}; border-color: {LICHEN}; color: {INK};}}
[data-testid="stBaseButton-primary"] p {{color: {INK};}}
/* section navigation: outlined pills, the active one fills lime */
[data-testid="stButtonGroup"] button {{border-radius: 12px !important; font-size: 13px; color: {LICHEN};}}
[data-testid="stButtonGroup"] button * {{font-family: {MONO}; font-size: 13px; letter-spacing: -0.02em;}}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{background: {LIME} !important; border-color: {LIME} !important;
                                                            color: {INK} !important;}}
[data-testid="stButtonGroup"] button[aria-checked="true"] * {{color: {INK} !important;}}
strong, b, th {{font-weight: 400 !important;}}
strong, b {{color: {PAPER};}}
[data-testid="stSliderThumbValue"] {{color: {PAPER};}}
[data-testid="stSidebar"] .stButton button {{padding-left: 8px; padding-right: 8px;}}
[data-testid="stSidebar"] .stButton button p {{font-size: 13px;}}
.gt-card {{border: 1px solid {GRAPHITE}; border-radius: 16px; padding: 20px; height: 100%;}}
.gt-card .value {{font-size: 42px; line-height: 1.05; letter-spacing: -0.02em; color: {PAPER}; margin: 12px 0 8px;
                  font-variant-numeric: tabular-nums;}}
.gt-card .sub, .gt-note {{font-size: 14px; line-height: 1.35; color: {LICHEN};}}
.gt-risk {{border: 1px solid {GRAPHITE}; border-radius: 16px; padding: 20px; height: 100%;}}
.gt-risk .cat {{display: flex; align-items: center; gap: 10px; margin: 10px 0 6px; font-size: 24px; letter-spacing: -0.006em;}}
.gt-dot {{flex: none; width: 6px; height: 6px; border-radius: 50%;}}
.gt-clock {{font-family: {MONO}; font-size: 15px; letter-spacing: -0.007em; margin: 4px 0 10px;}}
.gt-disclaimer {{font-family: {MONO}; font-size: 13px; letter-spacing: -0.02em; line-height: 1.5; color: {LICHEN};
                 border-top: 1px solid {GRAPHITE}; margin-top: 3rem; padding-top: 1rem;}}
</style>
""", unsafe_allow_html=True)

# Home-only CSS: full-width landing page, no sidebar or Streamlit header, brand row -> 12px -> pills -> 12px -> hero.
HOME_CSS = f"""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {{display: none !important;}}
[data-testid="stHeader"] {{display: none;}}
.block-container {{max-width: 100% !important; padding-top: 1rem !important; padding-left: 2rem !important;
                   padding-right: 2rem !important; padding-bottom: 0 !important;}}
/* brand row -> 12px -> section pills -> 12px -> hero */
.block-container [data-testid="stVerticalBlock"] {{gap: 12px !important;}}
.gt-brand {{font-size: 24px; letter-spacing: -0.02em; color: {PAPER}; line-height: 1;}}
.gt-brand span {{font-family: {MONO}; font-size: 12px; letter-spacing: -0.02em; text-transform: uppercase; color: {LICHEN};
                 margin-left: 12px;}}
.gt-brand {{margin-bottom: 0;}}
/* Streamlit pulls markdown blocks up by -1rem; cancel it for the brand row so the 12px gap holds */
[data-testid="stMarkdownContainer"]:has(> .gt-brand) {{margin-bottom: 0 !important;}}
iframe[title="app.glucotwin_home"] {{width: calc(100% + 4rem) !important; max-width: none !important; margin-left: -2rem;
                                    display: block; border: 0;}}
</style>"""


def page_home():
    if not home_available():
        st.info("Home page not found. Expected home/home.html inside the GlucoTwin folder.")
        return
    value = components.declare_component("glucotwin_home", path=str(home_component_dir()))(key="home", default=None)
    # "Open the dashboard" inside the page sends {page, n}; n is a timestamp so each click is handled once.
    if isinstance(value, dict) and value.get("page") in PAGES and value.get("n") != st.session_state.get("home_nav_n"):
        st.session_state.home_nav_n = value["n"]
        st.session_state.goto = value["page"]
        st.rerun()


# ---- Section selection first, so Home renders instantly without building the dashboard ----
goto = st.session_state.pop("goto", None)
if goto in PAGES:
    st.session_state.page = goto
elif st.session_state.get("page") not in PAGES:
    st.session_state.page = "Home"  # first visit opens on Home

if st.session_state.page == "Home":
    # The sidebar widgets are not drawn on Home, and Streamlit drops the state of keyed widgets that are not drawn
    # in a run. Re-assigning their keys keeps the chosen patient, clock and data source for the way back.
    for _key in ("pid", "clock", "source"):
        if _key in st.session_state:
            st.session_state[_key] = st.session_state[_key]
    st.markdown(HOME_CSS, unsafe_allow_html=True)
    st.markdown('<div class="gt-brand">GlucoTwin.<span>research prototype</span></div>', unsafe_allow_html=True)
    st.segmented_control("Section", PAGES, key="page", required=True, label_visibility="collapsed",
                         format_func=NAV_LABELS.get)
    page_home()
    st.stop()


# ==============================================================================================
# Loading (cached across reruns)
# ==============================================================================================
SOURCE_LABELS = {"synthetic": "Synthetic demo cohort", "real": "Imported real dataset"}


@st.cache_resource(show_spinner="Loading data and building features...")
def load_system(source: str):
    tables = ensure_data(source=source)
    return tables, build_dataset(tables)


def get_artifacts(source: str, tables, dataset):
    store = st.session_state.setdefault("artifacts_by_source", {})
    if source not in store:
        bar = st.progress(0.0, "Checking for trained models (retrains automatically if the data changed)...")
        try:
            store[source] = load_or_train(tables, dataset, source=source,
                                          progress=lambda f, m: bar.progress(min(f, 1.0), m))
        except ValueError as exc:
            bar.empty()
            st.error(f"Cannot train on this dataset: {exc}")
            st.stop()
        bar.empty()
    return store[source]


@st.cache_data(show_spinner=False)
def flagged_times(source: str, pid: str, config: str, artifacts_id: str) -> dict:
    """Timestamps where the risk model raises an alert - used only for the 'jump to next flag' demo control.
    Uses the precomputed causal feature table (identical to what the twin computes at that time)."""
    _, dataset = load_system(source)
    art = st.session_state.artifacts_by_source[source]
    df = dataset["features"]
    p = df[(df["patient_id"] == pid) & df["g_now"].notna()]
    out = {}
    for event, spec in EVENTS.items():
        rm = art["risk"][(event, config)]
        e = p[p[spec["eligible"]]]
        if not rm.available or e.empty:
            out[event] = []
            continue
        prob = rm.predict_proba(e)
        out[event] = list(e.loc[prob >= rm.threshold, "timestamp"])
    return out


def reset_patient_state():
    for key in ("pid", "clock", "history", "flag_msg"):
        st.session_state.pop(key, None)
    st.session_state.playing = False


sources = available_sources()
def sidebar_group(label: str, text: str, first: bool = False) -> None:
    """A labelled block in the sidebar control panel: mono label plus a one-line explanation."""
    st.markdown(f'<div class="gt-sgroup{" first" if first else ""}"><div class="gt-label">{label}</div>'
                f'<div class="cap">{text}</div></div>', unsafe_allow_html=True)


with st.sidebar:
    sidebar_group("Patient", "Choose whose twin to view. Held-out patients were never used for training.", first=True)
    if len(sources) > 1:
        source = st.radio("Data source", sources, key="source", format_func=SOURCE_LABELS.get,
                          on_change=reset_patient_state, horizontal=True)
    else:
        source = "synthetic"

with st.spinner("Loading the digital twin…"):
    tables, dataset = load_system(source)
    artifacts = get_artifacts(source, tables, dataset)
split = artifacts["split"]
patients = tables["patients"].set_index("patient_id")
SYN = is_synthetic(tables)
TIME_ONLY = split.mode == "time_only"
test_set, train_set = set(split.test_patients), set(split.train_patients)
_, _, TEST_MASK = split.masks(dataset["features"])


def pv(value, spec: str = "{}", missing: str = "unknown") -> str:
    """Format a possibly-missing profile value."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or (isinstance(value, str) and not value):
        return missing
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return str(value)


def cohort_label(pid: str) -> str:
    if TIME_ONLY:
        return "time-split"
    return "held-out" if pid in test_set else "training"


# ==============================================================================================
# Sidebar: patient, model, twin clock
# ==============================================================================================
def patient_label(pid: str) -> str:
    p = patients.loc[pid]
    return f"{pid} · {p['diabetes_type']} · {pv(p['age'], '{:.0f}y', '?y')} · {cohort_label(pid)}"


ordered = sorted(test_set) if TIME_ONLY else sorted(test_set) + sorted(train_set)


def demo_patient() -> str:
    """Default patient: the test patient with the most hypo windows in its test period
    (chosen for demonstration value, disclosed in the README)."""
    f = dataset["features"]
    counts = f.loc[TEST_MASK].groupby("patient_id")["label_hypo"].sum()
    return counts.idxmax() if len(counts) and counts.max() > 0 else ordered[0]


def default_clock(pid: str):
    g = dataset["grid"]
    t = g.loc[g["patient_id"] == pid, "timestamp"]
    start = max(split.cutoff_for(pid), t.min() + pd.Timedelta(hours=24))
    morning = start.normalize() + pd.Timedelta(hours=7)
    if morning <= start:
        morning += pd.Timedelta(days=1)
    return min(morning, t.max()).to_pydatetime()


def on_patient_change():
    st.session_state.clock = default_clock(st.session_state.pid)
    st.session_state.playing = False


if "pid" not in st.session_state or st.session_state.pid not in patients.index:
    st.session_state.pid = demo_patient()
    st.session_state.clock = default_clock(st.session_state.pid)
    st.session_state.playing = False
    st.session_state.history = {}

with st.sidebar:
    st.selectbox("Patient", ordered, key="pid", format_func=patient_label, on_change=on_patient_change,
                 label_visibility="collapsed")
    sidebar_group("Model", "Which data the twin uses: glucose only, plus wearables, or plus the health record.")
    configs = list(MODEL_CONFIGS)
    config = st.selectbox("Twin model configuration", configs, format_func=CONFIG_LABELS.get, label_visibility="collapsed",
                          index=configs.index(st.session_state.get("config_choice", configs[2])))
    st.session_state.config_choice = config

pid = st.session_state.pid
twin = PatientTwin(pid, dataset["grid"], dataset["static"], tables, artifacts)
min_clock = (twin.first_time + pd.Timedelta(hours=24)).to_pydatetime()
max_clock = twin.last_time.to_pydatetime()
if st.session_state.pop("advance", 0):
    st.session_state.clock = st.session_state.clock + timedelta(minutes=st.session_state.pop("advance_by", 5))
st.session_state.clock = min(max(st.session_state.clock, min_clock), max_clock)
PATIENT_CUTOFF = split.cutoff_for(pid)


def shift_clock(minutes: int):
    st.session_state.clock = min(max(st.session_state.clock + timedelta(minutes=minutes), min_clock), max_clock)


def toggle_play():
    st.session_state.playing = not st.session_state.playing


def jump_to_flag():
    flags = flagged_times(source, st.session_state.pid, st.session_state.get("cfg_for_flags", "full_twin"),
                          artifacts["meta"]["fingerprint"])
    now = pd.Timestamp(st.session_state.clock)
    # Low-glucose alerts first (clinically more urgent); fall back to high-glucose alerts.
    nxt = sorted(t for t in flags.get("hypo", []) if t > now + pd.Timedelta(minutes=30)) or         sorted(t for t in flags.get("hyper", []) if t > now + pd.Timedelta(minutes=30))
    if nxt:
        st.session_state.clock = (nxt[0] - pd.Timedelta(minutes=5)).to_pydatetime()
        st.session_state.flag_msg = None
    else:
        st.session_state.flag_msg = "No later model alert for this patient."


st.session_state.cfg_for_flags = config
with st.sidebar:
    sidebar_group("Twin clock", "Moves 'now'. Models only see data up to this time.")
    st.slider("Observations visible up to", min_value=min_clock, max_value=max_clock, key="clock",
              step=timedelta(minutes=C.SAMPLE_MINUTES), format="ddd DD MMM YYYY, HH:mm", label_visibility="collapsed")
    c1, c2, c3 = st.columns(3)
    c1.button("+5m", on_click=shift_clock, args=(5,), width="stretch")
    c2.button("+30m", on_click=shift_clock, args=(30,), width="stretch")
    c3.button("+2h", on_click=shift_clock, args=(120,), width="stretch")
    if (TIME_ONLY or pid in train_set) and pd.Timestamp(st.session_state.clock) < PATIENT_CUTOFF:
        st.warning("Before this patient's test period: this data was used for training/calibration, "
                   "so forecasts here are in-sample.")
    sidebar_group("Replay", "Plays the clock forward automatically. Next alert jumps to just before the next predicted low.")
    c4, c5 = st.columns(2)
    c4.button("⏸ Pause" if st.session_state.playing else "▶ Play", on_click=toggle_play, type="primary", width="stretch")
    c5.button("⚑ Next alert", on_click=jump_to_flag, width="stretch",
              help="Jump to just before the next low-glucose alert (or, if none, high-glucose alert) for this patient.")
    if st.session_state.get("flag_msg"):
        st.caption(st.session_state.flag_msg)
    speed = st.select_slider("Speed (seconds per step)", options=[0.25, 0.5, 1.0, 2.0],
                             value=st.session_state.get("speed_choice", 0.5))
    step_min = st.select_slider("Step (minutes)", options=[5, 15, 30], value=st.session_state.get("step_choice", 5))
    st.session_state.speed_choice, st.session_state.step_choice = speed, step_min
    st.markdown(f'<div class="gt-snote">Test period for this patient starts {PATIENT_CUTOFF:%a %d %b %Y %H:%M}. '
                "Data after the twin clock is hidden from the models.</div>", unsafe_allow_html=True)

twin.observe_until(st.session_state.clock)
state: TwinState = twin.state(config=config)

# Record each twin update so the timeline can show how the state evolved.
hist = st.session_state.history.setdefault((pid, config), {})


def snapshot(s: TwinState) -> dict:
    f120 = s.forecast.iloc[-1] if s.forecast is not None else None
    return dict(
        time=s.cursor, glucose=s.glucose_now,
        pred_120=None if f120 is None else f120["predicted"], lower_120=None if f120 is None else f120["lower"],
        upper_120=None if f120 is None else f120["upper"],
        p_hypo=s.risk.get("hypo", {}).get("probability"), p_hyper=s.risk.get("hyper", {}).get("probability"),
        cat_hypo=s.risk.get("hypo", {}).get("category", "Unavailable"),
        cat_hyper=s.risk.get("hyper", {}).get("category", "Unavailable"),
        can_forecast=s.can_forecast,
    )


if hist and abs(state.cursor - max(hist)) > pd.Timedelta(minutes=60):
    hist.clear()  # the clock jumped: start a fresh, contiguous timeline
if not hist:  # backfill: replay the previous 4 h in 15-min steps so the timeline starts with context
    back = PatientTwin(pid, dataset["grid"], dataset["static"], tables, artifacts)
    for minutes in range(240, 0, -15):
        t = state.cursor - pd.Timedelta(minutes=minutes)
        if t >= pd.Timestamp(min_clock):
            back.observe_until(t)
            hist[back.cursor] = snapshot(back.state(config=config, explain=False))
hist[state.cursor] = snapshot(state)


# ==============================================================================================
# Helpers
# ==============================================================================================
def card(label: str, value: str, sub: str = "") -> str:
    return f'<div class="gt-card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


def kpi(label: str, value: str, detail: str, meaning: str) -> str:
    """Overview KPI card: label -> value -> unit/detail -> what it means."""
    return (f'<div class="gt-card"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub">{detail}</div><div class="what">{meaning}</div></div>')


def grid(cards: list[str], kind: str = "") -> None:
    """Cards in one equal-height row (wraps to equal-height rows on narrow screens)."""
    st.markdown(f'<div class="gt-grid {kind}">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def section_intro(title: str, text: str) -> None:
    st.markdown(f'<div class="gt-intro"><div class="t" role="heading" aria-level="2">{title}</div>'
                f'<div class="d">{text}</div></div>', unsafe_allow_html=True)


def block(title: str, caption: str = "", prose: bool = False) -> None:
    """Heading + one caption line above every chart, table or group of cards (prose: Aspekta caption, 24px gap above)."""
    st.markdown(f'<div class="gt-block{" prose" if prose else ""}"><div class="t" role="heading" aria-level="3">{title}</div>'
                + (f'<div class="c">{caption}</div>' if caption else "") + "</div>", unsafe_allow_html=True)


def hairline() -> None:
    st.markdown('<hr class="gt-hr">', unsafe_allow_html=True)


def table(df: pd.DataFrame, max_height: int | None = None, key_column: bool = False) -> None:
    """Full-width table with wrapped (never truncated) text and one row style everywhere."""
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "—"
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:g}"
        return html.escape(str(v))
    num = [pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in df.columns]
    head = "".join(f'<th class="{"num" if n else ""}">{html.escape(str(c))}</th>' for c, n in zip(df.columns, num))
    def cls(i, n):
        return " ".join(c for c in ("num" if n else "", "key" if key_column and i == 0 else "") if c)
    rows = "".join("<tr>" + "".join(f'<td class="{cls(i, n)}">{cell(v)}</td>' for i, (v, n) in enumerate(zip(r, num)))
                   + "</tr>" for r in df.itertuples(index=False, name=None))
    style = f' style="max-height:{max_height}px"' if max_height else ""
    st.markdown(f'<div class="gt-table-wrap"{style}><table class="gt-table"><thead><tr>{head}</tr></thead>'
                f"<tbody>{rows}</tbody></table></div>", unsafe_allow_html=True)


def fmt(v, spec="{:.0f}", none="—"):
    return none if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)


def chip(text: str, dot: str | None = None) -> str:
    return f'<span class="gt-chip">' + (f'<i style="background:{dot}"></i>' if dot else "") + f"{html.escape(text)}</span>"


def freshness_chip(s: TwinState) -> str:
    """CGM data freshness for the app bar: missing, stale, imputed or fresh."""
    if s.minutes_since_obs is None:
        return chip("No CGM data", ALERT)
    if s.minutes_since_obs > C.STALE_AFTER_MINUTES:
        return chip(f"Stale CGM: {s.minutes_since_obs:.0f} min old", ALERT)
    if s.glucose_now_imputed:
        return chip("Current glucose imputed (short gap)", SERIOUS)
    return chip(f"CGM fresh · {s.minutes_since_obs:.0f} min", LIME)


def add_now_marker(fig, t):
    fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper", line=dict(color=LICHEN, dash="dot", width=1))
    fig.add_annotation(x=t, y=1.02, yref="paper", text="now", showarrow=False, font=dict(size=11, color=LICHEN))


def base_layout(fig, height=420, title=None, legend_below=None):
    below = bool(title) if legend_below is None else legend_below
    legend = (dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0) if below
              else dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=40 if title else 30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(text=title, x=0, xanchor="left", font=dict(family=SANS, size=18, color=PAPER, weight=400)),
                      legend=legend,
                      hovermode="x unified", font=dict(family=MONO, size=12, color=LICHEN),
                      hoverlabel=dict(bgcolor=INK, bordercolor=GRAPHITE, font=dict(family=SANS, color=PAPER)))
    axis = dict(gridcolor=GRAPHITE, linecolor=GRAPHITE, zerolinecolor=GRAPHITE)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def glucose_figure(s: TwinState, hours_back: float = 6, show_actual: bool = False, height: int = 430):
    obs = twin.observed(hours_back)
    fig = go.Figure()
    fig.add_hrect(y0=C.TARGET_RANGE[0], y1=C.TARGET_RANGE[1], fillcolor=COL["target"], line_width=0, layer="below")
    fig.add_hline(y=C.HYPO_THRESHOLD, line=dict(color=COL["hypo"], dash="dash", width=1),
                  annotation_text=f"{C.HYPO_THRESHOLD:.0f}", annotation_position="bottom right")
    fig.add_hline(y=C.HYPER_THRESHOLD, line=dict(color=COL["hyper"], dash="dash", width=1),
                  annotation_text=f"{C.HYPER_THRESHOLD:.0f}", annotation_position="top right")
    fig.add_trace(go.Scatter(x=obs["timestamp"], y=obs["glucose"], mode="lines", name="CGM (observed)",
                             line=dict(color=COL["obs"], width=2.2), connectgaps=False))
    imp = obs[obs["glucose_imputed"]]
    if len(imp):
        fig.add_trace(go.Scatter(x=imp["timestamp"], y=imp["glucose"], mode="markers", name="Imputed (≤30-min gap)",
                                 marker=dict(color=INK, size=8, line=dict(color=COL["imputed"], width=1.5))))
    if show_actual:
        fut = twin.revealed_future(2)
        fig.add_trace(go.Scatter(x=fut["timestamp"], y=fut["glucose"], mode="lines", name="Actual (hidden from model)",
                                 line=dict(color=COL["actual"], width=1.8, dash="dot")))
    if s.forecast is not None:
        f = s.forecast
        fig.add_trace(go.Scatter(x=f["time"], y=f["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=f["time"], y=f["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=COL["band"], name=f"{C.PREDICTION_INTERVAL:.0%} prediction interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=f["time"], y=f["predicted"], mode="lines+markers", name="Twin forecast",
                                 line=dict(color=COL["forecast"], width=2.4, dash="dash"), marker=dict(size=7)))
    add_now_marker(fig, s.cursor)
    top = float(np.nanmax(obs["glucose"].to_numpy(), initial=0))
    if s.forecast is not None:
        top = max(top, float(s.forecast["upper"].max()))
    fig.update_yaxes(title="Glucose (mg/dL)", range=[40, max(300, top + 30)])
    fig.update_xaxes(range=[s.cursor - pd.Timedelta(hours=hours_back), s.cursor + pd.Timedelta(hours=2.2)])
    return base_layout(fig, height)


def risk_card(event: str, info: dict) -> str:
    cat = info["category"]
    dot, border = CATEGORY_STYLE.get(cat, CATEGORY_STYLE["Unavailable"])
    if info["probability"] is not None:
        prob = info["probability"]
        main = f"{prob:.0%}" if prob >= 0.01 else "<1%"
        extra = "Model alert: above F1-optimised alert threshold" if info["alert"] else "Below model alert threshold"
        extra += f" ({info['threshold']:.0%})"
    elif info["currently_beyond"]:
        main, extra = "Now", "Current glucose is already beyond this threshold - not a prediction."
    else:
        main, extra = "—", info["message"] or "Risk model unavailable."
    return (f'<div class="gt-risk" style="border-color:{border}"><div class="gt-label">{info["name"]}</div>'
            f'<div class="cat"><span class="gt-dot" style="background:{dot}"></span>{cat} · {main}</div>'
            f'<div class="gt-note">{extra}</div></div>')


def contributions_figure(df: pd.DataFrame, unit: str, height=360):
    d = df.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["contribution"], y=d["description"], orientation="h",
        marker_color=[GROUP_COLORS.get(g, COL["other"]) for g in d["group"]],
        customdata=np.stack([d["group"], d["value"].astype(str)], axis=1),
        hovertemplate="%{y}<br>value: %{customdata[1]}<br>group: %{customdata[0]}<br>contribution: %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=LICHEN, width=1))
    fig.update_xaxes(title=f"Contribution ({unit})")
    return base_layout(fig, height).update_layout(hovermode="closest", showlegend=False)


# ==============================================================================================
# Header
# ==============================================================================================
prof = twin.profile

# CSS-only markdown blocks still take a 16px row gap each; collapse them so the app bar starts at the top.
st.markdown("""<style>
[data-testid="stMain"] [data-testid="stElementContainer"]:has(style) {display: none;}
.block-container {padding-left: 2.5rem; padding-right: 2.5rem;}
</style>""", unsafe_allow_html=True)
# One app bar: project name left; twin clock and data badges right, on one line.
if SYN:
    data_chip = chip("Simulated", LICHEN)
else:
    prov = patients["provenance"].dropna()
    data_chip = chip(f"Real data · {prov.iloc[0] if len(prov) else 'imported'}", PAPER)
st.markdown(f'<div class="gt-appbar"><div class="brand"><span class="name">GlucoTwin.</span>'
            f'<span class="tagline">AI patient digital twin · 2-hour glucose risk</span></div>'
            f'<div class="right"><span class="clock"><span>Twin clock</span>{state.cursor:%a %d %b %Y, %H:%M}</span>'
            f"{data_chip}{freshness_chip(state)}</div></div>", unsafe_allow_html=True)
# Patient strip: name + ID, then the facts as small chips.
cohort = "time-split patient" if TIME_ONLY else "held-out test patient" if pid in test_set else "training-cohort patient"
model_short = re.sub(r"^\d+\s*-\s*|\s*\(.*\)$", "", CONFIG_LABELS[config])
facts = [prof["diabetes_type"], pv(prof["age"], "{:.0f} y", "age unknown"), pv(prof["sex"], missing="sex unknown"),
         cohort, f"model: {model_short}"]
st.markdown(f'<div class="gt-patient"><span class="name">{html.escape(str(prof["display_name"]))}</span>'
            f'<span class="pid">{pid}</span>' + "".join(chip(str(f)) for f in facts) + "</div>", unsafe_allow_html=True)
# Section pills (the same widget, key and labels as on Home; exactly one per run).
page = st.segmented_control("Section", PAGES, key="page", required=True, label_visibility="collapsed",
                            format_func=NAV_LABELS.get)


# ==============================================================================================
# Pages
# ==============================================================================================
def page_overview():
    s = state
    section_intro("Overview", INTROS["Overview"])
    rate = f"{s.trend_rate:+.1f} mg/dL/min" if s.trend_rate is not None else "trend unavailable"
    p120 = s.forecast.iloc[-1] if s.forecast is not None else None
    sm = s.summary_24h
    hypo, hyper = s.risk.get("hypo"), s.risk.get("hyper")
    worst = max([i for i in (hypo, hyper) if i], key=lambda i: (i["probability"] or 0), default=None)
    grid([
        kpi("Glucose now", f"{fmt(s.glucose_now)} <span style='font-size:1.1rem'>{s.trend}</span>", f"mg/dL · {rate}",
            "Latest CGM reading and its trend per minute."),
        kpi("Forecast +2 h", fmt(None if p120 is None else p120["predicted"]),
            "mg/dL · " + (f"interval {p120['lower']:.0f}-{p120['upper']:.0f}" if p120 is not None else s.reason[:60]),
            "Predicted glucose in 2 hours, with its 80% interval."),
        kpi("Time in range", fmt(sm.get("tir"), "{:.0f}%"),
            f"below {fmt(sm.get('tbr'), '{:.1f}%')} · above {fmt(sm.get('tar'), '{:.0f}%')}",
            "Share of the last 24 h between 70 and 180 mg/dL."),
        kpi("Variability", fmt(sm.get("cv"), "{:.0f}%"),
            f"CV · mean {fmt(sm.get('mean'))} mg/dL · GMI {fmt(sm.get('gmi'), '{:.1f}%')}",
            "Coefficient of variation; above 36% means unstable glucose."),
        kpi("Risk (2 h)", "—" if not worst else worst["category"].replace("Currently beyond threshold", "Beyond now"),
            "hypo " + (fmt(hypo["probability"], "{:.0%}") if hypo else "—") + " · hyper "
            + (fmt(hyper["probability"], "{:.0%}") if hyper else "—"),
            "Higher of the low and high risk categories for the next 2 hours."),
    ])
    hairline()
    left, right = st.columns([3, 2], gap="large")
    with left:
        block("Digital-twin snapshot", "White: real CGM. Dashed: the twin's forecast. Band: 80% interval.", prose=True)
        st.plotly_chart(glucose_figure(s, hours_back=6, height=400), theme=None, key="ov_chart")
        if not s.can_forecast:
            st.warning(s.reason)
        block("Latest observations", "The newest value of each live signal and when it was last seen.", prose=True)
        obs = twin.observed(6)
        def last_time(col):
            v = obs.loc[obs[col].notna(), "timestamp"] if col in obs else pd.Series(dtype="datetime64[ns]")
            return "missing (6 h)" if v.empty else f"{v.iloc[-1]:%H:%M} ({(s.cursor - v.iloc[-1]).total_seconds() / 60:.0f} min ago)"
        real = obs[obs["glucose_observed"]]
        latest = pd.DataFrame([
            ("CGM glucose (real)", fmt(s.glucose_now), "missing" if real.empty else f"{real['timestamp'].iloc[-1]:%H:%M}"),
            ("Heart rate", fmt(s.wearable.get("hr_now"), "{:.0f} bpm"), last_time("heart_rate_bpm")),
            ("Steps, last 30 min", fmt(s.wearable.get("steps_30m")), last_time("steps")),
            ("Sleep, last 24 h", fmt(s.wearable.get("sleep_24h_h"), "{:.1f} h"), last_time("is_asleep")),
            ("Logged carbs, last 2 h", fmt(s.wearable.get("carbs_2h"), "{:.0f} g"), "event log"),
            ("Logged insulin, last 2 h", fmt(s.wearable.get("insulin_2h"), "{:.1f} U"), "event log"),
        ], columns=["Signal", "Value", "Last seen"])
        table(latest, key_column=True)
    with right:
        block("Synthetic patient profile" if SYN else "Patient profile", "Static health-record facts used by the twin.",
              prose=True)
        sf = twin.static_row
        prof_rows = {
            "Diabetes": f"{prof['diabetes_type']}, "
                        f"{pv(prof['years_since_diagnosis'], '{:.0f} y since diagnosis', 'duration unknown')}",
            "Therapy": prof["insulin_delivery"] if prof["insulin_delivery"] != "none" else "No insulin",
            "Latest HbA1c": pv(sf["hba1c_latest"], "{:.1f}%", "not recorded")
                            + (f" (change {sf['hba1c_change']:+.1f})" if pd.notna(sf["hba1c_change"]) else ""),
            "BMI": pv(prof["bmi"], "{:.1f}", "not recorded"),
            "Prior hypo events (12 mo)": pv(prof["prior_hypo_events_12m"], "{:.0f}", "not recorded"),
            "Medications": ", ".join(twin.medications["medication"]) or "None recorded",
        }
        table(pd.DataFrame(prof_rows.items(), columns=["Field", "Value"]), key_column=True)
        q = s.quality
        block("Data quality (last 24 h)",
              "Imputed = gaps ≤ 30 min filled in; longer gaps stay missing.", prose=True)
        quality = [("Glucose imputed", fmt(q.get("pct_glucose_imputed_24h"), "{:.1f}%")),
                   ("Glucose missing", fmt(q.get("pct_glucose_missing_24h"), "{:.1f}%")),
                   ("Heart rate missing", fmt(q.get("pct_hr_missing_24h"), "{:.1f}%")),
                   ("Model inputs missing now", f"{len(s.missing_features)} / {len(artifacts['forecast'].features[config])}")]
        st.markdown('<div class="gt-mini">' + "".join(f'<div><div class="label">{k}</div><div class="value">{v}</div></div>'
                                                      for k, v in quality) + "</div>", unsafe_allow_html=True)


def page_forecast():
    s = state
    section_intro("Glucose Forecast", INTROS["Glucose Forecast"])
    c1, c2, c3 = st.columns([2, 2, 3])
    hours = c1.select_slider("History shown", options=[3, 6, 12, 24], value=6, format_func=lambda h: f"{h} h")
    show_actual = c2.toggle("Reveal what actually happened", value=False,
                            help="Shows the next 2 h of simulated CGM data. The model never sees it - for replay evaluation only.")
    block("Glucose and the 2-hour forecast",
          "White line: real CGM readings. Dashed line: the twin's forecast. Shaded band: 80% interval. "
          "Dotted line (when revealed): what actually happened.")
    st.plotly_chart(glucose_figure(s, hours_back=hours, show_actual=show_actual, height=470), theme=None, key="fc_chart")
    if not s.can_forecast:
        st.warning(f"Forecast withheld: {s.reason}")
        return
    hairline()
    left, right = st.columns([3, 2], gap="large")
    with left:
        fam = artifacts["forecast"].selected_family[config]
        block("Forecast trajectory",
              f"Predicted glucose at each horizon with its {C.PREDICTION_INTERVAL:.0%} interval. Model: {FAMILY_LABELS[fam]}, "
              "one model per horizon; the interval is split-conformal and approximate for time series.")
        f = s.forecast.copy()
        f["time"] = f["time"].dt.strftime("%H:%M")
        f["horizon"] = f["horizon_min"].map(lambda m: "now" if m == 0 else f"+{m} min")
        f = f[["horizon", "time", "predicted", "lower", "upper"]].round(0)
        f.columns = ["Horizon", "Clock time", "Predicted (mg/dL)", "Interval low", "Interval high"]
        table(f)
    with right:
        block("Threshold crossing (next 2 h)",
              "Calibrated chance that glucose crosses each threshold; the dot shows the risk category.")
        st.markdown('<div class="gt-stack">' + "".join(risk_card(e, s.risk[e]) for e in ("hypo", "hyper") if e in s.risk)
                    + "</div>", unsafe_allow_html=True)
        crosses = s.forecast[(s.forecast["horizon_min"] > 0)]
        if (crosses["predicted"] < C.HYPO_THRESHOLD).any() or (crosses["lower"] < C.HYPO_THRESHOLD).any():
            st.info(f"Forecast trajectory or its interval dips below {C.HYPO_THRESHOLD:.0f} mg/dL.")
        if (crosses["predicted"] > C.HYPER_THRESHOLD).any():
            st.info(f"Forecast trajectory rises above {C.HYPER_THRESHOLD:.0f} mg/dL.")


def page_clinical():
    section_intro("Clinical Context", INTROS["Clinical Context"])
    obs = twin.observed(24).copy()
    for col in ("heart_rate_bpm", "steps", "is_asleep", "carbs_logged_g", "insulin_logged_u"):
        if col not in obs:
            obs[col] = np.nan  # signal not provided by this dataset
    missing_hr = obs["heart_rate_bpm"].isna().mean() * 100
    left, right = st.columns([3, 2], gap="large")
    with left:
        block("Wearable & logged lifestyle (last 24 h)",
              "Heart rate; steps with sleep marked; logged carbs and insulin. "
              + (f"Heart rate and steps are missing for {missing_hr:.0f}% of the period (e.g. watch charging) and are "
                 "left missing, not set to zero." if missing_hr > 0 else "No missing wearable data in this period."))
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.4, 0.3, 0.3],
                            subplot_titles=("Heart rate (bpm)", "Steps per 5 min · sleep shaded", "Logged carbs (g) & insulin (U)"))
        fig.add_trace(go.Scatter(x=obs["timestamp"], y=obs["heart_rate_bpm"], mode="lines", name="Heart rate",
                                 line=dict(color=PAPER, width=1.4), connectgaps=False), row=1, col=1)
        fig.add_trace(go.Bar(x=obs["timestamp"], y=obs["steps"], name="Steps", marker_color=COL["wearable"]), row=2, col=1)
        sleep = obs[obs["is_asleep"] == 1]
        if len(sleep):
            fig.add_trace(go.Scatter(x=sleep["timestamp"], y=[0] * len(sleep), mode="markers", name="Asleep",
                                     marker=dict(color=LICHEN, size=4, symbol="square")), row=2, col=1)
        carbs = obs[obs["carbs_logged_g"] > 0]
        ins = obs[obs["insulin_logged_u"] > 0]
        fig.add_trace(go.Bar(x=carbs["timestamp"], y=carbs["carbs_logged_g"], name="Carbs (g)", marker_color=COL["wearable"], width=900000), row=3, col=1)
        fig.add_trace(go.Bar(x=ins["timestamp"], y=ins["insulin_logged_u"], name="Insulin (U)", marker_color=PAPER, width=900000), row=3, col=1)
        fig.update_layout(bargap=0)
        st.plotly_chart(base_layout(fig, 560, legend_below=True).update_layout(hovermode="closest"), theme=None, key="wear")
    with right:
        block("Diagnoses", "Coded conditions from the health record.")
        cond = twin.conditions[["display", "code_system", "code", "onset_date"]].rename(
            columns={"display": "Condition", "code_system": "System", "code": "Code", "onset_date": "Onset"})
        table(cond)
        block("Medications", "Current medicines and their drug class.")
        table(twin.medications[["medication", "drug_class", "status"]].rename(
            columns={"medication": "Medication", "drug_class": "Class", "status": "Status"}))
        block("Laboratory results", "Lab tests recorded before monitoring started.")
        labs = twin.labs[["date", "test", "value", "unit"]].rename(columns=str.title)
        table(labs)
        block("HbA1c history", "Long-term glucose control (%) from past lab tests.")
        a1c = twin.labs[twin.labs["test"] == "HbA1c"]
        fig = go.Figure(go.Scatter(x=pd.to_datetime(a1c["date"]), y=a1c["value"], mode="lines+markers",
                                   line=dict(color=COL["ehr"])))
        fig.update_yaxes(title="HbA1c (%)")
        st.plotly_chart(base_layout(fig, 220), theme=None, key="a1c")

    hairline()
    block("How the twin fuses static and dynamic information",
          "At every update the twin re-reads the data up to the twin clock, recomputes its time-series features, adds the "
          "fixed health-record features, and passes them to the forecast and risk models.")
    fs = artifacts["forecast"].features
    cards = []
    for group, label in GROUP_LABELS.items():
        used = [f for f in FEATURE_GROUPS[group] if f in fs["full_twin"]]
        cards.append(card(label, f"{len(used)} features",
                          {"glucose": "Recent levels, trends, variability, time of day",
                           "wearable": "Heart rate, steps, sleep, logged carbs & insulin",
                           "ehr": "Age, diabetes type & duration, HbA1c, eGFR, medications, comorbidities, hypo history"}[group]))
    grid(cards, "three")


def page_risk():
    s = state
    section_intro("Risk & Explainability", INTROS["Risk & Explainability"])
    if not s.can_forecast:
        st.warning(f"No risk estimate: {s.reason}")
    block("Risk of a low or a high (next 2 h)",
          "Calibrated probability from each risk model; the dot shows its category. Definitions are in the panel below.")
    grid([risk_card(e, s.risk[e]) for e in ("hypo", "hyper") if e in s.risk], "two")
    with st.expander("Risk category definitions and label method"):
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in RISK_DEFINITIONS.items()))
        st.markdown(
            f"- **Label**: 1 if the CGM value crosses the threshold at any time in the next {C.HORIZON_MINUTES} min "
            f"(≥ {C.MIN_FUTURE_COVERAGE:.0%} of the window must have data). Thresholds ({C.HYPO_THRESHOLD:.0f} / "
            f"{C.HYPER_THRESHOLD:.0f} mg/dL) are configurable demo settings in `src/config.py`, not medical advice.\n"
            "- Only moments not already beyond the threshold are scored (a *new* crossing).\n"
            "- Probabilities are isotonic-calibrated on a calibration split; categories use fixed probability bands.")

    if s.can_forecast and s.contributions is not None:
        hairline()
        left, right = st.columns([3, 2], gap="large")
        with left:
            unit = "mg/dL" if "mg/dL" in s.explain_method else "model units"
            block("What drove the 2-hour glucose forecast",
                  f"Each bar is one input: right pushes the forecast up, left pushes it down ({unit}); colour = data source. "
                  f"Method: {s.explain_method}. Associations, not causes.")
            st.plotly_chart(contributions_figure(s.contributions, unit), theme=None, key="contrib")
        with right:
            block("Contribution by data source", "The same contributions summed per data source.")
            g = s.group_contributions.reindex(list(GROUP_LABELS.values())).dropna()
            fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=[GROUP_COLORS[i] for i in g.index]))
            fig.update_xaxes(title="Summed contribution")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(base_layout(fig, 250).update_layout(hovermode="closest"), theme=None, key="groups")
            block("Top inputs", "Each input's current value and its contribution to the forecast.")
            t = s.contributions[["description", "value", "contribution"]].copy()
            t["contribution"] = t["contribution"].round(2)
            table(t.rename(columns=str.title), max_height=250)
        if s.risk_contributions is not None:
            ev = s.risk_explain_event
            hairline()
            block(f"Drivers of the {ev} risk score",
                  f"Same view for the risk score of the more likely event. Method: {s.risk_explain_method}.")
            st.plotly_chart(contributions_figure(s.risk_contributions, "log-odds" if "log-odds" in s.risk_explain_method else "probability", 300), theme=None, key="rcontrib")

    hairline()
    block("Uncertainty & data warnings", "What limits confidence in the numbers above right now.")
    w = []
    if s.forecast is not None:
        f = s.forecast.iloc[-1]
        w.append(f"2-h forecast {f['predicted']:.0f} mg/dL with a {C.PREDICTION_INTERVAL:.0%} interval of "
                 f"{f['lower']:.0f}-{f['upper']:.0f} mg/dL (± {(f['upper'] - f['lower']) / 2:.0f}).")
    if s.glucose_now_imputed:
        w.append("Current glucose value is imputed across a short sensor gap.")
    if s.missing_features:
        w.append(f"{len(s.missing_features)} model inputs missing right now: "
                 + ", ".join(s.missing_features[:8]) + ("…" if len(s.missing_features) > 8 else "")
                 + ". The model handles missing values natively, but confidence is lower.")
    if s.minutes_since_obs is not None and s.minutes_since_obs > 5:
        w.append(f"Latest real CGM reading is {s.minutes_since_obs:.0f} min old.")
    for event, info in s.risk.items():
        if info["status"] != "ok":
            w.append(f"{event.capitalize()} model: {info['message']}")
    w.append("All inputs are simulated; performance on real patients is unknown.")
    for item in w:
        st.markdown(f"- {item}")

    hairline()
    block("Suggested clinician review prompts",
          "Informational prompts for human review only - not treatment, dosing, or triage instructions.")
    prompts = []
    for event, info in s.risk.items():
        if info["category"] in ("Elevated", "High"):
            word = "low" if event == "hypo" else "high"
            prompts.append(f"The twin estimates a {info['category'].lower()} probability of a {word} glucose excursion "
                           "in the next 2 h. Consider reviewing the recent CGM trend, logged meals and activity.")
        if info["currently_beyond"]:
            prompts.append("Current glucose is already outside the configured threshold - review the live CGM trace.")
    if s.missing_features or (s.minutes_since_obs or 0) > 5:
        prompts.append("Some inputs are missing or delayed - consider confirming the sensor and wearable are connected.")
    if not prompts:
        prompts.append("No elevated-risk flags at this time. Routine review of trends is sufficient for this demo.")
    for item in prompts:
        st.markdown(f"- {item}")

    with st.expander("Scenario sensitivity (how the model responds to hypothetical inputs)"):
        st.caption("Adds hypothetical logged inputs to the twin's current history and re-runs the model. This shows "
                   "**model sensitivity**, not a validated physiological simulation or a treatment recommendation.")
        a, b = st.columns(2)
        carbs = a.slider("Hypothetical carbs logged now (g)", 0, 90, 0, 5)
        walk = b.slider("Hypothetical walk in last 30 min (steps per 5 min)", 0, 700, 0, 50)
        if (carbs or walk) and s.can_forecast:
            alt = twin.state(config=config, scenario={"carbs_g": carbs, "walk_steps_per_5min": walk}, explain=False)
            cmp = pd.DataFrame({"Horizon": [f"+{h} min" for h in alt.forecast["horizon_min"][1:]],
                                "Current forecast": s.forecast["predicted"][1:].round(0).to_numpy(),
                                "With scenario": alt.forecast["predicted"][1:].round(0).to_numpy()})
            cmp["Difference"] = cmp["With scenario"] - cmp["Current forecast"]
            table(cmp)
            for event in ("hypo", "hyper"):
                p0, p1 = s.risk[event]["probability"], alt.risk[event]["probability"]
                if p0 is not None and p1 is not None:
                    st.markdown(f"- {event} probability: {p0:.0%} → {p1:.0%}")


def page_performance():
    ev = artifacts["evaluation"]
    sp = ev["split"]
    section_intro("Model Performance", INTROS["Model Performance"])
    block("Evaluation design", "How the data were split, so every reported number comes from data the models never trained on.")
    st.markdown(
        f"- **Split ({sp['mode']})**: {sp['description']}\n"
        "- Preprocessing statistics (medians, scalers, calibrators) are fitted on training data only.\n"
        f"- Model family (ridge vs. gradient boosting) is chosen by calibration-set MAE, not test MAE. "
        f"Backend: {artifacts['meta']['gbm_backend']}. Trained {artifacts['meta']['trained_at'][:16]} in "
        f"{artifacts['meta']['train_seconds']} s on {artifacts['meta'].get('n_patients', '?')} patients.")

    hairline()
    fm = ev["forecast_metrics"]
    block("Forecasting accuracy (test patients)",
          "Mean absolute error (MAE) and RMSE in mg/dL for each configuration and model at the chosen horizon; lower is better.")
    h = st.radio("Horizon", list(artifacts["forecast"].horizons), index=len(artifacts["forecast"].horizons) - 1,
                 horizontal=True, format_func=lambda m: f"{m} min")
    t = fm[fm["horizon_min"] == h].copy()
    show = t[["config_label", "family_label", "selected", "MAE", "RMSE", "calib_MAE", "interval_coverage", "n"]].rename(columns={
        "config_label": "Configuration", "family_label": "Model", "selected": "Used by twin", "calib_MAE": "Calibration MAE",
        "interval_coverage": f"{C.PREDICTION_INTERVAL:.0%} interval coverage"})
    table(show.round(2))
    left, right = st.columns(2, gap="large")
    with left:
        block("MAE by horizon", "Error at each forecast horizon for the model the twin uses; dotted = last-value baseline.")
        sel = fm[(fm["selected"]) | (fm["config"] == "persistence")]
        fig = go.Figure()
        config_col = {"persistence": LICHEN, "glucose_only": COL["glucose"], "glucose_wearable": COL["wearable"],
                      "full_twin": COL["ehr"]}  # colour follows the data source each configuration adds
        for lab, grp in sel.groupby("config_label", sort=True):
            cfg = grp["config"].iloc[0]
            fig.add_trace(go.Scatter(x=grp["horizon_min"], y=grp["MAE"], mode="lines+markers", name=lab, marker=dict(size=8),
                                     line=dict(color=config_col.get(cfg, LICHEN), width=2,
                                               dash="dot" if cfg == "persistence" else "solid")))
        fig.update_xaxes(title="Horizon (min)")
        fig.update_yaxes(title="MAE (mg/dL)")
        st.plotly_chart(base_layout(fig, 340).update_layout(hovermode="closest"), theme=None, key="mae_h")
    with right:
        block("Ablation with 95% CI", "Change in 2-hour error from each added data source; left of 0 = better, lime = a clear gain.")
        ab = ev["ablation"]
        fig = go.Figure(go.Bar(
            x=ab["mae_difference"], y=ab["comparison"], orientation="h",
            error_x=dict(type="data", symmetric=False, array=ab["ci_high"] - ab["mae_difference"],
                         arrayminus=ab["mae_difference"] - ab["ci_low"]),
            marker_color=[LIME if hi < 0 else ALERT if lo > 0 else LICHEN for lo, hi in zip(ab["ci_low"], ab["ci_high"])]))
        fig.update_traces(error_x_color=PAPER)
        fig.add_vline(x=0, line=dict(color=LICHEN, width=1))
        fig.update_xaxes(title="Δ MAE at 2 h (mg/dL) · negative = better")
        st.plotly_chart(base_layout(fig, 340).update_layout(hovermode="closest"), theme=None, key="abl")
    block("Ablation results",
          "Bootstrap resamples whole patient-days; a CI that includes 0 means the data show no clear improvement.")
    table(ev["ablation"][["comparison", "mae_difference", "ci_low", "ci_high", "conclusion"]].round(2))

    hairline()
    block("Risk classification (test patients)",
          "Alert quality per event and configuration. Compare PR-AUC with the base rate (what a random guess scores); "
          "positive windows overlap in time, so there are fewer independent events than 'positives'.")
    rm = ev["risk_metrics"]
    cols = ["event", "config_label", "status", "positives", "n", "base_rate", "threshold", "precision", "recall", "F1",
            "PR_AUC", "Brier", "ECE", "message"]
    table(rm[[c for c in cols if c in rm]].round(3))
    a, b = st.columns(2, gap="large")
    with a:
        block("Calibration curve", "Points on the diagonal mean predicted probabilities match how often events happened.")
        keys = list(ev["calibration"])
        if keys:
            key = st.selectbox("Calibration curve", keys, format_func=lambda k: f"{k[0]} · {CONFIG_LABELS[k[1]]}",
                               index=keys.index(("hypo", config)) if ("hypo", config) in keys else 0,
                               label_visibility="collapsed")
            ct = ev["calibration"][key]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=LICHEN, dash="dash"), name="Perfect"))
            fig.add_trace(go.Scatter(x=ct["mean_predicted"], y=ct["observed_rate"], mode="lines+markers", name="Model",
                                     text=ct["n"], hovertemplate="pred %{x:.2f}<br>obs %{y:.2f}<br>n=%{text}<extra></extra>",
                                     line=dict(color=PAPER), marker=dict(size=8)))
            fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
            fig.update_yaxes(title="Observed frequency", range=[0, 1])
            st.plotly_chart(base_layout(fig, 340).update_layout(hovermode="closest"), theme=None, key="calib")
    with b:
        block("Predicted vs actual (test sample)", "Each dot is one test moment at +2 h; the closer to the dashed line, the better.")
        sc = ev["scatter"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sc["actual"], y=sc["glucose_only"], mode="markers", name="Glucose only",
                                 marker=dict(size=4, color=COL["glucose"], opacity=0.4)))
        fig.add_trace(go.Scatter(x=sc["actual"], y=sc["full_twin"], mode="markers", name="Full twin",
                                 marker=dict(size=4, color=COL["ehr"], opacity=0.4)))
        fig.add_trace(go.Scatter(x=[40, 400], y=[40, 400], mode="lines", line=dict(color=LICHEN, dash="dash"), name="Ideal"))
        fig.update_xaxes(title="Actual glucose at +2 h")
        fig.update_yaxes(title="Predicted")
        st.plotly_chart(base_layout(fig, 340).update_layout(hovermode="closest"), theme=None, key="scatter")

    hairline()
    imp, method = artifacts["importance"][config]
    a, b = st.columns([2, 3], gap="large")
    with a:
        block("Share by data source", "Each source's share of mean |contribution| at the 2-hour horizon.")
        gsum = imp.groupby("group")["mean_abs_contribution"].sum()
        share = (100 * gsum / gsum.sum()).round(1)
        fig = go.Figure(go.Pie(labels=share.index, values=share.values, hole=0.55,
                               marker=dict(colors=[GROUP_COLORS.get(i, COL["other"]) for i in share.index],
                                           line=dict(color=INK, width=2))))
        st.plotly_chart(base_layout(fig, 300), theme=None, key="share")
    with b:
        block("Top features",
              f"Largest mean |contribution| on test rows ({method}). Importance is not causation; static features can also "
              "identify patients, one reason for held-out evaluation.")
        top = imp.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["mean_abs_contribution"], y=top["description"], orientation="h",
                               marker_color=[GROUP_COLORS.get(g, COL["other"]) for g in top["group"]]))
        st.plotly_chart(base_layout(fig, 300).update_layout(hovermode="closest"), theme=None, key="topf")
    if TIME_ONLY:
        st.caption("Time-only split: the same patients appear in training and test (different periods), so these "
                   "numbers measure personalised forecasting, not performance on new patients.")
    if SYN:
        st.warning("All metrics are computed on synthetic data produced by our own simulator. Because the simulator was "
                   "designed so that EHR attributes influence glucose dynamics, these results show the pipeline works - "
                   "they do not establish real-world clinical performance.")
    else:
        st.warning("Retrospective metrics on an imported dataset. They are not clinical validation: small cohorts, "
                   "a single data source and retrospective replay can all overstate real-world performance.")


def page_timeline():
    s = state
    rows = pd.DataFrame(sorted(hist.values(), key=lambda r: r["time"]))
    section_intro("Twin Timeline", INTROS["Twin Timeline"])
    block("Live twin state",
          "Press Play in the sidebar to stream new observations; each step re-reads the history and re-runs every model. "
          "Small figures show the change since the previous update.")
    prev = rows.iloc[-2] if len(rows) > 1 else None
    k = st.columns(4)
    def delta(cur, key, spec="{:+.0f}"):
        if prev is None or cur is None or prev[key] is None or pd.isna(prev[key]):
            return None
        return spec.format(cur - prev[key])
    k[0].metric("Glucose now", fmt(s.glucose_now, "{:.0f} mg/dL"), delta(s.glucose_now, "glucose"))
    p120 = None if s.forecast is None else float(s.forecast.iloc[-1]["predicted"])
    k[1].metric("Forecast +2 h", fmt(p120, "{:.0f} mg/dL"), delta(p120, "pred_120"))
    ph = s.risk.get("hypo", {}).get("probability")
    pr = s.risk.get("hyper", {}).get("probability")
    k[2].metric("Hypo risk (2 h)", fmt(ph, "{:.0%}"), delta(ph, "p_hypo", "{:+.0%}"), delta_color="inverse")
    k[3].metric("Hyper risk (2 h)", fmt(pr, "{:.0%}"), delta(pr, "p_hyper", "{:+.0%}"), delta_color="inverse")

    hairline()
    block("Earlier forecasts vs what happened",
          "Top: real glucose and the 2-hour forecasts plotted at the time they refer to. Bottom: risk over the replay. "
          "The twin only sees data left of the dotted 'now' line.")
    obs = twin.observed(10)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=("Observed glucose vs forecasts the twin made earlier", "Estimated 2-h risk over the replay"))
    fig.add_hrect(y0=C.TARGET_RANGE[0], y1=C.TARGET_RANGE[1], fillcolor=COL["target"], line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=obs["timestamp"], y=obs["glucose"], mode="lines", name="CGM observed",
                             line=dict(color=COL["obs"], width=2)), row=1, col=1)
    past = rows[rows["pred_120"].notna()].copy()
    if len(past):
        past["target_time"] = past["time"] + pd.Timedelta(minutes=C.HORIZON_MINUTES)
        fig.add_trace(go.Scatter(x=past["target_time"], y=past["pred_120"], mode="markers+lines", name="2-h forecast (plotted at target time)",
                                 line=dict(color=COL["forecast"], width=1, dash="dot"), marker=dict(size=5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=past["target_time"], y=past["upper_120"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=past["target_time"], y=past["lower_120"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=COL["band"], name="Interval", hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rows["time"], y=rows["p_hypo"], mode="lines+markers", name="Hypo probability",
                                 line=dict(color=COL["hypo"]), marker=dict(size=4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=rows["time"], y=rows["p_hyper"], mode="lines+markers", name="Hyper probability",
                                 line=dict(color=COL["hyper"]), marker=dict(size=4)), row=2, col=1)
    fig.add_hline(y=C.HYPO_THRESHOLD, line=dict(color=COL["hypo"], dash="dash", width=1), row=1, col=1)
    fig.add_vline(x=s.cursor, line=dict(color=LICHEN, dash="dot", width=1))
    fig.update_yaxes(title="mg/dL", row=1, col=1)
    fig.update_yaxes(title="probability", range=[0, 1], row=2, col=1)
    fig.update_xaxes(range=[s.cursor - pd.Timedelta(hours=10), s.cursor + pd.Timedelta(hours=2.2)])
    st.plotly_chart(base_layout(fig, 600, legend_below=True), theme=None, key="timeline")

    hairline()
    block("Twin update log", "One row per twin update, newest first; 'Category changed' marks a switch in a risk category.")
    if len(rows):
        log = rows.copy()
        log["changed"] = (log["cat_hypo"] != log["cat_hypo"].shift()) | (log["cat_hyper"] != log["cat_hyper"].shift())
        log = log.iloc[::-1].head(40)
        log["time"] = log["time"].dt.strftime("%a %H:%M")
        out = log[["time", "glucose", "pred_120", "p_hypo", "cat_hypo", "p_hyper", "cat_hyper", "changed"]].rename(columns={
            "time": "Twin clock", "glucose": "Glucose", "pred_120": "Forecast +2h", "p_hypo": "P(hypo)", "cat_hypo": "Hypo category",
            "p_hyper": "P(hyper)", "cat_hyper": "Hyper category", "changed": "Category changed"})
        table(out.round(2), max_height=320)
    if st.button("Clear timeline history"):
        hist.clear()
        st.rerun()


def page_about():
    section_intro("About & Limitations", INTROS["About & Limitations"])
    block("What makes this a digital twin, not just a prediction model",
          "Five properties that separate a twin from a one-off prediction.")
    st.markdown(
        "- **Patient-specific virtual representation**: one object per synthetic patient that holds their EHR layer "
        "(profile, conditions, medications, labs) and their live sensor layer under the same ID.\n"
        "- **Continuous synchronisation**: the twin state is recomputed from the observation stream at every 5-minute step "
        "(physical → virtual information flow).\n"
        "- **Forward simulation**: forecasts, intervals and threshold-crossing risks for the next 2 h, plus a model "
        "sensitivity tool for hypothetical inputs.\n"
        "- **Feedback to the clinician**: explanations, data-quality warnings and review prompts (virtual → physical "
        "information flow happens through a human, never automatically).\n"
        "- **Honest limits**: this prototype uses a statistical twin; it does not include a validated mechanistic "
        "physiology model, and it is not clinically validated.")
    hairline()
    if SYN:
        block("Simulation assumptions (all data is synthetic)",
              "Every rule the synthetic data generator follows; none are estimated from real data.")
        st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(SIMULATION_ASSUMPTIONS, 1)))
    else:
        block("Imported dataset", "Where the data come from and how missing fields are handled.")
        st.markdown(f"Source: **{SOURCE_LABELS[source]}** · {len(patients)} patients · "
                    f"{len(tables['timeseries']):,} rows. Static EHR features that the source does not provide are "
                    "left missing and dropped from the models automatically.")
    hairline()
    block("Limitations", "Known weaknesses to keep in mind when reading any number in this dashboard.")
    st.markdown(
        "- Synthetic data from our own simulator: models may learn the simulator's rules, and the EHR → glucose links were "
        "designed by us. Real-world performance is unknown.\n"
        f"- Small cohort ({len(patients)} patients, ~{(dataset['grid']['timestamp'].max() - dataset['grid']['timestamp'].min()).days} days span); "
        f"{len(split.test_patients)} test patients ({split.mode} split), so "
        "metrics have wide uncertainty - especially for rare hypoglycaemia windows.\n"
        "- Static EHR features can let models memorise patients; that is why we evaluate on unseen patients.\n"
        "- Conformal intervals assume exchangeability, which time series violate - test coverage is reported, not assumed.\n"
        "- No external validation (e.g., OhioT1DM under its data-use agreement), no prospective testing, no regulatory review.\n"
        "- Not a medical device; no treatment or insulin-dosing recommendations are made.")
    hairline()
    block("Data quality across the cohort", "Per patient: how much glucose and heart-rate data is missing or filled in.")
    q = dataset["quality"].copy()
    table(q.drop(columns=["first_timestamp", "last_timestamp"]).round(1), max_height=300)
    if dataset["issues"]:
        st.markdown("Validation issues: " + "; ".join(dataset["issues"]))
    hairline()
    block("Interoperability export" + (" (synthetic)" if SYN else ""),
          "Download this patient's record as a FHIR R4-style bundle, or their time series as CSV.")
    a, b = st.columns(2)
    a.download_button("Download FHIR R4-style Bundle (JSON)", fhir_json(pid, tables), file_name=f"{pid}_fhir_bundle.json",
                      mime="application/json", width="stretch")
    ts = tables["timeseries"]
    b.download_button("Download this patient's time-series (CSV)", ts[ts["patient_id"] == pid].to_csv(index=False),
                      file_name=f"{pid}_timeseries.csv", mime="text/csv", width="stretch")


def page_journey():
    section_intro("Patient Journey", INTROS["Patient Journey"])
    st.caption("A standalone teaching simulation of one ward patient (Bed 7), separate from the twin cohort: the sidebar "
               "patient and twin clock do not drive it. Press play inside the panel.")
    if not journey_available():
        st.info(f"Patient journey page not found. Expected demo/, styles/ and fonts/ in {JOURNEY_ROOT}.")
        return
    components.declare_component("patient_journey", path=str(journey_component_dir()))(key="patient_journey")


{
    "Overview": page_overview, "Glucose Forecast": page_forecast, "Clinical Context": page_clinical,
    "Risk & Explainability": page_risk, "Model Performance": page_performance, "Twin Timeline": page_timeline,
    "Patient Journey": page_journey,
    "About & Limitations": page_about,
}[page]()

st.markdown('<div class="gt-disclaimer">GlucoTwin is a research and educational prototype' +
            (' built on synthetic data. ' if SYN else ' (imported dataset, retrospective replay). ') +
            'It is not a medical device, has not been clinically validated, and does not provide diagnosis, treatment '
            'or insulin-dosing advice. Risk flags are model outputs for discussion only.</div>', unsafe_allow_html=True)

# ---- replay loop: advance the twin clock and rerun (all pages update consistently) ----
if st.session_state.playing:
    if st.session_state.clock >= max_clock:
        st.session_state.playing = False
    else:
        time.sleep(speed)
        st.session_state.advance, st.session_state.advance_by = 1, step_min
        st.rerun()
