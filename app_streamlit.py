"""
Streamlit dashboard for quantization benchmark results.
Design: Option C - blue accent, sidebar + main, functional nav
"""
import os
import pathlib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from quant_pipeline.core.pipeline import Pipeline

# Auto-create .streamlit/config.toml to force light mode
_cfg_dir  = pathlib.Path(__file__).parent / ".streamlit"
_cfg_file = _cfg_dir / "config.toml"
_cfg_dir.mkdir(exist_ok=True)
if not _cfg_file.exists() or "[theme]" not in _cfg_file.read_text():
    _cfg_file.write_text(
        '[theme]\n'
        'base = "light"\n'
        'primaryColor = "#2563eb"\n'
        'backgroundColor = "#f4f6f9"\n'
        'secondaryBackgroundColor = "#ffffff"\n'
        'textColor = "#111111"\n'
    )

st.set_page_config(
    page_title="QuantBench",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Session state -----------------------------------------------------------
if "page"           not in st.session_state: st.session_state.page           = "Benchmark report"
if "pred_result"    not in st.session_state: st.session_state.pred_result    = None
if "pred_last_text" not in st.session_state: st.session_state.pred_last_text = ""

PAGES = ["Benchmark report", "Robustness", "Sensitivity"]
page  = st.session_state.page

# ---- CSS ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Geist', sans-serif !important;
    box-sizing: border-box;
}

header[data-testid="stHeader"] { display: none; }
/* Hide sidebar collapse button and its tooltip */
button[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }
*:focus { outline: none !important; }

.stApp { background: #f4f6f9 !important; }

.block-container {
    padding: 28px 36px 40px 36px !important;
    max-width: 100% !important;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8eaed !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

/* Sidebar radio styled as nav */
section[data-testid="stSidebar"] .stRadio { margin: 0 !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 20px !important;
    margin: 0 !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #6b7280 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #f3f4f6 !important;
    color: #111 !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #eff6ff !important;
    color: #2563eb !important;
    font-weight: 600 !important;
    border-right: 2px solid #2563eb !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #2563eb !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stRadio label p {
    font-size: 13px !important;
    margin: 0 !important;
    color: inherit !important;
}
/* Hide radio circle */
section[data-testid="stSidebar"] .stRadio label > div:first-child { display: none !important; }

.sb-brand {
    padding: 18px 20px 14px;
    border-bottom: 1px solid #e8eaed;
    display: flex; align-items: center; gap: 9px;
    margin-bottom: 6px;
}
.sb-icon {
    width: 26px; height: 26px; background: #2563eb; border-radius: 6px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.sb-name { font-size: 14px; font-weight: 700; color: #111 !important; }
.sb-section {
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: #c4c4c4; padding: 10px 20px 4px;
    display: block;
}
.sb-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.sb-footer { padding: 14px 20px; border-top: 1px solid #e8eaed; font-size: 11px; color: #d1d5db; }

/* Radio as nav */
section[data-testid="stSidebar"] .stRadio { margin: 0 !important; padding: 0 !important; }
section[data-testid="stSidebar"] .stRadio > label { display: none !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; display: flex !important; flex-direction: column !important; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 20px !important; margin: 0 !important; border-radius: 0 !important;
    font-size: 13px !important; font-weight: 500 !important; color: #111111 !important;
    cursor: pointer !important; min-height: unset !important;
    width: 100% !important; display: block !important;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: #f3f4f6 !important; color: #111 !important; }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #eff6ff !important; color: #2563eb !important;
    font-weight: 600 !important; border-right: 2px solid #2563eb !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p { color: #2563eb !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] .stRadio label p { font-size: 13px !important; margin: 0 !important; color: inherit !important; width: 100% !important; }
section[data-testid="stSidebar"] .stRadio label > div:first-child { display: none !important; }
.sb-nav-item {
    display: block; padding: 8px 20px; font-size: 13px; font-weight: 400;
    color: #6b7280 !important; text-decoration: none; cursor: pointer;
}
.sb-nav-item:hover { background: #f3f4f6; color: #111 !important; }
.sb-nav-active {
    display: block; padding: 8px 20px; font-size: 13px; font-weight: 600;
    color: #2563eb !important; background: #eff6ff;
    border-right: 2px solid #2563eb; text-decoration: none;
}

/* ---- Page headings ---- */
.page-title {
    font-size: 22px; font-weight: 700; color: #111 !important;
    letter-spacing: -0.025em; margin-bottom: 4px;
}
.page-sub { font-size: 13px; color: #9ca3af !important; line-height: 1.5; margin-bottom: 22px; }

/* ---- Section labels ---- */
.section-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: #374151 !important; margin-bottom: 12px; display: block;
}

/* ---- KPI cards ---- */
.kpi-card {
    background: #fff !important; border: 1px solid #e8eaed;
    border-radius: 10px; padding: 16px 18px;
}
.kpi-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9ca3af !important; margin-bottom: 7px;
}
.kpi-value { font-size: 24px; font-weight: 700; color: #111 !important; letter-spacing: -0.03em; line-height: 1; }
.kpi-unit  { font-size: 13px; color: #9ca3af !important; margin-left: 2px; font-weight: 400; }
.kpi-sub   { font-size: 11px; color: #2563eb !important; margin-top: 6px; font-weight: 600; }

/* ---- Cards ---- */
.card { background: #fff !important; border: 1px solid #e8eaed; border-radius: 10px; overflow: hidden; margin-bottom: 14px; }
.card-header { padding: 13px 18px 11px; border-bottom: 1px solid #f3f4f6; display: flex; align-items: center; justify-content: space-between; }
.card-title { font-size: 13px; font-weight: 700; color: #111 !important; }
.card-meta  { font-size: 11px; color: #9ca3af !important; }
.card-body  { padding: 18px; }
.card-body-flush { padding: 0; }

/* ---- Data table ---- */
.data-table { width: 100%; border-collapse: collapse; }
.data-table thead th {
    padding: 9px 16px; text-align: left; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: #9ca3af !important; border-bottom: 1px solid #f3f4f6; background: #fafafa !important;
}
.data-table tbody td { padding: 10px 16px; color: #374151 !important; font-size: 13px; border-bottom: 1px solid #f9fafb; }
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover td { background: #f9fafb !important; }

/* ---- Mode pills ---- */
.pill { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; }
.pill-fp32 { background: #eff6ff; color: #1d4ed8 !important; }
.pill-fp16 { background: #f5f3ff; color: #6d28d9 !important; }
.pill-ptq  { background: #ecfdf5; color: #065f46 !important; }
.pill-qat  { background: #fff7ed; color: #9a3412 !important; }

/* ---- Accuracy bars ---- */
.acc-rows { display: flex; flex-direction: column; gap: 12px; }
.acc-row  { display: flex; align-items: center; gap: 12px; }
.acc-mode { display: flex; align-items: center; gap: 7px; width: 90px; flex-shrink: 0; }
.acc-dot  { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.acc-name { font-size: 12px; color: #6b7280 !important; }
.acc-track { flex: 1; height: 7px; border-radius: 4px; overflow: hidden; }
.acc-fill  { height: 100%; border-radius: 4px; }
.acc-val   { font-size: 12.5px; font-weight: 700; color: #111 !important; min-width: 44px; text-align: right; }

/* ---- Text input ---- */
div[data-testid="stTextInput"] label {
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important; color: #9ca3af !important;
}
div[data-testid="stTextInput"] input {
    background: #f9fafb !important; border: 1px solid #e5e7eb !important;
    border-radius: 7px !important; color: #111 !important;
    font-size: 13px !important; caret-color: #2563eb !important;
}
div[data-testid="stTextInput"] input:focus {
    background: #fff !important; border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
div[data-baseweb="input"] { box-shadow: none !important; }
div[data-testid="stTextInput"] input:-webkit-autofill,
div[data-testid="stTextInput"] input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0 999px #ffffff inset !important;
    -webkit-text-fill-color: #111 !important;
    transition: background-color 9999s ease-in-out 0s !important;
}

/* ---- Selectbox ---- */
div[data-testid="stSelectbox"] label {
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important; color: #9ca3af !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #f9fafb !important; border: 1px solid #e5e7eb !important;
    border-radius: 7px !important; color: #111 !important;
    font-size: 13px !important; min-height: 38px !important;
}
div[data-testid="stSelectbox"] svg { display: block !important; color: #6b7280 !important; }
div[data-baseweb="popover"] {
    background: #fff !important; border: 1px solid #e5e7eb !important;
    border-radius: 8px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] li { font-size: 13px !important; color: #111 !important; background: #fff !important; }
ul[data-testid="stSelectboxVirtualDropdown"] li:hover { background: #f3f4f6 !important; }
li[aria-selected="true"] { background: #eff6ff !important; color: #2563eb !important; }

/* ---- Run inference button ---- */
div[data-testid="stMain"] .stButton > button {
    background-color: #2563eb !important; color: #fff !important;
    font-weight: 700 !important; font-size: 13px !important;
    border: none !important; border-radius: 7px !important;
    padding: 9px 22px !important; width: 100% !important;
    opacity: 1 !important; cursor: pointer !important;
}
div[data-testid="stMain"] .stButton > button:hover,
div[data-testid="stMain"] .stButton > button:hover * {
    background-color: #1d4ed8 !important; color: #fff !important; opacity: 1 !important;
}
div[data-testid="stMain"] .stButton > button:active {
    background-color: #1e40af !important; color: #fff !important; opacity: 1 !important;
}
div[data-testid="stMain"] .stButton > button:disabled,
div[data-testid="stMain"] .stButton > button[disabled] {
    background-color: #2563eb !important; color: #fff !important; opacity: 1 !important;
}

/* ---- Result card ---- */
.result-card { border-radius: 8px; padding: 13px 16px; margin-top: 4px; border: 1px solid; }
.result-pos  { background: #f0fdf4 !important; border-color: #bbf7d0; }
.result-neg  { background: #fef2f2 !important; border-color: #fecaca; }
.r-label { font-size: 13px; font-weight: 700; display: block; margin-bottom: 2px; }
.r-pos   { color: #15803d !important; }
.r-neg   { color: #dc2626 !important; }
.r-meta  { font-size: 11px; color: #9ca3af !important; display: block; margin-bottom: 8px; }
.conf-track { background: #e5e7eb; border-radius: 3px; height: 4px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 3px; }

/* ---- Tabs ---- */
button[data-baseweb="tab"] { font-size: 13px !important; font-weight: 600 !important; color: #6b7280 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #2563eb !important; font-weight: 700 !important; }
div[data-baseweb="tab-highlight"] { background: #2563eb !important; }
div[data-baseweb="tab-list"] { border-bottom: 1px solid #e5e7eb !important; }

/* ---- Divider / footer ---- */
.divider { border: none; border-top: 1px solid #e8eaed; margin: 22px 0; }
.footer  { text-align: center; padding: 16px 0 8px; font-size: 11px; color: #d1d5db; }

/* ---- Empty state ---- */
.empty-state { text-align: center; padding: 48px 24px; color: #9ca3af; font-size: 13px; line-height: 1.7; }
.empty-state strong { display: block; font-size: 14px; font-weight: 700; color: #374151; margin-bottom: 6px; }
            
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    width: 100% !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    width: 100% !important;
    box-sizing: border-box !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:last-child {
    width: 100% !important;
    flex: 1 !important;
}
            
section[data-testid="stSidebar"] .stRadio {
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    width: 100% !important;
}
section[data-testid="stSidebar"] .stElementContainer {
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    width: 100% !important;
    padding: 0 !important;
}
                     
</style>

<script>
function applyButtonStyles() {
    document.querySelectorAll('[data-testid="stMain"] .stButton button').forEach(function(btn) {
        btn.style.setProperty('background-color', '#2563eb', 'important');
        btn.style.setProperty('color', '#ffffff', 'important');
        btn.style.setProperty('opacity', '1', 'important');
        btn.style.setProperty('border', 'none', 'important');
        btn.onmouseenter = function() {
            this.style.setProperty('background-color', '#1d4ed8', 'important');
            this.style.setProperty('color', '#ffffff', 'important');
        };
        btn.onmouseleave = function() {
            this.style.setProperty('background-color', '#2563eb', 'important');
            this.style.setProperty('color', '#ffffff', 'important');
        };
    });
    document.querySelectorAll('[data-testid="stTextInput"] input').forEach(function(inp) {
        inp.onfocus = function() {
            this.style.setProperty('border-color', '#2563eb', 'important');
            this.style.setProperty('box-shadow', '0 0 0 3px rgba(37,99,235,0.12)', 'important');
            var p = this.parentElement;
            while (p) { p.style.setProperty('box-shadow', 'none', 'important'); p = p.parentElement; if (p && p.dataset && p.dataset.testid === 'stTextInput') break; }
        };
        inp.onblur = function() {
            this.style.setProperty('border-color', '#e5e7eb', 'important');
            this.style.setProperty('box-shadow', 'none', 'important');
        };
    });
}
var _obs = new MutationObserver(function() { setTimeout(applyButtonStyles, 80); });
_obs.observe(document.body, { childList: true, subtree: true });
window.addEventListener('load', function() { setTimeout(applyButtonStyles, 300); });
</script>
""", unsafe_allow_html=True)


# ---- Constants ---------------------------------------------------------------
MODE_META = {
    "fp32":     {"pill": "pill-fp32", "color": "#2563eb", "track": "#dbeafe"},
    "fp16":     {"pill": "pill-fp16", "color": "#7c3aed", "track": "#ede9fe"},
    "int8_ptq": {"pill": "pill-ptq",  "color": "#059669", "track": "#d1fae5"},
    "int8_qat": {"pill": "pill-qat",  "color": "#d97706", "track": "#fef3c7"},
}
CHART_COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706"]
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#374151", family="Geist, sans-serif", size=11),
    margin=dict(t=12, b=4, l=0, r=0), height=280, showlegend=False,
)
AXIS = dict(gridcolor="#f3f4f6", zeroline=False, linecolor="#e5e7eb",
            tickfont=dict(size=11, color="#6b7280"))


# ---- Helpers -----------------------------------------------------------------
def pill_html(mode: str) -> str:
    key   = mode.replace("-", "_")
    meta  = MODE_META.get(key, {"pill": "", "color": "#888", "track": "#eee"})
    label = mode.upper().replace("_", "-")
    return f'<span class="pill {meta["pill"]}">{label}</span>'


def render_table(df: pd.DataFrame) -> str:
    cols   = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows   = ""
    for _, row in df.iterrows():
        cells = ""
        for i, c in enumerate(cols):
            val = str(row[c])
            if i == 0 and val.lower().replace("-", "_") in MODE_META:
                cells += f"<td>{pill_html(val)}</td>"
            else:
                cells += f"<td>{val}</td>"
        rows += f"<tr>{cells}</tr>"
    return f'<table class="data-table"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>'


def acc_bars_html(df: pd.DataFrame) -> str:
    max_acc   = df["accuracy"].max() if not df.empty else 1.0
    rows_html = ""
    for _, row in df.iterrows():
        key  = row["mode"].replace("-", "_")
        meta = MODE_META.get(key, {"color": "#888", "track": "#eee"})
        pct  = row["accuracy"] / max_acc * 100
        name = row["mode"].upper().replace("_", "-")
        rows_html += f"""
        <div class="acc-row">
            <div class="acc-mode">
                <div class="acc-dot" style="background:{meta['color']}"></div>
                <span class="acc-name">{name}</span>
            </div>
            <div class="acc-track" style="background:{meta['track']}">
                <div class="acc-fill" style="width:{pct:.1f}%;background:{meta['color']}"></div>
            </div>
            <span class="acc-val">{row['accuracy']:.1%}</span>
        </div>"""
    return f'<div class="acc-rows">{rows_html}</div>'


# ---- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-icon">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <rect x="1" y="1" width="5" height="5" rx="1.2" fill="white"/>
                <rect x="8" y="1" width="5" height="5" rx="1.2" fill="white" opacity="0.55"/>
                <rect x="1" y="8" width="5" height="5" rx="1.2" fill="white" opacity="0.55"/>
                <rect x="8" y="8" width="5" height="5" rx="1.2" fill="white" opacity="0.25"/>
            </svg>
        </div>
        <span class="sb-name">QuantBench</span>
    </div>
    <span class="sb-section">Analysis</span>
    """, unsafe_allow_html=True)

    selected = st.radio("nav", PAGES, index=PAGES.index(page), label_visibility="collapsed")
    if selected != page:
        st.session_state.page = selected
        st.rerun()

    st.markdown("""
    <span class="sb-section" style="margin-top:10px">Modes</span>
    <div style="padding:4px 20px 8px;display:flex;flex-direction:column;gap:7px;">
        <div style="display:flex;align-items:center;gap:8px;font-size:12.5px;color:#6b7280;">
            <span class="sb-dot" style="background:#2563eb"></span>FP32 baseline
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:12.5px;color:#6b7280;">
            <span class="sb-dot" style="background:#7c3aed"></span>FP16
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:12.5px;color:#6b7280;">
            <span class="sb-dot" style="background:#059669"></span>INT8-PTQ
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:12.5px;color:#6b7280;">
            <span class="sb-dot" style="background:#d97706"></span>INT8-QAT
        </div>
    </div>
    <div class="sb-footer">v1.0 &nbsp;·&nbsp; Model Quantization</div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE: Benchmark report
# ==============================================================================
if page == "Benchmark report":

    st.markdown("""
    <div class="page-title">Benchmark report</div>
    <div class="page-sub">FP32 &nbsp;·&nbsp; FP16 &nbsp;·&nbsp; INT8-PTQ &nbsp;·&nbsp; INT8-QAT &mdash; accuracy, latency, memory and robustness</div>
    """, unsafe_allow_html=True)

    # Prediction
    st.markdown('<span class="section-label">Single prediction</span>', unsafe_allow_html=True)
    col_pred, col_spacer = st.columns([1, 1.4])
    with col_pred:
        text = st.text_input("Input text", placeholder="Enter a sentence to classify...")
        mode_pred = st.selectbox("Precision mode", ["fp32", "fp16", "int8_ptq", "int8_qat"])

        if text != st.session_state.pred_last_text:
            st.session_state.pred_result    = None
            st.session_state.pred_last_text = text

        if st.button("Run inference", key="run_inference_btn"):
            if text.strip():
                with st.spinner("Running..."):
                    pipe   = Pipeline(precision=mode_pred)
                    result = pipe.predict(text)
                label_map = {0: "NEGATIVE", 1: "POSITIVE"}
                label = label_map.get(result["label"], str(result["label"]))
                st.session_state.pred_result = {
                    "label": label,
                    "conf":  result["confidence"],
                    "mode":  mode_pred,
                }
            else:
                st.warning("Please enter some text before running a prediction.")

        if st.session_state.pred_result:
            r    = st.session_state.pred_result
            css  = "result-pos" if r["label"] == "POSITIVE" else "result-neg"
            lcss = "r-pos"      if r["label"] == "POSITIVE" else "r-neg"
            fill = "#15803d"    if r["label"] == "POSITIVE" else "#dc2626"
            st.markdown(f"""
            <div class="result-card {css}">
                <span class="r-label {lcss}">{r['label']}</span>
                <span class="r-meta">{r['conf']:.1%} confidence &nbsp;&middot;&nbsp; {r['mode']}</span>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{r['conf']*100:.1f}%;background:{fill}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # KPI cards
    st.markdown('<span class="section-label">FP32 baseline metrics</span>', unsafe_allow_html=True)
    try:
        df   = pd.read_csv("outputs/results.csv")
        fp32 = df[df["mode"] == "fp32"].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, unit in [
            (c1, "Accuracy",    f"{fp32['accuracy']:.1%}",       ""),
            (c2, "Avg latency", f"{fp32['avg_latency_ms']:.1f}", "ms"),
            (c3, "P99 latency", f"{fp32['p99_latency_ms']:.1f}", "ms"),
            (c4, "Memory",      f"{fp32['memory_mb']:.0f}",      "MB"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{val}<span class="kpi-unit">{unit}</span></div>
                    <div class="kpi-sub">FP32 baseline</div>
                </div>
                """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Run `python scripts/run_benchmark.py` first to generate results.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Results table + accuracy bars
    st.markdown('<span class="section-label">All modes</span>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown('<div class="card"><div class="card-header"><span class="card-title">Results table</span><span class="card-meta">sorted by accuracy</span></div>', unsafe_allow_html=True)
        try:
            df   = pd.read_csv("outputs/results.csv")
            df_s = df.sort_values("accuracy", ascending=False)
            disp = df_s.copy()
            disp["accuracy"]       = df_s["accuracy"].map("{:.1%}".format)
            disp["avg_latency_ms"] = df_s["avg_latency_ms"].map("{:.1f} ms".format)
            disp["p99_latency_ms"] = df_s["p99_latency_ms"].map("{:.1f} ms".format)
            disp["memory_mb"]      = df_s["memory_mb"].map("{:.0f} MB".format)
            disp.columns           = ["Mode", "Accuracy", "Avg latency", "P99 latency", "Memory"]
            st.markdown('<div class="card-body-flush">' + render_table(disp) + '</div></div>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.markdown('<div class="card-body"><p style="font-size:12px;color:#9ca3af;">No results yet.</p></div></div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card"><div class="card-header"><span class="card-title">Accuracy by mode</span><span class="card-meta">% correct</span></div>', unsafe_allow_html=True)
        try:
            df = pd.read_csv("outputs/results.csv")
            st.markdown('<div class="card-body">' + acc_bars_html(df) + '</div></div>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.markdown('<div class="card-body"><p style="font-size:12px;color:#9ca3af;">No results yet.</p></div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Charts
    st.markdown('<span class="section-label">Latency &amp; memory</span>', unsafe_allow_html=True)
    try:
        df    = pd.read_csv("outputs/results.csv")
        modes = df["mode"].tolist()
        tab1, tab2, tab3 = st.tabs(["Accuracy", "Latency", "Memory"])

        with tab1:
            fig = go.Figure(go.Bar(
                x=modes, y=df["accuracy"], marker_color=CHART_COLORS, marker_line_width=0,
                text=[f"{v:.1%}" for v in df["accuracy"]], textposition="outside",
                textfont=dict(size=11, color="#6b7280"),
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, yaxis=dict(**AXIS, range=[0, 1.12], tickformat=".0%"), xaxis=dict(**AXIS))
            st.plotly_chart(fig, width="stretch")

        with tab2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Avg latency", x=modes, y=df["avg_latency_ms"],
                marker_color=CHART_COLORS, marker_line_width=0,
                text=[f"{v:.1f}" for v in df["avg_latency_ms"]], textposition="outside",
                textfont=dict(size=11, color="#6b7280"),
                hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} ms<extra></extra>",
            ))
            fig2.add_trace(go.Scatter(
                name="P99 latency", x=modes, y=df["p99_latency_ms"],
                mode="markers", marker=dict(symbol="diamond", size=9, color="#94a3b8"),
                hovertemplate="<b>%{x}</b><br>P99: %{y:.1f} ms<extra></extra>",
            ))
            fig2.update_layout(
                **{**CHART_LAYOUT, "showlegend": True},
                yaxis=dict(**AXIS, title="ms"), xaxis=dict(**AXIS),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#6b7280"),
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, width="stretch")

        with tab3:
            fig3 = go.Figure(go.Bar(
                x=modes, y=df["memory_mb"], marker_color=CHART_COLORS, marker_line_width=0,
                text=[f"{v:.0f} MB" for v in df["memory_mb"]], textposition="outside",
                textfont=dict(size=11, color="#6b7280"),
                hovertemplate="<b>%{x}</b><br>Memory: %{y:.0f} MB<extra></extra>",
            ))
            fig3.update_layout(**CHART_LAYOUT, yaxis=dict(**AXIS, title="MB"), xaxis=dict(**AXIS))
            st.plotly_chart(fig3, width="stretch")

        st.markdown('<span class="section-label" style="margin-top:4px">Raw data</span>', unsafe_allow_html=True)
        disp2 = df.copy()
        disp2["accuracy"]       = df["accuracy"].map("{:.1%}".format)
        disp2["avg_latency_ms"] = df["avg_latency_ms"].map("{:.2f} ms".format)
        disp2["p99_latency_ms"] = df["p99_latency_ms"].map("{:.2f} ms".format)
        disp2["memory_mb"]      = df["memory_mb"].map("{:.1f} MB".format)
        st.markdown('<div class="card"><div class="card-body-flush">' + render_table(disp2) + '</div></div>', unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("Run `python scripts/run_benchmark.py` first to generate results.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="footer">QuantBench &nbsp;&middot;&nbsp; Model Quantization Analysis Platform &nbsp;&middot;&nbsp; v1.0</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE: Robustness
# ==============================================================================
elif page == "Robustness":
    st.markdown("""
    <div class="page-title">Robustness</div>
    <div class="page-sub">Accuracy of each precision mode under different input perturbations.</div>
    """, unsafe_allow_html=True)
    try:
        rob_df  = pd.read_csv("outputs/robustness.csv")
        fig_rob = px.line(rob_df, x="perturbation", y="accuracy", color="mode",
                          markers=True, color_discrete_sequence=CHART_COLORS)
        fig_rob.update_traces(line=dict(width=2), marker=dict(size=6))
        fig_rob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#374151", family="Geist, sans-serif", size=11),
            yaxis=dict(range=[0, 1.05], gridcolor="#f3f4f6", tickformat=".0%",
                       title="Accuracy", zeroline=False, linecolor="#e5e7eb",
                       tickfont=dict(size=11, color="#6b7280")),
            xaxis=dict(gridcolor="#f3f4f6", title="Perturbation type",
                       zeroline=False, linecolor="#e5e7eb",
                       tickfont=dict(size=11, color="#6b7280")),
            legend=dict(bgcolor="rgba(0,0,0,0)", title="Mode", font=dict(size=11, color="#6b7280")),
            margin=dict(t=12, b=4, l=0, r=0), height=360,
        )
        st.markdown('<span class="section-label">Accuracy under perturbations</span>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-header"><span class="card-title">All modes</span><span class="card-meta">by perturbation type</span></div><div class="card-body">', unsafe_allow_html=True)
        st.plotly_chart(fig_rob, width="stretch")
        st.markdown('</div></div>', unsafe_allow_html=True)
        rob_disp = rob_df.copy()
        rob_disp["accuracy"] = rob_df["accuracy"].map("{:.2f}".format)
        st.markdown('<span class="section-label">Raw data</span>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-body-flush">' + render_table(rob_disp) + '</div></div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <div class="card"><div class="card-body">
            <div class="empty-state">
                <strong>No robustness data yet</strong>
                Run <code>python scripts/run_benchmark.py</code> to generate robustness results.
            </div>
        </div></div>""", unsafe_allow_html=True)


# ==============================================================================
# PAGE: Sensitivity
# ==============================================================================
elif page == "Sensitivity":
    st.markdown("""
    <div class="page-title">Layer sensitivity</div>
    <div class="page-sub">Per-layer accuracy delta when quantized individually to INT8.</div>
    """, unsafe_allow_html=True)
    try:
        from PIL import Image
        img = Image.open("outputs/sensitivity.png")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="card"><div class="card-header"><span class="card-title">Per-layer accuracy delta</span><span class="card-meta">INT8 quantization</span></div><div class="card-body">', unsafe_allow_html=True)
            st.image(img, width="stretch")
            st.markdown('<p style="text-align:center;font-size:11px;color:#9ca3af;margin-top:6px;">Positive delta = slight improvement. Zero delta = no accuracy loss.</p>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <div class="card"><div class="card-body">
            <div class="empty-state">
                <strong>No sensitivity data yet</strong>
                Run <code>python scripts/run_sensitivity.py</code> to generate sensitivity results.
            </div>
        </div></div>""", unsafe_allow_html=True)
