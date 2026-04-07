"""
Streamlit dashboard for quantization benchmark results.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from quant_pipeline.core.pipeline import Pipeline

st.set_page_config(
    page_title="Quantization Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

.stApp {
    background: #f8fafc;
    color: #0f172a;
}

/* Kill ALL purple/browser default focus outlines globally */
*:focus, *:focus-visible, *:focus-within {
    outline: none !important;
    box-shadow: none !important;
}

/* Hero panel */
.hero-panel {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 26px 40px 22px 40px;
    margin-bottom: 28px;
}
.hero-title {
    font-size: 1.4rem; font-weight: 800; color: #0f172a;
    letter-spacing: -0.025em; margin: 0 0 5px 0; line-height: 1.2;
}
.hero-desc { font-size: 0.875rem; color: #64748b; margin: 0; line-height: 1.6; }

header[data-testid="stHeader"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── Header ─────────────────────────────────────── */
.site-header {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 40px;
    height: 54px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
}
.site-header-left { display: flex; align-items: center; gap: 10px; }
.site-logo {
    width: 26px; height: 26px;
    background: #0f172a; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
}
.site-name { font-size: 0.88rem; font-weight: 600; color: #0f172a; }
.header-sep { width: 1px; height: 14px; background: #e2e8f0; }
.header-page { font-size: 0.82rem; color: #94a3b8; }
.site-badge {
    font-size: 0.67rem; font-weight: 500; color: #64748b;
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 20px; padding: 2px 8px;
}

/* ── Section label ──────────────────────────────── */
.section-label {
    font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: #374151; margin: 0 0 10px 0; display: block;
}

/* ── Metric cards ───────────────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 18px 20px;
    transition: box-shadow 0.15s, transform 0.15s;
}
.metric-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.06); transform: translateY(-1px); }
.m-label {
    font-size: 0.68rem; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 8px; display: block;
}
.m-value { font-size: 1.7rem; font-weight: 700; color: #0f172a; letter-spacing: -0.025em; line-height: 1; }
.m-unit { font-size: 0.78rem; color: #94a3b8; margin-left: 2px; }
.m-green { color: #16a34a; }

/* ── HTML table ─────────────────────────────────── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.85rem;
    margin-bottom: 8px;
}
.data-table thead tr {
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
}
.data-table thead th {
    padding: 11px 16px;
    text-align: left;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
}
.data-table tbody tr {
    border-bottom: 1px solid #f1f5f9;
    transition: background 0.1s;
}
.data-table tbody tr:last-child { border-bottom: none; }
.data-table tbody tr:hover { background: #f8fafc; }
.data-table tbody td {
    padding: 11px 16px;
    color: #374151;
    font-size: 0.855rem;
}
.data-table tbody td.idx { color: #94a3b8; font-size: 0.78rem; width: 40px; }
.data-table tbody td.num { font-variant-numeric: tabular-nums; }

/* ── Inputs ─────────────────────────────────────── */
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label {
    font-size: 0.82rem !important; font-weight: 600 !important; color: #1e293b !important;
}
div[data-testid="stTextInput"] input {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 7px !important;
    color: #0f172a !important;
    font-size: 0.875rem !important;
    outline: none !important;
    caret-color: #3b82f6 !important;
}
/* Fix browser autofill dark background */
div[data-testid="stTextInput"] input:-webkit-autofill,
div[data-testid="stTextInput"] input:-webkit-autofill:hover,
div[data-testid="stTextInput"] input:-webkit-autofill:focus,
div[data-testid="stTextInput"] input:-webkit-autofill:active {
    -webkit-box-shadow: 0 0 0 999px #ffffff inset !important;
    -webkit-text-fill-color: #0f172a !important;
    background-color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextInput"] input:focus-visible,
div[data-testid="stTextInput"] input:active {
    border: 1px solid #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    outline: none !important;
}
/* Target the BaseWeb input wrapper that causes the purple ring */
div[data-testid="stTextInput"] > div:focus-within,
div[data-testid="stTextInput"] > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    outline: none !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important; border: 1px solid #d1d5db !important;
    border-radius: 7px !important; color: #0f172a !important; font-size: 0.875rem !important;
}

/* Dropdown popup list */
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"],
div[data-baseweb="menu"] ul {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
}
div[data-baseweb="menu"] li,
div[data-baseweb="option"],
li[role="option"] {
    background: #ffffff !important;
    color: #0f172a !important;
    font-size: 0.875rem !important;
}
div[data-baseweb="menu"] li:hover,
div[data-baseweb="option"]:hover,
li[role="option"]:hover {
    background: #f1f5f9 !important;
    color: #0f172a !important;
}
li[aria-selected="true"],
div[data-baseweb="option"][aria-selected="true"] {
    background: #eff6ff !important;
    color: #3b82f6 !important;
}

/* ── Button ─────────────────────────────────────── */
.stButton > button {
    background: #0f172a !important; color: #ffffff !important;
    font-weight: 600 !important; font-size: 0.875rem !important;
    border: none !important; border-radius: 7px !important; padding: 10px 22px !important;
}
.stButton > button:hover { background: #1e293b !important; }

/* ── Result card ────────────────────────────────── */
.result-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px 20px; margin-top: 14px;
}
.result-pos { border-left: 4px solid #16a34a; }
.result-neg { border-left: 4px solid #dc2626; }
.r-label { font-size: 1rem; font-weight: 700; display: block; margin-bottom: 2px; }
.r-pos { color: #16a34a; }
.r-neg { color: #dc2626; }
.r-meta { font-size: 0.77rem; color: #94a3b8; display: block; margin-bottom: 8px; }
.conf-bg { background: #f1f5f9; border-radius: 3px; height: 4px; overflow: hidden; }
.conf-bar { height: 100%; border-radius: 3px; }

/* ── Tabs ───────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.82rem !important; font-weight: 500 !important;
    color: #64748b !important; padding: 9px 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #0f172a !important; font-weight: 700 !important; }
div[data-baseweb="tab-list"] { border-bottom: 1px solid #e2e8f0 !important; }

/* ── Divider / Footer ───────────────────────────── */
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }
.footer { text-align: center; padding: 14px 0 28px; font-size: 0.73rem; color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


def render_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a clean light HTML table."""
    cols = list(df.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for i, row in df.iterrows():
        cells = f'<td class="idx">{i}</td>'
        for c in cols:
            cells += f'<td class="num">{row[c]}</td>'
        rows += f"<tr>{cells}</tr>"
    return f"""
    <table class="data-table">
        <thead><tr><th></th>{header}</tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
    <div class="site-header-left">
        <div class="site-logo">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <rect x="1" y="1" width="5" height="5" rx="1" fill="white"/>
                <rect x="8" y="1" width="5" height="5" rx="1" fill="white" opacity="0.55"/>
                <rect x="1" y="8" width="5" height="5" rx="1" fill="white" opacity="0.55"/>
                <rect x="8" y="8" width="5" height="5" rx="1" fill="white" opacity="0.25"/>
            </svg>
        </div>
        <span class="site-name">QuantBench</span>
        <div class="header-sep"></div>
        <span class="header-page">Benchmark Report</span>
    </div>
    <span class="site-badge">v1.0</span>
</div>
""", unsafe_allow_html=True)

# ── Hero panel ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-panel">
    <p class="hero-title">Quantization Dashboard</p>
    <p class="hero-desc">Compare FP32, FP16, INT8-PTQ and INT8-QAT precision modes across accuracy, inference latency, memory footprint and input robustness.</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("outputs/results.csv")
    fp32 = df[df["mode"] == "fp32"].iloc[0]

    st.markdown('<span class="section-label">Baseline — FP32</span>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <span class="m-label">Accuracy</span>
            <span class="m-value m-green">{fp32['accuracy']:.1%}</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <span class="m-label">Avg Latency</span>
            <span class="m-value">{fp32['avg_latency_ms']:.1f}</span><span class="m-unit">ms</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <span class="m-label">P99 Latency</span>
            <span class="m-value">{fp32['p99_latency_ms']:.1f}</span><span class="m-unit">ms</span>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <span class="m-label">Memory</span>
            <span class="m-value">{fp32['memory_mb']:.0f}</span><span class="m-unit">MB</span>
        </div>""", unsafe_allow_html=True)

except FileNotFoundError:
    st.warning("Run `python scripts/run_benchmark.py` first to generate results.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
st.markdown('<span class="section-label">Single Prediction</span>', unsafe_allow_html=True)

text = st.text_input("Input Text", placeholder="Enter a sentence to classify...")
mode = st.selectbox("Precision Mode", ["fp32", "fp16", "int8_ptq", "int8_qat"])

if st.button("Run Prediction"):
    if text.strip():
        with st.spinner("Running inference..."):
            pipe = Pipeline(precision=mode)
            result = pipe.predict(text)
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        label = label_map.get(result['label'], str(result['label']))
        conf  = result['confidence']
        css   = "result-pos" if label == "POSITIVE" else "result-neg"
        lcss  = "r-pos"      if label == "POSITIVE" else "r-neg"
        fill  = "#16a34a"    if label == "POSITIVE" else "#dc2626"
        st.markdown(f"""
        <div class="result-card {css}">
            <span class="r-label {lcss}">{label}</span>
            <span class="r-meta">Confidence: {conf:.1%} &nbsp;&middot;&nbsp; Mode: {mode}</span>
            <div class="conf-bg">
                <div class="conf-bar" style="width:{conf*100:.1f}%;background:{fill}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text before running a prediction.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown('<span class="section-label">Benchmark Results</span>', unsafe_allow_html=True)

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#374151", family="Inter", size=12),
    margin=dict(t=16, b=8, l=0, r=0), height=300, showlegend=False,
)
AXIS = dict(gridcolor="#e2e8f0", zeroline=False, linecolor="#cbd5e1",
            tickfont=dict(size=12, color="#374151"))

try:
    df = pd.read_csv("outputs/results.csv")
    modes  = df["mode"].tolist()
    COLORS = ["#3b82f6", "#6366f1", "#8b5cf6", "#0ea5e9"]

    tab1, tab2, tab3 = st.tabs(["Accuracy", "Latency", "Memory"])

    with tab1:
        fig = go.Figure(go.Bar(
            x=modes, y=df["accuracy"], marker_color=COLORS, marker_line_width=0,
            text=[f"{v:.1%}" for v in df["accuracy"]], textposition="outside",
            textfont=dict(size=11, color="#374151"),
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>",
        ))
        fig.update_layout(**LAYOUT, yaxis=dict(**AXIS, range=[0,1.12], tickformat=".0%"), xaxis=dict(**AXIS))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Avg Latency", x=modes, y=df["avg_latency_ms"],
            marker_color="#3b82f6", marker_line_width=0,
            text=[f"{v:.1f}" for v in df["avg_latency_ms"]], textposition="outside",
            textfont=dict(size=11, color="#374151"),
            hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} ms<extra></extra>",
        ))
        fig2.add_trace(go.Bar(
            name="P99 Latency", x=modes, y=df["p99_latency_ms"],
            marker_color="#bfdbfe", marker_line_width=0,
            text=[f"{v:.1f}" for v in df["p99_latency_ms"]], textposition="outside",
            textfont=dict(size=11, color="#374151"),
            hovertemplate="<b>%{x}</b><br>P99: %{y:.1f} ms<extra></extra>",
        ))
        fig2.update_layout(**{**LAYOUT, "showlegend": True}, barmode="group",
                           yaxis=dict(**AXIS, title="ms"), xaxis=dict(**AXIS),
                           legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#64748b"),
                                       orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure(go.Bar(
            x=modes, y=df["memory_mb"], marker_color=COLORS, marker_line_width=0,
            text=[f"{v:.0f} MB" for v in df["memory_mb"]], textposition="outside",
            textfont=dict(size=11, color="#374151"),
            hovertemplate="<b>%{x}</b><br>Memory: %{y:.1f} MB<extra></extra>",
        ))
        fig3.update_layout(**LAYOUT, yaxis=dict(**AXIS, title="MB"), xaxis=dict(**AXIS))
        st.plotly_chart(fig3, use_container_width=True)

    # Format display copy
    disp = df.copy()
    disp["accuracy"]       = df["accuracy"].map("{:.1%}".format)
    disp["avg_latency_ms"] = df["avg_latency_ms"].map("{:.2f} ms".format)
    disp["p99_latency_ms"] = df["p99_latency_ms"].map("{:.2f} ms".format)
    disp["memory_mb"]      = df["memory_mb"].map("{:.1f} MB".format)

    st.markdown('<span class="section-label" style="margin-top:8px">Raw Data</span>', unsafe_allow_html=True)
    st.markdown(render_table(disp), unsafe_allow_html=True)

except FileNotFoundError:
    st.warning("Run `python scripts/run_benchmark.py` first to generate results.")

# ── Robustness ────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<span class="section-label">Robustness Under Perturbations</span>', unsafe_allow_html=True)

try:
    rob_df = pd.read_csv("outputs/robustness.csv")
    fig_rob = px.line(
        rob_df, x="perturbation", y="accuracy", color="mode", markers=True,
        color_discrete_sequence=["#3b82f6", "#6366f1", "#0ea5e9", "#8b5cf6"],
    )
    fig_rob.update_traces(line=dict(width=2), marker=dict(size=6))
    fig_rob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#374151", family="Inter", size=12),
        yaxis=dict(range=[0,1.05], gridcolor="#e2e8f0", tickformat=".0%",
                   title="Accuracy", zeroline=False, linecolor="#cbd5e1",
                   tickfont=dict(size=12, color="#374151")),
        xaxis=dict(gridcolor="#e2e8f0", title="Perturbation Type",
                   zeroline=False, linecolor="#cbd5e1",
                   tickfont=dict(size=12, color="#374151")),
        legend=dict(bgcolor="rgba(0,0,0,0)", title="Mode", font=dict(size=11, color="#64748b")),
        margin=dict(t=16, b=8, l=0, r=0), height=320,
    )
    st.plotly_chart(fig_rob, use_container_width=True)

    rob_disp = rob_df.copy()
    rob_disp["accuracy"] = rob_df["accuracy"].map("{:.2f}".format)
    st.markdown('<span class="section-label" style="margin-top:8px">Raw Robustness Data</span>', unsafe_allow_html=True)
    st.markdown(render_table(rob_disp), unsafe_allow_html=True)

except FileNotFoundError:
    st.info("Robustness data will appear after running the benchmark.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="footer">QuantBench &nbsp;&middot;&nbsp; Model Quantization Analysis Platform &nbsp;&middot;&nbsp; v1.0</div>', unsafe_allow_html=True)
