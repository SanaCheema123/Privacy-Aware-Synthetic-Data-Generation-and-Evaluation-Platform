"""Visual style for the PrivSynth dashboard: CSS, navbar, cards, badges and footer."""
from __future__ import annotations

import base64
from pathlib import Path

GREEN = "#16a34a"
DARK = "#0b0f0d"
RISK_COLORS = {"low": "#16a34a", "medium": "#ea8a0c", "high": "#dc2626", "unknown": "#6b7280"}
GEN_COLORS = {"Gaussian Copula": "#2563eb", "CTGAN": "#ea8a0c", "TVAE": "#0d9488", "TabDDPM": "#9333ea"}

CSS = """
<style>
/* hide default Streamlit chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebarCollapsedControl"] {display:none !important;}
.block-container {padding-top: 0.6rem !important; padding-bottom: 1rem; max-width: 1320px;}

/* ---------- navbar ---------- */
.st-key-navbar {
    background: linear-gradient(90deg, #070a08 0%, #0f1a13 60%, #11261a 100%);
    border-radius: 14px; padding: 10px 18px; margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,.18); position: sticky; top: 6px; z-index: 999;
}
.st-key-navbar [data-testid="stHorizontalBlock"] {align-items: center;}
.brand {display:flex; align-items:center; gap:12px;}
.brand img {height:44px;}
.brand .logo-fallback {height:44px; width:44px; border-radius:10px; background:#16a34a; color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:800; font-size:20px;}
.brand .t1 {color:#fff; font-weight:800; font-size:21px; letter-spacing:.3px; line-height:1.1;}
.brand .t2 {color:#86efac; font-size:11.5px; letter-spacing:.4px;}
.st-key-navbar button {
    background: transparent !important; border: 1px solid transparent !important; color:#d1fae5 !important;
    border-radius: 9px !important; font-weight:600 !important; padding: 6px 4px !important; min-height: 38px;
    white-space: nowrap !important;
}
.st-key-navbar button p {white-space: nowrap !important; font-size: 14.5px !important;}
.st-key-navbar button:hover {background: rgba(34,197,94,.14) !important; color:#fff !important;}
.st-key-navbar button[kind="primary"] {
    background: #16a34a !important; color:#fff !important; box-shadow: 0 2px 8px rgba(22,163,74,.45);
}
.status-chip {text-align:right; color:#a7f3d0; font-size:12px;}
.status-chip b {color:#fff;}

/* ---------- hero ---------- */
.hero {background: linear-gradient(120deg, #0b0f0d 0%, #14532d 100%); color:#fff; border-radius:16px;
       padding: 34px 38px; margin-bottom: 18px;}
.hero h1 {color:#fff; font-size: 34px; margin: 0 0 8px 0;}
.hero p {color:#d1fae5; font-size: 16px; max-width: 820px; margin:0;}
.hero .tag {display:inline-block; background: rgba(134,239,172,.16); color:#86efac; border:1px solid rgba(134,239,172,.35);
            border-radius: 20px; padding: 3px 12px; font-size: 12px; margin-bottom: 12px;}

/* ---------- cards ---------- */
.card {background:#fff; border:1px solid #e3ebe6; border-radius:14px; padding:18px 20px; height:100%;
       box-shadow: 0 1px 3px rgba(16,40,24,.06);}
.card h4 {margin: 6px 0 6px 0; color:#14532d; font-size:16px;}
.card p {color:#4b5b52; font-size:13.5px; margin:0;}
.card .icon {font-size: 24px;}
.kpi {background:#fff; border:1px solid #e3ebe6; border-left: 5px solid #16a34a; border-radius:12px;
      padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,40,24,.06);}
.kpi .label {color:#5f6f66; font-size:12.5px; text-transform: uppercase; letter-spacing: .5px;}
.kpi .value {color:#0b0f0d; font-size:26px; font-weight:800; margin-top:2px;}
.kpi .sub {color:#6b7c72; font-size:12px;}
.section-title {font-size: 20px; font-weight: 800; color:#0f2a1b; margin: 22px 0 4px 0;}
.section-sub {color:#5f6f66; font-size: 13.5px; margin-bottom: 12px;}
.badge {display:inline-block; color:#fff; border-radius:6px; padding: 2px 10px; font-size:12px; font-weight:700;}
.step {display:flex; gap:12px; align-items:flex-start; margin-bottom:12px;}
.step .num {background:#16a34a; color:#fff; border-radius:50%; width:28px; height:28px; flex: 0 0 28px;
            display:flex; align-items:center; justify-content:center; font-weight:700; font-size:13px;}
.step .txt b {color:#0f2a1b;} .step .txt span {color:#5f6f66; font-size:13px;}
.risk-card {background:#fff; border-radius:14px; padding:18px; border:1px solid #e3ebe6; text-align:center;}
.risk-card .name {font-weight:800; font-size:17px; color:#0f2a1b;}
.risk-card .lvl {font-size:22px; font-weight:800; margin: 6px 0;}

/* ---------- widgets ---------- */
div[data-testid="stTabs"] button[role="tab"] {font-weight:600;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#16a34a;}
.stButton > button[kind="primary"], .stDownloadButton > button {border-radius: 10px; font-weight: 700;}
[data-testid="stFileUploader"] section {border: 2px dashed #86efac; background:#f0fdf4; border-radius:12px;}
[data-testid="stExpander"] {background:#fff; border-radius:12px;}

/* ---------- footer ---------- */
.app-footer {margin-top: 40px; padding: 18px; text-align:center; color:#6b7c72; font-size:12.5px;
             border-top:1px solid #dfe8e2;}
.app-footer b {color:#14532d;}
</style>
"""


def logo_html(logo: Path | None, height: int = 44) -> str:
    if logo and Path(logo).exists():
        b64 = base64.b64encode(Path(logo).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="height:{height}px" alt="AIVONEX logo">'
    return '<div class="logo-fallback">A</div>'


def brand_block(logo: Path | None) -> str:
    return (f'<div class="brand">{logo_html(logo)}<div><div class="t1">PrivSynth</div>'
            f'<div class="t2">AIVONEX · Privacy Aware Synthetic Data</div></div></div>')


def kpi(label: str, value: str, sub: str = "", color: str = GREEN) -> str:
    return (f'<div class="kpi" style="border-left-color:{color}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def card(icon: str, title: str, text: str) -> str:
    return f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>'


def badge(level: str) -> str:
    return f'<span class="badge" style="background:{RISK_COLORS.get(level, "#6b7280")}">{level.upper()}</span>'


def section(title: str, sub: str = "") -> str:
    return f'<div class="section-title">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else "")


def step(n: int, title: str, text: str) -> str:
    return f'<div class="step"><div class="num">{n}</div><div class="txt"><b>{title}</b><br><span>{text}</span></div></div>'


FOOTER = ('<div class="app-footer"><b>PrivSynth</b> · Privacy Aware Synthetic Data Generation and Evaluation Platform'
          '<br>© AIVONEX SMC PVT LTD · Bahawalpur, Pakistan</div>')
