"""PrivSynth web dashboard (Streamlit) with a branded navigation bar.

Run from the project root:
    python -m streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import ui_style as ui  # noqa: E402
from privsynth.config import load_config, logo_path  # noqa: E402
from privsynth.data import detect_schema, load_csv  # noqa: E402
from privsynth.generators import DISPLAY_NAMES  # noqa: E402
from privsynth.imbalance import class_distribution, imbalance_ratio  # noqa: E402
from privsynth.pipeline import run_pipeline  # noqa: E402

st.set_page_config(page_title="PrivSynth | AIVONEX", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)

CFG = load_config()
LOGO = logo_path(CFG)
PAGES = ["Home", "Data", "Generate", "Results", "Privacy", "Export", "About"]
ICONS = {"Home": "🏠", "Data": "📂", "Generate": "🚀", "Results": "📊", "Privacy": "🛡️", "Export": "⬇️", "About": "ℹ️"}
GEN_INFO = {
    "gaussian_copula": ("📐", "Statistical baseline. Models each column's distribution and links them with a Gaussian copula. Very fast."),
    "ctgan": ("⚔️", "Conditional GAN with mode specific normalisation. Strong on mixed and multimodal columns."),
    "tvae": ("🧬", "Variational autoencoder for tables. Quick to train, can stay close to real records."),
    "tabddpm": ("🌫️", "Diffusion model with class conditioning. Usually the best fidelity and native class balancing."),
}
PLOT_LAYOUT = dict(template="plotly_white", margin=dict(l=20, r=20, t=60, b=20), font=dict(size=13),
                   title_font=dict(size=16, color="#0f2a1b"), title_x=0.01,
                   legend=dict(orientation="h", yanchor="top", y=-0.14, x=0.5, xanchor="center"))

ss = st.session_state
ss.setdefault("page", "Home")


# ======================================================================
# navbar
# ======================================================================
def go_to(page: str) -> None:
    ss.page = page


def load_uploaded() -> None:
    up = ss.get("uploader")
    if up is not None:
        ss.df = load_csv(io.BytesIO(up.getvalue()))
        ss.data_name = up.name
        ss.pop("run", None)


def load_demo() -> None:
    demo = ROOT / CFG["data"]["input_csv"]
    ss.df = load_csv(demo)
    ss.data_name = demo.name
    ss.pop("run", None)


with st.container(key="navbar"):
    cols = st.columns([3.0] + [1.0] * len(PAGES) + [1.7], gap="small")
    cols[0].markdown(ui.brand_block(LOGO), unsafe_allow_html=True)
    for c, p in zip(cols[1:-1], PAGES):
        c.button(f"{ICONS[p]} {p}", key=f"nav_{p}", use_container_width=True,
                 type="primary" if ss.page == p else "secondary", on_click=go_to, args=(p,))
    data_name = ss.get("data_name", "no dataset")
    run_txt = ss["run"]["run_id"].replace("run_", "") if "run" in ss else "no run yet"
    cols[-1].markdown(f'<div class="status-chip">Data: <b>{data_name}</b><br>Run: <b>{run_txt}</b></div>',
                      unsafe_allow_html=True)


def need_data() -> bool:
    if "df" not in ss:
        st.info("No dataset loaded yet. Open the **Data** page to upload a CSV or load the demo dataset.")
        st.button("Go to Data", on_click=go_to, args=("Data",), type="primary")
        return False
    return True


def need_run() -> bool:
    if "run" not in ss:
        st.info("No results yet. Configure and start a run on the **Generate** page.")
        st.button("Go to Generate", on_click=go_to, args=("Generate",), type="primary")
        return False
    return True


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


# ======================================================================
# pages
# ======================================================================
def page_home() -> None:
    html('<div class="hero"><span class="tag">Generative AI · Privacy · Evaluation</span>'
         '<h1>Privacy Aware Synthetic Data Platform</h1>'
         '<p>Create realistic synthetic versions of sensitive tabular datasets, then prove they are useful and safe: '
         'statistical fidelity, downstream machine learning performance and privacy attack resistance in one report.</p></div>')
    c = st.columns(4)
    c[0].markdown(ui.card("🤖", "4 Generators", "CTGAN, TVAE, Gaussian Copula and TabDDPM trained and compared side by side."), unsafe_allow_html=True)
    c[1].markdown(ui.card("⚖️", "Class Balancing", "Keep, balance or set custom class ratios using conditional sampling."), unsafe_allow_html=True)
    c[2].markdown(ui.card("📈", "Fidelity and Utility", "Distribution tests, correlation checks and train on synthetic, test on real models."), unsafe_allow_html=True)
    c[3].markdown(ui.card("🛡️", "Privacy Risk", "Exact matches, distance to closest record, disclosure and membership inference attacks."), unsafe_allow_html=True)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        html(ui.section("How it works"))
        for i, (t, d) in enumerate([
            ("Upload data", "CSV upload with automatic schema detection; IDs and free text are removed."),
            ("Hold out real rows", "A share of real rows is hidden from every generator for fair testing."),
            ("Train generators", "Each selected model learns the training data and produces synthetic rows."),
            ("Evaluate", "Fidelity, downstream utility and privacy risk are measured for each model."),
            ("Export", "Download synthetic CSVs, the HTML report, metrics and figures."),
        ], start=1):
            html(ui.step(i, t, d))
    with right:
        html(ui.section("Current session"))
        k = st.columns(2)
        if "df" in ss:
            k[0].markdown(ui.kpi("Dataset", ss["data_name"], f"{len(ss['df']):,} rows"), unsafe_allow_html=True)
        else:
            k[0].markdown(ui.kpi("Dataset", "None", "load one on the Data page", "#6b7280"), unsafe_allow_html=True)
        if "run" in ss:
            b = ss["run"]["leaderboard"].iloc[0]
            k[1].markdown(ui.kpi("Best generator", b["generator"], f"overall score {b['overall_score']:.3f}"), unsafe_allow_html=True)
        else:
            k[1].markdown(ui.kpi("Best generator", "None", "no run yet", "#6b7280"), unsafe_allow_html=True)
        st.write("")
        st.button("🚀  Start with your data", on_click=go_to, args=("Data",), type="primary", use_container_width=True)


def page_data() -> None:
    html(ui.section("Dataset", "Upload a CSV file or use the built in demo clinical dataset."))
    up_col, demo_col = st.columns([2.2, 1], gap="large")
    with up_col:
        st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", key="uploader",
                         on_change=load_uploaded)
    with demo_col:
        demo = ROOT / CFG["data"]["input_csv"]
        st.write("")
        st.button("📋  Load demo dataset", use_container_width=True, disabled=not demo.exists(), on_click=load_demo)
        st.caption("Demo: 4,000 artificial patients, imbalanced outcome." if demo.exists()
                   else "Run scripts\\make_sample_data.py to create the demo.")
    if "df" not in ss:
        return
    df = ss.df

    cols = list(df.columns)
    default_t = ss.get("target") or (CFG["data"]["target_column"] if CFG["data"]["target_column"] in cols else cols[-1])
    s1, s2 = st.columns([1, 2], gap="large")
    t = s1.selectbox("🎯 Target column", ["(none)"] + cols,
                     index=(cols.index(default_t) + 1) if default_t in cols else 0,
                     help="Used for class balancing and downstream model evaluation")
    ss.target = None if t == "(none)" else t
    ss.drop = s2.multiselect("🗑️ Columns to drop (names, emails, free text)", cols,
                             default=[c for c in ss.get("drop", []) if c in cols])
    schema = detect_schema(df, ss.target, int(CFG["data"]["max_categories"]), ss.drop)
    ss.schema = schema

    k = st.columns(5)
    k[0].markdown(ui.kpi("Rows", f"{len(df):,}"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Columns", f"{df.shape[1]}", f"{len(schema.modelled)} used"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Numerical", f"{len(schema.numerical)}", "incl. dates", "#2563eb"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Categorical", f"{len(schema.categorical)}", "", "#9333ea"), unsafe_allow_html=True)
    k[4].markdown(ui.kpi("Missing cells", f"{df.isna().mean().mean() * 100:.1f}%", "", "#ea8a0c"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔎 Schema", "📄 Preview", "⚖️ Class balance"])
    with tab1:
        sf = schema.summary_frame()
        html(ui.html_table(sf, progress={"missing_%": (0, 100)}, formats={"missing_%": "{:.1f}"}, max_height=460))
        if schema.dropped:
            st.caption(f"Dropped automatically or by you: {', '.join(schema.dropped)}")
    with tab2:
        html(ui.html_table(df.head(100), max_height=420))
    with tab3:
        if schema.task == "classification":
            d = df.dropna(subset=[ss.target])
            dist = class_distribution(d, ss.target)
            a, b = st.columns([2, 1], gap="large")
            fig = go.Figure(go.Bar(x=dist["class"], y=dist["count"], marker_color="#16a34a",
                                   text=[f"{s}%" for s in dist["share_%"]], textposition="outside"))
            fig.update_layout(**PLOT_LAYOUT, title=f"Class distribution of '{ss.target}'", height=340)
            a.plotly_chart(fig, use_container_width=True)
            r = imbalance_ratio(d, ss.target)
            b.markdown(ui.kpi("Imbalance ratio", f"{r:.1f} : 1", "majority / minority",
                              "#dc2626" if r > 4 else "#16a34a"), unsafe_allow_html=True)
            if r > 4:
                b.warning("Strong imbalance. Consider **balance** on the Generate page.")
        elif schema.task == "regression":
            st.info("Numeric target: downstream evaluation will use regression metrics (R², MAE).")
        else:
            st.info("Select a target column to see class balance.")
    st.button("Next: configure generation ➜", on_click=go_to, args=("Generate",), type="primary")


def page_generate() -> None:
    if not need_data():
        return
    schema = ss.schema
    html(ui.section("Generators", "Select the models to train and compare."))
    chosen = []
    cols = st.columns(4)
    for c, key in zip(cols, DISPLAY_NAMES):
        icon, text = GEN_INFO[key]
        with c.container(border=True):
            html(f"<div style='font-size:24px'>{icon}</div><b style='font-size:16px;color:#14532d'>{DISPLAY_NAMES[key]}</b>"
                 f"<p style='color:#5f6f66;font-size:13px;min-height:72px'>{text}</p>")
            if st.toggle("Use", value=key in CFG["generators"]["enabled"], key=f"use_{key}"):
                chosen.append(key)

    html(ui.section("Settings"))
    left, right = st.columns(2, gap="large")
    with left.container(border=True):
        st.markdown("**🧠 Training**")
        epochs = st.slider("CTGAN / TVAE epochs", 10, 1000, int(CFG["generators"]["ctgan"]["epochs"]), 10)
        steps = st.slider("TabDDPM training steps", 200, 20000, int(CFG["generators"]["tabddpm"]["steps"]), 200)
        holdout = st.slider("Holdout share (never seen by generators)", 0.10, 0.50,
                            float(CFG["data"]["holdout_size"]), 0.05)
    with right.container(border=True):
        st.markdown("**⚖️ Class imbalance and size**")
        is_cls = schema.task == "classification"
        strategy = st.radio("Strategy", ["none", "balance", "custom"], horizontal=True, disabled=not is_cls)
        st.caption("none keeps the real proportions · balance makes all classes equal · custom uses your ratios")
        n_rows = st.number_input("Synthetic rows (0 = same as real training rows)", 0, 1_000_000, 0, 500)
        ratios = {}
        if strategy == "custom" and is_cls:
            cats = schema.columns[ss.target].categories
            rc = st.columns(min(len(cats), 4))
            for i, cls in enumerate(cats):
                ratios[cls] = rc[i % len(rc)].number_input(f"{ss.target} = {cls}", 0.0, 1.0,
                                                           round(1 / len(cats), 2), 0.05, key=f"ratio_{cls}")

    st.write("")
    if st.button(f"🚀  Generate and evaluate ({len(chosen)} models)", type="primary",
                 disabled=not chosen, use_container_width=True):
        overrides = {
            "data": {"target_column": ss.target, "holdout_size": holdout, "drop_columns": ss.drop},
            "generators": {"enabled": chosen, "ctgan": {"epochs": int(epochs)}, "tvae": {"epochs": int(epochs)},
                           "tabddpm": {"steps": int(steps)}},
            "imbalance": {"strategy": strategy, "num_rows": int(n_rows) or None, "custom_ratios": ratios},
        }
        with st.status("Running the pipeline ...", expanded=True) as status:
            bar = st.progress(0.0)

            def prog(msg: str, frac: float) -> None:
                bar.progress(min(frac, 1.0))
                status.write(f"• {msg}")

            try:
                ss.run = run_pipeline(CFG, overrides, data=ss.df, progress=prog)
                status.update(label=f"Finished {ss.run['run_id']}", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Run failed", state="error")
                st.error(str(e))
                return
        st.success("Done. Your results are ready.")
        st.button("View results ➜", on_click=go_to, args=("Results",), type="primary")


def _score_bars(board: pd.DataFrame) -> go.Figure:
    metrics = ["fidelity_score", "utility_score", "privacy_score", "overall_score"]
    fig = go.Figure()
    for _, r in board.iterrows():
        fig.add_bar(name=r["generator"], x=[m.replace("_score", "").title() for m in metrics],
                    y=[r[m] for m in metrics], marker_color=ui.GEN_COLORS.get(r["generator"]))
    fig.update_layout(**PLOT_LAYOUT, barmode="group", yaxis=dict(range=[0, 1.05], title="score"),
                      title="Scores by generator (higher is better)", height=420)
    return fig


def _radar(board: pd.DataFrame) -> go.Figure:
    axes = ["Fidelity", "Utility", "Privacy", "Realism"]
    fig = go.Figure()
    for _, r in board.iterrows():
        realism = 1 - 2 * abs(r["detection_auc"] - 0.5)
        vals = [r["fidelity_score"], np.nan_to_num(r["utility_score"]), r["privacy_score"], realism]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], fill="toself", name=r["generator"],
                                      line_color=ui.GEN_COLORS.get(r["generator"]), opacity=0.65))
    fig.update_layout(**PLOT_LAYOUT, polar=dict(radialaxis=dict(range=[0, 1], tickfont=dict(size=10))),
                      title="Trade off profile", height=420)
    fig.update_layout(legend=dict(y=-0.08))
    return fig


def page_results() -> None:
    if not need_run():
        return
    run = ss.run
    board = run["leaderboard"]
    best = board.iloc[0]
    html(ui.section("Results overview", f"Run {run['run_id']} · {run['n_train']:,} training rows · "
                                        f"{run['n_holdout']:,} holdout rows · strategy: {run['imbalance_strategy']}"))
    k = st.columns(4)
    k[0].markdown(ui.kpi("Recommended", best["generator"], "highest overall score"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Overall score", f"{best['overall_score']:.3f}", "0.35 F + 0.35 U + 0.30 P"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Detection AUC", f"{best['detection_auc']:.3f}", "0.5 = indistinguishable", "#2563eb"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Privacy risk", best["privacy_risk"].upper(), f"MIA AUC {best['mia_auc']:.3f}",
                         ui.RISK_COLORS.get(best["privacy_risk"])), unsafe_allow_html=True)
    if run["failed"]:
        st.warning(f"Failed generators: {run['failed']}")

    a, b = st.columns([1.4, 1], gap="large")
    a.plotly_chart(_score_bars(board), use_container_width=True)
    b.plotly_chart(_radar(board), use_container_width=True)
    show = board.drop(columns=["key"]).copy()
    sc = ["fidelity_score", "utility_score", "privacy_score", "overall_score"]
    html(ui.html_table(show, progress={c: (0, 1) for c in sc}, badges=["privacy_risk"],
                       formats={"detection_auc": "{:.3f}", "mia_auc": "{:.3f}", "train_seconds": "{:.1f}"}))

    html(ui.section("Generator details"))
    pick = st.radio("Generator", list(run["results"]), horizontal=True,
                    format_func=lambda k: run["results"][k]["display_name"])
    r = run["results"][pick]
    for n in r.get("notes", []):
        st.warning(n)
    t1, t2, t3 = st.tabs(["📈 Fidelity", "🤖 Downstream utility", "🖼️ Figures"])
    with t1:
        f = r["fidelity"]
        k = st.columns(4)
        k[0].markdown(ui.kpi("Column shapes", f"{f['column_shape_score']:.3f}"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Pair trends", f"{f['pair_trend_score']:.3f}"), unsafe_allow_html=True)
        k[2].markdown(ui.kpi("Detection AUC", f"{f['detection_auc']:.3f}", "0.5 ideal", "#2563eb"), unsafe_allow_html=True)
        k[3].markdown(ui.kpi("Fidelity score", f"{f['fidelity_score']:.3f}"), unsafe_allow_html=True)
        ct = f["column_table"].sort_values("score")
        fig = go.Figure(go.Bar(x=ct["score"], y=ct["column"], orientation="h",
                               marker_color=["#2563eb" if t == "numerical" else "#9333ea" for t in ct["type"]],
                               text=[f"{v:.3f}" for v in ct["score"]], textposition="outside"))
        fig.update_layout(**PLOT_LAYOUT, title="Per column similarity (blue numerical, purple categorical)",
                          xaxis=dict(range=[0, 1.08]), height=max(320, 28 * len(ct)))
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        u = r["utility"]
        if u.get("available"):
            k = st.columns(3)
            k[0].markdown(ui.kpi(f"Real model ({u['primary_metric']})", f"{u['trtr_primary']:.3f}", "train real, test real", "#6b7280"), unsafe_allow_html=True)
            k[1].markdown(ui.kpi(f"Synthetic model ({u['primary_metric']})", f"{u['tstr_primary']:.3f}", "train synthetic, test real"), unsafe_allow_html=True)
            k[2].markdown(ui.kpi("Utility ratio", f"{u['utility_ratio']:.3f}", "1.0 = no loss"), unsafe_allow_html=True)
            tb = u["table"]
            prim = tb[tb.metric == u["primary_metric"]]
            fig = go.Figure([go.Bar(name="Real (TRTR)", x=prim["model"], y=prim["TRTR"], marker_color="#6b7280"),
                             go.Bar(name="Synthetic (TSTR)", x=prim["model"], y=prim["TSTR"], marker_color="#16a34a")])
            fig.update_layout(**PLOT_LAYOUT, barmode="group", title=f"{u['primary_metric']} on the real holdout set", height=360)
            st.plotly_chart(fig, use_container_width=True)
            html(ui.html_table(tb, formats={"TRTR": "{:.4f}", "TSTR": "{:.4f}"}))
        else:
            st.info("Select a target column on the Data page to enable downstream evaluation.")
    with t3:
        figs = r["figures"]
        for key, cap in [("distributions", "Real versus synthetic distributions"), ("association", "Column associations"),
                         ("pca", "PCA projection")]:
            if figs.get(key):
                st.image(str(figs[key]), caption=cap, use_container_width=True)


def page_privacy() -> None:
    if not need_run():
        return
    run = ss.run
    html(ui.section("Privacy risk report", "Empirical attacks against the synthetic data, using real holdout people as the control group."))
    cols = st.columns(len(run["results"]))
    for c, (key, r) in zip(cols, run["results"].items()):
        p = r["privacy"]
        color = ui.RISK_COLORS[p["overall_risk"]]
        c.markdown(f'<div class="risk-card" style="border-top:5px solid {color}"><div class="name">{r["display_name"]}</div>'
                   f'<div class="lvl" style="color:{color}">{p["overall_risk"].upper()} RISK</div>'
                   f'<div style="color:#5f6f66;font-size:13px">privacy score {p["privacy_score"]:.3f}</div></div>',
                   unsafe_allow_html=True)

    rows = []
    for key, r in run["results"].items():
        p = r["privacy"]
        rows.append({"generator": r["display_name"], "exact match %": p["exact_match_rate"] * 100,
                     "close records %": p["close_record_rate"] * 100, "NN disclosure %": p["nn_disclosure_share"] * 100,
                     "MIA AUC": p["mia_auc"], "TPR @ 1% FPR": p["mia_tpr_at_1pct_fpr"], "risk": p["overall_risk"]})
    st.write("")
    html(ui.html_table(pd.DataFrame(rows), badges=["risk"],
                       formats={"exact match %": "{:.2f}", "close records %": "{:.1f}", "NN disclosure %": "{:.1f}",
                                "MIA AUC": "{:.3f}", "TPR @ 1% FPR": "{:.3f}"}))

    pick = st.radio("Generator", list(run["results"]), horizontal=True, key="priv_pick",
                    format_func=lambda k: run["results"][k]["display_name"])
    r = run["results"][pick]
    p = r["privacy"]
    html("<ul>" + "".join(f"<li>{s}</li>" for s in r["privacy_text"]) + "</ul>")
    a, b = st.columns(2, gap="large")
    fig = go.Figure()
    fig.add_histogram(x=p["_dcr_holdout"], name="Real holdout → train", marker_color="#6b7280", opacity=0.6, histnorm="probability density")
    fig.add_histogram(x=p["_dcr_synth"], name="Synthetic → train", marker_color="#16a34a", opacity=0.6, histnorm="probability density")
    fig.add_vline(x=p["dcr_holdout_p5"], line_dash="dash", annotation_text="holdout 5th pct")
    fig.update_layout(**PLOT_LAYOUT, barmode="overlay", title="Distance to closest training record", height=380,
                      xaxis_title="distance")
    a.plotly_chart(fig, use_container_width=True)
    fpr, tpr = p["_mia_roc"]
    fig = go.Figure([go.Scatter(x=fpr, y=tpr, name=f"Attack (AUC {p['mia_auc']:.3f})", line=dict(color="#dc2626", width=3)),
                     go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(color="#6b7280", dash="dash"))])
    fig.update_layout(**PLOT_LAYOUT, title="Membership inference attack ROC", height=380,
                      xaxis_title="false positive rate", yaxis_title="true positive rate")
    b.plotly_chart(fig, use_container_width=True)
    with st.expander("How to read these metrics"):
        st.markdown("- **Exact match**: synthetic rows identical to a training row, should be near 0.\n"
                    "- **Close records**: synthetic rows closer to a real person than 95% of unseen people are; about 5% is normal.\n"
                    "- **NN disclosure**: share of synthetic rows whose nearest real person was a training member; 50% means no leak.\n"
                    "- **MIA AUC**: how well an attacker guesses who was in the training data; 0.5 is a coin flip.\n"
                    "- These are empirical estimates, not formal differential privacy guarantees.")


def page_export() -> None:
    if not need_run():
        return
    run = ss.run
    html(ui.section("Export", "Download synthetic datasets, the full evaluation report and all artefacts."))
    cols = st.columns(len(run["results"]))
    for c, (key, r) in zip(cols, run["results"].items()):
        path = Path(run["synthetic_dir"]) / f"synthetic_{key}.csv"
        with c.container(border=True):
            html(f"<b style='font-size:16px;color:#14532d'>{r['display_name']}</b><br>"
                 f"<span style='color:#5f6f66;font-size:13px'>{r['n_synthetic']:,} rows · {ui.badge(r['privacy']['overall_risk'])}</span>")
            st.download_button("⬇️ Synthetic CSV", path.read_bytes(), path.name, "text/csv",
                               key=f"dl_{key}", use_container_width=True)
            with st.expander("Preview"):
                html(ui.html_table(pd.read_csv(path).head(20), max_height=360))

    st.write("")
    a, b, c = st.columns(3)
    a.download_button("📄 HTML evaluation report", Path(run["report_html"]).read_bytes(), "evaluation_report.html",
                      "text/html", use_container_width=True, type="primary")
    mpath = Path(run["reports_dir"]) / "metrics.json"
    b.download_button("🧾 Metrics JSON", mpath.read_bytes(), "metrics.json", "application/json", use_container_width=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, tag in [(run["synthetic_dir"], "synthetic"), (run["reports_dir"], "reports"), (run["figures_dir"], "figures")]:
            for f in Path(folder).glob("*"):
                z.write(f, f"{run['run_id']}/{tag}/{f.name}")
    c.download_button("🗜️ Everything (zip)", buf.getvalue(), f"{run['run_id']}.zip", "application/zip", use_container_width=True)
    st.caption(f"Files are also saved on disk: {run['reports_dir']}")


def page_about() -> None:
    html(ui.section("About PrivSynth"))
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        st.markdown(
            "PrivSynth is a research platform for producing synthetic tabular data and checking whether it is "
            "**faithful**, **useful** and **private**. It was built by **AIVONEX SMC PVT LTD**.\n\n"
            "**Scoring**\n"
            "- Fidelity = mean of column shape score, pair trend score and detection realism\n"
            "- Utility = synthetic model score divided by real model score on the same holdout\n"
            "- Privacy = 1 minus the mean of four normalised risks\n"
            "- Overall = 0.35 × fidelity + 0.35 × utility + 0.30 × privacy\n\n"
            "**Generators**: Gaussian Copula, CTGAN and TVAE from the SDV library, and a class conditional "
            "TabDDPM style diffusion model implemented in PyTorch.")
    with b:
        html('<div class="card" style="text-align:center;background:#0b0f0d">'
             + ui.logo_html(LOGO, 90).replace('class="logo-fallback"', 'class="logo-fallback" style="margin:auto"')
             + '<h4 style="color:#86efac">AIVONEX SMC PVT LTD</h4>'
               '<p style="color:#d1fae5">Applied AI research and services<br>Bahawalpur, Pakistan</p></div>')


{"Home": page_home, "Data": page_data, "Generate": page_generate, "Results": page_results,
 "Privacy": page_privacy, "Export": page_export, "About": page_about}[ss.page]()
html(ui.FOOTER)
