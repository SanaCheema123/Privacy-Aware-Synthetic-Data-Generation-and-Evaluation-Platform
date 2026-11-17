"""Self contained HTML evaluation report with the AIVONEX logo (images embedded as base64)."""
from __future__ import annotations

import base64
import datetime as dt
import html
from pathlib import Path

import pandas as pd

RISK_COLORS = {"low": "#2e7d32", "medium": "#ef6c00", "high": "#c62828", "unknown": "#757575"}


def _b64(path: Path | None) -> str:
    if path is None or not Path(path).exists():
        return ""
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _img(path: Path | None, alt: str) -> str:
    src = _b64(path)
    return f'<figure><img src="{src}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>' if src else ""


def _table(df: pd.DataFrame, digits: int = 4) -> str:
    return df.to_html(index=False, float_format=lambda v: f"{v:.{digits}f}", border=0, classes="tbl", na_rep="n/a")


def _badge(level: str) -> str:
    return f'<span class="badge" style="background:{RISK_COLORS.get(level, "#757575")}">{level.upper()}</span>'


CSS = """
body{font-family:Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f5f7f6;color:#1d2521}
header{background:#0b0f0d;color:#fff;padding:22px 40px;display:flex;align-items:center;gap:20px}
header img{height:64px}
header h1{margin:0;font-size:22px}
header p{margin:4px 0 0;color:#a9d6b8;font-size:13px}
main{max-width:1200px;margin:auto;padding:24px 40px 60px}
section{background:#fff;border-radius:10px;padding:20px 26px;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h2{color:#14532d;border-bottom:2px solid #22c55e33;padding-bottom:6px}
h3{color:#1f3b2c}
.tbl{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
.tbl th{background:#14532d;color:#fff;padding:7px;text-align:left}
.tbl td{padding:6px 7px;border-bottom:1px solid #e6ece8}
.badge{color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:600}
.kpis{display:flex;gap:14px;flex-wrap:wrap}
.kpi{flex:1;min-width:160px;background:#f0f7f2;border-radius:8px;padding:12px 16px}
.kpi b{display:block;font-size:24px;color:#14532d}
figure{margin:14px 0}figure img{max-width:100%;border:1px solid #e3e8e5;border-radius:6px}
figcaption{font-size:12px;color:#5f6f66}
footer{text-align:center;color:#7a8a80;font-size:12px;padding:20px}
ul li{margin:4px 0}
"""


def build_html_report(run: dict, out_path: Path, logo: Path | None) -> Path:
    board: pd.DataFrame = run["leaderboard"]
    best = board.iloc[0]
    logo_src = _b64(logo)
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>PrivSynth report {run['run_id']}</title>"
             f"<style>{CSS}</style></head><body><header>"
             + (f'<img src="{logo_src}" alt="AIVONEX logo">' if logo_src else "")
             + f"<div><h1>Privacy Aware Synthetic Data Evaluation Report</h1>"
               f"<p>AIVONEX SMC PVT LTD · Run {run['run_id']} · {dt.datetime.now():%d %b %Y %H:%M}</p></div></header><main>"]

    # summary
    parts.append("<section><h2>1. Summary</h2><div class='kpis'>"
                 f"<div class='kpi'>Real rows<b>{run['n_rows']}</b></div>"
                 f"<div class='kpi'>Columns used<b>{len(run['schema'].modelled)}</b></div>"
                 f"<div class='kpi'>Target<b>{html.escape(str(run['schema'].target))}</b></div>"
                 f"<div class='kpi'>Recommended generator<b>{html.escape(best['generator'])}</b></div></div>"
                 f"<p>The dataset was split into a training part ({run['n_train']} rows) that every generator learned from "
                 f"and a holdout part ({run['n_holdout']} rows) that no generator ever saw. The holdout is the real test set "
                 f"for downstream models and the non member reference for privacy attacks. Class balancing strategy: "
                 f"<b>{html.escape(run['imbalance_strategy'])}</b>.</p>"
                 + ("<p><i>Because the synthetic class proportions were changed on purpose, column shape and detection "
                    "scores are naturally a little lower than they would be for a same distribution copy. Judge these "
                    "runs mainly by utility (minority class F1) and privacy.</i></p>"
                    if run["imbalance_strategy"] != "none" else "")
                 + f"<h3>Leaderboard</h3>{_table(board.drop(columns=['key']), 3)}"
                 f"<p>Overall score = 0.35 × fidelity + 0.35 × utility + 0.30 × privacy. All scores are between 0 and 1, higher is better.</p>"
                 f"{_img(run['figures'].get('leaderboard'), 'Generator comparison')}"
                 f"{_img(run['figures'].get('class_balance'), 'Class distribution')}</section>")

    # schema
    parts.append(f"<section><h2>2. Detected schema</h2>{_table(run['schema'].summary_frame(), 2)}</section>")

    # per generator
    for i, (key, res) in enumerate(run["results"].items(), start=3):
        name = res["display_name"]
        fid, util, priv = res["fidelity"], res["utility"], res["privacy"]
        figs = res["figures"]
        parts.append(f"<section><h2>{i}. {html.escape(name)}</h2>"
                     f"<p>Training time {res['fit_seconds']:.1f} s, sampling time {res['sample_seconds']:.1f} s, "
                     f"{res['n_synthetic']} synthetic rows.</p>"
                     + "".join(f"<p style='color:#c62828'><b>Note:</b> {html.escape(n)}</p>" for n in res.get("notes", []))
                     +
                     "<h3>Statistical fidelity</h3><div class='kpis'>"
                     f"<div class='kpi'>Column shapes<b>{fid['column_shape_score']:.3f}</b></div>"
                     f"<div class='kpi'>Pair trends<b>{fid['pair_trend_score']:.3f}</b></div>"
                     f"<div class='kpi'>Detection AUC<b>{fid['detection_auc']:.3f}</b></div>"
                     f"<div class='kpi'>Fidelity score<b>{fid['fidelity_score']:.3f}</b></div></div>"
                     f"<p>Detection AUC is how well a random forest separates real from synthetic rows; 0.5 is ideal.</p>"
                     f"{_table(fid['column_table'].dropna(axis=1, how='all'), 3)}"
                     f"{_img(figs.get('distributions'), 'Distributions')}{_img(figs.get('association'), 'Associations')}"
                     f"{_img(figs.get('pca'), 'PCA')}")
        if util.get("available"):
            parts.append("<h3>Downstream machine learning utility</h3><div class='kpis'>"
                         f"<div class='kpi'>TRTR {util['primary_metric']}<b>{util['trtr_primary']:.3f}</b></div>"
                         f"<div class='kpi'>TSTR {util['primary_metric']}<b>{util['tstr_primary']:.3f}</b></div>"
                         f"<div class='kpi'>Utility ratio<b>{util['utility_ratio']:.3f}</b></div></div>"
                         f"{_table(util['table'], 4)}")
        parts.append(f"<h3>Privacy risk report {_badge(priv['overall_risk'])}</h3><ul>"
                     + "".join(f"<li>{html.escape(s)}</li>" for s in res["privacy_text"]) + "</ul>"
                     + _table(pd.DataFrame([{k: v for k, v in priv.items() if not k.startswith('_') and not isinstance(v, dict)}]), 4)
                     + _img(figs.get("privacy"), "Privacy") + "</section>")

    if run["figures"].get("utility"):
        parts.append(f"<section><h2>Utility comparison</h2>{_img(run['figures']['utility'], 'Utility')}</section>")
    if run["figures"].get("losses"):
        parts.append(f"<section><h2>Training curves</h2>{_img(run['figures']['losses'], 'Losses')}</section>")

    parts.append("<section><h2>How to read the privacy metrics</h2><ul>"
                 "<li><b>Exact match rate</b>: synthetic rows identical to a training row. Should be close to 0.</li>"
                 "<li><b>DCR</b>: distance from each synthetic row to its closest training row. It is compared with how close "
                 "real unseen people naturally are to the training set.</li>"
                 "<li><b>Nearest neighbour disclosure</b>: with equal sized member and non member sets, 50% means the synthetic "
                 "data does not reveal who was in the training data.</li>"
                 "<li><b>Membership inference AUC</b>: success of an attacker guessing whether a person was in the training data. "
                 "0.5 means no better than a coin flip.</li>"
                 "<li>These are empirical risk estimates, not formal guarantees. For formal guarantees use a differentially private "
                 "generator (for example DP CTGAN or PATE GAN).</li></ul></section>")
    parts.append("</main><footer>Generated by PrivSynth · AIVONEX SMC PVT LTD</footer></body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path
