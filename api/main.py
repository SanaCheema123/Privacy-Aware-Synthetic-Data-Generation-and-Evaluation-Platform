"""PrivSynth REST API (FastAPI).

Run from the project root:
    uvicorn api.main:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from privsynth import __version__  # noqa: E402
from privsynth.config import load_config, logo_path, resolve  # noqa: E402
from privsynth.data import detect_schema, load_csv  # noqa: E402
from privsynth.generators import REGISTRY  # noqa: E402
from privsynth.pipeline import run_pipeline  # noqa: E402

CFG = load_config()
app = FastAPI(title="PrivSynth API · AIVONEX", version=__version__,
              description="Privacy aware synthetic tabular data generation and evaluation. "
                          "Logo: GET /logo")


async def _read(file: UploadFile):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")
    return load_csv(io.BytesIO(await file.read()))


@app.get("/", response_class=HTMLResponse)
def home():
    logo = '<img src="/logo" height="70">' if logo_path(CFG) else ""
    return f"<html><body style='font-family:sans-serif;padding:30px'>{logo}<h2>PrivSynth API</h2>" \
           f"<p>AIVONEX SMC PVT LTD · version {__version__}</p><a href='/docs'>Open interactive docs</a></body></html>"


@app.get("/logo")
def logo():
    p = logo_path(CFG)
    if not p:
        raise HTTPException(404, "Logo not found in assets/aivonex_logo.png")
    return FileResponse(p)


@app.get("/health")
def health():
    return {"status": "ok", "generators": list(REGISTRY)}


@app.post("/schema")
async def schema(file: UploadFile = File(...), target: str | None = Form(None)):
    df = await _read(file)
    s = detect_schema(df, target or None, int(CFG["data"]["max_categories"]))
    return {"rows": len(df), "columns": df.shape[1], "schema": s.to_dict()}


@app.post("/runs")
async def create_run(file: UploadFile = File(...), target: str | None = Form(None),
                     generators: str = Form("gaussian_copula,ctgan,tvae,tabddpm"),
                     imbalance: str = Form("none"), num_rows: int | None = Form(None),
                     epochs: int | None = Form(None), tabddpm_steps: int | None = Form(None)):
    """Upload a CSV, train the selected generators and evaluate them (synchronous)."""
    df = await _read(file)
    gens = [g.strip() for g in generators.split(",") if g.strip()]
    bad = set(gens) - set(REGISTRY)
    if bad:
        raise HTTPException(400, f"Unknown generators {bad}")
    ov = {"data": {"target_column": target or None}, "generators": {"enabled": gens},
          "imbalance": {"strategy": imbalance, "num_rows": num_rows}}
    if epochs:
        ov["generators"]["ctgan"] = {"epochs": epochs}
        ov["generators"]["tvae"] = {"epochs": epochs}
    if tabddpm_steps:
        ov["generators"]["tabddpm"] = {"steps": tabddpm_steps}
    try:
        run = run_pipeline(CFG, ov, data=df)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"run_id": run["run_id"], "leaderboard": json.loads(run["leaderboard"].to_json(orient="records")),
            "failed": run["failed"],
            "downloads": {"report": f"/runs/{run['run_id']}/report",
                          "metrics": f"/runs/{run['run_id']}/metrics",
                          "synthetic": {k: f"/runs/{run['run_id']}/synthetic/{k}" for k in run["results"]}}}


def _run_dir(kind: str, run_id: str) -> Path:
    if not run_id.startswith("run_") or "/" in run_id or ".." in run_id:
        raise HTTPException(400, "Invalid run id")
    d = resolve(CFG["outputs"][kind]) / run_id
    if not d.exists():
        raise HTTPException(404, "Run not found")
    return d


@app.get("/runs/{run_id}/report", response_class=HTMLResponse)
def get_report(run_id: str):
    return (_run_dir("reports_dir", run_id) / "evaluation_report.html").read_text(encoding="utf-8")


@app.get("/runs/{run_id}/metrics")
def get_metrics(run_id: str):
    return json.loads((_run_dir("reports_dir", run_id) / "metrics.json").read_text(encoding="utf-8"))


@app.get("/runs/{run_id}/synthetic/{generator}")
def get_synthetic(run_id: str, generator: str):
    if generator not in REGISTRY:
        raise HTTPException(400, "Unknown generator")
    p = _run_dir("synthetic_dir", run_id) / f"synthetic_{generator}.csv"
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(p, media_type="text/csv", filename=p.name)
