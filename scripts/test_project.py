"""Full project test: environment, unit tests and a quick pipeline run on every test dataset.

Usage (from the project root):
    python scripts\\test_project.py
Put test CSVs in data/raw/ with names starting with "test_". The last column is used as target.
A summary prints in the terminal and the full report is saved to outputs/reports/test_report.txt.
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REPORT = ROOT / "outputs" / "reports" / "test_report.txt"
QUICK = {"ctgan": {"epochs": 20}, "tvae": {"epochs": 20}, "tabddpm": {"steps": 400, "num_timesteps": 200}}

lines: list[str] = []
status: list[tuple[str, bool, str]] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


def step(name: str, ok: bool, detail: str = "") -> None:
    status.append((name, ok, detail))
    log(f"{'[GREEN]' if ok else '[RED]  '} {name}  {detail}")


log(f"PrivSynth project test · {dt.datetime.now():%Y-%m-%d %H:%M}")
log(f"Python {sys.version.split()[0]} · {sys.executable}\n")

# 1. environment
log("=== 1. Environment check ===")
r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_env.py")], capture_output=True, text=True)
log(r.stdout.strip())
step("Packages load", "Problem packages" not in r.stdout and r.returncode == 0)

# 2. unit tests
log("\n=== 2. Unit tests ===")
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore", str(ROOT / "tests")],
                   capture_output=True, text=True, cwd=ROOT)
summary = (r.stdout.strip().splitlines() or ["no output"])[-1]
log(r.stdout.strip()[-3000:])
step("Unit tests", r.returncode == 0, summary)

# 3. dashboard imports (no browser needed)
log("\n=== 3. Dashboard and API import ===")
r = subprocess.run([sys.executable, "-c", "import sys; sys.path[:0]=['app','src','.']; import ui_style, api.main; "
                    "import ast; ast.parse(open('app/streamlit_app.py', encoding='utf-8').read()); print('ok')"],
                   capture_output=True, text=True, cwd=ROOT)
step("Dashboard and API code", r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:150])

# 4. pipeline on every test dataset
log("\n=== 4. Quick pipeline on test datasets ===")
datasets = sorted((ROOT / "data" / "raw").glob("test_*.csv"))
if not datasets:
    datasets = [ROOT / "data" / "raw" / "sample_clinical.csv"]
    log("No test_*.csv files in data/raw, using sample_clinical.csv")
from privsynth.data import load_csv  # noqa: E402
from privsynth.pipeline import run_pipeline  # noqa: E402

for csv in datasets:
    t0 = time.time()
    try:
        df = load_csv(csv)
        target = df.columns[-1]
        counts = df[target].value_counts()
        strategy = "balance" if counts.max() / max(counts.min(), 1) > 4 else "none"
        log(f"\n--- {csv.name}: {len(df)} rows, target '{target}', strategy '{strategy}' ---")
        run = run_pipeline(None, {"data": {"target_column": target}, "generators": QUICK,
                                  "imbalance": {"strategy": strategy}}, data=df)
        board = run["leaderboard"].drop(columns=["key"])
        log(board.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        log(f"report: {run['report_html']}")
        ok = not run["failed"] and len(board) > 0
        step(f"Pipeline {csv.name}", ok, f"best {board.iloc[0]['generator']} "
                                         f"({board.iloc[0]['overall_score']:.3f}) in {time.time() - t0:.0f}s"
             + (f" failed: {run['failed']}" if run["failed"] else ""))
    except Exception as e:
        log(traceback.format_exc())
        step(f"Pipeline {csv.name}", False, str(e)[:150])

# summary
log("\n=== SUMMARY ===")
for name, ok, detail in status:
    log(f"{'GREEN' if ok else 'RED  '}  {name:<45} {detail}")
all_ok = all(ok for _, ok, _ in status)
log("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED, see details above")
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nFull report saved to: {REPORT}")
sys.exit(0 if all_ok else 1)
