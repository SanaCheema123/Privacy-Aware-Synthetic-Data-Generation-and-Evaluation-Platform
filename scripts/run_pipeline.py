"""Command line entry point.

Examples:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --csv data/raw/my_data.csv --target label
    python scripts/run_pipeline.py --generators ctgan tabddpm --imbalance balance
    python scripts/run_pipeline.py --quick            # fast smoke run with few epochs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from privsynth.pipeline import run_pipeline  # noqa: E402

QUICK = {"ctgan": {"epochs": 30}, "tvae": {"epochs": 30}, "tabddpm": {"steps": 600, "num_timesteps": 200}}


def main() -> None:
    ap = argparse.ArgumentParser(description="PrivSynth synthetic data generation and evaluation")
    ap.add_argument("--config", default=None, help="YAML config (default config/config.yaml)")
    ap.add_argument("--csv", default=None, help="input CSV path")
    ap.add_argument("--target", default=None, help="target column")
    ap.add_argument("--generators", nargs="+", default=None,
                    choices=["gaussian_copula", "ctgan", "tvae", "tabddpm"])
    ap.add_argument("--imbalance", default=None, choices=["none", "balance", "custom"])
    ap.add_argument("--rows", type=int, default=None, help="number of synthetic rows")
    ap.add_argument("--epochs", type=int, default=None, help="epochs for CTGAN and TVAE")
    ap.add_argument("--quick", action="store_true", help="few epochs for a fast test run")
    a = ap.parse_args()

    ov: dict = {"data": {}, "generators": {}, "imbalance": {}}
    if a.csv:
        ov["data"]["input_csv"] = a.csv
    if a.target:
        ov["data"]["target_column"] = a.target
    if a.generators:
        ov["generators"]["enabled"] = a.generators
    if a.imbalance:
        ov["imbalance"]["strategy"] = a.imbalance
    if a.rows:
        ov["imbalance"]["num_rows"] = a.rows
    if a.epochs:
        ov["generators"]["ctgan"] = {"epochs": a.epochs}
        ov["generators"]["tvae"] = {"epochs": a.epochs}
    if a.quick:
        for k, v in QUICK.items():
            ov["generators"].setdefault(k, {}).update(v)

    run = run_pipeline(a.config, ov)
    print("\n================ LEADERBOARD ================")
    print(run["leaderboard"].drop(columns=["key"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if run["failed"]:
        print("\nFailed generators:", run["failed"])
    print(f"\nHTML report     : {run['report_html']}")
    print(f"Synthetic CSVs  : {run['synthetic_dir']}")
    print(f"Figures         : {run['figures_dir']}")


if __name__ == "__main__":
    main()
