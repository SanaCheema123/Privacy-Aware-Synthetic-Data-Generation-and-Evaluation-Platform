"""Create a demo clinical style CSV (mixed types, missing values, ID and date columns, imbalanced target).

Usage:
    python scripts/make_sample_data.py --rows 4000 --out data/raw/sample_clinical.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(55, 14, rows), 18, 95).round()
    sex = rng.choice(["Female", "Male"], rows)
    bmi = np.clip(rng.normal(27 + 0.03 * (age - 55), 4.5, rows), 15, 50).round(1)
    smoker = rng.choice(["never", "former", "current"], rows, p=[0.55, 0.3, 0.15])
    sys_bp = np.clip(100 + 0.6 * age + 0.8 * (bmi - 25) + rng.normal(0, 12, rows), 85, 210).round()
    chol = np.clip(rng.normal(195 + 0.4 * age, 35, rows), 110, 340).round()
    glucose = np.clip(rng.lognormal(np.log(98 + 0.6 * (bmi - 25)), 0.18, rows), 60, 320).round()
    diabetes = (glucose > 126).astype(int)
    region = rng.choice(["Punjab", "Sindh", "KPK", "Balochistan", "Islamabad"], rows, p=[.45, .25, .15, .07, .08])
    visits = rng.poisson(2 + 0.03 * (age - 40).clip(0), rows)
    admit = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, rows), unit="D")
    logit = (-6.2 + 0.045 * age + 0.06 * (bmi - 25) + 0.018 * (sys_bp - 130) + 0.9 * diabetes
             + 0.7 * (smoker == "current") + 0.3 * (sex == "Male") + 0.08 * visits)
    outcome = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "patient_id": np.arange(100001, 100001 + rows), "admission_date": admit.strftime("%Y-%m-%d"),
        "age": age.astype(int), "sex": sex, "region": region, "bmi": bmi, "smoker": smoker,
        "systolic_bp": sys_bp.astype(int), "cholesterol": chol, "glucose": glucose,
        "diabetes": diabetes, "prior_visits": visits, "outcome": outcome})
    # realistic missingness
    for c, p in [("bmi", 0.03), ("cholesterol", 0.05), ("smoker", 0.02)]:
        df.loc[rng.random(rows) < p, c] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw/sample_clinical.csv")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make(a.rows, a.seed)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print("Outcome distribution:", df["outcome"].value_counts().to_dict())
