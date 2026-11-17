

# PrivSynth: Privacy Aware Synthetic Data Generation and Evaluation Platform
[screen-capture (12).webm](https://github.com/user-attachments/assets/b9a0ed7d-f2e5-4ccd-a264-1eae92d32603)

A generative AI platform that produces synthetic tabular datasets for research and evaluates
their statistical similarity, usefulness for machine learning, and privacy risk.
Built by AIVONEX SMC PVT LTD.

## Features

| Feature | Where |
|---------|-------|
| CSV upload and automatic schema detection (numerical, categorical, boolean, datetime, ID and free text detection) | `src/privsynth/data/schema.py` |
| Synthetic generation with CTGAN, TVAE, Gaussian Copula and TabDDPM | `src/privsynth/generators/` |
| Class imbalance handling (keep, balance, custom ratios) with conditional sampling | `src/privsynth/imbalance.py` |
| Real versus synthetic distribution comparison (KS, TVD, Wasserstein, correlation / Cramér's V, detection AUC, PCA) | `src/privsynth/evaluation/fidelity.py` |
| Downstream model training and evaluation (TSTR vs TRTR with 3 models) | `src/privsynth/evaluation/utility.py` |
| Privacy risk report (exact matches, DCR, NNDR, nearest neighbour disclosure, membership inference attack) | `src/privsynth/evaluation/privacy.py` |
| Synthetic dataset export (CSV in original types), HTML report, metrics JSON, leaderboard | `src/privsynth/pipeline.py` |
| Web dashboard | `app/streamlit_app.py` |
| REST API | `api/main.py` |

## Folder structure

```
PrivSynth_Platform/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml                 all settings (data, generators, imbalance, evaluation, outputs)
├── assets/
│   └── aivonex_logo.png            company logo used in reports, figures, dashboard and API
├── data/
│   ├── raw/                        input CSVs (sample_clinical.csv demo included)
│   └── synthetic/<run_id>/         exported synthetic_<generator>.csv files
├── src/privsynth/
│   ├── config.py                   config loading, seeds, paths
│   ├── logging_utils.py
│   ├── imbalance.py                class balancing plan
│   ├── pipeline.py                 end to end orchestration
│   ├── data/
│   │   ├── loader.py               CSV reading, train / holdout split
│   │   ├── schema.py               schema detection and type restoration
│   │   └── transformer.py          reversible numeric encoding
│   ├── generators/
│   │   ├── base.py                 common generator interface
│   │   ├── sdv_models.py           CTGAN, TVAE, Gaussian Copula
│   │   ├── tabddpm.py              diffusion generator (PyTorch)
│   │   └── __init__.py             registry
│   ├── evaluation/
│   │   ├── fidelity.py             statistical fidelity
│   │   ├── utility.py              downstream ML utility
│   │   └── privacy.py              privacy risk metrics
│   └── reporting/
│       ├── plots.py                all figures (auto saved as PNG)
│       └── report.py               self contained HTML report
├── scripts/
│   ├── make_sample_data.py         creates the demo dataset
│   └── run_pipeline.py             command line runner
├── app/streamlit_app.py            dashboard
├── api/main.py                     FastAPI service
├── tests/test_platform.py          unit and smoke tests (pytest)
├── docs/ARCHITECTURE.md
└── outputs/
    ├── figures/<run_id>/           PNG figures
    ├── reports/<run_id>/           evaluation_report.html, metrics.json, leaderboard.csv, schema.json
    ├── models/<run_id>/            trained generators (.pkl)
    └── logs/privsynth.log
```

## Setup (Windows PowerShell)

Use Python 3.11. Always call tools through `python -m` (pip, pytest, streamlit, uvicorn),
because Windows Application Control can block the small `.exe` launchers inside `.venv\Scripts`.

```powershell
cd D:\PrivSynth_Platform
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\check_env.py
```

`check_env.py` loads every package one by one and prints OK or BLOCKED, so a blocked DLL is easy to spot.

## Run

```powershell
cd D:\PrivSynth_Platform
python scripts\make_sample_data.py              # demo dataset
python scripts\run_pipeline.py --quick          # fast test, about 2 minutes on CPU
python scripts\run_pipeline.py                  # full training with config.yaml settings
python scripts\run_pipeline.py --csv data\raw\my_data.csv --target label --imbalance balance
python -m streamlit run app\streamlit_app.py    # dashboard at http://localhost:8501
python -m uvicorn api.main:app --port 8000      # API docs at http://127.0.0.1:8000/docs
python -m pytest -q                             # tests
```

## How to read the results

* **Fidelity**: column shape scores near 1 and detection AUC near 0.5 mean the synthetic data is hard to tell apart from the real data.
* **Utility**: utility ratio = TSTR / TRTR. A value near 1 means a model trained on synthetic data performs as well on real holdout data as a model trained on the real data.
* **Privacy**: exact match rate near 0, close record rate near 5%, nearest neighbour disclosure near 50% and membership inference AUC near 0.5 are good signs.
* There is always a tradeoff: a generator that copies the data scores perfectly on fidelity and terribly on privacy. The leaderboard balances the three.

## Notes and limitations

* TabDDPM here uses Gaussian diffusion over quantile normalised numerics and one hot categoricals (a common simplification of the paper's multinomial diffusion). It is fully class conditional, which makes it the strongest option for minority class oversampling.
* Privacy metrics are empirical estimates, not formal guarantees. For formal guarantees add a differentially private generator.
* Missing numeric values are modelled by SDV generators and median imputed for TabDDPM.
