"""Check that every required package (and its compiled DLLs) can load on this machine.

Usage (from the project root):
    python scripts\\check_env.py
Each package is imported in a separate process so one blocked DLL does not hide the others.
"""
import subprocess
import sys

CHECKS = [
    ("numpy", "import numpy; numpy.linalg.inv(numpy.eye(2)); print(numpy.__version__)"),
    ("pandas", "import pandas; print(pandas.__version__)"),
    ("scipy", "import scipy.stats, scipy; print(scipy.__version__)"),
    ("scikit-learn", "import sklearn.ensemble, sklearn.metrics, sklearn.neighbors, sklearn; print(sklearn.__version__)"),
    ("matplotlib", "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot; print(matplotlib.__version__)"),
    ("torch", "import torch; torch.randn(2, 2) @ torch.randn(2, 2); print(torch.__version__)"),
    ("sdv", "import sdv.single_table, sdv; print(sdv.__version__)"),
    ("streamlit", "import streamlit; print(streamlit.__version__)"),
    ("fastapi", "import fastapi, uvicorn; print(fastapi.__version__)"),
    ("pytest", "import pytest; print(pytest.__version__)"),
]

print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
bad = []
for name, code in CHECKS:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  OK       {name:<14}{r.stdout.strip()}")
    else:
        last = (r.stderr.strip().splitlines() or ["unknown error"])[-1]
        status = "BLOCKED" if "Application Control" in r.stderr or "WinError 4551" in r.stderr else "FAILED"
        print(f"  {status:<8} {name:<14}{last[:110]}")
        bad.append(name)

print("\nAll packages load correctly." if not bad else f"\nProblem packages: {', '.join(bad)}")
