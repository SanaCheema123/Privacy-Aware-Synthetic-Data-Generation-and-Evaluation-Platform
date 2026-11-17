"""Class imbalance handling: decide how many synthetic rows each class receives."""
from __future__ import annotations

import pandas as pd


def class_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    counts = df[target].astype(str).value_counts()
    return pd.DataFrame({"class": counts.index, "count": counts.values,
                         "share_%": (counts.values / counts.sum() * 100).round(2)})


def imbalance_ratio(df: pd.DataFrame, target: str) -> float:
    counts = df[target].astype(str).value_counts()
    return float(counts.max() / max(counts.min(), 1))


def plan_class_counts(real: pd.DataFrame, target: str, strategy: str = "none", num_rows: int | None = None,
                      custom_ratios: dict | None = None) -> dict[str, int]:
    """Return {class_label: n_rows} for synthetic generation.

    none     keep the real class proportions
    balance  same number of rows for every class
    custom   user ratios, for example {"0": 0.6, "1": 0.4}
    """
    counts = real[target].astype(str).value_counts()
    classes = counts.index.tolist()
    n = int(num_rows or len(real))
    if strategy == "balance":
        shares = {c: 1.0 / len(classes) for c in classes}
    elif strategy == "custom":
        ratios = {str(k): float(v) for k, v in (custom_ratios or {}).items()}
        unknown = set(ratios) - set(classes)
        if unknown:
            raise ValueError(f"custom_ratios contains unknown classes {unknown}; valid classes are {classes}")
        total = sum(ratios.values()) or 1.0
        shares = {c: ratios.get(c, 0.0) / total for c in classes}
    else:
        shares = (counts / counts.sum()).to_dict()
    plan = {c: int(round(n * s)) for c, s in shares.items()}
    diff = n - sum(plan.values())  # fix rounding so the total is exact
    if diff:
        top = max(plan, key=plan.get)
        plan[top] += diff
    return plan
