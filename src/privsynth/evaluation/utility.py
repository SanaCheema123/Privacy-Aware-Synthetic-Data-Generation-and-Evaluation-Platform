"""Downstream machine learning utility.

TRTR  Train on Real (training split), Test on Real holdout  -> upper reference
TSTR  Train on Synthetic,              Test on Real holdout  -> what the synthetic data is worth
Utility ratio = TSTR score / TRTR score (1.0 means no loss in usefulness).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.schema import Schema


def _models(task: str, seed: int) -> dict:
    if task == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed,
                                                    class_weight="balanced"),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        }
    return {
        "logistic_regression": Ridge(alpha=1.0),  # linear baseline for regression tasks
        "random_forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def _pipeline(model, num: list[str], cat: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def _prepare(df: pd.DataFrame, schema: Schema, num: list[str], cat: list[str]):
    X = df[num + cat].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat:
        X[c] = X[c].astype(str).astype(object)
    y = df[schema.target]
    if schema.task == "classification":
        # integer class codes (sklearn 1.9 mishandles class_weight with string labels)
        labels = schema.columns[schema.target].categories
        lookup = {str(c): i for i, c in enumerate(labels)}
        y = y.astype(str).map(lookup).fillna(-1).to_numpy(dtype=int)
    else:
        y = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    return X, y


def _score(task: str, model, X, y) -> dict:
    pred = model.predict(X)
    if task == "classification":
        out = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
               "f1_macro": f1_score(y, pred, average="macro", zero_division=0)}
        try:
            proba = model.predict_proba(X)
            classes = list(model.classes_)
            if len(classes) == 2:
                out["roc_auc"] = roc_auc_score((y == classes[1]).astype(int), proba[:, 1])
            else:
                out["roc_auc"] = roc_auc_score(y, proba, multi_class="ovr", labels=classes)
        except Exception:
            out["roc_auc"] = np.nan
        return out
    return {"r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred)}


def evaluate_utility(real_train: pd.DataFrame, holdout: pd.DataFrame, synth: pd.DataFrame, schema: Schema,
                     model_names: list[str], seed: int = 42, trtr_cache: dict | None = None) -> dict:
    if schema.target is None:
        return {"available": False, "reason": "no target column selected"}
    task = schema.task
    feats = [c for c in schema.modelled if c != schema.target]
    num = [c for c in feats if c in schema.numerical]
    cat = [c for c in feats if c in schema.categorical]
    Xte, yte = _prepare(holdout, schema, num, cat)
    primary = "f1_macro" if task == "classification" else "r2"
    rows = []
    trtr_cache = trtr_cache if trtr_cache is not None else {}
    all_models = _models(task, seed)
    for name in model_names:
        if name not in all_models:
            continue
        if name not in trtr_cache:
            Xr, yr = _prepare(real_train, schema, num, cat)
            m = _pipeline(all_models[name], num, cat).fit(Xr, yr)
            trtr_cache[name] = _score(task, m, Xte, yte)
        Xs, ys = _prepare(synth, schema, num, cat)
        if task == "classification" and len(np.unique(ys)) < 2:
            tstr = {k: np.nan for k in trtr_cache[name]}
        else:
            ms = _pipeline(_models(task, seed)[name], num, cat).fit(Xs, ys)
            tstr = _score(task, ms, Xte, yte)
        for metric in trtr_cache[name]:
            rows.append({"model": name, "metric": metric, "TRTR": trtr_cache[name][metric], "TSTR": tstr[metric]})
    table = pd.DataFrame(rows)
    prim = table[table.metric == primary]
    ratio = float(np.nanmean(prim["TSTR"] / prim["TRTR"].replace(0, np.nan))) if len(prim) else np.nan
    return {"available": True, "task": task, "primary_metric": primary, "table": table,
            "tstr_primary": float(prim["TSTR"].mean()), "trtr_primary": float(prim["TRTR"].mean()),
            "utility_ratio": ratio, "utility_score": float(np.clip(ratio, 0, 1)) if not np.isnan(ratio) else np.nan}
