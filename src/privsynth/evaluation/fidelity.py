"""Statistical fidelity: how closely synthetic data matches the real distribution.

Column shapes
    numerical   1 minus Kolmogorov Smirnov statistic, plus normalised Wasserstein distance
    categorical 1 minus total variation distance between category frequencies
Column pair trends
    numerical pairs   Pearson correlation difference
    categorical pairs Cramer's V difference
    mixed pairs       correlation ratio (eta) difference
Detection
    a classifier tries to tell real from synthetic rows; ROC AUC near 0.5 is ideal
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..data.schema import Schema
from ..data.transformer import TabularTransformer


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)


def _freqs(s: pd.Series) -> pd.Series:
    return s.astype(str).value_counts(normalize=True)


def column_shapes(real: pd.DataFrame, synth: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    for c in schema.numerical:
        r, s = _num(real[c]), _num(synth[c])
        if len(r) == 0 or len(s) == 0:
            continue
        ks = ks_2samp(r, s).statistic
        span = (r.max() - r.min()) or 1.0
        wd = wasserstein_distance(r, s) / span
        rows.append({"column": c, "type": "numerical", "metric": "KS complement", "score": 1 - ks,
                     "wasserstein_norm": wd, "real_mean": r.mean(), "synth_mean": s.mean(),
                     "real_std": r.std(), "synth_std": s.std()})
    for c in schema.categorical:
        pr, ps = _freqs(real[c]), _freqs(synth[c])
        idx = pr.index.union(ps.index)
        tvd = 0.5 * np.abs(pr.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).sum()
        rows.append({"column": c, "type": "categorical", "metric": "TV complement", "score": 1 - tvd,
                     "wasserstein_norm": np.nan, "real_mean": np.nan, "synth_mean": np.nan,
                     "real_std": np.nan, "synth_std": np.nan,
                     "category_coverage": float(ps.index.isin(pr.index).sum() / max(len(pr), 1))})
    return pd.DataFrame(rows)


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a.astype(str), b.astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def _eta(cat: pd.Series, num: pd.Series) -> float:
    d = pd.DataFrame({"c": cat.astype(str), "x": pd.to_numeric(num, errors="coerce")}).dropna()
    if d["x"].var() == 0 or len(d) < 3:
        return 0.0
    grand = d["x"].mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in d.groupby("c")["x"])
    total = ((d["x"] - grand) ** 2).sum()
    return float(np.sqrt(between / total)) if total > 0 else 0.0


def association_matrix(df: pd.DataFrame, schema: Schema, columns: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for a, b in itertools.combinations(columns, 2):
        an, bn = a in schema.numerical, b in schema.numerical
        if an and bn:
            v = pd.to_numeric(df[a], errors="coerce").corr(pd.to_numeric(df[b], errors="coerce"))
        elif not an and not bn:
            v = _cramers_v(df[a], df[b])
        else:
            v = _eta(df[b] if an else df[a], df[a] if an else df[b])
        m.loc[a, b] = m.loc[b, a] = 0.0 if pd.isna(v) else v
    return m


def pair_trends(real: pd.DataFrame, synth: pd.DataFrame, schema: Schema,
                max_columns: int = 25) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    cols = schema.modelled[:max_columns]
    r = association_matrix(real, schema, cols)
    s = association_matrix(synth, schema, cols)
    diff = (r - s).abs()
    iu = np.triu_indices(len(cols), 1)
    score = 1 - float(np.mean(diff.to_numpy()[iu]) / 2) if len(cols) > 1 else 1.0
    return score, r, s


def detection_auc(real: pd.DataFrame, synth: pd.DataFrame, schema: Schema, seed: int = 42,
                  max_rows: int = 5000) -> float:
    n = min(len(real), len(synth), max_rows)
    r = real.sample(n, random_state=seed)
    s = synth.sample(n, random_state=seed)
    tf = TabularTransformer(schema.numerical, schema.categorical).fit(pd.concat([r, s]))
    X = np.vstack([tf.transform(r), tf.transform(s)])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, n_jobs=-1, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def evaluate_fidelity(real: pd.DataFrame, synth: pd.DataFrame, schema: Schema, seed: int = 42) -> dict:
    shapes = column_shapes(real, synth, schema)
    pair_score, r_assoc, s_assoc = pair_trends(real, synth, schema)
    auc = detection_auc(real, synth, schema, seed)
    shape_score = float(shapes["score"].mean()) if len(shapes) else float("nan")
    detection_score = 1 - 2 * abs(auc - 0.5)
    overall = float(np.nanmean([shape_score, pair_score, detection_score]))
    return {
        "column_shape_score": shape_score,
        "pair_trend_score": pair_score,
        "detection_auc": auc,
        "detection_score": detection_score,
        "fidelity_score": overall,
        "column_table": shapes,
        "real_assoc": r_assoc,
        "synth_assoc": s_assoc,
    }
