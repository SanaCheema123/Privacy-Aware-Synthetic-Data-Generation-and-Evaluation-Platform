"""Privacy risk evaluation.

All metrics compare three sets: real TRAIN rows (members, seen by the generator),
real HOLDOUT rows (non members, never seen) and SYNTHETIC rows.

1. Exact match rate        share of synthetic rows that are copies of a training row
2. DCR (distance to closest record)
                           distance from each synthetic row to its nearest training row,
                           compared with the natural holdout to train distance. Many synthetic
                           rows much closer than real neighbours suggests memorisation.
3. NNDR                    nearest neighbour distance ratio (d1/d2); values near 0 mean a
                           synthetic row sits on top of one specific real person.
4. Nearest neighbour disclosure (DCR share)
                           for equal sized train and holdout sets, the share of synthetic rows
                           whose nearest real row is a TRAIN row. Ideal is 0.5; values well above
                           0.5 mean the synthetic data leaks who was in the training set.
5. Membership inference attack (distance based, black box)
                           the attacker scores each real candidate by how close it is to the
                           synthetic data. AUC of member vs non member separation; 0.5 means
                           the attacker learns nothing. We also report TPR at 1% FPR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from ..data.schema import Schema
from ..data.transformer import TabularTransformer


def _encode(train, holdout, synth, schema: Schema):
    # one hot scaled by 1/sqrt(2) so a category mismatch costs distance 1, like one std of a numeric column
    tf = TabularTransformer(schema.numerical, schema.categorical, mode="standard", cat_scale=1 / np.sqrt(2))
    tf.fit(train)
    return tf.transform(train), tf.transform(holdout), tf.transform(synth)


def risk_level(value: float, low: float, high: float, higher_is_riskier: bool = True) -> str:
    if np.isnan(value):
        return "unknown"
    v = value if higher_is_riskier else -value
    lo, hi = (low, high) if higher_is_riskier else (-low, -high)
    return "low" if v <= lo else ("medium" if v <= hi else "high")


def evaluate_privacy(train: pd.DataFrame, holdout: pd.DataFrame, synth: pd.DataFrame, schema: Schema,
                     seed: int = 42, max_rows: int = 5000) -> dict:
    rng = np.random.default_rng(seed)
    # full reference set (members) for exact match and DCR
    tr_full = train.sample(min(len(train), max_rows), random_state=seed).reset_index(drop=True)
    sy = synth.sample(min(len(synth), max_rows), random_state=seed).reset_index(drop=True)
    # equal sized member / non member sets for disclosure and membership inference
    n = min(len(tr_full), len(holdout))
    ho = holdout.sample(n, random_state=seed).reset_index(drop=True)
    Xfull, Xho, Xsy = _encode(tr_full, ho, sy, schema)
    Xtr = Xfull[rng.choice(len(Xfull), n, replace=False)]

    # 1. exact matches (on the encoded representation, rounded)
    tr_keys = set(map(tuple, np.round(Xfull, 6)))
    exact = float(np.mean([tuple(r) in tr_keys for r in np.round(Xsy, 6)]))

    # 2/3. DCR and NNDR synthetic -> train
    nn_tr = NearestNeighbors(n_neighbors=2).fit(Xfull)
    d_sy, _ = nn_tr.kneighbors(Xsy)
    dcr_sy = d_sy[:, 0]
    nndr = d_sy[:, 0] / np.maximum(d_sy[:, 1], 1e-12)
    # baseline: holdout -> train (how close real unseen people naturally are)
    d_ho, _ = nn_tr.kneighbors(Xho, n_neighbors=1)
    dcr_ho = d_ho[:, 0]
    p5 = float(np.percentile(dcr_ho, 5))
    close_rate = float(np.mean(dcr_sy < p5))  # expected about 0.05 for a safe generator

    # 4. nearest neighbour disclosure: is the closest real row a member or a non member?
    d_sy_tr_eq = NearestNeighbors(n_neighbors=1).fit(Xtr).kneighbors(Xsy)[0][:, 0]  # equal sized member set
    d_sy_ho = NearestNeighbors(n_neighbors=1).fit(Xho).kneighbors(Xsy)[0][:, 0]
    closer_to_train = d_sy_tr_eq < d_sy_ho
    ties = np.isclose(d_sy_tr_eq, d_sy_ho)
    dcr_share = float(np.mean(closer_to_train & ~ties) + 0.5 * np.mean(ties))

    # 5. membership inference (distance to closest synthetic record)
    nn_sy = NearestNeighbors(n_neighbors=1).fit(Xsy)
    d_mem = nn_sy.kneighbors(Xtr)[0][:, 0]
    d_non = nn_sy.kneighbors(Xho)[0][:, 0]
    scores = -np.r_[d_mem, d_non] + rng.normal(0, 1e-9, 2 * n)  # tiny jitter breaks ties
    labels = np.r_[np.ones(n), np.zeros(n)]
    mia_auc = float(roc_auc_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    tpr_at_1fpr = float(np.interp(0.01, fpr, tpr))
    advantage = float(max(tpr - fpr))

    levels = {
        "exact_match": risk_level(exact, 0.001, 0.01),
        "close_records": risk_level(close_rate, 0.07, 0.15),
        "nn_disclosure": risk_level(abs(dcr_share - 0.5), 0.03, 0.08),
        "membership_inference": risk_level(mia_auc - 0.5, 0.03, 0.08),
    }
    order = {"low": 0, "medium": 1, "high": 2, "unknown": 0}
    overall = max(levels.values(), key=lambda v: order[v])
    privacy_score = float(np.clip(1 - np.mean([
        min(exact / 0.01, 1), min(max(close_rate - 0.05, 0) / 0.15, 1),
        min(abs(dcr_share - 0.5) / 0.2, 1), min(max(mia_auc - 0.5, 0) / 0.2, 1)]), 0, 1))

    return {
        "n_compared": int(n),
        "exact_match_rate": exact,
        "dcr_synth_median": float(np.median(dcr_sy)),
        "dcr_synth_p5": float(np.percentile(dcr_sy, 5)),
        "dcr_holdout_median": float(np.median(dcr_ho)),
        "dcr_holdout_p5": p5,
        "close_record_rate": close_rate,
        "nndr_median": float(np.median(nndr)),
        "nn_disclosure_share": dcr_share,
        "mia_auc": mia_auc,
        "mia_advantage": advantage,
        "mia_tpr_at_1pct_fpr": tpr_at_1fpr,
        "risk_levels": levels,
        "overall_risk": overall,
        "privacy_score": privacy_score,
        "_dcr_synth": dcr_sy,
        "_dcr_holdout": dcr_ho,
        "_mia_roc": (fpr, tpr),
    }


def explain_privacy(p: dict) -> list[str]:
    """Plain language sentences for the report."""
    lines = [
        f"{p['exact_match_rate'] * 100:.2f}% of synthetic rows are exact copies of a training record "
        f"(risk: {p['risk_levels']['exact_match']}).",
        f"{p['close_record_rate'] * 100:.1f}% of synthetic rows are closer to a training record than 95% of real "
        f"unseen people are; about 5% is expected for a safe generator (risk: {p['risk_levels']['close_records']}).",
        f"For {p['nn_disclosure_share'] * 100:.1f}% of synthetic rows the nearest real person was a training member; "
        f"50% means no disclosure (risk: {p['risk_levels']['nn_disclosure']}).",
        f"A distance based membership inference attacker reaches ROC AUC {p['mia_auc']:.3f} "
        f"(0.5 means random guessing) and TPR {p['mia_tpr_at_1pct_fpr'] * 100:.1f}% at 1% FPR "
        f"(risk: {p['risk_levels']['membership_inference']}).",
    ]
    return lines
