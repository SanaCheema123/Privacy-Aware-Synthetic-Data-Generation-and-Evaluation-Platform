"""Figures for the evaluation report. Every figure is saved as PNG in outputs/figures/<run_id>/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from ..data.schema import Schema  # noqa: E402
from ..data.transformer import TabularTransformer  # noqa: E402

REAL_C, SYN_C, HOLD_C = "#2a6f97", "#e07a1f", "#6a994e"
GEN_COLORS = {"gaussian_copula": "#4c78a8", "ctgan": "#f58518", "tvae": "#54a24b", "tabddpm": "#b279a2"}

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.labelsize": 9, "font.size": 9})


def _add_logo(fig, logo: Path | None):
    if logo is None:
        return
    try:
        img = plt.imread(str(logo))
        ax = fig.add_axes([0.905, 0.925, 0.085, 0.07], anchor="NE", zorder=10)
        ax.imshow(img)
        ax.axis("off")
    except Exception:
        pass


def _save(fig, path: Path, logo: Path | None) -> Path:
    _add_logo(fig, logo)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distributions(real, synth, schema: Schema, title: str, path: Path, logo=None, max_cols=12) -> Path:
    cols = schema.modelled[:max_cols]
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        if c in schema.numerical:
            r = pd.to_numeric(real[c], errors="coerce").dropna()
            s = pd.to_numeric(synth[c], errors="coerce").dropna()
            bins = np.histogram_bin_edges(np.r_[r, s], bins=30)
            ax.hist(r, bins=bins, density=True, alpha=0.55, color=REAL_C, label="Real")
            ax.hist(s, bins=bins, density=True, alpha=0.55, color=SYN_C, label="Synthetic")
        else:
            pr = real[c].astype(str).value_counts(normalize=True)
            ps = synth[c].astype(str).value_counts(normalize=True)
            idx = pr.index.union(ps.index)[:15]
            x = np.arange(len(idx))
            ax.bar(x - 0.2, pr.reindex(idx, fill_value=0), 0.4, color=REAL_C, label="Real")
            ax.bar(x + 0.2, ps.reindex(idx, fill_value=0), 0.4, color=SYN_C, label="Synthetic")
            ax.set_xticks(x)
            ax.set_xticklabels([str(i)[:10] for i in idx], rotation=35, ha="right")
        ax.set_title(c)
    for ax in axes[len(cols):]:
        ax.axis("off")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Real versus synthetic distributions: {title}", fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, path, logo)


def plot_association(real_assoc, synth_assoc, title: str, path: Path, logo=None) -> Path:
    diff = (real_assoc - synth_assoc).abs()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, m, t, cmap, vmin in [(axes[0], real_assoc, "Real", "RdBu_r", -1), (axes[1], synth_assoc, "Synthetic", "RdBu_r", -1),
                                 (axes[2], diff, "Absolute difference", "Oranges", 0)]:
        im = ax.imshow(m.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=1)
        ax.set_xticks(range(len(m)))
        ax.set_yticks(range(len(m)))
        ax.set_xticklabels([c[:12] for c in m.columns], rotation=90, fontsize=7)
        ax.set_yticklabels([c[:12] for c in m.index], fontsize=7)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Column associations: {title}", fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, path, logo)


def plot_pca(real, synth, schema: Schema, title: str, path: Path, logo=None, seed=42) -> Path:
    n = min(len(real), len(synth), 3000)
    r, s = real.sample(n, random_state=seed), synth.sample(n, random_state=seed)
    tf = TabularTransformer(schema.numerical, schema.categorical).fit(r)
    pca = PCA(2, random_state=seed).fit(tf.transform(r))
    pr, ps = pca.transform(tf.transform(r)), pca.transform(tf.transform(s))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(pr[:, 0], pr[:, 1], s=6, alpha=0.4, color=REAL_C, label="Real")
    ax.scatter(ps[:, 0], ps[:, 1], s=6, alpha=0.4, color=SYN_C, label="Synthetic")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False)
    ax.set_title(f"PCA projection: {title}")
    return _save(fig, path, logo)


def plot_privacy(priv: dict, title: str, path: Path, logo=None) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    dsy, dho = priv["_dcr_synth"], priv["_dcr_holdout"]
    hi = np.percentile(np.r_[dsy, dho], 99)
    bins = np.linspace(0, hi, 40)
    axes[0].hist(dho, bins=bins, alpha=0.55, density=True, color=HOLD_C, label="Real holdout to train")
    axes[0].hist(dsy, bins=bins, alpha=0.55, density=True, color=SYN_C, label="Synthetic to train")
    axes[0].axvline(priv["dcr_holdout_p5"], color="k", ls="--", lw=1, label="Holdout 5th percentile")
    axes[0].set_xlabel("Distance to closest training record")
    axes[0].set_title("Distance to closest record (DCR)")
    axes[0].legend(frameon=False, fontsize=8)
    fpr, tpr = priv["_mia_roc"]
    axes[1].plot(fpr, tpr, color=SYN_C, lw=2, label=f"Attack AUC = {priv['mia_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Membership inference attack ROC")
    axes[1].legend(frameon=False)
    fig.suptitle(f"Privacy risk: {title}", fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, path, logo)


def plot_class_balance(real, synth_by_gen: dict, target: str, path: Path, logo=None) -> Path:
    data = {"Real (train)": real[target].astype(str).value_counts(normalize=True)}
    for g, df in synth_by_gen.items():
        data[g] = df[target].astype(str).value_counts(normalize=True)
    tab = pd.DataFrame(data).fillna(0)
    fig, ax = plt.subplots(figsize=(1.6 * len(tab.columns) + 3, 4))
    bottom = np.zeros(len(tab.columns))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(tab.index):
        ax.bar(tab.columns, tab.loc[cls], bottom=bottom, label=f"{target} = {cls}", color=cmap(i % 10))
        bottom += tab.loc[cls].to_numpy()
    ax.set_ylabel("Share of rows")
    ax.set_title("Class distribution: real versus synthetic")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path, logo)


def plot_leaderboard(board: pd.DataFrame, path: Path, logo=None) -> Path:
    metrics = [m for m in ["fidelity_score", "utility_score", "privacy_score", "overall_score"] if m in board]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(board), 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (_, row) in enumerate(board.iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, [row[m] for m in metrics], w, label=row["generator"],
               color=GEN_COLORS.get(row["key"], None))
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Generator comparison")
    ax.legend(frameon=False, ncol=len(board))
    return _save(fig, path, logo)


def plot_utility(util_tables: dict, primary: str, path: Path, logo=None) -> Path:
    rows = []
    for g, t in util_tables.items():
        sub = t[t.metric == primary]
        for _, r in sub.iterrows():
            rows.append({"generator": g, "model": r["model"], "TSTR": r["TSTR"], "TRTR": r["TRTR"]})
    df = pd.DataFrame(rows)
    models = df["model"].unique().tolist()
    gens = df["generator"].unique().tolist()
    x = np.arange(len(models))
    w = 0.8 / (len(gens) + 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    trtr = df.drop_duplicates("model").set_index("model")["TRTR"].reindex(models)
    ax.bar(x - 0.4 + w / 2, trtr, w, color="#555555", label="Real data (TRTR)")
    for i, g in enumerate(gens):
        vals = df[df.generator == g].set_index("model")["TSTR"].reindex(models)
        ax.bar(x - 0.4 + w * (i + 1.5), vals, w, label=f"{g} (TSTR)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models])
    ax.set_ylabel(primary.replace("_", " "))
    ax.set_title("Downstream model performance on the real holdout set")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path, logo)


def plot_losses(losses: dict, path: Path, logo=None) -> Path | None:
    losses = {k: v for k, v in losses.items() if v is not None and len(v)}
    if not losses:
        return None
    fig, axes = plt.subplots(1, len(losses), figsize=(5 * len(losses), 3.5))
    axes = np.atleast_1d(axes)
    for ax, (g, df) in zip(axes, losses.items()):
        xcol = "step" if "step" in df else ("Epoch" if "Epoch" in df else df.columns[0])
        for c in df.columns:
            if c != xcol and pd.api.types.is_numeric_dtype(df[c]):
                ax.plot(df[xcol], df[c], label=c, lw=1.2)
        ax.set_title(f"{g} training loss")
        ax.set_xlabel(xcol)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(fig, path, logo)
