"""CSV loading and real data splitting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV with light cleaning (strip column names, drop empty rows/columns)."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df.reset_index(drop=True)


def split_real(df: pd.DataFrame, target: str | None, holdout_size: float, seed: int,
               stratify: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split real data into a training part (seen by generators) and a holdout part.

    The holdout is used as the real test set for TSTR/TRTR and as the
    non member set for membership inference and DCR baselines.
    """
    strat = None
    if stratify and target is not None and target in df:
        counts = df[target].value_counts()
        if counts.min() >= 2 and len(counts) <= 50:
            strat = df[target]
    train, holdout = train_test_split(df, test_size=holdout_size, random_state=seed, stratify=strat)
    return train.reset_index(drop=True), holdout.reset_index(drop=True)
