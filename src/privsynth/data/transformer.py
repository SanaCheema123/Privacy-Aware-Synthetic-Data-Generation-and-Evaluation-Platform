"""Reversible numeric encoding of mixed tabular data.

Used by TabDDPM (quantile mode) and by distance based privacy metrics (standard mode).
Numerical columns: median imputation plus a missing indicator when needed, then
standard scaling or quantile to normal transform. Categorical columns: one hot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

MISSING_TOKEN = "__missing__"


class TabularTransformer:
    def __init__(self, numerical: list[str], categorical: list[str], mode: str = "standard",
                 seed: int = 42, cat_scale: float = 1.0):
        assert mode in ("standard", "quantile")
        self.numerical = list(numerical)
        self.categorical = list(categorical)
        self.mode = mode
        self.seed = seed
        self.cat_scale = cat_scale  # one hot values are scaled by this factor

    def fit(self, df: pd.DataFrame) -> "TabularTransformer":
        self.medians_ = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in self.numerical}
        if self.numerical:
            X = self._num_raw(df)
            if self.mode == "quantile":
                self.num_tf_ = QuantileTransformer(output_distribution="normal",
                                                   n_quantiles=max(min(len(df) // 5, 1000), 10),
                                                   subsample=1_000_000, random_state=self.seed)
            else:
                self.num_tf_ = StandardScaler()
            self.num_tf_.fit(X)
        self.cats_ = {}
        for c in self.categorical:
            vals = df[c].astype("object").where(df[c].notna(), MISSING_TOKEN).astype(str)
            self.cats_[c] = sorted(vals.unique().tolist())
        self.cat_sizes_ = [len(self.cats_[c]) for c in self.categorical]
        self.output_dim_ = len(self.numerical) + sum(self.cat_sizes_)
        return self

    def _num_raw(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.numerical:
            v = pd.to_numeric(df[c], errors="coerce").astype(float)
            cols.append(v.fillna(self.medians_[c]).to_numpy())
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.numerical:
            parts.append(self.num_tf_.transform(self._num_raw(df)))
        for c in self.categorical:
            cats = self.cats_[c]
            idx = {k: i for i, k in enumerate(cats)}
            vals = df[c].astype("object").where(df[c].notna(), MISSING_TOKEN).astype(str)
            oh = np.zeros((len(df), len(cats)))
            pos = vals.map(idx).to_numpy()
            ok = ~pd.isna(pos)
            oh[np.where(ok)[0], pos[ok].astype(int)] = 1.0
            parts.append(oh * self.cat_scale)
        return np.hstack(parts).astype(np.float32) if parts else np.zeros((len(df), 0), np.float32)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        out = {}
        k = len(self.numerical)
        if k:
            num = self.num_tf_.inverse_transform(np.asarray(X[:, :k], dtype=float))
            for j, c in enumerate(self.numerical):
                out[c] = num[:, j]
        pos = k
        for c, size in zip(self.categorical, self.cat_sizes_):
            block = X[:, pos:pos + size]
            labels = np.array(self.cats_[c], dtype=object)[np.argmax(block, axis=1)]
            labels[labels == MISSING_TOKEN] = np.nan
            out[c] = labels
            pos += size
        return pd.DataFrame(out)

    @property
    def cat_slices(self) -> list[tuple[str, int, int]]:
        res, pos = [], len(self.numerical)
        for c, size in zip(self.categorical, self.cat_sizes_):
            res.append((c, pos, pos + size))
            pos += size
        return res
