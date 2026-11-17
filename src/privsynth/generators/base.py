"""Common interface shared by every synthetic data generator."""
from __future__ import annotations

import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ..data.schema import Schema


class BaseGenerator(ABC):
    name: str = "base"
    display_name: str = "Base"
    supports_native_conditioning: bool = False

    def __init__(self, params: dict | None = None, seed: int = 42):
        self.params = params or {}
        self.seed = seed
        self.fit_seconds: float = 0.0
        self.schema: Schema | None = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, schema: Schema) -> "BaseGenerator":
        """df must already be a model frame (see Schema.to_model_frame)."""
        self.schema = schema
        t0 = time.time()
        self._fit(df, schema)
        self.fit_seconds = time.time() - t0
        return self

    @abstractmethod
    def _fit(self, df: pd.DataFrame, schema: Schema) -> None: ...

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame: ...

    def sample_conditional(self, target: str, class_counts: dict[str, int], max_rounds: int = 30) -> pd.DataFrame:
        """Generate an exact number of rows per class.

        Default implementation uses rejection sampling: draw batches and keep rows of
        the classes that still need rows. Generators that can condition natively
        override this method.
        """
        need = {str(k): int(v) for k, v in class_counts.items() if v > 0}
        chunks = []
        total = sum(need.values())
        for _ in range(max_rounds):
            if not any(need.values()):
                break
            batch = self.sample(max(total, 1000))
            batch[target] = batch[target].astype(str)
            for cls, k in list(need.items()):
                if k <= 0:
                    continue
                rows = batch[batch[target] == cls].head(k)
                if len(rows):
                    chunks.append(rows)
                    need[cls] -= len(rows)
        self.class_shortfall_ = {k: v for k, v in need.items() if v > 0}
        out = pd.concat(chunks, ignore_index=True) if chunks else self.sample(1).head(0)
        return out.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: Path) -> "BaseGenerator":
        with open(path, "rb") as f:
            return pickle.load(f)
