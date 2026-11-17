"""CTGAN, TVAE and Gaussian Copula generators built on the SDV library."""
from __future__ import annotations

import warnings

import pandas as pd

from ..data.schema import Schema
from .base import BaseGenerator


TABLE = "table"


def _as_frame(out) -> pd.DataFrame:
    """SDV 2.x returns {table_name: DataFrame}; SDV 1.x returns a DataFrame."""
    if isinstance(out, dict):
        return next(iter(out.values()))
    return out


def build_sdv_metadata(schema: Schema, columns: list[str]):
    from sdv.metadata import Metadata

    sdtype_map = {"numerical": "numerical", "datetime": "numerical",  # dates arrive as float days
                  "categorical": "categorical", "boolean": "categorical"}
    cols = {c: {"sdtype": sdtype_map[schema.columns[c].sdtype]} for c in columns}
    return Metadata.load_from_dict({"tables": {TABLE: {"columns": cols}}})


class _SDVGenerator(BaseGenerator):
    supports_native_conditioning = True

    def _make_synth(self, metadata):
        raise NotImplementedError

    def _fit(self, df: pd.DataFrame, schema: Schema) -> None:
        metadata = build_sdv_metadata(schema, list(df.columns))
        self.model_ = self._make_synth(metadata)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:  # SDV 1.x accepts a DataFrame
                self.model_.fit(df)
            except Exception as e:  # SDV 2.x expects {table_name: DataFrame}
                if "dictionary" not in str(e).lower():
                    raise
                self.model_.fit({TABLE: df})

    def sample(self, n: int) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _as_frame(self.model_.sample(num_rows=int(n)))

    def sample_conditional(self, target: str, class_counts: dict[str, int], max_rounds: int = 30) -> pd.DataFrame:
        from sdv.sampling import Condition

        conditions = [Condition(num_rows=int(k), column_values={target: str(cls)})
                      for cls, k in class_counts.items() if k > 0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = _as_frame(self.model_.sample_from_conditions(conditions=conditions, max_tries_per_batch=200))
        except Exception:  # fall back to rejection sampling
            return super().sample_conditional(target, class_counts, max_rounds)
        # SDV may silently return fewer rows for classes the model rarely produces
        got = out[target].astype(str).value_counts()
        short = {str(c): int(k) - int(got.get(str(c), 0)) for c, k in class_counts.items()}
        short = {c: k for c, k in short.items() if k > 0}
        self.class_shortfall_ = {}
        if short:
            extra = super().sample_conditional(target, short, max_rounds)
            out = pd.concat([out, extra], ignore_index=True)
        return out.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

    def loss_history(self) -> pd.DataFrame | None:
        try:
            return self.model_.get_loss_values()
        except Exception:
            return None


class GaussianCopulaGenerator(_SDVGenerator):
    name, display_name = "gaussian_copula", "Gaussian Copula"

    def _make_synth(self, metadata):
        from sdv.single_table import GaussianCopulaSynthesizer

        return GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True, enforce_rounding=True,
                                         default_distribution=self.params.get("default_distribution", "beta"))


class CTGANGenerator(_SDVGenerator):
    name, display_name = "ctgan", "CTGAN"

    def _make_synth(self, metadata):
        from sdv.single_table import CTGANSynthesizer

        return CTGANSynthesizer(metadata, epochs=int(self.params.get("epochs", 300)),
                                batch_size=int(self.params.get("batch_size", 500)),
                                pac=int(self.params.get("pac", 10)), verbose=False)


class TVAEGenerator(_SDVGenerator):
    name, display_name = "tvae", "TVAE"

    def _make_synth(self, metadata):
        from sdv.single_table import TVAESynthesizer

        return TVAESynthesizer(metadata, epochs=int(self.params.get("epochs", 300)),
                               batch_size=int(self.params.get("batch_size", 500)), verbose=False)
