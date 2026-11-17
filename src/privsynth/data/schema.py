"""Automatic schema detection for uploaded CSV files.

Every column gets one of these types:
    numerical   continuous or integer values
    categorical small set of discrete values (strings, low cardinality integers)
    boolean     two valued True/False style columns (handled as categorical)
    datetime    parsable dates, modelled as numeric days and restored afterwards
    id          unique identifiers, dropped (they carry privacy risk and no signal)
    text        high cardinality free text, dropped
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid|^index$|record|patient_?no|ssn|email|phone|name$)", re.I)
MODELLED_TYPES = {"numerical", "categorical", "boolean", "datetime"}


@dataclass
class ColumnInfo:
    name: str
    sdtype: str
    dtype: str
    n_unique: int
    missing_pct: float
    is_integer: bool = False
    categories: list = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    datetime_format: str | None = None
    decimals: int | None = None
    note: str = ""


@dataclass
class Schema:
    columns: dict[str, ColumnInfo]
    target: str | None = None
    task: str | None = None  # classification or regression

    # ---------- column groups ----------
    @property
    def modelled(self) -> list[str]:
        return [c for c, i in self.columns.items() if i.sdtype in MODELLED_TYPES]

    @property
    def numerical(self) -> list[str]:
        return [c for c in self.modelled if self.columns[c].sdtype in ("numerical", "datetime")]

    @property
    def categorical(self) -> list[str]:
        return [c for c in self.modelled if self.columns[c].sdtype in ("categorical", "boolean")]

    @property
    def dropped(self) -> list[str]:
        return [c for c, i in self.columns.items() if i.sdtype not in MODELLED_TYPES]

    # ---------- conversions ----------
    def to_model_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep modelled columns, convert datetimes to float days, categoricals to strings."""
        out = df[self.modelled].copy()
        for c in self.modelled:
            info = self.columns[c]
            if info.sdtype == "datetime":
                ts = pd.to_datetime(out[c], errors="coerce")
                out[c] = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
            elif info.sdtype in ("categorical", "boolean"):
                out[c] = out[c].astype("object").where(out[c].notna(), np.nan)
                out[c] = out[c].map(lambda v: v if pd.isna(v) else str(v)).astype("object")
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        return out

    def from_model_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore original types (integers, dates, category labels) after generation."""
        out = df.copy()
        for c in out.columns:
            info = self.columns.get(c)
            if info is None:
                continue
            if info.sdtype == "datetime":
                days = pd.to_numeric(out[c], errors="coerce")
                ts = pd.Timestamp("1970-01-01") + pd.to_timedelta(days, unit="D")
                out[c] = ts.dt.strftime(info.datetime_format or "%Y-%m-%d")
            elif info.sdtype == "numerical":
                vals = pd.to_numeric(out[c], errors="coerce")
                if info.min is not None and info.max is not None:
                    vals = vals.clip(info.min, info.max)
                if info.is_integer:
                    vals = vals.round().astype("Int64")
                elif info.decimals is not None:
                    vals = vals.round(info.decimals)
                out[c] = vals
            else:
                out[c] = out[c].astype("object")
        return out

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        return {"target": self.target, "task": self.task, "columns": {k: asdict(v) for k, v in self.columns.items()}}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(columns={k: ColumnInfo(**v) for k, v in d["columns"].items()}, target=d.get("target"), task=d.get("task"))

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for c, i in self.columns.items():
            rows.append({"column": c, "type": i.sdtype, "dtype": i.dtype, "unique": i.n_unique,
                         "missing_%": round(i.missing_pct, 2), "used": i.sdtype in MODELLED_TYPES, "note": i.note})
        return pd.DataFrame(rows)


def _looks_like_datetime(s: pd.Series) -> str | None:
    sample = s.dropna().astype(str).head(200)
    if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").all():
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.95:
            return fmt
    return None


def detect_schema(df: pd.DataFrame, target: str | None = None, max_categories: int = 30,
                  drop_columns: list[str] | None = None) -> Schema:
    drop_columns = set(drop_columns or [])
    n = len(df)
    cols: dict[str, ColumnInfo] = {}
    for c in df.columns:
        s = df[c]
        nun = int(s.nunique(dropna=True))
        info = ColumnInfo(name=c, sdtype="categorical", dtype=str(s.dtype), n_unique=nun,
                          missing_pct=float(s.isna().mean() * 100))
        if c in drop_columns:
            info.sdtype, info.note = "id", "dropped by config"
        elif nun <= 1:
            info.sdtype, info.note = "id", "constant column, dropped"
        elif pd.api.types.is_bool_dtype(s) or nun == 2 and c != target and not pd.api.types.is_float_dtype(s):
            info.sdtype = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            vals = s.dropna()
            is_int = bool(np.all(np.isclose(vals, np.round(vals))))
            info.is_integer = is_int
            info.min, info.max = float(vals.min()), float(vals.max())
            if not is_int:  # keep the precision of the real data (for example BMI with 1 decimal)
                info.decimals = int(min(6, vals.head(500).map(lambda v: len(f"{v:.6f}".rstrip("0").split(".")[1])).max()))
            if is_int and nun == n and ID_NAME_PATTERN.search(c):
                info.sdtype, info.note = "id", "unique identifier, dropped"
            elif is_int and nun <= min(10, max_categories) and c != target:
                info.sdtype, info.note = "categorical", "low cardinality integer"
            elif c == target and is_int and nun <= max_categories:
                info.sdtype, info.note = "categorical", "target classes"
            else:
                info.sdtype = "numerical"
        else:
            fmt = _looks_like_datetime(s)
            if fmt:
                info.sdtype, info.datetime_format = "datetime", fmt
            elif ID_NAME_PATTERN.search(c) or nun > 0.9 * n:
                info.sdtype, info.note = "id", "identifier or free text, dropped"
            elif nun > max_categories:
                info.sdtype, info.note = "text", f"more than {max_categories} categories, dropped"
            else:
                info.sdtype = "categorical"
        if info.sdtype in ("categorical", "boolean"):
            info.categories = sorted(map(str, s.dropna().unique().tolist()))
        cols[c] = info

    task = None
    if target is not None:
        if target not in cols:
            raise ValueError(f"Target column '{target}' not found in CSV columns: {list(df.columns)}")
        task = "classification" if cols[target].sdtype in ("categorical", "boolean") else "regression"
    return Schema(columns=cols, target=target, task=task)
