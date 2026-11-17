"""Configuration loading and path helpers."""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(resolve(path) if path else DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, overrides: dict) -> dict:
    """Return a copy of base with nested overrides applied."""
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def logo_path(cfg: dict) -> Path | None:
    p = resolve(cfg.get("project", {}).get("logo_path", "assets/aivonex_logo.png"))
    return p if p.exists() else None
