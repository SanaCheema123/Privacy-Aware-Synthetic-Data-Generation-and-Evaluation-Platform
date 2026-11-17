"""Generator registry."""
from __future__ import annotations

from .base import BaseGenerator
from .sdv_models import CTGANGenerator, GaussianCopulaGenerator, TVAEGenerator
from .tabddpm import TabDDPMGenerator

REGISTRY: dict[str, type[BaseGenerator]] = {
    "gaussian_copula": GaussianCopulaGenerator,
    "ctgan": CTGANGenerator,
    "tvae": TVAEGenerator,
    "tabddpm": TabDDPMGenerator,
}

DISPLAY_NAMES = {k: v.display_name for k, v in REGISTRY.items()}


def get_generator(name: str, params: dict | None = None, seed: int = 42) -> BaseGenerator:
    if name not in REGISTRY:
        raise KeyError(f"Unknown generator '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](params=params, seed=seed)


__all__ = ["BaseGenerator", "REGISTRY", "DISPLAY_NAMES", "get_generator"]
