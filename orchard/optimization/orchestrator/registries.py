"""
Optuna Component Registries.

Centralized definitions for:

- Sampler type registry (TPE, CmaES, Random)
- Pruner type registry (Median, Percentile, Hyperband, Nop)

These registries enable the factory pattern in builders.py and
provide a single point of maintenance for supported algorithms.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping

from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PercentilePruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

# type aliases for clarity
PrunerFactory = Callable[[], object]

# ==================== SAMPLER REGISTRY ====================

SAMPLER_REGISTRY: Mapping[str, type] = MappingProxyType(
    {
        "tpe": TPESampler,
        "cmaes": CmaEsSampler,
        "random": RandomSampler,
    }
)

# ==================== PRUNER REGISTRY ====================

PRUNER_REGISTRY: Mapping[str, PrunerFactory] = MappingProxyType(
    {
        "median": MedianPruner,
        "percentile": lambda: PercentilePruner(percentile=25.0),
        "hyperband": HyperbandPruner,
        "none": NopPruner,
    }
)
