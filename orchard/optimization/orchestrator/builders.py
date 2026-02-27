"""
Factory Functions for Optuna Components.

Provides builder functions that construct Optuna samplers, pruners,
and callbacks based on the Optuna configuration sub-model. Centralizes
the instantiation logic and provides clear error messages for invalid
configurations.

Functions:

- build_sampler: Create Optuna sampler from type string
- build_pruner: Create Optuna pruner from config
- build_callbacks: Construct optimization callbacks list
"""

from __future__ import annotations

import logging
from typing import cast

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PercentilePruner

from ...core import LOGGER_NAME, OptunaConfig
from ..early_stopping import get_early_stopping_callback
from .config import PRUNER_REGISTRY, SAMPLER_REGISTRY

logger = logging.getLogger(LOGGER_NAME)


# CONFIGURATION BUILDERS
def build_sampler(optuna_cfg: OptunaConfig) -> optuna.samplers.BaseSampler:
    """
    Create Optuna sampler from configuration.

    Args:
        optuna_cfg: Optuna sub-config with sampler_type

    Returns:
        Configured Optuna sampler instance

    Raises:
        ValueError: If sampler_type is not in SAMPLER_REGISTRY
    """
    sampler_cls = SAMPLER_REGISTRY.get(optuna_cfg.sampler_type)
    if sampler_cls is None:
        raise ValueError(
            f"Unknown sampler: {optuna_cfg.sampler_type}. "
            f"Valid options: {list(SAMPLER_REGISTRY.keys())}"
        )
    return sampler_cls()  # type: ignore[no-any-return]


def build_pruner(
    optuna_cfg: OptunaConfig,
) -> MedianPruner | PercentilePruner | HyperbandPruner | NopPruner:
    """
    Create Optuna pruner from configuration.

    Args:
        optuna_cfg: Optuna sub-config with enable_pruning and pruner_type

    Returns:
        Configured Optuna pruner instance (NopPruner if disabled)

    Raises:
        ValueError: If pruner_type is not in PRUNER_REGISTRY
    """
    if not optuna_cfg.enable_pruning:
        return NopPruner()

    pruner_factory = PRUNER_REGISTRY.get(optuna_cfg.pruner_type)
    if pruner_factory is None:
        raise ValueError(
            f"Unknown pruner: {optuna_cfg.pruner_type}. "
            f"Valid options: {list(PRUNER_REGISTRY.keys())}"
        )
    # type narrowing: PRUNER_REGISTRY values are concrete pruner factories
    return cast(MedianPruner | PercentilePruner | HyperbandPruner | NopPruner, pruner_factory())


def build_callbacks(optuna_cfg: OptunaConfig, monitor_metric: str) -> list:
    """
    Construct list of optimization callbacks from configuration.

    Currently supports:

        - Early stopping callback (based on metric threshold)

    Args:
        optuna_cfg: Optuna sub-config with early stopping parameters
        monitor_metric: Target metric name (from ``training.monitor_metric``)

    Returns:
        list of Optuna callback objects (may be empty)

    Example:
        >>> callbacks = build_callbacks(optuna_cfg, "auc")
        >>> len(callbacks)  # 0 or 1 depending on early_stopping config
    """
    early_stop_callback = get_early_stopping_callback(
        direction=optuna_cfg.direction,
        threshold=optuna_cfg.early_stopping_threshold,
        patience=optuna_cfg.early_stopping_patience,
        enabled=optuna_cfg.enable_early_stopping,
        metric_name=monitor_metric,
    )

    return [early_stop_callback] if early_stop_callback else []
