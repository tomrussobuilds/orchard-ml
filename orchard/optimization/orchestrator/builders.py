"""
Factory Functions for Optuna Components.

Provides builder functions that construct Optuna samplers, pruners,
and callbacks based on configuration strings. Centralizes the
instantiation logic and provides clear error messages for invalid
configurations.

Functions:
    - build_sampler: Create Optuna sampler from type string
    - build_pruner: Create Optuna pruner from config
    - build_callbacks: Construct optimization callbacks list
"""

# Standard Imports
import logging
from typing import List

# Third-Party Imports
import optuna
from optuna.pruners import NopPruner

# Internal Imports
from orchard.core import LOGGER_NAME, Config

from ..early_stopping import get_early_stopping_callback

# Relative Imports
from .config import PRUNER_REGISTRY, SAMPLER_REGISTRY

logger = logging.getLogger(LOGGER_NAME)


# CONFIGURATION BUILDERS
def build_sampler(sampler_type: str, cfg: Config) -> optuna.samplers.BaseSampler:
    """
    Create Optuna sampler from configuration string.

    Args:
        sampler_type: Sampler algorithm name ("tpe", "cmaes", "random", "grid")

    Returns:
        Configured Optuna sampler instance

    Raises:
        ValueError: If sampler_type is not in SAMPLER_REGISTRY

    Example:
        >>> sampler = build_sampler("tpe")
        >>> isinstance(sampler, optuna.samplers.TPESampler)
        True
    """
    sampler_cls = SAMPLER_REGISTRY.get(cfg.optuna.sampler_type)
    if sampler_cls is None:
        raise ValueError(
            f"Unknown sampler: {cfg.optuna.sampler_type}. "
            f"Valid options: {list(SAMPLER_REGISTRY.keys())}"
        )
    return sampler_cls()


def build_pruner(enable_pruning: bool, pruner_type: str, cfg: Config) -> optuna.pruners.BasePruner:
    """
    Create Optuna pruner from configuration.

    Args:
        enable_pruning: Whether pruning is enabled
        pruner_type: Pruner algorithm name ("median", "percentile", "hyperband", "none")

    Returns:
        Configured Optuna pruner instance (NopPruner if disabled)

    Raises:
        ValueError: If pruner_type is not in PRUNER_REGISTRY

    Example:
        >>> pruner = build_pruner(enable_pruning=True, pruner_type="median")
        >>> isinstance(pruner, optuna.pruners.MedianPruner)
        True
    """
    if not cfg.optuna.enable_pruning:
        return NopPruner()

    pruner_factory = PRUNER_REGISTRY.get(cfg.optuna.pruner_type)
    if pruner_factory is None:
        raise ValueError(
            f"Unknown pruner: {cfg.optuna.pruner_type}. "
            f"Valid options: {list(PRUNER_REGISTRY.keys())}"
        )
    return pruner_factory()


def build_callbacks(cfg: Config) -> List:
    """
    Construct list of optimization callbacks from configuration.

    Currently supports:
        - Early stopping callback (based on metric threshold)

    Args:
        cfg: Global configuration with optuna section

    Returns:
        List of Optuna callback objects (may be empty)

    Example:
        >>> callbacks = build_callbacks(cfg)
        >>> len(callbacks)  # 0 or 1 depending on early_stopping config
    """
    early_stop_callback = get_early_stopping_callback(
        direction=cfg.optuna.direction,
        threshold=cfg.optuna.early_stopping_threshold,
        patience=cfg.optuna.early_stopping_patience,
        enabled=cfg.optuna.enable_early_stopping,
        metric_name=cfg.optuna.metric_name,
    )

    return [early_stop_callback] if early_stop_callback else []
