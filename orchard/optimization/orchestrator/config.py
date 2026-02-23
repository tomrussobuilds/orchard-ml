"""
Optuna Configuration Constants and Registries.

Centralized definitions for:
    - Sampler type registry (TPE, CmaES, Random, Grid)
    - Pruner type registry (Median, Percentile, Hyperband)
    - Parameter-to-config mapping (training/architecture/augmentation sections)

These registries enable the factory pattern in builders.py and
provide a single point of maintenance for supported algorithms.
"""

from __future__ import annotations

import logging
from typing import Callable

from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PercentilePruner
from optuna.samplers import CmaEsSampler, GridSampler, RandomSampler, TPESampler

from ...core import LOGGER_NAME

# Global logger instance
logger = logging.getLogger(LOGGER_NAME)


# type aliases for clarity
SamplerFactory = Callable[[], object]
PrunerFactory = Callable[[], object]

# ==================== SAMPLER REGISTRY ====================

SAMPLER_REGISTRY: dict[str, type] = {
    "tpe": TPESampler,
    "cmaes": CmaEsSampler,
    "random": RandomSampler,
    "grid": GridSampler,
}
"""Registry mapping sampler type strings to Optuna sampler classes."""

# ==================== PRUNER REGISTRY ====================

PRUNER_REGISTRY: dict[str, PrunerFactory] = {
    "median": MedianPruner,
    "percentile": lambda: PercentilePruner(percentile=25.0),
    "hyperband": HyperbandPruner,
    "none": NopPruner,
}
"""Registry mapping pruner type strings to Optuna pruner factories."""

# ==================== PARAMETER MAPPING ====================

TRAINING_PARAMS: set[str] = {
    "learning_rate",
    "weight_decay",
    "momentum",
    "min_lr",
    "mixup_alpha",
    "label_smoothing",
    "batch_size",
    "scheduler_type",
    "scheduler_patience",
}
"""Hyperparameters that belong in the training section of Config."""

ARCHITECTURE_PARAMS: set[str] = {
    "dropout",
}
"""Hyperparameters that belong in the architecture section of Config."""

AUGMENTATION_PARAMS: set[str] = {
    "rotation_angle",
    "jitter_val",
    "min_scale",
}
"""Hyperparameters that belong in the augmentation section of Config."""

SPECIAL_PARAMS: dict[str, tuple[str, str]] = {
    "model_name": ("architecture", "name"),
    "weight_variant": ("architecture", "weight_variant"),
}

# ==================== HELPER FUNCTIONS ====================


def map_param_to_config_path(param_name: str) -> tuple[str, str]:
    """
    Map hyperparameter name to its location in Config hierarchy.

    Args:
        param_name: Name of the hyperparameter from Optuna trial

    Returns:
        Tuple of ``(section, field_name)`` for navigating the config dict

    Example:
        >>> result = map_param_to_config_path("learning_rate")
        >>> result
        ('training', 'learning_rate')
    """
    if param_name in TRAINING_PARAMS:
        return ("training", param_name)
    elif param_name in ARCHITECTURE_PARAMS:
        return ("architecture", param_name)
    elif param_name in AUGMENTATION_PARAMS:
        return ("augmentation", param_name)
    elif param_name in SPECIAL_PARAMS:
        return SPECIAL_PARAMS[param_name]
    else:
        # Fallback: assume it's a training parameter
        logger.warning(f"Unknown parameter '{param_name}', defaulting to training section")
        return ("training", param_name)
