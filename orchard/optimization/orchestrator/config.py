"""
Optuna Configuration Constants and Registries.

Centralized definitions for:
    - Sampler type registry (TPE, CmaES, Random, Grid)
    - Pruner type registry (Median, Percentile, Hyperband)
    - Parameter-to-config mapping (training/model/augmentation sections)

These registries enable the factory pattern in builders.py and
provide a single point of maintenance for supported algorithms.
"""

# Standard Imports
import logging
from typing import Callable, Dict, Set, Tuple

# Third-Party Imports
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PercentilePruner
from optuna.samplers import CmaEsSampler, GridSampler, RandomSampler, TPESampler

# Internal Imports
from orchard.core import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Type aliases for clarity
SamplerFactory = Callable[[], object]
PrunerFactory = Callable[[], object]

# ==================== SAMPLER REGISTRY ====================

SAMPLER_REGISTRY: Dict[str, type] = {
    "tpe": TPESampler,
    "cmaes": CmaEsSampler,
    "random": RandomSampler,
    "grid": GridSampler,
}
"""Registry mapping sampler type strings to Optuna sampler classes."""

# ==================== PRUNER REGISTRY ====================

PRUNER_REGISTRY: Dict[str, PrunerFactory] = {
    "median": MedianPruner,
    "percentile": lambda: PercentilePruner(percentile=25.0),
    "hyperband": HyperbandPruner,
    "none": NopPruner,
}
"""Registry mapping pruner type strings to Optuna pruner factories."""

# ==================== PARAMETER MAPPING ====================

TRAINING_PARAMS: Set[str] = {
    "learning_rate",
    "weight_decay",
    "momentum",
    "min_lr",
    "mixup_alpha",
    "label_smoothing",
    "batch_size",
    "cosine_fraction",
    "scheduler_patience",
}
"""Hyperparameters that belong in the training section of Config."""

MODEL_PARAMS: Set[str] = {
    "dropout",
}
"""Hyperparameters that belong in the model section of Config."""

AUGMENTATION_PARAMS: Set[str] = {
    "rotation_angle",
    "jitter_val",
    "min_scale",
}
"""Hyperparameters that belong in the augmentation section of Config."""

SPECIAL_PARAMS: Dict[str, Tuple[str, str]] = {
    "model_name": ("model", "name"),
    "weight_variant": ("model", "weight_variant"),
}

# ==================== HELPER FUNCTIONS ====================


def map_param_to_config_path(param_name: str) -> Tuple[str, str]:
    """
    Map hyperparameter name to its location in Config hierarchy.

    Args:
        param_name: Name of the hyperparameter from Optuna trial

    Returns:
        Tuple of (section, key) for navigating the config dict

    Example:
        >>> section, key = map_param_to_config_path("learning_rate")
        >>> # Returns: ("training", "learning_rate")
        >>> config_dict[section][key] = 0.001
    """
    if param_name in TRAINING_PARAMS:
        return ("training", param_name)
    elif param_name in MODEL_PARAMS:
        return ("model", param_name)
    elif param_name in AUGMENTATION_PARAMS:
        return ("augmentation", param_name)
    elif param_name in SPECIAL_PARAMS:
        return SPECIAL_PARAMS[param_name]
    else:
        # Fallback: assume it's a training parameter
        logger.warning(f"Unknown parameter '{param_name}', defaulting to training section")
        return ("training", param_name)
