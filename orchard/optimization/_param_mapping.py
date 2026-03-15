"""
Hyperparameter-to-Config Section Mapping.

Single source of truth for mapping Optuna's flat parameter namespace
to Config's hierarchical structure. Used by both trial construction
(``TrialConfigBuilder``) and best-config export (``build_best_config_dict``).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

# ==================== PARAMETER MAPPING (single source of truth) ===========

PARAM_MAPPING: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "training": frozenset(
            {
                "optimizer_type",
                "learning_rate",
                "weight_decay",
                "momentum",
                "min_lr",
                "mixup_alpha",
                "label_smoothing",
                "criterion_type",
                "focal_gamma",
                "batch_size",
                "scheduler_type",
                "scheduler_patience",
            }
        ),
        "architecture": frozenset({"dropout", "weight_variant"}),
        "augmentation": frozenset({"rotation_angle", "jitter_val", "min_scale"}),
    }
)

SPECIAL_PARAMS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "model_name": ("architecture", "name"),
    }
)

# Derived set for fast lookup in map_param_to_config_path
_SECTION_LOOKUP: Mapping[str, str] = MappingProxyType(
    {param: section for section, params in PARAM_MAPPING.items() for param in params}
)


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
    if param_name in SPECIAL_PARAMS:
        return SPECIAL_PARAMS[param_name]
    section = _SECTION_LOOKUP.get(param_name)
    if section is not None:
        return (section, param_name)
    raise ValueError(
        f"Unknown hyperparameter '{param_name}'. "
        f"Add it to PARAM_MAPPING or SPECIAL_PARAMS in _param_mapping.py."
    )
