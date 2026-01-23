"""
Configuration Package Initialization.

Provides a unified, flat public API for configuration components while
avoiding eager imports of heavy or optional dependencies (e.g. torch).

All heavy modules are imported lazily via __getattr__ (PEP 562).
"""

# Standard Imports
from __future__ import annotations

from importlib import import_module
from typing import Any

# PUBLIC API
__all__ = [
    "Config",
    "HardwareConfig",
    "TelemetryConfig",
    "TrainingConfig",
    "AugmentationConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ModelConfig",
    "InfrastructureManager",
    "ValidatedPath",
    "OptunaConfig",
]

# LAZY IMPORTS MAPPING
_LAZY_IMPORTS: dict[str, str] = {
    "Config": "orchard.core.config.engine",
    "HardwareConfig": "orchard.core.config.hardware_config",
    "TelemetryConfig": "orchard.core.config.telemetry_config",
    "TrainingConfig": "orchard.core.config.training_config",
    "AugmentationConfig": "orchard.core.config.augmentation_config",
    "DatasetConfig": "orchard.core.config.dataset_config",
    "EvaluationConfig": "orchard.core.config.evaluation_config",
    "ModelConfig": "orchard.core.config.models_config",
    "InfrastructureManager": "orchard.core.config.infrastructure_config",
    "ValidatedPath": "orchard.core.config.types",
    "OptunaConfig": "orchard.core.config.optuna_config",
}


# LAZY LOADER FUNCTION
def __getattr__(name: str) -> Any:
    """
    Lazily import configuration components on first access.

    Prevents importing heavy dependencies (e.g. torch) unless the
    corresponding configuration class is actually used.
    """
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_IMPORTS[name])
    attr = getattr(module, name)

    # Cache on module for future access
    globals()[name] = attr
    return attr


# DIR SUPPORT
def __dir__() -> list[str]:
    return sorted(__all__)
