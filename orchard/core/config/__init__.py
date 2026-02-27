"""
Configuration Package Initialization.

Provides a unified, flat public API for configuration components while
avoiding eager imports of heavy or optional dependencies (e.g. torch).

Architecture:

- Lazy Import Pattern (PEP 562): Uses __getattr__ for on-demand loading
- Deferred Dependencies: torch and pydantic loaded only when needed
- Flat API: All configs accessible from orchard.core.config namespace
- Caching: Loaded modules cached in globals() for performance

Implementation:

1. ``__all__``: Public API contract listing all available configs
2. ``_LAZY_IMPORTS``: Mapping from config names to module paths
3. ``__getattr__``: Dynamic loader triggered on first access
4. ``__dir__``: IDE/introspection support for auto-completion

Example:
    >>> from orchard.core.config import Config, HardwareConfig
    >>> # torch is NOT imported yet (lazy loading)
    >>> cfg = Config.from_recipe(Path("recipes/config_mini_cnn.yaml"))
    >>> # NOW torch is imported (triggered by Config access)
"""

from importlib import import_module
from typing import Any

__all__ = [
    "Config",
    "HardwareConfig",
    "TelemetryConfig",
    "TrainingConfig",
    "AugmentationConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ArchitectureConfig",
    "InfrastructureManager",
    "InfraManagerProtocol",
    "ValidatedPath",
    "OptunaConfig",
    "FloatRange",
    "IntRange",
    "SearchSpaceOverrides",
    "ExportConfig",
    "TrackingConfig",
    "_CrossDomainValidator",
]

# LAZY IMPORTS MAPPING
_PKG = "orchard.core.config"
_OPTUNA_MOD = f"{_PKG}.optuna_config"
_INFRA_MOD = f"{_PKG}.infrastructure_config"

_LAZY_IMPORTS: dict[str, str] = {
    "Config": f"{_PKG}.manifest",
    "HardwareConfig": f"{_PKG}.hardware_config",
    "TelemetryConfig": f"{_PKG}.telemetry_config",
    "TrainingConfig": f"{_PKG}.training_config",
    "AugmentationConfig": f"{_PKG}.augmentation_config",
    "DatasetConfig": f"{_PKG}.dataset_config",
    "EvaluationConfig": f"{_PKG}.evaluation_config",
    "ArchitectureConfig": f"{_PKG}.architecture_config",
    "InfrastructureManager": _INFRA_MOD,
    "InfraManagerProtocol": _INFRA_MOD,
    "ValidatedPath": f"{_PKG}.types",
    "OptunaConfig": _OPTUNA_MOD,
    "FloatRange": _OPTUNA_MOD,
    "IntRange": _OPTUNA_MOD,
    "SearchSpaceOverrides": _OPTUNA_MOD,
    "ExportConfig": f"{_PKG}.export_config",
    "TrackingConfig": f"{_PKG}.tracking_config",
    "_CrossDomainValidator": f"{_PKG}.manifest",
}


# LAZY LOADER FUNCTION
def __getattr__(name: str) -> Any:
    """
    Lazily import configuration components on first access.

    Implements PEP 562 module-level __getattr__ to defer heavy dependency
    imports (torch, pydantic) until the corresponding class is used.

    Args:
        name: Name of the configuration class to import.

    Returns:
        The requested configuration class.

    Raises:
        AttributeError: If name is not in the public API (__all__).
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
    """
    Support for dir() and IDE auto-completion.

    Returns:
        Sorted list of public configuration class names
    """
    return sorted(__all__)
