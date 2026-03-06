"""
Hyperparameter Search Space Definitions for Optuna.

Provides ``SearchSpaceRegistry``, a centralized catalogue of parameter
sampling functions, and the ``get_search_space`` factory that assembles
preset search spaces (``quick`` or ``full``) for Optuna studies.

All bounds are read from a ``SearchSpaceOverrides`` Pydantic V2 model,
so ranges can be fully customized via YAML through
``OptunaConfig.search_space_overrides`` without code changes. Defaults
are tuned for image classification and respect the type
constraints defined in ``core/config/types.py``.

Key Functions:
    ``get_search_space``: Factory that returns a preset search space
        dict (``quick`` or ``full``), optionally including model
        architecture selection.

Key Components:
    ``SearchSpaceRegistry``: Resolution-aware registry of sampling
        functions for optimization, regularization, batch size,
        scheduler, and augmentation parameters.

Example:
    >>> space = get_search_space("full", resolution=224, include_models=True)
    >>> study.optimize(OptunaObjective(cfg, space, device), n_trials=50)
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

from ..exceptions import OrchardConfigError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from ..core.config.optuna_config import SearchSpaceOverrides

_SamplerFn = Callable[..., Any]


def _default_overrides() -> SearchSpaceOverrides:
    """Lazy import to avoid circular dependency at module level."""
    from ..core.config.optuna_config import SearchSpaceOverrides

    return SearchSpaceOverrides()


_VIT_WEIGHT_VARIANTS: list[str | None] = [
    None,  # Default variant
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_tiny_patch16_224.augreg_in21k",
]


def _vit_weight_variant_sampler(trial: Any) -> str | None:
    """Conditionally sample ViT weight variant (only active when model_name is vit_tiny)."""
    if trial.params.get("model_name") == "vit_tiny":
        result: str | None = trial.suggest_categorical("weight_variant", _VIT_WEIGHT_VARIANTS)
        return result
    return None


# SEARCH SPACE DEFINITIONS
class SearchSpaceRegistry:
    """
    Centralized registry of hyperparameter search distributions.

    Reads bounds from a SearchSpaceOverrides instance, enabling full
    YAML customization of search ranges without code changes.

    Each method returns a dict of {param_name: suggest_function} where
    suggest_function takes a Trial object and returns a sampled value.

    Args:
        overrides: Configurable search range bounds. Uses defaults if None.
    """

    def __init__(self, overrides: SearchSpaceOverrides | None = None) -> None:
        self.ov = overrides if overrides is not None else _default_overrides()

    def get_optimization_space(self) -> Mapping[str, _SamplerFn]:
        """
        Core optimization hyperparameters (learning rate, weight decay, etc.).

        Returns:
            Immutable mapping of parameter names to sampling functions
        """
        ov = self.ov
        return MappingProxyType(
            {
                "optimizer_type": lambda trial: trial.suggest_categorical(
                    "optimizer_type",
                    ov.optimizer_type,
                ),
                "learning_rate": lambda trial: trial.suggest_float(
                    "learning_rate",
                    ov.learning_rate.low,
                    ov.learning_rate.high,
                    log=ov.learning_rate.log,
                ),
                "weight_decay": lambda trial: trial.suggest_float(
                    "weight_decay",
                    ov.weight_decay.low,
                    ov.weight_decay.high,
                    log=ov.weight_decay.log,
                ),
                "momentum": lambda trial: trial.suggest_float(
                    "momentum",
                    ov.momentum.low,
                    ov.momentum.high,
                ),
                "min_lr": lambda trial: trial.suggest_float(
                    "min_lr",
                    ov.min_lr.low,
                    ov.min_lr.high,
                    log=ov.min_lr.log,
                ),
            }
        )

    def get_loss_space(self) -> Mapping[str, _SamplerFn]:
        """
        Loss function parameters (criterion type, focal gamma, label smoothing).

        ``focal_gamma`` is only sampled when ``criterion_type == "focal"``,
        otherwise defaults to 2.0.  ``label_smoothing`` is only sampled
        when ``criterion_type == "cross_entropy"``, otherwise defaults to 0.0.

        Returns:
            Immutable mapping of loss-related parameter samplers
        """
        # label_smoothing lives here (not in regularization) because it is
        # mutually exclusive with focal_gamma — only the active loss's param
        # is sampled; the other gets a safe default via trial.params dispatch.
        ov = self.ov
        return MappingProxyType(
            {
                "criterion_type": lambda trial: trial.suggest_categorical(
                    "criterion_type",
                    ov.criterion_type,
                ),
                "focal_gamma": lambda trial: (
                    trial.suggest_float(
                        "focal_gamma",
                        ov.focal_gamma.low,
                        ov.focal_gamma.high,
                    )
                    if trial.params.get("criterion_type") == "focal"
                    else 2.0
                ),
                "label_smoothing": lambda trial: (
                    trial.suggest_float(
                        "label_smoothing",
                        ov.label_smoothing.low,
                        ov.label_smoothing.high,
                    )
                    if trial.params.get("criterion_type") == "cross_entropy"
                    else 0.0
                ),
            }
        )

    def get_regularization_space(self) -> Mapping[str, _SamplerFn]:
        """
        Regularization strategies (mixup, dropout).

        Returns:
            Immutable mapping of regularization parameter samplers
        """
        ov = self.ov
        return MappingProxyType(
            {
                "mixup_alpha": lambda trial: trial.suggest_float(
                    "mixup_alpha",
                    ov.mixup_alpha.low,
                    ov.mixup_alpha.high,
                ),
                "dropout": lambda trial: trial.suggest_float(
                    "dropout",
                    ov.dropout.low,
                    ov.dropout.high,
                ),
            }
        )

    def get_batch_size_space(self, resolution: int = 28) -> Mapping[str, _SamplerFn]:
        """
        Batch size as categorical (resolution-aware).

        Args:
            resolution: Input image resolution (e.g. 28, 32, 64, 128, 224)

        Returns:
            Immutable mapping with batch_size sampler
        """
        if resolution >= 224:
            batch_choices = list(self.ov.batch_size_high_res)
        else:
            batch_choices = list(self.ov.batch_size_low_res)

        return MappingProxyType(
            {
                "batch_size": lambda trial: trial.suggest_categorical("batch_size", batch_choices),
            }
        )

    def get_scheduler_space(self) -> Mapping[str, _SamplerFn]:
        """
        Learning rate scheduler parameters.

        Returns:
            Immutable mapping of scheduler-related samplers
        """
        ov = self.ov
        return MappingProxyType(
            {
                "scheduler_type": lambda trial: trial.suggest_categorical(
                    "scheduler_type",
                    ov.scheduler_type,
                ),
                "scheduler_patience": lambda trial: trial.suggest_int(
                    "scheduler_patience",
                    ov.scheduler_patience.low,
                    ov.scheduler_patience.high,
                ),
            }
        )

    def get_augmentation_space(self) -> Mapping[str, _SamplerFn]:
        """
        Data augmentation intensity parameters.

        Returns:
            Immutable mapping of augmentation samplers
        """
        ov = self.ov
        return MappingProxyType(
            {
                "rotation_angle": lambda trial: trial.suggest_int(
                    "rotation_angle",
                    ov.rotation_angle.low,
                    ov.rotation_angle.high,
                ),
                "jitter_val": lambda trial: trial.suggest_float(
                    "jitter_val",
                    ov.jitter_val.low,
                    ov.jitter_val.high,
                ),
                "min_scale": lambda trial: trial.suggest_float(
                    "min_scale",
                    ov.min_scale.low,
                    ov.min_scale.high,
                ),
            }
        )

    def get_full_space(self, resolution: int = 28) -> Mapping[str, _SamplerFn]:
        """
        Combined search space with all available parameters.

        Args:
            resolution: Input image resolution for batch size calculation

        Returns:
            Immutable unified mapping of all parameter samplers
        """
        full_space: dict[str, _SamplerFn] = {}
        full_space.update(self.get_optimization_space())
        full_space.update(self.get_loss_space())
        full_space.update(self.get_regularization_space())
        full_space.update(self.get_batch_size_space(resolution))
        full_space.update(self.get_scheduler_space())
        full_space.update(self.get_augmentation_space())
        return MappingProxyType(full_space)

    def get_quick_space(self, resolution: int = 28) -> Mapping[str, _SamplerFn]:
        """
        Reduced search space for fast exploration (most impactful params).

        Focuses on:

        - Learning rate (most critical)
        - Weight decay
        - Batch size (resolution-aware)
        - Dropout

        Args:
            resolution: Input image resolution for batch size calculation

        Returns:
            Immutable mapping of high-impact parameter samplers
        """
        space: dict[str, _SamplerFn] = {}
        space.update(self.get_optimization_space())
        space.update(
            {
                "batch_size": self.get_batch_size_space(resolution)["batch_size"],
                "dropout": self.get_regularization_space()["dropout"],
            }
        )
        return MappingProxyType(space)

    @staticmethod
    def get_model_space_224() -> Mapping[str, _SamplerFn]:
        """Search space for 224x224 architectures with weight variants."""
        return MappingProxyType(
            {
                "model_name": lambda trial: trial.suggest_categorical(
                    "model_name", ["resnet_18", "efficientnet_b0", "vit_tiny", "convnext_tiny"]
                ),
                "weight_variant": _vit_weight_variant_sampler,
            }
        )

    @staticmethod
    def get_model_space_28() -> Mapping[str, _SamplerFn]:
        """Search space for 28x28 architectures."""
        return MappingProxyType(
            {
                "model_name": lambda trial: trial.suggest_categorical(
                    "model_name", ["resnet_18", "mini_cnn"]
                ),
            }
        )


# PRESET CONFIGURATIONS
def get_search_space(
    preset: str = "quick",
    resolution: int = 28,
    include_models: bool = False,
    model_pool: list[str] | None = None,
    overrides: SearchSpaceOverrides | None = None,
) -> Mapping[str, Any]:
    """
    Factory function to retrieve a search space preset.

    Args:
        preset: Name of the preset ("quick", "full", etc.)
        resolution: Input image resolution (affects batch_size choices)
        include_models: If True, includes model architecture selection
        model_pool: Restrict model search to these architectures.
            When None, uses all built-in models for the target resolution.
        overrides: Configurable search range bounds (uses defaults if None)

    Returns:
        Immutable mapping of parameter samplers keyed by parameter name

    Raises:
        OrchardConfigError: If preset name not recognized
    """
    registry = SearchSpaceRegistry(overrides)

    if preset == "quick":
        space = dict(registry.get_quick_space(resolution))
    elif preset == "full":
        space = dict(registry.get_full_space(resolution))
    else:
        raise OrchardConfigError(f"Unknown preset '{preset}'. Available: quick, full")

    if include_models:
        if model_pool is not None:
            space.update(_build_model_space_from_pool(model_pool))
        elif resolution >= 224:
            space.update(registry.get_model_space_224())
        else:
            space.update(registry.get_model_space_28())

    return MappingProxyType(space)


def _build_model_space_from_pool(pool: list[str]) -> Mapping[str, _SamplerFn]:
    """
    Build model search space from a user-specified pool of architectures.

    Args:
        pool: list of model names to include in the search.

    Returns:
        Immutable mapping with model_name sampler (and weight_variant if vit_tiny is in pool).
    """
    space: dict[str, _SamplerFn] = {
        "model_name": lambda trial, _p=pool: trial.suggest_categorical("model_name", _p),
    }

    if "vit_tiny" in pool:
        space["weight_variant"] = _vit_weight_variant_sampler

    return MappingProxyType(space)
