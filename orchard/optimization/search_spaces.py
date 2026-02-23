"""
Hyperparameter Search Space Definitions for Optuna.

Defines search distributions for each optimizable parameter using
configurable bounds from SearchSpaceOverrides (Pydantic V2).

Search ranges respect the type constraints in core/config/types.py
and default to domain-expert values for medical imaging, but can be
fully customized via YAML through OptunaConfig.search_space_overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover
    from ..core.config.optuna_config import SearchSpaceOverrides


def _default_overrides() -> SearchSpaceOverrides:
    """Lazy import to avoid circular dependency at module level."""
    from ..core.config.optuna_config import SearchSpaceOverrides

    return SearchSpaceOverrides()


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

    def get_optimization_space(self) -> dict[str, Callable]:
        """
        Core optimization hyperparameters (learning rate, weight decay, etc.).

        Returns:
            dict mapping parameter names to sampling functions
        """
        ov = self.ov
        return {
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

    def get_regularization_space(self) -> dict[str, Callable]:
        """
        Regularization strategies (mixup, label smoothing, dropout).

        Returns:
            dict of regularization parameter samplers
        """
        ov = self.ov
        return {
            "mixup_alpha": lambda trial: trial.suggest_float(
                "mixup_alpha",
                ov.mixup_alpha.low,
                ov.mixup_alpha.high,
            ),
            "label_smoothing": lambda trial: trial.suggest_float(
                "label_smoothing",
                ov.label_smoothing.low,
                ov.label_smoothing.high,
            ),
            "dropout": lambda trial: trial.suggest_float(
                "dropout",
                ov.dropout.low,
                ov.dropout.high,
            ),
        }

    def get_batch_size_space(self, resolution: int = 28) -> dict[str, Callable]:
        """
        Batch size as categorical (resolution-aware).

        Args:
            resolution: Input image resolution (28 or 224)

        Returns:
            dict with batch_size sampler
        """
        if resolution >= 224:
            batch_choices = list(self.ov.batch_size_high_res)
        else:
            batch_choices = list(self.ov.batch_size_low_res)

        return {
            "batch_size": lambda trial: trial.suggest_categorical("batch_size", batch_choices),
        }

    def get_scheduler_space(self) -> dict[str, Callable]:
        """
        Learning rate scheduler parameters.

        Returns:
            dict of scheduler-related samplers
        """
        ov = self.ov
        return {
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

    def get_augmentation_space(self) -> dict[str, Callable]:
        """
        Data augmentation intensity parameters.

        Returns:
            dict of augmentation samplers
        """
        ov = self.ov
        return {
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

    def get_full_space(self, resolution: int = 28) -> dict[str, Callable]:
        """
        Combined search space with all available parameters.

        Args:
            resolution: Input image resolution for batch size calculation

        Returns:
            Unified dict of all parameter samplers
        """
        full_space: dict[str, Callable] = {}
        full_space.update(self.get_optimization_space())
        full_space.update(self.get_regularization_space())
        full_space.update(self.get_batch_size_space(resolution))
        full_space.update(self.get_scheduler_space())
        full_space.update(self.get_augmentation_space())
        return full_space

    def get_quick_space(self, resolution: int = 28) -> dict[str, Callable]:
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
            dict of high-impact parameter samplers
        """
        space: dict[str, Callable] = {}
        space.update(self.get_optimization_space())
        space.update(
            {
                "batch_size": self.get_batch_size_space(resolution)["batch_size"],
                "dropout": self.get_regularization_space()["dropout"],
            }
        )
        return space

    @staticmethod
    def get_model_space_224() -> dict[str, Callable]:
        """Search space for 224x224 architectures with weight variants."""
        return {
            "model_name": lambda trial: trial.suggest_categorical(
                "model_name", ["resnet_18", "efficientnet_b0", "vit_tiny", "convnext_tiny"]
            ),
            "weight_variant": lambda trial: (
                trial.suggest_categorical(
                    "weight_variant",
                    [
                        None,  # Default variant
                        "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                        "vit_tiny_patch16_224.augreg_in21k",
                    ],
                )
                if trial.params.get("model_name") == "vit_tiny"
                else None
            ),
        }

    @staticmethod
    def get_model_space_28() -> dict[str, Callable]:
        """Search space for 28x28 architectures."""
        return {
            "model_name": lambda trial: trial.suggest_categorical(
                "model_name", ["resnet_18", "mini_cnn"]
            ),
        }

    def get_full_space_with_models(self, resolution: int = 28) -> dict[str, Callable]:
        """
        Combined search space including model architecture selection.

        Args:
            resolution: Input image resolution (determines model choices)

        Returns:
            Unified dict of all parameter samplers + model selection
        """
        full_space = self.get_full_space(resolution)

        if resolution >= 224:
            full_space.update(self.get_model_space_224())
        else:
            full_space.update(self.get_model_space_28())

        return full_space


# PRESET CONFIGURATIONS
def get_search_space(
    preset: str = "quick",
    resolution: int = 28,
    include_models: bool = False,
    model_pool: list[str] | None = None,
    overrides: SearchSpaceOverrides | None = None,
) -> dict[str, Any]:
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
        dict[str, Any]: Dictionary of parameter samplers keyed by parameter name

    Raises:
        ValueError: If preset name not recognized
    """
    registry = SearchSpaceRegistry(overrides)

    if preset == "quick":
        space = registry.get_quick_space(resolution)
    elif preset == "full":
        space = registry.get_full_space(resolution)
    else:
        raise ValueError(f"Unknown preset '{preset}'. Available: quick, full")

    if include_models:
        if model_pool is not None:
            space.update(_build_model_space_from_pool(model_pool))
        elif resolution >= 224:
            space.update(registry.get_model_space_224())
        else:
            space.update(registry.get_model_space_28())

    return space


def _build_model_space_from_pool(pool: list[str]) -> dict[str, Callable]:
    """
    Build model search space from a user-specified pool of architectures.

    Args:
        pool: list of model names to include in the search.

    Returns:
        dict with model_name sampler (and weight_variant if vit_tiny is in pool).
    """
    space: dict[str, Callable] = {
        "model_name": lambda trial, _p=pool: trial.suggest_categorical("model_name", _p),
    }

    if "vit_tiny" in pool:
        space["weight_variant"] = lambda trial: (
            trial.suggest_categorical(
                "weight_variant",
                [
                    None,
                    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                    "vit_tiny_patch16_224.augreg_in21k",
                ],
            )
            if trial.params.get("model_name") == "vit_tiny"
            else None
        )

    return space
