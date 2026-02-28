"""
Trial configuration building utilities.

This module provides a builder responsible for creating trial-specific
Config instances for Optuna experiments, handling parameter mapping,
metadata preservation, and validation.
"""

from __future__ import annotations

from typing import Any

from ...core import Config


# CONFIG BUILDER
class TrialConfigBuilder:
    """
    Builds trial-specific Config instances for Optuna trials.

    Handles parameter mapping from Optuna's flat namespace to Config's
    hierarchical structure, preserves dataset metadata excluded from
    serialization, and validates via Pydantic.

    Attributes:
        base_cfg: Base configuration template
        optuna_epochs: Number of epochs for Optuna trials (from cfg.optuna.epochs)
        base_metadata: Cached dataset metadata

    Example:
        >>> builder = TrialConfigBuilder(base_cfg)
        >>> trial_params = {"learning_rate": 0.001, "dropout": 0.3}
        >>> trial_cfg = builder.build(trial_params)
    """

    # Mapping of parameter names to config sections
    PARAM_MAPPING = {
        "training": [
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
        ],
        "architecture": ["dropout", "weight_variant"],
        "augmentation": ["rotation_angle", "jitter_val", "min_scale"],
    }

    def __init__(self, base_cfg: Config) -> None:
        """
        Initialize config builder.

        Args:
            base_cfg: Base configuration template
        """
        self.base_cfg = base_cfg
        self.optuna_epochs = base_cfg.optuna.epochs
        self.base_metadata = base_cfg.dataset._ensure_metadata

    def build(self, trial_params: dict[str, Any]) -> Config:
        """
        Build trial-specific Config with parameter overrides.

        Args:
            trial_params: Sampled hyperparameters from Optuna

        Returns:
            Validated Config instance with trial parameters
        """
        config_dict = self.base_cfg.model_dump()

        # Preserve resolution
        if config_dict["dataset"].get("resolution") is None:
            config_dict["dataset"]["resolution"] = self.base_cfg.dataset.resolution

        # Re-inject metadata (excluded from serialization)
        config_dict["dataset"]["metadata"] = self.base_metadata

        # Override epochs for Optuna trials
        config_dict["training"]["epochs"] = self.optuna_epochs

        # Cap mixup_epochs to trial length (prevents _check_mixup_epochs ValueError)
        config_dict["training"]["mixup_epochs"] = min(
            config_dict["training"]["mixup_epochs"], self.optuna_epochs
        )

        # Apply trial-specific overrides
        self._apply_param_overrides(config_dict, trial_params)

        return Config(**config_dict)

    def _apply_param_overrides(
        self, config_dict: dict[str, Any], trial_params: dict[str, Any]
    ) -> None:
        """
        Apply parameter overrides to config dict (in-place).

        Maps flat parameter names to hierarchical config structure
        using PARAM_MAPPING. Handles special cases for model_name → architecture.name
        and skips None values for weight_variant (used for non-ViT models).

        Args:
            config_dict: Config dictionary to modify
            trial_params: Parameters to apply
        """
        for param_name, value in trial_params.items():
            # Skip None values for weight_variant (returned for non-ViT models)
            if param_name == "weight_variant" and value is None:
                continue

            # Special case: model_name maps to architecture.name
            if param_name == "model_name":
                config_dict["architecture"]["name"] = value
                continue

            # Standard mapping
            for section, params in self.PARAM_MAPPING.items():
                if param_name in params:
                    config_dict[section][param_name] = value
                    break
