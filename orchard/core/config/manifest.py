"""
Training Pipeline Configuration Manifest.

Defines the hierarchical ``Config`` schema that aggregates specialized
sub-configs (Hardware, Dataset, Architecture, Training, Evaluation,
Augmentation, Optuna, Export) into a single immutable manifest.

Layout:

- ``Config`` — main Pydantic model, ordered as:
  Fields & model validator, Properties (``run_slug``, ``num_workers``),
  Serialization (``dump_portable``, ``dump_serialized``),
  ``from_recipe`` — primary factory (``orchard`` CLI)
- ``_CrossDomainValidator`` — cross-domain validation logic
  (AMP vs Device, LR bounds, Mixup scheduling, resolution/model pairing)
- ``_deep_set`` — dot-notation dict helper for CLI overrides
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...exceptions import OrchardConfigError
from ..io import load_config_from_yaml
from ..metadata.wrapper import get_registry
from ..paths import (
    HIGHRES_THRESHOLD,
    METRIC_LOSS,
    METRIC_MAP,
    METRIC_MAP_50,
    METRIC_MAP_75,
    PROJECT_ROOT,
)
from .architecture_config import ArchitectureConfig
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .export_config import ExportConfig
from .hardware_config import HardwareConfig
from .optuna_config import OptunaConfig
from .telemetry_config import TelemetryConfig
from .tracking_config import TrackingConfig
from .training_config import TrainingConfig
from .types import TaskType

# Architecture resolution constraints (add new architectures here).
# These are semantic subsets of SUPPORTED_RESOLUTIONS (core.paths.constants);
# keep in sync when adding new resolutions.
_MODELS_LOW_RES = frozenset({"mini_cnn"})
_MODELS_224_ONLY = frozenset({"efficientnet_b0", "vit_tiny", "convnext_tiny"})
_MODELS_DETECTION = frozenset({"fasterrcnn"})

# Resolution constraints (subsets of core.paths.constants.SUPPORTED_RESOLUTIONS)
_RESOLUTIONS_LOW_RES: Final[frozenset[int]] = frozenset({28, 32, 64})
_RESOLUTIONS_224_ONLY: Final[frozenset[int]] = frozenset({224})

# Optuna override conflict detection: dotted config keys tuned per preset.
# Derived from optimization/_param_mapping.py PARAM_MAPPING + SearchSpaceRegistry.
_OPTUNA_QUICK_KEYS: Final[frozenset[str]] = frozenset(
    {
        "training.optimizer_type",
        "training.learning_rate",
        "training.weight_decay",
        "training.momentum",
        "training.min_lr",
        "training.batch_size",
        "architecture.dropout",
    }
)

_OPTUNA_FULL_EXTRA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "training.criterion_type",
        "training.focal_gamma",
        "training.label_smoothing",
        "training.mixup_alpha",
        "training.scheduler_type",
        "training.scheduler_patience",
        "augmentation.rotation_angle",
        "augmentation.jitter_val",
        "augmentation.min_scale",
    }
)


# OVERRIDE UTILITIES
def _deep_set(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    Set a nested dict value using a dot-separated key path.

    Creates intermediate dicts as needed. Used by ``Config.from_recipe``
    to apply CLI overrides before Pydantic instantiation.

    Args:
        data: Target dictionary to modify in-place
        dotted_key: Dot-separated path (e.g. ``"training.epochs"``)
        value: Value to set at the leaf key
    """
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _warn_optuna_override_conflicts(
    overrides: dict[str, Any],
    search_space_preset: str,
) -> None:
    """
    Warn when ``--set`` overrides target parameters that Optuna will tune.

    These overrides apply to the base config but are silently overwritten
    per trial by Optuna's search space, so the user's intent is lost.

    Args:
        overrides: Flat dict of dotted override keys from ``--set``.
        search_space_preset: Active Optuna preset (``"quick"`` or ``"full"``).
    """
    tunable = _OPTUNA_QUICK_KEYS
    if search_space_preset == "full":
        tunable = tunable | _OPTUNA_FULL_EXTRA_KEYS

    conflicts = sorted(tunable & overrides.keys())
    if not conflicts:
        return

    import warnings

    _stacklevel = 3  # pragma: no mutate
    warnings.warn(  # pragma: no mutate
        f"--set overrides {conflicts} will be ignored: "
        f"Optuna '{search_space_preset}' search space tunes these parameters per trial. "
        f"To fix a parameter, narrow the search space in optuna.search_space_overrides "
        f"or use a custom preset that excludes it.",
        stacklevel=_stacklevel,
    )


# MAIN CONFIGURATION
class Config(BaseModel):
    """
    Main experiment manifest aggregating specialized sub-configurations.

    Serves as the Single Source of Truth (SSOT) for all experiment parameters.
    Validates cross-domain logic (AMP/device compatibility, resolution/model pairing)
    and provides factory methods for YAML and CLI instantiation.

    Attributes:
        task_type: ML task type (currently ``"classification"``)
        hardware: Device selection, threading, reproducibility settings
        telemetry: Logging, paths, experiment naming
        training: Optimizer, scheduler, epochs, regularization
        augmentation: Data augmentation and TTA parameters
        dataset: Dataset selection, resolution, normalization
        evaluation: Metrics, visualization, reporting settings
        architecture: Architecture selection, pretrained weights
        optuna: Hyperparameter optimization configuration (optional)
        export: Model export configuration for ONNX (optional)
        tracking: Experiment tracking configuration for MLflow (optional)

    Example:
        >>> from orchard.core import Config
        >>> cfg = Config.from_recipe(Path("recipes/config_mini_cnn.yaml"))
        >>> cfg.architecture.name
        'mini_cnn'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_type: TaskType = Field(
        default="classification",
        description="ML task type driving strategy dispatch for losses, metrics, and evaluation.",
    )
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    optuna: OptunaConfig | None = Field(default=None)
    export: ExportConfig | None = Field(default=None)
    tracking: TrackingConfig | None = Field(default=None)

    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        """
        Cross-domain validation enforcing consistency across sub-configs.

        Invokes _CrossDomainValidator to check:

        - Model/resolution compatibility (ResNet-18 → 28x28)
        - Training epochs bounds (mixup_epochs ≤ epochs)
        - Hardware/feature alignment (AMP requires GPU)
        - Pretrained/channel consistency (pretrained → RGB)
        - Optimizer bounds (min_lr < learning_rate)

        Returns:
            Validated Config instance with auto-corrections applied

        Raises:
            OrchardConfigError: On irrecoverable validation failures
        """
        return _CrossDomainValidator.validate(self)

    # -- Properties ----------------------------------------------------------

    @property
    def run_slug(self) -> str:
        """
        Generate unique experiment folder identifier.

        Combines dataset name and model name for human-readable
        run identification in output directories. Slashes in
        architecture names (e.g. ``timm/convnext_base``) are
        replaced with underscores to keep paths flat.

        Returns:
            String in format '{dataset_name}_{model_name}'.
        """
        safe_name = self.architecture.name.replace("/", "_")
        return f"{self.dataset.dataset_name}_{safe_name}"

    @property
    def num_workers(self) -> int:
        """
        Get effective DataLoader workers from hardware policy.

        Delegates to hardware config which respects reproducibility
        constraints (returns 0 if reproducible mode enabled).

        Returns:
            Number of DataLoader worker processes.
        """
        return self.hardware.effective_num_workers

    # -- Serialization -------------------------------------------------------

    def dump_portable(self) -> dict[str, Any]:
        """
        Serialize config with environment-agnostic paths.

        Converts absolute filesystem paths to project-relative paths
        (e.g., '/home/user/project/dataset' -> './dataset') to prevent
        host-specific path leakage in exported configurations.

        Returns:
            Dictionary with all paths converted to portable relative strings.
        """
        full_data = self.model_dump()
        full_data["hardware"] = self.hardware.model_dump()
        full_data["telemetry"] = self.telemetry.to_portable_dict()

        # Sanitize dataset root path
        # default {} never reached (model_dump always has "dataset"), equivalent mutant
        dataset_section = full_data.get("dataset", {})  # pragma: no mutate
        data_root = dataset_section.get("data_root")

        if data_root:
            dr_path = Path(data_root)
            if dr_path.is_relative_to(PROJECT_ROOT):
                relative_dr = dr_path.relative_to(PROJECT_ROOT)
                full_data["dataset"]["data_root"] = f"./{relative_dr}"

        return full_data

    def dump_serialized(self) -> dict[str, Any]:
        """
        Convert config to JSON-compatible dict for YAML serialization.

        Uses Pydantic's json mode to ensure all values are serializable
        (Path objects become strings, enums become values, etc.).

        Returns:
            Dictionary with all values JSON-serializable for YAML export.
        """
        # Pydantic mode= is case-insensitive, so "json"/"JSON" are equivalent mutants
        return self.model_dump(mode="json")  # pragma: no mutate

    # -- Factory: YAML recipe (primary, used by ``orchard`` CLI) -------------

    @classmethod
    def from_recipe(
        cls,
        recipe_path: Path,
        overrides: dict[str, Any] | None = None,
    ) -> "Config":
        """
        Factory from YAML recipe with optional dot-notation overrides.

        Loads the recipe, applies overrides to the raw dict *before*
        Pydantic instantiation, resolves dataset metadata, and returns
        a validated Config. This is the preferred entry point for the
        ``orchard`` CLI.

        Args:
            recipe_path: Path to YAML recipe file
            overrides: Flat dict of dot-notation keys to values
                       (e.g. ``{"training.epochs": 20}``)

        Returns:
            Validated Config instance

        Raises:
            FileNotFoundError: If recipe_path does not exist
            ValueError: If recipe is missing ``dataset.name``
            KeyError: If dataset not found in registry

        Example:
            >>> cfg = Config.from_recipe(Path("recipes/config_mini_cnn.yaml"))
            >>> cfg = Config.from_recipe(
            ...     Path("recipes/config_mini_cnn.yaml"),
            ...     overrides={"training.epochs": 20, "training.seed": 123},
            ... )
        """
        raw_data = load_config_from_yaml(recipe_path)

        if overrides:
            for dotted_key, value in overrides.items():
                _deep_set(raw_data, dotted_key, value)

        dataset_section = raw_data.get("dataset", {})
        ds_name = dataset_section.get("name")
        if not ds_name:
            raise OrchardConfigError(f"Recipe '{recipe_path}' must specify 'dataset.name'")

        resolution = dataset_section.get("resolution", 28)
        task_type = raw_data.get("task_type", "classification")
        wrapper = get_registry(resolution, task_type)

        if ds_name not in wrapper.registry:
            available = list(wrapper.registry.keys())
            raise OrchardConfigError(
                f"Dataset '{ds_name}' not found at resolution {resolution}. "
                f"Available at {resolution}px: {available}"
            )

        metadata = wrapper.get_dataset(ds_name)
        raw_data.setdefault("dataset", {})["metadata"] = metadata

        if overrides and raw_data.get("optuna") is not None:
            optuna_section = raw_data["optuna"]
            preset = optuna_section.get("search_space_preset", "full")
            _warn_optuna_override_conflicts(overrides, preset)

        return cls(**raw_data)


# CROSS-DOMAIN VALIDATION
class _CrossDomainValidator:
    """
    Internal cross-domain validator (no public API).
    """

    @classmethod
    def validate(cls, config: Config) -> Config:
        """
        Run all cross-domain validation checks.
        """
        cls._check_architecture_resolution(config)
        cls._check_mixup_epochs(config)
        cls._check_amp_device(config)
        cls._check_pretrained_channels(config)
        cls._check_lr_bounds(config)
        cls._check_cpu_highres_performance(config)
        cls._check_min_dataset_size(config)
        cls._check_quantization_architecture(config)
        cls._check_detection_config(config)
        return config

    @classmethod
    def _check_architecture_resolution(cls, config: Config) -> None:
        """
        Validate architecture-resolution compatibility.

        Enforces that each built-in model is used with its supported resolution(s):

        - Low-resolution (28, 32, 64): mini_cnn
        - 224x224 only: efficientnet_b0, vit_tiny, convnext_tiny
        - Multi-resolution (all supported): resnet_18

        timm models (prefixed with ``timm/``) bypass this check as they
        support variable resolutions managed by the user.

        Raises:
            OrchardConfigError: If architecture and resolution are incompatible.
        """
        model_name = config.architecture.name.lower()

        # timm models handle their own resolution requirements
        if model_name.startswith("timm/"):
            return

        resolution = config.dataset.resolution

        if model_name in _MODELS_LOW_RES and resolution not in _RESOLUTIONS_LOW_RES:
            raise OrchardConfigError(
                f"'{config.architecture.name}' requires resolution "
                f"{sorted(_RESOLUTIONS_LOW_RES)}, got {resolution}. "
                f"Use a 224x224 architecture "
                f"(efficientnet_b0, vit_tiny, convnext_tiny) or resnet_18."
            )

        if model_name in _MODELS_224_ONLY and resolution not in _RESOLUTIONS_224_ONLY:
            raise OrchardConfigError(
                f"'{config.architecture.name}' requires resolution=224, got {resolution}. "
                f"Use resnet_18 or mini_cnn for low resolution."
            )

    @classmethod
    def _check_mixup_epochs(cls, config: Config) -> None:
        """
        Validate mixup scheduling within training bounds.

        Raises:
            OrchardConfigError: If mixup_epochs exceeds total epochs.
        """
        if config.training.mixup_epochs > config.training.epochs:
            raise OrchardConfigError(
                f"mixup_epochs ({config.training.mixup_epochs}) exceeds "
                f"total epochs ({config.training.epochs})"
            )

    @classmethod
    def _check_amp_device(cls, config: Config) -> None:
        """
        Validate AMP-device alignment.

        Auto-disables AMP on CPU with a warning instead of failing,
        since this is a recoverable misconfiguration (e.g. GPU recipe
        running on a CPU-only machine).

        Note:
            Uses ``object.__setattr__`` to bypass Pydantic frozen
            restriction.  This is intentional: AMP auto-disable is a
            UX convenience that must happen after device resolution,
            which occurs during model validation (post-freeze).
        """
        if not config.hardware.supports_amp and config.training.use_amp:
            import warnings

            warnings.warn(
                "AMP requires GPU (CUDA/MPS) but CPU detected. Disabling AMP automatically.",
                UserWarning,
                stacklevel=4,
            )
            object.__setattr__(config.training, "use_amp", False)

    @classmethod
    def _check_pretrained_channels(cls, config: Config) -> None:
        """
        Validate pretrained model channel requirements.

        Pretrained models require RGB (3 channels). Grayscale datasets
        must use force_rgb=True or disable pretraining.

        Raises:
            OrchardConfigError: If pretrained model used with non-RGB input.
        """
        if config.architecture.pretrained and config.dataset.effective_in_channels != 3:
            raise OrchardConfigError(
                f"Pretrained {config.architecture.name} requires RGB (3 channels), "
                f"but dataset will provide {config.dataset.effective_in_channels} channels. "
                f"set 'force_rgb: true' in dataset config or disable pretraining"
            )

    @classmethod
    def _check_lr_bounds(cls, config: Config) -> None:
        """
        Validate learning rate bounds consistency.

        Raises:
            OrchardConfigError: If min_lr >= learning_rate.
        """
        if config.training.min_lr >= config.training.learning_rate:
            raise OrchardConfigError(
                f"min_lr ({config.training.min_lr}) must be less than "
                f"learning_rate ({config.training.learning_rate})"
            )

    @classmethod
    def _check_cpu_highres_performance(cls, config: Config) -> None:
        """
        Warn when training at high resolution on CPU.

        Emits a UserWarning when the resolved device is CPU and the
        dataset resolution is 224px or above, as this combination
        results in significantly slower training.
        """
        if (
            config.hardware.device.lower().startswith("cpu")
            and config.dataset.resolution >= HIGHRES_THRESHOLD
        ):
            import warnings

            warnings.warn(
                f"Training at resolution {config.dataset.resolution}px on CPU "
                f"will be significantly slower than on a GPU accelerator.",
                UserWarning,
                stacklevel=4,
            )

    @classmethod
    def _check_quantization_architecture(cls, config: Config) -> None:
        """
        Warn when aggressive quantization targets a very small model.

        4-bit quantization (int4/uint4) on mini_cnn causes disproportionate
        precision loss because the model has very few parameters and small
        convolution kernels.
        """
        if config.export is None or not config.export.quantize:
            return
        if (
            config.export.quantization_type in ("int4", "uint4")
            and config.architecture.name.lower() == "mini_cnn"
        ):
            import warnings

            warnings.warn(
                f"4-bit quantization ({config.export.quantization_type}) on "
                f"mini_cnn is likely to degrade accuracy severely. "
                f"Consider int8/uint8 or a larger architecture.",
                UserWarning,
                stacklevel=4,
            )

    @classmethod
    def _check_min_dataset_size(cls, config: Config) -> None:
        """
        Warn when max_samples is too small for reliable class-balanced training.

        Emits a UserWarning when max_samples is set but less than 10 per class,
        which may cause unreliable class balancing and noisy metrics.
        Only applies to classification tasks.
        """
        if config.task_type != "classification":
            return
        if config.dataset.max_samples is None:
            return
        num_classes = config.dataset.num_classes
        if config.dataset.max_samples < num_classes:
            raise OrchardConfigError(
                f"max_samples ({config.dataset.max_samples}) must be >= num_classes "
                f"({num_classes}). Class balancing requires at least one sample per class."
            )
        if config.dataset.max_samples < 10 * num_classes:
            import warnings

            warnings.warn(
                f"max_samples ({config.dataset.max_samples}) is less than "
                f"10x num_classes ({num_classes}). Class balancing may be unreliable.",
                UserWarning,
                stacklevel=4,
            )

    @classmethod
    def _check_detection_config(cls, config: Config) -> None:
        """
        Validate detection-specific constraints.

        Enforces that detection tasks use compatible architectures,
        resolutions, and training settings.
        """
        if config.task_type != "detection":
            return

        model_name = config.architecture.name.lower()

        # Detection requires a detection-capable architecture
        if model_name not in _MODELS_DETECTION:
            raise OrchardConfigError(
                f"Architecture '{config.architecture.name}' is not compatible with "
                f"task_type='detection'. Use a detection model (e.g. 'fasterrcnn')."
            )

        # Detection models need reasonable resolution (FPN expects >= 224)
        if config.dataset.resolution < HIGHRES_THRESHOLD:
            raise OrchardConfigError(
                f"Detection requires resolution >= {HIGHRES_THRESHOLD}, "
                f"got {config.dataset.resolution}."
            )

        # monitor_metric must be a detection metric
        _valid_detection_metrics = frozenset(
            {METRIC_LOSS, METRIC_MAP, METRIC_MAP_50, METRIC_MAP_75}
        )
        if config.training.monitor_metric not in _valid_detection_metrics:
            raise OrchardConfigError(
                f"monitor_metric '{config.training.monitor_metric}' is not valid "
                f"for detection tasks. Use one of: {sorted(_valid_detection_metrics)}."
            )

        # MixUp is not meaningful for bounding-box tasks
        if config.training.mixup_alpha > 0:
            raise OrchardConfigError(
                "MixUp (mixup_alpha > 0) is not compatible with detection tasks. "
                "Set mixup_alpha: 0.0 in training config."
            )

        # Spatial augmentations modify images but NOT bounding boxes,
        # producing misaligned targets.  Auto-disable with warning
        # (same pattern as AMP-on-CPU auto-disable).
        overrides: list[str] = []
        aug = config.augmentation
        if aug.hflip > 0:
            object.__setattr__(aug, "hflip", 0.0)
            overrides.append("hflip → 0.0")
        if aug.rotation_angle > 0:
            object.__setattr__(aug, "rotation_angle", 0)
            overrides.append("rotation_angle → 0")
        if aug.min_scale < 1.0:
            object.__setattr__(aug, "min_scale", 1.0)
            overrides.append("min_scale → 1.0")
        if overrides:
            import warnings

            warnings.warn(
                "Spatial augmentations are not compatible with detection "
                "(they transform the image but not the bounding boxes). "
                f"Auto-disabled: {', '.join(overrides)}.",
                UserWarning,
                stacklevel=4,
            )

        # Warn about classification-only training params that are silently ignored
        if config.training.label_smoothing > 0:
            import warnings

            warnings.warn(
                "label_smoothing is ignored for detection tasks "
                "(detection models compute losses internally).",
                UserWarning,
                stacklevel=4,
            )

        if config.training.use_tta:
            import warnings

            warnings.warn(
                "use_tta is ignored for detection tasks "
                "(test-time augmentation is not supported for detection).",
                UserWarning,
                stacklevel=4,
            )

        if config.training.criterion_type != "cross_entropy":
            import warnings

            warnings.warn(
                f"criterion_type '{config.training.criterion_type}' is ignored for "
                "detection tasks (detection models use built-in losses).",
                UserWarning,
                stacklevel=4,
            )

        _focal_gamma_default = TrainingConfig.model_fields["focal_gamma"].default
        if config.training.focal_gamma != _focal_gamma_default:
            import warnings

            warnings.warn(
                "focal_gamma is ignored for detection tasks "
                "(detection models use built-in losses).",
                UserWarning,
                stacklevel=4,
            )

        if config.training.weighted_loss:
            import warnings

            warnings.warn(
                "weighted_loss is ignored for detection tasks "
                "(detection models use built-in losses).",
                UserWarning,
                stacklevel=4,
            )
