"""
Vision Pipeline Configuration Manifest.

Defines the hierarchical ``Config`` schema that aggregates specialized
sub-configs (Hardware, Dataset, Architecture, Training, Evaluation,
Augmentation, Optuna, Export) into a single immutable manifest.

Layout:
    * ``Config`` — main Pydantic model, ordered as:
        1. Fields & model validator
        2. Properties (``run_slug``, ``num_workers``)
        3. Serialization (``dump_portable``, ``dump_serialized``)
        4. ``from_recipe`` — primary factory (``orchard`` CLI)
        5. ``from_args`` / ``from_yaml`` — legacy factories (``forge.py``)
        6. Private argparse helpers
    * ``_CrossDomainValidator`` — cross-domain validation logic
      (AMP vs Device, LR bounds, Mixup scheduling, resolution/model pairing)
    * ``_deep_set`` — dot-notation dict helper for CLI overrides
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..io import load_config_from_yaml
from ..metadata import DatasetMetadata
from ..metadata.wrapper import DatasetRegistryWrapper
from ..paths import PROJECT_ROOT
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


# MAIN CONFIGURATION
class Config(BaseModel):
    """
    Main experiment manifest aggregating specialized sub-configurations.

    Serves as the Single Source of Truth (SSOT) for all experiment parameters.
    Validates cross-domain logic (AMP/device compatibility, resolution/model pairing)
    and provides factory methods for YAML and CLI instantiation.

    Attributes:
        hardware: Device selection, threading, reproducibility settings
        telemetry: Logging, paths, experiment naming
        training: Optimizer, scheduler, epochs, regularization
        augmentation: Data augmentation and TTA parameters
        dataset: Dataset selection, resolution, normalization
        evaluation: Metrics, visualization, reporting settings
        architecture: Architecture selection, pretrained weights
        optuna: Hyperparameter optimization configuration (optional)
        export: Model export configuration for ONNX/TorchScript (optional)

    Example:
        >>> from orchard.core import Config
        >>> cfg = Config.from_recipe(Path("recipes/config_mini_cnn.yaml"))
        >>> cfg.architecture.name
        'mini_cnn'
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True, frozen=True)

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    optuna: Optional[OptunaConfig] = Field(default=None)
    export: Optional[ExportConfig] = Field(default=None)
    tracking: Optional[TrackingConfig] = Field(default=None)

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
            ValueError: On irrecoverable validation failures
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

    def dump_portable(self) -> Dict[str, Any]:
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
        dataset_section = full_data.get("dataset", {})
        data_root = dataset_section.get("data_root")

        if data_root:
            dr_path = Path(data_root)
            if dr_path.is_relative_to(PROJECT_ROOT):
                relative_dr = dr_path.relative_to(PROJECT_ROOT)
                full_data["dataset"]["data_root"] = f"./{relative_dr}"

        return full_data

    def dump_serialized(self) -> Dict[str, Any]:
        """
        Convert config to JSON-compatible dict for YAML serialization.

        Uses Pydantic's json mode to ensure all values are serializable
        (Path objects become strings, enums become values, etc.).

        Returns:
            Dictionary with all values JSON-serializable for YAML export.
        """
        return self.model_dump(mode="json")

    # -- Factory: YAML recipe (primary, used by ``orchard`` CLI) -------------

    @classmethod
    def from_recipe(
        cls,
        recipe_path: Path,
        overrides: Optional[Dict[str, Any]] = None,
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
            raise ValueError(f"Recipe '{recipe_path}' must specify 'dataset.name'")

        resolution = dataset_section.get("resolution", 28)
        wrapper = DatasetRegistryWrapper(resolution=resolution)

        if ds_name not in wrapper.registry:
            available = list(wrapper.registry.keys())
            raise KeyError(
                f"Dataset '{ds_name}' not found at resolution {resolution}. "
                f"Available at {resolution}px: {available}"
            )

        metadata = wrapper.get_dataset(ds_name)
        raw_data.setdefault("dataset", {})["metadata"] = metadata

        cfg = cls(**raw_data)
        return cls.model_validate(cfg)

    # -- Factory: argparse (legacy, used by ``forge.py``) --------------------

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Factory from CLI arguments.

        Args:
            args: Parsed argparse namespace

        Returns:
            Configured instance with resolved metadata
        """
        ds_meta = cls._resolve_dataset_metadata(args)
        return cls._build_from_yaml_or_args(args, ds_meta)

    @classmethod
    def from_yaml(cls, yaml_path: Path, metadata: DatasetMetadata) -> "Config":
        """
        Factory from YAML file with metadata injection.

        Args:
            yaml_path: Path to config YAML
            metadata: Dataset metadata

        Returns:
            Hydrated Config instance
        """
        return cls._hydrate_yaml(yaml_path, metadata)

    # -- Private helpers (argparse path) -------------------------------------

    @classmethod
    def _resolve_dataset_metadata(cls, args: argparse.Namespace) -> DatasetMetadata:
        """
        Fetches dataset specs from registry.

        Args:
            args: Parsed CLI arguments

        Returns:
            Dataset metadata object

        Raises:
            ValueError: If dataset not found or name missing
        """
        ds_name_raw = getattr(args, "dataset", None)
        if not ds_name_raw:
            raise ValueError("Dataset name required via --dataset or config file")

        resolution = getattr(args, "resolution", 28)
        ds_name = ds_name_raw.lower()

        try:
            return DatasetConfig._resolve_metadata(ds_name, resolution)
        except KeyError:
            raise ValueError(
                f"Dataset '{ds_name_raw}' not found in registry for resolution {resolution}"
            )

    @classmethod
    def _build_from_yaml_or_args(
        cls, args: argparse.Namespace, ds_meta: DatasetMetadata
    ) -> "Config":
        """
        Constructs Config from YAML or CLI arguments.

        **PRECEDENCE ORDER:**
        1. If --config provided → YAML values take precedence (CLI ignored)
        2. If no --config → CLI arguments used
        3. Fallback → Pydantic field defaults

        This ensures YAML recipes are the authoritative source when specified,
        preventing configuration drift in reproducible research.

        Args:
            args: Parsed argparse namespace
            ds_meta: Initial dataset metadata (may be overridden by YAML)

        Returns:
            Configured instance with correct metadata
        """
        if getattr(args, "config", None):
            return cls._build_from_yaml(args, ds_meta)
        return cls._build_from_cli(args)

    @classmethod
    def _build_from_yaml(cls, args: argparse.Namespace, ds_meta: DatasetMetadata) -> "Config":
        """
        Build Config from YAML file with metadata resolution.

        Extracts dataset/resolution from YAML, re-resolves metadata if needed,
        and hydrates config through the YAML path.

        Args:
            args: Parsed argparse namespace (must have config attribute)
            ds_meta: Initial dataset metadata (may be overridden by YAML)

        Returns:
            Configured instance with correct metadata
        """
        yaml_path = Path(args.config)
        raw_data = load_config_from_yaml(yaml_path)

        # Extract critical values from YAML
        dataset_section = raw_data.get("dataset", {})
        yaml_dataset_name = dataset_section.get("name")
        yaml_resolution = dataset_section.get("resolution")
        yaml_img_size = dataset_section.get("img_size")

        # Inject into args so sub-configs see them
        if yaml_resolution is not None:
            args.resolution = yaml_resolution
        if yaml_img_size is not None:
            args.img_size = yaml_img_size
        if yaml_dataset_name:
            args.dataset = yaml_dataset_name

        # Re-resolve metadata with correct resolution if dataset specified
        if yaml_dataset_name:
            resolution = yaml_resolution if yaml_resolution is not None else 28
            wrapper = DatasetRegistryWrapper(resolution=resolution)

            if yaml_dataset_name not in wrapper.registry:
                available = list(wrapper.registry.keys())
                raise KeyError(
                    f"Dataset '{yaml_dataset_name}' not found at resolution {resolution}. "
                    f"Available at {resolution}px: {available}"
                )

            ds_meta = wrapper.get_dataset(yaml_dataset_name)

        return cls.from_yaml(yaml_path, metadata=ds_meta)

    @classmethod
    def _build_from_cli(cls, args: argparse.Namespace) -> "Config":
        """
        Build Config from CLI arguments only.

        Args:
            args: Parsed argparse namespace

        Returns:
            Configured instance from CLI values
        """
        return cls(
            hardware=HardwareConfig.from_args(args),
            telemetry=TelemetryConfig.from_args(args),
            training=TrainingConfig.from_args(args),
            augmentation=AugmentationConfig.from_args(args),
            dataset=DatasetConfig.from_args(args),
            architecture=ArchitectureConfig.from_args(args),
            evaluation=EvaluationConfig.from_args(args),
            optuna=OptunaConfig.from_args(args) if hasattr(args, "study_name") else None,
            export=ExportConfig.from_args(args) if hasattr(args, "format") else None,
        )

    @classmethod
    def _hydrate_yaml(cls, yaml_path: Path, metadata: DatasetMetadata) -> "Config":
        """
        Loads YAML config and injects dataset metadata.

        Metadata must be passed in from _build_from_yaml_or_args
        after proper resolution handling.

        Args:
            yaml_path: Path to config YAML
            metadata: Pre-resolved dataset metadata for correct resolution

        Returns:
            Validated Config instance
        """
        raw_data = load_config_from_yaml(yaml_path)

        # Inject metadata into dataset section BEFORE instantiation
        if "dataset" not in raw_data:
            raw_data["dataset"] = {}

        # Store metadata object (will be excluded from serialization)
        raw_data["dataset"]["metadata"] = metadata

        # Instantiate config with pre-injected metadata
        cfg = cls(**raw_data)

        return cls.model_validate(cfg)


# CROSS-DOMAIN VALIDATION
class _CrossDomainValidator:
    """Internal cross-domain validator (no public API)."""

    @classmethod
    def validate(cls, config: Config) -> Config:
        """Run all cross-domain validation checks."""
        cls._check_architecture_resolution(config)
        cls._check_mixup_epochs(config)
        cls._check_amp_device(config)
        cls._check_pretrained_channels(config)
        cls._check_lr_bounds(config)
        return config

    @classmethod
    def _check_architecture_resolution(cls, config: Config) -> None:
        """
        Validate architecture-resolution compatibility.

        Enforces that each built-in model is used with its supported resolution(s):
            - 28x28 only: mini_cnn
            - 224x224 only: efficientnet_b0, vit_tiny, convnext_tiny
            - Multi-resolution (28x28, 224x224): resnet_18

        timm models (prefixed with ``timm/``) bypass this check as they
        support variable resolutions managed by the user.

        Raises:
            ValueError: If architecture and resolution are incompatible.
        """
        model_name = config.architecture.name.lower()

        # timm models handle their own resolution requirements
        if model_name.startswith("timm/"):
            return

        resolution = config.dataset.resolution

        resolution_28_only = {"mini_cnn"}
        resolution_224_only = {"efficientnet_b0", "vit_tiny", "convnext_tiny"}
        multi_resolution = {"resnet_18"}

        if model_name in resolution_28_only and resolution != 28:
            raise ValueError(
                f"'{config.architecture.name}' requires resolution=28, got {resolution}. "
                f"Use a 224x224 architecture (efficientnet_b0, vit_tiny, convnext_tiny) "
                f"or resnet_18 for high resolution."
            )

        if model_name in resolution_224_only and resolution != 224:
            raise ValueError(
                f"'{config.architecture.name}' requires resolution=224, got {resolution}. "
                f"Use resnet_18 or mini_cnn for low resolution."
            )

        if model_name in multi_resolution and resolution not in (28, 224):
            raise ValueError(
                f"'{config.architecture.name}' supports resolutions 28 or 224, "
                f"got {resolution}."
            )

    @classmethod
    def _check_mixup_epochs(cls, config: Config) -> None:
        """
        Validate mixup scheduling within training bounds.

        Raises:
            ValueError: If mixup_epochs exceeds total epochs.
        """
        if config.training.mixup_epochs > config.training.epochs:
            raise ValueError(
                f"mixup_epochs ({config.training.mixup_epochs}) exceeds "
                f"total epochs ({config.training.epochs})"
            )

    @classmethod
    def _check_amp_device(cls, config: Config) -> None:
        """
        Validate AMP-device alignment.

        Auto-disables AMP on CPU with a warning instead of failing,
        since this is a recoverable misconfiguration.
        """
        if config.hardware.device == "cpu" and config.training.use_amp:
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
            ValueError: If pretrained model used with non-RGB input.
        """
        if config.architecture.pretrained and config.dataset.effective_in_channels != 3:
            raise ValueError(
                f"Pretrained {config.architecture.name} requires RGB (3 channels), "
                f"but dataset will provide {config.dataset.effective_in_channels} channels. "
                f"Set 'force_rgb: true' in dataset config or disable pretraining"
            )

    @classmethod
    def _check_lr_bounds(cls, config: Config) -> None:
        """
        Validate learning rate bounds consistency.

        Raises:
            ValueError: If min_lr >= learning_rate.
        """
        if config.training.min_lr >= config.training.learning_rate:
            raise ValueError(
                f"min_lr ({config.training.min_lr}) must be less than "
                f"learning_rate ({config.training.learning_rate})"
            )


# OVERRIDE UTILITIES
def _deep_set(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
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
