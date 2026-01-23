"""
Vision Pipeline Configuration & Orchestration Engine.

Declarative core defining the hierarchical schema and validation logic
for the pipeline. Transforms raw inputs (CLI, YAML) into a structured,
type-safe manifest synchronizing hardware state with experiment logic.

Key Features:
    * Hierarchical aggregation: Unifies specialized sub-configs (Hardware,
      Dataset, Model, Training, Evaluation, Augmentation, Optuna) into a single,
      immutable object.
    * Cross-domain validation: Complex logic checks (AMP vs Device, LR bounds,
      Mixup scheduling) spanning multiple sub-modules
    * Metadata-driven injection: Centralizes dataset specification resolution,
      ensuring architectural synchronization
    * Factory polymorphism: Dual entry points via YAML files or CLI arguments

Strict validation during initialization guarantees logically sound and
reproducible execution context for the RootOrchestrator.
"""

# Standard Imports
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

# Third-Party Imports
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Relative Imports
from ..io import load_config_from_yaml
from ..metadata.wrapper import DatasetRegistryWrapper
from ..paths import PROJECT_ROOT
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .hardware_config import HardwareConfig
from .models_config import ModelConfig
from .optuna_config import OptunaConfig
from .telemetry_config import TelemetryConfig
from .training_config import TrainingConfig


# MAIN CONFIGURATION
class Config(BaseModel):
    """
    Main experiment manifest aggregating specialized sub-configurations.

    Provides the validated blueprint for RootOrchestrator to execute experiments.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True, frozen=True)

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optuna: Optional[OptunaConfig] = Field(default=None)

    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        # Use object.__setattr__ to bypass frozen restriction
        if self.dataset.metadata is None:
            object.__setattr__(self, "metadata", {})

        # 1. Model-specific resolution constraints
        if "resnet_18_adapted" in self.model.name.lower() and self.dataset.resolution != 28:
            raise ValueError(
                f"resnet_18_adapted requires resolution=28, got {self.dataset.resolution}"
            )

        # 2. Training logic
        if self.training.mixup_epochs > self.training.epochs:
            raise ValueError(
                f"mixup_epochs ({self.training.mixup_epochs}) exceeds "
                f"total epochs ({self.training.epochs})"
            )

        # 3. Hardware-feature alignment (auto-disable AMP on CPU)
        if self.hardware.device == "cpu" and self.training.use_amp:
            import warnings

            warnings.warn(
                "AMP requires GPU (CUDA/MPS) but CPU detected. Disabling AMP automatically.",
                UserWarning,
                stacklevel=2,
            )
            object.__setattr__(self.training, "use_amp", False)

        # 4. Model-dataset consistency
        if self.model.pretrained and self.dataset.effective_in_channels != 3:
            raise ValueError(
                f"Pretrained {self.model.name} requires RGB (3 channels), "
                f"but dataset will provide {self.dataset.effective_in_channels} channels. "
                f"Set 'force_rgb: true' in dataset config or disable pretraining"
            )

        # 5. Optimizer bounds
        if self.training.min_lr >= self.training.learning_rate:
            msg = (
                f"min_lr ({self.training.min_lr}) must be less than "
                f"learning_rate ({self.training.learning_rate})"
            )
            raise ValueError(msg)

        return self

    @property
    def run_slug(self) -> str:
        """Unique experiment folder identifier."""
        return f"{self.dataset.dataset_name}_{self.model.name}"

    @property
    def num_workers(self) -> int:
        """Effective DataLoader workers from hardware policy."""
        return self.hardware.effective_num_workers

    def dump_portable(self) -> Dict[str, Any]:
        """
        Serializes config with environment-agnostic paths.

        Converts absolute paths to relative anchors from PROJECT_ROOT.
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
        """Converts config to JSON-compatible dict for YAML serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def _resolve_dataset_metadata(cls, args: argparse.Namespace) -> dict:
        """
        Fetches dataset specs from registry.

        Args:
            args: Parsed CLI arguments

        Returns:
            Dataset metadata dictionary

        Raises:
            ValueError: If dataset not found or name missing
        """
        ds_name_raw = getattr(args, "dataset", None)
        if not ds_name_raw:
            raise ValueError("Dataset name required via --dataset or config file")

        resolution = getattr(args, "resolution", 28)
        ds_name = ds_name_raw.lower()

        wrapper = DatasetRegistryWrapper(resolution=resolution)
        if ds_name not in wrapper.registry:
            raise ValueError(
                f"Dataset '{ds_name_raw}' not found in registry for resolution {resolution}"
            )

        return wrapper.get_dataset(ds_name)

    @classmethod
    def _hydrate_yaml(cls, yaml_path: Path, metadata: dict) -> "Config":
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

    @classmethod
    def from_yaml(cls, yaml_path: Path, metadata: dict) -> "Config":
        """
        Factory from YAML file with metadata injection.

        Args:
            yaml_path: Path to config YAML
            metadata: Dataset metadata

        Returns:
            Hydrated Config instance
        """
        return cls._hydrate_yaml(yaml_path, metadata)

    @classmethod
    def _build_from_yaml_or_args(cls, args: argparse.Namespace, ds_meta: dict) -> "Config":
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
            yaml_path = Path(args.config)
            raw_data = load_config_from_yaml(yaml_path)

            # STEP 1: Extract critical values from YAML
            dataset_section = raw_data.get("dataset", {})
            yaml_dataset_name = dataset_section.get("name")
            yaml_resolution = dataset_section.get("resolution")
            yaml_img_size = dataset_section.get("img_size")

            # STEP 2: Inject into args so sub-configs see them
            if yaml_resolution is not None:
                args.resolution = yaml_resolution
            if yaml_img_size is not None:
                args.img_size = yaml_img_size
            if yaml_dataset_name:
                args.dataset = yaml_dataset_name

            # STEP 3: Re-resolve metadata with correct resolution
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

            # STEP 4: Use YAML hydration path with correct metadata
            return cls.from_yaml(yaml_path, metadata=ds_meta)

        # Build from CLI args (standard path)
        return cls(
            hardware=HardwareConfig.from_args(args),
            telemetry=TelemetryConfig.from_args(args),
            training=TrainingConfig.from_args(args),
            augmentation=AugmentationConfig.from_args(args),
            dataset=DatasetConfig.from_args(args),
            model=ModelConfig.from_args(args),
            evaluation=EvaluationConfig.from_args(args),
            optuna=OptunaConfig.from_args(args) if hasattr(args, "study_name") else None,
        )

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
