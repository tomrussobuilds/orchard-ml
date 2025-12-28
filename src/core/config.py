"""
Configuration Engine and Schema Definitions

This module serves as the Single Source of Truth (SSOT) for experiment 
parameters and reproducibility. It defines a strict, hierarchical data 
structure using Pydantic to ensure type safety and immutability.

Key Features:
    * Hierarchical Schema: Decouples system, training, augmentation, and 
      dataset-specific parameters into specialized sub-configurations.
    * Validation & Type Safety: Enforces runtime constraints (e.g., value ranges, 
      hardware availability) and prevents accidental modification via frozen models.
    * Environment Awareness: Orchestrates hardware-dependent settings like 
      device selection and optimal worker allocation.
    * CLI Integration: Provides a factory bridge to transform raw CLI namespaces 
      (parsed externally) into validated, immutable Config objects.
"""
# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import argparse
from pathlib import Path
import tempfile
from typing import Annotated, Optional

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from pydantic import (
    BaseModel, Field, ConfigDict, field_validator, model_validator
    )

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .system import detect_best_device, get_num_workers
from .paths import DATASET_DIR, OUTPUTS_ROOT
from .io import load_config_from_yaml

# =========================================================================== #
#                                TYPE ALIASES                                 #
# =========================================================================== #

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

# =========================================================================== #
#                                SUB-CONFIGURATIONS                           #
# =========================================================================== #

class SystemConfig(BaseModel):
    """Sub-configuration for system paths and hardware settings."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    device: str = Field(default_factory=detect_best_device)
    data_dir: Path = Field(default=DATASET_DIR)
    output_dir: Path = Field(default=OUTPUTS_ROOT)
    save_model: bool = True
    log_interval: PositiveInt = Field(default=10)
    project_name: str = "medmnist_experiment"

    @property
    def lock_file_path(self) -> Path:
        """Dynamically generates a cross-platform lock file path."""
        return Path(tempfile.gettempdir()) / f"{self.project_name}.lock"

    @field_validator("data_dir", "output_dir", mode="after")
    @classmethod
    def ensure_directories_exist(cls, v: Path) -> Path:
        "Ensure paths are absolute and create folders if missing."
        v.mkdir(parents=True, exist_ok=True)
        return v.resolve()

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """
        SSOT Validation: Ensures the requested device actually exists on this system.
        If the requested accelerator (cuda/mps) is unavailable, it self-corrects to 'cpu'.
        """
        if v == "auto":
            return detect_best_device()
        
        requested = v.lower()
        if "cuda" in requested and not torch.cuda.is_available():
            return "cpu"
        if "mps" in requested and not torch.backends.mps.is_available():
            return "cpu"
        return requested

class TrainingConfig(BaseModel):
    """Sub-configuration for core training hyperparameters."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    seed: int = 42
    batch_size: PositiveInt = Field(default=128)
    epochs: PositiveInt = Field(default=60)
    patience: NonNegativeInt = Field(default=15)
    learning_rate: PositiveFloat = Field(default=0.008)
    min_lr: PositiveFloat = Field(default=1e-6)
    momentum: Probability = Field(default=0.9)
    weight_decay: NonNegativeFloat = Field(default=5e-4)
    label_smoothing: Annotated[float, Field(default=0.0, ge=0.0, le=0.2)]
    mixup_alpha: NonNegativeFloat = Field(
        default=0.002,
        description="Mixup interpolation coefficient"
        )
    mixup_epochs: NonNegativeInt = Field(
        default=30,
        description="Number of epochs to apply mixup")
    use_tta: bool = True
    cosine_fraction: Probability = Field(default=0.5)
    use_amp: bool = False
    grad_clip: Optional[PositiveFloat] = Field(
        default=1.0,
        description="Max norm for gradient clipping; None to disable"
    )
    

class AugmentationConfig(BaseModel):
    """Sub-configuration for data augmentation parameters."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    hflip: Probability = Field(default=0.5)
    rotation_angle: Annotated[int, Field(default=10, ge=0, le=180)]
    jitter_val: NonNegativeFloat = Field(default=0.2)
    min_scale: Probability = Field(default=0.9)

    tta_translate: float = Field(
        default=2.0,
        description="Pixel shift for TTA"
    )
    tta_scale: float = Field(
        default=1.1,
        description="Scale factor for TTA"
    )
    tta_blur_sigma: float = Field(
        default=0.4,
        description="Gaussian blur sigma for TTA"
    )


class DatasetConfig(BaseModel):
    """Sub-configuration for dataset-specific metadata and sampling."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    dataset_name: str = "BloodMNIST"
    max_samples: Optional[PositiveInt] = Field(default=20000)
    use_weighted_sampler: bool = True
    in_channels: PositiveInt = Field(default=3)
    num_classes: PositiveInt = Field(default=8)
    img_size: PositiveInt = Field(
        default=28,
        description="Target square resolution for the model input"
        )
    force_rgb: bool = Field(
        default=True,
        description="Convert grayscale to 3-channel to enable ImageNet weights"
    )
    mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    std: tuple[float, ...] = (0.5, 0.5, 0.5)
    normalization_info: str = "N/A"
    is_anatomical: bool = True
    is_texture_based: bool = True

    @property
    def effective_in_channels(self) -> int:
        """Returns the actual number of channels the model will see"""
        return 3 if self.force_rgb else self.in_channels
    

class EvaluationConfig(BaseModel):
    """Sub-configuration for model evaluation and reporting."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    n_samples: PositiveInt = Field(default=12)
    fig_dpi: PositiveInt = Field(default=200)
    img_size: tuple[int, int] = (10, 10)
    cmap_confusion: str = "Blues"
    plot_style: str = "seaborn-v0_8-muted"
    report_format: str = "xlsx"

    @field_validator("report_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        supported = ["xlsx", "csv", "json"]
        if v.lower() not in supported:
            raise ValueError(f"Format {v} not supported. Use {supported}")
        return v.lower()
    
    save_confusion_matrix: bool = True
    save_predictions_grid: bool = True
    grid_cols: PositiveInt = Field(default=4)
    fig_size_predictions: tuple[int, int] = (12, 8)


# =========================================================================== #
#                                MAIN CONFIGURATION                          #
# =========================================================================== #

class Config(BaseModel):
    """
    Main configuration manifest.
    
    Acts as the root container for all sub-configurations and provides 
    the `from_args` factory to bridge raw CLI arguments into this 
    validated schema.
    """
    model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            frozen=True
    )
    
    # Nested configurations - Explicit access required (e.g., cfg.training.seed)
    system: SystemConfig = Field(default_factory=SystemConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    num_workers: NonNegativeInt = Field(default_factory=get_num_workers)
    model_name: str = "ResNet-18 Adapted"
    pretrained: bool = True


    @model_validator(mode="after")
    def validate_logic(self) -> "Config":
        """Cross-field logic validation after instantiation."""
        if self.training.mixup_epochs > self.training.epochs:
            raise ValueError(
                f"mixup_epochs ({self.training.mixup_epochs}) cannot exceed "
                f"epochs ({self.training.epochs})."
            )
        is_cpu = self.system.device == "cpu"
        if is_cpu and self.training.use_amp:
            raise ValueError("AMP cannot be enabled when using CPU device.")
        if self.pretrained and self.dataset.in_channels == 1 and not self.dataset.force_rgb:
            raise ValueError(
                "Pretrained models require 3-channel input. "
                "Set force_rgb=True in dataset config."
            )
        
        return self

    @field_validator("num_workers")
    @classmethod
    def check_cpu_count(cls, v: int) -> int:
        cpu_count = os.cpu_count() or 1
        return min(v, cpu_count)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """
        Factory method to create a validated Config instance from a YAML file.
        """
        raw_data = load_config_from_yaml(yaml_path)
        return cls(**raw_data)        
            
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Factory method to create a validated Config instance from a CLI namespace.
        """
        if hasattr(args, 'config') and args.config:
            return cls.from_yaml(Path(args.config))
        # -------------------------------------------------------------------------- #
        from .metadata import DATASET_REGISTRY

        # 1. Dataset metadata lookup
        dataset_key = args.dataset.lower()
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{args.dataset}' not found in DATASET_REGISTRY.")
        ds_meta = DATASET_REGISTRY[dataset_key]

        # 2. Early logic: Decide whether to force RGB based on model and dataset needs
        if hasattr(args, 'force_rgb') and args.force_rgb is not None:
            should_force_rgb = args.force_rgb
        else:
            should_force_rgb = (ds_meta.in_channels == 1) and args.pretrained

        # 3. Hardware-aware logic: Resolve device and validate AMP support
        final_device_str = args.device if hasattr(args, 'device') else "auto"

        # 4. Dataset limits
        final_max_samples = args.max_samples if (hasattr(args, 'max_samples') and args.max_samples > 0) else None

        # 5. Atomic construction of the Config object
        return cls(
            model_name=args.model_name,
            pretrained=args.pretrained,
            num_workers=args.num_workers,
            system=SystemConfig(
                device=final_device_str,
                data_dir=Path(args.data_dir),
                output_dir=Path(args.output_dir),
                save_model=args.save_model,
                log_interval=args.log_interval,
                project_name=getattr(args, 'project_name', "medmnist_experiment")
            ),
            training=TrainingConfig(
                seed=args.seed,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                mixup_alpha=args.mixup_alpha,
                mixup_epochs=args.mixup_epochs,
                use_tta=args.use_tta,
                cosine_fraction=args.cosine_fraction,
                use_amp=args.use_amp,
                grad_clip=args.grad_clip,
                label_smoothing=getattr(args, 'label_smoothing', 0.0),
                min_lr=getattr(args, 'min_lr', 1e-6)
            ),
            augmentation=AugmentationConfig(
                hflip=args.hflip,
                rotation_angle=args.rotation_angle,
                jitter_val=args.jitter_val,
                min_scale=getattr(args, 'min_scale', 0.9),
                tta_translate=getattr(args, 'tta_translate', 2.0),
                tta_scale=getattr(args, 'tta_scale', 1.1),
                tta_blur_sigma=getattr(args, 'tta_blur_sigma', 0.4)
            ),
            dataset=DatasetConfig(
                dataset_name=ds_meta.name,
                max_samples=final_max_samples,
                use_weighted_sampler=getattr(args, 'use_weighted_sampler', True),
                in_channels=ds_meta.in_channels,
                num_classes=len(ds_meta.classes),
                mean=ds_meta.mean,
                std=ds_meta.std,
                normalization_info=f"Mean={ds_meta.mean}, Std={ds_meta.std}",
                is_anatomical=ds_meta.is_anatomical,
                is_texture_based=ds_meta.is_texture_based,
                force_rgb=should_force_rgb,
                img_size=getattr(args, 'img_size', 28)
            ),
            evaluation=EvaluationConfig(
                n_samples=getattr(args, 'n_samples', 12),
                fig_dpi=getattr(args, 'fig_dpi', 200),
                plot_style=getattr(args, 'plot_style', "seaborn-v0_8-muted"),
                report_format=getattr(args, 'report_format', "xlsx")
            )
        )