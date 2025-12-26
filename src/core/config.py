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
import torch
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict, field_validator

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .system import detect_best_device, get_num_workers
from .paths import DATASET_DIR, OUTPUTS_ROOT

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
    log_interval: int = Field(default=10, gt=0)

    @field_validator("data_dir", "output_dir", mode="after")
    @classmethod
    def ensure_directories_exist(cls, v: Path) -> Path:
        "Ensure paths are absolute and create folders if missing."
        v.mkdir(parents=True, exist_ok=True)
        return v.resolve()

    @field_validator("device")
    @classmethod
    def validate_hardware_availability(cls, v: str) -> str:
        """
        SSOT Validation: Ensures the requested device actually exists on this system.
        If the requested accelerator (cuda/mps) is unavailable, it self-corrects to 'cpu'.
        """
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
    batch_size: int = Field(default=128, gt=0)
    epochs: int = Field(default=60, gt=0)
    patience: int = Field(default=15, ge=0)
    learning_rate: float = Field(default=0.008, gt=0)
    min_lr: float = Field(default=1e-6)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    weight_decay: float = Field(default=5e-4, ge=0.0)
    mixup_alpha: float = Field(default=0.002, ge=0.0)
    mixup_epochs: int = Field(default=30, ge=0)
    use_tta: bool = True
    cosine_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    use_amp: bool = False
    grad_clip: float | None = Field(default=1.0, gt=0)


class AugmentationConfig(BaseModel):
    """Sub-configuration for data augmentation parameters."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    hflip: float = Field(default=0.5, ge=0.0, le=1.0)
    rotation_angle: int = Field(default=10, ge=0, le=180)
    jitter_val: float = Field(default=0.2, ge=0.0)

    # TTA
    tta_translate: float = Field(default=2.0, description="Pixel shift for TTA")
    tta_scale: float = Field(default=1.1, description="Scale factor for TTA")
    tta_blur_sigma: float = Field(default=0.4, description="Gaussian blur sigma for TTA")


class DatasetConfig(BaseModel):
    """Sub-configuration for dataset-specific metadata and sampling."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    dataset_name: str = "BloodMNIST"
    max_samples: int | None = Field(default=20000, gt=0)
    use_weighted_sampler: bool = True
    in_channels: int = 3
    num_classes: int = 8
    mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    std: tuple[float, ...] = (0.5, 0.5, 0.5)
    normalization_info: str = "N/A"
    is_anatomical: bool = True

class EvaluationConfig(BaseModel):
    """Sub-configuration for model evaluation and reporting."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    n_samples: int = Field(default=12, gt=0)
    fig_dpi: int = Field(default=200, gt=0)
    img_size: tuple[int, int] = (10, 10)
    cmap_confusion: str = "Blues"
    plot_style: str = "seaborn-v0_8-muted"

    report_format: str = "xlsx"

    @field_validator("report_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        supported = ["xlsx", "csv", "json"]
        if v.lower() not in supported:
            raise ValueError(f"Format {v} not supoprted. Use {supported}")
        return v.lower()
    
    save_confusion_matrix: bool = True
    save_predictions_grid: bool = True
    grid_cols: int = 4
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
    
    num_workers: int = Field(default_factory=get_num_workers)
    model_name: str = "ResNet-18 Adapted"
    pretrained: bool = True

    @field_validator("num_workers")
    @classmethod
    def check_cpu_count(cls, v: int) -> int:
        cpu_count = os.cpu_count() or 1
        return min(v, cpu_count)

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Factory method to create a Config instance from CLI arguments."""
        from .metadata import DATASET_REGISTRY

        detected_device = detect_best_device()
        final_device = args.device if args.device != "auto" else detected_device

        actual_use_amp = args.use_amp and ("cuda" in final_device)

        dataset_key = args.dataset.lower()
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{args.dataset}' not found in DATASET_REGISTRY.")
        
        ds_meta = DATASET_REGISTRY[dataset_key]
        final_max = args.max_samples if args.max_samples > 0 else None
        
        return cls(
            model_name=args.model_name,
            pretrained=args.pretrained,
            num_workers=args.num_workers,
            system=SystemConfig(
                device=final_device,
                data_dir=Path(args.data_dir),
                output_dir=Path(args.output_dir),
                save_model=args.save_model,
                log_interval=args.log_interval
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
                use_tta=args.use_tta,
                cosine_fraction=args.cosine_fraction,
                mixup_epochs=getattr(args, 'mixup_epochs', args.epochs // 2),
                use_amp=actual_use_amp,
                grad_clip=args.grad_clip
            ),
            augmentation=AugmentationConfig(
                hflip=args.hflip,
                rotation_angle=args.rotation_angle,
                jitter_val=args.jitter_val,
                tta_translate=getattr(args, 'tta_translate', 2.0),
                tta_scale=getattr(args, 'tta_scale', 1.1),
                tta_blur_sigma=getattr(args, 'tta_blur_sigma', 0.4)
            ),
            dataset=DatasetConfig(
                dataset_name=ds_meta.name,
                max_samples=final_max,
                use_weighted_sampler=getattr(args, 'use_weighted_sampler', True),
                in_channels=ds_meta.in_channels,
                num_classes=len(ds_meta.classes),
                mean=ds_meta.mean,
                std=ds_meta.std,
                normalization_info=f"Mean={ds_meta.mean}, Std={ds_meta.std}",
                is_anatomical=ds_meta.is_anatomical
            ),
            evaluation=EvaluationConfig(
                n_samples=getattr(args, 'n_samples', 12),
                fig_dpi=getattr(args, 'fig_dpi', 200),
                plot_style=getattr(args, 'plot_style', "seaborn-v0_8-muted"),
                report_format=getattr(args, 'report_format', "xlsx")
            )
        )