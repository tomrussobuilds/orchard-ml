"""
Configuration and Command-Line Interface Module

This module defines the training hyperparameters using Pydantic for validation
and type safety. It also provides the argument parsing logic for the 
command-line interface (CLI).
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import os
import argparse

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                                HELPER FUNCTIONS
# =========================================================================== #

def _get_num_workers_config() -> int:
    """
    Calculates the default value for num_workers based on the environment.

    If DOCKER_REPRODUCIBILITY_MODE is set to '1' or 'TRUE', it returns 0
    to force single-thread execution for bit-per-bit determinism.
    
    Returns:
        int: The determined number of data loader workers (0 or 4).
    """
    is_docker_reproducible = os.environ.get("DOCKER_REPRODUCIBILITY_MODE", "0").upper() in ("1", "TRUE")
    return 0 if is_docker_reproducible else 4

# =========================================================================== #
#                                CONFIGURATION
# =========================================================================== #

class Config(BaseModel):
    """Configuration class for training hyperparameters using Pydantic validation."""
    model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            frozen=True
    )
    
    # Core Hyperparameters
    seed: int = 42
    batch_size: int = Field(default=128, gt=0)
    num_workers: int = Field(default_factory=_get_num_workers_config)
    epochs: int = Field(default=60, gt=0)
    patience: int = Field(default=15, ge=0)
    learning_rate: float = Field(default=0.008, gt=0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    weight_decay: float = Field(default=5e-4, ge=0.0)
    mixup_alpha: float = Field(default=0.002, ge=0.0)
    use_tta: bool = True
    cosine_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Data Augmentation Parameters
    hflip: float = Field(default=0.5, ge=0.0, le=1.0)
    rotation_angle: int = Field(default=10, ge=0, le=180)
    jitter_val: float = Field(default=0.2, ge=0.0)

    model_name: str = "ResNet-18 Adapted"
    dataset_name: str = "BloodMNIST"

    # Dataset Metadata (to be populated from DATASET_REGISTRY)
    normalization_info : str =  "N/A"
    in_channels : int = 3
    num_classes : int = 8
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

# =========================================================================== #
#                                ARGUMENT PARSING
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    """
    Configure and analyze command line arguments for the training script.

    Returns:
        argparse.Namespace: An object containing all parsed command line arguments.
    """
    from .dataset_metadata import DATASET_REGISTRY

    parser = argparse.ArgumentParser(
        description="MedMNIST training pipeline based on adapted ResNet-18.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Aggiunge i default automaticamente nell'help
    )
    
    default_cfg = Config()

    # Group: Training Hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    
    train_group.add_argument(
        '--epochs',
        type=int,
        default=default_cfg.epochs
    )
    train_group.add_argument(
        '--batch_size',
        type=int,
        default=default_cfg.batch_size
    )
    train_group.add_argument(
        '--lr', '--learning_rate',
        type=float,
        default=default_cfg.learning_rate
    )
    train_group.add_argument(
        '--seed',
        type=int,
        default=default_cfg.seed
    )
    train_group.add_argument(
        '--patience',
        type=int,
        default=default_cfg.patience
    )
    train_group.add_argument(
        '--momentum',
        type=float,
        default=default_cfg.momentum
    )
    train_group.add_argument(
        '--weight_decay',
        type=float,
        default=default_cfg.weight_decay
    )
    train_group.add_argument(
        '--cosine_fraction',
        type=float,
        default=default_cfg.cosine_fraction,
        help="Fraction of total epochs to apply cosine annealing before switching to ReduceLROnPlateau."
    )
    # Group: Regularization & Augmentation
    aug_group = parser.add_argument_group("Regularization & Augmentation")
    
    aug_group.add_argument(
        '--mixup_alpha',
        type=float,
        default=default_cfg.mixup_alpha
    )
    aug_group.add_argument(
        '--no_tta',
        action='store_false',
        dest='use_tta',
        default=default_cfg.use_tta,
        help="Disable TTA during final evaluation."
    )
    aug_group.add_argument(
        '--hflip',
        type=float,
        default=default_cfg.hflip
    )
    aug_group.add_argument(
        '--rotation_angle',
        type=int,
        default=default_cfg.rotation_angle
    )
    aug_group.add_argument(
        '--jitter_val',
        type=float,
        default=default_cfg.jitter_val
    )
    
    # Group: Dataset Selection
    dataset_group = parser.add_argument_group("Dataset Configuration")

    dataset_group.add_argument(
        '--dataset',
        type=str,
        default="bloodmnist",
        choices=DATASET_REGISTRY.keys(),
        help="Target MedMNIST dataset."
    )

    # Group: Model Selection
    model_group = parser.add_argument_group("Model Configuration")

    model_group.add_argument(
        '--model_name',
        type=str,
        default="ResNet-18 Adapted",
        help="Architecture identifier."
    )

    return parser.parse_args()