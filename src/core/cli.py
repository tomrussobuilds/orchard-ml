"""
Argument Parsing Module

This module handles the command-line interface (CLI) for the training pipeline.
It bridges terminal inputs with the hierarchical Pydantic configuration, 
enabling dynamic control over hardware allocation, directory structures, 
and experimental hyperparameters.
"""

# =========================================================================== #
#                               Standard Imports                              #
# =========================================================================== #
import argparse

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .config import Config

# =========================================================================== #
#                                ARGUMENT PARSING                             #
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    """
    Configure and analyze command line arguments for the training script.

    Provides a comprehensive CLI to manage training schedules, hardware 
    settings (device, workers), filesystem paths, and model configurations.

    Returns:
        argparse.Namespace: An object containing all parsed command line arguments.
    """
    from .metadata import DATASET_REGISTRY

    parser = argparse.ArgumentParser(
        description="MedMNIST training pipeline based on adapted ResNet-18.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # We use a default Config instance to access the default values defined in sub-configs
    default_cfg = Config()

    # Group: System & Hardware
    sys_group = parser.add_argument_group("System & Hardware")
    
    sys_group.add_argument(
        '--device',
        type=str,
        default=default_cfg.system.device,
        help="Computing device (cpu, cuda, mps)."
    )
    sys_group.add_argument(
        '--num_workers',
        type=int,
        default=default_cfg.num_workers,
        help="Number of subprocesses for data loading."
    )

    # Group: Paths & Logging
    path_group = parser.add_argument_group("Paths & Logging")

    path_group.add_argument(
        '--data_dir',
        type=str,
        default=str(default_cfg.system.data_dir),
        help="Path to directory containing raw MedMNIST .npz files."
    )
    path_group.add_argument(
        '--output_dir',
        type=str,
        default=str(default_cfg.system.output_dir),
        help="Base directory for experiment outputs and runs."
    )
    path_group.add_argument(
        '--log_interval',
        type=int,
        default=default_cfg.system.log_interval,
        help="How many batches to wait before logging training status."
    )
    path_group.add_argument(
        '--no_save',
        action='store_false',
        dest='save_model',
        default=default_cfg.system.save_model,
        help="Disable saving the best model checkpoint."
    )
    path_group.add_argument(
        '--resume',
        type=str,
        default=None,
        help="Path to a .pth checkpoint to resume training or run evaluation."
    )
    path_group.add_argument(
        '--project_name',
        type=str,
        default=default_cfg.system.project_name,
        help="Logical name for the experiment suite."
    )
    
    # Group: Training Hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    
    train_group.add_argument(
        '--epochs',
        type=int,
        default=default_cfg.training.epochs
    )
    train_group.add_argument(
        '--batch_size',
        type=int,
        default=default_cfg.training.batch_size
    )
    train_group.add_argument(
        '--lr', '--learning_rate',
        type=float,
        default=default_cfg.training.learning_rate
    )
    train_group.add_argument(
        '--seed',
        type=int,
        default=default_cfg.training.seed
    )
    train_group.add_argument(
        '--patience',
        type=int,
        default=default_cfg.training.patience
    )
    train_group.add_argument(
        '--momentum',
        type=float,
        default=default_cfg.training.momentum
    )
    train_group.add_argument(
        '--weight_decay',
        type=float,
        default=default_cfg.training.weight_decay
    )
    train_group.add_argument(
        '--cosine_fraction',
        type=float,
        default=default_cfg.training.cosine_fraction,
        help="Fraction of total epochs to apply cosine annealing before switching to ReduceLROnPlateau."
    )
    # Coerenza con cfg.training.use_amp
    train_group.add_argument(
        '--use_amp',
        action='store_true',
        default=default_cfg.training.use_amp,
        help="Enable Automatic Mixed Precision (FP16) during training."
    )
    train_group.add_argument(
        '--no_amp',
        action='store_false',
        dest='use_amp',
        help="Disable Automatic Mixed Precision."
    )
    train_group.add_argument(
        '--grad_clip',
        type=float,
        default=default_cfg.training.grad_clip,
        help="Maximum norm for gradient clipping (set to 0 to disable)."
    )
    train_group.add_argument(
        '--label_smoothing',
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0 to 1.0)."
    )

    # Group: Regularization & Augmentation
    aug_group = parser.add_argument_group("Regularization & Augmentation")
    
    aug_group.add_argument(
        '--mixup_alpha',
        type=float,
        default=default_cfg.training.mixup_alpha
    )
    aug_group.add_argument(
        '--mixup_epochs',
        type=int,
        default=default_cfg.training.mixup_epochs,
        help="Number of epochs to apply MixUp augmentation."
    )
    aug_group.add_argument(
        '--no_tta',
        action='store_false',
        dest='use_tta',
        default=default_cfg.training.use_tta,
        help="Disable TTA during final evaluation."
    )
    aug_group.add_argument(
        '--hflip',
        type=float,
        default=default_cfg.augmentation.hflip
    )
    aug_group.add_argument(
        '--rotation_angle',
        type=int,
        default=default_cfg.augmentation.rotation_angle
    )
    aug_group.add_argument(
        '--jitter_val',
        type=float,
        default=default_cfg.augmentation.jitter_val
    )

    # Group: Dataset Selection and Configuration
    dataset_group = parser.add_argument_group("Dataset Configuration")

    dataset_group.add_argument(
        '--dataset',
        type=str,
        default="bloodmnist",
        choices=DATASET_REGISTRY.keys(),
        help="Target MedMNIST dataset."
    )
    dataset_group.add_argument(
        '--max_samples',
        type=int,
        default=default_cfg.dataset.max_samples or 0,
        help="Max training samples (Use 0 or -1 for full dataset)."
    )
    dataset_group.add_argument(
        '--balanced',
        action='store_true',
        dest='use_weighted_sampler',
        default=default_cfg.dataset.use_weighted_sampler,
        help="Use WeightedRandomSampler to handle class imbalance."
    )
    dataset_group.add_argument(
        '--force_rgb',
        action='store_true',
        dest='force_rgb',
        default=None, 
        help="Force conversion of grayscale to RGB."
    )
    dataset_group.add_argument(
        '--no_force_rgb',
        action='store_false',
        dest='force_rgb',
        help="Disable grayscale to RGB conversion."
    )

    # Group: Model Selection
    model_group = parser.add_argument_group("Model Configuration")

    model_group.add_argument(
        '--model_name',
        type=str,
        default=default_cfg.model_name,
        help="Architecture identifier."
    )
    model_group.add_argument(
        '--pretrained',
        action='store_true',
        default=default_cfg.pretrained,
        help="Load ImageNet weights for the backbone (default: True)."
    )
    model_group.add_argument(
        '--no_pretrained',
        action='store_false',
        dest='pretrained',
        help="Initialize model with random weights."
    )

    # Group: Evaluation & Reporting
    eval_group = parser.add_argument_group("Evaluation & Reporting")

    eval_group.add_argument(
        '--n_samples',
        type=int,
        default=default_cfg.evaluation.n_samples,
        help="Number of images to display in the prediction grid."
    )
    eval_group.add_argument(
        '--fig_dpi',
        type=int,
        default=default_cfg.evaluation.fig_dpi,
        help="Resolution (DPI) for saved plots."
    )
    eval_group.add_argument(
        '--report_format',
        type=str,
        default=default_cfg.evaluation.report_format,
        choices=["xlsx", "csv", "json"],
        help="Format for the final experiment summary."
    )
    eval_group.add_argument(
        '--plot_style',
        type=str,
        default=default_cfg.evaluation.plot_style,
        help="Matplotlib style for visualizations (e. g., 'ggplot', 'bmh')."
    )

    return parser.parse_args()