"""
Argument Parsing Module.

Handles command-line interface (CLI) for the training pipeline.
Bridges terminal inputs with hierarchical Pydantic configuration.
"""

import argparse
from pathlib import Path

from .config.augmentation_config import AugmentationConfig
from .config.evaluation_config import EvaluationConfig
from .config.hardware_config import HardwareConfig
from .config.telemetry_config import TelemetryConfig
from .config.training_config import TrainingConfig


# ARGUMENT PARSING
def parse_args() -> argparse.Namespace:
    """
    Configure and parse command-line arguments for the training script.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="MedMNIST training pipeline with multi-resolution support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Instantiate configs with defaults
    # ArchitectureConfig and DatasetConfig omitted (require mandatory data-dependent fields)
    hardware_def = HardwareConfig()
    telemetry_def = TelemetryConfig()
    train_def = TrainingConfig()
    eval_def = EvaluationConfig()
    aug_def = AugmentationConfig()

    # ===== Global Strategy =====
    strat_group = parser.add_argument_group("Global Strategy")

    strat_group.add_argument(
        "--config",
        type=str,
        default="recipes/config_mini_cnn.yaml",
        help="Path to YAML config file (default: MiniCNN on BloodMNIST 28x28)",
    )
    strat_group.add_argument(
        "--project_name",
        type=str,
        default=hardware_def.project_name,
        help="Experiment suite name (for logging and locks)",
    )
    strat_group.add_argument(
        "--reproducible",
        action="store_true",
        dest="reproducible",
        help="Enforce strict determinism (deterministic algorithms, num_workers=0)",
    )

    # ===== System & Hardware =====
    sys_group = parser.add_argument_group("System & Hardware")

    sys_group.add_argument(
        "--device",
        type=str,
        default=hardware_def.device,
        help="Computing device (cpu, cuda, mps, auto)",
    )
    sys_group.add_argument(
        "--num_workers",
        type=int,
        dest="num_workers",
        default=None,
        help="Data loading subprocesses",
    )

    # ===== Paths & Logging =====
    path_group = parser.add_argument_group("Paths & Logging")

    path_group.add_argument(
        "--data_dir",
        type=str,
        default=str(telemetry_def.data_dir),
        help="Directory containing MedMNIST .npz files",
    )
    path_group.add_argument(
        "--output_dir",
        type=str,
        default=str(telemetry_def.output_dir),
        help="Base directory for experiment outputs",
    )
    path_group.add_argument(
        "--log_interval",
        type=int,
        default=telemetry_def.log_interval,
        help="Batches between training status logs",
    )
    path_group.add_argument(
        "--no_save",
        action="store_false",
        dest="save_model",
        default=telemetry_def.save_model,
        help="Disable best model checkpoint saving",
    )
    path_group.add_argument(
        "--resume",
        "--checkpoint",
        type=str,
        default=None,
        dest="resume",
        help="Path to .pth checkpoint for resuming training or export",
    )

    # ===== Training Hyperparameters =====
    train_group = parser.add_argument_group("Training Hyperparameters")

    train_group.add_argument("--epochs", type=int, default=train_def.epochs)
    train_group.add_argument("--batch_size", type=int, default=train_def.batch_size)
    train_group.add_argument("--lr", "--learning_rate", type=float, default=train_def.learning_rate)
    train_group.add_argument("--seed", type=int, default=train_def.seed)
    train_group.add_argument("--patience", type=int, default=train_def.patience)
    train_group.add_argument("--momentum", type=float, default=train_def.momentum)
    train_group.add_argument("--weight_decay", type=float, default=train_def.weight_decay)
    train_group.add_argument(
        "--cosine_fraction",
        type=float,
        default=train_def.cosine_fraction,
        help="Fraction of epochs for cosine annealing before plateau scheduler",
    )
    train_group.add_argument(
        "--use_amp",
        action="store_true",
        default=train_def.use_amp,
        help="Enable Automatic Mixed Precision (FP16)",
    )
    train_group.add_argument(
        "--no_amp", action="store_false", dest="use_amp", help="Disable Automatic Mixed Precision"
    )
    train_group.add_argument(
        "--grad_clip",
        type=float,
        default=train_def.grad_clip,
        help="Max gradient norm (0 to disable)",
    )
    train_group.add_argument(
        "--label_smoothing", type=float, default=0.0, help="Label smoothing factor (0.0-1.0)"
    )
    train_group.add_argument(
        "--scheduler_type",
        type=str,
        default=train_def.scheduler_type,
        choices=["cosine", "plateau", "step", "none"],
        help="Learning rate decay strategy",
    )
    train_group.add_argument(
        "--min_lr", type=float, default=train_def.min_lr, help="Minimum LR floor"
    )
    train_group.add_argument(
        "--scheduler_patience",
        type=int,
        default=train_def.scheduler_patience,
        help="Epochs before LR decay (plateau only)",
    )
    train_group.add_argument(
        "--scheduler_factor",
        type=float,
        default=train_def.scheduler_factor,
        help="LR decay multiplicative factor",
    )
    train_group.add_argument(
        "--step_size",
        type=int,
        default=train_def.step_size,
        help="LR decay period in epochs (step only)",
    )
    train_group.add_argument(
        "--criterion_type",
        type=str,
        default=train_def.criterion_type,
        choices=["cross_entropy", "bce_logit", "focal"],
        help="Loss function (bce_logit for multi-label)",
    )
    train_group.add_argument(
        "--focal_gamma",
        type=float,
        default=train_def.focal_gamma,
        help="Focal Loss gamma (higher = more focus on hard samples)",
    )

    # ===== Regularization & Augmentation =====
    aug_group = parser.add_argument_group("Regularization & Augmentation")

    aug_group.add_argument("--mixup_alpha", type=float, default=train_def.mixup_alpha)
    aug_group.add_argument(
        "--mixup_epochs", type=int, default=train_def.mixup_epochs, help="Epochs to apply MixUp"
    )
    aug_group.add_argument(
        "--no_tta",
        action="store_false",
        dest="use_tta",
        default=train_def.use_tta,
        help="Disable TTA during evaluation",
    )
    aug_group.add_argument("--hflip", type=float, default=aug_def.hflip)
    aug_group.add_argument("--rotation_angle", type=int, default=aug_def.rotation_angle)
    aug_group.add_argument("--jitter_val", type=float, default=aug_def.jitter_val)

    # ===== Dataset Configuration =====
    dataset_group = parser.add_argument_group("Dataset Configuration")

    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="bloodmnist",
        help="Target dataset (MedMNIST or custom datasets like galaxy10)",
    )
    dataset_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max training samples (0 or -1 for full dataset)",
    )
    dataset_group.add_argument(
        "--balanced",
        action="store_true",
        dest="use_weighted_sampler",
        default=False,
        help="Use WeightedRandomSampler for class imbalance",
    )
    dataset_group.add_argument(
        "--force_rgb",
        action="store_true",
        dest="force_rgb",
        default=None,
        help="Force grayscale to RGB conversion",
    )
    dataset_group.add_argument(
        "--no_force_rgb",
        action="store_false",
        dest="force_rgb",
        help="Disable grayscale to RGB conversion",
    )
    dataset_group.add_argument(
        "--is_anatomical",
        type=lambda x: str(x).lower() == "true",
        default=None,
        help="Override anatomical orientation flag (None uses registry default)",
    )
    dataset_group.add_argument(
        "--is_texture_based",
        type=lambda x: str(x).lower() == "true",
        default=None,
        help="Override texture-based flag (None uses registry default)",
    )
    dataset_group.add_argument(
        "--resolution", type=int, choices=[28, 224], default=28, help="Dataset image resolution"
    )

    # ===== Model Configuration =====
    model_group = parser.add_argument_group("Model Configuration")

    model_group.add_argument(
        "--model_name", type=str, default="mini_cnn", help="Architecture identifier"
    )
    model_group.add_argument(
        "--pretrained", action="store_true", default=False, help="Load ImageNet weights"
    )
    model_group.add_argument(
        "--no_pretrained",
        action="store_false",
        dest="pretrained",
        help="Random weight initialization",
    )

    # ===== Evaluation & Reporting =====
    eval_group = parser.add_argument_group("Evaluation & Reporting")

    eval_group.add_argument(
        "--n_samples", type=int, default=eval_def.n_samples, help="Images in prediction grid"
    )
    eval_group.add_argument(
        "--fig_dpi", type=int, default=eval_def.fig_dpi, help="Plot resolution (DPI)"
    )
    eval_group.add_argument(
        "--report_format",
        type=str,
        default=eval_def.report_format,
        choices=["xlsx", "csv", "json"],
        help="Experiment summary format",
    )
    eval_group.add_argument(
        "--plot_style",
        type=str,
        default=eval_def.plot_style,
        help="Matplotlib style (e.g., 'ggplot', 'seaborn-v0_8-muted')",
    )

    # ===== Model Export =====
    export_group = parser.add_argument_group("Model Export (ONNX/TorchScript)")

    export_group.add_argument(
        "--format",
        type=str,
        choices=["onnx", "torchscript", "both"],
        default=None,
        help="Export format (triggers export mode when specified)",
    )

    export_group.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Export output path (auto-generated if None)",
    )

    export_group.add_argument(
        "--opset_version",
        type=int,
        default=18,
        help="ONNX opset version (18=latest, clean export with no warnings)",
    )

    export_group.add_argument(
        "--dynamic_axes",
        action="store_true",
        default=True,
        help="Enable dynamic batch size for ONNX export",
    )

    export_group.add_argument(
        "--no_dynamic_axes",
        action="store_false",
        dest="dynamic_axes",
        help="Disable dynamic batch size",
    )

    export_group.add_argument(
        "--do_constant_folding",
        action="store_true",
        default=True,
        help="Optimize constant operations at export time",
    )

    export_group.add_argument(
        "--torchscript_method",
        type=str,
        choices=["trace", "script"],
        default="trace",
        help="TorchScript conversion method",
    )

    export_group.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Apply INT8 quantization",
    )

    export_group.add_argument(
        "--quantization_backend",
        type=str,
        choices=["qnnpack", "fbgemm"],
        default="qnnpack",
        help="Quantization backend (qnnpack=mobile, fbgemm=x86)",
    )

    export_group.add_argument(
        "--validate_export",
        action="store_true",
        default=True,
        help="Validate exported model against PyTorch",
    )

    export_group.add_argument(
        "--no_validate_export",
        action="store_false",
        dest="validate_export",
        help="Skip export validation",
    )

    export_group.add_argument(
        "--validation_samples",
        type=int,
        default=10,
        help="Number of samples for export validation",
    )

    export_group.add_argument(
        "--max_deviation",
        type=float,
        default=1e-5,
        help="Maximum allowed output deviation between PyTorch and exported model",
    )

    optuna_group = parser.add_argument_group("Optuna Hyperparameter Optimization")

    # Study configuration
    optuna_group.add_argument(
        "--study_name", type=str, default="medmnist_optimization", help="Name of the Optuna study"
    )

    optuna_group.add_argument(
        "--n_trials", type=int, default=50, help="Number of optimization trials"
    )

    optuna_group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optimization timeout in seconds (None = unlimited)",
    )

    # Optimization target
    optuna_group.add_argument(
        "--metric_name",
        type=str,
        default="auc",
        choices=["loss", "accuracy", "auc"],
        help="Metric to optimize",
    )

    optuna_group.add_argument(
        "--direction",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Optimization direction",
    )

    # Search strategy
    optuna_group.add_argument(
        "--sampler_type",
        type=str,
        default="tpe",
        choices=["tpe", "cmaes", "random", "grid"],
        help="Hyperparameter sampling algorithm",
    )

    optuna_group.add_argument(
        "--search_space_preset",
        type=str,
        default="full",
        choices=["quick", "full", "optimization_only", "regularization_only"],
        help="Predefined search space configuration",
    )

    # Pruning
    optuna_group.add_argument(
        "--enable_pruning",
        type=bool,
        default=True,
        help="Enable early stopping of unpromising trials",
    )

    optuna_group.add_argument(
        "--pruner_type",
        type=str,
        default="median",
        choices=["median", "percentile", "hyperband", "none"],
        help="Pruning algorithm",
    )

    optuna_group.add_argument(
        "--pruning_warmup_epochs",
        type=int,
        default=10,
        help="Min epochs before pruning can trigger",
    )

    # Storage
    optuna_group.add_argument(
        "--storage_type",
        type=str,
        default="sqlite",
        choices=["sqlite", "memory", "postgresql"],
        help="Backend for study persistence",
    )

    optuna_group.add_argument(
        "--storage_path",
        type=Path,
        default=None,
        help="Path to SQLite database or PostgreSQL connection string",
    )

    # Execution
    optuna_group.add_argument(
        "--n_jobs", type=int, default=1, help="Parallel trials (1=sequential, -1=all cores)"
    )

    optuna_group.add_argument(
        "--load_if_exists", type=bool, default=True, help="Resume existing study or create new"
    )

    # Reporting
    optuna_group.add_argument(
        "--save_plots",
        type=bool,
        default=True,
        help="Generate and save optimization visualizations",
    )

    optuna_group.add_argument(
        "--save_best_config", type=bool, default=True, help="Export best trial as YAML config"
    )

    return parser.parse_args()
