"""
Main Execution Script for BloodMNIST Classification

This script orchestrates the entire training and evaluation pipeline:
1. Parses command-line arguments and sets up configuration.
2. Ensures environment reproducibility and prevents duplicate runs.
3. Loads the BloodMNIST dataset and creates PyTorch DataLoaders.
4. Initializes and adapts the model via the Models Factory.
5. Executes the training loop with the ModelTrainer.
6. Loads the best checkpoint and performs final evaluation, including TTA.
7. Generates visualization reports and saves structured Excel data.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import logging
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import (
    Config, Logger, parse_args, set_seed, kill_duplicate_processes, get_device, 
    DATASET_REGISTRY, RunPaths, setup_static_directories, ensure_single_instance
)
from scripts.data_handler import (
    load_medmnist, get_dataloaders, show_sample_images, get_augmentations_transforms
)
from scripts.models import get_model
from scripts.trainer import ModelTrainer
from scripts.evaluation import run_final_evaluation

# =========================================================================== #
#                               MAIN EXECUTION
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

def main() -> None:
    """
    The main function that controls the entire BloodMNIST training and evaluation flow.
    """
    
    # 1. Configuration Setup
    args = parse_args()

    dataset_key = args.dataset.lower()
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{args.dataset}' is not recognized. "
                         f"Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    ds_meta = DATASET_REGISTRY[dataset_key]

    cfg = Config(
        model_name=args.model_name,
        dataset_name=ds_meta.name,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        mixup_alpha=args.mixup_alpha,
        use_tta=args.use_tta,
        hflip=args.hflip,
        rotation_angle=args.rotation_angle,
        jitter_val=args.jitter_val
    )
    
    # Initialize Seed
    set_seed(cfg.seed)

    # 2. Environment Initialization
    lock_path = Path("/tmp/bloodmnist_training.lock")
    
    # Setup base project structure
    setup_static_directories()
    
    # Initialize dynamic paths for the current run
    paths = RunPaths(cfg.model_name, cfg.dataset_name)
    
    # Setup logger with run-specific file
    Logger.setup(
        name=paths.project_id,
        log_dir=paths.logs
    )
    legacy_logger = logging.getLogger("medmnist_pipeline")
    run_logger = logging.getLogger(paths.project_id)
    legacy_logger.handlers = run_logger.handlers
    legacy_logger.setLevel(run_logger.level)
    
    ensure_single_instance(lock_file=lock_path,logger=run_logger)
    kill_duplicate_processes(logger=run_logger)
    device = get_device(logger=run_logger)
    
    run_logger.info(f"Run Directory initialized: {paths.root}")
    run_logger.info(
        f"Hyperparameters: LR={cfg.learning_rate:.4f}, Momentum={cfg.momentum:.2f}, "
        f"Batch={cfg.batch_size}, Epochs={cfg.epochs}, MixUp={cfg.mixup_alpha}, "
        f"TTA={'Enabled' if cfg.use_tta else 'Disabled'}"
    )

    # Retrieve dataset metadata from registry
    ds_meta = DATASET_REGISTRY[cfg.dataset_name.lower()]
    run_logger.info(f"Dataset selected: {cfg.dataset_name} with {ds_meta.num_classes} classes.")

    # 3. Data Loading and Preparation
    data = load_medmnist(ds_meta)
    
    # Optional: Visual check of samples (saved to run-specific figures directory)
    show_sample_images(
        images=data.X_train,
        labels=data.y_train,
        classes=ds_meta.classes,
        save_path=paths.figures / "dataset_samples.png",
        cfg=cfg
    )

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

    # 4. Model Initialization (Factory Pattern)
    # Corrected: Passing both device and cfg as required by our __init__.py factory
    model = get_model(device=device, cfg=cfg)

    # 5. Training Execution
    run_logger.info("Starting training pipeline".center(60, "="))

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        output_dir=paths.models # Save checkpoints to the run-specific directory
    )
    best_path, train_losses, val_accuracies = trainer.train()

    # Load the best weights found during training
    model.load_state_dict(torch.load(best_path, map_location=device))
    run_logger.info(f"Loaded best checkpoint weights from: {best_path}")

    # 6. Final Evaluation (Metrics & Plots)
    # Get augmentation info string for reporting
    aug_info = get_augmentations_transforms(cfg)

    macro_f1, test_acc = run_final_evaluation(
        model=model,
        test_loader=test_loader,
        test_images=data.X_test,
        test_labels=data.y_test,
        class_names=ds_meta.classes,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        device=device,
        paths=paths,
        cfg=cfg,
        use_tta=cfg.use_tta,
        aug_info=aug_info
    )

    # Final Summary Logging
    run_logger.info(
        f"PIPELINE COMPLETED → "
        f"Test Acc: {test_acc:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Results saved in: {paths.root}"
    )


# =========================================================================== #
#                               ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    main()