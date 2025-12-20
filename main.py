"""
Main Execution Script for BloodMNIST Classification

This script orchestrates the entire training and evaluation pipeline:
1. Parses command-line arguments and sets up configuration.
2. Ensures environment reproducibility and prevents duplicate runs.
3. Loads the BloodMNIST dataset and creates PyTorch DataLoaders.
4. Initializes and adapts the ResNet-18 model.
5. Executes the training loop with the ModelTrainer.
6. Loads the best checkpoint and performs final evaluation, including TTA.
7. Generates visualization reports (plots, confusion matrix).
8. Builds and saves the final structured training report to Excel.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Final
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.utils import (
    Config, Logger, parse_args, set_seed, kill_duplicate_processes, get_device, 
    NPZ_PATH, REPORTS_DIR
)
from scripts.data_handler import (
    load_bloodmnist, get_dataloaders, show_sample_images
    )
from scripts.models import get_model
from scripts.trainer import ModelTrainer
from scripts.evaluation import (
    run_final_evaluation, create_structured_report
)

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()

# =========================================================================== #
#                               MAIN EXECUTION
# =========================================================================== #

def main() -> None:
    """
    The main function that controls the entire BloodMNIST training and evaluation flow.
    """
    
    # 1. Configuration Setup
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        mixup_alpha=args.mixup_alpha,
        # Argument --no_tta means use_tta is False
        use_tta=not args.no_tta,
        hflip=args.hflip,
        rotation_angle=args.rotation_angle,
        jitter_val=args.jitter_val
    )

    # Seed
    set_seed(cfg.seed)

    # 2. Environment Initialization
    kill_duplicate_processes()
    device = get_device(logger=logger)

    logger.info(
        f"Hyperparameters: LR={cfg.learning_rate:.4f}, Momentum={cfg.momentum:.2f}, WeightDecay={cfg.weight_decay:.1e}, "
        f"Batch={cfg.batch_size}, Epochs={cfg.epochs}, Rot={cfg.rotation_angle}, Jitter={cfg.jitter_val}, HFlip={cfg.hflip}, "
        f"MixUp={cfg.mixup_alpha}, Seed={cfg.seed}, TTA={'Enabled' if cfg.use_tta else 'Disabled'}"
    )

    # 3. Data Loading and Preparation
    data = load_bloodmnist(NPZ_PATH)
    logger.info(
        f"Dataset loaded → Train:{len(data.X_train)} | "
        f"Val:{len(data.X_val)} | "
        f"Test:{len(data.X_test)}"
    )

    # Optional visualization of sample images (saved to figures directory)
    show_sample_images(data)

    # Create PyTorch DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

    # 4. Model Initialization
    model = get_model(device=device)

    # 5. Training Execution
    logger.info("Starting training".center(60, "="))

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=cfg
    )
    best_path, train_losses, val_accuracies = trainer.train()

    # Load the state dictionary of the best performing model (from validation)
    model.load_state_dict(torch.load(best_path, map_location=device))
    logger.info(f"Best model loaded from {best_path}")

    # 6. Final Evaluation and Reporting Generation
    macro_f1, test_acc = run_final_evaluation(
        model=model,
        test_loader=test_loader,
        data=data,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        device=device,
        use_tta=cfg.use_tta,
    )

    # 7. Build and Save Structured Report
    report = create_structured_report(
        val_accuracies=val_accuracies,
        macro_f1=macro_f1,
        test_acc=test_acc,
        train_losses=train_losses,
        best_path=best_path,
        cfg=cfg
    )

    # Log final metrics cleanly
    logger.info(
        f"FINAL RESULTS → "
        f"Test Accuracy: {report.test_accuracy:.4f} | "
        f"Macro F1: {report.test_macro_f1:.4f} | "
        f"Best Val Accuracy: {report.best_val_accuracy:.4f}"
    )
    logger.info("Training & evaluation completed successfully!")

    # Save Excel report
    excel_path = REPORTS_DIR / "training_report.xlsx"
    report.save(excel_path)


# =========================================================================== #
#                               ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    main()