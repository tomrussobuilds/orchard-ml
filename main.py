"""
Main Execution Script for MedMNIST Classification Pipeline

This orchestrator manages the lifecycle of a deep learning experiment, applying 
an adapted ResNet-18 architecture to various MedMNIST datasets (e.g., BloodMNIST). 

Key Pipeline Features:
1. Dynamic Configuration: Metadata-driven setup (mean/std, classes, channels) 
   leveraging a centralized Dataset Registry.
2. Root Orchestration: Centralized environment setup via RootOrchestrator 
   (seeding, locking, directory management, and logging).
3. Data Management: Handles automated loading, subset mocking for testing, 
   and robust PyTorch DataLoader creation with configurable augmentations.
4. Model Orchestration: Factory-based initialization of specialized architectures.
5. Training & Recovery: Executes standardized training loops with automated 
   checkpointing of the best model based on validation performance.
6. Comprehensive Evaluation: Performs final testing with Test-Time Augmentation (TTA), 
   generates diagnostic visualizations (Confusion Matrices, Loss Curves), and 
   exports structured performance reports in Excel format.
"""
# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
# torch import removed: hardware abstraction handled by RootOrchestrator

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import (
    Config, parse_args, DATASET_REGISTRY, RootOrchestrator
)
from src.data_handler import (
    load_medmnist, get_dataloaders, show_sample_images, get_augmentations_description
)
from src.models import get_model
from src.trainer import ModelTrainer
from src.evaluation import run_final_evaluation

# =========================================================================== #
#                               MAIN EXECUTION
# =========================================================================== #

def main() -> None:
    """
    The main function that controls the entire training and evaluation flow.
    """
    
    # 1. Configuration & Root Orchestration
    args = parse_args()
    cfg = Config.from_args(args)
    
    # Initialize Core Services (Seed, Paths, Logs, Locks)
    # This encapsulates the system-level boilerplate previously in main
    orchestrator = RootOrchestrator(cfg)
    paths = orchestrator.initialize_core_services()
    
    # Access the logger initialized by the orchestrator
    run_logger = orchestrator.run_logger

    # NEW: Retrieve hardware device object via orchestrator abstraction
    device = orchestrator.get_device()

    # Retrieve dataset metadata from registry
    ds_meta = DATASET_REGISTRY[cfg.dataset.dataset_name.lower()]
    run_logger.info(f"Dataset selected: {cfg.dataset.dataset_name} with {cfg.dataset.num_classes} classes.")

    # 2. Data Loading and Preparation
    # 'data' is now a metadata container for Lazy Loading
    data = load_medmnist(ds_meta)

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)
    
    # Optional: Visual check of samples (saved to run-specific figures directory)
    show_sample_images(
        loader=train_loader,
        classes=ds_meta.classes,
        save_path=paths.figures / "dataset_samples.png",
        cfg=cfg
    )

    # 3. Model Initialization (Factory Pattern)
    # The device object is passed directly, maintaining framework independence in main
    model = get_model(device=device, cfg=cfg)

    # 4. Training Execution
    run_logger.info("Starting training pipeline".center(60, "="))

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        output_dir=paths.models
    )
    best_path, train_losses, val_accuracies = trainer.train()

    # 5. Model Recovery & Weight Loading
    orchestrator.load_weights(model, best_path)

    # 6. Final Evaluation (Metrics & Plots)
    aug_info = get_augmentations_description(cfg)

    # test_images and test_labels set to None to trigger Lazy extraction from loader
    macro_f1, test_acc = run_final_evaluation(
        model=model,
        test_loader=test_loader,
        test_images=None,
        test_labels=None,
        class_names=ds_meta.classes,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        device=device,
        paths=paths,
        cfg=cfg,
        use_tta=cfg.training.use_tta,
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