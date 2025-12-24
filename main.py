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
    Main orchestrator that controls the end-to-end training and evaluation flow.
    """
    
    # 1. Configuration & Root Orchestration
    args         = parse_args()
    cfg          = Config.from_args(args)
    orchestrator = RootOrchestrator(cfg)
    
    # Initialize Core Services (Seed, Paths, Logs, Locks)
    paths        = orchestrator.initialize_core_services()
    run_logger   = orchestrator.run_logger
    device       = orchestrator.get_device()
    
    # Retrieve dataset metadata
    ds_meta      = DATASET_REGISTRY[cfg.dataset.dataset_name.lower()]

    try:
        # --- 2. Data Preparation ---
        run_logger.info(f" Preparing Dataset: {cfg.dataset.dataset_name} ".center(60, "-"))
        
        data    = load_medmnist(ds_meta)
        loaders = get_dataloaders(data, cfg)
        train_loader, val_loader, test_loader = loaders
        
        show_sample_images(
            loader    = train_loader,
            classes   = ds_meta.classes,
            save_path = paths.figures / "dataset_samples.png",
            cfg       = cfg
        )

        # --- 3. Model & Training Execution ---
        run_logger.info(f" Starting Pipeline: {cfg.model.model_type} ".center(60, "#"))

        model   = get_model(device=device, cfg=cfg)
        trainer = ModelTrainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            device       = device,
            cfg          = cfg,
            output_dir   = paths.models
        )
        
        # Start training and capture history
        best_path, train_losses, val_accuracies = trainer.train()

        # --- 4. Model Recovery & Evaluation ---
        run_logger.info(" Final Evaluation Phase ".center(60, "-"))
        
        orchestrator.load_weights(model, best_path)
        
        macro_f1, test_acc = run_final_evaluation(
            model          = model,
            test_loader    = test_loader,
            test_images    = None,
            test_labels    = None,
            class_names    = ds_meta.classes,
            train_losses   = train_losses,
            val_accuracies = val_accuracies,
            device         = device,
            paths          = paths,
            cfg            = cfg,
            use_tta        = cfg.training.use_tta,
            aug_info       = get_augmentations_description(cfg)
        )

        # --- 5. Structured Summary Logging ---
        summary = (
            f"\n{'#'*60}\n"
            f"{' PIPELINE EXECUTION SUMMARY ':^60}\n"
            f"{'-'*60}\n"
            f"  » Dataset:      {cfg.dataset.dataset_name}\n"
            f"  » Architecture: {cfg.model.model_type}\n"
            f"  » Test Acc:     {test_acc:>8.2%}\n"
            f"  » Macro F1:     {macro_f1:>8.4f}\n"
            f"  » Artifacts:    {paths.root}\n"
            f"{'#'*60}"
        )
        run_logger.info(summary)

    except Exception as e:
        run_logger.error(f"Pipeline crashed during execution: {e}", exc_info=True)
        raise e
        
    finally:
        run_logger.info(f"Cleanup finished. Run directory: {paths.root}")


# =========================================================================== #
#                               ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    main()