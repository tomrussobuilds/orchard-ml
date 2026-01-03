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
   exports structured performance reports.
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
from src.trainer import (
    ModelTrainer, get_criterion, get_optimizer, get_scheduler
)
from src.evaluation import run_final_evaluation

# =========================================================================== #
#                               MAIN EXECUTION
# =========================================================================== #

def main() -> None:
    """
    Main orchestrator that controls the end-to-end training and evaluation flow.
    
    This function initializes the environment through the RootOrchestrator,
    coordinates the data loading phase, executes the training loop via ModelTrainer,
    and concludes with a rigorous evaluation and reporting phase.
    """
    
    # 1. Configuration & Root Orchestration
    # We parse the CLI arguments and build the SSOT (Single Source of Truth)
    args         = parse_args()
    cfg          = Config.from_args(args)
    
    # Using RootOrchestrator as a Context Manager to handle lifecycle, 
    # resource guarding (locks), and automatic cleanup.
    with RootOrchestrator(cfg) as orchestrator:
        
        # Access the synchronized services provided by the orchestrator
        paths        = orchestrator.paths
        run_logger   = orchestrator.run_logger
        device       = orchestrator.get_device()
        
        # Retrieve dataset metadata from registry using the validated slug
        # Note: cfg.dataset.dataset_name is now a property resolved via metadata
        ds_meta      = DATASET_REGISTRY[cfg.dataset.metadata.name.lower()]

        try:
            # --- 2. Data Preparation ---
            run_logger.info(
                f"\n{'━' * 80}\n{' DATA PREPARATION ':^80}\n{'━' * 80}"
            )
            
            # Loading data based on metadata and creating DataLoader instances
            data    = load_medmnist(ds_meta)
            loaders = get_dataloaders(data, cfg)
            train_loader, val_loader, test_loader = loaders
            
            # Visual diagnostic: save sample images to verify augmentations/normalization
            show_sample_images(
                loader    = train_loader,
                classes   = ds_meta.classes,
                save_path = paths.get_fig_path("dataset_samples.png"),
                cfg       = cfg
            )

            # --- 3. Model & Training Execution ---
            pipeline_title = f" STARTING PIPELINE: {cfg.model.name.upper()} "
            run_logger.info(
                f"\n{'#' * 80}\n{pipeline_title:^80}\n{'#' * 80}"
            )

            # Factory-based model initialization
            model   = get_model(device=device, cfg=cfg)

            # Optimization components
            criterion = get_criterion(cfg)
            optimizer = get_optimizer(model, cfg)
            scheduler = get_scheduler(optimizer, cfg)

            # The Trainer encapsulates the training loop logic and validation
            trainer = ModelTrainer(
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                optimizer    = optimizer,
                scheduler    = scheduler,
                criterion    = criterion,
                device       = device,
                cfg          = cfg,
                output_path  = paths.best_model_path
            )
            
            # Start training and capture history for final plotting
            best_path, train_losses, val_accuracies = trainer.train()

            # --- 4. Model Recovery & Evaluation ---
            run_logger.info(
                f"\n{'━' * 80}\n{' FINAL EVALUATION PHASE ':^80}\n{'━' * 80}"
            )
            
            # Recover the best weights (determined by validation) for final testing
            orchestrator.load_weights(model, paths.best_model_path)
            
            # Execute comprehensive testing (including TTA if enabled)
            macro_f1, test_acc = run_final_evaluation(
                model          = model,
                test_loader    = test_loader,
                train_losses   = train_losses,
                val_accuracies = val_accuracies,
                class_names    = ds_meta.classes,
                paths          = paths,
                cfg            = cfg,
                aug_info       = get_augmentations_description(cfg),
                log_path       = paths.logs / "session.log"
            )

            # --- 5. Structured Summary Logging ---
            summary = (
                f"\n{'#'*80}\n"
                f"{' PIPELINE EXECUTION SUMMARY ':^80}\n"
                f"{'━'*80}\n"
                f"  » Dataset:      {cfg.dataset.dataset_name}\n"
                f"  » Architecture: {cfg.model.name}\n"
                f"  » Test Acc:     {test_acc:>8.2%}\n"
                f"  » Macro F1:     {macro_f1:>8.4f}\n"
                f"  » Device:       {orchestrator.get_device()}\n"
                f"  » Artifacts:    {paths.root}\n"
                f"{'#'*80}"
            )
            run_logger.info(summary)
        
        except KeyboardInterrupt:
            run_logger.warning("\n[!] Interrupted by user. Cleaning up and exiting...")
        except Exception as e:
            run_logger.error(f"\n[!] Pipeline crashed during execution: {e}", exc_info=True)
            raise e
            
        finally:
            # The Context Manager (__exit__) handles orchestrator.cleanup() automatically.
            # This releases the infrastructure lock and closes logging handlers.
            if 'paths' in locals() and paths:
                run_logger.info(f"Pipeline Shutdown completed. Run directory: {paths.root}")


# =========================================================================== #
#                               ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    main()