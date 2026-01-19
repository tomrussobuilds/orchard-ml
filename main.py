"""
Training Pipeline Entry Point for VisionForge.

Orchestrates end-to-end classification experiments with complete lifecycle
management: environment setup, data loading, model training, evaluation,
and artifact generation.

Workflow:
    1. Configuration parsing (CLI/YAML)
    2. Environment initialization via RootOrchestrator
    3. Data loading with metadata-driven validation
    4. Model training with AMP, MixUp, and checkpointing
    5. Final evaluation with Test-Time Augmentation
    6. Report generation (Excel, plots, logs)

Usage:
    # Train with YAML recipe (recommended)
    python main.py --config recipes/config_resnet_18_adapted.yaml

    # Quick test with CLI overrides
    python main.py --dataset bloodmnist --epochs 10 --batch_size 64

    # High-resolution training (GPU required)
    python main.py --config recipes/config_vit_tiny.yaml

Key Features:
    - Type-safe configuration via Pydantic V2
    - Automatic GPU detection and fallback
    - Kernel-level file locking for cluster safety
    - Deterministic run isolation (BLAKE2b hashing)
    - Comprehensive artifact suite (models, plots, reports)
"""

# =========================================================================== #
#                            INTERNAL IMPORTS                                 #
# =========================================================================== #
from orchard.core import (
    DATASET_REGISTRY,
    Config,
    LogStyle,
    RootOrchestrator,
    log_training_summary,
    parse_args,
)
from orchard.data_handler import (
    get_augmentations_description,
    get_dataloaders,
    load_medmnist,
    show_samples_for_dataset,
)
from orchard.evaluation import run_final_evaluation
from orchard.models import get_model
from orchard.trainer import ModelTrainer, get_criterion, get_optimizer, get_scheduler

# =========================================================================== #
#                           MAIN EXECUTION                                    #
# =========================================================================== #


def main() -> None:
    """
    Main orchestrator for training pipeline execution.

    Coordinates the complete training lifecycle from configuration parsing
    to final evaluation. Utilizes RootOrchestrator context manager for
    resource safety and automatic cleanup.

    Workflow:
        1. Parse CLI arguments (YAML config or direct flags)
        2. Build unified Config with Pydantic validation
        3. Initialize orchestrator (device, filesystem, logging)
        4. Load dataset with metadata injection
        5. Train model with validation checkpointing
        6. Evaluate on test set with optional TTA
        7. Generate comprehensive reports and artifacts

    Raises:
        KeyboardInterrupt: User interrupted training (graceful cleanup)
        Exception: Any fatal error during pipeline (logged and re-raised)
    """
    # Parse CLI arguments (supports both YAML and direct flags)
    args = parse_args()

    # Build configuration (triggers Pydantic validation)
    cfg = Config.from_args(args)

    # Use orchestrator context manager for resource safety
    # Guarantees cleanup even if pipeline crashes
    with RootOrchestrator(cfg) as orchestrator:

        # Access synchronized services provided by orchestrator
        paths = orchestrator.paths
        run_logger = orchestrator.run_logger
        device = orchestrator.get_device()

        # Retrieve dataset metadata from registry
        ds_meta = DATASET_REGISTRY[cfg.dataset.metadata.name.lower()]

        try:
            # ================================================================ #
            #                       DATA PREPARATION                           #
            # ================================================================ #
            run_logger.info("")
            run_logger.info(LogStyle.HEAVY)
            run_logger.info(f"{'DATA PREPARATION':^80}")
            run_logger.info(LogStyle.HEAVY)

            # Load dataset and create DataLoader instances
            data = load_medmnist(ds_meta)
            loaders = get_dataloaders(data, cfg)
            train_loader, val_loader, test_loader = loaders

            # Visual diagnostic: save augmentation samples
            show_samples_for_dataset(
                loader=train_loader,
                classes=ds_meta.classes,
                dataset_name=cfg.dataset.dataset_name,
                run_paths=paths,
                num_samples=cfg.evaluation.n_samples,
                resolution=cfg.dataset.resolution,
            )

            # ================================================================ #
            #                     MODEL TRAINING                               #
            # ================================================================ #
            run_logger.info("")
            run_logger.info(LogStyle.DOUBLE)
            run_logger.info(f"{'TRAINING PIPELINE: ' + cfg.model.name.upper():^80}")
            run_logger.info(LogStyle.DOUBLE)

            # Initialize model and optimization components
            model = get_model(device=device, cfg=cfg)
            criterion = get_criterion(cfg)
            optimizer = get_optimizer(model, cfg)
            scheduler = get_scheduler(optimizer, cfg)

            # Execute training loop
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=device,
                cfg=cfg,
                output_path=paths.best_model_path,
            )

            _, train_losses, val_metrics_history = trainer.train()

            # ================================================================ #
            #                     FINAL EVALUATION                             #
            # ================================================================ #
            run_logger.info("")
            run_logger.info(LogStyle.HEAVY)
            run_logger.info(f"{'FINAL EVALUATION PHASE':^80}")
            run_logger.info(LogStyle.HEAVY)

            # Execute comprehensive testing (with optional TTA)
            macro_f1, test_acc = run_final_evaluation(
                model=model,
                test_loader=test_loader,
                train_losses=train_losses,
                val_metrics_history=val_metrics_history,
                class_names=ds_meta.classes,
                paths=paths,
                cfg=cfg,
                aug_info=get_augmentations_description(cfg),
                log_path=paths.logs / "session.log",
            )

            # ================================================================ #
            #                     PIPELINE SUMMARY                             #
            # ================================================================ #
            log_training_summary(
                cfg=cfg,
                test_acc=test_acc,
                macro_f1=macro_f1,
                device=device,
                paths=paths,
                logger_instance=run_logger,
            )

        except KeyboardInterrupt:
            run_logger.warning(
                f"\n{LogStyle.WARNING} Interrupted by user. Cleaning up and exiting..."
            )

        except Exception as e:
            run_logger.error(f"\n{LogStyle.WARNING} Pipeline crashed: {e}", exc_info=True)
            raise

        finally:
            # Context manager handles automatic cleanup
            if "paths" in locals() and paths:
                run_logger.info(f"Pipeline shutdown complete. Run directory: {paths.root}")


# =========================================================================== #
#                           ENTRY POINT                                       #
# =========================================================================== #

if __name__ == "__main__":
    main()
