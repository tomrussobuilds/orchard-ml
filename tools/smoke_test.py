"""
Smoke Test Module for MedMNIST Pipeline

This script performs a rapid, end-to-end execution of the training and 
evaluation pipeline. It verifies the 5-stage orchestration logic:
1. Environment Initialization: Setup via RootOrchestrator and hardware abstraction.
2. Data Preparation: Metadata loading and lightweight DataLoader creation.
3. Training Execution: Verification of Factory-based optimization and training loops.
4. Model Recovery & Testing: Weights restoration and TTA-enabled inference.
5. Reporting & Summary: Diagnostic visualization and Excel report generation.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import (
    Config, parse_args, DATASET_REGISTRY, RootOrchestrator
)
from src.data_handler import (
    load_medmnist, get_dataloaders, get_augmentations_description
)
from src.models import get_model
from src.trainer import (
    ModelTrainer, get_criterion, get_optimizer, get_scheduler
)
from src.evaluation import run_final_evaluation

# =========================================================================== #
#                               SMOKE TEST EXECUTION                          #
# =========================================================================== #

def run_smoke_test(args: argparse.Namespace) -> None:
    """
    Orchestrates a lightweight version of the main pipeline to ensure 
    code stability and prevent regression bugs.
    """
    # 1. Configuration Setup & Override
    # Create Config and force minimal parameters for rapid execution
    base_cfg = Config.from_args(args)
    
    cfg = base_cfg.model_copy(update={
        "num_workers": 0,
        "training": base_cfg.training.model_copy(update={
            "epochs": 1,           # Minimal epochs
            "batch_size": 4,       # Small batch
            "use_amp": False,      # Disable AMP for stability in test
        }),
        "dataset": base_cfg.dataset.model_copy(update={
            "max_samples": 32,     # Minimal data subset
        })
    })

    # --- Stage 1: Environment Initialization ---
    with RootOrchestrator(cfg) as orchestrator:
        paths = orchestrator.paths
        run_logger = orchestrator.run_logger
        device = orchestrator.get_device()
        
        # Resolve dataset metadata
        ds_meta = DATASET_REGISTRY[cfg.dataset.metadata.name.lower()]

        run_logger.info(f"\n{'━'*60}\n{' RUNNING SMOKE TEST: ' + ds_meta.name.upper():^60}\n{'━'*60}")

        try:
            # --- Stage 2: Data Preparation ---
            run_logger.info("Stage 2: Initializing DataHolders...")
            data = load_medmnist(ds_meta)
            train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

            # --- Stage 3: Training Execution ---
            run_logger.info("Stage 3: Testing Model & Optimizer Factories...")
            model = get_model(device=device, cfg=cfg)

            # Using official project factories instead of manual instantiation
            criterion = get_criterion(cfg)
            optimizer = get_optimizer(model, cfg)
            scheduler = get_scheduler(optimizer, cfg)

            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=device,
                cfg=cfg,
                output_path=paths.best_model_path
            )
            
            # Capture history (List[dict] for val_metrics)
            _, train_losses, val_metrics_history = trainer.train()

            # --- Stage 4: Model Recovery & Evaluation ---
            run_logger.info("Stage 4: Recovering weights and running inference...")
            
            if not paths.best_model_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {paths.best_model_path}")
            
            orchestrator.load_weights(model, paths.best_model_path)
            
            # --- Stage 5: Reporting & Summary ---
            run_logger.info("Stage 5: Verifying Reporting & Visualization utilities...")
            
            _, test_acc = run_final_evaluation(
                model=model,
                test_loader=test_loader,
                train_losses=train_losses,
                val_metrics_history=val_metrics_history,
                class_names=ds_meta.classes,
                paths=paths,
                cfg=cfg,
                aug_info=get_augmentations_description(cfg),
                log_path=paths.logs / "smoke_test.log"
            )

            # Final Summary Block
            run_logger.info(f"\n{'#'*60}\n{' SMOKE TEST PASSED SUCCESSFULLY ':^60}\n{'#'*60}")
            run_logger.info(f"Final Test Accuracy: {test_acc:.4f}")
            run_logger.info(f"Artifacts preserved in: {paths.root}")

        except Exception as e:
            run_logger.error(f"SMOKE TEST CRITICAL FAILURE: {str(e)}", exc_info=True)
            raise e

# =========================================================================== #
#                               ENTRY POINT                                   #
# =========================================================================== #

if __name__ == "__main__":
    cli_args = parse_args()
    run_smoke_test(args=cli_args)