"""
Smoke Test Module for MedMNIST Pipeline

This script performs a rapid, end-to-end execution of the training and 
evaluation pipeline. It uses a minimal subset of data and a single epoch 
to verify:
1. Model initialization and forward/backward passes.
2. Checkpoint saving and loading.
3. Visualization utility compatibility (specifically training curves and matrices).
4. Reporting and directory structure integrity.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import torch
import argparse

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from scripts.core import (
    Config, Logger, set_seed, DATASET_REGISTRY, RunPaths, 
    setup_static_directories
)
from scripts.data_handler import (
    load_medmnist, get_dataloaders, get_augmentations_description
)
from scripts.models import get_model
from scripts.trainer import ModelTrainer
from scripts.evaluation import run_final_evaluation

# =========================================================================== #
#                               SMOKE TEST EXECUTION                          #
# =========================================================================== #

def run_smoke_test(args: argparse.Namespace) -> None:
    """
    Orchestrates a lightweight version of the main pipeline to ensure 
    code stability and prevent regression bugs.
    """
    # Setup Config
    # Create Config from CLI args using your Pydantic factory
    cfg = Config.from_args(args)

    # Use model_copy to override values because the Config class is 'frozen=True'
    cfg = cfg.model_copy(update={
        "num_workers": 0,
        "training": cfg.training.model_copy(update={
            "epochs": 1,
            "batch_size": 4,
        }),
        "dataset": cfg.dataset.model_copy(update={
            "max_samples": 16,
            "use_weighted_sampler": False
        })
    })
    
    dataset_key = args.dataset.lower()
    ds_meta = DATASET_REGISTRY[dataset_key]

    # Environment Initialization
    setup_static_directories()

    # Define execution paths
    paths = RunPaths(f"SMOKE_TEST_{cfg.model_name}", cfg.dataset.dataset_name)

    # Setup Logger
    Logger.setup(
        name=paths.project_id,
        log_dir=paths.logs
    )
    run_logger = logging.getLogger(paths.project_id)
    
    header_text = f" INITIALIZING SMOKE TEST: {cfg.dataset.dataset_name.upper()} "
    divider = "=" * max(60, len(header_text))
    
    run_logger.info(divider)
    run_logger.info(header_text.center(len(divider), " "))
    run_logger.info(divider)
    
    run_logger.info(f"Environment verified. Working directory: {paths.root}")

    set_seed(cfg.training.seed)
    device = torch.device("cpu")
    
    run_logger.info("Starting Smoke Test: Environment verified.")

    # Data Loading (Lazy Metadata)
    data = load_medmnist(ds_meta)
    run_logger.info(f"Generating DataLoaders with max_samples={cfg.dataset.max_samples}...")
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

    # Model Factory Check
    model = get_model(device=device, cfg=cfg)
    run_logger.info(f"Model {cfg.model_name} instantiated on {device}.")

    # Training Loop Execution
    run_logger.info("Executing training epoch...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        output_dir=paths.models
    )
    
    # The trainer returns the exact path where it saved the checkpoint
    best_path, train_losses, val_accuracies = trainer.train()

    # Final Evaluation & Visualization Verification
    run_logger.info("Running final evaluation and reporting...")

    # Verification: Ensure the file was actually written before loading
    if not best_path.exists():
        run_logger.error(f"Checkpoint missing! Expected at: {best_path}")
        raise FileNotFoundError(
            f"Checkpoint not found in: {best_path}"
        )
    
    # Load the best weights using the path provided by the trainer
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    run_logger.info(f"Successfully loaded checkpoint from {best_path.name}")
    
    aug_info = get_augmentations_description(cfg)

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

    run_logger.info(f"SMOKE TEST PASSED: Acc {test_acc:.4f} | F1 {macro_f1:.4f}")
    run_logger.info(f"\nSmoke test completed. Check outputs in: {paths.root}\n")


# =========================================================================== #
#                               ENTRY POINT                                   #
# =========================================================================== #

if __name__ == "__main__":
    from scripts.core import parse_args
    cli_args = parse_args()
    try:
        run_smoke_test(args=cli_args)
    except Exception as e:
        # Fallback to basic logging if run_logger initialization fails
        logging.error(f"SMOKE TEST FAILED: {str(e)}", exc_info=True)
        raise