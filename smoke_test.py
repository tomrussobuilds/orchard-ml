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
from src.core import (
    Config, RootOrchestrator, DATASET_REGISTRY, parse_args
)
from src.data_handler import (
    load_medmnist, get_dataloaders, get_augmentations_description
)
from src.models import get_model
from src.trainer import ModelTrainer
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
    # Create Config from CLI args and force minimal parameters for rapid testing
    base_cfg = Config.from_args(args)
    
    cfg = base_cfg.model_copy(update={
        "num_workers": 0,
        "training": base_cfg.training.model_copy(update={
            "epochs": 1,
            "batch_size": 4,
        }),
        "dataset": base_cfg.dataset.model_copy(update={
            "max_samples": 16,
            "use_weighted_sampler": False
        })
    })

    # 2. Root Orchestration
    # The RootOrchestrator handles directory creation, logging, and safety
    orchestrator = RootOrchestrator(cfg)
    paths = orchestrator.initialize_core_services()
    run_logger = orchestrator.run_logger

    header_text = f" INITIALIZING SMOKE TEST: {cfg.dataset.dataset_name.upper()} "
    divider = "=" * max(60, len(header_text))
    
    run_logger.info(divider)
    run_logger.info(header_text.center(len(divider), " "))
    run_logger.info(divider)
    
    # Force CPU for smoke testing to ensure portability
    device = torch.device("cpu")
    run_logger.info(f"Smoke test execution started on {device}.")

    # 3. Data Loading (Lazy Metadata)
    ds_meta = DATASET_REGISTRY[cfg.dataset.dataset_name.lower()]
    data = load_medmnist(ds_meta)
    run_logger.info(f"Generating DataLoaders with max_samples={cfg.dataset.max_samples}...")
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

    # 4. Model Factory Check
    model = get_model(device=device, cfg=cfg)
    run_logger.info(f"Model {cfg.model_name} instantiated.")

    # 5. Training Loop Execution
    run_logger.info("Executing training epoch...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        output_dir=paths.models
    )
    
    best_path, train_losses, val_accuracies = trainer.train()

    # 6. Final Evaluation & Visualization Verification
    run_logger.info("Running final evaluation and reporting...")

    if not best_path.exists():
        run_logger.error(f"Checkpoint missing! Expected at: {best_path}")
        raise FileNotFoundError(f"Checkpoint not found in: {best_path}")
    
    # Load weights to verify checkpoint integrity
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
    cli_args = parse_args()
    try:
        run_smoke_test(args=cli_args)
    except Exception as e:
        # Emergency logging if the orchestrator/logger fails
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"SMOKE TEST FAILED: {str(e)}", exc_info=True)
        raise