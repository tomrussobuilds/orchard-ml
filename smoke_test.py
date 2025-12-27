"""
Smoke Test Module for MedMNIST Pipeline

This script performs a rapid, end-to-end execution of the training and 
evaluation pipeline. It uses a minimal subset of data and a single epoch 
to verify the 6 core pillars of the system:
1. Model initialization and forward/backward passes (ResNet-18 Adaptation).
2. Checkpoint saving and loading (Atomic Weight Persistence).
3. Visualization utility compatibility (Training curves and Confusion Matrices).
4. Reporting and directory structure integrity (Excel & Path Orchestration).
5. Data Flow integrity (Torchvision V2 Transforms & RGB/Gray handling).
6. System Safeguards (Kernel-level locking and Hardware abstraction).
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import argparse

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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
            "use_amp": False,
            "grad_clip": 1.0
        }),
        "dataset": base_cfg.dataset.model_copy(update={
            "max_samples": 16,
            "use_weighted_sampler": False
        })
    })

    # 2. Root Orchestration via Context Manager
    # The RootOrchestrator handles directory creation, logging, and hardware abstraction.
    # The context manager ensures that the system lock is released even if the test fails.
    with RootOrchestrator(cfg) as orchestrator:
        paths = orchestrator.paths
        run_logger = orchestrator.run_logger
        device = orchestrator.get_device()

        header_text = f" INITIALIZING SMOKE TEST: {cfg.dataset.dataset_name.upper()} "
        divider = "=" * max(60, len(header_text))
        
        run_logger.info(divider)
        run_logger.info(header_text.center(len(divider), " "))
        run_logger.info(divider)

        run_logger.info(f"Smoke test execution started on {device}.")

        try:
            # 3. Data Loading (Lazy Metadata)
            ds_meta = DATASET_REGISTRY[cfg.dataset.dataset_name.lower()]
            data = load_medmnist(ds_meta)
            run_logger.info(f"Generating DataLoaders with max_samples={cfg.dataset.max_samples}...")
            train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

            # 4. Model Factory Check
            model = get_model(device=device, cfg=cfg)
            run_logger.info(f"Model {cfg.model_name} instantiated.")

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

            # 5. Training Loop Execution
            run_logger.info("Executing training epoch...")
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
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
            
            orchestrator.load_weights(model, best_path)
            
            aug_info = get_augmentations_description(cfg)

            macro_f1, test_acc = run_final_evaluation(
                model=model,
                test_loader=test_loader,
                class_names=ds_meta.classes,
                train_losses=train_losses,
                val_accuracies=val_accuracies,
                paths=paths,
                cfg=cfg,
                aug_info=aug_info
            )

            run_logger.info(f"SMOKE TEST PASSED: Acc {test_acc:.4f} | F1 {macro_f1:.4f}")
            run_logger.info(f"\nSmoke test completed. Check outputs in: {paths.root}\n")

        except Exception as e:
            run_logger.error(f"Smoke test pipeline failed: {str(e)}", exc_info=True)
            raise e

# =========================================================================== #
#                               ENTRY POINT                                   #
# =========================================================================== #

if __name__ == "__main__":
    cli_args = parse_args()
    run_smoke_test(args=cli_args)