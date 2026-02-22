"""
Smoke Test Module for Orchard ML Pipeline.

Performs rapid end-to-end validation of the complete training pipeline
with minimal computational overhead. Verifies all 5 orchestration stages:
1. Environment initialization via RootOrchestrator
2. Data loading with metadata injection
3. Model training with factory pattern
4. Weight recovery and inference
5. Report generation and artifact export

Usage:
    python -m tests.smoke_test
    python -m tests.smoke_test --dataset bloodmnist --architecture mini_cnn

Expected Runtime: ~30 seconds on GPU, ~2 minutes on CPU
"""

from __future__ import annotations

import argparse
import os

from orchard.core import Config, DatasetRegistryWrapper, LogStyle, RootOrchestrator
from orchard.data_handler import get_augmentations_description, get_dataloaders, load_dataset
from orchard.evaluation import run_final_evaluation
from orchard.models import get_model
from orchard.trainer import ModelTrainer, get_criterion, get_optimizer, get_scheduler


def _build_smoke_config(dataset: str = "bloodmnist", architecture: str = "mini_cnn") -> Config:
    """Build a minimal Config for smoke testing (no YAML, no from_args)."""
    return Config(
        dataset={"name": dataset, "resolution": 28, "max_samples": 32},
        architecture={"name": architecture, "pretrained": False},
        training={
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "use_amp": False,
            "mixup_alpha": 0.0,
            "mixup_epochs": 0,
        },
        hardware={"device": "cpu", "project_name": "smoke-test", "reproducible": True},
        telemetry={"data_dir": "./dataset", "output_dir": "./outputs"},
    )


# SMOKE TEST EXECUTION
def run_smoke_test(cfg: Config) -> None:
    """
    Orchestrates lightweight pipeline validation for regression testing.

    Args:
        cfg: Pre-built Config with minimal resource requirements.

    Raises:
        Exception: Any failure during pipeline execution.
    """

    # PIPELINE VALIDATION
    with RootOrchestrator(cfg) as orchestrator:
        paths = orchestrator.paths
        run_logger = orchestrator.run_logger
        device = orchestrator.get_device()

        # Resolve dataset metadata with resolution-specific registry
        wrapper = DatasetRegistryWrapper(resolution=cfg.dataset.resolution)
        ds_meta = wrapper.get_dataset(cfg.dataset.metadata.name.lower())

        run_logger.info("")
        run_logger.info(LogStyle.HEAVY)
        run_logger.info(f"{'SMOKE TEST: ' + ds_meta.name.upper():^80}")
        run_logger.info(LogStyle.HEAVY)

        try:
            # DATA PREPARATION
            run_logger.info("[Stage 1/5] Checking environment for CI/synthetic dataset...")
            if os.getenv("CI"):
                from orchard.data_handler.synthetic import create_synthetic_dataset

                data = create_synthetic_dataset()
            else:
                data = load_dataset(ds_meta)

            run_logger.info("[Stage 2/5] Initializing DataLoaders...")
            train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

            # MODEL TRAINING
            run_logger.info("[Stage 3/5] Testing Model & Optimizer Factories...")
            model = get_model(device=device, cfg=cfg)
            criterion = get_criterion(cfg.training)
            optimizer = get_optimizer(model, cfg.training)
            scheduler = get_scheduler(optimizer, cfg.training)

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

            # MODEL EVALUATION
            run_logger.info("[Stage 4/5] Recovering weights and running inference...")

            if not paths.best_model_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {paths.best_model_path}")

            trainer.load_best_weights()

            # REPORT GENERATION
            run_logger.info("[Stage 5/5] Verifying reporting utilities...")

            _, test_acc = run_final_evaluation(
                model=model,
                test_loader=test_loader,
                train_losses=train_losses,
                val_metrics_history=val_metrics_history,
                class_names=ds_meta.classes,
                paths=paths,
                cfg=cfg,
                aug_info=get_augmentations_description(cfg),
                log_path=paths.logs / "smoke_test.log",
            )

            # TEST SUMMARY
            run_logger.info("")
            run_logger.info(LogStyle.DOUBLE)
            run_logger.info(f"{'SMOKE TEST PASSED':^80}")
            run_logger.info(LogStyle.DOUBLE)
            run_logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Test Accuracy : {test_acc:.2%}")
            run_logger.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} Dataset       : {ds_meta.display_name}"
            )
            run_logger.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} Architecture  : {cfg.architecture.name}"
            )
            run_logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Device        : {device}")
            run_logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Artifacts     : {paths.root}")
            run_logger.info(LogStyle.DOUBLE)
            run_logger.info("")

        except Exception as e:
            run_logger.error(f"\n{LogStyle.WARNING} SMOKE TEST FAILED: {str(e)}", exc_info=True)
            raise


# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchard ML Smoke Test")
    parser.add_argument("--dataset", default="bloodmnist", help="Dataset name")
    parser.add_argument("--architecture", default="mini_cnn", help="Architecture name")
    args = parser.parse_args()

    cfg = _build_smoke_config(dataset=args.dataset, architecture=args.architecture)
    run_smoke_test(cfg)
