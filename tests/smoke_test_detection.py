"""
Smoke Test Module for Orchard ML Detection Pipeline.

Performs rapid end-to-end validation of the detection training pipeline
with synthetic data and minimal computational overhead. Verifies:
1. Environment initialization via RootOrchestrator
2. Synthetic detection data loading with collate
3. FasterRCNN model training via task registry adapters
4. Weight recovery and inference
5. Detection evaluation (mAP metrics)

Usage:
    python -m tests.smoke_test_detection

Expected Runtime: ~2 minutes on GPU, ~5 minutes on CPU
"""

from __future__ import annotations

from orchard.architectures import get_model
from orchard.core import Config, LogStyle, RootOrchestrator
from orchard.core.metadata.wrapper import DetectionRegistryWrapper
from orchard.core.task_registry import get_task
from orchard.data_handler import get_dataloaders
from orchard.data_handler.diagnostic import create_synthetic_detection_dataset
from orchard.data_handler.dispatcher import DatasetData
from orchard.trainer import ModelTrainer, get_optimizer, get_scheduler


def _build_detection_smoke_config() -> Config:
    """Build a minimal Config for detection smoke testing."""
    wrapper = DetectionRegistryWrapper(resolution=224)
    metadata = wrapper.get_dataset("pennfudan")

    return Config(
        task_type="detection",
        dataset={
            "name": "pennfudan",
            "resolution": 224,
            "max_samples": 20,
            "metadata": metadata,
        },
        architecture={"name": "fasterrcnn", "pretrained": False},
        training={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.005,
            "optimizer_type": "sgd",
            "use_amp": False,
            "mixup_alpha": 0.0,
            "mixup_epochs": 0,
            "monitor_metric": "map",
            "monitor_direction": "maximize",
        },
        hardware={"device": "cpu", "project_name": "smoke-test-detection", "reproducible": True},
        telemetry={"output_dir": "./outputs"},
        augmentation={
            "hflip": 0.0,
            "rotation_angle": 0,
        },
    )


def run_detection_smoke_test(cfg: Config) -> None:
    """
    Orchestrates lightweight detection pipeline validation.

    Args:
        cfg: Pre-built detection Config with minimal resource requirements.

    Raises:
        Exception: Any failure during pipeline execution.
    """
    with RootOrchestrator(cfg) as orchestrator:
        paths = orchestrator.paths
        run_logger = orchestrator.run_logger
        device = orchestrator.get_device()

        run_logger.info("")  # type: ignore
        run_logger.info(LogStyle.HEAVY)  # type: ignore
        run_logger.info(f"{'DETECTION SMOKE TEST':^80}")  # type: ignore
        run_logger.info(LogStyle.HEAVY)  # type: ignore

        try:
            # SYNTHETIC DATA GENERATION
            run_logger.info("[Stage 1/5] Generating synthetic detection data...")  # type: ignore
            synth = create_synthetic_detection_dataset(
                num_classes=1,
                samples=20,
                resolution=224,
                channels=3,
                name="synthetic_detection",
            )
            data = DatasetData(
                path=synth.image_path,
                name=synth.name,
                is_rgb=True,
                num_classes=synth.num_classes,
                annotation_path=synth.annotation_path,
            )

            # DATALOADERS
            run_logger.info("[Stage 2/5] Initializing detection DataLoaders...")  # type: ignore
            train_loader, val_loader, test_loader = get_dataloaders(
                data,
                cfg.dataset,
                cfg.training,
                cfg.augmentation,
                cfg.num_workers,
                task_type="detection",
            )

            # MODEL + TASK ADAPTERS
            run_logger.info("[Stage 3/5] Testing FasterRCNN & task adapters...")  # type: ignore
            model = get_model(device=device, dataset_cfg=cfg.dataset, arch_cfg=cfg.architecture)
            task = get_task("detection")
            criterion = task.criterion_factory.get_criterion(cfg.training)
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
                training=cfg.training,
                output_path=paths.best_model_path,  # type: ignore
                training_step=task.training_step,
                validation_metrics=task.validation_metrics,
            )

            _, train_losses, val_metrics_history = trainer.train()

            # WEIGHT RECOVERY
            run_logger.info("[Stage 4/5] Recovering weights and verifying checkpoint...")  # type: ignore
            if not paths.best_model_path.exists():  # type: ignore
                raise FileNotFoundError(f"Checkpoint not found: {paths.best_model_path}")  # type: ignore

            trainer.load_best_weights()

            # DETECTION EVALUATION
            run_logger.info("[Stage 5/5] Running detection evaluation (mAP)...")  # type: ignore
            test_metrics = task.eval_pipeline.run_evaluation(
                model=model,
                test_loader=test_loader,
                train_losses=train_losses,
                val_metrics_history=val_metrics_history,
                class_names=["person", "object"],
                paths=paths,  # type: ignore
                training=cfg.training,
                dataset=cfg.dataset,
                augmentation=cfg.augmentation,
                evaluation=cfg.evaluation,
                arch_name=cfg.architecture.name,
            )

            # VERIFY METRICS
            assert "map" in test_metrics, "mAP metric missing from test results"
            assert "map_50" in test_metrics, "mAP@50 metric missing from test results"
            assert "map_75" in test_metrics, "mAP@75 metric missing from test results"

            # TEST SUMMARY
            run_logger.info("")  # type: ignore
            run_logger.info(LogStyle.DOUBLE)  # type: ignore
            run_logger.info(f"{'DETECTION SMOKE TEST PASSED':^80}")  # type: ignore
            run_logger.info(LogStyle.DOUBLE)  # type: ignore
            run_logger.info(  # type: ignore
                f"{LogStyle.INDENT}{LogStyle.SUCCESS} mAP     : {test_metrics['map']:.4f}"
            )
            run_logger.info(  # type: ignore
                f"{LogStyle.INDENT}{LogStyle.ARROW} mAP@50  : {test_metrics['map_50']:.4f}"
            )
            run_logger.info(  # type: ignore
                f"{LogStyle.INDENT}{LogStyle.ARROW} mAP@75  : {test_metrics['map_75']:.4f}"
            )
            run_logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Device  : {device}")  # type: ignore
            run_logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Artifacts: {paths.root}")  # type: ignore
            run_logger.info(LogStyle.DOUBLE)  # type: ignore
            run_logger.info("")  # type: ignore

        except Exception as e:
            run_logger.error(  # type: ignore
                f"\n{LogStyle.WARNING} DETECTION SMOKE TEST FAILED: {str(e)}", exc_info=True
            )
            raise


# ENTRY POINT
if __name__ == "__main__":
    cfg = _build_detection_smoke_config()
    run_detection_smoke_test(cfg)
