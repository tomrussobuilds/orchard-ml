"""
Pipeline Phase Functions.

Reusable functions for each phase of the ML lifecycle, designed to work
with a shared RootOrchestrator for unified artifact management.

Phases:
    1. Optimization: Optuna hyperparameter search
    2. Training: Model training with validation and checkpointing
    3. Export: ONNX model export with validation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import optuna
import torch
import torch.nn as nn

from orchard.core import (
    LOGGER_NAME,
    Config,
    DatasetRegistryWrapper,
    LogStyle,
    log_optimization_summary,
)

if TYPE_CHECKING:  # pragma: no cover
    from orchard.core import RootOrchestrator

from orchard.data_handler import (
    get_augmentations_description,
    get_dataloaders,
    load_dataset,
    show_samples_for_dataset,
)
from orchard.evaluation import run_final_evaluation
from orchard.export import export_to_onnx
from orchard.models import get_model
from orchard.optimization import run_optimization
from orchard.optimization.orchestrator import export_best_config
from orchard.trainer import ModelTrainer, get_criterion, get_optimizer, get_scheduler

logger = logging.getLogger(LOGGER_NAME)


def run_optimization_phase(
    orchestrator: RootOrchestrator,
    cfg: Config | None = None,
) -> Tuple[optuna.Study, Path | None]:
    """
    Execute hyperparameter optimization phase.

    Runs Optuna study with configured trials, pruning, and early stopping.
    Generates visualizations (if enabled) and exports best configuration.

    Args:
        orchestrator: Active RootOrchestrator providing paths, device, logger
        cfg: Optional config override (defaults to orchestrator's config)

    Returns:
        Tuple of (completed study, path to best_config.yaml or None)

    Example:
        >>> with RootOrchestrator(cfg) as orch:
        ...     study, best_config_path = run_optimization_phase(orch)
        ...     print(f"Best AUC: {study.best_value:.4f}")
    """
    cfg = cfg or orchestrator.cfg
    paths = orchestrator.paths
    device = orchestrator.get_device()
    run_logger = orchestrator.run_logger

    # Type guards for MyPy
    assert run_logger is not None, "Logger not initialized"
    assert paths is not None, "Paths not initialized"

    run_logger.info("")
    run_logger.info(LogStyle.DOUBLE)
    run_logger.info(f"{'HYPERPARAMETER OPTIMIZATION':^80}")
    run_logger.info(LogStyle.DOUBLE)

    # Execute Optuna study
    study = run_optimization(cfg=cfg, device=device, paths=paths)

    # Export best config for subsequent training
    best_config_path = export_best_config(study, cfg, paths)

    log_optimization_summary(
        study=study,
        cfg=cfg,
        device=device,
        paths=paths,
    )

    if best_config_path:
        run_logger.info(f"Best config exported to: {best_config_path}")

    return study, best_config_path


def run_training_phase(
    orchestrator: RootOrchestrator,
    cfg: Config | None = None,
) -> Tuple[Path, List[float], List[dict], nn.Module, float, float]:
    """
    Execute model training phase.

    Loads dataset, initializes model, runs training with validation,
    and performs final evaluation on test set.

    Args:
        orchestrator: Active RootOrchestrator providing paths, device, logger
        cfg: Optional config override (defaults to orchestrator's config)

    Returns:
        Tuple of (best_model_path, train_losses, val_metrics, model, macro_f1, test_acc)

    Example:
        >>> with RootOrchestrator(cfg) as orch:
        ...     best_path, losses, metrics, model, f1, acc = run_training_phase(orch)
        ...     print(f"Test Accuracy: {acc:.4f}")
    """
    cfg = cfg or orchestrator.cfg
    paths = orchestrator.paths
    device = orchestrator.get_device()
    run_logger = orchestrator.run_logger

    # Type guards for MyPy
    assert run_logger is not None, "Logger not initialized"
    assert paths is not None, "Paths not initialized"

    # Dataset metadata
    wrapper = DatasetRegistryWrapper(resolution=cfg.dataset.resolution)
    ds_meta = wrapper.get_dataset(cfg.dataset.dataset_name.lower())

    # DATA PREPARATION
    run_logger.info("")
    run_logger.info(LogStyle.HEAVY)
    run_logger.info(f"{'DATA PREPARATION':^80}")
    run_logger.info(LogStyle.HEAVY)

    data = load_dataset(ds_meta)
    loaders = get_dataloaders(data, cfg)
    train_loader, val_loader, test_loader = loaders

    show_samples_for_dataset(
        loader=train_loader,
        classes=ds_meta.classes,
        dataset_name=cfg.dataset.dataset_name,
        run_paths=paths,
        num_samples=cfg.evaluation.n_samples,
        resolution=cfg.dataset.resolution,
        cfg=cfg,
    )

    # MODEL TRAINING
    run_logger.info("")
    run_logger.info(LogStyle.DOUBLE)
    run_logger.info(f"{'TRAINING PIPELINE - ' + cfg.model.name.upper():^80}")
    run_logger.info(LogStyle.DOUBLE)

    model = get_model(device=device, cfg=cfg)
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
        output_path=paths.best_model_path,
    )

    best_model_path, train_losses, val_metrics_history = trainer.train()

    # FINAL EVALUATION
    run_logger.info("")
    run_logger.info(LogStyle.HEAVY)
    run_logger.info(f"{'FINAL EVALUATION':^80}")
    run_logger.info(LogStyle.HEAVY)

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

    return best_model_path, train_losses, val_metrics_history, model, macro_f1, test_acc


def run_export_phase(
    orchestrator: RootOrchestrator,
    checkpoint_path: Path,
    cfg: Config | None = None,
    export_format: str = "onnx",
    opset_version: int = 18,
) -> Path | None:
    """
    Execute model export phase.

    Exports trained model to production format (ONNX) with validation.

    Args:
        orchestrator: Active RootOrchestrator providing paths, device, logger
        checkpoint_path: Path to trained model checkpoint (.pth)
        cfg: Optional config override (defaults to orchestrator's config)
        export_format: Export format ("onnx" or "none")
        opset_version: ONNX opset version (default: 18)

    Returns:
        Path to exported model, or None if export_format is "none"

    Example:
        >>> with RootOrchestrator(cfg) as orch:
        ...     best_path, *_ = run_training_phase(orch)
        ...     onnx_path = run_export_phase(orch, best_path)
        ...     print(f"Exported to: {onnx_path}")
    """
    if export_format == "none":
        return None

    cfg = cfg or orchestrator.cfg
    paths = orchestrator.paths
    run_logger = orchestrator.run_logger

    # Type guards for MyPy
    assert run_logger is not None, "Logger not initialized"
    assert paths is not None, "Paths not initialized"

    run_logger.info("")
    run_logger.info(LogStyle.HEAVY)
    run_logger.info(f"{'MODEL EXPORT':^80}")
    run_logger.info(LogStyle.HEAVY)

    # Determine input shape from config
    resolution = cfg.dataset.resolution
    in_channels = 3 if cfg.dataset.force_rgb else 1
    input_shape = (in_channels, resolution, resolution)

    # Export path (directory managed by RunPaths)
    onnx_path = paths.exports / "model.onnx"

    # Reload model architecture (on CPU for export)
    export_model = get_model(device=torch.device("cpu"), cfg=cfg)

    export_to_onnx(
        model=export_model,
        checkpoint_path=checkpoint_path,
        output_path=onnx_path,
        input_shape=input_shape,
        opset_version=opset_version,
        dynamic_axes=True,
        validate=True,
    )

    run_logger.info(f"Model exported to: {onnx_path}")

    return onnx_path
