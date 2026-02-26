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
from typing import TYPE_CHECKING, NamedTuple

import optuna
import torch
import torch.nn as nn

from ..core import (
    LOGGER_NAME,
    Config,
    DatasetRegistryWrapper,
    LogStyle,
    log_optimization_summary,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core import RootOrchestrator
    from ..tracking import TrackerProtocol

from ..architectures import get_model
from ..data_handler import (
    get_augmentations_description,
    get_dataloaders,
    load_dataset,
    show_samples_for_dataset,
)
from ..evaluation import run_final_evaluation
from ..export import (
    benchmark_onnx_inference,
    export_to_onnx,
    quantize_model,
    validate_export,
)
from ..optimization import run_optimization
from ..trainer import (
    ModelTrainer,
    compute_class_weights,
    get_criterion,
    get_optimizer,
    get_scheduler,
)

logger = logging.getLogger(LOGGER_NAME)

_ERR_LOGGER_NOT_INIT = "Logger not initialized"
_ERR_PATHS_NOT_INIT = "Paths not initialized"


class TrainingResult(NamedTuple):
    """Structured return type for :func:`run_training_phase`."""

    best_model_path: Path
    train_losses: list[float]
    val_metrics: list[dict]
    model: nn.Module
    macro_f1: float
    test_acc: float
    test_auc: float


def run_optimization_phase(
    orchestrator: RootOrchestrator,
    cfg: Config | None = None,
    tracker: TrackerProtocol | None = None,
) -> tuple[optuna.Study, Path | None]:
    """
    Execute hyperparameter optimization phase.

    Runs Optuna study with configured trials, pruning, and early stopping.
    Generates visualizations (if enabled) and exports best configuration.

    Args:
        orchestrator: Active RootOrchestrator providing paths, device, logger
        cfg: Optional config override (defaults to orchestrator's config)
        tracker: Optional experiment tracker for MLflow nested trial logging

    Returns:
        tuple of (completed study, path to best_config.yaml or None)

    Example:
        >>> with RootOrchestrator(cfg) as orch:
        ...     study, best_config_path = run_optimization_phase(orch)
        ...     print(f"Best AUC: {study.best_value:.4f}")
    """
    cfg = cfg or orchestrator.cfg
    paths = orchestrator.paths
    device = orchestrator.get_device()
    run_logger = orchestrator.run_logger

    # type guards for MyPy
    assert run_logger is not None, _ERR_LOGGER_NOT_INIT  # nosec B101
    assert paths is not None, _ERR_PATHS_NOT_INIT  # nosec B101

    LogStyle.log_phase_header(run_logger, "HYPERPARAMETER OPTIMIZATION", LogStyle.DOUBLE)

    # Execute Optuna study (includes post-processing: visualizations, best config export)
    study = run_optimization(cfg=cfg, device=device, paths=paths, tracker=tracker)

    log_optimization_summary(
        study=study,
        cfg=cfg,
        device=device,
        paths=paths,
    )

    # Best config path is in reports dir (exported by orchestrator if save_best_config=True)
    candidate = paths.reports / "best_config.yaml"
    best_config_path: Path | None = candidate if candidate.exists() else None

    return study, best_config_path


def run_training_phase(
    orchestrator: RootOrchestrator,
    cfg: Config | None = None,
    tracker: TrackerProtocol | None = None,
) -> TrainingResult:
    """
    Execute model training phase.

    Loads dataset, initializes model, runs training with validation,
    and performs final evaluation on test set.

    Args:
        orchestrator: Active RootOrchestrator providing paths, device, logger
        cfg: Optional config override (defaults to orchestrator's config)
        tracker: Optional experiment tracker for MLflow metric logging

    Returns:
        TrainingResult named tuple with best_model_path, train_losses,
        val_metrics, model, macro_f1, test_acc, test_auc.

    Example:
        >>> with RootOrchestrator(cfg) as orch:
        ...     result = run_training_phase(orch)
        ...     print(f"Test Accuracy: {result.test_acc:.4f}")
    """
    cfg = cfg or orchestrator.cfg
    paths = orchestrator.paths
    device = orchestrator.get_device()
    run_logger = orchestrator.run_logger

    # type guards for MyPy
    assert run_logger is not None, _ERR_LOGGER_NOT_INIT  # nosec B101
    assert paths is not None, _ERR_PATHS_NOT_INIT  # nosec B101

    # Dataset metadata
    wrapper = DatasetRegistryWrapper(resolution=cfg.dataset.resolution)
    ds_meta = wrapper.get_dataset(cfg.dataset.dataset_name.lower())

    # DATA PREPARATION
    LogStyle.log_phase_header(run_logger, "DATA PREPARATION")

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
    LogStyle.log_phase_header(
        run_logger, "TRAINING PIPELINE - " + cfg.architecture.name.upper(), LogStyle.DOUBLE
    )

    model = get_model(device=device, cfg=cfg)

    class_weights = None
    if cfg.training.weighted_loss:
        train_labels = train_loader.dataset.labels.flatten()  # type: ignore[attr-defined]
        class_weights = compute_class_weights(train_labels, ds_meta.num_classes, device)

    criterion = get_criterion(cfg.training, class_weights=class_weights)
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
        tracker=tracker,
    )

    best_model_path, train_losses, val_metrics_history = trainer.train()

    # FINAL EVALUATION
    LogStyle.log_phase_header(run_logger, "FINAL EVALUATION")

    macro_f1, test_acc, test_auc = run_final_evaluation(
        model=model,
        test_loader=test_loader,
        train_losses=train_losses,
        val_metrics_history=val_metrics_history,
        class_names=ds_meta.classes,
        paths=paths,
        cfg=cfg,
        aug_info=get_augmentations_description(cfg, ds_meta=ds_meta),
        log_path=paths.logs / "session.log",
        tracker=tracker,
    )

    return TrainingResult(
        best_model_path=best_model_path,
        train_losses=train_losses,
        val_metrics=val_metrics_history,
        model=model,
        macro_f1=macro_f1,
        test_acc=test_acc,
        test_auc=test_auc,
    )


def run_export_phase(
    orchestrator: RootOrchestrator,
    checkpoint_path: Path,
    cfg: Config | None = None,
) -> Path | None:
    """
    Execute model export phase.

    Exports trained model to production format (ONNX) with validation.
    Export format and opset version are read from ``cfg.export``.

    Args:
        orchestrator: Active RootOrchestrator providing paths, device, logger
        checkpoint_path: Path to trained model checkpoint (.pth)
        cfg: Optional config override (defaults to orchestrator's config)

    Returns:
        Path to exported model, or None if export config is absent

    Example:
        >>> with RootOrchestrator(cfg) as orch:
        ...     best_path, *_ = run_training_phase(orch)
        ...     onnx_path = run_export_phase(orch, best_path)
        ...     print(f"Exported to: {onnx_path}")
    """
    cfg = cfg or orchestrator.cfg

    if cfg.export is None:
        return None

    paths = orchestrator.paths
    run_logger = orchestrator.run_logger

    # type guards for MyPy
    assert run_logger is not None, _ERR_LOGGER_NOT_INIT  # nosec B101
    assert paths is not None, _ERR_PATHS_NOT_INIT  # nosec B101

    LogStyle.log_phase_header(run_logger, "MODEL EXPORT")

    # Determine input shape from config (must match get_model's channel resolution)
    resolution = cfg.dataset.resolution
    input_shape = (cfg.dataset.effective_in_channels, resolution, resolution)

    # Export path (directory managed by RunPaths)
    onnx_path = paths.exports / "model.onnx"

    # Reload model architecture (on CPU for export)
    export_model = get_model(device=torch.device("cpu"), cfg=cfg, verbose=False)

    export_cfg = cfg.export  # guaranteed non-None (checked above)
    export_to_onnx(
        model=export_model,
        checkpoint_path=checkpoint_path,
        output_path=onnx_path,
        input_shape=input_shape,
        opset_version=export_cfg.opset_version,
        dynamic_axes=export_cfg.dynamic_axes,
        do_constant_folding=export_cfg.do_constant_folding,
        validate=export_cfg.validate_export,
    )

    # Post-export quantization
    quantized_path = None
    if export_cfg.quantize:
        quantized_path = quantize_model(
            onnx_path=onnx_path,
            backend=export_cfg.quantization_backend,
        )

    # Numerical validation: compare PyTorch vs ONNX outputs
    if export_cfg.validate_export:
        is_valid = validate_export(
            pytorch_model=export_model,
            onnx_path=onnx_path,
            input_shape=input_shape,
            num_samples=export_cfg.validation_samples,
            max_deviation=export_cfg.max_deviation,
        )
        if is_valid is False:
            logger.warning(
                f"  {LogStyle.WARNING} Numerical validation failed: "
                "ONNX outputs diverge from PyTorch model"
            )

    # Inference latency benchmark
    if export_cfg.benchmark:
        benchmark_onnx_inference(
            onnx_path=onnx_path,
            input_shape=input_shape,
            seed=cfg.training.seed,
            label="ONNX",
        )
        if quantized_path:
            benchmark_onnx_inference(
                onnx_path=quantized_path,
                input_shape=input_shape,
                seed=cfg.training.seed,
                label="Quantized",
            )

    logger.info(f"  {LogStyle.SUCCESS} Export completed")
    logger.info(f"    {LogStyle.ARROW} Output            : {onnx_path.name}")
    if quantized_path:
        logger.info(f"    {LogStyle.ARROW} Quantized         : {quantized_path.name}")

    return onnx_path
