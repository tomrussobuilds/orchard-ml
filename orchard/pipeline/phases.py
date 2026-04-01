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
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import optuna
import torch
import torch.nn as nn

from ..core import (
    LOGGER_NAME,
    Config,
    LogStyle,
    Reporter,
    log_optimization_summary,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core import RootOrchestrator
    from ..core.config import ExportConfig
    from ..tracking import TrackerProtocol

from ..architectures import get_model
from ..core.task_registry import get_task
from ..data_handler import (
    VisionDataset,
    get_augmentations_description,
    get_dataloaders,
    load_dataset,
    show_samples_for_dataset,
)
from ..export import (
    benchmark_onnx_inference,
    export_to_onnx,
    quantize_model,
    validate_export,
)
from ..optimization import run_optimization
from ..optimization.orchestrator.exporters import BEST_CONFIG_FILENAME
from ..trainer import (
    ModelTrainer,
    compute_class_weights,
    get_optimizer,
    get_scheduler,
)

logger = logging.getLogger(LOGGER_NAME)

_ERR_LOGGER_NOT_INIT = "Logger not initialized"
_ERR_PATHS_NOT_INIT = "Paths not initialized"
_QUANTIZED_TOLERANCE_FACTOR = 10


class TrainingResult(NamedTuple):
    """
    Structured return type for :func:`run_training_phase`.

    Attributes:
        best_model_path (Path): Filesystem path to the best checkpoint.
        train_losses (list[float]): Per-epoch training loss history.
        val_metrics (list[Mapping[str, float]]): Per-epoch validation metrics.
        model (nn.Module): Model with best weights restored.
        test_metrics (Mapping[str, float]): Final metrics on the test set.
    """

    best_model_path: Path
    train_losses: list[float]
    val_metrics: list[Mapping[str, float]]
    model: nn.Module
    test_metrics: Mapping[str, float]


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

    Reporter.log_phase_header(run_logger, "HYPERPARAMETER OPTIMIZATION", LogStyle.DOUBLE)

    # Execute Optuna study (includes post-processing: visualizations, best config export)
    study = run_optimization(cfg=cfg, device=device, paths=paths, tracker=tracker)

    log_optimization_summary(
        study=study,
        cfg=cfg,
        device=device,
        paths=paths,
    )

    # Best config path is in reports dir (exported by orchestrator if save_best_config=True)
    candidate = paths.reports / BEST_CONFIG_FILENAME
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
        val_metrics, model, test_metrics.

    Example:
        >>> with RootOrchestrator(cfg) as orch:
        ...     result = run_training_phase(orch)
        ...     print(f"Test metrics: {result.test_metrics}")
    """
    cfg = cfg or orchestrator.cfg
    paths = orchestrator.paths
    device = orchestrator.get_device()
    run_logger = orchestrator.run_logger

    # type guards for MyPy
    assert run_logger is not None, _ERR_LOGGER_NOT_INIT  # nosec B101
    assert paths is not None, _ERR_PATHS_NOT_INIT  # nosec B101

    # Dataset metadata (respects data_root override via _ensure_metadata)
    ds_meta = cfg.dataset._ensure_metadata

    # DATA PREPARATION
    Reporter.log_phase_header(run_logger, "DATA PREPARATION")

    data = load_dataset(ds_meta)
    loaders = get_dataloaders(
        data,
        cfg.dataset,
        cfg.training,
        cfg.augmentation,
        cfg.num_workers,
        task_type=cfg.task_type,
    )
    train_loader, val_loader, test_loader = loaders

    # show_samples_for_dataset assumes stacked Tensor batches — detection
    # batches (list[Tensor]) would crash on denormalization.
    if cfg.task_type != "detection":  # pragma: no mutate
        show_samples_for_dataset(
            loader=train_loader,
            dataset_name=cfg.dataset.dataset_name,
            run_paths=paths,
            mean=cfg.dataset.mean,
            std=cfg.dataset.std,
            arch_name=cfg.architecture.name,
            fig_dpi=cfg.evaluation.fig_dpi,
            num_samples=cfg.evaluation.n_samples,
            resolution=cfg.dataset.resolution,
        )

    # MODEL TRAINING
    Reporter.log_phase_header(
        run_logger, "TRAINING PIPELINE - " + cfg.architecture.name.upper(), LogStyle.DOUBLE
    )

    model = get_model(device=device, dataset_cfg=cfg.dataset, arch_cfg=cfg.architecture)

    class_weights = None
    if cfg.task_type == "classification" and cfg.training.weighted_loss:
        ds = cast(VisionDataset, train_loader.dataset)  # pragma: no mutate
        train_labels = ds.labels.flatten()
        class_weights = compute_class_weights(train_labels, ds_meta.num_classes, device)

    task = get_task(cfg.task_type)
    criterion = task.criterion_factory.get_criterion(cfg.training, class_weights=class_weights)
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
        output_path=paths.best_model_path,
        tracker=tracker,
        training_step=task.training_step,
        validation_metrics=task.validation_metrics,
    )

    best_model_path, train_losses, val_metrics_history = trainer.train()

    # FINAL EVALUATION
    Reporter.log_phase_header(run_logger, "FINAL EVALUATION")

    test_metrics = task.eval_pipeline.run_evaluation(
        model=model,
        test_loader=test_loader,
        train_losses=train_losses,
        val_metrics_history=val_metrics_history,
        class_names=ds_meta.classes,
        paths=paths,
        training=cfg.training,
        dataset=cfg.dataset,
        augmentation=cfg.augmentation,
        evaluation=cfg.evaluation,
        arch_name=cfg.architecture.name,
        aug_info=get_augmentations_description(
            cfg.augmentation,
            cast(int, cfg.dataset.img_size),  # pragma: no mutate
            cfg.training.mixup_alpha,
            ds_meta=ds_meta,
        ),
        tracker=tracker,
    )

    return TrainingResult(
        best_model_path=best_model_path,
        train_losses=train_losses,
        val_metrics=val_metrics_history,
        model=model,
        test_metrics=test_metrics,
    )


def _validate_exported_models(
    pytorch_model: nn.Module,
    onnx_path: Path,
    quantized_path: Path | None,
    input_shape: tuple[int, int, int],
    export_cfg: ExportConfig,
) -> None:
    """Validate ONNX (and optionally quantized) model against PyTorch outputs."""
    is_valid = validate_export(
        pytorch_model=pytorch_model,
        onnx_path=onnx_path,
        input_shape=input_shape,
        num_samples=export_cfg.validation_samples,
        max_deviation=export_cfg.max_deviation,
        label=export_cfg.format.upper(),
    )
    # `is False` (not `not is_valid`): None means onnxruntime is absent,
    # which is a skip — only warn when validation actually ran and failed.
    if is_valid is False:
        logger.warning(
            "  %s Numerical validation failed: ONNX outputs diverge from PyTorch model",
            LogStyle.WARNING,
        )

    if quantized_path is not None:
        q_valid = validate_export(
            pytorch_model=pytorch_model,
            onnx_path=quantized_path,
            input_shape=input_shape,
            num_samples=export_cfg.validation_samples,
            max_deviation=export_cfg.max_deviation * _QUANTIZED_TOLERANCE_FACTOR,
            label=export_cfg.quantization_type.upper(),
        )
        if q_valid is False:
            logger.error(
                "  %s Quantized model validation failed: outputs diverge beyond 10x tolerance",
                LogStyle.FAILURE,
            )


def _benchmark_exported_models(
    onnx_path: Path,
    quantized_path: Path | None,
    input_shape: tuple[int, int, int],
    export_cfg: ExportConfig,
    seed: int,
) -> None:
    """Run inference latency benchmarks on ONNX (and optionally quantized) model."""
    benchmark_onnx_inference(
        onnx_path=onnx_path,
        input_shape=input_shape,
        num_runs=export_cfg.benchmark_runs,
        seed=seed,
        label=export_cfg.format.upper(),
    )
    if quantized_path:
        benchmark_onnx_inference(
            onnx_path=quantized_path,
            input_shape=input_shape,
            num_runs=export_cfg.benchmark_runs,
            seed=seed,
            label=export_cfg.quantization_type.upper(),
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

    # TODO(detection): Remove pragma when ONNX export is implemented for detection
    if cfg.task_type == "detection":  # pragma: no mutate
        import warnings  # pragma: no mutate

        warnings.warn(  # pragma: no mutate
            "ONNX export is not yet supported for detection tasks. Skipping.",  # pragma: no mutate
            UserWarning,  # pragma: no mutate
            stacklevel=2,  # pragma: no mutate
        )
        return None  # pragma: no mutate

    paths = orchestrator.paths
    run_logger = orchestrator.run_logger

    # type guards for MyPy
    assert run_logger is not None, _ERR_LOGGER_NOT_INIT  # nosec B101
    assert paths is not None, _ERR_PATHS_NOT_INIT  # nosec B101

    Reporter.log_phase_header(run_logger, "MODEL EXPORT")

    # Determine input shape from config (must match get_model's channel resolution)
    resolution = cfg.dataset.resolution
    input_shape = (cfg.dataset.effective_in_channels, resolution, resolution)

    # Export path (directory managed by RunPaths)
    onnx_path = paths.exports / "model.onnx"

    # Reload model architecture (on CPU for export)
    export_model = get_model(
        device=torch.device("cpu"),
        dataset_cfg=cfg.dataset,
        arch_cfg=cfg.architecture,
        verbose=False,
    )

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
            weight_type=export_cfg.quantization_type,
        )

    # Numerical validation: compare PyTorch vs ONNX outputs
    if export_cfg.validate_export:
        _validate_exported_models(export_model, onnx_path, quantized_path, input_shape, export_cfg)

    # Inference latency benchmark
    if export_cfg.benchmark:
        _benchmark_exported_models(
            onnx_path, quantized_path, input_shape, export_cfg, cfg.training.seed
        )

    logger.info("  %s Export completed", LogStyle.SUCCESS)
    logger.info("    %s Output            : %s", LogStyle.ARROW, onnx_path.name)
    if quantized_path:
        logger.info("    %s Quantized         : %s", LogStyle.ARROW, quantized_path.name)

    return onnx_path
