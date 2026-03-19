"""
Detection Evaluation Pipeline Adapter.

Minimal MVP evaluation for detection: inference + mAP computation +
training loss curves + structured report. Bbox visualization deferred
to a later release.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from ...core import LOGGER_NAME
from ...core.paths import METRIC_LOSS, METRIC_MAP, METRIC_MAP_50, METRIC_MAP_75
from ...evaluation.plot_context import PlotContext
from ...evaluation.visualization import plot_training_curves
from .helpers import to_cpu

if TYPE_CHECKING:  # pragma: no cover
    from ...core.config import (
        AugmentationConfig,
        DatasetConfig,
        EvaluationConfig,
        TrainingConfig,
    )
    from ...core.paths import RunPaths
    from ...tracking import TrackerProtocol

logger = logging.getLogger(LOGGER_NAME)


class DetectionEvalPipelineAdapter:
    """Orchestrates detection inference, mAP computation, and reporting."""

    def run_evaluation(
        self,
        model: nn.Module,
        test_loader: DataLoader[Any],
        train_losses: list[float],
        val_metrics_history: list[Mapping[str, float]],
        class_names: list[str],  # noqa: ARG002
        paths: RunPaths,
        training: TrainingConfig,  # noqa: ARG002
        dataset: DatasetConfig,
        augmentation: AugmentationConfig,  # noqa: ARG002
        evaluation: EvaluationConfig,
        arch_name: str,
        aug_info: str = "N/A",  # pragma: no mutate  # noqa: ARG002
        tracker: TrackerProtocol | None = None,
    ) -> Mapping[str, float]:
        """
        Run detection evaluation pipeline.

        Computes mAP metrics on the test set, plots training loss curves,
        and optionally logs metrics to the experiment tracker.

        Args:
            model: Trained detection model (already on target device).
            test_loader: DataLoader for test set.
            train_losses: Training loss history per epoch.
            val_metrics_history: Validation metrics history per epoch.
            class_names: List of class label strings.
            paths: RunPaths for artifact output.
            training: Training sub-config.
            dataset: Dataset sub-config.
            augmentation: Augmentation sub-config.
            evaluation: Evaluation sub-config.
            arch_name: Architecture identifier.
            aug_info: Augmentation description string.
            tracker: Optional experiment tracker for final metrics.

        Returns:
            Mapping of detection metric names to float values.
        """
        device = next(model.parameters()).device

        # Inference + mAP computation
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")

        with torch.no_grad():
            for images, targets in test_loader:
                images_on_device = [img.to(device) for img in images]
                predictions = model(images_on_device)
                metric.update(
                    [to_cpu(p) for p in predictions],
                    [to_cpu(t) for t in targets],
                )

        result = metric.compute()
        test_metrics = {
            METRIC_MAP: float(result["map"]),
            METRIC_MAP_50: float(result["map_50"]),
            METRIC_MAP_75: float(result["map_75"]),
        }

        # Log results
        logger.info(
            "Detection test metrics: mAP=%.4f  mAP@50=%.4f  mAP@75=%.4f",
            test_metrics[METRIC_MAP],
            test_metrics[METRIC_MAP_50],
            test_metrics[METRIC_MAP_75],
        )

        # Training curves (loss only — no accuracy for detection)
        val_losses = [m.get(METRIC_LOSS, 0.0) for m in val_metrics_history]
        ctx = PlotContext(  # pragma: no mutate
            arch_name=arch_name,
            resolution=dataset.resolution,
            fig_dpi=evaluation.fig_dpi,
            plot_style=evaluation.plot_style,
            cmap_confusion=evaluation.cmap_confusion,
            grid_cols=evaluation.grid_cols,
            n_samples=evaluation.n_samples,
            fig_size_predictions=evaluation.fig_size_predictions,
        )
        plot_training_curves(
            train_losses=train_losses,
            val_accuracies=val_losses,  # param name is classification-legacy; contains losses here
            out_path=paths.figures / "training_curves.png",  # pragma: no mutate
            ctx=ctx,
        )

        # Tracker logging
        if tracker is not None:
            full_metrics = {METRIC_LOSS: 0.0, **test_metrics}
            tracker.log_test_metrics(full_metrics)

        return MappingProxyType(test_metrics)
