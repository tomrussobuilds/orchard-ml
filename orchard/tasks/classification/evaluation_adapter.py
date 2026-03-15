"""
Classification Evaluation Pipeline Adapter.

Wraps :func:`orchard.evaluation.evaluation_pipeline.run_final_evaluation`
to satisfy :class:`~orchard.core.task_protocols.TaskEvalPipeline`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn
from torch.utils.data import DataLoader

from ...evaluation.evaluation_pipeline import run_final_evaluation

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from ...core.config import (
        AugmentationConfig,
        DatasetConfig,
        EvaluationConfig,
        TrainingConfig,
    )
    from ...core.paths import RunPaths
    from ...tracking import TrackerProtocol


class ClassificationEvalPipelineAdapter:
    """Orchestrates classification inference, visualization, and reporting."""

    def run_evaluation(
        self,
        model: nn.Module,
        test_loader: DataLoader[Any],
        train_losses: list[float],
        val_metrics_history: list[Mapping[str, float]],
        class_names: list[str],
        paths: RunPaths,
        training: TrainingConfig,
        dataset: DatasetConfig,
        augmentation: AugmentationConfig,
        evaluation: EvaluationConfig,
        arch_name: str,
        aug_info: str = "N/A",  # pragma: no mutate
        tracker: TrackerProtocol | None = None,
    ) -> tuple[float, float, float]:
        """
        Delegate to the existing final evaluation pipeline.

        Args:
            model: Trained model (already on target device).
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
            3-tuple of (macro_f1, test_acc, test_auc).
        """
        return run_final_evaluation(
            model=model,
            test_loader=test_loader,
            train_losses=train_losses,
            val_metrics_history=val_metrics_history,
            class_names=class_names,
            paths=paths,
            training=training,
            dataset=dataset,
            augmentation=augmentation,
            evaluation=evaluation,
            arch_name=arch_name,
            aug_info=aug_info,
            tracker=tracker,
        )
