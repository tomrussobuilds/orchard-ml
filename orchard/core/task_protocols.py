"""
Task-Agnostic Strategy Protocols.

Defines the structural contracts that task-specific components must satisfy.
Each protocol represents one dimension of task-specific behavior:

- **Criterion**: Loss function construction
- **Validation metrics**: Per-epoch metric computation
- **Eval pipeline**: Full evaluation orchestration (inference, visualization,
  reporting) as a single cohesive unit

Concrete implementations live under ``orchard.tasks.<task_type>/``
and are registered in the :mod:`orchard.core.task_registry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from ..tracking import TrackerProtocol
    from .config import (
        AugmentationConfig,
        DatasetConfig,
        EvaluationConfig,
        TrainingConfig,
    )
    from .paths import RunPaths


@runtime_checkable
class TaskCriterionFactory(Protocol):
    """Protocol for task-specific loss function construction."""

    def get_criterion(
        self,
        training: TrainingConfig,
        class_weights: torch.Tensor | None = None,
    ) -> nn.Module:
        """
        Build a loss function from training config.

        Args:
            training: Training sub-config with criterion parameters.
            class_weights: Optional per-class weights for imbalanced datasets.

        Returns:
            Configured loss module.
        """
        ...  # pragma: no cover


@runtime_checkable
class TaskValidationMetrics(Protocol):
    """Protocol for per-epoch validation metric computation."""

    def compute_validation_metrics(
        self,
        model: nn.Module,
        val_loader: DataLoader[Any],
        criterion: nn.Module,
        device: torch.device,
    ) -> Mapping[str, float]:
        """
        Compute task-specific validation metrics.

        Must return an immutable mapping with at least a ``"loss"`` key.

        Args:
            model: Neural network model to evaluate.
            val_loader: Validation data provider.
            criterion: Loss function.
            device: Hardware target (CUDA/MPS/CPU).

        Returns:
            Immutable mapping of metric name to value.
        """
        ...  # pragma: no cover


@runtime_checkable
class TaskEvalPipeline(Protocol):
    """Protocol for end-to-end evaluation (inference + visualization + reporting)."""

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
        aug_info: str = "N/A",
        tracker: TrackerProtocol | None = None,
    ) -> tuple[float, float, float]:
        """
        Execute the complete evaluation pipeline.

        Coordinates inference, visualization, and structured reporting
        as a single task-specific unit.

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
        ...  # pragma: no cover
