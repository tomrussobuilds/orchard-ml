"""
Test Suite for Classification Task Adapters.

Tests that each adapter correctly delegates to its underlying function
and that the auto-registration mechanism works.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.tasks.classification.criterion_adapter import (
    ClassificationCriterionAdapter,
)
from orchard.tasks.classification.evaluation_adapter import (
    ClassificationEvalPipelineAdapter,
)
from orchard.tasks.classification.metrics_adapter import (
    ClassificationMetricsAdapter,
)
from orchard.tasks.classification.training_step_adapter import (
    ClassificationTrainingStepAdapter,
)

# ── ClassificationCriterionAdapter ────────────────────────────────────────────


@pytest.mark.unit
@patch("orchard.tasks.classification.criterion_adapter.get_criterion")
def test_criterion_adapter_delegates(mock_get_criterion: MagicMock) -> None:
    """CriterionAdapter delegates to trainer.setup.get_criterion."""
    mock_loss = MagicMock(spec=nn.Module)
    mock_get_criterion.return_value = mock_loss
    training_cfg = MagicMock()

    adapter = ClassificationCriterionAdapter()
    result = adapter.get_criterion(training_cfg)

    mock_get_criterion.assert_called_once_with(training_cfg, class_weights=None)
    assert result is mock_loss


@pytest.mark.unit
@patch("orchard.tasks.classification.criterion_adapter.get_criterion")
def test_criterion_adapter_passes_class_weights(mock_get_criterion: MagicMock) -> None:
    """CriterionAdapter forwards class_weights to get_criterion."""
    mock_loss = MagicMock(spec=nn.Module)
    mock_get_criterion.return_value = mock_loss
    training_cfg = MagicMock()
    weights = torch.tensor([0.5, 1.5])

    adapter = ClassificationCriterionAdapter()
    adapter.get_criterion(training_cfg, class_weights=weights)

    call_kwargs = mock_get_criterion.call_args.kwargs
    assert torch.equal(call_kwargs["class_weights"], weights)


@pytest.mark.unit
@patch("orchard.tasks.classification.criterion_adapter.get_criterion")
def test_criterion_adapter_default_weights_none(mock_get_criterion: MagicMock) -> None:
    """CriterionAdapter defaults class_weights to None."""
    mock_get_criterion.return_value = MagicMock()
    training_cfg = MagicMock()

    ClassificationCriterionAdapter().get_criterion(training_cfg)

    assert mock_get_criterion.call_args.kwargs["class_weights"] is None


# ── ClassificationMetricsAdapter ──────────────────────────────────────────────


@pytest.mark.unit
@patch("orchard.tasks.classification.metrics_adapter.validate_epoch")
def test_metrics_adapter_delegates(mock_validate: MagicMock) -> None:
    """MetricsAdapter delegates to trainer.engine.validate_epoch."""
    mock_metrics = {"loss": 0.3, "accuracy": 0.9, "auc": 0.85, "f1": 0.88}
    mock_validate.return_value = mock_metrics

    model = MagicMock(spec=nn.Module)
    loader = MagicMock()
    criterion = MagicMock(spec=nn.Module)
    device = torch.device("cpu")

    adapter = ClassificationMetricsAdapter()
    result = adapter.compute_validation_metrics(model, loader, criterion, device)

    mock_validate.assert_called_once_with(model, loader, criterion, device)
    assert result is mock_metrics


@pytest.mark.unit
@patch("orchard.tasks.classification.metrics_adapter.validate_epoch")
def test_metrics_adapter_returns_mapping(mock_validate: MagicMock) -> None:
    """MetricsAdapter returns the mapping from validate_epoch unchanged."""
    expected = {"loss": 0.25, "accuracy": 0.92, "auc": 0.88, "f1": 0.90}
    mock_validate.return_value = expected

    adapter = ClassificationMetricsAdapter()
    result = adapter.compute_validation_metrics(
        MagicMock(), MagicMock(), MagicMock(), torch.device("cpu")
    )

    assert result["loss"] == pytest.approx(0.25)
    assert result["accuracy"] == pytest.approx(0.92)


# ── ClassificationEvalPipelineAdapter ─────────────────────────────────────────


@pytest.mark.unit
@patch("orchard.tasks.classification.evaluation_adapter.run_final_evaluation")
def test_eval_adapter_delegates(mock_eval: MagicMock) -> None:
    """EvalPipelineAdapter delegates to run_final_evaluation."""
    mock_eval.return_value = (0.85, 0.90, 0.92)

    adapter = ClassificationEvalPipelineAdapter()
    result = adapter.run_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.5, 0.4],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a", "b"],
        paths=MagicMock(),
        training=MagicMock(),
        dataset=MagicMock(),
        augmentation=MagicMock(),
        evaluation=MagicMock(),
        arch_name="mini_cnn",
    )

    mock_eval.assert_called_once()
    assert result == {"f1": 0.85, "accuracy": 0.90, "auc": 0.92}


@pytest.mark.unit
@patch("orchard.tasks.classification.evaluation_adapter.run_final_evaluation")
def test_eval_adapter_passes_all_kwargs(mock_eval: MagicMock) -> None:
    """EvalPipelineAdapter forwards all kwargs to run_final_evaluation."""
    mock_eval.return_value = (0.85, 0.90, 0.92)

    model = MagicMock()
    test_loader = MagicMock()
    paths = MagicMock()
    training = MagicMock()
    dataset = MagicMock()
    augmentation = MagicMock()
    evaluation = MagicMock()
    tracker = MagicMock()

    adapter = ClassificationEvalPipelineAdapter()
    adapter.run_evaluation(
        model=model,
        test_loader=test_loader,
        train_losses=[0.5],
        val_metrics_history=[{"acc": 0.9}],
        class_names=["a", "b"],
        paths=paths,
        training=training,
        dataset=dataset,
        augmentation=augmentation,
        evaluation=evaluation,
        arch_name="resnet",
        aug_info="test_aug",
        tracker=tracker,
    )

    kw = mock_eval.call_args.kwargs
    assert kw["model"] is model
    assert kw["test_loader"] is test_loader
    assert kw["train_losses"] == [0.5]
    assert kw["val_metrics_history"] == [{"acc": 0.9}]
    assert kw["class_names"] == ["a", "b"]
    assert kw["paths"] is paths
    assert kw["training"] is training
    assert kw["dataset"] is dataset
    assert kw["augmentation"] is augmentation
    assert kw["evaluation"] is evaluation
    assert kw["arch_name"] == "resnet"
    assert kw["aug_info"] == "test_aug"
    assert kw["tracker"] is tracker


@pytest.mark.unit
@patch("orchard.tasks.classification.evaluation_adapter.run_final_evaluation")
def test_eval_adapter_default_aug_info(mock_eval: MagicMock) -> None:
    """EvalPipelineAdapter uses default aug_info='N/A' when not specified."""
    mock_eval.return_value = (0.8, 0.85, 0.9)

    adapter = ClassificationEvalPipelineAdapter()
    adapter.run_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[],
        val_metrics_history=[],
        class_names=[],
        paths=MagicMock(),
        training=MagicMock(),
        dataset=MagicMock(),
        augmentation=MagicMock(),
        evaluation=MagicMock(),
        arch_name="cnn",
    )

    assert mock_eval.call_args.kwargs["aug_info"] == "N/A"


@pytest.mark.unit
@patch("orchard.tasks.classification.evaluation_adapter.run_final_evaluation")
def test_eval_adapter_default_tracker_none(mock_eval: MagicMock) -> None:
    """EvalPipelineAdapter passes tracker=None by default."""
    mock_eval.return_value = (0.8, 0.85, 0.9)

    adapter = ClassificationEvalPipelineAdapter()
    adapter.run_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[],
        val_metrics_history=[],
        class_names=[],
        paths=MagicMock(),
        training=MagicMock(),
        dataset=MagicMock(),
        augmentation=MagicMock(),
        evaluation=MagicMock(),
        arch_name="cnn",
    )

    assert mock_eval.call_args.kwargs["tracker"] is None


@pytest.mark.unit
@patch("orchard.tasks.classification.evaluation_adapter.run_final_evaluation")
def test_eval_adapter_returns_mapping(mock_eval: MagicMock) -> None:
    """EvalPipelineAdapter converts the 3-tuple to a metric mapping."""
    mock_eval.return_value = (0.91, 0.93, 0.95)

    adapter = ClassificationEvalPipelineAdapter()
    result = adapter.run_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[],
        val_metrics_history=[],
        class_names=[],
        paths=MagicMock(),
        training=MagicMock(),
        dataset=MagicMock(),
        augmentation=MagicMock(),
        evaluation=MagicMock(),
        arch_name="cnn",
    )

    assert result["f1"] == pytest.approx(0.91)
    assert result["accuracy"] == pytest.approx(0.93)
    assert result["auc"] == pytest.approx(0.95)


# ── ClassificationTrainingStepAdapter ─────────────────────────────────────────


@pytest.mark.unit
def test_training_step_adapter_without_mixup() -> None:
    """TrainingStepAdapter calls model(inputs) and criterion(outputs, targets)."""
    model = MagicMock(spec=nn.Module)
    logits = torch.randn(4, 10)
    model.return_value = logits

    criterion = MagicMock(spec=nn.Module)
    loss_val = torch.tensor(0.5)
    criterion.return_value = loss_val

    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))

    adapter = ClassificationTrainingStepAdapter()
    result = adapter.compute_training_loss(model, inputs, targets, criterion)

    model.assert_called_once_with(inputs)
    criterion.assert_called_once_with(logits, targets)
    assert result is loss_val


@pytest.mark.unit
def test_training_step_adapter_with_mixup() -> None:
    """TrainingStepAdapter applies MixUp and computes blended loss."""
    model = MagicMock(spec=nn.Module)
    logits = torch.randn(4, 10)
    model.return_value = logits

    criterion = MagicMock(spec=nn.Module)
    criterion.return_value = torch.tensor(1.0)

    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    mixed_inputs = torch.randn(4, 1, 28, 28)
    y_a = torch.randint(0, 10, (4,))
    y_b = torch.randint(0, 10, (4,))
    lam = 0.7

    mixup_fn = MagicMock(return_value=(mixed_inputs, y_a, y_b, lam))

    adapter = ClassificationTrainingStepAdapter()
    result = adapter.compute_training_loss(model, inputs, targets, criterion, mixup_fn)

    mixup_fn.assert_called_once_with(inputs, targets)
    model.assert_called_once_with(mixed_inputs)
    assert criterion.call_count == 2
    # Verify criterion receives (outputs, y_a) and (outputs, y_b)
    first_call, second_call = criterion.call_args_list
    assert torch.equal(first_call[0][0], logits)
    assert torch.equal(first_call[0][1], y_a)
    assert torch.equal(second_call[0][0], logits)
    assert torch.equal(second_call[0][1], y_b)
    expected_loss = lam * criterion.return_value + (1 - lam) * criterion.return_value
    assert torch.isclose(result, expected_loss)


@pytest.mark.unit
def test_training_step_adapter_mixup_none_means_no_mixup() -> None:
    """TrainingStepAdapter with mixup_fn=None uses direct forward."""
    model = MagicMock(spec=nn.Module)
    logits = torch.randn(4, 10)
    model.return_value = logits

    criterion = MagicMock(spec=nn.Module)
    criterion.return_value = torch.tensor(0.3)

    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))

    adapter = ClassificationTrainingStepAdapter()
    result = adapter.compute_training_loss(model, inputs, targets, criterion, None)

    model.assert_called_once_with(inputs)
    criterion.assert_called_once_with(logits, targets)
    assert result is criterion.return_value


@pytest.mark.unit
def test_training_step_adapter_mixup_blended_loss_value() -> None:
    """TrainingStepAdapter computes lam * loss_a + (1-lam) * loss_b."""
    model = MagicMock(spec=nn.Module)
    model.return_value = torch.randn(4, 10)

    loss_a = torch.tensor(2.0)
    loss_b = torch.tensor(4.0)
    criterion = MagicMock(spec=nn.Module, side_effect=[loss_a, loss_b])

    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))
    lam = 0.6

    mixup_fn = MagicMock(return_value=(inputs, targets, targets, lam))

    adapter = ClassificationTrainingStepAdapter()
    result = adapter.compute_training_loss(model, inputs, targets, criterion, mixup_fn)

    expected = 0.6 * 2.0 + 0.4 * 4.0
    assert result.item() == pytest.approx(expected)


# ── Auto-registration ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_tasks_init_triggers_registration() -> None:
    """Importing orchard.tasks registers the classification task."""
    from orchard.core.task_registry import _TASK_REGISTRY

    # The conftest or other test imports will have triggered registration.
    # We verify classification is present.
    assert "classification" in _TASK_REGISTRY

    components = _TASK_REGISTRY["classification"]
    assert isinstance(components.criterion_factory, ClassificationCriterionAdapter)
    assert isinstance(components.training_step, ClassificationTrainingStepAdapter)
    assert isinstance(components.validation_metrics, ClassificationMetricsAdapter)
    assert isinstance(components.eval_pipeline, ClassificationEvalPipelineAdapter)
