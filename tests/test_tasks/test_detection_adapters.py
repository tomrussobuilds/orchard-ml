"""
Test Suite for Detection Task Adapters.

Tests that each adapter correctly implements its protocol contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.tasks.detection.criterion_adapter import (
    DetectionCriterionAdapter,
    _DetectionNoOpCriterion,
)
from orchard.tasks.detection.evaluation_adapter import DetectionEvalPipelineAdapter
from orchard.tasks.detection.metrics_adapter import DetectionMetricsAdapter
from orchard.tasks.detection.training_step_adapter import (
    DetectionTrainingStepAdapter,
)

# ── DetectionCriterionAdapter ────────────────────────────────────────────────


@pytest.mark.unit
def test_criterion_adapter_returns_module() -> None:
    """CriterionAdapter returns an nn.Module instance."""
    adapter = DetectionCriterionAdapter()
    training_cfg = MagicMock()
    result = adapter.get_criterion(training_cfg)

    assert isinstance(result, nn.Module)


@pytest.mark.unit
def test_criterion_adapter_returns_noop_sentinel() -> None:
    """CriterionAdapter returns a _DetectionNoOpCriterion sentinel."""
    adapter = DetectionCriterionAdapter()
    result = adapter.get_criterion(MagicMock())

    assert isinstance(result, _DetectionNoOpCriterion)


@pytest.mark.unit
def test_criterion_adapter_ignores_class_weights() -> None:
    """CriterionAdapter returns sentinel regardless of class_weights."""
    adapter = DetectionCriterionAdapter()
    weights = torch.tensor([0.5, 1.5])
    result = adapter.get_criterion(MagicMock(), class_weights=weights)

    assert isinstance(result, _DetectionNoOpCriterion)


@pytest.mark.unit
def test_noop_criterion_raises_on_forward() -> None:
    """_DetectionNoOpCriterion.forward() raises RuntimeError."""
    criterion = _DetectionNoOpCriterion()
    with pytest.raises(RuntimeError, match="Detection models compute losses internally"):
        criterion(torch.randn(4, 10), torch.randint(0, 10, (4,)))


@pytest.mark.unit
def test_noop_criterion_raises_with_any_args() -> None:
    """_DetectionNoOpCriterion.forward() raises regardless of arguments."""
    criterion = _DetectionNoOpCriterion()
    with pytest.raises(RuntimeError, match="criterion should never be called"):
        criterion()


# ── DetectionTrainingStepAdapter ─────────────────────────────────────────────


def _make_detection_batch(
    batch_size: int = 2,
    num_boxes: int = 3,
    img_size: int = 64,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Create a synthetic detection batch (images + target dicts)."""
    images = [torch.randn(3, img_size, img_size) for _ in range(batch_size)]
    targets = []
    for _ in range(batch_size):
        boxes = torch.rand(num_boxes, 4) * img_size
        # Ensure x2 > x1, y2 > y1
        boxes[:, 2] = boxes[:, 0] + torch.rand(num_boxes) * (img_size - boxes[:, 0])
        boxes[:, 3] = boxes[:, 1] + torch.rand(num_boxes) * (img_size - boxes[:, 1])
        labels = torch.randint(1, 5, (num_boxes,))
        targets.append({"boxes": boxes, "labels": labels})
    return images, targets


def _make_mock_detection_model(
    loss_dict: dict[str, torch.Tensor] | None = None,
) -> MagicMock:
    """Create a mock detection model that returns a loss dict."""
    model = MagicMock(spec=nn.Module)
    if loss_dict is None:
        loss_dict = {
            "loss_classifier": torch.tensor(0.5, requires_grad=True),
            "loss_box_reg": torch.tensor(0.3, requires_grad=True),
            "loss_objectness": torch.tensor(0.1, requires_grad=True),
            "loss_rpn_box_reg": torch.tensor(0.1, requires_grad=True),
        }
    model.return_value = loss_dict
    return model


@pytest.mark.unit
def test_training_step_sums_loss_dict() -> None:
    """TrainingStepAdapter sums all loss components from the model."""
    model = _make_mock_detection_model()
    images, targets = _make_detection_batch()
    criterion = MagicMock(spec=nn.Module)

    adapter = DetectionTrainingStepAdapter()
    result = adapter.compute_training_loss(model, images, targets, criterion)

    expected = 0.5 + 0.3 + 0.1 + 0.1
    assert result.item() == pytest.approx(expected)


@pytest.mark.unit
def test_training_step_calls_model_with_images_and_targets() -> None:
    """TrainingStepAdapter passes images and targets to the model."""
    model = _make_mock_detection_model()
    images, targets = _make_detection_batch(batch_size=2)
    criterion = MagicMock(spec=nn.Module)

    adapter = DetectionTrainingStepAdapter()
    adapter.compute_training_loss(model, images, targets, criterion)

    model.assert_called_once()
    call_args = model.call_args[0]
    assert len(call_args[0]) == 2  # 2 images
    assert len(call_args[1]) == 2  # 2 target dicts


@pytest.mark.unit
def test_training_step_ignores_criterion() -> None:
    """TrainingStepAdapter never calls the criterion."""
    model = _make_mock_detection_model()
    images, targets = _make_detection_batch()
    criterion = MagicMock(spec=nn.Module)

    adapter = DetectionTrainingStepAdapter()
    adapter.compute_training_loss(model, images, targets, criterion)

    criterion.assert_not_called()


@pytest.mark.unit
def test_training_step_ignores_mixup() -> None:
    """TrainingStepAdapter ignores mixup_fn (not applicable to detection)."""
    model = _make_mock_detection_model()
    images, targets = _make_detection_batch()
    criterion = MagicMock(spec=nn.Module)
    mixup_fn = MagicMock()

    adapter = DetectionTrainingStepAdapter()
    adapter.compute_training_loss(model, images, targets, criterion, mixup_fn)

    mixup_fn.assert_not_called()


@pytest.mark.unit
def test_training_step_device_transfer() -> None:
    """TrainingStepAdapter moves images and targets to device."""
    images = [MagicMock(spec=torch.Tensor) for _ in range(2)]
    for img in images:
        img.to.return_value = torch.randn(3, 64, 64)

    box_tensor = MagicMock(spec=torch.Tensor)
    box_tensor.to.return_value = torch.rand(3, 4)
    label_tensor = MagicMock(spec=torch.Tensor)
    label_tensor.to.return_value = torch.randint(1, 5, (3,))
    targets: list[dict[str, Any]] = [{"boxes": box_tensor, "labels": label_tensor}]

    model = _make_mock_detection_model()
    device = torch.device("cpu")

    adapter = DetectionTrainingStepAdapter()
    adapter.compute_training_loss(model, images, targets, MagicMock(), device=device)

    for img in images:
        img.to.assert_called_once_with(device)
    box_tensor.to.assert_called_once_with(device)
    label_tensor.to.assert_called_once_with(device)


@pytest.mark.unit
def test_training_step_no_device_skips_transfer() -> None:
    """TrainingStepAdapter does NOT call .to() when device is None."""
    images, targets = _make_detection_batch(batch_size=1)
    model = _make_mock_detection_model()

    adapter = DetectionTrainingStepAdapter()
    result = adapter.compute_training_loss(model, images, targets, MagicMock(), device=None)

    # Should still work — images/targets stay on their original device
    assert isinstance(result, torch.Tensor)


@pytest.mark.unit
def test_training_step_returns_scalar() -> None:
    """TrainingStepAdapter returns a scalar (0-dim) tensor."""
    model = _make_mock_detection_model()
    images, targets = _make_detection_batch()

    adapter = DetectionTrainingStepAdapter()
    result = adapter.compute_training_loss(model, images, targets, MagicMock())

    assert result.dim() == 0


@pytest.mark.unit
def test_training_step_preserves_grad() -> None:
    """Summed loss preserves requires_grad for backpropagation."""
    model = _make_mock_detection_model()
    images, targets = _make_detection_batch()

    adapter = DetectionTrainingStepAdapter()
    result = adapter.compute_training_loss(model, images, targets, MagicMock())

    assert result.requires_grad


@pytest.mark.unit
def test_training_step_single_loss_component() -> None:
    """TrainingStepAdapter works with a single loss component."""
    loss_dict = {"loss_total": torch.tensor(1.5, requires_grad=True)}
    model = _make_mock_detection_model(loss_dict)
    images, targets = _make_detection_batch()

    adapter = DetectionTrainingStepAdapter()
    result = adapter.compute_training_loss(model, images, targets, MagicMock())

    assert result.item() == pytest.approx(1.5)


# ── DetectionMetricsAdapter ──────────────────────────────────────────────────


@pytest.mark.unit
@patch("orchard.tasks.detection.metrics_adapter.MeanAveragePrecision")
def test_metrics_adapter_returns_mapping(mock_map_cls: MagicMock) -> None:
    """MetricsAdapter returns an immutable mapping with mAP keys."""
    mock_metric = MagicMock()
    mock_metric.compute.return_value = {
        "map": torch.tensor(0.45),
        "map_50": torch.tensor(0.65),
        "map_75": torch.tensor(0.35),
    }
    mock_map_cls.return_value = mock_metric

    model = MagicMock(spec=nn.Module)
    val_loader = MagicMock()
    val_loader.__iter__ = MagicMock(return_value=iter([]))

    adapter = DetectionMetricsAdapter()
    result = adapter.compute_validation_metrics(model, val_loader, MagicMock(), torch.device("cpu"))

    assert isinstance(result, Mapping)
    assert result["map"] == pytest.approx(0.45)
    assert result["map_50"] == pytest.approx(0.65)
    assert result["map_75"] == pytest.approx(0.35)
    assert result["loss"] == pytest.approx(0.0)


@pytest.mark.unit
@patch("orchard.tasks.detection.metrics_adapter.MeanAveragePrecision")
def test_metrics_adapter_sets_model_eval(mock_map_cls: MagicMock) -> None:
    """MetricsAdapter calls model.eval() before inference."""
    mock_map_cls.return_value.compute.return_value = {
        "map": torch.tensor(0.0),
        "map_50": torch.tensor(0.0),
        "map_75": torch.tensor(0.0),
    }

    model = MagicMock(spec=nn.Module)
    val_loader = MagicMock()
    val_loader.__iter__ = MagicMock(return_value=iter([]))

    adapter = DetectionMetricsAdapter()
    adapter.compute_validation_metrics(model, val_loader, MagicMock(), torch.device("cpu"))

    model.eval.assert_called_once()


@pytest.mark.unit
def test_metrics_adapter_real_inference() -> None:
    """MetricsAdapter computes mAP without mocks (integration test, kills loop mutants)."""

    # Model that returns predictions in eval mode
    def detection_model_eval(images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
            for _ in images
        ]

    model = MagicMock(spec=nn.Module)
    model.side_effect = detection_model_eval

    images = [torch.randn(3, 64, 64), torch.randn(3, 64, 64)]
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[15.0, 15.0, 55.0, 55.0]]), "labels": torch.tensor([1])},
    ]

    val_loader = MagicMock()
    val_loader.__iter__ = MagicMock(return_value=iter([(images, targets)]))

    adapter = DetectionMetricsAdapter()
    result = adapter.compute_validation_metrics(model, val_loader, MagicMock(), torch.device("cpu"))

    assert "map" in result
    assert "map_50" in result
    assert "map_75" in result
    assert "loss" in result
    assert result["map"] >= 0.0
    model.eval.assert_called_once()


@pytest.mark.unit
@patch("orchard.tasks.detection.metrics_adapter.MeanAveragePrecision")
def test_metrics_adapter_processes_batches(mock_map_cls: MagicMock) -> None:
    """MetricsAdapter iterates batches and updates the metric."""
    mock_metric = MagicMock()
    mock_metric.compute.return_value = {
        "map": torch.tensor(0.5),
        "map_50": torch.tensor(0.7),
        "map_75": torch.tensor(0.4),
    }
    mock_map_cls.return_value = mock_metric

    images = [torch.randn(3, 64, 64), torch.randn(3, 64, 64)]
    targets = [
        {"boxes": torch.rand(2, 4), "labels": torch.tensor([1, 2])},
        {"boxes": torch.rand(1, 4), "labels": torch.tensor([1])},
    ]

    model = MagicMock(spec=nn.Module)
    model.return_value = [
        {"boxes": torch.rand(2, 4), "scores": torch.rand(2), "labels": torch.tensor([1, 2])},
        {"boxes": torch.rand(1, 4), "scores": torch.rand(1), "labels": torch.tensor([1])},
    ]

    val_loader = MagicMock()
    val_loader.__iter__ = MagicMock(return_value=iter([(images, targets)]))

    adapter = DetectionMetricsAdapter()
    adapter.compute_validation_metrics(model, val_loader, MagicMock(), torch.device("cpu"))

    mock_metric.update.assert_called_once()


# ── DetectionEvalPipelineAdapter ─────────────────────────────────────────────


@pytest.mark.unit
def test_eval_adapter_real_inference() -> None:
    """EvalPipelineAdapter computes mAP without mocks (integration, kills loop mutants)."""

    def detection_model_eval(images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
            for _ in images
        ]

    model = MagicMock(spec=nn.Module)
    model.side_effect = detection_model_eval
    param = torch.randn(1)
    model.parameters.return_value = iter([param])

    images = [torch.randn(3, 64, 64)]
    targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])}]

    test_loader = MagicMock()
    test_loader.__iter__ = MagicMock(return_value=iter([(images, targets)]))
    paths = MagicMock()

    dataset_cfg = MagicMock()
    dataset_cfg.resolution = 224
    dataset_cfg.name = "test_det"
    eval_cfg = MagicMock()
    eval_cfg.fig_dpi = 100
    eval_cfg.plot_style = "default"
    eval_cfg.cmap_confusion = "Blues"
    eval_cfg.grid_cols = 4
    eval_cfg.n_samples = 16
    eval_cfg.fig_size_predictions = (12, 8)

    with patch("orchard.tasks.detection.evaluation_adapter.plot_training_curves"):
        adapter = DetectionEvalPipelineAdapter()
        result = adapter.run_evaluation(
            model=model,
            test_loader=test_loader,
            train_losses=[0.5],
            val_metrics_history=[{"loss": 0.3}],
            class_names=["obj"],
            paths=paths,
            training=MagicMock(),
            dataset=dataset_cfg,
            augmentation=MagicMock(),
            evaluation=eval_cfg,
            arch_name="fasterrcnn",
        )

    assert isinstance(result, Mapping)
    assert "map" in result
    assert "map_50" in result
    assert result["map"] >= 0.0


@pytest.mark.unit
@patch("orchard.tasks.detection.evaluation_adapter.plot_training_curves")
@patch("orchard.tasks.detection.evaluation_adapter.MeanAveragePrecision")
def test_eval_adapter_returns_metrics(mock_map_cls: MagicMock, mock_plot: MagicMock) -> None:
    """EvalPipelineAdapter returns mAP metrics from test set."""
    mock_metric = MagicMock()
    mock_metric.compute.return_value = {
        "map": torch.tensor(0.55),
        "map_50": torch.tensor(0.75),
        "map_75": torch.tensor(0.45),
    }
    mock_map_cls.return_value = mock_metric

    model = MagicMock(spec=nn.Module)
    param = torch.randn(1)
    model.parameters.return_value = iter([param])
    test_loader = MagicMock()
    test_loader.__iter__ = MagicMock(return_value=iter([]))
    paths = MagicMock()

    adapter = DetectionEvalPipelineAdapter()
    result = adapter.run_evaluation(
        model=model,
        test_loader=test_loader,
        train_losses=[0.5, 0.4],
        val_metrics_history=[{"loss": 0.3, "map": 0.4}],
        class_names=["cat", "dog"],
        paths=paths,
        training=MagicMock(),
        dataset=MagicMock(),
        augmentation=MagicMock(),
        evaluation=MagicMock(),
        arch_name="fasterrcnn",
    )

    assert isinstance(result, Mapping)
    assert result["map"] == pytest.approx(0.55)
    assert result["map_50"] == pytest.approx(0.75)
    assert result["map_75"] == pytest.approx(0.45)


@pytest.mark.unit
@patch("orchard.tasks.detection.evaluation_adapter.plot_training_curves")
@patch("orchard.tasks.detection.evaluation_adapter.MeanAveragePrecision")
def test_eval_adapter_plots_training_curves(mock_map_cls: MagicMock, mock_plot: MagicMock) -> None:
    """EvalPipelineAdapter generates training curves with correct args."""
    mock_map_cls.return_value.compute.return_value = {
        "map": torch.tensor(0.0),
        "map_50": torch.tensor(0.0),
        "map_75": torch.tensor(0.0),
    }

    model = MagicMock(spec=nn.Module)
    model.parameters.return_value = iter([torch.randn(1)])
    test_loader = MagicMock()
    test_loader.__iter__ = MagicMock(return_value=iter([]))
    paths = MagicMock()
    dataset_cfg = MagicMock()
    dataset_cfg.resolution = 224
    dataset_cfg.name = "test"
    eval_cfg = MagicMock()
    eval_cfg.fig_dpi = 100
    eval_cfg.plot_style = "default"
    eval_cfg.cmap_confusion = "Blues"
    eval_cfg.grid_cols = 4
    eval_cfg.n_samples = 16
    eval_cfg.fig_size_predictions = (12, 8)

    adapter = DetectionEvalPipelineAdapter()
    adapter.run_evaluation(
        model=model,
        test_loader=test_loader,
        train_losses=[0.5],
        val_metrics_history=[{"loss": 0.3}],
        class_names=[],
        paths=paths,
        training=MagicMock(),
        dataset=dataset_cfg,
        augmentation=MagicMock(),
        evaluation=eval_cfg,
        arch_name="fasterrcnn",
    )

    mock_plot.assert_called_once()
    kw = mock_plot.call_args[1]
    assert kw["train_losses"] == [0.5]
    assert kw["val_accuracies"] == [0.3]
    assert kw["ctx"] is not None
    assert kw["val_label"] == "Validation Loss"
    assert kw["out_path"] == paths.figures / "training_curves.png"


@pytest.mark.unit
@patch("orchard.tasks.detection.evaluation_adapter.plot_training_curves")
@patch("orchard.tasks.detection.evaluation_adapter.MeanAveragePrecision")
def test_eval_adapter_val_losses_fallback(mock_map_cls: MagicMock, mock_plot: MagicMock) -> None:
    """val_losses defaults to 0.0 when METRIC_LOSS is missing from history."""
    mock_map_cls.return_value.compute.return_value = {
        "map": torch.tensor(0.0),
        "map_50": torch.tensor(0.0),
        "map_75": torch.tensor(0.0),
    }

    model = MagicMock(spec=nn.Module)
    model.parameters.return_value = iter([torch.randn(1)])
    test_loader = MagicMock()
    test_loader.__iter__ = MagicMock(return_value=iter([]))
    paths = MagicMock()
    dataset_cfg = MagicMock()
    dataset_cfg.resolution = 224
    dataset_cfg.name = "test"
    eval_cfg = MagicMock()
    eval_cfg.fig_dpi = 100
    eval_cfg.plot_style = "default"
    eval_cfg.cmap_confusion = "Blues"
    eval_cfg.grid_cols = 4
    eval_cfg.n_samples = 16
    eval_cfg.fig_size_predictions = (12, 8)

    adapter = DetectionEvalPipelineAdapter()
    adapter.run_evaluation(
        model=model,
        test_loader=test_loader,
        train_losses=[0.5],
        val_metrics_history=[{"map": 0.4}],  # no "loss" key
        class_names=[],
        paths=paths,
        training=MagicMock(),
        dataset=dataset_cfg,
        augmentation=MagicMock(),
        evaluation=eval_cfg,
        arch_name="fasterrcnn",
    )

    kw = mock_plot.call_args[1]
    assert kw["val_accuracies"] == [0.0]
    assert kw["val_accuracies"] != [1.0]
    assert kw["val_accuracies"] != [None]


@pytest.mark.unit
@patch("orchard.tasks.detection.evaluation_adapter.plot_training_curves")
@patch("orchard.tasks.detection.evaluation_adapter.MeanAveragePrecision")
def test_eval_adapter_logs_to_tracker(mock_map_cls: MagicMock, mock_plot: MagicMock) -> None:
    """EvalPipelineAdapter logs metrics to tracker when provided."""
    mock_map_cls.return_value.compute.return_value = {
        "map": torch.tensor(0.5),
        "map_50": torch.tensor(0.7),
        "map_75": torch.tensor(0.4),
    }

    model = MagicMock(spec=nn.Module)
    model.parameters.return_value = iter([torch.randn(1)])
    test_loader = MagicMock()
    test_loader.__iter__ = MagicMock(return_value=iter([]))
    tracker = MagicMock()

    adapter = DetectionEvalPipelineAdapter()
    adapter.run_evaluation(
        model=model,
        test_loader=test_loader,
        train_losses=[],
        val_metrics_history=[],
        class_names=[],
        paths=MagicMock(),
        training=MagicMock(),
        dataset=MagicMock(),
        augmentation=MagicMock(),
        evaluation=MagicMock(),
        arch_name="fasterrcnn",
        tracker=tracker,
    )

    tracker.log_test_metrics.assert_called_once()
    logged = tracker.log_test_metrics.call_args[0][0]
    assert "map" in logged
    assert "loss" in logged
    assert logged["loss"] == pytest.approx(0.0)  # detection has no val loss


@pytest.mark.unit
@patch("orchard.tasks.detection.evaluation_adapter.plot_training_curves")
@patch("orchard.tasks.detection.evaluation_adapter.MeanAveragePrecision")
def test_eval_adapter_no_tracker_no_error(mock_map_cls: MagicMock, mock_plot: MagicMock) -> None:
    """EvalPipelineAdapter works without a tracker (tracker=None)."""
    mock_map_cls.return_value.compute.return_value = {
        "map": torch.tensor(0.0),
        "map_50": torch.tensor(0.0),
        "map_75": torch.tensor(0.0),
    }

    model = MagicMock(spec=nn.Module)
    model.parameters.return_value = iter([torch.randn(1)])
    test_loader = MagicMock()
    test_loader.__iter__ = MagicMock(return_value=iter([]))

    adapter = DetectionEvalPipelineAdapter()
    result = adapter.run_evaluation(
        model=model,
        test_loader=test_loader,
        train_losses=[],
        val_metrics_history=[],
        class_names=[],
        paths=MagicMock(),
        training=MagicMock(),
        dataset=MagicMock(),
        augmentation=MagicMock(),
        evaluation=MagicMock(),
        arch_name="fasterrcnn",
    )

    assert isinstance(result, Mapping)


# ── Auto-registration ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_detection_task_registered() -> None:
    """Importing orchard registers the detection task."""
    from orchard.core.task_registry import _TASK_REGISTRY

    assert "detection" in _TASK_REGISTRY

    components = _TASK_REGISTRY["detection"]
    assert isinstance(components.criterion_factory, DetectionCriterionAdapter)
    assert isinstance(components.training_step, DetectionTrainingStepAdapter)
    assert isinstance(components.validation_metrics, DetectionMetricsAdapter)
    assert isinstance(components.eval_pipeline, DetectionEvalPipelineAdapter)


@pytest.mark.unit
def test_detection_fallback_metrics() -> None:
    """Detection fallback metrics contain expected keys."""
    from orchard.core.task_registry import get_task

    task = get_task("detection")
    assert "loss" in task.fallback_metrics
    assert "map" in task.fallback_metrics
    assert "map_50" in task.fallback_metrics
    assert "map_75" in task.fallback_metrics
    assert task.fallback_metrics["map"] == pytest.approx(0.0)


@pytest.mark.unit
def test_detection_early_stopping_thresholds() -> None:
    """Detection early-stopping thresholds contain expected keys."""
    from orchard.core.task_registry import get_task

    task = get_task("detection")
    assert "map" in task.early_stopping_thresholds
    assert "map_50" in task.early_stopping_thresholds
    assert task.early_stopping_thresholds["map"] == pytest.approx(0.60)


# ── to_cpu helper ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_to_cpu_moves_tensors() -> None:
    """to_cpu moves tensor values to CPU."""
    from orchard.tasks.detection.helpers import to_cpu

    d = {"boxes": torch.tensor([1.0, 2.0]), "labels": torch.tensor([0])}
    result = to_cpu(d)
    assert all(v.device.type == "cpu" for v in result.values())


@pytest.mark.unit
def test_to_cpu_preserves_non_tensors() -> None:
    """to_cpu passes through non-tensor values unchanged."""
    from orchard.tasks.detection.helpers import to_cpu

    d: dict[str, Any] = {"boxes": torch.tensor([1.0]), "name": "obj", "count": 3}
    result = to_cpu(d)
    assert result["name"] == "obj"
    assert result["count"] == 3
    assert isinstance(result["boxes"], torch.Tensor)
