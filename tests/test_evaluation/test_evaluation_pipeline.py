"""
Smoke Tests for Evaluation Pipeline Module.

Quick coverage tests to validate pipeline orchestration.
These are minimal tests to boost coverage from 0% to ~20%.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchard.evaluation.evaluation_pipeline import run_final_evaluation


# PIPELINE: SMOKE TESTS
@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_run_final_evaluation_returns_tuple(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Test run_final_evaluation returns (macro_f1, test_acc, test_auc) tuple."""
    mock_evaluate.return_value = (
        [0, 1, 2],
        [0, 1, 2],
        {"accuracy": 0.95, "auc": 0.98},
        0.94,
    )

    # Mock report
    mock_report_obj = MagicMock()
    mock_report.return_value = mock_report_obj

    # Create mocks
    mock_architecture = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = False
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.effective_is_anatomical = True
    mock_cfg.dataset.effective_is_texture_based = False
    mock_cfg.architecture.name = "test_architecture"
    mock_cfg.dataset.resolution = 28

    result = run_final_evaluation(
        model=mock_architecture,
        test_loader=mock_loader,
        train_losses=[0.5, 0.3, 0.1],
        val_metrics_history=[{"accuracy": 0.8}, {"accuracy": 0.9}],
        class_names=["class0", "class1"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name=mock_cfg.architecture.name,
    )

    assert isinstance(result, tuple)
    assert len(result) == 3
    macro_f1, test_acc, test_auc = result
    assert macro_f1 == pytest.approx(0.94)
    assert test_acc == pytest.approx(0.95)
    assert test_auc == pytest.approx(0.98)


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_run_final_evaluation_calls_evaluate_model(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Test run_final_evaluation calls evaluate_model with correct params."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_architecture = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = True
    mock_cfg.dataset.metadata.is_anatomical = False
    mock_cfg.dataset.metadata.is_texture_based = True
    mock_cfg.dataset.effective_is_anatomical = False
    mock_cfg.dataset.effective_is_texture_based = True
    mock_cfg.architecture.name = "test"
    mock_cfg.dataset.resolution = 28

    run_final_evaluation(
        model=mock_architecture,
        test_loader=mock_loader,
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["class0"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name=mock_cfg.architecture.name,
    )

    mock_evaluate.assert_called_once()
    call_args = mock_evaluate.call_args
    # Positional args: model, test_loader
    assert call_args[0][0] is mock_architecture
    assert call_args[0][1] is mock_loader
    # Keyword args
    call_kwargs = call_args.kwargs
    assert call_kwargs["device"] is not None  # resolved from model.parameters()
    assert call_kwargs["use_tta"] is True
    assert call_kwargs["is_anatomical"] is False
    assert call_kwargs["is_texture_based"] is True


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_run_final_evaluation_calls_visualizations(
    mock_report: MagicMock,
    mock_show_pred: MagicMock,
    mock_curves: MagicMock,
    mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Test run_final_evaluation calls all visualization functions."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_architecture = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = False
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.effective_is_anatomical = True
    mock_cfg.dataset.effective_is_texture_based = False
    mock_cfg.architecture.name = "test"
    mock_cfg.dataset.resolution = 28

    run_final_evaluation(
        model=mock_architecture,
        test_loader=mock_loader,
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["class0"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name=mock_cfg.architecture.name,
    )

    mock_confusion.assert_called_once()
    cm_kw = mock_confusion.call_args.kwargs
    assert cm_kw["all_labels"] is not None
    assert cm_kw["all_preds"] is not None
    assert cm_kw["classes"] == ["class0"]
    assert cm_kw["ctx"] is not None

    mock_curves.assert_called_once()
    curves_kw = mock_curves.call_args.kwargs
    assert curves_kw["ctx"] is not None

    mock_show_pred.assert_called_once()
    sp_kw = mock_show_pred.call_args.kwargs
    assert sp_kw["model"] is mock_architecture
    assert sp_kw["loader"] is mock_loader
    assert sp_kw["device"] is not None
    assert sp_kw["classes"] == ["class0"]
    assert sp_kw["ctx"] is not None


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_run_final_evaluation_creates_report(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Test run_final_evaluation creates and saves report."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report_obj = MagicMock()
    mock_report.return_value = mock_report_obj

    mock_architecture = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = False
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.effective_is_anatomical = True
    mock_cfg.dataset.effective_is_texture_based = False
    mock_cfg.architecture.name = "test_architecture"
    mock_cfg.dataset.resolution = 28

    run_final_evaluation(
        model=mock_architecture,
        test_loader=mock_loader,
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["class0"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name=mock_cfg.architecture.name,
    )

    mock_report.assert_called_once()
    mock_report_obj.save.assert_called_once_with(
        mock_paths.final_report_path, fmt=mock_cfg.evaluation.report_format
    )


# MUTATION-RESILIENT: CONDITIONAL LOGIC, EXACT VALUES, TRACKER
@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_confusion_matrix_skipped_when_disabled(
    mock_report: MagicMock,
    mock_show_pred: MagicMock,
    mock_curves: MagicMock,
    mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify confusion matrix is NOT called when save_confusion_matrix=False."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_model = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.evaluation.save_confusion_matrix = False
    mock_cfg.evaluation.save_predictions_grid = True

    run_final_evaluation(
        model=mock_model,
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
    )

    mock_confusion.assert_not_called()
    mock_show_pred.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_predictions_grid_skipped_when_disabled(
    mock_report: MagicMock,
    mock_show_pred: MagicMock,
    mock_curves: MagicMock,
    mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify predictions grid is NOT called when save_predictions_grid=False."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_model = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.evaluation.save_confusion_matrix = True
    mock_cfg.evaluation.save_predictions_grid = False

    run_final_evaluation(
        model=mock_model,
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
    )

    mock_show_pred.assert_not_called()
    mock_confusion.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_arch_tag_replaces_slash(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify arch_name with '/' is converted to '_' in figure paths."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.evaluation.save_confusion_matrix = True
    mock_cfg.evaluation.save_predictions_grid = True

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="timm/mobilenet_v3",
    )

    # All get_fig_path calls must use "timm_mobilenet_v3" (underscore, not slash)
    for call in mock_paths.get_fig_path.call_args_list:
        arg = call[0][0]
        assert "/" not in arg
        assert "timm_mobilenet_v3" in arg


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_tracker_receives_metrics(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify tracker.log_test_metrics is called with correct values."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.92, "auc": 0.97}, 0.91)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_tracker = MagicMock()

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
        tracker=mock_tracker,
    )

    mock_tracker.log_test_metrics.assert_called_once_with(
        {"accuracy": 0.92, "f1": 0.91, "auc": 0.97}
    )


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_tracker_not_called_when_none(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify tracker is not accessed when tracker=None."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()

    # Should not raise — tracker is None
    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
        tracker=None,
    )


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_val_acc_list_extraction(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify val_acc_list is correctly extracted from val_metrics_history."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()

    history = [
        {"accuracy": 0.7, "f1": 0.6},
        {"accuracy": 0.85, "f1": 0.8},
        {"accuracy": 0.92, "f1": 0.9},
    ]

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.5, 0.3, 0.1],
        val_metrics_history=history,  # type: ignore
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
    )

    call_kwargs = mock_curves.call_args.kwargs
    assert call_kwargs["val_accuracies"] == [0.7, 0.85, 0.92]
    assert call_kwargs["train_losses"] == [0.5, 0.3, 0.1]


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_final_log_path(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify create_structured_report receives log_path = paths.logs / 'session.log'."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
    )

    call_kwargs = mock_report.call_args.kwargs
    assert call_kwargs["log_path"] == Path("/mock/logs/session.log")


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_evaluate_model_receives_resolution(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify evaluate_model receives the dataset resolution kwarg."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.dataset.resolution = 224

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
    )

    call_kwargs = mock_evaluate.call_args.kwargs
    assert call_kwargs["resolution"] == 224


# ---------------------------------------------------------------------------
# Mutation-killing tests: PlotContext, exact kwargs, return order
# ---------------------------------------------------------------------------


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
@patch("orchard.evaluation.evaluation_pipeline.PlotContext")
def test_plot_context_receives_exact_kwargs(
    mock_ctx_cls: MagicMock,
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify PlotContext is constructed with ALL exact kwargs from config."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()
    mock_ctx_cls.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset.mean = (0.5,)
    mock_cfg.dataset.std = (0.2,)
    mock_cfg.dataset.effective_is_anatomical = True
    mock_cfg.dataset.effective_is_texture_based = False
    mock_cfg.training.use_tta = False
    mock_cfg.evaluation.fig_dpi = 150
    mock_cfg.evaluation.plot_style = "seaborn"
    mock_cfg.evaluation.cmap_confusion = "Blues"
    mock_cfg.evaluation.grid_cols = 4
    mock_cfg.evaluation.n_samples = 16
    mock_cfg.evaluation.fig_size_predictions = (10, 8)

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="resnet_18",
    )

    ctx_kwargs = mock_ctx_cls.call_args.kwargs
    assert ctx_kwargs["arch_name"] == "resnet_18"
    assert ctx_kwargs["resolution"] == 28
    assert ctx_kwargs["fig_dpi"] == 150
    assert ctx_kwargs["plot_style"] == "seaborn"
    assert ctx_kwargs["cmap_confusion"] == "Blues"
    assert ctx_kwargs["grid_cols"] == 4
    assert ctx_kwargs["n_samples"] == 16
    assert ctx_kwargs["fig_size_predictions"] == (10, 8)
    assert ctx_kwargs["mean"] == (0.5,)
    assert ctx_kwargs["std"] == (0.2,)
    assert ctx_kwargs["use_tta"] is False
    assert ctx_kwargs["is_anatomical"] is True
    assert ctx_kwargs["is_texture_based"] is False


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_exact_figure_path_strings(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify get_fig_path is called with exact f-string values."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.dataset.resolution = 28
    mock_cfg.evaluation.save_confusion_matrix = True
    mock_cfg.evaluation.save_predictions_grid = True

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="mini_cnn",
    )

    fig_calls = [c[0][0] for c in mock_paths.get_fig_path.call_args_list]
    assert "confusion_matrix_mini_cnn_28.png" in fig_calls
    assert "training_curves_mini_cnn_28.png" in fig_calls
    assert "sample_predictions_mini_cnn_28.png" in fig_calls


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_evaluate_model_exact_kwargs(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify evaluate_model is called with ALL expected kwargs."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.dataset.resolution = 224
    mock_cfg.dataset.effective_is_anatomical = True
    mock_cfg.dataset.effective_is_texture_based = False
    mock_cfg.training.use_tta = True

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
    )

    kw = mock_evaluate.call_args.kwargs
    assert kw["use_tta"] is True
    assert kw["is_anatomical"] is True
    assert kw["is_texture_based"] is False
    assert kw["aug_cfg"] is mock_cfg.augmentation
    assert kw["resolution"] == 224


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_create_report_exact_kwargs(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify create_structured_report receives ALL expected kwargs."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report_obj = MagicMock()
    mock_report.return_value = mock_report_obj

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.dataset.resolution = 28

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.5, 0.3],
        val_metrics_history=[{"accuracy": 0.8}, {"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test_arch",
        aug_info="custom_aug",
    )

    kw = mock_report.call_args.kwargs
    assert kw["val_metrics"] == [{"accuracy": 0.8}, {"accuracy": 0.9}]
    assert kw["test_metrics"] == {"accuracy": 0.9, "auc": 0.95}
    assert kw["macro_f1"] == pytest.approx(0.88)
    assert kw["train_losses"] == [0.5, 0.3]
    assert kw["best_path"] == Path("/mock/model.pth")
    assert kw["log_path"] == Path("/mock/logs/session.log")
    assert kw["arch_name"] == "test_arch"
    assert kw["dataset"] is mock_cfg.dataset
    assert kw["training"] is mock_cfg.training
    assert kw["aug_info"] == "custom_aug"


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_aug_info_default_is_na(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify aug_info defaults to 'N/A' when not specified."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()

    run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
    )

    kw = mock_report.call_args.kwargs
    assert kw["aug_info"] == "N/A"


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_return_tuple_order(
    mock_report: MagicMock,
    _mock_show_pred: MagicMock,
    _mock_curves: MagicMock,
    _mock_confusion: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    """Verify return order is (macro_f1, test_acc, test_auc)."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.85, "auc": 0.91}, 0.82)
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()

    result = run_final_evaluation(
        model=MagicMock(),
        test_loader=MagicMock(),
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["a"],
        paths=mock_paths,
        training=mock_cfg.training,
        dataset=mock_cfg.dataset,
        augmentation=mock_cfg.augmentation,
        evaluation=mock_cfg.evaluation,
        arch_name="test",
    )

    # Return is (macro_f1, test_acc, test_auc)
    assert result[0] == pytest.approx(0.82)  # macro_f1
    assert result[1] == pytest.approx(0.85)  # test_acc
    assert result[2] == pytest.approx(0.91)  # test_auc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
