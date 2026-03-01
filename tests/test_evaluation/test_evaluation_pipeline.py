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
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
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
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
    """Test run_final_evaluation calls evaluate_model with correct params."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9}, 0.88)
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
    call_kwargs = mock_evaluate.call_args.kwargs
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
    mock_report,
    mock_show_pred,
    mock_curves,
    mock_confusion,
    mock_evaluate,
):
    """Test run_final_evaluation calls all visualization functions."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9}, 0.88)
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
    mock_curves.assert_called_once()
    mock_show_pred.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_run_final_evaluation_creates_report(
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
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
    mock_report,
    mock_show_pred,
    mock_curves,
    mock_confusion,
    mock_evaluate,
):
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
    mock_report,
    mock_show_pred,
    mock_curves,
    mock_confusion,
    mock_evaluate,
):
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
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
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
def test_test_auc_fallback_to_nan(
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
    """Verify test_auc falls back to NaN when 'auc' is missing from test_metrics."""
    import math

    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9}, 0.88)  # no "auc" key
    mock_report.return_value = MagicMock()

    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/mock/fig.png"))
    mock_paths.best_model_path = Path("/mock/model.pth")
    mock_paths.logs = Path("/mock/logs")
    mock_paths.final_report_path = Path("/mock/report.xlsx")

    mock_cfg = MagicMock()

    _, _, test_auc = run_final_evaluation(
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

    assert math.isnan(test_auc)


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_tracker_receives_metrics(
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
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

    mock_tracker.log_test_metrics.assert_called_once_with(test_acc=0.92, macro_f1=0.91)


@pytest.mark.unit
@patch("orchard.evaluation.evaluation_pipeline.evaluate_model")
@patch("orchard.evaluation.evaluation_pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.evaluation_pipeline.plot_training_curves")
@patch("orchard.evaluation.evaluation_pipeline.show_predictions")
@patch("orchard.evaluation.evaluation_pipeline.create_structured_report")
def test_tracker_not_called_when_none(
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
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
    mock_report,
    _mock_show_pred,
    mock_curves,
    _mock_confusion,
    mock_evaluate,
):
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
        val_metrics_history=history,
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
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
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
    mock_report,
    _mock_show_pred,
    _mock_curves,
    _mock_confusion,
    mock_evaluate,
):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
