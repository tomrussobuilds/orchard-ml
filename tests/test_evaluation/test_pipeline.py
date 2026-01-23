"""
Smoke Tests for Evaluation Pipeline Module.

Quick coverage tests to validate pipeline orchestration.
These are minimal tests to boost coverage from 0% to ~20%.
"""

# Standard Imports
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third-Party Imports
import pytest

# Internal Imports
from orchard.evaluation.pipeline import run_final_evaluation


# PIPELINE: SMOKE TESTS
@pytest.mark.unit
@patch("orchard.evaluation.pipeline.evaluate_model")
@patch("orchard.evaluation.pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.pipeline.plot_training_curves")
@patch("orchard.evaluation.pipeline.show_predictions")
@patch("orchard.evaluation.pipeline.create_structured_report")
def test_run_final_evaluation_returns_tuple(
    mock_report,
    mock_show_pred,
    mock_curves,
    mock_confusion,
    mock_evaluate,
):
    """Test run_final_evaluation returns (macro_f1, test_acc) tuple."""
    # Mock evaluate_model to return test data
    mock_evaluate.return_value = (
        [0, 1, 2],  # all_preds
        [0, 1, 2],  # all_labels
        {"accuracy": 0.95, "auc": 0.98},  # test_metrics
        0.94,  # macro_f1
    )

    # Mock report
    mock_report_obj = MagicMock()
    mock_report.return_value = mock_report_obj

    # Create mocks
    mock_model = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/tmp/fig.png"))
    mock_paths.best_model_path = Path("/tmp/model.pth")
    mock_paths.logs = Path("/tmp/logs")
    mock_paths.final_report_path = Path("/tmp/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = False
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.model.name = "test_model"
    mock_cfg.dataset.resolution = 28

    result = run_final_evaluation(
        model=mock_model,
        test_loader=mock_loader,
        train_losses=[0.5, 0.3, 0.1],
        val_metrics_history=[{"accuracy": 0.8}, {"accuracy": 0.9}],
        class_names=["class0", "class1"],
        paths=mock_paths,
        cfg=mock_cfg,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    macro_f1, test_acc = result
    assert macro_f1 == 0.94
    assert test_acc == 0.95


@pytest.mark.unit
@patch("orchard.evaluation.pipeline.evaluate_model")
@patch("orchard.evaluation.pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.pipeline.plot_training_curves")
@patch("orchard.evaluation.pipeline.show_predictions")
@patch("orchard.evaluation.pipeline.create_structured_report")
def test_run_final_evaluation_calls_evaluate_model(
    mock_report,
    mock_show_pred,
    mock_curves,
    mock_confusion,
    mock_evaluate,
):
    """Test run_final_evaluation calls evaluate_model with correct params."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9}, 0.88)
    mock_report.return_value = MagicMock()

    mock_model = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/tmp/fig.png"))
    mock_paths.best_model_path = Path("/tmp/model.pth")
    mock_paths.logs = Path("/tmp/logs")
    mock_paths.final_report_path = Path("/tmp/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = True
    mock_cfg.dataset.metadata.is_anatomical = False
    mock_cfg.dataset.metadata.is_texture_based = True
    mock_cfg.model.name = "test"
    mock_cfg.dataset.resolution = 28

    run_final_evaluation(
        model=mock_model,
        test_loader=mock_loader,
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["class0"],
        paths=mock_paths,
        cfg=mock_cfg,
    )

    # Verify evaluate_model was called
    mock_evaluate.assert_called_once()
    call_kwargs = mock_evaluate.call_args.kwargs
    assert call_kwargs["use_tta"] is True
    assert call_kwargs["is_anatomical"] is False
    assert call_kwargs["is_texture_based"] is True


@pytest.mark.unit
@patch("orchard.evaluation.pipeline.evaluate_model")
@patch("orchard.evaluation.pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.pipeline.plot_training_curves")
@patch("orchard.evaluation.pipeline.show_predictions")
@patch("orchard.evaluation.pipeline.create_structured_report")
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

    mock_model = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/tmp/fig.png"))
    mock_paths.best_model_path = Path("/tmp/model.pth")
    mock_paths.logs = Path("/tmp/logs")
    mock_paths.final_report_path = Path("/tmp/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = False
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.model.name = "test"
    mock_cfg.dataset.resolution = 28

    run_final_evaluation(
        model=mock_model,
        test_loader=mock_loader,
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["class0"],
        paths=mock_paths,
        cfg=mock_cfg,
    )

    # Verify all visualizations were called
    mock_confusion.assert_called_once()
    mock_curves.assert_called_once()
    mock_show_pred.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.pipeline.evaluate_model")
@patch("orchard.evaluation.pipeline.plot_confusion_matrix")
@patch("orchard.evaluation.pipeline.plot_training_curves")
@patch("orchard.evaluation.pipeline.show_predictions")
@patch("orchard.evaluation.pipeline.create_structured_report")
def test_run_final_evaluation_creates_report(
    mock_report,
    mock_show_pred,
    mock_curves,
    mock_confusion,
    mock_evaluate,
):
    """Test run_final_evaluation creates and saves report."""
    mock_evaluate.return_value = ([0], [0], {"accuracy": 0.9, "auc": 0.95}, 0.88)
    mock_report_obj = MagicMock()
    mock_report.return_value = mock_report_obj

    mock_model = MagicMock()
    mock_loader = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_fig_path = MagicMock(return_value=Path("/tmp/fig.png"))
    mock_paths.best_model_path = Path("/tmp/model.pth")
    mock_paths.logs = Path("/tmp/logs")
    mock_paths.final_report_path = Path("/tmp/report.xlsx")

    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.use_tta = False
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.model.name = "test"
    mock_cfg.dataset.resolution = 28

    run_final_evaluation(
        model=mock_model,
        test_loader=mock_loader,
        train_losses=[0.1],
        val_metrics_history=[{"accuracy": 0.9}],
        class_names=["class0"],
        paths=mock_paths,
        cfg=mock_cfg,
    )

    # Verify report was created and saved
    mock_report.assert_called_once()
    mock_report_obj.save.assert_called_once_with(mock_paths.final_report_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
