"""
Unit Test Suite for the Reporting & Experiment Summarization Module.

This suite validates the integrity of the TrainingReport Pydantic model,
the Excel export logic, and the factory function for report generation.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pandas as pd

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import pytest

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from orchard.evaluation import TrainingReport, create_structured_report

# =========================================================================== #
#                                    MOCKS                                    #
# =========================================================================== #


@pytest.fixture
def mock_config():
    """Provides a mocked Config object with necessary nested attributes."""
    cfg = MagicMock()
    cfg.model.name = "ResNet18"
    cfg.dataset.dataset_name = "PathMNIST"
    cfg.dataset.metadata.is_texture_based = True
    cfg.dataset.metadata.is_anatomical = False
    cfg.dataset.metadata.normalization_info = "ImageNet"
    cfg.training.use_tta = False
    cfg.training.learning_rate = 0.001
    cfg.training.batch_size = 32
    cfg.training.seed = 42
    cfg.augmentation.model_dump.return_value = {"horizontal_flip": True, "rotation": 15}
    return cfg


@pytest.fixture
def sample_report_data():
    """Provides a valid dictionary of data for TrainingReport instantiation."""
    return {
        "model": "ResNet18",
        "dataset": "PathMNIST",
        "best_val_accuracy": 0.95,
        "best_val_auc": 0.98,
        "test_accuracy": 0.94,
        "test_auc": 0.97,
        "test_macro_f1": 0.93,
        "is_texture_based": True,
        "is_anatomical": False,
        "use_tta": False,
        "epochs_trained": 50,
        "learning_rate": 0.001,
        "batch_size": 32,
        "seed": 42,
        "augmentations": "Flip: True",
        "normalization": "Standard",
        "model_path": "/tmp/model.pth",
        "log_path": "/tmp/train.log",
    }


# =========================================================================== #
#                                 UNIT TESTS                                  #
# =========================================================================== #


@pytest.mark.unit
def test_training_report_instantiation(sample_report_data):
    """Test if TrainingReport correctly validates and stores input data."""
    report = TrainingReport(**sample_report_data)
    assert report.model == "ResNet18"
    assert report.best_val_accuracy == 0.95
    assert isinstance(report.timestamp, str)


@pytest.mark.unit
def test_to_vertical_df(sample_report_data):
    """Test the conversion of the Pydantic model to a vertical DataFrame."""
    report = TrainingReport(**sample_report_data)
    df = report.to_vertical_df()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Parameter", "Value"]
    assert "best_val_accuracy" in df["Parameter"].values
    assert 0.95 in df["Value"].values


@pytest.mark.unit
@patch("pandas.ExcelWriter")
@patch("pathlib.Path.mkdir")
def test_report_save_success(mock_mkdir, mock_writer, sample_report_data):
    """Test the save method to ensure ExcelWriter is called with correct parameters."""
    report = TrainingReport(**sample_report_data)
    test_path = Path("test_report.xlsx")

    report.save(test_path)

    # Check if directory was created
    mock_mkdir.assert_called_once()
    # Check if ExcelWriter was instantiated
    mock_writer.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.reporting.logger")
def test_report_save_failure(mock_logger, sample_report_data):
    """Test error handling when Excel saving fails."""
    report = TrainingReport(**sample_report_data)

    with patch.object(TrainingReport, "to_vertical_df", side_effect=Exception("Write Error")):
        report.save(Path("error.xlsx"))

        mock_logger.error.assert_called()


@pytest.mark.unit
def test_create_structured_report(mock_config):
    """Test the factory function that aggregates metrics into a TrainingReport."""
    val_metrics = [{"accuracy": 0.8, "auc": 0.85}, {"accuracy": 0.9, "auc": 0.92}]
    test_metrics = {"accuracy": 0.88, "auc": 0.91}
    train_losses = [0.5, 0.4, 0.3]

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        macro_f1=0.87,
        train_losses=train_losses,
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        cfg=mock_config,
    )

    assert isinstance(report, TrainingReport)
    assert report.best_val_accuracy == 0.9
    assert report.epochs_trained == 3
    assert report.test_accuracy == 0.88
    assert "Horizontal flip" in report.augmentations


@pytest.mark.unit
def test_excel_formatting_logic(sample_report_data):
    """Test the internal _apply_excel_formatting helper using mocks for XlsxWriter."""
    report = TrainingReport(**sample_report_data)
    df = report.to_vertical_df()

    # Mock XlsxWriter components
    mock_writer = MagicMock()
    mock_workbook = MagicMock()
    mock_worksheet = MagicMock()

    mock_writer.book = mock_workbook
    mock_writer.sheets = {"Detailed Report": mock_worksheet}

    # Trigger internal method
    report._apply_excel_formatting(mock_writer, df)

    # Verify column setting and header writing
    assert mock_worksheet.write.called
    mock_worksheet.set_column.assert_any_call("A:A", 25, ANY)
    mock_worksheet.set_column.assert_any_call("B:B", 70)


# =========================================================================== #
#                                 ENTRY POINT                                 #
# =========================================================================== #

if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-m", "unit"])
