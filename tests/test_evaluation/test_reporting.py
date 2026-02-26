"""
Unit Test Suite for the Reporting & Experiment Summarization Module.

This suite validates the integrity of the TrainingReport Pydantic model,
the Excel export logic, and the factory function for report generation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from orchard.evaluation import TrainingReport, create_structured_report


# MOCKS
@pytest.fixture
def mock_config():
    """Provides a mocked Config object with necessary nested attributes."""
    cfg = MagicMock()
    cfg.architecture.name = "mini_cnn"
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
        "architecture": "mini_cnn",
        "dataset": "PathMNIST",
        "best_val_accuracy": 0.95,
        "best_val_auc": 0.98,
        "best_val_f1": 0.94,
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
        "model_path": "/mock/model.pth",
        "log_path": "/mock/train.log",
    }


# UNIT TESTS
@pytest.mark.unit
def test_training_report_instantiation(sample_report_data):
    """Test if TrainingReport correctly validates and stores input data."""
    report = TrainingReport(**sample_report_data)
    assert report.architecture == "mini_cnn"
    assert report.best_val_accuracy == pytest.approx(0.95)
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

    mock_mkdir.assert_called_once()
    mock_writer.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.reporting.logger")
@patch("pathlib.Path.mkdir")
def test_report_save_failure(mock_mkdir, mock_logger, sample_report_data):
    """Test error handling when Excel saving fails."""
    report = TrainingReport(**sample_report_data)

    with patch.object(TrainingReport, "to_vertical_df", side_effect=ValueError("Write Error")):
        report.save(Path("error.xlsx"))

        mock_logger.error.assert_called()


@pytest.mark.unit
def test_create_structured_report(mock_config):
    """Test the factory function that aggregates metrics into a TrainingReport."""
    val_metrics = [
        {"accuracy": 0.8, "auc": 0.85, "f1": 0.78},
        {"accuracy": 0.9, "auc": 0.92, "f1": 0.89},
    ]
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
        aug_info="HFlip(True), Rotation(15°)",
    )

    assert isinstance(report, TrainingReport)
    assert report.best_val_accuracy == pytest.approx(0.9)
    assert report.epochs_trained == 3
    assert report.test_accuracy == pytest.approx(0.88)
    assert "HFlip" in report.augmentations


@pytest.mark.unit
def test_excel_formatting_logic(sample_report_data):
    """Test the internal _apply_excel_formatting helper using mocks for XlsxWriter."""
    report = TrainingReport(**sample_report_data)
    df = report.to_vertical_df()

    mock_writer = MagicMock()
    mock_workbook = MagicMock()
    mock_worksheet = MagicMock()

    mock_writer.book = mock_workbook
    mock_writer.sheets = {"Detailed Report": mock_worksheet}

    report._apply_excel_formatting(mock_writer, df)

    assert mock_worksheet.write.called
    mock_worksheet.set_column.assert_any_call("A:A", 25, ANY)
    mock_worksheet.set_column.assert_any_call("B:B", 70)


@pytest.mark.unit
def test_report_save_creates_xlsx_file(sample_report_data, tmp_path):
    """Test that save() actually creates an .xlsx file with correct suffix."""
    report = TrainingReport(**sample_report_data)

    test_path = tmp_path / "report"

    report.save(test_path)

    expected_path = tmp_path / "report.xlsx"
    assert expected_path.exists()
    assert expected_path.suffix == ".xlsx"


@pytest.mark.unit
def test_report_save_with_existing_xlsx_suffix(sample_report_data, tmp_path):
    """Test that save() handles path that already has .xlsx suffix."""
    report = TrainingReport(**sample_report_data)

    test_path = tmp_path / "report.xlsx"

    report.save(test_path)

    assert test_path.exists()


@pytest.mark.unit
def test_report_save_csv(sample_report_data, tmp_path):
    """Test save() creates a .csv file when fmt='csv'."""
    report = TrainingReport(**sample_report_data)
    path = tmp_path / "report"

    report.save(path, fmt="csv")

    csv_path = tmp_path / "report.csv"
    assert csv_path.exists()
    assert csv_path.suffix == ".csv"
    content = csv_path.read_text()
    assert "architecture" in content


@pytest.mark.unit
def test_report_save_json(sample_report_data, tmp_path):
    """Test save() creates a .json file when fmt='json'."""
    report = TrainingReport(**sample_report_data)
    path = tmp_path / "report"

    report.save(path, fmt="json")

    json_path = tmp_path / "report.json"
    assert json_path.exists()
    assert json_path.suffix == ".json"
    content = json_path.read_text()
    assert "architecture" in content


@pytest.mark.unit
def test_create_structured_report_handles_empty_val_metrics(mock_config):
    """Test create_structured_report with empty validation metrics."""
    val_metrics = []
    test_metrics = {"accuracy": 0.88, "auc": 0.91}
    train_losses = [0.5]

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        macro_f1=0.87,
        train_losses=train_losses,
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        cfg=mock_config,
        aug_info="N/A",
    )
    assert isinstance(report, TrainingReport)


@pytest.mark.unit
def test_create_structured_report_filters_nan_auc(mock_config):
    """Test best_val_auc filters NaN values, picks valid max."""
    val_metrics = [
        {"accuracy": 0.8, "auc": float("nan"), "f1": 0.78},
        {"accuracy": 0.9, "auc": 0.92, "f1": 0.89},
        {"accuracy": 0.85, "auc": float("nan"), "f1": 0.82},
    ]
    test_metrics = {"accuracy": 0.88, "auc": 0.91}

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        macro_f1=0.87,
        train_losses=[0.5, 0.4, 0.3],
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        cfg=mock_config,
        aug_info="N/A",
    )

    assert report.best_val_auc == pytest.approx(0.92)


@pytest.mark.unit
def test_create_structured_report_all_nan_auc(mock_config):
    """Test best_val_auc defaults to 0.0 when all AUC values are NaN."""
    val_metrics = [
        {"accuracy": 0.8, "auc": float("nan"), "f1": 0.78},
        {"accuracy": 0.9, "auc": float("nan"), "f1": 0.89},
    ]
    test_metrics = {"accuracy": 0.88, "auc": 0.91}

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        macro_f1=0.87,
        train_losses=[0.5, 0.4],
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        cfg=mock_config,
        aug_info="N/A",
    )

    assert report.best_val_auc == pytest.approx(0.0)


# ENTRY POINT
if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-m", "unit"])
