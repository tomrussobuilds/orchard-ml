"""
Unit Test Suite for the Reporting & Experiment Summarization Module.

This suite validates the integrity of the TrainingReport Pydantic model,
the Excel export logic, and the factory function for report generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest

from orchard.evaluation import TrainingReport, create_structured_report


# MOCKS
@pytest.fixture
def mock_config() -> None:
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
    return cfg  # type: ignore


@pytest.fixture
def sample_report_data() -> None:
    """Provides a valid dictionary of data for TrainingReport instantiation."""
    return {  # type: ignore
        "architecture": "mini_cnn",
        "dataset": "PathMNIST",
        "best_val_metrics": {"accuracy": 0.95, "auc": 0.98, "f1": 0.94},
        "test_metrics": {"accuracy": 0.94, "auc": 0.97, "f1": 0.93},
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
def test_training_report_instantiation(sample_report_data: Any) -> None:
    """Test if TrainingReport correctly validates and stores input data."""
    report = TrainingReport(**sample_report_data)
    assert report.architecture == "mini_cnn"
    assert report.best_val_metrics["accuracy"] == pytest.approx(0.95)
    assert isinstance(report.timestamp, str)


@pytest.mark.unit
def test_to_vertical_df(sample_report_data: Any) -> None:
    """Test the conversion of the Pydantic model to a vertical DataFrame."""
    report = TrainingReport(**sample_report_data)
    df = report.to_vertical_df()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Parameter", "Value"]
    # Metric dicts are flattened: best_val_metrics.accuracy → best_val_accuracy
    assert "best_val_accuracy" in df["Parameter"].values
    assert 0.95 in df["Value"].values


@pytest.mark.unit
@patch("pandas.ExcelWriter")
@patch("pathlib.Path.mkdir")
def test_report_save_success(  # type: ignore
    mock_mkdir: MagicMock, mock_writer: MagicMock, sample_report_data
) -> None:
    """Test the save method to ensure ExcelWriter is called with correct parameters."""
    report = TrainingReport(**sample_report_data)
    test_path = Path("test_report.xlsx")

    report.save(test_path)

    mock_mkdir.assert_called_once()
    mock_writer.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.reporting.logger")
@patch("pathlib.Path.mkdir")
def test_report_save_failure(  # type: ignore
    mock_mkdir: MagicMock, mock_logger: MagicMock, sample_report_data
) -> None:
    """Test error handling: logger.error receives the exception."""
    report = TrainingReport(**sample_report_data)

    with patch.object(TrainingReport, "to_vertical_df", side_effect=ValueError("Write Error")):
        report.save(Path("error.xlsx"))

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0]
        assert isinstance(call_args[1], ValueError)


@pytest.mark.unit
def test_create_structured_report(mock_config: MagicMock) -> None:
    """Test the factory function that aggregates metrics into a TrainingReport."""
    val_metrics = [
        {"accuracy": 0.8, "auc": 0.85, "f1": 0.78},
        {"accuracy": 0.9, "auc": 0.92, "f1": 0.89},
    ]
    test_metrics = {"accuracy": 0.88, "auc": 0.91, "f1": 0.87}
    train_losses = [0.5, 0.4, 0.3]

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=train_losses,
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        arch_name=mock_config.architecture.name,
        dataset=mock_config.dataset,
        training=mock_config.training,
        aug_info="HFlip(True), Rotation(15°)",
    )

    assert isinstance(report, TrainingReport)
    assert report.best_val_metrics["accuracy"] == pytest.approx(0.9)
    assert report.epochs_trained == 3
    assert report.test_metrics["accuracy"] == pytest.approx(0.88)
    assert report.augmentations is not None
    assert "HFlip" in report.augmentations
    # Kill default task_type mutant: classification keeps domain flags
    assert report.is_texture_based is not None
    assert report.is_anatomical is not None


@pytest.mark.unit
def test_excel_formatting_logic(sample_report_data: Any) -> None:
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
def test_report_save_creates_xlsx_file(sample_report_data: Any, tmp_path: Path) -> None:
    """Test that save() actually creates an .xlsx file with correct suffix."""
    report = TrainingReport(**sample_report_data)

    test_path = tmp_path / "report"

    report.save(test_path)

    expected_path = tmp_path / "report.xlsx"
    assert expected_path.exists()
    assert expected_path.suffix == ".xlsx"

    # Verify xlsx has no index column (index=False)
    df = pd.read_excel(expected_path)
    assert list(df.columns) == ["Parameter", "Value"]


@pytest.mark.unit
def test_report_save_with_existing_xlsx_suffix(sample_report_data: Any, tmp_path: Path) -> None:
    """Test that save() handles path that already has .xlsx suffix."""
    report = TrainingReport(**sample_report_data)

    test_path = tmp_path / "report.xlsx"

    report.save(test_path)

    assert test_path.exists()


@pytest.mark.unit
def test_report_save_csv(sample_report_data: Any, tmp_path: Path) -> None:
    """Test save() creates a .csv with correct format (no index column)."""
    report = TrainingReport(**sample_report_data)
    path = tmp_path / "report"

    report.save(path, fmt="csv")

    csv_path = tmp_path / "report.csv"
    assert csv_path.exists()
    assert csv_path.suffix == ".csv"
    content = csv_path.read_text()
    assert "architecture" in content
    # index=False means first column is "Parameter", not row numbers
    first_line = content.strip().split("\n")[0]
    assert first_line == "Parameter,Value"


@pytest.mark.unit
def test_report_save_json(sample_report_data: Any, tmp_path: Path) -> None:
    """Test save() creates .json with orient='records' and indent=2."""
    import json

    report = TrainingReport(**sample_report_data)
    path = tmp_path / "report"

    report.save(path, fmt="json")

    json_path = tmp_path / "report.json"
    assert json_path.exists()
    assert json_path.suffix == ".json"
    data = json.loads(json_path.read_text())
    # orient="records" produces a list of dicts
    assert isinstance(data, list)
    assert isinstance(data[0], dict)
    # indent=2 means the output is formatted (not compact)
    raw = json_path.read_text()
    assert "\n" in raw  # indented output has newlines


@pytest.mark.unit
def test_create_structured_report_aug_info_none_fallback(mock_config: MagicMock) -> None:
    """Test aug_info=None falls back to 'N/A'."""
    val_metrics = [{"accuracy": 0.8, "auc": 0.85, "f1": 0.78}]
    test_metrics = {"accuracy": 0.88, "auc": 0.91, "f1": 0.87}

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=[0.5],
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        arch_name=mock_config.architecture.name,
        dataset=mock_config.dataset,
        training=mock_config.training,
        aug_info=None,
    )
    assert report.augmentations == "N/A"


@pytest.mark.unit
def test_create_structured_report_resolves_paths(mock_config: MagicMock, tmp_path: Path) -> None:
    """Test model_path and log_path are resolved to absolute paths."""
    model_file = tmp_path / "best.pth"
    model_file.touch()
    log_file = tmp_path / "run.log"
    log_file.touch()

    val_metrics = [{"accuracy": 0.8, "auc": 0.85, "f1": 0.78}]
    test_metrics = {"accuracy": 0.88, "auc": 0.91, "f1": 0.87}

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=[0.5],
        best_path=model_file,
        log_path=log_file,
        arch_name=mock_config.architecture.name,
        dataset=mock_config.dataset,
        training=mock_config.training,
    )
    assert report.model_path == str(model_file.resolve())
    assert report.log_path == str(log_file.resolve())


@pytest.mark.unit
def test_report_save_creates_nested_parent_dirs(sample_report_data: Any, tmp_path: Path) -> None:
    """Test save() creates nested parent directories (parents=True)."""
    report = TrainingReport(**sample_report_data)
    path = tmp_path / "deep" / "nested" / "report"

    report.save(path, fmt="csv")

    assert (tmp_path / "deep" / "nested" / "report.csv").exists()


@pytest.mark.unit
def test_create_structured_report_handles_empty_val_metrics(mock_config: MagicMock) -> None:
    """Test create_structured_report with empty validation metrics."""
    val_metrics = []  # type: ignore
    test_metrics = {"accuracy": 0.88, "auc": 0.91, "f1": 0.87}
    train_losses = [0.5]

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=train_losses,
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        arch_name=mock_config.architecture.name,
        dataset=mock_config.dataset,
        training=mock_config.training,
        aug_info="N/A",
    )
    assert isinstance(report, TrainingReport)
    assert report.best_val_metrics == {}


@pytest.mark.unit
def test_create_structured_report_filters_nan_auc(mock_config: MagicMock) -> None:
    """Test best_val_metrics filters NaN values, picks valid max."""
    val_metrics = [
        {"accuracy": 0.8, "auc": float("nan"), "f1": 0.78},
        {"accuracy": 0.9, "auc": 0.92, "f1": 0.89},
        {"accuracy": 0.85, "auc": float("nan"), "f1": 0.82},
    ]
    test_metrics = {"accuracy": 0.88, "auc": 0.91, "f1": 0.87}

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=[0.5, 0.4, 0.3],
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        arch_name=mock_config.architecture.name,
        dataset=mock_config.dataset,
        training=mock_config.training,
        aug_info="N/A",
    )

    assert report.best_val_metrics["auc"] == pytest.approx(0.92)


@pytest.mark.unit
def test_create_structured_report_all_nan_auc(mock_config: MagicMock) -> None:
    """Test best_val_metrics defaults to 0.0 when all AUC values are NaN."""
    val_metrics = [
        {"accuracy": 0.8, "auc": float("nan"), "f1": 0.78},
        {"accuracy": 0.9, "auc": float("nan"), "f1": 0.89},
    ]
    test_metrics = {"accuracy": 0.88, "auc": 0.91, "f1": 0.87}

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=[0.5, 0.4],
        best_path=Path("/models/best.pth"),
        log_path=Path("/logs/run.log"),
        arch_name=mock_config.architecture.name,
        dataset=mock_config.dataset,
        training=mock_config.training,
        aug_info="N/A",
    )

    assert report.best_val_metrics["auc"] == pytest.approx(0.0)


@pytest.mark.unit
def test_report_save_default_fmt_is_xlsx(sample_report_data: Any, tmp_path: Path) -> None:
    """Test save() defaults to xlsx format when fmt is not specified."""
    report = TrainingReport(**sample_report_data)
    path = tmp_path / "report"

    report.save(path)

    assert (tmp_path / "report.xlsx").exists()


@pytest.mark.unit
@patch("orchard.evaluation.reporting.logger")
def test_report_save_xlsx_no_error_logged(  # type: ignore
    mock_logger: MagicMock, sample_report_data, tmp_path: Path
) -> None:
    """Test save() does not log errors on success (catches _apply_excel_formatting regressions)."""
    report = TrainingReport(**sample_report_data)
    path = tmp_path / "report"

    report.save(path)

    mock_logger.error.assert_not_called()


@pytest.mark.unit
def test_report_save_json_indent(sample_report_data: Any, tmp_path: Path) -> None:
    """Test save() json uses indent=2 specifically (not 3)."""
    report = TrainingReport(**sample_report_data)
    path = tmp_path / "report"

    report.save(path, fmt="json")

    raw = (tmp_path / "report.json").read_text()
    lines = raw.split("\n")
    # indent=2: inner keys should be indented with exactly 4 spaces (2 for array + 2 for object)
    key_lines = [line for line in lines if '"Parameter"' in line]
    assert len(key_lines) > 0
    assert all(line.startswith("    ") for line in key_lines)
    # With indent=3 those would start with 6 spaces instead
    assert not all(line.startswith("      ") for line in key_lines)


# ── Detection report tests ───────────────────────────────────────────────────


@pytest.mark.unit
def test_to_vertical_df_skips_none_fields() -> None:
    """None fields are excluded from the vertical DataFrame."""
    report = TrainingReport(
        architecture="fasterrcnn",
        dataset="pennfudan",
        best_val_metrics={"map": 0.85},
        test_metrics={"map": 0.80},
        epochs_trained=10,
        learning_rate=0.005,
        batch_size=4,
        seed=42,
        normalization="Mean: (0.5,) | Std: (0.2,)",
        model_path="/tmp/best.pth",
        log_path="/tmp/session.log",
        # classification-only fields left as None
    )
    df = report.to_vertical_df()
    params = set(df["Parameter"].tolist())
    assert "is_texture_based" not in params
    assert "is_anatomical" not in params
    assert "use_tta" not in params
    assert "augmentations" not in params
    # Task-agnostic fields present (after None fields — kills continue→break mutant)
    assert "architecture" in params
    assert "best_val_map" in params
    assert "test_map" in params
    assert "normalization" in params
    assert "model_path" in params
    assert "seed" in params


@pytest.mark.unit
def test_create_structured_report_detection_excludes_classification_fields(
    mock_config: MagicMock,
) -> None:
    """Detection report excludes is_texture_based, is_anatomical, use_tta, augmentations."""
    val_metrics = [{"map": 0.7, "loss": 0.0}]
    test_metrics = {"map": 0.65, "map_50": 0.90}

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=[0.5, 0.3],
        best_path=Path("/tmp/best.pth"),
        log_path=Path("/tmp/session.log"),
        arch_name="fasterrcnn",
        dataset=mock_config.dataset,
        training=mock_config.training,
        task_type="detection",
    )

    assert report.is_texture_based is None
    assert report.is_anatomical is None
    assert report.use_tta is None
    assert report.augmentations is None
    # Loss sentinel dropped
    assert "loss" not in report.best_val_metrics


@pytest.mark.unit
def test_create_structured_report_classification_keeps_all_fields(
    mock_config: MagicMock,
) -> None:
    """Classification report keeps all fields."""
    val_metrics = [{"accuracy": 0.9, "auc": 0.92}]
    test_metrics = {"accuracy": 0.88}

    report = create_structured_report(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=[0.5],
        best_path=Path("/tmp/best.pth"),
        log_path=Path("/tmp/session.log"),
        arch_name="resnet_18",
        dataset=mock_config.dataset,
        training=mock_config.training,
        aug_info="HFlip(0.5)",
        task_type="classification",
    )

    assert report.is_texture_based is not None
    assert report.is_anatomical is not None
    assert report.use_tta is not None
    assert report.augmentations == "HFlip(0.5)"


# ENTRY POINT
if __name__ == "__main__":
    pytest.main([__file__, "-vv", "-m", "unit"])
