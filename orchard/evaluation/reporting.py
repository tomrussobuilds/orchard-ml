"""
Reporting & Experiment Summarization Module.

This module orchestrates the generation of human-readable artifacts following
the completion of a training pipeline. It leverages Pydantic for strict
validation of experiment results and transforms raw metrics into structured,
professionally formatted Excel summaries.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..core import LOGGER_NAME, DatasetConfig, LogStyle, TrainingConfig
from ..core.paths import METRIC_ACCURACY, METRIC_AUC, METRIC_F1

logger = logging.getLogger(LOGGER_NAME)


# EXCEL REPORTS
class TrainingReport(BaseModel):
    """
    Validated data container for summarizing a complete training experiment.

    This model serves as a Schema for the final experimental metadata. It stores
    hardware, hyperparameter, and performance states to ensure full reproducibility
    and traceability of the medical imaging pipeline.

    Attributes:
        timestamp (str): ISO formatted execution time.
        model (str): Identifier of the architecture used.
        dataset (str): Name of the dataset.
        best_val_accuracy (float): Peak accuracy achieved on validation set.
        best_val_f1 (float): Peak macro-averaged F1 on validation set.
        test_accuracy (float): Final accuracy on the unseen test set.
        test_macro_f1 (float): Macro-averaged F1 score (key for imbalanced data).
        is_texture_based (bool): Whether texture-preserving logic was applied.
        is_anatomical (bool): Whether anatomical orientation constraints were enforced.
        use_tta (bool): Indicates if Test-Time Augmentation was active.
        epochs_trained (int): Total number of optimization cycles completed.
        learning_rate (float): Initial learning rate used by the optimizer.
        batch_size (int): Samples processed per iteration.
        augmentations (str): Descriptive string of the transformation pipeline.
        normalization (str): Mean/Std statistics applied to the input tensors.
        model_path (str): Absolute path to the best saved checkpoint.
        log_path (str): Absolute path to the session execution log.
        seed (int): Global RNG seed for experiment replication.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Model & Data Identity
    architecture: str
    dataset: str

    # Core Metrics
    best_val_accuracy: float
    best_val_auc: float
    best_val_f1: float
    test_accuracy: float
    test_auc: float
    test_macro_f1: float

    # Domain Logic Flags
    is_texture_based: bool
    is_anatomical: bool
    use_tta: bool

    # Hyperparameters
    epochs_trained: int
    learning_rate: float
    batch_size: int
    seed: int

    # Metadata Strings
    augmentations: str
    normalization: str
    model_path: str
    log_path: str

    def to_vertical_df(self) -> pd.DataFrame:
        """
        Converts the Pydantic model into a vertical pandas DataFrame.

        Returns:
            pd.DataFrame: A two-column DataFrame (Parameter, Value) for Excel export.
        """
        data = self.model_dump()
        return pd.DataFrame(list(data.items()), columns=["Parameter", "Value"])

    def save(self, path: Path, fmt: str = "xlsx") -> None:
        """
        Saves the report to disk in the requested format.

        Supported formats:
            - ``xlsx``: Professional Excel with conditional formatting.
            - ``csv``: Flat CSV (two columns: Parameter, Value).
            - ``json``: Pretty-printed JSON array.

        Args:
            path: Base file path (suffix is replaced to match *fmt*).
            fmt: Output format — one of ``"xlsx"``, ``"csv"``, ``"json"``.
        """
        fmt = fmt.lower()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = self.to_vertical_df()
            if fmt == "csv":
                path = path.with_suffix(".csv")
                df.to_csv(path, index=False)
            elif fmt == "json":
                path = path.with_suffix(".json")
                df.to_json(path, orient="records", indent=2)
            else:
                path = path.with_suffix(".xlsx")
                with pd.ExcelWriter(
                    path,
                    engine="xlsxwriter",
                    engine_kwargs={"options": {"nan_inf_to_errors": True}},
                ) as writer:
                    df.to_excel(writer, sheet_name="Detailed Report", index=False)
                    self._apply_excel_formatting(writer, df)

            logger.info(  # pragma: no mutant
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'Summary Report':<18}: {path.name}"
            )
        except Exception as e:  # xlsxwriter raises non-standard exceptions
            logger.error(f"Failed to generate report: {e}")

    def _apply_excel_formatting(self, writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
        """Internal helper to apply styles, formats and column widths to the worksheet."""
        workbook = writer.book  # pragma: no mutant
        worksheet = writer.sheets["Detailed Report"]  # pragma: no mutant

        # Formatting Definitions
        header_format = workbook.add_format(  # pragma: no mutant
            {"bold": True, "bg_color": "#D7E4BC", "border": 1, "align": "center"}
        )

        # Base format for Floats.
        float_format = workbook.add_format(  # pragma: no mutant
            {
                "border": 1,
                "align": "left",
                "valign": "top",
                "text_wrap": True,
                "font_size": 10,
                "num_format": "0.0000",
            }
        )

        # Base format for Integers.
        int_format = workbook.add_format(  # pragma: no mutant
            {"border": 1, "num_format": "0", "align": "left"}
        )

        # String format
        string_format = workbook.add_format(  # pragma: no mutant
            {"border": 1, "align": "left", "valign": "vcenter", "text_wrap": True}
        )

        # Column Setup
        for row_idx, (_, value) in enumerate(df.values):  # pragma: no mutant
            if isinstance(value, float):  # pragma: no mutant
                fmt = float_format  # pragma: no mutant
            elif isinstance(value, int) and not isinstance(value, bool):  # pragma: no mutant
                fmt = int_format  # pragma: no mutant
            else:  # pragma: no mutant
                fmt = string_format  # pragma: no mutant
            worksheet.write(row_idx + 1, 1, value, fmt)  # pragma: no mutant

        worksheet.set_column(  # pragma: no mutant
            "A:A", 25, workbook.add_format({"border": 1, "bold": True})
        )
        worksheet.set_column("B:B", 70)  # pragma: no mutant

        for col_num, value in enumerate(df.columns.values):  # pragma: no mutant
            worksheet.write(0, col_num, value, header_format)  # pragma: no mutant


def create_structured_report(
    val_metrics: Sequence[dict],
    test_metrics: dict,
    macro_f1: float,
    train_losses: Sequence[float],
    best_path: Path,
    log_path: Path,
    arch_name: str,
    dataset: DatasetConfig,
    training: TrainingConfig,
    aug_info: str | None = None,
) -> TrainingReport:
    """
    Constructs a TrainingReport object using final metrics and configuration.

    This factory method aggregates disparate pipeline results into a single
    validated container, resolving paths and extracting augmentation summaries.

    Args:
        val_metrics: History of per-epoch validation metric dicts.
        test_metrics: Final test-set metric dict (accuracy, auc, etc.).
        macro_f1: Final Macro F1 score on test set.
        train_losses: History of per-epoch training losses.
        best_path: Path to the saved model weights.
        log_path: Path to the run log file.
        arch_name: Architecture identifier (e.g. ``"resnet_18"``).
        dataset: Dataset sub-config with metadata, name, and normalization info.
        training: Training sub-config with hyperparameters and flags.
        aug_info: Pre-formatted augmentation string.

    Returns:
        TrainingReport: A validated Pydantic model ready for export.
    """
    # Augmentation info is expected from caller; fallback to "N/A"
    aug_info = aug_info or "N/A"

    def _safe_max(key: str) -> float:
        """Return the best non-NaN value for *key* across validation epochs."""
        values = [m[key] for m in val_metrics if not math.isnan(m[key])]
        return max(values, default=0.0)

    best_val_acc = _safe_max(METRIC_ACCURACY)
    best_val_auc = _safe_max(METRIC_AUC)
    best_val_f1 = _safe_max(METRIC_F1)

    return TrainingReport(
        architecture=arch_name,
        dataset=dataset.dataset_name,
        best_val_accuracy=best_val_acc,
        best_val_auc=best_val_auc,
        best_val_f1=best_val_f1,
        test_accuracy=test_metrics[METRIC_ACCURACY],
        test_auc=test_metrics[METRIC_AUC],
        test_macro_f1=macro_f1,
        is_texture_based=dataset.metadata.is_texture_based,
        is_anatomical=dataset.metadata.is_anatomical,
        use_tta=training.use_tta,
        epochs_trained=len(train_losses),
        learning_rate=training.learning_rate,
        batch_size=training.batch_size,
        augmentations=aug_info,
        normalization=dataset.metadata.normalization_info,
        model_path=str(best_path.resolve()),
        log_path=str(log_path.resolve()),
        seed=training.seed,
    )
