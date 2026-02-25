"""
Reporting & Experiment Summarization Module

This module orchestrates the generation of human-readable artifacts following
the completion of a training pipeline. It leverages Pydantic for strict
validation of experiment results and transforms raw metrics into structured,
professionally formatted Excel summaries.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..core import LOGGER_NAME, Config, LogStyle
from ..core.paths import METRIC_ACCURACY, METRIC_AUC
from ..data_handler import get_augmentations_description

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

            logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Summary Report':<18}: {path.name}")
        except Exception as e:  # xlsxwriter raises non-standard exceptions
            logger.error(f"Failed to generate report: {e}")

    def _apply_excel_formatting(self, writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
        """
        Internal helper to apply styles, formats and column widths to the worksheet.
        """
        workbook = writer.book
        worksheet = writer.sheets["Detailed Report"]

        # Formatting Definitions
        header_format = workbook.add_format(
            {"bold": True, "bg_color": "#D7E4BC", "border": 1, "align": "center"}
        )

        # Base format for Floats.
        float_format = workbook.add_format(
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
        int_format = workbook.add_format({"border": 1, "num_format": "0", "align": "left"})

        # String format
        string_format = workbook.add_format(
            {"border": 1, "align": "left", "valign": "vcenter", "text_wrap": True}
        )

        # Column Setup
        for row_idx, (_, value) in enumerate(df.values):
            if isinstance(value, float):
                fmt = float_format
            elif isinstance(value, int) and not isinstance(value, bool):
                fmt = int_format
            else:
                fmt = string_format
            worksheet.write(row_idx + 1, 1, value, fmt)

        worksheet.set_column("A:A", 25, workbook.add_format({"border": 1, "bold": True}))
        worksheet.set_column("B:B", 70)

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)


def create_structured_report(
    val_metrics: Sequence[dict],
    test_metrics: dict,
    macro_f1: float,
    train_losses: Sequence[float],
    best_path: Path,
    log_path: Path,
    cfg: Config,
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
        cfg: Validated global configuration.
        aug_info: Pre-formatted augmentation string.

    Returns:
        TrainingReport: A validated Pydantic model ready for export.
    """
    # Auto-generate augmentation info if not provided
    aug_info = aug_info or get_augmentations_description(cfg)

    best_val_acc = max((m[METRIC_ACCURACY] for m in val_metrics), default=0.0)
    best_val_auc = max((m[METRIC_AUC] for m in val_metrics), default=0.0)

    return TrainingReport(
        architecture=cfg.architecture.name,
        dataset=cfg.dataset.dataset_name,
        best_val_accuracy=best_val_acc,
        best_val_auc=best_val_auc,
        test_accuracy=test_metrics[METRIC_ACCURACY],
        test_auc=test_metrics[METRIC_AUC],
        test_macro_f1=macro_f1,
        is_texture_based=cfg.dataset.metadata.is_texture_based,
        is_anatomical=cfg.dataset.metadata.is_anatomical,
        use_tta=cfg.training.use_tta,
        epochs_trained=len(train_losses),
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        augmentations=aug_info,
        normalization=cfg.dataset.metadata.normalization_info,
        model_path=str(best_path.resolve()),
        log_path=str(log_path.resolve()),
        seed=cfg.training.seed,
    )
