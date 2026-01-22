"""
Reporting & Experiment Summarization Module

This module orchestrates the generation of human-readable artifacts following
the completion of a training pipeline. It leverages Pydantic for strict
validation of experiment results and transforms raw metrics into structured,
professionally formatted Excel summaries.
"""

import logging

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from datetime import datetime
from pathlib import Path
from typing import Sequence

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from orchard.core import LOGGER_NAME, Config

# =========================================================================== #
#                               EXCEL REPORTS                                 #
# =========================================================================== #

logger = logging.getLogger(LOGGER_NAME)


class TrainingReport(BaseModel):
    """
    Validated data container for summarizing a complete training experiment.

    This model serves as a Schema for the final experimental metadata. It stores
    hardware, hyperparameter, and performance states to ensure full reproducibility
    and traceability of the medical imaging pipeline.

    Attributes:
        timestamp (str): ISO formatted execution time.
        model (str): Identifier of the architecture used.
        dataset (str): Name of the MedMNIST subset.
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
    model: str
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

    def save(self, path: Path) -> None:
        """
        Saves the report DataFrame to an Excel file with professional formatting.

        Applies conditional formatting and column widths to ensure the report
        is presentation-ready.

        Args:
            path (Path): Filesystem path where the .xlsx file will be created.
        """
        try:

            path = path.with_suffix(".xlsx")
            path.parent.mkdir(parents=True, exist_ok=True)

            df = self.to_vertical_df()

            with pd.ExcelWriter(
                path, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}
            ) as writer:
                df.to_excel(writer, sheet_name="Detailed Report", index=False)

                self._apply_excel_formatting(writer, df)

            logger.info(f"Summary Excel report saved → {path.name}")
        except Exception as e:
            logger.error(f"Failed to generate Excel report: {e}")

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
        val_accuracies (Sequence[float]): History of validation accuracies.
        macro_f1 (float): Final Macro F1 score on test set.
        test_acc (float): Final accuracy on test set.
        train_losses (Sequence[float]): History of training losses.
        best_path (Path): Path to the saved model weights.
        log_path (Path): Path to the run log file.
        cfg (Config): Validated global configuration.
        aug_info (str, optional): Pre-formatted augmentation string.

    Returns:
        TrainingReport: A validated Pydantic model ready for export.
    """
    # Auto-generate augmentation info if not provided
    aug_info = aug_info or _format_augmentation_string(cfg)

    best_val_acc = max((m["accuracy"] for m in val_metrics), default=0.0)
    best_val_auc = max((m["auc"] for m in val_metrics), default=0.0)

    return TrainingReport(
        model=cfg.model.name,
        dataset=cfg.dataset.dataset_name,
        best_val_accuracy=best_val_acc,
        best_val_auc=best_val_auc,
        test_accuracy=test_metrics["accuracy"],
        test_auc=test_metrics["auc"],
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


def _format_augmentation_string(cfg: Config) -> str:
    """Internal helper to transform the aug dictionary into a readable string."""
    aug_dict = cfg.augmentation.model_dump()
    return ", ".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in aug_dict.items()])
