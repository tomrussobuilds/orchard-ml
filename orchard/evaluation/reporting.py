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
from typing import TYPE_CHECKING, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..core import LOGGER_NAME, DatasetConfig, LogStyle, TrainingConfig

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

logger = logging.getLogger(LOGGER_NAME)


# EXCEL REPORTS
class TrainingReport(BaseModel):
    """
    Validated data container for summarizing a complete training experiment.

    This model serves as a Schema for the final experimental metadata. It stores
    hardware, hyperparameter, and performance states to ensure full reproducibility
    and traceability of the training pipeline.

    Attributes:
        timestamp (str): ISO formatted execution time.
        architecture (str): Identifier of the architecture used.
        dataset (str): Name of the dataset.
        best_val_metrics (dict[str, float]): Peak values for each validation metric.
        test_metrics (dict[str, float]): Final metric values on the unseen test set.
        is_texture_based (bool): Whether texture-preserving logic was applied.
        is_anatomical (bool): Whether anatomical orientation constraints were enforced.
        use_tta (bool): Indicates if Test-Time Augmentation was active.
        epochs_trained (int): Total number of optimization cycles completed.
        learning_rate (float): Initial learning rate used by the optimizer.
        batch_size (int): Samples processed per iteration.
        seed (int): Global RNG seed for experiment replication.
        augmentations (str): Descriptive string of the transformation pipeline.
        normalization (str): Mean/Std statistics applied to the input tensors.
        model_path (str): Absolute path to the best saved checkpoint.
        log_path (str): Absolute path to the session execution log.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Model & Data Identity
    architecture: str
    dataset: str

    # Core Metrics (task-agnostic)
    best_val_metrics: dict[str, float]
    test_metrics: dict[str, float]

    # Domain Logic Flags (classification only — None for detection)
    is_texture_based: bool | None = None
    is_anatomical: bool | None = None
    use_tta: bool | None = None

    # Hyperparameters
    epochs_trained: int
    learning_rate: float
    batch_size: int
    seed: int

    # Metadata Strings
    augmentations: str | None = None
    normalization: str
    model_path: str
    log_path: str

    def to_vertical_df(self) -> pd.DataFrame:
        """
        Converts the Pydantic model into a vertical pandas DataFrame.

        Metric dict fields (``best_val_metrics``, ``test_metrics``) are flattened
        into prefixed rows (e.g. ``best_val_accuracy``, ``test_f1``).

        Returns:
            pd.DataFrame: A two-column DataFrame (Parameter, Value) for Excel export.
        """
        rows: list[tuple[str, object]] = []
        for key, value in self.model_dump().items():
            if value is None:
                continue
            if isinstance(value, dict):
                prefix = key.removesuffix("_metrics")
                for sub_key, sub_value in value.items():
                    rows.append((f"{prefix}_{sub_key}", sub_value))
            else:
                rows.append((key, value))
        return pd.DataFrame(rows, columns=["Parameter", "Value"])

    def save(
        self, path: Path, fmt: str = "xlsx"  # pragma: no mutate  # .lower() normalizes any case
    ) -> None:
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
                df.to_csv(path, index=False)  # pragma: no mutate  # None ≡ False in pandas
            elif fmt == "json":
                path = path.with_suffix(".json")
                df.to_json(path, orient="records", indent=2)
            else:
                path = path.with_suffix(".xlsx")
                with pd.ExcelWriter(  # pragma: no mutate
                    path,
                    engine="xlsxwriter",  # pragma: no mutate
                    engine_kwargs={"options": {"nan_inf_to_errors": True}},  # pragma: no mutate
                ) as writer:
                    # fmt: off
                    df.to_excel(writer, sheet_name="Detailed Report", index=False)  # pragma: no mutate
                    # fmt: on
                    self._apply_excel_formatting(writer, df)

            logger.info(
                "%s%s %-18s: %s", LogStyle.INDENT, LogStyle.ARROW, "Summary Report", path.name
            )
        except Exception as e:  # xlsxwriter raises non-standard exceptions
            logger.error("Failed to generate report: %s", e)

    def _apply_excel_formatting(self, writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
        """Internal helper to apply styles, formats and column widths to the worksheet."""
        workbook = writer.book  # pragma: no mutate
        worksheet = writer.sheets["Detailed Report"]  # pragma: no mutate

        # Formatting Definitions
        header_format = workbook.add_format(  # pragma: no mutate
            {"bold": True, "bg_color": "#D7E4BC", "border": 1, "align": "center"}
        )

        # Base format for Floats.
        float_format = workbook.add_format(  # pragma: no mutate
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
        int_format = workbook.add_format(  # pragma: no mutate
            {"border": 1, "num_format": "0", "align": "left"}
        )

        # String format
        string_format = workbook.add_format(  # pragma: no mutate
            {"border": 1, "align": "left", "valign": "vcenter", "text_wrap": True}
        )

        # Column Setup
        for row_idx, (_, value) in enumerate(df.values):  # pragma: no mutate
            if isinstance(value, float):  # pragma: no mutate
                fmt = float_format  # pragma: no mutate
            elif isinstance(value, int) and not isinstance(value, bool):  # pragma: no mutate
                fmt = int_format  # pragma: no mutate
            else:  # pragma: no mutate
                fmt = string_format  # pragma: no mutate
            worksheet.write(row_idx + 1, 1, value, fmt)  # pragma: no mutate

        worksheet.set_column(  # pragma: no mutate
            "A:A", 25, workbook.add_format({"border": 1, "bold": True})
        )
        worksheet.set_column("B:B", 70)  # pragma: no mutate

        for col_num, value in enumerate(df.columns.values):  # pragma: no mutate
            worksheet.write(0, col_num, value, header_format)  # pragma: no mutate


def create_structured_report(
    val_metrics: Sequence[Mapping[str, float]],
    test_metrics: Mapping[str, float],
    train_losses: Sequence[float],
    best_path: Path,
    log_path: Path,
    arch_name: str,
    dataset: DatasetConfig,
    training: TrainingConfig,
    aug_info: str | None = None,
    task_type: str = "classification",  # pragma: no mutate
) -> TrainingReport:
    """
    Constructs a TrainingReport object using final metrics and configuration.

    This factory method aggregates disparate pipeline results into a single
    validated container, resolving paths and extracting augmentation summaries.

    Args:
        val_metrics: History of per-epoch validation metric dicts.
        test_metrics: Final test-set metric mapping (all task metrics included).
        train_losses: History of per-epoch training losses.
        best_path: Path to the saved model weights.
        log_path: Path to the run log file.
        arch_name: Architecture identifier (e.g. ``"resnet_18"``).
        dataset: Dataset sub-config with metadata, name, and normalization info.
        training: Training sub-config with hyperparameters and flags.
        aug_info: Pre-formatted augmentation string.
        task_type: Task type for conditional field inclusion.

    Returns:
        TrainingReport: A validated Pydantic model ready for export.
    """

    def _safe_max(key: str) -> float:
        """Return the best non-NaN value for *key* across validation epochs."""
        values = [m[key] for m in val_metrics if not math.isnan(m[key])]
        return max(values, default=0.0)

    # Build best-val metrics generically from all keys present in history
    all_keys: set[str] = set()
    for m in val_metrics:
        all_keys.update(m.keys())
    best_val_metrics = {key: _safe_max(key) for key in sorted(all_keys)}

    # Classification-only fields (None → excluded from report)
    is_classification = task_type == "classification"

    # Detection: drop the METRIC_LOSS sentinel (always 0.0, not meaningful)
    if not is_classification:
        best_val_metrics.pop("loss", None)  # pragma: no mutate

    return TrainingReport(
        architecture=arch_name,
        dataset=dataset.dataset_name,
        best_val_metrics=best_val_metrics,
        test_metrics=dict(test_metrics),
        is_texture_based=dataset.metadata.is_texture_based if is_classification else None,
        is_anatomical=dataset.metadata.is_anatomical if is_classification else None,
        use_tta=training.use_tta if is_classification else None,
        epochs_trained=len(train_losses),
        learning_rate=training.learning_rate,
        batch_size=training.batch_size,
        augmentations=(aug_info or "N/A") if is_classification else None,
        normalization=dataset.metadata.normalization_info,
        model_path=str(best_path.resolve()),
        log_path=str(log_path.resolve()),
        seed=training.seed,
    )
