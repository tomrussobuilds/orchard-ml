"""
Reporting & Experiment Summarization Module

This module orchestrates the generation of human-readable artifacts following 
the completion of a training pipeline. It leverages Pydantic for strict 
validation of experiment results and transforms raw metrics into structured, 
professionally formatted Excel summaries.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from datetime import datetime
from pathlib import Path
from typing import Sequence
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, LOGGER_NAME

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
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True
    )

    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Model & Data Identity
    model: str
    dataset: str
    
    # Core Metrics
    best_val_accuracy: float
    test_accuracy: float
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
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_vertical_df()

        try:
            with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Detailed Report', index=False)

                workbook = writer.book
                worksheet = writer.sheets['Detailed Report']

                # Formatting Definitions
                header_format = workbook.add_format({
                    'bold': True, 
                    'bg_color': '#D7E4BC', 
                    'border': 1, 
                    'align': 'center'
                })
                
                # Base format for column B: includes 4-decimal precision for numbers
                # but stays flexible for strings with wrapping.
                value_format = workbook.add_format({
                    'border': 1, 
                    'align': 'left', 
                    'valign': 'top', 
                    'text_wrap': True, 
                    'font_size': 10,
                    'num_format': '0.0000'  # Critical: Forces 4 decimal places for floats
                })
                
                # Parameter column format (Column A)
                param_format = workbook.add_format({
                    'border': 1, 
                    'align': 'left', 
                    'valign': 'vcenter', 
                    'bold': False
                })

                # Column Setup: 
                # Column A (Parameters) is narrow and clean.
                # Column B (Values) is wide to accommodate paths and augmentations.
                worksheet.set_column('A:A', 25, param_format)
                worksheet.set_column('B:B', 70, value_format)

                # Overwrite headers with style
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)         

            logger.info(f"Summary Excel report saved to → {path.name}")
        except Exception as e:
            logger.error(f"Failed to generate Excel report: {e}")


def create_structured_report(
    val_accuracies: Sequence[float],
    macro_f1: float,
    test_acc: float,
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
    if aug_info is None:
        aug_dict = cfg.augmentation.model_dump()
        aug_info = ", ".join([f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in aug_dict.items()])

    return TrainingReport(
        model=cfg.model.name,
        dataset=cfg.dataset.dataset_name,
        best_val_accuracy=max(val_accuracies) if val_accuracies else 0.0,
        test_accuracy=test_acc,
        test_macro_f1=macro_f1,
        is_texture_based=cfg.dataset.is_texture_based,
        is_anatomical=cfg.dataset.is_anatomical,
        use_tta=cfg.training.use_tta,
        epochs_trained=len(train_losses),
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        augmentations=aug_info,
        normalization=cfg.dataset.normalization_info,
        model_path=str(best_path.resolve()),
        log_path=str(log_path.resolve()),
        seed=cfg.training.seed,
    )