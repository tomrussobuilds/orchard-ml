"""
Reporting Module

This module defines the structured training report and utilities for generating
final experiment summaries in Excel format and YAML configuration files.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence, Any, Dict, TYPE_CHECKING
import logging
import yaml

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import pandas as pd

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import Config
if TYPE_CHECKING:
    from scripts.core import RunPaths

# =========================================================================== #
#                               EXCEL REPORTS
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

@dataclass(frozen=True)
class TrainingReport:
    """Structured data container for summarizing a complete training experiment."""
    timestamp: str
    model: str
    dataset: str
    best_val_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    epochs_trained: int
    learning_rate: float
    batch_size: int
    augmentations: str
    normalization: str
    model_path: str
    log_path: str
    seed: int

    def to_vertical_df(self) -> pd.DataFrame:
        """Converts the report dataclass into a vertical pandas DataFrame."""
        data = asdict(self)
        return pd.DataFrame(list(data.items()), columns=["Parameter", "Value"]) 

    def save(self, path: Path) -> None:
        """Saves the report DataFrame to an Excel file with professional formatting."""
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_vertical_df()

        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Detailed Report', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Detailed Report']

            # Professional Styling
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#D7E4BC', 'border': 1, 'align': 'center'
            })
            base_format = workbook.add_format({
                'border': 1, 'align': 'left', 'valign': 'vcenter'
            })
            wrap_format = workbook.add_format({
                'border': 1, 'text_wrap': True, 'valign': 'top', 'font_size': 10
            })

            worksheet.set_column('A:A', 25, base_format)
            worksheet.set_column('B:B', 60, wrap_format)

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)         

        logger.info(f"Training report saved to → {path}")


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
    Constructs a TrainingReport object using the final metrics and configuration.
    """
    # Use provided aug_info or fallback to a dynamic check if needed
    if aug_info is None:
        try:
            from scripts.data_handler import get_augmentations_transforms
            aug_info = get_augmentations_transforms(cfg)
        except ImportError:
            aug_info = "N/A"

    return TrainingReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model=cfg.model_name,
        dataset=cfg.dataset.dataset_name,
        best_val_accuracy=max(val_accuracies) if val_accuracies else 0.0,
        test_accuracy=test_acc,
        test_macro_f1=macro_f1,
        epochs_trained=len(train_losses),
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        augmentations=aug_info,
        normalization=cfg.dataset.normalization_info,
        model_path=str(best_path),
        log_path=str(log_path),
        seed=cfg.training.seed,
    )

def save_report_as_yaml(
        config: Config,
        run_paths: 'RunPaths'
) -> Path:
    """
    Saves the configuration dictionary as a YAML file, ensuring Python-specific 
    types like tuples are converted to standard lists for better compatibility.
    """
    yaml_path = run_paths.get_config_path()
    
    try:
        config_data = asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)

        def clean_config(data: Any) -> Any:
            """Recursively converts tuples to lists and Paths to strings."""
            if isinstance(data, dict):
                return {k: clean_config(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [clean_config(i) for i in data]
            elif isinstance(data, Path):
                return str(data)
            return data
        
        cleaned_data = clean_config(config_data)
        
        with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(
                cleaned_data,
                yaml_file,
                default_flow_style=False,
                sort_keys=False
            )

        logger.info(f"Configuration saved to YAML at → {yaml_path}")
        return yaml_path
    except Exception as e:
        logger.error(f"Failed to save configuration YAML: {str(e)}", exc_info=True)
        raise e
    
def load_config_from_yaml(
        yaml_path: Path
    ) -> Dict[str, Any]:
    """
    Loads a configuration dictionary from a YAML file.

    Args:
        yaml_path (Path): The path to the YAML configuration file.
    
    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)