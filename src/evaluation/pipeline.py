"""
Evaluation and Reporting Package

This package coordinates model inference, performance visualization, 
and structured experiment reporting using a memory-efficient Lazy approach.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple, List
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, RunPaths
from .engine import evaluate_model
from .visualization import (
    plot_confusion_matrix, 
    plot_training_curves, 
    show_predictions
)
from .reporting import create_structured_report, save_report_as_yaml

# =========================================================================== #
#                               EVALUATION PIPELINE                           #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")


def run_final_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    test_images: np.ndarray = None,
    test_labels: np.ndarray = None,
    class_names: List[str] = [],
    train_losses: List[float] = [],
    val_accuracies: List[float] = [],
    device: torch.device = torch.device("cpu"),
    paths: RunPaths = None,
    cfg: Config = None,
    use_tta: bool = False,
    aug_info: str = "N/A"
) -> Tuple[float, float]:
    """
    Executes the complete evaluation pipeline. Supports both direct array 
    input and Lazy Loading from DataLoaders for visualization samples.
    """
    
    # --- 1) Inference & Metrics ---
    # Perform full test set inference (supports TTA)
    all_preds, all_labels, test_acc, macro_f1 = evaluate_model(
        model, test_loader, device, use_tta=use_tta
    )

    # --- 2) Visualizations ---
    # Standard metrics plots
    plot_confusion_matrix(
        all_labels=all_labels,
        all_preds=all_preds,
        classes=class_names,
        out_path=paths.figures / "confusion_matrix.png",
        cfg=cfg
    )

    plot_training_curves(
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        out_dir=paths.figures,
        cfg=cfg
    )

    # Lazy Visualization Logic: Extract sample batch if full arrays are missing
    if test_images is None or test_labels is None:
        logger.info("Lazy Mode: Extracting sample batch from test_loader for prediction grid...")
        # Get one batch from the test loader
        sample_batch = next(iter(test_loader))
        # Convert to NumPy and fix dimensions: (B, C, H, W) -> (B, H, W, C)
        vis_images = sample_batch[0].cpu().numpy().transpose(0, 2, 3, 1)
        vis_labels = sample_batch[1].cpu().numpy().flatten()
        # Correlate with the first N predictions generated earlier
        vis_preds = all_preds[:len(vis_labels)]
    else:
        vis_images = test_images
        vis_labels = test_labels
        vis_preds = all_preds

    # Generate the visual grid of predictions
    show_predictions(
        images=vis_images,
        true_labels=vis_labels,
        preds=vis_preds,
        classes=class_names,
        n=12,
        save_path=paths.figures / "sample_predictions.png",
        cfg=cfg
    )

    # --- 3) Structured Reporting ---
    # Export parameters for reproducibility
    save_report_as_yaml(config=cfg, run_paths=paths)
    
    # Locate the best model checkpoint dynamically
    best_model_filename = f"best_model_{cfg.model_name.lower().replace(' ', '_')}.pth"
    
    # Compile final Excel summary
    report = create_structured_report(
        val_accuracies=val_accuracies,
        macro_f1=macro_f1,
        test_acc=test_acc,
        train_losses=train_losses,
        best_path=paths.models / best_model_filename,
        log_path=paths.logs / "run.log",
        cfg=cfg,
        aug_info=aug_info
    )
    report.save(paths.reports / "training_summary.xlsx")

    logger.info(f"Evaluation finished. Metrics: Acc {test_acc:.4f}, F1 {macro_f1:.4f}")
    
    return macro_f1, test_acc