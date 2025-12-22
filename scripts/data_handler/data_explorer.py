"""
Data Visualization Module

This module provides utilities for inspecting the dataset visually, 
specifically by generating grids of sample images from the raw NumPy arrays.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from pathlib import Path
from typing import List

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from scripts.core import Config

# ========================================================================== #
#                             VISUALIZATION UTILITIES                        #
# ========================================================================== #  
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")


def show_sample_images(
        images: np.ndarray,
        labels: np.ndarray,
        classes: List[str],
        save_path: Path,
        cfg: Config
    ) -> None:  
    """
    Generates and saves a figure showing 9 random samples from the training set.

    Args:
        images (np.ndarray): NumPy array of training images.
        labels (np.ndarray): NumPy array of training labels.
        classes (List[str]): List of class names for labeling.
        save_path (Path): Full path where the figure will be saved.
        cfg (Config): Configuration object for title metadata.
    """
    # Safety check: avoid crashing if the dataset is surprisingly small
    num_samples = min(len(images), 9)
    indices = np.random.choice(len(images), size=num_samples, replace=False)

    plt.figure(figsize=(9, 9))
    for i, idx in enumerate(indices):
        img = images[idx]
        label_idx = int(labels[idx])

        plt.subplot(3, 3, i + 1)

        # Handle grayscale (1 channel), Channel-First, or Channel-Last images
        if img.ndim == 3 and img.shape[-1] == 3:
            plt.imshow(img)
        elif img.ndim == 3 and img.shape[0] == 3:
            plt.imshow(img.transpose(1, 2, 0))
        else:
            plt.imshow(img.squeeze(), cmap='gray')

        class_name = classes[label_idx] if label_idx < len(classes) else f"ID: {label_idx}"
        plt.title(f"{label_idx} — {class_name}", fontsize=11)
        plt.axis("off")

    model_title = cfg.model_name if cfg else "Model"
    plt.suptitle(f"{model_title} — 9 Random Samples from Training Set", fontsize=16)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    # Ensure the parent directory exists (safety for RunPaths)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample images saved to → {save_path}")