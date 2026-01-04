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
import torch
import matplotlib.pyplot as plt

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, LOGGER_NAME
from .factory import DataLoader

# ========================================================================== #
#                             VISUALIZATION UTILITIES                        #
# ========================================================================== #  
# Global logger instance
logger = logging.getLogger(LOGGER_NAME)


def show_sample_images(
        loader: DataLoader,
        classes: List[str],
        save_path: Path,
        cfg: Config,
    ) -> None:  
    """
    Extracts a batch from the DataLoader and saves a grid of sample images 
    with their corresponding labels to verify data integrity and augmentations.

    Args:
        loader (DataLoader): The PyTorch DataLoader to sample from.
        classes (list[str]): List of class names for label mapping.
        save_path (Path): Full path (including filename) to save the resulting image.
        cfg (Config): Configuration object for metadata (mean, std).
        num_samples (int): Number of images to display in the grid. Defaults to 16.
    """
    # Extract one batch of data from the loader
    try:
        batch_images, batch_labels = next(iter(loader))
    except StopIteration:
        logger.error("DataLoader is empty. Cannot generate sample images.")
        return
    
    actual_samples = min(len(batch_images), 9)
    
    plt.figure(figsize=(9, 9))
    
    # Denormalization constants from Config
    mean = torch.tensor(cfg.dataset.mean).view(-1, 1, 1)
    std = torch.tensor(cfg.dataset.std).view(-1, 1, 1)

    for i in range(actual_samples):
        # Convert tensor to numpy and denormalize for proper visualization
        img_tensor = batch_images[i]
        
        # Reverse normalization: img = (tensor * std) + mean
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        img = img_tensor.cpu().numpy()
        label_idx = int(batch_labels[i])

        plt.subplot(3, 3, i + 1)

        # Handle grayscale (1 channel), Channel-First (PyTorch standard), or Channel-Last
        if img.ndim == 3 and img.shape[0] == 3:
            # PyTorch CHW to Matplotlib HWC
            plt.imshow(img.transpose(1, 2, 0))
        elif img.ndim == 3 and img.shape[0] == 1:
            # Grayscale case
            plt.imshow(img.squeeze(), cmap='gray')
        elif img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            # Fallback for other formats
            plt.imshow(img)

        class_name = classes[label_idx] if label_idx < len(classes) else f"ID: {label_idx}"
        plt.title(f"{label_idx} — {class_name}", fontsize=11)
        plt.axis("off")

    model_title = cfg.model.name if cfg else "Model"
    plt.suptitle(f"{model_title} — 9 Samples from Training Loader", fontsize=16)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample images saved → {save_path}")