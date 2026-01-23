"""
Data Visualization Module

Utilities to inspect MedMNIST datasets visually by generating grids of sample images
from raw tensors or NumPy arrays. Supports grayscale and RGB images and optional
denormalization via Config. Figures are saved inside the run's output directory
managed by RunPaths.
"""

# Standard Imports
import logging
from pathlib import Path
from typing import List, Optional

# Third-Party Imports
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# Internal Imports
from orchard.core import LOGGER_NAME, Config, RunPaths

from .factory import DataLoader

logger = logging.getLogger(LOGGER_NAME)


# VISUALIZATION UTILITIES
def show_sample_images(
    loader: DataLoader,
    save_path: Path,
    cfg: Optional[Config] = None,
    num_samples: int = 16,
    title_prefix: Optional[str] = None,
) -> None:
    """
    Extracts a batch from the DataLoader and saves a grid of sample images
    with their corresponding labels to verify data integrity and augmentations.

    Args:
        loader (DataLoader): The PyTorch DataLoader to sample from.
        save_path (Path): Full path (including filename) to save the resulting image.
        cfg (Config, optional): Configuration object for metadata (mean, std).
        num_samples (int, optional): Number of images to display in the grid. Defaults to 16.
        title_prefix (str, optional): Optional string to prepend to the figure title.
    """
    try:
        batch_images, _ = next(iter(loader))
    except StopIteration:
        logger.error("DataLoader is empty. Cannot generate sample images.")
        return

    actual_samples = min(len(batch_images), num_samples)
    images = batch_images[:actual_samples]

    # Apply denormalization if Config is provided
    if cfg is not None:
        mean = torch.tensor(cfg.dataset.mean).view(-1, 1, 1)
        std = torch.tensor(cfg.dataset.std).view(-1, 1, 1)
        images = images * std + mean

    images = torch.clamp(images, 0, 1)

    # Create a grid
    grid = make_grid(images, nrow=int(actual_samples**0.5), padding=2)

    # Convert to numpy HWC for matplotlib
    plt.imshow(
        (
            grid.squeeze(0).cpu().numpy()
            if images.shape[1] == 1
            else grid.permute(1, 2, 0).cpu().numpy()
        ),
        cmap="gray" if images.shape[1] == 1 else None,
    )

    # Figure title
    model_title = cfg.model.name if cfg else "Model"
    title_str = f"{model_title} — {actual_samples} Samples"
    if title_prefix:
        title_str = f"{title_prefix} — {title_str}"
    plt.title(title_str, fontsize=14)

    plt.axis("off")
    plt.tight_layout()

    # Ensure target directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample images saved → {save_path}")


def show_samples_for_dataset(
    loader: DataLoader,
    classes: List[str],
    dataset_name: str,
    run_paths: RunPaths,
    cfg: Optional[Config] = None,
    num_samples: int = 16,
    resolution: int | None = None,
) -> None:
    """
    Generates a grid of sample images from a dataset and saves it in the
    run-specific figures directory. Optionally uses Config to display
    the model name and apply denormalization.

    Args:
        loader (DataLoader): PyTorch DataLoader to sample images from.
        classes (List[str]): List of class names (unused here, for metadata).
        dataset_name (str): Name of the dataset, used in the filename and title.
        run_paths (RunPaths): RunPaths instance to resolve figure saving path.
        cfg (Config, optional): Config object with model.name, dataset.mean/std.
        num_samples (int, optional): Number of images to include in the grid.
        resolution (int | None, optional): Resolution to include in filename to avoid overwriting.
    """
    res_str = f"_{resolution}x{resolution}" if resolution else ""
    save_path = run_paths.get_fig_path(f"{dataset_name}/sample_grid{res_str}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    show_sample_images(
        loader=loader,
        save_path=save_path,
        cfg=cfg,
        num_samples=num_samples,
        title_prefix=f"{dataset_name}{res_str}",
    )
