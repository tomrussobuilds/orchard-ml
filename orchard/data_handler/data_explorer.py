"""
Data Visualization Module.

Utilities to inspect datasets visually by generating grids of sample images
from raw tensors or NumPy arrays. Supports grayscale and RGB images and optional
denormalization. Figures are saved inside the run's output directory managed by
RunPaths.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from ..core import LOGGER_NAME, LogStyle, RunPaths
from .loader import DataLoader

logger = logging.getLogger(LOGGER_NAME)

_DEFAULT_DPI = 200


# VISUALIZATION UTILITIES
def show_sample_images(
    loader: DataLoader,
    save_path: Path,
    *,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    arch_name: str = "Model",
    fig_dpi: int = _DEFAULT_DPI,
    num_samples: int = 16,
    title_prefix: str | None = None,
) -> None:
    """
    Extract a batch from the DataLoader and save a grid of sample images.

    Saves images with their corresponding labels to verify data integrity and augmentations.

    Args:
        loader: The PyTorch DataLoader to sample from.
        save_path: Full path (including filename) to save the resulting image.
        mean: Per-channel mean for denormalization.
        std: Per-channel std for denormalization.
        arch_name: Architecture name for the figure title.
        fig_dpi: DPI for the saved figure.
        num_samples: Number of images to display in the grid.
        title_prefix: Optional string to prepend to the figure title.
    """
    try:
        batch_images, _ = next(iter(loader))
    except StopIteration:
        logger.error("DataLoader is empty. Cannot generate sample images.")
        return

    actual_samples = min(len(batch_images), num_samples)
    images = batch_images[:actual_samples]

    # Apply denormalization if mean/std are provided
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)
        images = images * std_t + mean_t

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
    title_str = f"{arch_name} — {actual_samples} Samples"
    if title_prefix:
        title_str = f"{title_prefix} — {title_str}"
    plt.title(title_str, fontsize=14)

    plt.axis("off")
    plt.tight_layout()

    # Ensure target directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=fig_dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Sample Grid':<18}: {save_path.name}")


def show_samples_for_dataset(
    loader: DataLoader,
    classes: list[str],  # noqa: ARG001
    dataset_name: str,
    run_paths: RunPaths,
    *,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    arch_name: str = "Model",
    fig_dpi: int = _DEFAULT_DPI,
    num_samples: int = 16,
    resolution: int | None = None,
) -> None:
    """
    Generate a grid of sample images from a dataset and save to the figures directory.

    Args:
        loader: PyTorch DataLoader to sample images from.
        classes: List of class names (unused here, for metadata).
        dataset_name: Name of the dataset, used in the filename and title.
        run_paths: RunPaths instance to resolve figure saving path.
        mean: Per-channel mean for denormalization.
        std: Per-channel std for denormalization.
        arch_name: Architecture name for the figure title.
        fig_dpi: DPI for the saved figure.
        num_samples: Number of images to include in the grid.
        resolution: Resolution to include in filename to avoid overwriting.
    """
    res_str = f"_{resolution}x{resolution}" if resolution else ""
    save_path = run_paths.get_fig_path(f"{dataset_name}/sample_grid{res_str}.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    show_sample_images(
        loader=loader,
        save_path=save_path,
        mean=mean,
        std=std,
        arch_name=arch_name,
        fig_dpi=fig_dpi,
        num_samples=num_samples,
        title_prefix=f"{dataset_name}{res_str}",
    )
