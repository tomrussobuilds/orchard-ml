"""
Detection Collate Function.

Detection models expect ``list[Tensor]`` images and ``list[dict]`` targets
rather than stacked tensor batches. This module provides the custom collate
function for detection DataLoaders.
"""

from __future__ import annotations

from typing import Any

import torch


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """
    Collate detection samples into list-based batches.

    Unlike the default PyTorch collate (which stacks tensors), detection
    requires list-based batching because each image can have a different
    number of bounding boxes.

    Args:
        batch: List of (image, target_dict) tuples from the dataset.

    Returns:
        Tuple of (list of image tensors, list of target dicts).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
