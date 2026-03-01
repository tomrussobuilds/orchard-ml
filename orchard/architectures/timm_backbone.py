"""
Generic timm Backbone Builder.

Provides a universal builder for any model available in the timm
(PyTorch Image Models) registry. Delegates channel adaptation,
head replacement, and weight loading entirely to timm's native API.

Usage via YAML config::

    architecture:
      name: "timm/convnext_base.fb_in22k"
      pretrained: true
      dropout: 0.2

The ``timm/`` prefix is stripped by the factory before reaching this builder.
"""

from __future__ import annotations

import timm
import torch.nn as nn

from ..core.config import ArchitectureConfig


def build_timm_model(
    num_classes: int,
    in_channels: int,
    *,
    arch_cfg: ArchitectureConfig,
) -> nn.Module:
    """
    Construct any timm-registered model with automatic adaptation.

    timm.create_model handles:

    - Pretrained weight loading (from HuggingFace Hub or torch.hub)
    - Classification head replacement (num_classes)
    - Input channel adaptation with weight morphing (in_chans)
    - Dropout rate injection (drop_rate)

    Args:
        num_classes: Number of output classes for the classification head.
        in_channels: Number of input channels (1=grayscale, 3=RGB).
        arch_cfg: Architecture sub-config with name, pretrained, dropout.

    Returns:
        Adapted timm model (device placement handled by factory).

    Raises:
        ValueError: If the timm model identifier is not found in the registry.
    """
    model_id = arch_cfg.name.split("/", 1)[1]

    try:
        model = timm.create_model(
            model_id,
            pretrained=arch_cfg.pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            drop_rate=arch_cfg.dropout,
        )
    except Exception as e:  # timm raises diverse internal errors
        raise ValueError(
            f"Failed to create timm model '{model_id}'. "
            f"Verify the identifier is valid: https://huggingface.co/timm. "
            f"Original error: {e}"
        ) from e

    return model
