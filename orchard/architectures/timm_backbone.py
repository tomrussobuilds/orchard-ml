"""
Generic timm Backbone Builder.

Provides a universal builder for any model available in the timm
(PyTorch Image Models) registry. Delegates channel adaptation,
head replacement, and weight loading entirely to timm's native API.

Usage via YAML config:
    architecture:
      name: "timm/convnext_base.fb_in22k"
      pretrained: true
      dropout: 0.2

The ``timm/`` prefix is stripped by the factory before reaching this builder.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from ..core import Config


def build_timm_model(
    device: torch.device, num_classes: int, in_channels: int, cfg: Config
) -> nn.Module:
    """
    Construct any timm-registered model with automatic adaptation.

    timm.create_model handles:
        - Pretrained weight loading (from HuggingFace Hub or torch.hub)
        - Classification head replacement (num_classes)
        - Input channel adaptation with weight morphing (in_chans)
        - Dropout rate injection (drop_rate)

    Args:
        device: Target hardware for model placement.
        num_classes: Number of output classes for the classification head.
        in_channels: Number of input channels (1=grayscale, 3=RGB).
        cfg: Global config with architecture.name, pretrained, dropout.

    Returns:
        Adapted timm model deployed to device.

    Raises:
        ValueError: If the timm model identifier is not found in the registry.
    """
    model_id = cfg.architecture.name.split("/", 1)[1]

    try:
        model = timm.create_model(
            model_id,
            pretrained=cfg.architecture.pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            drop_rate=cfg.architecture.dropout,
        )
    except Exception as e:  # timm raises diverse internal errors
        raise ValueError(
            f"Failed to create timm model '{model_id}'. "
            f"Verify the identifier is valid: https://huggingface.co/timm. "
            f"Original error: {e}"
        ) from e

    return model.to(device)
