"""
Models Factory Module.

Implements the Factory Pattern using a registry-based approach to decouple
model instantiation from execution logic. Architectures are dynamically
adapted to geometric constraints (channels, classes) resolved at runtime.

Architecture:

- Registry Pattern: Internal _MODEL_REGISTRY maps names to builders
- Dynamic Adaptation: Structural parameters derived from DatasetConfig
- Device Management: Automatic model transfer to target accelerator

Key Components:
    get_model: Factory function for architecture resolution and instantiation
    _MODEL_REGISTRY: Internal mapping of architecture names to builders

Example:
    >>> from orchard.architectures.factory import get_model
    >>> model = get_model(device=device, cfg=cfg)
    >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import torch.nn as nn

from ..core import LOGGER_NAME, Config, LogStyle
from .convnext_tiny import build_convnext_tiny
from .efficientnet_b0 import build_efficientnet_b0
from .mini_cnn import build_mini_cnn
from .resnet_18 import build_resnet18
from .timm_backbone import build_timm_model
from .vit_tiny import build_vit_tiny

# LOGGER CONFIGURATION
logger = logging.getLogger(LOGGER_NAME)


@contextmanager
def _suppress_download_noise() -> Iterator[None]:
    """
    Suppress tqdm progress bars and download logging from torch.hub and huggingface_hub.
    """
    prev = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    noisy_loggers = [
        logging.getLogger("torch.hub"),
        logging.getLogger("huggingface_hub.utils._http"),
    ]
    old_levels = [lg.level for lg in noisy_loggers]
    for lg in noisy_loggers:
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = prev
        for lg, level in zip(noisy_loggers, old_levels):
            lg.setLevel(level)


_BuilderFn = Callable[..., nn.Module]

_MODEL_REGISTRY: dict[str, _BuilderFn] = {
    "resnet_18": build_resnet18,
    "efficientnet_b0": build_efficientnet_b0,
    "convnext_tiny": build_convnext_tiny,
    "vit_tiny": build_vit_tiny,
    "mini_cnn": build_mini_cnn,
    # Extension point: register your custom architecture here
    # "your_model": build_your_model,
}


def _dispatch_builder(
    model_name_lower: str,
    device: torch.device,
    num_classes: int,
    in_channels: int,
    cfg: Config,
) -> nn.Module:
    """
    Dispatch to the correct builder with narrowed parameters.
    """
    arch = cfg.architecture

    if model_name_lower.startswith("timm/"):
        return build_timm_model(
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            arch_cfg=arch,
        )

    builder = _MODEL_REGISTRY.get(model_name_lower)
    if builder is None:
        error_msg = f"Architecture '{arch.name}' is not registered in the Factory."
        logger.error(f" {LogStyle.WARNING} {error_msg}")
        raise ValueError(error_msg)

    if builder is build_mini_cnn:
        return build_mini_cnn(
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=arch.dropout,
        )
    if builder is build_resnet18:
        return build_resnet18(
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=arch.pretrained,
            resolution=cfg.dataset.resolution,
        )
    if builder is build_vit_tiny:
        return build_vit_tiny(
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=arch.pretrained,
            weight_variant=arch.weight_variant,
        )
    # efficientnet_b0, convnext_tiny: pretrained only
    return builder(
        device,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=arch.pretrained,
    )


# MODEL FACTORY LOGIC
def get_model(device: torch.device, cfg: Config, verbose: bool = True) -> nn.Module:
    """
    Factory function to resolve, instantiate, and prepare architectures.

    It maps configuration identifiers to specific builder functions via an
    internal registry. Structural parameters like input channels and class
    cardinality are derived from the 'effective' geometry resolved by
    the DatasetConfig.

    Args:
        device: Hardware accelerator target.
        cfg: Global configuration manifest with resolved metadata.
        verbose: Suppress builder-internal INFO logging.

    Returns:
        nn.Module: The instantiated model synchronized with the target device.

    Example:
        >>> model = get_model(device=device, cfg=cfg)
        >>> batch = torch.randn(8, cfg.dataset.effective_in_channels,
        ...                     cfg.dataset.img_size, cfg.dataset.img_size).to(device)
        >>> logits = model(batch)

    Raises:
        ValueError: If the requested architecture is not found in the registry.
    """
    # Resolve structural dimensions from Single Source of Truth (Config)
    in_channels = cfg.dataset.effective_in_channels
    num_classes = cfg.dataset.num_classes
    model_name_lower = cfg.architecture.name.lower()

    if verbose:
        logger.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Architecture':<18}: "
            f"{cfg.architecture.name} | "
            f"Input: {cfg.dataset.img_size}x{cfg.dataset.img_size}x{in_channels} | "
            f"Output: {num_classes} classes"
        )

    # Instance construction and adaptation.
    # When verbose=False (e.g. export phase), suppress builder-internal INFO logs
    # to avoid duplicating messages already shown during training.
    _prev_level = logger.level
    if not verbose:
        logger.setLevel(logging.WARNING)
    try:
        with _suppress_download_noise():
            model = _dispatch_builder(model_name_lower, device, num_classes, in_channels, cfg)
    finally:
        logger.setLevel(_prev_level)

    # Parameter telemetry
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Deployed':<18}: "
            f"{str(device).upper()} | Parameters: {total_params:,}"
        )
        logger.info("")

    return model
