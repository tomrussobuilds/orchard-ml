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

- ``get_model``: Factory function for architecture resolution and instantiation
- ``_MODEL_REGISTRY``: Internal mapping of architecture names to builders

Example:
    >>> from orchard.architectures.factory import get_model
    >>> model = get_model(device, dataset_cfg=cfg.dataset, arch_cfg=cfg.architecture)
    >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import torch.nn as nn

from ..core import LOGGER_NAME, ArchitectureConfig, DatasetConfig, LogStyle
from .convnext_tiny import build_convnext_tiny
from .efficientnet_b0 import build_efficientnet_b0
from .mini_cnn import build_mini_cnn
from .resnet_18 import build_resnet18
from .timm_backbone import build_timm_model
from .vit_tiny import build_vit_tiny

# LOGGER CONFIGURATION
logger = logging.getLogger(LOGGER_NAME)


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


# MODEL FACTORY LOGIC
def get_model(
    device: torch.device,
    dataset_cfg: DatasetConfig,
    arch_cfg: ArchitectureConfig,
    verbose: bool = True,
) -> nn.Module:
    """
    Factory function to resolve, instantiate, and prepare architectures.

    It maps configuration identifiers to specific builder functions via an
    internal registry. Structural parameters like input channels and class
    cardinality are derived from the 'effective' geometry resolved by
    the DatasetConfig.

    Args:
        device: Hardware accelerator target.
        dataset_cfg: Dataset sub-config with resolved metadata.
        arch_cfg: Architecture sub-config with model selection.
        verbose: Suppress builder-internal INFO logging.

    Returns:
        nn.Module: The instantiated model synchronized with the target device.

    Example:
        >>> model = get_model(device, dataset_cfg=cfg.dataset, arch_cfg=cfg.architecture)

    Raises:
        ValueError: If the requested architecture is not found in the registry.
    """
    # Resolve structural dimensions from sub-configs
    in_channels = dataset_cfg.effective_in_channels
    num_classes = dataset_cfg.num_classes
    model_name_lower = arch_cfg.name.lower()

    if verbose:
        logger.info(  # pragma: no mutant
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Architecture':<18}: "
            f"{arch_cfg.name} | "
            f"Input: {dataset_cfg.img_size}x{dataset_cfg.img_size}x{in_channels} | "
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
            model = _dispatch_builder(
                model_name_lower, num_classes, in_channels, arch_cfg, dataset_cfg.resolution
            )
    finally:
        logger.setLevel(_prev_level)

    # Centralised device placement (builders stay device-agnostic)
    model = model.to(device)

    # Parameter telemetry
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(  # pragma: no mutant
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Deployed':<18}: "
            f"{str(device).upper()} | Parameters: {total_params:,}"
        )
        logger.info("")  # pragma: no mutant

    return model


# INTERNAL HELPERS
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


def _dispatch_builder(
    model_name_lower: str,
    num_classes: int,
    in_channels: int,
    arch_cfg: ArchitectureConfig,
    resolution: int,
) -> nn.Module:
    """
    Dispatch to the correct architecture builder with narrowed parameters.

    Resolves ``model_name_lower`` against the ``_MODEL_REGISTRY`` (or the
    ``timm/`` prefix convention) and forwards only the parameters each
    builder actually needs. Device placement is handled by ``get_model``.

    Args:
        model_name_lower: Lowercased architecture identifier.
        num_classes: Number of output classes.
        in_channels: Number of input image channels.
        arch_cfg: Architecture sub-config with model-specific options.
        resolution: Input spatial resolution (e.g. 28, 64, 224).

    Returns:
        Instantiated model on CPU.

    Raises:
        ValueError: If the architecture is not found in the registry.
    """
    # Explicit dispatch: each builder has a distinct signature (dropout,
    # resolution, weight_variant …) so a single Callable protocol cannot
    # unify them without an opaque **kwargs bag.  At 6 architectures the
    # trade-off favours readability; for 20+ a BuildContext ABC would pay off.
    if model_name_lower.startswith("timm/"):
        return build_timm_model(
            num_classes=num_classes,
            in_channels=in_channels,
            arch_cfg=arch_cfg,
        )

    builder = _MODEL_REGISTRY.get(model_name_lower)
    if builder is None:
        error_msg = f"Architecture '{arch_cfg.name}' is not registered in the Factory."
        logger.error(f" {LogStyle.FAILURE} {error_msg}")
        raise ValueError(error_msg)

    if builder is build_mini_cnn:
        return build_mini_cnn(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=arch_cfg.dropout,
        )
    if builder is build_resnet18:
        return build_resnet18(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=arch_cfg.pretrained,
            resolution=resolution,
        )
    if builder is build_vit_tiny:
        return build_vit_tiny(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=arch_cfg.pretrained,
            weight_variant=arch_cfg.weight_variant,
        )
    # efficientnet_b0, convnext_tiny: pretrained only
    return builder(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=arch_cfg.pretrained,
    )
