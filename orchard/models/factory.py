"""
Models Factory Module

This module implements the Factory Pattern using a registry-based approach
to decouple model instantiation from the main execution logic.
It ensures that architectures are dynamically adapted to the geometric
constraints (channels, classes) resolved at runtime.
"""

# Standard Imports
import logging

# Third-Party Imports
import torch
import torch.nn as nn

# Internal Imports
from orchard.core import LOGGER_NAME, Config

from .efficientnet_b0 import build_efficientnet_b0
from .mini_cnn import build_mini_cnn
from .resnet_18_adapted import build_resnet18_adapted
from .vit_tiny import build_vit_tiny

# LOGGER CONFIGURATION
logger = logging.getLogger(LOGGER_NAME)


# MODEL FACTORY LOGIC
def get_model(device: torch.device, cfg: Config) -> nn.Module:
    """
    Factory function to resolve, instantiate, and prepare architectures.

    It maps configuration identifiers to specific builder functions via an
    internal registry. Structural parameters like input channels and class
    cardinality are derived from the 'effective' geometry resolved by
    the DatasetConfig.

    Args:
        device: Hardware accelerator target.
        cfg: Global configuration manifest with resolved metadata.

    Returns:
        nn.Module: The instantiated model synchronized with the target device.

    Raises:
        ValueError: If the requested architecture is not found in the registry.
    """
    # Internal Imports
    _MODEL_REGISTRY = {
        "resnet_18_adapted": build_resnet18_adapted,
        "efficientnet_b0": build_efficientnet_b0,
        "vit_tiny": build_vit_tiny,
        "mini_cnn": build_mini_cnn,
    }

    # Resolve structural dimensions from Single Source of Truth (Config)
    in_channels = cfg.dataset.effective_in_channels
    num_classes = cfg.dataset.num_classes
    model_name_lower = cfg.model.name.lower()

    logger.info(
        f"Initializing Architecture: {cfg.model.name} | "
        f"Input: {cfg.dataset.img_size}x{cfg.dataset.img_size}x{in_channels} | "
        f"Output: {num_classes} classes"
    )

    # Architecture resolution via Registry lookup
    builder = _MODEL_REGISTRY.get(model_name_lower)
    if not builder:
        error_msg = f"Architecture '{cfg.model.name}' is not registered in the Factory."
        logger.error(f" [!] {error_msg}")
        raise ValueError(error_msg)

    # Instance construction and adaptation
    model = builder(device=device, cfg=cfg, in_channels=in_channels, num_classes=num_classes)

    # Final deployment and parameter telemetry
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model deployed to {str(device).upper()} | " f"Total Parameters: {total_params:,}")

    return model
