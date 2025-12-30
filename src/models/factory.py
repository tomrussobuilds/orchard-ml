"""
Models Factory Module

This module implements the Factory Pattern to decouple model instantiation 
from the main execution logic. It routes requests to specific architecture 
definitions based on the configuration provided.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, DATASET_REGISTRY, LOGGER_NAME
from .resnet_18_adapted import build_resnet18_adapted

# =========================================================================== #
#                                MODEL FACTORY LOGIC                          #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger(LOGGER_NAME)

def get_model(
    device: torch.device,
    cfg: Config
) -> nn.Module:
    """
    Factory function to instantiate and prepare the model.

    Based on the 'model_name' defined in the Config object, this function 
    calls the appropriate builder and moves the resulting model to the 
    specified hardware device.

    Args:
        device (torch.device): The hardware device (CPU/CUDA) to host the model.
        cfg (Config): The global configuration object containing model metadata.

    Returns:
        nn.Module: The instantiated and hardware-assigned PyTorch model.

    Raises:
        ValueError: If the requested model_name is not registered in the factory.
    """
    
    # Normalize model name for robust matching
    model_name_lower = cfg.model.name.lower()
    
    num_classes = cfg.dataset.num_classes
    in_channels = cfg.dataset.effective_in_channels

    logger.info(f"Instantiating model '{cfg.model.name}' | "
                f"Num Classes: {num_classes}, | In Channels: {in_channels}"
)
    # Routing logic (Factory Pattern)
    if "resnet_18_adapted" in model_name_lower:
        # Currently routes to the adapted ResNet-18 implementation
        model = build_resnet18_adapted(
            device=device, 
            num_classes=num_classes, 
            in_channels=in_channels,
            cfg=cfg
        )
    else:
        error_msg = f"Model architecture '{cfg.model.name}' is not recognized by the Factory."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized on {str(device).upper()} | "
                f"Total Parameters: {total_params:,}"
                )
    
    return model