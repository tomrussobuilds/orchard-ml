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
from scripts.core import Config, DATASET_REGISTRY
from .resnet_18_adapted import build_resnet18_adapted

# =========================================================================== #
#                                MODEL FACTORY LOGIC                          #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

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
    model_name_lower = cfg.model_name.lower()
    
    # FIXED: Normalize the dataset name to lowercase to match DATASET_REGISTRY keys
    dataset_key = cfg.dataset_name.lower()
    
    # FIXED: Use the normalized key for the check and class count retrieval
    if dataset_key not in DATASET_REGISTRY:
        available_datasets = ", ".join(DATASET_REGISTRY.keys())
        error_msg = (
            f"Dataset '{cfg.dataset_name}' not found in registry. "
            f"Available options are: {available_datasets}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    metadata = DATASET_REGISTRY[dataset_key] 
    num_classes = len(metadata.classes)
    in_channels = metadata.in_channels

    # Routing logic (Factory Pattern)
    if "resnet-18 adapted" in model_name_lower:
        # Currently routes to the adapted ResNet-18 implementation
        return build_resnet18_adapted(
            device=device, 
            num_classes=num_classes, 
            in_channels=in_channels,
            cfg=cfg
        )
    else:
        error_msg = f"Model architecture '{cfg.model_name}' is not recognized by the Factory."
        logger.error(error_msg)
        raise ValueError(error_msg)