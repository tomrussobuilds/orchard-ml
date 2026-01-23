"""
Model Checkpoint & Weight Management.

Handles secure restoration of model states and device mapping for neural networks.
"""

# Standard Imports
from pathlib import Path

# Third-Party Imports
import torch


#  WEIGHT MANAGEMENT
def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    """
    Restores model state from a checkpoint using secure weight-only loading.

    Args:
        model (torch.nn.Module): The model instance to populate.
        path (Path): Filesystem path to the checkpoint file.
        device (torch.device): Target device for mapping the tensors.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    # weights_only=True is used for security (avoids arbitrary code execution)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
