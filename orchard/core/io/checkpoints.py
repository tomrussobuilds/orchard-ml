"""
Model Checkpoint & Weight Management.

Handles secure restoration of model states and device mapping for neural networks.
Uses PyTorch's weights_only=True for security hardening against arbitrary code
execution attacks via malicious checkpoints.

Key Features:
    * Secure weight loading with weights_only=True
    * Device-aware tensor mapping (CPU/CUDA/MPS)
    * Existence validation before restoration
"""

from pathlib import Path

import torch


#  WEIGHT MANAGEMENT
def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    """
    Restores model state from a checkpoint using secure weight-only loading.

    Loads PyTorch state_dict from disk with security hardening (weights_only=True)
    to prevent arbitrary code execution. Automatically maps tensors to target device.

    Args:
        model: The model instance to populate with loaded weights
        path: Filesystem path to the checkpoint file (.pth)
        device: Target device for mapping the loaded tensors

    Raises:
        FileNotFoundError: If the checkpoint file does not exist at path

    Example:
        >>> model = build_resnet18_adapted(device, num_classes=10, in_channels=1, cfg)
        >>> checkpoint_path = Path("outputs/run_123/models/best_model.pth")
        >>> load_model_weights(model, checkpoint_path, device)
    """
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    # weights_only=True is used for security (avoids arbitrary code execution)
    state_dict = torch.load(path, map_location=device, weights_only=True)

    # Validate architecture compatibility before loading
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    if model_keys != checkpoint_keys:
        missing = model_keys - checkpoint_keys
        unexpected = checkpoint_keys - model_keys
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)[:5]}")
        if unexpected:
            parts.append(f"unexpected keys: {sorted(unexpected)[:5]}")
        raise RuntimeError(
            f"Checkpoint architecture mismatch ({', '.join(parts)}). "
            "Ensure the config matches the architecture used during training."
        )

    model.load_state_dict(state_dict)
