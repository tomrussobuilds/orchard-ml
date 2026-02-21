"""
Export Validation Utilities.

Validates exported models by comparing outputs against original PyTorch models.
Ensures numerical consistency and correctness after export.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ..core import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def validate_export(
    pytorch_model: nn.Module,
    onnx_path: Path,
    input_shape: Tuple[int, int, int],
    num_samples: int = 10,
    max_deviation: float = 1e-5,
) -> bool:
    """
    Validate ONNX export against PyTorch model.

    Compares outputs from PyTorch and ONNX models on random inputs
    to ensure numerical consistency after export.

    Args:
        pytorch_model: Original PyTorch model (with loaded weights)
        onnx_path: Path to exported ONNX model
        input_shape: Input tensor shape (C, H, W)
        num_samples: Number of random samples to test
        max_deviation: Maximum allowed absolute difference

    Returns:
        True if validation passes, False otherwise

    Example:
        >>> model.load_state_dict(torch.load("checkpoint.pth"))
        >>> valid = validate_export(model, Path("model.onnx"))
        >>> if valid:
        ...     print("Export validated successfully!")
    """
    # Check if ONNX file exists
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    try:
        import onnxruntime as ort

        logger.info("Validating ONNX export against PyTorch model")
        logger.info(f"  Test samples: {num_samples}")
        logger.info(f"  Max deviation: {max_deviation}")

        # Load ONNX model
        session = ort.InferenceSession(str(onnx_path))

        pytorch_model.eval()
        pytorch_model.cpu()

        max_diff = 0.0

        with torch.no_grad():
            for i in range(num_samples):
                # Generate random input
                x_torch = torch.randn(1, *input_shape)
                x_numpy = x_torch.numpy().astype(np.float32)

                # PyTorch inference
                y_torch = pytorch_model(x_torch).numpy()

                # ONNX inference
                y_onnx = session.run(None, {"input": x_numpy})[0]

                # Compare outputs
                diff = np.abs(y_torch - y_onnx).max()
                max_diff = max(max_diff, diff)

                if diff > max_deviation:
                    logger.error(
                        f"✗ Validation failed on sample {i + 1}: "
                        f"max diff = {diff:.2e} (threshold = {max_deviation:.2e})"
                    )
                    return False

        logger.info(f"✓ Validation passed! Max deviation: {max_diff:.2e}")
        return True

    except ImportError as e:
        logger.warning(f"onnxruntime not installed. Skipping validation: {e}")
        return False
    except (RuntimeError, ValueError) as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise
