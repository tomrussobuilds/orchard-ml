"""
Model Export Package.

Provides utilities for exporting trained PyTorch models to production formats
(ONNX, TorchScript) with validation and optimization support.

Example:
    >>> from orchard.export import export_to_onnx
    >>> export_to_onnx(
    ...     model=trained_model,
    ...     checkpoint_path="outputs/best_model.pth",
    ...     output_path="exports/model.onnx",
    ...     input_shape=(3, 224, 224),
    ... )
"""

from .onnx_exporter import export_to_onnx
from .validation import validate_export

__all__ = ["export_to_onnx", "validate_export"]
