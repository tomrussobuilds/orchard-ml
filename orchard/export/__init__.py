"""
Model Export Package.

Provides utilities for exporting trained PyTorch models to ONNX format
with validation, benchmarking, and optimization support.

Example:
    >>> from orchard.export import export_to_onnx
    >>> export_to_onnx(
    ...     model=trained_model,
    ...     checkpoint_path="outputs/best_model.pth",
    ...     output_path="exports/model.onnx",
    ...     input_shape=(3, 224, 224),
    ... )
"""

from .onnx_exporter import benchmark_onnx_inference, export_to_onnx, quantize_model
from .validation import validate_export

__all__ = ["benchmark_onnx_inference", "export_to_onnx", "quantize_model", "validate_export"]
