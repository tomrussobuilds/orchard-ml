"""
Export Configuration Schema.

Pydantic v2 schema defining model export parameters for ONNX.
Supports optimization and validation settings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .types import PositiveFloat, PositiveInt


# EXPORT CONFIGURATION
class ExportConfig(BaseModel):
    """
    Model export configuration for production deployment.

    Defines ONNX export settings, optimization level, and validation parameters.

    Attributes:
        format: Export format (only 'onnx' supported).
        opset_version: ONNX opset version (18 recommended).
        dynamic_axes: Enable dynamic batch size for flexible inference.
        do_constant_folding: Optimize constant operations during export.
        quantize: Enable post-training quantization.
        quantization_type: Weight type — int8/uint8 for server, int4/uint4 for edge.
        quantization_backend: Backend — qnnpack for ARM/mobile, fbgemm for x86.
        validate_export: Validate exported model matches PyTorch output.
        validation_samples: Number of samples for export validation.
        max_deviation: Maximum allowed output deviation for validation.
        benchmark: Run inference latency benchmark after export.

    Example:
        >>> cfg = ExportConfig(format="onnx", opset_version=18)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ==================== Export Format ====================
    # Reserved for future multi-format support (e.g., TorchScript, CoreML)
    format: Literal["onnx"] = Field(
        default="onnx", description="Export format (only ONNX supported)"
    )

    # ==================== ONNX Settings ====================
    opset_version: PositiveInt = Field(
        default=18,
        description="ONNX opset version (18=latest, no conversion warnings). "
        "Lower versions may trigger fallback.",
    )

    dynamic_axes: bool = Field(
        default=True, description="Enable dynamic batch size (required for inference)"
    )

    do_constant_folding: bool = Field(
        default=True, description="Optimize constant operations at export time"
    )

    # ==================== Quantization ====================
    quantize: bool = Field(default=False, description="Apply post-training quantization")

    quantization_type: Literal["int8", "uint8", "int4", "uint4"] = Field(
        default="int8",
        description="Weight quantization type (int4/uint4 for edge, int8/uint8 for server)",
    )

    quantization_backend: Literal["qnnpack", "fbgemm"] = Field(
        default="qnnpack", description="Quantization backend (qnnpack=mobile, fbgemm=x86)"
    )

    # ==================== Validation ====================
    validate_export: bool = Field(
        default=True, description="Validate exported model against PyTorch"
    )

    validation_samples: PositiveInt = Field(
        default=10, description="Number of samples for validation"
    )

    max_deviation: PositiveFloat = Field(
        default=1e-4,
        description="Maximum allowed output deviation between PyTorch and exported model",
    )

    # ==================== Benchmark ====================
    benchmark: bool = Field(
        default=False, description="Run ONNX inference latency benchmark after export"
    )

    benchmark_runs: PositiveInt = Field(
        default=100, description="Number of inference runs for latency averaging"
    )
