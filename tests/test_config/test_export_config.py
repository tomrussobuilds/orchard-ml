"""
Test Suite for ExportConfig.

Tests model export configuration, format validation, ONNX
parameters, quantization settings, and validation options.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchard.core.config import ExportConfig


# EXPORT CONFIG: DEFAULTS
@pytest.mark.unit
def test_export_config_defaults():
    """Test ExportConfig with default values."""
    config = ExportConfig()

    assert config.format == "onnx"
    assert config.opset_version == 18
    assert config.dynamic_axes is True
    assert config.do_constant_folding is True
    assert config.quantize is False
    assert config.quantization_backend == "qnnpack"
    assert config.validate_export is True
    assert config.validation_samples == 10
    assert config.max_deviation == pytest.approx(1e-4)
    assert config.benchmark is False


@pytest.mark.unit
def test_export_config_custom_values():
    """Test ExportConfig with custom parameters."""
    config = ExportConfig(
        format="onnx",
        opset_version=16,
        dynamic_axes=False,
        quantize=True,
        quantization_backend="fbgemm",
    )

    assert config.format == "onnx"
    assert config.opset_version == 16
    assert config.dynamic_axes is False
    assert config.quantize is True
    assert config.quantization_backend == "fbgemm"


# EXPORT CONFIG: FORMAT VALIDATION
@pytest.mark.unit
def test_valid_formats():
    """Test only ONNX format is accepted."""
    config = ExportConfig(format="onnx")
    assert config.format == "onnx"


@pytest.mark.unit
def test_invalid_format_rejected():
    """Test non-onnx formats are rejected."""
    for fmt in ["torchscript", "both", "invalid_format"]:
        with pytest.raises(ValidationError):
            ExportConfig(format=fmt)


# EXPORT CONFIG: ONNX PARAMETERS
@pytest.mark.unit
def test_opset_version_positive():
    """Test opset_version must be positive."""
    config = ExportConfig(opset_version=16)
    assert config.opset_version == 16

    with pytest.raises(ValidationError):
        ExportConfig(opset_version=0)

    with pytest.raises(ValidationError):
        ExportConfig(opset_version=-1)


@pytest.mark.unit
def test_dynamic_axes_boolean():
    """Test dynamic_axes accepts boolean values."""
    config_true = ExportConfig(dynamic_axes=True)
    assert config_true.dynamic_axes is True

    config_false = ExportConfig(dynamic_axes=False)
    assert config_false.dynamic_axes is False


@pytest.mark.unit
def test_constant_folding_boolean():
    """Test do_constant_folding accepts boolean values."""
    config_true = ExportConfig(do_constant_folding=True)
    assert config_true.do_constant_folding is True

    config_false = ExportConfig(do_constant_folding=False)
    assert config_false.do_constant_folding is False


# EXPORT CONFIG: QUANTIZATION
@pytest.mark.unit
def test_quantization_disabled_by_default():
    """Test quantization is disabled by default."""
    config = ExportConfig()
    assert config.quantize is False


@pytest.mark.unit
def test_quantization_can_be_enabled():
    """Test quantization can be enabled."""
    config = ExportConfig(quantize=True)
    assert config.quantize is True


@pytest.mark.unit
def test_valid_quantization_backends():
    """Test valid quantization backends are accepted."""
    for backend in ["qnnpack", "fbgemm"]:
        config = ExportConfig(quantization_backend=backend)
        assert config.quantization_backend == backend


@pytest.mark.unit
def test_invalid_quantization_backend_rejected():
    """Test invalid quantization backend is rejected."""
    with pytest.raises(ValidationError):
        ExportConfig(quantization_backend="invalid_backend")


# EXPORT CONFIG: VALIDATION PARAMETERS
@pytest.mark.unit
def test_validation_enabled_by_default():
    """Test export validation is enabled by default."""
    config = ExportConfig()
    assert config.validate_export is True


@pytest.mark.unit
def test_validation_can_be_disabled():
    """Test export validation can be disabled."""
    config = ExportConfig(validate_export=False)
    assert config.validate_export is False


@pytest.mark.unit
def test_validation_samples_positive():
    """Test validation_samples must be positive."""
    config = ExportConfig(validation_samples=50)
    assert config.validation_samples == 50

    with pytest.raises(ValidationError):
        ExportConfig(validation_samples=0)

    with pytest.raises(ValidationError):
        ExportConfig(validation_samples=-10)


@pytest.mark.unit
def test_max_deviation_accepts_float():
    """Test max_deviation accepts float values."""
    config = ExportConfig(max_deviation=1e-3)
    assert config.max_deviation == pytest.approx(1e-3)

    config2 = ExportConfig(max_deviation=0.001)
    assert config2.max_deviation == pytest.approx(0.001)


# EXPORT CONFIG: IMMUTABILITY
@pytest.mark.unit
def test_config_is_frozen():
    """Test ExportConfig is immutable after creation."""
    config = ExportConfig()

    with pytest.raises(ValidationError):
        config.format = "changed"


@pytest.mark.unit
def test_config_forbids_extra_fields():
    """Test ExportConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        ExportConfig(unknown_param="value")


# EXPORT CONFIG: INTEGRATION SCENARIOS
@pytest.mark.unit
def test_onnx_export_config():
    """Test typical ONNX export configuration."""
    config = ExportConfig(
        format="onnx",
        opset_version=16,
        dynamic_axes=True,
        do_constant_folding=True,
        validate_export=True,
        validation_samples=20,
    )

    assert config.format == "onnx"
    assert config.opset_version == 16
    assert config.dynamic_axes is True
    assert config.validate_export is True


@pytest.mark.unit
def test_quantized_export_config():
    """Test quantized export configuration."""
    config = ExportConfig(
        format="onnx",
        quantize=True,
        quantization_backend="fbgemm",
        validate_export=True,
        max_deviation=1e-3,
    )

    assert config.quantize is True
    assert config.quantization_backend == "fbgemm"
    assert config.max_deviation == pytest.approx(1e-3)


@pytest.mark.unit
def test_benchmark_can_be_enabled():
    """Test benchmark flag can be enabled."""
    config = ExportConfig(benchmark=True)
    assert config.benchmark is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
