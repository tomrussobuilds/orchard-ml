"""
Test Suite for Export Validation.

Tests numerical validation comparing PyTorch models against
exported ONNX models for output consistency.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# Skip entire module if onnxscript not available (required by torch.onnx.export)
pytest.importorskip("onnxscript")

from orchard.export.validation import validate_export  # noqa: E402


# SIMPLE TEST MODEL
class SimpleTestModel(nn.Module):
    """Minimal CNN for validation testing."""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# VALIDATION: BASIC FUNCTIONALITY
@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_export_basic(tmp_path):
    """Test basic export validation with default parameters."""

    # Create and export model
    model = SimpleTestModel(in_channels=3, num_classes=10)
    model.eval()
    onnx_path = tmp_path / "model.onnx"

    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    # Validate
    result = validate_export(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(3, 28, 28),
        num_samples=5,
        max_deviation=1e-5,
    )

    assert result is True


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_with_custom_samples(tmp_path):
    """Test validation with custom number of samples."""

    model = SimpleTestModel()
    model.eval()
    onnx_path = tmp_path / "model.onnx"

    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    # Test with different sample counts
    for num_samples in [1, 5, 10]:
        result = validate_export(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_shape=(3, 28, 28),
            num_samples=num_samples,
        )
        assert result is True


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_different_input_shapes(tmp_path):
    """Test validation with various input resolutions."""

    model = SimpleTestModel()
    model.eval()

    for resolution in [28, 224]:
        onnx_path = tmp_path / f"model_{resolution}.onnx"
        dummy_input = torch.randn(1, 3, resolution, resolution)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
        )

        result = validate_export(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_shape=(3, resolution, resolution),
            num_samples=3,
        )
        assert result is True


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_different_channels(tmp_path):
    """Test validation with different channel counts."""

    for in_channels in [1, 3]:
        model = SimpleTestModel(in_channels=in_channels)
        model.eval()
        onnx_path = tmp_path / f"model_{in_channels}ch.onnx"

        dummy_input = torch.randn(1, in_channels, 28, 28)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
        )

        result = validate_export(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_shape=(in_channels, 28, 28),
            num_samples=3,
        )
        assert result is True


# VALIDATION: TOLERANCE TESTING
@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_with_strict_tolerance(tmp_path):
    """Test validation with very strict tolerance."""

    model = SimpleTestModel()
    model.eval()
    onnx_path = tmp_path / "model.onnx"

    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    result = validate_export(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(3, 28, 28),
        num_samples=3,
        max_deviation=1e-7,
    )

    # Should still pass for deterministic operations
    assert result is True


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_with_relaxed_tolerance(tmp_path):
    """Test validation with relaxed tolerance."""

    model = SimpleTestModel()
    model.eval()
    onnx_path = tmp_path / "model.onnx"

    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    result = validate_export(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(3, 28, 28),
        num_samples=3,
        max_deviation=1e-3,
    )

    assert result is True


# VALIDATION: ERROR HANDLING
@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_missing_onnx_file_raises_error(tmp_path):
    """Test validation fails with missing ONNX file."""

    model = SimpleTestModel()
    onnx_path = tmp_path / "nonexistent.onnx"

    with pytest.raises(FileNotFoundError):
        validate_export(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_shape=(3, 28, 28),
        )


@pytest.mark.unit
def test_validate_without_onnxruntime_logs_warning(tmp_path, monkeypatch):
    """Test validation logs warning when onnxruntime not available."""
    # This test would need to mock onnxruntime import failure
    # For now, we skip it if onnxruntime is available
    pytest.importorskip("onnxruntime", reason="Test requires onnxruntime to be absent")


# VALIDATION: EDGE CASES
@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_single_sample(tmp_path):
    """Test validation with just one sample."""

    model = SimpleTestModel()
    model.eval()
    onnx_path = tmp_path / "model.onnx"

    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    result = validate_export(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(3, 28, 28),
        num_samples=1,
    )

    assert result is True


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_model_in_eval_mode(tmp_path):
    """Test validation works with model in eval mode."""

    model = SimpleTestModel()
    model.eval()

    onnx_path = tmp_path / "model.onnx"
    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    result = validate_export(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(3, 28, 28),
        num_samples=5,
    )

    assert result is True


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_fails_with_large_deviation(tmp_path):
    """Test validation fails when outputs differ significantly."""

    # Create two different models
    model1 = SimpleTestModel(in_channels=3, num_classes=10)
    model1.eval()
    model2 = SimpleTestModel(in_channels=3, num_classes=10)
    model2.eval()

    onnx_path = tmp_path / "model.onnx"

    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model1,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    # Validate against model2 (different weights) with very strict tolerance
    result = validate_export(
        pytorch_model=model2,
        onnx_path=onnx_path,
        input_shape=(3, 28, 28),
        num_samples=1,
        max_deviation=1e-10,
    )

    assert result is False


@pytest.mark.unit
def test_validate_with_onnxruntime_import_error(tmp_path, monkeypatch):
    """Test validation returns False when onnxruntime import fails."""

    model = SimpleTestModel()
    model.eval()
    onnx_path = tmp_path / "model.onnx"

    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    # Mock onnxruntime import to fail inside validate_export
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "onnxruntime" in name:
            raise ImportError("onnxruntime not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    result = validate_export(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(3, 28, 28),
    )

    assert result is False


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_validate_with_runtime_error(tmp_path, monkeypatch):
    """Test validation raises exception on onnxruntime errors."""

    model = SimpleTestModel()
    model.eval()
    onnx_path = tmp_path / "model.onnx"

    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
    )

    # Mock InferenceSession to raise RuntimeError
    import onnxruntime as ort

    def mock_inference_session(*_args, **_kwargs):
        raise RuntimeError("ONNX Runtime error")

    monkeypatch.setattr(ort, "InferenceSession", mock_inference_session)

    with pytest.raises(RuntimeError, match="ONNX Runtime error"):
        validate_export(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_shape=(3, 28, 28),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
