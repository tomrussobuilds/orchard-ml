"""
Test Suite for ONNX Exporter.

Tests ONNX export functionality including model conversion,
dynamic batch sizes, validation, and error handling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# Skip entire module if onnxscript not available (required by torch.onnx.export)
pytest.importorskip("onnxscript")

from orchard.export.onnx_exporter import (  # noqa: E402
    benchmark_onnx_inference,
    export_to_onnx,
)


# SIMPLE TEST MODEL
class SimpleTestModel(nn.Module):
    """Minimal CNN for export testing."""

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


# ONNX EXPORT: BASIC FUNCTIONALITY
@pytest.mark.unit
def test_export_to_onnx_basic(tmp_path):
    """Test basic ONNX export with default parameters."""

    # Create test model and checkpoint
    model = SimpleTestModel(in_channels=3, num_classes=10)
    checkpoint_path = tmp_path / "test_model.pth"
    output_path = tmp_path / "test_model.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    # Export to ONNX
    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        validate=False,
    )

    # Verify output file exists
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.unit
def test_export_with_dynamic_axes(tmp_path):
    """Test ONNX export with dynamic batch dimension."""

    model = SimpleTestModel(in_channels=1, num_classes=5)
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model_dynamic.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(1, 28, 28),
        dynamic_axes=True,
        validate=False,
    )

    assert output_path.exists()


@pytest.mark.unit
def test_export_without_dynamic_axes(tmp_path):
    """Test ONNX export with fixed batch dimension."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model_static.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        dynamic_axes=False,
        validate=False,
    )

    assert output_path.exists()


@pytest.mark.unit
def test_export_with_different_input_shapes(tmp_path):
    """Test ONNX export with various input resolutions."""

    model = SimpleTestModel()

    for resolution in [28, 224]:
        checkpoint_path = tmp_path / f"model_{resolution}.pth"
        output_path = tmp_path / f"model_{resolution}.onnx"

        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

        export_to_onnx(
            model=model,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            input_shape=(3, resolution, resolution),
            validate=False,
        )

        assert output_path.exists()


@pytest.mark.unit
def test_export_different_channels(tmp_path):
    """Test ONNX export with different input channel counts."""

    for in_channels in [1, 3]:
        model = SimpleTestModel(in_channels=in_channels, num_classes=8)
        checkpoint_path = tmp_path / f"model_{in_channels}ch.pth"
        output_path = tmp_path / f"model_{in_channels}ch.onnx"

        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

        export_to_onnx(
            model=model,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            input_shape=(in_channels, 28, 28),
            validate=False,
        )

        assert output_path.exists()


# ONNX EXPORT: OPSET VERSIONS
@pytest.mark.unit
def test_export_with_opset_13(tmp_path):
    """Test ONNX export with opset version 13 (stable)."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model_opset13.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        opset_version=13,
        validate=False,
    )

    assert output_path.exists()


@pytest.mark.unit
def test_export_with_opset_16(tmp_path):
    """Test ONNX export with opset version 16 (latest features)."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model_opset16.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        opset_version=16,
        validate=False,
    )

    assert output_path.exists()


# ONNX EXPORT: CONSTANT FOLDING
@pytest.mark.unit
def test_export_with_constant_folding(tmp_path):
    """Test ONNX export with constant folding enabled."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model_folded.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        do_constant_folding=True,
        validate=False,
    )

    assert output_path.exists()


@pytest.mark.unit
def test_export_without_constant_folding(tmp_path):
    """Test ONNX export without constant folding."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model_unfolded.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        do_constant_folding=False,
        validate=False,
    )

    assert output_path.exists()


# ONNX EXPORT: VALIDATION
@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_export_with_validation_enabled(tmp_path):
    """Test ONNX export with validation (requires onnxruntime)."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model_validated.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        validate=True,
    )

    assert output_path.exists()


# ONNX EXPORT: ERROR HANDLING
@pytest.mark.unit
def test_export_missing_checkpoint_raises_error(tmp_path):
    """Test ONNX export fails gracefully with missing checkpoint."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "nonexistent.pth"
    output_path = tmp_path / "model.onnx"

    with pytest.raises(FileNotFoundError):
        export_to_onnx(
            model=model,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            input_shape=(3, 28, 28),
            validate=False,
        )


@pytest.mark.unit
def test_export_creates_output_directory(tmp_path):
    """Test ONNX export creates output directory if needed."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_dir = tmp_path / "exports" / "nested"
    output_path = output_dir / "model.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        validate=False,
    )

    assert output_path.exists()
    assert output_dir.exists()


# ONNX EXPORT: BENCHMARK
@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_benchmark_onnx_inference(tmp_path):
    """Test ONNX inference benchmarking (requires onnxruntime)."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        validate=False,
    )

    # Benchmark should complete without errors
    benchmark_onnx_inference(
        onnx_path=output_path,
        input_shape=(3, 28, 28),
        num_runs=10,
    )


@pytest.mark.unit
def test_export_with_raw_state_dict(tmp_path):
    """Test ONNX export with raw state_dict (no model_state_dict wrapper)."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model.onnx"

    # Save raw state_dict without wrapper
    torch.save(model.state_dict(), checkpoint_path)

    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        validate=False,
    )

    assert output_path.exists()


@pytest.mark.unit
def test_export_without_onnx_validation(tmp_path, monkeypatch):
    """Test export when onnx package not available for validation."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    # Mock onnx import to fail
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("onnx not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Should complete without validation
    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_shape=(3, 28, 28),
        validate=True,
    )

    assert output_path.exists()


@pytest.mark.unit
def test_export_with_onnx_validation_failure(tmp_path, monkeypatch):
    """Test export handles ONNX validation exceptions."""

    model = SimpleTestModel()
    checkpoint_path = tmp_path / "model.pth"
    output_path = tmp_path / "model.onnx"

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    # Mock onnx.checker.check_model to raise an exception
    import onnx

    def mock_check_model(_):
        raise RuntimeError("ONNX validation error")

    monkeypatch.setattr(onnx.checker, "check_model", mock_check_model)

    with pytest.raises(RuntimeError, match="ONNX validation error"):
        export_to_onnx(
            model=model,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            input_shape=(3, 28, 28),
            validate=True,
        )


@pytest.mark.unit
def test_benchmark_without_onnxruntime(tmp_path, monkeypatch):
    """Test benchmark returns -1.0 when onnxruntime not available."""

    # Create a dummy ONNX file
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("dummy")

    # Mock onnxruntime import to fail
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    result = benchmark_onnx_inference(onnx_path, input_shape=(3, 28, 28))
    assert result == -1.0


@pytest.mark.unit
def test_benchmark_with_invalid_onnx(tmp_path):
    """Test benchmark returns -1.0 with invalid ONNX file."""

    # Create an invalid ONNX file
    onnx_path = tmp_path / "invalid.onnx"
    onnx_path.write_text("not a valid onnx file")

    result = benchmark_onnx_inference(onnx_path, input_shape=(3, 28, 28), num_runs=1)
    assert result == -1.0


@pytest.mark.unit
def test_benchmark_onnx_inference_success(monkeypatch):
    """Test benchmark_onnx_inference runs full inference loop."""

    mock_session = MagicMock()
    mock_session.run.return_value = None

    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)

    mock_rng = MagicMock()
    mock_rng.integers.return_value = np.zeros((1, 3, 224, 224), dtype=np.float32)

    mock_np = MagicMock()
    mock_np.random.default_rng.return_value = mock_rng
    mock_np.float32 = np.float32

    monkeypatch.setitem(sys.modules, "numpy", mock_np)

    latency = benchmark_onnx_inference(
        Path("fake_model.onnx"),
        input_shape=(3, 224, 224),
        num_runs=5,
    )

    assert latency >= 0
    assert mock_session.run.call_count == 10 + 5


# ONNX QUANTIZATION
def _export_simple_model(tmp_path):
    """Helper: export a SimpleTestModel and return (model, onnx_path)."""
    model = SimpleTestModel(in_channels=3, num_classes=10)
    checkpoint_path = tmp_path / "model.pth"
    onnx_path = tmp_path / "model.onnx"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    export_to_onnx(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=onnx_path,
        input_shape=(3, 28, 28),
        validate=False,
    )
    return model, onnx_path


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_quantize_model_basic(tmp_path):
    """Test basic INT8 quantization of exported ONNX model."""
    from orchard.export.onnx_exporter import quantize_model

    _, onnx_path = _export_simple_model(tmp_path)
    quantized_path = quantize_model(onnx_path)

    assert quantized_path is not None
    assert quantized_path.exists()
    assert quantized_path.stat().st_size > 0
    assert quantized_path.stat().st_size <= onnx_path.stat().st_size


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_quantize_model_default_output_name(tmp_path):
    """Test quantized model uses model_quantized.onnx by default."""
    from orchard.export.onnx_exporter import quantize_model

    _, onnx_path = _export_simple_model(tmp_path)
    quantized_path = quantize_model(onnx_path)

    assert quantized_path is not None
    assert quantized_path.name == "model_quantized.onnx"
    assert quantized_path.parent == onnx_path.parent


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_quantize_model_custom_output_path(tmp_path):
    """Test quantization with custom output path."""
    from orchard.export.onnx_exporter import quantize_model

    _, onnx_path = _export_simple_model(tmp_path)
    custom_path = tmp_path / "custom" / "quantized.onnx"
    custom_path.parent.mkdir(parents=True)

    quantized_path = quantize_model(onnx_path, output_path=custom_path)

    assert quantized_path is not None
    assert quantized_path == custom_path
    assert quantized_path.exists()


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_quantize_model_fbgemm_backend(tmp_path):
    """Test quantization with fbgemm backend (per-channel, x86)."""
    from orchard.export.onnx_exporter import quantize_model

    _, onnx_path = _export_simple_model(tmp_path)
    quantized_path = quantize_model(onnx_path, backend="fbgemm")

    assert quantized_path is not None
    assert quantized_path.exists()


@pytest.mark.unit
@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="Requires onnxruntime",
)
def test_quantize_model_qnnpack_backend(tmp_path):
    """Test quantization with qnnpack backend (per-tensor, mobile/ARM)."""
    from orchard.export.onnx_exporter import quantize_model

    _, onnx_path = _export_simple_model(tmp_path)
    quantized_path = quantize_model(onnx_path, backend="qnnpack")

    assert quantized_path is not None
    assert quantized_path.exists()


@pytest.mark.unit
def test_quantize_model_without_onnxruntime(tmp_path, monkeypatch):
    """Test quantize_model returns None when onnxruntime.quantization unavailable."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "onnxruntime.quantization" in name:
            raise ImportError("onnxruntime.quantization not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from orchard.export.onnx_exporter import quantize_model

    # Create a dummy ONNX file
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_text("dummy")

    result = quantize_model(onnx_path)
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
