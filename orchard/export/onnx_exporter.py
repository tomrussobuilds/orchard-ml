"""
ONNX Export, Quantization, and Benchmarking.

End-to-end production pipeline for converting trained PyTorch checkpoints
to optimized ONNX graphs. The module is consumed by the CLI ``orchard``
export command and operates entirely on CPU.

Key Functions:

- ``export_to_onnx``: Trace-based export with dynamic batch axes, constant
  folding, and optional ``onnx.checker`` validation.
- ``quantize_model``: INT8 dynamic post-training quantization via
  onnxruntime (qnnpack for ARM, fbgemm for x86).
- ``benchmark_onnx_inference``: Warm-up + timed inference loop returning
  average latency in milliseconds.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from ..core import LOGGER_NAME, LogStyle

logger = logging.getLogger(LOGGER_NAME)


def export_to_onnx(
    model: nn.Module,
    checkpoint_path: Path,
    output_path: Path,
    input_shape: tuple[int, int, int],
    opset_version: int = 18,
    dynamic_axes: bool = True,
    do_constant_folding: bool = True,
    validate: bool = True,
) -> None:
    """
    Export trained PyTorch model to ONNX format.

    Args:
        model: PyTorch model architecture (uninitialized weights OK)
        checkpoint_path: Path to trained .pth checkpoint
        output_path: Output path for .onnx file
        input_shape: Input tensor shape (C, H, W)
        opset_version: ONNX opset version (18=latest, clean export with no warnings)
        dynamic_axes: Enable dynamic batch size (required for production)
        do_constant_folding: Optimize constant operations at export
        validate: Validate exported model with ONNX checker

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        RuntimeError: If state_dict loading fails (architecture mismatch).
        ValueError: If ONNX validation fails (when validate=True).

    Example:
        >>> export_to_onnx(
        ...     model=EfficientNet(),
        ...     checkpoint_path=Path("outputs/best_model.pth"),
        ...     output_path=Path("exports/model.onnx"),
        ... )
    """
    logger.info("  [Source]")
    logger.info(f"    {LogStyle.BULLET} Checkpoint        : {checkpoint_path.name}")
    logger.info("")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Move model to CPU for ONNX export
    model.cpu()
    model.eval()

    # Create dummy input (batch_size=1 for tracing)
    dummy_input = torch.randn(1, *input_shape)

    logger.info("  [Export Settings]")
    logger.info(f"    {LogStyle.BULLET} Format            : ONNX (opset {opset_version})")
    logger.info(f"    {LogStyle.BULLET} Input shape       : {tuple(dummy_input.shape)}")
    logger.info(f"    {LogStyle.BULLET} Dynamic axes      : {dynamic_axes}")
    logger.info("")

    # Prepare dynamic axes configuration
    if dynamic_axes:
        dynamic_axes_config = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    else:
        dynamic_axes_config = None

    # Export to ONNX (suppress verbose PyTorch warnings for cleaner output)
    # Temporarily suppress torch.onnx internal loggers to avoid roi_align warnings
    onnx_loggers = [
        logging.getLogger("torch.onnx._internal.exporter._schemas"),
        logging.getLogger("torch.onnx._internal.exporter"),
    ]
    original_levels = [log.level for log in onnx_loggers]

    try:
        # Raise log level to ERROR to suppress WARNING messages
        for onnx_logger in onnx_loggers:
            onnx_logger.setLevel(logging.ERROR)

        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            # Suppress warnings and stdout prints (e.g. ONNX rewrite rules)
            warnings.simplefilter("ignore")

            torch.onnx.export(
                model,
                (dummy_input,),  # Wrap in tuple for mypy type checking
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes_config,
                verbose=False,
            )
    finally:
        # Restore original log levels
        for onnx_logger, original_level in zip(onnx_loggers, original_levels):
            onnx_logger.setLevel(original_level)

    # Validate exported model
    if validate:
        try:
            import onnx

            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

            logger.info("  [Validation]")
            logger.info(f"    {LogStyle.BULLET} ONNX check        : {LogStyle.SUCCESS} Valid")
            size_mb = _onnx_file_size_mb(output_path)
            logger.info(f"    {LogStyle.BULLET} Model size        : {size_mb:.2f} MB")

        except ImportError:
            logger.warning(
                f"    {LogStyle.WARNING} onnx package not installed. Skipping validation."
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"    {LogStyle.FAILURE} ONNX validation failed: {e}")
            raise

    logger.info("")


def quantize_model(
    onnx_path: Path,
    output_path: Path | None = None,
    backend: str = "qnnpack",
) -> Path | None:
    """
    Apply INT8 dynamic post-training quantization to an ONNX model.

    Uses onnxruntime.quantization to reduce model size and improve
    inference speed with minimal accuracy loss.

    Args:
        onnx_path: Path to the exported ONNX model
        output_path: Path for quantized model (defaults to model_quantized.onnx
                     in the same directory)
        backend: Quantization backend ("qnnpack" for mobile/ARM, "fbgemm" for x86)

    Returns:
        Path to the quantized ONNX model, or None if quantization failed

    Example:
        >>> quantized = quantize_model(Path("exports/model.onnx"))
        >>> print(f"Quantized model: {quantized}")
    """
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        logger.warning(
            f"    {LogStyle.WARNING} onnxruntime.quantization not available. "
            "Skipping quantization."
        )
        return None

    if output_path is None:
        output_path = onnx_path.parent / "model_quantized.onnx"

    logger.info("  [Quantization]")
    logger.info(f"    {LogStyle.BULLET} Backend           : {backend}")

    preprocessed_path = onnx_path.parent / "model_preprocessed.onnx"
    try:
        # Clear intermediate value_info to avoid shape conflicts from dynamo exporter
        model_proto = onnx.load(str(onnx_path))
        while len(model_proto.graph.value_info) > 0:
            model_proto.graph.value_info.pop()
        onnx.save(model_proto, str(preprocessed_path))

        per_channel = backend == "fbgemm"

        # Suppress onnxruntime's "Please consider to run pre-processing before
        # quantization" warning.  We already handle pre-processing ourselves by
        # clearing value_info from the ONNX graph (see above) to work around
        # shape conflicts produced by the PyTorch dynamo-based exporter.
        # Running onnxruntime's own quant_pre_process() is unnecessary and would
        # re-introduce the conflicting shape annotations we just removed.
        import logging as _logging

        _root = _logging.getLogger()
        _prev_level = _root.level
        _root.setLevel(_logging.ERROR)
        try:
            quantize_dynamic(
                model_input=str(preprocessed_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
                per_channel=per_channel,
            )
        finally:
            _root.setLevel(_prev_level)

        original_mb = _onnx_file_size_mb(onnx_path)
        quantized_mb = _onnx_file_size_mb(output_path)
        ratio = original_mb / quantized_mb if quantized_mb > 0 else 0

        logger.info(
            f"    {LogStyle.BULLET} Size              : "
            f"{original_mb:.2f} MB → {quantized_mb:.2f} MB ({ratio:.1f}x)"
        )
        logger.info(f"    {LogStyle.BULLET} Status            : {LogStyle.SUCCESS} Done")
        logger.info("")

        return output_path

    except Exception as e:  # onnxruntime raises non-standard exceptions
        logger.error(f"    {LogStyle.FAILURE} Quantization failed: {e}")
        return None
    finally:
        preprocessed_path.unlink(missing_ok=True)


def benchmark_onnx_inference(
    onnx_path: Path,
    input_shape: tuple[int, int, int],
    num_runs: int = 100,
    seed: int = 42,
    label: str = "ONNX",
) -> float:
    """
    Benchmark ONNX model inference speed.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape (C, H, W)
        num_runs: Number of inference runs for averaging
        seed: Random seed for reproducible dummy input
        label: Display label for the benchmark log header

    Returns:
        Average inference time in milliseconds

    Example:
        >>> latency = benchmark_onnx_inference(Path("model.onnx"))
        >>> print(f"Latency: {latency:.2f}ms")
    """
    try:
        import time

        import numpy as np
        import onnxruntime as ort

        logger.info(f"  [Benchmark — {label}]")

        # Create inference session
        session = ort.InferenceSession(str(onnx_path))

        # Prepare dummy input using random Generator
        rng = np.random.default_rng(seed)
        dummy_input = rng.random(size=(1, *input_shape), dtype=np.float32) * 255

        # Warmup
        for _ in range(10):
            session.run(None, {"input": dummy_input})

        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            session.run(None, {"input": dummy_input})
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / num_runs) * 1000
        logger.info(f"    {LogStyle.BULLET} Runs              : {num_runs}")
        logger.info(f"    {LogStyle.BULLET} Avg latency       : {avg_latency_ms:.2f}ms")
        logger.info("")

        return avg_latency_ms

    except ImportError:
        logger.warning("onnxruntime not installed. Skipping benchmark.")
        return -1.0
    except Exception as e:  # onnxruntime raises non-standard exceptions
        logger.error(f"Benchmark failed: {e}")
        return -1.0


def _onnx_file_size_mb(path: Path) -> float:
    """
    Total size of an ONNX model including external data files (e.g. .data).
    """
    size = path.stat().st_size
    external = path.parent / f"{path.name}.data"
    if external.exists():
        size += external.stat().st_size
    return size / (1024 * 1024)
