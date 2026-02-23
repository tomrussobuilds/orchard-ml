"""
ONNX Model Exporter.

Converts trained PyTorch models to ONNX format for production deployment.
Supports dynamic batch sizes, optimization, and validation.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from ..core import LOGGER_NAME
from ..core.logger import LogStyle

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

    Example:
        >>> export_to_onnx(
        ...     model=EfficientNet(),
        ...     checkpoint_path=Path("outputs/best_model.pth"),
        ...     output_path=Path("exports/model.onnx"),
        ... )
    """
    logger.info("  [Source]")
    logger.info(f"    {LogStyle.BULLET} Checkpoint        : {checkpoint_path.name}")

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

            # Report model size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            logger.info("  [Validation]")
            logger.info(f"    {LogStyle.BULLET} ONNX check        : {LogStyle.SUCCESS} Valid")
            logger.info(f"    {LogStyle.BULLET} Model size        : {file_size_mb:.2f} MB")

        except ImportError:
            logger.warning(
                f"    {LogStyle.WARNING} onnx package not installed. Skipping validation."
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"    {LogStyle.WARNING} ONNX validation failed: {e}")
            raise

    logger.info("")


def benchmark_onnx_inference(
    onnx_path: Path,
    input_shape: tuple[int, int, int],
    num_runs: int = 100,
    seed: int = 42,
) -> float:
    """
    Benchmark ONNX model inference speed.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape (C, H, W)
        num_runs: Number of inference runs for averaging
        seed: Random seed for reproducible dummy input

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

        logger.info("  [Benchmark]")

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
        logger.info(f"    {LogStyle.BULLET} Avg latency       : " f"{avg_latency_ms:.2f}ms")

        return avg_latency_ms

    except ImportError:
        logger.warning("onnxruntime not installed. Skipping benchmark.")
        return -1.0
    except Exception as e:  # noqa: broad-except — onnxruntime raises non-standard exceptions
        logger.error(f"Benchmark failed: {e}")
        return -1.0
