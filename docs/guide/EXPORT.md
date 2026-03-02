← [Back to Home](../index.md)

<h1 align="center">Model Export Guide</h1>

Convert trained Orchard ML models to ONNX for production deployment.

<h2>How It Works</h2>

Add an `export:` section to any YAML recipe. The pipeline will **train, evaluate, and export** in a single run:

```bash
orchard run recipes/config_efficientnet_b0.yaml
```

After training completes, the export phase automatically:
1. Loads the best checkpoint from `checkpoints/`
2. Traces the model and converts to ONNX
3. Validates PyTorch vs exported output (optional)
4. Benchmarks inference latency (optional)

```
outputs/20260219_galaxy10_efficientnetb0_abc123/
  checkpoints/
    best_efficientnetb0.pth     # PyTorch checkpoint
  exports/
    model.onnx                  # Production-ready ONNX export
```

<h2>Configuration</h2>

All export behavior is controlled via the `export:` section of your YAML recipe.

**Minimal (defaults are sensible):**

```yaml
export:
  format: onnx
```

**Full reference:**

```yaml
export:
  # Format
  format: onnx                    # only ONNX supported

  # ONNX settings
  opset_version: 18               # 18 = latest, no conversion warnings
  dynamic_axes: true              # dynamic batch size for flexible inference
  do_constant_folding: true       # fold constants at export time

  # Quantization
  quantize: false                 # apply post-training quantization
  quantization_type: int8         # int8 | uint8 | int4 | uint4
  quantization_backend: qnnpack   # qnnpack (mobile/ARM) | fbgemm (x86)

  # Validation
  validate_export: true           # compare PyTorch vs exported output
  validation_samples: 10          # number of samples for validation
  max_deviation: 1.0e-04          # max allowed numerical deviation

  # Benchmark
  benchmark: false                # run ONNX inference latency benchmark
```

| Field | Default | Description |
|-------|---------|-------------|
| `format` | `onnx` | Export format (only ONNX supported) |
| `opset_version` | `18` | ONNX opset version (18 recommended) |
| `dynamic_axes` | `true` | Enable dynamic batch size for inference flexibility |
| `do_constant_folding` | `true` | Optimize constant operations during export |
| `quantize` | `false` | Apply post-training quantization |
| `quantization_type` | `int8` | Weight type: `int8`, `uint8` (server), `int4`, `uint4` (edge) |
| `quantization_backend` | `qnnpack` | Quantization backend: `qnnpack` (mobile/ARM), `fbgemm` (x86) |
| `validate_export` | `true` | Run numerical validation after export |
| `validation_samples` | `10` | Number of random samples for validation |
| `max_deviation` | `1e-4` | Maximum allowed output deviation (PyTorch vs ONNX) |
| `benchmark` | `false` | Run inference latency benchmark after export |

<h2>Quantization</h2>

Orchard ML supports dynamic post-training quantization via ONNX Runtime with four weight types:

| Weight Type | Bits | Target | Notes |
|-------------|------|--------|-------|
| `int8` | 8 | Server / general | Default, all layers quantized |
| `uint8` | 8 | Server / general | Unsigned variant |
| `int4` | 4 | Edge / mobile | Only FC layers quantized (Conv stays FP32) |
| `uint4` | 4 | Edge / mobile | Unsigned variant |

**Server deployment (INT8):**

```yaml
export:
  format: onnx
  quantize: true
  quantization_type: int8
  quantization_backend: fbgemm   # x86 servers
```

**Edge deployment (INT4):**

```yaml
export:
  format: onnx
  quantize: true
  quantization_type: int4
  quantization_backend: qnnpack   # mobile / ARM
```

| Backend | Target Hardware | Quantization Style |
|---------|----------------|--------------------|
| `qnnpack` | Mobile / ARM | Per-tensor |
| `fbgemm` | x86 servers | Per-channel |

After export, the output directory will contain both models:

```
exports/
  model.onnx                  # Full-precision original
  model_quantized.onnx        # Quantized (INT8/INT4/...)
```

> **Note:** 4-bit quantization (INT4/UINT4) only quantizes Gemm nodes (fully-connected
> layers). Conv layers remain at full precision because ONNX Runtime's 4-bit packing
> does not support convolution weights. This is the standard approach for edge-deployed
> vision models.
>
> The `validate_export` check runs against the original (non-quantized) ONNX model.
> If `benchmark: true` is set, both models are benchmarked for latency comparison.

<h2>Troubleshooting</h2>

<h3>Validation failed</h3>

If numerical deviations exceed `max_deviation`, relax the tolerance or set `validate_export: false`.

<h3>Missing onnxscript</h3>

```bash
pip install onnx onnxruntime onnxscript
```

<h3>Export warnings</h3>

`opset_version: 18` produces clean output. Lower versions may emit harmless conversion warnings.

<h2>Next Steps</h2>

- Deploy with [ONNX Runtime](https://onnxruntime.ai/)
- Optimize for edge devices with [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
- Convert to TensorRT for NVIDIA GPUs
