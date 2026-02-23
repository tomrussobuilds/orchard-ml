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
  quantize: false                 # apply INT8 post-training quantization
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
| `quantize` | `false` | Apply INT8 post-training quantization |
| `quantization_backend` | `qnnpack` | Quantization backend: `qnnpack` (mobile/ARM), `fbgemm` (x86) |
| `validate_export` | `true` | Run numerical validation after export |
| `validation_samples` | `10` | Number of random samples for validation |
| `max_deviation` | `1e-4` | Maximum allowed output deviation (PyTorch vs ONNX) |
| `benchmark` | `false` | Run inference latency benchmark after export |

<h2>Quantization</h2>

Orchard ML supports INT8 dynamic post-training quantization via ONNX Runtime.
Quantization reduces model size by 2-4x and can improve inference speed on compatible hardware.

**Enable quantization:**

```yaml
export:
  format: onnx
  quantize: true
  quantization_backend: fbgemm   # x86 servers
```

| Backend | Target Hardware | Quantization Style |
|---------|----------------|--------------------|
| `qnnpack` | Mobile / ARM | Per-tensor |
| `fbgemm` | x86 servers | Per-channel |

After export, the output directory will contain both models:

```
exports/
  model.onnx                  # Full-precision original
  model_quantized.onnx        # INT8 quantized
```

> **Note:** INT8 quantization introduces small numerical deviations compared to the
> full-precision model. The `validate_export` check runs against the original
> (non-quantized) ONNX model. If `benchmark: true` is set, both models are benchmarked
> for latency comparison.

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
