<h1 align="center">Model Export Guide</h1>

Convert trained Orchard ML models to ONNX format for production deployment.

<h2>Quick Start</h2>

<h3>Integrated Export (Recommended)</h3>

Add an `export:` section to your YAML recipe to export automatically after training:

```yaml
# recipes/config_efficientnet_b0.yaml
dataset:
  name: galaxy10
  resolution: 224
model:
  name: efficientnet_b0
training:
  epochs: 60
export:
  format: onnx
  opset_version: 18
```

```bash
orchard run recipes/config_efficientnet_b0.yaml
```

The pipeline will train, evaluate, and export to ONNX in a single run.

<h3>Standalone Export (via forge.py)</h3>

To export a previously trained checkpoint without re-training, use `forge.py` directly:

```bash
python forge.py \
    --checkpoint outputs/20260201_galaxy10_efficientnetb0_4a29ea_221107/models/best_efficientnetb0.pth \
    --dataset galaxy10 \
    --resolution 224 \
    --model_name efficientnet_b0 \
    --format onnx
```

**Output** (clean, no warnings with opset 18):
```
2026-02-14 13:12:37 - INFO - ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2026-02-14 13:12:37 - INFO -                       MODEL EXPORT
2026-02-14 13:12:37 - INFO - ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2026-02-14 13:12:37 - INFO -   [Source]
2026-02-14 13:12:37 - INFO -     • Checkpoint        : best_minicnn.pth
2026-02-14 13:12:37 - INFO -   [Export Settings]
2026-02-14 13:12:37 - INFO -     • Format            : ONNX (opset 18)
2026-02-14 13:12:37 - INFO -     • Input shape       : (1, 3, 28, 28)
2026-02-14 13:12:37 - INFO -     • Dynamic axes      : True
2026-02-14 13:12:38 - INFO -   [Validation]
2026-02-14 13:12:38 - INFO -     • ONNX check        : ✓ Valid
2026-02-14 13:12:38 - INFO -     • Model size        : 0.02 MB
2026-02-14 13:12:38 - INFO -
2026-02-14 13:12:38 - INFO -   ✓ Export completed
2026-02-14 13:12:38 - INFO -     » Output            : model.onnx
```

**Important**: The `--dataset` parameter is used only to determine model architecture metadata (resolution, channels, classes). No actual data is loaded during export.

<h2>Dataset Registry</h2>

Use `--dataset` to specify architecture metadata for your trained model.

<h3>Supported Datasets</h3>

**MedMNIST** (28x28 and 224x224):
- `bloodmnist`, `pathmnist`, `chestmnist`, `dermamnist`, etc.

**Space/Astronomy** (224x224 only):
- `galaxy10` - Galaxy morphology classification (10 classes, RGB)

For datasets **not** in the registry, use any dataset name with matching resolution and class count - only the metadata is used.

<h2>Common Options</h2>

> [!NOTE]
> The options below use `forge.py` for standalone export of existing checkpoints.
> For integrated export during training, configure the `export:` section in your YAML recipe.

<h3>Quantization (Reduce model size)</h3>

**Mobile deployment:**
```bash
python forge.py \
    --checkpoint outputs/run_xyz/models/best_model.pth \
    --dataset galaxy10 \
    --resolution 224 \
    --model_name efficientnet_b0 \
    --format onnx \
    --quantize \
    --quantization_backend qnnpack
```

**Server deployment:**
```bash
python forge.py \
    --checkpoint outputs/run_xyz/models/best_model.pth \
    --dataset galaxy10 \
    --resolution 224 \
    --model_name efficientnet_b0 \
    --format onnx \
    --quantize \
    --quantization_backend fbgemm
```

<h3>Custom output path</h3>

```bash
python forge.py \
    --checkpoint outputs/run_xyz/models/best_model.pth \
    --dataset galaxy10 \
    --resolution 224 \
    --model_name efficientnet_b0 \
    --format onnx \
    --output_path /path/to/production/model_v1.0.onnx
```

<h3>Strict validation</h3>

```bash
python forge.py \
    --checkpoint outputs/run_xyz/models/best_model.pth \
    --dataset galaxy10 \
    --resolution 224 \
    --model_name efficientnet_b0 \
    --format onnx \
    --validation_samples 100 \
    --max_deviation 1e-7
```

<h3>Skip validation (faster export)</h3>

```bash
python forge.py \
    --checkpoint outputs/run_xyz/models/best_model.pth \
    --dataset galaxy10 \
    --resolution 224 \
    --model_name efficientnet_b0 \
    --format onnx \
    --validate_export False
```

<h2>CLI Reference (forge.py standalone export)</h2>

<h3>Required Arguments</h3>

| Argument | Description |
|----------|-------------|
| `--checkpoint` / `--resume` | Path to trained `.pth` checkpoint |
| `--format` | Export format: `onnx`, `torchscript`, or `both` |

<h3>Model Architecture (required)</h3>

| Argument | Description | Example |
|----------|-------------|---------|
| `--model_name` | Model architecture | `efficientnet_b0` |
| `--dataset` | Dataset for metadata | `pathmnist`, `bloodmnist` |
| `--resolution` | Input size | `28`, `224` |

<h3>ONNX Options</h3>

| Argument | Default | Description |
|----------|---------|-------------|
| `--opset_version` | `18` | ONNX opset (18=latest, no warnings) |
| `--dynamic_axes` | `True` | Variable batch size |
| `--do_constant_folding` | `True` | Optimize constants |

<h3>Quantization Options</h3>

| Argument | Default | Description |
|----------|---------|-------------|
| `--quantize` | `False` | Enable INT8 quantization |
| `--quantization_backend` | `qnnpack` | `qnnpack` (mobile) or `fbgemm` (server) |

<h3>Validation Options</h3>

| Argument | Default | Description |
|----------|---------|-------------|
| `--validate_export` | `True` | Compare PyTorch vs ONNX |
| `--validation_samples` | `10` | Number of test samples |
| `--max_deviation` | `1e-5` | Max allowed difference |

<h2>Troubleshooting</h2>

<h3>Checkpoint not found</h3>

Checkpoints are in the `models/` subdirectory:
```bash
--checkpoint outputs/run_xyz/models/best_model.pth
```

<h3>Class count mismatch</h3>

Match `--dataset` to your trained model:
- Galaxy10 → `--dataset galaxy10` (10 classes)
- PathMNIST → `--dataset pathmnist` (9 classes)
- BloodMNIST → `--dataset bloodmnist` (8 classes)

<h3>Validation failed</h3>

Relax tolerance with `--max_deviation 1e-4` or skip with `--validate_export False`

<h3>Missing onnxscript</h3>

```bash
pip install onnx onnxruntime onnxscript
```

<h3>Export warnings</h3>

Default `--opset_version 18` produces clean output with no warnings. If using lower versions (<18), you may see opset conversion warnings that are safe to ignore.

---

<h2>Dataset Reference</h2>

| Resolution | Medical Datasets | Space/Astronomy |
|------------|------------------|-----------------|
| 28x28 | `bloodmnist`, `dermamnist`, `organmnist_axial` | - |
| 224x224 | `pathmnist`, `chestmnist`, `octmnist` | `galaxy10` |

---

<h2>What Happens During Export</h2>

1. **Checkpoint loading**: Loads trained weights from `.pth` file
2. **Model reconstruction**: Rebuilds architecture from config/CLI args
3. **ONNX conversion**: Exports to ONNX format with optimization
4. **Validation** (optional): Compares 10 random samples between PyTorch and ONNX
5. **Benchmarking** (optional): Measures inference latency

**Output location**: Same directory as checkpoint, with `.onnx` extension

Example:
```
outputs/run_xyz/models/best_efficientnetb0.pth
                     → best_efficientnetb0.onnx
```

<h2>Next Steps</h2>

- Deploy with [ONNX Runtime](https://onnxruntime.ai/)
- Optimize for edge devices with [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
- Convert to TensorRT for NVIDIA GPUs
