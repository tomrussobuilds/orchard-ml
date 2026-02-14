# Model Export Guide

Convert trained VisionForge models to ONNX format for production deployment.

## Quick Start

Export a trained model by specifying the checkpoint and architecture parameters:

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

## Dataset Registry

Use `--dataset` to specify architecture metadata for your trained model.

### Supported Datasets

**MedMNIST** (28x28 and 224x224):
- `bloodmnist`, `pathmnist`, `chestmnist`, `dermamnist`, etc.

**Space/Astronomy** (224x224 only):
- `galaxy10` - Galaxy morphology classification (10 classes, RGB)

For datasets **not** in the registry, use any dataset name with matching resolution and class count - only the metadata is used.

## Common Options

### Quantization (Reduce model size)

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

### Custom output path

```bash
python forge.py \
    --checkpoint outputs/run_xyz/models/best_model.pth \
    --dataset galaxy10 \
    --resolution 224 \
    --model_name efficientnet_b0 \
    --format onnx \
    --output_path /path/to/production/model_v1.0.onnx
```

### Strict validation

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

### Skip validation (faster export)

```bash
python forge.py \
    --checkpoint outputs/run_xyz/models/best_model.pth \
    --dataset galaxy10 \
    --resolution 224 \
    --model_name efficientnet_b0 \
    --format onnx \
    --validate_export False
```

## CLI Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint` / `--resume` | Path to trained `.pth` checkpoint |
| `--format` | Export format: `onnx`, `torchscript`, or `both` |

### Model Architecture (required)

| Argument | Description | Example |
|----------|-------------|---------|
| `--model_name` | Model architecture | `efficientnet_b0` |
| `--dataset` | Dataset for metadata | `pathmnist`, `bloodmnist` |
| `--resolution` | Input size | `28`, `224` |

### ONNX Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--opset_version` | `18` | ONNX opset (18=latest, no warnings) |
| `--dynamic_axes` | `True` | Variable batch size |
| `--do_constant_folding` | `True` | Optimize constants |

### Quantization Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--quantize` | `False` | Enable INT8 quantization |
| `--quantization_backend` | `qnnpack` | `qnnpack` (mobile) or `fbgemm` (server) |

### Validation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--validate_export` | `True` | Compare PyTorch vs ONNX |
| `--validation_samples` | `10` | Number of test samples |
| `--max_deviation` | `1e-5` | Max allowed difference |

## Troubleshooting

### Checkpoint not found
Checkpoints are in the `models/` subdirectory:
```bash
--checkpoint outputs/run_xyz/models/best_model.pth
```

### Class count mismatch
Match `--dataset` to your trained model:
- Galaxy10 → `--dataset galaxy10` (10 classes)
- PathMNIST → `--dataset pathmnist` (9 classes)
- BloodMNIST → `--dataset bloodmnist` (8 classes)

### Validation failed
Relax tolerance with `--max_deviation 1e-4` or skip with `--validate_export False`

### Missing onnxscript
```bash
pip install onnx onnxruntime onnxscript
```

### Export warnings
Default `--opset_version 18` produces clean output with no warnings. If using lower versions (<18), you may see opset conversion warnings that are safe to ignore.

## Dataset Reference

| Resolution | Medical Datasets | Space/Astronomy |
|------------|------------------|-----------------|
| 28x28 | `bloodmnist`, `dermamnist`, `organmnist_axial` | - |
| 224x224 | `pathmnist`, `chestmnist`, `octmnist` | `galaxy10` |

## What Happens During Export

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

## Next Steps

- Deploy with [ONNX Runtime](https://onnxruntime.ai/)
- Optimize for edge devices with [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
- Convert to TensorRT for NVIDIA GPUs
