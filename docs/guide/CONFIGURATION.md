← [Back to Main README](../../README.md)

# Configuration Guide

## Usage Patterns

### Configuration-Driven Execution

**Recommended Method:** YAML recipes ensure full reproducibility and version control.

```bash
# Verify environment (~30 seconds)
python -m tests.smoke_test

# Train with presets (28×28 resolution, CPU-compatible)
python forge.py --config recipes/config_resnet_18.yaml     # ~15 min GPU, ~2.5h CPU
python forge.py --config recipes/config_mini_cnn.yaml              # ~2-3 min GPU, ~10 min CPU

# Train with presets (224×224 resolution, GPU required)
python forge.py --config recipes/config_efficientnet_b0.yaml       # ~30 min each trial
python forge.py --config recipes/config_vit_tiny.yaml              # ~25-35 min each trial
```

### CLI Overrides

For rapid experimentation (not recommended for production):

```bash
# Quick test on different dataset
python forge.py --dataset dermamnist --epochs 10 --batch_size 64

# Custom learning rate schedule
python forge.py --lr 0.001 --min_lr 1e-7 --epochs 100

# Disable augmentations
python forge.py --mixup_alpha 0 --no_tta
```

> [!WARNING]
> **Configuration Precedence Order:**
> 1. **YAML file** (highest priority - if `--config` is provided)
> 2. **CLI arguments** (only used when no `--config` specified)
> 3. **Defaults** (from Pydantic field definitions)
>
> **When `--config` is provided, YAML values override CLI arguments.** This prevents configuration drift but means CLI flags are ignored. For reproducible research, always use YAML recipes.

---

## Configuration Reference

### Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `epochs` | int | 60 | [1, 1000] | Training epochs |
| `batch_size` | int | 128 | [1, 2048] | Samples per batch |
| `learning_rate` | float | 0.008 | (1e-8, 1.0) | Initial SGD learning rate |
| `min_lr` | float | 1e-6 | (0, lr) | Minimum LR for scheduler |
| `weight_decay` | float | 5e-4 | [0, 0.2] | L2 regularization |
| `momentum` | float | 0.9 | [0, 1] | SGD momentum |
| `mixup_alpha` | float | 0.002 | [0, 1] | MixUp strength (0=disabled) |
| `label_smoothing` | float | 0.0 | [0, 0.3] | Label smoothing factor |
| `dropout` | float | 0.0 | [0, 0.9] | Dropout probability |
| `seed` | int | 42 | - | Global random seed |
| `reproducible` | bool | False | - | Enable strict determinism |

### Augmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hflip` | float | 0.5 | Horizontal flip probability |
| `rotation_angle` | int | 10 | Max rotation degrees |
| `jitter_val` | float | 0.2 | ColorJitter intensity |
| `min_scale` | float | 0.95 | Minimum RandomResizedCrop scale |
| `no_tta` | bool | False | Disable test-time augmentation |

### Model Parameters

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `model_name` | str | "resnet_18" | `resnet_18`, `mini_cnn` (28×28); `efficientnet_b0`, `vit_tiny` (224×224) |
| `pretrained` | bool | True | Use ImageNet weights (N/A for MiniCNN) |
| `weight_variant` | str | None | ViT-specific pretrained variant (e.g., `augreg_in21k_ft_in1k`) |
| `force_rgb` | bool | True | Convert grayscale to 3-channel |
| `resolution` | int | 28 | [28, 224] |

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | "bloodmnist" | MedMNIST identifier |
| `data_root` | Path | `./dataset` | Dataset directory |
| `max_samples` | int | None | Cap training samples (debugging) |
| `use_weighted_sampler` | bool | True | Balance class distribution |

---

## Extending to New Datasets

The framework is designed for zero-code dataset integration via the registry system:

### 1. Add Dataset Metadata

Edit `orchard/core/metadata/medmnist_v2_28x28.py` or `medmnist_v2_224x224.py`:

```python
DATASET_REGISTRY = {
    "custom_dataset": DatasetMetadata(
        name="custom_dataset",
        num_classes=10,
        in_channels=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        native_resolution=28,
        class_names=["class0", "class1", ...],
        url="https://example.com/dataset.npz",
        md5="abc123...",
        is_anatomical=False,
        is_texture_based=True
    ),
}
```

### 2. Train Immediately

```bash
python forge.py --dataset custom_dataset --epochs 30
```

No code changes required—the configuration engine automatically resolves metadata.

---
