← [Back to Home](../index.md)

<h1 align="center">Configuration Guide</h1>

<h2>Usage Patterns</h2>

<h3>Configuration-Driven Execution</h3>

**Recommended Method:** YAML recipes ensure full reproducibility and version control.

```bash
# Verify environment (~30 seconds)
python -m tests.smoke_test

# Train with presets (28×28 resolution, CPU-compatible)
orchard run recipes/config_resnet_18.yaml             # ~10-15 min GPU, ~2.5h CPU
orchard run recipes/config_mini_cnn.yaml              # ~2-3 min GPU, ~10 min CPU

# 32×32 resolution (CIFAR-10/100)
orchard run recipes/config_cifar10_mini_cnn.yaml      # ~3-5 min GPU
orchard run recipes/config_cifar10_resnet_18.yaml     # ~10-15 min GPU

# 128×128 resolution (GPU, timm models)
orchard run recipes/config_timm_efficientnet_lite0_128.yaml  # ~10 min GPU
orchard run recipes/config_timm_convnextv2_nano_128.yaml     # ~15 min GPU

# Train with presets (224×224 resolution, GPU required)
orchard run recipes/config_efficientnet_b0.yaml       # ~30 min each trial
orchard run recipes/config_vit_tiny.yaml              # ~25-35 min each trial
```

<h3>CLI Overrides</h3>

Use `--set` to override individual values without editing the YAML recipe:

```bash
# Quick test on different dataset
orchard run recipes/config_resnet_18.yaml --set dataset.name=dermamnist --set training.epochs=10

# Custom learning rate schedule
orchard run recipes/config_resnet_18.yaml --set training.learning_rate=0.001 --set training.min_lr=1e-7

# Disable augmentations
orchard run recipes/config_resnet_18.yaml --set augmentation.mixup_alpha=0
```

> [!TIP]
> **Configuration Precedence Order:**
> 1. **`--set` overrides** (highest priority)
> 2. **YAML recipe values**
> 3. **Defaults** (from Pydantic field definitions)
>
> The `--set` flag uses dot-notation paths matching the YAML structure (`training.epochs=30`, `dataset.name=pathmnist`). Values are auto-cast to the appropriate type (int, float, bool, null).

---

<h2>Configuration Reference</h2>

<h3>Core Parameters</h3>

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `epochs` | int | 60 | [1, 1000] | Training epochs |
| `batch_size` | int | 16 | [1, 128] | Samples per batch |
| `learning_rate` | float | 0.008 | (1e-8, 1.0) | Initial SGD learning rate |
| `min_lr` | float | 1e-6 | (1e-8, 1.0) | Minimum LR for scheduler |
| `weight_decay` | float | 5e-4 | [0, 0.2] | L2 regularization |
| `momentum` | float | 0.9 | [0, 1) | SGD momentum |
| `mixup_alpha` | float | 0.2 | [0, ∞) | MixUp strength (0=disabled) |
| `label_smoothing` | float | 0.0 | [0, 0.3] | Label smoothing factor |
| `seed` | int | 42 | - | Global random seed |
| `reproducible` | bool | False | - | Enable strict determinism |
| `use_tta` | bool | True | - | Enable test-time augmentation |

<h3>Augmentation Parameters</h3>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hflip` | float | 0.5 | Horizontal flip probability |
| `rotation_angle` | int | 10 | Max rotation degrees |
| `jitter_val` | float | 0.2 | ColorJitter intensity |
| `min_scale` | float | 0.9 | Minimum RandomResizedCrop scale |
| `tta_mode` | str | "full" | TTA strategy: `full` or `light` |
| `tta_translate` | float | 2.0 | Pixel translation range for TTA ensemble |
| `tta_scale` | float | 1.1 | Scale factor for TTA zoom augmentation |
| `tta_blur_sigma` | float | 0.4 | Gaussian blur sigma for TTA smoothing |
| `tta_blur_kernel_size` | int | 3 | Gaussian blur kernel size for TTA (must be odd) |

<h3>Architecture Parameters</h3>

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `name` | str | "mini_cnn" | `resnet_18`, `mini_cnn` (28/32/64); `timm/*` (128); `efficientnet_b0`, `vit_tiny` (224) |
| `pretrained` | bool | False | Use ImageNet weights (N/A for MiniCNN) |
| `dropout` | float | 0.2 | [0, 0.9] · Dropout probability (wired for mini_cnn, timm) |
| `weight_variant` | str | None | ViT-specific pretrained variant (e.g., `augreg_in21k_ft_in1k`) |

<h3>Dataset Parameters</h3>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | "bloodmnist" | Dataset identifier (MedMNIST, CIFAR-10/100, Galaxy10) |
| `data_root` | Path | `./dataset` | Dataset directory |
| `resolution` | int | 28 | Target resolution: {28, 32, 64, 128, 224} |
| `force_rgb` | bool | True | Convert grayscale to 3-channel |
| `max_samples` | int | None | Cap training samples (debugging) |
| `use_weighted_sampler` | bool | True | Balance class distribution |
| `val_ratio` | float | 0.10 | Validation/test split ratio |
| `lazy_loading` | bool | True | Memory-map NPZ files instead of loading into RAM |

---

<h2>Extending to New Datasets</h2>

The framework is designed for zero-code dataset integration via the registry system:

<h3>1. Add Dataset Metadata</h3>

Create a new domain file in `orchard/core/metadata/domains/` (e.g., `custom.py`):

```python
REGISTRY_28: Final[dict[str, DatasetMetadata]] = {
    "custom_dataset": DatasetMetadata(
        name="custom_dataset",
        display_name="Custom Dataset",
        md5_checksum="abc123...",
        url="https://example.com/dataset.npz",
        path=DATASET_DIR / "custom_dataset_28.npz",
        classes=["class_a", "class_b", "class_c"],
        in_channels=3,
        native_resolution=28,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        is_anatomical=False,
        is_texture_based=True,
    ),
}
```

<h3>2. Train Immediately</h3>

```bash
orchard run recipes/config_resnet_18.yaml --set dataset.name=custom_dataset --set training.epochs=30
```

No code changes required—the configuration engine automatically resolves metadata.

---
