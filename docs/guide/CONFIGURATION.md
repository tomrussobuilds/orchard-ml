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

<h3>Augmentation Parameters</h3>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hflip` | float | 0.5 | Horizontal flip probability |
| `rotation_angle` | int | 10 | Max rotation degrees |
| `jitter_val` | float | 0.2 | ColorJitter intensity |
| `min_scale` | float | 0.95 | Minimum RandomResizedCrop scale |
| `no_tta` | bool | False | Disable test-time augmentation |

<h3>Model Parameters</h3>

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `model_name` | str | "resnet_18" | `resnet_18`, `mini_cnn` (28/32/64); `efficientnet_b0`, `vit_tiny` (224) |
| `pretrained` | bool | True | Use ImageNet weights (N/A for MiniCNN) |
| `weight_variant` | str | None | ViT-specific pretrained variant (e.g., `augreg_in21k_ft_in1k`) |
| `force_rgb` | bool | True | Convert grayscale to 3-channel |
| `resolution` | int | 28 | {28, 32, 64, 224} |

<h3>Dataset Parameters</h3>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | "bloodmnist" | Dataset identifier (MedMNIST, CIFAR-10/100, Galaxy10) |
| `data_root` | Path | `./dataset` | Dataset directory |
| `max_samples` | int | None | Cap training samples (debugging) |
| `use_weighted_sampler` | bool | True | Balance class distribution |

---

<h2>Extending to New Datasets</h2>

The framework is designed for zero-code dataset integration via the registry system:

<h3>1. Add Dataset Metadata</h3>

Edit the appropriate domain file in `orchard/core/metadata/domains/` (e.g., `medical.py` or `space.py`):

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

<h3>2. Train Immediately</h3>

```bash
orchard run recipes/config_resnet_18.yaml --set dataset.name=custom_dataset --set training.epochs=30
```

No code changes required—the configuration engine automatically resolves metadata.

---
