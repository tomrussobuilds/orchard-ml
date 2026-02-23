← [Back to Home](../index.md)

<h1 align="center">Hyperparameter Optimization Guide</h1>

<h3>Quick Start</h3>

```bash
# Install Optuna (if not already present)
pip install optuna plotly timm  # timm required for ViT support

# Run optimization with presets (28×28 MedMNIST)
orchard run recipes/optuna_resnet_18.yaml          # 50 trials, ~15 min GPU, ~2.5h CPU
orchard run recipes/optuna_mini_cnn.yaml           # 50 trials, ~1-2 min GPU, ~5 min CPU

# 32×32 resolution (CIFAR-10/100)
orchard run recipes/optuna_cifar100_mini_cnn.yaml  # 50 trials, ~1-2h GPU
orchard run recipes/optuna_cifar100_resnet_18.yaml # 50 trials, ~3-4h GPU

# 224×224 resolution (includes weight variant search for ViT)
orchard run recipes/optuna_efficientnet_b0.yaml    # 20 trials, ~1.5-5h GPU
orchard run recipes/optuna_vit_tiny.yaml           # 20 trials, ~3-5h GPU

# Custom search via --set overrides
orchard run recipes/optuna_resnet_18.yaml \
    --set dataset.name=pathmnist \
    --set optuna.n_trials=20 \
    --set training.epochs=10 \
    --set optuna.search_space_preset=quick

# Resume interrupted study
orchard run recipes/optuna_vit_tiny.yaml \
    --set optuna.load_if_exists=true
```

<h3>Search Space Coverage</h3>

Select a preset via `search_space_preset`:

| Preset | Parameters | Use case |
|---|---|---|
| **`full`** (default) | 13+ parameters | Comprehensive search |
| **`quick`** | 4 parameters | Rapid exploration |
| **`architectures`** | Full + model search | Best model-hyperparameter combo |

**Full Space** parameters:
- **Optimization**: `learning_rate`, `weight_decay`, `momentum`, `min_lr`
- **Regularization**: `mixup_alpha`, `label_smoothing`, `dropout`
- **Scheduling**: `cosine_fraction`, `scheduler_patience`
- **Augmentation**: `rotation_angle`, `jitter_val`, `min_scale`
- **Batch Size**: Resolution-aware categorical choices
  - ≤32×32: `batch_size_low_res` — [16, 32, 48, 64]
  - 224×224: `batch_size_high_res` — [8, 12, 16] (OOM-safe for 8GB VRAM)
- **Architecture** (requires `enable_model_search: true`):
  - ≤64×64: [`resnet_18`, `mini_cnn`]
  - 224×224: [`resnet_18`, `efficientnet_b0`, `convnext_tiny`, `vit_tiny`]
- **Weight Variants** (ViT only, 224×224):
  - `vit_tiny_patch16_224.augreg_in21k_ft_in1k`
  - `vit_tiny_patch16_224.augreg_in21k`
  - Default variant

**Quick Space**: `learning_rate`, `weight_decay`, `batch_size`, `dropout`

<h3>Model Search</h3>

Enable `enable_model_search` to let Optuna automatically explore all registered architectures for the target resolution alongside hyperparameters:

```yaml
optuna:
  n_trials: 20
  enable_model_search: true   # Explore architectures automatically
```

When enabled, the optimizer treats the model architecture as an additional categorical hyperparameter, selecting from all models compatible with the configured resolution. This is the recommended approach for finding the best architecture–hyperparameter combination without manual experimentation.

<h3>Optimization Workflow</h3>

```bash
# Phase 1: Comprehensive search (configurable trials, early stopping enabled)
orchard run recipes/optuna_efficientnet_b0.yaml

# Phase 2: Review results
firefox outputs/*/figures/param_importances.html
firefox outputs/*/figures/optimization_history.html

# Phase 3: Train with best config (60 epochs, full evaluation)
orchard run outputs/*/reports/best_config.yaml
```

<h3>Artifacts Generated</h3>

See the **[Artifact Reference Guide](ARTIFACTS.md)** for complete documentation of all generated files.

<h3>Customization</h3>

<h4>Search Space Overrides (YAML-based)</h4>

Override any parameter from the [search space](#search-space-coverage) directly in your recipe YAML, without code changes:

```yaml
optuna:
  search_space_preset: full
  search_space_overrides:
    learning_rate:
      low: 1.0e-04           # Narrower range for stable convergence
      high: 5.0e-03
      log: true
    dropout:
      low: 0.15
      high: 0.4
    batch_size_low_res:       # Categorical: provide a list
      - 32
      - 48
      - 64
```

**Float parameters** require `low`, `high`, and optionally `log: true` (default `false`) for log-scale sampling. **Categorical parameters** are plain lists.

All parameters listed in [Search Space Coverage](#search-space-coverage) can be overridden.

---


