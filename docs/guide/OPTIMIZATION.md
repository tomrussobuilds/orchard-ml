← [Back to Main README](../../README.md)

# Hyperparameter Optimization Guide

## 🎯 Hyperparameter Optimization

### Quick Start

```bash
# Install Optuna (if not already present)
pip install optuna plotly timm  # timm required for ViT support

# Run optimization with presets
python forge.py --config recipes/optuna_resnet_18.yaml  # 50 trials, ~3 min GPU, ~2.5h CPU
python forge.py --config recipes/optuna_mini_cnn.yaml           # 50 trials, ~1-2 min GPU, ~5 min CPU

# 224×224 resolution (includes weight variant search for ViT)
python forge.py --config recipes/optuna_efficientnet_b0.yaml    # 20 trials, ~1.5-5h GPU
python forge.py --config recipes/optuna_vit_tiny.yaml           # 20 trials, ~3-5h GPU

# Custom search (20 trials, 10 epochs each)
python forge.py --dataset pathmnist \
    --n_trials 20 \
    --epochs 10 \
    --search_space_preset quick

# Resume interrupted study
python forge.py --config recipes/optuna_vit_tiny.yaml \
    --load_if_exists true
```

### Search Space Coverage

**Full Space** (13+ parameters):
- **Optimization**: `learning_rate`, `weight_decay`, `momentum`, `min_lr`
- **Regularization**: `mixup_alpha`, `label_smoothing`, `dropout`
- **Scheduling**: `cosine_fraction`, `scheduler_patience`
- **Augmentation**: `rotation_angle`, `jitter_val`, `min_scale`
- **Batch Size**: Resolution-aware categorical choices
  - 28×28: [16, 32, 48, 64]
  - 224×224: [8, 12, 16] (OOM-safe for 8GB VRAM)
- **Architecture** (resolution-specific, requires `enable_model_search: true`):
  - 28×28: [`resnet_18`, `mini_cnn`]
  - 224×224: [`resnet_18`, `efficientnet_b0`, `convnext_tiny`, `vit_tiny`]
- **Weight Variants** (ViT only, 224×224):
  - `vit_tiny_patch16_224.augreg_in21k_ft_in1k`
  - `vit_tiny_patch16_224.augreg_in21k`
  - Default variant

**Quick Space** (4 parameters):
- `learning_rate`, `weight_decay`, `batch_size`, `dropout`

### Model Search

Enable `enable_model_search` to let Optuna automatically explore all registered architectures for the target resolution alongside hyperparameters:

```yaml
optuna:
  n_trials: 20
  enable_model_search: true   # Explore architectures automatically
```

When enabled, the optimizer treats the model architecture as an additional categorical hyperparameter, selecting from all models compatible with the configured resolution. This is the recommended approach for finding the best architecture–hyperparameter combination without manual experimentation.

### Optimization Workflow

```bash
# Phase 1: Comprehensive search (configurable trials, early stopping enabled)
python forge.py --config recipes/optuna_efficientnet_b0.yaml

# Phase 2: Review results
firefox outputs/*/figures/param_importances.html
firefox outputs/*/figures/optimization_history.html

# Phase 3: Train with best config (60 epochs, full evaluation)
python forge.py --config outputs/*/reports/best_config.yaml
```

### Artifacts Generated

See the **[Artifact Reference Guide](ARTIFACTS.md)** for complete documentation of all generated files.

### Customization

Edit search spaces in `orchard/optimization/search_spaces.py`:

```python
class CustomSearchSpace:
    @staticmethod
    def get_optimization_space() -> Dict[str, Callable]:
        return {
            "learning_rate": lambda trial: trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True
            ),
            "weight_decay": lambda trial: trial.suggest_float(
                "weight_decay", 1e-5, 1e-3, log=True
            ),
        }
```

---
