← [Back to Main README](../../README.md)

# Hyperparameter Optimization Guide

## 🎯 Hyperparameter Optimization

### Quick Start

```bash
# Install Optuna (if not already present)
pip install optuna plotly timm  # timm required for ViT support

# Run optimization with presets
python optimize.py --config recipes/optuna_resnet_18_adapted.yaml  # 50 trials, ~3 min GPU, ~2.5h CPU
python optimize.py --config recipes/optuna_mini_cnn.yaml           # 50 trials, ~1-2 min GPU, ~5 min CPU

# 224×224 resolution (includes weight variant search for ViT)
python optimize.py --config recipes/optuna_efficientnet_b0.yaml    # 20 trials, ~1.5-5h GPU
python optimize.py --config recipes/optuna_vit_tiny.yaml           # 20 trials, ~3-5h GPU

# Custom search (20 trials, 10 epochs each)
python optimize.py --dataset pathmnist \
    --n_trials 20 \
    --epochs 10 \
    --search_space_preset quick

# Resume interrupted study
python optimize.py --config recipes/optuna_vit_tiny.yaml \
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
- **Architecture** (resolution-specific):
  - 28×28: [`resnet_18_adapted`, `mini_cnn`]
  - 224×224: [`efficientnet_b0`, `vit_tiny`]
- **Weight Variants** (ViT only, 224×224):
  - `vit_tiny_patch16_224.augreg_in21k_ft_in1k`
  - `vit_tiny_patch16_224.augreg_in21k`
  - Default variant

**Quick Space** (4 parameters):
- `learning_rate`, `weight_decay`, `batch_size`, `dropout`

### Optimization Workflow

```bash
# Phase 1: Comprehensive search (configurable trials, early stopping enabled)
python optimize.py --config recipes/optuna_efficientnet_b0.yaml

# Phase 2: Review results
firefox outputs/*/figures/param_importances.html
firefox outputs/*/figures/optimization_history.html

# Phase 3: Train with best config (60 epochs, full evaluation)
python main.py --config outputs/*/reports/best_config.yaml
```

### Artifacts Generated

### Optimization Run Artifacts
```
outputs/20260123_organcmnist_efficientnetb0_a3f7c2/
├── figures/
│   ├── param_importances.html      # Interactive importance plot
│   ├── optimization_history.html   # Trial progression
│   ├── slice.html                  # 1D parameter effects
│   └── parallel_coordinate.html    # Multi-dimensional view
├── reports/
│   ├── best_config.yaml            # Optimized configuration
│   ├── study_summary.json          # All trials metadata
│   └── top_10_trials.xlsx          # Best configurations
├── database/
│   └── study.db                    # SQLite storage for resumption
└── logs/
    └── orchestrator_YYYYMMDD_HHMMSS.log
```

### Standard Training Run Artifacts
```
outputs/20260123_bloodmnist_minicnn_b4a8f1/
├── figures/
│   ├── confusion_matrix_mini_cnn.png       # Classification matrix
│   ├── sample_predictions_mini_cnn.png     # Model predictions visualization
│   ├── training_curves_mini_cnn.png        # Loss/accuracy plots
│   ├── training_curves_mini_cnn.npz        # Raw metrics data
│   └── bloodmnist/                         # Dataset-specific visualizations
│       └── sample_grid_28x28.png           # 12 samples with augmentations
├── reports/
│   ├── training_summary.xlsx               # Complete metrics spreadsheet
│   └── config_snapshot_YYYYMMDD_HHMMSS.yaml # Configuration used for run
├── models/
│   └── best_mini_cnn.pth                   # Best model weights (by val AUC)
└── logs/
    └── orchestrator_YYYYMMDD_HHMMSS.log
```

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
