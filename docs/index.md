# Orchard ML

**Type-safe deep learning framework for reproducible computer vision research.**

Orchard ML provides a complete pipeline from data loading to production deployment,
with Pydantic v2 validated configuration, Optuna hyperparameter optimization, and
ONNX export with quantization.

## Key Features

- **Type-safe configuration** -- Pydantic v2 frozen models with cross-domain validation
- **6 built-in architectures** -- MiniCNN, ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-Tiny, plus 1000+ via timm
- **14 datasets** -- MedMNIST, CIFAR-10/100, Galaxy10
- **Optuna integration** -- Hyperparameter search with pruning and model search
- **ONNX export** -- Production-ready export with INT8 quantization and benchmarking
- **MLflow tracking** -- Local experiment tracking with SQLite backend
- **Full reproducibility** -- Deterministic seeding, config snapshots, artifact management

## Quick Start

```bash
pip install orchard-ml
orchard run recipes/config_mini_cnn.yaml
```

## Documentation

- [User Guide](guide/FRAMEWORK.md) -- Framework overview, configuration, and workflows
- [API Reference](reference/) -- Auto-generated from source code
