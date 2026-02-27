← [Back to Main README](../README.md)

<h1 align="center">Orchard Package</h1>

**Orchard ML core package** - Type-safe deep learning framework components.

<h2>Package Structure</h2>

```
orchard/
├── cli_app.py                  # Typer CLI entry point (orchard run / orchard init)
├── core/                       # Framework nucleus
│   ├── config/                 # Pydantic V2 schemas (14 modules)
│   │   ├── manifest.py         # Main Config (SSOT)
│   │   ├── hardware_config.py  # Device, threading, determinism
│   │   ├── training_config.py  # Optimizer, scheduler, regularization
│   │   ├── dataset_config.py   # Data loading, resolution, normalization
│   │   ├── augmentation_config.py  # MixUp, TTA, transforms
│   │   ├── evaluation_config.py    # Metrics, visualization
│   │   ├── architecture_config.py  # Architecture selection
│   │   ├── optuna_config.py    # Hyperparameter optimization
│   │   ├── tracking_config.py  # MLflow tracking settings
│   │   ├── telemetry_config.py # Filesystem, logging policy, experiment ID
│   │   ├── export_config.py    # ONNX export parameters
│   │   ├── infrastructure_config.py # Resource lifecycle, flock, process mgmt
│   │   └── types.py            # Semantic types & validation primitives
│   ├── environment/            # Hardware abstraction
│   │   ├── hardware.py         # Device detection, CPU/GPU/MPS
│   │   ├── reproducibility.py  # Seeding, determinism
│   │   ├── distributed.py      # Rank detection, DDP guards
│   │   ├── timing.py           # Execution timing utilities
│   │   ├── policy.py           # TTA mode selection
│   │   └── guards.py           # Process management, flock
│   ├── io/                     # Serialization utilities
│   │   ├── checkpoints.py      # Model weight loading
│   │   ├── serialization.py    # YAML config I/O, requirements dump
│   │   └── data_io.py          # Dataset validation
│   ├── logger/                 # Telemetry system
│   │   ├── logger.py           # Logger setup
│   │   ├── reporter.py         # Environment reporting
│   │   ├── styles.py           # Log formatting & styling
│   │   └── progress.py         # Progress tracking utilities
│   ├── metadata/               # Dataset registry
│   │   ├── base.py             # DatasetMetadata schema
│   │   ├── domains/            # Domain-specific registries
│   │   │   ├── medical.py      # Medical imaging (MedMNIST)
│   │   │   ├── space.py        # Astronomical imaging
│   │   │   └── benchmark.py    # Standard benchmarks (CIFAR-10/100)
│   │   └── wrapper.py          # Multi-resolution registry wrapper
│   ├── paths/                  # Path management
│   │   ├── constants.py        # Static paths (PROJECT_ROOT, etc.)
│   │   └── run_paths.py        # Dynamic workspace paths
│   └── orchestrator.py         # RootOrchestrator (7-phase lifecycle)
├── data_handler/               # Data loading pipeline
│   ├── fetcher.py              # Fetch dispatcher + loading interface
│   ├── fetchers/               # Domain-specific download modules
│   │   ├── medmnist_fetcher.py # MedMNIST NPZ download with retries & MD5
│   │   ├── galaxy10_converter.py # Galaxy10 HDF5 download & NPZ conversion
│   │   └── cifar_converter.py  # CIFAR-10/100 torchvision → NPZ conversion
│   ├── dataset.py              # VisionDataset (eager/lazy loading)
│   ├── loader.py               # DataLoaderFactory
│   ├── transforms.py           # Augmentation pipelines (torchvision V2)
│   ├── data_explorer.py        # Visualization utilities
│   └── synthetic.py            # Synthetic data generation
├── architectures/              # Architecture factory
│   ├── factory.py              # Model registry & builder
│   ├── resnet_18.py            # ResNet-18 multi-resolution (28/32/64/128/224)
│   ├── mini_cnn.py             # Compact CNN (~95K params, 28/32/64)
│   ├── efficientnet_b0.py      # EfficientNet for 224×224
│   ├── convnext_tiny.py        # ConvNeXt-Tiny for 224×224
│   ├── vit_tiny.py             # Vision Transformer for 224×224
│   ├── timm_backbone.py        # Timm pass-through support
│   └── _morphing.py            # Pretrained weight adaptation (channel mismatch)
├── trainer/                    # Training loop
│   ├── engine.py               # Core train/validation logic + mixup
│   ├── trainer.py              # ModelTrainer orchestrator
│   ├── _loop.py                # Shared TrainingLoop kernel + AMP/MixUp factories
│   ├── _scheduling.py          # Scheduler stepping utility
│   ├── losses.py               # FocalLoss implementation
│   └── setup.py                # Optimizer/scheduler/criterion factories
├── evaluation/                 # Metrics and visualization
│   ├── evaluator.py            # Evaluation orchestration
│   ├── evaluation_pipeline.py  # Full evaluation pipeline
│   ├── metrics.py              # AUC, F1, Accuracy, Macro-F1
│   ├── tta.py                  # Test-time augmentation (adaptive)
│   ├── visualization.py        # Confusion matrix, curves
│   └── reporting.py            # Excel report generation
├── pipeline/                   # Pipeline phase orchestration
│   └── phases.py               # Training, optimization, export phases
├── export/                     # Model export for production
│   ├── onnx_exporter.py        # ONNX export with quantization
│   └── validation.py           # PyTorch vs ONNX validation
├── tracking/                   # Experiment tracking
│   └── tracker.py              # MLflow integration (optional, local SQLite)
└── optimization/               # Optuna integration
    ├── objective/              # Trial execution logic
    │   ├── objective.py        # OptunaObjective
    │   ├── config_builder.py   # Trial config override
    │   ├── training_executor.py    # Trial training
    │   └── metric_extractor.py # Metric extraction
    ├── orchestrator/           # Study management
    │   ├── orchestrator.py     # OptunaOrchestrator
    │   ├── config.py           # Study configuration
    │   ├── builders.py         # Sampler/pruner builders
    │   ├── exporters.py        # Results export (YAML, Excel)
    │   ├── utils.py            # Utility helpers
    │   └── visualizers.py      # Plotly visualizations
    ├── search_spaces.py        # Hyperparameter distributions
    └── early_stopping.py       # Convergence detection
```

<h2>Architecture Principles</h2>

<h3>1. Dependency Injection</h3>

All modules receive `Config` as dependency - no global state:
```python
model = get_model(device=device, cfg=cfg)
loaders = get_dataloaders(data, cfg)
trainer = ModelTrainer(model=model, cfg=cfg, ...)
```

<h3>2. Single Source of Truth (SSOT)</h3>

`Config` is the immutable configuration manifest validated by Pydantic V2:
- Cross-domain validation (AMP ↔ device, pretrained ↔ RGB)
- Late-binding metadata injection (dataset specs from registry)
- Path portability (relative anchoring from PROJECT_ROOT)

<h3>3. Separation of Concerns</h3>

- **core/**: Framework infrastructure (config, hardware, logging)
- **data_handler/**: Data loading only
- **architectures/**: Architecture definitions only
- **trainer/**: Training loop only
- **evaluation/**: Metrics & visualization only
- **pipeline/**: Phase orchestration (training, optimization, export)
- **export/**: ONNX export and validation
- **tracking/**: MLflow experiment tracking (optional)
- **optimization/**: Optuna wrapper only

<h3>4. Protocol-Based Design</h3>

Use protocols for testability:
```python
class InfraManagerProtocol(Protocol):
    def prepare_environment(self, cfg, logger) -> None: ...
    def release_resources(self, cfg, logger) -> None: ...
```

<h2>Key Extension Points</h2>

<h3>Adding New Datasets</h3>

Register in the appropriate domain file (e.g., `orchard/core/metadata/domains/medical.py`):
```python
REGISTRY_224: Final[Dict[str, DatasetMetadata]] = {
    "custom_dataset": DatasetMetadata(
        name="custom_dataset",
        num_classes=10,
        in_channels=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        native_resolution=224,
        is_anatomical=False,
        is_texture_based=True,
    ),
}
```
Export from `orchard/core/metadata/domains/__init__.py` to make it available.

<h3>Adding New Architectures</h3>

1. Create builder in `orchard/architectures/your_model.py`:
```python
def build_your_model(device, cfg, in_channels, num_classes):
    # Implementation
    return model
```

2. Register in `orchard/architectures/factory.py`:
```python
_MODEL_REGISTRY["your_model"] = build_your_model
```

<h3>Adding New Optimizers</h3>

Extend `orchard/trainer/setup.py`:
```python
def get_optimizer(model, cfg):
    if cfg.training.optimizer_type == "adam":
        return torch.optim.Adam(...)
    # Add new case
```

<h2>Further Reading</h2>

- **[Framework Guide](../docs/guide/FRAMEWORK.md)** - System design, technical deep dive
- **[Architecture Guide](../docs/guide/ARCHITECTURE.md)** - Supported models and weight transfer
- **[Configuration Guide](../docs/guide/CONFIGURATION.md)** - All config parameters
- **[Optimization Guide](../docs/guide/OPTIMIZATION.md)** - Optuna integration, search spaces, pruning
- **[Export Guide](../docs/guide/EXPORT.md)** - ONNX export, quantization, validation
- **[Tracking Guide](../docs/guide/TRACKING.md)** - MLflow local setup, run comparison
- **[Docker Guide](../docs/guide/DOCKER.md)** - Container build, GPU-accelerated execution
- **[Artifact Guide](../docs/guide/ARTIFACTS.md)** - Output directory structure, artifact differences
- **[Testing Guide](../docs/guide/TESTING.md)** - Test suite organization
