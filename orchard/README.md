← [Back to Main README](../README.md)

<h1 align="center">Orchard Package</h1>

**Orchard ML core package** - Type-safe deep learning framework components.

<h2>Package Structure</h2>

```
orchard/
├── cli_app.py                              # Typer CLI (orchard init / run)
├── exceptions.py                           # Framework-wide custom exceptions
├── core/                                   # Framework nucleus
│   ├── config/                             # Pydantic V2 schemas
│   │   ├── manifest.py                     # Main Config (SSOT)
│   │   ├── hardware_config.py              # Device, threading, determinism
│   │   ├── training_config.py              # Optimizer, scheduler, regularization
│   │   ├── dataset_config.py               # Data loading, resolution, normalization
│   │   ├── augmentation_config.py          # MixUp, TTA, transforms
│   │   ├── evaluation_config.py            # Metrics, visualization
│   │   ├── architecture_config.py          # Architecture selection
│   │   ├── optuna_config.py                # Hyperparameter optimization
│   │   ├── tracking_config.py              # MLflow tracking settings
│   │   ├── telemetry_config.py             # Filesystem, logging, experiment ID
│   │   ├── export_config.py                # ONNX export parameters
│   │   ├── infrastructure_config.py        # Resource lifecycle, flock, process mgmt
│   │   └── types.py                        # Semantic types & validation primitives
│   ├── environment/                        # Hardware abstraction
│   │   ├── hardware.py                     # Device detection, CPU/GPU/MPS
│   │   ├── reproducibility.py              # Seeding, determinism
│   │   ├── distributed.py                  # Rank detection, DDP guards
│   │   ├── timing.py                       # Execution timing utilities
│   │   ├── policy.py                       # TTA mode selection
│   │   └── guards.py                       # Process management, flock
│   ├── io/                                 # Serialization utilities
│   │   ├── checkpoints.py                  # Model weight loading
│   │   ├── serialization.py                # YAML config I/O, requirements dump
│   │   └── data_io.py                      # Dataset validation
│   ├── logger/                             # Telemetry system
│   │   ├── logger.py                       # Logger setup
│   │   ├── env_reporter.py                 # Environment reporting
│   │   └── progress.py                     # Progress tracking utilities
│   ├── metadata/                           # Dataset registry
│   │   ├── base.py                         # DatasetMetadata schema
│   │   ├── wrapper.py                      # Registry wrappers + get_registry() factory
│   │   └── domains/                        # Domain-specific registries
│   │       ├── classification/             # Classification domains
│   │       │   ├── medical.py              # MedMNIST v2 (YAML-driven, 4 resolutions)
│   │       │   ├── medical.yaml            # MedMNIST manifest (md5, urls, stats)
│   │       │   ├── space.py                # Galaxy10 DECals (224px)
│   │       │   └── benchmark.py            # CIFAR-10/100 (32px)
│   │       └── detection/                  # Detection domains
│   │           └── pennfudan.py            # PennFudan pedestrians (224px)
│   ├── paths/                              # Path management
│   │   ├── constants.py                    # Metric keys, log styles, static values
│   │   ├── root.py                         # Project root discovery, derived paths
│   │   └── run_paths.py                    # Dynamic workspace paths
│   ├── task_protocols.py                   # Task abstraction protocols
│   ├── task_registry.py                    # Task component registry
│   └── orchestrator.py                     # RootOrchestrator (7-phase lifecycle)
├── data_handler/                           # Data loading pipeline
│   ├── dispatcher.py                       # Fetch dispatcher + loading interface
│   ├── fetchers/                           # Domain-specific download modules
│   │   ├── medmnist_fetcher.py             # MedMNIST NPZ download + retries + MD5
│   │   ├── galaxy10_converter.py           # Galaxy10 HDF5 → NPZ conversion
│   │   ├── cifar_converter.py              # CIFAR-10/100 torchvision → NPZ
│   │   └── pennfudan_fetcher.py            # PennFudan ZIP → bbox NPZ conversion
│   ├── dataset.py                          # VisionDataset (eager/lazy loading)
│   ├── detection_dataset.py                # DetectionDataset (bbox annotations)
│   ├── collate.py                          # Detection collate (list-based batches)
│   ├── loader.py                           # DataLoaderFactory
│   ├── transforms.py                       # Augmentation pipelines (torchvision V2)
│   ├── data_explorer.py                    # Visualization utilities
│   └── diagnostic/                         # Health check & smoke test utilities
│       ├── synthetic.py                    # Synthetic classification data
│       ├── synthetic_detection.py          # Synthetic detection data
│       └── temp_loader.py                  # Lightweight DataLoader for diagnostics
├── architectures/                          # Architecture factory
│   ├── factory.py                          # Model registry & dispatch
│   ├── resnet_18.py                        # ResNet-18 (28/32/64/128/224)
│   ├── mini_cnn.py                         # Compact CNN (~95K params, 28/32/64)
│   ├── efficientnet_b0.py                  # EfficientNet-B0 (224px)
│   ├── convnext_tiny.py                    # ConvNeXt-Tiny (224px)
│   ├── vit_tiny.py                         # ViT-Tiny (224px, multiple weight variants)
│   ├── fasterrcnn.py                       # Faster R-CNN ResNet-50-FPN v2 (detection)
│   ├── timm_backbone.py                    # timm/ pass-through (1000+ models)
│   └── _morphing.py                        # Pretrained weight adaptation
├── trainer/                                # Training loop
│   ├── engine.py                           # Core train/validation logic + mixup
│   ├── trainer.py                          # ModelTrainer orchestrator
│   ├── _loop.py                            # TrainingLoop kernel + AMP/MixUp factories
│   ├── _scheduling.py                      # Scheduler stepping utility
│   ├── losses.py                           # FocalLoss implementation
│   └── setup.py                            # Optimizer/scheduler/criterion factories
├── evaluation/                             # Metrics and visualization
│   ├── evaluator.py                        # Evaluation orchestration
│   ├── evaluation_pipeline.py              # Classification evaluation pipeline
│   ├── metrics.py                          # AUC, F1, Accuracy, Macro-F1
│   ├── plot_context.py                     # PlotContext DTO for matplotlib
│   ├── tta.py                              # Test-time augmentation (adaptive)
│   ├── visualization.py                    # Confusion matrix, training curves
│   └── reporting.py                        # Excel report generation
├── pipeline/                               # Pipeline phase orchestration
│   └── phases.py                           # Training, optimization, export phases
├── export/                                 # Model export for production
│   ├── onnx_exporter.py                    # ONNX export with quantization
│   └── validation.py                       # PyTorch vs ONNX validation
├── tracking/                               # Experiment tracking
│   └── tracker.py                          # MLflow integration (local SQLite)
├── tasks/                                  # Task-specific adapters
│   ├── classification/                     # Classification task
│   │   ├── criterion_adapter.py            # CrossEntropy / Focal loss
│   │   ├── metrics_adapter.py              # Accuracy, F1, AUC
│   │   ├── evaluation_adapter.py           # Full evaluation pipeline
│   │   └── training_step_adapter.py        # Classification forward pass
│   └── detection/                          # Detection task
│       ├── criterion_adapter.py            # No-op (built-in model loss)
│       ├── metrics_adapter.py              # mAP via torchmetrics
│       ├── evaluation_adapter.py           # mAP evaluation + training curves
│       ├── training_step_adapter.py        # Detection forward pass (list I/O)
│       └── helpers.py                      # Shared utilities (to_cpu)
└── optimization/                           # Optuna integration
    ├── _param_mapping.py                   # PARAM_MAPPING / SPECIAL_PARAMS
    ├── objective/                          # Trial execution logic
    │   ├── objective.py                    # OptunaObjective
    │   ├── config_builder.py               # Trial config override
    │   ├── training_executor.py            # Trial training
    │   └── metric_extractor.py             # Metric extraction
    ├── orchestrator/                       # Study management
    │   ├── orchestrator.py                 # OptunaOrchestrator
    │   ├── registries.py                   # Sampler/pruner registries
    │   ├── builders.py                     # Sampler/pruner builders
    │   ├── exporters.py                    # Results export (YAML, Excel)
    │   ├── utils.py                        # Utility helpers
    │   └── visualizers.py                  # Plotly visualizations
    ├── search_spaces.py                    # Hyperparameter distributions
    └── early_stopping.py                   # Convergence detection
```

<h2>Architecture Principles</h2>

<h3>1. Dependency Injection</h3>

All modules receive narrowed sub-configs as dependencies - no global state:
```python
model = get_model(device=device, dataset_cfg=cfg.dataset, arch_cfg=cfg.architecture)
loaders = get_dataloaders(data, cfg.dataset, cfg.training, cfg.augmentation, cfg.num_workers)
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
- **tasks/**: Task-specific adapters (strategy pattern via registry)
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

Create a new domain file (e.g., `orchard/core/metadata/domains/custom.py`):
```python
REGISTRY_224: Final[dict[str, DatasetMetadata]] = {
    "custom_dataset": DatasetMetadata(
        name="custom_dataset",
        display_name="Custom Dataset",
        md5_checksum="abc123...",
        url="https://example.com/dataset.npz",
        path=DATASET_DIR / "custom_dataset_224.npz",
        classes=["class_a", "class_b", "class_c"],
        in_channels=3,
        native_resolution=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        is_anatomical=False,
        is_texture_based=True,
    ),
}
```
Export from `orchard/core/metadata/domains/__init__.py` to make it available.

<h3>Adding New Architectures</h3>

1. Create builder in `orchard/architectures/your_model.py`:
```python
def build_your_model(
    num_classes: int,
    in_channels: int,
    *,
    pretrained: bool,
) -> nn.Module:
    model = ...  # Build your model
    return model  # Device placement handled by get_model()
```

2. Register in `orchard/architectures/factory.py` inside the `MappingProxyType` dict:
```python
_MODEL_REGISTRY: MappingProxyType[str, _BuilderFn] = MappingProxyType(
    {
        ...
        "your_model": build_your_model,
    }
)
```

<h3>Adding New Optimizers</h3>

Extend `orchard/trainer/setup.py`:
```python
def get_optimizer(model: nn.Module, training: TrainingConfig) -> optim.Optimizer:
    if training.optimizer_type == "adamw":
        return torch.optim.AdamW(...)
    # Add new case

def get_scheduler(optimizer: optim.Optimizer, training: TrainingConfig) -> LRScheduler:
    if training.scheduler_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(...)
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
