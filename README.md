<h1 align="center">Orchard ML</h1>
<p align="center"><strong>Type-safe deep learning framework for reproducible computer vision research</strong></p>

---

<!-- Badges Section -->
<table align="center">
<tr>
<td align="right"><strong>CI & Quality</strong></td>
<td>
  <a href="https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml"><img src="https://github.com/tomrussobuilds/orchard-ml/actions/workflows/ci.yml/badge.svg" alt="CI/CD Pipeline"></a>
  <a href="https://codecov.io/gh/tomrussobuilds/orchard-ml"><img src="https://codecov.io/gh/tomrussobuilds/orchard-ml/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=tomrussobuilds_orchard-ml"><img src="https://sonarcloud.io/api/project_badges/measure?project=tomrussobuilds_orchard-ml&metric=alert_status" alt="Quality Gate"></a>
</td>
</tr>
<tr>
<td align="right"><strong>Platform</strong></td>
<td>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14--dev-blue?logo=python&logoColor=white" alt="Python">
  <a href="https://pypi.org/project/orchard-ml/"><img src="https://img.shields.io/pypi/v/orchard-ml?color=blue&logo=pypi&logoColor=white&v=1" alt="PyPI"></a>
  <a href="docs/guide/DOCKER.md"><img src="https://img.shields.io/badge/Docker-CUDA%2012.1-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</td>
</tr>
<tr>
<td align="right"><strong>Stack</strong></td>
<td>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/docs/timm"><img src="https://img.shields.io/badge/timm-models-FF9D00?logo=huggingface&logoColor=white" alt="timm"></a>
  <a href="https://docs.pydantic.dev/"><img src="https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white" alt="Pydantic"></a>
  <a href="https://optuna.org/"><img src="https://img.shields.io/badge/Optuna-3.0%2B-00ADD8?logo=optuna&logoColor=white" alt="Optuna"></a>
  <a href="https://onnx.ai/"><img src="https://img.shields.io/badge/ONNX-export-005CED?logo=onnx&logoColor=white" alt="ONNX"></a>
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white" alt="MLflow"></a>
</td>
</tr>
<tr>
<td align="right"><strong>Code Style</strong></td>
<td>
  <!-- Dynamic badges — updated by .github/workflows/badges.yml via Gist -->
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tomrussobuilds/7835190af6011e9051b673c8be974f8a/raw/black.json" alt="Black"></a>
  <a href="https://pycqa.github.io/isort/"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tomrussobuilds/7835190af6011e9051b673c8be974f8a/raw/isort.json" alt="isort"></a>
  <a href="https://flake8.pycqa.org/"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tomrussobuilds/7835190af6011e9051b673c8be974f8a/raw/flake8.json" alt="Flake8"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tomrussobuilds/7835190af6011e9051b673c8be974f8a/raw/mypy.json" alt="mypy"></a>
  <a href="https://bandit.readthedocs.io/"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tomrussobuilds/7835190af6011e9051b673c8be974f8a/raw/bandit.json" alt="Bandit"></a>
  <a href="https://radon.readthedocs.io/"><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tomrussobuilds/7835190af6011e9051b673c8be974f8a/raw/radon.json" alt="Radon"></a>
</td>
</tr>
</table>

---

<h2>Table of Contents</h2>

- [**Overview**](#overview)
- [**Hardware Requirements**](#hardware-requirements)
- [**Quick Start**](#quick-start)
- [**Colab Notebooks**](#colab-notebooks)
- [**Experiment Management**](#experiment-management)
- [**Documentation**](#documentation)
- [**Citation**](#citation)
- [**Roadmap**](#roadmap)
- [**License**](#license)

---

<h2>Overview</h2>

**Orchard ML** is a research-grade `PyTorch` training framework engineered for reproducible, scalable computer vision experiments across diverse domains. Built on [MedMNIST v2](https://zenodo.org/records/6496656) medical imaging datasets and expanded to astronomical imaging ([Galaxy10 DECals](https://zenodo.org/records/10845026)) and standard benchmarks ([CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)), it provides a domain-agnostic platform supporting multi-resolution architectures (28×28, 32×32, 64×64, 224×224), automated hyperparameter optimization, and cluster-safe execution.

**Key Differentiators:**
- **Type-Safe Configuration Engine**: `Pydantic V2`-based declarative manifests eliminate runtime errors
- **Idempotent Lifecycle Orchestration**: `RootOrchestrator` coordinates a 7-phase initialization sequence (seeding, filesystem, logging, infrastructure locks, telemetry) via Context Manager with full dependency injection
- **Zero-Conflict Execution**: Kernel-level file locking (`fcntl`) prevents concurrent runs from corrupting shared resources
- **Intelligent Hyperparameter Search**: `Optuna` integration with TPE sampling and Median Pruning
- **Hardware-Agnostic**: Auto-detection and optimization for `CPU`/`CUDA`/`MPS` backends
- **Audit-Grade Traceability**: `BLAKE2b`-hashed run directories with full `YAML` snapshots

**Supported Architectures:**

| Resolution | Architectures | Parameters | Use Case |
|-----------|--------------|-----------|----------|
| **28 / 32 / 64 / 224** | `ResNet-18` | ~11M | Multi-resolution baseline, transfer learning |
| **28 / 32 / 64** | `MiniCNN` | ~95K | Fast prototyping, ablation studies |
| **224×224** | `EfficientNet-B0` | ~4.0M | Efficient compound scaling |
| **224×224** | `ConvNeXt-Tiny` | ~27.8M | Modern ConvNet design |
| **224×224** | `ViT-Tiny` | ~5.5M | Patch-based attention, multiple weight variants |

> [!TIP]
> **1000+ additional architectures via [timm](https://huggingface.co/docs/timm)**: Any model in the `timm` registry can be used by prefixing the name with `timm/` in your recipe:
> ```yaml
> architecture:
>   name: "timm/mobilenetv3_small_100"   # ~1.5M params, edge-friendly
>   pretrained: true
> ```
> This works with MobileNet, DenseNet, RegNet, EfficientNet-V2, and any other architecture supported by `timm`. See `recipes/config_timm_mobilenetv3.yaml` for a ready-to-use example.

---

<h2>Hardware Requirements</h2>

<h3>CPU Training (28×28 / 32×32 / 64×64)</h3>

- **Supported Resolutions**: 28×28, 32×32, 64×64
- **Time**: ~2.5 hours (`ResNet-18`, 28×28, 60 epochs, 16 cores)
- **Time**: ~5-10 minutes (`MiniCNN`, 28×28, 60 epochs, 16 cores)
- **Architectures**: `ResNet-18`, `MiniCNN`
- **Use Case**: Development, testing, limited hardware environments

<h3>GPU Training (All Resolutions)</h3>

- **28×28 Resolution**:
  - `MiniCNN`: ~2-3 minutes (60 epochs)
  - `ResNet-18`: ~10-15 minutes (60 epochs)
- **32×32 Resolution** (CIFAR-10/100):
  - `MiniCNN`: ~3-5 minutes (60 epochs)
  - `ResNet-18`: ~15-20 minutes (60 epochs)
- **64×64 Resolution**:
  - `MiniCNN`: ~3-5 minutes (60 epochs)
  - `ResNet-18`: ~15-20 minutes (60 epochs)
- **224×224 Resolution**:
  - `EfficientNet-B0`: ~30 minutes per trial (15 epochs)
  - `ViT-Tiny`: ~25-35 minutes per trial (15 epochs)
- **VRAM**: 8GB recommended for 224×224 resolution
- **Architectures**: `ResNet-18`, `EfficientNet-B0`, `ConvNeXt-Tiny`, `ViT-Tiny`

> [!WARNING]
> **224×224 training on CPU is not recommended** - it would take 10+ hours per trial. High-resolution training requires GPU acceleration. Only 28×28 resolution has been tested and validated for CPU training.

> [!NOTE]
> **Apple Silicon (`MPS`)**: The codebase includes `MPS` backend support (device detection, seeding, memory management), but it has not been tested on real hardware. If you encounter issues, please open an issue.

> [!NOTE]
> **Data Format**: Orchard ML operates on `NPZ` archives as its canonical data format. All datasets are downloaded or converted to `NPZ` before entering the training pipeline. Custom datasets in other formats (HDF5, DICOM, TIFF) can be integrated by adding a conversion step in a dedicated fetcher module — see the [Galaxy10 fetcher](orchard/data_handler/fetchers/) for a reference implementation.

**Representative Benchmarks** (RTX 5070 Laptop GPU):

| Task | Architecture | Resolution | Device | Time | Notes |
|------|-------------|-----------|--------|------|-------|
| **Smoke Test** | `MiniCNN` | 28×28 | CPU/GPU | <30s | 1-epoch sanity check |
| **Quick Training** | `MiniCNN` | 28×28 | GPU | ~2-3 min | 60 epochs |
| **Quick Training** | `MiniCNN` | 28×28 | CPU (16 cores) | ~5-10 min | 60 epochs, CPU-validated |
| **Mid-Res Training** | `MiniCNN` | 64×64 | GPU | ~3-5 min | 60 epochs |
| **Transfer Learning** | `ResNet-18` | 28×28 | GPU | ~5 min | 60 epochs |
| **Transfer Learning** | `ResNet-18` | 28×28 | CPU (16 cores) | ~2.5h | 60 epochs, CPU-validated |
| **High-Res Training** | `EfficientNet-B0` | 224×224 | GPU | ~30 min/trial | 15 epochs per trial, **GPU required** |
| **High-Res Training** | `ViT-Tiny` | 224×224 | GPU | ~25-35 min/trial | 15 epochs per trial, **GPU required** |
| **Optimization Study** | `EfficientNet-B0` | 224×224 | GPU | ~2h | 4 trials (early stop at AUC≥0.9999) |
| **Optimization Study** | Various | 224×224 | GPU | ~1.5-5h | 20 trials, highly variable |

> [!NOTE]
> **Timing Variance**: Optimization times are highly dependent on early stopping criteria, pruning configuration, and dataset complexity:
> - **Early Stopping**: Studies may finish in 1-3 hours if performance thresholds are met quickly (e.g., AUC ≥ 0.9999 after 4 trials)
> - **Full Exploration**: Without early stopping, 20 trials can extend to 5+ hours
> - **Pruning Impact**: Median pruning can save 30-50% of total time by terminating underperforming trials

---

<h2>Quick Start</h2>

<h3>Step 1: Environment Setup</h3>

**Option A**: Install from source (recommended)
```bash
git clone https://github.com/tomrussobuilds/orchard-ml.git
```

Navigate into the project directory and install in editable mode:
```bash
cd orchard-ml
pip install -e .
```

With development tools (linting, testing, type checking):
```bash
pip install -e ".[dev]"
```

**Option B**: Install from PyPI
```bash
pip install orchard-ml
orchard init            # generates recipe.yaml with all defaults
orchard run recipe.yaml
```

<h3>Step 2: Verify Installation (Optional)</h3>

```bash
# Run 1-epoch sanity check (~30 seconds, CPU/GPU)
# Downloads BloodMNIST 28×28 by default
python -m tests.smoke_test

# Note: You can skip this step - datasets are auto-downloaded on first run
```

<h3>Step 3: Training Workflow</h3>

Orchard ML uses the `orchard` CLI as the **single entry point** for all workflows. The pipeline behavior is controlled entirely by the `YAML` recipe:

- **Training only**: Use a `config_*.yaml` file (no `optuna:` section)
- **Optimization + Training**: Use an `optuna_*.yaml` file (has `optuna:` section)
- **With Export**: Add an `export:` section to your config

```bash
orchard --version                          # Verify installation
orchard run --help                         # Show available options
```

<h4>Training Only (Quick start)</h4>

```bash
# 28×28 resolution (CPU-compatible)
orchard run recipes/config_mini_cnn.yaml              # ~2-3 min GPU, ~5-10 min CPU
orchard run recipes/config_resnet_18.yaml             # ~10-15 min GPU, ~2.5h CPU

# 32×32 resolution (CIFAR-10/100)
orchard run recipes/config_cifar10_mini_cnn.yaml      # ~3-5 min GPU
orchard run recipes/config_cifar10_resnet_18.yaml     # ~10-15 min GPU

# 64×64 resolution (CPU/GPU)
orchard run recipes/config_mini_cnn_64.yaml           # ~3-5 min GPU

# 224×224 resolution (GPU required)
orchard run recipes/config_efficientnet_b0.yaml       # ~30 min GPU
orchard run recipes/config_vit_tiny.yaml              # ~25-35 min GPU

# Override any config value on the fly
orchard run recipes/config_mini_cnn.yaml --set training.epochs=20 --set training.seed=99
```

**What happens:**
- Dataset auto-downloaded to `./dataset/`
- Training runs for 60 epochs with early stopping
- Results saved to timestamped directory in `outputs/`

---

<h4>Hyperparameter Optimization + Training (Full pipeline)</h4>

```bash
# 28×28 resolution - fast iteration
orchard run recipes/optuna_mini_cnn.yaml              # ~5 min GPU, ~5-10 min CPU
orchard run recipes/optuna_resnet_18.yaml             # ~15 min GPU

# 32×32 resolution - CIFAR-10/100
orchard run recipes/optuna_cifar100_mini_cnn.yaml     # ~1-2h GPU
orchard run recipes/optuna_cifar100_resnet_18.yaml    # ~3-4h GPU

# 224×224 resolution - requires GPU
orchard run recipes/optuna_efficientnet_b0.yaml       # ~1.5-5h*, GPU
orchard run recipes/optuna_vit_tiny.yaml              # ~3-5h*, GPU

# *Time varies due to early stopping (may finish in 1-3h if target AUC reached)
```

**What happens:**
1. **Optimization**: Explores hyperparameter combinations with `Optuna`
2. **Training**: Full 60-epoch training with best hyperparameters found
3. **Artifacts**: Interactive plots, best_config.yaml, model weights

> [!TIP]
> **Model Search**: Enable `optuna.enable_model_search: true` in your `YAML` config to let `Optuna` automatically explore all registered architectures for the target resolution. Use `optuna.model_pool` to restrict the search to a subset of architectures (e.g. `["vit_tiny", "efficientnet_b0"]`).

**View optimization results:**
```bash
firefox outputs/*/figures/param_importances.html       # Which hyperparameters matter most
firefox outputs/*/figures/optimization_history.html    # Trial progression
```

---

<h4>Model Export (Production deployment)</h4>

All training configs (`config_*.yaml`) include `ONNX` export by default:
```bash
orchard run recipes/config_efficientnet_b0.yaml
# → Training + ONNX export to outputs/*/exports/model.onnx
```

See the [Export Guide](docs/guide/EXPORT.md) for configuration options (format, quantization, validation).

---

<h2>Colab Notebooks</h2>

Try Orchard ML directly in Google Colab — no local setup required:

| Notebook | Description | Runtime | Time |
|----------|-------------|---------|------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomrussobuilds/orchard-ml/blob/main/notebooks/01_quickstart_bloodmnist_cpu.ipynb) **[Quick Start: BloodMNIST CPU](notebooks/01_quickstart_bloodmnist_cpu.ipynb)** | `MiniCNN` training on `BloodMNIST` 28×28 — end-to-end training, evaluation, and `ONNX` export | CPU | ~15 min |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tomrussobuilds/orchard-ml/blob/main/notebooks/02_galaxy10_optuna_model_search.ipynb) **[Optuna Model Search: Galaxy10 GPU](notebooks/02_galaxy10_optuna_model_search.ipynb)** | Automatic architecture search (`EfficientNet-B0`, `ViT-Tiny`, `ConvNeXt-Tiny`, `ResNet-18`) on Galaxy10 224×224 with `Optuna` | T4 GPU | ~30-45 min |

---

<h2>Experiment Management</h2>

Every run generates a complete artifact suite for total traceability. Both training-only and optimization workflows share the same `RunPath` orchestrator, producing `BLAKE2b`-hashed timestamped directories.

**[Browse Sample Artifacts](./docs/artifacts)** — Excel reports, `YAML` configs, and diagnostic plots from real training runs.
See the [full artifact tree](docs/artifacts/artifacts_structure.png) for the complete directory layout — logs, model weights, and HTML plots are generated locally and not tracked in the repo.

**[Browse Recipe Configs](./recipes)** — Ready-to-use `YAML` configurations for every architecture and workflow.
Copy the closest recipe, tweak the parameters, and run:
```bash
cp recipes/config_efficientnet_b0.yaml my_run.yaml
# edit hyperparameters, swap dataset/model, add or remove sections (optuna, export, tracking)
orchard run my_run.yaml
```

---

<h2>Documentation</h2>

| Guide | Covers |
|-------|--------|
| [Framework Guide](docs/guide/FRAMEWORK.md) | System architecture diagrams, design principles, component deep-dives |
| [Architecture Guide](docs/guide/ARCHITECTURE.md) | Supported model architectures, weight transfer, grayscale adaptation, `MixUp` |
| [Configuration Guide](docs/guide/CONFIGURATION.md) | Full parameter reference, usage patterns, adding new datasets |
| [Optimization Guide](docs/guide/OPTIMIZATION.md) | `Optuna` integration, search space config, pruning strategies, visualization |
| [Docker Guide](docs/guide/DOCKER.md) | Container build instructions, GPU-accelerated execution, reproducibility mode |
| [Export Guide](docs/guide/EXPORT.md) | `ONNX` export pipeline, quantization options, validation and benchmarking |
| [Tracking Guide](docs/guide/TRACKING.md) | `MLflow` local setup, dashboard and run comparison, programmatic querying |
| [Artifact Guide](docs/guide/ARTIFACTS.md) | Output directory structure, training vs optimization artifact differences |
| [Testing Guide](docs/guide/TESTING.md) | Test suite, quality automation scripts, CI/CD pipeline details |
| [`orchard/`](orchard/README.md) / [`tests/`](tests/README.md) | Internal package structure, module responsibilities, extension points |

<h2>Citation</h2>

```bibtex
@software{orchardml2026,
  author = {Tommaso Russo},
  title  = {Orchard ML: Type-Safe Deep Learning Framework},
  year   = {2026},
  url    = {https://github.com/tomrussobuilds/orchard-ml},
  note   = {PyTorch framework with Pydantic V2 configuration and Optuna optimization}
}
```

---

<h2>Roadmap</h2>

- **Expanded Dataset Domains**: Climate, remote sensing, microscopy
- **Multi-modal Support**: Detection, segmentation hooks
- **Distributed Training**: `DDP`, `FSDP` support for multi-GPU

---

<h2>License</h2>

MIT License - See [LICENSE](LICENSE) for details.

<h2>Contributing</h2>

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

For detailed guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

<h2>Contact</h2>

For questions or collaboration: [GitHub Issues](https://github.com/tomrussobuilds/orchard-ml/issues)