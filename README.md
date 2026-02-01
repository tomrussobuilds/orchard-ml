# 🔮 VisionForge: Type-Safe Deep Learning Framework

---

<!-- Badges Section -->
<p align="center">
  <!-- CI/CD & Quality -->
  <a href="https://github.com/tomrussobuilds/visionforge/actions/workflows/ci.yml">
    <img src="https://github.com/tomrussobuilds/visionforge/actions/workflows/ci.yml/badge.svg" alt="CI/CD Pipeline">
  </a>
  <a href="https://codecov.io/gh/tomrussobuilds/visionforge">
    <img src="https://codecov.io/gh/tomrussobuilds/visionforge/branch/main/graph/badge.svg" alt="Coverage">
  </a>
  <a href="https://docs.pytest.org/">
    <img src="https://img.shields.io/badge/tested%20with-pytest-blue?logo=pytest&logoColor=white" alt="Tested with pytest">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
</p>

<p align="center">
  <!-- Core Technologies -->
  <img src="https://img.shields.io/badge/python-3.10--3.14-blue?logo=python&logoColor=white" alt="Python">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://docs.pydantic.dev/">
    <img src="https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white" alt="Pydantic">
  </a>
  <a href="https://optuna.org/">
    <img src="https://img.shields.io/badge/Optuna-3.0%2B-00ADD8?logo=optuna&logoColor=white" alt="Optuna">
  </a>
</p>

<p align="center">
  <!-- Code Quality & Status -->
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
  </a>
  <a href="https://pycqa.github.io/isort/">
    <img src="https://img.shields.io/badge/imports-isort-1674b1?logo=python&logoColor=white" alt="isort">
  </a>
  <a href="https://flake8.pycqa.org/">
    <img src="https://img.shields.io/badge/linting-flake8-brightgreen?logo=python&logoColor=white" alt="Flake8">
  </a>
  <img src="https://img.shields.io/badge/tests-890%2B-success" alt="Tests">
  <img src="https://img.shields.io/badge/Architecture-Decoupled-blueviolet" alt="Architecture">
  <img src="https://img.shields.io/badge/status-Active-success" alt="Status">
  <a href="https://github.com/tomrussobuilds/visionforge/issues">
    <img src="https://img.shields.io/github/issues/tomrussobuilds/visionforge" alt="GitHub Issues">
  </a>
</p>

---

## 📌 Table of Contents

- [🎯 Overview](#-overview)
- [⚡ Hardware Requirements](#-hardware-requirements)
- [🚀 Quick Start](#-quick-start)
- [📚 Documentation Hub](#-documentation-hub)
- [📊 Experiment Management](#-experiment-management)
- [📚 Citation](#-citation)
- [🗺 Development Roadmap](#-development-roadmap)
- [📄 License](#-license)

---

## 🎯 Overview

**VisionForge** is a research-grade PyTorch training framework engineered for reproducible, scalable computer vision experiments across diverse domains. Built on [MedMNIST v2](https://zenodo.org/records/6496656) medical imaging datasets and expanded to astronomical imaging ([Galaxy10 DECals](https://zenodo.org/records/10845026)), it provides a domain-agnostic platform supporting multi-resolution architectures (28×28 to 224×224+), automated hyperparameter optimization, and cluster-safe execution.

**Key Differentiators:**
- **Type-Safe Configuration Engine**: Pydantic V2-based declarative manifests eliminate runtime errors
- **Zero-Conflict Execution**: Kernel-level file locking (`fcntl`) prevents concurrent runs from corrupting shared resources
- **Intelligent Hyperparameter Search**: Optuna integration with TPE sampling and Median Pruning
- **Hardware-Agnostic**: Auto-detection and optimization for CPU/CUDA/MPS backends
- **Audit-Grade Traceability**: BLAKE2b-hashed run directories with full YAML snapshots

**Supported Architectures:**

| Resolution | Architectures | Parameters | Use Case |
|-----------|--------------|-----------|----------|
| **28×28** | ResNet-18-Adapted | ~11M | Transfer learning baseline |
| **28×28** | MiniCNN | ~94K | Fast prototyping, ablation studies |
| **224×224** | EfficientNet-B0 | ~4.0M | Efficient compound scaling |
| **224×224** | ViT-Tiny | ~5.5M | Patch-based attention, multiple weight variants |

---

## ⚡ Hardware Requirements

### CPU Training (28×28 Only)
- **Supported Resolution**: 28×28 **only**
- **Time**: ~2.5 hours (ResNet-18-Adapted, 60 epochs, 16 cores)
- **Time**: ~10 minutes (MiniCNN, 60 epochs, 16 cores)
- **Architectures**: ResNet-18-Adapted, MiniCNN
- **Use Case**: Development, testing, limited hardware environments

### GPU Training (All Resolutions)
- **28×28 Resolution**: 
  - MiniCNN: ~2-3 minutes (60 epochs)
  - ResNet-18-Adapted: ~5 minutes (60 epochs)
- **224×224 Resolution**: 
  - EfficientNet-B0: ~30 minutes per trial (15 epochs)
  - ViT-Tiny: ~25-35 minutes per trial (30 epochs)
- **VRAM**: 8GB recommended for 224×224 resolution
- **Architectures**: All (ResNet-18-Adapted, MiniCNN, EfficientNet-B0, ViT-Tiny)

> [!WARNING]
> **224×224 training on CPU is not recommended** - it would take 10+ hours per trial. High-resolution training requires GPU acceleration. Only 28×28 resolution has been tested and validated for CPU training.

**Representative Benchmarks** (RTX 5070 Laptop GPU):

| Task | Architecture | Resolution | Device | Time | Notes |
|------|-------------|-----------|--------|------|-------|
| **Smoke Test** | MiniCNN | 28×28 | CPU/GPU | <30s | 1-epoch sanity check |
| **Quick Training** | MiniCNN | 28×28 | GPU | ~2-3 min | 60 epochs |
| **Quick Training** | MiniCNN | 28×28 | CPU (16 cores) | ~30 min | 60 epochs, CPU-validated |
| **Transfer Learning** | ResNet-18-Adapted | 28×28 | GPU | ~5 min | 60 epochs |
| **Transfer Learning** | ResNet-18-Adapted | 28×28 | CPU (16 cores) | ~2.5h | 60 epochs, CPU-validated |
| **High-Res Training** | EfficientNet-B0 | 224×224 | GPU | ~30 min | 15 epochs, **GPU required** |
| **High-Res Training** | ViT-Tiny | 224×224 | GPU | ~25-35 min | 30 epochs, **GPU required** |
| **Optimization Study** | EfficientNet-B0 | 224×224 | GPU | ~2h | 4 trials (early stop at AUC≥0.9999) |
| **Optimization Study** | Various | 224×224 | GPU | ~1.5-5h | 20 trials, highly variable |

>[!Note]
>**Timing Variance**: Optimization times are highly dependent on early stopping criteria, pruning configuration, and dataset complexity:
>- **Early Stopping**: Studies may finish in 1-3 hours if performance thresholds are met quickly (e.g., AUC ≥ 0.9999 after 4 trials)
>- **Full Exploration**: Without early stopping, 20 trials can extend to 5+ hours
>- **Pruning Impact**: Median pruning can save 30-50% of total time by terminating underperforming trials

---

## 🚀 Quick Start

### Step 1: Environment Setup
```bash
# Clone and install dependencies
git clone https://github.com/tomrussobuilds/visionforge.git
cd visionforge
pip install -r requirements.txt
```

### Step 2: Verify Installation (Optional)
```bash
# Run 1-epoch sanity check (~30 seconds, CPU/GPU)
# Downloads BloodMNIST 28×28 by default
python -m tests.smoke_test

# Note: You can skip this step - main.py will auto-download datasets as needed
```

### Step 3: Training Workflow

VisionForge supports a **three-phase workflow** optimized for different hardware configurations:

#### **Phase 1: Baseline Training** (Learn the framework)

Start with a quick training run to familiarize yourself with the pipeline:
```bash
# 28×28 resolution (CPU-compatible)
python main.py --config recipes/config_mini_cnn.yaml              # ~2-3 min GPU, ~5-10 min CPU (faster with best_config)

# or with transfer learning
python main.py --config recipes/config_resnet_18_adapted.yaml     # ~15 min GPU, ~2.5h CPU

# 224×224 resolution (GPU required)
python main.py --config recipes/config_efficientnet_b0.yaml       # ~30 min GPU per trial
python main.py --config recipes/config_vit_tiny.yaml              # ~25-35 min GPU per trial
```

**What happens:**
- Dataset is auto-downloaded to `./dataset/` (cached for future runs)
- Training runs for 60 epochs with early stopping
- Results saved to timestamped directory in `outputs/`

---

#### **Phase 2: Hyperparameter Optimization** (Find best configuration)

Use Optuna to automatically search for optimal hyperparameters:
```bash
# 28×28 resolution - fast iteration
python optimize.py --config recipes/optuna_mini_cnn.yaml              # ~5 min GPU, ~10 min CPU (50 trials with pruning/early_stoppping)
python optimize.py --config recipes/optuna_resnet_18_adapted.yaml     # ~5-10 min GPU (50 trials with pruning/early_stopping)

# 224×224 resolution - requires GPU
python optimize.py --config recipes/optuna_efficientnet_b0.yaml       # ~1.5-5h*, 20 trials, GPU
python optimize.py --config recipes/optuna_vit_tiny.yaml              # ~3-5h*, 20 trials, GPU

# *Time varies significantly due to early stopping (may finish in 1-3h if target AUC reached)
```

**What happens:**
- Explores hyperparameter combinations (learning rate, regularization, augmentation, etc.)
- Each trial trains for 15 epochs with pruning
- Stops early if performance threshold met (e.g., AUC ≥ 0.9999)
- Generates interactive visualization plots and best configuration YAML

**View optimization results:**
```bash
firefox outputs/*/figures/param_importances.html       # See which hyperparameters matter most
firefox outputs/*/figures/optimization_history.html    # Track trial progression
```

---

#### **Phase 3: Production Training** (Train with optimized settings)

Use the best hyperparameters found during optimization for full training:
```bash
# Train with optimized configuration (60 epochs, full validation)
python main.py --config outputs/YYYYMMDD_dataset_model_hash/reports/best_config.yaml
```

**What happens:**
- Full 60-epoch training with best hyperparameters
- Complete evaluation with test-time augmentation
- Final artifacts: model weights, metrics spreadsheet, visualizations

---

### Step 4: Explore Results

All outputs are isolated in timestamped directories:
```bash
ls outputs/YYYYMMDD_dataset_model_hash/
├── figures/          # Confusion matrices, training curves, sample  predictions
├── reports/          # Excel summaries, best_config.yaml (optimization only)
├── models/           # Trained model weights (.pth)
└── database/         # Optuna study database (optimization only)
```

**Key files:**
- `reports/training_summary.xlsx` - Complete metrics and hyperparameters
- `models/best_*.pth` - Best model weights (by validation AUC)
- `reports/best_config.yaml` - Optimized configuration (optimization runs only)

---

### Hardware Recommendations

**28×28 Resolution:**
- ✅ CPU: Functional (~10 min - 2.5h per training run depending on architecture)
- ✅ GPU: Recommended (~2-5 min per training run)
- 💡 Best for: Rapid prototyping, learning the framework, limited hardware

**224×224 Resolution:**
- ❌ CPU: Not recommended (10+ hours per trial)
- ✅ GPU: Required (~25-35 min per trial)
- 💡 Best for: Production training, modern architectures (EfficientNet, ViT)

---

## 📚 Documentation Hub

Comprehensive guides for advanced usage and system internals:

### 🏗️ Architecture & Design
**[Architecture Guide](docs/guide/ARCHITECTURE.md)**
- Core features and technical design principles
- System architecture diagrams
- Dependency relationships
- Component deep-dives (Config engine, Infrastructure safety, Reproducibility)

### ⚙️ Configuration & Customization
**[Configuration Guide](docs/guide/CONFIGURATION.md)**
- Complete parameter reference
- Usage patterns and best practices
- Extending to new datasets (zero-code integration)

### 🎯 Hyperparameter Optimization
**[Optimization Guide](docs/guide/OPTIMIZATION.md)**
- Optuna integration details
- Search space configuration
- Pruning strategies
- Visualization and result analysis

### 🐳 Containerization
**[Docker Training Guide](docs/guide/DOCKER.md)**
- Container build instructions
- GPU-accelerated execution
- Strict reproducibility mode

### 🧪 Testing & Quality
**[Testing Guide](docs/guide/TESTING.md)**
- Test suite organization (890+ tests)
- Quality check automation (`check_quality.sh`)
- Smoke tests and health checks
- CI/CD pipeline details

### 📦 Package Internals
**[orchard/ Package Documentation](orchard/README.md)**
- Internal architecture principles
- Package structure and module responsibilities
- Extension points for developers

**[tests/ Package Documentation](tests/README.md)**
- Test organization and categories
- Running specific test suites
- Coverage reporting

---

## 📊 Experiment Management

Every run generates a complete artifact suite for total traceability. Both training-only and optimization workflows share the same `RunPath` orchestrator, producing BLAKE2b-hashed timestamped directories with consistent structure.

> **Note:** The artifact structure below shows a complete optimization + training workflow. Training-only runs produce a simplified subset (without optimization-specific files like `study.db`, `best_config.yaml`, HTML plots). See [Testing Guide](docs/guide/TESTING.md) and [Optimization Guide](docs/guide/OPTIMIZATION.md) for workflow-specific details.

**Artifact Structure (Optimization + Training):**
```
outputs/20260123_organcmnist_efficientnetb0_a3f7c2/
├── figures/
│   ├── confusion_matrix_efficientnet_b0_224.png
│   ├── training_curves_efficientnet_b0_224.png
│   ├── sample_predictions_efficientnet_b0_224.png
│   ├── param_importances.html          # Interactive importance plot (optimization)
│   ├── optimization_history.html       # Trial progression (optimization)
│   └── parallel_coordinates.html       # Hyperparameter relationships (optimization)
├── reports/
│   ├── training_summary.xlsx           # Comprehensive metrics spreadsheet
│   ├── best_config.yaml                # Optimized configuration (optimization)
│   ├── study_summary.json              # All trials metadata (optimization)
│   └── top_10_trials.xlsx              # Best configurations (optimization)
├── models/
│   └── best_efficientnetb0.pth         # Trained model weights
└── database/
    └── study.db                        # SQLite storage for resumption (optimization)
```

> [!IMPORTANT]
> ### 📂 [View Sample Artifacts](./docs/artifacts)
> Explore Excel reports, YAML configs, and diagnostic plots from real experiments.

## 📚 Citation

```bibtex
@software{visionforge2025,
  author = {Tommaso Russo},
  title  = {VisionForge: Type-Safe Deep Learning Framework},
  year   = {2025},
  url    = {https://github.com/tomrussobuilds/visionforge},
  note   = {PyTorch framework with Pydantic configuration and Optuna optimization}
}
```

---

## 🗺 Development Roadmap

### ✅ Phase 1: Foundation (Completed)
- Architecture adaptation (3×3 stem, MaxPool removal)
- Pydantic-based configuration engine
- Infrastructure safety (flock, process management)

### ✅ Phase 2: Automation (Completed)
- YAML-driven execution model
- Optuna hyperparameter optimization
- Multi-resolution support (28×28, 224×224)
- Comprehensive test suite (890+ tests, 100% coverage)
- CI/CD pipeline with GitHub Actions

### ✅ Phase 3: Modern Architectures (Completed)
- **Vision Transformer (ViT-Tiny)**: Patch-based attention with 3 weight variants
- **MiniCNN**: Compact baseline for rapid prototyping (~50K parameters)
- **Weight Variant Search**: Automatic exploration of ImageNet-1k/21k pretraining strategies

### ✅ Phase 4: Quality Assurance & Multi-Domain Support (Completed)
- **Test Coverage**: 100% across 890+ tests (unit, integration, smoke)
- **Artifact Export**: HTML visualizations (parameter importance, optimization history, slices, parallel coordinates)
- **Comprehensive Reporting**: Excel summaries, JSON metadata, YAML snapshots
- **Multi-Domain Architecture**: Medical imaging (MedMNIST v2) + Astronomical imaging (Galaxy10 DECals)
- **Domain-Specific Converters**: HDF5 to NPZ pipeline for non-standard formats

### 🎯 Current Status
- **Test Coverage**: 100% across 890+ tests (minimal pragma for defensive guards)
- **Architectures**: 4 total (2 for 28×28, 2 for 224×224)
  - 28×28: ResNet-18-Adapted, MiniCNN
  - 224×224: EfficientNet-B0, ViT-Tiny
- **Resolutions**: 2 (28×28, 224×224)
- **Domains**: Medical imaging (MedMNIST v2), Astronomical imaging (Galaxy10 DECals)
- **Export Formats**: PyTorch (.pth), HTML visualizations, Excel reports

### 🔮 Future Enhancements (Roadmap)
- **ONNX Export**: Model serialization for deployment
- **Additional Architectures**: ConvNeXt, EfficientNet-V2, DeiT
- **Expanded Dataset Domains**: Climate, remote sensing, microscopy
- **Multi-modal Support**: Detection, segmentation hooks
- **Distributed Training**: DDP, FSDP support for multi-GPU

> **Development Philosophy**: 
> Incremental feature addition with maintained test coverage. New capabilities are added carefully to preserve the framework's stability and reproducibility guarantees.


---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

## 📧 Contact

For questions or collaboration: [GitHub Issues](https://github.com/tomrussobuilds/visionforge/issues)