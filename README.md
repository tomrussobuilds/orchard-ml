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
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
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
  <img src="https://img.shields.io/badge/tests-800%2B-success" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-%E2%86%92100%25-brightgreen" alt="Coverage Goal">
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
- [✨ Core Features](#-core-features)
- [🏗 System Architecture](#-system-architecture)
- [📊 Experiment Management](#-experiment-management)
- [🧩 Dependency Graph](#-dependency-graph)
- [🔬 Technical Deep Dive](#-technical-deep-dive)
- [📁 Project Structure](#-project-structure)
- [💻 Usage Patterns](#-usage-patterns)
- [🎯 Hyperparameter Optimization](#-hyperparameter-optimization)
- [✅ Environment Verification](#-environment-verification)
- [🐳 Containerized Deployment](#-containerized-deployment)
- [📊 Configuration Reference](#-configuration-reference)
- [🔄 Extending to New Datasets](#-extending-to-new-datasets)
- [🧪 Testing & Quality Assurance](#-testing--quality-assurance)
- [📚 Citation](#-citation)
- [🗺 Development Roadmap](#-development-roadmap)

---

## 🎯 Overview

**VisionForge** is a research-grade PyTorch training framework engineered for reproducible, scalable computer vision experiments. Originally designed for medical imaging (MedMNIST v2), it has evolved into a domain-agnostic platform supporting multi-resolution architectures (28×28 to 224×224+), automated hyperparameter optimization, and cluster-safe execution.

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
| **28×28** | MiniCNN | ~50K | Fast prototyping, ablation studies |
| **224×224** | EfficientNet-B0 | ~5.3M | Efficient compound scaling |
| **224×224** | ViT-Tiny | ~5.7M | Patch-based attention, multiple weight variants |

---

## ⚡ Hardware Requirements

### CPU Training (28×28 Only)
- **Supported Resolution**: 28×28 **only**
- **Time**: ~2.5 hours (ResNet-18-Adapted, 60 epochs, 16 cores)
- **Time**: ~30 minutes (MiniCNN, 60 epochs, 16 cores)
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
# Clone and setup environment
git clone https://github.com/tomrussobuilds/visionforge.git
cd visionforge
pip install -r requirements.txt
```

### Step 2: Smoke Test (30 seconds - GPU/CPU)
```bash
# Verify installation with 1-epoch sanity check
# Downloads BloodMNIST 28×28 by default (or specify --dataset/--resolution)
python -m tests.smoke_test
```

### Step 3: Choose Your Workflow

#### 🔬 **Workflow A: Fast Prototyping** (28×28, CPU-friendly)
```bash
# Quick baseline (~2-3 min GPU, ~30 min CPU)
python main.py --config recipes/config_mini_cnn.yaml

# Transfer learning baseline (~5 min GPU, ~2.5h CPU)
python main.py --config recipes/config_resnet_18_adapted.yaml
```

#### 🎯 **Workflow B: Hyperparameter Optimization → Best Config** (Recommended)
```bash
# 1. Run optimization study (GPU recommended)
python optimize.py --config recipes/optuna_mini_cnn.yaml              # 28×28 (~1-2h)
python optimize.py --config recipes/optuna_efficientnet_b0.yaml       # 224×224 (~1.5-5h)

# 2. View optimization results (interactive HTML plots)
firefox outputs/*/figures/param_importances.html

# 3. Train with optimized hyperparameters (60 epochs, full validation)
python main.py --config outputs/*/reports/best_config.yaml
```

#### 🏆 **Workflow C: State-of-the-Art** (224×224, GPU required)
```bash
# EfficientNet-B0 (~30 min per trial, GPU)
python main.py --config recipes/config_efficientnet_b0.yaml

# Vision Transformer (~25-35 min per trial, GPU)
python main.py --config recipes/config_vit_tiny.yaml
```

### Step 4: Explore Results
```bash
# All artifacts in timestamped directories
ls outputs/YYYYMMDD_dataset_model_hash/
├── figures/          # Confusion matrices, training curves, sample predictions
├── reports/          # Excel summaries, study reports, best configs
├── checkpoints/      # Model weights (.pth)
└── database/         # Optuna SQLite studies (optimization only)
```

---

## ✨ Core Features

### 🔒 Enterprise-Grade Execution Safety

**Tiered Configuration Engine (SSOT)**  
Built on Pydantic V2, the configuration system acts as a **Single Source of Truth**, transforming raw inputs (CLI/YAML) into an immutable, type-safe execution blueprint:

- **Late-Binding Metadata Injection**: Dataset specifications (normalization constants, class mappings) are resolved from a centralized registry at instantiation time
- **Cross-Domain Validation**: Post-construction logic guards prevent unstable states (e.g., enforcing RGB input for pretrained weights, validating AMP compatibility)
- **Path Portability**: Automatic serialization converts absolute paths to environment-agnostic anchors for cross-platform reproducibility

**Infrastructure Guard Layer**  
An independent `InfrastructureManager` bridges declarative configs with physical hardware:

- **Mutual Exclusion via `flock`**: Kernel-level advisory locking ensures only one training instance per workspace (prevents VRAM race conditions)
- **Process Sanitization**: `psutil` wrapper identifies and terminates ghost Python processes
- **HPC-Aware Safety**: Auto-detects cluster schedulers (SLURM/PBS/LSF) and suspends aggressive process cleanup to preserve multi-user stability

**Deterministic Run Isolation**  
Every execution generates a unique workspace using:
```
outputs/YYYYMMDD_DS_MODEL_HASH6/
```
Where `HASH6` is a BLAKE2b cryptographic digest (3-byte, deterministic) computed from the training configuration. Even minor hyperparameter variations produce isolated directories, preventing resource overlap and ensuring auditability.

### 🔬 Reproducibility Architecture

**Dual-Layer Reproducibility Strategy:**
1. **Standard Mode**: Global seeding (Seed 42) with performance-optimized algorithms
2. **Strict Mode**: Bit-perfect reproducibility via:
   - `torch.use_deterministic_algorithms(True)`
   - `worker_init_fn` for multi-process RNG synchronization
   - Auto-scaling to `num_workers=0` when determinism is critical

**Data Integrity Validation:**
- MD5 checksum verification for dataset downloads
- `validate_npz_keys` structural integrity checks before memory allocation

### ⚡ Performance Optimization

**Hybrid RAM Management:**
- **Small Datasets** (<50K samples): Full RAM caching for maximum throughput
- **Large Datasets** (>100K samples): Indexed slicing to prevent OOM errors

**Dynamic Path Anchoring:**
- "Search-up" logic locates project root via markers (`.git`, `README.md`)
- Ensures absolute path stability regardless of invocation directory

**Graceful Logger Reconfiguration:**
- Initial logs route to `STDOUT` for immediate feedback
- Hot-swap to timestamped file handler post-initialization without trace loss

### 🎯 Intelligent Hyperparameter Search

**Optuna Integration Features:**
- **TPE Sampling**: Tree-structured Parzen Estimator for efficient search space exploration
- **Median Pruning**: Early stopping of underperforming trials (30-50% time savings)
- **Persistent Studies**: SQLite storage enables resume-from-checkpoint
- **Type-Safe Constraints**: All search spaces respect Pydantic validation bounds
- **Auto-Visualization**: Parameter importance plots, optimization history, parallel coordinates

---

## 🏗 System Architecture

The framework implements **Separation of Concerns (SoC)** with five core layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      RootOrchestrator                           │
│              (Lifecycle Manager & Context)                      │
│                                                                 │
│  Responsibilities:                                              │
│  • Phase 1-7 initialization sequence                            │
│  • Resource acquisition & cleanup (Context Manager)             │
│  • Device resolution & caching                                  │
└────────────┬─────────────────────────┬──────────────────────────┘
             │                         │
             │ uses                    │ uses
             │                         │
    ┌────────▼──────────┐     ┌────────▼───────────────┐
    │                   │     │                        │
    │  Config Engine    │     │  InfrastructureManager │
    │  (Pydantic V2)    │     │  (flock/psutil)        │
    │                   │     │                        │
    │  • Type safety    │     │  • Process cleanup     │
    │  • Validation     │     │  • Kernel locks        │
    │  • Metadata       │     │  • HPC detection       │
    │    injection      │     │                        │
    └───────────────────┘     └────────────────────────┘
             │
             │ provides config to
             │
    ┌────────▼───────────────────────────────────────────────┐
    │                                                        │
    │              Execution Pipeline                        │
    │                                                        │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
    │  │   Data   │  │  Model   │  │ Trainer  │              │
    │  │ Handler  │→ │ Factory  │→ │  Engine  │              │
    │  └──────────┘  └──────────┘  └────┬─────┘              │
    │                                   │                    │
    │                             ┌─────▼──────┐             │
    │                             │ Evaluation │             │
    │                             │  Pipeline  │             │
    │                             └────────────┘             │
    │                                                        │
    └────────────────────────────┬───────────────────────────┘
                                 │
                                 │ alternative path
                                 │
                        ┌────────▼──────────────┐
                        │  Optimization Engine  │
                        │      (Optuna)         │
                        │                       │
                        │  • Study management   │
                        │  • Trial execution    │
                        │  • Pruning logic      │
                        │  • Visualization      │
                        └───────────────────────┘
```

**Key Design Principles:**

1. Orchestrator owns both Config and InfrastructureManager
2. Config is the SSOT - all modules receive it as dependency
3. InfrastructureManager is stateless utility for OS-level operations
4. Execution pipeline is linear: Data → Model → Training → Eval
5. Optimization wraps the entire pipeline for each trial

---

## 📊 Experiment Management

Every run generates a complete artifact suite for total traceability:

**Artifact Structure:**
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
├── checkpoints/
│   └── best_efficientnetb0.pth         # Trained model weights
└── database/
    └── study.db                        # SQLite storage for resumption (optimization)
```

> [!IMPORTANT]
> ### 📂 [View Sample Artifacts](./docs/artifacts)
> Explore Excel reports, YAML configs, and diagnostic plots from real experiments.

---

## 🧩 Dependency Graph

<p align="center">
<img src="docs/framework_map.svg?v=4" width="900" alt="System Dependency Graph">
</p>

> *Generated via `pydeps`. Highlights the centralized Config hub and modular architecture.*

<details>
<summary>🛠️ Regenerate Dependency Graph</summary>

```bash
pydeps orchard \
    --cluster \
    --max-bacon=0 \
    --max-module-depth=4 \
    --only orchard \
    --noshow \
    -T svg \
    -o docs/framework_map.svg
```

**Requirements:** `pydeps` + Graphviz (`sudo apt install graphviz` or `brew install graphviz`)

</details>

---

## 🔬 Technical Deep Dive

### Architecture Adaptation

**ResNet-18 for 28×28 Resolution**

Standard ResNet-18 is optimized for 224×224 ImageNet inputs. Direct application to 28×28 domains causes catastrophic information loss. Our adaptation strategy:

| Layer | Standard ResNet-18 | VisionForge Adapted | Rationale |
|-------|-------------------|---------------------|-----------|
| **Input Conv** | 7×7, stride=2, pad=3 | **3×3, stride=1, pad=1** | Preserve spatial resolution |
| **Max Pooling** | 3×3, stride=2 | **Identity (bypassed)** | Prevent 75% feature loss |
| **Stage 1 Input** | 56×56 (from 224) | **28×28 (from 28)** | Native resolution entry |

**Key Modifications:**
1. **Stem Redesign**: Replacing large-receptive-field convolution avoids immediate downsampling
2. **Pooling Removal**: MaxPool bypass maintains full spatial fidelity into residual stages
3. **Bicubic Weight Transfer**: Pretrained 7×7 weights are spatially interpolated to 3×3 geometry

---

**MiniCNN for 28×28 Resolution**

A compact, custom architecture designed specifically for low-resolution medical imaging:

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Architecture** | 3 conv blocks + global pooling | Fast convergence with minimal parameters |
| **Parameters** | ~50K | 220× fewer than ResNet-18-Adapted |
| **Input Processing** | 28×28 → 14×14 → 7×7 → 1×1 | Progressive spatial compression |
| **Regularization** | Configurable dropout before FC | Overfitting prevention |

**Advantages:**
- **Speed**: 2-3 minutes for full 60-epoch training on GPU
- **Efficiency**: Ideal for rapid prototyping and ablation studies
- **Interpretability**: Simple architecture for educational purposes

---

**EfficientNet-B0 for 224×224 Resolution**

Implements compound scaling (depth, width, resolution) for optimal parameter efficiency:

| Feature | Specification | Benefit |
|---------|--------------|---------|
| **Architecture** | Mobile Inverted Bottleneck Convolution (MBConv) | Memory-efficient feature extraction |
| **Parameters** | ~5.3M | 50% fewer than ResNet-50 |
| **Pretrained Weights** | ImageNet-1k | Strong initialization for transfer learning |
| **Input Adaptation** | Dynamic first-layer modification for grayscale | Preserves pretrained knowledge via weight morphing |

---

**Vision Transformer Tiny (ViT-Tiny) for 224×224 Resolution**

Patch-based attention architecture with multiple pretrained weight variants:

| Feature | Specification | Benefit |
|---------|--------------|---------|
| **Architecture** | 12-layer transformer encoder | Global context modeling via self-attention |
| **Parameters** | ~5.7M | Comparable to EfficientNet-B0 |
| **Patch Size** | 16×16 (196 patches from 224×224) | Efficient sequence length for transformers |
| **Weight Variants** | ImageNet-1k, ImageNet-21k, ImageNet-21k→1k fine-tuned | Optuna-searchable pretraining strategies |

**Supported Weight Variants:**
1. `vit_tiny_patch16_224.augreg_in21k_ft_in1k`: ImageNet-21k pretrained, fine-tuned on 1k (recommended)
2. `vit_tiny_patch16_224.augreg_in21k`: ImageNet-21k pretrained (requires custom head tuning)
3. `vit_tiny_patch16_224`: ImageNet-1k baseline

---

### Mathematical Weight Transfer

To retain ImageNet-learned feature detectors when adapting to grayscale inputs, we apply bicubic interpolation for CNNs and channel averaging for ViT:

**CNN Weight Morphing (ResNet, EfficientNet):**

**Source Tensor:**
```math
W_{\text{src}} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}
```

**Transformation:**
```math
W_{\text{dest}} = \mathcal{I}_{\text{bicubic}}(W_{\text{src}}, \text{size}=(K', K'))
```

For grayscale adaptation:
```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W_{\text{src}}[:, c, :, :]
```

**ViT Patch Embedding Adaptation:**
```math
W_{\text{gray}} = \text{mean}(W_{\text{src}}, \text{dim}=1) \quad \text{where} \quad W_{\text{src}} \in \mathbb{R}^{192 \times 3 \times 16 \times 16}
```

**Result:** Preserves learned edge detectors and texture patterns while adapting to custom input geometry.

---

### Training Regularization

**MixUp Augmentation** synthesizes training samples via convex combinations:

```math
\tilde{x} = \lambda x_i + (1 - \lambda) x_j \quad \text{where} \quad \lambda \sim \text{Beta}(\alpha, \alpha)
```

```math
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
```

This prevents overfitting on small-scale textures and improves generalization.

---

## 📁 Project Structure

```
visionforge/
├── main.py                         # Training entry point
├── optimize.py                     # Hyperparameter search entry point
├── Dockerfile                      # Multi-stage reproducible build
├── requirements.txt                # Pinned dependencies
├── recipes/                        # YAML configuration presets
│   ├── config_resnet_18_adapted.yaml       # 28×28 transfer learning
│   ├── config_mini_cnn.yaml                # 28×28 fast baseline
│   ├── config_efficientnet_b0.yaml         # 224×224 efficient scaling
│   ├── config_vit_tiny.yaml                # 224×224 transformer
│   ├── optuna_resnet_18_adapted.yaml       # Optimization for 28×28
│   ├── optuna_mini_cnn.yaml                # Architecture search 28×28
│   ├── optuna_efficientnet_b0.yaml         # Optimization for 224×224
│   └── optuna_vit_tiny.yaml                # Weight variant search 224×224
├── tests/                          # Test suite (800+ tests, →100% coverage)
│   ├── smoke_test.py               # 1-epoch E2E verification (~30s)
│   ├── health_check.py             # Dataset integrity validation
│   ├── test_config/                # Config engine tests (210 tests)
│   ├── test_core/                  # Core utilities tests (95 tests)
│   ├── test_data_handler/          # Data loading tests (87 tests)
│   ├── test_evaluation/            # Metrics & viz tests (63 tests)
│   ├── test_models/                # Architecture tests (44 tests)
│   ├── test_optimization/          # Optuna integration tests (78 tests)
│   └── test_trainer/               # Training loop tests (142 tests)
├── orchard/                        # Core framework package
│   ├── core/                       # Framework nucleus
│   │   ├── config/                 # Pydantic schemas (8 modules)
│   │   ├── environment/            # Hardware abstraction
│   │   ├── io/                     # Serialization utilities
│   │   ├── logger/                 # Telemetry system
│   │   ├── metadata/               # Dataset registry
│   │   ├── paths/                  # Path management
│   │   ├── cli.py                  # Argument parser
│   │   └── orchestrator.py         # Lifecycle coordinator
│   ├── data_handler/               # Loading strategies
│   ├── models/                     # Architecture factory
│   │   ├── resnet_18_adapted.py    # Adapted ResNet for 28×28
│   │   ├── mini_cnn.py             # Compact CNN for 28×28
│   │   ├── efficientnet_b0.py      # EfficientNet for 224×224
│   │   ├── vit_tiny.py             # Vision Transformer for 224×224
│   │   └── factory.py              # Architecture registry
│   ├── trainer/                    # Training loop
│   │   ├── engine.py               # Core training/validation logic
│   │   ├── trainer.py              # ModelTrainer orchestrator
│   │   ├── losses.py               # Custom loss functions (FocalLoss)
│   │   └── setup.py                # Optimizer/scheduler setup
│   ├── evaluation/                 # Metrics and visualization
│   │   ├── engine.py               # Evaluation orchestration
│   │   ├── metrics.py              # AUC, F1, Accuracy
│   │   ├── visualization.py        # Plots (confusion matrix, curves)
│   │   └── reporting.py            # Excel report generation
│   └── optimization/               # Optuna integration
│       ├── objective/              # Trial execution logic
│       ├── orchestrator/           # Study management & visualization
│       ├── search_spaces.py        # Hyperparameter distributions
│       └── early_stopping.py       # Convergence detection
└── outputs/                        # Isolated run workspaces
    └── YYYYMMDD_dataset_model_hash/
        ├── figures/                # Visualizations (PNG, HTML)
        ├── reports/                # Metrics (XLSX, JSON, YAML)
        ├── models/                 # Model weights (.pth)
        └── database/               # Optuna study (SQLite)
```

---

## 💻 Usage Patterns

### Configuration-Driven Execution

**Recommended Method:** YAML recipes ensure full reproducibility and version control.

```bash
# Verify environment (~30 seconds)
python -m tests.smoke_test

# Train with presets (28×28 resolution, CPU-compatible)
python main.py --config recipes/config_resnet_18_adapted.yaml     # ~5 min GPU, ~2.5h CPU
python main.py --config recipes/config_mini_cnn.yaml              # ~2-3 min GPU, ~30 min CPU

# Train with presets (224×224 resolution, GPU required)
python main.py --config recipes/config_efficientnet_b0.yaml       # ~30 min per trial
python main.py --config recipes/config_vit_tiny.yaml              # ~25-35 min per trial
```

### CLI Overrides

For rapid experimentation (not recommended for production):

```bash
# Quick test on different dataset
python main.py --dataset dermamnist --epochs 10 --batch_size 64

# Custom learning rate schedule
python main.py --lr 0.001 --min_lr 1e-7 --epochs 100

# Disable augmentations
python main.py --mixup_alpha 0 --no_tta
```

> [!WARNING]
> **Configuration Precedence Order:**
> 1. **YAML file** (highest priority - if `--config` is provided)
> 2. **CLI arguments** (only used when no `--config` specified)
> 3. **Defaults** (from Pydantic field definitions)
>
> **When `--config` is provided, YAML values override CLI arguments.** This prevents configuration drift but means CLI flags are ignored. For reproducible research, always use YAML recipes.

---

## 🎯 Hyperparameter Optimization

### Quick Start

```bash
# Install Optuna (if not already present)
pip install optuna plotly timm  # timm required for ViT support

# Run optimization with presets
python optimize.py --config recipes/optuna_resnet_18_adapted.yaml  # 50 trials, ~3-5h
python optimize.py --config recipes/optuna_mini_cnn.yaml           # 50 trials, ~1-2h

# 224×224 resolution (includes weight variant search for ViT)
python optimize.py --config recipes/optuna_efficientnet_b0.yaml    # 20 trials, ~1.5-5h
python optimize.py --config recipes/optuna_vit_tiny.yaml           # 20 trials, ~3-5h

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
└── database/
    └── study.db                    # SQLite storage for resumption
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

## ✅ Environment Verification

**Smoke Test** (1-epoch sanity check):
```bash
# Default: BloodMNIST 28×28
python -m tests.smoke_test

# Custom dataset/resolution
python -m tests.smoke_test --dataset pathmnist --resolution 224
```

**Output:** Validates full pipeline in <30 seconds:
- Dataset loading and preprocessing
- Model instantiation and weight transfer
- Training loop execution
- Evaluation metrics computation
- Excel/PNG artifact generation

**Health Check** (dataset integrity):
```bash
python -m tests.health_check --dataset bloodmnist
```

**Output:** Verifies:
- MD5 checksum matching
- NPZ key structure (`train_images`, `train_labels`, `val_images`, etc.)
- Sample count validation

---

## 🐳 Containerized Deployment

### Build Image

```bash
docker build -t visionforge:latest .
```

### Execution Modes

**Standard Mode** (Performance Optimized):
```bash
docker run -it --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -e TORCH_HOME=/tmp/torch_cache \
  -e MPLCONFIGDIR=/tmp/matplotlib_cache \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18_adapted.yaml
```

**Strict Reproducibility Mode** (Bit-Perfect Determinism):
```bash
docker run -it --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -e IN_DOCKER=TRUE \
  -e DOCKER_REPRODUCIBILITY_MODE=TRUE \
  -e TORCH_HOME=/tmp/torch_cache \
  -e MPLCONFIGDIR=/tmp/matplotlib_cache \
  -e PYTHONHASHSEED=42 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  visionforge:latest \
  --config recipes/config_resnet_18_adapted.yaml
```

> [!NOTE]
> - `TORCH_HOME` and `MPLCONFIGDIR` prevent permission errors in containerized environments
> - `CUBLAS_WORKSPACE_CONFIG` is required for CUDA determinism
> - `--gpus all` requires NVIDIA Container Toolkit

---

## 📊 Configuration Reference

### Core Parameters

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

### Augmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hflip` | float | 0.5 | Horizontal flip probability |
| `rotation_angle` | int | 10 | Max rotation degrees |
| `jitter_val` | float | 0.2 | ColorJitter intensity |
| `min_scale` | float | 0.95 | Minimum RandomResizedCrop scale |
| `no_tta` | bool | False | Disable test-time augmentation |

### Model Parameters

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `model_name` | str | "resnet_18_adapted" | `resnet_18_adapted`, `mini_cnn` (28×28); `efficientnet_b0`, `vit_tiny` (224×224) |
| `pretrained` | bool | True | Use ImageNet weights (N/A for MiniCNN) |
| `weight_variant` | str | None | ViT-specific pretrained variant (e.g., `augreg_in21k_ft_in1k`) |
| `force_rgb` | bool | True | Convert grayscale to 3-channel |
| `resolution` | int | 28 | [28, 224] |

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | "bloodmnist" | MedMNIST identifier |
| `data_root` | Path | `./dataset` | Dataset directory |
| `max_samples` | int | None | Cap training samples (debugging) |
| `use_weighted_sampler` | bool | True | Balance class distribution |

---

## 🔄 Extending to New Datasets

The framework is designed for zero-code dataset integration via the registry system:

### 1. Add Dataset Metadata

Edit `orchard/core/metadata/medmnist_v2_28x28.py` or `medmnist_v2_224x224.py`:

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

### 2. Train Immediately

```bash
python main.py --dataset custom_dataset --epochs 30
```

No code changes required—the configuration engine automatically resolves metadata.

---

## 🧪 Testing & Quality Assurance

### Test Suite

VisionForge includes a comprehensive test suite with **800+ tests** targeting **→100% code coverage**:

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=orchard --cov-report=html

# Run specific test categories
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only

# Run parallel tests (faster)
pytest tests/ -n auto
```

### Test Categories

- **Unit Tests** (650+ tests): Config validation, metadata injection, type safety
- **Integration Tests** (150+ tests): End-to-end pipeline validation, YAML hydration
- **Smoke Tests**: 1-epoch sanity checks (~30 seconds)
- **Health Checks**: Dataset integrity validation


### Continuous Integration

GitHub Actions automatically run on every push:

- ✅ Code quality checks (Black, isort, Flake8)
- ✅ Unit tests across Python 3.10, 3.11, 3.12
- ✅ Smoke tests (E2E validation)
- ✅ Dataset health checks
- ✅ Security scanning (Bandit, Safety)
- ✅ Code coverage reporting (Codecov)

View the latest build status: [![CI/CD](https://github.com/tomrussobuilds/visionforge/actions/workflows/ci.yml/badge.svg)](https://github.com/tomrussobuilds/visionforge/actions/workflows/ci.yml)

---

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
- Comprehensive test suite (800+ tests)
- CI/CD pipeline with GitHub Actions

### ✅ Phase 3: Modern Architectures (Completed)
- **Vision Transformer (ViT-Tiny)**: Patch-based attention with 3 weight variants
- **MiniCNN**: Compact baseline for rapid prototyping (~50K parameters)
- **Architecture Search**: Optuna-driven model selection for both resolutions
- **Weight Variant Search**: Automatic exploration of ImageNet-1k/21k pretraining strategies
- **ONNX Export**: Model serialization for deployment (via `onnx` extra)

### 🔮 Phase 4: Advanced Features (In Progress)
- **Additional Architectures**: ConvNeXt, EfficientNet-V2, DeiT
- **Domain Extension**: Abstract dataset registry for non-medical domains
- **Multi-modal Support**: Detection, segmentation hooks
- **Distributed Training**: DDP, FSDP support for multi-GPU
- **Advanced Export**: TorchScript optimization, quantization, ONNX Runtime
- **Benchmark Suite**: Standardized architecture comparison framework

### 🎯 Current Status
- **Test Coverage**: 98% (→100% goal, 800+ tests)
- **Architectures**: 4 total (2 for 28×28, 2 for 224×224)
  - 28×28: ResNet-18-Adapted, MiniCNN
  - 224×224: EfficientNet-B0, ViT-Tiny
- **Resolutions**: 2 (28×28, 224×224)
- **Export Formats**: PyTorch (.pth), ONNX (.onnx with optional dependency)

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

---

<p align="center">
<strong>Built with ❤️ for reproducible research</strong>
</p>