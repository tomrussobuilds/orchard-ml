← [Back to Main README](../../README.md)

# Architecture & Design

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
- **Small Datasets** : Full RAM caching for maximum throughput
- **Large Datasets** : Indexed slicing to prevent OOM errors

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

## 🧩 Dependency Graph

<p align="center">
<img src="../framework_map.svg?v=4" width="900" alt="System Dependency Graph">
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
| **Parameters** | ~94K | 220× fewer than ResNet-18-Adapted |
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
| **Parameters** | ~4.0M | 50% fewer than ResNet-50 |
| **Pretrained Weights** | ImageNet-1k | Strong initialization for transfer learning |
| **Input Adaptation** | Dynamic first-layer modification for grayscale | Preserves pretrained knowledge via weight morphing |

---

**Vision Transformer Tiny (ViT-Tiny) for 224×224 Resolution**

Patch-based attention architecture with multiple pretrained weight variants:

| Feature | Specification | Benefit |
|---------|--------------|---------|
| **Architecture** | 12-layer transformer encoder | Global context modeling via self-attention |
| **Parameters** | ~5.5M | Comparable to EfficientNet-B0 |
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
