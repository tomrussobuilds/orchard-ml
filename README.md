# 🩺 MedMNIST Classification with Adapted ResNet-18

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

![Status](https://img.shields.io/badge/status-WIP-orange)
![Issues](https://img.shields.io/github/issues/tomrussobuilds/medmnist)
![Stars](https://img.shields.io/github/stars/tomrussobuilds/medmnist?style=social)

**97.78% Test Accuracy • 0.9752 Macro F1 • Single pretrained ResNet-18 • 28×28 images**

## 📌 Table of Contents
* [🚀 Getting Started](#-getting-started)
* [✨ Key Features](#-key-features)
* [🏗 Architecture Details](#-architecture-details)
* [🔬 Training Regularization](#-training-regularization)
* [📁 Project Structure](#-project-structure)
* [⚙️ Requirements & Installation](#️-requirements--installation)
* [💻 Usage (Local )](#-usage-local)
* [🐳 Docker Execution](#-docker-execution)
* [✅ Environment Verification (Smoke Test)](#-environment-verification-smoke-test)
* [📊 Command Line Arguments](#-command-line-arguments)
* [🗺 Research Goals](#-research-goals)

This repository provides a highly reproducible, robust training framework for the **MedMNIST v2** suite (supporting BloodMNIST, DermaMNIST, etc.) using an adapted pretrained ResNet-18 architecture. The goal is to demonstrate solid performance using a minimal configuration that adheres to modern PyTorch best practices.


### Final Results (60 epochs, seed 42)

| Metric                  | Value     |
|-------------------------|-----------|
| Best Validation Accuracy| **96.96%** |
| Test Accuracy (with TTA)| **97.78%** |
| Test Macro F1 (with TTA)| **0.9752** |

### Confusion Matrix
<img src="docs/media/confusion_matrix.png" width="400">

### Training Curves
<img src="docs/media/training_curves.png" width="400">

→ Confusion matrix, training curves, sample predictions and Excel report are automatically saved.

---

## 🚀 Getting Started

### 1. Installation & Environment
Ensure you have the project structure correctly set up with `src/` as a package:
```bash
# Clone the repository
git clone <github.com/tomrussobuilds/medmnist.git>
cd med_mnist

# (Optional) Add src to PYTHONPATH to enable absolute imports
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

### ✨ Key Features & Defensive Engineering

This pipeline is engineered for unattended, robust execution in research environments and containerized clusters. It moves beyond simple classification by implementing low-level system safeguards:

**Kernel-Level Singleton** (ensure_single_instance): Utilizes the fcntl.flock Unix syscall to acquire an Exclusive Lock (LOCK_EX | LOCK_NB) on a physical lock-file. This prevents race conditions on GPU VRAM or checkpoint corruption by ensuring only one training instance is active at a time.

**Atomic Run Isolation**: Managed via the RunPaths utility, every execution generates a unique workspace (outputs/YYYYMMDD_HHMMSS/). Logs, high-resolution plots, and Excel reports are isolated to prevent historical data overwrites.

**Proactive Process Guard**: Integrates psutil to identify and terminate ghost Python processes sharing the same entry-point, optimizing resource allocation in shared HPC or Docker environments.

**Data Integrity & Validation**: Implements MD5 Checksum verification for dataset downloads and a strict validate_npz_keys check to ensure the structural integrity of the MedMNIST .npz files before memory allocation.

**Deterministic Pipeline**: Guaranteed bit-per-bit reproducibility (Seed 42) via torch.backends.cudnn.deterministic and custom worker_init_fn to handle RNG seeding across multi-process DataLoaders.

**System Utilities**: The system.py module serves as a low-level abstraction layer that manages hardware device selection, ensures process-level atomicity through kernel-level file locking, and enforces strict environment-wide reproducibility.

**Continuous Stability Guard** (smoke_test.py): A dedicated diagnostic script that executes a "micro-pipeline" (1 epoch, minimal data subset). It validates the entire execution chain—from weight interpolation to Excel reporting—in less than 30 seconds, ensuring no regressions after architectural changes.

**Metadata-Driven SSOT** (Dataset Registry): The pipeline eliminates hardcoded constants. A centralized registry automatically configures normalization parameters (mean/std), class mappings, and channel counts (RGB vs Grayscale) based on the selected MedMNIST subset.

**Hybrid RAM Management**: Optimized for varying hardware constraints. The system automatically performs full RAM caching for smaller datasets to maximize throughput, while utilizing indexed slicing for massive datasets (like TissueMNIST) to prevent OOM (Out-of-Memory) errors.

**Dynamic Path Anchoring**: Leveraging a "Search-up" logic, the system dynamically locates the project root by searching for markers (`.git` or `README.md`). This ensures absolute path stability regardless of whether the script is launched from the root, `src/`, or a subfolder.

**Graceful Logger Reconfiguration**: Implements a two-stage logging lifecycle. Initial logs are routed to `STDOUT` for immediate feedback; once the Orchestrator initializes the run directory, the logger seamlessly hot-swaps to include a timestamped file handler without losing previous trace data.

---


### 🏗 Architecture Details: ResNet-18 for 28×28

Standard ResNet-18 is designed for $224 \times 224$ inputs. When applied to the $28 \times 28$ MedMNIST manifold, the standard architecture suffers from aggressive information loss due to its initial downsampling layers. To preserve critical morphological details, the backbone has been modified:

| Layer | Standard ResNet-18 | Adapted ResNet-18 (Ours) | Adaptation Strategy |
| :--- | :--- | :--- | :--- |
| **Input Conv** | $7 \times 7$, stride 2, pad 3 | **$3 \times 3$, stride 1, pad 1** | Bicubic Weight Interpolation |
| **Max Pooling** | $3 \times 3$, stride 2 | **Disabled (Identity)** | Maintain spatial resolution |
| **Stage 1 Input** | $56 \times 56$ (from 224) | **$28 \times 28$ (from 28)** | Preserve native resolution |



**Key Modifications:**
1.  **Stem Adaptation**: The initial large-receptive-field convolution is replaced with a $3 \times 3$ kernel. By setting `stride=1`, we avoid losing 75% of the pixel data in the first layer.
2.  **Downsampling Removal**: The initial MaxPool layer is bypassed. In a standard ResNet, the feature map would be reduced to $14 \times 14$ before even reaching the first residual block. Our adaptation enters the residual stages at the full $28 \times 28$ resolution.
3.  **Bicubic Weight Transfer**: To maintain the representational power of the ImageNet-pretrained backbone, weights from the original $7 \times 7$ stem are mapped to the adapted $3 \times 3$ geometry using bicubic interpolation. This ensures the model starts with learned feature detectors rather than random noise.

---

### 🔬 Mathematical Weight Transfer

To retain the representational power of the pretrained backbone, we do not initialize the new $3 \times 3$ kernel randomly. Instead, we perform a spatial transformation on the weight tensor:

**Source Tensor**: Pretrained ImageNet weights 

$$W_{src} \in \mathbb{R}^{C_{out} \times C_{in} \times 7 \times 7}$$

**Interpolation**: Application of a bicubic resizing function $f$ across spatial dimensions:

$$W_{dest} = f(W_{src}, \text{size}=(3, 3))$$

**Result**: A $3 \times 3$ kernel that preserves the edge-detection patterns learned on ImageNet but optimized for a tighter receptive field.

Note: This process ensures that the model starts training with "meaningful" filters rather than noise, leading to faster convergence and higher accuracy on small-scale medical textures.

---

### 🔬 Training Regularization

To improve generalization on the $28 \times 28$ manifold, the pipeline implements MixUp during training. New samples are synthesized as a convex combination of two random samples from the mini-batch:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$
$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

Where $\lambda \in [0, 1]$ is drawn from a $\text{Beta}(\alpha, \alpha)$ distribution.

---

### 📁 Project Structure

```bash
med_mnist/
├── main.py                      # Global entry point: CLI parsing and RootOrchestrator lifecycle.
├── smoke_test.py                # Rapid diagnostic tool: End-to-End pipeline verification (1 epoch).
├── health_check.py              # System diagnostic: MD5 integrity, NPZ keys, and sample generation.
├── Dockerfile                   # Image definition: Multi-stage build for reproducibility.
├── requirements.txt             # Python dependencies: Torch 2.0+, V2 Transforms, Pydantic 2.0.
├── src/                         # Modular package: Core classification framework logic.
│   ├── core/                    # Config & Constants: Centralized logic and SSOT.
│   │   ├── metadata/            # Dataset Registry: Schema definitions and class mappings.
│   │   ├── config.py            # Pydantic models for structured configuration validation.
│   │   └── system.py            # System safeguards: Exclusive locks and process management.
│   ├── data_handler/            # Loading & Augmentation:
│   │   ├── dataset.py           # MedMNIST logic: RAM caching and index management.
│   │   └── transforms.py        # V2 Augmentations: Optimized RGB/Gray pipelines.
│   ├── models/                  # Model Factory:
│   │   └── resnet_18_adapted.py # Adapted ResNet-18 with Bicubic Weight Interpolation.
│   ├── trainer/                 # Training Loop:
│   │   ├── trainer.py           # Main Trainer class: Early stopping and checkpointing.
│   │   └── engine.py            # Functional core: MixUp and epoch-level logic.
│   └── evaluation/              # Analytics & Reporting:
│       ├── engine.py            # Performance scoring (Macro F1, Accuracy).
│       └── reporting.py         # Automated Excel and visualization generation.
└── outputs/                     # Results (ignored): Isolated workspace for each run.
    └── YYYYMMDD_HHMMSS/         # Unique run directory: Logs, models, and figures.
```

### ⚙️ Requirements & Installation

Install dependencies easily with pip:

```bash
pip install -r requirements.txt
```

Install dependencies easily with pip, or check the full list here:

[📦 See Full Requirements](requirements.txt)


### 💻 Usage (Local)

Run the script from the project root. It will default to the fast mode (`num_workers=4`).

```bash
# 1. Setup Environment (Required to resolve src package)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. Verify System
python health_check.py

# 3. Run Training (Registry handles automated download)
python main.py --dataset bloodmnist
```
Note: The entry point script is now `main.py`.

The script will automatically:

- Download BloodMNIST if missing
- Train for max 60 epochs with early stopping (`patience=15`)
- Save the best model → `outputs/YYYYMMDD_HHMMSS_bloodmnist_resnet18/models/best_model.pth`
- Generate figures, confusion matrix, Excel report → `figures/` and `reports/`

---

### ✅ Environment Verification (Smoke Test)
Before starting a full training session, it is highly recommended to run the diagnostic smoke test. This ensures that your local environment, PyTorch versions, and visualization libraries are fully compatible:

```bash
python smoke_test.py
```

This will run a 1-epoch training on a tiny subset of a MedMNIST dataset and verify the generation of all output files.

---

### 🐳 Docker Execution (Recommended for Portability)

The pipeline is containerized using the included `Dockerfile`.

| Mode | Command | `num_workers` | Guarantees |
| :--- | :--- | :---: | :--- |
| **Fast Mode (Default)** | `docker run bloodmnist_image` | 4 | Best performance on CPU, **Not fully deterministic** |
| **Strict Reproducibility** | `-e DOCKER_REPRODUCIBILITY_MODE=TRUE` | 0 | **100% Deterministic** (bit-per-bit), but slower |

---

Build the image locally using the provided Dockerfile. This ensures all dependencies and environment are correctly configured.

```bash
sudo docker build -t bloodmnist_image .
```

Run in Strict Reproducibility Mode (Recommended for testing):

```bash
sudo docker run -it --rm \
  -e DOCKER_REPRODUCIBILITY_MODE=TRUE \
  -v $(pwd)/dataset:/app/dataset \
  bloodmnist_image
```

### 📊 Command Line Arguments

You can fully configure training from the command line (via `main.py`).

| Arg | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| --epochs | int | 60 | Maximum number of training epochs. |
| --batch_size | int | 128 | Batch size for data loaders. |
| --lr | float | 0.008 | Initial learning rate for the SGD optimizer. |
| --seed | int | 42 | Random seed for reproducibility (influences PyTorch, NumPy, Python). |
| --mixup_alpha | float | 0.002 | $\alpha$ parameter for MixUp regularization. Set to 0 to disable MixUp. |
| --patience | int | 15 | Early stopping patience (epochs without validation improvement). |
| --momentum | float | 0.9 | Momentum factor for the SGD optimizer. |
| --weight_decay | float | 5e-4 | Weight decay (L2 penalty) for the optimizer |
| --no_tta | flag | (disabled) | Flag to disable Test-Time Augmentation (TTA) during final evaluation. |
| --hflip | float | 0.5 | Probability of Horizontal Flip augmentation. |
| --rotation_angle | int | 10 | Max rotation angle for random rotations. |
| --jitter_val | float | 0.2 | Strength of Color Jitter (brightness/contrast). |
| --dataset | str | "bloodmnist" | MedMNIST dataset identifier (e.g., bloodmnist, dermamnist). |
| --model_name | str | "ResNet-18 Adapted" | Identifier for logging and folder naming. |


Examples:

Run without TTA

```bash
python main.py --no_tta
```
Train for 100 epochs

```bash
python main.py --epochs 100
```
Lower LR for finer tuning

```bash
python main.py --lr 0.001
```
Disable MixUp

```bash
python main.py --mixup_alpha 0
```
Custom batch size & seed

```bash
python main.py --batch_size 64 --seed 123
```
### Scaling to other MedMNIST datasets
Thanks to the new registry system, you can train on different datasets without changing the code:

```bash
python main.py --dataset dermamnist --lr 0.005 --epochs 100
```

### Citation

If you use this repository in academic work or derivative projects:

```bibtex
@misc{medmnist_resnet18,
  title  = {MedMNIST Classification with Adapted ResNet-18},
  author = {Tommaso Russo},
  year   = {2025},
  url    = {https://github.com/tomrussobuilds/medmnist}
}
```
### 🗺 Research Goals & Roadmap

This project serves as a sandbox for medical imaging experimentation. While the Adapted ResNet-18 has demonstrated that classic architectures—when paired with modern training techniques like MixUp and Stem-Adaptation—remain extremely competitive for MedMNIST, it is only the starting point.

  - Phase 1 (Completed): Optimization on 28×28 BloodMNIST. Implementation of kernel stem resizing and MaxPool removal to preserve spatial resolution.

  - Phase 2 (Near Term): Scaling the pipeline across the entire MedMNIST v2 suite (e.g., DermaMNIST, OrganMNIST) via the DATASET_REGISTRY to validate hyperparameter robustness.

  - Phase 3 (Mid Term): Moving beyond ResNet to evaluate how modern backbones like EfficientNet-V2 and ConvNeXt handle the small-scale medical manifold.

  - Phase 4 (Long Term): Transitioning the repo into a versatile benchmarking framework, extending support for high-resolution formats (224×224) and multi-modal integration.
