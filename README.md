# 🩺 MedMNIST Classification with Adapted ResNet-18

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white)

![Architecture](https://img.shields.io/badge/Architecture-Decoupled-blueviolet)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![License](https://img.shields.io/badge/license-MIT-green)

![Status](https://img.shields.io/badge/status-WIP-orange)
![Issues](https://img.shields.io/github/issues/tomrussobuilds/medmnist)

---

## 📌 Table of Contents

* [📊 Experiment Artifacts & Reporting](#-experiment-artifacts--reporting)
* [🚀 Getting Started](#-getting-started)
* [✨ Key Features](#-key-features--defensive-engineering)
* [🏗 Architecture Details](#-architecture-details)
* [🧩 Internal Dependency Mapping](#-internal-dependency-mapping)
* [📁 Project Structure](#-project-structure)
* [⚙️ Requirements & Installation](#️-requirements--installation)
* [💻 Usage (Local)](#-usage-local)
* [🐳 Docker Execution](#-docker-execution-recommended-for-portability)
* [✅ Environment Verification (Smoke Test)](#-environment-verification-smoke-test)
* [📊 Command Line Arguments](#-command-line-arguments)
* [🗺 Research Goals](#-research-goals--roadmap)

This repository provides a highly reproducible, robust training framework for the **MedMNIST v2** suite (supporting BloodMNIST, DermaMNIST, etc.) using an adapted pretrained ResNet-18 architecture. The goal is to demonstrate solid performance using a minimal configuration that adheres to modern PyTorch best practices.

---

### 📊 Experiment Artifacts & Reporting

Every run is fully documented through a suite of automatically generated artifacts. This ensures total traceability and rapid qualitative assessment.

* **Qualitative Results**: High-resolution grids with correct/incorrect label highlighting.
* **Quantitative Performance**: Comprehensive `.xlsx` reports (Single Source of Truth) containing epoch logs and class-wise metrics.
* **Traceability**: Every run mirrors its exact Pydantic configuration state.

> [!IMPORTANT]
> ### 📂 [Explore All Experiment Artifacts & Samples](./docs/artifacts)
> Click the link above to view sample Excel reports, YAML configs, and full-resolution diagnostic plots.

---

### Visual Diagnostics

<table style="width: 100%; border: none;">
  <tr>
    <td style="width: 50%; text-align: center; border: none; vertical-align: top;">
      <b>Confusion Matrix</b><br>
      <img src="docs/artifacts/confusion_matrix.png" style="width: 100%; max-width: 300px;">
    </td>
    <td style="width: 40%; text-align: center; border: none; vertical-align: top;">
      <b>Training Dynamics</b><br>
      <img src="docs/artifacts/training_curves.png" style="width: 100%;">
    </td>
  </tr>
</table>

### Sample Predictions

<p>
<img src="docs/artifacts/sample_predictions.png" width="500">
</p>

## 🚀 Getting Started

### 1. Installation & Environment
Ensure you have the project structure correctly set up with `src/` as a package:
```bash
# Clone the repository
git clone https://github.com/tomrussobuilds/medmnist.git
cd med_mnist

# (Optional) Add src to PYTHONPATH to enable absolute imports
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

### ✨ Key Features & Defensive Engineering

This pipeline is engineered for unattended, robust execution in research environments and containerized clusters. It moves beyond simple classification by implementing low-level system safeguards:

**Kernel-Level Singleton** (`ensure_single_instance`): Utilizes the `fcntl.flock` Unix syscall to acquire an Exclusive Lock (`LOCK_EX | LOCK_NB`) on a physical lock-file. This prevents race conditions on GPU VRAM or checkpoint corruption by ensuring only one training instance is active at a time.

**Atomic Run Isolation**: Managed via the `RunPaths` utility, every execution generates a unique workspace (`outputs/YYYYMMDD_HHMMSS/`). Logs, high-resolution plots, and Excel reports are isolated to prevent historical data overwrites.

**Proactive Process Guard**: Integrates `psutil` to identify and terminate ghost Python processes sharing the same entry-point. Controlled by the `allow_process_kill` flag, it includes a safety check that automatically disables sanitization in shared environments (detecting `SLURM_JOB_ID` or `PBS_JOBID`) to prevent accidental termination of other users' tasks.

**Data Integrity & Validation**: Implements `MD5` Checksum verification for dataset downloads and a strict `validate_npz_keys` check to ensure the structural integrity of the MedMNIST `.npz` files before memory allocation.

**Deterministic Pipeline**: Implements a dual-layer reproducibility strategy. Beyond global seeding (`Seed 42`), it features a Strict Mode that enforces bit-per-bit reproducibility by activating deterministic GPU kernels (`torch.use_deterministic_algorithms`) and synchronizing multi-process RNG via `worker_init_fn`, automatically scaling to zero workers when total determinism is required.

**System Utilities**: The `environment` module serves as a low-level abstraction layer that manages hardware device selection, ensures process-level atomicity through kernel-level file locking, and enforces strict environment-wide reproducibility.

**Continuous Stability Guard** (`smoke_test.py`): A dedicated diagnostic script that executes a "micro-pipeline" (1 epoch, minimal data subset). It validates the entire execution chain—from weight interpolation to Excel reporting—in less than 30 seconds, ensuring no regressions after architectural changes.

**Metadata-Driven SSOT** (Dataset Registry): The pipeline eliminates hardcoded constants. A centralized registry automatically configures normalization parameters (`mean`/`std`), class mappings, and channel counts (`RGB` vs `Grayscale`) based on the selected MedMNIST subset.

**Hybrid RAM Management**: Optimized for varying hardware constraints. The system automatically performs full RAM caching for smaller datasets to maximize throughput, while utilizing indexed slicing for massive datasets (like `TissueMNIST`) to prevent OOM (Out-of-Memory) errors.

**Dynamic Path Anchoring**: Leveraging a "Search-up" logic, the system dynamically locates the project root by searching for markers (`.git` or `README.md`). This ensures absolute path stability regardless of whether the script is launched from the root, `src/`, or a subfolder.

**Graceful Logger Reconfiguration**: Implements a two-stage logging lifecycle. Initial logs are routed to `STDOUT` for immediate feedback; once the `Orchestrator` initializes the run directory, the logger seamlessly hot-swaps to include a timestamped file handler without losing previous trace data.

---

### 🧩 Internal Dependency Mapping
The framework is designed with strict **Separation of Concerns (SoC)**. Below is the architectural graph showing the decoupling between the core engine, the data handlers, and the reporting silos.

<p align="center">
  <img src="docs/artifacts/framework_map.svg" width="850" alt="Framework Map">
</p>

> *Generated via pydeps. Highlighting the centralized Config hub and the linear flow from Orchestrator to Trainer.*

---


### 🏗 Architecture Details

Standard ResNet-18 is designed for $224 \times 224$ inputs. When applied to the $28 \times 28$ MedMNIST manifold, the standard architecture suffers from aggressive information loss due to its initial downsampling layers. To preserve critical morphological details, the backbone has been modified:

| Layer | Standard ResNet-18 | Adapted ResNet-18 (Ours) | Adaptation Strategy |
| :--- | :--- | :--- | :--- |
| **Input Conv** | $7 \times 7$, stride 2, pad 3 | **$3 \times 3$, stride 1, pad 1** | Bicubic Weight Interpolation |
| **Max Pooling** | $3 \times 3$, stride 2 | **Bypassed (Identity)** | Maintain spatial resolution |
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
├── Dockerfile                   # Image definition: Multi-stage build for reproducibility.
├── requirements.txt             # Python dependencies: Torch 2.0+, V2 Transforms, Pydantic 2.0.
├── tools/                       # Diagnostic & Validation Tools:
│   ├── health_check.py          # Global diagnostic: MD5 integrity, NPZ keys, & samples.
│   ├── smoke_test.py            # Rapid E2E verification: 1-epoch diagnostic.
│   └── unit_test.py             # Initial unit testing suite (WIP).
├── src/                         # Modular package: Core framework logic.
│   ├── core/                    # System Hub: Centralized SSOT.
│   │   ├── config/              # Configuration & Schema Hub:
│   │   │   ├── engine.py        # Logic for YAML parsing & validation.
│   │   │   └── types.py         # Custom Pydantic types & annotations.
│   │   ├── logger/              # Telemetry & Artifact Tracking:
│   │   │   ├── logger.py        # Console & File logging setup.
│   │   │   └── reporter.py      # Experiment reporting engine.
│   │   ├── metadata/            # Dataset Registry & Class mappings.
│   │   ├── environment/         # Infrastructure Layer:
│   │   │   ├── hardware.py      # Hardware discovery (CUDA/MPS) & threading.
│   │   │   └── reproducibility.py # Determinism & Seeding protocols.
│   │   ├── orchestrator.py      # Lifecycle Master & Coordination.
│   │   └── processes.py         # Resource Guard: Kernel-level locking (fcntl).
│   ├── data_handler/            # Loading & Augmentation:
│   │   ├── dataset.py           # MedMNIST logic: RAM caching & indexing.
│   │   └── transforms.py        # V2 Augmentations: Optimized pipelines.
│   ├── models/                  # Model Factory:
│   │   └── resnet_18_adapted.py # Adapted ResNet-18 with Weight Interpolation.
│   ├── trainer/                 # Training Loop & MixUp logic.
│   └── evaluation/              # Analytics, Scoring & Reporting.
└── outputs/                     # Results: Isolated workspace for each run.
```
---

### ⚙️ Requirements & Installation


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
python -m tools.health_check

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
python -m tools.smoke_test
```

This will run a 1-epoch training on a tiny subset of a MedMNIST dataset and verify the generation of all output files.

---

### 🐳 Docker Execution (Recommended for Portability)

The pipeline is containerized using the included `Dockerfile`.

Build the image locally using the provided Dockerfile. This ensures all dependencies and environment are correctly configured.

```bash
sudo docker build -t bloodmnist_image .
```

**Run Experiments**

You can choose between Strict Reproducibility (for testing/validation) and Standard Mode (for performance).

Option A: Strict Reproducibility Mode (Deterministic) Enforces bit-perfect results by disabling multi-processing and forcing deterministic GPU kernels.

```bash
sudo docker run -it --rm \
  -u $(id -u):$(id -g) \
  -e IN_DOCKER=TRUE \
  -e DOCKER_REPRODUCIBILITY_MODE=TRUE \
  -e TORCH_HOME=/tmp/torch_cache \
  -e MPLCONFIGDIR=/tmp/matplotlib_cache \
  -e PYTHONHASHSEED=42 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  bloodmnist_image --dataset bloodmnist
```

Option B: Standard Mode (High Performance) Optimized for training speed using multi-core data loading and standard algorithms.

```bash
sudo docker run -it --rm \
  -u $(id -u):$(id -g) \
  -e IN_DOCKER=TRUE \
  -e TORCH_HOME=/tmp/torch_cache \
  -e MPLCONFIGDIR=/tmp/matplotlib_cache \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/outputs:/app/outputs \
  bloodmnist_image --dataset pathmnist --batch_size 256
```

> [!IMPORTANT]
> The flags `-e TORCH_HOME=/tmp/torch_cache` and `-e MPLCONFIGDIR=/tmp/matplotlib_cache` are mandatory when running with a specific user ID (`-u`) to avoid `Permission Denied` errors in the container's root filesystem.
> The `CUBLAS_WORKSPACE_CONFIG` is also required for bit-perfect `CUDA` determinism.

---

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
| --reproducible | bool | False | Enables strict deterministic algorithms and forces num_workers=0. |
--allow_process_kill | bool | True | Enables/disables termination of duplicate processes (auto-disabled on Clusters). |

---

**Examples**:

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
