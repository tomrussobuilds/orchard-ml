# BloodMNIST Classification with Adapted ResNet-18

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-red)
![Torchaudio](https://img.shields.io/badge/Torchaudio-2.0%2B-yellow)


**97.22% Test Accuracy • 96.95 Macro F1 • Single pretrained ResNet-18 • 28×28 images**

This repository provides a reproducible training pipeline for the BloodMNIST (from MedMNIST v2) using an adapted pretrained ResNet-18, demonstrating solid performance with a straightforward setup.

### Confusion Matrix
<img src="figures/confusion_matrix_resnet18.png" width="400">

### Training Curves
<img src="figures/training_curves.png" width="400">

### Sample Predictions
<img src="figures/sample_predictions.png" width="400">

### Final Results (60 epochs, seed 42)
| Metric                  | Value     |
|-------------------------|-----------|
| Best Validation Accuracy| **97.43%** |
| Test Accuracy (with TTA)| **97.22%** |
| Test Macro F1 (with TTA)| **0.9695** |

→ Confusion matrix, training curves, sample predictions and Excel report are automatically saved.

---

### Why this repo exists

I wanted to see how far a **single pretrained ResNet-18** could go on the tiny 28×28 BloodMNIST dataset with proper adaptation and modern training practices — no Ensembles, no ViTs, no custom backbones.

Spoiler: a carefully adapted ResNet-18 performs surprisingly well, even on 28×28 medical images.

---

### Key Features & Design Choices

- **ResNet-18 adapted for 28×28**:  
  – Replaced 7×7 conv with 3×3 (preserves spatial info)  
  – Removed initial MaxPool → full 28×28 feature maps until the end  
  – ImageNet pretrained weights transferred via bicubic upsampling of the first conv

- Strong but reasonable data augmentation + very light **MixUp** (α = 0.001 – kept silent on purpose, higher values hurt here)

- Cosine annealing for first ~33 epochs → ReduceLROnPlateau afterwards

- Test-Time Augmentation (7 deterministic transforms, averaged)

- Automatic dataset download with MD5 validation and atomic write

- Full reproducibility (fixed seeds, deterministic CuDNN)

- Exhaustive logging, Excel report, confusion matrix, training curves, sample predictions

- A ridiculous amount of defensive utilities born from real pain at 5 AM debugging sessions (see below)

---

### The Small Utilities That Save Large Headaches

A few tiny helpers included in this repo were added after very real 5AM debugging incidents:

- **`get_base_dir()`** — ensures outputs never end up in unexpected system locations  
- **`kill_duplicate_processes()`** — stops accidental multi-launches that hog all RAM  
- **`ensure_mnist_npz()`** — safe dataset download with retries, MD5 check, and atomic write  
- Graceful process cleanup, checksum utilities, debug-safe file creation, etc.

They may look overkill, but they make the whole training pipeline safe to run unattended.

---

### Project Structure

```bash
bloodmnist/
│
├── train_bloodmnist.py       # Main training script
│
├── dataset/                  # BloodMNIST dataset
├── logs/                     # Logs to file
├── figures/                  # Auto-generated plots
├── reports/                  # Excel report + logs
└── models/                   # Saved checkpoints
```

### Requirements

```bash
pip install -r requirements.txt
```

Install dependencies easily with pip, or check the full list here:

[📦 See Full Requirements](requirements.txt)


### Usage

```bash
git clone https://github.com/tomrussobuilds/bloodmnist.git
cd bloodmnist
python train_bloodmnist.py
```

You can also check the training script directly: 

[📄 train_bloodmnist.py](train_bloodmnist.py)

That’s it.
The script will:

- Download BloodMNIST if missing
- Train for max 60 epochs with early stopping (patience=15)
- Save the best model → models/resnet18_bloodmnist_best.pth
- Generate figures, confusion matrix, Excel report → figures/ and reports/

### Reproducibility

Everything is deterministic (seed 42). Run the script twice → same validation curve, same final accuracy.

### Citation

If you use this repository in academic work or derivative projects:

```bibtex
@misc{bloodmnist_resnet18,
  title  = {BloodMNIST Classification with Adapted ResNet-18},
  author = {Tommaso Russo},
  year   = {2025},
  url    = {https://github.com/tomrussobuilds/bloodmnist}
}
```

### Conclusion

This project shows how a classic, lightweight architecture like ResNet-18 can perform extremely well on a compact medical-image dataset when paired with a careful training setup.  

The goal is not to chase leaderboard scores, but to provide a **clean, stable, reproducible** pipeline that others can reuse or extend with minimal friction.

If you find this project useful, feedback and suggestions are always welcome.
