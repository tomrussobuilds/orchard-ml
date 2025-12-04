# BloodMNIST Classification with Adapted ResNet-18

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-red)
![Torchaudio](https://img.shields.io/badge/Torchaudio-2.0%2B-yellow)


**97.22% Test Accuracy • 96.95 Macro F1 • Single pretrained ResNet-18 • 28×28 images**

This repository contains a complete, reproducible training pipeline for the **BloodMNIST** (from MedMNIST v2) using a carefully adapted pretrained ResNet-18, achieving **state-of-the-art-level performance** with minimal tricks.

### Confusion Matrix  
![Confusion Matrix](figures/confusion_matrix_resnet18.png)

### Training Curves  
![Training Curves](figures/training_curves.png)

### Sample Predictions  
![Sample Predictions](figures/sample_predictions.png)

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

Spoiler: it goes *very* far.

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

### The "Born from Pain" Utilities

These tiny functions saved my sanity more than once:

- `get_base_dir()` – because once, at 5 AM, after deleting the `figures/` folder to test a clean run, the script happily saved all plots directly into the system trash. I laughed for 10 minutes straight when I realized.
- `kill_duplicate_processes()` – prevents accidentally launching 17 identical trainings that eat all RAM
- `ensure_mnist_npz()` – downloads with retries, atomic write, strict MD5 + ZIP header check (never train on a half-downloaded files again)
- MD5 checksum, graceful process killing, etc.

They look overkill. They are. They are also the reason this script can run unattended overnight without exploding.

---

### Requirements

```bash
pip install -r requirements.txt
```

Install dependencies easily with pip, or check the full list here:

[📦 See Full Requirements](requirements.txt)


### Usage

```bash
git clone https://github.com/yourusername/bloodmnist-resnet18.git
cd bloodmnist-resnet18
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

### Conclusion

This project shows how a simple, well-understood architecture like ResNet-18 can still perform very well on a compact dataset such as BloodMNIST when combined with a careful training setup and reproducible utilities.
The goal was not to push the limits of the benchmark, but to build a clean, reliable and fully deterministic pipeline that others can reuse or adapt with minimal effort.

If this repository is helpful or you decide to build on it, feedback and suggestions are always welcome.
Thanks for taking the time to explore the project.