"""
BloodMNIST training pipeline based on an adapted ResNet-18.

This script includes:
- dataset download with MD5 validation and atomic writes,
- preprocessing, augmentations, and dataloaders for BloodMNIST,
- a ResNet-18 adapted for 28×28 inputs (3×3 conv1, removed maxpool),
- training with MixUp and SGD,
- a two-phase learning rate schedule:
    • CosineAnnealingLR for the first stage,
    • ReduceLROnPlateau for adaptive fine-tuning,
- early stopping based on validation accuracy,
- optional Test-Time Augmentation (TTA),
- evaluation (accuracy, macro-F1, confusion matrix),
- saving training curves, sample predictions, logs and an Excel report.

Designed to be easy to run, reproducible, and suitable for small medical-image datasets.
"""

from __future__ import annotations

# Standard Library
import os
import hashlib
import logging
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Final, List, Tuple, Sequence

# Third Party
import psutil
# Use Agg backend for matplotlib (non-interactive)
# Can comment out these two lines if interactive plotting or GUI is desired
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


# =========================================================================== #
#                               CONFIG & CONSTANTS
# =========================================================================== #

def get_base_dir():
    """Return directory of this script or CWD if run interactively."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()
    
BASE_DIR = get_base_dir()

# Directories
DATASET_DIR: Final[Path] = BASE_DIR / "dataset"
FIGURES_DIR: Final[Path] = BASE_DIR / "figures"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
LOG_DIR: Final[Path] = BASE_DIR / "logs"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"

for d in (DATASET_DIR, FIGURES_DIR, MODELS_DIR, LOG_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Dataset
NPZ_PATH: Final[Path] = DATASET_DIR / "bloodmnist.npz"
# MD5 of bloodmnist.npz from MedMNIST
EXPECTED_MD5: Final[str] = "7053d0359d879ad8a5505303e11de1dc"
URL: Final[str] = "https://zenodo.org/record/5208230/files/bloodmnist.npz?download=1"

# Class names from official BloodMNIST taxonomy
BLOODMNIST_CLASSES: Final[list[str]] = [
    "basophil", "eosinophil", "erythroblast", "immature granulocyte",
    "lymphocyte", "monocyte", "neutrophil", "platelet"
]


# Training hyperparameters
SEED: Final[int] = 42
BATCH_SIZE: Final[int] = 128
NUM_WORKERS: Final[int] = 4
EPOCHS: Final[int] = 60
PATIENCE: Final[int] = 15
LEARNING_RATE: Final[float] = 0.008
MIXUP_ALPHA: Final[float] = 0.001

# =========================================================================== #
#                                 LOGGING 
# =========================================================================== #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    )
logger = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file: Path = LOG_DIR / f"training_{timestamp}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# =========================================================================== #
#                                 SEED 
# =========================================================================== #

def set_seed(seed: int = SEED) -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# =========================================================================== #
#                               UTILITIES
# =========================================================================== #

def md5_checksum(path: Path) -> str:
    """Calculate MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_npz_keys(data) -> None:
    """Validate that the NPZ dataset contains all expected keys."""
    required_keys = {
        "train_images", "train_labels",
        "val_images", "val_labels",
        "test_images", "test_labels",
    }

    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"NPZ file is missing required keys: {missing}")

def kill_duplicate_processes(script_name: str = None):
    """Kills all Python processes that execute the same script but the current one."""
    if script_name is None:
        script_name = os.path.basename(__file__)
    
    current_pid = os.getpid()
    killed = 0

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] not in ('python', 'python3', 'python.exe'):
                continue
            cmdline = proc.cmdline()
            if len(cmdline) > 1 and script_name in cmdline[-1]:
                if proc.pid != current_pid:
                    proc.terminate()
                    killed += 1
                    logger.info(f"Killed duplicate process PID {proc.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed:
        logger.info(f"Killed {killed} duplicate process(es). Proceeding...")
        time.sleep(1)

# =========================================================================== #
#                  DOWNLOAD WITH RETRY AND ATOMIC WRITING
# =========================================================================== #

def ensure_mnist_npz(target_npz: Path, retries: int = 5, delay: float = 5.0) -> Path:
    """
    Download BloodMNIST with retry, atomic write, and strict validation.
    Returns the path to a valid .npz file.
    """
    def _is_valid(path: Path) -> bool:
        if not path.exists() or path.stat().st_size < 50_000:
            return False
        if path.read_bytes()[:2] != b"PK":  # header ZIP
            return False
        return md5_checksum(path) == EXPECTED_MD5

    # File already validated → Exit (fast path)
    if _is_valid(target_npz):
        logger.info(f"Valid dataset found: {target_npz}")
        return target_npz

    # File existing but corrupted → remove
    if target_npz.exists():
        logger.warning(f"Corrupted dataset found, deleting: {target_npz}")
        target_npz.unlink()

    logger.info(f"Downloading BloodMNIST from {URL}")

    tmp_path = target_npz.with_suffix(".tmp")
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(URL, timeout=60)
            response.raise_for_status()

            # Write atomically
            tmp_path.write_bytes(response.content)

            if not _is_valid(tmp_path):
                raise ValueError("Downloaded file failed validation (wrong size/header/MD5)")

            tmp_path.replace(target_npz)
            logger.info(f"Successfully downloaded and verified: {target_npz}")
            return target_npz

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()

            if attempt == retries:
                logger.error(f"Failed to download dataset after {retries} attempts")
                raise RuntimeError("Could not download BloodMNIST dataset") from e

            logger.warning(f"Attempt {attempt}/{retries} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError("Unexpected error in dataset download")

# =========================================================================== #
#                               DATA LOADING
# =========================================================================== #

@dataclass(frozen=True)
class BloodMNISTData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_bloodmnist(npz_path: Path = NPZ_PATH) -> BloodMNISTData:
    """
    Load the NPZ dataset.
    If the NPZ does not exist or is corrupted, calls ensure_mnist_npz().
    """
    path = ensure_mnist_npz(npz_path)

    logger.info(f"Loading dataset from {path}")

    with np.load(npz_path, mmap_mode="r") as data:
        validate_npz_keys(data)
        logger.info(f"Keys in NPZ file: {data.files}")

        return BloodMNISTData(
            X_train=data["train_images"],
            y_train=data["train_labels"].ravel(),
            X_val=data["val_images"],
            y_val=data["val_labels"].ravel(),
            X_test=data["test_images"],
            y_test=data["test_labels"].ravel(),
        )

# =========================================================================== #
#                               DATASET & DATALOADER
# =========================================================================== #

class BloodMNISTDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 transform: transforms.Compose | None = None):
        self.images = images.astype(np.float32) / 255.0
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def get_dataloaders(
        data: BloodMNISTData,
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Strong augmentation for small dataset
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = BloodMNISTDataset(data.X_train, data.y_train, transform=train_transform)
    val_ds   = BloodMNISTDataset(data.X_val,   data.y_val,   transform=val_transform)
    test_ds  = BloodMNISTDataset(data.X_test,  data.y_test,  transform=val_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

# =========================================================================== #
#                               MODEL
# =========================================================================== #

def get_model(device: torch.device) -> nn.Module:
    """Removes maxpool to retain spatial resolution at 28x28."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old_conv = model.conv1

    new_conv = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    with torch.no_grad():
        w = old_conv.weight
        w = F.interpolate(w, size=(3,3), mode='bicubic', align_corners=True)
        new_conv.weight[:] = w
    
    model.conv1 = new_conv
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 8)
    model = model.to(device)
    logger.info(
        "ResNet-18 successfully ADAPTED for 28×28 inputs "
        "(3×3 conv1 + maxpool removed + head changed to 8 classes)"
    )

    return model

# =========================================================================== #
#                               MIXUP UTILITY
# =========================================================================== #

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp augmentation.
    Returns mixed inputs, pairs of targets, and mixing coefficient.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    if device is not None:
        index = torch.randperm(batch_size, device=device)
    else:
        index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """MixUp loss: weighted combination of two losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =========================================================================== #
#                           TEST TIME AUGMENTATION
# =========================================================================== #

def tta_predict(
        model: nn.Module,
        img: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
    """Apply 7-variant deterministic TTA."""
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.rot90(x, k=1, dims=[2,3]),
        lambda x: torch.rot90(x, k=3, dims=[2,3]),
        lambda x: TF.gaussian_blur(x, kernel_size=3, sigma=0.8),
        lambda x: (x + 0.015 * torch.rand_like(x)).clamp(0,1),
        lambda x: x,
    ]
    
    preds = []
    with torch.no_grad():
        img = img.unsqueeze(0).to(device) 
        for t in transforms:
            aug = t(img)
            assert aug.ndim == 4 and aug.shape[1] in (1,3), f"Invalid shape: {aug.shape}"
            logits = model(aug)
            preds.append(F.softmax(logits, dim=1))
    
    return torch.mean(torch.stack(preds), dim=0)


# =========================================================================== #
#                               TRAINING
# =========================================================================== #

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[Path, List[float], List[float]]:
    """Train the model and return (best_model_path, train_losses, val_accuracies)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS*1.5,
        eta_min=1e-4
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        threshold=1e-4,
        cooldown=0,
        min_lr=1e-5,
    )

    best_acc = 0.0
    epochs_no_improve = 0
    best_path = MODELS_DIR / "resnet18_bloodmnist_best.pth"
    train_losses = []
    val_accuracies = []

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            if MIXUP_ALPHA > 0:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, MIXUP_ALPHA, device
                )
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(
                {"loss": f"{running_loss / ((progress_bar.n + 1) * inputs.size(0)):.4f}"}
            )

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc = correct / total
        val_accuracies.append(val_acc)

        # Scheduler step
        if epoch <= 33:
            # First phase: cosine warm descent
            cosine_scheduler.step()
        else:
            # Second phase: adaptive LR reduction
            plateau_scheduler.step(val_acc)

        # Logging & Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model! Val Acc: {val_acc:.4f} ↑")
        else:
            epochs_no_improve += 1
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Best: {best_acc:.4f} | LR: {current_lr:.6f}"
        )

        if epochs_no_improve >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    return best_path, train_losses, val_accuracies

# =========================================================================== #
#                               EVALUATION & PLOTS
# =========================================================================== #

def evaluate_model(
        model: nn.Module,
        test_loader: DataLoader,
        data: BloodMNISTData,
        device: torch.device,
        use_tta=False
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets = targets.numpy()

            if use_tta:
                batch_preds = []
                for img in inputs:
                    pred = tta_predict(model, img, device)
                    batch_preds.append(pred.argmax(dim=1).item())
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(batch_preds)
            all_labels.extend(targets)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) | "
                f"Macro F1: {macro_f1:.4f}"
    )
    if use_tta:
        logger.info("TEST-TIME AUGMENTATION ENABLED")

    return all_preds, all_labels, accuracy, macro_f1


def show_predictions(dataset, preds, n=12, save_path=None):
    plt.figure(figsize=(12,9))
    indices = np.random.choice(len(dataset.X_test), n, replace=False)

    rows = 3
    cols = 4
    for i, idx in enumerate(indices):
        img = dataset.X_test[idx]
        true_label = int(dataset.y_test[idx])
        pred_label = int(preds[idx])

        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        color = "green" if true_label == pred_label else "red"
        plt.title(f"T:{BLOODMNIST_CLASSES[true_label]}\nP:{BLOODMNIST_CLASSES[pred_label]}",
                  color=color, fontsize=10
        )
        plt.axis("off")

    plt.suptitle("Test predictions - ResNet-18 on BloodMNIST", fontsize=16)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
            
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Sample predictions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(train_losses, val_accuracies, out_path):
    fig, ax1 = plt.subplots()

    ax1.plot(train_losses, 'r-', label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='r')

    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, 'b-', label="Validation Accuracy")
    ax2.set_ylabel("Accuracy", color='b')

    plt.title("Training Loss & Validation Accuracy")
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()

def make_plots_and_reports(
    model: nn.Module,
    test_loader: DataLoader,
    data: BloodMNISTData,
    train_losses: list[float],
    val_accuracies: list[float],
    device: torch.device,
) -> Tuple[float, float]:
    """Generate all figures and save predictions after training."""
    model.eval()

    # Test set evaluation
    all_preds, all_labels, test_acc, macro_f1 = evaluate_model(
        model, test_loader, data, device, use_tta=True
    )

    logger.info(f"TEST MACRO F1-SCORE: {macro_f1:.4f} | Test Accuracy: {test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=BLOODMNIST_CLASSES,
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format='.3f')
    plt.title(
        "Confusion Matrix – ResNet-18 on BloodMNIST (normalized by true class)",
        fontsize=14, pad=20
    )
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix_resnet18.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix → {cm_path}")

    # Training curves
    plot_training_curves(train_losses, val_accuracies, FIGURES_DIR / "training_curves.png")
    np.savez(
        FIGURES_DIR / "training_curves.npz",
        train_losses=train_losses,
        val_accuracies=val_accuracies,
    )
    logger.info(f"Training curves → {FIGURES_DIR / 'training_curves.png'}")

    # Sample predictions
    pred_path = FIGURES_DIR / "sample_predictions.png"
    show_predictions(data, all_preds, n=12, save_path=pred_path)
    logger.info(f"Sample predictions → {pred_path}")

    return macro_f1, test_acc

# =========================================================================== #
#                               SAMPLE IMAGES
# =========================================================================== #

def show_sample_images(data: BloodMNISTData, save_path: Path | None = None) -> None:  
    if save_path is None:
        save_path = FIGURES_DIR / "bloodmnist_samples.png"

    if save_path.exists():
        logger.info(f"Sample images already exist → {save_path}")
        return

    indices = np.random.choice(len(data.X_train), size=9, replace=False)

    plt.figure(figsize=(9, 9))
    for i, idx in enumerate(indices):
        img = data.X_train[idx]
        label = int(data.y_train[idx])

        plt.subplot(3, 3, i + 1)

        if img.ndim == 3 and img.shape[-1] == 3:
            plt.imshow(img)
        else:
            plt.imshow(img.squeeze(), cmap='gray')

        plt.title(f"{label} — {BLOODMNIST_CLASSES[label]}", fontsize=11)
        plt.axis("off")

    plt.suptitle("BloodMNIST — 9 random samples from training set", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample images saved → {save_path}")

# =========================================================================== #
#                               EXCEL REPORTS
# =========================================================================== #

@dataclass(frozen=True)
class TrainingReport:
    timestamp: str
    model: str
    dataset: str
    best_val_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    epochs_trained: int
    learning_rate: float
    batch_size: int
    augmentations: str
    normalization: str
    model_path: str
    log_path: str
    seed: int = SEED

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the report into a single-row DataFrame (perfect for Excel)."""
        return pd.DataFrame([asdict(self)])

    def save(self, path: Path | str) -> None:
        """Save the report in Excel (overwrite or create the file)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_excel(path, index=False)
        logger.info(f"Training report saved → {path}")


def build_training_report(
    val_accuracies: Sequence[float],
    macro_f1: float,
    test_acc: float,
    train_losses: Sequence[float],
    best_path: Path,
) -> TrainingReport:
    """
    Builds a structured report with all the metadata from the experiment.
    """
    return TrainingReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model="ResNet-18 (28×28 adapted, ImageNet pretrained)",
        dataset="BloodMNIST",
        best_val_accuracy=max(val_accuracies),
        test_accuracy=test_acc,
        test_macro_f1=macro_f1,
        epochs_trained=len(train_losses),
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        augmentations="HorizontalFlip(0.5), Rotation(±10), ColorJitter, RandomResizedCrop(0.8-1.0)",
        normalization="ImageNet mean/std",
        model_path=str(best_path),
        log_path=str(log_file),
        seed=SEED,
    )

# =========================================================================== #
#                               MAIN
# =========================================================================== #

def main() -> None:
    kill_duplicate_processes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset
    data = load_bloodmnist(NPZ_PATH)
    logger.info(f"Dataset loaded → Train:{len(data.X_train)} | "
                f"Val:{len(data.X_val)} | "
                f"Test:{len(data.X_test)}"
    )

    # Random samples (only the first time)
    # Can comment out this line if you're not interested in visualizing samples,
    # but it's useful for dataset exploration
    show_sample_images(data)

    # DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(data)

    # Model
    model = get_model(device=device)

    # Training
    logger.info("Starting training".center(60, "="))
    
    best_path, train_losses, val_accuracies = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
 
    # Load best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    logger.info(f"Best model loaded from {best_path}")

    # Final report
    macro_f1, test_acc = make_plots_and_reports(
        model=model,
        test_loader=test_loader,
        data=data,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        device=device,
    )

    logger.info(f"FINAL RESULT → Test Accuracy: {test_acc:.4f} | "
                f"Macro F1: {macro_f1:.4f} | "
                f"Best Val: {max(val_accuracies):.4f}"
    )
    logger.info("Training & evaluation completed successfully!")

    # Final Report
    report = build_training_report(
        val_accuracies=val_accuracies,
        macro_f1=macro_f1,
        test_acc=test_acc,
        train_losses=train_losses,
        best_path=best_path,
    )

    logger.info(
        f"FINAL RESULT → Test Accuracy: {report.test_accuracy:.4f} | "
        f"Macro F1: {report.test_macro_f1:.4f} | "
        f"Best Val: {report.best_val_accuracy:.4f}"
    )

    # Save Excel
    excel_path = REPORTS_DIR / "training_report.xlsx"
    report.save(excel_path)

# =========================================================================== #
#                               ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    main()
