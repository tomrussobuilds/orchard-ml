"""
Evaluation and Reporting Module

This module contains functions for model evaluation, including standard and
Test-Time Augmentation (TTA) prediction. It also provides utilities for
visualization, such as plotting training curves, confusion matrices, and sample
predictions, and for generating a structured final training report saved in Excel format.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Tuple, Sequence, Final, List
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms.functional as TF

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from .utils import (
    Logger, Config, FIGURES_DIR, BLOODMNIST_CLASSES, log_file
)
from .data_handler import BloodMNISTData

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


# =========================================================================== #
#                               EVALUATION UTILITIES
# =========================================================================== #

def tta_predict_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Performs Test-Time Augmentation (TTA) inference on a batch of inputs.

    Applies a set of 6 standard augmentations (flips, rotations, light blur,
    and Gaussian noise) in addition to the original input (7 total). Predictions
    from all augmented versions are averaged in the probability space.

    Args:
        model (nn.Module): The trained PyTorch model.
        inputs (torch.Tensor): The batch of test images.
        device (torch.device): The device to run the inference on.

    Returns:
        torch.Tensor: The averaged softmax probability predictions (mean ensemble).
    """
    model.eval()
    inputs = inputs.to(device)
    
    # Define a list of augmented versions of the input batch
    augs: List[torch.Tensor] = [
        inputs,
        torch.flip(inputs, dims=[3]), # Horizontal flip
        torch.rot90(inputs, k=1, dims=[2, 3]), # 90 degree rotation
        torch.rot90(inputs, k=3, dims=[2, 3]), # 270 degree rotation
        TF.gaussian_blur(inputs, kernel_size=3, sigma=0.8), # Light Gaussian blur
        # Add small Gaussian noise and clamp
        (inputs + 0.015 * torch.randn_like(inputs)).clamp(0, 1),
    ]
    
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for aug in augs:
            logits = model(aug)
            # Use softmax output for averaging
            preds.append(F.softmax(logits, dim=1))
    
    # Stack all predictions and take the mean along the batch dimension
    return torch.stack(preds).mean(0)


def evaluate_model(
        model: nn.Module,
        test_loader: DataLoader,
        data: BloodMNISTData,
        device: torch.device,
        use_tta: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Evaluates the model on the test set, optionally using Test-Time Augmentation (TTA).

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        data (BloodMNISTData): The dataset object (used here for type hint consistency).
        device (torch.device): The device (CPU/CUDA) to run the evaluation on.
        use_tta (bool, optional): Whether to enable TTA prediction. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float]:
            all_preds: Array of model predictions.
            all_labels: Array of true labels.
            accuracy: Test set accuracy.
            macro_f1: Test set Macro F1-score.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Targets are needed as NumPy array for metrics calculation later
            targets_np = targets.numpy()

            if use_tta:
                outputs = tta_predict_batch(model, inputs, device)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()
            
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(batch_preds)
            all_labels.extend(targets_np)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate performance metrics
    accuracy = np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    log_message = (
        f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) | "
        f"Macro F1: {macro_f1:.4f}"
    )
    if use_tta:
        log_message += " | TEST-TIME AUGMENTATION ENABLED"
    
    logger.info(log_message)

    return all_preds, all_labels, accuracy, macro_f1


# =========================================================================== #
#                               VISUALIZATION UTILITIES
# =========================================================================== #

def show_predictions(dataset: BloodMNISTData, preds: np.ndarray, n: int = 12, save_path: Path | None = None) -> None:
    """
    Displays a grid of randomly selected test images with their true and
    predicted labels, highlighting correct vs. incorrect predictions.

    Args:
        dataset (BloodMNISTData): The loaded dataset object (for test data access).
        preds (np.ndarray): The array of model predictions for the test set.
        n (int): The number of samples to display (must be multiple of 4).
        save_path (Path | None): Path to save the figure. If None, the plot is shown.
    """
    # Ensure n is a multiple of 4 for a clean 3x4 grid or similar
    if n > len(dataset.X_test):
        n = len(dataset.X_test)
    
    rows = int(np.ceil(n / 4))
    cols = 4

    plt.figure(figsize=(12, 3 * rows))
    # Randomly select N indices from the test set
    indices = np.random.choice(len(dataset.X_test), n, replace=False)

    for i, idx in enumerate(indices):
        img = dataset.X_test[idx]
        true_label = int(dataset.y_test[idx])
        pred_label = int(preds[idx])

        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        color = "green" if true_label == pred_label else "red"
        
        plt.title(
            f"T:{BLOODMNIST_CLASSES[true_label]}\nP:{BLOODMNIST_CLASSES[pred_label]}",
            color=color, fontsize=10
        )
        plt.axis("off")

    plt.suptitle("Test Predictions — ResNet-18 on BloodMNIST", fontsize=16)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Sample predictions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(train_losses: Sequence[float], val_accuracies: Sequence[float], out_path: Path) -> None:
    """
    Plots the training loss and validation accuracy curves on a dual-axis plot.

    Args:
        train_losses (Sequence[float]): List of training losses per epoch.
        val_accuracies (Sequence[float]): List of validation accuracies per epoch.
        out_path (Path): Path to save the generated plot.
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot Training Loss on the left axis (ax1)
    ax1.plot(train_losses, 'r-', label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Validation Accuracy on the right axis (ax2)
    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, 'b-', label="Validation Accuracy")
    ax2.set_ylabel("Accuracy", color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title("Training Loss & Validation Accuracy", fontsize=14)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(all_labels: np.ndarray, all_preds: np.ndarray, out_path: Path) -> None:
    """
    Generates and saves a normalized confusion matrix plot.

    Args:
        all_labels (np.ndarray): Array of true labels.
        all_preds (np.ndarray): Array of predicted labels.
        out_path (Path): Path to save the generated plot.
    """
    # Calculate the normalized confusion matrix (rows sum to 1)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=BLOODMNIST_CLASSES,
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format='.3f')

    plt.title("Confusion Matrix – ResNet-18 on BloodMNIST", fontsize=14, pad=20)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {out_path}")

def save_training_curves(train_losses: Sequence[float], val_accuracies: Sequence[float], out_dir: Path) -> None:
    """Plots and saves training curves and their raw data to disk."""
    plot_training_curves(train_losses, val_accuracies, out_dir / "training_curves.png")

    # Save raw data for later analysis
    np.savez(
        out_dir / "training_curves.npz",
        train_losses=train_losses,
        val_accuracies=val_accuracies,
    )
    logger.info(f"Training curves data saved → {out_dir / 'training_curves.npz'}")

def save_sample_predictions(data: BloodMNISTData, all_preds: np.ndarray, out_path: Path) -> None:
    """Generates and saves a figure showing sample predictions."""
    show_predictions(data, all_preds, n=12, save_path=out_path)
    logger.info(f"Sample predictions figure saved → {out_path}")


# =========================================================================== #
#                               REPORT GENERATION
# =========================================================================== #

def generate_all_reports(
    model: nn.Module,
    test_loader: DataLoader,
    data: BloodMNISTData,
    train_losses: List[float],
    val_accuracies: List[float],
    device: torch.device,
    use_tta: bool = False
) -> Tuple[float, float]:
    """
    Executes the full evaluation pipeline, generating all figures and metrics.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader.
        data (BloodMNISTData): The structured dataset.
        train_losses (List[float]): Loss history from training.
        val_accuracies (List[float]): Validation accuracy history.
        device (torch.device): The device used for inference.
        use_tta (bool, optional): Whether TTA is enabled for final evaluation.

    Returns:
        Tuple[float, float]: The test macro F1-score and test accuracy.
    """
    
    # --- 1) Evaluate Model Performance ---
    all_preds, all_labels, test_acc, macro_f1 = evaluate_model(
        model, test_loader, data, device, use_tta=use_tta
    )

    # --- 2) Confusion Matrix Figure ---
    plot_confusion_matrix(
        all_labels,
        all_preds,
        FIGURES_DIR / "confusion_matrix_resnet18.png"
    )

    # --- 3) Training Curves Figure & Data ---
    save_training_curves(
        train_losses,
        val_accuracies,
        FIGURES_DIR
    )

    # --- 4) Sample Predictions Figure ---
    save_sample_predictions(
        data,
        all_preds,
        FIGURES_DIR / "sample_predictions.png"
    )

    logger.info(f"Evaluation and reporting complete → Accuracy={test_acc:.4f}, Macro F1={macro_f1:.4f}")
    return macro_f1, test_acc


# =========================================================================== #
#                               SAMPLE IMAGES
# =========================================================================== #

def show_sample_images(data: BloodMNISTData, save_path: Path | None = None) -> None:  
    """
    Generates and saves a figure showing 9 random samples from the training set.

    Args:
        data (BloodMNISTData): The structured dataset (to access training images).
        save_path (Path | None, optional): Path to save the figure.
                                           Defaults to FIGURES_DIR/bloodmnist_samples.png.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "bloodmnist_samples.png"

    if save_path.exists():
        logger.info(f"Sample images figure already exists → {save_path}")
        return

    indices = np.random.choice(len(data.X_train), size=9, replace=False)

    plt.figure(figsize=(9, 9))
    for i, idx in enumerate(indices):
        img = data.X_train[idx]
        label = int(data.y_train[idx])

        plt.subplot(3, 3, i + 1)

        # Handle grayscale (1 channel) or color (3 channels) images
        if img.ndim == 3 and img.shape[-1] == 3:
            plt.imshow(img)
        else:
            plt.imshow(img.squeeze(), cmap='gray')

        plt.title(f"{label} — {BLOODMNIST_CLASSES[label]}", fontsize=11)
        plt.axis("off")

    plt.suptitle("BloodMNIST — 9 Random Samples from Training Set", fontsize=16)
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample images saved → {save_path}")

# =========================================================================== #
#                               EXCEL REPORTS
# =========================================================================== #

@dataclass(frozen=True)
class TrainingReport:
    """Structured data container for summarizing a complete training experiment."""
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
    seed: int = Config().seed

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the report dataclass into a single-row Pandas DataFrame."""
        return pd.DataFrame([asdict(self)])

    def save(self, path: Path | str) -> None:
        """Saves the report DataFrame to an Excel file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use to_excel for saving (index=False prevents saving row numbers)
        self.to_dataframe().to_excel(path, index=False)
        logger.info(f"Training report saved → {path}")


def build_training_report(
    val_accuracies: Sequence[float],
    macro_f1: float,
    test_acc: float,
    train_losses: Sequence[float],
    best_path: Path,
    cfg: Config,
) -> TrainingReport:
    """
    Constructs a TrainingReport object using the final metrics and configuration.

    Args:
        val_accuracies (Sequence[float]): List of validation accuracies (to find max).
        macro_f1 (float): Final test macro F1-score.
        test_acc (float): Final test accuracy.
        train_losses (Sequence[float]): List of training losses (to count epochs).
        best_path (Path): Path to the saved model.
        cfg (Config): The configuration object used for the run.

    Returns:
        TrainingReport: The fully populated report object.
    """
    return TrainingReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model="ResNet-18 (28×28 adapted, ImageNet pretrained)",
        dataset="BloodMNIST",
        best_val_accuracy=max(val_accuracies),
        test_accuracy=test_acc,
        test_macro_f1=macro_f1,
        epochs_trained=len(train_losses),
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        augmentations="HorizontalFlip(0.5), Rotation(±10), ColorJitter, RandomResizedCrop(0.9-1.0)",
        normalization="ImageNet mean/std",
        model_path=str(best_path),
        log_path=str(log_file),
        seed=cfg.seed,
    )