"""
Trainer Package Facade

This package exposes the central ModelTrainer class, the optimization factories,
and the low-level execution engines, providing a unified interface for the
training lifecycle.
"""

from .engine import compute_auc, mixup_data, train_one_epoch, validate_epoch
from .setup import compute_class_weights, get_criterion, get_optimizer, get_scheduler
from .trainer import ModelTrainer

__all__ = [
    "ModelTrainer",
    "train_one_epoch",
    "validate_epoch",
    "compute_auc",
    "mixup_data",
    "get_optimizer",
    "get_scheduler",
    "get_criterion",
    "compute_class_weights",
]
