"""
Trainer Package Facade

This package exposes the ModelTrainer class and utility functions for
the training lifecycle.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .engine import (
    mixup_data, mixup_criterion, train_one_epoch, validate_epoch
    )

from .trainer import ModelTrainer