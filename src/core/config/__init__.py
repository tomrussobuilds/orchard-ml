"""
Configuration Package Initialization.

This package exposes the core Experiment Manifest (Config) and its 
constituent sub-schemas. By centralizing exports here, the rest of the 
application can interact with the configuration engine through a unified 
interface while maintaining a modular internal file structure.
"""

# =========================================================================== #
#                                 Main Engine                                 #
# =========================================================================== #
from .engine import Config

# =========================================================================== #
#                                Sub-Configurations                           #
# =========================================================================== #
from .system_config import SystemConfig
from .training_config import TrainingConfig
from .augmentation_config import AugmentationConfig
from .dataset_config import DatasetConfig
from .evaluation_config import EvaluationConfig
from .models_config import ModelConfig

# =========================================================================== #
#                                    Types                                    #
# =========================================================================== #
from .types import ValidatedPath

# =========================================================================== #
#                                   EXPORTS                                   #
# =========================================================================== #
__all__ = [
    "Config",
    "SystemConfig",
    "TrainingConfig",
    "AugmentationConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "ValidatedPath",
    "ModelConfig"
]