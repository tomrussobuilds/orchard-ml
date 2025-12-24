"""
Core Utilities Package

This package exposes the essential components for configuration, logging, 
system management, project constants and the dynamic dataset registry.
It also includes the RootOrchestrator to manage experiment lifecycle initialization.
"""

# =========================================================================== #
#                                Configuration
# =========================================================================== #
from .config import (
    Config,
    DatasetConfig,
    TrainingConfig,
    AugmentationConfig
)

# =========================================================================== #
#                                Constants & Paths
# =========================================================================== #
from .constants import (
    PROJECT_ROOT, 
    DATASET_DIR,
    OUTPUTS_ROOT,
    STATIC_DIRS,
    RunPaths,
    setup_static_directories
)

# =========================================================================== #
#                                Dataset Registry
# =========================================================================== #
# Now importing from the metadata package facade
from .metadata import (
    DatasetMetadata,
    DATASET_REGISTRY
)

# =========================================================================== #
#                                Environment Orchestration                    #
# =========================================================================== #
from .orchestrator import RootOrchestrator

# =========================================================================== #
#                                Logging                                      #
# =========================================================================== #
from .logger import Logger

# =========================================================================== #
#                                System Management                            #
# =========================================================================== #
from .system import (
    set_seed, 
    detect_best_device, 
    md5_checksum, 
    validate_npz_keys, 
    kill_duplicate_processes,
    ensure_single_instance,
    get_cuda_name
)

# =========================================================================== #
#                                Command Line Interface
# =========================================================================== #
from .cli import parse_args