"""
Core Utilities Package

This package exposes the essential components for configuration, logging, 
system management, project constants, and the dynamic dataset registry.
It also includes the RootOrchestrator to manage experiment lifecycle initialization.
"""

# =========================================================================== #
#                                Configuration
# =========================================================================== #
from .config import (
    Config,
    SystemConfig,
    DatasetConfig,
    TrainingConfig,
    AugmentationConfig,
    EvaluationConfig,
)

# =========================================================================== #
#                                Constants & Paths
# =========================================================================== #
from .paths import (
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
    get_num_workers,
    kill_duplicate_processes,
    ensure_single_instance,
    get_cuda_name,
    to_device_obj,
    load_model_weights,
    configure_system_libraries,
    release_single_instance,
    apply_cpu_threads,
    determine_tta_mode,
    apply_cpu_threads,
)

# =========================================================================== #
#                                Input/Output Utilities                       #
# =========================================================================== #
from .io import (
    save_config_as_yaml,
    load_config_from_yaml,
    validate_npz_keys,
    md5_checksum
)

# =========================================================================== #
#                                Command Line Interface
# =========================================================================== #
from .cli import parse_args