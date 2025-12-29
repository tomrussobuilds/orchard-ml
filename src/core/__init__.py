"""
Core Utilities Package

This package exposes the essential components for configuration, logging, 
system management, project constants, and the dynamic dataset registry.
It also includes the RootOrchestrator to manage experiment lifecycle initialization.
"""

# =========================================================================== #
#                                Configuration                                #
# =========================================================================== #
from .config import (
    Config,
    SystemConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    AugmentationConfig,
    EvaluationConfig,
)

# =========================================================================== #
#                                Constants & Paths                            #
# =========================================================================== #
from .paths import (
    PROJECT_ROOT, 
    DATASET_DIR,
    OUTPUTS_ROOT,
    STATIC_DIRS,
    LOGGER_NAME,
    RunPaths,
    setup_static_directories
)

# =========================================================================== #
#                                Dataset Registry                             #
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
#                                Environment & Hardware                       #
# =========================================================================== #
from .environment import (
    set_seed, 
    detect_best_device, 
    get_num_workers,
    get_cuda_name,
    to_device_obj,
    configure_system_libraries,
    apply_cpu_threads,
    determine_tta_mode,
    worker_init_fn
)

# =========================================================================== #
#                                Process Management                           #
# =========================================================================== #
from .processes import (
    ensure_single_instance,
    release_single_instance,
    kill_duplicate_processes
)

# =========================================================================== #
#                                Input/Output Utilities                       #
# =========================================================================== #
from .io import (
    save_config_as_yaml,
    load_config_from_yaml,
    load_model_weights,
    validate_npz_keys,
    md5_checksum
)

# =========================================================================== #
#                                Command Line Interface                       #
# =========================================================================== #
from .cli import parse_args