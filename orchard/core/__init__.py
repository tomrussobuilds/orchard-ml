"""
Core Utilities Package

This package exposes the essential components for configuration, logging,
system management, project constants, and the dynamic dataset registry.
It also includes the RootOrchestrator to manage experiment lifecycle initialization.
"""

# =========================================================================== #
#                                Command Line Interface                       #
# =========================================================================== #
from .cli import parse_args

# =========================================================================== #
#                                Configuration                                #
# =========================================================================== #
from .config import (
    AugmentationConfig,
    Config,
    DatasetConfig,
    EvaluationConfig,
    HardwareConfig,
    ModelConfig,
    OptunaConfig,
    TelemetryConfig,
    TrainingConfig,
)

# =========================================================================== #
#                                Environment & Hardware                       #
# =========================================================================== #
from .environment import (
    apply_cpu_threads,
    configure_system_libraries,
    detect_best_device,
    determine_tta_mode,
    ensure_single_instance,
    get_cuda_name,
    get_num_workers,
    release_single_instance,
    set_seed,
    to_device_obj,
    worker_init_fn,
)

# =========================================================================== #
#                                Input/Output Utilities                       #
# =========================================================================== #
from .io import (
    load_config_from_yaml,
    load_model_weights,
    md5_checksum,
    save_config_as_yaml,
    validate_npz_keys,
)

# =========================================================================== #
#                                Logging                                      #
# =========================================================================== #
from .logger import (
    Logger,
    LogStyle,
    Reporter,
    log_best_config_export,
    log_optimization_header,
    log_optimization_summary,
    log_study_summary,
    log_training_summary,
    log_trial_params_compact,
    log_trial_start,
)

# =========================================================================== #
#                                Dataset Registry                             #
# =========================================================================== #
from .metadata import DATASET_REGISTRY, DatasetMetadata

# =========================================================================== #
#                                Environment Orchestration                    #
# =========================================================================== #
from .orchestrator import RootOrchestrator

# =========================================================================== #
#                                Constants & Paths                            #
# =========================================================================== #
from .paths import (
    DATASET_DIR,
    LOGGER_NAME,
    OUTPUTS_ROOT,
    PROJECT_ROOT,
    STATIC_DIRS,
    RunPaths,
    get_project_root,
    setup_static_directories,
)

# =========================================================================== #
#                                Public Interface                             #
# =========================================================================== #
__all__ = [
    # Configuration
    "Config",
    "HardwareConfig",
    "TelemetryConfig",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "AugmentationConfig",
    "EvaluationConfig",
    "OptunaConfig",
    # Constants & Paths
    "PROJECT_ROOT",
    "DATASET_DIR",
    "OUTPUTS_ROOT",
    "STATIC_DIRS",
    "LOGGER_NAME",
    "RunPaths",
    "setup_static_directories",
    "get_project_root",
    # Metadata
    "DatasetMetadata",
    "DATASET_REGISTRY",
    # Orchestration
    "RootOrchestrator",
    # Logging
    "Logger",
    "Reporter",
    "log_optimization_header",
    "log_study_summary",
    "log_best_config_export",
    "LogStyle",
    "log_trial_start",
    "log_trial_params_compact",
    "log_optimization_summary",
    "log_training_summary",
    # Environment
    "set_seed",
    "detect_best_device",
    "get_num_workers",
    "get_cuda_name",
    "to_device_obj",
    "configure_system_libraries",
    "apply_cpu_threads",
    "determine_tta_mode",
    "worker_init_fn",
    "ensure_single_instance",
    "release_single_instance",
    "kill_duplicate_processes",
    # I/O
    "save_config_as_yaml",
    "load_config_from_yaml",
    "load_model_weights",
    "validate_npz_keys",
    "md5_checksum",
    # CLI
    "parse_args",
]
