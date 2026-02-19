"""
Core Utilities Package

This package exposes the essential components for configuration, logging,
system management, project constants, and the dynamic dataset registry.
It also includes the RootOrchestrator to manage experiment lifecycle initialization.
"""

# Configuration
from .config import (
    ArchitectureConfig,
    AugmentationConfig,
    Config,
    DatasetConfig,
    EvaluationConfig,
    HardwareConfig,
    OptunaConfig,
    TelemetryConfig,
    TrainingConfig,
)

# Environment & Hardware
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

# Input/Output Utilities
from .io import (
    load_config_from_yaml,
    load_model_weights,
    md5_checksum,
    save_config_as_yaml,
    validate_npz_keys,
)

# Logging
from .logger import (
    Logger,
    LogStyle,
    Reporter,
    log_best_config_export,
    log_optimization_header,
    log_optimization_summary,
    log_pipeline_summary,
    log_study_summary,
    log_training_summary,
    log_trial_params_compact,
    log_trial_start,
)

# Dataset Registry
from .metadata import DATASET_REGISTRY, DatasetMetadata, DatasetRegistryWrapper

# Environment Orchestration
from .orchestrator import InfraManagerProtocol, RootOrchestrator, TimeTracker, TimeTrackerProtocol

# Constants & Paths
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

__all__ = [
    # Configuration
    "Config",
    "HardwareConfig",
    "TelemetryConfig",
    "DatasetConfig",
    "ArchitectureConfig",
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
    "DatasetRegistryWrapper",
    "DATASET_REGISTRY",
    # Orchestration
    "RootOrchestrator",
    "InfraManagerProtocol",
    "TimeTracker",
    "TimeTrackerProtocol",
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
    "log_pipeline_summary",
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
    # I/O
    "save_config_as_yaml",
    "load_config_from_yaml",
    "load_model_weights",
    "validate_npz_keys",
    "md5_checksum",
]
