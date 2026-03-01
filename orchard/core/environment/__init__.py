"""
Environment & Infrastructure Abstraction Layer.

This package centralizes hardware acceleration discovery, system-level
optimizations, and reproducibility protocols. It provides a unified interface
to ensure consistent execution across Local, HPC, and Docker environments.
"""

# Distributed Environment Detection (from .distributed)
from .distributed import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)

# Process & Resource Guards (from .guards)
from .guards import (
    DuplicateProcessCleaner,
    ensure_single_instance,
    release_single_instance,
)

# Hardware, Device & Policy Management (from .hardware, .policy)
from .hardware import (
    apply_cpu_threads,
    configure_system_libraries,
    detect_best_device,
    get_accelerator_name,
    get_num_workers,
    get_vram_info,
    has_mps_backend,
    to_device_obj,
)
from .policy import determine_tta_mode

# Determinism & Seeding (from .reproducibility)
from .reproducibility import set_seed, worker_init_fn

# Timing (from .timing)
from .timing import TimeTracker, TimeTrackerProtocol

__all__ = [
    # Hardware & Policy
    "configure_system_libraries",
    "detect_best_device",
    "to_device_obj",
    "get_num_workers",
    "apply_cpu_threads",
    "get_accelerator_name",
    "has_mps_backend",
    "determine_tta_mode",
    "get_vram_info",
    # Reproducibility
    "set_seed",
    "worker_init_fn",
    # Guards
    "ensure_single_instance",
    "release_single_instance",
    "DuplicateProcessCleaner",
    # Distributed
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    # Timing
    "TimeTracker",
    "TimeTrackerProtocol",
]
