"""
Environment & Infrastructure Abstraction Layer.

This package centralizes hardware acceleration discovery, system-level
optimizations, and reproducibility protocols. It provides a unified interface
to ensure consistent execution across Local, HPC, and Docker environments.
"""

#                                Exposed Interface                            #
# 3. Process & Resource Guards (from .guards)
from .guards import DuplicateProcessCleaner, ensure_single_instance, release_single_instance

# 1. Hardware, Device & Policy Management (from .hardware, .policy)
from .hardware import (
    apply_cpu_threads,
    configure_system_libraries,
    detect_best_device,
    get_cuda_name,
    get_num_workers,
    get_vram_info,
    to_device_obj,
)
from .policy import determine_tta_mode

# 2. Determinism & Seeding (from .reproducibility)
from .reproducibility import is_repro_mode_requested, set_seed, worker_init_fn

#                                     Exports                                 #
__all__ = [
    # Hardware & Policy
    "configure_system_libraries",
    "detect_best_device",
    "to_device_obj",
    "get_num_workers",
    "apply_cpu_threads",
    "get_cuda_name",
    "determine_tta_mode",
    "get_vram_info",
    # Reproducibility
    "set_seed",
    "worker_init_fn",
    "is_repro_mode_requested",
    # Guards
    "ensure_single_instance",
    "release_single_instance",
    "DuplicateProcessCleaner",
]
