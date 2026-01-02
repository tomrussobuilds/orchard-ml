"""
Environment & Infrastructure Abstraction Layer.

This package centralizes hardware acceleration discovery, system-level 
optimizations, and reproducibility protocols. It provides a unified interface 
to ensure consistent execution across Local, HPC, and Docker environments.
"""

# =========================================================================== #
#                                Exposed Interface                            #
# =========================================================================== #

# 1. Hardware & Device Management (from .device)
from .hardware import (
    configure_system_libraries,
    detect_best_device,
    to_device_obj,
    get_num_workers,
    apply_cpu_threads,
    get_cuda_name,
    determine_tta_mode
)

# 2. Determinism & Seeding (from .reproducibility)
from .reproducibility import (
    set_seed,
    worker_init_fn,
    is_repro_mode_requested
)

# =========================================================================== #
#                                     Exports                                 #
# =========================================================================== #

__all__ = [
    # Hardware
    "configure_system_libraries",
    "detect_best_device",
    "to_device_obj",
    "get_num_workers",
    "apply_cpu_threads",
    "get_cuda_name",
    "determine_tta_mode",
    
    # Reproducibility
    "set_seed",
    "worker_init_fn",
    "is_repro_mode_requested"
]