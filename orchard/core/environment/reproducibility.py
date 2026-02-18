"""
Reproducibility Environment.

Ensures deterministic behavior across Python, NumPy, and PyTorch by
centralizing RNG seeding, DataLoader worker initialization, and strict
algorithmic determinism enforcement.

Two reproducibility levels are supported:

    Standard (strict=False):
        Seeds all PRNGs and disables cuDNN auto-tuner. Sufficient for
        most experiments — results are reproducible across runs on the
        same hardware, but non-deterministic kernels (e.g. atomicAdd
        in cuBLAS) may cause minor floating-point variations.

    Strict (strict=True):
        Enables ``torch.use_deterministic_algorithms(True)`` on all backends
        (CUDA, MPS, CPU) and configures ``CUBLAS_WORKSPACE_CONFIG`` when CUDA
        is available. Forces ``num_workers=0`` via HardwareConfig to eliminate
        multiprocessing non-determinism. Incurs a 5-30% performance penalty
        on GPU workloads.

Detection:
    Strict mode is activated by either a CLI flag (``--reproducible``) or the
    ``DOCKER_REPRODUCIBILITY_MODE=TRUE`` environment variable, checked by
    ``is_repro_mode_requested()``.
"""

import logging
import os
import random

import numpy as np
import torch


# REPRODUCIBILITY LOGIC
def is_repro_mode_requested(cli_flag: bool = False) -> bool:
    """Detect if strict reproducibility mode is requested.

    Checks both the CLI flag and the ``DOCKER_REPRODUCIBILITY_MODE``
    environment variable. Either source is sufficient to enable strict mode.

    Args:
        cli_flag: Value passed from the command line argument.

    Returns:
        True if strict mode should be enabled.
    """
    docker_flag = os.environ.get("DOCKER_REPRODUCIBILITY_MODE", "FALSE").upper() == "TRUE"
    return cli_flag or docker_flag


def set_seed(seed: int, strict: bool = False) -> None:
    """Seed all PRNGs and optionally enforce deterministic algorithms.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA + MPS).
    In strict mode, additionally forces deterministic kernels at the
    cost of reduced performance.

    Note:
        ``PYTHONHASHSEED`` is set here for completeness, but CPython reads it
        only at interpreter startup — the runtime assignment has no effect on
        the running process. The project Dockerfile handles this correctly
        (``ENV PYTHONHASHSEED=0``). For bare-metal runs, prefix the command:
        ``PYTHONHASHSEED=42 python forge.py``. Full bit-exact determinism
        additionally requires ``strict=True`` and ``num_workers=0`` (both
        enforced automatically in Docker via ``DOCKER_REPRODUCIBILITY_MODE``).

    Args:
        seed: The seed value to set across all PRNGs.
        strict: If True, enforces deterministic algorithms (5-30% perf penalty).
    """
    random.seed(seed)

    # Best-effort: effective only if set before interpreter startup (see Note)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if strict:
        logging.warning(
            "PYTHONHASHSEED=%s set at runtime, but CPython reads it only at "
            "interpreter startup — this assignment has no effect on the "
            "running process. Full determinism requires the variable to be "
            "set *before* the interpreter starts (the project Dockerfile "
            "does this via ENV PYTHONHASHSEED=0). For bare-metal runs use: "
            "PYTHONHASHSEED=%s python forge.py",
            seed,
            seed,
        )

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if strict:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if strict:
        torch.use_deterministic_algorithms(True)
        logging.info("STRICT REPRODUCIBILITY ENABLED: Using deterministic algorithms.")


def worker_init_fn(worker_id: int) -> None:
    """Initialize PRNGs for a DataLoader worker subprocess.

    Each worker receives a unique but deterministic sub-seed derived from
    the parent seed, ensuring augmentation diversity while maintaining
    reproducibility across runs.

    Called automatically by DataLoader when ``num_workers > 0``.
    In strict reproducibility mode, ``num_workers`` is forced to 0 by
    HardwareConfig, so this function is never invoked.

    Args:
        worker_id: Subprocess ID provided by DataLoader (0-based).
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    # Derive unique sub-seed: deterministic per (parent_seed, worker_id)
    base_seed = worker_info.seed
    seed = (base_seed + worker_id) % 2**32

    # Synchronize all major PRNGs for this worker
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
