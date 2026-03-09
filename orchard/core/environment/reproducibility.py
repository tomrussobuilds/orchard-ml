"""
Reproducibility Environment.

Ensures deterministic behavior across Python, NumPy, and PyTorch by
centralizing RNG seeding, DataLoader worker initialization, and strict
algorithmic determinism enforcement.

Three reproducibility levels are supported:

- **Standard** (``strict=False``): Seeds all PRNGs and disables cuDNN
  auto-tuner. Sufficient for most experiments — results are reproducible
  across runs on the same hardware, but non-deterministic kernels
  (e.g. atomicAdd in cuBLAS) may cause minor floating-point variations.
- **Strict** (``strict=True``): Enables
  ``torch.use_deterministic_algorithms(True)`` on all backends (CUDA, MPS,
  CPU) and configures ``CUBLAS_WORKSPACE_CONFIG`` when CUDA is available.
  Forces ``num_workers=0`` via HardwareConfig to eliminate multiprocessing
  non-determinism. Incurs a 5-30% performance penalty on GPU workloads.
- **Strict warn-only** (``strict=True, warn_only=True``): Same as strict,
  but non-deterministic operations emit warnings instead of raising errors.
  Useful for discovering which operations lack deterministic kernels without
  crashing the experiment.

Strict mode is controlled by ``HardwareConfig.use_deterministic_algorithms``,
resolved from the recipe YAML or direct Config construction.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


# REPRODUCIBILITY LOGIC
# Defaults are tested, but mutmut's test-mapping doesn't link default-value
# mutations to the exercising tests — false-positive survivors without pragma.
def set_seed(seed: int, strict: bool = False, warn_only: bool = False) -> None:  # pragma: no mutate
    """
    Seed all PRNGs and optionally enforce deterministic algorithms.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA + MPS).
    In strict mode, additionally forces deterministic kernels at the
    cost of reduced performance.

    Note:
        ``PYTHONHASHSEED`` is set here for completeness, but CPython reads it
        only at interpreter startup — the runtime assignment has no effect on
        the running process. The project Dockerfile handles this correctly
        (``ENV PYTHONHASHSEED=0``). For bare-metal runs, prefix the command:
        ``PYTHONHASHSEED=42 orchard run <recipe>``. Full bit-exact determinism
        additionally requires ``strict=True`` and ``num_workers=0`` (both
        enforced automatically in Docker via ``DOCKER_REPRODUCIBILITY_MODE``).

    Args:
        seed: The seed value to set across all PRNGs.
        strict: If True, enforces deterministic algorithms (5-30% perf penalty).
        warn_only: If True (and strict=True), uses warn-only mode for
            ``torch.use_deterministic_algorithms`` — logs warnings instead of
            raising errors for non-deterministic ops. Ignored when strict
            is False.
    """
    random.seed(seed)

    # Best-effort: effective only if set before interpreter startup (see Note)
    already_set = os.environ.get("PYTHONHASHSEED") == str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if strict and not already_set:
        _stacklevel = 2  # pragma: no mutate
        warnings.warn(
            f"PYTHONHASHSEED={seed} set at runtime, but CPython reads it only at "
            "interpreter startup. For bare-metal determinism: "
            f"PYTHONHASHSEED={seed} orchard run <recipe>",
            stacklevel=_stacklevel,
        )

    np.random.seed(seed)
    torch.manual_seed(seed)

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if has_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if strict:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if has_mps:
        torch.mps.manual_seed(seed)

    if strict:
        if has_mps:
            _stacklevel = 2  # pragma: no mutate
            warnings.warn(
                "MPS backend has partial determinism support in PyTorch. "
                "Some operations may not have deterministic implementations. "
                "Consider using CPU for fully deterministic experiments.",
                stacklevel=_stacklevel,
            )
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize PRNGs for a DataLoader worker subprocess.

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
