"""
Reproducibility Environment.

This module ensures deterministic behavior across libraries (PyTorch, NumPy, Python).
It handles global seeding, worker initialization for DataLoaders, and enforces 
strict deterministic algorithms when requested via CLI or Docker environment.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import random
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch

# =========================================================================== #
#                            Reproducibility Logic                            #
# =========================================================================== #

def is_repro_mode_requested(cli_flag: bool = False) -> bool:
    """
    Detects if strict reproducibility mode is requested via CLI or environment.
    If either the command line flag is set or the DOCKER_REPRODUCIBILITY_MODE or
    environment variable is set to "TRUE", strict mode is enabled.
    
    Args:
        cli_flag (bool): Value passed from the command line argument.
        
    Returns:
        bool: True if strict mode should be enabled.
    """
    docker_flag = os.environ.get("DOCKER_REPRODUCIBILITY_MODE", "FALSE").upper() == "TRUE"
    return cli_flag or docker_flag

def set_seed(seed: int, strict: bool = False) -> None:
    """
    Ensures deterministic behavior across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to set.
        strict (bool): If True, enforces deterministic algorithms (may impact performance).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        if strict:
            # Enforce bit-per-bit reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            logging.info("STRICT REPRODUCIBILITY ENABLED: Using deterministic algorithms.")
        else:
            # Standard reproducible mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id: int) -> None:
    """
    Initializes random number generators for DataLoader workers to ensure 
    augmentation diversity and reproducibility.
    
    Args:
        worker_id (int): Subprocess ID provided by DataLoader.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    # Combine parent seed with worker ID for a unique sub-seed
    base_seed = worker_info.seed 
    seed = (base_seed + worker_id) % 2**32

    # Synchronize all major PRNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)