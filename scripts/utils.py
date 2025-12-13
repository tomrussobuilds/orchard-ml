"""
BloodMNIST Classification Training Pipeline

This script defines the configuration, constants, utility functions, and command-line
argument parsing for a deep learning model training pipeline focused on the BloodMNIST
dataset. It includes utilities for directory setup, MD5 checksum validation, logging,
process management (to prevent duplicate runs), and setting up reproducibility.
The core structure provides the foundation for training a classification model
(e.g., an adapted ResNet) on blood cell images.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import logging
import sys
import os
import hashlib
import random
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Final
from dataclasses import dataclass

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import psutil
import numpy as np
import torch

# Configuration to prevent Python from writing .pyc files (optional but cleaner)
sys.dont_write_bytecode = True

# =========================================================================== #
#                               CONFIG & CONSTANTS
# =========================================================================== #

def get_base_dir() -> Path:
    """Return the base directory of the script or the Current Working Directory (CWD).

    Returns:
        Path: The absolute path to the directory containing this script.
    """
    try:
        # Use the directory of the currently executing file
        return Path(__file__).resolve().parent
    except NameError:
        # Fallback for interactive environments (e.g., Jupyter, standard interpreter)
        return Path.cwd()
    
PROJECT_ROOT: Final[Path] = get_base_dir().parent

# Directories
DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "figures"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"

# Ensure all necessary directories exist
for d in (DATASET_DIR, FIGURES_DIR, MODELS_DIR, LOG_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Dataset details
NPZ_PATH: Final[Path] = DATASET_DIR / "bloodmnist.npz"

# MD5 of bloodmnist.npz from MedMNIST for validation
EXPECTED_MD5: Final[str] = "7053d0359d879ad8a5505303e11de1dc"
URL: Final[str] = "https://zenodo.org/record/5208230/files/bloodmnist.npz?download=1"

# Class names from official BloodMNIST taxonomy
BLOODMNIST_CLASSES: Final[list[str]] = [
    "basophil", "eosinophil", "erythroblast", "immature granulocyte",
    "lymphocyte", "monocyte", "neutrophil", "platelet"
]

def _get_num_workers_config() -> int:
    """
    Calculates the default value for num_workers based on the environment variable.

    If DOCKER_REPRODUCIBILITY_MODE is set to '1' or 'TRUE' it returns 0
    to force single-thread execution and ensure bit-per-bit determinism
    in containerized environments. Otherwise, it returns 4 for faster loading.
    
    Returns:
        int: The determined number of data loader workers (0 or 4).
    """
    # Check the environment variable for strict reproducibility mode
    is_docker_reproducible = os.environ.get("DOCKER_REPRODUCIBILITY_MODE", "0").upper() in ("1", "TRUE")

    # Return 0 for strict mode 4 otherwise
    return 0 if is_docker_reproducible else 4

# Training hyperparameters configuration
@dataclass(frozen=True)
class Config:
    """Configuration class for training hyperparameters."""
    seed: int = 42
    batch_size: int = 128
    num_workers: int = _get_num_workers_config()
    epochs: int = 60
    patience: int = 15
    learning_rate: float = 0.008
    momentum: float = 0.9
    weight_decay: float = 5e-4
    mixup_alpha: float = 0.002
    use_tta: bool = True

def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across NumPy, Python's random, and PyTorch.

    Args:
        seed (int): The integer seed value to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(Config().seed)

def get_device(logger: logging.Logger) -> torch.device:
    """
    Determinies the appropriate device (CUDA or CPU) for computation.
    
    Args:
        logger (logging.Logger): The logger instance to report the selected device.
    
    Returns:
        torch.device: The selected device object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Log the device selection
    logger.info(f"Using device: {device}")
    
    return device 

class Logger:
    """Configurable logger with rotating file handler and stdout output.

    This class ensures a single, well-configured logger instance is used across the
    application, supporting both console output and log file rotation.

    Parameters:
        name (str): Logger name.
        log_dir (Path): Directory where log files are stored.
        log_to_file (bool): Enable/disable log file output.
        level (int): Default logging level.
        max_bytes (int): Max size before rotation.
        backup_count (int): Number of rotated files to keep.
    """

    _loggers = {}

    def __init__(
        self,
        name: str = "bloodmnist_pipeline",
        log_dir: Path = Path("logs"),
        log_to_file: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_to_file = log_to_file
        self.level = logging.DEBUG if os.getenv("DEBUG") == "1" else level 
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        if name in Logger._loggers:
            self.logger = Logger._loggers[name]
        else:
            self.logger = logging.getLogger(name)
            self._setup_logger()
            Logger._loggers[name] = self.logger

    def _setup_logger(self):
        """Internal method to configure logging handlers and formatter."""
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        self.logger.setLevel(self.level)
        self.logger.propagate = False

        # Console handler (StreamHandler)
        # Check if a StreamHandler is already present to avoid duplicates
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(formatter)
            self.logger.addHandler(stream)

        # Rotating File handler
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"{self.name}_{timestamp}.log"

            # Check if a RotatingFileHandler is already present to avoid duplicates
            if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
                file_handler = RotatingFileHandler(
                    filename,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count,
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

                global log_file
                log_file = filename

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance.

        Returns:
            logging.Logger: The configured logger object.
        """
        return self.logger

# Global logger instance
logger: logging.Logger = Logger(log_dir=LOG_DIR).get_logger()

# =========================================================================== #
#                                 Utility Functions
# =========================================================================== #

def md5_checksum(path: Path) -> str:
    """Calculate the MD5 checksum of a file in chunks for efficiency.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        # Read the file in 8192-byte chunks
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """Validate that the loaded NPZ dataset contains all expected keys.

    Args:
        data (np.lib.npyio.NpzFile): The loaded NPZ file object (e.g., from np.load).

    Raises:
        ValueError: If any required key is missing from the NPZ file.
    """
    required_keys = {
        "train_images", "train_labels",
        "val_images", "val_labels",
        "test_images", "test_labels",
    }

    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"NPZ file is missing required keys: {missing}")
    

def kill_duplicate_processes(script_name: str = None) -> None:
    """Kills all Python processes executing the same script, excluding the current one.

    This prevents accidental multiple runs of the training script.

    Args:
        script_name (str, optional): The filename of the script to check.
                                     Defaults to the current script's filename.
    """
    if script_name is None:
        script_name = os.path.basename(__file__)
    
    current_pid = os.getpid()
    killed = 0
    # Common Python executable names
    python_executables =  ('python', 'python3', 'python.exe')

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a Python process
            if proc.info['name'] not in python_executables:
                continue
            
            cmdline = proc.cmdline()
            
            # Skip the current process
            if proc.pid == current_pid:
                    continue           
            
            is_match = False
            # Check if the script name is the last argument or the second to last (e.g., 'python script.py')
            if cmdline and cmdline[-1] == script_name:
                    is_match = True
            elif len(cmdline) >= 2 and cmdline[-2] == script_name:
                    is_match = True
            
            if is_match:
                proc.terminate()
                killed += 1
                logger.info(f"Killed duplicate process PID {proc.pid}")
        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed:
        logger.info(f"Killed {killed} duplicate process(es). Waiting 1 second for cleanup...")
        time.sleep(1)



# =========================================================================== #
#                               Argument Parsing
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    """Configure and analyze command line arguments for the training script.

    Returns:
        argparse.Namespace: An object containing all parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="BloodMNIST training pipeline based on adapted ResNet-18."
    )
    
    # Get default config values
    default_config = Config()

    parser.add_argument(
        '--epochs',
        type=int,
        default=default_config.epochs,
        help=f"Number of training epochs. Default: {default_config.epochs}"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=default_config.batch_size,
        help=f"Batch size for data loaders. Default: {default_config.batch_size}"
    )
    parser.add_argument(
        '--lr',
        '--learning_rate',
        type=float,
        default=default_config.learning_rate,
        help=f"Initial learning rate for SGD optimizer. Default: {default_config.learning_rate}"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=default_config.seed,
        help=f"Random seed for reproducibility. Default: {default_config.seed}"
    )
    parser.add_argument(
        '--mixup_alpha',
        type=float,
        default=default_config.mixup_alpha,
        help=f"Alpha parameter for MixUp regularization. Set to 0 to disable. Default: {default_config.mixup_alpha}"
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=default_config.patience,
        help=f"Early stopping patience (epochs without improvement). Default: {default_config.patience}"
    )
    parser.add_argument(
        '--no_tta',
        action='store_true',
        # Setting the opposite of default_config.use_tta for clarity
        help="Disable Test-Time Augmentation (TTA) during final evaluation. (TTA is enabled by default)."
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=default_config.momentum,
        help=f"Momentum factor for the SGD optimizer. Default: {default_config.momentum}"
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=default_config.weight_decay,
        help=f"Weight decay (L2 penalty) for the optimizer. Default: {default_config.weight_decay}"
    )
    return parser.parse_args()