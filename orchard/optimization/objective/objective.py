"""
Optuna Objective Function for Vision Pipeline.

Provides OptunaObjective, a highly testable objective function for Optuna
hyperparameter optimization with dependency injection and specialized components.

Architecture:
    - TrialConfigBuilder: Builds trial-specific configurations
    - MetricExtractor: Handles metric extraction and best-value tracking
    - TrialTrainingExecutor: Executes training loops with pruning
    - OptunaObjective: High-level orchestration with dependency injection

Key features:
    - Complete dependency injection for testability
    - Protocol-based abstractions for mocking
    - Single source of truth (all settings from cfg.optuna.*)
    - Memory-efficient cleanup between trials
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import logging
from typing import Any, Dict, Optional, Protocol

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import optuna
import torch

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core import LOGGER_NAME, Config, log_trial_start
from orchard.data_handler import MedMNISTData, get_dataloaders, load_medmnist
from orchard.models import get_model
from orchard.trainer import get_criterion, get_optimizer, get_scheduler

# =========================================================================== #
#                         Relative Imports                                    #
# =========================================================================== #
from .config_builder import TrialConfigBuilder
from .metric_extractor import MetricExtractor
from .training_executor import TrialTrainingExecutor

logger = logging.getLogger(LOGGER_NAME)


# =========================================================================== #
#                          PROTOCOLS                                          #
# =========================================================================== #


class DatasetLoaderProtocol(Protocol):
    """Protocol for dataset loading (enables dependency injection)."""

    def __call__(self, metadata) -> MedMNISTData:
        """Load dataset from metadata."""
        ...


class DataloaderFactoryProtocol(Protocol):
    """Protocol for dataloader creation (enables dependency injection)."""

    def __call__(self, medmnist_data: MedMNISTData, cfg: Config, is_optuna: bool = False) -> tuple:
        """Create train/val/test dataloaders."""
        ...


class ModelFactoryProtocol(Protocol):
    """Protocol for model creation (enables dependency injection)."""

    def __call__(self, device: torch.device, cfg: Config) -> torch.nn.Module:
        """Create and initialize model."""
        ...


# =========================================================================== #
#                          MAIN OBJECTIVE                                     #
# =========================================================================== #


class OptunaObjective:
    """
    Optuna objective function with dependency injection.

    Orchestrates hyperparameter optimization trials by:
    - Building trial-specific configurations
    - Creating data loaders, models, and optimizers
    - Executing training with pruning
    - Tracking and returning best metrics

    All external dependencies are injectable for testability:
    - dataset_loader: Dataset loading function
    - dataloader_factory: DataLoader creation function
    - model_factory: Model instantiation function

    Attributes:
        cfg: Base configuration (single source of truth)
        search_space: Hyperparameter search space
        device: Training device (CPU/CUDA/MPS)
        config_builder: Builds trial-specific configs
        metric_extractor: Handles metric extraction
        medmnist_data: Cached dataset (loaded once, reused across trials)

    Example:
        >>> objective = OptunaObjective(
        ...     cfg=config,
        ...     search_space=search_space,
        ...     device=torch.device("cuda"),
        ... )
        >>> study = optuna.create_study(direction="maximize")
        >>> study.optimize(objective, n_trials=50)
    """

    def __init__(
        self,
        cfg: Config,
        search_space: Dict[str, Any],
        device: torch.device,
        # Optional dependency injection for testing
        dataset_loader: Optional[DatasetLoaderProtocol] = None,
        dataloader_factory: Optional[DataloaderFactoryProtocol] = None,
        model_factory: Optional[ModelFactoryProtocol] = None,
    ):
        """
        Initialize Optuna objective.

        Args:
            cfg: Base configuration (reads optuna.* settings)
            search_space: Hyperparameter search space
            device: Training device
            dataset_loader: Dataset loading function (default: load_medmnist)
            dataloader_factory: DataLoader factory (default: get_dataloaders)
            model_factory: Model factory (default: get_model)
        """
        self.cfg = cfg
        self.search_space = search_space
        self.device = device

        # Dependency injection with defaults
        self._dataset_loader = dataset_loader or load_medmnist
        self._dataloader_factory = dataloader_factory or get_dataloaders
        self._model_factory = model_factory or get_model

        # Components (read metric_name from cfg.optuna for single source of truth)
        self.config_builder = TrialConfigBuilder(cfg)
        self.metric_extractor = MetricExtractor(cfg.optuna.metric_name)

        # Load dataset once (reused across all trials)
        self.medmnist_data = self._dataset_loader(self.config_builder.base_metadata)

        logger.info(f"Objective initialized with metric: {cfg.optuna.metric_name}")
        logger.info(f"Dataset: {self.config_builder.base_metadata.name}")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Execute single Optuna trial.

        Samples hyperparameters, builds trial configuration, trains model,
        and returns best validation metric.

        Args:
            trial: Optuna trial object

        Returns:
            Best validation metric achieved during training

        Raises:
            optuna.TrialPruned: If trial is pruned during training
        """
        # Sample parameters
        params = self._sample_params(trial)

        # Build trial config
        trial_cfg = self.config_builder.build(params)

        # Log trial start
        log_trial_start(trial.number, params)

        try:
            # Setup training components
            train_loader, val_loader, _ = self._dataloader_factory(
                self.medmnist_data, trial_cfg, is_optuna=True
            )
            model = self._model_factory(self.device, trial_cfg)
            optimizer = get_optimizer(model, trial_cfg)
            scheduler = get_scheduler(optimizer, trial_cfg)
            criterion = get_criterion(trial_cfg)

            # Execute training
            executor = TrialTrainingExecutor(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                cfg=trial_cfg,
                device=self.device,
                metric_extractor=self.metric_extractor,
            )

            best_metric = executor.execute(trial)

            return best_metric

        finally:
            # Cleanup GPU memory between trials
            self._cleanup()

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from search space.

        Supports both dict-based search spaces and objects with sample_params method.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled hyperparameters
        """
        if hasattr(self.search_space, "sample_params"):
            return self.search_space.sample_params(trial)
        return {key: fn(trial) for key, fn in self.search_space.items()}

    def _cleanup(self) -> None:
        """
        Clean up GPU memory between trials.

        Note: Orchestrator handles full resource cleanup. This only clears GPU cache.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
