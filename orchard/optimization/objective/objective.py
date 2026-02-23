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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import optuna
import torch

from ...core import LOGGER_NAME, Config, log_trial_start

if TYPE_CHECKING:  # pragma: no cover
    from ...tracking import TrackerProtocol

from ...architectures import get_model
from ...data_handler import DatasetData, get_dataloaders, load_dataset
from ...trainer import get_criterion, get_optimizer, get_scheduler

# Relative Imports
from .config_builder import TrialConfigBuilder
from .metric_extractor import MetricExtractor
from .training_executor import TrialTrainingExecutor

logger = logging.getLogger(LOGGER_NAME)


# PROTOCOLS
class DatasetLoaderProtocol(Protocol):
    """Protocol for dataset loading (enables dependency injection)."""

    def __call__(self, metadata) -> DatasetData:
        """Load dataset from metadata."""
        ...  # pragma: no cover


class DataloaderFactoryProtocol(Protocol):
    """Protocol for dataloader creation (enables dependency injection)."""

    def __call__(self, dataset_data: DatasetData, cfg: Config, is_optuna: bool = False) -> tuple:
        """Create train/val/test dataloaders."""
        ...  # pragma: no cover


class ModelFactoryProtocol(Protocol):
    """Protocol for model creation (enables dependency injection)."""

    def __call__(self, device: torch.device, cfg: Config) -> torch.nn.Module:
        """Create and initialize model."""
        ...  # pragma: no cover


# MAIN OBJECTIVE
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
        dataset_data: Cached dataset (loaded once, reused across trials)

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
        search_space: dict[str, Any],
        device: torch.device,
        dataset_loader: DatasetLoaderProtocol | None = None,
        dataloader_factory: DataloaderFactoryProtocol | None = None,
        model_factory: ModelFactoryProtocol | None = None,
        tracker: TrackerProtocol | None = None,
    ) -> None:
        """
        Initialize Optuna objective.

        Args:
            cfg: Base configuration (reads optuna.* settings)
            search_space: Hyperparameter search space
            device: Training device
            dataset_loader: Dataset loading function (default: load_dataset)
            dataloader_factory: DataLoader factory (default: get_dataloaders)
            model_factory: Model factory (default: get_model)
            tracker: Optional experiment tracker for nested trial logging
        """
        self.cfg = cfg
        self.search_space = search_space
        self.device = device
        self.tracker = tracker

        # Dependency injection with defaults
        self._dataset_loader = dataset_loader or load_dataset
        self._dataloader_factory = dataloader_factory or get_dataloaders
        self._model_factory = model_factory or get_model

        # Components (read metric_name from cfg.optuna for single source of truth)
        self.config_builder = TrialConfigBuilder(cfg)
        self.metric_extractor = MetricExtractor(
            cfg.optuna.metric_name, direction=cfg.optuna.direction
        )

        # Load dataset once (reused across all trials)
        self.dataset_data = self._dataset_loader(self.config_builder.base_metadata)

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Execute single Optuna trial.

        Samples hyperparameters, builds trial configuration, trains model,
        and returns best validation metric. Failed trials return the worst
        possible metric instead of crashing the study.

        Args:
            trial: Optuna trial object

        Returns:
            Best validation metric achieved during training,
            or worst-case metric if the trial fails.

        Raises:
            optuna.TrialPruned: If trial is pruned during training
        """
        # Reset per-trial metric tracking
        self.metric_extractor.reset()

        # Sample parameters
        params = self._sample_params(trial)

        # Build trial config
        trial_cfg = self.config_builder.build(params)

        # Inject recipe-level flags for logging (not Optuna params)
        log_params = {**params, "pretrained": self.cfg.architecture.pretrained}

        # Log trial start
        log_trial_start(trial.number, log_params)

        # Start nested MLflow run for this trial
        if self.tracker is not None:
            self.tracker.start_optuna_trial(trial.number, log_params)

        try:
            # Setup training components
            train_loader, val_loader, _ = self._dataloader_factory(
                self.dataset_data, trial_cfg, is_optuna=True
            )
            model = self._model_factory(self.device, trial_cfg)
            optimizer = get_optimizer(model, trial_cfg.training)
            scheduler = get_scheduler(optimizer, trial_cfg.training)
            criterion = get_criterion(trial_cfg.training)

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

        except optuna.TrialPruned:
            raise

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {type(e).__name__}: {e}")
            return self._worst_metric()

        finally:
            # End nested MLflow run for this trial
            if self.tracker is not None:
                best_metric_val = self.metric_extractor.best_metric
                self.tracker.end_optuna_trial(best_metric_val)

            # Cleanup GPU memory between trials
            self._cleanup()

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
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

    def _worst_metric(self) -> float:
        """
        Return worst possible metric based on optimization direction.

        Used as a fallback return value when a trial fails, ensuring
        the study continues without being biased by the failed trial.

        Returns:
            float("inf") for minimize, 0.0 for maximize.
        """
        if self.cfg.optuna.direction == "minimize":
            return float("inf")
        return 0.0

    def _cleanup(self) -> None:
        """
        Clean up GPU/MPS memory between trials.

        Note: Orchestrator handles full resource cleanup. This only clears accelerator cache.
        """
        # Per-trial cache flush (mirrors InfraManager.flush_compute_cache for session teardown)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
