"""
Optuna Study Orchestrator — Core Implementation.

Provides ``OptunaOrchestrator``, the lifecycle manager that coordinates
study creation, trial execution, and artifact generation for
hyperparameter optimization. Specialized tasks are delegated to focused
submodules (``builders``, ``exporters``, ``visualizers``).

Key Functions:
    ``run_optimization``: Convenience function that wires the
        orchestrator to the pipeline and returns the completed study.

Key Components:
    ``OptunaOrchestrator``: Study lifecycle manager that assembles
        sampler, pruner, callbacks, and objective, then drives
        ``study.optimize()``. Base ``Config`` is seamlessly overridden
        per trial via ``TrialConfigBuilder``.

Typical Usage:
    >>> from orchard.optimization.orchestrator import run_optimization
    >>> study = run_optimization(cfg=config, device=device, paths=paths)
    >>> print(f"Best trial: {study.best_trial.number}")
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import optuna

from ...core import (
    LOGGER_NAME,
    Config,
    RunPaths,
    log_optimization_header,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...tracking import TrackerProtocol

from ..objective.objective import OptunaObjective
from ..search_spaces import get_search_space
from .builders import build_callbacks, build_pruner, build_sampler
from .exporters import export_best_config, export_study_summary, export_top_trials
from .utils import has_completed_trials
from .visualizers import generate_visualizations

logger = logging.getLogger(LOGGER_NAME)


# STUDY ORCHESTRATOR
class OptunaOrchestrator:
    """
    High-level manager for Optuna hyperparameter optimization studies.

    Coordinates the complete optimization lifecycle: study creation, trial execution,
    and post-processing artifact generation. Integrates with Orchard ML's Config
    and RunPaths infrastructure, delegating specialized tasks (sampler/pruner building,
    visualization, export) to focused submodules.

    This orchestrator serves as the entry point for hyperparameter tuning, wrapping
    Optuna's API with Orchard ML-specific configuration and output management.

    Attributes:
        cfg (Config): Template configuration that will be overridden per trial
        device (torch.device): Hardware target for training (CPU/CUDA/MPS)
        paths (RunPaths): Output directory structure for artifacts and results

    Example:
        >>> orchestrator = OptunaOrchestrator(cfg=config, device=device, paths=paths)
        >>> study = orchestrator.optimize()
        >>> print(f"Best AUC: {study.best_value:.3f}")
        >>> # Artifacts saved to paths.figures/ and paths.exports/
    """

    def __init__(
        self,
        cfg: Config,
        device,
        paths: RunPaths,
        tracker: TrackerProtocol | None = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            cfg (Config): Base Config to override per trial
            device (torch.device): PyTorch device for training
            paths (RunPaths): Root directory for outputs
            tracker (TrackerProtocol | None): Optional experiment tracker for nested trial logging
        """
        self.cfg = cfg
        self.device = device
        self.paths = paths
        self.tracker = tracker

    def create_study(self) -> optuna.Study:
        """Create or load Optuna study with configured sampler and pruner.

        Returns:
            Configured Optuna study instance
        """
        sampler = build_sampler(self.cfg.optuna)
        pruner = build_pruner(self.cfg.optuna)
        storage_url = self.cfg.optuna.get_storage_url(self.paths)

        study = optuna.create_study(
            study_name=self.cfg.optuna.study_name,
            direction=self.cfg.optuna.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=self.cfg.optuna.load_if_exists,
        )

        return study

    def optimize(self) -> optuna.Study:
        """Execute hyperparameter optimization.

        Returns:
            Completed study with trial results
        """
        # Suppress Optuna's internal INFO logs (e.g. "A new study created in RDB")
        # before create_study(); our own header in phases.py is sufficient
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = self.create_study()
        search_space = get_search_space(
            self.cfg.optuna.search_space_preset,
            resolution=self.cfg.dataset.resolution,
            include_models=self.cfg.optuna.enable_model_search,
            model_pool=self.cfg.optuna.model_pool,
            overrides=self.cfg.optuna.search_space_overrides,
        )

        objective = OptunaObjective(
            cfg=self.cfg,
            search_space=search_space,
            device=self.device,
            tracker=self.tracker,
        )

        # Configure callbacks and log our structured header
        log_optimization_header(self.cfg)

        callbacks = build_callbacks(self.cfg.optuna, self.cfg.training.monitor_metric)

        study.set_user_attr("n_trials", self.cfg.optuna.n_trials)

        interrupted = False
        try:
            study.optimize(
                objective,
                n_trials=self.cfg.optuna.n_trials,
                timeout=self.cfg.optuna.timeout,
                n_jobs=self.cfg.optuna.n_jobs,
                show_progress_bar=self.cfg.optuna.show_progress_bar,
                callbacks=callbacks,
            )
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Optimization interrupted by user. Saving partial results...")

        self._post_optimization_processing(study)

        if interrupted:
            logger.warning(
                "Continuing to training in 5 seconds... (Ctrl+C again to abort pipeline)"
            )
            time.sleep(5)  # grace period for the user to fully abort

        return study

    def _post_optimization_processing(self, study: optuna.Study) -> None:
        """Execute all post-optimization tasks.

        Args:
            study: Completed Optuna study
        """
        if not has_completed_trials(study):
            logger.warning(
                "No completed trials. Skipping visualizations, best config, "
                "and detailed exports."
            )
            export_study_summary(study, self.paths)
            return

        if self.cfg.optuna.save_plots:
            generate_visualizations(study, self.paths.figures)

        export_study_summary(study, self.paths)
        export_top_trials(study, self.paths, self.cfg.training.monitor_metric)

        if self.cfg.optuna.save_best_config:
            export_best_config(study, self.cfg, self.paths)


def run_optimization(
    cfg: Config,
    device,
    paths: RunPaths,
    tracker: TrackerProtocol | None = None,
) -> optuna.Study:
    """
    Convenience function to run complete optimization pipeline.

    Args:
        cfg (Config): Global configuration with optuna section
        device (torch.device): PyTorch device for training
        paths (RunPaths): RunPaths instance for output management
        tracker (TrackerProtocol | None): Optional experiment tracker for nested trial logging

    Returns:
        Completed Optuna study with trial results

    Example:
        >>> study = run_optimization(cfg=config, device="cuda", paths=paths)
        >>> print(f"Best AUC: {study.best_value:.3f}")
    """
    orchestrator = OptunaOrchestrator(cfg=cfg, device=device, paths=paths, tracker=tracker)
    return orchestrator.optimize()
