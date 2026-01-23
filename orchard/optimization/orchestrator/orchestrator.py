"""
OptunaOrchestrator Core Implementation.

Primary orchestration class that coordinates study lifecycle:
creation, optimization, and post-processing. Delegates specific
tasks to specialized modules (builders, exporters, visualizers).

This module contains only high-level coordination logic. All
implementation details are delegated to focused submodules.
"""

# Standard Imports
import logging

# Third-Party Imports
import optuna

# Internal Imports
from orchard.core import (
    LOGGER_NAME,
    Config,
    LogStyle,
    RunPaths,
    log_optimization_header,
    log_study_summary,
)

# Relative Imports
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
    High-level manager for Optuna optimization studies.

    Coordinates study creation, execution, and artifact generation
    while integrating with existing Config and logging infrastructure.

    Attributes:
        cfg: Template configuration for trials
        device: Computation device (CPU/CUDA/MPS)
        paths: RunPaths instance for output management

    Methods:
        create_study: Create or load Optuna study
        optimize: Execute full optimization pipeline
    """

    def __init__(self, cfg: Config, device, paths: RunPaths):
        """Initialize orchestrator.

        Args:
            cfg: Base Config to override per trial
            device: PyTorch device
            paths: Root directory for outputs
        """
        self.cfg = cfg
        self.device = device
        self.paths = paths

        logger.info(f"OptunaOrchestrator initialized. Output: {paths.root}")

    def create_study(self) -> optuna.Study:
        """Create or load Optuna study with configured sampler and pruner.

        Returns:
            Configured Optuna study instance
        """
        # Pass cfg to builders
        sampler = build_sampler(self.cfg.optuna.sampler_type, self.cfg)
        pruner = build_pruner(self.cfg.optuna.enable_pruning, self.cfg.optuna.pruner_type, self.cfg)
        storage_url = self.cfg.optuna.get_storage_url(self.paths)

        logger.info(LogStyle.DOUBLE)
        logger.info(f"Starting Optuna optimization: {self.cfg.optuna.n_trials} trials")
        logger.info(LogStyle.DOUBLE)

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
        study = self.create_study()
        search_space = get_search_space(
            self.cfg.optuna.search_space_preset, resolution=self.cfg.dataset.resolution
        )

        objective = OptunaObjective(
            cfg=self.cfg,
            search_space=search_space,
            device=self.device,
        )

        # Configure logging and callbacks
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        log_optimization_header(self.cfg)

        # Pass cfg to build_callbacks
        callbacks = build_callbacks(self.cfg)

        # Run optimization with error handling
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
            logger.warning("Optimization interrupted by user. Saving partial results...")

        # Post-optimization processing
        self._post_optimization_processing(study)

        return study

    def _post_optimization_processing(self, study: optuna.Study) -> None:
        """Execute all post-optimization tasks.

        Args:
            study: Completed Optuna study
        """
        # Always log summary (works even without completed trials)
        log_study_summary(study, self.cfg.optuna.metric_name)

        # Only proceed with exports if we have completed trials
        if not has_completed_trials(study):
            logger.warning(
                "No completed trials. Skipping visualizations, best config, "
                "and detailed exports."
            )
            # Still export study summary (contains metadata about failed trials)
            export_study_summary(study, self.paths, self.cfg.optuna.metric_name)
            return

        # Generate artifacts for successful study
        if self.cfg.optuna.save_plots:
            generate_visualizations(study, self.paths.figures)

        if self.cfg.optuna.save_best_config:
            export_best_config(study, self.cfg, self.paths)

        export_study_summary(study, self.paths, self.cfg.optuna.metric_name)
        export_top_trials(study, self.paths, self.cfg.optuna.metric_name)


def run_optimization(cfg: Config, device, paths: RunPaths) -> optuna.Study:
    """
    Convenience function to run complete optimization pipeline.

    Args:
        cfg: Global configuration with optuna section
        device: PyTorch device for training
        paths: RunPaths instance for output management

    Returns:
        Completed Optuna study with trial results

    Example:
        >>> study = run_optimization(cfg=config, device="cuda", paths=paths)
        >>> print(f"Best AUC: {study.best_value:.3f}")
    """
    orchestrator = OptunaOrchestrator(cfg=cfg, device=device, paths=paths)
    return orchestrator.optimize()
