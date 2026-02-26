"""
Optuna Study Orchestrator Package.

High-level coordination for hyperparameter optimization studies.
Provides a modular architecture for study creation, execution,
visualization, and result export.

Key Components:
    ``OptunaOrchestrator``: Primary study lifecycle manager.
    ``run_optimization``: Convenience function for full pipeline
        execution (study creation → trial loop → artifact export).
    ``export_best_config``, ``export_study_summary``,
        ``export_top_trials``: Post-study artifact generators.

Typical Usage:
    >>> from orchard.optimization.orchestrator import run_optimization
    >>> study = run_optimization(cfg=config, device=device, paths=paths)
"""

from .exporters import export_best_config, export_study_summary, export_top_trials
from .orchestrator import OptunaOrchestrator, run_optimization

__all__ = [
    "OptunaOrchestrator",
    "run_optimization",
    "export_best_config",
    "export_study_summary",
    "export_top_trials",
]
