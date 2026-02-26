"""
Pipeline Orchestration Module.

Provides reusable phase functions for the complete ML lifecycle:

- Optimization Phase: Optuna hyperparameter search
- Training Phase: Model training with best parameters
- Export Phase: ONNX model export

These functions are designed to work with a shared RootOrchestrator,
enabling unified artifact management and logging across all phases.

Example:
    >>> from orchard.pipeline import (
    ...     run_optimization_phase,
    ...     run_training_phase,
    ...     run_export_phase,
    ... )
    >>> with RootOrchestrator(cfg) as orch:
    ...     study = run_optimization_phase(orch)
    ...     best_path, losses, metrics = run_training_phase(orch)
    ...     onnx_path = run_export_phase(orch, best_path)
"""

from .phases import (
    TrainingResult,
    run_export_phase,
    run_optimization_phase,
    run_training_phase,
)

__all__ = [
    "TrainingResult",
    "run_optimization_phase",
    "run_training_phase",
    "run_export_phase",
]
