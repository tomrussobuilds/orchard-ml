"""
Utility Functions for Study Analysis.

Provides helper functions for working with Optuna studies:

- Extract completed trials
- Check if study has successful completions
- Compute trial statistics

These utilities eliminate code duplication across orchestrator modules.
"""

from __future__ import annotations

import optuna


# HELPER FUNCTIONS
def get_completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """
    Extract all successfully completed trials from study.

    Args:
        study: Optuna study instance

    Returns:
        list of trials with state == TrialState.COMPLETE

    Example:
        >>> completed = get_completed_trials(study)
        >>> len(completed)  # Number of successful trials
    """
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def has_completed_trials(study: optuna.Study) -> bool:
    """
    Check if study has at least one completed trial.

    Args:
        study: Optuna study instance

    Returns:
        True if at least one trial completed successfully

    Example:
        >>> if has_completed_trials(study):
        ...     export_best_config(study, cfg, paths)
    """
    return len(get_completed_trials(study)) > 0
