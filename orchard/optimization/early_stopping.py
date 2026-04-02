"""
Early Stopping Callback for Optuna Studies.

Provides ``StudyEarlyStoppingCallback`` and its factory
``get_early_stopping_callback``, which terminate an Optuna study when a
user-defined (or metric-specific default) performance threshold is
sustained for a configurable number of consecutive trials.

Key Functions:
    ``get_early_stopping_callback``: Factory that resolves sensible
        default thresholds for common metrics (AUC, accuracy, F1,
        loss, MAE, MSE) and returns a configured callback.

Key Components:
    ``StudyEarlyStoppingCallback``: Optuna callback that tracks
        consecutive threshold hits and calls ``study.stop()`` once
        ``patience`` is reached. Direction-aware (maximize/minimize).

Example:
    >>> callback = get_early_stopping_callback("auc", "maximize")
    >>> study.optimize(objective, callbacks=[callback])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from ..core import LOGGER_NAME, LogStyle, Reporter

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

logger = logging.getLogger(LOGGER_NAME)


# EARLY STOPPING CALLBACK
class StudyEarlyStoppingCallback:
    """
    Callback to stop Optuna study when target metric is achieved.

    Prevents wasteful computation when near-perfect performance is reached
    (e.g., AUC > 0.9999).

    Example:
        callback = StudyEarlyStoppingCallback(
            threshold=0.9999,
            direction="maximize",
            patience=3
        )
        study.optimize(objective, callbacks=[callback])

    Attributes:
        threshold (float): Metric value that triggers early stopping.
        direction (str): "maximize" or "minimize".
        patience (int): Number of trials meeting threshold before stopping.
        _count (int): Internal counter for consecutive threshold hits.
    """

    def __init__(  # pragma: no mutate (defaults: trampoline can't intercept)
        self, threshold: float, direction: str = "maximize", patience: int = 2, enabled: bool = True
    ) -> None:
        """
        Initialize early stopping callback.

        Args:
            threshold: Target metric value (e.g., 0.9999 for AUC)
            direction: "maximize" or "minimize" (should match study direction)
            patience: Number of consecutive trials meeting threshold before stop
            enabled: Whether callback is active (allows runtime disable)
        """
        self.threshold = threshold
        self.direction = direction
        self.patience = patience
        self.enabled = enabled
        self._count = 0

        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got '{direction}'")

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """
        Callback invoked after each trial completion.

        Args:
            study: Optuna study instance
            trial: Completed trial

        Side Effects:
            Calls ``study.stop()`` when early stopping criteria are met.
        """
        if not self.enabled:
            return

        if trial.state != TrialState.COMPLETE:
            self._count = 0
            return

        value = trial.value
        if value is None:
            self._count = 0
            return
        threshold_met = (
            value >= self.threshold if self.direction == "maximize" else value <= self.threshold
        )

        if not threshold_met:
            self._count = 0
            return

        # Threshold met
        self._count += 1
        cmp = "≥" if self.direction == "maximize" else "≤"  # pragma: no mutate
        logger.info(
            "%s%s Trial %d reached threshold (%.6f %s %.6f) [%d/%d]",
            LogStyle.INDENT,
            LogStyle.SUCCESS,
            trial.number,
            value,
            cmp,
            self.threshold,
            self._count,
            self.patience,
        )

        if self._count < self.patience:
            return

        # Calculate trials saved
        total_trials = study.user_attrs.get("n_trials")
        trials_saved: int | str
        if isinstance(total_trials, int):
            trials_saved = total_trials - (trial.number + 1)
        else:
            trials_saved = "N/A"

        # Use LogStyle for consistent formatting
        Reporter.log_phase_header(
            logger,
            "EARLY STOPPING: Target performance achieved!",
            LogStyle.DOUBLE,
        )
        logger.info(
            "%s%s Metric           : %.6f",
            LogStyle.INDENT,
            LogStyle.SUCCESS,
            value,
        )
        logger.info(
            "%s%s Threshold        : %.6f",
            LogStyle.INDENT,
            LogStyle.ARROW,
            self.threshold,
        )
        logger.info(
            "%s%s Trials completed : %d",
            LogStyle.INDENT,
            LogStyle.ARROW,
            trial.number + 1,
        )
        logger.info(
            "%s%s Trials saved     : %s",
            LogStyle.INDENT,
            LogStyle.SUCCESS,
            trials_saved,
        )
        logger.info(LogStyle.DOUBLE)
        logger.info("")

        study.stop()


# CONFIGURATION HELPER
def get_early_stopping_callback(  # pragma: no mutate (defaults: trampoline can't intercept)
    metric_name: str,
    direction: str,
    threshold: float | None = None,
    patience: int = 2,
    enabled: bool = True,
    task_thresholds: Mapping[str, float] | None = None,
) -> StudyEarlyStoppingCallback | None:
    """
    Factory function to create appropriate early stopping callback.

    Lookup order: explicit *threshold* → *task_thresholds* from the task
    registry → ``None`` (disabled with warning).

    Args:
        metric_name: Name of metric being optimized (e.g., "auc", "accuracy")
        direction: "maximize" or "minimize"
        threshold: Custom threshold (if None, uses task default)
        patience: Trials meeting threshold before stopping
        enabled: Whether callback is active
        task_thresholds: Task-specific metric→threshold mapping from the
            task registry.

    Returns:
        Configured callback or None if disabled
    """
    if not enabled:
        return None

    if threshold is None:
        if task_thresholds is not None:
            threshold = task_thresholds.get(metric_name.lower())

        if threshold is None:
            logger.warning(
                "No default threshold for metric '%s'. "
                "Early stopping disabled. set threshold manually to enable.",
                metric_name,
            )
            return None

    return StudyEarlyStoppingCallback(  # pragma: no mutate (trampoline can't intercept kwargs)
        threshold=threshold, direction=direction, patience=patience, enabled=enabled
    )
