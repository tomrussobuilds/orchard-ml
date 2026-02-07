"""
Early Stopping Callback for Optuna Studies.

Terminates optimization when a satisfactory metric threshold is reached,
preventing wasteful computation when near-perfect performance is achieved.
"""

import logging
from typing import Optional

from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

from orchard.core import LOGGER_NAME, LogStyle

logger = logging.getLogger(LOGGER_NAME)


# EARLY STOPPING CALLBACK
class StudyEarlyStoppingCallback:
    """
    Callback to stop Optuna study when target metric is achieved.

    Prevents wasteful computation when near-perfect performance is reached
    (e.g., AUC > 0.9999 for classification tasks).

    Usage:
        callback = StudyEarlyStoppingCallback(
            threshold=0.9999,
            direction="maximize",
            patience=3
        )
        study.optimize(objective, callbacks=[callback])

    Attributes:
        threshold: Metric value that triggers early stopping
        direction: "maximize" or "minimize"
        patience: Number of trials meeting threshold before stopping
        _count: Internal counter for consecutive threshold hits
    """

    def __init__(
        self, threshold: float, direction: str = "maximize", patience: int = 2, enabled: bool = True
    ):
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

        Raises:
            optuna.TrialPruned: Signals study termination
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
        logger.info(
            f"Trial {trial.number} reached threshold "
            f"({value:.6f} "
            f"{'≥' if self.direction == 'maximize' else '≤'} "
            f"{self.threshold:.6f}) "
            f"[{self._count}/{self.patience}]"
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
        logger.info("")
        logger.info(LogStyle.DOUBLE)
        logger.info(f"{'EARLY STOPPING: Target performance achieved!':^80}")
        logger.info(LogStyle.DOUBLE)
        logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Metric           : {value:.6f}")
        logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Threshold        : {self.threshold:.6f}")
        logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Trials completed : {trial.number + 1}")
        logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Trials saved     : {trials_saved}")
        logger.info(LogStyle.DOUBLE)
        logger.info("")

        study.stop()


# CONFIGURATION HELPER
def get_early_stopping_callback(
    metric_name: str,
    direction: str,
    threshold: Optional[float] = None,
    patience: int = 2,
    enabled: bool = True,
) -> Optional[StudyEarlyStoppingCallback]:
    """
    Factory function to create appropriate early stopping callback.

    Provides sensible defaults for common metrics.

    Args:
        metric_name: Name of metric being optimized (e.g., "auc", "accuracy")
        direction: "maximize" or "minimize"
        threshold: Custom threshold (if None, uses metric-specific default)
        patience: Trials meeting threshold before stopping
        enabled: Whether callback is active

    Returns:
        Configured callback or None if disabled
    """
    if not enabled:
        return None

    # Default thresholds for common metrics
    DEFAULT_THRESHOLDS = {
        "maximize": {
            "auc": 0.9999,  # Near-perfect AUC
            "accuracy": 0.995,  # 99.5% accuracy
            "f1": 0.98,  # 98% F1 score
        },
        "minimize": {
            "loss": 0.01,  # Very low loss
            "mae": 0.01,  # Mean absolute error
            "mse": 0.001,  # Mean squared error
        },
    }

    if threshold is None:
        threshold = DEFAULT_THRESHOLDS.get(direction, {}).get(metric_name.lower(), None)

        if threshold is None:
            logger.warning(
                f"No default threshold for metric '{metric_name}'. "
                f"Early stopping disabled. Set threshold manually to enable."
            )
            return None

    logger.info(
        f"Early stopping enabled: "
        f"{metric_name} {'≥' if direction == 'maximize' else '≤'} {threshold:.6f} "
        f"(patience={patience})"
    )

    return StudyEarlyStoppingCallback(
        threshold=threshold, direction=direction, patience=patience, enabled=enabled
    )
