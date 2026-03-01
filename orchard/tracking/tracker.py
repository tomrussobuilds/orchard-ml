"""
MLflow Tracker Implementation.

Provides MLflowTracker for experiment tracking and NoOpTracker as a silent
fallback when MLflow is unavailable or tracking is disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from ..core import LOGGER_NAME, LogStyle
from ..core.paths import METRIC_ACCURACY, METRIC_AUC, METRIC_F1, METRIC_LOSS

logger = logging.getLogger(LOGGER_NAME)

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys for MLflow params."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TrackerProtocol(Protocol):
    """
    Protocol defining the experiment tracker interface.

    Both MLflowTracker and NoOpTracker implement this protocol,
    enabling type-safe dependency injection without inheritance.
    """

    def start_run(self, cfg: Any, run_name: str, tracking_uri: str) -> None: ...

    def log_epoch(self, epoch: int, train_loss: float, val_metrics: dict, lr: float) -> None: ...

    def log_test_metrics(self, test_acc: float, macro_f1: float) -> None: ...

    def log_artifact(self, path: Path) -> None: ...

    def log_artifacts_dir(self, directory: Path) -> None: ...

    def start_optuna_trial(self, trial_number: int, params: dict[str, Any]) -> None: ...

    def end_optuna_trial(self, best_metric: float) -> None: ...

    def end_run(self) -> None: ...


class NoOpTracker:  # pragma: no cover
    """
    Silent no-op tracker used when MLflow is disabled or unavailable.

    All methods mirror the MLflowTracker interface but perform no operations,
    ensuring zero overhead when tracking is not active.
    """

    def start_run(self, cfg: Any, run_name: str, tracking_uri: str) -> None:  # noqa: ARG002
        """No-op: skip MLflow run initialization."""

    def log_epoch(  # noqa: ARG002
        self, epoch: int, train_loss: float, val_metrics: dict, lr: float
    ) -> None:
        """No-op: skip per-epoch metric logging."""

    def log_test_metrics(self, test_acc: float, macro_f1: float) -> None:  # noqa: ARG002
        """No-op: skip test metric logging."""

    def log_artifact(self, path: Path) -> None:  # noqa: ARG002
        """No-op: skip artifact logging."""

    def log_artifacts_dir(self, directory: Path) -> None:  # noqa: ARG002
        """No-op: skip directory artifact logging."""

    def start_optuna_trial(self, trial_number: int, params: dict[str, Any]) -> None:  # noqa: ARG002
        """No-op: skip nested Optuna trial run creation."""

    def end_optuna_trial(self, best_metric: float) -> None:  # noqa: ARG002
        """No-op: skip nested Optuna trial run closure."""

    def end_run(self) -> None:
        """No-op: skip MLflow run closure."""


class MLflowTracker:  # pragma: no cover
    """
    MLflow-based experiment tracker for Orchard ML runs.

    Manages a single MLflow run lifecycle: parameter logging, per-epoch metrics,
    final test metrics, and artifact collection. Supports nested runs for
    Optuna trial tracking.

    Attributes:
        experiment_name (str): MLflow experiment name.
    """

    def __init__(self, experiment_name: str = "orchard-ml") -> None:
        self.experiment_name = experiment_name
        self._parent_run_id: str | None = None

    def start_run(self, cfg: Any, run_name: str, tracking_uri: str) -> None:
        """
        Start an MLflow run and log all config parameters.

        Args:
            cfg: Config object with dump_serialized() method.
            run_name: Human-readable run name (typically paths.run_id).
            tracking_uri: SQLite URI for local MLflow storage.
        """
        # Temporarily suppress all INFO logs during MLflow/Alembic DB init
        prev_level = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=run_name)
        finally:
            logging.disable(prev_level)
        active_run = mlflow.active_run()
        self._parent_run_id = active_run.info.run_id if active_run else None

        # Log flattened config as params (MLflow has 500-char value limit)
        flat_params = _flatten_dict(cfg.dump_serialized())
        safe_params = {k: str(v)[:500] for k, v in flat_params.items() if v is not None}
        mlflow.log_params(safe_params)

        logger.info(f"  {LogStyle.ARROW} MLflow run started")  # pragma: no mutant

    def log_epoch(self, epoch: int, train_loss: float, val_metrics: dict, lr: float) -> None:
        """
        Log per-epoch training metrics.

        Args:
            epoch: Current epoch number (1-based).
            train_loss: Training loss for this epoch.
            val_metrics: Validation metrics dict with 'loss', 'accuracy', 'auc', 'f1'.
            lr: Current learning rate.
        """
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_metrics[METRIC_LOSS],
                "val_accuracy": val_metrics[METRIC_ACCURACY],
                "val_auc": val_metrics.get(METRIC_AUC, 0.0),
                "val_f1": val_metrics.get(METRIC_F1, 0.0),
                "learning_rate": lr,
            },
            step=epoch,
        )

    def log_test_metrics(self, test_acc: float, macro_f1: float) -> None:
        """
        Log final test set metrics.

        Args:
            test_acc: Test set accuracy.
            macro_f1: Test set macro-averaged F1 score.
        """
        mlflow.log_metrics({"test_accuracy": test_acc, "test_macro_f1": macro_f1})

    def log_artifact(self, path: Path) -> None:
        """
        Log a single file as an MLflow artifact.

        Args:
            path: Path to file to log.
        """
        if path.exists():
            mlflow.log_artifact(str(path))

    def log_artifacts_dir(self, directory: Path) -> None:
        """
        Log all files in a directory as MLflow artifacts.

        Args:
            directory: Directory whose contents to log.
        """
        if directory.exists() and directory.is_dir():
            mlflow.log_artifacts(str(directory))

    def start_optuna_trial(self, trial_number: int, params: dict[str, Any]) -> None:
        """
        Start a nested MLflow run for an Optuna trial.

        Args:
            trial_number: Optuna trial number.
            params: Sampled hyperparameters for this trial.
        """
        mlflow.start_run(
            run_name=f"trial_{trial_number:03d}",
            nested=True,
        )
        safe_params = {k: str(v)[:500] for k, v in params.items()}
        mlflow.log_params(safe_params)

    def end_optuna_trial(self, best_metric: float) -> None:
        """
        End the current nested Optuna trial run.

        Args:
            best_metric: Best validation metric achieved in this trial.
        """
        mlflow.log_metric("best_trial_metric", best_metric)
        mlflow.end_run()

    def end_run(self) -> None:
        """End the active MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info(f"  {LogStyle.ARROW} MLflow run ended")  # pragma: no mutant

    def __enter__(self) -> MLflowTracker:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.end_run()


def create_tracker(cfg: Any) -> TrackerProtocol:
    """
    Factory: returns MLflowTracker if tracking is configured, else NoOpTracker.

    Args:
        cfg: Config object. If cfg.tracking is set and enabled, returns MLflowTracker.

    Returns:
        Active tracker instance.
    """
    tracking_cfg = getattr(cfg, "tracking", None)
    if tracking_cfg is None or not tracking_cfg.enabled:
        return NoOpTracker()

    if not _MLFLOW_AVAILABLE:
        logger.warning(
            "Tracking enabled in config but mlflow is not installed. "
            "Install with: pip install orchard-ml[tracking]"
        )
        return NoOpTracker()

    return MLflowTracker(experiment_name=tracking_cfg.experiment_name)
