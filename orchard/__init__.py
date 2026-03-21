"""
Orchard ML: type-Safe Deep Learning for Reproducible Research.

Top-level convenience API re-exporting the most commonly used components
from subpackages, so users and the ``orchard`` CLI can write:

    from orchard import Config, RootOrchestrator, get_model
"""

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("orchard-ml")

from .architectures import get_model
from .core import Config, LogStyle, RootOrchestrator, log_pipeline_summary
from .core.paths import (
    METRIC_ACCURACY,
    METRIC_AUC,
    METRIC_F1,
    METRIC_LOSS,
    METRIC_MAP,
    METRIC_MAP_50,
    METRIC_MAP_75,
    MLRUNS_DB,
)
from .core.task_registry import TaskComponents, register_task
from .exceptions import (
    OrchardConfigError,
    OrchardDatasetError,
    OrchardDeviceError,
    OrchardError,
    OrchardExportError,
    OrchardInfrastructureError,
)
from .pipeline import run_export_phase, run_optimization_phase, run_training_phase
from .tasks import (
    ClassificationCriterionAdapter,
    ClassificationEvalPipelineAdapter,
    ClassificationMetricsAdapter,
    ClassificationTrainingStepAdapter,
    DetectionCriterionAdapter,
    DetectionEvalPipelineAdapter,
    DetectionMetricsAdapter,
    DetectionTrainingStepAdapter,
)
from .tracking import create_tracker

# ── Task Registration ─────────────────────────────────────────────────────
_CLASSIFICATION_FALLBACK = {
    METRIC_LOSS: 999.0,
    METRIC_ACCURACY: 0.0,
    METRIC_AUC: 0.0,
    METRIC_F1: 0.0,
}

_CLASSIFICATION_EARLY_STOP = {
    METRIC_AUC: 0.9999,
    METRIC_ACCURACY: 0.995,
    METRIC_F1: 0.98,
    METRIC_LOSS: 0.01,
}

register_task(
    "classification",
    TaskComponents(
        criterion_factory=ClassificationCriterionAdapter(),
        training_step=ClassificationTrainingStepAdapter(),
        validation_metrics=ClassificationMetricsAdapter(),
        eval_pipeline=ClassificationEvalPipelineAdapter(),
        fallback_metrics=_CLASSIFICATION_FALLBACK,
        early_stopping_thresholds=_CLASSIFICATION_EARLY_STOP,
    ),
)

_DETECTION_FALLBACK = {
    METRIC_LOSS: 999.0,
    METRIC_MAP: 0.0,
    METRIC_MAP_50: 0.0,
    METRIC_MAP_75: 0.0,
}

_DETECTION_EARLY_STOP = {
    METRIC_MAP: 0.85,
    METRIC_MAP_50: 0.95,
    METRIC_MAP_75: 0.80,
    METRIC_LOSS: 0.01,
}

register_task(
    "detection",
    TaskComponents(
        criterion_factory=DetectionCriterionAdapter(),
        training_step=DetectionTrainingStepAdapter(),
        validation_metrics=DetectionMetricsAdapter(),
        eval_pipeline=DetectionEvalPipelineAdapter(),
        fallback_metrics=_DETECTION_FALLBACK,
        early_stopping_thresholds=_DETECTION_EARLY_STOP,
    ),
)

__all__ = [
    "__version__",
    # Core
    "Config",
    "LogStyle",
    "RootOrchestrator",
    "log_pipeline_summary",
    # Paths
    "MLRUNS_DB",
    # Architectures
    "get_model",
    # Pipeline
    "run_export_phase",
    "run_optimization_phase",
    "run_training_phase",
    # Tracking
    "create_tracker",
    # Exceptions
    "OrchardError",
    "OrchardConfigError",
    "OrchardDatasetError",
    "OrchardDeviceError",
    "OrchardExportError",
    "OrchardInfrastructureError",
]
