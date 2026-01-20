"""
Optuna Hyperparameter Optimization Module.

Provides components for automated hyperparameter search:
    - SearchSpaceRegistry: Predefined search distributions
    - OptunaObjective: Objective function for optimization
    - OptunaOrchestrator: Study lifecycle management
    - run_optimization: Convenience function for complete workflow
"""

# =========================================================================== #
#                             Internal Imports                                #
# =========================================================================== #
from .early_stopping import StudyEarlyStoppingCallback, get_early_stopping_callback
from .objective import MetricExtractor, OptunaObjective, TrialConfigBuilder, TrialTrainingExecutor
from .orchestrator import OptunaOrchestrator, run_optimization
from .search_spaces import SearchSpaceRegistry, get_search_space

# =========================================================================== #
#                              Public API                                     #
# =========================================================================== #
__all__ = [
    "SearchSpaceRegistry",
    "get_search_space",
    "OptunaObjective",
    "TrialTrainingExecutor",
    "MetricExtractor",
    "TrialConfigBuilder",
    "OptunaOrchestrator",
    "run_optimization",
    "StudyEarlyStoppingCallback",
    "get_early_stopping_callback",
]
