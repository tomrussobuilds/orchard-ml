"""
Study Result Export Functions.

Handles serialization of Optuna study results to various formats:
    - Best trial configuration (YAML)
    - Complete study metadata (JSON)
    - Top K trials comparison (Excel)

All export functions handle edge cases (no completed trials, missing
timestamps) and provide informative logging.
"""

# Standard Imports
import json
import logging
from typing import Dict, List, Optional

# Third-Party Imports
import optuna
import pandas as pd

# Internal Imports
from orchard.core import LOGGER_NAME, Config, RunPaths, log_best_config_export, save_config_as_yaml

# Relative Imports
from .config import map_param_to_config_path
from .utils import get_completed_trials, has_completed_trials

logger = logging.getLogger(LOGGER_NAME)


# CONFIG EXPORT
def export_best_config(study: optuna.Study, cfg: Config, paths: RunPaths) -> None:
    """
    Export best trial configuration as YAML file.

    Creates a new Config instance with best hyperparameters applied,
    validates it, and saves to reports/best_config.yaml.

    Args:
        study: Completed Optuna study with at least one successful trial
        cfg: Template configuration (used for non-optimized parameters)
        paths: RunPaths instance for output location

    Note:
        Skips export with warning if no completed trials exist.

    Example:
        >>> export_best_config(study, cfg, paths)
        # Creates: {paths.reports}/best_config.yaml
    """
    if not has_completed_trials(study):
        logger.warning("No completed trials. Cannot export best config.")
        return

    # Build config dict with best parameters
    config_dict = build_best_config_dict(study.best_params, cfg)

    # Create and validate new config
    best_config = Config(**config_dict)

    # Save to YAML
    output_path = paths.reports / "best_config.yaml"
    save_config_as_yaml(best_config, output_path)
    log_best_config_export(output_path)

    return output_path


def export_study_summary(study: optuna.Study, paths: RunPaths, metric_name: str) -> None:
    """
    Export complete study metadata to JSON.

    Serializes all trials with parameters, values, states, timestamps,
    and durations. Handles studies with zero completed trials gracefully.

    Args:
        study: Optuna study (may contain failed/pruned trials)
        paths: RunPaths instance for output location
        metric_name: Name of optimization metric (for metadata)

    Output Structure:
        {
            "study_name": str,
            "direction": str,
            "n_trials": int,
            "n_completed": int,
            "best_trial": {...} or null,
            "trials": [...]
        }

    Example:
        >>> export_study_summary(study, paths, "auc")
        # Creates: {paths.reports}/study_summary.json
    """
    completed = get_completed_trials(study)

    # Build best trial data (may be None if no completed trials)
    best_trial_data = build_best_trial_data(study, completed)

    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "best_trial": best_trial_data,
        "trials": [build_trial_data(trial) for trial in study.trials],
    }

    output_path = paths.reports / "study_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved study summary to {output_path}")


def export_top_trials(
    study: optuna.Study, paths: RunPaths, metric_name: str, top_k: int = 10
) -> None:
    """
    Export top K trials to Excel spreadsheet.

    Creates human-readable comparison table of best-performing trials
    with hyperparameters, metric values, and durations.

    Args:
        study: Completed Optuna study with at least one successful trial
        paths: RunPaths instance for output location
        metric_name: Name of optimization metric (for column header)
        top_k: Number of top trials to export (default: 10)

    DataFrame Columns:
        - Rank: 1-based ranking
        - Trial: Trial number
        - {METRIC_NAME}: Objective value
        - {param_name}: Each hyperparameter
        - Duration (s): Trial duration if available

    Example:
        >>> export_top_trials(study, paths, "auc", top_k=10)
        # Creates: {paths.reports}/top_10_trials.xlsx
    """
    completed = get_completed_trials(study)
    if not completed:
        logger.warning("No completed trials. Cannot export top trials.")
        return

    # Sort by value (ascending for minimize, descending for maximize)
    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=reverse)[:top_k]

    # Build DataFrame
    df = build_top_trials_dataframe(sorted_trials, metric_name)

    # Save to Excel
    output_path = paths.reports / "top_10_trials.xlsx"
    df.to_excel(output_path, index=False, engine="openpyxl")

    logger.info(f"Saved top {len(sorted_trials)} trials to {output_path}")


# ==================== HELPER FUNCTIONS ====================


def build_best_config_dict(best_params: Dict, cfg: Config) -> Dict:
    """
    Construct config dictionary from best trial parameters.

    Maps Optuna parameters back to Config structure using
    map_param_to_config_path and restores full training epochs.

    Args:
        best_params: Dictionary from study.best_params
        cfg: Template config for structure and defaults

    Returns:
        Config dictionary ready for validation
    """
    config_dict = cfg.model_dump()

    for param_name, value in best_params.items():
        section, key = map_param_to_config_path(param_name)
        config_dict[section][key] = value

    # Restore normal epochs for final training (not Optuna short epochs)
    config_dict["training"]["epochs"] = cfg.training.epochs

    return config_dict


def build_best_trial_data(
    study: optuna.Study, completed: List[optuna.trial.FrozenTrial]
) -> Optional[Dict]:
    """
    Build best trial metadata dictionary.

    Args:
        study: Optuna study instance
        completed: List of completed trials

    Returns:
        Dictionary with best trial info, or None if no completed trials
    """
    if not completed:
        return None

    try:
        best = study.best_trial
        return {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "datetime_start": best.datetime_start.isoformat() if best.datetime_start else None,
            "datetime_complete": (
                best.datetime_complete.isoformat() if best.datetime_complete else None
            ),
        }
    except ValueError:
        # No best trial available
        return None


def build_trial_data(trial: optuna.trial.FrozenTrial) -> Dict:
    """
    Build trial metadata dictionary.

    Handles missing timestamps gracefully and computes duration
    when both start and complete times are available.

    Args:
        trial: Frozen trial from study

    Returns:
        Dictionary with trial information
    """
    duration = None
    if trial.datetime_complete and trial.datetime_start:
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()

    return {
        "number": trial.number,
        "value": trial.value,
        "params": trial.params,
        "state": trial.state.name,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": (
            trial.datetime_complete.isoformat() if trial.datetime_complete else None
        ),
        "duration_seconds": duration,
    }


def build_top_trials_dataframe(
    sorted_trials: List[optuna.trial.FrozenTrial], metric_name: str
) -> pd.DataFrame:
    """
    Build DataFrame from sorted trials.

    Args:
        sorted_trials: List of trials sorted by performance
        metric_name: Name of optimization metric (for column header)

    Returns:
        Pandas DataFrame with trial comparison data
    """
    rows = []
    for rank, trial in enumerate(sorted_trials, 1):
        row = {
            "Rank": rank,
            "Trial": trial.number,
            f"{metric_name.upper()}": trial.value,
        }
        row.update(trial.params)

        # Add duration if available
        if trial.datetime_complete and trial.datetime_start:
            duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
            row["Duration (s)"] = int(duration)

        rows.append(row)

    return pd.DataFrame(rows)
