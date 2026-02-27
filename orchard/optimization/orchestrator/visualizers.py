"""
Optuna Visualization Generation.

Creates interactive Plotly visualizations for study analysis:

- Optimization history (metric over trials)
- Parameter importance (feature importance for hyperparameters)
- Slice plots (1D parameter effects)
- Parallel coordinate plot (multi-dimensional view)

All functions handle missing dependencies (plotly) and plot generation
failures gracefully with informative logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import optuna

from ...core import LOGGER_NAME, LogStyle
from .utils import has_completed_trials

logger = logging.getLogger(LOGGER_NAME)


# VISUALIZATION GENERATION
def generate_visualizations(study: optuna.Study, output_dir: Path) -> None:
    """
    Generate and save all Optuna visualization plots.

    Creates interactive HTML plots for study analysis. Skips generation
    if no completed trials exist or if plotly is not installed.

    Args:
        study: Completed Optuna study with at least one successful trial
        output_dir: Directory to save HTML plot files (typically paths.figures)

    Generated plots:

    - ``optimization_history.html``: Metric progression over trials
    - ``param_importances.html``: Hyperparameter importance ranking
    - ``slice.html``: Individual parameter effects
    - ``parallel_coordinate.html``: Multi-dimensional parameter view

    Note:
        Requires plotly installation. Logs warning if not available.

    Example:
        >>> generate_visualizations(study, paths.figures)
        # Creates 4 .html files in paths.figures/
    """
    if not has_completed_trials(study):
        logger.warning("No completed trials. Skipping visualizations.")
        return

    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )

        plots = {
            "optimization_history": plot_optimization_history,
            "param_importances": plot_param_importances,
            "slice": plot_slice,
            "parallel_coordinate": plot_parallel_coordinate,
        }

        for plot_name, plot_fn in plots.items():
            save_plot(study, plot_name, plot_fn, output_dir)  # type: ignore[arg-type]

    except ImportError:
        logger.warning(
            "plotly not installed. Skipping visualization generation. "
            "Install with: pip install plotly"
        )


def save_plot(
    study: optuna.Study, plot_name: str, plot_fn: Callable[..., Any], output_dir: Path
) -> None:
    """
    Save a single Optuna visualization plot.

    Wraps plot generation in exception handling to prevent failures
    from blocking the optimization pipeline.

    Args:
        study: Optuna study instance
        plot_name: Human-readable plot name (for logging)
        plot_fn: Optuna plotting function (e.g., plot_optimization_history)
        output_dir (Path): Directory for output HTML files

    Example:
        >>> from optuna.visualization import plot_optimization_history
        >>> save_plot(study, "history", plot_optimization_history, Path("./figures"))
    """
    try:
        fig = plot_fn(study)
        output_path = output_dir / f"{plot_name}.html"
        fig.write_html(str(output_path))
        logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {plot_name:<22}: {output_path.name}")
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Failed to generate {plot_name}: {e}")
