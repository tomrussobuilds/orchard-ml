"""
Orchard ML Command-Line Interface.

Provides the ``orchard`` entry point for running ML pipelines
from YAML recipes with optional parameter overrides.

Usage:
    orchard run recipes/config_mini_cnn.yaml
    orchard run recipes/optuna_mini_cnn.yaml --set training.epochs=30
"""

from pathlib import Path
from typing import Annotated, Any, Optional

import typer

app = typer.Typer(
    name="orchard",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from importlib.metadata import version as pkg_version

        typer.echo(f"orchard-ml {pkg_version('orchard-ml')}")
        raise typer.Exit()


@app.callback()
def main(
    _: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Orchard ML: Type-Safe Deep Learning for Reproducible Research."""
    ...  # pragma: no cover


# OVERRIDE PARSING UTILITIES
def _auto_cast(value: str) -> Any:
    """
    Cast a CLI string to the appropriate Python scalar type.
    Handles booleans, None, integers, floats, and falls back to string.
    """
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_overrides(raw: list[str]) -> dict[str, Any]:
    """
    Parse ``key.path=value`` strings into a flat override dict.

    Args:
        raw: List of "dotted.key=value" strings from ``--set`` flags

    Returns:
        Dict mapping dotted keys to auto-casted values

    Raises:
        typer.BadParameter: If an item has no ``=`` or an empty key
    """
    overrides: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise typer.BadParameter(f"Override must use key=value format, got: '{item}'")
        key, _, val = item.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"Empty key in override: '{item}'")
        overrides[key] = _auto_cast(val.strip())
    return overrides


@app.command()
def run(
    recipe: Annotated[
        Path,
        typer.Argument(help="Path to YAML recipe file."),
    ],
    set_: Annotated[
        Optional[list[str]],
        typer.Option(
            "--set",
            help="Override config value (repeatable): key.path=value",
        ),
    ] = None,
) -> None:
    """Run the ML pipeline from a YAML recipe."""
    from orchard import (
        MLRUNS_DB,
        Config,
        LogStyle,
        RootOrchestrator,
        create_tracker,
        log_pipeline_summary,
        run_export_phase,
        run_optimization_phase,
        run_training_phase,
    )

    if not recipe.exists():
        typer.echo(f"Error: recipe not found: {recipe}", err=True)
        raise typer.Exit(code=1)

    overrides = _parse_overrides(set_ or [])
    cfg = Config.from_recipe(recipe, overrides=overrides or None)

    with RootOrchestrator(cfg) as orchestrator:
        paths = orchestrator.paths
        run_logger = orchestrator.run_logger
        assert paths is not None and run_logger is not None  # nosec B101

        tracker = create_tracker(cfg)
        tracking_uri = f"sqlite:///{MLRUNS_DB}"
        tracker.start_run(cfg=cfg, run_name=paths.run_id, tracking_uri=tracking_uri)

        training_cfg = cfg

        try:
            # Phase 1: Optimization (if optuna config present)
            if cfg.optuna is not None:
                _, best_config_path = run_optimization_phase(orchestrator, tracker=tracker)
                if best_config_path and best_config_path.exists():
                    training_cfg = Config.from_recipe(best_config_path)
                    run_logger.info(f"Using optimized config: {best_config_path.name}")
            else:
                run_logger.info("Skipping optimization (no optuna config)")

            # Phase 2: Training
            best_model_path, _, _, _, macro_f1, test_acc = run_training_phase(
                orchestrator, cfg=training_cfg, tracker=tracker
            )

            # Phase 3: Export (if export config present)
            onnx_path = None
            if cfg.export is not None:
                onnx_path = run_export_phase(
                    orchestrator,
                    checkpoint_path=best_model_path,
                    cfg=training_cfg,
                    export_format=cfg.export.format,
                    opset_version=cfg.export.opset_version,
                )

            # Log final artifacts
            tracker.log_artifacts_dir(paths.figures)
            tracker.log_artifact(paths.final_report_path)
            tracker.log_artifact(paths.get_config_path())

            log_pipeline_summary(
                test_acc=test_acc,
                macro_f1=macro_f1,
                best_model_path=best_model_path,
                run_dir=paths.root,
                duration=orchestrator.time_tracker.elapsed_formatted,
                onnx_path=onnx_path,
                logger_instance=run_logger,
            )

        except KeyboardInterrupt:
            run_logger.warning(f"{LogStyle.WARNING} Interrupted by user.")
            raise SystemExit(1)

        except Exception as e:
            run_logger.error(f"{LogStyle.WARNING} Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            tracker.end_run()
