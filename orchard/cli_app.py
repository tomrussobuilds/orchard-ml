"""
Orchard ML Command-Line Interface.

Provides the ``orchard`` entry point with two commands:

- ``orchard init`` — generate a starter recipe YAML with all defaults
- ``orchard run``  — execute a training pipeline from a YAML recipe

Usage:
    orchard init
    orchard run recipes/config_mini_cnn.yaml
    orchard run recipes/optuna_mini_cnn.yaml --set training.epochs=30
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    name="orchard",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# ── App callback ────────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from importlib.metadata import version as pkg_version

        typer.echo(f"orchard-ml {pkg_version('orchard-ml')}")
        raise typer.Exit()


@app.callback()
def main(
    _: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Orchard ML: type-Safe Deep Learning for Reproducible Research."""
    ...  # pragma: no cover


# ── Commands ────────────────────────────────────────────────────────────────


@app.command()
def init(
    output: Annotated[
        Path,
        typer.Argument(help="Output YAML file path."),
    ] = Path("recipe.yaml"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing file."),
    ] = False,
) -> None:
    """Generate a starter recipe with all config fields and defaults."""
    import yaml

    if output.exists() and not force:
        typer.echo(f"Error: '{output}' already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)

    data = _build_init_dict()
    yaml_body = yaml.dump(
        data, default_flow_style=False, sort_keys=False, indent=4, allow_unicode=True
    )
    content = _INIT_HEADER.format(filename=output.name) + yaml_body

    output.write_text(content, encoding="utf-8")
    typer.echo(f"Recipe created: {output}")
    typer.echo(f"Run it with:   orchard run {output}")


@app.command()
def run(
    recipe: Annotated[
        Path,
        typer.Argument(help="Path to YAML recipe file."),
    ],
    set_: Annotated[
        list[str] | None,
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

        orchestrator.log_environment_report()

        training_cfg = cfg

        try:
            # Phase 1: Optimization (if optuna config present)
            if cfg.optuna is not None:
                _, best_config_path = run_optimization_phase(orchestrator, tracker=tracker)
                if best_config_path and best_config_path.exists():
                    training_cfg = Config.from_recipe(best_config_path)
                    run_logger.info(
                        f"{LogStyle.INDENT}{LogStyle.ARROW} {'Optimized Config':<18}: "
                        f"{best_config_path.name}"
                    )
            else:
                run_logger.info(
                    f"{LogStyle.INDENT}{LogStyle.ARROW} Skipping optimization (no optuna config)"
                )

            # Phase 2: Training
            result = run_training_phase(orchestrator, cfg=training_cfg, tracker=tracker)

            # Phase 3: Export (if export config present)
            onnx_path = None
            if cfg.export is not None:
                onnx_path = run_export_phase(
                    orchestrator,
                    checkpoint_path=result.best_model_path,
                    cfg=training_cfg,
                )

            # Log final artifacts
            tracker.log_artifacts_dir(paths.figures)
            tracker.log_artifact(paths.final_report_path)
            tracker.log_artifact(paths.get_config_path())

            log_pipeline_summary(
                test_acc=result.test_acc,
                macro_f1=result.macro_f1,
                test_auc=result.test_auc,
                best_model_path=result.best_model_path,
                run_dir=paths.root,
                duration=orchestrator.time_tracker.elapsed_formatted,
                onnx_path=onnx_path,
                logger_instance=run_logger,
            )

        except KeyboardInterrupt:
            run_logger.warning(f"{LogStyle.WARNING} Interrupted by user.")
            raise SystemExit(1)

        except Exception as e:  # top-level catch-all for logging; re-raises
            run_logger.error(f"{LogStyle.WARNING} Pipeline failed: {e}", exc_info=True)
            raise

        finally:
            tracker.end_run()


# ── Private helpers ─────────────────────────────────────────────────────────

_INIT_HEADER = """\
# ==============================================================================
# Orchard ML — Starter Recipe (generated by `orchard init`)
# ==============================================================================
# Usage:   orchard run {filename}
# Docs:    https://github.com/tomrussobuilds/orchard-ml
#
# Edit the values you need. Remove optional sections to disable them.
# Optional sections: tracking, export, optuna
# ==============================================================================

"""


def _auto_cast(value: str) -> Any:
    """
    Cast a CLI string to the appropriate Python scalar type.

    Args:
        value: Raw string from the command line.

    Returns:
        Converted bool, None, int, float, or the original string.
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
        raw: list of "dotted.key=value" strings from ``--set`` flags.

    Returns:
        dict mapping dotted keys to auto-casted values.

    Raises:
        typer.BadParameter: If an item has no ``=`` or an empty key.
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


def _build_init_dict() -> dict[str, Any]:
    """
    Build a complete config dict with all defaults for recipe generation.

    Returns:
        Ordered dict with every config section dumped via ``model_dump(mode="json")``,
        paths sanitized to portable relative strings, and device reset to ``"auto"``.
    """
    from orchard.core.config import (
        ArchitectureConfig,
        AugmentationConfig,
        DatasetConfig,
        EvaluationConfig,
        ExportConfig,
        HardwareConfig,
        OptunaConfig,
        TelemetryConfig,
        TrackingConfig,
        TrainingConfig,
    )

    dump = lambda m: m.model_dump(mode="json")  # noqa: E731

    ds = dump(DatasetConfig())
    ds.pop("metadata", None)
    ds.pop("img_size", None)
    ds["data_root"] = "./dataset"

    hw = dump(HardwareConfig())
    hw["device"] = "auto"

    tel = dump(TelemetryConfig())
    tel["data_dir"] = "./dataset"
    tel["output_dir"] = "./outputs"

    return {
        "dataset": ds,
        "architecture": dump(ArchitectureConfig()),
        "training": dump(TrainingConfig()),
        "augmentation": dump(AugmentationConfig()),
        "hardware": hw,
        "telemetry": tel,
        "evaluation": dump(EvaluationConfig()),
        "tracking": dump(TrackingConfig()),
        "export": dump(ExportConfig()),
        "optuna": dump(OptunaConfig()),
    }
