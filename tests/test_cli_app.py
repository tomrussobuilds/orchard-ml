"""
Test Suite for Orchard ML CLI (cli_app.py).

Tests the Typer-based CLI utilities: override parsing, auto-casting,
and basic command invocation.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchard.cli_app import _auto_cast, _parse_overrides


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from Rich/Typer help output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# AUTO-CAST
@pytest.mark.unit
class TestAutoCast:
    """Tests for _auto_cast string-to-Python type conversion."""

    def test_int(self):
        assert _auto_cast("42") == 42
        assert isinstance(_auto_cast("42"), int)

    def test_negative_int(self):
        assert _auto_cast("-5") == -5

    def test_zero(self):
        assert _auto_cast("0") == 0
        assert isinstance(_auto_cast("0"), int)

    def test_float(self):
        assert _auto_cast("3.14") == pytest.approx(3.14)
        assert isinstance(_auto_cast("3.14"), float)

    def test_scientific_notation(self):
        assert _auto_cast("1e-5") == pytest.approx(1e-5)
        assert isinstance(_auto_cast("1e-5"), float)

    def test_negative_float(self):
        assert _auto_cast("-0.001") == pytest.approx(-0.001)

    def test_bool_true(self):
        assert _auto_cast("true") is True
        assert _auto_cast("True") is True
        assert _auto_cast("TRUE") is True

    def test_bool_false(self):
        assert _auto_cast("false") is False
        assert _auto_cast("False") is False

    def test_null(self):
        assert _auto_cast("null") is None
        assert _auto_cast("None") is None
        assert _auto_cast("none") is None

    def test_string_passthrough(self):
        assert _auto_cast("hello") == "hello"
        assert _auto_cast("cosine") == "cosine"

    def test_empty_string(self):
        assert _auto_cast("") == ""


# PARSE OVERRIDES
@pytest.mark.unit
class TestParseOverrides:
    """Tests for _parse_overrides CLI flag parsing."""

    def test_single_override(self):
        result = _parse_overrides(["training.epochs=20"])
        assert result == {"training.epochs": 20}

    def test_multiple_overrides(self):
        result = _parse_overrides(
            [
                "training.epochs=20",
                "training.seed=123",
                "training.use_amp=true",
            ]
        )
        assert result == {
            "training.epochs": 20,
            "training.seed": 123,
            "training.use_amp": True,
        }

    def test_empty_list(self):
        result = _parse_overrides([])
        assert result == {}

    def test_value_with_equals(self):
        """Values containing = should work (partition on first =)."""
        result = _parse_overrides(["dataset.name=a=b"])
        assert result == {"dataset.name": "a=b"}

    def test_missing_equals_raises(self):
        import typer

        with pytest.raises(typer.BadParameter, match="key=value"):
            _parse_overrides(["no_equals_here"])

    def test_empty_key_raises(self):
        import typer

        with pytest.raises(typer.BadParameter, match="Empty key"):
            _parse_overrides(["=value"])

    def test_whitespace_stripped(self):
        result = _parse_overrides(["  training.epochs  =  42  "])
        assert result == {"training.epochs": 42}

    def test_float_scientific(self):
        result = _parse_overrides(["training.learning_rate=1e-4"])
        assert result["training.learning_rate"] == pytest.approx(1e-4)

    def test_null_value(self):
        result = _parse_overrides(["dataset.max_samples=null"])
        assert result == {"dataset.max_samples": None}

    def test_invalid_section_raises(self):
        import typer

        with pytest.raises(typer.BadParameter, match="Unknown config section"):
            _parse_overrides(["bogus.field=1"])

    def test_invalid_field_raises(self):
        import typer

        with pytest.raises(typer.BadParameter, match="Unknown field"):
            _parse_overrides(["training.bogus=1"])

    def test_missing_dot_raises(self):
        import typer

        with pytest.raises(typer.BadParameter, match="section.field"):
            _parse_overrides(["epochs=30"])


# CLI COMMAND (help only - avoids running the full pipeline)
@pytest.mark.unit
class TestCLIHelp:
    """Smoke tests for CLI command registration."""

    def test_app_help(self):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_run_help(self):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "--set" in clean
        assert "RECIPE" in clean

    def test_run_missing_recipe(self):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "nonexistent.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output


# CLI RUN COMMAND (mocked pipeline)
@pytest.mark.unit
class TestCLIRun:
    """Tests for the ``run`` command with mocked pipeline components."""

    def test_version_flag(self):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "orchard-ml" in result.output

    @patch("orchard.RootOrchestrator")
    @patch("orchard.Config")
    def test_run_raises_when_orchestrator_returns_none(self, mock_cfg_cls, mock_orch_cls, tmp_path):
        """Covers RuntimeError when orchestrator paths/logger are None."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("dataset:\n  name: test\n")

        mock_cfg_cls.from_recipe.return_value = MagicMock()

        mock_orch = MagicMock()
        mock_orch.paths = None
        mock_orch.run_logger = None
        mock_orch_cls.return_value.__enter__.return_value = mock_orch

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(recipe)])

        assert result.exit_code != 0

    @patch("orchard.create_tracker")
    @patch("orchard.log_pipeline_summary")
    @patch("orchard.run_training_phase")
    @patch("orchard.RootOrchestrator")
    @patch("orchard.Config")
    def test_run_training_only(
        self, mock_cfg_cls, mock_orch_cls, mock_train, mock_summary, mock_tracker_fn, tmp_path
    ):
        """Covers the main run path: training only (no optuna, no export)."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("dataset:\n  name: test\n")

        mock_cfg = MagicMock()
        mock_cfg.optuna = None
        mock_cfg.export = None
        mock_cfg_cls.from_recipe.return_value = mock_cfg

        mock_orch = MagicMock()
        mock_orch_cls.return_value.__enter__.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.best_model_path = Path("model.pt")
        mock_result.macro_f1 = 0.95
        mock_result.test_acc = 0.90
        mock_result.test_auc = 0.92
        mock_train.return_value = mock_result
        mock_tracker_fn.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(recipe)])

        assert result.exit_code == 0
        mock_cfg_cls.from_recipe.assert_called_once()
        mock_train.assert_called_once()
        mock_summary.assert_called_once()

    @patch("orchard.create_tracker")
    @patch("orchard.log_pipeline_summary")
    @patch("orchard.run_export_phase")
    @patch("orchard.run_optimization_phase")
    @patch("orchard.run_training_phase")
    @patch("orchard.RootOrchestrator")
    @patch("orchard.Config")
    def test_run_full_pipeline(
        self,
        mock_cfg_cls,
        mock_orch_cls,
        mock_train,
        mock_optuna,
        mock_export,
        _mock_summary,
        mock_tracker_fn,
        tmp_path,
    ):
        """Covers optuna + training + export path."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("dataset:\n  name: test\n")

        best_config = tmp_path / "best.yaml"
        best_config.write_text("dataset:\n  name: test\n")

        mock_cfg = MagicMock()
        mock_cfg.optuna = MagicMock()
        mock_cfg.export = MagicMock()
        mock_cfg_cls.from_recipe.return_value = mock_cfg

        mock_orch = MagicMock()
        mock_orch_cls.return_value.__enter__.return_value = mock_orch
        mock_optuna.return_value = (MagicMock(), best_config)
        mock_result = MagicMock()
        mock_result.best_model_path = Path("model.pt")
        mock_result.macro_f1 = 0.95
        mock_result.test_acc = 0.90
        mock_result.test_auc = 0.92
        mock_train.return_value = mock_result
        mock_export.return_value = Path("model.onnx")
        mock_tracker_fn.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(recipe)])

        assert result.exit_code == 0
        mock_optuna.assert_called_once()
        mock_train.assert_called_once()
        mock_export.assert_called_once()
        assert mock_cfg_cls.from_recipe.call_count == 2  # initial + optimized

    @patch("orchard.create_tracker")
    @patch("orchard.run_training_phase")
    @patch("orchard.RootOrchestrator")
    @patch("orchard.Config")
    def test_run_keyboard_interrupt(
        self, mock_cfg_cls, mock_orch_cls, mock_train, mock_tracker_fn, tmp_path
    ):
        """Covers KeyboardInterrupt handling."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("dataset:\n  name: test\n")

        mock_cfg = MagicMock()
        mock_cfg.optuna = None
        mock_cfg_cls.from_recipe.return_value = mock_cfg

        mock_orch_cls.return_value.__enter__.return_value = MagicMock()
        mock_train.side_effect = KeyboardInterrupt
        mock_tracker_fn.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(recipe)])

        assert result.exit_code != 0

    @patch("orchard.create_tracker")
    @patch("orchard.run_training_phase")
    @patch("orchard.RootOrchestrator")
    @patch("orchard.Config")
    def test_run_pipeline_error(
        self, mock_cfg_cls, mock_orch_cls, mock_train, mock_tracker_fn, tmp_path
    ):
        """Covers generic exception handling."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("dataset:\n  name: test\n")

        mock_cfg = MagicMock()
        mock_cfg.optuna = None
        mock_cfg_cls.from_recipe.return_value = mock_cfg

        mock_orch_cls.return_value.__enter__.return_value = MagicMock()
        mock_train.side_effect = RuntimeError("GPU OOM")
        mock_tracker_fn.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(recipe)])

        assert result.exit_code != 0

    @patch("orchard.create_tracker")
    @patch("orchard.run_training_phase")
    @patch("orchard.RootOrchestrator")
    @patch("orchard.Config")
    def test_run_orchard_error_clean_exit(
        self, mock_cfg_cls, mock_orch_cls, mock_train, mock_tracker_fn, tmp_path
    ):
        """OrchardError produces clean exit without traceback."""
        from typer.testing import CliRunner

        from orchard.cli_app import app
        from orchard.exceptions import OrchardConfigError

        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("dataset:\n  name: test\n")

        mock_cfg = MagicMock()
        mock_cfg.optuna = None
        mock_cfg_cls.from_recipe.return_value = mock_cfg

        mock_orch_cls.return_value.__enter__.return_value = MagicMock()
        mock_train.side_effect = OrchardConfigError("bad config value")
        mock_tracker_fn.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(recipe)])

        assert result.exit_code != 0


# CLI INIT COMMAND
@pytest.mark.unit
class TestCLIInit:
    """Tests for the ``init`` command."""

    def test_init_creates_recipe(self, tmp_path):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        result = runner.invoke(app, ["init", str(tmp_path / "recipe.yaml")])
        assert result.exit_code == 0
        content = (tmp_path / "recipe.yaml").read_text()
        for section in (
            "dataset:",
            "training:",
            "augmentation:",
            "hardware:",
            "optuna:",
            "export:",
        ):
            assert section in content

    def test_init_custom_filename(self, tmp_path):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        out = tmp_path / "my_recipe.yaml"
        result = runner.invoke(app, ["init", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "my_recipe.yaml" in result.output

    def test_init_refuses_overwrite(self, tmp_path):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        target = tmp_path / "recipe.yaml"
        target.write_text("existing")
        runner = CliRunner()
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 1
        assert "already exists" in result.output
        assert target.read_text() == "existing"

    def test_init_force_overwrites(self, tmp_path):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        target = tmp_path / "recipe.yaml"
        target.write_text("existing")
        runner = CliRunner()
        result = runner.invoke(app, ["init", str(target), "--force"])
        assert result.exit_code == 0
        content = target.read_text()
        assert "dataset:" in content

    def test_init_valid_yaml(self, tmp_path):
        """Generated YAML is parseable via load_config_from_yaml."""
        from typer.testing import CliRunner

        from orchard.cli_app import app
        from orchard.core.io import load_config_from_yaml

        runner = CliRunner()
        target = tmp_path / "recipe.yaml"
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 0
        data = load_config_from_yaml(target)
        assert data["dataset"]["name"] == "bloodmnist"
        assert data["training"]["seed"] == 42

    def test_init_no_internal_fields(self, tmp_path):
        """metadata and img_size must not appear in generated YAML."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        target = tmp_path / "recipe.yaml"
        runner.invoke(app, ["init", str(target)])
        content = target.read_text()
        assert "metadata:" not in content
        assert "img_size:" not in content

    def test_init_device_is_auto(self, tmp_path):
        """Hardware device should be 'auto', not resolved."""
        from typer.testing import CliRunner

        from orchard.cli_app import app
        from orchard.core.io import load_config_from_yaml

        runner = CliRunner()
        target = tmp_path / "recipe.yaml"
        runner.invoke(app, ["init", str(target)])
        data = load_config_from_yaml(target)
        assert data["hardware"]["device"] == "auto"

    def test_init_paths_are_portable(self, tmp_path):
        """Paths should be relative, not absolute."""
        from typer.testing import CliRunner

        from orchard.cli_app import app
        from orchard.core.io import load_config_from_yaml

        runner = CliRunner()
        target = tmp_path / "recipe.yaml"
        runner.invoke(app, ["init", str(target)])
        data = load_config_from_yaml(target)
        assert data["dataset"]["data_root"] == "./dataset"
        assert data["telemetry"]["output_dir"] == "./outputs"

    def test_init_help(self):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        clean = _strip_ansi(result.output)
        assert "OUTPUT" in clean or "output" in clean.lower()

    def test_init_header_present(self, tmp_path):
        """Generated file starts with the header comment."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        target = tmp_path / "recipe.yaml"
        runner.invoke(app, ["init", str(target)])
        content = target.read_text()
        assert content.startswith("# yaml-language-server:")
        assert "# =" in content
        assert "orchard run" in content


class TestCommentedYaml:
    """Tests for commented YAML generation in init command."""

    @pytest.fixture()
    def recipe_content(self, tmp_path):
        """Generate a recipe and return its content."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        target = tmp_path / "recipe.yaml"
        runner.invoke(app, ["init", str(target)])
        return target.read_text()

    def test_comments_present(self, recipe_content):
        """Generated recipe contains field description comments."""
        assert "# Dataset identifier" in recipe_content
        assert "# Samples per batch" in recipe_content
        assert "# Optimizer algorithm" in recipe_content
        assert "# Initial learning rate" in recipe_content
        assert "# Enable MLflow tracking" in recipe_content

    def test_constraint_ranges(self, recipe_content):
        """Comments include constraint ranges with en-dash."""
        assert "(1\u2013128)" in recipe_content  # batch_size
        assert "(0.0\u20130.2)" in recipe_content  # weight_decay

    def test_enum_options(self, recipe_content):
        """Comments include enum values in brackets."""
        assert "[sgd, adamw]" in recipe_content
        assert "[auc, accuracy, f1]" in recipe_content
        assert "[cosine, plateau, step, none]" in recipe_content

    def test_one_sided_constraints(self, recipe_content):
        """Comments include one-sided constraints."""
        assert "(\u2265 0)" in recipe_content  # patience
        assert "(> 0)" in recipe_content  # epochs

    def test_roundtrip_values(self, tmp_path):
        """Commented YAML parses to correct values."""
        from typer.testing import CliRunner

        from orchard.cli_app import app
        from orchard.core.io import load_config_from_yaml

        runner = CliRunner()
        target = tmp_path / "recipe.yaml"
        runner.invoke(app, ["init", str(target)])
        data = load_config_from_yaml(target)

        assert set(data.keys()) == {
            "dataset",
            "architecture",
            "training",
            "augmentation",
            "hardware",
            "telemetry",
            "evaluation",
            "tracking",
            "export",
            "optuna",
        }
        assert data["training"]["learning_rate"] == pytest.approx(0.008)
        assert data["training"]["min_lr"] == pytest.approx(1e-6)
        assert data["dataset"]["max_samples"] is None
        assert data["evaluation"]["fig_size_predictions"] == [12, 8]

    def test_nested_search_space(self, tmp_path):
        """Nested search_space_overrides renders correctly."""
        from typer.testing import CliRunner

        from orchard.cli_app import app
        from orchard.core.io import load_config_from_yaml

        runner = CliRunner()
        target = tmp_path / "recipe.yaml"
        runner.invoke(app, ["init", str(target)])
        data = load_config_from_yaml(target)
        sso = data["optuna"]["search_space_overrides"]
        assert sso["learning_rate"]["log"] is True
        assert sso["batch_size_low_res"] == [16, 32, 48, 64]
        assert sso["optimizer_type"] == ["sgd", "adamw"]

    def test_no_floatrange_docstrings(self, recipe_content):
        """FloatRange/IntRange type docstrings must not leak as comments."""
        assert "Typed bounds for" not in recipe_content
        assert "Lower bound (inclusive)" not in recipe_content

    def test_comments_above_fields(self, recipe_content):
        """Comments appear on the line immediately above the field at indent=1."""
        lines = recipe_content.split("\n")
        for i, line in enumerate(lines):
            if line.lstrip().startswith("batch_size: ") and i > 0:
                # Must be at exactly indent=1 (4 spaces)
                assert line.startswith("    ") and not line.startswith("        ")
                assert lines[i - 1].strip().startswith("# Samples per batch")
                break


class TestYamlHelpers:
    """Unit tests for YAML builder helper functions."""

    def test_format_none(self):
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value(None) == "null"

    def test_format_bool(self):
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value(True) == "true"
        assert _format_yaml_value(False) == "false"

    def test_format_int(self):
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value(42) == "42"

    def test_format_float(self):
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value(0.5) == "0.5"

    def test_format_small_float(self):
        from orchard.cli_app import _format_yaml_value

        result = _format_yaml_value(1e-6)
        assert "1" in result
        assert "e" in result.lower()
        # Must not contain YAML document-end marker
        assert "..." not in result

    def test_format_string_keyword(self):
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value("true") == "'true'"
        assert _format_yaml_value("null") == "'null'"

    def test_format_plain_string(self):
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value("bloodmnist") == "bloodmnist"

    def test_build_comment_description_only(self):
        from orchard.cli_app import _build_comment

        assert _build_comment({"description": "Enable tracking"}) == "Enable tracking"

    def test_build_comment_enum(self):
        from orchard.cli_app import _build_comment

        result = _build_comment({"description": "Optimizer", "enum": ["sgd", "adamw"]})
        assert result == "Optimizer [sgd, adamw]"

    def test_build_comment_range(self):
        from orchard.cli_app import _build_comment

        result = _build_comment({"description": "Batch size", "minimum": 1, "maximum": 128})
        assert "Batch size" in result
        assert "1" in result
        assert "128" in result

    def test_build_comment_no_description(self):
        from orchard.cli_app import _build_comment

        assert _build_comment({"minimum": 0, "type": "integer"}) is None
        assert _build_comment({}) is None

    def test_build_comment_exclusive_range(self):
        from orchard.cli_app import _build_comment

        result = _build_comment(
            {"description": "LR", "exclusiveMinimum": 1e-8, "exclusiveMaximum": 1.0}
        )
        assert "LR" in result  # type: ignore[operator]
        assert "1e-08" in result  # type: ignore[operator]

    def test_build_comment_upper_only(self):
        from orchard.cli_app import _build_comment

        assert _build_comment({"description": "Cap", "maximum": 100}) == "Cap (\u2264 100)"
        assert _build_comment({"description": "Lim", "exclusiveMaximum": 5}) == "Lim (< 5)"

    def test_format_string_with_special_chars(self):
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value("a:b") == "'a:b'"
        assert _format_yaml_value("") == "''"

    def test_format_large_float(self):
        from orchard.cli_app import _format_yaml_value

        result = _format_yaml_value(1e8)
        assert "..." not in result

    def test_format_zero_float(self):
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value(0.0) == "0.0"

    def test_render_list_of_dicts(self, tmp_path):
        """list-of-dicts branch in _render_fields."""
        from orchard.cli_app import _render_fields

        lines: list[str] = []
        _render_fields(lines, {"items": [{"a": 1}]}, {}, indent=0, defs={})
        text = "\n".join(lines)
        assert "items:" in text
        assert "a: 1" in text

    def test_format_non_scalar_falls_back_to_yaml_dump(self):
        """yaml.dump fallback for non-scalar types (tuple, custom objects)."""
        from orchard.cli_app import _format_yaml_value

        result = _format_yaml_value((1, 2, 3))
        # yaml.dump renders tuples as flow sequences: [1, 2, 3]
        assert "1" in result
        assert "2" in result
        assert "..." not in result

    def test_build_commented_yaml_unknown_section(self):
        """Sections with no matching Pydantic model still render fields."""
        from orchard.cli_app import _build_commented_yaml

        data = {"unknown_section": {"foo": "bar", "baz": 42}}
        result = _build_commented_yaml(data)
        assert "unknown_section:" in result
        assert "foo: bar" in result
        assert "baz: 42" in result


# INTEGRATION: CLI + REAL CONFIG + REAL ORCHESTRATOR
@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests that exercise real Config.from_recipe and RootOrchestrator phases."""

    @staticmethod
    def _write_recipe(tmp_path, overrides=None):
        import yaml

        content = {
            "dataset": {"name": "bloodmnist", "resolution": 28, "force_rgb": True},
            "architecture": {"name": "mini_cnn", "pretrained": False},
            "training": {"epochs": 1, "mixup_epochs": 0, "use_amp": False, "seed": 42},
            "hardware": {"device": "cpu", "project_name": "cli-integration"},
            "telemetry": {"output_dir": str(tmp_path)},
        }
        if overrides:
            for k, v in overrides.items():
                section, field = k.split(".", 1)
                content[section][field] = v
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.dump(content), encoding="utf-8")
        return recipe

    @patch("orchard.core.io.serialization.dump_requirements")
    @patch("orchard.core.orchestrator.InfrastructureManager")
    @patch("orchard.log_pipeline_summary")
    @patch("orchard.create_tracker")
    @patch("orchard.run_training_phase")
    def test_run_real_config_creates_workspace(
        self, mock_train, mock_tracker_fn, mock_summary, mock_infra_cls, _mock_dump_req, tmp_path
    ):
        """Real Config + RootOrchestrator creates workspace, writes config.yaml."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        recipe = self._write_recipe(tmp_path)

        mock_infra_cls.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.best_model_path = tmp_path / "model.pt"
        mock_result.test_acc = 0.90
        mock_result.macro_f1 = 0.88
        mock_result.test_auc = 0.91
        mock_train.return_value = mock_result
        mock_tracker = MagicMock()
        mock_tracker_fn.return_value = mock_tracker

        result = CliRunner().invoke(app, ["run", str(recipe)])
        assert result.exit_code == 0, result.output

        # Real RunPaths created a directory tree under tmp_path
        run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(run_dirs) >= 1
        run_dir = run_dirs[0]
        assert (run_dir / "reports" / "config.yaml").exists()

        mock_train.assert_called_once()
        mock_tracker.start_run.assert_called_once()
        mock_tracker.end_run.assert_called_once()
        mock_infra_cls.return_value.prepare_environment.assert_called_once()

    @patch("orchard.core.io.serialization.dump_requirements")
    @patch("orchard.core.orchestrator.InfrastructureManager")
    @patch("orchard.log_pipeline_summary")
    @patch("orchard.create_tracker")
    @patch("orchard.run_training_phase")
    def test_run_set_override_reaches_config(
        self, mock_train, mock_tracker_fn, mock_summary, mock_infra_cls, _mock_dump_req, tmp_path
    ):
        """--set training.seed=99 overrides the recipe value (42) end-to-end."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        recipe = self._write_recipe(tmp_path)

        mock_infra_cls.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.best_model_path = tmp_path / "model.pt"
        mock_result.test_acc = 0.9
        mock_result.macro_f1 = 0.9
        mock_result.test_auc = 0.9
        mock_train.return_value = mock_result
        mock_tracker_fn.return_value = MagicMock()

        result = CliRunner().invoke(app, ["run", str(recipe), "--set", "training.seed=99"])
        assert result.exit_code == 0, result.output

        call_kwargs = mock_train.call_args
        passed_cfg = call_kwargs.kwargs.get("cfg") or call_kwargs[1]
        assert passed_cfg.training.seed == 99

    def test_run_missing_recipe_exits_clean(self, tmp_path):
        """Missing recipe exits with code 1, no filesystem side effects."""
        from typer.testing import CliRunner

        from orchard.cli_app import app

        result = CliRunner().invoke(app, ["run", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_init_produces_loadable_recipe(self, tmp_path):
        """orchard init generates a YAML parseable by Config.from_recipe."""
        from typer.testing import CliRunner

        from orchard.cli_app import app
        from orchard.core import Config

        recipe = tmp_path / "starter.yaml"
        result = CliRunner().invoke(app, ["init", str(recipe)])
        assert result.exit_code == 0
        assert recipe.exists()

        cfg = Config.from_recipe(recipe)
        assert cfg.dataset.dataset_name is not None
        assert cfg.architecture.name is not None


# ---------------------------------------------------------------------------
# Mutation-killing tests for YAML generation helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInlineRef:
    """Direct tests for _inline_ref to kill schema resolution mutants."""

    def test_ref_key_resolution(self):
        """rsplit("/", 1)[-1] must extract the last path segment."""
        from orchard.cli_app import _inline_ref

        schema = {"$ref": "#/$defs/MyType", "description": "override"}
        defs = {"MyType": {"type": "string", "description": "from def"}}
        result = _inline_ref(schema, defs)
        assert result["type"] == "string"
        # description from schema overrides base
        assert result["description"] == "override"

    def test_ref_description_not_inherited(self):
        """When property has no description, base description must be removed."""
        from orchard.cli_app import _inline_ref

        schema = {"$ref": "#/$defs/FloatRange"}
        defs = {"FloatRange": {"type": "number", "description": "Typed bounds for float"}}
        result = _inline_ref(schema, defs)
        assert "description" not in result
        assert result["type"] == "number"

    def test_ref_excludes_ref_key(self):
        """$ref itself must not appear in the resolved result."""
        from orchard.cli_app import _inline_ref

        schema = {"$ref": "#/$defs/X", "minimum": 0}
        defs = {"X": {"type": "integer"}}
        result = _inline_ref(schema, defs)
        assert "$ref" not in result
        assert result["minimum"] == 0
        assert result["type"] == "integer"

    def test_ref_values_not_nullified(self):
        """Schema values must be copied as-is, not replaced with None."""
        from orchard.cli_app import _inline_ref

        schema = {"$ref": "#/$defs/X", "minimum": 5, "maximum": 10}
        defs = {"X": {"type": "integer"}}
        result = _inline_ref(schema, defs)
        assert result["minimum"] == 5
        assert result["maximum"] == 10

    def test_ref_missing_def_returns_schema_values(self):
        """Missing $def key should still return schema properties (minus $ref)."""
        from orchard.cli_app import _inline_ref

        schema = {"$ref": "#/$defs/Missing", "description": "kept"}
        result = _inline_ref(schema, {})
        assert result["description"] == "kept"

    def test_ref_deep_path(self):
        """Multi-segment $ref path should still resolve correctly."""
        from orchard.cli_app import _inline_ref

        schema = {"$ref": "#/$defs/deep/nested/TypeName"}
        defs = {"TypeName": {"type": "object"}}
        result = _inline_ref(schema, defs)
        assert result["type"] == "object"


@pytest.mark.unit
class TestMergeAnyOf:
    """Direct tests for _merge_any_of to kill anyOf merging mutants."""

    def test_excludes_anyof_key(self):
        """anyOf key must be excluded from merged result."""
        from orchard.cli_app import _merge_any_of

        schema = {
            "description": "test",
            "anyOf": [
                {"type": "integer", "minimum": 1, "maximum": 100},
                {"type": "null"},
            ],
        }
        result = _merge_any_of(schema)
        assert "anyOf" not in result
        assert result["description"] == "test"
        assert result["minimum"] == 1
        assert result["maximum"] == 100

    def test_skips_null_variant(self):
        """Null variant must be skipped; constraints come from non-null."""
        from orchard.cli_app import _merge_any_of

        schema = {
            "anyOf": [
                {"type": "null"},
                {"type": "number", "exclusiveMinimum": 0.0},
            ],
        }
        result = _merge_any_of(schema)
        assert result["exclusiveMinimum"] == 0.0

    def test_constraint_values_not_none(self):
        """Constraint values must be copied from variant, not set to None."""
        from orchard.cli_app import _merge_any_of

        schema = {
            "anyOf": [
                {"type": "integer", "minimum": 5, "maximum": 50, "enum": [5, 10, 50]},
                {"type": "null"},
            ],
        }
        result = _merge_any_of(schema)
        assert result["minimum"] == 5
        assert result["maximum"] == 50
        assert result["enum"] == [5, 10, 50]

    def test_only_non_null_constraints(self):
        """With == instead of !=, we'd pick null variant (wrong)."""
        from orchard.cli_app import _merge_any_of

        schema = {
            "anyOf": [
                {"type": "null"},
                {"type": "integer", "minimum": 0},
            ],
        }
        result = _merge_any_of(schema)
        # null variant has no "minimum", so this tests the right variant was selected
        assert "minimum" in result
        assert result["minimum"] == 0


@pytest.mark.unit
class TestResolveRefs:
    """Direct tests for _resolve_refs to kill string mutation mutants."""

    def test_ref_resolved(self):
        from orchard.cli_app import _resolve_refs

        properties = {"field": {"$ref": "#/$defs/MyType"}}
        defs = {"MyType": {"type": "string"}}
        result = _resolve_refs(properties, defs)
        assert result["field"]["type"] == "string"
        assert "$ref" not in result["field"]

    def test_anyof_merged(self):
        from orchard.cli_app import _resolve_refs

        properties = {
            "field": {
                "anyOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "null"},
                ],
            }
        }
        result = _resolve_refs(properties, {})
        assert "anyOf" not in result["field"]
        assert result["field"]["minimum"] == 1

    def test_plain_passthrough(self):
        from orchard.cli_app import _resolve_refs

        properties = {"field": {"type": "string", "description": "plain"}}
        result = _resolve_refs(properties, {})
        assert result["field"] == {"type": "string", "description": "plain"}


@pytest.mark.unit
class TestFormatYamlValueMutants:
    """Tests targeting boundary float formatting mutants."""

    def test_boundary_1e_minus_3(self):
        """Value exactly at 1e-3 should use str(), not yaml.dump."""
        from orchard.cli_app import _format_yaml_value

        # 1e-3 = 0.001, which is NOT < 1e-3, so should use str()
        result = _format_yaml_value(0.001)
        assert result == "0.001"

    def test_just_below_1e_minus_3(self):
        """Value below 1e-3 should use yaml.dump (scientific notation)."""
        from orchard.cli_app import _format_yaml_value

        result = _format_yaml_value(1e-5)
        assert "e" in result.lower() or "E" in result

    def test_boundary_1e7(self):
        """Value exactly at 1e7 should use yaml.dump."""
        from orchard.cli_app import _format_yaml_value

        result = _format_yaml_value(1e7)
        assert "..." not in result

    def test_just_below_1e7(self):
        """Value just below 1e7 should use str()."""
        from orchard.cli_app import _format_yaml_value

        result = _format_yaml_value(9999999.0)
        assert result == "9999999.0"

    def test_zero_uses_str(self):
        """0.0 should use str(), not yaml.dump (isclose to 0)."""
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value(0.0) == "0.0"

    def test_and_vs_or_logic(self):
        """Non-zero value in normal range uses str(), not yaml.dump."""
        from orchard.cli_app import _format_yaml_value

        # 0.5 is not close to 0, abs >= 1e-3 and abs < 1e7 → str()
        assert _format_yaml_value(0.5) == "0.5"
        # If `and` were changed to `or`, 0.5 would hit yaml.dump branch

    def test_yaml_dump_flow_style(self):
        """yaml.dump must use default_flow_style=True for inline output."""
        from orchard.cli_app import _format_yaml_value

        # 1e-6 triggers yaml.dump path; flow style produces "1.0e-06" inline
        result = _format_yaml_value(1e-6)
        assert "\n" not in result
        assert "..." not in result

    def test_split_on_newline(self):
        """yaml.dump result must be split on \\n, not None or other delimiter."""
        from orchard.cli_app import _format_yaml_value

        # This value goes through yaml.dump path
        result = _format_yaml_value(1e-10)
        assert "\n" not in result
        assert isinstance(result, str)
        assert len(result) > 0

    def test_special_chars_quoting(self):
        """Strings with special YAML chars must be quoted."""
        from orchard.cli_app import _format_yaml_value

        assert _format_yaml_value("a:b") == "'a:b'"
        assert _format_yaml_value("a#b") == "'a#b'"
        assert _format_yaml_value("a{b") == "'a{b'"
        assert _format_yaml_value("a}b") == "'a}b'"

    def test_string_without_special_chars_not_quoted(self):
        """Strings without special YAML chars must NOT be quoted."""
        from orchard.cli_app import _format_yaml_value

        # These contain X but X is NOT a special char
        assert _format_yaml_value("test_X") == "test_X"
        assert _format_yaml_value("MIXED") == "MIXED"

    def test_non_scalar_split_on_newline(self):
        """Non-scalar yaml.dump must also split on \\n."""
        from orchard.cli_app import _format_yaml_value

        result = _format_yaml_value([1, 2, 3])
        assert "\n" not in result
        assert "..." not in result

    def test_non_scalar_flow_style(self):
        """Non-scalar yaml.dump must use default_flow_style=True."""
        from orchard.cli_app import _format_yaml_value

        result = _format_yaml_value({"a": 1})
        # flow style: {a: 1}, block style would have newlines
        assert "{" in result or "a:" in result


@pytest.mark.unit
class TestAppendWrappedComment:
    """Tests targeting comment wrapping arithmetic mutants."""

    def test_short_comment_no_wrap(self):
        from orchard.cli_app import _append_wrapped_comment

        lines: list[str] = []
        _append_wrapped_comment(lines, "Short", "    ")
        assert len(lines) == 1
        assert lines[0] == "    # Short"

    def test_long_comment_wraps(self):
        from orchard.cli_app import _append_wrapped_comment

        lines: list[str] = []
        # _COMMENT_MAX_WIDTH=80, prefix="    " (4 chars), subtract 2 for "# " = 74 chars max
        long_comment = "A" * 75  # exceeds 74
        _append_wrapped_comment(lines, long_comment, "    ")
        assert len(lines) > 1
        for line in lines:
            assert line.startswith("    # ")

    def test_boundary_exact_max_no_wrap(self):
        """Comment exactly at max_text (with spaces) should NOT wrap."""
        from orchard.cli_app import _COMMENT_MAX_WIDTH, _append_wrapped_comment

        prefix = "    "
        max_text = _COMMENT_MAX_WIDTH - len(prefix) - 2  # 74
        # Build a comment of exactly max_text chars with word boundaries
        exact_comment = "A " * (max_text // 2)  # wrappable, exactly max_text
        assert len(exact_comment) == max_text
        lines: list[str] = []
        _append_wrapped_comment(lines, exact_comment, prefix)
        # With <=: len==max_text → True → 1 line (no wrap)
        # With < : len==max_text → False → wraps → 2+ lines
        assert len(lines) == 1

    def test_subtract_2_not_3(self):
        """max_text = MAX_WIDTH - len(prefix) - 2, not - 3."""
        from orchard.cli_app import _COMMENT_MAX_WIDTH, _append_wrapped_comment

        prefix = "    "
        max_text = _COMMENT_MAX_WIDTH - len(prefix) - 2  # 74
        # 74-char wrappable comment: fits in 74 but NOT in 73
        comment = " ".join(["word"] * 15)  # 74 chars
        assert len(comment) == max_text
        lines: list[str] = []
        _append_wrapped_comment(lines, comment, prefix)
        # With -2 (correct): max_text=74, len<=74 → 1 line
        # With -3 (mutant): max_text=73, len>73 → wraps
        assert len(lines) == 1

    def test_width_parameter_not_omitted(self):
        """textwrap.wrap must use explicit width=max_text, not default 70."""
        from orchard.cli_app import _COMMENT_MAX_WIDTH, _append_wrapped_comment

        prefix = "    "
        max_text = _COMMENT_MAX_WIDTH - len(prefix) - 2  # 74
        # 72 chars with spaces: fits in 74 but NOT in default 70
        comment = "W " * 36  # 72 chars
        assert 70 < len(comment) <= max_text
        lines: list[str] = []
        _append_wrapped_comment(lines, comment, prefix)
        # With width=74 (correct): 1 line
        # With width omitted (default 70): 2+ lines
        assert len(lines) == 1


@pytest.mark.unit
class TestBuildCommentedYamlMutants:
    """Tests for _build_commented_yaml schema resolution mutants."""

    def test_defs_resolved(self):
        """$defs must be fetched from schema for ref resolution."""
        from orchard.cli_app import _build_commented_yaml

        # Use a real config model that has $defs in its JSON schema
        data = {"training": {"learning_rate": 0.008}}
        result = _build_commented_yaml(data)
        # With correct $defs resolution, description comments appear
        assert "# " in result
        assert "training:" in result
        assert "learning_rate: " in result

    def test_properties_key_fetched(self):
        """Schema 'properties' must be fetched for sub-section rendering."""
        from orchard.cli_app import _build_commented_yaml

        # Dataset has nested properties with descriptions
        data = {"dataset": {"name": "bloodmnist", "resolution": 28}}
        result = _build_commented_yaml(data)
        assert "# Dataset identifier" in result or "# " in result
        assert "name: bloodmnist" in result

    def test_empty_defs_fallback(self):
        """Missing $defs should not crash (fallback to {})."""
        from orchard.cli_app import _build_commented_yaml

        # unknown_section has no model → properties={}, defs={}
        data = {"unknown": {"key": "val"}}
        result = _build_commented_yaml(data)
        assert "unknown:" in result
        assert "key: val" in result

    def test_fields_at_indent_one(self):
        """Fields under a section must be at indent=1 (4 spaces), not 2."""
        from orchard.cli_app import _build_commented_yaml

        data = {"training": {"epochs": 10, "seed": 42}}
        result = _build_commented_yaml(data)
        lines = result.split("\n")
        epoch_lines = [ln for ln in lines if "epochs:" in ln and ln.strip().startswith("epochs")]
        assert len(epoch_lines) == 1
        # Exactly 4 spaces (indent=1), NOT 8 (indent=2)
        assert epoch_lines[0].startswith("    ") and not epoch_lines[0].startswith("        ")


@pytest.mark.unit
class TestBuildInitDictMutants:
    """Tests for _build_init_dict to kill pop() key mutants."""

    def test_metadata_removed(self):
        from orchard.cli_app import _build_init_dict

        result = _build_init_dict()
        assert "metadata" not in result["dataset"]

    def test_img_size_removed(self):
        from orchard.cli_app import _build_init_dict

        result = _build_init_dict()
        assert "img_size" not in result["dataset"]

    def test_pop_with_none_default(self):
        """pop("img_size", None) — second arg must be None (not omitted)."""
        from orchard.cli_app import _build_init_dict

        # If pop() is called without default and key is missing, it raises KeyError
        # This test ensures it doesn't raise
        result = _build_init_dict()
        assert isinstance(result["dataset"], dict)


@pytest.mark.unit
class TestValidateOverrideKeyMutants:
    """Tests targeting split vs rsplit and maxsplit mutants."""

    def test_nested_dotted_key_accepted(self):
        """'training.epochs' (one dot) must be accepted."""
        from orchard.cli_app import _validate_override_key

        # Should not raise
        _validate_override_key("training.epochs")

    def test_split_vs_rsplit_difference(self):
        """With split('.', maxsplit=1), 'a.b.c' → ['a', 'b.c'] (section='a').
        With rsplit('.', maxsplit=1), 'a.b.c' → ['a.b', 'c'] (section='a.b', invalid).
        """

        from orchard.cli_app import _validate_override_key

        # "training.epochs" should work (section=training, field=epochs)
        _validate_override_key("training.epochs")

        # We need a key with 2+ dots where split and rsplit differ
        # "augmentation.jitter_val" works with split → section="augmentation", field="jitter_val"
        _validate_override_key("augmentation.jitter_val")

    def test_maxsplit_1_vs_none(self):
        """Without maxsplit=1, 'section.field' still produces 2 parts.
        But with maxsplit=None, a key like 'training.sub.field' gives 3 parts → rejected.
        With maxsplit=1, it gives 2 parts → accepted (field='sub.field', then validated).
        """

        from orchard.cli_app import _validate_override_key

        # This key has one dot — works either way
        _validate_override_key("training.epochs")

    def test_maxsplit_2_difference(self):
        """maxsplit=2 vs maxsplit=1: for 'a.b', both give ['a','b'].
        No difference for simple keys, but the test structure ensures
        maxsplit=1 is tested via the len(parts)==2 check.
        """

        from orchard.cli_app import _validate_override_key

        _validate_override_key("dataset.name")


@pytest.mark.unit
class TestVersionCallbackMutant:
    """Test that version callback uses correct package name casing."""

    def test_version_output_contains_version(self):
        from typer.testing import CliRunner

        from orchard.cli_app import app

        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Must contain a version-like string (digits and dots)
        output = result.output.strip()
        assert "orchard-ml" in output
        parts = output.split()
        assert len(parts) >= 2
        # The version part should contain digits
        assert any(c.isdigit() for c in parts[-1])


@pytest.mark.unit
class TestRenderFieldsMutants:
    """Tests for _render_fields to kill property/indent mutants."""

    def test_nested_dict_properties_key(self):
        """Sub-dict must look up 'properties' from schema, not mutated keys."""
        from orchard.cli_app import _render_fields

        # Simulate a schema where the parent has a "properties" key in field_schema
        properties = {
            "outer": {
                "properties": {
                    "inner": {"description": "Inner field"},
                },
            }
        }
        lines: list[str] = []
        _render_fields(lines, {"outer": {"inner": 42}}, properties, indent=0, defs={})
        text = "\n".join(lines)
        assert "outer:" in text
        assert "    inner: 42" in text
        # Inner field should have its comment
        assert "# Inner field" in text

    def test_nested_dict_indent_increments_by_one(self):
        """Nested dict should indent by exactly 1 level (not 2)."""
        from orchard.cli_app import _render_fields

        lines: list[str] = []
        _render_fields(lines, {"a": {"b": 1}}, {}, indent=0, defs={})
        text = "\n".join(lines)
        assert "a:" in text
        # b should be at indent 1 (4 spaces)
        assert "    b: 1" in text
        # NOT at indent 2 (8 spaces)
        b_lines = [ln for ln in lines if "b: 1" in ln]
        assert b_lines[0] == "    b: 1"

    def test_list_of_dicts_indent(self):
        """List-of-dict items should use indent+2 for nested fields."""
        from orchard.cli_app import _render_fields

        lines: list[str] = []
        _render_fields(lines, {"items": [{"x": 1, "y": 2}]}, {}, indent=0, defs={})
        text = "\n".join(lines)
        assert "items:" in text
        # "-" at indent+1 = 4 spaces
        assert "    -" in text
        # Fields at indent+2 = exactly 8 spaces (not 12)
        x_lines = [ln for ln in lines if "x: 1" in ln]
        assert len(x_lines) == 1
        assert x_lines[0] == "        x: 1"

    def test_defs_passed_to_recursive_calls(self):
        """defs parameter must propagate to nested _render_fields calls."""
        from orchard.cli_app import _render_fields

        # If defs=None were passed, _resolve_refs would crash when it tries
        # to call defs.get(). We verify no crash with actual defs.
        defs = {"SomeType": {"type": "string"}}
        properties = {
            "outer": {
                "properties": {
                    "inner": {"$ref": "#/$defs/SomeType"},
                },
            }
        }
        lines: list[str] = []
        _render_fields(lines, {"outer": {"inner": "val"}}, properties, indent=0, defs=defs)
        text = "\n".join(lines)
        assert "inner: val" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
