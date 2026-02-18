"""
Test Suite for Orchard ML CLI (cli_app.py).

Tests the Typer-based CLI utilities: override parsing, auto-casting,
and basic command invocation.
"""

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
        result = _parse_overrides(["key=a=b"])
        assert result == {"key": "a=b"}

    def test_missing_equals_raises(self):
        import typer

        with pytest.raises(typer.BadParameter, match="key=value"):
            _parse_overrides(["no_equals_here"])

    def test_empty_key_raises(self):
        import typer

        with pytest.raises(typer.BadParameter, match="Empty key"):
            _parse_overrides(["=value"])

    def test_whitespace_stripped(self):
        result = _parse_overrides(["  key  =  42  "])
        assert result == {"key": 42}

    def test_float_scientific(self):
        result = _parse_overrides(["training.lr=1e-4"])
        assert result["training.lr"] == pytest.approx(1e-4)

    def test_null_value(self):
        result = _parse_overrides(["optuna=null"])
        assert result == {"optuna": None}


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
        mock_train.return_value = (Path("model.pt"), None, None, None, 0.95, 0.90)
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
        mock_cfg.export.format = "onnx"
        mock_cfg.export.opset_version = 18
        mock_cfg_cls.from_recipe.return_value = mock_cfg

        mock_orch = MagicMock()
        mock_orch_cls.return_value.__enter__.return_value = mock_orch
        mock_optuna.return_value = (MagicMock(), best_config)
        mock_train.return_value = (Path("model.pt"), None, None, None, 0.95, 0.90)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
