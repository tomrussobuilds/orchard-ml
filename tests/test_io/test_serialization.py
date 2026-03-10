"""
Smoke Tests for Configuration Serialization Module.

Tests to validate YAML serialization and deserialization.

"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from orchard.core.io.serialization import (
    _persist_yaml_atomic,
    _sanitize_for_yaml,
    dump_git_info,
    dump_requirements,
    load_config_from_yaml,
    save_config_as_yaml,
)


# SANITIZE FOR YAML
@pytest.mark.unit
def test_sanitize_for_yaml_path_objects():
    """Test _sanitize_for_yaml converts Path objects to strings."""
    data = {"output_dir": Path("/mock/outputs"), "log_file": Path("/mock/log.txt")}

    result = _sanitize_for_yaml(data)

    assert result["output_dir"] == "/mock/outputs"
    assert result["log_file"] == "/mock/log.txt"
    assert isinstance(result["output_dir"], str)


@pytest.mark.unit
def test_sanitize_for_yaml_nested_structures():
    """Test _sanitize_for_yaml handles nested dicts and lists."""
    data = {
        "paths": {"data": Path("/data"), "models": Path("/models")},
        "sizes": [28, 224, Path("/path/file")],
    }

    result = _sanitize_for_yaml(data)

    assert result["paths"]["data"] == "/data"
    assert result["paths"]["models"] == "/models"
    assert result["sizes"][2] == "/path/file"


@pytest.mark.unit
def test_sanitize_for_yaml_primitives():
    """Test _sanitize_for_yaml preserves primitive types."""
    data = {"int": 42, "float": 3.14, "str": "test", "bool": True, "none": None}

    result = _sanitize_for_yaml(data)

    assert result == data


@pytest.mark.unit
def test_sanitize_for_yaml_tuples():
    """Test _sanitize_for_yaml converts tuples to lists."""
    data = {"tuple": (1, 2, Path("/path"))}

    result = _sanitize_for_yaml(data)

    assert result["tuple"] == [1, 2, "/path"]
    assert isinstance(result["tuple"], list)


# SAVE CONFIG AS YAML
@pytest.mark.unit
def test_save_config_as_yaml_with_dict(tmp_path):
    """Test save_config_as_yaml saves dictionary to YAML."""
    config = {"model": "resnet", "epochs": 10, "lr": 0.001}
    yaml_path = tmp_path / "config.yaml"

    result = save_config_as_yaml(config, yaml_path)

    assert result == yaml_path
    assert yaml_path.exists()

    with open(yaml_path, "r") as f:
        loaded = yaml.safe_load(f)
    assert loaded == config


@pytest.mark.unit
def test_save_config_as_yaml_with_model_dump():
    """Test save_config_as_yaml handles Pydantic model_dump."""
    mock_config = MagicMock(spec=["model_dump"])
    mock_config.model_dump.return_value = {"key": "value"}

    with patch("orchard.core.io.serialization._persist_yaml_atomic") as mock_persist:
        yaml_path = Path("/mock/test.yaml")

        save_config_as_yaml(mock_config, yaml_path)

        # Should call model_dump
        assert mock_config.model_dump.called
        mock_persist.assert_called_once()


@pytest.mark.unit
def test_save_config_as_yaml_with_dump_portable():
    """Test save_config_as_yaml prioritizes dump_portable over model_dump."""
    mock_config = MagicMock()
    mock_config.dump_portable.return_value = {"portable": True}
    mock_config.model_dump.return_value = {"portable": False}

    with patch("orchard.core.io.serialization._persist_yaml_atomic") as mock_persist:
        yaml_path = Path("/mock/test.yaml")

        save_config_as_yaml(mock_config, yaml_path)

        mock_config.dump_portable.assert_called_once()
        mock_config.model_dump.assert_not_called()

        mock_persist.assert_called_once_with({"portable": True}, yaml_path)


@pytest.mark.unit
def test_save_config_as_yaml_with_paths(tmp_path):
    """Test save_config_as_yaml converts Path objects to strings."""
    config = {"output_dir": Path("/mock/outputs"), "log": Path("/mock/log.txt")}
    yaml_path = tmp_path / "config.yaml"

    save_config_as_yaml(config, yaml_path)

    with open(yaml_path, "r") as f:
        loaded = yaml.safe_load(f)

    assert loaded["output_dir"] == "/mock/outputs"
    assert loaded["log"] == "/mock/log.txt"


@pytest.mark.unit
def test_save_config_as_yaml_creates_directory(tmp_path):
    """Test save_config_as_yaml creates parent directories if needed."""
    config = {"test": "value"}
    yaml_path = tmp_path / "nested" / "dir" / "config.yaml"

    save_config_as_yaml(config, yaml_path)

    assert yaml_path.exists()
    assert yaml_path.parent.exists()


@pytest.mark.unit
def test_save_config_as_yaml_invalid_data():
    """Test save_config_as_yaml raises ValueError for unserializable data."""
    mock_config = MagicMock(spec=["model_dump"])
    mock_config.model_dump.side_effect = Exception("Cannot serialize")

    yaml_path = Path("/mock/test.yaml")

    with pytest.raises(ValueError):
        save_config_as_yaml(mock_config, yaml_path)


@pytest.mark.unit
def test_save_config_as_yaml_io_error(tmp_path):
    """Test that save_config_as_yaml logs and raises an OSError / PermissionError."""

    config = {"model": "resnet"}

    yaml_path = tmp_path / "config.yaml"

    with (
        patch("orchard.core.io.serialization._persist_yaml_atomic") as mock_persist,
        patch("orchard.core.io.serialization.logging.getLogger") as mock_logger,
    ):
        mock_persist.side_effect = OSError("Disk is full")
        mock_logger.return_value = MagicMock()

        with pytest.raises(OSError, match="Disk is full"):
            save_config_as_yaml(config, yaml_path)

        mock_logger.return_value.error.assert_called_once()


# LOAD CONFIG FROM YAML
@pytest.mark.unit
def test_load_config_from_yaml_success(tmp_path):
    """Test load_config_from_yaml loads valid YAML file."""
    config = {"model": "efficientnet", "batch_size": 32}
    yaml_path = tmp_path / "config.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    loaded = load_config_from_yaml(yaml_path)

    assert loaded == config


@pytest.mark.unit
def test_load_config_from_yaml_file_not_found():
    """Test load_config_from_yaml raises FileNotFoundError for missing file."""
    yaml_path = Path("/nonexistent/config.yaml")

    with pytest.raises(FileNotFoundError, match="not found"):
        load_config_from_yaml(yaml_path)


@pytest.mark.unit
def test_load_config_from_yaml_complex_structure(tmp_path):
    """Test load_config_from_yaml handles nested structures."""
    config = {
        "model": {"name": "vit", "pretrained": True},
        "training": {"epochs": 100, "lr": 0.001},
        "paths": ["/data", "/outputs"],
    }
    yaml_path = tmp_path / "config.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    loaded = load_config_from_yaml(yaml_path)

    assert loaded == config
    assert loaded["model"]["name"] == "vit"


# PERSIST YAML ATOMIC
@pytest.mark.unit
def test_persist_yaml_atomic_creates_file(tmp_path):
    """Test _persist_yaml_atomic creates file and writes data."""
    data = {"key": "value"}
    yaml_path = tmp_path / "test.yaml"

    _persist_yaml_atomic(data, yaml_path)

    assert yaml_path.exists()
    with open(yaml_path, "r") as f:
        loaded = yaml.safe_load(f)
    assert loaded == data


@pytest.mark.unit
def test_persist_yaml_atomic_creates_parent_dir(tmp_path):
    """Test _persist_yaml_atomic creates parent directories."""
    data = {"test": "data"}
    yaml_path = tmp_path / "nested" / "dir" / "config.yaml"

    _persist_yaml_atomic(data, yaml_path)

    assert yaml_path.parent.exists()
    assert yaml_path.exists()


# DUMP REQUIREMENTS
@pytest.mark.unit
def test_dump_requirements_writes_file(tmp_path):
    """Test dump_requirements creates a file with Python version header."""
    output = tmp_path / "requirements.txt"
    dump_requirements(output)

    assert output.exists()
    content = output.read_text()
    assert content.startswith("# Python")


@pytest.mark.unit
def test_dump_requirements_handles_subprocess_failure(tmp_path):
    """Test dump_requirements gracefully handles subprocess failure."""
    output = tmp_path / "requirements.txt"

    with patch("subprocess.run", side_effect=OSError("mock pip failure")):
        dump_requirements(output)  # should not raise

    assert not output.exists()


# AUDIT SAVER
@pytest.mark.unit
def test_audit_saver_delegates_to_free_functions(tmp_path):
    """Test AuditSaver.save_config delegates to save_config_as_yaml."""
    from orchard.core.io.serialization import AuditSaver

    saver = AuditSaver()
    yaml_path = tmp_path / "config.yaml"
    data = {"key": "value"}

    result = saver.save_config(data=data, yaml_path=yaml_path)

    assert result == yaml_path
    assert yaml_path.exists()
    assert yaml.safe_load(yaml_path.read_text()) == {"key": "value"}


@pytest.mark.unit
def test_audit_saver_dump_requirements_delegates(tmp_path):
    """Test AuditSaver.dump_requirements delegates to dump_requirements."""
    from orchard.core.io.serialization import AuditSaver

    saver = AuditSaver()
    output = tmp_path / "requirements.txt"

    with patch("orchard.core.io.serialization.dump_requirements") as mock_dump:
        saver.dump_requirements(output)
        mock_dump.assert_called_once_with(output)


# PERSIST YAML ATOMIC: FORMAT VERIFICATION
@pytest.mark.unit
def test_persist_yaml_atomic_uses_block_style(tmp_path):
    """Test _persist_yaml_atomic writes block style (default_flow_style=False)."""
    data = {"nested": {"a": 1, "b": 2}}
    yaml_path = tmp_path / "test.yaml"

    _persist_yaml_atomic(data, yaml_path)

    content = yaml_path.read_text()
    # Block style uses newlines and indentation, not {a: 1, b: 2}
    assert "{" not in content
    assert "nested:" in content


@pytest.mark.unit
def test_persist_yaml_atomic_preserves_key_order(tmp_path):
    """Test _persist_yaml_atomic preserves insertion order (sort_keys=False)."""
    from collections import OrderedDict

    data = OrderedDict([("zebra", 1), ("apple", 2), ("mango", 3)])
    yaml_path = tmp_path / "test.yaml"

    _persist_yaml_atomic(data, yaml_path)

    content = yaml_path.read_text()
    z_pos = content.index("zebra")
    a_pos = content.index("apple")
    m_pos = content.index("mango")
    assert z_pos < a_pos < m_pos


@pytest.mark.unit
def test_persist_yaml_atomic_indent_is_4(tmp_path):
    """Test _persist_yaml_atomic uses 4-space indentation."""
    data = {"parent": {"child": "value"}}
    yaml_path = tmp_path / "test.yaml"

    _persist_yaml_atomic(data, yaml_path)

    content = yaml_path.read_text()
    # 4-space indented child key
    assert "    child: value" in content


@pytest.mark.unit
def test_persist_yaml_atomic_allows_unicode(tmp_path):
    """Test _persist_yaml_atomic writes unicode chars directly (allow_unicode=True)."""
    data = {"name": "café résumé"}
    yaml_path = tmp_path / "test.yaml"

    _persist_yaml_atomic(data, yaml_path)

    content = yaml_path.read_text()
    # With allow_unicode=True, unicode chars are written directly
    assert "café" in content
    assert "résumé" in content


@pytest.mark.unit
def test_persist_yaml_atomic_calls_flush_and_fsync(tmp_path):
    """Test _persist_yaml_atomic calls flush() and os.fsync()."""
    import os as _os

    data = {"key": "value"}
    yaml_path = tmp_path / "test.yaml"

    flush_called = False
    fsync_called = False
    original_fsync = _os.fsync

    class TrackingFile:
        """Wrapper to track flush/fsync calls."""

        def __init__(self, real_file):
            self._real = real_file

        def write(self, data):
            return self._real.write(data)

        def flush(self):
            nonlocal flush_called
            flush_called = True
            return self._real.flush()

        def fileno(self):
            return self._real.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._real.__exit__(*args)

    original_open = open

    def patched_open(path, mode="r", **kwargs):
        f = original_open(path, mode, **kwargs)
        if "w" in mode:
            return TrackingFile(f)
        return f

    def tracking_fsync(fd):
        nonlocal fsync_called
        fsync_called = True
        return original_fsync(fd)

    with (
        patch("builtins.open", side_effect=patched_open),
        patch("os.fsync", side_effect=tracking_fsync),
    ):
        _persist_yaml_atomic(data, yaml_path)

    assert flush_called, "flush() was not called"
    assert fsync_called, "os.fsync() was not called"


@pytest.mark.unit
def test_persist_yaml_atomic_creates_nested_parents(tmp_path):
    """Test _persist_yaml_atomic creates deeply nested parent dirs (parents=True)."""
    data = {"k": "v"}
    yaml_path = tmp_path / "a" / "b" / "c" / "d" / "config.yaml"

    _persist_yaml_atomic(data, yaml_path)

    assert yaml_path.exists()
    assert yaml_path.parent.exists()


@pytest.mark.unit
def test_persist_yaml_atomic_existing_dir_ok(tmp_path):
    """Test _persist_yaml_atomic doesn't fail on existing dir (exist_ok=True)."""
    data = {"k": "v"}
    yaml_path = tmp_path / "config.yaml"

    # Write twice to same location — second should not fail
    _persist_yaml_atomic(data, yaml_path)
    _persist_yaml_atomic(data, yaml_path)

    assert yaml_path.exists()


@pytest.mark.unit
def test_persist_yaml_atomic_writes_utf8(tmp_path):
    """Test _persist_yaml_atomic writes file with utf-8 encoding."""
    data = {"emoji": "✓", "accents": "àèìòù"}
    yaml_path = tmp_path / "test.yaml"

    _persist_yaml_atomic(data, yaml_path)

    raw_bytes = yaml_path.read_bytes()
    # Verify the file is valid UTF-8
    content = raw_bytes.decode("utf-8")
    assert "✓" in content
    assert "àèìòù" in content


# DUMP REQUIREMENTS: DETAILED VERIFICATION
@pytest.mark.unit
def test_dump_requirements_subprocess_args(tmp_path):
    """Test dump_requirements calls subprocess.run with correct arguments."""
    import sys

    output = tmp_path / "requirements.txt"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="package==1.0\n")
        dump_requirements(output)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        # Verify the command
        assert call_args.args[0] == [sys.executable, "-m", "pip", "freeze", "--local"]
        # Verify kwargs
        assert call_args.kwargs["capture_output"] is True
        assert call_args.kwargs["text"] is True
        assert call_args.kwargs["timeout"] == 30


@pytest.mark.unit
def test_dump_requirements_header_format(tmp_path):
    """Test dump_requirements writes correct Python version header."""
    import sys

    output = tmp_path / "requirements.txt"
    dump_requirements(output)

    content = output.read_text()
    expected_version = sys.version.split()[0]
    assert content.startswith(f"# Python {expected_version}\n")


@pytest.mark.unit
def test_dump_requirements_includes_stdout(tmp_path):
    """Test dump_requirements includes pip freeze output after header."""
    output = tmp_path / "requirements.txt"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="torch==2.0.0\nnumpy==1.24.0\n")
        dump_requirements(output)

    content = output.read_text()
    assert "torch==2.0.0" in content
    assert "numpy==1.24.0" in content


@pytest.mark.unit
def test_dump_requirements_writes_utf8(tmp_path):
    """Test dump_requirements writes file with utf-8 encoding."""
    output = tmp_path / "requirements.txt"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="pkg==1.0\n")
        dump_requirements(output)

    # Verify it's valid UTF-8
    raw = output.read_bytes()
    raw.decode("utf-8")  # Should not raise


@pytest.mark.unit
def test_dump_requirements_handles_timeout(tmp_path):
    """Test dump_requirements handles TimeoutExpired gracefully."""
    import subprocess

    output = tmp_path / "requirements.txt"

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=30),
    ):
        dump_requirements(output)  # Should not raise

    assert not output.exists()


# SAVE CONFIG AS YAML: MODEL_DUMP MODE
@pytest.mark.unit
def test_save_config_as_yaml_model_dump_uses_json_mode(tmp_path):
    """Test save_config_as_yaml calls model_dump with mode='json'."""
    mock_config = MagicMock(spec=["model_dump"])
    mock_config.model_dump.return_value = {"key": "value"}

    yaml_path = tmp_path / "config.yaml"
    save_config_as_yaml(mock_config, yaml_path)

    mock_config.model_dump.assert_called_once_with(mode="json")


@pytest.mark.unit
def test_save_config_as_yaml_returns_path(tmp_path):
    """Test save_config_as_yaml returns the yaml_path."""
    yaml_path = tmp_path / "config.yaml"
    result = save_config_as_yaml({"a": 1}, yaml_path)

    assert result == yaml_path
    assert result is yaml_path


@pytest.mark.unit
def test_save_config_as_yaml_raw_dict_passthrough(tmp_path):
    """Test save_config_as_yaml passes raw dict directly (no dump_portable/model_dump)."""
    data = {"raw": True, "nested": {"x": 1}}
    yaml_path = tmp_path / "config.yaml"

    save_config_as_yaml(data, yaml_path)

    loaded = yaml.safe_load(yaml_path.read_text())
    assert loaded == data


@pytest.mark.unit
def test_save_config_as_yaml_sanitizes_paths_in_model_dump(tmp_path):
    """Test save_config_as_yaml sanitizes Path objects from model_dump output."""
    mock_config = MagicMock(spec=["model_dump"])
    mock_config.model_dump.return_value = {"path": Path("/some/path"), "val": 42}

    yaml_path = tmp_path / "config.yaml"
    save_config_as_yaml(mock_config, yaml_path)

    loaded = yaml.safe_load(yaml_path.read_text())
    assert loaded["path"] == "/some/path"
    assert isinstance(loaded["path"], str)


@pytest.mark.unit
def test_save_config_as_yaml_error_message_contains_cause():
    """Test save_config_as_yaml ValueError message includes original error."""
    mock_config = MagicMock(spec=["model_dump"])
    mock_config.model_dump.side_effect = RuntimeError("custom error XYZ")

    with pytest.raises(ValueError, match="custom error XYZ"):
        save_config_as_yaml(mock_config, Path("/fake/path.yaml"))


# LOAD CONFIG FROM YAML: EDGE CASES
@pytest.mark.unit
def test_load_config_from_yaml_error_message_contains_path():
    """Test FileNotFoundError message includes the path."""
    yaml_path = Path("/nonexistent/specific_file.yaml")

    with pytest.raises(FileNotFoundError, match="specific_file.yaml"):
        load_config_from_yaml(yaml_path)


@pytest.mark.unit
def test_load_config_from_yaml_reads_utf8(tmp_path):
    """Test load_config_from_yaml reads UTF-8 encoded files correctly."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("name: café\n", encoding="utf-8")

    loaded = load_config_from_yaml(yaml_path)
    assert loaded["name"] == "café"


@pytest.mark.unit
def test_load_config_from_yaml_existing_file_does_not_raise(tmp_path):
    """Test load_config_from_yaml does not raise for existing file (not negated check)."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("key: value\n")

    result = load_config_from_yaml(yaml_path)
    assert result == {"key": "value"}


# SANITIZE FOR YAML: EDGE CASES
@pytest.mark.unit
def test_sanitize_for_yaml_empty_dict():
    """Test _sanitize_for_yaml handles empty dict."""
    assert _sanitize_for_yaml({}) == {}


@pytest.mark.unit
def test_sanitize_for_yaml_empty_list():
    """Test _sanitize_for_yaml handles empty list."""
    assert _sanitize_for_yaml([]) == []


@pytest.mark.unit
def test_sanitize_for_yaml_deeply_nested():
    """Test _sanitize_for_yaml recurses through deeply nested structures."""
    data = {"a": {"b": {"c": {"d": Path("/deep")}}}}
    result = _sanitize_for_yaml(data)
    assert result["a"]["b"]["c"]["d"] == "/deep"
    assert isinstance(result["a"]["b"]["c"]["d"], str)


@pytest.mark.unit
def test_sanitize_for_yaml_list_of_paths():
    """Test _sanitize_for_yaml converts all paths in a list."""
    data = [Path("/a"), Path("/b"), Path("/c")]
    result = _sanitize_for_yaml(data)
    assert result == ["/a", "/b", "/c"]
    assert all(isinstance(x, str) for x in result)


@pytest.mark.unit
def test_sanitize_for_yaml_mixed_list():
    """Test _sanitize_for_yaml handles mixed types in list."""
    data = [1, "text", Path("/path"), {"key": Path("/val")}, [Path("/nested")]]
    result = _sanitize_for_yaml(data)
    assert result == [1, "text", "/path", {"key": "/val"}, ["/nested"]]


@pytest.mark.unit
def test_sanitize_for_yaml_non_path_non_collection_passthrough():
    """Test _sanitize_for_yaml returns non-Path scalars unchanged."""
    assert _sanitize_for_yaml(42) == 42
    assert _sanitize_for_yaml(3.14) == 3.14
    assert _sanitize_for_yaml("hello") == "hello"
    assert _sanitize_for_yaml(True) is True
    assert _sanitize_for_yaml(None) is None


@pytest.mark.unit
def test_sanitize_for_yaml_tuple_returns_list():
    """Test _sanitize_for_yaml converts tuple to list (type check)."""
    result = _sanitize_for_yaml((1, 2, 3))
    assert isinstance(result, list)
    assert not isinstance(result, tuple)


# AUDIT SAVER: FULL INTEGRATION
@pytest.mark.unit
def test_audit_saver_save_config_returns_correct_path(tmp_path):
    """Test AuditSaver.save_config returns the yaml_path."""
    from orchard.core.io.serialization import AuditSaver

    saver = AuditSaver()
    yaml_path = tmp_path / "config.yaml"

    result = saver.save_config(data={"k": "v"}, yaml_path=yaml_path)
    assert result == yaml_path


@pytest.mark.unit
def test_audit_saver_dump_requirements_actually_writes(tmp_path):
    """Test AuditSaver.dump_requirements writes a file."""
    from orchard.core.io.serialization import AuditSaver

    saver = AuditSaver()
    output = tmp_path / "requirements.txt"

    saver.dump_requirements(output)
    assert output.exists()
    assert output.read_text().startswith("# Python")


# LOGGER.ERROR MUTANT KILLERS — save_config_as_yaml serialization error
@pytest.mark.unit
def test_save_config_as_yaml_serialization_error_logs_exact_message():
    """Kill mutants 24-30: assert exact logger.error message for serialization failures."""
    import logging

    mock_config = MagicMock(spec=["model_dump"])
    mock_config.model_dump.side_effect = RuntimeError("boom")

    logger = logging.getLogger("OrchardML")
    with patch.object(logger, "error") as mock_error:
        with pytest.raises(ValueError):
            save_config_as_yaml(mock_config, Path("/fake.yaml"))

    mock_error.assert_called_once_with(
        "Serialization failed: object structure is incompatible. Error: %s",
        mock_config.model_dump.side_effect,
    )


# LOGGER.ERROR MUTANT KILLERS — save_config_as_yaml IO error
@pytest.mark.unit
def test_save_config_as_yaml_io_error_logs_exact_message(tmp_path):
    """Kill mutants 36-44: assert exact logger.error message for IO errors."""
    import logging

    yaml_path = tmp_path / "config.yaml"
    logger = logging.getLogger("OrchardML")

    with (
        patch("orchard.core.io.serialization._persist_yaml_atomic") as mock_persist,
        patch.object(logger, "error") as mock_error,
    ):
        mock_persist.side_effect = OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            save_config_as_yaml({"a": 1}, yaml_path)

    mock_error.assert_called_once_with(
        "IO Error: Could not write YAML to %s. Error: %s",
        yaml_path,
        mock_persist.side_effect,
    )


# LOGGER.ERROR MUTANT KILLERS — dump_requirements error
@pytest.mark.unit
def test_dump_requirements_error_logs_exact_message(tmp_path):
    """Kill mutants 32-38: assert exact logger.error message for dump_requirements failures."""
    import logging

    output = tmp_path / "requirements.txt"
    logger = logging.getLogger("OrchardML")

    err = OSError("pip not found")
    with (
        patch("subprocess.run", side_effect=err),
        patch.object(logger, "error") as mock_error,
    ):
        dump_requirements(output)

    mock_error.assert_called_once_with("Failed to dump requirements: %s", err)


# LOGGER_NAME MUTANT KILLERS
@pytest.mark.unit
def test_save_config_as_yaml_uses_orchard_logger():
    """Kill save_config mutmut_2: assert getLogger called with LOGGER_NAME."""
    mock_config = MagicMock(spec=["model_dump"])
    mock_config.model_dump.side_effect = RuntimeError("x")

    with patch("orchard.core.io.serialization.logging.getLogger") as mock_gl:
        mock_gl.return_value = MagicMock()
        with pytest.raises(ValueError):
            save_config_as_yaml(mock_config, Path("/fake.yaml"))

    mock_gl.assert_called_once_with("OrchardML")


@pytest.mark.unit
def test_dump_requirements_uses_orchard_logger(tmp_path):
    """Kill dump_requirements mutmut_2: assert getLogger called with LOGGER_NAME."""
    output = tmp_path / "requirements.txt"

    with (
        patch("subprocess.run", side_effect=OSError("fail")),
        patch("orchard.core.io.serialization.logging.getLogger") as mock_gl,
    ):
        mock_gl.return_value = MagicMock()
        dump_requirements(output)

    mock_gl.assert_called_once_with("OrchardML")


# DUMP_PORTABLE STRING MUTANT KILLERS
@pytest.mark.unit
def test_save_config_as_yaml_dump_portable_exact_attr_name(tmp_path):
    """Kill mutants 7-8: ensure hasattr checks exact 'dump_portable' string."""

    class PortableConfig:
        def dump_portable(self):
            return {"from_portable": True}

    yaml_path = tmp_path / "config.yaml"
    save_config_as_yaml(PortableConfig(), yaml_path)

    loaded = yaml.safe_load(yaml_path.read_text())
    assert loaded == {"from_portable": True}


@pytest.mark.unit
def test_save_config_as_yaml_no_dump_portable_falls_to_model_dump(tmp_path):
    """Kill mutant 7-8: object WITHOUT dump_portable uses model_dump path."""

    class ModelConfig:
        def model_dump(self, mode=None):
            return {"from_model": True}

    yaml_path = tmp_path / "config.yaml"
    save_config_as_yaml(ModelConfig(), yaml_path)

    loaded = yaml.safe_load(yaml_path.read_text())
    assert loaded == {"from_model": True}


# PERSIST YAML ATOMIC: STRONGER FORMAT TESTS
@pytest.mark.unit
def test_persist_yaml_atomic_sort_keys_false(tmp_path):
    """Kill mutants 26, 30: sort_keys=False must preserve insertion order, not sort."""
    data = {"z_last": 1, "a_first": 2, "m_middle": 3}
    yaml_path = tmp_path / "test.yaml"

    _persist_yaml_atomic(data, yaml_path)

    content = yaml_path.read_text()
    lines = [ln.strip() for ln in content.strip().splitlines()]
    assert lines[0].startswith("z_last")
    assert lines[1].startswith("a_first")
    assert lines[2].startswith("m_middle")


@pytest.mark.unit
def test_persist_yaml_atomic_indent_exactly_4(tmp_path):
    """Kill mutant 31: indent must be exactly 4, not 5."""
    data = {"parent": {"child": {"grandchild": "val"}}}
    yaml_path = tmp_path / "test.yaml"

    _persist_yaml_atomic(data, yaml_path)

    content = yaml_path.read_text()
    # 4-space indent at level 1
    assert "    child:" in content
    # 8-space indent at level 2
    assert "        grandchild: val" in content
    # Must NOT have 5 or 10-space indent
    assert "     child:" not in content


# ENCODING MUTANT KILLERS — mock open to verify encoding kwarg
@pytest.mark.unit
def test_persist_yaml_atomic_passes_utf8_encoding(tmp_path):
    """Kill mutants 9, 12: verify open() is called with encoding='utf-8'."""
    data = {"k": "v"}
    yaml_path = tmp_path / "test.yaml"
    original_open = open

    open_calls = []

    def tracking_open(*args, **kwargs):
        open_calls.append(kwargs)
        return original_open(*args, **kwargs)

    with patch("builtins.open", side_effect=tracking_open):
        _persist_yaml_atomic(data, yaml_path)

    write_calls = [c for c in open_calls if c.get("encoding") is not None or "encoding" in c]
    assert any(c.get("encoding") == "utf-8" for c in write_calls)


@pytest.mark.unit
def test_load_config_from_yaml_passes_utf8_encoding(tmp_path):
    """Kill mutants load_5, load_8: verify open() with encoding='utf-8'."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("key: value\n")

    original_open = open
    open_calls = []

    def tracking_open(*args, **kwargs):
        open_calls.append((args, kwargs))
        return original_open(*args, **kwargs)

    with patch("builtins.open", side_effect=tracking_open):
        load_config_from_yaml(yaml_path)

    # Find the call with encoding kwarg
    assert any(kw.get("encoding") == "utf-8" for _, kw in open_calls)


@pytest.mark.unit
def test_dump_requirements_writes_utf8_encoding(tmp_path):
    """Kill mutants dump_26, dump_28: verify write_text uses encoding='utf-8'."""
    output = tmp_path / "requirements.txt"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="pkg==1.0\n")

        with patch.object(type(output), "write_text", wraps=output.write_text) as mock_wt:
            dump_requirements(output)
            mock_wt.assert_called_once()
            call_kwargs = mock_wt.call_args
            assert call_kwargs.kwargs.get("encoding") == "utf-8" or (
                len(call_kwargs.args) >= 2 and "utf-8" in str(call_kwargs)
            )


# DUMP GIT INFO
@pytest.mark.unit
def test_dump_git_info_writes_commit_and_branch(tmp_path):
    """Test dump_git_info writes commit hash, short hash, branch, and dirty status."""

    output = tmp_path / "git_info.txt"

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0)
        if cmd == ["git", "rev-parse", "HEAD"]:
            result.stdout = "abc123def456\n"
        elif cmd == ["git", "rev-parse", "--short", "HEAD"]:
            result.stdout = "abc123d\n"
        elif cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            result.stdout = "main\n"
        elif cmd == ["git", "status", "--porcelain"]:
            result.stdout = " M file.py\n"
        else:
            result.returncode = 1
            result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        dump_git_info(output)

    assert output.exists()
    content = output.read_text()
    assert "commit: abc123def456" in content
    assert "commit_short: abc123d" in content
    assert "branch: main" in content
    assert "dirty: True" in content

    # Kill join-separator mutant: each line must be exactly one key-value pair
    lines = content.strip().splitlines()
    assert len(lines) == 4
    assert lines[0] == "commit: abc123def456"
    assert lines[1] == "commit_short: abc123d"
    assert lines[2] == "branch: main"
    assert lines[3] == "dirty: True"


@pytest.mark.unit
def test_dump_git_info_clean_working_tree(tmp_path):
    """Test dump_git_info reports dirty: False for clean working tree."""
    output = tmp_path / "git_info.txt"

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0)
        if cmd == ["git", "rev-parse", "HEAD"]:
            result.stdout = "deadbeef\n"
        elif cmd == ["git", "rev-parse", "--short", "HEAD"]:
            result.stdout = "deadbee\n"
        elif cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            result.stdout = "feature\n"
        elif cmd == ["git", "status", "--porcelain"]:
            result.stdout = "\n"
        else:
            result.returncode = 1
            result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        dump_git_info(output)

    content = output.read_text()
    assert "dirty: False" in content


@pytest.mark.unit
def test_dump_git_info_handles_file_not_found(tmp_path):
    """Test dump_git_info gracefully handles git not installed (FileNotFoundError)."""
    output = tmp_path / "git_info.txt"

    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        dump_git_info(output)  # Should not raise

    assert not output.exists()


@pytest.mark.unit
def test_dump_git_info_handles_timeout(tmp_path):
    """Test dump_git_info gracefully handles subprocess timeout."""
    import subprocess

    output = tmp_path / "git_info.txt"

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5),
    ):
        dump_git_info(output)  # Should not raise

    assert not output.exists()


@pytest.mark.unit
def test_dump_git_info_handles_os_error(tmp_path):
    """Test dump_git_info gracefully handles OSError."""
    output = tmp_path / "git_info.txt"

    with patch("subprocess.run", side_effect=OSError("permission denied")):
        dump_git_info(output)  # Should not raise

    assert not output.exists()


@pytest.mark.unit
def test_dump_git_info_no_output_when_all_commands_fail(tmp_path):
    """Test dump_git_info writes nothing when all git commands fail (returncode != 0)."""
    output = tmp_path / "git_info.txt"

    def fake_run(cmd, **kwargs):
        return MagicMock(returncode=1, stdout="")

    with patch("subprocess.run", side_effect=fake_run):
        dump_git_info(output)

    assert not output.exists()


@pytest.mark.unit
def test_dump_git_info_partial_git_output(tmp_path):
    """Test dump_git_info handles partial results (e.g. commit ok, branch fails)."""
    output = tmp_path / "git_info.txt"

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        if cmd == ["git", "rev-parse", "HEAD"]:
            result.returncode = 0
            result.stdout = "abc123\n"
        elif cmd == ["git", "rev-parse", "--short", "HEAD"]:
            result.returncode = 0
            result.stdout = "abc\n"
        else:
            result.returncode = 1
            result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        dump_git_info(output)

    content = output.read_text()
    assert "commit: abc123" in content
    assert "commit_short: abc" in content
    assert "branch" not in content
    assert "dirty" not in content


@pytest.mark.unit
def test_dump_git_info_writes_utf8(tmp_path):
    """Test dump_git_info writes file with utf-8 encoding."""
    output = tmp_path / "git_info.txt"

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0)
        if cmd == ["git", "rev-parse", "HEAD"]:
            result.stdout = "abc123\n"
        elif cmd == ["git", "rev-parse", "--short", "HEAD"]:
            result.stdout = "abc\n"
        elif cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            result.stdout = "main\n"
        elif cmd == ["git", "status", "--porcelain"]:
            result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        with patch.object(type(output), "write_text", wraps=output.write_text) as mock_wt:
            dump_git_info(output)
            mock_wt.assert_called_once()
            call_kwargs = mock_wt.call_args
            assert call_kwargs.kwargs.get("encoding") == "utf-8" or (
                len(call_kwargs.args) >= 2 and "utf-8" in str(call_kwargs)
            )


@pytest.mark.unit
def test_dump_git_info_uses_orchard_logger(tmp_path):
    """Test dump_git_info uses the OrchardML logger."""
    import logging

    output = tmp_path / "git_info.txt"
    logger = logging.getLogger("OrchardML")

    with (
        patch("subprocess.run", side_effect=OSError("fail")),
        patch.object(logger, "debug") as mock_debug,
    ):
        dump_git_info(output)

    mock_debug.assert_called_once()


@pytest.mark.unit
def test_dump_git_info_error_logs_exact_message(tmp_path):
    """Kill log string mutants: assert exact logger.debug message for git errors."""
    import logging

    output = tmp_path / "git_info.txt"
    logger = logging.getLogger("OrchardML")
    err = FileNotFoundError("git not found")

    with (
        patch("subprocess.run", side_effect=err),
        patch.object(logger, "debug") as mock_debug,
    ):
        dump_git_info(output)

    mock_debug.assert_called_once_with("Could not capture git info: %s", err)


@pytest.mark.unit
def test_dump_git_info_subprocess_args(tmp_path):
    """Test dump_git_info calls subprocess.run with correct timeout and kwargs."""
    output = tmp_path / "git_info.txt"
    calls = []

    def tracking_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return MagicMock(returncode=1, stdout="")

    with patch("subprocess.run", side_effect=tracking_run):
        dump_git_info(output)

    assert len(calls) == 4
    for cmd, kwargs in calls:
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 5


@pytest.mark.unit
def test_dump_git_info_content_ends_with_newline(tmp_path):
    """Test dump_git_info output ends with a trailing newline."""
    output = tmp_path / "git_info.txt"

    def fake_run(cmd, **kwargs):
        result = MagicMock(returncode=0)
        if cmd == ["git", "rev-parse", "HEAD"]:
            result.stdout = "abc\n"
        elif cmd == ["git", "rev-parse", "--short", "HEAD"]:
            result.stdout = "ab\n"
        elif cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            result.stdout = "main\n"
        elif cmd == ["git", "status", "--porcelain"]:
            result.stdout = ""
        return result

    with patch("subprocess.run", side_effect=fake_run):
        dump_git_info(output)

    content = output.read_text()
    assert content.endswith("\n")


# AUDIT SAVER: DUMP GIT INFO DELEGATION
@pytest.mark.unit
def test_audit_saver_dump_git_info_delegates(tmp_path):
    """Test AuditSaver.dump_git_info delegates to module-level dump_git_info."""
    from orchard.core.io.serialization import AuditSaver

    saver = AuditSaver()
    output = tmp_path / "git_info.txt"

    with patch("orchard.core.io.serialization.dump_git_info") as mock_dump:
        saver.dump_git_info(output)
        mock_dump.assert_called_once_with(output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
