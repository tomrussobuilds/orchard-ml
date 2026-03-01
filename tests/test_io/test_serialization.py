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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
