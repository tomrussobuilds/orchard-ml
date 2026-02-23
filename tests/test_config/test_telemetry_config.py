"""
Test Suite for TelemetryConfig.

Tests filesystem resolution, logging configuration,
and portable path serialization.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orchard.core.config import TelemetryConfig
from orchard.core.paths import PROJECT_ROOT


# TELEMETRY CONFIG: DEFAULTS
@pytest.mark.unit
def test_telemetry_config_defaults():
    """Test TelemetryConfig with default values."""
    config = TelemetryConfig()

    assert str(config.data_dir) == "./dataset"
    assert str(config.output_dir) == "./outputs"
    assert config.log_interval == 10
    assert config.log_level == "INFO"


@pytest.mark.unit
def test_telemetry_config_custom_values():
    """Test TelemetryConfig with custom parameters."""
    config = TelemetryConfig(
        data_dir=Path("./custom_data"),
        output_dir=Path("./custom_out"),
        log_level="DEBUG",
        log_interval=5,
    )

    assert config.data_dir == Path("./custom_data").resolve()
    assert config.output_dir == Path("./custom_out").resolve()
    assert config.log_level == "DEBUG"
    assert config.log_interval == 5


# TELEMETRY CONFIG: VALIDATION
@pytest.mark.unit
def test_log_interval_bounds():
    """Test log_interval must be in [1, 1000]."""

    config = TelemetryConfig(log_interval=1)
    assert config.log_interval == 1

    config = TelemetryConfig(log_interval=1000)
    assert config.log_interval == 1000

    with pytest.raises(ValidationError):
        TelemetryConfig(log_interval=0)

    with pytest.raises(ValidationError):
        TelemetryConfig(log_interval=2000)


@pytest.mark.unit
def test_log_level_valid_values():
    """Test log_level accepts valid logging levels."""
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        config = TelemetryConfig(log_level=level)
        assert config.log_level == level


@pytest.mark.unit
def test_log_level_invalid_value():
    """Test log_level rejects invalid values."""
    with pytest.raises(ValidationError):
        TelemetryConfig(log_level="INVALID")

    with pytest.raises(ValidationError):
        TelemetryConfig(log_level="TRACE")


# TELEMETRY CONFIG: PATH RESOLUTION
@pytest.mark.unit
def test_path_sanitization():
    """Test paths are resolved to absolute forms."""
    config = TelemetryConfig(data_dir=Path("~/dataset"), output_dir=Path("./outputs"))

    assert config.data_dir.is_absolute()
    assert config.output_dir.is_absolute()


# TELEMETRY CONFIG: PORTABILITY
@pytest.mark.unit
def test_to_portable_dict():
    """Test to_portable_dict() converts to relative paths."""
    config = TelemetryConfig(data_dir=PROJECT_ROOT / "dataset", output_dir=PROJECT_ROOT / "outputs")

    portable = config.to_portable_dict()

    assert portable["data_dir"] == "./dataset"
    assert portable["output_dir"] == "./outputs"


@pytest.mark.unit
def test_to_portable_dict_absolute_outside_project():
    """Test to_portable_dict() preserves absolute paths outside PROJECT_ROOT."""
    external_path = Path("/mock/external_data")
    config = TelemetryConfig(data_dir=external_path)

    portable = config.to_portable_dict()

    assert str(external_path) in portable["data_dir"]


@pytest.mark.unit
def test_to_portable_dict_preserves_other_fields():
    """Test to_portable_dict() preserves non-path fields."""
    config = TelemetryConfig(log_level="DEBUG", log_interval=5)

    portable = config.to_portable_dict()

    assert portable["log_level"] == "DEBUG"
    assert portable["log_interval"] == 5


# TELEMETRY CONFIG: EMPTY YAML HANDLING
@pytest.mark.unit
def test_handle_empty_config():
    """Test empty YAML section (None) is handled correctly."""
    config = TelemetryConfig(**TelemetryConfig.handle_empty_config(None))

    assert str(config.data_dir) == "./dataset"
    assert config.log_level == "INFO"


@pytest.mark.unit
def test_handle_empty_config_with_values():
    """Test handle_empty_config passes through non-None values."""
    data = {"log_level": "DEBUG", "log_interval": 20}
    result = TelemetryConfig.handle_empty_config(data)

    assert result == data


# TELEMETRY CONFIG: IMMUTABILITY
@pytest.mark.unit
def test_config_is_frozen():
    """Test TelemetryConfig is immutable after creation."""
    config = TelemetryConfig()

    with pytest.raises(ValidationError):
        config.log_level = "DEBUG"


@pytest.mark.unit
def test_config_forbids_extra_fields():
    """Test TelemetryConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        TelemetryConfig(unknown_field="value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
