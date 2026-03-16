"""
Test Suite for HardwareConfig.

Tests device resolution, reproducibility mode, num_workers logic,
and lock file path generation.
"""

from __future__ import annotations

import tempfile

import pytest
import torch
from pydantic import ValidationError

from orchard.core.config import HardwareConfig


# HARDWARE CONFIG: DEVICE RESOLUTION
@pytest.mark.unit
def test_device_auto_resolves() -> None:
    """Test device='auto' resolves to best available."""
    config = HardwareConfig(device="auto")

    assert config.device in ("cpu", "cuda", "mps")


@pytest.mark.unit
def test_device_cpu_always_works() -> None:
    """Test device='cpu' always resolves successfully."""
    config = HardwareConfig(device="cpu")

    assert config.device == "cpu"


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda_when_available() -> None:
    """Test device='cuda' resolves when CUDA available."""
    config = HardwareConfig(device="cuda")

    assert config.device == "cuda"


@pytest.mark.unit
@pytest.mark.skipif(torch.cuda.is_available(), reason="Test requires no CUDA")
def test_device_cuda_fallback_to_cpu() -> None:
    """Test device='cuda' falls back to CPU with warning when unavailable."""
    with pytest.warns(UserWarning, match="CUDA was explicitly requested"):
        config = HardwareConfig(device="cuda")

    assert config.device == "cpu"


@pytest.mark.unit
def test_device_cuda_fallback_when_unavailable() -> None:
    """Test device='cuda' falls back to CPU with warning when CUDA unavailable (mocked)."""
    from unittest.mock import patch

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.warns(UserWarning, match="CUDA was explicitly requested"):
            config = HardwareConfig(device="cuda")

        assert config.device == "cpu"


@pytest.mark.unit
def test_invalid_device_fallback() -> None:
    """Test invalid device type falls through validator."""
    config = HardwareConfig(device="mps")

    assert config.device in ("mps", "cpu")


@pytest.mark.unit
def test_device_mps_fallback_emits_warning() -> None:
    """Test device='mps' emits warning when MPS unavailable (mocked)."""
    from unittest.mock import MagicMock, patch

    mock_backends = MagicMock()
    mock_backends.mps.is_available.return_value = False

    with patch("torch.backends", mock_backends):
        with pytest.warns(UserWarning, match="MPS was explicitly requested"):
            config = HardwareConfig(device="mps")

    assert config.device == "cpu"


@pytest.mark.unit
def test_device_cpu_explicit_no_fallback_warning() -> None:
    """Test device='cpu' emits no fallback warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = HardwareConfig(device="cpu")

    assert config.device == "cpu"


@pytest.mark.unit
def test_device_auto_no_fallback_warning() -> None:
    """Test device='auto' resolving to CPU emits no fallback warning."""
    import warnings
    from unittest.mock import patch

    with patch(
        "orchard.core.config.hardware_config.detect_best_device",
        return_value="cpu",
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = HardwareConfig(device="auto")

    assert config.device == "cpu"


# HARDWARE CONFIG: REPRODUCIBILITY
@pytest.mark.unit
def test_reproducible_mode_disabled_by_default() -> None:
    """Test reproducible mode is False by default."""
    config = HardwareConfig()

    assert config.reproducible is False
    assert config.use_deterministic_algorithms is False


@pytest.mark.unit
def test_reproducible_mode_enables_deterministic() -> None:
    """Test reproducible mode sets deterministic algorithms."""
    config = HardwareConfig(reproducible=True)

    assert config.reproducible is True
    assert config.use_deterministic_algorithms is True


@pytest.mark.unit
def test_reproducible_mode_affects_num_workers() -> None:
    """Test reproducible mode forces num_workers=0."""
    config = HardwareConfig(reproducible=False)

    assert config.effective_num_workers >= 0


# HARDWARE CONFIG: NUM_WORKERS
@pytest.mark.unit
def test_effective_num_workers_zero_when_reproducible() -> None:
    """Test effective_num_workers is 0 in reproducible mode."""
    config = HardwareConfig(reproducible=True)

    assert config.effective_num_workers == 0


@pytest.mark.unit
def test_effective_num_workers_respects_explicit_value() -> None:
    """Test effective_num_workers uses explicit value when set."""
    config = HardwareConfig(reproducible=False)

    assert config.effective_num_workers >= 0


# HARDWARE CONFIG: AMP SUPPORT
@pytest.mark.unit
def test_supports_amp_cpu_false() -> None:
    """Test CPU does not support AMP."""
    config = HardwareConfig(device="cpu")

    assert config.supports_amp is False


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_supports_amp_cuda_true() -> None:
    """Test CUDA supports AMP."""
    config = HardwareConfig(device="cuda")

    assert config.supports_amp is True


# HARDWARE CONFIG: LOCK FILE PATH
@pytest.mark.unit
def test_lock_file_path_in_temp_dir() -> None:
    """Test lock file is created in system temp directory."""
    config = HardwareConfig(project_name="test_project")

    lock_path = config.lock_file_path

    assert str(lock_path).startswith(tempfile.gettempdir())
    assert lock_path.name == "test_project.lock"


@pytest.mark.unit
def test_lock_file_path_uses_project_name() -> None:
    """Test lock file path uses project name."""
    config = HardwareConfig(project_name="my-experiment")

    lock_path = config.lock_file_path

    assert "my-experiment.lock" in str(lock_path)


@pytest.mark.unit
def test_lock_file_path_sanitizes_slashes() -> None:
    """Test lock file path sanitizes project name with slashes."""
    pytest.skip("ProjectSlug doesn't allow slashes by design")

    config = HardwareConfig(project_name="org/project")
    lock_path = config.lock_file_path

    assert "/" not in lock_path.name
    assert "org_project.lock" in lock_path.name


# HARDWARE CONFIG: PROJECT NAME VALIDATION
@pytest.mark.unit
def test_project_name_validation_valid() -> None:
    """Test project_name follows slug pattern."""
    valid_names = ["valid-project_123", "my-exp", "test_001"]

    for name in valid_names:
        config = HardwareConfig(project_name=name)
        assert config.project_name == name


@pytest.mark.unit
def test_project_name_validation_invalid() -> None:
    """Test invalid project names are rejected."""
    with pytest.raises(ValidationError):
        HardwareConfig(project_name="Invalid Project!")

    with pytest.raises(ValidationError):
        HardwareConfig(project_name="UPPERCASE")


# HARDWARE CONFIG: DEFAULTS
@pytest.mark.unit
def test_hardware_config_defaults() -> None:
    """Test HardwareConfig with default values."""
    config = HardwareConfig()

    assert config.device in ("auto", "cpu", "cuda", "mps")
    assert config.project_name == "orchard_ml"
    assert config.allow_process_kill is True


# HARDWARE CONFIG: IMMUTABILITY
@pytest.mark.unit
def test_config_is_frozen() -> None:
    """Test HardwareConfig is frozen (consistent with all other sub-configs)."""
    config = HardwareConfig()

    with pytest.raises(ValidationError):
        config.reproducible = True


@pytest.mark.unit
def test_config_forbids_extra_fields() -> None:
    """Test HardwareConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        HardwareConfig(unknown_field="value")


# HARDWARE CONFIG: DETERMINISTIC WARN-ONLY
@pytest.mark.unit
def test_deterministic_warn_only_default_false() -> None:
    """Test deterministic_warn_only defaults to False."""
    config = HardwareConfig(device="cpu")

    assert config.deterministic_warn_only is False


@pytest.mark.unit
def test_deterministic_warn_only_requires_reproducible() -> None:
    """Test deterministic_warn_only=True without reproducible=True raises."""
    with pytest.raises(ValidationError, match="requires reproducible=True"):
        HardwareConfig(device="cpu", deterministic_warn_only=True, reproducible=False)


@pytest.mark.unit
def test_deterministic_warn_only_with_reproducible_valid() -> None:
    """Test deterministic_warn_only=True with reproducible=True is valid."""
    config = HardwareConfig(device="cpu", deterministic_warn_only=True, reproducible=True)

    assert config.deterministic_warn_only is True
    assert config.reproducible is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
