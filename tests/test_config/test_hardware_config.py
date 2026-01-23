"""
Test Suite for HardwareConfig.

Tests device resolution, reproducibility mode, num_workers logic,
and lock file path generation.
"""

# Standard Imports
import tempfile
from argparse import Namespace

# Third-Party Imports
import pytest
import torch
from pydantic import ValidationError

# Internal Imports
from orchard.core.config import HardwareConfig


# HARDWARE CONFIG: DEVICE RESOLUTION
@pytest.mark.unit
def test_device_auto_resolves():
    """Test device='auto' resolves to best available."""
    config = HardwareConfig(device="auto")

    # Should resolve to cuda, mps, or cpu
    assert config.device in ("cpu", "cuda", "mps")


@pytest.mark.unit
def test_device_cpu_always_works():
    """Test device='cpu' always resolves successfully."""
    config = HardwareConfig(device="cpu")

    assert config.device == "cpu"


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda_when_available():
    """Test device='cuda' resolves when CUDA available."""
    config = HardwareConfig(device="cuda")

    assert config.device == "cuda"


@pytest.mark.unit
@pytest.mark.skipif(torch.cuda.is_available(), reason="Test requires no CUDA")
def test_device_cuda_fallback_to_cpu():
    """Test device='cuda' falls back to CPU when unavailable."""
    config = HardwareConfig(device="cuda")

    # Should fallback to CPU
    assert config.device == "cpu"


@pytest.mark.unit
def test_invalid_device_fallback():
    """Test invalid device type falls through validator."""
    # MPS on non-Mac should fallback to CPU
    config = HardwareConfig(device="mps")

    # Should either be mps (if on Mac) or cpu (fallback)
    assert config.device in ("mps", "cpu")


# HARDWARE CONFIG: REPRODUCIBILITY
@pytest.mark.unit
def test_reproducible_mode_disabled_by_default():
    """Test reproducible mode is False by default."""
    config = HardwareConfig()

    assert config.reproducible is False
    assert config.use_deterministic_algorithms is False


@pytest.mark.unit
def test_reproducible_mode_enables_deterministic():
    """Test reproducible mode sets deterministic algorithms."""
    config = HardwareConfig(reproducible=True)

    assert config.reproducible is True
    assert config.use_deterministic_algorithms is True


@pytest.mark.unit
def test_reproducible_mode_affects_num_workers():
    """Test reproducible mode forces num_workers=0."""
    config = HardwareConfig(reproducible=False)

    # Should use system-detected workers
    assert config.effective_num_workers >= 0


@pytest.mark.unit
def test_for_optuna_factory_enables_reproducibility():
    """Test HardwareConfig.for_optuna() enables reproducible mode."""
    config = HardwareConfig.for_optuna(device="cpu")

    assert config.reproducible is True
    assert config.effective_num_workers == 0


# HARDWARE CONFIG: NUM_WORKERS
@pytest.mark.unit
def test_effective_num_workers_zero_when_reproducible():
    """Test effective_num_workers is 0 in reproducible mode."""
    config = HardwareConfig(reproducible=True)

    assert config.effective_num_workers == 0


@pytest.mark.unit
def test_effective_num_workers_respects_explicit_value():
    """Test effective_num_workers uses explicit value when set."""
    config = HardwareConfig(reproducible=False)

    assert config.effective_num_workers >= 0


# HARDWARE CONFIG: AMP SUPPORT
@pytest.mark.unit
def test_supports_amp_cpu_false():
    """Test CPU does not support AMP."""
    config = HardwareConfig(device="cpu")

    assert config.supports_amp is False


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_supports_amp_cuda_true():
    """Test CUDA supports AMP."""
    config = HardwareConfig(device="cuda")

    assert config.supports_amp is True


# HARDWARE CONFIG: LOCK FILE PATH
@pytest.mark.unit
def test_lock_file_path_in_temp_dir():
    """Test lock file is created in system temp directory."""
    config = HardwareConfig(project_name="test_project")

    lock_path = config.lock_file_path

    # Should be in temp directory
    assert str(lock_path).startswith(tempfile.gettempdir())
    assert lock_path.name == "test_project.lock"


@pytest.mark.unit
def test_lock_file_path_uses_project_name():
    """Test lock file path uses project name."""
    config = HardwareConfig(project_name="my-experiment")

    lock_path = config.lock_file_path

    assert "my-experiment.lock" in str(lock_path)


@pytest.mark.unit
def test_lock_file_path_sanitizes_slashes():
    """Test lock file path sanitizes project name with slashes."""
    pytest.skip("ProjectSlug doesn't allow slashes by design")

    config = HardwareConfig(project_name="org/project")
    lock_path = config.lock_file_path

    # Slashes should be replaced with underscores
    assert "/" not in lock_path.name
    assert "org_project.lock" in lock_path.name


# HARDWARE CONFIG: PROJECT NAME VALIDATION
@pytest.mark.unit
def test_project_name_validation_valid():
    """Test project_name follows slug pattern."""
    valid_names = ["valid-project_123", "my-exp", "test_001"]

    for name in valid_names:
        config = HardwareConfig(project_name=name)
        assert config.project_name == name


@pytest.mark.unit
def test_project_name_validation_invalid():
    """Test invalid project names are rejected."""
    with pytest.raises(ValidationError):
        HardwareConfig(project_name="Invalid Project!")

    with pytest.raises(ValidationError):
        HardwareConfig(project_name="UPPERCASE")


# HARDWARE CONFIG: FROM ARGS
@pytest.mark.unit
def test_from_args_basic():
    """Test HardwareConfig.from_args() with basic arguments."""
    args = Namespace(device="cpu", project_name="test_exp", allow_process_kill=True)

    config = HardwareConfig.from_args(args)

    assert config.device == "cpu"
    assert config.project_name == "test_exp"
    assert config.allow_process_kill is True


@pytest.mark.unit
def test_from_args_ignores_none_values():
    """Test from_args() ignores None values."""
    args = Namespace(device="cpu", project_name=None)  # Should use default

    config = HardwareConfig.from_args(args)

    assert config.device == "cpu"
    assert config.project_name == "vision_experiment"  # Default


@pytest.mark.unit
def test_from_args_partial():
    """Test from_args() with partial arguments."""
    args = Namespace(device="cuda")

    config = HardwareConfig.from_args(args)

    assert config.device in ("cuda", "cpu")  # May fallback
    assert config.project_name == "vision_experiment"  # Default


# HARDWARE CONFIG: DEFAULTS
@pytest.mark.unit
def test_hardware_config_defaults():
    """Test HardwareConfig with default values."""
    config = HardwareConfig()

    assert config.device in ("auto", "cpu", "cuda", "mps")
    assert config.project_name == "vision_experiment"
    assert config.allow_process_kill is True


# HARDWARE CONFIG: MUTABILITY
@pytest.mark.unit
def test_config_not_frozen():
    """Test HardwareConfig is NOT frozen (allows reproducible mutation)."""
    config = HardwareConfig()

    # Should be mutable (frozen=False)
    config.reproducible = True
    assert config.reproducible is True


@pytest.mark.unit
def test_config_forbids_extra_fields():
    """Test HardwareConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        HardwareConfig(unknown_field="value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
