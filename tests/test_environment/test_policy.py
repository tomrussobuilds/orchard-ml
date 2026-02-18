"""
Test Suite for Execution & Optimization Policies.

Tests TTA mode determination logic based on hardware
availability and configuration constraints.
"""

import pytest

from orchard.core.environment import determine_tta_mode


# TTA MODE: DISABLED
@pytest.mark.unit
def test_determine_tta_mode_disabled_cpu():
    """Test TTA mode returns DISABLED when use_tta is False on CPU."""
    result = determine_tta_mode(use_tta=False, device_type="cpu")
    assert result == "DISABLED"


@pytest.mark.unit
def test_determine_tta_mode_disabled_cuda():
    """Test TTA mode returns DISABLED when use_tta is False on CUDA."""
    result = determine_tta_mode(use_tta=False, device_type="cuda")
    assert result == "DISABLED"


@pytest.mark.unit
def test_determine_tta_mode_disabled_mps():
    """Test TTA mode returns DISABLED when use_tta is False on MPS."""
    result = determine_tta_mode(use_tta=False, device_type="mps")
    assert result == "DISABLED"


# TTA MODE: DEFAULT (FULL)
@pytest.mark.unit
def test_determine_tta_mode_cpu_default():
    """Test TTA mode returns FULL for CPU when using default tta_mode."""
    result = determine_tta_mode(use_tta=True, device_type="cpu")
    assert result == "FULL (CPU)"


# TTA MODE: EXPLICIT LIGHT
@pytest.mark.unit
def test_determine_tta_mode_cpu_light():
    """Test TTA mode returns LIGHT for CPU when tta_mode is explicitly 'light'."""
    result = determine_tta_mode(use_tta=True, device_type="cpu", tta_mode="light")
    assert result == "LIGHT (CPU)"


# TTA MODE: ACCELERATED
@pytest.mark.unit
def test_determine_tta_mode_cuda_full():
    """Test TTA mode returns FULL for CUDA acceleration."""
    result = determine_tta_mode(use_tta=True, device_type="cuda")
    assert result == "FULL (CUDA)"


@pytest.mark.unit
def test_determine_tta_mode_mps_full():
    """Test TTA mode returns FULL for MPS (Apple Silicon) acceleration."""
    result = determine_tta_mode(use_tta=True, device_type="mps")
    assert result == "FULL (MPS)"


# TTA MODE: EDGE CASES
@pytest.mark.unit
def test_determine_tta_mode_case_sensitivity():
    """Test device_type case handling (should work with any case)."""
    result = determine_tta_mode(use_tta=True, device_type="cuda")
    assert "CUDA" in result

    result = determine_tta_mode(use_tta=True, device_type="CUDA")
    assert "CUDA" in result


@pytest.mark.unit
def test_determine_tta_mode_unknown_device():
    """Test TTA mode with unknown device type."""
    result = determine_tta_mode(use_tta=True, device_type="xpu")
    assert result == "FULL (XPU)"


@pytest.mark.unit
def test_determine_tta_mode_all_devices():
    """Test all known device types with TTA enabled (default tta_mode='full')."""
    devices = ["cpu", "cuda", "mps"]
    expected = [
        "FULL (CPU)",
        "FULL (CUDA)",
        "FULL (MPS)",
    ]

    for device, expected_result in zip(devices, expected):
        result = determine_tta_mode(use_tta=True, device_type=device)
        assert result == expected_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
