"""
Unit tests for hyperparameter-to-config section mapping.

Tests the mapping functions and registries in
orchard/optimization/_param_mapping.py.
"""

from __future__ import annotations

import pytest

from orchard.optimization._param_mapping import map_param_to_config_path
from orchard.optimization.orchestrator.registries import (
    PRUNER_REGISTRY,
    SAMPLER_REGISTRY,
)

# ---------------------------------------------------------------------------
# SAMPLER / PRUNER registries
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sampler_registry_has_tpe() -> None:
    """Test SAMPLER_REGISTRY contains TPE."""
    assert "tpe" in SAMPLER_REGISTRY


@pytest.mark.unit
def test_pruner_registry_has_median() -> None:
    """Test PRUNER_REGISTRY contains Median."""
    assert "median" in PRUNER_REGISTRY


# ---------------------------------------------------------------------------
# map_param_to_config_path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_map_param_to_config_path_training() -> None:
    """Test mapping training parameter."""
    section, key = map_param_to_config_path("learning_rate")
    assert section == "training"
    assert key == "learning_rate"


@pytest.mark.unit
def test_map_param_to_config_path_architecture() -> None:
    """Test mapping architecture parameter."""
    section, key = map_param_to_config_path("dropout")
    assert section == "architecture"
    assert key == "dropout"


@pytest.mark.unit
def test_map_param_to_config_path_augmentation() -> None:
    """Test mapping augmentation parameter."""
    section, key = map_param_to_config_path("rotation_angle")
    assert section == "augmentation"
    assert key == "rotation_angle"

    section, key = map_param_to_config_path("jitter_val")
    assert section == "augmentation"
    assert key == "jitter_val"

    section, key = map_param_to_config_path("min_scale")
    assert section == "augmentation"
    assert key == "min_scale"


@pytest.mark.unit
def test_map_param_to_config_path_special_model_name() -> None:
    """Test mapping special parameter: model_name."""
    section, key = map_param_to_config_path("model_name")
    assert section == "architecture"
    assert key == "name"


@pytest.mark.unit
def test_map_param_to_config_path_special_weight_variant() -> None:
    """Test mapping special parameter: weight_variant."""
    section, key = map_param_to_config_path("weight_variant")
    assert section == "architecture"
    assert key == "weight_variant"


@pytest.mark.unit
def test_map_param_to_config_path_unknown_raises() -> None:
    """Test unknown parameter raises ValueError."""
    with pytest.raises(ValueError, match="Unknown hyperparameter 'unknown_param'"):
        map_param_to_config_path("unknown_param")
