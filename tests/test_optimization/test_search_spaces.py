"""
Unit tests for SearchSpaceRegistry and get_search_space function.

These tests validate the functionality of the hyperparameter search space definitions and retrieval functions.
They ensure that search spaces are correctly defined and resolved for different configurations.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from optuna.trial import Trial
from pydantic import ValidationError

from orchard.core.config.optuna_config import FloatRange, IntRange, SearchSpaceOverrides
from orchard.optimization import SearchSpaceRegistry, get_search_space


# SEARCH SPACE REGISTRY: DEFAULT OVERRIDES
@pytest.mark.unit
def test_get_optimization_space():
    """Test retrieval of core optimization hyperparameters."""
    registry = SearchSpaceRegistry()
    space = registry.get_optimization_space()

    assert "learning_rate" in space
    assert "weight_decay" in space
    assert "momentum" in space
    assert "min_lr" in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_float.side_effect = lambda *args, **kwargs: 0.001
    learning_rate = space["learning_rate"](trial_mock)
    assert 1e-5 <= learning_rate <= 1e-2
    assert learning_rate == pytest.approx(0.001)


@pytest.mark.unit
def test_get_regularization_space():
    """Test retrieval of regularization strategies."""
    registry = SearchSpaceRegistry()
    space = registry.get_regularization_space()

    assert "mixup_alpha" in space
    assert "label_smoothing" in space
    assert "dropout" in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_float = lambda *args, **kwargs: 0.01

    mixup_alpha = space["mixup_alpha"](trial_mock)
    assert 0.0 <= mixup_alpha <= 0.4


@pytest.mark.unit
def test_get_batch_size_space():
    """Test retrieval of batch size space with resolution-aware choices."""
    registry = SearchSpaceRegistry()

    space_224 = registry.get_batch_size_space(resolution=224)
    assert "batch_size" in space_224
    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical.side_effect = lambda _param, _choices: 12
    batch_size_224 = space_224["batch_size"](trial_mock)
    assert batch_size_224 in [8, 12, 16]
    assert batch_size_224 == 12

    space_28 = registry.get_batch_size_space(resolution=28)
    assert "batch_size" in space_28
    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical.side_effect = lambda _param, _choices: 32
    batch_size_28 = space_28["batch_size"](trial_mock)
    assert batch_size_28 in [16, 32, 48, 64]
    assert batch_size_28 == 32


@pytest.mark.unit
def test_get_full_space():
    """Test retrieval of combined full search space."""
    registry = SearchSpaceRegistry()
    space = registry.get_full_space(resolution=28)

    assert "learning_rate" in space
    assert "mixup_alpha" in space
    assert "batch_size" in space
    assert "scheduler_patience" in space
    assert "rotation_angle" in space


@pytest.mark.unit
def test_get_quick_space():
    """Test retrieval of the reduced quick search space."""
    registry = SearchSpaceRegistry()
    space = registry.get_quick_space(resolution=28)

    assert "learning_rate" in space
    assert "batch_size" in space
    assert "dropout" in space
    assert "weight_decay" in space
    assert "scheduler_patience" not in space


@pytest.mark.unit
def test_get_model_space_224():
    """Test retrieval of model space for 224x224 resolution."""
    space = SearchSpaceRegistry.get_model_space_224()

    assert "model_name" in space
    assert "weight_variant" in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical.side_effect = lambda _param, _choices: "vit_tiny"
    model_name = space["model_name"](trial_mock)
    assert model_name in ["efficientnet_b0", "vit_tiny"]
    assert model_name == "vit_tiny"


@pytest.mark.unit
def test_get_model_space_28():
    """Test retrieval of model space for 28x28 resolution."""
    space = SearchSpaceRegistry.get_model_space_28()

    assert "model_name" in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical.side_effect = lambda _param, _choices: "resnet_18"
    model_name = space["model_name"](trial_mock)
    assert model_name in ["resnet_18", "mini_cnn"]
    assert model_name == "resnet_18"


# PRESET FACTORY
@pytest.mark.unit
def test_get_search_space_invalid_preset():
    """Test behavior when an invalid preset is passed to get_search_space."""
    with pytest.raises(ValueError):
        get_search_space(preset="invalid_preset", resolution=28)


@pytest.mark.unit
def test_get_search_space_valid_presets():
    """Test behavior when valid presets are passed to get_search_space."""
    space_quick = get_search_space(preset="quick", resolution=28)
    assert "learning_rate" in space_quick
    assert "batch_size" in space_quick
    assert "dropout" in space_quick

    space_full = get_search_space(preset="full", resolution=28)
    assert "learning_rate" in space_full
    assert "mixup_alpha" in space_full
    assert "batch_size" in space_full


@pytest.mark.unit
def test_get_search_space_with_models_resolution_224():
    """Test get_search_space with include_models=True for 224x224 resolution."""
    space = get_search_space(preset="quick", resolution=224, include_models=True)

    assert "model_name" in space
    assert "weight_variant" in space
    assert "learning_rate" in space
    assert "batch_size" in space


@pytest.mark.unit
def test_get_search_space_with_models_resolution_28():
    """Test get_search_space with include_models=True for 28x28 resolution."""
    space = get_search_space(preset="quick", resolution=28, include_models=True)

    assert "model_name" in space
    assert "weight_variant" not in space
    assert "learning_rate" in space
    assert "batch_size" in space


# CUSTOM OVERRIDES
@pytest.mark.unit
def test_custom_overrides_applied():
    """Test that custom SearchSpaceOverrides are used by the registry."""
    custom_ov = SearchSpaceOverrides(
        learning_rate=FloatRange(low=1e-3, high=1e-1, log=True),
        batch_size_low_res=[64, 128, 256],
    )
    registry = SearchSpaceRegistry(custom_ov)

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_float = MagicMock(return_value=0.05)

    space = registry.get_optimization_space()
    space["learning_rate"](trial_mock)

    trial_mock.suggest_float.assert_any_call("learning_rate", 1e-3, 1e-1, log=True)


@pytest.mark.unit
def test_custom_batch_size_overrides():
    """Test custom batch size choices via overrides."""
    custom_ov = SearchSpaceOverrides(
        batch_size_low_res=[32, 64, 128],
        batch_size_high_res=[4, 8],
    )
    registry = SearchSpaceRegistry(custom_ov)

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical = MagicMock(return_value=64)

    space = registry.get_batch_size_space(resolution=28)
    space["batch_size"](trial_mock)

    trial_mock.suggest_categorical.assert_called_with("batch_size", [32, 64, 128])


@pytest.mark.unit
def test_get_search_space_with_overrides():
    """Test get_search_space factory passes overrides to registry."""
    custom_ov = SearchSpaceOverrides(
        batch_size_high_res=[4, 8, 12],
    )
    space = get_search_space(preset="full", resolution=224, overrides=custom_ov)

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_float = MagicMock(return_value=0.001)
    trial_mock.suggest_int = MagicMock(return_value=5)
    trial_mock.suggest_categorical = MagicMock(return_value=8)

    space["batch_size"](trial_mock)

    trial_mock.suggest_categorical.assert_called_with("batch_size", [4, 8, 12])


# SEARCH SPACE OVERRIDES VALIDATION
@pytest.mark.unit
def test_float_range_rejects_invalid_bounds():
    """Test FloatRange raises ValueError when low >= high."""
    with pytest.raises(ValidationError, match="strictly less than"):
        FloatRange(low=0.5, high=0.5)

    with pytest.raises(ValidationError, match="strictly less than"):
        FloatRange(low=1.0, high=0.1)


@pytest.mark.unit
def test_int_range_rejects_invalid_bounds():
    """Test IntRange raises ValueError when low >= high."""
    with pytest.raises(ValidationError, match="strictly less than"):
        IntRange(low=10, high=10)

    with pytest.raises(ValidationError, match="strictly less than"):
        IntRange(low=20, high=5)


@pytest.mark.unit
def test_search_space_overrides_defaults():
    """Test SearchSpaceOverrides has sensible defaults."""
    ov = SearchSpaceOverrides()

    assert ov.learning_rate.low == pytest.approx(1e-5)
    assert ov.learning_rate.high == pytest.approx(1e-2)
    assert ov.learning_rate.log is True
    assert ov.batch_size_low_res == [16, 32, 48, 64]
    assert ov.batch_size_high_res == [8, 12, 16]
    assert ov.dropout.low == pytest.approx(0.1)
    assert ov.dropout.high == pytest.approx(0.5)


@pytest.mark.unit
def test_search_space_overrides_forbids_extra():
    """Test SearchSpaceOverrides rejects unknown fields."""
    with pytest.raises(ValidationError):
        SearchSpaceOverrides(unknown_param=FloatRange(low=0.0, high=1.0))


# MODEL POOL FILTERING
@pytest.mark.unit
def test_get_search_space_with_model_pool():
    """Test get_search_space uses model_pool when provided."""
    pool = ["resnet_18", "mini_cnn"]
    space = get_search_space(
        preset="quick",
        resolution=28,
        include_models=True,
        model_pool=pool,
    )

    assert "model_name" in space
    assert "weight_variant" not in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical = MagicMock(return_value="resnet_18")

    space["model_name"](trial_mock)
    trial_mock.suggest_categorical.assert_called_with("model_name", pool)


@pytest.mark.unit
def test_get_search_space_model_pool_with_vit():
    """Test model_pool includes weight_variant when vit_tiny is in pool."""
    pool = ["efficientnet_b0", "vit_tiny"]
    space = get_search_space(
        preset="full",
        resolution=224,
        include_models=True,
        model_pool=pool,
    )

    assert "model_name" in space
    assert "weight_variant" in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical = MagicMock(return_value="vit_tiny")
    trial_mock.params = {"model_name": "vit_tiny"}

    space["model_name"](trial_mock)
    trial_mock.suggest_categorical.assert_called_with("model_name", pool)


@pytest.mark.unit
def test_get_search_space_model_pool_without_vit():
    """Test model_pool excludes weight_variant when vit_tiny is absent."""
    pool = ["resnet_18", "efficientnet_b0"]
    space = get_search_space(
        preset="full",
        resolution=224,
        include_models=True,
        model_pool=pool,
    )

    assert "model_name" in space
    assert "weight_variant" not in space


@pytest.mark.unit
def test_get_search_space_model_pool_none_uses_defaults():
    """Test model_pool=None falls back to resolution-based defaults."""
    space_224 = get_search_space(
        preset="quick", resolution=224, include_models=True, model_pool=None
    )
    space_28 = get_search_space(preset="quick", resolution=28, include_models=True, model_pool=None)

    # 224 default includes weight_variant (vit_tiny is in default pool)
    assert "weight_variant" in space_224
    # 28 default does not include weight_variant
    assert "weight_variant" not in space_28


@pytest.mark.unit
def test_get_search_space_model_pool_with_timm():
    """Test model_pool works with timm/ prefixed names."""
    pool = ["resnet_18", "timm/mobilenetv3_small_100"]
    space = get_search_space(
        preset="quick",
        resolution=224,
        include_models=True,
        model_pool=pool,
    )

    assert "model_name" in space
    assert "weight_variant" not in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical = MagicMock(return_value="timm/mobilenetv3_small_100")

    space["model_name"](trial_mock)
    trial_mock.suggest_categorical.assert_called_with("model_name", pool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
