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
def test_get_loss_space_cross_entropy():
    """Test loss space samples label_smoothing for cross_entropy and defaults focal_gamma."""
    registry = SearchSpaceRegistry()
    space = registry.get_loss_space()

    assert "criterion_type" in space
    assert "focal_gamma" in space
    assert "label_smoothing" in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical = MagicMock(return_value="cross_entropy")
    trial_mock.suggest_float = MagicMock(return_value=0.1)
    trial_mock.params = {"criterion_type": "cross_entropy"}

    assert space["criterion_type"](trial_mock) == "cross_entropy"
    assert space["focal_gamma"](trial_mock) == 2.0  # default, not sampled
    assert space["label_smoothing"](trial_mock) == 0.1  # sampled
    trial_mock.suggest_float.assert_called_once()


@pytest.mark.unit
def test_get_loss_space_focal():
    """Test loss space samples focal_gamma for focal and defaults label_smoothing."""
    registry = SearchSpaceRegistry()
    space = registry.get_loss_space()

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical = MagicMock(return_value="focal")
    trial_mock.suggest_float = MagicMock(return_value=3.0)
    trial_mock.params = {"criterion_type": "focal"}

    assert space["criterion_type"](trial_mock) == "focal"
    assert space["focal_gamma"](trial_mock) == 3.0  # sampled
    assert space["label_smoothing"](trial_mock) == 0.0  # default, not sampled
    trial_mock.suggest_float.assert_called_once()


@pytest.mark.unit
def test_get_regularization_space():
    """Test retrieval of regularization strategies."""
    registry = SearchSpaceRegistry()
    space = registry.get_regularization_space()

    assert "mixup_alpha" in space
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
    assert "criterion_type" in space
    assert "focal_gamma" in space
    assert "label_smoothing" in space
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
    assert ov.criterion_type == ["cross_entropy", "focal"]
    assert ov.focal_gamma.low == pytest.approx(0.5)
    assert ov.focal_gamma.high == pytest.approx(5.0)


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
def test_custom_criterion_type_overrides():
    """Test custom criterion_type restricts to a single loss type."""
    custom_ov = SearchSpaceOverrides(
        criterion_type=["focal"],
        focal_gamma=FloatRange(low=1.0, high=3.0),
    )
    registry = SearchSpaceRegistry(custom_ov)
    space = registry.get_loss_space()

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical = MagicMock(return_value="focal")
    trial_mock.suggest_float = MagicMock(return_value=2.0)
    trial_mock.params = {"criterion_type": "focal"}

    space["criterion_type"](trial_mock)
    trial_mock.suggest_categorical.assert_called_with("criterion_type", ["focal"])

    space["focal_gamma"](trial_mock)
    trial_mock.suggest_float.assert_called_with("focal_gamma", 1.0, 3.0)


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


# MUTATION-RESILIENT: EXACT BOUNDS VERIFICATION
@pytest.mark.unit
def test_optimization_space_passes_exact_bounds():
    """Verify suggest_float receives exact overrides bounds for every optimization param."""
    registry = SearchSpaceRegistry()
    ov = registry.ov
    space = registry.get_optimization_space()

    trial = MagicMock(spec=Trial)
    trial.suggest_float = MagicMock(return_value=0.001)

    space["learning_rate"](trial)
    trial.suggest_float.assert_called_with(
        "learning_rate", ov.learning_rate.low, ov.learning_rate.high, log=True
    )

    trial.suggest_float.reset_mock()
    space["weight_decay"](trial)
    trial.suggest_float.assert_called_with(
        "weight_decay", ov.weight_decay.low, ov.weight_decay.high, log=True
    )

    trial.suggest_float.reset_mock()
    space["momentum"](trial)
    trial.suggest_float.assert_called_with("momentum", ov.momentum.low, ov.momentum.high)

    trial.suggest_float.reset_mock()
    space["min_lr"](trial)
    trial.suggest_float.assert_called_with("min_lr", ov.min_lr.low, ov.min_lr.high, log=True)


@pytest.mark.unit
def test_regularization_space_passes_exact_bounds():
    """Verify suggest_float receives exact bounds for mixup_alpha and dropout."""
    registry = SearchSpaceRegistry()
    ov = registry.ov
    space = registry.get_regularization_space()

    trial = MagicMock(spec=Trial)
    trial.suggest_float = MagicMock(return_value=0.2)

    space["mixup_alpha"](trial)
    trial.suggest_float.assert_called_with("mixup_alpha", ov.mixup_alpha.low, ov.mixup_alpha.high)

    trial.suggest_float.reset_mock()
    space["dropout"](trial)
    trial.suggest_float.assert_called_with("dropout", ov.dropout.low, ov.dropout.high)


@pytest.mark.unit
def test_scheduler_space_passes_exact_bounds():
    """Verify scheduler space passes correct types and bounds."""
    registry = SearchSpaceRegistry()
    ov = registry.ov
    space = registry.get_scheduler_space()

    trial = MagicMock(spec=Trial)
    trial.suggest_categorical = MagicMock(return_value="cosine")
    trial.suggest_int = MagicMock(return_value=5)

    space["scheduler_type"](trial)
    trial.suggest_categorical.assert_called_with("scheduler_type", ov.scheduler_type)

    space["scheduler_patience"](trial)
    trial.suggest_int.assert_called_with(
        "scheduler_patience", ov.scheduler_patience.low, ov.scheduler_patience.high
    )


@pytest.mark.unit
def test_augmentation_space_passes_exact_bounds():
    """Verify augmentation space passes correct bounds to suggest_int/suggest_float."""
    registry = SearchSpaceRegistry()
    ov = registry.ov
    space = registry.get_augmentation_space()

    trial = MagicMock(spec=Trial)
    trial.suggest_int = MagicMock(return_value=10)
    trial.suggest_float = MagicMock(return_value=0.1)

    space["rotation_angle"](trial)
    trial.suggest_int.assert_called_with(
        "rotation_angle", ov.rotation_angle.low, ov.rotation_angle.high
    )

    space["jitter_val"](trial)
    trial.suggest_float.assert_any_call("jitter_val", ov.jitter_val.low, ov.jitter_val.high)

    trial.suggest_float.reset_mock()
    space["min_scale"](trial)
    trial.suggest_float.assert_called_with("min_scale", ov.min_scale.low, ov.min_scale.high)


@pytest.mark.unit
def test_loss_space_passes_exact_bounds_focal():
    """Verify focal_gamma uses exact overrides bounds when criterion is focal."""
    registry = SearchSpaceRegistry()
    ov = registry.ov
    space = registry.get_loss_space()

    trial = MagicMock(spec=Trial)
    trial.suggest_float = MagicMock(return_value=2.5)
    trial.params = {"criterion_type": "focal"}

    space["focal_gamma"](trial)
    trial.suggest_float.assert_called_with("focal_gamma", ov.focal_gamma.low, ov.focal_gamma.high)


@pytest.mark.unit
def test_loss_space_passes_exact_bounds_label_smoothing():
    """Verify label_smoothing uses exact overrides bounds when criterion is cross_entropy."""
    registry = SearchSpaceRegistry()
    ov = registry.ov
    space = registry.get_loss_space()

    trial = MagicMock(spec=Trial)
    trial.suggest_float = MagicMock(return_value=0.1)
    trial.params = {"criterion_type": "cross_entropy"}

    space["label_smoothing"](trial)
    trial.suggest_float.assert_called_with(
        "label_smoothing", ov.label_smoothing.low, ov.label_smoothing.high
    )


@pytest.mark.unit
def test_loss_space_criterion_type_choices():
    """Verify criterion_type passes exact overrides list to suggest_categorical."""
    registry = SearchSpaceRegistry()
    ov = registry.ov
    space = registry.get_loss_space()

    trial = MagicMock(spec=Trial)
    trial.suggest_categorical = MagicMock(return_value="cross_entropy")

    space["criterion_type"](trial)
    trial.suggest_categorical.assert_called_with("criterion_type", ov.criterion_type)


@pytest.mark.unit
def test_batch_size_resolution_boundary():
    """Verify resolution=223 uses low_res list and resolution=224 uses high_res list."""
    registry = SearchSpaceRegistry()
    ov = registry.ov

    trial = MagicMock(spec=Trial)
    trial.suggest_categorical = MagicMock(return_value=32)

    space_low = registry.get_batch_size_space(resolution=223)
    space_low["batch_size"](trial)
    trial.suggest_categorical.assert_called_with("batch_size", list(ov.batch_size_low_res))

    trial.suggest_categorical.reset_mock()
    trial.suggest_categorical = MagicMock(return_value=12)
    space_high = registry.get_batch_size_space(resolution=224)
    space_high["batch_size"](trial)
    trial.suggest_categorical.assert_called_with("batch_size", list(ov.batch_size_high_res))


@pytest.mark.unit
def test_model_space_224_exact_model_list():
    """Verify get_model_space_224 passes the exact list of 224-resolution models."""
    space = SearchSpaceRegistry.get_model_space_224()

    trial = MagicMock(spec=Trial)
    trial.suggest_categorical = MagicMock(return_value="resnet_18")

    space["model_name"](trial)
    trial.suggest_categorical.assert_called_with(
        "model_name", ["resnet_18", "efficientnet_b0", "vit_tiny", "convnext_tiny"]
    )


@pytest.mark.unit
def test_model_space_224_weight_variant_vit():
    """Verify weight_variant is sampled when model_name is vit_tiny, None otherwise."""
    space = SearchSpaceRegistry.get_model_space_224()

    # When model_name IS vit_tiny → sample weight_variant
    trial_vit = MagicMock(spec=Trial)
    trial_vit.params = {"model_name": "vit_tiny"}
    trial_vit.suggest_categorical = MagicMock(return_value=None)

    space["weight_variant"](trial_vit)
    trial_vit.suggest_categorical.assert_called_with(
        "weight_variant",
        [None, "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit_tiny_patch16_224.augreg_in21k"],
    )

    # When model_name is NOT vit_tiny → returns None, no suggest call
    trial_other = MagicMock(spec=Trial)
    trial_other.params = {"model_name": "resnet_18"}
    result = space["weight_variant"](trial_other)
    assert result is None


@pytest.mark.unit
def test_model_space_28_exact_model_list():
    """Verify get_model_space_28 passes exact list of 28-resolution models."""
    space = SearchSpaceRegistry.get_model_space_28()

    trial = MagicMock(spec=Trial)
    trial.suggest_categorical = MagicMock(return_value="mini_cnn")

    space["model_name"](trial)
    trial.suggest_categorical.assert_called_with("model_name", ["resnet_18", "mini_cnn"])


@pytest.mark.unit
def test_quick_space_has_exactly_six_params():
    """Verify quick space contains exactly the expected high-impact params."""
    registry = SearchSpaceRegistry()
    space = registry.get_quick_space(resolution=28)

    expected = {"learning_rate", "weight_decay", "momentum", "min_lr", "batch_size", "dropout"}
    assert set(space.keys()) == expected


@pytest.mark.unit
def test_full_space_has_all_params():
    """Verify full space is the union of all sub-spaces."""
    registry = SearchSpaceRegistry()
    space = registry.get_full_space(resolution=28)

    expected = {
        "learning_rate",
        "weight_decay",
        "momentum",
        "min_lr",
        "criterion_type",
        "focal_gamma",
        "label_smoothing",
        "mixup_alpha",
        "dropout",
        "batch_size",
        "scheduler_type",
        "scheduler_patience",
        "rotation_angle",
        "jitter_val",
        "min_scale",
    }
    assert set(space.keys()) == expected


@pytest.mark.unit
def test_get_search_space_resolution_224_uses_model_space_224():
    """Verify factory dispatches to model_space_224 for resolution >= 224."""
    space = get_search_space(preset="quick", resolution=224, include_models=True)
    # 224 default includes convnext_tiny which is NOT in 28-resolution space
    trial = MagicMock(spec=Trial)
    trial.suggest_categorical = MagicMock(return_value="convnext_tiny")
    space["model_name"](trial)
    call_args = trial.suggest_categorical.call_args[0]
    assert "convnext_tiny" in call_args[1]


@pytest.mark.unit
def test_build_model_space_from_pool_exact_pool():
    """Verify _build_model_space_from_pool passes exact pool list to suggest_categorical."""
    from orchard.optimization.search_spaces import _build_model_space_from_pool

    pool = ["efficientnet_b0", "resnet_18"]
    space = _build_model_space_from_pool(pool)

    trial = MagicMock(spec=Trial)
    trial.suggest_categorical = MagicMock(return_value="resnet_18")

    space["model_name"](trial)
    trial.suggest_categorical.assert_called_with("model_name", pool)
    assert "weight_variant" not in space


@pytest.mark.unit
def test_build_model_space_from_pool_with_vit_includes_weight_variant():
    """Verify _build_model_space_from_pool adds weight_variant when vit_tiny in pool."""
    from orchard.optimization.search_spaces import _build_model_space_from_pool

    pool = ["vit_tiny", "resnet_18"]
    space = _build_model_space_from_pool(pool)
    assert "weight_variant" in space

    trial = MagicMock(spec=Trial)
    trial.params = {"model_name": "vit_tiny"}
    trial.suggest_categorical = MagicMock(return_value=None)

    space["weight_variant"](trial)
    trial.suggest_categorical.assert_called_with(
        "weight_variant",
        [None, "vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit_tiny_patch16_224.augreg_in21k"],
    )


@pytest.mark.unit
def test_build_model_space_from_pool_vit_non_vit_returns_none():
    """Verify weight_variant returns None when model_name is not vit_tiny."""
    from orchard.optimization.search_spaces import _build_model_space_from_pool

    pool = ["vit_tiny", "resnet_18"]
    space = _build_model_space_from_pool(pool)

    trial = MagicMock(spec=Trial)
    trial.params = {"model_name": "resnet_18"}
    result = space["weight_variant"](trial)
    assert result is None


# ---------------------------------------------------------------------------
# Default parameter value tests (kill mutants on function signatures)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_search_space_default_preset_is_quick():
    """Calling with no preset should use 'quick' (same result as explicit)."""
    default_result = get_search_space()
    explicit_result = get_search_space(preset="quick")
    assert set(default_result.keys()) == set(explicit_result.keys())


@pytest.mark.unit
def test_get_search_space_default_resolution_is_28():
    """Default resolution=28 produces low-res batch sizes."""
    space = get_search_space()
    trial = MagicMock(spec=Trial)
    space["batch_size"](trial)
    trial.suggest_categorical.assert_called_once_with("batch_size", [16, 32, 48, 64])


@pytest.mark.unit
def test_get_search_space_default_include_models_is_false():
    """Default include_models=False means no model_name key."""
    space = get_search_space()
    assert "model_name" not in space


@pytest.mark.unit
def test_get_search_space_invalid_preset_error_message():
    """ValueError message must mention the bad preset name."""
    with pytest.raises(ValueError, match="bogus"):
        get_search_space(preset="bogus")


@pytest.mark.unit
def test_get_quick_space_default_resolution_is_28():
    """get_quick_space() with no args should use resolution=28."""
    registry = SearchSpaceRegistry()
    space = registry.get_quick_space()
    trial = MagicMock(spec=Trial)
    space["batch_size"](trial)
    trial.suggest_categorical.assert_called_once_with("batch_size", [16, 32, 48, 64])


@pytest.mark.unit
def test_get_full_space_default_resolution_is_28():
    """get_full_space() with no args should use resolution=28."""
    registry = SearchSpaceRegistry()
    space = registry.get_full_space()
    trial = MagicMock(spec=Trial)
    space["batch_size"](trial)
    trial.suggest_categorical.assert_called_once_with("batch_size", [16, 32, 48, 64])


@pytest.mark.unit
def test_get_batch_size_space_default_resolution_is_28():
    """get_batch_size_space() with no args should use resolution=28."""
    registry = SearchSpaceRegistry()
    space = registry.get_batch_size_space()
    trial = MagicMock(spec=Trial)
    space["batch_size"](trial)
    trial.suggest_categorical.assert_called_once_with("batch_size", [16, 32, 48, 64])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
