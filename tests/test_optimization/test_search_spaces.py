"""
Unit tests for SearchSpaceRegistry and get_search_space function.

These tests validate the functionality of the hyperparameter search space definitions and retrieval functions.
They ensure that search spaces are correctly defined and resolved for different configurations.
"""

# Standard Imports
from unittest.mock import MagicMock

# Third-Party Imports
import pytest
from optuna.trial import Trial

# Internal Imports
from orchard.optimization import FullSearchSpace, SearchSpaceRegistry, get_search_space


# TEST CASES
@pytest.mark.unit
def test_get_optimization_space():
    """Test retrieval of core optimization hyperparameters."""
    space = SearchSpaceRegistry.get_optimization_space()

    assert "learning_rate" in space
    assert "weight_decay" in space
    assert "momentum" in space
    assert "min_lr" in space

    # Check that the suggested values are within the expected bounds
    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_float.side_effect = lambda param, low, high, log: 0.001
    learning_rate = space["learning_rate"](trial_mock)
    assert 1e-5 <= learning_rate <= 1e-2
    assert learning_rate == 0.001


@pytest.mark.unit
def test_get_regularization_space():
    """Test retrieval of regularization strategies."""
    space = SearchSpaceRegistry.get_regularization_space()

    assert "mixup_alpha" in space
    assert "label_smoothing" in space
    assert "dropout" in space

    # Check value ranges for regularization parameters
    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_float.side_effect = lambda param, low, high: 0.2
    mixup_alpha = space["mixup_alpha"](trial_mock)
    assert 0.0 <= mixup_alpha <= 0.4
    assert mixup_alpha == 0.2


@pytest.mark.unit
def test_get_batch_size_space():
    """Test retrieval of batch size space with resolution-aware choices."""
    # Test with high resolution (224)
    space_224 = SearchSpaceRegistry.get_batch_size_space(resolution=224)
    assert "batch_size" in space_224
    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical.side_effect = lambda param, choices: 12
    batch_size_224 = space_224["batch_size"](trial_mock)
    assert batch_size_224 in [8, 12, 16]
    assert batch_size_224 == 12

    # Test with low resolution (28)
    space_28 = SearchSpaceRegistry.get_batch_size_space(resolution=28)
    assert "batch_size" in space_28
    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical.side_effect = lambda param, choices: 32
    batch_size_28 = space_28["batch_size"](trial_mock)
    assert batch_size_28 in [16, 32, 48, 64]
    assert batch_size_28 == 32


@pytest.mark.unit
def test_get_full_space():
    """Test retrieval of combined full search space."""
    space = SearchSpaceRegistry.get_full_space(resolution=28)

    # Ensure all key spaces are present
    assert "learning_rate" in space
    assert "mixup_alpha" in space
    assert "batch_size" in space
    assert "scheduler_patience" in space
    assert "rotation_angle" in space


@pytest.mark.unit
def test_get_quick_space():
    """Test retrieval of the reduced quick search space."""
    space = SearchSpaceRegistry.get_quick_space(resolution=28)

    # Only a subset of parameters should be included
    assert "learning_rate" in space
    assert "batch_size" in space
    assert "dropout" in space
    assert "weight_decay" in space

    # Ensure irrelevant parameters (like "scheduler_patience") are excluded
    assert "scheduler_patience" not in space


@pytest.mark.unit
def test_get_model_space_224():
    """Test retrieval of model space for 224x224 resolution."""
    space = SearchSpaceRegistry.get_model_space_224()

    # Ensure model_name and weight_variant are present for 224x224
    assert "model_name" in space
    assert "weight_variant" in space

    # Test model choice based on trial
    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical.side_effect = lambda param, choices: "vit_tiny"
    model_name = space["model_name"](trial_mock)
    assert model_name in ["efficientnet_b0", "vit_tiny"]
    assert model_name == "vit_tiny"


@pytest.mark.unit
def test_get_model_space_28():
    """Test retrieval of model space for 28x28 resolution."""
    space = SearchSpaceRegistry.get_model_space_28()

    # Ensure model_name is present for 28x28
    assert "model_name" in space

    trial_mock = MagicMock(spec=Trial)
    trial_mock.suggest_categorical.side_effect = lambda param, choices: "resnet_18_adapted"
    model_name = space["model_name"](trial_mock)
    assert model_name in ["resnet_18_adapted", "mini_cnn"]
    assert model_name == "resnet_18_adapted"


@pytest.mark.unit
def test_get_full_space_with_models():
    """Test retrieval of full space with model selection based on resolution."""
    # Test with low resolution (28)
    space_28 = SearchSpaceRegistry.get_full_space_with_models(resolution=28)
    assert "model_name" in space_28
    assert "batch_size" in space_28

    # Test with high resolution (224)
    space_224 = SearchSpaceRegistry.get_full_space_with_models(resolution=224)
    assert "model_name" in space_224
    assert "weight_variant" in space_224


@pytest.mark.unit
def test_get_search_space_invalid_preset():
    """Test behavior when an invalid preset is passed to get_search_space."""
    with pytest.raises(ValueError):
        get_search_space(preset="invalid_preset", resolution=28)


@pytest.mark.unit
def test_get_search_space_valid_presets():
    """Test behavior when valid presets are passed to get_search_space."""
    # Test quick preset
    space_quick = get_search_space(preset="quick", resolution=28)
    assert "learning_rate" in space_quick
    assert "batch_size" in space_quick
    assert "dropout" in space_quick

    # Test full preset
    space_full = get_search_space(preset="full", resolution=28)
    assert "learning_rate" in space_full
    assert "mixup_alpha" in space_full
    assert "batch_size" in space_full


@pytest.mark.unit
def test_full_search_space_sample_params():
    """Test resolution-aware sampling in the FullSearchSpace class."""
    full_space = FullSearchSpace(resolution=28)

    # Mock trial
    trial_mock = MagicMock(spec=Trial)

    # Mocking suggest_float for both log=True and log=False cases
    trial_mock.suggest_float = MagicMock()
    trial_mock.suggest_float.return_value = 0.001

    # Mocking suggest_categorical
    trial_mock.suggest_categorical = MagicMock()
    trial_mock.suggest_categorical.return_value = 32
    sampled_params = full_space.sample_params(trial_mock)

    assert sampled_params["batch_size"] in [16, 32, 48, 64]
    assert sampled_params["batch_size"] == 32
    assert "learning_rate" in sampled_params
    assert "momentum" in sampled_params
    assert "dropout" in sampled_params
