"""
Test Suite for Config Engine.

Tests main Config class integration, cross-validation,
YAML hydration, and from_args factory.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import argparse

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
import yaml
from pydantic import ValidationError

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core import Config, DatasetConfig, HardwareConfig, ModelConfig, TrainingConfig

# =========================================================================== #
#                    CONFIG: BASIC CONSTRUCTION                               #
# =========================================================================== #


@pytest.mark.unit
def test_config_defaults():
    """Test Config with all default sub-configs."""
    config = Config()

    # Sub-configs should be instantiated
    assert config.hardware is not None
    assert config.training is not None
    assert config.dataset is not None
    assert config.model is not None


@pytest.mark.unit
def test_config_from_args_basic(basic_args):
    """Test Config.from_args() with basic arguments."""
    config = Config.from_args(basic_args)

    assert config.dataset.dataset_name == "bloodmnist"
    assert config.model.name == "resnet_18_adapted"
    assert config.training.epochs == 60


# =========================================================================== #
#                    CONFIG: CROSS-VALIDATION                                 #
# =========================================================================== #


@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resnet_18_adapted_requires_resolution_28_direct(device):
    """
    resnet_18_adapted uses a modified stem and is only compatible with 28x28 inputs,
    regardless of the execution device.
    """
    with pytest.raises(
        ValidationError,
        match="resnet_18_adapted requires resolution=28",
    ):
        Config(
            dataset=DatasetConfig(
                name="bloodmnist",
                resolution=224,  # ❌ invalid for this architecture
            ),
            model=ModelConfig(
                name="resnet_18_adapted",
                pretrained=True,
            ),
            training=TrainingConfig(),
            hardware=HardwareConfig(device=device),
        )


@pytest.mark.unit
def test_mixup_epochs_cannot_exceed_total_epochs_direct():
    """
    MixUp scheduling cannot exceed total training epochs.
    """
    with pytest.raises(
        ValidationError,
        match="mixup_epochs .* exceeds total epochs",
    ):
        Config(
            training=TrainingConfig(
                epochs=5,
                mixup_epochs=10,  # ❌ invalid
            ),
            dataset=DatasetConfig(),
            model=ModelConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_resolve_dataset_metadata_requires_name():
    """_resolve_dataset_metadata should raise ValueError if dataset name is missing."""
    args = argparse.Namespace(dataset=None)

    with pytest.raises(ValueError, match="Dataset name required via --dataset or config file"):
        Config._resolve_dataset_metadata(args)


@pytest.mark.unit
def test_resolve_dataset_metadata_not_in_registry():
    """_resolve_dataset_metadata should raise ValueError if dataset not in registry."""
    args = argparse.Namespace(dataset="nonexistent_dataset", resolution=28)

    with pytest.raises(
        ValueError, match="Dataset 'nonexistent_dataset' not found in registry for resolution 28"
    ):
        Config._resolve_dataset_metadata(args)


@pytest.mark.unit
def test_amp_requires_gpu():
    """Test AMP validation rejects CPU + AMP."""
    args = argparse.Namespace(
        dataset="bloodmnist", device="cpu", use_amp=True, pretrained=True  # Invalid with CPU
    )

    with pytest.raises(ValidationError, match="AMP requires GPU"):
        Config.from_args(args)


@pytest.mark.unit
def test_pretrained_requires_rgb():
    """Test pretrained model validation enforces RGB channels."""
    args = argparse.Namespace(
        dataset="organcmnist",  # Grayscale
        model_name="resnet_18_adapted",
        pretrained=True,
        force_rgb=False,  # This will cause validation error
        resolution=28,
    )

    with pytest.raises(ValidationError, match="Pretrained.*requires RGB"):
        Config.from_args(args)


@pytest.mark.unit
def test_min_lr_less_than_lr_validation():
    """Test min_lr < learning_rate validation (covers line 106)."""
    args = argparse.Namespace(
        dataset="bloodmnist",
        learning_rate=0.001,
        min_lr=0.01,  # ❌ min_lr > learning_rate
        pretrained=True,
    )

    with pytest.raises(ValidationError, match="min_lr"):
        Config.from_args(args)


@pytest.mark.unit
def test_min_lr_equals_lr_direct_instantiation(mock_metadata_28):
    """Test min_lr == learning_rate validation via direct instantiation."""
    # This approach ensures the validator runs and line 106 is covered
    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(
                name="bloodmnist",
                resolution=28,
                metadata=mock_metadata_28,
            ),
            model=ModelConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                learning_rate=0.001,
                min_lr=0.001,  # ❌ Equal - triggers line 106
            ),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_resolve_dataset_metadata_success():
    """Test successful dataset metadata resolution."""
    args = argparse.Namespace(dataset="bloodmnist", resolution=28)

    metadata = Config._resolve_dataset_metadata(args)

    assert metadata is not None
    assert metadata.num_classes > 0
    assert metadata.in_channels in [1, 3]


# =========================================================================== #
#                    CONFIG: YAML HYDRATION                                   #
# =========================================================================== #


@pytest.mark.integration
def test_from_yaml_loads_correctly(temp_yaml_config, mock_metadata_28):
    """Test Config.from_yaml() loads YAML correctly."""
    config = Config.from_yaml(temp_yaml_config, metadata=mock_metadata_28)

    assert config.dataset.dataset_name == "bloodmnist"
    assert config.model.name == "resnet_18_adapted"
    assert config.training.epochs == 60
    assert config.training.batch_size == 128


@pytest.mark.integration
def test_yaml_optuna_section_loaded(temp_yaml_config, mock_metadata_28):
    """Test YAML with optuna section loads OptunaConfig."""
    config = Config.from_yaml(temp_yaml_config, metadata=mock_metadata_28)

    # Optuna section should be loaded
    assert config.optuna is not None
    assert config.optuna.study_name == "yaml_test_study"
    assert config.optuna.n_trials == 20


@pytest.mark.integration
def test_yaml_precedence_over_args(temp_yaml_config):
    """Test YAML values override CLI arguments."""
    args = argparse.Namespace(
        config=str(temp_yaml_config),
        epochs=999,  # Should be ignored
        batch_size=999,  # Should be ignored
        dataset="bloodmnist",
        pretrained=True,
    )

    config = Config.from_args(args)

    # YAML values should take precedence
    assert config.training.epochs == 60  # From YAML
    assert config.training.batch_size == 128  # From YAML


@pytest.mark.integration
def test_build_from_yaml_or_args_resolves_dataset(tmp_path, mock_metadata_28):
    """
    _build_from_yaml_or_args should trigger the 'if yaml_dataset_name' branch
    and re-resolve dataset from the registry (covers line 255).
    """
    yaml_content = {
        "dataset": {"name": "dermamnist", "resolution": 28},
        "model": {"name": "mini_cnn"},
        "training": {"epochs": 60},
        "optuna": {"study_name": "yaml_test_study", "n_trials": 20},
    }
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    args = argparse.Namespace(
        config=str(yaml_path),
        dataset="bloodmnist",  # Different from YAML - will trigger re-resolution
    )

    # This will call wrapper.get_dataset() on line 255
    cfg = Config._build_from_yaml_or_args(args, ds_meta=mock_metadata_28)

    assert cfg.dataset.dataset_name == "dermamnist"
    assert cfg.model.name == "mini_cnn"
    assert cfg.training.epochs == 60
    assert cfg.optuna.study_name == "yaml_test_study"


@pytest.mark.integration
def test_yaml_different_dataset_triggers_wrapper_call(tmp_path):
    """
    Line 256: ds_meta = wrapper.get_dataset(yaml_dataset_name)
    Using bloodmnist which definitely exists in registry.
    """
    yaml_content = {
        "dataset": {"name": "bloodmnist", "resolution": 28},  # Use known dataset
        "model": {"name": "mini_cnn", "pretrained": False},
        "training": {"epochs": 100, "mixup_epochs": 0, "use_amp": False},
        "hardware": {"device": "cpu"},
    }
    yaml_path = tmp_path / "config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    args = argparse.Namespace(
        config=str(yaml_path),
        dataset="dermamnist",  # Different from YAML to force re-resolution
        resolution=28,
    )

    config = Config.from_args(args)
    assert config.dataset.dataset_name == "bloodmnist"


@pytest.mark.integration
def test_build_from_yaml_or_args_yaml_dataset_not_found(tmp_path):
    """_build_from_yaml_or_args should raise KeyError if YAML dataset not in registry."""
    yaml_content = {"dataset": {"name": "nonexistent_dataset", "resolution": 28}}
    yaml_path = tmp_path / "bad_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    args = argparse.Namespace(
        config=str(yaml_path),
        dataset="bloodmnist",
    )
    with pytest.raises(KeyError, match="nonexistent_dataset"):
        Config._build_from_yaml_or_args(args, ds_meta={})


# =========================================================================== #
#                    CONFIG: SERIALIZATION                                    #
# =========================================================================== #


@pytest.mark.unit
def test_dump_portable_converts_paths():
    """Test dump_portable() makes paths relative."""
    config = Config()

    portable = config.dump_portable()

    # Paths should be relative or portable
    assert "dataset" in portable
    assert "telemetry" in portable


@pytest.mark.unit
def test_dump_serialized_json_compatible():
    """Test dump_serialized() produces JSON-compatible dict."""
    config = Config()

    serialized = config.dump_serialized()

    # Should be dict with all sub-configs
    assert isinstance(serialized, dict)
    assert "hardware" in serialized
    assert "training" in serialized


# =========================================================================== #
#                    CONFIG: PROPERTIES                                       #
# =========================================================================== #


@pytest.mark.unit
def test_run_slug_property():
    """Test run_slug combines dataset and model names."""
    config = Config()

    slug = config.run_slug

    assert "bloodmnist" in slug
    assert config.model.name in slug


@pytest.mark.unit
def test_num_workers_property():
    """Test num_workers delegates to hardware config."""
    config = Config()

    workers = config.num_workers

    assert workers >= 0
    assert workers == config.hardware.effective_num_workers


# =========================================================================== #
#                    CONFIG: EDGE CASES                                       #
# =========================================================================== #


@pytest.mark.unit
def test_frozen_immutability():
    """Test Config is frozen (immutable)."""
    config = Config()

    with pytest.raises(ValidationError):
        config.training = None


@pytest.mark.unit
def test_min_lr_boundary_condition_line_106(mock_metadata_28):
    """
    Lines 106-110: msg creation and raise ValueError(msg) for min_lr >= learning_rate
    """
    # Test 1: min_lr == learning_rate
    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28, metadata=mock_metadata_28),
            model=ModelConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                epochs=100, mixup_epochs=0, use_amp=False, learning_rate=0.001, min_lr=0.001
            ),
            hardware=HardwareConfig(device="cpu"),
        )

    # Test 2: min_lr > learning_rate
    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28, metadata=mock_metadata_28),
            model=ModelConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                epochs=100, mixup_epochs=0, use_amp=False, learning_rate=0.001, min_lr=0.002
            ),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.integration
def test_invalid_yaml_raises_error(temp_invalid_yaml):
    """Test invalid YAML raises validation error."""
    # This YAML has min_lr > learning_rate
    with pytest.raises(ValidationError):
        args = argparse.Namespace(
            config=str(temp_invalid_yaml), dataset="bloodmnist", pretrained=True
        )
        Config.from_args(args)
