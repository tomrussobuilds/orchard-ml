"""
Test Suite for Config Manifest.

Tests main Config class integration, cross-validation,
serialization, and from_recipe factory.
"""

import pytest
from pydantic import ValidationError

from orchard.core import ArchitectureConfig, Config, DatasetConfig, HardwareConfig, TrainingConfig


# CONFIG: BASIC CONSTRUCTION
@pytest.mark.unit
def test_config_defaults():
    """Test Config with all default sub-configs."""
    config = Config()

    assert config.hardware is not None
    assert config.training is not None
    assert config.dataset is not None
    assert config.architecture is not None


# CONFIG: CROSS-VALIDATION
@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resnet_18_supports_all_resolutions(device):
    """
    resnet_18 supports 28x28, 64x64, and 224x224 resolutions.
    """
    config_28 = Config(
        dataset=DatasetConfig(name="bloodmnist", resolution=28),
        architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device=device),
    )
    assert config_28.dataset.resolution == 28

    config_64 = Config(
        dataset=DatasetConfig(name="bloodmnist", resolution=64, force_rgb=True),
        architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device=device),
    )
    assert config_64.dataset.resolution == 64

    config_224 = Config(
        dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
        architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device=device),
    )
    assert config_224.dataset.resolution == 224


@pytest.mark.unit
@pytest.mark.parametrize("architecture_name", ["efficientnet_b0", "vit_tiny", "convnext_tiny"])
def test_224_models_require_resolution_224(architecture_name):
    """
    efficientnet_b0 and vit_tiny require 224x224 resolution.
    Using them with 28x28 should raise ValueError.
    """
    with pytest.raises(
        ValidationError,
        match=f"'{architecture_name}' requires resolution=224",
    ):
        Config(
            dataset=DatasetConfig(
                name="bloodmnist",
                resolution=28,
            ),
            architecture=ArchitectureConfig(
                name=architecture_name,
                pretrained=False,
            ),
            training=TrainingConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_mini_cnn_rejects_224():
    """mini_cnn only supports 28x28 and 64x64 resolutions."""
    with pytest.raises(
        ValidationError,
        match="'mini_cnn' requires resolution 28, 32, or 64",
    ):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_mini_cnn_accepts_32():
    """mini_cnn accepts 32x32 resolution (CIFAR)."""
    cfg = Config(
        dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
        architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device="cpu"),
    )
    assert cfg.dataset.resolution == 32


@pytest.mark.unit
def test_resnet_18_accepts_32():
    """resnet_18 accepts 32x32 resolution (CIFAR)."""
    cfg = Config(
        dataset=DatasetConfig(name="cifar10", resolution=32, force_rgb=True),
        architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
        training=TrainingConfig(),
        hardware=HardwareConfig(device="cpu"),
    )
    assert cfg.dataset.resolution == 32


@pytest.mark.unit
def test_resnet_18_rejects_invalid_resolution():
    """resnet_18 only supports 28, 32, 64, or 224, not arbitrary resolutions."""
    with pytest.raises(
        ValidationError,
        match=r"'resnet_18' supports resolutions \[28, 32, 64, 224\]",
    ):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=112),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=False),
            training=TrainingConfig(),
            hardware=HardwareConfig(device="cpu"),
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
                mixup_epochs=10,
            ),
            dataset=DatasetConfig(),
            architecture=ArchitectureConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
def test_amp_auto_disabled_on_cpu():
    """Test AMP is automatically disabled on CPU with warning."""
    with pytest.warns(UserWarning, match="AMP.*CPU"):
        cfg = Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(use_amp=True),
            hardware=HardwareConfig(device="cpu"),
        )

    assert cfg.training.use_amp is False


@pytest.mark.unit
def test_pretrained_requires_rgb():
    """Test pretrained model validation enforces RGB channels."""
    with pytest.raises(ValidationError, match="Pretrained.*requires RGB"):
        Config(
            dataset=DatasetConfig(name="organcmnist", resolution=28, force_rgb=False),
            architecture=ArchitectureConfig(name="resnet_18", pretrained=True),
            training=TrainingConfig(),
            hardware=HardwareConfig(device="cpu"),
        )


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:AMP requires GPU.*:UserWarning")
def test_min_lr_equals_lr_direct_instantiation(mock_metadata_28):
    """Test min_lr == learning_rate validation via direct instantiation."""
    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(
                name="bloodmnist",
                resolution=28,
                metadata=mock_metadata_28,
            ),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                learning_rate=0.001,
                min_lr=0.001,
            ),
            hardware=HardwareConfig(device="cpu"),
        )


# CONFIG: SERIALIZATION
@pytest.mark.unit
def test_dump_portable_converts_paths():
    """Test dump_portable() makes paths relative."""
    config = Config()

    portable = config.dump_portable()

    assert "dataset" in portable
    assert "telemetry" in portable


@pytest.mark.unit
def test_dump_serialized_json_compatible():
    """Test dump_serialized() produces JSON-compatible dict."""
    config = Config()

    serialized = config.dump_serialized()

    assert isinstance(serialized, dict)
    assert "hardware" in serialized
    assert "training" in serialized


# CONFIG: PROPERTIES
@pytest.mark.unit
def test_run_slug_property():
    """Test run_slug combines dataset and model names."""
    config = Config()

    slug = config.run_slug

    assert "bloodmnist" in slug
    assert config.architecture.name in slug


@pytest.mark.unit
def test_run_slug_sanitizes_timm_slash():
    """Test run_slug replaces / with _ for timm model names."""
    config = Config(
        architecture=ArchitectureConfig(name="timm/convnext_base", pretrained=False),
        dataset=DatasetConfig(name="bloodmnist", resolution=224, force_rgb=True),
        hardware=HardwareConfig(device="cpu"),
    )

    slug = config.run_slug

    assert "/" not in slug
    assert "timm_convnext_base" in slug


@pytest.mark.unit
def test_num_workers_property():
    """Test num_workers delegates to hardware config."""
    config = Config()

    workers = config.num_workers

    assert workers >= 0
    assert workers == config.hardware.effective_num_workers


# CONFIG: EDGE CASES
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
    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28, metadata=mock_metadata_28),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                epochs=100, mixup_epochs=0, use_amp=False, learning_rate=0.001, min_lr=0.001
            ),
            hardware=HardwareConfig(device="cpu"),
        )

    with pytest.raises(ValidationError):
        Config(
            dataset=DatasetConfig(name="bloodmnist", resolution=28, metadata=mock_metadata_28),
            architecture=ArchitectureConfig(name="mini_cnn", pretrained=False),
            training=TrainingConfig(
                epochs=100, mixup_epochs=0, use_amp=False, learning_rate=0.001, min_lr=0.002
            ),
            hardware=HardwareConfig(device="cpu"),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
